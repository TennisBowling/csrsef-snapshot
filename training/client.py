import asyncio
import websockets
import json
import torch
from llama_cpp import Llama
import ast
import re
from typing import List, Dict
import logging
from concurrent.futures import ThreadPoolExecutor
import uuid
import colorlog
import os

logging.basicConfig(level=logging.INFO)

handler = colorlog.StreamHandler()
handler.setFormatter(
    colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )
)

logger = colorlog.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class LLMClient:
    def __init__(self, server_url: str = "wss://llm.tennisbowling.com"):
        self.server_url = server_url
        self.client_id = str(uuid.uuid4())
        self.gpu_count = torch.mps.device_count()
        self.model_name = "qwen2.5-coder-14b-instruct-q5_k_m"
        if self.gpu_count == 0:
            raise RuntimeError("No GPUs detected")
        logging.info(f"Initialized client {self.client_id} with {self.gpu_count} GPUs")

        self.models = {}
        for gpu_id in range(self.gpu_count):
            self.models[gpu_id] = self.load_model(gpu_id)

    def load_model(self, gpu_id: int) -> Llama:
        return Llama(
            model_path="qwen2.5-coder-14b-instruct-q5_k_m.gguf",
            n_gpu_layers=-1,
            n_ctx=8192,
            n_batch=512,
            n_threads=int(os.cpu_count()/self.gpu_count),
            n_threads_batch=int(os.cpu_count()/self.gpu_count),
            gpu_device=gpu_id,
            verbose=True,
        )

    def extract_code(self, text: str) -> str:
        code_block_match = re.search(r"```python(.*?)```", text, re.DOTALL)
        if not code_block_match:
            code_block_match = re.search(r"```(.*?)```", text, re.DOTALL)

        if code_block_match:
            code = code_block_match.group(1).strip()
        else:
            code = text.strip()

        try:
            ast.parse(code)
            logger.info("Model response is good")
            return code
        except SyntaxError:
            cleaned_code = re.sub(r"^[^a-zA-Z]*", "", code)
            try:
                ast.parse(cleaned_code)
                logger.info("Model response is good")
                return cleaned_code
            except SyntaxError:
                logger.warning("Retrying model response")
                return None

    def process_item(self, item: Dict, gpu_id: int) -> Dict:
        model = self.models[gpu_id]
        max_attempts = 1
        attempt = 0

        while attempt < max_attempts:
            prompt = f"""Code solution for:
{item['Body']}
Provide only the code solution in Python, no explanations."""

            response = model.create_completion(
                prompt,
                max_tokens=1024,
                temperature=0.7,  # Lower temperature for more deterministic output
                top_p=0.9,  # Adjust top_p if needed
                repeat_penalty=1.1,
                stop=["<|im_end|>"]
            )

            model_output = response["choices"][0]["text"]
            extracted_code = self.extract_code(model_output)

            if extracted_code is not None:
                return {
                    "Body": item["Body"],
                    "Score": item["Score"],
                    "Title": item["Title"],
                    "AiAnswer": extracted_code,
                    "success": True,
                }

            attempt += 1
            logger.warning(
                f"Attempt {attempt} failed to produce valid code, retrying..."
            )

        # If all attempts fail, return failure indicator
        return {
            "Body": item["Body"],
            "Score": item["Score"],
            "Title": item["Title"],
            "success": False,
        }

    async def process_batch_async(self, batch_data: List[Dict]) -> List[Dict]:
        loop = asyncio.get_event_loop()
        results = []

        async def process_item_async(item, gpu_id):
            with ThreadPoolExecutor(max_workers=1) as executor:
                return await loop.run_in_executor(executor, self.process_item, item, gpu_id)

        tasks = []
        for i, item in enumerate(batch_data):
            gpu_id = i % self.gpu_count
            tasks.append(process_item_async(item, gpu_id))

        results = await asyncio.gather(*tasks)
        return results

    async def run(self):
        while True:
            try:
                async with websockets.connect(
                    self.server_url,
                    ping_interval=20,
                    ping_timeout=60,
                    close_timeout=60,
                    max_size=500 * 1024 * 1024,  # 100MB max message size
                ) as websocket:
                    logger.info(f"Connected to server at {self.server_url}")

                    while True:
                        try:
                            # Request work
                            request = json.dumps(
                                {
                                    "command": "get_work",
                                    "client_id": self.client_id,
                                }
                            )
                            logger.debug(f"Sending request: {request}")
                            await websocket.send(request)

                            response = await websocket.recv()
                            logger.debug(f"Received response: {response[:200]}...")
                            response_data = json.loads(response)
                            command = response_data.get("command")

                            if command == "process_batch":
                                batch_id = response_data["batch_id"]
                                batch_data = response_data["data"]

                                logger.info(
                                    f"Processing batch {batch_id} with {len(batch_data)} items"
                                )
                                results = []
                                failures = False

                                results = await self.process_batch_async(batch_data)
                                for result in results:
                                    if not result['success']:
                                        failures = True
                                        break

                                if failures:
                                    # If any item failed, send no_return
                                    logger.warning(
                                        f"Sending no_return for batch {batch_id}"
                                    )
                                    no_return_data = json.dumps(
                                        {"command": "no_return", "batch_id": batch_id}
                                    )
                                    await websocket.send(no_return_data)
                                else:
                                    submit_data = json.dumps(
                                        {
                                            "command": "submit_results",
                                            "batch_id": batch_id,
                                            "results": [
                                                {
                                                    k: v
                                                    for k, v in res.items()
                                                    if k != "success"
                                                }
                                                for res in results
                                            ],
                                            "model": self.model_name,
                                        }
                                    )
                                    logger.debug(
                                        f"Submitting results for batch {batch_id}"
                                    )
                                    await websocket.send(submit_data)

                            elif command == "no_work":
                                logger.info("No work available, waiting...")
                                await asyncio.sleep(10)

                            # Request status update
                            await websocket.send(json.dumps({"command": "status"}))
                            status = json.loads(await websocket.recv())
                            logger.info(f"Status update: {status}")

                            # Small delay between iterations
                            await asyncio.sleep(1)

                        except websockets.exceptions.ConnectionClosedError as e:
                            logger.error(
                                f"Connection closed error during communication: {str(e)}"
                            )
                            raise  # Re-raise to trigger reconnection

                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error: {str(e)}")
                            continue

                        except Exception as e:
                            logger.error(f"Error during communication: {str(e)}")
                            continue

            except websockets.exceptions.ConnectionClosedError as e:
                logger.error(
                    f"Connection closed (code: {e.code}, reason: {e.reason})"
                )
                await asyncio.sleep(5)

            except websockets.exceptions.InvalidStatusCode as e:
                logger.error(f"Invalid status code: {e.status_code}")
                await asyncio.sleep(5)

            except Exception as e:
                logger.error(f"Unexpected error: {type(e).__name__}: {str(e)}")
                await asyncio.sleep(5)

if __name__ == "__main__":
    client = LLMClient()
    asyncio.run(client.run())