import asyncio
from tqdm import tqdm
import websockets
import json
import logging
import colorlog
import signal
from typing import List, Dict, Optional
from openai import AsyncAzureOpenAI
import aiofiles
import tiktoken
from datetime import datetime

handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s',
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'green',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'red,bg_white',
    }
))

logger = colorlog.getLogger('batchprocessor')
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class BatchProcessor:
    def __init__(self):
        self.api_key = ""
        self.model_name = "gpt-4o-batch"
        self.client = AsyncAzureOpenAI(
            api_key=self.api_key,
            azure_endpoint="https://csrsef.openai.azure.com/",
            api_version="2024-10-01-preview",
            timeout=300
        )
        self.encoding = tiktoken.encoding_for_model('gpt-4o')
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.stopping = False
        self.lock = asyncio.Lock()
        self.batch_data = []
        self.results_mapping = {}

    async def get_valid_job(self):
        while True:
            async with self.lock:
                await self.websocket.send(json.dumps({"command": "get_work"}))
                job = json.loads(await self.websocket.recv())
                
            if job["command"] != "no_work":
                return job
            
            logger.info("No work available, retrying...")

        
    async def collect_jobs(self, num_jobs: int = 100):
        logger.info(f"Starting to collect {num_jobs} jobs...")

        input_tokens = 0
        
        for i in tqdm(range(num_jobs)):
            job = await self.get_valid_job()
            
            if job["command"] == "process_batch":
                code_content = job["data"][0]["code"] or ""
                file_path = job["data"][0]["full_path"] or ""
                
                prompt = f"""Rewrite this Python code without changing its functionality:\n\n{code_content}"""
                input_tokens += len(self.encoding.encode(prompt))

                batch_entry = {
                    "custom_id": f"req_{i}",
                    "method": "POST",
                    "url": "/chat/completions",
                    "body": {
                        "model": self.model_name,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a Python code rewriter. Slightly rewrite the provided code without changing its functionality. Only return the rewritten code, no explanations."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    }
                }
                
                # Store mapping for later processing
                self.results_mapping[f"req_{i}"] = {
                    "batch_id": job["batch_id"],
                    "code": code_content,
                    "full_path": file_path
                }
                
                self.batch_data.append(batch_entry)
            
                
        logger.info(f"Finished collecting {len(self.batch_data)} jobs, a total of {input_tokens} input tokens")
        
    async def create_batch_file(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"batch_input_{timestamp}.jsonl"
        
        async with aiofiles.open(filename, 'w') as f:
            for entry in self.batch_data:
                await f.write(json.dumps(entry) + '\n')
                
        logger.info(f"Created batch file: {filename}")
        return filename
        
    async def submit_batch(self, input_filename: str):
        try:                
            file_response = await self.client.files.create(
                file=open(input_filename, "rb"),
                purpose="batch",
            )

            while True:
                logger.info("Waiting for file to be processed")
                await asyncio.sleep(3)
                status = await self.client.files.retrieve(file_response.id)
                if status.status == "processed":
                    break

            logger.info(f"Calling batch endpoint, file is {file_response}")
            
            batch = await self.client.batches.create(
                input_file_id=file_response.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
            
            logger.info(f"Created batch with ID: {batch.id}")
            return batch.id
            
        except Exception as e:
            logger.error(f"Error submitting batch: {str(e)}")
            return None
            
    async def wait_for_completion(self, batch_id: str):
        while True:
            try:
                batch = await self.client.batches.retrieve(batch_id)
                status = batch.status
                
                logger.info(f"Batch status: {status}, batch progress: {batch.request_counts.completed}/{batch.request_counts.total}")
                
                if status == "completed":
                    return batch.output_file_id
                elif status in ["failed", "expired", "cancelled"]:
                    logger.error(f"Batch failed with status: {status}")
                    return False
                    
                await asyncio.sleep(180)  # Check every 3m
                
            except Exception as e:
                logger.error(f"Error checking batch status: {str(e)}")
                await asyncio.sleep(180)
                
    async def process_results(self, batch_id: str):
        try:
            response = await self.client.files.content(batch_id)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_filename = f"batch_results_{timestamp}.jsonl"
            
            async with aiofiles.open(results_filename, 'w') as f:
                await f.write(response.text)
                
            logger.info(f"Saved results to: {results_filename}")
            
            async with aiofiles.open(results_filename, 'r') as f:
                async for line in f:
                    result = json.loads(line)
                    custom_id = result["custom_id"]
                    
                    if custom_id in self.results_mapping:
                        original_data = self.results_mapping[custom_id]
                        
                        # Prepare result for submission
                        submit_data = {
                            "command": "submit_results",
                            "batch_id": original_data["batch_id"],
                            "results": [{
                                "code": original_data["code"],
                                "full_path": original_data["full_path"],
                                "analysis": result["response"]["body"]["choices"][0]["message"]["content"],
                                "success": True
                            }],
                            "model": self.model_name
                        }
                        
                        async with self.lock:
                            await self.websocket.send(json.dumps(submit_data))
                            await self.websocket.send(json.dumps({"command": "status"}))
                            status_response = await self.websocket.recv()
                        logger.info(f"Submitted result for {custom_id}: {status_response}")
                            
                        await asyncio.sleep(0.05)  # Rate limiting
                        
        except Exception as e:
            logger.error(f"Error processing results: {str(e)}")
            
    
    async def run(self):
            
        async with websockets.connect(
            "ws://192.168.1.10:8765",
            ping_interval=10,
            ping_timeout=600,
            close_timeout=60,
            max_size=100 * 1024 * 1024
        ) as websocket:
            self.websocket = websocket
            
            try:
                # Collect jobs
                await self.collect_jobs()
                
                if not self.batch_data:
                    logger.warning("No jobs collected, exiting")
                    return
                    
                input_file = await self.create_batch_file()
                batch_id = await self.submit_batch(input_file)
                
                if not batch_id:
                    logger.error("Failed to submit batch")
                    return
                    
                res_id = await self.wait_for_completion(batch_id)
                if res_id:
                    await self.process_results(res_id)
                    
            except Exception as e:
                logger.error(f"Error in main execution: {str(e)}")
                
            finally:
                self.stopping = True
                
processor = BatchProcessor()
asyncio.run(processor.run())