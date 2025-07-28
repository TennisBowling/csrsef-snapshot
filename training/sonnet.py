from claude_api.session import SessionData
from claude_api.client import ClaudeAPIClient, SendMessageResponse
from claude_api.session import get_session_data
import asyncio
import websockets
import json
import logging
import colorlog
import signal
from typing import List, Dict, Optional
from openai import AsyncOpenAI
import regex as re
from functools import partial
import ast

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

logger = colorlog.getLogger('llmclient')
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class LLMClient:
    def __init__(self):
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.stopping = False
        self.lock = asyncio.Lock()
        self.workers = 0
        self.cookie_header_value = "sessionKey=sk-ant-"
        self.user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.3 Safari/605.1.15"
        self.organization_id = "f739f9fb-0e82-412f-bb02-15fd01bff61f"

        self.model_name = "claude-3-opus-20240229"

        self.session = SessionData(self.cookie_header_value, self.user_agent, self.organization_id)
        self.client = ClaudeAPIClient(self.session, model_name=self.model_name)

    
    def extract_code(self, text: str) -> Optional[str]:
        pattern = r"```python\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
            
        pattern = r"```\n?(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
            
        return None

    def validate_code(self, code: str) -> bool:
        try:
            ast.parse(code)
            return True
        except (SyntaxError, Exception):
            return False
        
    async def send_req(self, question: str) -> str | None:
        prompt = f"""Code solution for:
{question}
Provide only the code solution in Python, no explanations."""

        try:
            loop = asyncio.get_running_loop()
            chat_id = await loop.run_in_executor(None, self.client.create_chat)

            part = partial(self.client.send_message, chat_id, prompt)

            res = (await loop.run_in_executor(None, part)).answer

            if not res:
                logger.warning("Model didn't return anything")
                return None

            code = self.extract_code(res) or res
            if not self.validate_code(code):
                logger.warning("Not able to extract valid python from model")
                return None
            
            return res
        
        except Exception as e:
            logger.error(f"Error processing: {str(e)}")
            return None
        
    async def do_work(self):
        self.workers += 1
        logger.info("Creating worker!")
        async with self.lock:
            await self.websocket.send(json.dumps({"command": "get_work"}))
            job = json.loads(await self.websocket.recv())
        
        if job["command"] == "process_batch":
            batch_id = job["batch_id"]
            question = job["data"][0]["Body"]
            score = job["data"][0]["Score"]
            title = job["data"][0]["Title"]

            response = await self.send_req(question)
            if not response:
                async with self.lock:
                    await self.websocket.send(json.dumps({
                        "command": "no_return",
                        "batch_id": batch_id,
                    }))
                self.workers -= 1
                return
                
            result = {
                'Body': question,
                'Score': score,
                'Title': title,
                'AiAnswer': response,
                'success': True
            }

            async with self.lock:
                await self.websocket.send(json.dumps({
                    "command": "submit_results",
                    "batch_id": batch_id,
                    "results": [result],
                    "model": self.model_name
                }))

                await self.websocket.send(json.dumps({"command": "status"}))
                logger.info(await self.websocket.recv())
                self.workers -= 1

        elif job["command"] == "no_work":
            await asyncio.sleep(3)
            self.workers -= 1
            return

    def shutdown(self):
        logger.info("Shutting down")
        self.stopping = True

    async def run(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            asyncio.get_running_loop().add_signal_handler(sig, self.shutdown)

        async with websockets.connect("wss://llm.tennisbowling.com", ping_interval=5, ping_timeout=600, close_timeout=60, max_size=100 * 1024 * 1024) as websocket:
            self.websocket = websocket
            while not self.stopping:
                asyncio.create_task(self.do_work())
                await asyncio.sleep(6)
            
            logger.info("Waiting for workers to finish")
            while self.workers != 0:
                await asyncio.sleep(0.2)

            logger.info("Done")

client = LLMClient()
asyncio.run(client.run())