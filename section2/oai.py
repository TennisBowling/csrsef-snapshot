import asyncio
import websockets
import json
import logging
from typing import List, Dict
from openai import OpenAI
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from queue import Queue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIWorker:
    def __init__(self, work_queue: Queue, result_queue: Queue, batch_size: int = 32):
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.batch_size = batch_size
        self.client = OpenAI(api_key="")

    def process_batch(self, items: List[dict]) -> List[dict]:
        codes = [item['code'] for item in items]
        results = []

        for i in range(0, len(codes), self.batch_size):
            batch_codes = codes[i:i + self.batch_size]
            batch_items = items[i:i + self.batch_size]

            try:
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch_codes,
                    encoding_format="float"
                )

                for j, item in enumerate(batch_items):
                    results.append({
                        'id': item['id'],
                        'code': item['code'],
                        'ast_data': item['ast_data'],
                        'ast_depth': item['ast_depth'],
                        'embedding': response.data[j].embedding
                    })
            except Exception as e:
                logger.error(f"Error getting embeddings from OpenAI: {e}")
                continue

        return results

    def run(self):
        while True:
            batch = self.work_queue.get()
            if batch is None:
                break

            try:
                results = self.process_batch(batch)
                self.result_queue.put(results)
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                self.result_queue.put([])

class EmbeddingClient:
    def __init__(self, num_workers: int = 4, worker_batch_size: int = 32):
        self.num_workers = num_workers
        self.work_queues = [Queue() for _ in range(num_workers)]
        self.result_queue = Queue()
        self.workers: List[Thread] = []
        self.openai_workers: List[OpenAIWorker] = []
        self.worker_batch_size = worker_batch_size

    async def request_work(self, websocket):
        await websocket.send(json.dumps({
            'type': 'request_work'
        }))
        response = await websocket.recv()
        return json.loads(response)

    async def handle_server_communication(self, websocket):
        while True:
            try:
                # Actively request work instead of waiting
                data = await self.request_work(websocket)

                if data['type'] == 'work' and 'items' in data:
                    items = data['items']
                    if not items:
                        logger.info("No work available, waiting...")
                        await asyncio.sleep(5)
                        continue

                    num_items = len(items)
                    worker_index = 0
                    
                    # Distribute work across workers
                    for i in range(0, num_items, 1):
                        self.work_queues[worker_index].put([items[i]])
                        worker_index = (worker_index + 1) % self.num_workers

                    # Collect results
                    results = []
                    for _ in range(num_items):
                        batch_result = self.result_queue.get()
                        results.extend(batch_result)

                    await websocket.send(json.dumps({
                        'type': 'results',
                        'items': results
                    }))

                    logger.info(f"Processed and sent back {len(results)} items")

            except websockets.exceptions.ConnectionClosed:
                logger.warning("Connection closed, attempting to reconnect...")
                break
            except Exception as e:
                logger.error(f"Error in server communication: {e}")
                await asyncio.sleep(5)
                continue

    async def connect_websocket(self):
        while True:
            try:
                async with websockets.connect('wss://llm.tennisbowling.com') as websocket:
                    logger.info("Connected to embedding server")
                    await self.handle_server_communication(websocket)
            except Exception as e:
                logger.error(f"WebSocket connection error: {e}")
                await asyncio.sleep(5)

    def start(self):
        for worker_id in range(self.num_workers):
            worker = OpenAIWorker(self.work_queues[worker_id], self.result_queue, batch_size=self.worker_batch_size)
            thread = Thread(target=worker.run, daemon=True)
            self.workers.append(thread)
            self.openai_workers.append(worker)
            thread.start()

        asyncio.get_event_loop().run_until_complete(self.connect_websocket())

    def cleanup(self):
        for queue in self.work_queues:
            queue.put(None)

        for worker in self.workers:
            worker.join()

if __name__ == "__main__":
    client = EmbeddingClient(num_workers=4, worker_batch_size=1)
    try:
        client.start()
    except KeyboardInterrupt:
        logger.info("Shutting down client...")
        client.cleanup()