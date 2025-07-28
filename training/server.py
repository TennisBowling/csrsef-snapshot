import asyncio
import websockets
import json
import pandas as pd
import numpy as np
from typing import Dict, Set, List
import logging
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
import os
import glob
import hashlib
import gc
import aiosqlite

logging.basicConfig(level=logging.INFO)

@dataclass
class WorkBatch:
    batch_id: str
    rows: List[dict]
    assigned_to: str = None
    assigned_time: datetime = None
    completed: bool = False
    skipped: bool = False
    chunk_id: int = None

class LLMServer:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        batch_size: int = 1,
        target_rows: int = 400000,
        db_path: str = "batches.db",
        dataset_path: str = "so_dataset.sqlite"
    ):
        self.host = host
        self.port = port
        self.batch_size = batch_size
        self.checkpoint_size = 200
        self.clients: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.work_batches: Dict[str, WorkBatch] = {}
        self.completed_results = []
        self.data_loaded = False
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.target_rows = target_rows  
        self.current_dataset_position = 0  
        self.initial_load_size = 100000
        self.load_increment_size = 10000
        self.all_data_processed = False

        self.work_batches_lock = asyncio.Lock()

        self.db_conn = None
        self.db_path = db_path
        self.dataset_conn = None
        self.dataset_path = dataset_path

        self.is_loading_data = False

    async def async_init_db(self):
        self.db_conn = await aiosqlite.connect(self.db_path)
        async with self.db_conn.cursor() as c:
            await c.execute("""
                CREATE TABLE IF NOT EXISTS completed_batches (
                    batch_id TEXT PRIMARY KEY
                )
            """)
            await c.execute("""
                CREATE TABLE IF NOT EXISTS blacklisted_batches (
                    batch_id TEXT PRIMARY KEY
                )
            """)
            await c.execute("""
                CREATE TABLE IF NOT EXISTS skipped_batches (
                    batch_id TEXT PRIMARY KEY
                )
            """)
            await self.db_conn.commit()

        self.dataset_conn = await aiosqlite.connect(self.dataset_path)
        self.dataset_conn.row_factory = aiosqlite.Row

    async def init_db(self):
        await self.async_init_db()

    async def _load_data_chunk(self, num_rows: int):
        if self.is_loading_data:
            logging.info("Data loading already in progress, skipping...")
            return
        
        self.is_loading_data = True
        logging.info(f"Loading up to {num_rows} rows from position {self.current_dataset_position}...")

        chunk_id = self.current_dataset_position

        try:
            async with self.dataset_conn.execute("""
                SELECT Body, Score, Title, row_id
                FROM posts 
                WHERE row_id > ? 
                ORDER BY row_id 
                LIMIT ?
            """, (self.current_dataset_position, num_rows)) as cursor:
                
                rows = await cursor.fetchall()
            
            if not rows:
                logging.info("No more data to load.")
                self.all_data_processed = True
                return

            records = []
            last_row_id = self.current_dataset_position  # Track the actual last row we process
            
            for idx, row in enumerate(rows):
                record = dict(row)
                last_row_id = record['row_id']  # Update with each row's ID
                del record['row_id']
                
                for key, value in record.items():
                    if value is None:
                        continue
                    elif isinstance(value, (int, float)):
                        record[key] = value
                    elif hasattr(value, 'isoformat'):
                        record[key] = value.isoformat()
                records.append(record)

                if idx > 0 and idx % 10000 == 0:
                    logging.info(f"Processed {idx} rows in this chunk")

            new_batches = 0
            for i in range(0, len(records), self.batch_size):
                batch_rows = records[i:i + self.batch_size]
                batch_id = self.generate_deterministic_batch_id(batch_rows)

                if not await self.is_batch_processed(batch_id):
                    async with self.work_batches_lock:
                        if batch_id not in self.work_batches:
                            batch = WorkBatch(
                                batch_id=batch_id,
                                rows=batch_rows,
                                chunk_id=chunk_id
                            )
                            self.work_batches[batch.batch_id] = batch
                            new_batches += 1

            self.current_dataset_position = last_row_id
            
            logging.info(f"Loaded {len(records)} new rows, created {new_batches} new batches.")

        except Exception as e:
            logging.error(f"Error loading data chunk: {e}")
        finally:
            self.is_loading_data = False

    async def __adel__(self):
        if self.db_conn:
            await self.db_conn.close()
        if self.dataset_conn:
            await self.dataset_conn.close()

    async def save_completed_batch_id(self, batch_id: str):
        async with self.db_conn.cursor() as c:
            await c.execute("INSERT OR IGNORE INTO completed_batches (batch_id) VALUES (?)", (batch_id,))
            await self.db_conn.commit()

    async def save_blacklisted_batch_id(self, batch_id: str):
        async with self.db_conn.cursor() as c:
            await c.execute("INSERT OR IGNORE INTO blacklisted_batches (batch_id) VALUES (?)", (batch_id,))
            await self.db_conn.commit()

    async def save_skipped_batch_id(self, batch_id: str):
        async with self.db_conn.cursor() as c:
            await c.execute("INSERT OR IGNORE INTO skipped_batches (batch_id) VALUES (?)", (batch_id,))
            await self.db_conn.commit()

    def generate_deterministic_batch_id(self, rows: List[dict]) -> str:
        row_str = json.dumps(rows, sort_keys=True)
        return hashlib.sha256(row_str.encode()).hexdigest()[:32]

    async def cleanup_old_batches(self):
        async with self.work_batches_lock:
            batches_to_remove = []
            for batch_id, batch in self.work_batches.items():
                if await self.is_batch_processed(batch_id):
                    batches_to_remove.append(batch_id)

            for batch_id in batches_to_remove:
                del self.work_batches[batch_id]

        gc.collect()
    
    async def is_batch_processed(self, batch_id: str) -> bool:
        async with self.db_conn.cursor() as c:
            await c.execute("""
                SELECT 1 FROM completed_batches WHERE batch_id = ?
                UNION ALL
                SELECT 1 FROM blacklisted_batches WHERE batch_id = ?
                UNION ALL
                SELECT 1 FROM skipped_batches WHERE batch_id = ?
            """, (batch_id, batch_id, batch_id))
            result = await c.fetchone()
            return bool(result)

    async def load_dataset(self):
        if not self.data_loaded:
            logging.info("Loading initial dataset...")
            await self._load_data_chunk(self.initial_load_size)
            self.data_loaded = True
            remaining_batches = len(self.work_batches)
            logging.info(f"Initial dataset loaded. Remaining batches: {remaining_batches}")

    async def get_next_batch(self, client_id: str):
        # Lock around any iteration over self.work_batches
        async with self.work_batches_lock:
            batch_keys = list(self.work_batches.keys())
            for batch_id in batch_keys:
                batch = self.work_batches.get(batch_id)
                if not batch:
                    continue
                if (
                    not batch.completed
                    and not batch.skipped
                    and not await self.is_batch_processed(batch.batch_id)
                    and (
                        batch.assigned_to is None
                        or (datetime.now() - batch.assigned_time).total_seconds() > 3600
                    )
                ):
                    batch.assigned_to = client_id
                    batch.assigned_time = datetime.now()
                    return batch
        async with self.work_batches_lock:
            if (
                not self.all_data_processed
                and not self.is_loading_data
                and len(self.work_batches) < (self.initial_load_size * 2 / self.batch_size)
            ):
                asyncio.create_task(self.load_more_data())
        return None

    async def checkpoint_results(self):
        if not self.completed_results:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = os.path.join(self.checkpoint_dir, f'checkpoint_{timestamp}.parquet')

        logging.info(f"Creating checkpoint with {len(self.completed_results)} results")

        df = pd.DataFrame(self.completed_results)
        df.to_parquet(checkpoint_file)

        self.completed_results = []

        await self.cleanup_old_batches()

        logging.info(f"Checkpoint saved to {checkpoint_file}")

    async def handle_client(self, websocket, path):
        client_id = str(uuid.uuid4())
        self.clients[client_id] = websocket
        logging.info(f"New client connected: {client_id}")

        try:
            async for message in websocket:
                try:
                    logging.debug(f"Received message from client {client_id}: {message[:200]}...")
                    data = json.loads(message)
                    command = data.get('command')
                    logging.info(f"Processing command '{command}' from client {client_id}")

                    if command == 'get_work':
                        batch = await self.get_next_batch(client_id)
                        if batch:
                            response = {
                                'command': 'process_batch',
                                'batch_id': batch.batch_id,
                                'data': batch.rows
                            }
                            logging.info(f"Sending batch {batch.batch_id} to client {client_id}")
                            await websocket.send(json.dumps(response))
                        else:
                            logging.info(f"No work available for client {client_id}")
                            await websocket.send(json.dumps({'command': 'no_work'}))

                    elif command == 'submit_results':
                        batch_id = data.get('batch_id')
                        results = data.get('results')
                        model = data.get('model', 'unknown')
                        logging.info(f"Received results for batch {batch_id} from client {client_id} using model {model}")

                        if batch_id in self.work_batches:
                            self.work_batches[batch_id].completed = True
                            await self.save_completed_batch_id(batch_id)

                            for result in results:
                                result['model_used'] = model

                            self.completed_results.extend(results)
                            logging.info(f"Processed results for batch {batch_id}")

                            await self.cleanup_old_batches()

                            if len(self.completed_results) >= self.checkpoint_size:
                                await self.checkpoint_results()

                            if (await self.count_completed_in_db() >= self.target_rows // self.batch_size
                                and self.all_data_processed):
                                logging.info("Target row count reached and all data processed, saving final results...")
                                if self.completed_results:
                                    await self.checkpoint_results()
                                await self.save_results()

                    elif command == 'no_return':
                        batch_id = data.get('batch_id')
                        logging.info(f"Client {client_id} reported no return for batch {batch_id}")
                        if batch_id in self.work_batches:
                            self.work_batches[batch_id].skipped = True
                            await self.save_skipped_batch_id(batch_id)
                            await self.save_blacklisted_batch_id(batch_id)

                            await self.cleanup_old_batches()

                            if (not self.is_loading_data
                                and len(self.work_batches) < self.initial_load_size * 2 / self.batch_size
                                and not self.all_data_processed):
                                asyncio.create_task(self.load_more_data())

                    elif command == 'status':
                        total_batches = self.target_rows // self.batch_size
                        completed_batches = await self.count_completed_in_db()
                        status_data = {
                            'command': 'status_update',
                            'total_batches': total_batches,
                            'completed_batches': completed_batches,
                            'connected_clients': len(self.clients),
                            'current_results_buffer': len(self.completed_results),
                            'remaining_batches': len([b for b in self.work_batches.values() if not b.completed])
                        }
                        logging.info(f"Sending status update to client {client_id}: {status_data}")
                        await websocket.send(json.dumps(status_data))

                except json.JSONDecodeError as e:
                    logging.error(f"JSON decode error from client {client_id}: {str(e)}")
                    continue

                except Exception as e:
                    logging.error(f"Error processing message from client {client_id}: {str(e)}")
                    continue

        except websockets.exceptions.ConnectionClosedOK:
            logging.info(f"Client {client_id} disconnected normally")
        except websockets.exceptions.ConnectionClosedError as e:
            logging.error(f"Client {client_id} connection closed with error: {str(e)}")
        except Exception as e:
            logging.error(f"Unexpected error with client {client_id}: {str(e)}")
        finally:
            if client_id in self.clients:
                del self.clients[client_id]
            logging.info(f"Client {client_id} removed from active clients")

    async def count_completed_in_db(self) -> int:
        async with self.db_conn.execute("SELECT COUNT(*) FROM completed_batches") as c:
            result = await c.fetchone()
            return result[0]

    async def save_results(self):
        logging.info("Saving final results...")
        all_results_df = pd.DataFrame()
        checkpoints = sorted(glob.glob(os.path.join(self.checkpoint_dir, 'checkpoint_*.parquet')))

        for checkpoint in checkpoints:
            logging.info(f"Loading checkpoint file: {checkpoint}")
            try:
                df = pd.read_parquet(checkpoint)
                all_results_df = pd.concat([all_results_df, df], ignore_index=True)
                logging.info(f"Loaded {len(df)} rows from {checkpoint}")
                del df
                gc.collect()
            except Exception as e:
                logging.error(f"Error loading checkpoint file {checkpoint}: {e}")

        if not all_results_df.empty:
            try:
                all_results_df.to_parquet('so_ai_solved.parquet')
                logging.info(f"Final results saved to so_ai_solved.parquet with {len(all_results_df)} rows")
            except Exception as e:
                logging.error(f"Error saving final results: {e}")

        for checkpoint in checkpoints:
            try:
                os.remove(checkpoint)
                logging.info(f"Deleted checkpoint file: {checkpoint}")
            except Exception as e:
                logging.error(f"Error deleting checkpoint file {checkpoint}: {e}")
        
        logging.info("Final save completed and cleanup done.")

    async def load_more_data(self):
        if self.all_data_processed or self.is_loading_data:
            logging.info(f"All data processed or loading already in progress.")
            return

        logging.info("Loading more data...")
        await self._load_data_chunk(self.load_increment_size)

    async def start(self):
        self.init_db_task = asyncio.create_task(self.async_init_db())
        await self.init_db_task
        await self.load_dataset()
        server = await websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            ping_interval=5,
            ping_timeout=600,
            max_size=100 * 1024 * 1024,
            compression=None
        )
        logging.info(f"Server started on ws://{self.host}:{self.port}")
        await asyncio.Future()

if __name__ == "__main__":
    server = LLMServer()
    asyncio.run(server.start())