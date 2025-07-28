import asyncio
import websockets
import json
import sqlite3
import asyncpg
import logging
from datetime import datetime, timedelta, timezone
import aiosqlite
from typing import Dict, List, Set, Optional, Deque
from dataclasses import dataclass
import signal
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WorkItem:
    id: int
    code: str
    ast_data: str
    ast_depth: int

class EmbeddingServer:
    def __init__(self, source_db_path: str, state_db_path: str):
        self.source_db_path = source_db_path
        self.state_db_path = state_db_path
        self.active_workers: Dict[str, Set[int]] = {}
        self.batch_size = 2
        self.assignment_timeout = timedelta(minutes=10)
        self.pg_pool = None
        self.shutdown_event = asyncio.Event()
        self.total_items = 0
        self.completed_items = 0
        self.last_progress_log = datetime.now(timezone.utc)
        self.progress_update_interval = timedelta(seconds=10)
        self.pending_work_items: Deque[WorkItem] = deque()
        self.cache_size = 1000
        self.min_cache_threshold = 100
        self.cache_lock = asyncio.Lock()
        self.init_complete = asyncio.Event()

    async def init_postgres(self):
        self.pg_pool = await asyncpg.create_pool(
            user='tennisbowling',
            password='tennispass',
            database='tennisbowling',
            host='127.0.0.1'
        )

        async with self.pg_pool.acquire() as conn:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS code_embeddings (
                    id SERIAL PRIMARY KEY,
                    code TEXT NOT NULL,
                    ast_data JSONB NOT NULL,
                    ast_depth INTEGER NOT NULL,
                    embedding vector(1536) NOT NULL
                );
            ''')

    async def init_state_db(self):
        try:
            logger.info(f"Initializing state database at {self.state_db_path}...")
            async with aiosqlite.connect(self.state_db_path) as db:
                logger.info("Connected to state database, creating tables...")
                await db.execute('''
                    CREATE TABLE IF NOT EXISTS work_items (
                        id INTEGER PRIMARY KEY,
                        source_id INTEGER NOT NULL,
                        code TEXT NOT NULL,
                        ast_data TEXT NOT NULL,
                        ast_depth INTEGER NOT NULL,
                        status TEXT NOT NULL
                    )
                ''')
                logger.info("Created work_items table")
                await db.execute('CREATE INDEX IF NOT EXISTS idx_status ON work_items(status)')
                logger.info("Created index")
                await db.commit()

                async with db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='work_items'") as cursor:
                    if not await cursor.fetchone():
                        raise Exception("Table creation failed - table does not exist after creation")

            logger.info("State database initialization complete")
        except Exception as e:
            logger.error(f"Failed to initialize state database: {str(e)}", exc_info=True)
            raise

    async def update_progress_stats(self):
        async with aiosqlite.connect(self.state_db_path) as db:
            async with db.execute('SELECT COUNT(*) FROM work_items') as cursor:
                self.total_items = (await cursor.fetchone())[0]

            async with db.execute('SELECT COUNT(*) FROM work_items WHERE status = "completed"') as cursor:
                self.completed_items = (await cursor.fetchone())[0]

            completion_percentage = (self.completed_items / self.total_items * 100) if self.total_items > 0 else 0
            active_workers_count = len(self.active_workers)

            logger.info(f"Progress update:")
            logger.info(f"Total items: {self.total_items:,}")
            logger.info(f"Completed: {self.completed_items:,} ({completion_percentage:.2f}%)")
            logger.info(f"Active workers: {active_workers_count}")
            logger.info(f"Remaining: {self.total_items - self.completed_items:,}")

    async def load_work_items_chunk(self):
        async with self.cache_lock:
            # Only load more if cache is getting low
            if len(self.pending_work_items) >= self.min_cache_threshold:
                return

            items_to_load = self.cache_size - len(self.pending_work_items)

            async with aiosqlite.connect(self.state_db_path) as state_db:
                # First try to load pending items from state DB
                async with state_db.execute('''
                    SELECT id, code, ast_data, ast_depth
                    FROM work_items
                    WHERE status = "pending"
                    LIMIT ?
                ''', (items_to_load,)) as cursor:
                    async for row in cursor:
                        self.pending_work_items.append(WorkItem(
                            id=row[0],
                            code=row[1],
                            ast_data=row[2],
                            ast_depth=row[3]
                        ))

                    if len(self.pending_work_items) >= self.min_cache_threshold:
                        return

                # If we still need more, load from source DB
                remaining_items = items_to_load - len(self.pending_work_items)
                if remaining_items > 0:
                    async with state_db.execute('SELECT MAX(source_id) FROM work_items') as cursor:
                        max_source_id = (await cursor.fetchone())[0] or 0

                    # Use connection pooling for source DB
                    source_conn = sqlite3.connect(self.source_db_path)
                    try:
                        source_cur = source_conn.cursor()
                        
                        # Instead of using NOT IN, use a WHERE id > max_source_id approach
                        source_cur.execute('''
                            SELECT id, code, ast_data, ast_depth
                            FROM code_blocks
                            WHERE id > ?
                            ORDER BY id
                            LIMIT ?
                        ''', (max_source_id, remaining_items))

                        rows = source_cur.fetchall()
                        if rows:
                            await state_db.executemany('''
                                INSERT INTO work_items (source_id, code, ast_data, ast_depth, status)
                                VALUES (?, ?, ?, ?, 'pending')
                            ''', [(row[0], row[1], row[2], row[3]) for row in rows])

                            await state_db.commit()

                            async with state_db.execute('''
                                SELECT id, code, ast_data, ast_depth
                                FROM work_items
                                WHERE status = 'pending'
                                ORDER BY id DESC
                                LIMIT ?
                            ''', (remaining_items,)) as cursor:
                                inserted_items_count = 0
                                async for row in cursor:
                                    self.pending_work_items.appendleft(WorkItem(
                                        id=row[0],
                                        code=row[1],
                                        ast_data=row[2],
                                        ast_depth=row[3]
                                    ))
                                    inserted_items_count += 1
                                logger.info(f"Loaded {inserted_items_count} new work items from source DB into state DB and cache.")

                    finally:
                        source_conn.close()

    async def get_next_batch(self, worker_id: str) -> List[WorkItem]:
        # First check if we need to load more items
        if len(self.pending_work_items) < self.batch_size:
            await self.load_work_items_chunk()

        async with self.cache_lock:
            work_items = []
            async with aiosqlite.connect(self.state_db_path) as db:
                while len(work_items) < self.batch_size and self.pending_work_items:
                    work_item = self.pending_work_items.popleft()
                    work_items.append(work_item)

                    # Mark as assigned in database
                    await db.execute('''
                        UPDATE work_items
                        SET status = 'assigned'
                        WHERE id = ?
                    ''', (work_item.id,))

                await db.commit()

            return work_items

    async def store_results(self, worker_id: str, results: List[dict]):
        async with self.pg_pool.acquire() as pg_conn:
            logger.info(f"Storing {len(results)} results from worker {worker_id}") # Added log here
            async with aiosqlite.connect(self.state_db_path) as sqlite_conn:
                for result in results:
                    embedding_str = str(result['embedding'])

                    # Store in PostgreSQL
                    await pg_conn.execute('''
                        INSERT INTO code_embeddings (code, ast_data, ast_depth, embedding)
                        VALUES ($1, $2, $3, $4)
                    ''', result['code'], result['ast_data'], result['ast_depth'], embedding_str)

                    await sqlite_conn.execute('''
                        UPDATE work_items
                        SET status = 'completed'
                        WHERE id = ?
                    ''', (result['id'],))

                await sqlite_conn.commit()

        if datetime.now(timezone.utc) - self.last_progress_log > self.progress_update_interval:
            await self.update_progress_stats()
            self.last_progress_log = datetime.now(timezone.utc)

    async def handle_client(self, websocket, path):
        await self.init_complete.wait()
        worker_id = str(id(websocket))
        self.active_workers[worker_id] = set()
        logger.info(f"New worker connected: {worker_id}")
        await self.update_progress_stats()

        try:
          while not self.shutdown_event.is_set():
            message = await websocket.recv()
            data = json.loads(message)

            if data.get("type") == "request_work":
              # Client is asking for work: get a batch and send it back
              work_items = await self.get_next_batch(worker_id)
              if not work_items:
               await asyncio.sleep(5)
               continue
              work_data = [{
                'id': item.id,
                'code': item.code,
                'ast_data': item.ast_data,
                'ast_depth': item.ast_depth
              } for item in work_items]

              await websocket.send(json.dumps({
                'type': 'work',
                'items': work_data
              }))
            elif data.get("type") == "results":
                asyncio.create_task(self.store_results(worker_id, data.get("items", [])))
            else:
                logger.error(f"Unknown message type from {worker_id}: {data.get('type')}")

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Worker {worker_id} disconnected")
        finally:
            del self.active_workers[worker_id]
            await self.update_progress_stats()

    async def start(self):
        try:
            logger.info("Starting database initialization...")
            await self.init_postgres()
            await self.init_state_db()

            # Signal initialization is complete
            self.init_complete.set()
            logger.info("Database initialization complete")

            server = await websockets.serve(self.handle_client, "0.0.0.0", 8765)
            logger.info("WebSocket server started on port 8765")

            loop = asyncio.get_event_loop()
            loop.add_signal_handler(signal.SIGTERM, lambda: self.shutdown_event.set())
            loop.add_signal_handler(signal.SIGINT, lambda: self.shutdown_event.set())

            async def progress_update_loop():
                while not self.shutdown_event.is_set():
                    await self.update_progress_stats()
                    await asyncio.sleep(10)  # Update every 10 seconds now

            progress_task = asyncio.create_task(progress_update_loop())

            try:
                await self.shutdown_event.wait()
            finally:
                logger.info("Shutting down server...")
                server.close()
                await server.wait_closed()

                # Cancel progress update loop
                progress_task.cancel()
                try:
                    await progress_task
                except asyncio.CancelledError:
                    pass

                if self.pg_pool:
                    await self.pg_pool.close()
                logger.info("Server shutdown complete")
        except Exception as e:
            logger.error(f"Failed to start server: {str(e)}", exc_info=True)
            raise

if __name__ == "__main__":
    server = EmbeddingServer(
        source_db_path="github_ast.sqlite",
        state_db_path="embedding_state.sqlite"
    )

    asyncio.run(server.start())