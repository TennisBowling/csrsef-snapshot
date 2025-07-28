import asyncio
import asyncpg
import pickle
import math
from typing import List
import multiprocessing as mp
from tqdm import tqdm
import warnings
from functools import partial

warnings.filterwarnings("ignore")

class ProcessedResult:
    def __init__(self, data: List[tuple]):
        self.data = data

BATCH_SIZE = 3000
NUM_PROCESSES = 100
NUM_WORKERS_PER_PROCESS = 10

connection_string = 'postgres://tennisbowling:tennispass@time.tennisbowling.com/tennisbowling?sslmode=prefer'

async def process_batch(batch_data: List[tuple], worker_id: int, process_id: int):
    conn = await asyncpg.connect(connection_string)
    total_inserted = 0
    
    try:
        for sub_batch in chunks(batch_data, BATCH_SIZE):
            try:
                await conn.execute('''
                    INSERT INTO code_blocks (code, ast, ast_depth, code_hash)
                    SELECT * FROM unnest($1::text[], $2::text[], $3::integer[], $4::text[])
                    ON CONFLICT (code_hash) DO NOTHING
                ''', 
                    [row[0] for row in sub_batch],
                    [row[1] for row in sub_batch],
                    [row[2] for row in sub_batch]
                )
                total_inserted += len(sub_batch)
                
            except Exception as e:
                print(f"Error in process {process_id}, worker {worker_id}: {e}")
                continue
                
        return total_inserted
    finally:
        await conn.close()

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

async def process_chunk(chunk_data: List[tuple], process_id: int):
    sub_chunks = list(chunks(chunk_data, len(chunk_data) // NUM_WORKERS_PER_PROCESS))
    tasks = [
        process_batch(sub_chunk, worker_id, process_id)
        for worker_id, sub_chunk in enumerate(sub_chunks)
    ]
    results = await asyncio.gather(*tasks)
    return sum(results)

def process_in_subprocess(chunk_data: List[tuple], process_id: int):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(process_chunk(chunk_data, process_id))
    finally:
        loop.close()

def main():
    pickle_path = '/gscratch/scrubbed/enzovt/processed_results.pickle'
    
    print("Loading pickle file...")
    with open(pickle_path, 'rb') as f:
        all_data, _ = pickle.load(f)
    
    flattened_data = []
    for result in all_data:
        flattened_data.extend(result.data)
    all_data = None
    
    print(f"Loaded {len(flattened_data)} rows")
    
    chunk_size = math.ceil(len(flattened_data) / NUM_PROCESSES)
    data_chunks = list(chunks(flattened_data, chunk_size))
    flattened_data = None
    
    print(f"Processing using {NUM_PROCESSES} processes "
          f"with {NUM_WORKERS_PER_PROCESS} async workers each...")
    
    with mp.Pool(NUM_PROCESSES) as pool:
        process_args = [
            (chunk, i) for i, chunk in enumerate(data_chunks)
        ]
        
        total_processed = 0
        for count in tqdm(
            pool.starmap(process_in_subprocess, process_args),
            total=len(data_chunks),
            desc="Processing chunks"
        ):
            total_processed += count
    
    print(f"Processing complete:")
    print(f"Total rows processed: {total_processed}")

if __name__ == '__main__':
    main()