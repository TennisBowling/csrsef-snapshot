import ast
import asyncio
import asyncpg
import numpy as np
import pyarrow.parquet as pq
from typing import List, Tuple
import multiprocessing as mp
from tqdm import tqdm
import json
import hashlib
from concurrent.futures import ProcessPoolExecutor
import warnings

warnings.filterwarnings('ignore')

CONNECTION_STRING = 'postgres://tennisbowling:tennispass@time.tennisbowling.com/tennisbowling?sslmode=prefer'
BATCH_SIZE = 100
NUM_PROCESSES = 100

async def create_table():
    conn = await asyncpg.connect(CONNECTION_STRING)
    try:
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS code_blocks (
                id SERIAL PRIMARY KEY,
                code TEXT NOT NULL,
                ast TEXT NOT NULL,
                ast_depth INTEGER NOT NULL,
                full_path TEXT NOT NULL,
                code_hash TEXT NOT NULL
            )
        ''')
        
        print("Created table if it didn't exist")
        try:
            await conn.execute('ALTER TABLE code_blocks ADD CONSTRAINT unique_code_hash UNIQUE (code_hash)')
        except asyncpg.exceptions.DuplicateTableError:
            pass
    finally:
        await conn.close()

class CodeSplitter:
    def __init__(self, min_ast_depth: int = 3, min_node_types: int = 4, min_lines: int = 8):
        self.min_ast_depth = min_ast_depth
        self.min_node_types = min_node_types
        self.min_lines = min_lines
        
    def get_ast_depth(self, node: ast.AST) -> int:
        if not isinstance(node, ast.AST):
            return 0
        return 1 + max((self.get_ast_depth(child) for child in ast.iter_child_nodes(node)), default=0)
    
    def get_node_types(self, node: ast.AST) -> set:
        types = {type(node)}
        for child in ast.iter_child_nodes(node):
            types.update(self.get_node_types(child))
        return types
    
    def is_valid_block(self, node: ast.AST) -> Tuple[bool, int]:
        if all(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(node)):
            return False, 0
            
        depth = self.get_ast_depth(node)
        unique_types = len(self.get_node_types(node))
        
        return (depth >= self.min_ast_depth and 
                unique_types >= self.min_node_types), depth
    
    def ast_to_dict(self, node: ast.AST) -> dict:
        if isinstance(node, ast.AST):
            fields = {}
            for field in node._fields:
                value = getattr(node, field)
                if isinstance(value, list):
                    fields[field] = [self.ast_to_dict(x) for x in value]
                else:
                    fields[field] = self.ast_to_dict(value)
            return {
                '_type': node.__class__.__name__,
                '_fields': fields
            }
        elif isinstance(node, (str, int, float, bool, type(None))):
            return node
        else:
            raise TypeError(f"Unexpected type: {type(node)}")
    
    def extract_code_blocks(self, code: str, full_path: str) -> List[Tuple[str, str, int, str, str]]:
        try:
            tree = ast.parse(code)
        except:
            return []
            
        blocks = []
        current_lines = []
        
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if current_lines:
                    standalone_block = '\n'.join(current_lines).strip()
                    try:
                        parsed_block = ast.parse(standalone_block)
                        if len(standalone_block.splitlines()) >= self.min_lines:
                            is_valid, depth = self.is_valid_block(parsed_block)
                            if is_valid:
                                ast_json = json.dumps(self.ast_to_dict(parsed_block))
                                code_hash = hashlib.md5(standalone_block.encode()).hexdigest()
                                blocks.append((standalone_block, ast_json, depth, full_path, code_hash))
                    except:
                        pass
                    current_lines = []
                
                try:
                    node_code = ast.get_source_segment(code, node)
                    if node_code and len(node_code.splitlines()) >= self.min_lines:
                        is_valid, depth = self.is_valid_block(node)
                        if is_valid:
                            ast_json = json.dumps(self.ast_to_dict(node))
                            code_hash = hashlib.md5(node_code.encode()).hexdigest()
                            blocks.append((node_code, ast_json, depth, full_path, code_hash))
                except:
                    continue
            else:
                try:
                    line = ast.get_source_segment(code, node)
                    if line:
                        current_lines.append(line)
                except:
                    continue
        
        if current_lines:
            standalone_block = '\n'.join(current_lines).strip()
            try:
                parsed_block = ast.parse(standalone_block)
                if len(standalone_block.splitlines()) >= self.min_lines:
                    is_valid, depth = self.is_valid_block(parsed_block)
                    if is_valid:
                        ast_json = json.dumps(self.ast_to_dict(parsed_block))
                        code_hash = hashlib.md5(standalone_block.encode()).hexdigest()
                        blocks.append((standalone_block, ast_json, depth, full_path, code_hash))
            except:
                pass
        
        return blocks


def process_chunk(chunk_data):
    chunk_df, min_ast_depth, min_node_types = chunk_data
    splitter = CodeSplitter(min_ast_depth=min_ast_depth, min_node_types=min_node_types)
    
    print(f"Initial chunk size: {len(chunk_df)} rows")
    
    chunk_df = chunk_df[chunk_df['code'].notna() & (chunk_df['code'] != '')]
    print(f"After empty filtering: {len(chunk_df)} rows")
    
    all_blocks = []
    processed_files = 0
    valid_files = 0
    
    for _, row in chunk_df.iterrows():
        processed_files += 1
        try:
            blocks = splitter.extract_code_blocks(row['code'], row['full_path'])
            if blocks:
                valid_files += 1
                all_blocks.extend(blocks)
                
    
        except Exception as e:
            print(f"Error processing file {row['full_path']}: {e}")
            continue
    
    print(f"Chunk processing complete:")
    print(f"- Processed {processed_files} files")
    print(f"- Found valid blocks in {valid_files} files")
    print(f"- Total blocks found: {len(all_blocks)}")
    
    return all_blocks

async def process_batch(batch_data: List[tuple]):
    if not batch_data:
        return 0
    
    print(f"Attempting to insert batch of {len(batch_data)} blocks")
    
    conn = await asyncpg.connect(CONNECTION_STRING)
    try:
        result = await conn.execute('''
            INSERT INTO code_blocks (code, ast, ast_depth, full_path, code_hash)
            SELECT * FROM unnest($1::text[], $2::text[], $3::integer[], $4::text[], $5::text[])
            ON CONFLICT (code_hash) DO NOTHING
            RETURNING id
        ''', 
            [row[0] for row in batch_data],
            [row[1] for row in batch_data],
            [row[2] for row in batch_data],
            [row[3] for row in batch_data],
            [row[4] for row in batch_data]
        )
        print(f"Insert result: {result}")
        if isinstance(result, str):
            inserted = int(result.split()[-1])
        else:
            inserted = len(result)
        print(f"Inserted {inserted} new blocks into database")
        return inserted
    except Exception as e:
        print(f"Error in batch insert: {e}")
        print(f"First row of failed batch: {batch_data[0]}")
        return 0
    finally:
        await conn.close()

async def main():
    await create_table()
    
    print("Reading parquet file...")
    df = pq.read_table('github.parquet').to_pandas()
    print(f"Total rows in parquet: {len(df)}")
    
    df = df[df['code'].notna() & (df['code'] != '')]
    print(f"Rows after filtering empty: {len(df)}")
    
    chunk_size = max(1, len(df) // (NUM_PROCESSES * 4))
    chunks = [(chunk, 3, 4) for chunk in np.array_split(df, len(df) // chunk_size + 1)]
    print(f"Split into {len(chunks)} chunks")
    
    print(f"Processing {len(df)} files using {NUM_PROCESSES} processes...")
    total_blocks = 0
    
    with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
        current_batch = []
        
        for future in tqdm(futures, desc="Processing chunks"):
            try:
                blocks = future.result()
                if blocks:
                    print(f"Received {len(blocks)} blocks from chunk")
                    current_batch.extend(blocks)
                    
                    while len(current_batch) >= BATCH_SIZE:
                        print(f"Processing batch of size {BATCH_SIZE}")
                        inserted = await process_batch(current_batch[:BATCH_SIZE])
                        total_blocks += inserted
                        current_batch = current_batch[BATCH_SIZE:]
                        
            except Exception as e:
                print(f"Error processing chunk: {e}")
                continue
        
        while current_batch:
            batch_size = min(len(current_batch), BATCH_SIZE)
            print(f"Processing final batch of {batch_size} blocks")
            inserted = await process_batch(current_batch[:batch_size])
            total_blocks += inserted
            current_batch = current_batch[batch_size:]
    
    print(f"Processing complete:")
    print(f"Total blocks inserted: {total_blocks}")

if __name__ == '__main__':
    asyncio.run(main())