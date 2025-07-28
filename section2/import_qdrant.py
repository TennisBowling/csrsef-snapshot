from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import PointStruct
from qdrant_client.models import Distance, VectorParams
import numpy as np
import asyncpg
import asyncio
from tqdm import tqdm

async def get_total_rows(pool):
    async with pool.acquire() as conn:
        return await conn.fetchval('SELECT COUNT(*) FROM code_embeddings_raw')

async def fetch_batch(pool, offset, batch_size):
    async with pool.acquire() as conn:
        return await conn.fetch('''
            SELECT id, code, ast_data, ast_depth, embedding, full_path 
            FROM code_embeddings_raw 
            ORDER BY id 
            LIMIT $1 OFFSET $2
        ''', batch_size, offset)

async def process_batch(client, rows):
    points = [
        PointStruct(
            id=row['id'],
            vector=np.array(row['embedding']).tolist(),
            payload={
                'code': row['code'],
                'ast_data': row['ast_data'],
                'ast_depth': row['ast_depth'],
                'full_path': row['full_path']
            }
        )
        for row in rows
    ]
    
    await client.upsert(
        collection_name="code_embeddings",
        points=points
    )

async def main():
    client = AsyncQdrantClient(host="localhost", port=6333, timeout=500)
    pool = await asyncpg.create_pool(
        'postgres://tennisbowling:tennispass@time.tennisbowling.com/tennisbowling'
    )

    try:
        await client.create_collection(
            collection_name="code_embeddings",
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )

        total_rows = await get_total_rows(pool)
        batch_size = 10000
        
        with tqdm(total=total_rows, desc="Migrating data") as pbar:
            offset = 0
            while offset < total_rows:
                rows = await fetch_batch(pool, offset, batch_size)
                
                if not rows:
                    break
                
                await process_batch(client, rows)
                
                pbar.update(len(rows))
                offset += batch_size

    finally:
        await pool.close()
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())