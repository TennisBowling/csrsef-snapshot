import ast
import time
import zss
import asyncio
import json
from typing import List, Any, Optional, Dict
import openai
from openai import AsyncOpenAI
from qdrant_client import QdrantClient, models
from tqdm.asyncio import tqdm_asyncio
import numpy as np
import traceback
import os

class ASTComparer:
    
    def __init__(self, normalize_whitespace: bool = True, 
                 ignore_variable_names: bool = True,
                 ignore_literal_values: bool = False):
        self.normalize_whitespace = normalize_whitespace
        self.ignore_variable_names = ignore_variable_names
        self.ignore_literal_values = ignore_literal_values
        
    def _get_children(self, node: ast.AST) -> List[ast.AST]:
        if not isinstance(node, ast.AST):
            return []
        
        return [
            child for child in ast.iter_child_nodes(node)
            if isinstance(child, ast.AST) and
            not isinstance(child, (ast.Load, ast.Store, ast.Del))
        ]
    
    def _get_node_label(self, node: ast.AST) -> str:
        if not isinstance(node, ast.AST):
            return str(node)
            
        label = node.__class__.__name__
        
        if isinstance(node, ast.Name):
            if not self.ignore_variable_names:
                label += f"_{node.id}"
        elif isinstance(node, ast.Constant):
            if not self.ignore_literal_values:
                label += f"_{type(node.value).__name__}_{node.value}"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                label += f"_{node.func.id}"
        elif isinstance(node, ast.BinOp):
            label += f"_{node.op.__class__.__name__}"
        elif isinstance(node, ast.Compare):
            ops = '_'.join(op.__class__.__name__ for op in node.ops)
            label += f"_{ops}"
            
        return label
    
    def _node_distance(self, a: str, b: str) -> int:
        if a == b:
            return 0
        
        a_parts = a.split('_')
        b_parts = b.split('_')
        
        if a_parts[0] != b_parts[0]:
            return 10
            
        return sum(1 for i in range(min(len(a_parts), len(b_parts)))
                  if a_parts[i] != b_parts[i])
    
    def compare(self, code1: str, code2: str) -> float:
        try:
            tree1 = ast.parse(code1)
            tree2 = ast.parse(code2)
            
            distance = zss.simple_distance(
                tree1, tree2,
                get_children=self._get_children,
                get_label=self._get_node_label,
                label_dist=self._node_distance
            )
            
            # Normalize by tree sizes to get a relative score
            size1 = sum(1 for _ in ast.walk(tree1))
            size2 = sum(1 for _ in ast.walk(tree2))
            max_size = max(size1, size2)
            
            return 1 - (distance / (max_size * 10))  # 10 is max node distance
            
        except SyntaxError:
            raise ValueError("Invalid Python code provided")

async def generate_rewritten_code(client: AsyncOpenAI, original_code: str) -> str:
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a Python code rewriter. Slightly rewrite the provided code without changing its functionality. Only return the rewritten code, no explanations."
                },
                {
                    "role": "user",
                    "content": f"Slightly rewrite this Python code without changing its functionality:\n\n{original_code}"
                }
            ],
            temperature=0.7,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error getting rewritten code: {str(e)}")
        raise

async def generate_embedding(client: AsyncOpenAI, code: str) -> List[float]:
    try:
        embedding_response = await client.embeddings.create(
            input=code,
            model="text-embedding-3-small"
        )
        return embedding_response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        raise

async def process_code_sample(
    client: AsyncOpenAI,
    qdrant: QdrantClient,
    comparer: ASTComparer,
    point_id: str,
    original_code: str,
    sem: asyncio.Semaphore
) -> Dict[str, Any]:
    result = {
        "id": point_id,
        "original_code": original_code,
        "rewritten_code": None,
        "original_ranking": -1,
        "error": None,
    }
    
    try:
        # Use semaphore to limit concurrent API calls
        async with sem:
            # 1. Rewrite the code using GPT-4o-mini
            rewritten_code = await generate_rewritten_code(client, original_code)
            result["rewritten_code"] = rewritten_code
            
            # 2. Generate embedding for the rewritten code
            rewritten_embedding = await generate_embedding(client, rewritten_code)
        
        # 3. Find top 100 closest embeddings
        matches = qdrant.query_points(
            collection_name="code_embeddings",
            query=rewritten_embedding,
            limit=100
        )
        
        # 4. Run TED comparison between rewritten code and each result
        ted_results = []
        for matched_point in matches.points:
            matched_code = matched_point.payload.get('code', '')
            if matched_code:
                try:
                    ted = comparer.compare(rewritten_code, matched_code)
                    ted_results.append((ted, matched_code, matched_point.id))
                except Exception:
                    continue
        
        ted_results.sort(key=lambda item: item[0], reverse=True)
        
        # 5. Find where the original code ranks
        for i, (_, _, matched_id) in enumerate(ted_results):
            if matched_id == point_id:
                result["original_ranking"] = i + 1  # 1-based indexing
                break
    
    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
    
    return result

async def process_in_batches(
    client: AsyncOpenAI,
    qdrant: QdrantClient,
    comparer: ASTComparer,
    points,
    batch_size=50,
    concurrency_limit=20
):
    all_results = []
    total_points = len(points)
    
    sem = asyncio.Semaphore(concurrency_limit)
    
    os.makedirs("results", exist_ok=True)
    
    for i in range(0, total_points, batch_size):
        batch_points = points[i:min(i+batch_size, total_points)]
        batch_number = i//batch_size + 1
        total_batches = (total_points + batch_size - 1)//batch_size
        
        print(f"Processing batch {batch_number}/{total_batches} ({len(batch_points)} points)")
        
        tasks = []
        for point in batch_points:
            code = point.payload.get('code', '')
            if code:
                task = process_code_sample(
                    client, 
                    qdrant, 
                    comparer, 
                    point.id, 
                    code,
                    sem
                )
                tasks.append(task)
        
        batch_results = await tqdm_asyncio.gather(
            *tasks,
            desc=f"Batch {batch_number}/{total_batches}"
        )
        
        all_results.extend(batch_results)
        
        batch_file = f"results/batch_{batch_number}.json"
        with open(batch_file, "w") as f:
            json.dump(batch_results, f, indent=2)
        
        # Print batch statistics
        batch_rankings = [r["original_ranking"] for r in batch_results if r["original_ranking"] > 0]
        errors = [r for r in batch_results if r["error"] is not None]
        
        print(f"Batch {batch_number} stats:")
        print(f"  Found original: {len(batch_rankings)}/{len(batch_results)}")
        print(f"  Errors: {len(errors)}/{len(batch_results)}")
        if batch_rankings:
            print(f"  Average ranking: {np.mean(batch_rankings):.2f}")
            print(f"  In top-1: {len([r for r in batch_rankings if r <= 1]) / len(batch_rankings) * 100:.2f}%")
            print(f"  In top-10: {len([r for r in batch_rankings if r <= 10]) / len(batch_rankings) * 100:.2f}%")
    
    return all_results

async def main():
    print("Initializing clients...")
    qdrant = QdrantClient(url="http://127.0.0.1:6333", timeout=1000)
    oai = AsyncOpenAI(api_key="")
    comparer = ASTComparer(
        normalize_whitespace=True,
        ignore_variable_names=True,
        ignore_literal_values=False
    )
    
    print("Fetching 50,000 random code samples...")
    start_time = time.time()
    sampled_points = qdrant.query_points(
        collection_name="code_embeddings",
        query=models.SampleQuery(sample=models.Sample.RANDOM),
        limit=100
    )
    
    print(f"Retrieved {len(sampled_points.points)} samples in {time.time() - start_time:.2f} seconds")
    
    total_start_time = time.time()
    results = await process_in_batches(
        oai, 
        qdrant, 
        comparer, 
        sampled_points.points,
        batch_size=50,
        concurrency_limit=20
    )
    total_time = time.time() - total_start_time
    
    clean_results = [r for r in results if not isinstance(r, Exception)]
    rankings = [r["original_ranking"] for r in clean_results if r["original_ranking"] > 0]
    
    stats = {
        "total_samples": len(clean_results),
        "found_original": len(rankings),
        "not_found_original": len([r for r in clean_results if r["original_ranking"] == -1]),
        "mean_ranking": float(np.mean(rankings)) if rankings else None,
        "median_ranking": float(np.median(rankings)) if rankings else None,
        "ranking_distribution": {str(i): rankings.count(i) for i in range(1, 101) if rankings.count(i) > 0},
        "top_1_percent": len([r for r in rankings if r <= 1]) / len(rankings) * 100 if rankings else None,
        "top_5_percent": len([r for r in rankings if r <= 5]) / len(rankings) * 100 if rankings else None,
        "top_10_percent": len([r for r in rankings if r <= 10]) / len(rankings) * 100 if rankings else None,
        "total_time_seconds": total_time,
    }
    
    with open("results/final_results.json", "w") as f:
        json.dump(clean_results, f, indent=2)
    
    with open("results/statistics.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print("Final statistics:")
    print(f"Total samples processed: {stats['total_samples']}")
    print(f"Found original code: {stats['found_original']} ({stats['found_original']/stats['total_samples']*100:.2f}%)")
    print(f"Not found: {stats['not_found_original']} ({stats['not_found_original']/stats['total_samples']*100:.2f}%)")
    if rankings:
        print(f"Average ranking position: {stats['mean_ranking']:.2f}")
        print(f"Median ranking position: {stats['median_ranking']}")
        print(f"In top-1: {stats['top_1_percent']:.2f}%")
        print(f"In top-5: {stats['top_5_percent']:.2f}%")
        print(f"In top-10: {stats['top_10_percent']:.2f}%")
    print(f"Total processing time: {total_time/60:.2f} minutes")
    print("Results saved to results/final_results.json and results/statistics.json")

if __name__ == "__main__":
    asyncio.run(main())
