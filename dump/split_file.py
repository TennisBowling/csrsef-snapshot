import ast
import pyarrow.parquet as pq
import pandas as pd
from typing import List, Tuple
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import random

def remove_comments(code: str) -> str:
    # Remove comments while preserving line structure
    lines = []
    for line in code.splitlines():
        if '#' in line:
            line = line.split('#')[0]
        if line.strip():
            lines.append(line.rstrip())
    return '\n'.join(lines)

class CodeSplitter:
    def __init__(self, min_ast_depth: int = 3, min_node_types: int = 4):
        self.min_ast_depth = min_ast_depth
        self.min_node_types = min_node_types
        
    def get_ast_depth(self, node: ast.AST) -> int:
        if not isinstance(node, ast.AST):
            return 0
        return 1 + max((self.get_ast_depth(child) for child in ast.iter_child_nodes(node)), default=0)
    
    def get_node_types(self, node: ast.AST) -> set:
        types = {type(node)}
        for child in ast.iter_child_nodes(node):
            types.update(self.get_node_types(child))
        return types
    
    def is_valid_block(self, node: ast.AST) -> bool:
        depth = self.get_ast_depth(node)
        unique_types = len(self.get_node_types(node))
        return depth >= self.min_ast_depth and unique_types >= self.min_node_types
    
    def extract_code_blocks(self, code: str, repo_name: str) -> List[tuple]:
        try:
            tree = ast.parse(code)
        except:
            return []
            
        blocks = []
        standalone_code = []
        current_lines = []
        
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if current_lines:
                    standalone_block = '\n'.join(current_lines).strip()
                    try:
                        parsed_block = ast.parse(standalone_block)
                        lines = remove_comments(standalone_block).splitlines()
                        
                        if (self.is_valid_block(parsed_block) and 
                            len(lines) >= 8 and
                            not all(isinstance(n, (ast.Import, ast.ImportFrom)) for n in parsed_block.body)):
                            proc_code = remove_comments(standalone_block)
                            blocks.append((standalone_block, proc_code, repo_name))
                    except:
                        pass
                    current_lines = []
                
                try:
                    node_code = ast.get_source_segment(code, node)
                    if node_code and self.is_valid_block(node):
                        proc_code = remove_comments(node_code)
                        blocks.append((node_code, proc_code, repo_name))
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
                lines = remove_comments(standalone_block).splitlines()
                
                if (self.is_valid_block(parsed_block) and 
                    len(lines) >= 8 and
                    not all(isinstance(n, (ast.Import, ast.ImportFrom)) for n in parsed_block.body)):
                    proc_code = remove_comments(standalone_block)
                    blocks.append((standalone_block, proc_code, repo_name))
            except:
                pass
        
        return blocks

def process_chunk(args):
    chunk_df, min_ast_depth, min_node_types = args
    splitter = CodeSplitter(min_ast_depth=min_ast_depth, min_node_types=min_node_types)
    all_blocks = []
    
    for _, row in chunk_df.iterrows():
        try:
            blocks = splitter.extract_code_blocks(row['code'], row['repo_name'])
            all_blocks.extend(blocks)
        except:
            continue
    
    return all_blocks

def main():
    print("Reading parquet file...")
    df = pq.read_table('github.parquet').to_pandas()
    
    n_cores = cpu_count()
    chunk_size = max(1, len(df) // (n_cores * 4))
    chunks = np.array_split(df, len(df) // chunk_size + 1)
    process_args = [(chunk, 3, 4) for chunk in chunks]
    
    print(f"Processing {len(df)} files using {n_cores} cores...")
    with Pool(n_cores) as pool:
        chunk_results = list(tqdm(
            pool.imap(process_chunk, process_args),
            total=len(chunks),
            desc="Processing files",
            unit="chunk"
        ))
    
    all_blocks = [block for chunk in chunk_results for block in chunk]
    total_blocks = len(all_blocks)
    
    # Group by processed code to find duplicates
    duplicates = defaultdict(list)
    for orig_code, proc_code, repo_name in all_blocks:
        duplicates[proc_code].append((orig_code, repo_name))
    
    group_sizes = [len(group) for group in duplicates.values()]
    print("Duplicate group statistics:")
    print(f"Total unique processed codes: {len(duplicates)}")
    print(f"Average group size: {sum(group_sizes)/len(group_sizes):.2f}")
    print(f"Max group size: {max(group_sizes)}")
    
    # Save unique blocks
    final_blocks = []
    for blocks in duplicates.values():
        final_blocks.append({'code': blocks[0][0], 'repo_name': blocks[0][1]})
    
    print("Deduplication Statistics:")
    print(f"Total blocks before deduplication: {total_blocks}")
    print(f"Unique blocks after deduplication: {len(final_blocks)}")
    print(f"Removed duplicates: {total_blocks - len(final_blocks)}")
    
    print("Saving results...")
    result_df = pd.DataFrame(final_blocks)
    result_df.to_parquet('split.parquet', index=False)
    
    print(f"Done! Final output contains {len(result_df)} unique code blocks.")

if __name__ == '__main__':
    main()