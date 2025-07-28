import pyarrow.parquet as pq
import pandas as pd
from typing import List, Tuple
import ast
import multiprocessing as mp
from dataclasses import dataclass
from tqdm.auto import tqdm
import numpy as np

# Common programming construct nodes we want to check for
PROGRAMMING_NODES = {
    'FunctionDef',    # Function definitions
    'ClassDef',       # Class definitions
    'If',            # If statements
    'For',           # For loops
    'While',         # While loops
    'Try',           # Try blocks
    'With',          # With statements
    'Call',          # Function calls
    'Assign',        # Assignments
    'Return',        # Return statements
    'Lambda',        # Lambda functions
}

@dataclass
class CodeMetrics:
    depth: int
    node_types: int
    prog_constructs: int
    is_complex: bool

def get_ast_complexity(code: str, min_depth=3, min_node_types=4, min_prog_constructs=2) -> CodeMetrics:
    try:
        tree = ast.parse(code)
        
        # Check if the code only contains imports
        has_only_imports = True
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom, ast.Module)):
                has_only_imports = False
                break
                
        if has_only_imports:
            return CodeMetrics(0, 0, 0, False)
        
        # Get metrics
        depth = _get_max_depth(tree)
        node_types = len(_count_unique_node_types(tree))
        prog_constructs = _count_programming_constructs(tree)
        
        is_complex = (
            depth >= min_depth and 
            node_types >= min_node_types and
            prog_constructs >= min_prog_constructs
        )
        
        return CodeMetrics(depth, node_types, prog_constructs, is_complex)
        
    except Exception:
        return CodeMetrics(0, 0, 0, False)  # TODO: Filter these out later

def _count_programming_constructs(node) -> int:
    count = 1 if type(node).__name__ in PROGRAMMING_NODES else 0
    for child in ast.iter_child_nodes(node):
        count += _count_programming_constructs(child)
    return count
        
def _get_max_depth(node, current_depth=0) -> int:
    max_child_depth = current_depth
    for child in ast.iter_child_nodes(node):
        child_depth = _get_max_depth(child, current_depth + 1)
        max_child_depth = max(max_child_depth, child_depth)
    return max_child_depth

def _count_unique_node_types(node) -> set:
    node_types = {type(node).__name__}
    for child in ast.iter_child_nodes(node):
        node_types.update(_count_unique_node_types(child))
    return node_types

def process_batch(batch_data: Tuple[pd.DataFrame, str, dict]) -> pd.DataFrame:
    batch_df, code_column, params = batch_data
    
    results = []
    for code in batch_df[code_column]:
        metrics = get_ast_complexity(
            code,
            min_depth=params['min_depth'],
            min_node_types=params['min_node_types'],
            min_prog_constructs=params['min_prog_constructs']
        )
        results.append(metrics)
    
    batch_df['ast_depth'] = [m.depth for m in results]
    batch_df['ast_node_types'] = [m.node_types for m in results]
    batch_df['prog_constructs'] = [m.prog_constructs for m in results]
    batch_df['is_complex'] = [m.is_complex for m in results]
    
    return batch_df

def analyze_code_complexity(
    input_parquet: str,
    output_parquet: str,
    code_column: str,
    min_depth: int = 3,
    min_node_types: int = 4,
    min_prog_constructs: int = 2,
    batch_size: int = 10000,
    n_processes: int = mp.cpu_count()
) -> None:
    # Parameters to pass to each process
    params = {
        'min_depth': min_depth,
        'min_node_types': min_node_types,
        'min_prog_constructs': min_prog_constructs
    }
    
    print(f"Processing {input_parquet} in batches...")
    
    pf = pq.ParquetFile(input_parquet)
    total_rows = pf.metadata.num_rows
    batches = []
    
    for batch_df in tqdm(pf.iter_batches(batch_size=batch_size), 
                        total=total_rows // batch_size + 1,
                        desc="Preparing batches"):
        batch_df = batch_df.to_pandas()
        batches.append((batch_df, code_column, params))
    
    # Process batches in parallel
    print(f"Processing with {n_processes} processes...")
    with mp.Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap(process_batch, batches),
            total=len(batches),
            desc='Processing batches'
        ))
    
    print("Combining results...")
    final_df = pd.concat(results, ignore_index=True)
    
    complex_code = final_df[final_df['is_complex']]
    print(f"Processed {len(final_df)} code snippets:")
    print(f"Complex code snippets: {len(complex_code)} ({len(complex_code)/len(final_df)*100:.1f}%)")
    print(f"Average AST depth: {final_df['ast_depth'].mean():.2f}")
    print(f"Average node types: {final_df['ast_node_types'].mean():.2f}")
    print(f"Average programming constructs: {final_df['prog_constructs'].mean():.2f}")
    
    print(f"Saving results to {output_parquet}")
    final_df.to_parquet(output_parquet)

if __name__ == '__main__':
    analyze_code_complexity(
        'github.parquet',
        'github_processed.parquet',
        code_column='code',
        batch_size=10000,
        n_processes=120
    )