import pyarrow.parquet as pq
import pandas as pd
from collections import Counter
from typing import List, Tuple
import ast
import difflib
from dataclasses import dataclass
import numpy as np
import multiprocessing as mp
from functools import partial
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings("error")

PROGRAMMING_NODES = {
    'FunctionDef',
    'ClassDef',
    'If',
    'For',
    'While',
    'Try',
    'With',
    'Call',
    'Assign',
    'Return',
    'Lambda'
}


@dataclass
class CodeSimilarity:
    code: str
    uniqueness_score: float = 0.0

class ASTComplexityChecker:
    def __init__(self, min_depth=3, min_node_types=4, min_prog_constructs=2):
        self.min_depth = min_depth
        self.min_node_types = min_node_types
        self.min_prog_constructs = min_prog_constructs
        
    def get_ast_complexity(self, code: str) -> tuple[bool, int, int, int]:
        try:
            tree = ast.parse(code)
            
            has_only_imports = True
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Import, ast.ImportFrom, ast.Module)):
                    has_only_imports = False
                    break
                    
            if has_only_imports:
                return False, 0, 0, 0
                
            depth = self._get_max_depth(tree)
            node_types = self._count_unique_node_types(tree)
            prog_constructs = self._count_programming_constructs(tree)
            
            is_complex = (
                depth >= self.min_depth and 
                len(node_types) >= self.min_node_types and
                prog_constructs >= self.min_prog_constructs
            )
            
            return is_complex, depth, len(node_types), prog_constructs
            
        except (Exception, SyntaxWarning):
            return False, 0, 0, 0
            
    def _count_programming_constructs(self, node) -> int:
        count = 1 if type(node).__name__ in PROGRAMMING_NODES else 0
        for child in ast.iter_child_nodes(node):
            count += self._count_programming_constructs(child)
        return count
            
    def _get_max_depth(self, node, current_depth=0) -> int:
        max_child_depth = current_depth
        for child in ast.iter_child_nodes(node):
            child_depth = self._get_max_depth(child, current_depth + 1)
            max_child_depth = max(max_child_depth, child_depth)
        return max_child_depth
    
    def _count_unique_node_types(self, node) -> set:
        node_types = {type(node).__name__}
        for child in ast.iter_child_nodes(node):
            node_types.update(self._count_unique_node_types(child))
        return node_types

class ASTComparer:
    def __init__(self):
        self.node_types = set()
        
    def get_ast_structure(self, code: str) -> List[str]:
        try:
            tree = ast.parse(code)
            return self._get_node_structure(tree)
        except (Exception, SyntaxWarning):
            return []
    
    def _get_node_structure(self, node) -> List[str]:
        structure = [type(node).__name__]
        for child in ast.iter_child_nodes(node):
            structure.extend(self._get_node_structure(child))
        return structure

    def calculate_similarity(self, code1: str, code2: str) -> float:
        struct1 = self.get_ast_structure(code1)
        struct2 = self.get_ast_structure(code2)
        
        if not struct1 or not struct2:
            return 0.0
            
        matcher = difflib.SequenceMatcher(None, struct1, struct2)
        return matcher.ratio()

def process_batch(batch_data: Tuple[List[str], List[str], int, int]) -> List[float]:
    batch_codes, all_codes, sample_size, batch_idx = batch_data
    comparer = ASTComparer()
    n_all_codes = len(all_codes)
    batch_scores = []
    
    desc = f'Batch {batch_idx}'
    for code in tqdm(batch_codes, desc=desc, leave=False):
        compare_indices = np.random.choice(
            range(n_all_codes),
            min(sample_size, n_all_codes),
            replace=False
        )
        
        similarities = [
            comparer.calculate_similarity(code, all_codes[j])
            for j in compare_indices
        ]
        
        avg_similarity = np.mean(similarities) if similarities else 0
        batch_scores.append(1 - avg_similarity)
        
    return batch_scores

def find_most_unique_code_parallel(df: pd.DataFrame, 
                                 code_column: str, 
                                 top_percent: float,
                                 sample_size: int = 1000,
                                 n_processes: int = 24,
                                 min_ast_depth: int = 3,
                                 min_node_types: int = 4,
                                 min_prog_constructs: int = 2) -> pd.DataFrame:
    complexity_checker = ASTComplexityChecker(
        min_depth=min_ast_depth,
        min_node_types=min_node_types,
        min_prog_constructs=min_prog_constructs
    )
    
    valid_codes = []
    valid_indices = []
    depths = []
    node_type_counts = []
    prog_construct_counts = []
    
    print("Filtering code snippets...")
    for idx, code in enumerate(tqdm(df[code_column])):
        is_complex, depth, node_types, prog_constructs = complexity_checker.get_ast_complexity(code)
        if is_complex:
            valid_codes.append(code)
            valid_indices.append(idx)
            depths.append(depth)
            node_type_counts.append(node_types)
            prog_construct_counts.append(prog_constructs)
            
    print(f"Found {len(valid_codes)} code snippets with actual programming constructs")
    print(f"Average AST depth: {np.mean(depths):.2f}")
    print(f"Average unique node types: {np.mean(node_type_counts):.2f}")
    print(f"Average programming constructs: {np.mean(prog_construct_counts):.2f}")
    
    n_codes = len(valid_codes)
    
    batch_size = max(1, n_codes // n_processes)
    batches = []
    
    for i in range(0, n_codes, batch_size):
        batch_codes = valid_codes[i:i + batch_size]
        batches.append((batch_codes, valid_codes, sample_size, len(batches)))
    
    with mp.Pool(processes=n_processes) as pool:
        batch_results = list(tqdm(
            pool.imap(process_batch, batches),
            total=len(batches),
            desc='Overall Progress',
            position=0
        ))
    
    uniqueness_scores = []
    for batch_scores in batch_results:
        uniqueness_scores.extend(batch_scores)
    
    print("Calculating final results...")
    
    result_df = df.iloc[valid_indices].copy()
    result_df['ast_depth'] = depths
    result_df['ast_node_types'] = node_type_counts
    result_df['uniqueness_score'] = uniqueness_scores
    
    cutoff_index = int(len(result_df) * (top_percent / 100))
    result = result_df.nlargest(cutoff_index, 'uniqueness_score')
    
    print(f"Done! Kept {len(result)} most unique code snippets.")
    return result

if __name__ == '__main__':
    df = pd.read_parquet('cleaned_ai_dataset.parquet')
    result = find_most_unique_code_parallel(
        df, 
        "code", 
        top_percent=90,
        sample_size=1000,
        n_processes=110,
        min_ast_depth=3,
        min_node_types=4
    )
    result.to_parquet('ai_final_processed.parquet')