import pandas as pd
from typing import Tuple, List
from tqdm import tqdm
import ast
import numpy as np
from torch.utils.data import Dataset
import torch


class ASTSubtreeExtractor:
    def __init__(self,
                 max_depth: int = 10,
                 max_subtree_size: int = 70,
                 max_nodes: int = 150):
        self.max_depth = max_depth
        self.max_subtree_size = max_subtree_size
        self.max_nodes = max_nodes
        self.node_types = {'PAD': 0}

    def get_node_type(self, node: ast.AST) -> str:
        return node.__class__.__name__

    def get_node_id(self, node_type: str) -> int:
        if node_type not in self.node_types:
            self.node_types[node_type] = len(self.node_types)
        return self.node_types[node_type]

    def extract_subtrees(self, code: str) -> np.ndarray:
        try:
            tree = ast.parse(code)
            subtrees = []

            queue: List[Tuple[ast.AST, int]] = [(tree, 0)]
            while queue and len(subtrees) < self.max_subtree_size:
                node, depth = queue.pop(0)

                if depth < self.max_depth:
                    subtree = self._get_subtree_sequence(node)
                    if len(subtree) > 0:
                        padded_subtree = np.zeros(self.max_nodes, dtype=np.int64)
                        padded_subtree[:min(len(subtree), self.max_nodes)] = subtree[:self.max_nodes]
                        subtrees.append(padded_subtree)

                    for child in ast.iter_child_nodes(node):
                        queue.append((child, depth + 1))

            if not subtrees:
                return np.zeros((self.max_subtree_size, self.max_nodes), dtype=np.int64)

            subtrees_array = np.array(subtrees, dtype=np.int64)
            if len(subtrees) < self.max_subtree_size:
                padding = np.zeros((self.max_subtree_size - len(subtrees), self.max_nodes), dtype=np.int64)
                subtrees_array = np.vstack((subtrees_array, padding))

            return subtrees_array[:self.max_subtree_size]

        except Exception as e:
            print(f"Failed to parse code: {e}")
            return np.zeros((self.max_subtree_size, self.max_nodes), dtype=np.int64)

    def _get_subtree_sequence(self, root: ast.AST) -> List[int]:
        sequence = []
        stack = [root]

        while stack and len(sequence) < self.max_nodes:
            node = stack.pop()
            node_type = self.get_node_type(node)
            sequence.append(self.get_node_id(node_type))

            for child in reversed(list(ast.iter_child_nodes(node))):
                stack.append(child)

        return sequence


class SANNDataset(Dataset):
    def __init__(self,
                 ai_code_df: pd.DataFrame,
                 human_code_df: pd.DataFrame,
                 extractor: ASTSubtreeExtractor,
                 is_test: bool = False):

        self.ai_code = ai_code_df.reset_index(drop=True)
        self.human_code = human_code_df.reset_index(drop=True)
        
        self.extractor = extractor
        self.samples = []

        print(f"Processing {'test' if is_test else 'training'} AI code samples: {len(self.ai_code)}")
        for code in tqdm(self.ai_code['code']):
            subtrees = self.extractor.extract_subtrees(code)
            self.samples.append((subtrees, 0))  # 0 for AI

        print(f"Processing {'test' if is_test else 'training'} human code samples: {len(self.human_code)}")
        for code in tqdm(self.human_code['code']):
            subtrees = self.extractor.extract_subtrees(code)
            self.samples.append((subtrees, 1))  # 1 for human

        sample_shapes = [subtrees.shape for subtrees, _ in self.samples]
        print(f"Sample shapes: {set(sample_shapes)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        subtrees, label = self.samples[idx]
        return torch.from_numpy(subtrees), label

def custom_collate_fn(batch):
    subtrees = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    return subtrees, labels