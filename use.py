import ast
import json
import warnings
import argparse
import torch
import pickle
from typing import Tuple, Dict, Any, List, Set
import numpy as np
import math
from torch import nn
import torch.nn.functional as F
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def remove_comments(code: str) -> str:
    lines = []
    for line in code.splitlines():
        if '#' in line:
            line = line.split('#')[0]
        if line.strip():
            lines.append(line.rstrip())
    return '\n'.join(lines)

class ASTProcessor:
    def __init__(self, min_ast_depth: int = 3, min_node_types: int = 4, min_lines: int = 8):
        self.min_ast_depth = min_ast_depth
        self.min_node_types = min_node_types
        self.min_lines = min_lines

    def get_ast_depth(self, node: ast.AST) -> int:
        if not isinstance(node, ast.AST):
            return 0
        return 1 + max((self.get_ast_depth(child) for child in ast.iter_child_nodes(node)), default=0)
    
    def get_node_types(self, node: ast.AST) -> Set[type]:
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
    
    def get_source_segment(self, code: str, node: ast.AST) -> str:
        try:
            start_lineno = node.lineno - 1
            end_lineno = node.end_lineno if hasattr(node, 'end_lineno') else start_lineno + 1
            
            lines = code.splitlines()
            relevant_lines = lines[start_lineno:end_lineno]
            
            if hasattr(node, 'col_offset'):
                first_line = relevant_lines[0][node.col_offset:]
                relevant_lines[0] = first_line
            
            return '\n'.join(relevant_lines)
        except Exception as e:
            logger.error(f"Error in get_source_segment: {e}")
            return ""
            
    def extract_code_blocks(self, code: str) -> List[Tuple[str, str, int]]:
        try:
            tree = ast.parse(code.strip())
        except Exception as e:
            logger.error(f"Parse error: {e}")
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
                                blocks.append(standalone_block)
                    except Exception:
                        pass
                    current_lines = []
                
                try:
                    node_code = self.get_source_segment(code, node)
                    if node_code and len(node_code.splitlines()) >= self.min_lines:
                        is_valid, depth = self.is_valid_block(node)
                        if is_valid:
                            blocks.append(node_code)
                except Exception:
                    continue
            else:
                try:
                    line = self.get_source_segment(code, node)
                    if line:
                        current_lines.append(line)
                except Exception:
                    continue
        
        if current_lines:
            standalone_block = '\n'.join(current_lines).strip()
            try:
                parsed_block = ast.parse(standalone_block)
                if len(standalone_block.splitlines()) >= self.min_lines:
                    is_valid, depth = self.is_valid_block(parsed_block)
                    if is_valid:
                        blocks.append(standalone_block)
            except Exception:
                pass
        
        return blocks

class ASTSubtreeExtractor:
    def __init__(self, 
                 max_depth: int = 10,
                 max_subtree_size: int = 70,
                 max_nodes: int = 150,
                 node_types: dict = None):
        self.max_depth = max_depth
        self.max_subtree_size = max_subtree_size
        self.max_nodes = max_nodes
        self.node_types = node_types if node_types else {'PAD': 0}
        
    def get_node_type(self, node: ast.AST) -> str:
        return node.__class__.__name__
    
    def get_node_id(self, node_type: str) -> int:
        return self.node_types.get(node_type, 0)
    
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
            logger.error(f"Failed to parse code: {e}")
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

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        return self.gamma * (x - mean) / (var + self.eps).sqrt() + self.beta

class SANN(nn.Module):
    def __init__(self,
                 node_vocab_size: int,
                 embedding_dim: int = 128,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.node_embedding = nn.Embedding(node_vocab_size, embedding_dim, padding_idx=0)
        nn.init.normal_(self.node_embedding.weight, mean=0, std=embedding_dim ** -0.5)
        
        self.register_buffer(
            "pos_encoding",
            self._create_positional_encoding(200, embedding_dim)
        )
        
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=embedding_dim * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True
            ) for _ in range(num_layers)
        ])
        
        self.norm = LayerNorm(embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim // 2)
        self.fc3 = nn.Linear(embedding_dim // 2, 2)
        self.dropout = nn.Dropout(dropout)
        
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        pos = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pos_encoding = torch.zeros(1, max_len, d_model)
        pos_encoding[0, :, 0::2] = torch.sin(pos * div_term)
        pos_encoding[0, :, 1::2] = torch.cos(pos * div_term)
        return pos_encoding

    def forward(self, subtrees: torch.Tensor) -> torch.Tensor:
        batch_size, num_subtrees, num_nodes = subtrees.shape
        
        node_emb = self.node_embedding(subtrees)
        subtree_emb = node_emb.mean(dim=2)
        subtree_emb = subtree_emb + self.pos_encoding[:, :num_subtrees, :]
        
        pad_mask = (subtrees.sum(dim=-1) == 0)
        
        x = subtree_emb
        for layer in self.transformer_layers:
            x = layer(x, src_key_padding_mask=pad_mask)
            
        x = self.norm(x)
        pooled = torch.mean(x, dim=1)
        
        residual = pooled
        x = self.dropout(F.gelu(self.fc1(pooled)))
        x = x + residual
        
        x = self.dropout(F.gelu(self.fc2(x)))
        logits = self.fc3(x)
        
        return F.softmax(logits, dim=1)

def analyze_code_block(code_block: str, model: nn.Module, extractor: ASTSubtreeExtractor, device: torch.device) -> Tuple[int, torch.Tensor]:
    subtrees = extractor.extract_subtrees(code_block)
    subtrees_tensor = torch.from_numpy(subtrees).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(subtrees_tensor)
        probabilities = output[0]
        prediction = torch.argmax(output[0]).item()

    return prediction, probabilities

def analyze_file(file_path: str, model_path: str, node_types_path: str) -> None:
    with open(node_types_path, 'rb') as f:
        node_types = pickle.load(f)

    processor = ASTProcessor()
    extractor = ASTSubtreeExtractor(node_types=node_types)

    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)

    config = checkpoint['config']
    model = SANN(
        node_vocab_size=len(node_types),
        embedding_dim=int(config['embedding_dim']),
        num_heads=int(config['num_heads']),
        num_layers=int(config['num_layers']),
        dropout=config['dropout']
    )

    state_dict = checkpoint['model_state_dict']
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return

    code_blocks = processor.extract_code_blocks(code)
    
    logger.info(f"Analysis Results for {file_path}:")
    logger.info(f"Found {len(code_blocks)} valid code blocks")
    
    for i, block in enumerate(code_blocks, 1):
        prediction, probabilities = analyze_code_block(block, model, extractor, device)
        
        logger.info(f"Block {i}:")
        logger.info(f"First line: {block.split('\n')[0][:100]}...")
        logger.info(f"Prediction: {'AI-generated' if prediction == 0 else 'Human-written'}")
        logger.info(f"Confidence Scores:")
        logger.info(f"  AI-generated: {probabilities[0]*100:.2f}%")
        logger.info(f"  Human-written: {probabilities[1]*100:.2f}%")

def main():
    parser = argparse.ArgumentParser(description='Analyze Python files for AI vs Human authorship')
    parser.add_argument('--file', required=True, help='Path to the Python file to analyze')
    parser.add_argument('--model', required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--node-types', required=True, help='Path to the node types pickle file')
    
    args = parser.parse_args()
    analyze_file(args.file, args.model, args.node_types)

if __name__ == "__main__":
    main()




