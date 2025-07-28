import ast
import os
import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
import logging
from tqdm import tqdm
import torch.nn.functional as F
import math

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 0:
    logger.info(f"Using {torch.cuda.device_count()} GPUs!")

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

class SANNDataset(Dataset):
    def __init__(self, 
                 ai_code_df: pd.DataFrame,
                 human_code_df: pd.DataFrame,
                 extractor: ASTSubtreeExtractor,
                 test_split: float = 0.1,
                 is_test: bool = False):
        
        ai_test_size = int(len(ai_code_df) * test_split)
        human_test_size = int(len(human_code_df) * test_split)
        
        ai_indices = np.random.permutation(len(ai_code_df))
        human_indices = np.random.permutation(len(human_code_df))
        
        if is_test:
            ai_indices = ai_indices[:ai_test_size]
            human_indices = human_indices[:human_test_size]
        else:
            ai_indices = ai_indices[ai_test_size:]
            human_indices = human_indices[human_test_size:]
            
        self.ai_code = ai_code_df.iloc[ai_indices].reset_index(drop=True)
        self.human_code = human_code_df.iloc[human_indices].reset_index(drop=True)
        
        self.extractor = extractor
        self.samples = []
        
        logger.info("Processing AI code samples...")
        for code in tqdm(self.ai_code['code']):
            subtrees = self.extractor.extract_subtrees(code)
            self.samples.append((subtrees, 0))
                
        logger.info("Processing human code samples...")
        for code in tqdm(self.human_code['code']):
            subtrees = self.extractor.extract_subtrees(code)
            self.samples.append((subtrees, 1))
        
        sample_shapes = [subtrees.shape for subtrees, _ in self.samples]
        logger.info(f"Sample shapes: {set(sample_shapes)}")
                
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        subtrees, label = self.samples[idx]
        return torch.from_numpy(subtrees), label

def custom_collate_fn(batch):
    subtrees = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    return subtrees, labels

class SANN(nn.Module):
    def __init__(self,
                 node_vocab_size: int,
                 embedding_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dropout: float = 0.5):
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
        x = torch.mean(x, dim=1)
        
        residual = x
        x = self.dropout(F.gelu(self.fc1(x)))
        x = x + residual
        
        x = self.dropout(F.gelu(self.fc2(x)))
        logits = self.fc3(x)
        
        return logits

def train_epoch(model: nn.Module,
                dataloader: DataLoader,
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler.LRScheduler,
                device: torch.device,
                grad_clip: float = 1.0) -> Tuple[float, float]:
    
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch_idx, (subtrees, labels) in enumerate(progress_bar):
        subtrees = subtrees.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(subtrees)
        loss = criterion(outputs, labels)
        
        if torch.isnan(loss):
            logger.error(f"NaN loss detected at batch {batch_idx}")
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        pred = outputs.argmax(dim=1)
        correct += pred.eq(labels).sum().item()
        total += len(labels)
        
        progress_bar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'acc': 100. * correct / total,
            'lr': optimizer.param_groups[0]['lr']
        })
    
    return total_loss / len(dataloader), 100. * correct / total

def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for subtrees, labels in tqdm(dataloader, desc="Testing"):
            subtrees = subtrees.to(device)
            labels = labels.to(device)
            
            outputs = model(subtrees)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    metrics = {}
    classes = {0: "AI", 1: "Human"}
    
    for class_idx in [0, 1]:
        tp = np.sum((all_preds == class_idx) & (all_labels == class_idx))
        fp = np.sum((all_preds == class_idx) & (all_labels != class_idx))
        fn = np.sum((all_preds != class_idx) & (all_labels == class_idx))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[classes[class_idx]] = {
            "precision": precision * 100,
            "recall": recall * 100,
            "f1": f1 * 100
        }
    
    overall_acc = 100 * np.mean(all_preds == all_labels)
    
    macro_f1 = (metrics["AI"]["f1"] + metrics["Human"]["f1"]) / 2
    
    logger.info(f"Overall Accuracy: {overall_acc:.2f}%")
    logger.info(f"Macro-averaged F1 Score: {macro_f1:.2f}%\n")
    
    for class_name in ["AI", "Human"]:
        logger.info(f"{class_name} Code Metrics:")
        logger.info(f"  Precision: {metrics[class_name]['precision']:.2f}%")
        logger.info(f"  Recall: {metrics[class_name]['recall']:.2f}%")
        logger.info(f"  F1 Score: {metrics[class_name]['f1']:.2f}%")
    
    return total_loss / len(dataloader), overall_acc, macro_f1

def main():
    logger.info("Loading datasets...")
    ai_df = pd.read_parquet('ai.parquet')
    human_df = pd.read_parquet('human.parquet')
    
    extractor = ASTSubtreeExtractor()
    
    dataset_cache = "/gscratch/scrubbed/enzovt/dataset.pickle"
    if os.path.exists(dataset_cache):
        logger.info("Loading cached datasets...")
        with open(dataset_cache, 'rb') as f:
            cached_data = pickle.load(f)
            train_dataset = cached_data['train']
            test_dataset = cached_data['test']

        with open("/gscratch/scrubbed/enzovt/node_types.pickle", 'rb') as f:
            extractor.node_types = pickle.load(f)
    else:
        logger.info("Creating new datasets...")
        train_dataset = SANNDataset(ai_df, human_df, extractor, test_split=0.1, is_test=False)
        test_dataset = SANNDataset(ai_df, human_df, extractor, test_split=0.1, is_test=True)
        
        with open(dataset_cache, 'wb') as f:
            pickle.dump({
                'train': train_dataset,
                'test': test_dataset
            }, f)
    
    logger.info("Done")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=32,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=32,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    model = SANN(
        node_vocab_size=len(extractor.node_types),
    )

    logger.info(f"Node types: {extractor.node_types}")
    with open("/gscratch/scrubbed/enzovt/node_types.pickle", 'wb') as f:
        pickle.dump(extractor.node_types, f)
        logger.info("wrote node types")
    logger.info(f"vocab size is {len(extractor.node_types)}")
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.2,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    num_training_steps = len(train_dataloader) * 1
    num_warmup_steps = num_training_steps // 10
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-4,
        total_steps=num_training_steps,
        pct_start=0.1,
        anneal_strategy='cos',
        cycle_momentum=False
    )
    
    logger.info("Starting training...")
    best_acc = 0

    num_training_steps = len(train_dataloader)

    for epoch in range(1):
        logger.info(f"Epoch {epoch + 1}/1")
        train_loss, train_acc = train_epoch(
            model, train_dataloader, criterion, optimizer, scheduler, device
        )
        test_loss, test_acc, macro_f1 = evaluate(model, test_dataloader, criterion, device)
        
        logger.info(f"Saving model for epoch {epoch + 1}")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_acc': test_acc,
            'macro_f1': macro_f1
        }, f'model_epoch_{epoch + 1}.pt')
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'macro_f1': macro_f1
            }, 'best_model.pt')

if __name__ == "__main__":
    main()