import ast
import os
os.environ['MPLCONFIGDIR'] = "/gscratch/scrubbed/enzovt/matplotlib_cache/"

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
from torch.utils.tensorboard.writer import SummaryWriter


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
                 is_test: bool = False):

        self.ai_code = ai_code_df.reset_index(drop=True)
        self.human_code = human_code_df.reset_index(drop=True)
        
        self.extractor = extractor
        self.samples = []

        logger.info(f"Processing {'test' if is_test else 'training'} AI code samples: {len(self.ai_code)}")
        for code in tqdm(self.ai_code['code']):
            subtrees = self.extractor.extract_subtrees(code)
            self.samples.append((subtrees, 0))  # 0 for AI

        logger.info(f"Processing {'test' if is_test else 'training'} human code samples: {len(self.human_code)}")
        for code in tqdm(self.human_code['code']):
            subtrees = self.extractor.extract_subtrees(code)
            self.samples.append((subtrees, 1))  # 1 for human

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
                 embedding_dim: int = 128,  # Double from 128 to 256
                 num_heads: int = 4,    # Increase from 4 to 8 heads
                 num_layers: int = 2,   # Double layers from 2 to 4
                 dropout: float = 0.6): # Slightly increase dropout
        super().__init__()

        self.embedding_dim = embedding_dim

        node_vocab_size = max(1, node_vocab_size)
        
        # Node embedding with scaled initialization
        self.node_embedding = nn.Embedding(node_vocab_size, embedding_dim, padding_idx=0)
        nn.init.normal_(self.node_embedding.weight, mean=0, std=embedding_dim ** -0.5)

        # Positional encoding for longer sequences
        self.register_buffer(
            "pos_encoding",
            self._create_positional_encoding(250, embedding_dim)
        )

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=embedding_dim * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True  # Pre-norm architecture for stability
            ) for _ in range(num_layers)
        ])

        # Layer normalization
        self.norm = LayerNorm(embedding_dim)

        # Classification layers with residual connections
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

    def forward(self, subtrees: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, Dict]:
        batch_size, num_subtrees, num_nodes = subtrees.shape

        # Store intermediate activations
        activations = {}

        # This handles the potential CUDA indexing error
        max_vocab_idx = self.node_embedding.weight.size(0) - 1
        subtrees = torch.clamp(subtrees, 0, max_vocab_idx)

        # Node embeddings
        node_emb = self.node_embedding(subtrees)  # [B, S, N, E]
        activations['node_embeddings'] = node_emb

        # Average node embeddings for each subtree
        subtree_emb = node_emb.mean(dim=2)  # [B, S, E]
        activations['subtree_embeddings'] = subtree_emb

        pos_len = min(num_subtrees, self.pos_encoding.size(1))
        if pos_len < num_subtrees:
            logger.warning(f"Sequence length {num_subtrees} exceeds positional encoding length {self.pos_encoding.size(1)}")
            # Expand positional encoding if needed
            subtree_emb[:, :pos_len, :] = subtree_emb[:, :pos_len, :] + self.pos_encoding[:, :pos_len, :]
        else:
            subtree_emb = subtree_emb + self.pos_encoding[:, :num_subtrees, :]

        pad_mask = (subtrees.sum(dim=-1) == 0)

        x = subtree_emb
        attention_weights = []
        for i, layer in enumerate(self.transformer_layers):
            if return_attention:
                try:
                    x = layer.forward(x, src_key_padding_mask=pad_mask)
                    attn_output_weights = layer.self_attn.forward(x, x, x, need_weights=True)[1]
                    attention_weights.append(attn_output_weights)
                except Exception as e:
                    logger.error(f"Error in transformer layer {i}: {e}")
                    # Continue with current x value if there's an error
            else:
                try:
                    x = layer(x, src_key_padding_mask=pad_mask)
                except Exception as e:
                    logger.error(f"Error in transformer layer {i}: {e}")
                    # Continue with current x value
            
            activations[f'transformer_layer_{i}'] = x

        # Global average pooling
        x = self.norm(x)
        pooled = torch.mean(x, dim=1)
        activations['pooled'] = pooled

        # Classification with residual connections
        residual = pooled
        x = self.dropout(F.gelu(self.fc1(pooled)))
        x = x + residual
        activations['fc1'] = x

        x = self.dropout(F.gelu(self.fc2(x)))
        activations['fc2'] = x

        logits = self.fc3(x)
        activations['logits'] = logits

        if return_attention:
            return logits, {'activations': activations, 'attention_weights': attention_weights}
        return logits, {'activations': activations}
    
def calculate_macro_f1(predictions, labels):
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    f1_scores = []
    for class_idx in [0, 1]:
        # True positives, false positives, false negatives
        tp = np.sum((predictions == class_idx) & (labels == class_idx))
        fp = np.sum((predictions == class_idx) & (labels != class_idx))
        fn = np.sum((predictions != class_idx) & (labels == class_idx))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    
    return np.mean(f1_scores)

def run_epoch(model, dataloader, optimizer, criterion, device, is_training=True):
    total_loss = 0
    predictions = []
    labels_list = []
    
    with torch.set_grad_enabled(is_training):
        for subtrees, labels in tqdm(dataloader):
            try:
                subtrees = subtrees.to(device)
                labels = labels.to(device)
                
                outputs, _ = model(subtrees)
                loss = criterion(outputs, labels)
                
                if is_training:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                total_loss += loss.item()
                predictions.extend(outputs.argmax(dim=1).cpu().numpy())
                labels_list.extend(labels.cpu().numpy())
            except RuntimeError as e:
                logger.error(f"Error processing batch: {str(e)}")
                if "CUDA error" in str(e) or "device-side assert" in str(e):
                    logger.warning("CUDA error detected, skipping batch")
                    if is_training:
                        optimizer.zero_grad()  # Clear any partial gradients
                    continue
                else:
                    raise  # Re-raise the error if it's not a CUDA error
    
    # If we processed at least some batches
    if len(predictions) > 0:
        accuracy = np.mean(np.array(predictions) == np.array(labels_list))
        f1 = calculate_macro_f1(predictions, labels_list)
        avg_loss = total_loss / max(1, len(predictions) // labels.size(0))  # Avoid division by zero
    else:
        logger.warning("No batches were successfully processed in this epoch")
        accuracy = 0.0
        f1 = 0.0
        avg_loss = float('inf')
    
    return avg_loss, accuracy, f1

def train(train_dataset, test_dataset, num_node_types, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Split training data into train and validation
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    train_dataloader = DataLoader(
        train_subset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=custom_collate_fn
    )
    
    val_dataloader = DataLoader(
        val_subset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    model = SANN(
        node_vocab_size=num_node_types,
        embedding_dim=config["embedding_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        dropout=config["dropout"]
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )
    
    writer = SummaryWriter(log_dir='/gscratch/scrubbed/enzovt/runs/sann_training')
    
    best_val_f1 = 0
    patience = 5
    patience_counter = 0
    
    for epoch in range(config["num_epochs"]):
        # Training phase
        model.train()
        train_loss, train_acc, train_f1 = run_epoch(
            model, train_dataloader, optimizer, criterion, device, is_training=True
        )
        
        # Validation phase
        model.eval()
        val_loss, val_acc, val_f1 = run_epoch(
            model, val_dataloader, None, criterion, device, is_training=False
        )
        
        # Test phase
        test_loss, test_acc, test_f1 = run_epoch(
            model, test_dataloader, None, criterion, device, is_training=False
        )
        
        # Log metrics
        logger.info(f"Epoch {epoch+1}/{config['num_epochs']}")
        logger.info(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        logger.info(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        logger.info(f"Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}")
        
        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Accuracy/test', test_acc, epoch)
        writer.add_scalar('F1/train', train_f1, epoch)
        writer.add_scalar('F1/val', val_f1, epoch)
        writer.add_scalar('F1/test', test_f1, epoch)
        
        # Early stopping check
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "config": config
            }
            torch.save(checkpoint, "/gscratch/scrubbed/enzovt/best_model.pt")
            logger.info(f"Saved new best model with validation F1: {val_f1:.4f}")
        else:
            patience_counter += 1
            logger.info(f"No improvement for {patience_counter} epochs")
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    writer.close()
    
    logger.info("Loading best model for final evaluation")
    checkpoint = torch.load("/gscratch/scrubbed/enzovt/best_model.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    
    model.eval()
    test_loss, test_acc, test_f1 = run_epoch(
        model, test_dataloader, None, criterion, device, is_training=False
    )
    
    logger.info(f"Final test results - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}")
    
    return model, test_acc, test_f1

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        progress = tqdm(dataloader, desc="Evaluating")
        for batch_idx, (subtrees, labels) in enumerate(progress):
            subtrees = subtrees.to(device)
            labels = labels.to(device)

            outputs, _ = model(subtrees)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            pred = outputs.argmax(dim=1)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            progress.set_postfix({
                'loss': f'{total_loss/(batch_idx+1):.4f}'
            })

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = np.mean(all_preds == all_labels)
    
    f1_scores = []
    for class_idx in [0, 1]:
        tp = np.sum((all_preds == class_idx) & (all_labels == class_idx))
        fp = np.sum((all_preds == class_idx) & (all_labels != class_idx))
        fn = np.sum((all_preds != class_idx) & (all_labels == class_idx))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    
    macro_f1 = np.mean(f1_scores)

    return total_loss / len(dataloader), accuracy, macro_f1

def analyze_datasets(ai_df: pd.DataFrame, human_df: pd.DataFrame,
                    min_length: int = 100,
                    max_length: int = 5000):


    # Original statistics
    logger.info("Original Dataset Statistics:")
    logger.info(f"AI samples: {len(ai_df)}")
    logger.info(f"Human samples: {len(human_df)}")

    # Code length statistics before filtering
    ai_lengths = ai_df['code'].str.len()
    human_lengths = human_df['code'].str.len()

    logger.info("Original Code Length Statistics:")
    logger.info(f"AI code - Mean: {ai_lengths.mean():.1f}, Median: {ai_lengths.median():.1f}, "
               f"Std: {ai_lengths.std():.1f}")
    logger.info(f"Human code - Mean: {human_lengths.mean():.1f}, Median: {human_lengths.median():.1f}, "
               f"Std: {human_lengths.std():.1f}")

    ai_df_filtered = ai_df[ai_df['code'].str.len().between(min_length, max_length)]
    human_df_filtered = human_df[human_df['code'].str.len().between(min_length, max_length)]

    # Statistics after filtering
    ai_lengths_filtered = ai_df_filtered['code'].str.len()
    human_lengths_filtered = human_df_filtered['code'].str.len()

    logger.info(f"Filtered Dataset Statistics (length between {min_length} and {max_length} chars):")
    logger.info(f"AI samples: {len(ai_df_filtered)}")
    logger.info(f"Human samples: {len(human_df_filtered)}")
    logger.info("Filtered Code Length Statistics:")
    logger.info(f"AI code - Mean: {ai_lengths_filtered.mean():.1f}, Median: {ai_lengths_filtered.median():.1f}, "
               f"Std: {ai_lengths_filtered.std():.1f}")
    logger.info(f"Human code - Mean: {human_lengths_filtered.mean():.1f}, Median: {human_lengths_filtered.median():.1f}, "
               f"Std: {human_lengths_filtered.std():.1f}")


    return ai_df_filtered, human_df_filtered

def create_length_stratified_sample(df: pd.DataFrame, n_samples: int) -> pd.DataFrame:

    df = df.copy()

    df.loc[:, 'length'] = df['code'].str.len()
    df.loc[:, 'length_bin'] = pd.qcut(df['length'], q=10, labels=False)  # 10 quantile bins

    # Sample equally from each bin
    samples_per_bin = n_samples // 10
    stratified_sample = pd.DataFrame()

    for bin_idx in range(10):
        bin_df = df[df['length_bin'] == bin_idx]
        if len(bin_df) > samples_per_bin:
            sampled = bin_df.sample(n=samples_per_bin, random_state=42)
        else:
            sampled = bin_df  # Take all samples if bin has fewer than needed
        stratified_sample = pd.concat([stratified_sample, sampled])

    return stratified_sample.drop(['length', 'length_bin'], axis=1)

def create_train_test_split(ai_df, human_df, test_split=0.1, random_state=123):
    ai_indices = np.random.RandomState(random_state).permutation(len(ai_df))
    human_indices = np.random.RandomState(random_state).permutation(len(human_df))
    
    ai_test_size = int(len(ai_df) * test_split)
    human_test_size = int(len(human_df) * test_split)
    
    # Split indices
    ai_train_idx = ai_indices[ai_test_size:]
    ai_test_idx = ai_indices[:ai_test_size]
    human_train_idx = human_indices[human_test_size:]
    human_test_idx = human_indices[:human_test_size]
    
    ai_train = ai_df.iloc[ai_train_idx]
    ai_test = ai_df.iloc[ai_test_idx]
    human_train = human_df.iloc[human_train_idx]
    human_test = human_df.iloc[human_test_idx]
    
    return ai_train, ai_test, human_train, human_test

def main():
    logger.info("Starting data loading process:")
    logger.info("Loading AI dataset from parquet...")
    ai_df = pd.read_parquet('/gscratch/stf/lleibm/enzo/ai.parquet', columns=["code"], memory_map=True)
    logger.info("Loading human dataset from parquet...")
    human_df = pd.read_parquet('/gscratch/stf/lleibm/enzo/human.parquet', columns=["code"], memory_map=True)
    logger.info(f"Initial dataset sizes - AI: {len(ai_df):,} samples, Human: {len(human_df):,} samples")

    logger.info("Filtering and analyzing datasets:")
    ai_df_filtered, human_df_filtered = analyze_datasets(
        ai_df,
        human_df,
        min_length=100,
        max_length=5000
    )

    n_samples = min(len(ai_df_filtered), len(human_df_filtered))
    logger.info(f"Creating stratified samples with {n_samples:,} samples per class...")
    logger.info("Stratifying AI dataset...")
    ai_df_stratified = create_length_stratified_sample(ai_df_filtered, n_samples)
    logger.info("Stratifying human dataset...")
    human_df_stratified = create_length_stratified_sample(human_df_filtered, n_samples)
    logger.info("Stratification complete")

    # Split data into train and test sets
    logger.info("Creating train/test split:")
    ai_train, ai_test, human_train, human_test = create_train_test_split(
        ai_df_stratified, 
        human_df_stratified
    )

    logger.info("Preparing training and test datasets:")
    logger.info("Initializing AST subtree extractor...")
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
        train_dataset = SANNDataset(ai_train, human_train, extractor, is_test=False)
        test_dataset = SANNDataset(ai_test, human_test, extractor, is_test=True)
        
        with open(dataset_cache, 'wb') as f:
            pickle.dump({
                'train': train_dataset,
                'test': test_dataset
            }, f)
            
        with open("/gscratch/scrubbed/enzovt/node_types.pickle", 'wb') as f:
            pickle.dump(extractor.node_types, f)

    # Define manual hyperparameters
    config = {
        "embedding_dim": 128,         # Reduced from 128 to 64
        "num_heads": 8,              # Reduced from 8 to 4 
        "num_layers": 4,             # Reduced from 4 to 2
        "dropout": 0.3,
        "batch_size": 8,            # Reduced from 32 to 16
        "lr": 1e-6,                  # Increased from 1e-6 to 5e-6
        "weight_decay": 1e-3,
        "label_smoothing": 0.1,
        "num_epochs": 4
    }

    logger.info("Model configuration:")
    for key, value in config.items():
        logger.info(f"{key}: {value}")

    # Train the model
    logger.info("Starting model training:")
    model, test_acc, test_f1 = train(
        train_dataset, 
        test_dataset, 
        len(extractor.node_types),
        config
    )
    
    logger.info(f"Final results:")
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    logger.info(f"Test Macro F1: {test_f1:.4f}")
    
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "node_types": extractor.node_types,
        "test_results": {
            "accuracy": test_acc,
            "f1": test_f1
        }
    }, "/gscratch/scrubbed/enzovt/sann_final_model.pt")
    
    logger.info("Final model saved to /gscratch/scrubbed/enzovt/sann_final_model.pt")

if __name__ == "__main__":
    main()