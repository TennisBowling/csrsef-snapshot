import ast
from collections import Counter
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
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 0:
    logger.info(f"Using {torch.cuda.device_count()} gpus")

def analyze_datasets(ai_df: pd.DataFrame, human_df: pd.DataFrame, 
                    min_length: int = 100,
                    max_length: int = 5000):
    
    logger.info("Original Dataset Statistics:")
    logger.info(f"AI samples: {len(ai_df)}")
    logger.info(f"Human samples: {len(human_df)}")
    
    ai_lengths = ai_df['code'].str.len()
    human_lengths = human_df['code'].str.len()
    
    logger.info("Original Code Length Statistics:")
    logger.info(f"AI code - Mean: {ai_lengths.mean():.1f}, Median: {ai_lengths.median():.1f}, "
               f"Std: {ai_lengths.std():.1f}")
    logger.info(f"Human code - Mean: {human_lengths.mean():.1f}, Median: {human_lengths.median():.1f}, "
               f"Std: {human_lengths.std():.1f}")
    
    ai_df_filtered = ai_df[ai_df['code'].str.len().between(min_length, max_length)]
    human_df_filtered = human_df[human_df['code'].str.len().between(min_length, max_length)]
    
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
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.hist(ai_lengths, bins=50, alpha=0.5, label='AI', density=True)
    plt.hist(human_lengths, bins=50, alpha=0.5, label='Human', density=True)
    plt.xlabel('Code Length (characters)')
    plt.ylabel('Density')
    plt.title('Original Code Length Distribution')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.hist(ai_lengths_filtered, bins=50, alpha=0.5, label='AI', density=True)
    plt.hist(human_lengths_filtered, bins=50, alpha=0.5, label='Human', density=True)
    plt.xlabel('Code Length (characters)')
    plt.ylabel('Density')
    plt.title('Filtered Code Length Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.close()
    
    return ai_df_filtered, human_df_filtered

def create_length_stratified_sample(df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    df = df.copy()
    
    df.loc[:, 'length'] = df['code'].str.len()
    df.loc[:, 'length_bin'] = pd.qcut(df['length'], q=10, labels=False)
    
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

class ASTSubtreeExtractor:
    def __init__(self, 
                 max_depth: int = 10,           # Codebases with more than 10 levels of nesting are very rare
                 max_subtree_size: int = 70,    # Probably has diminishing returns, minimizes noise from redundant subtrees
                 max_nodes: int = 150):         # Complex classes could have near 200 but we can go to 150
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
                return np.zeros((self.max_subtree_size, self.max_nodes), dtype=np.int64)    # Should probably check the number of zero'd subtrees we have
                
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
                 is_training: bool = True):
        
        self.ai_code = ai_code_df.reset_index(drop=True)
        self.human_code = human_code_df.reset_index(drop=True)
        
        self.extractor = extractor
        self.samples = []
        
        logger.info(f"Processing {'test' if not is_training else 'training'} AI code samples: {len(self.ai_code)}")
        for code in tqdm(self.ai_code['code']):
            subtrees = self.extractor.extract_subtrees(code)
            self.samples.append((subtrees, 0))  # 0 for AI
                
        logger.info(f"Processing {'test' if not is_training else 'training'} human code samples: {len(self.human_code)}")
        for code in tqdm(self.human_code['code']):
            subtrees = self.extractor.extract_subtrees(code)
            self.samples.append((subtrees, 1))  # 1 for human
        
        sample_shapes = [subtrees.shape for subtrees, _ in self.samples]
        logger.info(f"Sample shapes: {set(sample_shapes)}")

        non_zero_counts = [np.count_nonzero(subtree) for subtree, _ in self.samples]
        logger.info(f"Non-zero nodes per sample: Avg={np.mean(non_zero_counts):.1f} ± {np.std(non_zero_counts):.1f}")
        self._analyze_dataset_statistics()
    
    def _analyze_dataset_statistics(self):
        logger.info(f"Dataset Statistics:")
        
        # Sample counts
        ai_count = len([x for x in self.samples if x[1] == 0])
        human_count = len([x for x in self.samples if x[1] == 1])
        logger.info(f"AI samples: {ai_count}")
        logger.info(f"Human samples: {human_count}")
        
        # Analyze subtree statistics
        subtree_sizes = []
        node_counts = []
        non_zero_ratios = []
        
        for subtrees, _ in self.samples:
            subtree_sizes.append(len(subtrees))
            node_counts.append(np.count_nonzero(subtrees))
            non_zero_ratios.append(np.count_nonzero(subtrees) / subtrees.size)
        
        logger.info("Subtree Statistics:")
        logger.info(f"Average subtrees per sample: {np.mean(subtree_sizes):.1f} ± {np.std(subtree_sizes):.1f}")
        logger.info(f"Average nodes per sample: {np.mean(node_counts):.1f} ± {np.std(node_counts):.1f}")
        logger.info(f"Average non-zero ratio: {np.mean(non_zero_ratios):.3f} ± {np.std(non_zero_ratios):.3f}")
        
        # Log node type distribution
        node_types = []
        for subtrees, _ in self.samples:
            node_types.extend(subtrees[subtrees != 0].flatten())
        
        node_type_counts = Counter(node_types)
        logger.info("Top 10 node types:")
        for node_id, count in node_type_counts.most_common(10):
            node_type = [k for k, v in self.extractor.node_types.items() if v == node_id][0]
            logger.info(f"{node_type}: {count}")
        
        # Visualize node type distribution
        plt.figure(figsize=(12, 6))
        counts = [count for _, count in node_type_counts.most_common(20)]
        labels = [k for k, v in self.extractor.node_types.items() 
                 if v in [node_id for node_id, _ in node_type_counts.most_common(20)]]
        
        plt.bar(range(len(counts)), counts)
        plt.xticks(range(len(counts)), labels, rotation=45, ha='right')
        plt.xlabel('Node Type')
        plt.ylabel('Count')
        plt.title('Top 20 Node Type Distribution')
        plt.tight_layout()
        plt.close()
                
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
                 embedding_dim: int = 128,  # These might need some changing
                 num_heads: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        self.node_embedding = nn.Embedding(node_vocab_size, embedding_dim, padding_idx=0)
        nn.init.normal_(self.node_embedding.weight, mean=0, std=embedding_dim ** -0.5)
        
        # Positional encoding
        self.register_buffer(
            "pos_encoding",
            self._create_positional_encoding(200, embedding_dim)
        )
        
        # Transformer layers
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
        
        # Layer normalization
        self.norm = LayerNorm(embedding_dim)
        
        # CNN layers
        self.conv = nn.Sequential(
            # Input shape: [batch_size, 1, num_subtrees, embedding_dim]
            nn.Conv2d(
                in_channels=1,           # Treat as single-channel "image"
                out_channels=32,         # Learn 32 feature maps
                kernel_size=(3, 3),      # 3x3 kernel to capture local groups
                padding=1                # Preserve spatial dimensions
            ),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # Reduce spatial size by half
            
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)) # Collapse to [batch_size, 64, 1, 1]
        )

        # Adjust final classification layers to accept combined features
        self.fc1 = nn.Linear(embedding_dim + 64, embedding_dim + 64)  # Keep dimension
        self.fc2 = nn.Linear(embedding_dim + 64, embedding_dim // 2)
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
        activations = {}

        # Node Embeddings
        node_emb = self.node_embedding(subtrees)  # [B, S, N, E]
        activations['node_embeddings'] = node_emb

        # Subtree Embeddings
        subtree_emb = node_emb.mean(dim=2)  # [B, S, E]
        activations['subtree_embeddings'] = subtree_emb

        # Positional Encoding
        subtree_emb = subtree_emb + self.pos_encoding[:, :num_subtrees, :]

        # Transformer Layers
        pad_mask = (subtrees.sum(dim=-1) == 0)  # [B, S]
        x = subtree_emb
        attention_weights = []
        
        for i, layer in enumerate(self.transformer_layers):
            if return_attention:
                # Modified forward to capture attention weights
                x, attn_weights = layer.self_attn(
                    x, x, x,
                    key_padding_mask=pad_mask,
                    need_weights=True
                )
                attention_weights.append(attn_weights)
                x = layer(x)  # Continue through rest of layer
            else:
                x = layer(x, src_key_padding_mask=pad_mask)
                
            activations[f'transformer_layer_{i}'] = x

        # Global Pooling
        x = self.norm(x)
        transformer_pooled = torch.mean(x, dim=1)  # [B, E]
        activations['transformer_pooled'] = transformer_pooled

        # CNN Pathway
        # Reshape for CNN input: [B, 1, S, E] (treat as 1-channel image)
        cnn_input = subtree_emb.unsqueeze(1)  # Add channel dimension
        
        # Forward through CNN
        cnn_features = self.conv(cnn_input)  # [B, 64, 1, 1]
        cnn_features = cnn_features.squeeze()  # [B, 64]
        activations['cnn_features'] = cnn_features

        # Feature Fusion
        combined = torch.cat([transformer_pooled, cnn_features], dim=1)  # [B, E + 64]
        activations['combined_features'] = combined

        # Classification Head
        residual = combined  # [B, E + 64]
        x = self.dropout(F.gelu(self.fc1(combined)))  # [B, E + 64]
        x = x + residual  # Now compatible
        activations['fc1'] = x

        x = self.dropout(F.gelu(self.fc2(x)))  # [B, E//2]
        activations['fc2'] = x

        logits = self.fc3(x)  # [B, 2]
        activations['logits'] = logits

        if return_attention:
            return logits, {
                'activations': activations,
                'attention_weights': attention_weights,
                'cnn_features': cnn_features
            }
        return logits, {'activations': activations}

def train_epoch(model: nn.Module,
                dataloader: DataLoader,
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler.LRScheduler,
                device: torch.device,
                epoch: int,
                grad_clip: float = 1.0) -> Tuple[float, float]:
    
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch_idx, (subtrees, labels) in enumerate(progress_bar):
        global_step = epoch * len(dataloader) + batch_idx
        
        # Move to device
        subtrees = subtrees.to(device)
        labels = labels.to(device)
        
        # Forward pass with attention weights and activations
        optimizer.zero_grad()
        outputs, extras = model(subtrees, return_attention=True)
        loss = criterion(outputs, labels)
        
        if torch.isnan(loss):
            logger.error(f"NaN loss detected at batch {batch_idx}")
            continue
        
        # Backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        # Track metrics
        total_loss += loss.item()
        pred = outputs.argmax(dim=1)
        correct += pred.eq(labels).sum().item()
        total += len(labels)
        
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'acc': 100. * correct / total,
            'lr': optimizer.param_groups[0]['lr']
        })
    
    # End of epoch logging
    epoch_loss = total_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device, epoch: int):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (subtrees, labels) in enumerate(tqdm(dataloader, desc="Testing")):
            subtrees = subtrees.to(device)
            labels = labels.to(device)
            
            outputs, extras = model(subtrees, return_attention=True)
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
        # True Positives, False Positives, False Negatives, True Negatives
        tp = np.sum((all_preds == class_idx) & (all_labels == class_idx))
        fp = np.sum((all_preds == class_idx) & (all_labels != class_idx))
        fn = np.sum((all_preds != class_idx) & (all_labels == class_idx))
        tn = np.sum((all_preds != class_idx) & (all_labels != class_idx))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Store metrics
        metrics[classes[class_idx]] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity
        }
    
    overall_acc = np.mean(all_preds == all_labels)
    macro_f1 = (metrics["AI"]["f1"] + metrics["Human"]["f1"]) / 2
    
    # Log results to console
    logger.info(f"Evaluation Results for Epoch {epoch}:")
    logger.info(f"Overall Accuracy: {overall_acc:.2f}")
    logger.info(f"Macro-averaged F1 Score: {macro_f1:.2f}")
    logger.info(f"Average Loss: {total_loss / len(dataloader):.4f}\n")
    
    for class_name in ["AI", "Human"]:
        logger.info(f"{class_name} Code Metrics:")
        logger.info(f"  Precision: {metrics[class_name]['precision']:.2f}")
        logger.info(f"  Recall: {metrics[class_name]['recall']:.2f}")
        logger.info(f"  F1 Score: {metrics[class_name]['f1']:.2f}")
        logger.info(f"  Specificity: {metrics[class_name]['specificity']:.2f}")
    
    return total_loss / len(dataloader), overall_acc, macro_f1

def main():
    logger.info("Loading datasets...")
    ai_df = pd.read_parquet('ai.parquet', columns=["code"], memory_map=True)
    human_df = pd.read_parquet('human.parquet', columns=["code"], memory_map=True)

    ai_df_filtered, human_df_filtered = analyze_datasets(
        ai_df, 
        human_df,
        min_length=100,    # Filter very short samples
        max_length=5000    # Filter extremely long samples
    )

    n_samples = min(len(ai_df_filtered), len(human_df_filtered))
    
    ai_df_stratified = create_length_stratified_sample(ai_df_filtered, n_samples)
    human_df_stratified = create_length_stratified_sample(human_df_filtered, n_samples)
    
    # Set a fixed random seed for reproducibility
    np.random.seed(42)
    
    test_split = 0.1
    
    # Shuffle indices
    ai_indices = np.random.permutation(len(ai_df_stratified))
    human_indices = np.random.permutation(len(human_df_stratified))
    
    ai_test_size = int(len(ai_df_stratified) * test_split)
    human_test_size = int(len(human_df_stratified) * test_split)
    
    # Split the dataframes
    ai_train_df = ai_df_stratified.iloc[ai_indices[ai_test_size:]].reset_index(drop=True)
    ai_test_df = ai_df_stratified.iloc[ai_indices[:ai_test_size]].reset_index(drop=True)
    
    human_train_df = human_df_stratified.iloc[human_indices[human_test_size:]].reset_index(drop=True)
    human_test_df = human_df_stratified.iloc[human_indices[:human_test_size]].reset_index(drop=True)
    
    logger.info(f"Train set: {len(ai_train_df)} AI samples, {len(human_train_df)} human samples")
    logger.info(f"Test set: {len(ai_test_df)} AI samples, {len(human_test_df)} human samples")
    
    dataset_cache = "/gscratch/scrubbed/enzovt/dataset.pickle"

    extractor = ASTSubtreeExtractor()

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
        train_dataset = SANNDataset(ai_train_df, human_train_df, extractor, is_training=True)
        test_dataset = SANNDataset(ai_test_df, human_test_df, extractor, is_training=False)
        
        # Cache the datasets
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
        num_workers=os.cpu_count() or 0,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=os.cpu_count() or 0,
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
    
    # Training setup with learning rate scheduler
    ai_samples = len(train_dataset.ai_code)
    human_samples = len(train_dataset.human_code)
    total = ai_samples + human_samples
    class_weights = torch.tensor([total/human_samples, total/ai_samples]).to(device)  # Inverse frequency

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.1  # Reduced smoothing
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-4,  # Reduced learning rate
        weight_decay=0.01,  # L2 regularization
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    num_epochs = 50
    steps_per_epoch = len(train_dataloader)
    total_steps = num_epochs * steps_per_epoch
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-4,
        total_steps=total_steps,  # Use total steps for all epochs
        pct_start=0.3,
        anneal_strategy='cos',
        cycle_momentum=False
    )
    
    # Training loop
    logger.info("Starting training...")
    best_acc = 0

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss, train_acc = train_epoch(
            model, train_dataloader, criterion, optimizer, scheduler, device, epoch
        )
        test_loss, test_acc, macro_f1 = evaluate(model, test_dataloader, criterion, device, epoch)


        logger.info(f"Saving model for epoch {epoch + 1}")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_acc': test_acc,
            'macro_f1': macro_f1
        }, f'model_epoch_{epoch + 1}.pt')
        
        # Also save if it's the best (now considering both accuracy and F1)
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