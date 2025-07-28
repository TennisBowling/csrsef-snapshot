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
from torch.utils.tensorboard.writer import SummaryWriter
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import io
from PIL import Image
import torchvision

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

writer = SummaryWriter('runs/sann_training')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 0:
    logger.info(f"Using {torch.cuda.device_count()} gpus")

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    plt.close()
    return torchvision.transforms.ToTensor()(image)

def log_gradients_and_weights(model, step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            writer.add_histogram(f'gradients/{name}', param.grad.data, step)
            writer.add_histogram(f'weights/{name}', param.data, step)
            writer.add_scalar(f'gradient_norm/{name}', param.grad.norm().item(), step)
            writer.add_scalar(f'weight_norm/{name}', param.norm().item(), step)

def log_layer_activations(activations, step, layer_name):
    if activations is not None:
        writer.add_histogram(f'activations/{layer_name}', activations, step)
        writer.add_scalar(f'activations/{layer_name}_mean', activations.mean().item(), step)
        writer.add_scalar(f'activations/{layer_name}_std', activations.std().item(), step)

def log_attention_weights(attention_weights, step, layer_idx):
    if len(attention_weights.shape) > 2:
        attention_weights = attention_weights.mean(dim=0)
    elif len(attention_weights.shape) == 1:
        attention_weights = attention_weights.unsqueeze(-1)

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(attention_weights.cpu().numpy(), cmap='viridis')
    plt.title(f'Attention Weights - Layer {layer_idx}')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    plt.close()
    writer.add_image(f'attention_weights/layer_{layer_idx}', torchvision.transforms.ToTensor()(image), step)

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

        # For test set, we want equal numbers of AI and human samples
        if is_test:
            min_test_size = min(
                int(len(ai_code_df) * test_split),
                int(len(human_code_df) * test_split)
            )

            # Random permutation for both datasets
            ai_indices = np.random.permutation(len(ai_code_df))
            human_indices = np.random.permutation(len(human_code_df))

            # Take equal numbers of samples for test set
            ai_indices = ai_indices[:min_test_size]
            human_indices = human_indices[:min_test_size]

        else:
            # For training set, we keep original proportions
            ai_test_size = int(len(ai_code_df) * test_split)
            human_test_size = int(len(human_code_df) * test_split)

            # Random permutation for both datasets
            ai_indices = np.random.permutation(len(ai_code_df))
            human_indices = np.random.permutation(len(human_code_df))

            ai_indices = ai_indices[ai_test_size:]
            human_indices = human_indices[human_test_size:]

        self.ai_code = ai_code_df.iloc[ai_indices].reset_index(drop=True)
        self.human_code = human_code_df.iloc[human_indices].reset_index(drop=True)

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

        # Node embedding with scaled initialization
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

        # Node embeddings
        node_emb = self.node_embedding(subtrees)  # [B, S, N, E]
        activations['node_embeddings'] = node_emb

        # Average node embeddings for each subtree
        subtree_emb = node_emb.mean(dim=2)  # [B, S, E]
        activations['subtree_embeddings'] = subtree_emb

        subtree_emb = subtree_emb + self.pos_encoding[:, :num_subtrees, :]

        pad_mask = (subtrees.sum(dim=-1) == 0)

        x = subtree_emb
        attention_weights = []
        for i, layer in enumerate(self.transformer_layers):
            if return_attention:
                x = layer.forward(x, src_key_padding_mask=pad_mask)
                attn_output_weights = layer.self_attn.forward(x, x, x, need_weights=True)[1]
                attention_weights.append(attn_output_weights)
            else:
                x = layer(x, src_key_padding_mask=pad_mask)
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

        # Log gradients and weights periodically
        if batch_idx % 50000 == 0:
            log_gradients_and_weights(model, global_step)

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

        # Log batch metrics to TensorBoard
        writer.add_scalar('train/batch_loss', loss.item(), global_step)
        writer.add_scalar('train/batch_accuracy', 100. * pred.eq(labels).sum().item() / len(labels), global_step)
        writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], global_step)

        # Log layer activations periodically
        if batch_idx % 50000 == 0:
            for name, activation in extras['activations'].items():
                if isinstance(activation, torch.Tensor):
                    log_layer_activations(activation.detach(), global_step, name)

            # Log attention weights
            if 'attention_weights' in extras:
                for layer_idx, attn_weights in enumerate(extras['attention_weights']):
                    attn = attn_weights[0]  # Get first batch
                    if len(attn.shape) > 2:
                        attn = attn.mean(dim=0)  # Average across heads if multiple
                    log_attention_weights(attn.detach(), global_step, layer_idx)

        progress_bar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'acc': 100. * correct / total,
            'lr': optimizer.param_groups[0]['lr']
        })

    # End of epoch logging
    epoch_loss = total_loss / len(dataloader)
    epoch_acc = 100. * correct / total

    # Log epoch metrics
    writer.add_scalar('train/epoch_loss', epoch_loss, epoch)
    writer.add_scalar('train/epoch_accuracy', epoch_acc, epoch)

    cm = confusion_matrix(all_labels, all_preds)
    cm_image = plot_confusion_matrix(cm, ['AI', 'Human'])
    writer.add_image('train/confusion_matrix', cm_image, epoch)

    return epoch_loss, epoch_acc

def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device, epoch: int):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    attention_maps = []
    layer_activations = {}
    logged_attention_this_epoch = False  # Flag to track attention logging

    with torch.no_grad():
        for batch_idx, (subtrees, labels) in enumerate(tqdm(dataloader, desc="Testing")):
            subtrees = subtrees.to(device)
            labels = labels.to(device)

            outputs, extras = model(subtrees, return_attention=True)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            pred = outputs.argmax(dim=1)

            # Store predictions and labels for metric calculation
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Log batch metrics to TensorBoard
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('val/batch_loss', loss.item(), global_step)
            writer.add_scalar('val/batch_accuracy', 100. * pred.eq(labels).sum().item() / len(labels), global_step)

            # Store attention weights and activations periodically
            if batch_idx % 50000 == 0 and not logged_attention_this_epoch:  # Log attention only for the first such batch
                if 'attention_weights' in extras:
                    attention_maps.extend([
                        attn[0, 0].cpu().numpy()  # First head of first sample in batch
                        for attn in extras['attention_weights']
                    ])
                    logged_attention_this_epoch = True # Set the flag to True for the rest of the epoch

                # Log layer activations
                for name, activation in extras['activations'].items():
                    if isinstance(activation, torch.Tensor):
                        if name not in layer_activations:
                            layer_activations[name] = []
                        layer_activations[name].append(activation.mean().cpu().item())
                        log_layer_activations(activation.detach(), global_step, f'val_{name}')

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

        # Log class metrics to TensorBoard
        writer.add_scalar(f'val/{classes[class_idx]}_precision', precision, epoch)
        writer.add_scalar(f'val/{classes[class_idx]}_recall', recall, epoch)
        writer.add_scalar(f'val/{classes[class_idx]}_f1', f1, epoch)
        writer.add_scalar(f'val/{classes[class_idx]}_specificity', specificity, epoch)

    overall_acc = np.mean(all_preds == all_labels)
    macro_f1 = (metrics["AI"]["f1"] + metrics["Human"]["f1"]) / 2

    writer.add_scalar('val/epoch_loss', total_loss / len(dataloader), epoch)
    writer.add_scalar('val/epoch_accuracy', overall_acc, epoch)
    writer.add_scalar('val/macro_f1', macro_f1, epoch)

    cm = confusion_matrix(all_labels, all_preds)
    cm_image = plot_confusion_matrix(cm, ['AI', 'Human'])
    writer.add_image('val/confusion_matrix', cm_image, epoch)

    # Log attention maps
    if attention_maps:
        fig, axes = plt.subplots(1, len(attention_maps), figsize=(5*len(attention_maps), 4))
        if len(attention_maps) == 1:
            axes = [axes]
        for idx, attn_map in enumerate(attention_maps):
            # Reshape attention map if necessary
            if len(attn_map.shape) == 1:
                attn_map = attn_map.reshape(-1, 1)
            sns.heatmap(attn_map, ax=axes[idx], cmap='viridis')
            axes[idx].set_title(f'Layer {idx} Attention')
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        writer.add_image('val/attention_maps', torchvision.transforms.ToTensor()(image), epoch)
        plt.close()

    # Log layer activation distributions
    for name, values in layer_activations.items():
        writer.add_histogram(f'val/activations/{name}', np.array(values), epoch)

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

    # Visualize filtered code length distributions
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

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    writer.add_image('dataset/length_distribution',
                     torchvision.transforms.ToTensor()(Image.open(buf)), 0)
    plt.close()

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

    logger.info("Stratified Sample Statistics:")
    logger.info(f"AI samples: {len(ai_df_stratified)}")
    logger.info(f"Human samples: {len(human_df_stratified)}")

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
        train_dataset = SANNDataset(ai_df_stratified, human_df_stratified, extractor, test_split=0.1, is_test=False)
        test_dataset = SANNDataset(ai_df_stratified, human_df_stratified, extractor, test_split=0.1, is_test=True)

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
        num_workers=40,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=40,
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
    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,  # Reduced learning rate
        weight_decay=0.3,  # L2 regularization
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Cosine learning rate scheduler with warmup
    num_training_steps = len(train_dataloader) * 1  # 10 epochs
    num_warmup_steps = num_training_steps // 10

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-5,
        total_steps=num_training_steps,
        pct_start=0.35,  # Warmup for 30% of training
        anneal_strategy='cos',
        cycle_momentum=False
    )

    # Training loop
    logger.info("Starting training...")
    best_acc = 0

    num_training_steps = len(train_dataloader)


    for epoch in range(4):
        logger.info(f"Epoch {epoch + 1}/1")
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