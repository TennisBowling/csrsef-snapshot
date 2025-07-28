import ast
import os
os.environ['MPLCONFIGDIR'] = "/gscratch/scrubbed/enzovt/matplotlib_cache/"
import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.air.integrations.wandb import WandbLoggerCallback
import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple
import logging
from tqdm import tqdm
import torch.nn.functional as F
import math
from dataset import SANNDataset, ASTSubtreeExtractor, custom_collate_fn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



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
        avg_loss = total_loss / max(1, len(predictions) // labels.size(0))
    else:
        logger.warning("No batches were successfully processed in this epoch")
        accuracy = 0.0
        f1 = 0.0
        avg_loss = float('inf')
    
    return avg_loss, accuracy, f1

def train(config):
    from dataset import SANNDataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        ai_train = None
        human_train = None
        ai_test = None
        human_test = None
        train_dataset = SANNDataset(ai_train, human_train, extractor, is_test=False)
        test_dataset = SANNDataset(ai_test, human_test, extractor, is_test=True)
        
        #with open(dataset_cache, 'wb') as f:
        #    pickle.dump({
        #        'train': train_dataset,
        #        'test': test_dataset
        #    }, f)
            
        #with open("/gscratch/scrubbed/enzovt/node_types.pickle", 'wb') as f:
        #    pickle.dump(extractor.node_types, f)
        pass
    
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    train_dataloader = DataLoader(
        train_subset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
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
        node_vocab_size=len(extractor.node_types),
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
    
    
    best_val_f1 = 0
    patience = 5
    patience_counter = 0
    
    for epoch in range(config["num_epochs"]):
        model.train()
        train_loss, train_acc, train_f1 = run_epoch(
            model, train_dataloader, optimizer, criterion, device, is_training=True
        )
        
        model.eval()
        val_loss, val_acc, val_f1 = run_epoch(
            model, val_dataloader, None, criterion, device, is_training=False
        )
        
        test_loss, test_acc, test_f1 = run_epoch(
            model, test_dataloader, None, criterion, device, is_training=False
        )
        
        logger.info(f"Epoch {epoch+1}/{config['num_epochs']}")
        logger.info(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        logger.info(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        logger.info(f"Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}")


        tune.report(
            train_loss=train_loss,
            train_accuracy=train_acc,
            train_f1=train_f1,
            val_loss=val_loss,
            val_accuracy=val_acc,
            val_f1=val_f1,
            test_loss=test_loss,
            test_accuracy=test_acc,
            test_f1=test_f1,
            epoch=epoch
        )
        
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
        
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    
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


    return ai_df_filtered, human_df_filtered

def create_length_stratified_sample(df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    df = df.copy()

    df.loc[:, 'length'] = df['code'].str.len()
    df.loc[:, 'length_bin'] = pd.qcut(df['length'], q=10, labels=False)

    samples_per_bin = n_samples // 10
    stratified_sample = pd.DataFrame()

    for bin_idx in range(10):
        bin_df = df[df['length_bin'] == bin_idx]
        if len(bin_df) > samples_per_bin:
            sampled = bin_df.sample(n=samples_per_bin, random_state=42)
        else:
            sampled = bin_df
        stratified_sample = pd.concat([stratified_sample, sampled])

    return stratified_sample.drop(['length', 'length_bin'], axis=1)

def create_train_test_split(ai_df, human_df, test_split=0.1, random_state=123):
    ai_indices = np.random.RandomState(random_state).permutation(len(ai_df))
    human_indices = np.random.RandomState(random_state).permutation(len(human_df))
    
    ai_test_size = int(len(ai_df) * test_split)
    human_test_size = int(len(human_df) * test_split)
    
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
    ray.init()

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

    logger.info("Creating train/test split:")
    ai_train, ai_test, human_train, human_test = create_train_test_split(
        ai_df_stratified, 
        human_df_stratified
    )


    config = {
        "embedding_dim": tune.choice([16, 32, 64, 128, 256, 512]),
        "num_heads": tune.choice([2, 4, 8, 16]),
        "num_layers": tune.choice([2, 4, 6]),
        "dropout": tune.choice([0.1, 0.3, 0.5]),
        "batch_size": tune.choice([8, 16, 32, 64]),
        "lr": tune.loguniform(1e-7, 1e-5),
        "weight_decay": tune.loguniform(1e-4, 1e-1),
        "label_smoothing": tune.choice([0.0, 0.1, 0.3]),
        "grad_clip": tune.uniform(0.1, 3.0),
        "num_epochs": 4
    }

    logger.info("Model configuration:")
    for key, value in config.items():
        logger.info(f"{key}: {value}")

    logger.info("Starting model training:")
    trainable = tune.with_resources(train, { "gpu": 0.5 })

    scheduler = AsyncHyperBandScheduler(
        max_t=4,
        grace_period=1,
        reduction_factor=3,
        brackets=3
    )

    tuner = tune.Tuner(
        trainable,
        param_space=config,
        tune_config=tune.TuneConfig(num_samples=-1, metric="val_f1", mode="max", scheduler=scheduler, max_concurrent_trials=4),
        run_config=tune.RunConfig(name="sann", storage_path="/gscratch/scrubbed/enzovt/ray/", callbacks=[WandbLoggerCallback(project="sann", group="hyperparameter_search", log_config=True)])
    )
    
    logger.info("Starting tuning execution...")
    results = tuner.fit()
    logger.info("Hyperparameter optimization completed")

    best_result = results.get_best_result()
    if best_result and hasattr(best_result, 'metrics') and best_result.metrics:
        logger.info(f"Best trial config: {best_result.config}")
        logger.info(f"Best trial final macro F1: {best_result.metrics.get('macro_f1', 'N/A')}")
        logger.info(f"Best trial final accuracy: {best_result.metrics.get('accuracy', 'N/A')}")

        with open('best_config.pkl', 'wb') as f:
            pickle.dump(best_result.config, f)
    else:
        logger.warning("No successful trials found")

    ray.shutdown()

if __name__ == "__main__":
    main()