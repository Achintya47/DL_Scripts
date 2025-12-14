# Hierarchical Transformer encoder for variable-length 2D task systems (PyTorch).
# - Supports arbitrary number of tasks (4,8,...,N)
# - Each task is 2D
# - Max input chunk size = 20
#
# Added torchsummary summaries of each major component + full model.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import pandas as pd
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

# -------------------- Utilities --------------------
def make_padding_mask(lengths, max_len=None):
    if max_len is None:
        max_len = int(torch.max(lengths).item())
    B = lengths.size(0)
    idxs = torch.arange(max_len, device=lengths.device)
    mask = idxs.unsqueeze(0).expand(B, max_len) >= lengths.unsqueeze(1)
    return mask  # True for padded positions

def chunk_tensor(x, chunk_size):
    B, L, D = x.shape
    n_chunks = (L + chunk_size - 1) // chunk_size
    padded_len = n_chunks * chunk_size
    if padded_len > L:
        pad = torch.zeros(B, padded_len - L, D, device=x.device, dtype=x.dtype)
        x = torch.cat([x, pad], dim=1)
    x = x.view(B, n_chunks, chunk_size, D)
    return x, n_chunks

def chunk_mask(lengths, chunk_size, n_chunks=None):
    B = lengths.size(0)
    if n_chunks is None:
        n_chunks = (int(torch.max(lengths).item()) + chunk_size - 1) // chunk_size
    max_len = n_chunks * chunk_size
    idxs = torch.arange(max_len, device=lengths.device)
    base_mask = idxs.unsqueeze(0).expand(B, max_len) >= lengths.unsqueeze(1)
    base_mask = base_mask.view(B, n_chunks, chunk_size)
    return base_mask

# -------------------- Per-task encoder --------------------
class PerTaskEncoder(nn.Module):
    def __init__(self, in_dim=2, emb_dim=64, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
        )
    def forward(self, x):
        return self.net(x)  # (B, L, emb_dim)

# -------------------- Local Transformer Block --------------------
class LocalTransformerBlock(nn.Module):
    def __init__(self, d_model=64, n_heads=4, mlp_dim=128, dropout=0.1, max_rel=16):
        super().__init__()
        assert d_model % n_heads == 0
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.max_rel = max_rel
        self.rel_bias = nn.Parameter(torch.zeros(2*max_rel + 1, n_heads))
        nn.init.normal_(self.rel_bias, std=1e-2)

    def forward(self, x, key_padding_mask=None):
        B, Lw, D = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, Lw, self.n_heads, self.d_head).transpose(1,2) for t in qkv]
        scores = torch.matmul(q, k.transpose(-2,-1)) * self.scale
        idx = torch.arange(Lw, device=x.device)
        rel_pos = idx[None,:] - idx[:,None]
        rel_clip = torch.clamp(rel_pos, -self.max_rel, self.max_rel) + self.max_rel
        bias = self.rel_bias[rel_clip].permute(2,0,1)
        scores = scores + bias.unsqueeze(0)
        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :].to(torch.bool)
            scores = scores.masked_fill(mask, float("-1e9"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1,2).contiguous().view(B, Lw, D)
        out = self.out(out)
        x = x + out
        x = self.norm1(x)
        m = self.mlp(x)
        x = x + m
        x = self.norm2(x)
        return x

# -------------------- Window pooling --------------------
class WindowPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, d_model))
        self.scale = d_model ** -0.5
    def forward(self, x, key_padding_mask=None):
        B, Lw, D = x.shape
        q = self.q.unsqueeze(0).expand(B, -1, -1)
        scores = torch.matmul(q, x.transpose(-2,-1)) * self.scale
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1), float("-1e9"))
        attn = torch.softmax(scores, dim=-1)
        pooled = torch.matmul(attn, x)
        return pooled.squeeze(1)

# -------------------- Global Transformer --------------------
class GlobalTransformer(nn.Module):
    def __init__(self, n_layers=2, d_model=64, n_heads=4, mlp_dim=128, dropout=0.1, max_rel=32, max_len=256):
        super().__init__()
        self.layers = nn.ModuleList([LocalTransformerBlock(d_model, n_heads, mlp_dim, dropout, max_rel) for _ in range(n_layers)])
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.max_len = max_len
    def forward(self, x, key_padding_mask=None):
        B, n_chunks, D = x.shape
        if n_chunks > self.max_len:
            raise ValueError("n_chunks > max_len of global encoder")
        pos = torch.arange(n_chunks, device=x.device)[None,:].expand(B, n_chunks)
        x = x + self.pos_emb(pos)
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return x

# -------------------- Full hierarchical model --------------------
class HierarchicalSchedulabilityModel(nn.Module):
    def __init__(self, task_in_dim=2, emb_dim=64, chunk_size=20,
                 local_layers=1, local_heads=4, global_layers=2, global_heads=4,
                 dropout=0.1, use_domain_feats=True, domain_feat_dim=1):
        super().__init__()
        self.per_task = PerTaskEncoder(in_dim=task_in_dim, emb_dim=emb_dim, hidden_dim=emb_dim*2, dropout=dropout)
        self.chunk_size = chunk_size
        self.local_block = LocalTransformerBlock(d_model=emb_dim, n_heads=local_heads, mlp_dim=emb_dim*2, dropout=dropout, max_rel=chunk_size//2)
        self.window_pool = WindowPooling(emb_dim)
        self.global_enc = GlobalTransformer(n_layers=global_layers, d_model=emb_dim, n_heads=global_heads, mlp_dim=emb_dim*2, dropout=dropout, max_rel=128, max_len=1024)
        self.global_pool_q = nn.Parameter(torch.randn(1, emb_dim))
        self.use_domain_feats = use_domain_feats
        self.domain_feat_dim = domain_feat_dim
        cls_in = emb_dim + (domain_feat_dim if use_domain_feats else 0)
        self.classifier = nn.Sequential(
            nn.Linear(cls_in, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, tasks, lengths, domain_feats=None):
        B, L, _ = tasks.shape
        device = tasks.device
        x = self.per_task(tasks)
        chunked, n_chunks = chunk_tensor(x, self.chunk_size)
        c_mask = chunk_mask(lengths, self.chunk_size, n_chunks)
        Bc = B * n_chunks
        chunked_flat = chunked.view(Bc, self.chunk_size, -1)
        chunk_mask_flat = c_mask.view(Bc, self.chunk_size)
        y = self.local_block(chunked_flat, key_padding_mask=chunk_mask_flat)
        pooled_windows = self.window_pool(y, key_padding_mask=chunk_mask_flat)
        pooled_windows = pooled_windows.view(B, n_chunks, -1)
        window_pad = c_mask.all(dim=-1)
        g = self.global_enc(pooled_windows, key_padding_mask=window_pad)
        q = self.global_pool_q.unsqueeze(0).expand(B, -1, -1)
        scores = torch.matmul(q, g.transpose(-2,-1)).squeeze(1)
        scores = scores.masked_fill(window_pad, float("-1e9"))
        weights = torch.softmax(scores, dim=-1)
        global_pooled = torch.matmul(weights.unsqueeze(1), g).squeeze(1)
        if self.use_domain_feats:
            if domain_feats is None:
                domain_feats = torch.zeros(B, self.domain_feat_dim, device=device)
            out = torch.cat([global_pooled, domain_feats], dim=-1)
        else:
            out = global_pooled
        logits = self.classifier(out).squeeze(-1)
        probs = torch.sigmoid(logits)
        return probs, logits


# -------------------- Dataset --------------------
class MultiSheetTaskDataset(Dataset):
    def __init__(self, excel_files_info):
        """
        excel_files_info: List of dictionaries with file info
        e.g., [
            {'file': 'n=4.xlsx', 'n_tasks': 4, 'sheets': ['n=4 u=0.5', 'n=4 u=0.55', ...]},
            {'file': 'n=8.xlsx', 'n_tasks': 8, 'sheets': ['n=8 u=0.5', 'n=8 u=0.55', ...]}
        ]
        """
        self.all_data = []
        
        def clean_cell(x):
            if isinstance(x, str):
                x = x.replace('"', '')
                x = x.replace("\n", "")
                x = x.replace("_x000D_", "")
                x = x.strip()
                return x
            return x
        
        for file_info in excel_files_info:
            excel_file = file_info['file']
            n_tasks = file_info['n_tasks']
            sheet_names = file_info['sheets']
            
            print(f"Loading {excel_file} with {n_tasks} tasks...")
            
            for sheet_name in sheet_names:
                try:
                    # Load specific sheet
                    df = pd.read_excel(excel_file, sheet_name=sheet_name)
                    
                    # Clean column names
                    df.columns = df.columns.str.strip()
                    
                    # Clean cells
                    df = df.map(clean_cell)
                    
                    # Convert to numeric
                    for col in df.columns:
                        try:
                            df[col] = pd.to_numeric(df[col])
                        except (ValueError, TypeError):
                            pass
                    
                    # Extract task features (C_i, P_i pairs)
                    task_cols = [f"C_{i}" for i in range(1, n_tasks + 1)] + [f"P_{i}" for i in range(1, n_tasks + 1)]
                    task_features = df[task_cols].astype(float).values
                    
                    # Calculate utilization
                    utilization = []
                    for i in range(len(df)):
                        util = sum(df.iloc[i][f"C_{j}"] / df.iloc[i][f"P_{j}"] for j in range(1, n_tasks + 1))
                        utilization.append(util)
                    
                    # Extract utilization from sheet name (e.g., "n=4 u=0.5" -> 0.5)
                    target_utilization = float(sheet_name.split('u=')[1])
                    
                    # Labels (EDF only)
                    labels = df["EDF"].astype(float).values
                    
                    # Store each sample
                    for i in range(len(task_features)):
                        sample = {
                            'task_features': task_features[i],
                            'n_tasks': n_tasks,
                            'utilization': utilization[i],
                            'target_utilization': target_utilization,
                            'label': labels[i],
                            'sheet': sheet_name
                        }
                        self.all_data.append(sample)
                    
                    print(f"  - Loaded {len(df)} samples from sheet '{sheet_name}'")
                    
                except Exception as e:
                    print(f"  - Error loading sheet '{sheet_name}': {e}")
        
        print(f"Total samples loaded: {len(self.all_data)}")
        
        # Create statistics
        task_counts = {}
        for sample in self.all_data:
            n = sample['n_tasks']
            task_counts[n] = task_counts.get(n, 0) + 1
        
        print("Task distribution:", task_counts)
    
    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, idx):
        sample = self.all_data[idx]
        
        # Reshape task features: (C1, C2, ..., P1, P2, ...) -> [(C1,P1), (C2,P2), ...]
        n_tasks = sample['n_tasks']
        raw_features = sample['task_features']
        
        # Split into C and P arrays
        c_values = raw_features[:n_tasks]
        p_values = raw_features[n_tasks:]
        
        # Create task sequence: [[C1,P1], [C2,P2], ...]
        task_feats = torch.tensor([[c_values[i], p_values[i]] for i in range(n_tasks)], dtype=torch.float32)
        
        # Domain features: [actual_utilization, target_utilization, n_tasks]
        domain_feats = torch.tensor([
            sample['utilization'],
            sample['target_utilization'],
            sample['n_tasks']
        ], dtype=torch.float32)
        
        # Label
        label = torch.tensor(sample['label'], dtype=torch.float32)
        
        # Length
        length = torch.tensor(n_tasks, dtype=torch.long)
        
        return task_feats, length, domain_feats, label

# Legacy single-sheet dataset (kept for compatibility)
class TaskDataset(Dataset):
    def __init__(self, excel_file, sheet_name=None):
        # Load Excel
        if sheet_name:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
        else:
            df = pd.read_excel(excel_file)

        # Clean column names (remove leading/trailing spaces)
        df.columns = df.columns.str.strip()

        # Clean all string-like cells: remove quotes, \n, _x000D_, etc.
        def clean_cell(x):
            if isinstance(x, str):
                x = x.replace('"', '')         # remove quotes
                x = x.replace("\n", "")        # remove newlines
                x = x.replace("_x000D_", "")   # remove Excel carriage return
                x = x.strip()
                return x
            return x

        # Use map instead of deprecated applymap
        df = df.map(clean_cell)

        # Convert everything possible to numeric (catch exceptions explicitly)
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass  # Keep original values if conversion fails

        # Task-level features (C_i, P_i pairs for each task set)
        self.task_features = df[[f"C_{i}" for i in range(1, 5)] +
                                [f"P_{i}" for i in range(1, 5)]].astype(float).values

        # Calculate utilization as domain feature
        utilization = []
        for i in range(len(df)):
            util = sum(df.iloc[i][f"C_{j}"] / df.iloc[i][f"P_{j}"] for j in range(1, 5))
            utilization.append(util)
        
        self.domain_features = np.array(utilization).reshape(-1, 1)

        # Labels (EDF only - single output)
        self.labels = df["EDF"].astype(float).values

    def __len__(self):
        return len(self.task_features)

    def __getitem__(self, idx):
        # Take row -> (C1..C4, P1..P4)
        row = self.task_features[idx]

        # Reshape into sequence of 4 tasks, each with [C, P]
        task_feats = torch.tensor(row.reshape(4, 2), dtype=torch.float32)

        # Domain-level features (utilization)
        domain_feats = torch.tensor(self.domain_features[idx], dtype=torch.float32)

        # Labels (EDF only)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        length = torch.tensor(4, dtype=torch.long)  # 4 tasks

        return task_feats, length, domain_feats, label


def collate_fn_variable(batch):
    """Collate function for variable-length sequences"""
    tasks, lengths, domain_feats, labels = zip(*batch)
    
    # Find maximum sequence length in this batch
    max_len = max(len(task_seq) for task_seq in tasks)
    batch_size = len(tasks)
    
    # Pad sequences to max length
    padded_tasks = torch.zeros(batch_size, max_len, 2)  # 2 for [C, P]
    actual_lengths = torch.zeros(batch_size, dtype=torch.long)
    
    for i, task_seq in enumerate(tasks):
        seq_len = len(task_seq)
        padded_tasks[i, :seq_len, :] = task_seq
        actual_lengths[i] = seq_len
    
    domain_feats = torch.stack(domain_feats, dim=0)
    labels = torch.stack(labels, dim=0)
    
    return padded_tasks, actual_lengths, domain_feats, labels


# -------------------- Training Loop --------------------
def train_model(model, train_loader, val_loader, device, epochs=20, lr=1e-3):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_correct, train_total = 0, 0
        
        for tasks, lengths, domain_feats, labels in train_loader:
            tasks, lengths, domain_feats, labels = (
                tasks.to(device), lengths.to(device),
                domain_feats.to(device), labels.to(device)
            )
            optimizer.zero_grad()
            probs, _ = model(tasks, lengths, domain_feats)
            loss = criterion(probs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * tasks.size(0)
            
            # Training accuracy
            preds = (probs > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.numel()
            
        train_losses.append(running_loss / len(train_loader.dataset))
        train_accs.append(train_correct / train_total)

        # Validation
        model.eval()
        val_loss = 0.0
        correct, total = 0, 0
        all_probs, all_labels = [], []
        
        with torch.no_grad():
            for tasks, lengths, domain_feats, labels in val_loader:
                tasks, lengths, domain_feats, labels = (
                    tasks.to(device), lengths.to(device),
                    domain_feats.to(device), labels.to(device)
                )
                probs, _ = model(tasks, lengths, domain_feats)
                loss = criterion(probs, labels)
                val_loss += loss.item() * tasks.size(0)

                preds = (probs > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.numel()
                
                # Store for detailed analysis
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_losses.append(val_loss / len(val_loader.dataset))
        val_acc = correct / total
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_losses[-1]:.4f} | Train Acc: {train_accs[-1]:.4f} | "
              f"Val Loss: {val_losses[-1]:.4f} | Val Acc: {val_acc:.4f}")

    # Comprehensive visualizations
    plot_training_results(train_losses, val_losses, train_accs, val_accs, all_probs, all_labels)
    
    return train_losses, val_losses, train_accs, val_accs

def plot_training_results(train_losses, val_losses, train_accs, val_accs, all_probs, all_labels):
    """Create comprehensive training visualization plots"""
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Loss curves
    plt.subplot(2, 4, 1)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Accuracy curves
    plt.subplot(2, 4, 2)
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0.98, 1.002])  # Zoom in on high accuracy range
    
    # 3. Loss vs Accuracy correlation
    plt.subplot(2, 4, 3)
    plt.scatter(val_losses, val_accs, c=epochs, cmap='viridis', s=50, alpha=0.7)
    plt.colorbar(label='Epoch')
    plt.title('Validation Loss vs Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Validation Loss')
    plt.ylabel('Validation Accuracy')
    plt.grid(True, alpha=0.3)
    
    # 4. Prediction distribution
    plt.subplot(2, 4, 4)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Separate by true labels
    schedulable_probs = all_probs[all_labels == 1]
    non_schedulable_probs = all_probs[all_labels == 0]
    
    plt.hist(schedulable_probs, bins=30, alpha=0.7, label=f'Schedulable (n={len(schedulable_probs)})', 
             color='green', density=True)
    plt.hist(non_schedulable_probs, bins=30, alpha=0.7, label=f'Non-schedulable (n={len(non_schedulable_probs)})', 
             color='red', density=True)
    plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Decision Threshold')
    plt.title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Learning rate analysis (smoothed)
    plt.subplot(2, 4, 5)
    # Calculate moving averages for smoother curves
    window = 5
    if len(train_losses) >= window:
        train_smooth = np.convolve(train_losses, np.ones(window)/window, mode='valid')
        val_smooth = np.convolve(val_losses, np.ones(window)/window, mode='valid')
        smooth_epochs = range(window, len(train_losses) + 1)
        plt.plot(smooth_epochs, train_smooth, 'b-', label='Train Loss (Smoothed)', linewidth=2)
        plt.plot(smooth_epochs, val_smooth, 'r-', label='Val Loss (Smoothed)', linewidth=2)
    plt.title('Smoothed Loss Curves', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Convergence analysis
    plt.subplot(2, 4, 6)
    loss_diff = np.abs(np.array(train_losses) - np.array(val_losses))
    plt.plot(epochs, loss_diff, 'g-', linewidth=2)
    plt.title('Train-Val Loss Gap', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('|Train Loss - Val Loss|')
    plt.grid(True, alpha=0.3)
    
    # 7. Performance metrics summary
    plt.subplot(2, 4, 7)
    final_metrics = {
        'Final Train Acc': train_accs[-1],
        'Final Val Acc': val_accs[-1],
        'Best Val Acc': max(val_accs),
        'Final Train Loss': train_losses[-1],
        'Final Val Loss': val_losses[-1],
        'Min Val Loss': min(val_losses)
    }
    
    metrics_names = list(final_metrics.keys())
    metrics_values = list(final_metrics.values())
    colors = ['skyblue' if 'Acc' in name else 'lightcoral' for name in metrics_names]
    
    bars = plt.bar(range(len(metrics_names)), metrics_values, color=colors, alpha=0.7)
    plt.title('Final Performance Metrics', fontsize=14, fontweight='bold')
    plt.xticks(range(len(metrics_names)), metrics_names, rotation=45, ha='right')
    plt.ylabel('Value')
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 8. Class distribution and confusion matrix info
    plt.subplot(2, 4, 8)
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    
    # Confusion matrix components
    predictions = (np.array(all_probs) > 0.5).astype(int)
    tp = np.sum((all_labels == 1) & (predictions == 1))
    tn = np.sum((all_labels == 0) & (predictions == 0))
    fp = np.sum((all_labels == 0) & (predictions == 1))
    fn = np.sum((all_labels == 1) & (predictions == 0))
    
    confusion_data = np.array([[tn, fp], [fn, tp]])
    im = plt.imshow(confusion_data, cmap='Blues', alpha=0.8)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks([0, 1], ['Non-Schedulable', 'Schedulable'])
    plt.yticks([0, 1], ['Non-Schedulable', 'Schedulable'])
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f'{confusion_data[i, j]}', ha='center', va='center', 
                    fontsize=16, fontweight='bold', color='white' if confusion_data[i, j] > confusion_data.max()/2 else 'black')
    
    plt.colorbar(im, shrink=0.8)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed analysis
    print("\n" + "="*80)
    print("DETAILED PERFORMANCE ANALYSIS")
    print("="*80)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Final Training Accuracy: {train_accs[-1]:.6f}")
    print(f"Final Validation Accuracy: {val_accs[-1]:.6f}")
    print(f"Best Validation Accuracy: {max(val_accs):.6f} (Epoch {val_accs.index(max(val_accs)) + 1})")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"F1-Score: {f1_score:.6f}")
    print(f"True Positives: {tp}, True Negatives: {tn}")
    print(f"False Positives: {fp}, False Negatives: {fn}")
    
    schedulable_ratio = counts[1] / len(all_labels) if len(counts) > 1 else 1.0
    print(f"Dataset Balance: {schedulable_ratio:.1%} schedulable, {(1-schedulable_ratio):.1%} non-schedulable")
    
    # Overfitting analysis
    final_gap = abs(train_losses[-1] - val_losses[-1])
    avg_gap = np.mean([abs(t - v) for t, v in zip(train_losses[-10:], val_losses[-10:])])
    print(f"Final Train-Val Loss Gap: {final_gap:.6f}")
    print(f"Average Loss Gap (last 10 epochs): {avg_gap:.6f}")
    
    if avg_gap < 0.01:
        print("✅ Model shows good generalization (low train-val gap)")
    elif avg_gap < 0.05:
        print("⚠️  Model shows moderate generalization")
    else:
        print("❌ Model may be overfitting (high train-val gap)")
    
    print("="*80)


# -------------------- Run --------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Option 1: Use the new multi-sheet, multi-file dataset for variable-length training
    print("\n" + "="*60)
    print("TRAINING ON VARIABLE-LENGTH TASK DATASETS")
    print("="*60)
    
    # Define your Excel files and their sheets
    excel_files_info = [
        {
            'file': 'C:/Users/anura/Downloads/Scheduling/n=4.xlsx',
            'n_tasks': 4,
            'sheets': ['n=4 u=0.5', 'n=4 u=0.55', 'n=4 u=0.6', 'n=4 u=0.65', 'n=4 u=0.7', 'n=4 u=0.75', 'n=4 u=0.8']
        },
        {
            'file': 'C:/Users/anura/Downloads/Scheduling/n=8.xlsx',  # Replace with your actual 8-task file
            'n_tasks': 8,
            'sheets': ['n=8 u=0.5', 'n=8 u=0.55', 'n=8 u=0.6', 'n=8 u=0.65', 'n=8 u=0.7', 'n=8 u=0.75', 'n=8 u=0.8']
        }
    ]
    
    # Create dataset
    dataset = MultiSheetTaskDataset(excel_files_info)
    print(f"Total dataset size: {len(dataset)}")
    
    # Train/validation split (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders with variable-length collate function
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32,  # Increased batch size for more variety per batch
        shuffle=True, 
        collate_fn=collate_fn_variable
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False, 
        collate_fn=collate_fn_variable
    )
    
    # Create model with updated domain feature dimension (3: util, target_util, n_tasks)
    model = HierarchicalSchedulabilityModel(
        task_in_dim=2,
        emb_dim=64,
        chunk_size=20,
        use_domain_feats=True,
        domain_feat_dim=3  # [utilization, target_utilization, n_tasks]
    ).to(device)
    
    print("Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test a single batch to verify everything works
    print("\nTesting model with sample batch...")
    sample_batch = next(iter(train_loader))
    tasks, lengths, domain_feats, labels = [x.to(device) for x in sample_batch]
    
    print(f"Sample batch shapes:")
    print(f"  Tasks: {tasks.shape}")
    print(f"  Lengths: {lengths.shape}")
    print(f"  Domain features: {domain_feats.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Unique task lengths: {torch.unique(lengths).tolist()}")
    
    # Test forward pass
    with torch.no_grad():
        probs, logits = model(tasks, lengths, domain_feats)
        print(f"  Output probs shape: {probs.shape}")
        print(f"  Sample predictions: {probs[:5].cpu().numpy()}")
    
    print("\nStarting training...")
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, device, epochs=50, lr=1e-3
    )
    
  