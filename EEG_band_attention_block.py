"""
Multi-Band EEG Transformer for Auditory Attention Decoding
Based on the provided architecture with Band Attention and Temporal Self-Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import h5py
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm
import logging
from scipy import signal as scipy_signal
import random

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Configuration
class Config:
    # Data parameters
    n_channels = 32
    n_bands = 5
    n_classes = 2
    window_size = 2048
    sampling_rate = 2048
    
    # Spectrogram parameters (reduced for memory efficiency)
    # Output shape: (n_bands=5, n_channels=32, freq_bins=8, time_bins=16)
    nperseg = 128
    noverlap = 64
    nfft = 256
    
    # Model architecture
    d_model = 128  # Embedding dimension
    n_heads = 8
    n_layers = 4
    dim_feedforward = 512
    dropout = 0.1
    
    # Training parameters
    batch_size = 16  # Reduced from 32 to save memory
    num_epochs = 100
    learning_rate = 1e-4
    weight_decay = 1e-5
    warmup_epochs = 10
    
    # Stochastic augmentation
    aug_prob = 0.5
    noise_std = 0.1
    time_shift_max = 100
    channel_dropout_prob = 0.1
    
    # Paths
    data_dir = "./preprocessed_data"
    checkpoint_dir = "./checkpoints"
    log_file = "training.log"
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_logger(log_file: str):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def print_memory_usage():
    """Print current memory usage"""
    if torch.cuda.is_available():
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    
    import psutil
    process = psutil.Process()
    print(f"CPU Memory: {process.memory_info().rss / 1e9:.2f} GB")


class EEGDataset(Dataset):
    """
    Dataset for loading preprocessed EEG data and generating spectrograms on-the-fly
    Uses lazy loading to avoid memory issues
    """
    def __init__(self, subject_ids: List[int], data_dir: str, config: Config, 
                 augment: bool = False):
        self.config = config
        self.augment = augment
        self.data_dir = Path(data_dir)
        self.subject_ids = subject_ids
        
        # Store file paths and indices instead of loading all data
        self.file_paths = []
        self.sample_indices = []
        
        total_samples = 0
        for subject_id in subject_ids:
            h5_file = self.data_dir / f"subject_{subject_id:02d}_preprocessed.h5"
            
            with h5py.File(h5_file, 'r') as f:
                n_samples = f['EEG/clean'].shape[0]
                
                for i in range(n_samples):
                    self.file_paths.append(h5_file)
                    self.sample_indices.append(i)
                    total_samples += 1
        
        print(f"Loaded {len(subject_ids)} subjects: {total_samples} samples")
        
        # Cache for band filters to avoid recomputing
        self._filter_cache = {}
    
    def __len__(self):
        return len(self.file_paths)
    
    def _get_filter_coeffs(self, low: float, high: float):
        """Cache filter coefficients"""
        key = (low, high)
        if key not in self._filter_cache:
            nyquist = self.config.sampling_rate / 2
            b, a = scipy_signal.butter(4, [low/nyquist, high/nyquist], btype='band')
            self._filter_cache[key] = (b, a)
        return self._filter_cache[key]
    
    def augment_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply stochastic augmentation to EEG signal
        Args:
            signal: (window_size, n_channels)
        Returns:
            Augmented signal
        """
        if not self.augment or random.random() > self.config.aug_prob:
            return signal
        
        signal = signal.copy()
        
        # 1. Add Gaussian noise
        if random.random() < 0.5:
            noise = np.random.normal(0, self.config.noise_std, signal.shape)
            signal = signal + noise * np.std(signal)
        
        # 2. Time shifting
        if random.random() < 0.5:
            shift = random.randint(-self.config.time_shift_max, self.config.time_shift_max)
            signal = np.roll(signal, shift, axis=0)
        
        # 3. Channel dropout
        if random.random() < 0.3:
            n_drop = int(self.config.n_channels * self.config.channel_dropout_prob)
            drop_channels = random.sample(range(self.config.n_channels), n_drop)
            signal[:, drop_channels] = 0
        
        # 4. Amplitude scaling
        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)
            signal = signal * scale
        
        return signal
    
    def generate_spectrograms(self, signal: np.ndarray) -> np.ndarray:
        """
        Generate LIGHTWEIGHT spectrograms for all 5 frequency bands
        Optimized to reduce memory usage with fixed output shape
        Args:
            signal: (window_size, n_channels)
        Returns:
            spectrograms: (n_bands, n_channels, freq_bins, time_bins)
        """
        # Frequency bands
        bands = [
            (0.5, 4),   # delta
            (4, 8),     # theta
            (8, 13),    # alpha
            (13, 30),   # beta
            (30, 45)    # gamma
        ]
        
        # Reduced spectrogram parameters for memory efficiency
        nperseg = 128  # Reduced from 256
        noverlap = 64   # Reduced from 128
        nfft = 256      # Reduced from 512
        
        # Target output size for consistency
        target_freq_bins = 8
        target_time_bins = 16
        
        spectrograms = []
        
        for low, high in bands:
            # Get cached filter coefficients
            b, a = self._get_filter_coeffs(low, high)
            
            band_specs = []
            
            for ch in range(self.config.n_channels):
                # Filter signal for this band
                filtered = scipy_signal.filtfilt(b, a, signal[:, ch])
                
                # Generate spectrogram with reduced resolution
                f, t, Sxx = scipy_signal.spectrogram(
                    filtered,
                    fs=self.config.sampling_rate,
                    nperseg=nperseg,
                    noverlap=noverlap,
                    nfft=nfft,
                    scaling='spectrum'  # Use spectrum instead of density
                )
                
                # Select frequencies within band
                freq_mask = (f >= low) & (f <= high)
                Sxx_band = Sxx[freq_mask, :]
                
                # Log power with numerical stability
                Sxx_band = np.log1p(Sxx_band)  # log(1+x) more stable than log10
                
                # Resize to fixed shape using interpolation
                from scipy.ndimage import zoom
                
                if Sxx_band.shape[0] > 0 and Sxx_band.shape[1] > 0:
                    freq_zoom = target_freq_bins / Sxx_band.shape[0]
                    time_zoom = target_time_bins / Sxx_band.shape[1]
                    
                    Sxx_resized = zoom(Sxx_band, (freq_zoom, time_zoom), order=1)
                    
                    # Ensure exact shape
                    Sxx_resized = Sxx_resized[:target_freq_bins, :target_time_bins]
                else:
                    # Fallback: zeros if spectrogram is empty
                    Sxx_resized = np.zeros((target_freq_bins, target_time_bins))
                
                band_specs.append(Sxx_resized)
            
            # Stack channels: (n_channels, freq_bins, time_bins)
            band_spec = np.stack(band_specs, axis=0).astype(np.float32)  # Use float32
            spectrograms.append(band_spec)
        
        # Stack bands: (n_bands, n_channels, freq_bins, time_bins)
        spectrograms = np.stack(spectrograms, axis=0)
        
        return spectrograms
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Lazy load from HDF5 file
        h5_file = self.file_paths[idx]
        sample_idx = self.sample_indices[idx]
        
        with h5py.File(h5_file, 'r') as f:
            # Load single sample
            signal = f['EEG/clean'][sample_idx]  # (2048, 32)
            label = f['Labels'][sample_idx]
            
            # Flatten label if needed
            if isinstance(label, np.ndarray) and label.ndim > 0:
                label = label.item()
        
        # Apply augmentation
        signal = self.augment_signal(signal)
        
        # Generate spectrograms on-the-fly
        spectrograms = self.generate_spectrograms(signal)  # (5, 32, freq, time)
        
        # Convert to tensors
        spectrograms = torch.from_numpy(spectrograms).float()
        label = torch.tensor(label, dtype=torch.long)
        
        return spectrograms, label


class BandAttentionBlock(nn.Module):
    """
    Band Attention Block from the architecture
    Applies attention across frequency bands
    """
    def __init__(self, d_model: int, n_bands: int):
        super().__init__()
        self.d_model = d_model
        self.n_bands = n_bands
        
        # Max and Avg pooling
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_bands, d_model, H, W)
        Returns:
            attention: (batch, n_bands, 1, 1, 1)
        """
        batch, n_bands, d_model, H, W = x.shape
        
        # Pool across spatial dimensions for each band
        max_pooled = self.max_pool(x.view(batch * n_bands, d_model, H, W))  # (B*bands, d, 1, 1)
        avg_pooled = self.avg_pool(x.view(batch * n_bands, d_model, H, W))  # (B*bands, d, 1, 1)
        
        # Flatten and reshape
        max_pooled = max_pooled.view(batch, n_bands, d_model)  # (B, bands, d)
        avg_pooled = avg_pooled.view(batch, n_bands, d_model)  # (B, bands, d)
        
        # Concatenate max and avg
        pooled = torch.cat([max_pooled, avg_pooled], dim=-1)  # (B, bands, 2*d)
        
        # Apply MLP to get attention weight per band
        attention = self.mlp(pooled)  # (B, bands, 1)
        
        # Reshape for broadcasting: (B, bands, 1, 1, 1)
        attention = attention.unsqueeze(-1).unsqueeze(-1)
        
        return attention


class SpatialSelfAttentionBlock(nn.Module):
    """
    Spatial Self-Attention Block using Multi-Head Attention
    Processes spatial (channel) relationships
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x: (batch, seq_len, d_model)
        """
        # Pre-norm
        x_norm = self.layer_norm(x)
        
        # Multi-head attention
        attn_out, _ = self.mha(x_norm, x_norm, x_norm)
        
        # Residual connection
        x = x + self.dropout(attn_out)
        
        return x


class TemporalSelfAttentionBlock(nn.Module):
    """
    Temporal Self-Attention Block using Multi-Head Attention
    Processes temporal relationships
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x: (batch, seq_len, d_model)
        """
        # Multi-head attention with residual
        x_norm = self.layer_norm(x)
        attn_out, _ = self.mha(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)
        
        # Feed-forward with residual
        x = x + self.ffn(x)
        
        return x


class MultiBandEEGTransformer(nn.Module):
    """
    Complete Multi-Band EEG Transformer Architecture
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Multi-band feature extraction (CNN on spectrograms)
        # Input: (n_channels, freq_bins=8, time_bins=16)
        self.band_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(config.n_channels, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),  # 8x16 -> 4x8
                nn.Conv2d(128, config.d_model, kernel_size=3, padding=1),
                nn.BatchNorm2d(config.d_model),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4))  # Fixed spatial size 4x4
            )
            for _ in range(config.n_bands)
        ])
        
        # Band Attention Block
        self.band_attention = BandAttentionBlock(config.d_model, config.n_bands)
        
        # Spatial dimension: 4x4 = 16 positions per band
        self.spatial_pos_embedding = nn.Parameter(torch.randn(1, 16, config.d_model))
        
        # Spatial Self-Attention Blocks
        self.spatial_blocks = nn.ModuleList([
            SpatialSelfAttentionBlock(config.d_model, config.n_heads, config.dropout)
            for _ in range(config.n_layers // 2)
        ])
        
        # Temporal dimension
        self.temporal_pos_embedding = nn.Parameter(torch.randn(1, config.n_bands, config.d_model))
        
        # Temporal Self-Attention Blocks
        self.temporal_blocks = nn.ModuleList([
            TemporalSelfAttentionBlock(config.d_model, config.n_heads, config.dropout)
            for _ in range(config.n_layers // 2)
        ])
        
        # Classification head
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model))
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.n_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_bands, n_channels, freq, time)
        Returns:
            logits: (batch, n_classes)
        """
        batch_size = x.shape[0]
        
        # Extract features from each band
        band_features = []
        for i in range(self.config.n_bands):
            feat = self.band_encoders[i](x[:, i])  # (B, d_model, 4, 4)
            band_features.append(feat)
        
        # Stack: (B, n_bands, d_model, 4, 4)
        band_features = torch.stack(band_features, dim=1)
        
        # Apply band attention
        band_attn = self.band_attention(band_features)  # (B, n_bands, 1, 1, 1)
        band_features = band_features * band_attn  # Weighted by attention
        
        # Reshape for spatial self-attention
        # (B, n_bands, d_model, 4, 4) -> (B*n_bands, 16, d_model)
        spatial_features = band_features.flatten(3).permute(0, 1, 3, 2)  # (B, bands, 16, d)
        spatial_features = spatial_features.reshape(batch_size * self.config.n_bands, 16, self.config.d_model)
        
        # Add spatial positional embedding
        spatial_features = spatial_features + self.spatial_pos_embedding
        
        # Apply spatial self-attention blocks
        for block in self.spatial_blocks:
            spatial_features = block(spatial_features)
        
        # Aggregate spatial features (mean pooling)
        spatial_features = spatial_features.mean(dim=1)  # (B*bands, d_model)
        spatial_features = spatial_features.view(batch_size, self.config.n_bands, self.config.d_model)
        
        # Add temporal positional embedding
        temporal_features = spatial_features + self.temporal_pos_embedding
        
        # Apply temporal self-attention blocks
        for block in self.temporal_blocks:
            temporal_features = block(temporal_features)
        
        # Add CLS token for classification
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        temporal_features = torch.cat([cls_tokens, temporal_features], dim=1)
        
        # Use CLS token for classification
        cls_output = temporal_features[:, 0]  # (B, d_model)
        
        # Classify
        logits = self.classifier(cls_output)  # (B, n_classes)
        
        return logits


class Trainer:
    """Training loop with validation and checkpointing"""
    def __init__(self, model: nn.Module, config: Config, logger):
        self.model = model.to(config.device)
        self.config = config
        self.logger = logger
        
        # Optimizer and scheduler
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Tracking
        self.best_val_acc = 0.0
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch with on-the-fly spectrogram generation"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Gradient accumulation to simulate larger batch size
        accumulation_steps = 2
        
        pbar = tqdm(dataloader, desc="Training")
        self.optimizer.zero_grad()
        
        for batch_idx, (spectrograms, labels) in enumerate(pbar):
            # Spectrograms generated on-the-fly in DataLoader, moved to GPU
            spectrograms = spectrograms.to(self.config.device, non_blocking=True)
            labels = labels.to(self.config.device, non_blocking=True)
            
            # Forward pass
            logits = self.model(spectrograms)
            loss = self.criterion(logits, labels)
            loss = loss / accumulation_steps  # Scale loss
            
            # Backward pass
            loss.backward()
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Metrics
            total_loss += loss.item() * accumulation_steps
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # CRITICAL: Delete spectrograms immediately after use
            del spectrograms, labels, logits, loss
            
            # Clear cache every few batches
            if batch_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            pbar.set_postfix({'loss': total_loss / (batch_idx + 1), 'acc': 100. * correct / total})
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Validate the model with on-the-fly spectrogram generation"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (spectrograms, labels) in enumerate(tqdm(dataloader, desc="Validation")):
                # Spectrograms generated on-the-fly, moved to GPU
                spectrograms = spectrograms.to(self.config.device, non_blocking=True)
                labels = labels.to(self.config.device, non_blocking=True)
                
                logits = self.model(spectrograms)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # CRITICAL: Delete spectrograms immediately after use
                del spectrograms, labels, logits, loss
                
                # Clear cache every few batches
                if batch_idx % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch: int, val_acc: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'config': self.config
        }
        
        # Save latest
        torch.save(checkpoint, self.checkpoint_dir / 'latest_checkpoint.pth')
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best_model.pth')
            self.logger.info(f"Saved best model with validation accuracy: {val_acc:.2f}%")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Complete training loop"""
        self.logger.info("Starting training...")
        self.logger.info(f"Device: {self.config.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.config.num_epochs):
            self.logger.info(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            self.logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Learning rate scheduling
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f"Learning Rate: {current_lr:.6f}")
            
            # Save checkpoint
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
            
            self.save_checkpoint(epoch, val_acc, is_best)
        
        self.logger.info(f"\nTraining completed! Best validation accuracy: {self.best_val_acc:.2f}%")


def main():
    # Setup
    config = Config()
    logger = setup_logger(config.log_file)
    
    logger.info("="*80)
    logger.info("Multi-Band EEG Transformer Training")
    logger.info("="*80)
    
    # Print initial memory
    print("\nInitial Memory Usage:")
    print_memory_usage()
    
    # Define subject splits (12 subjects total)
    all_subjects = list(range(1, 13))  # Subjects 1-12
    
    # Split: 8 train, 2 validation, 2 test (hold out for final eval)
    train_subjects = [1, 2, 3, 4, 5, 6, 7, 8]
    val_subjects = [9, 10]
    test_subjects = [11, 12]  # Not used during training
    
    logger.info(f"Train subjects: {train_subjects}")
    logger.info(f"Validation subjects: {val_subjects}")
    logger.info(f"Test subjects (held out): {test_subjects}")
    
    # Create datasets
    logger.info("\nLoading datasets...")
    train_dataset = EEGDataset(train_subjects, config.data_dir, config, augment=True)
    val_dataset = EEGDataset(val_subjects, config.data_dir, config, augment=False)
    
    print("\nMemory after dataset creation:")
    print_memory_usage()
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,  # Reduced workers to save memory
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,  # Reduced workers
        pin_memory=True,
        persistent_workers=True
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    
    # Create model
    logger.info("\nBuilding model...")
    model = MultiBandEEGTransformer(config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = Trainer(model, config, logger)
    
    # Train
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()