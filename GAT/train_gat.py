"""
Training script for GAT model on EEG seizure detection.
"""

import os
import sys
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from seiz_eeg.dataset import EEGDataset
from gat_model import EEG_GAT, create_distance_adjacency_matrix
from hjorth_features import create_hjorth_transforms


def seed_everything(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_signal_transforms():
    """Create Hjorth signal transformation functions."""
    hjorth_transform, _ = create_hjorth_transforms()
    
    return hjorth_transform, hjorth_transform


def create_datasets(data_root, signal_transform, test_size=0.2, random_state=42):
    """Create training and validation datasets."""
    clips_tr = pd.read_parquet(data_root / "train/segments.parquet")
    
    # Split by patients
    patients = clips_tr['signals_path'].unique()
    train_patients, val_patients = train_test_split(
        patients, test_size=test_size, random_state=random_state
    )
    
    # Create training and validation clips
    train_clips = clips_tr[clips_tr['signals_path'].isin(train_patients)]
    val_clips = clips_tr[clips_tr['signals_path'].isin(val_patients)]
    
    # Create datasets
    train_dataset = EEGDataset(
        train_clips,
        signals_root=data_root / "train",
        signal_transform=signal_transform,
        prefetch=True
    )
    
    val_dataset = EEGDataset(
        val_clips,
        signals_root=data_root / "train",
        signal_transform=signal_transform,
        prefetch=True
    )
    
    return train_dataset, val_dataset


def create_dataloaders(train_dataset, val_dataset, batch_size=256):
    """Create PyTorch DataLoaders."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader


def train_epoch(model, train_loader, criterion, optimizer, device, distances=None):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for x_batch, y_batch in tqdm(train_loader, desc="Training", leave=False):
        # Move data to device
        x_batch = x_batch.float().to(device)
        y_batch = y_batch.float().unsqueeze(1).to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(x_batch)  # Model uses its own adjacency matrix
        loss = criterion(logits, y_batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def validate_epoch(model, val_loader, criterion, device, distances=None):
    """Validate the model for one epoch."""
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for x_batch, y_batch in tqdm(val_loader, desc="Validation", leave=False):
            # Move data to device
            x_batch = x_batch.float().to(device)
            y_batch = y_batch.float().unsqueeze(1).to(device)
            
            # Forward pass
            logits = model(x_batch)  # Model uses its own adjacency matrix
            loss = criterion(logits, y_batch)
            
            # Get predictions
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()
            
            # Collect results
            total_loss += loss.item()
            all_labels.extend(y_batch.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    all_labels = np.array(all_labels).flatten()
    all_preds = np.array(all_preds).flatten()
    all_probs = np.array(all_probs).flatten()
    
    avg_loss = total_loss / len(val_loader)
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    auc = roc_auc_score(all_labels, all_probs)
    
    return avg_loss, f1, precision, recall, auc


def train_gat_model(
    data_root,
    distances_path,
    num_epochs=50,
    batch_size=256,
    learning_rate=1e-3,
    hidden_dim=64,
    num_heads=8,
    num_layers=3,
    dropout=0.3,
    class_weight=4.0,
    patience=10,
    seed=42
):
    """
    Train the GAT model.
    
    Args:
        data_root: Path to data directory
        distances_path: Path to distances CSV file
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        hidden_dim: Hidden dimension for GAT layers
        num_heads: Number of attention heads
        num_layers: Number of GAT layers
        dropout: Dropout rate
        class_weight: Weight for positive class in loss function
        patience: Early stopping patience
        seed: Random seed
    """
    # Set random seed
    seed_everything(seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create signal transforms
    hjorth_transform, _ = create_hjorth_transforms()
    
    # Create datasets
    print("Creating datasets...")
    train_dataset, val_dataset = create_datasets(data_root, hjorth_transform)
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, batch_size)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Load distance matrix
    distances = create_distance_adjacency_matrix(distances_path, device)
    print(f"Adjacency matrix shape: {distances.shape}")
    
    # Create model
    model = EEG_GAT(
        num_electrodes=19,
        input_dim=15,  # Hjorth features (5 bands × 3 parameters)
        hidden_dim=hidden_dim,
        num_classes=1,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        adjacency_matrix=distances  # Pass the adjacency matrix to the model
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of model parameters: {num_params}")
    
    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.FloatTensor([class_weight]).to(device)
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_f1': [],
        'val_precision': [],
        'val_recall': [],
        'val_auc': []
    }
    
    best_f1 = 0.0
    patience_counter = 0
    
    print("Starting training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Training
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation
        val_loss, val_f1, val_precision, val_recall, val_auc = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step(val_f1)
        
        # Save metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_auc'].append(val_auc)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, F1: {val_f1:.4f}, "
              f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, AUC: {val_auc:.4f}")
        
        # Early stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), "best_gat_hjorth_model.pth")
            print(f"New best F1 score: {best_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    # Load best model
    model.load_state_dict(torch.load("best_gat_hjorth_model.pth"))
    
    return model, history


def plot_training_history(history):
    """Plot training history."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # F1 Score
    axes[0, 1].plot(history['val_f1'], label='Val F1', color='orange')
    axes[0, 1].set_title('F1 Score')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision and Recall
    axes[1, 0].plot(history['val_precision'], label='Precision', color='green')
    axes[1, 0].plot(history['val_recall'], label='Recall', color='red')
    axes[1, 0].set_title('Precision and Recall')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # AUC
    axes[1, 1].plot(history['val_auc'], label='Val AUC', color='purple')
    axes[1, 1].set_title('AUC Score')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('AUC')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('gat_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main training function."""
    # Configuration
    data_root = Path("/home/stnikoli/nml_project/data")
    distances_path = "/home/stnikoli/nml_project/data/distances_3d.csv"
    
    # Train model
    model, history = train_gat_model(
        data_root=data_root,
        distances_path=distances_path,
        num_epochs=20,
        batch_size=256,
        learning_rate=1e-4,
        hidden_dim=64,
        num_heads=8,
        num_layers=3,
        dropout=0.3,
        class_weight=4.0,
        patience=10,
        seed=42
    )
    
    # Plot training history
    plot_training_history(history)
    
    print("Training completed!")
    print(f"Best validation F1 score: {max(history['val_f1']):.4f}")


if __name__ == "__main__":
    main()
