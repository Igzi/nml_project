"""
Evaluation and submission script for GAT model.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from seiz_eeg.dataset import EEGDataset
from gat_model import EEG_GAT, create_distance_adjacency_matrix
from hjorth_features import create_hjorth_transforms
from train_gat import seed_everything


def create_test_dataset(data_root, signal_transform):
    """Create test dataset for submission."""
    test_clips = pd.read_parquet(data_root / "test/segments.parquet")
    
    test_dataset = EEGDataset(
        test_clips,
        signals_root=data_root / "test",
        signal_transform=signal_transform,
        prefetch=True,
        return_id=True  # Return sample IDs instead of labels
    )
    
    return test_dataset


def generate_submission(
    model_path="best_gat_hjorth_model.pth",
    data_root="/home/stnikoli/nml_project/data",
    distances_path="/home/stnikoli/nml_project/data/distances_3d.csv",
    output_file="submission_gat_hjorth.csv",
    batch_size=256
):
    """Generate submission file using trained GAT model with Hjorth features."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load test data
    data_root = Path(data_root)
    clips_te = pd.read_parquet(data_root / "test/segments.parquet")
    print(f"Loaded {len(clips_te)} test segments")
    
    # Create Hjorth feature transform
    _, hjorth_transform = create_hjorth_transforms()
    
    # Create test dataset
    test_dataset = EEGDataset(
        clips_te,
        signals_root=data_root / "test",
        signal_transform=hjorth_transform,
        prefetch=False,
        return_id=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Load model
    adjacency_matrix = create_distance_adjacency_matrix(distances_path, device)
    
    model = EEG_GAT(
        num_electrodes=19,
        input_dim=15,  # 5 frequency bands × 3 Hjorth parameters
        hidden_dim=64,
        num_classes=1,
        num_heads=8,
        num_layers=3,
        dropout=0.3,
        adjacency_matrix=adjacency_matrix
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print("Model loaded successfully")
    
    # Generate predictions
    all_predictions = []
    all_ids = []
    
    with torch.no_grad():
        for x_batch, id_batch in tqdm(test_loader, desc="Generating predictions"):
            x_batch = x_batch.to(device)
            
            # Forward pass
            logits = model(x_batch)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > 0.5).int()
            
            all_predictions.extend(predictions.cpu().numpy().flatten())
            all_ids.extend(id_batch)
    
    # Create submission DataFrame
    submission_df = pd.DataFrame({
        "id": all_ids,
        "label": all_predictions
    })
    
    return submission_df


def evaluate_with_framework():
    """Evaluate using the provided evaluation framework."""
    # Import evaluation function
    sys.path.append(str(Path(__file__).parent.parent))
    from evaluation import evaluate
    
    # Configuration
    data_root = Path("../data")
    distances_path = "../data/distances_3d.csv"
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load distance matrix
    adjacency_matrix = create_distance_adjacency_matrix(distances_path, device)
    
    # Create signal transform
    _, hjorth_transform = create_hjorth_transforms()
    
    # Load training data
    clips_tr = pd.read_parquet(data_root / "train/segments.parquet")
    
    # Create loss function
    class_weight = 4.0
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.FloatTensor([class_weight]).to(device)
    )
    
    # Evaluate using framework
    evaluate(
        EEG_GAT,
        clips_tr,
        signals_root=data_root / "train",
        num_epochs=20,
        learning_rate=1e-3,
        model_args={
            "num_electrodes": 19,
            "input_dim": 15,  # 5 frequency bands × 3 Hjorth parameters
            "hidden_dim": 64,
            "num_classes": 1,
            "num_heads": 8,
            "num_layers": 3,
            "dropout": 0.3,
            "adjacency_matrix": adjacency_matrix
        },
        criterion=criterion,
        signal_transform=hjorth_transform,
        batch_size=256,
        prefetch=True
    )


def main():
    """Main evaluation and submission function."""
    # Set random seed
    seed_everything(42)
    
    # Configuration
    data_root = Path("../data")
    distances_path = "../data/distances_3d.csv"
    model_path = "best_gat_hjorth_model.pth"
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load distance matrix
    adjacency_matrix = create_distance_adjacency_matrix(distances_path, device)
    
    # Create signal transform
    _, hjorth_transform = create_hjorth_transforms()
    
    # Load trained model
    model = EEG_GAT(
        num_electrodes=19,
        input_dim=15,  # 5 frequency bands × 3 Hjorth parameters
        hidden_dim=64,
        num_classes=1,
        num_heads=8,
        num_layers=3,
        dropout=0.3,
        adjacency_matrix=adjacency_matrix
    ).to(device)
    
    # Load model weights if available
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model weights from {model_path}")
    except FileNotFoundError:
        print(f"Model weights not found at {model_path}")
        print("Please train the model first using train_gat.py")
        return
    
    # Create test dataset
    print("Creating test dataset...")
    test_dataset = create_test_dataset(data_root, hjorth_transform)
    print(f"Test samples: {len(test_dataset)}")
    
    # Generate predictions
    print("Generating predictions...")
    submission_df = generate_submission(
        model_path=model_path,
        data_root=data_root,
        distances_path=distances_path,
        output_file="gat_hjorth_submission.csv"
    )
    
    # Save submission file
    submission_filename = "gat_hjorth_submission.csv"
    submission_df.to_csv(submission_filename, index=False)
    print(f"Submission file saved as {submission_filename}")
    
    # Print some statistics
    print(f"Total predictions: {len(submission_df)}")
    print(f"Positive predictions: {submission_df['label'].sum()}")
    print(f"Positive rate: {submission_df['label'].mean():.4f}")


if __name__ == "__main__":
    main()
