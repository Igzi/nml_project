"""
Comparative analysis script for different GAT configurations and baseline models.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from seiz_eeg.dataset import EEGDataset
from gat_model import EEG_GAT, create_distance_adjacency_matrix
from train_gat import create_signal_transforms, create_datasets, seed_everything


class SimpleMLPBaseline(nn.Module):
    """Simple MLP baseline for comparison."""
    
    def __init__(self, input_dim, hidden_dim=128, num_classes=1, dropout=0.3):
        super(SimpleMLPBaseline, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x, distances=None):
        # Flatten input
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.mlp(x)


def evaluate_model(model, val_loader, criterion, device, distances=None, model_name="Model"):
    """Evaluate a model on validation set."""
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.float().to(device)
            y_batch = y_batch.float().unsqueeze(1).to(device)
            
            logits = model(x_batch, distances)
            loss = criterion(logits, y_batch)
            
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()
            
            total_loss += loss.item()
            all_labels.extend(y_batch.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_labels = np.array(all_labels).flatten()
    all_preds = np.array(all_preds).flatten()
    all_probs = np.array(all_probs).flatten()
    
    avg_loss = total_loss / len(val_loader)
    
    # Calculate metrics
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    auc = roc_auc_score(all_labels, all_probs)
    
    print(f"\n{model_name} Results:")
    print(f"Loss: {avg_loss:.4f}")
    print(f"F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, AUC: {auc:.4f}")
    
    return {
        'loss': avg_loss,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc,
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels
    }


def compare_gat_configurations():
    """Compare different GAT configurations."""
    # Set random seed
    seed_everything(42)
    
    # Configuration
    data_root = Path("../data")
    distances_path = "../data/distances_3d.csv"
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load distance matrix
    distances = create_distance_adjacency_matrix(distances_path, device)
    
    # Create signal transforms and datasets
    _, fft_filtering = create_signal_transforms()
    train_dataset, val_dataset = create_datasets(data_root, fft_filtering)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([4.0]).to(device))
    
    # Different configurations to test
    configurations = [
        {
            'name': 'GAT-Small',
            'model_class': EEG_GAT,
            'params': {
                'num_electrodes': 19,
                'input_dim': 29,
                'hidden_dim': 32,
                'num_classes': 1,
                'num_heads': 4,
                'num_layers': 2,
                'dropout': 0.3
            },
            'use_distances': True
        },
        {
            'name': 'GAT-Medium',
            'model_class': EEG_GAT,
            'params': {
                'num_electrodes': 19,
                'input_dim': 29,
                'hidden_dim': 64,
                'num_classes': 1,
                'num_heads': 8,
                'num_layers': 3,
                'dropout': 0.3
            },
            'use_distances': True
        },
        {
            'name': 'GAT-Large',
            'model_class': EEG_GAT,
            'params': {
                'num_electrodes': 19,
                'input_dim': 29,
                'hidden_dim': 128,
                'num_classes': 1,
                'num_heads': 8,
                'num_layers': 4,
                'dropout': 0.4
            },
            'use_distances': True
        },
        {
            'name': 'MLP-Baseline',
            'model_class': SimpleMLPBaseline,
            'params': {
                'input_dim': 19 * 29,  # Flattened input
                'hidden_dim': 128,
                'num_classes': 1,
                'dropout': 0.3
            },
            'use_distances': False
        }
    ]
    
    results = {}
    
    for config in configurations:
        print(f"\n{'='*50}")
        print(f"Training {config['name']}")
        print(f"{'='*50}")
        
        # Create model
        model = config['model_class'](**config['params']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Number of parameters: {num_params}")
        
        # Quick training (fewer epochs for comparison)
        num_epochs = 15
        
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.float().to(device)
                y_batch = y_batch.float().unsqueeze(1).to(device)
                
                optimizer.zero_grad()
                
                if config['use_distances']:
                    logits = model(x_batch, distances)
                else:
                    logits = model(x_batch)
                
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")
        
        # Evaluate model
        dist_arg = distances if config['use_distances'] else None
        result = evaluate_model(
            model, val_loader, criterion, device, 
            distances=dist_arg, model_name=config['name']
        )
        result['num_parameters'] = num_params
        results[config['name']] = result
    
    return results


def plot_comparison_results(results):
    """Plot comparison results."""
    # Create comparison DataFrame
    comparison_data = []
    for model_name, result in results.items():
        comparison_data.append({
            'Model': model_name,
            'F1': result['f1'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'AUC': result['auc'],
            'Parameters': result['num_parameters']
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Plot metrics comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    metrics = ['F1', 'Precision', 'Recall', 'AUC']
    colors = ['skyblue', 'lightgreen', 'salmon', 'gold']
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        row, col = i // 2, i % 2
        
        bars = axes[row, col].bar(df['Model'], df[metric], color=color, alpha=0.7)
        axes[row, col].set_title(f'{metric} Comparison')
        axes[row, col].set_ylabel(metric)
        axes[row, col].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, df[metric]):
            axes[row, col].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('gat_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print results table
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print(df.to_string(index=False, float_format='%.4f'))
    
    return df


def analyze_attention_weights(model, val_loader, distances, device, num_samples=5):
    """Analyze attention weights from the GAT model."""
    model.eval()
    
    # This is a simplified analysis - in practice, you'd need to modify
    # the GAT model to return attention weights
    print("\nAttention Analysis:")
    print("Note: Full attention analysis would require model modification")
    print("to extract and visualize attention weights from GAT layers.")
    
    # For now, just show electrode connectivity based on distance matrix
    plt.figure(figsize=(10, 8))
    
    # Convert distance matrix to numpy for visualization
    dist_np = distances.cpu().numpy()
    
    # Create heatmap of electrode connectivity
    electrode_names = [
        'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
        'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'
    ]
    
    plt.imshow(dist_np, cmap='viridis', alpha=0.8)
    plt.colorbar(label='Connectivity Strength')
    plt.title('Electrode Connectivity Matrix (Distance-based)')
    plt.xlabel('Electrodes')
    plt.ylabel('Electrodes')
    
    # Add electrode labels
    plt.xticks(range(19), electrode_names, rotation=45)
    plt.yticks(range(19), electrode_names)
    
    plt.tight_layout()
    plt.savefig('electrode_connectivity.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main comparison function."""
    print("Starting GAT Model Comparison...")
    
    # Compare different configurations
    results = compare_gat_configurations()
    
    # Plot results
    comparison_df = plot_comparison_results(results)
    
    # Analyze best performing model
    best_model = comparison_df.loc[comparison_df['F1'].idxmax(), 'Model']
    print(f"\nBest performing model: {best_model}")
    
    print("\nComparison completed!")


if __name__ == "__main__":
    main()
