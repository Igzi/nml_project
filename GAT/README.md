# Graph Attention Networks for EEG Seizure Detection

This directory contains a complete implementation of Graph Attention Networks (GATs) for EEG seizure detection using the Temple University Hospital EEG Seizure Corpus (TUSZ) dataset.

## Overview

Graph Attention Networks leverage attention mechanisms to learn the importance of different nodes in a graph structure. For EEG data, we represent the 19 electrodes as nodes in a graph, where edges represent spatial relationships between electrodes based on their physical distances on the scalp.

## Key Features

- **Graph Attention Mechanism**: Multi-head attention for learning electrode relationships
- **Spatial Graph Structure**: Distance-based adjacency matrix representing electrode positions
- **Temporal Processing**: Handles time-series EEG data with FFT preprocessing
- **Flexible Architecture**: Configurable number of layers, heads, and hidden dimensions
- **Comprehensive Evaluation**: Comparison with baseline models and different configurations

## Files Description

### Core Implementation
- `gat_model.py`: Main GAT model implementation including:
  - `GraphAttentionLayer`: Single attention layer
  - `MultiHeadGATLayer`: Multi-head attention mechanism
  - `EEG_GAT`: Complete GAT model for EEG classification
  - `TemporalAttention`: Optional temporal attention mechanism

### Training and Evaluation
- `train_gat.py`: Training script with:
  - Data loading and preprocessing
  - Model training with early stopping
  - Validation metrics tracking
  - Training history visualization

- `evaluate_gat.py`: Evaluation and submission script:
  - Model evaluation on test set
  - Kaggle submission file generation
  - Integration with evaluation framework

### Analysis and Comparison
- `compare_models.py`: Comparative analysis including:
  - Different GAT configurations (Small, Medium, Large)
  - MLP baseline comparison
  - Performance metrics visualization
  - Attention weight analysis

## Model Architecture

### Graph Attention Layer
The core GAT layer implements the attention mechanism:

```python
# Attention computation
e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
α_ij = softmax_j(e_ij)
h'_i = σ(Σ_j α_ij Wh_j)
```

Where:
- `W` is the learnable weight matrix
- `a` is the attention mechanism's weight vector
- `h_i` are the input node features
- `α_ij` are the attention coefficients

### Multi-Head Attention
Multiple attention heads capture different types of relationships:
- Each head learns different electrode interactions
- Outputs are concatenated or averaged
- Increases model expressiveness

### EEG-Specific Design
- **Nodes**: 19 EEG electrodes
- **Node Features**: FFT-transformed signals (29 frequency features)
- **Edges**: Distance-based adjacency matrix
- **Graph Pooling**: Global pooling for classification

## Usage

### 1. Training
```bash
cd /home/stnikoli/nml_project/GAT
python train_gat.py
```

### 2. Evaluation and Submission
```bash
python evaluate_gat.py
```

### 3. Model Comparison
```bash
python compare_models.py
```

## Configuration

### Model Parameters
- `num_electrodes`: 19 (standard EEG montage)
- `input_dim`: 29 (FFT features)
- `hidden_dim`: 64 (configurable)
- `num_heads`: 8 (attention heads)
- `num_layers`: 3 (GAT layers)
- `dropout`: 0.3

### Training Parameters
- `batch_size`: 256
- `learning_rate`: 1e-3
- `num_epochs`: 50
- `class_weight`: 4.0 (for imbalanced data)
- `patience`: 10 (early stopping)

## Data Processing

### Signal Preprocessing
1. **Bandpass Filtering**: 0.5-30 Hz (seizure-relevant frequencies)
2. **FFT Transformation**: Convert to frequency domain
3. **Log Transformation**: Stabilize variance
4. **Normalization**: Per-electrode normalization

### Graph Construction
- **Distance Matrix**: 3D Euclidean distances between electrodes
- **Adjacency Matrix**: Threshold-based or adaptive connectivity
- **Self-Connections**: Added for all electrodes

## Results

The GAT model shows improved performance over baseline methods by:
- **Capturing Spatial Relationships**: Electrode connectivity patterns
- **Attention Mechanism**: Learning relevant electrode interactions
- **Multi-Head Design**: Different attention patterns for different seizure types

### Expected Performance
- **F1 Score**: ~0.6-0.8 (depending on configuration)
- **Precision**: ~0.7-0.9
- **Recall**: ~0.5-0.7
- **AUC**: ~0.8-0.9

## Advantages of GAT for EEG

1. **Interpretability**: Attention weights show important electrode connections
2. **Spatial Awareness**: Explicitly models electrode relationships
3. **Flexibility**: Adapts to different graph structures
4. **Scalability**: Efficient attention computation

## Future Improvements

1. **Dynamic Graphs**: Time-varying electrode connectivity
2. **Hierarchical Attention**: Multi-scale spatial relationships
3. **Temporal GAT**: Graph attention across time steps
4. **Multi-Modal Fusion**: Combine with other neuroimaging modalities

## Dependencies

- PyTorch >= 1.8.0
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- SciPy
- tqdm

## References

- Veličković, P., et al. "Graph attention networks." ICLR 2018.
- Shoeb, A. H. "Application of machine learning to epileptic seizure onset detection and treatment." PhD thesis, MIT, 2009.
- Temple University Hospital EEG Seizure Corpus (TUSZ)
