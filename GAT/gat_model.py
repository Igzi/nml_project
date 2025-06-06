"""
Graph Attention Network (GAT) implementation for EEG seizure detection.

This module implements a GAT model that processes EEG data represented as graphs,
where nodes correspond to electrodes and edges encode spatial relationships.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np        


class GraphAttentionLayer(nn.Module):
    """
    Single Graph Attention Layer implementation.
    
    Args:
        in_features: Number of input features per node
        out_features: Number of output features per node
        dropout: Dropout rate
        alpha: LeakyReLU negative slope
        concat: Whether to concatenate or average multi-head outputs
    """
    
    def __init__(self, in_features, out_features, dropout=0.3, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        # Linear transformation for input features
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        # Attention mechanism parameters
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        
        # Dropout and activation
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters using Glorot initialization."""
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
    
    def forward(self, h, adj):
        """
        Forward pass of the GAT layer.
        
        Args:
            h: Input node features [N, in_features]
            adj: Adjacency matrix [N, N]
            
        Returns:
            Output node features [N, out_features]
        """
        # Linear transformation
        Wh = torch.mm(h, self.W)  # [N, out_features]
        N = Wh.size()[0]
        
        # Compute attention coefficients
        # Create all pairs for attention computation
        Wh1 = torch.mm(Wh, self.a[:self.out_features, :])  # [N, 1]
        Wh2 = torch.mm(Wh, self.a[self.out_features:, :])  # [N, 1]
        
        # Broadcast and add to get attention scores for all pairs
        e = Wh1 + Wh2.T  # [N, N]
        e = self.leakyrelu(e)
        
        # Mask attention scores using adjacency matrix
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        # Softmax normalization
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)
        
        # Apply attention to node features
        h_prime = torch.mm(attention, Wh)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class MultiHeadGATLayer(nn.Module):
    """
    Multi-head Graph Attention Layer.
    
    Args:
        in_features: Number of input features per node
        out_features: Number of output features per node
        num_heads: Number of attention heads
        dropout: Dropout rate
        alpha: LeakyReLU negative slope
        concat: Whether to concatenate or average multi-head outputs
    """
    
    def __init__(self, in_features, out_features, num_heads=8, dropout=0.3, alpha=0.2, concat=True):
        super(MultiHeadGATLayer, self).__init__()
        self.num_heads = num_heads
        self.concat = concat
        
        # Create multiple attention heads
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(in_features, out_features, dropout, alpha, concat)
            for _ in range(num_heads)
        ])
        
        if concat:
            self.out_features = out_features * num_heads
        else:
            self.out_features = out_features
    
    def forward(self, h, adj):
        """
        Forward pass of multi-head GAT layer.
        
        Args:
            h: Input node features [N, in_features]
            adj: Adjacency matrix [N, N]
            
        Returns:
            Output node features [N, out_features * num_heads] or [N, out_features]
        """
        if self.concat:
            # Concatenate outputs from all heads
            head_outputs = [att(h, adj) for att in self.attentions]
            return torch.cat(head_outputs, dim=1)
        else:
            # Average outputs from all heads
            head_outputs = [att(h, adj) for att in self.attentions]
            return torch.mean(torch.stack(head_outputs), dim=0)


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism for time series data.
    
    Args:
        hidden_dim: Hidden dimension for attention computation
        dropout: Dropout rate
    """
    
    def __init__(self, hidden_dim, dropout=0.3):
        super(TemporalAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        """
        Apply temporal attention to sequence.
        
        Args:
            x: Input sequence [batch_size, seq_len, hidden_dim]
            
        Returns:
            Attended output [batch_size, hidden_dim]
        """
        # Compute attention scores
        attention_scores = self.attention(x)  # [batch_size, seq_len, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, seq_len, 1]
        
        # Apply attention weights
        attended_output = torch.sum(attention_weights * x, dim=1)  # [batch_size, hidden_dim]
        
        return attended_output, attention_weights


class EEG_GAT(nn.Module):
    """
    Graph Attention Network for EEG seizure detection.
    
    This model processes EEG data as graphs where:
    - Nodes represent EEG electrodes (19 channels)
    - Edges represent spatial relationships between electrodes
    - Node features are frequency domain representations of EEG signals
    
    Args:
        num_electrodes: Number of EEG electrodes (nodes in graph)
        input_dim: Input feature dimension per electrode
        hidden_dim: Hidden dimension for GAT layers
        num_classes: Number of output classes (1 for binary classification)
        num_heads: Number of attention heads in GAT layers
        num_layers: Number of GAT layers
        dropout: Dropout rate
        use_temporal_attention: Whether to use temporal attention
    """
    
    def __init__(self, num_electrodes=19, input_dim=15, hidden_dim=64, num_classes=1, 
                 num_heads=8, num_layers=2, dropout=0.3, use_temporal_attention=False,
                 adjacency_matrix=None):
        super(EEG_GAT, self).__init__()
        
        self.num_electrodes = num_electrodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.use_temporal_attention = use_temporal_attention
        
        # Store adjacency matrix
        if adjacency_matrix is not None:
            self.register_buffer('adjacency_matrix', adjacency_matrix)
        else:
            # Create a fully connected graph if no adjacency matrix provided
            adj = torch.ones(num_electrodes, num_electrodes)
            self.register_buffer('adjacency_matrix', adj)
        
        # Input projection layer
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        
        # First GAT layer
        self.gat_layers.append(
            MultiHeadGATLayer(hidden_dim, hidden_dim // num_heads, num_heads, dropout, concat=True)
        )
        
        # Intermediate GAT layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                MultiHeadGATLayer(hidden_dim, hidden_dim // num_heads, num_heads, dropout, concat=True)
            )
        
        # Final GAT layer (no concatenation)
        if num_layers > 1:
            self.gat_layers.append(
                MultiHeadGATLayer(hidden_dim, hidden_dim, num_heads=1, dropout=dropout, concat=False)
            )
        
        # Temporal attention mechanism
        if use_temporal_attention:
            self.temporal_attention = TemporalAttention(hidden_dim, dropout)
        
        # Graph-level pooling
        self.graph_pooling = nn.Sequential(
            nn.Linear(hidden_dim * num_electrodes, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
    
    def create_adjacency_matrix(self, distances, threshold_percentile=50):
        """
        Create adjacency matrix from distance matrix.
        
        Args:
            distances: Distance matrix [num_electrodes, num_electrodes]
            threshold_percentile: Percentile threshold for creating edges
            
        Returns:
            Binary adjacency matrix
        """
        # Convert distances to similarities (inverse relationship)
        similarities = 1.0 / (distances + 1e-8)
        
        # Set diagonal to 1 (self-connections)
        similarities.fill_diagonal_(1.0)
        
        # Create binary adjacency matrix based on threshold
        threshold = torch.percentile(similarities, threshold_percentile)
        adj_matrix = (similarities >= threshold).float()
        
        return adj_matrix
    
    def forward(self, x, distances=None):
        """
        Forward pass of the EEG-GAT model.
        
        Args:
            x: Input EEG data [batch_size, seq_len, num_electrodes] from FFT transform
            distances: Distance matrix [num_electrodes, num_electrodes]
            
        Returns:
            Classification logits [batch_size, num_classes]
        """
        batch_size = x.size(0)
        
        # Apply input projection to each electrode's features
        # x is [batch_size, 19, 354], we want to project the 354 features to hidden_dim
        x = self.input_projection(x)  # [batch_size, 19, hidden_dim]
        
        # Use stored adjacency matrix or create one if distances provided
        if distances is not None:
            adj_matrix = self.create_adjacency_matrix(distances)
        elif hasattr(self, 'adjacency_matrix'):
            adj_matrix = self.adjacency_matrix
        else:
            # Default to fully connected graph
            adj_matrix = torch.ones(self.num_electrodes, self.num_electrodes).to(x.device)
        
        # Process each sample in the batch
        batch_outputs = []
        
        for i in range(batch_size):
            # Get node features for current sample
            h = x[i]  # [num_electrodes, hidden_dim]
            
            # Apply GAT layers
            for layer_idx, gat_layer in enumerate(self.gat_layers):
                h = gat_layer(h, adj_matrix)
                
                # Apply batch normalization
                if layer_idx < len(self.batch_norms):
                    h = h.unsqueeze(0)  # Add batch dimension
                    h = self.batch_norms[layer_idx](h.transpose(1, 2)).transpose(1, 2)
                    h = h.squeeze(0)  # Remove batch dimension
                
                # Add residual connection for deeper networks
                if layer_idx > 0 and h.size(1) == self.hidden_dim:
                    h = h + x[i]  # Residual connection
            
            batch_outputs.append(h)
        
        # Stack batch outputs
        x = torch.stack(batch_outputs)  # [batch_size, num_electrodes, hidden_dim]
        
        # Graph-level pooling
        x = x.view(batch_size, -1)  # [batch_size, num_electrodes * hidden_dim]
        x = self.graph_pooling(x)  # [batch_size, hidden_dim]
        
        # Final classification
        logits = self.classifier(x)  # [batch_size, num_classes]
        
        return logits


def create_distance_adjacency_matrix(distances_path, device, distance_threshold=None):
    """
    Create adjacency matrix from electrode distances.
    
    Args:
        distances_path: Path to distances CSV file
        device: PyTorch device
        distance_threshold: Distance threshold for creating edges (None for adaptive)
        
    Returns:
        Adjacency matrix as PyTorch tensor
    """
    # Load distance matrix
    distances = np.genfromtxt(distances_path, skip_header=1, delimiter=',')[:, -1].reshape(19, 19)
    distances = torch.tensor(distances, dtype=torch.float32).to(device)
    
    if distance_threshold is None:
        # Use adaptive threshold based on median distance
        distance_threshold = torch.median(distances[distances > 0])
    
    # Create adjacency matrix (closer electrodes have stronger connections)
    adj_matrix = (distances <= distance_threshold).float()
    
    # Add self-connections
    adj_matrix.fill_diagonal_(1.0)
    
    return adj_matrix


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = EEG_GAT(
        num_electrodes=19,
        input_dim=354,  # FFT features (correct dimension)
        hidden_dim=64,
        num_classes=1,
        num_heads=8,
        num_layers=3,
        dropout=0.3
    ).to(device)
    
    # Test input
    batch_size = 32
    input_dim = 15  # Hjorth features (5 bands × 3 parameters)
    num_electrodes = 19
    
    x = torch.randn(batch_size, input_dim, num_electrodes).to(device)
    
    # Forward pass
    output = model(x)
    print(f"Model output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
