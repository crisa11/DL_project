import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.nn import GCNConv, GINConv, GatedGraphConv, global_mean_pool, global_add_pool
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform
from torch_geometric.utils import add_self_loops, degree
from conv import GNN_node, GNN_node_Virtualnode


# ============ ENHANCED ARCHITECTURE ============

class SymmetricCELoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target)
        pred = F.softmax(logits, dim=1)
        one_hot = F.one_hot(target, num_classes=logits.size(1)).float()
        rce = (-torch.sum(pred * torch.log(one_hot + 1e-6), dim=1)).mean()
        return self.alpha * ce + self.beta * rce

class RandomWalkPositionalEncoding(nn.Module):
    """
    Enhancement 6: Positional Encoding using Random Walk
    This helps the model understand the structure of the graph better
    """
    def __init__(self, walk_length=16, embed_dim=16):
        super().__init__()
        self.walk_length = walk_length
        self.embedding = nn.Linear(walk_length, embed_dim)
        
    def forward(self, edge_index, num_nodes, batch=None):
        """Compute random walk features for each node"""
        device = edge_index.device
        
        # Add self-loops for the random walk
        edge_index_with_loops, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        
        # Compute degree for normalization
        row, col = edge_index_with_loops
        deg = degree(col, num_nodes, dtype=torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        # Initialize random walk probabilities
        rw_probs = torch.zeros(num_nodes, self.walk_length, device=device)
        
        # Starting probability (uniform within each graph)
        if batch is None:
            prob = torch.ones(num_nodes, device=device) / num_nodes
        else:
            prob = torch.zeros(num_nodes, device=device)
            for b in batch.unique():
                mask = batch == b
                prob[mask] = 1.0 / mask.sum()
        
        # Perform random walks
        for step in range(self.walk_length):
            rw_probs[:, step] = prob
            # Normalized adjacency matrix multiplication
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            new_prob = torch.zeros_like(prob)
            for i, (r, c) in enumerate(zip(row, col)):
                new_prob[c] += prob[r] * norm[i]
            prob = new_prob * 0.9 + prob * 0.1  # Add slight self-loop probability
            
        # Transform to embeddings
        return self.embedding(rw_probs)

class FeedForwardBlock(nn.Module):
    """
    Enhancement 5: Feed-Forward Network
    Adds more expressiveness to each layer
    """
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.norm(x + residual)  # Residual connection within FFN
        return x

class GatedGCNBlock(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=3):
        super().__init__()
        self.lin   = nn.Linear(in_dim, out_dim)
        self.ggnn  = GatedGraphConv(out_channels=out_dim, num_layers=num_layers)
        self.act   = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.lin(x)                   # 1) proietta in out_dim
        x = self.ggnn(x, edge_index)      # 2) messaggi “gated”
        x = self.act(x)                   # 3) attivazione
        return x

class GNNPlusLayer(nn.Module):
    """
    A single GNN+ layer incorporating all enhancements
    """
    def __init__(self, in_dim, out_dim, gnn_type='gcn', dropout=0.1, 
                 use_edge_feat=True, use_ffn=True, use_residual=True):
        super().__init__()
        
        self.use_ffn = use_ffn
        self.use_residual = use_residual and (in_dim == out_dim)
        
        # Choose GNN type
        if gnn_type == 'gcn':
            self.conv = GCNConv(in_dim, out_dim)
        elif gnn_type == 'gcn-gated':
            # Usa il wrapper per Linear → GatedGraphConv → ReLU
            self.conv = GatedGCNBlock(in_dim, out_dim, num_layers=3)
        elif gnn_type == 'gin':
            gin_nn = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim)
            )
            self.conv = GINConv(gin_nn)
        
        # Enhancement 2: Normalization
        self.norm = nn.BatchNorm1d(out_dim)
        
        # Enhancement 3: Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Enhancement 5: Feed-forward network
        if self.use_ffn:
            self.ffn = FeedForwardBlock(out_dim, dropout)
            
        # For residual connections when dimensions don't match
        if use_residual and in_dim != out_dim:
            self.residual_proj = nn.Linear(in_dim, out_dim)
            self.use_residual = True
    
    def forward(self, x, edge_index, edge_attr=None):
        # Store input for residual
        residual = x
        
        # Apply GNN convolution
        x = self.conv(x, edge_index)
        
        # Apply normalization and activation
        x = self.norm(x)
        x = F.relu(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Enhancement 4: Residual connection
        if self.use_residual:
            if hasattr(self, 'residual_proj'):
                residual = self.residual_proj(residual)
            x = x + residual
        
        # Enhancement 5: Feed-forward network
        if self.use_ffn:
            x = self.ffn(x)
            
        return x


class EnhancedGCN(nn.Module):
    """
    GNN+ enhanced model based on the paper
    This replaces your SimpleGCN with all 6 enhancements
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=5, 
                 gnn_type='gin', dropout=0.1, use_positional_encoding=True,
                 pooling='mean', noise_robust=False):
        super().__init__()
        
        # For your zero-feature nodes
        self.embedding = nn.Embedding(1, input_dim)
        
        # Enhancement 6: Positional encoding
        self.use_pe = use_positional_encoding
        if self.use_pe:
            self.pe_encoder = RandomWalkPositionalEncoding(walk_length=16, embed_dim=16)
            actual_input_dim = input_dim + 16
        else:
            actual_input_dim = input_dim
        
        # Input projection
        self.input_proj = nn.Linear(actual_input_dim, hidden_dim)
        
        # Stack of GNN+ layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = GNNPlusLayer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                gnn_type=gnn_type,
                dropout=dropout,
                use_ffn=True,
                use_residual=True
            )
            self.layers.append(layer)
        
        # Output layers
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Pooling method
        self.pooling = pooling
        if pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling == 'add':
            self.pool = global_add_pool
            
        # For noise-robust training
        #self.noise_robust = noise_robust
        #self.symmetric_ce = SymmetricCELoss(alpha=1.0, beta=1.0)

    def apply_layers(self, node_feats, e_idx, e_attr):
        h = node_feats
        for layer in self.layers:
            # se il layer ha bisogno di edge_attr, passa anche quello
            h = layer(h, e_idx, edge_attr=e_attr)
        return h
    
    def drop_edges(self, edge_index, edge_attr, drop_prob):
        mask = torch.rand(edge_index.size(1), device=edge_index.device) >= drop_prob
        return edge_index[:, mask], edge_attr[mask]
        
    def forward(self, data, return_all=False, isTrain=False):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Initial embedding for zero features
        x = self.embedding(x)

        if isTrain:
            noise = torch.randn_like(x)*0.2
            x_noisy = x + noise
        else:
            x_noisy = None
    
        # Add positional encoding if enabled
        if self.use_pe:
            pe = self.pe_encoder(edge_index, x.size(0), batch)
            x = torch.cat([x, pe], dim=-1)
            if isTrain:
                x_noisy = torch.cat([x_noisy, pe], dim=-1)
        
        # Project to hidden dimension
        x = self.input_proj(x)
        if isTrain:
            x_noisy = self.input_proj(x_noisy)
        
        # Apply GNN+ layers
        # 5a) Forward “pulito”
        h_clean = self.apply_layers(x, edge_index, edge_attr)
        g_clean = self.pool(h_clean, batch)
        out_clean = self.output_mlp(g_clean)

        if not isTrain:
            return out_clean
        
        # 5b) Forward “rumoroso” senza caduta di archi
        h_noisy = self.apply_layers(x_noisy, edge_index, edge_attr)
        g_noisy = self.pool(h_noisy, batch)
        out_noisy = self.output_mlp(g_noisy)

        # 5c) Forward “rumoroso + edge-dropping”
        ei_pert, ea_pert = self.drop_edges(edge_index, edge_attr, drop_prob=0.2)
        h_pert = self.apply_layers(x_noisy, ei_pert, ea_pert)
        g_pert = self.pool(h_pert, batch)
        out_pert = self.output_mlp(g_pert)
        
        # 6) Cosa restituire
        if return_all:
            # ad esempio tuple (pulito, noisy, noisy+perturbed)
            return out_clean, out_noisy, out_pert
        else:
            return out_clean
    
    '''
    def compute_loss(self, pred, target):
        """
        Compute loss with optional noise robustness
        """
        if self.noise_robust:
            # Symmetric Cross Entropy for noise robustness
            return self.symmetric_ce(pred, target)
        else:
            return F.cross_entropy(pred, target)
    '''
