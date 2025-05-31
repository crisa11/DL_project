import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils
from torch.nn import Linear
from torch_geometric.nn import Linear
from torch_geometric.nn import global_mean_pool

class GatedGCNLayer(torch.nn.Module):
    """Individual GatedGCN layer implementation"""

    def __init__(self, in_dim_node, in_dim_edge, out_dim, dropout, residual, aggr='add', **kwargs):
        super().__init__(**kwargs)
        self.in_dim_node, self.in_dim_edge, self.out_dim = in_dim_node, in_dim_edge, out_dim
        self.activation = nn.ReLU()
        self.A = Linear(in_dim_node, out_dim, bias=True)
        self.B = Linear(in_dim_node, out_dim, bias=True)
        self.C = Linear(in_dim_edge, out_dim, bias=True)
        self.D = Linear(in_dim_node, out_dim, bias=True)
        self.E = Linear(in_dim_node, out_dim, bias=True)

        self.act_fn_x, self.act_fn_e = self.activation, self.activation
        self.dropout_rate, self.residual_enabled, self.e_prop = dropout, residual, None
        self.aggr = aggr

        
        self.bn_node_x = nn.BatchNorm1d(out_dim)
        self.bn_edge_e = nn.BatchNorm1d(out_dim)

        self.residual_proj_node = Linear(in_dim_node, out_dim,
                                         bias=False) if residual and in_dim_node != out_dim else nn.Identity()
        self.residual_proj_edge = Linear(in_dim_edge, out_dim,
                                         bias=False) if residual and in_dim_edge != out_dim else nn.Identity()

    def _ff_block(self, x):
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def forward(self, x_in_node, edge_idx, edge_in_attr):
        x_ident, e_ident = x_in_node, edge_in_attr
        Ax, Bx, Ce, Dx, Ex = self.A(x_in_node), self.B(x_in_node), self.C(edge_in_attr), self.D(x_in_node), self.E(
            x_in_node)

        if edge_idx.numel() > 0:
            row, col = edge_idx
            e_ij = Dx[row] + Ex[col] + Ce
            self.e_prop = e_ij
            aggr_out = torch_geometric.utils.scatter(torch.sigmoid(e_ij) * Bx[col], row, 0, dim_size=x_in_node.size(0),
                                                     reduce=self.aggr)
            x_trans, e_trans = Ax + aggr_out, self.e_prop
        else:
            x_trans, e_trans = Ax, torch.zeros((0, self.out_dim), device=x_in_node.device, dtype=x_in_node.dtype)

        x_trans = self.bn_node_x(x_trans)
        if e_trans.numel() > 0:
             e_trans = self.bn_edge_e(e_trans)

        x_trans = self.act_fn_x(x_trans)
        if e_trans.numel() > 0:
            e_trans = self.act_fn_e(e_trans)

        x_trans = F.dropout(x_trans, self.dropout_rate, training=self.training)
        if e_trans.numel() > 0:
            e_trans = F.dropout(e_trans, self.dropout_rate, training=self.training)

        x_final = self.residual_proj_node(x_ident) + x_trans if self.residual_enabled else x_trans
        e_final = (self.residual_proj_edge(
            e_ident) + e_trans) if self.residual_enabled and e_trans.numel() > 0 else e_trans

        return x_final, e_final

class GatedGCN(torch.nn.Module):

    def __init__(self, emb_dim, n_layers, gnn_emb,  dropout=0.5, residual=True):
        super().__init__()

        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.gnn_emb = gnn_emb

        # Node encoder
        self.node_encoder = nn.Embedding(1, self.emb_dim)

        # Edge encoder
        self.edge_encoder = Linear(7, self.emb_dim)

        current_node_dim = self.emb_dim
        
        current_edge_dim = self.emb_dim

        self.gnn_layers = nn.ModuleList()
        for i in range(self.n_layers):
            in_node = current_node_dim if i == 0 else self.gnn_emb
            in_edge = current_edge_dim if i == 0 else self.gnn_emb
            self.gnn_layers.append(GatedGCNLayer(in_node, in_edge, self.gnn_emb, dropout, residual)
            )
        self.pool = global_mean_pool
        self.head = Linear(self.gnn_emb, 6)

    def forward(self, data):
        x, edge_idx, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        if x.dtype == torch.long:
            x_base = self.node_encoder(x.squeeze(-1))
        else:
            x_base = self.node_encoder(x.long().squeeze(-1))

        e_attr_enc = torch.empty((0, self.emb_dim), device=x.device, dtype=x_base.dtype)
        if hasattr(edge_attr, 'numel') and edge_attr.numel() > 0:
            if edge_attr.size(0) > 0:
                expected_edge_dim_from_encoder = self.edge_encoder.weight.shape[1]
                if edge_attr.shape[1] == expected_edge_dim_from_encoder:
                    e_attr_enc = self.edge_encoder(edge_attr)
                else:
                    if edge_idx.numel() > 0:
                        num_edges = edge_idx.shape[1]
                        e_attr_enc = torch.zeros((num_edges, self.emb_dim),
                                                 device=x.device, dtype=x_base.dtype)

        current_x = x_base
        current_e = e_attr_enc

        for layer in self.gnn_layers:
            current_x, current_e = layer(current_x, edge_idx, current_e)

        graph_x = self.pool(current_x, batch)
        return self.head(graph_x)