import json
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, degree
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import argparse
from torch_geometric.data import Dataset, Data
import gzip
import os

def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

class GraphDataset(Dataset):
    def __init__(self, filename, transform=None, pre_transform=None):
        self.raw = filename
        self.graphs = self.loadGraphs(self.raw)
        super().__init__(None, transform, pre_transform)

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]

    @staticmethod
    def loadGraphs(path):
        print(f"Loading graphs from {path}...")
        print("This may take a few minutes, please wait...")
        with gzip.open(path, "rt", encoding="utf-8") as f:
            graphs_dicts = json.load(f)
        graphs = []
        for graph_dict in tqdm(graphs_dicts, desc="Processing graphs", unit="graph"):
            graphs.append(dictToGraphObject(graph_dict))
        return graphs


def dictToGraphObject(graph_dict):
    edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long)
    edge_attr = torch.tensor(graph_dict["edge_attr"], dtype=torch.float) if graph_dict["edge_attr"] else None
    num_nodes = graph_dict["num_nodes"]
    y = torch.tensor(graph_dict["y"][0], dtype=torch.long) if graph_dict["y"] is not None else None
    return Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes, y=y)


def get_rw_landing_probs(edge_index, num_nodes, k_max=16):
    if num_nodes == 0: return torch.zeros((0, k_max), device=edge_index.device)
    if edge_index.numel() == 0: return torch.zeros((num_nodes, k_max), device=edge_index.device)

    source, _ = edge_index[0], edge_index[1]
    deg = degree(source, num_nodes=num_nodes, dtype=torch.float)
    deg_inv = deg.pow(-1.)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)

    try:
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)
    except RuntimeError as e:
        max_idx = edge_index.max().item() if edge_index.numel() > 0 else -1
        print(f"Error in to_dense_adj: {e}. Max_idx: {max_idx}, Num_nodes: {num_nodes}. Returning zeros for RWSE.")
        return torch.zeros((num_nodes, k_max), device=edge_index.device)

    P_dense = deg_inv.view(-1, 1) * adj
    rws_list = []
    if num_nodes == 0: return torch.zeros((0, k_max), device=edge_index.device)
    Pk = torch.eye(num_nodes, device=edge_index.device)

    for _ in range(1, k_max + 1):
        if Pk.numel() == 0 or P_dense.numel() == 0:
            return torch.zeros((num_nodes, k_max), device=edge_index.device)
        try:
            Pk = Pk @ P_dense
        except RuntimeError as e:
            print(
                f"RuntimeError during Pk @ P_dense: {e}. Shapes Pk:{Pk.shape}, P_dense:{P_dense.shape}. Returning zeros.")
            return torch.zeros((num_nodes, k_max), device=edge_index.device)
        rws_list.append(torch.diag(Pk))

    return torch.stack(rws_list, dim=1) if rws_list else torch.zeros((num_nodes, k_max), device=edge_index.device)


def process_graph_data(graph_dict, is_test_set=False, graph_idx_info=""):
    num_nodes = graph_dict.get('num_nodes', 0)
    if not isinstance(num_nodes, int) or num_nodes < 0: num_nodes = 0

    x = torch.zeros(num_nodes, 1, dtype=torch.long)

    raw_edge_index = graph_dict.get('edge_index', [])
    raw_edge_attr = graph_dict.get('edge_attr', [])
    edge_attr_dim = graph_dict.get('edge_attr_dim', 7)
    
    if not isinstance(edge_attr_dim, int) or edge_attr_dim <= 0: edge_attr_dim = 7

    if num_nodes == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, edge_attr_dim), dtype=torch.float)
    else:
        edge_index = torch.tensor(raw_edge_index, dtype=torch.long)
        edge_attr = torch.tensor(raw_edge_attr, dtype=torch.float)
        if edge_index.numel() > 0 and edge_index.shape[0] != 2:
            print(f"Warning: Invalid edge_index shape for graph {graph_idx_info}. Clearing edges.")
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, edge_attr_dim), dtype=torch.float)

        if edge_attr.numel() > 0:
            if edge_attr.shape[1] != edge_attr_dim:
                print(
                    f"Warning: Mismatch edge_attr_dim (expected {edge_attr_dim}, got {edge_attr.shape[1]}) for graph {graph_idx_info}. Attempting to adapt or clearing.")
                if 'edge_attr_dim' in graph_dict: 
                    edge_attr = torch.empty((0, edge_attr_dim), dtype=torch.float)
                    if edge_index.shape[1] > 0:
                        print(f"  Cleared edge_attr for {graph_idx_info} due to dim mismatch with specification.")

        if edge_attr.numel() > 0 and edge_index.shape[1] != edge_attr.shape[0]:
            print(
                f"Warning: Mismatch edge_index/edge_attr count for graph {graph_idx_info}. Clearing edge_attr if no edges, or both if inconsistent.")
            if edge_index.shape[1] == 0:
                edge_attr = torch.empty((0, edge_attr_dim), dtype=torch.float)

    y_val_raw = graph_dict.get('y')
    y_val = -1
    if is_test_set:  
        y_val = -1
    elif y_val_raw is not None:
        temp_y = y_val_raw
        while isinstance(temp_y, list):  
            if len(temp_y) == 1:
                temp_y = temp_y[0]
            else:
                temp_y = -1
                break  
        if isinstance(temp_y, int):
            y_val = temp_y
        else:
            y_val = -1 

    if y_val == -1 and not is_test_set:
        print(f"Warning: 'y' missing or malformed for TRAIN/VAL graph {graph_idx_info}. Using -1. Data issue.")
    y = torch.tensor([y_val], dtype=torch.long)

    data_obj = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, num_nodes=num_nodes)

    if data_obj.num_nodes > 0 and data_obj.num_edges > 0:
        data_obj.rwse_pe = get_rw_landing_probs(data_obj.edge_index, data_obj.num_nodes, k_max=16)
    else:
        data_obj.rwse_pe = torch.zeros((data_obj.num_nodes, 16))
    return data_obj


def load_from_gz(gz_file_path):
    try:
        with gzip.open(gz_file_path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {gz_file_path}: {e}")
        return None

def load_dataset(train_path=None, val_split_ratio=0.2, dataset_name="dataset"):

    print(f"Dataset: {dataset_name}")

    dataset_splits = {'train': [], 'val': []}
    
    if train_path is None:
        print("No training path provided. Skipping data loading and returning empty splits.")
        return dataset_splits

    print(f"Loading training data from: {train_path}")
    train_json_list = load_from_gz(train_path)

    raw_train_graphs = []
    if train_json_list is not None:
        for i, g_data in tqdm(enumerate(train_json_list), total=len(train_json_list),
                                desc=f"  {dataset_name} Train Data"):
            if not isinstance(g_data, dict):
                continue
            processed_graph = process_graph_data(g_data, is_test_set=False,
                                                    graph_idx_info=f"{dataset_name}_train_{i}")
            raw_train_graphs.append(processed_graph)
    else:
        print(f"Failed to load training data from {train_path}")

    dataset_splits['train'] = []
    dataset_splits['val'] = []

    train_labels = [g.y.item() for g in raw_train_graphs if g.y.item() != -1]

    can_stratify = False
    if train_labels:
        unique_labels, counts = np.unique(train_labels, return_counts=True)
        if len(unique_labels) > 1 and all(c >= 2 for c in counts):
            can_stratify = True

    
    if can_stratify:
        dataset_splits['train'], dataset_splits['val'] = train_test_split(
            raw_train_graphs, test_size=val_split_ratio, random_state=42, stratify=train_labels
        )
        print(f"Used stratified split for {dataset_name}")
    else:
        dataset_splits['train'], dataset_splits['val'] = train_test_split(
            raw_train_graphs, test_size=val_split_ratio, random_state=42
        )
        print(f"Used random split for {dataset_name}")

    return dataset_splits