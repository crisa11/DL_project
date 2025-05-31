import torch
from torch_geometric.data import Batch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import os
from tqdm import tqdm 
import pandas as pd

def set_seed(seed=13):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)

def drop_edges(batch: Batch, drop_prob: float = 0.2) -> Batch:
    """
    Drop edges from a PyG Batch by processing each Data object individually.
    Returns a valid Batch usable with .to_data_list().
    """
    if drop_prob == 0.0:  # Optimization: if no drop, return original
        return batch

    data_list = batch.to_data_list()
    new_data_list = []

    for data in data_list:
        if data.edge_index is None or data.edge_index.size(1) == 0:
            new_data_list.append(data.clone())  # No edges to drop
            continue

        edge_index = data.edge_index
        num_edges = edge_index.size(1)

        keep_mask = torch.rand(num_edges, device=edge_index.device) > drop_prob
        new_edge_index = edge_index[:, keep_mask]

        new_data = data.clone()
        new_data.edge_index = new_edge_index

        if 'edge_attr' in data and data.edge_attr is not None:
            new_data.edge_attr = data.edge_attr[keep_mask]

        new_data_list.append(new_data)

    return Batch.from_data_list(new_data_list)

def smooth_one_hot(targets, n_classes, smoothing=0.0, device='cpu'):
    assert 0 <= smoothing < 1
    with torch.no_grad():
        confidence = 1.0 - smoothing
        label_shape = torch.Size((targets.size(0), n_classes))
        smooth_targets = torch.full(label_shape, smoothing / (n_classes - 1), device=device)
        smooth_targets.scatter_(1, targets.unsqueeze(1), confidence)
    return smooth_targets

class GCELoss(nn.Module):
    def __init__(self, q=0.5, label_smoothing=0.0):
        super().__init__()
        self.q = q
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        n_classes = logits.size(1)
        device = logits.device

        if self.label_smoothing > 0:
            targets_smoothed = smooth_one_hot(targets, n_classes, self.label_smoothing, device)
            p_t = torch.sum(probs * targets_smoothed, dim=1)
        else:
            p_t = probs[torch.arange(logits.size(0)), targets]

        loss = (1 - p_t ** self.q) / self.q
        return loss.mean()
    

class SCELoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        n_classes = logits.size(1)
        device = logits.device

        # Standard CE with label smoothing if enabled
        if self.label_smoothing > 0:
            targets_smoothed = smooth_one_hot(targets, n_classes, self.label_smoothing, device)
            ce = (-targets_smoothed * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        else:
            ce = F.cross_entropy(logits, targets)

        # Reverse CE term
        if self.label_smoothing > 0:
            rce = (-torch.sum(probs * torch.log(targets_smoothed + 1e-7), dim=1)).mean()
        else:
            one_hot = F.one_hot(targets, num_classes=n_classes).float()
            rce = (-torch.sum(probs * torch.log(one_hot + 1e-7), dim=1)).mean()

        return self.alpha * ce + self.beta * rce

def evaluate(data_loader, model, device, calculate_accuracy=False):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Iterating eval graphs", unit="batch"):
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            if calculate_accuracy:
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)
                total_loss += criterion(output, data.y).item()
            else:
                predictions.extend(pred.cpu().numpy())
    if calculate_accuracy:
        accuracy = correct / total
        return  total_loss / len(data_loader),accuracy
    return predictions

def save_predictions(predictions, test_path):
    script_dir = os.getcwd() 
    submission_folder = os.path.join(script_dir, "submission")
    test_dir_name = os.path.basename(os.path.dirname(test_path))
    
    os.makedirs(submission_folder, exist_ok=True)
    
    output_csv_path = os.path.join(submission_folder, f"testset_{test_dir_name}.csv")
    
    test_graph_ids = list(range(len(predictions)))
    output_df = pd.DataFrame({
        "id": test_graph_ids,
        "pred": predictions
    })
    
    output_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")