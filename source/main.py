import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GINConv, GATConv, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.utils import add_self_loops, degree
from loadData import GraphDataset
import os
from model import EnhancedGCN
from tqdm import tqdm
import sys
import platform
import pandas as pd
import numpy as np
import random
import copy
from collections import defaultdict
import warnings
import logging
from utils import *
from sklearn.metrics import f1_score
warnings.filterwarnings('ignore')

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
    TPU_AVAILABLE = True
    _XLA_WORLD_SIZE = xm.xrt_world_size()
    _IS_MASTER = xm.is_master_ordinal()
except ImportError:
    TPU_AVAILABLE = False
    _XLA_WORLD_SIZE = 1

def optimizer_step(optimizer):
    """One optimizer step compatible with TPU or regular PyTorch."""
    if TPU_AVAILABLE:
        xm.optimizer_step(optimizer, barrier=True)
    else:
        optimizer.step()

def mp_dataloader(loader, device):
    """Wrap DataLoader for multiprocess TPU training (no‑op elsewhere)."""
    if TPU_AVAILABLE:
        return pl.MpDeviceLoader(loader, device)
    return loader


def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)  
    return data


class GCODLoss(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, logits, u_B, y_B, y_soft_B, a_train):
        """
        logits: Tensor of shape (B, C) from f_theta(Z_B)
        u_B: Tensor of shape (B,) (trainable logits)
        y_B: Tensor of shape (B, C) (one-hot)
        y_soft_B: Tensor of shape (B, C) (soft labels)
        a_train: float (training accuracy between 0 and 1)
        """

        # L1: modified cross-entropy with soft supervision
        ce_loss = self.ce(logits, y_B.argmax(dim=1))  # standard CE
        weight_matrix = y_B * u_B.unsqueeze(1) # shape: (B, C)
        soft_ce = torch.sum(weight_matrix * y_soft_B, dim=1).mean()
        L1 = ce_loss + a_train * soft_ce

        # L2: soft label regression
        weighted_soft = y_soft_B + y_B * u_B.unsqueeze(1)  # shape: (B, C)
        L2 = F.mse_loss(weighted_soft, y_B)

        # L3: KL divergence between distributions from logits and u_B
        p = F.softmax(logits, dim=1)               # shape (B, C)
        log_p = torch.log(p + 1e-8)                # numerical stability
    
        # q from u_B: we first make sure u_B is positive
        u_B_clamped = torch.clamp(u_B, min=1e-6)   # avoid log(0)
        log_u_B = torch.log(u_B_clamped)           # shape (B,)
        q_scalar = torch.sigmoid(-log_u_B)         # shape (B,) in (0,1)
    
        # expand to match class dim
        q = q_scalar.unsqueeze(1).expand_as(p)     # shape (B, C)
    
        L3 = (1 - a_train) * F.kl_div(log_p, q, reduction='batchmean', log_target=False)

        return L1, L2, L3

def train(data_loader, model, optimizer, optimizer_u_B, criterion, device, save_checkpoints, checkpoint_path, current_epoch, u_B, a_train, warmup_epochs=10, consistency_start_epoch=15, alpha=0.3):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data in tqdm(data_loader, desc="Iterating training graphs", unit="batch"):
        data = data.to(device)
        optimizer.zero_grad()
        optimizer_u_B.zero_grad()

        logits, _ ,logits_perturbed = model(data, return_all=True, isTrain=True)
        B, C = logits.shape

        y_B = F.one_hot(data.y, num_classes=C).float()
        centroids = compute_centroids(logits.detach(), data.y, C)
        y_soft_B = compute_soft_labels(logits.detach(), centroids)

        if current_epoch < warmup_epochs:
            total_batch_loss = F.cross_entropy(logits, data.y)
        else:
            L1, L2, L3 = criterion(logits, u_B[:B], y_B, y_soft_B, a_train)
            main_loss = L1 + L3
            if current_epoch >= consistency_start_epoch:
                consistency_loss = F.mse_loss(logits, logits_perturbed)
                total_batch_loss = main_loss + alpha * consistency_loss + L2
            else:
                total_batch_loss = main_loss + L2

        total_batch_loss.backward()
        optimizer.step()
        optimizer_u_B.step()

        total_loss += total_batch_loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.y.size(0)

    if save_checkpoints:
        checkpoint_file = f"{checkpoint_path}_epoch_{current_epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_file)
        print(f"Checkpoint saved at {checkpoint_file}")

    acc = correct / total
    return total_loss / len(data_loader), acc


def evaluate(data_loader, model, device, calculate_accuracy=False):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    true_labels = []
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
                predictions.extend(pred.cpu().numpy())
                true_labels.extend(data.y.cpu().numpy())
            else:
                predictions.extend(pred.cpu().numpy())
    if calculate_accuracy:
        accuracy = correct / total
        f1 = f1_score(true_labels, predictions, average='macro')
        return  total_loss / len(data_loader),accuracy, f1
    return predictions


def main(args):
    global model, optimizer, criterion, device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model parameters - these are based on the paper's recommendations
    input_dim = 300  # Your original input dimension
    hidden_dim = 512  # Increased from 64 for better capacity
    output_dim = 6   # Number of classes
    
    # GNN+ specific parameters
    num_layers = 4   # Paper shows 3-20 layers work well
    gnn_type = 'gin'  # Can be 'gcn' or 'gin' or 'gcn-gated'
    dropout = 0.2   # Paper shows low dropout (0.0-0.2) works best
    use_positional_encoding = False
    pooling = 'mean'  # Your original uses mean pooling
    #noise_robust = args.noise_robust  # Enable if you have noisy labels
    
    # Initialize the enhanced model
    model = EnhancedGCN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        gnn_type=gnn_type,
        dropout=dropout,
        use_positional_encoding=use_positional_encoding,
        pooling=pooling,
        #noise_robust=noise_robust
    ).to(device)

    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Optimizer with weight decay as per paper
    u_B = torch.nn.Parameter(torch.zeros(args.batch_size, device=device), requires_grad=True)
    optimizer_u_B = torch.optim.Adam([u_B], lr=1)
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       mode='min',
                                                       factor=0.5,
                                                       patience=10,
                                                       min_lr=1e-5)
    
    criterion = GCODLoss(num_classes = 6)

    test_dir_name = os.path.basename(os.path.dirname(args.test_path))
    script_dir = os.getcwd()
    logs_folder = os.path.join(script_dir, "logs", test_dir_name)
    log_file = os.path.join(logs_folder, "training.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())
    checkpoint_path = os.path.join(script_dir, "checkpoints", f"model_{test_dir_name}_best.pth")
    checkpoints_folder = os.path.join(script_dir, "checkpoints", test_dir_name)
    os.makedirs(checkpoints_folder, exist_ok=True)

    # Train dataset and loader (if train_path is provided)
    if args.train_path:
        train_dataset = GraphDataset(args.train_path, transform=add_zeros)
        val_size = int(0.15 * len(train_dataset))
        train_size = len(train_dataset) - val_size
        generator = torch.Generator().manual_seed(13)
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=generator)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        # Training loop
        num_epochs = args.epochs
        best_val_accuracy = 0.0

        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        a_train = 0.0

        if args.num_checkpoints > 1:
            checkpoint_intervals = [int((i + 1) * num_epochs / args.num_checkpoints) for i in range(args.num_checkpoints)]
        else:
            checkpoint_intervals = [num_epochs]

        for epoch in range(num_epochs):
            train_loss, train_acc = train(
                                                    train_loader, model, optimizer, optimizer_u_B, criterion, device,
                                                    save_checkpoints=(epoch + 1 in checkpoint_intervals), 
                                                    checkpoint_path=os.path.join(checkpoints_folder, f"model_{test_dir_name}"),
                                                    current_epoch=epoch, u_B=u_B, a_train=a_train
                                                )
            a_train = train_acc
            val_loss, val_acc, val_f1 = evaluate(val_loader, model, device, calculate_accuracy=True)
            #print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val_loss:{val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f} ")

            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            scheduler.step(val_loss)

            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Best model updated and saved at {checkpoint_path}")
            
        
        plot_training_progress(train_losses, train_accuracies, os.path.join(logs_folder, "plots"))
        plot_training_progress(val_losses, val_accuracies, os.path.join(logs_folder, "plotsVal"))
        plot_from_logfile(log_file, os.path.join(logs_folder, "plots"))
        plot_val_from_logfile(log_file, os.path.join(logs_folder, "plotsVal"))


    elif os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded best model from {checkpoint_path}")

    # Prepare test dataset and loader
    test_dataset = GraphDataset(args.test_path, transform=add_zeros)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0) 
    # Evaluate and save test predictions
    predictions = evaluate(test_loader, model, device, calculate_accuracy=False)
    test_graph_ids = list(range(len(predictions)))  # Generate IDs for graphs

    # Save predictions to CSV
    test_dir_name = os.path.dirname(args.test_path).split(os.sep)[-1]
    output_csv_path = os.path.join(f"submission/testset_{test_dir_name}.csv")
    output_df = pd.DataFrame({
        "id": test_graph_ids,
        "pred": predictions
    })
    output_df.to_csv(output_csv_path, index=False)
    print(f"Test predictions saved to {output_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a GCN model on graph datasets.")
    parser.add_argument("--train_path", type=str, default=None, help="Path to the training dataset (optional).")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")
    parser.add_argument("--lr", type=int, default=0.001, help="Learning rate (default 0.001)")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs (default 200)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default 32)")
    parser.add_argument("--num_checkpoints", type=int, default=10, help="Number of checkpoints to save.")
    args = parser.parse_args()
    main(args)