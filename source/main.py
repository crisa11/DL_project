import argparse
import torch
from torch_geometric.loader import DataLoader
from loadData import GraphDataset
import os
import pandas as pd
from model import *
import random
import numpy as np
from tqdm import tqdm
from utils import *
from losses import *
import logging
from torch.utils.data import random_split
import gc


def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)  
    return data


def train(data_loader, current_epoch, checkpoint_path="./checkpoints", save_checkpoints=True):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for data in tqdm(data_loader, desc='Iterating training graphs', unit='batch'):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.y.size(0)
        del output, loss, data
    if save_checkpoints:
        checkpoint_file = f"{checkpoint_path}_epoch_{current_epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_file)
        print(f"Checkpoint saved at {checkpoint_file}")

    return total_loss / len(data_loader), correct/total


def evaluate(data_loader, calculate_accuracy=True):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    total_loss = 0
    with torch.no_grad():
        for data in tqdm(data_loader, desc='Iterating evaluation graphs', unit='batch', leave=False):
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
        return total_loss/len(data_loader), accuracy
    
    return predictions


def main(args):
    global model, optimizer, criterion, device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

    # Parameters for the GNN model
    input_dim  = 1      # dummy feature dimension (i nodi non hanno feature, usiamo embedding su 1 “label”)
    hidden_dim = 128    # dimensione degli embedding/node hidden
    output_dim = 6      # numero di classi da predire
    dropout    = 0.5    # dropout ratio
    batch_size = 32     # dimensione del batch
    num_layer  = 5      # numero di message‐passing layers
    gnn   = 'gcn'  # 'gin' | 'gin-virtual' | 'gcn' | 'gcn-virtual'

    # Initialize the model, optimizer, and loss criterion
    #model = CustomGCN(input_dim, hidden_dim, output_dim, dropout).to(device)  
    if gnn == 'gin':
        model = GNN(gnn_type='gin', num_class=output_dim, num_layer=num_layer, emb_dim=hidden_dim, drop_ratio=dropout, virtual_node=False).to(device)
    elif gnn == 'gin-virtual':
        model = GNN(gnn_type='gin', num_class=output_dim, num_layer=num_layer, emb_dim=hidden_dim, drop_ratio=dropout, virtual_node=True).to(device)
    elif gnn == 'gcn':
        model = GNN(gnn_type='gcn', num_class=output_dim, num_layer=num_layer, emb_dim=hidden_dim, drop_ratio=dropout, virtual_node=False).to(device)
    elif gnn == 'gcn-virtual':
        model = GNN(gnn_type='gcn', num_class=output_dim, num_layer=num_layer, emb_dim=hidden_dim, drop_ratio=dropout, virtual_node=True).to(device)
    else:
        raise ValueError('Invalid GNN type')
    
    if args.no_half:
        model = model.float()
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       mode='min',
                                                       factor=0.5,
                                                       patience=10,
                                                       min_lr=1e-5)
    #criterion = torch.nn.CrossEntropyLoss()
    criterion = SCELoss(alpha=1.0, beta=1.0, device=device, num_classes=6)

    # Prepare test dataset and loader
    test_dataset = GraphDataset(args.test_path, transform=add_zeros)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)  

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
        val_size = int(0.2 * len(train_dataset))
        train_size = len(train_dataset) - val_size
        generator = torch.Generator().manual_seed(12)
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=generator)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        # Training loop
        num_epochs = 100
        best_val_accuracy = 0.0

        #train_losses = []
        #train_accuracies = []
        #val_losses = []
        #val_accuracies = []

        if args.num_checkpoints > 1:
            checkpoint_intervals = [int((i + 1) * num_epochs / args.num_checkpoints) for i in range(args.num_checkpoints)]
        else:
            checkpoint_intervals = [num_epochs]

        for epoch in range(num_epochs):
            train_loss, train_acc = train(train_loader, epoch, save_checkpoints=(epoch + 1 in checkpoint_intervals), 
                                          checkpoint_path=os.path.join(checkpoints_folder, f"model_{test_dir_name}"))
            val_loss, val_acc = evaluate(val_loader, calculate_accuracy=True)
            #print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val_loss:{val_loss:.4f}, Val Acc: {val_acc:.4f}, ")

            #train_losses.append(train_loss)
            #train_accuracies.append(train_acc)
            #val_losses.append(val_loss)
            #val_accuracies.append(val_acc)

            scheduler.step(val_loss)

            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Best model updated and saved at {checkpoint_path}")
            
            if torch.backends.mps.is_available():
                gc.collect()
                torch.mps.empty_cache()
        
        #plot_training_progress(train_losses, train_accuracies, os.path.join(logs_folder, "plots"))
        #plot_training_progress(val_losses, val_accuracies, os.path.join(logs_folder, "plotsVal"))
        plot_from_logfile(log_file, os.path.join(logs_folder, "plots"))
        plot_val_from_logfile(log_file, os.path.join(logs_folder, "plotsVal"))


    elif os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded best model from {checkpoint_path}")

    
    # Evaluate and save test predictions
    predictions = evaluate(test_loader, calculate_accuracy=False)
    test_graph_ids = list(range(len(predictions)))  # Generate IDs for graphs

    # Save predictions to CSV
    test_dir_name = os.path.dirname(args.test_path).split(os.sep)[-1]
    output_csv_path = os.path.join(f"submission/testset_{test_dir_name}.csv")
    output_df = pd.DataFrame({
        "GraphID": test_graph_ids,
        "Class": predictions
    })
    output_df.to_csv(output_csv_path, index=False)
    print(f"Test predictions saved to {output_csv_path}")


if __name__ == "__main__":
    seed_torch()
    parser = argparse.ArgumentParser(description="Train and evaluate a GCN model on graph datasets.")
    parser.add_argument("--train_path", type=str, default=None, help="Path to the training dataset (optional).")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")
    parser.add_argument("--num_checkpoints", type=int, default=10, help="Number of checkpoints to save.")
    parser.add_argument('--no-half', action='store_true')
    args = parser.parse_args()
    main(args)