import os
import random
import time
import copy
import json
import glob
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score
from source.GNNPLUS.utils import drop_edges, set_seed



class Trainer:
    def __init__(self, model, batch_size, epochs, edge_dropping, checkpoints_path, logs_path):
        """
        Initialize the training system.

        Args:
            config: Configuration object with training parameters
            model_class: Model class to instantiate (e.g., MyLocalGatedGCN)
            checkpoints_path: Path to directory where checkpoints will be saved
            logs_path: Path to directory where training logs will be saved
        """
        set_seed()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.edge_dropping = edge_dropping
        self.checkpoints_path = checkpoints_path
        self.logs_path = logs_path

        os.makedirs(os.path.dirname(self.logs_path), exist_ok=True)

        # Create directories if they don't exist
        os.makedirs(self.checkpoints_path, exist_ok=True)
        #os.makedirs(self.logs_path, exist_ok=True)

        # For logging
        self.current_dataset_name = None
        self.training_logs = []

    def create_data_loaders(self, train_data, val_data=None):
        train_loader = None
        if train_data and len(train_data) > 0:
            train_loader = DataLoader(train_data, batch_size=self.batch_size,shuffle=True)

        val_loader = None
        if val_data and len(val_data) > 0:
            val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader

    def set_optimizers_and_schedulers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-4, weight_decay=1e-5)

        def lr_lambda_fn(current_epoch_internal):
            if current_epoch_internal < 10:
                return float(current_epoch_internal + 1) / float(10 + 1)
            progress = float(current_epoch_internal - 10) / \
                       float(max(1, self.epochs - 10))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

        scheduler = None
        if self.epochs > 10:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_fn)

        return optimizer, scheduler

    def _save_checkpoint(self, epoch, dataset_name, best=False):
        if best==True:
            model_path = os.path.join(os.path.dirname(self.checkpoints_path), f'model_{dataset_name}_best.pth')
            torch.save(self.model.state_dict(), model_path)
        else:
            model_path = os.path.join(self.checkpoints_path, f'model_{dataset_name}_epoch_{epoch}.pth')
            torch.save(self.model.state_dict(), model_path)
            
    
    def train_epoch(self, loader, optimizer, criterion):
        self.model.train()
        total_loss, processed_graphs = 0, 0

        correct = 0
        total = 0

        for data in loader:
            data = data.to(self.device)
            data = drop_edges(data, self.edge_dropping)


            optimizer.zero_grad()
            out = self.model(data)
            target_y = data.y.squeeze()
            if target_y.ndim == 0:
                target_y = target_y.unsqueeze(0)

            loss = criterion(out, target_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.num_graphs
            processed_graphs += data.num_graphs

            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)

        return total_loss / processed_graphs, correct / total


    def train(self, dataset_name, train_data, criterion, val_data=None, start_epoch=1):

        self.current_dataset_name = dataset_name

        train_loader, val_loader = self.create_data_loaders(train_data, val_data)


        optimizer, scheduler = self.set_optimizers_and_schedulers()

        print("Starting training...")
        best_val_acc = 0.0

        for epoch in range(start_epoch, self.epochs + 1):
            start_time = time.time()

             # Single Model
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)

            if epoch%20==0:
                print(f"Saved checkpoint at epoch {epoch}.")
                self._save_checkpoint(epoch, dataset_name, best=False)
           
            val_loss, val_acc = self.evaluation(val_loader, criterion)

            if val_acc > best_val_acc:
                print(f"New model saved as 'model_{dataset_name}_best.pth' with {val_acc:.4f}.")
                self._save_checkpoint(epoch, dataset_name, best=True)
                best_val_acc = val_acc

            cur_lr = optimizer.param_groups[0]['lr']

            print(f"Epoch {epoch:03d}/{self.epochs:03d} | train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | val_acc: {val_acc:.4f} | LR: {cur_lr:.1e}")

            if scheduler: scheduler.step()
            
            with open(self.logs_path, 'a') as f:
                f.write(f"Epoch {epoch}: train_loss: {train_loss}, val_loss: {val_loss}, val_acc: {val_acc}\n")

        print("\nTraining finished.")

    def evaluation(self, loader, criterion):
        self.model.eval()
        total_loss, processed_graphs = 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for data_batch in loader:
                data_batch = data_batch.to(self.device)
                out = self.model(data_batch)

                preds = out.argmax(dim=1)
                all_preds.append(preds.cpu())

                target_y = data_batch.y.squeeze()
                if target_y.ndim == 0:
                    target_y = target_y.unsqueeze(0)

                all_labels.append(target_y.cpu())

                valid_targets = target_y != -1
                if valid_targets.any():
                    loss = criterion(out[valid_targets], target_y[valid_targets])
                    total_loss += loss.item() * valid_targets.sum().item()
            
                processed_graphs += data_batch.num_graphs if hasattr(data_batch, 'num_graphs') else data_batch.x.size(0)

            all_preds_np = torch.cat(all_preds).numpy()
            all_labels_np = torch.cat(all_labels).numpy()

            valid_indices = all_labels_np != -1
            num_valid_targets = np.sum(valid_indices)
        
            if num_valid_targets > 0:
                accuracy = accuracy_score(all_labels_np[valid_indices], all_preds_np[valid_indices])

            effective_loss = total_loss / num_valid_targets if num_valid_targets > 0 else 0.0
        
        return effective_loss, accuracy