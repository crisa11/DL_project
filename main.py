import argparse
import os
import torch
from torch_geometric.loader import DataLoader
from source.loadData import GraphDataset
from source.utils import set_seed
import pandas as pd
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.utils.data import random_split
from source.models import GNN 
from source.utils import NoisyCrossEntropyLoss, GCELoss
from source.GNNPLUS.run import run


# Set the random seed
set_seed()

def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

def train(data_loader, model, optimizer, criterion, device, save_checkpoints, checkpoint_path, current_epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for data in tqdm(data_loader, desc="Iterating training graphs", unit="batch"):
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

        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())
    
    f1 = f1_score(all_labels, all_preds, average='macro')
    # Save checkpoints if required
    if save_checkpoints:
        checkpoint_file = f"{checkpoint_path}_epoch_{current_epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_file)
        print(f"Checkpoint saved at {checkpoint_file}")

    accuracy = correct / total

    return total_loss / len(data_loader), accuracy, f1

def evaluate(data_loader, model, device, calculate_accuracy=False):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    targets = []
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for data in tqdm(data_loader, desc="Iterating eval graphs", unit="batch"):
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())
            if calculate_accuracy:
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)
                total_loss += criterion(output, data.y)
                
    if calculate_accuracy:
        accuracy = correct / total
        avg_loss = total_loss / len(data_loader)
        #f1 = f1_score(targets, predictions, average='macro')
        return avg_loss, accuracy
    return predictions

def save_predictions(predictions, test_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
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


def plot_training_progress(train_losses, train_accuracies, output_dir):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 6))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Training Loss", color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Training Accuracy", color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy per Epoch')

    # Save plots in the current directory
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_progress.png"))
    plt.close()

def main(args):

    # Identify dataset folder (A, B, C, or D)
    test_dir_name = os.path.basename(os.path.dirname(args.test_path))

    if test_dir_name == 'A' or test_dir_name == 'B' or test_dir_name == 'C' :
        run(args)
    else:

        # Get the directory where the main script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
        num_checkpoints = args.num_checkpoints if args.num_checkpoints else 3

        
        # Setup logging
        logs_folder = os.path.join(script_dir, "logs", test_dir_name)
        log_file = os.path.join(logs_folder, "train.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
        logging.getLogger().addHandler(logging.StreamHandler())  # Console output as well

        # Define checkpoint path relative to the script's directory
        checkpoint_path = os.path.join(script_dir, "checkpoints", f"model_{test_dir_name}_best.pth")
        checkpoints_folder = os.path.join(script_dir, "checkpoints", test_dir_name)
        os.makedirs(checkpoints_folder, exist_ok=True)

        if args.gnn == 'gin':
            model = GNN(gnn_type = 'gin', num_class = 6, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
        elif args.gnn == 'gin-virtual':
            model = GNN(gnn_type = 'gin', num_class = 6, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
        elif args.gnn == 'gcn':
            model = GNN(gnn_type = 'gcn', num_class = 6, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
        elif args.gnn == 'gcn-virtual':
            model = GNN(gnn_type = 'gcn', num_class = 6, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
        else:
            raise ValueError('Invalid GNN type')
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        if args.baseline_mode == 2:
            criterion = NoisyCrossEntropyLoss(args.noise_prob)
        elif args.baseline_mode == 1:
            criterion = torch.nn.CrossEntropyLoss()
        elif args.baseline_mode==3 and test_dir_name=='B':
            criterion = GCELoss(q=0.9)
        else:
            criterion = GCELoss()

        # Load pre-trained model for inference
        if os.path.exists(checkpoint_path) and not args.train_path:

            if test_dir_name=='D':
                model = GNN(gnn_type='gin', num_class = 6, num_layer =5, emb_dim =300, drop_ratio =0.5, virtual_node = True).to(device)


            model.load_state_dict(torch.load(checkpoint_path))
            print(f"Loaded best model from {checkpoint_path}")

        # Prepare test dataset and loader
        test_dataset = GraphDataset(args.test_path, transform=add_zeros)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        log_train = f"logs/{test_dir_name}/train.log"
        os.makedirs(os.path.dirname(log_train), exist_ok=True)

        # If train_path is provided, train the model
        if args.train_path:
            full_dataset = GraphDataset(args.train_path, transform=add_zeros)
            val_size = int(0.2 * len(full_dataset))
            train_size = len(full_dataset) - val_size

            
            generator = torch.Generator().manual_seed(12)
            train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

            # Training loop
            num_epochs = args.epochs
            best_accuracy = 0.0
            train_losses = []
            train_accuracies = []
            val_losses = []
            val_accuracies = []

            # Calculate intervals for saving checkpoints
            if num_checkpoints > 1:
                checkpoint_intervals = [int((i + 1) * num_epochs / num_checkpoints) for i in range(num_checkpoints)]
            else:
                checkpoint_intervals = [num_epochs]

            for epoch in range(num_epochs):
                train_loss, train_acc, train_f1 = train(
                    train_loader, model, optimizer, criterion, device,
                    save_checkpoints=(epoch + 1 in checkpoint_intervals),
                    checkpoint_path=os.path.join(checkpoints_folder, f"model_{test_dir_name}"),
                    current_epoch=epoch
                )
                val_loss, val_acc = evaluate(val_loader, model, device, calculate_accuracy=True)
                logging.info(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                
                # Save logs for training progress
                train_losses.append(train_loss)
                train_accuracies.append(train_acc)
                with open(log_train, 'a') as f:
                    f.write(f"Epoch {epoch + 1}: train_loss: {train_loss}, train_acc: {train_acc}, val_acc: {val_acc}\n")

                # Save best model
                if val_acc > best_accuracy:
                    best_accuracy = val_acc
                    torch.save(model.state_dict(), checkpoint_path)
                    print(f"Best model updated and saved at {checkpoint_path}")

            # Plot training progress in current directory
            plot_training_progress(train_losses, train_accuracies, os.path.join(logs_folder, "plots"))
            plot_training_progress(val_losses, val_accuracies, os.path.join(logs_folder, "plotsVal"))

        # Generate predictions for the test set using the best model
        model.load_state_dict(torch.load(checkpoint_path))
        predictions = evaluate(test_loader, model, device, calculate_accuracy=False)
        save_predictions(predictions, args.test_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate GNN models on graph datasets.")
    parser.add_argument("--train_path", type=str, help="Path to the training dataset (optional).")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")
    parser.add_argument("--num_checkpoints", type=int,  default=5, help="Number of checkpoints to save during training.")
    parser.add_argument('--device', type=int, default=1, help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin-virtual', help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5, help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5, help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300, help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--baseline_mode', type=int, default= 3, help='1 (CE), 2 (Noisy CE), 3 (GCE) (default=3)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 10)')
    
    args = parser.parse_args()
    main(args)