import argparse
import os
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from source.GNNPLUS.model import GatedGCN
from source.GNNPLUS.trainer import Trainer
from source.GNNPLUS.loadData import GraphDataset,add_zeros, load_dataset
from source.GNNPLUS.utils import set_seed, GCELoss, SCELoss, evaluate, save_predictions


def run(args):
    #args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    set_seed()

    # Load test data
    test_dir_name = os.path.basename(os.path.dirname(args.test_path))
    dataset_splits = load_dataset(train_path=args.train_path, val_split_ratio=0.2, dataset_name=test_dir_name)

    val_data = dataset_splits.get('val', None)
    train_data = dataset_splits.get('train', None)

    log_dir = os.path.join(script_dir, "../../logs", test_dir_name)
    os.makedirs(log_dir, exist_ok=True)
    log_train = os.path.join(log_dir, "train.txt")
    checkpoint_path = os.path.join(script_dir, "../../checkpoints", f"model_{test_dir_name}_best.pth")
    checkpoints_folder = os.path.join(script_dir, "../../checkpoints", test_dir_name)
    os.makedirs(checkpoints_folder, exist_ok=True)
    
    if test_dir_name == 'A':
        model = GatedGCN(emb_dim=128, n_layers=3, gnn_emb=256)
        criterion = GCELoss()
        trainer = Trainer(model, batch_size=32, epochs=300, edge_dropping=0.2, checkpoints_path=checkpoints_folder, logs_path=log_train)
    elif test_dir_name == 'B':
        model = GatedGCN(emb_dim=128, n_layers=3, gnn_emb=128)
        criterion = GCELoss(q=0.9)
        trainer = Trainer(model, batch_size=32, epochs=300, edge_dropping=0.2, checkpoints_path=checkpoints_folder, logs_path=log_train) 
    elif test_dir_name == 'C':
        model = GatedGCN(emb_dim=256, n_layers=3, gnn_emb=512)
        criterion = GCELoss()
        trainer = Trainer(model, batch_size=32, epochs=300, edge_dropping=0.2, checkpoints_path=checkpoints_folder, logs_path=log_train) 
    else:
        model = GatedGCN(emb_dim=256, n_layers=3, gnn_emb=256)
        criterion = GCELoss()
        trainer = Trainer(model, batch_size=32, epochs=300, edge_dropping=0.25, checkpoints_path=checkpoints_folder, logs_path=log_train) 


    if args.train_path and train_data and val_data:
        trainer.train(dataset_name=test_dir_name, train_data=train_data, val_data=val_data,criterion=criterion)
        test_dataset = GraphDataset(args.test_path, transform=add_zeros)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        model.load_state_dict(torch.load(checkpoint_path))
        predictions = evaluate(test_loader, model, device, calculate_accuracy=False)
        save_predictions(predictions, args.test_path)
    elif checkpoint_path:
        test_dataset = GraphDataset(args.test_path, transform=add_zeros)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        model.load_state_dict(torch.load(checkpoint_path,  map_location=torch.device('cpu')))
        predictions = evaluate(test_loader, model, device, calculate_accuracy=False)
        save_predictions(predictions, args.test_path)

#if __name__ == "__main__":
#    parser = argparse.ArgumentParser(description="Train and evaluate GNN models on graph datasets.")
#    parser.add_argument("--train_path", type=str, help="Path to the training dataset (optional).")
#    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")
#    
#    args = parser.parse_args()
#    main(args)