import random
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import re


def seed_torch(seed=1111):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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


def plot_from_logfile(log_file_path, output_dir, title_prefix="Training"):
    epochs = []
    losses = []
    accuracies = []

    with open(log_file_path, "r") as f:
        for line in f:
            match = re.search(
                r"Epoch\s+(\d+)/\d+,\s+Loss:\s+([\d.]+),\s+Train Acc:\s+([\d.]+)",
                line
            )
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                acc = float(match.group(3))
                epochs.append(epoch)
                losses.append(loss)
                accuracies.append(acc)

    if not epochs:
        print(f"[WARNING] No metrics found in log: {log_file_path}")
        return

    plt.figure(figsize=(12, 6))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label="Loss", color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{title_prefix} Loss per Epoch')

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, label="Accuracy", color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{title_prefix} Accuracy per Epoch')

    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{title_prefix.lower()}_progress.png"))
    plt.close()

def plot_val_from_logfile(log_file_path, output_dir, title_prefix="Validation"):
    epochs = []
    val_losses = []
    val_accuracies = []

    with open(log_file_path, "r") as f:
        for line in f:
            match = re.search(
                r"Epoch\s+(\d+)/\d+.*?Val_loss:([\d.]+),\s+Val Acc:\s+([\d.]+)",
                line
            )
            if match:
                epoch = int(match.group(1))
                val_loss = float(match.group(2))
                val_acc = float(match.group(3))
                epochs.append(epoch)
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)

    if not epochs:
        print(f"[WARNING] No validation metrics found in log: {log_file_path}")
        return

    plt.figure(figsize=(12, 6))

    # Plot val loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, val_losses, label="Val Loss", color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Val Loss')
    plt.title(f'{title_prefix} Loss per Epoch')

    # Plot val accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label="Val Accuracy", color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Val Accuracy')
    plt.title(f'{title_prefix} Accuracy per Epoch')

    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{title_prefix.lower()}_progress.png"))
    plt.close()
