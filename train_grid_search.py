import argparse
import torch
import torch.optim as optim
from Dataset import load_cifar10, load_food101
from model_ViT import CurriculumVisionTransformer
from model_CnvNext import CurriculumConvNeXt
from Explainability import Explainability
from medmnist_loader import Dataset
import torch.nn as nn
import wandb
from torchmetrics import Accuracy, F1Score
import numpy as np
import random
import itertools
import os


# Set a random seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Main function
def main():
    parser = argparse.ArgumentParser(description='Train model with curriculum learning and explainability.')

    # Add argument for dataset
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'food101', 'bloodmnist'], required=True,
                        help='Dataset to use (cifar10, food101, bloodmnist)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--model', type=str, choices=['vit', 'cnn'], default='vit',
                        help='Choose the model architecture (vit or cnn)')

    args = parser.parse_args()

    # Grid Search Parameters
    learning_rates = [0.0005, 0.0001]
    batch_sizes = [128, 256]
    weight_decays = [0.0, 1e-4]

    # Initialize tracking variables
    best_val_acc = 0.0
    best_hyperparams = None
    best_model_path = "best_model.pth"


    columns = ["Learning Rate", "Batch Size", "Weight Decay", "Validation Accuracy", "Validation F1 Score"]
    results_table = wandb.Table(columns=columns)

    for lr, batch_size, weight_decay in itertools.product(learning_rates, batch_sizes, weight_decays):
        print(f"Training with lr={lr}, batch_size={batch_size}, weight_decay={weight_decay}")

        run_name = f"lr={lr}_bs={batch_size}_wd={weight_decay}"
        wandb.init(project="grid-search-curriculum", name=run_name, config={
            "learning_rate": lr,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "epochs": args.epochs
        })

        # Load the dataset
        if args.dataset == 'cifar10':
            num_classes = 10
            train_loader, val_loader, test_loader = load_cifar10(batch_size=batch_size)
        elif args.dataset == 'food101':
            num_classes = 101
            train_loader, val_loader, test_loader = load_food101(batch_size=batch_size)
        elif args.dataset == 'bloodmnist':
            num_classes = 8
            dataset = Dataset(batch_size=batch_size, name='B', image_size=28, as_rgb=True)
            train_loader, val_loader, test_loader = dataset.train_loader, dataset.val_loader, dataset.test_loader

        if args.model == 'vit':
            model = CurriculumVisionTransformer(num_classes=num_classes, pretrained=False)
        elif args.model == 'cnn':
            model = CurriculumConvNeXt(num_classes=num_classes, pretrained=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Initialize metrics
        accuracy_metric = Accuracy(task='multiclass', num_classes=num_classes).to(device)
        f1_metric = F1Score(task='multiclass', num_classes=num_classes, average='macro').to(device)


        for epoch in range(args.epochs):
            model.train()
            total_train_loss, total_train_acc, total_train_f1 = 0.0, 0.0, 0.0

            # Training loop
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                if len(labels.shape) > 1 and labels.shape[1] > 1:
                    labels = torch.argmax(labels, dim=1)
                labels = labels.squeeze()
                optimizer.zero_grad()

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Calculate accuracy and F1 score
                acc = accuracy_metric(outputs, labels)
                f1 = f1_metric(outputs, labels)
                total_train_loss += loss.item()
                total_train_acc += acc.item()
                total_train_f1 += f1.item()

            # Validation loop
            model.eval()
            total_val_acc, total_val_f1 = 0.0, 0.0
            with torch.no_grad():
                for val_images, val_labels in val_loader:
                    val_images, val_labels = val_images.to(device), val_labels.to(device)
                    if len(val_labels.shape) > 1 and val_labels.shape[1] > 1:
                        val_labels = torch.argmax(val_labels, dim=1)
                    val_labels = val_labels.squeeze()
                    val_outputs = model(val_images)

                    # Calculate validation accuracy and F1 score
                    val_acc = accuracy_metric(val_outputs, val_labels)
                    val_f1 = f1_metric(val_outputs, val_labels)
                    total_val_acc += val_acc.item()
                    total_val_f1 += val_f1.item()

            # Average metrics
            avg_val_acc = total_val_acc / len(val_loader)
            avg_val_f1 = total_val_f1 / len(val_loader)

            # Log metrics to WandB
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": total_train_loss / len(train_loader),
                "train_accuracy": total_train_acc / len(train_loader),
                "train_f1_score": total_train_f1 / len(train_loader),
                "val_accuracy": avg_val_acc,
                "val_f1_score": avg_val_f1,
                "learning_rate": lr,
                "batch_size": batch_size,
                "weight_decay": weight_decay,
            })

            # Save the best model
            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                best_hyperparams = {"learning_rate": lr, "batch_size": batch_size, "weight_decay": weight_decay}
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved with validation accuracy: {best_val_acc}")


        results_table.add_data(lr, batch_size, weight_decay, avg_val_acc, avg_val_f1)

        wandb.finish()


    wandb.init(project="grid-search-curriculum", name="final-results")
    wandb.log({"Hyperparameter Comparison Table": results_table})

    wandb.config.update(best_hyperparams)
    wandb.log({"best_validation_accuracy": best_val_acc})

    print(f"Best Hyperparameters: {best_hyperparams}")
    print(f"Best Validation Accuracy: {best_val_acc}")


if __name__ == "__main__":
    main()


