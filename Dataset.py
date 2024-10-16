import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
import random
import multiprocessing
import wandb
from datasets import load_dataset
from PIL import Image

# Set random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

# Initialize Weights & Biases
wandb.init(project="cl_exp", config={
    "batch_size": 256,
    "validation_split": 0.1,
    "random_seed": RANDOM_SEED,
    "selected_classes_cifar10": list(range(10)),  # All classes in CIFAR-10
    # For Tiny ImageNet, we'll define selected classes later
})

num_workers = 4

# Transformations
cifar_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
])

food101_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
])


# Function to load CIFAR-10 dataset
def load_cifar10(batch_size=256, validation_split=0.1):
    print("Loading CIFAR-10 dataset...")

    # Load CIFAR-10 dataset
    full_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=cifar_transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=cifar_transform)

    selected_classes = list(range(10))  # All classes in CIFAR-10
    wandb.config.update({"selected_classes_cifar10": selected_classes})

    # Get all indices and labels
    all_indices = list(range(len(full_dataset)))
    all_labels = full_dataset.targets

    # Filter indices by selected classes
    class_indices = {class_label: [] for class_label in selected_classes}
    for idx, label in zip(all_indices, all_labels):
        if label in selected_classes:
            class_indices[label].append(idx)

    # Limit samples per class and split into train and validation
    train_indices = []
    val_indices = []
    for class_label in selected_classes:
        indices = class_indices[class_label]
        random.shuffle(indices)
        num_total = len(indices)
        num_train = int(num_total * (1 - validation_split))
        train_indices.extend(indices[:num_train][:2000])  # Limit to 1,000 per class
        val_indices.extend(indices[num_train:num_train + 500])  # Limit to 200 per class

    # Create samplers
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # Create data loaders
    train_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    val_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers)

    # Process test dataset
    test_indices = []
    test_labels = test_dataset.targets
    test_all_indices = list(range(len(test_dataset)))
    class_test_indices = {class_label: [] for class_label in selected_classes}
    for idx, label in zip(test_all_indices, test_labels):
        if label in selected_classes:
            class_test_indices[label].append(idx)
    for class_label in selected_classes:
        indices = class_test_indices[class_label]
        random.shuffle(indices)
        test_indices.extend(indices[:500])  # Limit to 200 per class

    test_sampler = SubsetRandomSampler(test_indices)
    test_loader = DataLoader(test_dataset, batch_size=1, sampler=test_sampler, num_workers=num_workers)

    print("CIFAR-10 dataset loaded successfully!")
    print(f"Train set size: {len(train_indices)}")
    print(f"Validation set size: {len(val_indices)}")
    print(f"Test set size: {len(test_indices)}")

    # Log dataset sizes
    wandb.config.update({
        "train_size_cifar10": len(train_indices),
        "validation_size_cifar10": len(val_indices),
        "test_size_cifar10": len(test_indices),
    })

    return train_loader, val_loader, test_loader


def load_food101(batch_size=256, num_workers=4):

    # Load the Food-101 dataset
    train_dataset = datasets.Food101(root='./data', split='train', download=True, transform=food101_transform)
    test_dataset = datasets.Food101(root='./data', split='test', download=True, transform=food101_transform)
    # Access the class_to_idx mapping
    class_to_idx = train_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Get all test indices
    all_test_indices = list(range(len(test_dataset)))

    # Get the labels for the test dataset
    all_test_labels = test_dataset._labels

    # Organize indices by class
    class_indices = {class_label: [] for class_label in range(101)}  # Food-101 has 101 classes
    for idx, label in zip(all_test_indices, all_test_labels):
        class_indices[label].append(idx)


    for class_label in class_indices:
        random.shuffle(class_indices[class_label])

    # Split indices into validation (150 images) and test (100 images) sets per class
    val_indices = []
    test_indices = []

    for class_label in class_indices:
        val_indices.extend(class_indices[class_label][:150])  # First 150 for validation
        test_indices.extend(class_indices[class_label][150:250])  # Next 100 for testing

    # samplers
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # DataLoaders for training, validation, and test datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=8, sampler=test_sampler, num_workers=num_workers)

    print(f"Train set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_indices)}")
    print(f"Test set size: {len(test_indices)}")

    return train_loader, val_loader, test_loader


