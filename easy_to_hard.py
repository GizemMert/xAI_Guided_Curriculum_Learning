import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Subset,  Dataset
import numpy as np
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights



pretrained_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
pretrained_model.eval()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pretrained_model = pretrained_model.to(device)
criterion = nn.CrossEntropyLoss(reduction='none')
def compute_confidence_scores(train_loader):
    confidences = {}

    pretrained_model.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.cuda() if torch.cuda.is_available() else images

            outputs = pretrained_model(images)
            probs = F.softmax(outputs, dim=1)

            max_confidence, _ = torch.max(probs, dim=1)

            # Record the confidence score for each sample
            for i, conf in enumerate(max_confidence):
                confidences[batch_idx * train_loader.batch_size + i] = conf.item()

    return confidences

def group_by_confidence(confidences):
    sorted_confidences = sorted(confidences.items(), key=lambda x: x[1], reverse=True)
    num_samples = len(sorted_confidences)
    num_easy = int(0.3 * num_samples)
    num_medium = int(0.4 * num_samples)

    easy_samples = [x[0] for x in sorted_confidences[:num_easy]]  # Top 30% confidence = Easy samples
    medium_samples = [x[0] for x in sorted_confidences[num_easy:num_easy + num_medium]]  # Next 40% = Medium
    hard_samples = [x[0] for x in sorted_confidences[num_easy + num_medium:]]  # Bottom 30% = Hard

    return easy_samples, medium_samples, hard_samples

# Function to compute loss scores using the pretrained ResNet50 model
def compute_pretrained_loss_scores(train_loader):
    losses = {}

    pretrained_model.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            if len(labels.shape) > 1 and labels.shape[1] > 1:
                labels = torch.argmax(labels, dim=1)
            labels = labels.squeeze()

            # Forward pass through the pretrained ResNet50 model
            outputs = pretrained_model(images)


            # Compute sample-wise loss using CrossEntropyLoss
            sample_losses = criterion(outputs, labels)

            # Record the loss score for each sample
            for i, loss in enumerate(sample_losses):
                losses[batch_idx * train_loader.batch_size + i] = loss.item()

    return losses

def group_by_loss(losses):
    sorted_losses = sorted(losses.items(), key=lambda x: x[1], reverse=False)
    num_samples = len(sorted_losses)
    num_easy = int(0.3 * num_samples)
    num_medium = int(0.4 * num_samples)

    easy_samples = [x[0] for x in sorted_losses[:num_easy]]  # Top 30% = Easy samples
    medium_samples = [x[0] for x in sorted_losses[num_easy:num_easy + num_medium]]  # Next 40% = Medium
    hard_samples = [x[0] for x in sorted_losses[num_easy + num_medium:]]  # Bottom 30% = Hard

    return easy_samples, medium_samples, hard_samples


# Easy to hard data loader
def curriculum_dataloader_xai(epoch, total_epochs, original_dataset, train_loader, easy_samples, medium_samples, hard_samples, explainability_tracker):
    indices = easy_samples.copy()  # Start with easy samples

    start_medium_samples_epoch = 30
    start_hard_samples_epoch = 60

    # Gradually add medium samples based on explainability improvements and epoch
    if epoch >= start_medium_samples_epoch and epoch < start_hard_samples_epoch:
        if explainability_tracker.epochs_without_improvement == 0:  # Add only if scores are improving
            num_medium_to_add = int(((epoch - start_medium_samples_epoch) / (start_hard_samples_epoch - start_medium_samples_epoch)) * len(medium_samples))
            indices += medium_samples[:num_medium_to_add]

    curriculum_dataset = Subset(original_dataset, indices)
    return DataLoader(curriculum_dataset, batch_size=train_loader.batch_size, shuffle=True, num_workers=train_loader.num_workers)

def curriculum_dataloader_epoch(epoch, total_epochs, original_dataset, train_loader, easy_samples, medium_samples, hard_samples):
    indices = easy_samples.copy()


    start_medium_samples_epoch = 20
    start_hard_samples_epoch = 60


    if start_medium_samples_epoch <= epoch < start_hard_samples_epoch:

        num_medium_to_add = int(((epoch - start_medium_samples_epoch) / (start_hard_samples_epoch - start_medium_samples_epoch)) * len(medium_samples))
        indices += medium_samples[:num_medium_to_add]


    elif epoch >= start_hard_samples_epoch:
        indices += medium_samples + hard_samples


    curriculum_dataset = Subset(original_dataset, indices)

    return DataLoader(curriculum_dataset, batch_size=train_loader.batch_size, shuffle=True, num_workers=train_loader.num_workers)


