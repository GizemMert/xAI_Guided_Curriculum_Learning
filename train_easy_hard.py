import argparse
from timm.models.focalnet import FocalNetStage
import torch
import torch.optim as optim
import time
from Dataset import load_cifar10, load_food101
from model_ViT import CurriculumVisionTransformer
from model_Mixer import CurriculumMLPMixer
from model_CnvNext import CurriculumConvNeXt
from easy_to_hard import compute_confidence_scores, group_by_confidence, compute_pretrained_loss_scores, group_by_loss, curriculum_dataloader_epoch, curriculum_dataloader_xai
from sufficiency_easyhard_schedule import ExplainabilityEvaluator
from curriculum_scheduler_dropout import CurriculumScheduler
from Explainability import Explainability
import torch.nn as nn
import wandb
from medmnist_loader import Dataset
from torchmetrics import Accuracy, F1Score
from captum.metrics import infidelity, sensitivity_max
import numpy as np
import random
from visualize import visualize_attr_maps
import matplotlib.pyplot as plt


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ExplainabilityTracker:
    def __init__(self, patience=3):


        self.best_score = None
        self.epochs_without_improvement = 0
        self.patience = patience

    def update(self, current_score):
        """
        Args:
            - current_score: The current explainability score (lower is better).
        """
        if self.best_score is None or current_score < self.best_score:  # lower score is better
            self.best_score = current_score
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        if self.epochs_without_improvement >= self.patience:
            print(f"No improvement in explainability for {self.patience} epochs. Pausing sample addition.")

# Denormalization function for CIFAR-10
def denormalize(image, mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    if image.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    image = image * std + mean
    return image.clamp(0, 1)

# Denormalization function for BloodMNIST
def denormalize_blood(image, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    if image.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    image = image * std + mean
    return image.clamp(0, 1)


# Function to log fixed images with dropout applied
def log_fixed_images_with_dropout(original_images, dropout_images, epoch, dataset):
    for i in range(len(original_images)):

        if dataset == 'cifar10':
            denormalize_fn = denormalize
        elif dataset == 'bloodmnist':
            denormalize_fn = denormalize_blood
        original_image = denormalize_fn(original_images[i].cpu().detach()).numpy().transpose(1, 2, 0)
        dropout_image = denormalize_fn(dropout_images[i].cpu().detach()).numpy().transpose(1, 2, 0)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        # Original Image
        ax[0].imshow(original_image)
        ax[0].set_title(f"Original Image {i + 1} (Epoch {epoch})")
        ax[0].axis('off')

        # Dropout Applied Image
        ax[1].imshow(dropout_image)
        ax[1].set_title(f"Dropout Applied {i + 1} (Epoch {epoch})")
        ax[1].axis('off')

        wandb.log({f"Dropout_Image_{i + 1}_epoch_{epoch}": wandb.Image(fig)}, commit=False)
        plt.close(fig)


# Function to log attribution maps
def log_attributions(attributions, images, epoch, dataset, method_name="Attribution"):
    for i in range(len(images)):

        if dataset == 'cifar10':
            denormalize_fn = denormalize
        elif dataset == 'bloodmnist':
            denormalize_fn = denormalize_blood

        image = denormalize_fn(images[i].cpu().detach()).numpy().transpose(1, 2, 0)  # CHW -> HWC for RGB
        attribution = attributions[i].cpu().detach().numpy()

        if attribution.shape[0] == 3:
            attribution = np.mean(attribution, axis=0)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        # Plot original image
        ax[0].imshow(image)
        ax[0].set_title(f"Original Image {i+1}")
        ax[0].axis('off')

        # Plot attribution heatmap (2D)
        ax[1].imshow(attribution, cmap='jet', alpha=0.5)
        ax[1].set_title(f"{method_name} Attribution {i+1} (Epoch {epoch})")
        ax[1].axis('off')


        wandb.log({f"{method_name}_Attribution_{i+1}_epoch_{epoch}": wandb.Image(fig)}, commit=False)
        plt.close(fig)

# To store results across runs
test_accuracies = []
test_f1_scores = []
sufficiency_15_list = []
sufficiency_30_list = []
sufficiency_50_list = []
infidelity_scores = []
sensitivity_scores = []


def log_mean_and_std(mean, std, metric_name, args):
    wandb.log({
        f"{metric_name}_mean": mean,
        f"{metric_name}_std": std,
        "model": args.model,
        "dataset": args.dataset,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "explainer": args.explainer,
        "scheduler": args.scheduler,
        "explainer_for_scheduler": args.explain_schedular,
        "sample_selection_method": args.sample_selection,
        "metric": args.metric,

    })



def main():
    # Set random seeds for reproducibility
    # seeds = [0, 1, 2]  # Change the seeds as needed
    # for seed in seeds:
    #    set_seed(seed)

    parser = argparse.ArgumentParser(description='Train model with curriculum learning and explainability.')

    # Model and dataset settings
    parser.add_argument('--model', type=str, choices=['vit', 'cnn', 'mlpmix'], default='vit',
                        help='Choose the model architecture (vit, cnn or mlpmix)')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'food101', 'bloodmnist'], required=True,
                        help='Dataset to use (cifar10, food101, bloodmnist)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training and validation')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (L2 penalty)')

    # Explainability settings
    parser.add_argument('--explainer', type=str,
                        choices=['saliency', 'guided_backprop', 'compute_input_x_gradient'],
                        default='saliency',
                        help='Gradient-based explainer method to use')

    # Scheduler for dropout adjustments
    parser.add_argument('--scheduler', type=str,
                        choices=['epoch', 'sufficiency', 'infidelity', 'sensitivity', 'easyhard'],
                        default='epoch', help='Scheduler type to use for curriculum learning.')
    parser.add_argument('--explain_scheduler', type=str,
                        choices=['sufficiency', 'sensitivity', 'epoch'],
                        default='sufficiency', help='xAI type to guide scheduler to use for curriculum learning.')

    parser.add_argument('--sample_selection', type=str, choices=['confidence', 'loss'], default='confidence',
                        help='Sample selection strategy (confidence or loss)')

    # Metrics to evaluate
    parser.add_argument('--metric', type=str, nargs='+', choices=['sufficiency', 'infidelity', 'sensitivity'],
                        help='Evaluation metrics to compute (sufficiency, infidelity, sensitivity)')

    args = parser.parse_args()

    run_name = f"{args.scheduler}-{args.explainer}"

    wandb.init(project="curriculum-learning", entity="gizem-mert", name=run_name, config=args)

    # Load the dataset
    if args.dataset == 'cifar10':
        num_classes = 10
        train_loader, val_loader, test_loader = load_cifar10(batch_size=args.batch_size)
        original_train_dataset = train_loader.dataset
    elif args.dataset == 'food101':
        num_classes = 101
        train_loader, val_loader, test_loader = load_food101(batch_size=args.batch_size)
        original_train_dataset = train_loader.dataset
    elif args.dataset == 'bloodmnist':
        num_classes = 8
        dataset = Dataset(batch_size=args.batch_size, name='B', image_size=28, as_rgb=True)
        train_loader, val_loader, test_loader = dataset.train_loader, dataset.val_loader, dataset.test_loader
        original_train_dataset = train_loader.dataset
        wandb.log({
            "train_size": len(train_loader.dataset),
            "val_size": len(val_loader.dataset),
            "test_size": len(test_loader.dataset)
        })

    if args.scheduler == 'easyhard':
        # Select sample selection strategy based on the argument
        if args.sample_selection == 'confidence':
            print("Computing confidence scores for the training dataset...")
            confidences = compute_confidence_scores(train_loader)
            easy_samples, medium_samples, hard_samples = group_by_confidence(confidences)

        elif args.sample_selection == 'loss':
            print("Computing loss scores using the pretrained ResNet50 model...")
            losses = compute_pretrained_loss_scores(train_loader)
            easy_samples, medium_samples, hard_samples = group_by_loss(losses)

    # Select model based on user input
    if args.model == 'vit':
        model = CurriculumVisionTransformer(num_classes=num_classes, pretrained=False)
    elif args.model == 'cnn':
        model = CurriculumConvNeXt(num_classes=num_classes, pretrained=False)
    elif args.model == 'mlpmix':
        model = CurriculumMLPMixer(num_classes=num_classes, pretrained=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)


    # Initialize metrics
    accuracy_metric = Accuracy(task='multiclass', num_classes=num_classes).to(device)
    f1_metric = F1Score(task='multiclass', num_classes=num_classes, average='macro').to(device)

    # Initialize curriculum scheduler and explainability
    scheduler = CurriculumScheduler(
        initial_theta=1.0, final_theta=0.5, warmup_epochs=30, total_epochs=args.epochs, gamma=0.005,
        decay_type='exponential',
        metric_type=args.explain_scheduler
    )

    explainability = Explainability(model)
    evaluator = ExplainabilityEvaluator(model)
    explainability_tracker = ExplainabilityTracker(patience=3)

    best_val_acc = 0.0
    model_name = f"{args.dataset}_{args.model}_{args.explainer}_{args.scheduler}_{args.sample_selection}_{args.explain_scheduler}.pt"

    # Define the path to save the model
    best_model_path = f"models/{model_name}"

    def perturb_fn(inputs):
        noise = torch.tensor(np.random.normal(0, 0.003, inputs.shape)).float().to(inputs.device)
        return noise, inputs - noise

    early_stopping_counter = 0
    patience = 100
    total_training_time = 0.0

    # Select fixed images for visualization
    fixed_images, fixed_labels = [], []
    for i, (img, label) in enumerate(train_loader):
        if len(fixed_images) < 3:
            fixed_images.append(img[0].unsqueeze(0))
            fixed_labels.append(label[0].unsqueeze(0))
        else:
            break

    fixed_images = torch.cat(fixed_images, dim=0).to(device)
    fixed_labels = torch.cat(fixed_labels, dim=0).to(device)
    subsampling_ratio = 0.5
    for epoch in range(args.epochs):
        start_time = time.time()
        if args.scheduler == 'easyhard':
            if args.explain_scheduler == 'epoch':
                train_loader = curriculum_dataloader_epoch(epoch, args.epochs, original_train_dataset, train_loader,
                                                     easy_samples, medium_samples, hard_samples)
            else:
                # Update train_loader dynamically for curriculum learning with xAI
                train_loader = curriculum_dataloader_xai(epoch, args.epochs, original_train_dataset, train_loader,
                                                 easy_samples, medium_samples, hard_samples, explainability_tracker)


        model.train()
        total_train_loss = 0.0
        total_train_acc = 0.0
        total_train_f1 = 0.0

        sufficiency_score_sum = 0.0
        infid_score_sum = 0.0
        sens_score_sum = 0.0
        sufficiency_count = 0
        infidelity_count = 0
        sensitivity_count = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            if len(labels.shape) > 1 and labels.shape[1] > 1:
                labels = torch.argmax(labels, dim=1)
            labels = labels.squeeze()
            images.requires_grad = True

            # Forward pass through the model
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()



            if subsampling_ratio < 1.0:
                subsample_size = int(len(images) * subsampling_ratio)
                subsample_indices = torch.randperm(len(images))[:subsample_size]
                images_subsample = images[subsample_indices]
                labels_subsample = labels[subsample_indices]
            else:
                images_subsample = images
                labels_subsample = labels

            # Compute attributions on the subsampled images
            if args.explainer == 'saliency':
                attributions = explainability.compute_saliency(images_subsample, target=labels_subsample)
                explainer_method = explainability.compute_saliency
            elif args.explainer == 'guided_backprop':
                attributions = explainability.compute_guided_backprop(images_subsample, target=labels_subsample)
                explainer_method = explainability.compute_guided_backprop
            elif args.explainer == 'compute_input_x_gradient':
                attributions = explainability.compute_input_x_gradient(images_subsample, target=labels_subsample)
                explainer_method = explainability.compute_input_x_gradient

            explainability_score = None
            if args.scheduler == 'easyhard':
                if args.explain_scheduler == 'sufficiency':
                    explainability_score = evaluator.sufficiency(images_subsample, attributions)
                    # print(f"Type of explainability_score: {type(explainability_score)}")
                    sufficiency_score_sum += explainability_score
                    sufficiency_count += len(images)
                if args.explain_scheduler == 'sensitivity':
                    explainability_score = explainability.calculate_sensitivity(explainer_method, images_subsample, target=labels_subsample)
                    sens_score_sum += explainability_score.item()
                    sensitivity_count += 1


            # if explainability_score is None:
            # raise ValueError("Explainability score was not computed properly.")

            # Apply input-level dropout based on the chosen scheduler
            if args.scheduler == 'easyhard':

                dropout_probability = None
            else:
                # Apply dropout based on the selected scheduler
                if args.scheduler == 'epoch':
                    images = scheduler.apply_dropout(images, epoch=epoch)  # Epoch-based dropout
                    dropout_probability = scheduler.get_retain_probability(epoch=epoch)
                elif args.scheduler in ['sufficiency', 'infidelity', 'sensitivity']:
                    # For explainability-based schedulers (e.g., sufficiency, infidelity, sensitivity)
                    print(f"Explainability score: {explainability_score}")
                    images = scheduler.apply_dropout(images, explainability_score=explainability_score)
                    dropout_probability = scheduler.get_retain_probability(
                        explainability_score=explainability_score)

            # Compute accuracy and F1 score for training
            acc = accuracy_metric(outputs, labels)
            f1 = f1_metric(outputs, labels)
            total_train_loss += loss.item()
            total_train_acc += acc.item()
            total_train_f1 += f1.item()

        if args.explain_scheduler == 'sufficiency':
            avg_sufficiency_score = sufficiency_score_sum / sufficiency_count if sufficiency_count > 0 else None
            if args.scheduler == 'easyhard':
                if avg_sufficiency_score is not None:
                    if isinstance(avg_sufficiency_score, torch.Tensor):
                        explainability_tracker.update(avg_sufficiency_score.mean().item())
                    else:
                        explainability_tracker.update(
                            float(avg_sufficiency_score))

        if args.explain_scheduler == 'sensitivity':
            avg_sens_score = sens_score_sum / sensitivity_count if sensitivity_count > 0 else None
            if args.scheduler == 'easyhard':
                if avg_sens_score is not None:

                    if isinstance(avg_sens_score, torch.Tensor):
                        explainability_tracker.update(avg_sens_score.mean().item())  # Tensor case
                    else:
                        explainability_tracker.update(float(avg_sens_score))  # Ensure it's treated as a float

        # Validation loop
        model.eval()
        val_loss = 0.0
        total_val_acc = 0.0
        total_val_f1 = 0.0
        evl_train_loss = 0.0
        evl_train_acc = 0.0
        evl_train_f1 = 0.0

        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                if len(val_labels.shape) > 1 and val_labels.shape[1] > 1:
                    val_labels = torch.argmax(val_labels, dim=1)
                val_labels = val_labels.squeeze()
                val_outputs = model(val_images)
                val_loss += criterion(val_outputs, val_labels).item()

                # Compute accuracy and F1 score for validation
                acc = accuracy_metric(val_outputs, val_labels)
                f1 = f1_metric(val_outputs, val_labels)
                total_val_acc += acc.item()
                total_val_f1 += f1.item()

            for ev_train_images, ev_train_labels in train_loader:
                ev_train_images, ev_train_labels = ev_train_images.to(device), ev_train_labels.to(device)
                if len(ev_train_labels.shape) > 1 and ev_train_labels.shape[1] > 1:
                    ev_train_labels = torch.argmax(ev_train_labels, dim=1)
                ev_train_labels = ev_train_labels.squeeze()
                ev_train_outputs = model(ev_train_images)
                evl_train_loss += criterion(ev_train_outputs, ev_train_labels).item()

                # Compute accuracy and F1 score for validation
                acc = accuracy_metric(ev_train_outputs, ev_train_labels)
                f1 = f1_metric(ev_train_outputs, ev_train_labels)
                evl_train_acc += acc.item()
                evl_train_f1 += f1.item()

        # Save the model if validation accuracy improves
        train_acc = total_train_acc / len(train_loader)
        train_f1 = total_train_f1 / len(train_loader)
        val_acc = total_val_acc / len(val_loader)
        val_f1 = total_val_f1 / len(val_loader)

        # After each epoch, compute the average explainability scores (if calculated)
        avg_sufficiency_score = sufficiency_score_sum / sufficiency_count if sufficiency_count > 0 else None
        avg_infid_score = infid_score_sum / infidelity_count if infidelity_count > 0 else None
        avg_sens_score = sens_score_sum / sensitivity_count if sensitivity_count > 0 else None

        if args.scheduler != 'easyhard':
            # Visualize and log images after applying dropout
            dropout_fixed_images = scheduler.apply_dropout(fixed_images, explainability_score=explainability_score,
                                                       epoch=epoch)
            log_fixed_images_with_dropout(fixed_images, dropout_fixed_images, epoch, args.dataset)
            log_attributions(attributions[:3], fixed_images[:3], epoch, args.dataset, method_name=args.explainer)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with validation accuracy: {best_val_acc}")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f"Early stopping triggered. No improvement in {patience} epochs.")
                break

        print(f"Epoch [{epoch + 1}/{args.epochs}]")
        print(
            f"Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        end_time = time.time()
        epoch_duration = end_time - start_time
        total_training_time += epoch_duration
        wandb.log({
            "epoch": epoch + 1,
            "epoch_duration": epoch_duration,
            "backpropagation_samples": (batch_idx + 1) * images.size(0),
            "samples_explained": sufficiency_count + sensitivity_count,
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
            "train_loss": total_train_loss / len(train_loader),
            "val_loss": val_loss / len(val_loader),
            "train_accuracy": total_train_acc / len(train_loader),
            "train_f1_score": total_train_f1 / len(train_loader),
            "val_accuracy": total_val_acc / len(val_loader),
            "val_f1_score": total_val_f1 / len(val_loader),
            "learning_rate": args.learning_rate,
            "dropout_probability": dropout_probability,
            "avg_sufficiency": avg_sufficiency_score,
            "avg_infidelity": avg_infid_score,
            "avg_sensitivity": avg_sens_score,
            "evl_train_loss": evl_train_loss / len(train_loader),
            "evl_train_accuracy": evl_train_acc / len(train_loader),
            "evl_train_f1_score": evl_train_f1 / len(train_loader),
        })

    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval()
    explainability = Explainability(model)
    total_test_acc = 0.0
    total_test_f1 = 0.0

    # Initialize sufficiency for different proportions
    sufficiency_scores_sum = {0.15: 0.0, 0.30: 0.0, 0.50: 0.0}
    infid_score_sum = 0.0
    sens_score_sum = 0.0

    sufficiency_count = 0
    infidelity_count = 0
    sensitivity_count = 0

    idx = 0

    with torch.no_grad():
        for test_images, test_labels in test_loader:
            test_images, test_labels = test_images.to(device), test_labels.to(device)
            if len(test_labels.shape) > 1 and test_labels.shape[1] > 1:
                test_labels = torch.argmax(test_labels, dim=1)
            test_labels = test_labels.squeeze()
            test_outputs = model(test_images)

            # Compute accuracy and F1 score for test dataset
            acc = accuracy_metric(test_outputs, test_labels)
            f1 = f1_metric(test_outputs, test_labels)
            total_test_acc += acc.item()
            total_test_f1 += f1.item()

            # Compute explainability scores on test data (using chosen explainer)
            if args.explainer == 'saliency':
                attributions = explainability.compute_saliency(test_images, target=test_labels)
                explainer_method = explainability.compute_saliency
            elif args.explainer == 'guided_backprop':
                attributions = explainability.compute_guided_backprop(images, target=labels)
                explainer_method = explainability.compute_guided_backprop
            elif args.explainer == 'compute_input_x_gradient':
                attributions = explainability.compute_input_x_gradient(test_images, target=test_labels)
                explainer_method = explainability.compute_input_x_gradient

            # Visualize and log attribution maps for the first few test images
            if idx < 4:
                for i in range(min(2, test_images.size(0))):
                    experiment_name = wandb.run.name if wandb.run.name else "experiment"
                    visualize_attr_maps(test_images[i], attributions[i], args.explainer, idx=idx,
                                        experiment_name=experiment_name)
                    idx += 1

            # Compute and accumulate explainability metrics
            if args.metric:
                if 'sufficiency' in args.metric:
                    # Calculate sufficiency scores for different retention proportions
                    sufficiency_scores = explainability.calculate_sufficiency(test_images, attributions,
                                                                              retention_proportions=[0.15, 0.30,
                                                                                                     0.50])
                    for prop, score in sufficiency_scores.items():
                        sufficiency_scores_sum[prop] += score
                    sufficiency_count += len(test_images)

                if 'infidelity' in args.metric:
                    infid_score = explainability.calculate_infidelity(perturb_fn, test_images, attributions,
                                                                      target=test_labels)
                    infid_score_sum += infid_score.item()
                    infidelity_count += 1

                if 'sensitivity' in args.metric:
                    sens_score = explainability.calculate_sensitivity(explainer_method, test_images,
                                                                      target=test_labels)
                    sens_score_sum += sens_score.item()
                    sensitivity_count += 1

        # Compute average explainability metrics
        avg_sufficiency_scores = {prop: sufficiency_scores_sum[prop] / sufficiency_count for prop in
                                  sufficiency_scores_sum} if sufficiency_count > 0 else None
        avg_infid_score = infid_score_sum / infidelity_count if infidelity_count > 0 else None
        avg_sens_score = sens_score_sum / sensitivity_count if sensitivity_count > 0 else None

        # Create a W&B table to log the explainability metrics
        columns = ["Data Type", "Accuracy", "F1 Score", "Sufficiency 15%", "Sufficiency 30%", "Sufficiency 50%",
                   "Infidelity", "Sensitivity"]
        metrics_table = wandb.Table(columns=columns)

        # Set the data type based on the dataset used
        data_type = "CIFAR-10" if args.dataset == 'cifar10' else "Food-101" if args.dataset == 'food101' else "Unknown Dataset"

        accuracy = total_test_acc / len(test_loader)
        f1_score = total_test_f1 / len(test_loader)
        test_accuracies.append(accuracy)
        test_f1_scores.append(f1_score)
        sufficiency_15_list.append(avg_sufficiency_scores[0.15])
        sufficiency_30_list.append(avg_sufficiency_scores[0.30])
        sufficiency_50_list.append(avg_sufficiency_scores[0.50])
        infidelity_scores.append(avg_infid_score)
        sensitivity_scores.append(avg_sens_score)

        metrics_table.add_data(data_type, accuracy, f1_score,
                               avg_sufficiency_scores[0.15], avg_sufficiency_scores[0.30],
                               avg_sufficiency_scores[0.50],
                               avg_infid_score, avg_sens_score)

        wandb.log({"Test Metrics Table": metrics_table})


        wandb.log({
            # "seed": seed,
            "test_samples": len(test_loader.dataset),
            "total_training_time": total_training_time,
            "test_accuracy": accuracy,
            "test_f1_score": f1_score,
            "avg_sufficiency_15%": avg_sufficiency_scores[0.15],
            "avg_sufficiency_30%": avg_sufficiency_scores[0.30],
            "avg_sufficiency_50%": avg_sufficiency_scores[0.50],
            "avg_infidelity_score_test": avg_infid_score,
            "avg_sensitivity_score_test": avg_sens_score,
        })
        wandb.finish()
        """
        # After all runs, log mean and std for each metric
        log_mean_and_std(np.mean(test_accuracies), np.std(test_accuracies), "Test_Accuracy", args)
        log_mean_and_std(np.mean(test_f1_scores), np.std(test_f1_scores), "Test_F1_Score", args)
        log_mean_and_std(np.mean(sufficiency_15_list), np.std(sufficiency_15_list), "Sufficiency_15", args)
        log_mean_and_std(np.mean(sufficiency_30_list), np.std(sufficiency_30_list), "Sufficiency_30", args)
        log_mean_and_std(np.mean(sufficiency_50_list), np.std(sufficiency_50_list), "Sufficiency_50", args)
        log_mean_and_std(np.mean(infidelity_scores), np.std(infidelity_scores), "Infidelity_Score", args)
        log_mean_and_std(np.mean(sensitivity_scores), np.std(sensitivity_scores), "Sensitivity_Score", args)
        """

if __name__ == "__main__":
    main()
