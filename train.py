import argparse
from timm.models.focalnet import FocalNetStage
import torch
import torch.optim as optim
from Dataset import load_cifar10, load_food101
from model_ViT import CurriculumVisionTransformer
from model_CnvNext import CurriculumConvNeXt
from curriculum_scheduler_dropout import CurriculumScheduler
from Explainability import Explainability
import torch.nn as nn
import wandb
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

# Denormalization function for CIFAR-10
def denormalize(image, mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    if image.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    image = image * std + mean
    return image.clamp(0, 1)


# Function to log fixed images with dropout applied
def log_fixed_images_with_dropout(original_images, dropout_images, epoch):
    for i in range(len(original_images)):

        original_image = denormalize(original_images[i].cpu().detach()).numpy().transpose(1, 2, 0)
        dropout_image = denormalize(dropout_images[i].cpu().detach()).numpy().transpose(1, 2, 0)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        
        # Original Image
        ax[0].imshow(original_image)
        ax[0].set_title(f"Original Image {i+1} (Epoch {epoch})")
        ax[0].axis('off')

        # Dropout Applied Image
        ax[1].imshow(dropout_image)
        ax[1].set_title(f"Dropout Applied {i+1} (Epoch {epoch})")
        ax[1].axis('off')

        wandb.log({f"Dropout_Image_{i+1}_epoch_{epoch}": wandb.Image(fig)}, commit=False)
        plt.close(fig)


# Function to log attribution maps
def log_attributions(attributions, images, epoch, method_name="Attribution"):
    for i in range(len(images)):

        image = denormalize(images[i].cpu().detach()).numpy().transpose(1, 2, 0)  # CHW -> HWC for RGB
        attribution = attributions[i].cpu().detach().numpy()

        if attribution.shape[0] == 3:
            attribution = np.mean(attribution, axis=0)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        # Plot original image
        ax[0].imshow(image)
        ax[0].set_title(f"Original Image {i+1}")
        ax[0].axis('off')

        # Plot attribution heatmap
        ax[1].imshow(attribution, cmap='jet', alpha=0.5)
        ax[1].set_title(f"{method_name} Attribution {i+1} (Epoch {epoch})")
        ax[1].axis('off')


        wandb.log({f"{method_name}_Attribution_{i+1}_epoch_{epoch}": wandb.Image(fig)}, commit=False)
        plt.close(fig)

def main():
    #set_seed(24)

    parser = argparse.ArgumentParser(description='Train model with curriculum learning and explainability.')

    # Model and dataset settings
    parser.add_argument('--model', type=str, choices=['vit', 'cnn'], default='vit',
                        help='Choose the model architecture (vit or cnn)')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'food101'], required=True,
                        help='Dataset to use (cifar10 or food101)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training and validation')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')

    # Explainability settings
    parser.add_argument('--explainer', type=str,
                        choices=['saliency', 'integrated_gradients', 'compute_input_x_gradient'], default='saliency',
                        help='Gradient-based explainer method to use')

    # Scheduler for dropout adjustments
    parser.add_argument('--scheduler', type=str, choices=['epoch', 'sufficiency', 'infidelity', 'sensitivity'],
                        default='epoch',
                        help='Curriculum dropout scheduling method based on explainability metric (epoch, sufficiency, infidelity, sensitivity)')

    # Metrics
    parser.add_argument('--metric', type=str, nargs='+', choices=['sufficiency', 'infidelity', 'sensitivity'],
                        help='Evaluation metrics to compute (sufficiency, infidelity, sensitivity)')

    args = parser.parse_args()

    
    run_name = f"{args.scheduler}-{args.explainer}"


    wandb.init(project="curriculum-learning", entity="gizem-mert", name=run_name, config=args)

    # Load the dataset
    if args.dataset == 'cifar10':
        num_classes = 10
        train_loader, val_loader, test_loader = load_cifar10(batch_size=args.batch_size)
    elif args.dataset == 'food101':
        num_classes = 101
        train_loader, val_loader, test_loader = load_food101(batch_size=args.batch_size)

    # Select model based on user input
    if args.model == 'vit':
        model = CurriculumVisionTransformer(num_classes=num_classes, pretrained=False)
    elif args.model == 'cnn':
        model = CurriculumConvNeXt(num_classes=num_classes, pretrained=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Initialize metrics
    accuracy_metric = Accuracy(task='multiclass', num_classes=num_classes).to(device)
    f1_metric = F1Score(task='multiclass', num_classes=num_classes, average='macro').to(device)

    # Initialize curriculum scheduler and explainability
    scheduler = CurriculumScheduler(
        initial_theta=1.0, final_theta=0.5, warmup_epochs=30, total_epochs=args.epochs, gamma=0.005, decay_type='exponential',
        metric_type= args.scheduler
    )

    explainability = Explainability(model)

    best_val_acc = 0.0
    model_name = f"{args.dataset}_{args.model}_{args.explainer}_{args.scheduler}.pt"

    # Define the path to save the model
    best_model_path = f"models/{model_name}"

    def perturb_fn(inputs):
        noise = torch.tensor(np.random.normal(0, 0.003, inputs.shape)).float().to(inputs.device)
        return noise, inputs - noise

    early_stopping_counter = 0  
    patience = 20

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
    
    for epoch in range(args.epochs):
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
            
            images.requires_grad = True
            # 1. Select explainer dynamically based on the argument
            if args.explainer == 'saliency':
                attributions = explainability.compute_saliency(images, target=labels)
                explainer_method = explainability.compute_saliency
            elif args.explainer == 'integrated_gradients':
                attributions = explainability.compute_integrated_gradients(images, target=labels)
                explainer_method = explainability.compute_integrated_gradients
            elif args.explainer == 'compute_input_x_gradient':
                attributions = explainability.compute_input_x_gradient(images, target=labels)
                explainer_method = explainability.compute_input_x_gradient

            # 2. Compute explainability score using the appropriate method (e.g., sufficiency, infidelity, sensitivity)
            explainability_score = None
            if args.scheduler != 'epoch':
                if args.scheduler == 'sufficiency':
                    explainability_score = explainability.calculate_sufficiency(images, attributions)
                    sufficiency_score_sum += explainability_score.sum().item()
                    sufficiency_count += len(images)
                elif args.scheduler == 'infidelity':
                    explainability_score = explainability.calculate_infidelity(perturb_fn, images, attributions, target=labels)
                    infid_score_sum += explainability_score.item()
                    infidelity_count += 1
                elif args.scheduler == 'sensitivity':
                    explainability_score = explainability.calculate_sensitivity(explainer_method, images, target=labels)
                    sens_score_sum += explainability_score.item()
                    sensitivity_count += 1
            
           # if explainability_score is None:
               # raise ValueError("Explainability score was not computed properly.")


            # Apply input-level dropout based on the chosen scheduler
            if args.scheduler == 'epoch':
                images = scheduler.apply_dropout(images, epoch=epoch)  # Epoch-based dropout
                dropout_probability = scheduler.get_retain_probability(epoch=epoch)
            else:
                print(f"Explainability score: {explainability_score}")
                images = scheduler.apply_dropout(images, explainability_score=explainability_score)  # Explainability-based dropout
                dropout_probability = scheduler.get_retain_probability(explainability_score=explainability_score)

            # Forward pass through the model
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Compute accuracy and F1 score for training
            acc = accuracy_metric(outputs, labels)
            f1 = f1_metric(outputs, labels)
            total_train_loss += loss.item()
            total_train_acc += acc.item()
            total_train_f1 += f1.item()



        # Validation loop
        model.eval()
        total_val_acc = 0.0
        total_val_f1 = 0.0

        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_outputs = model(val_images)

                # Compute accuracy and F1 score for validation
                acc = accuracy_metric(val_outputs, val_labels)
                f1 = f1_metric(val_outputs, val_labels)
                total_val_acc += acc.item()
                total_val_f1 += f1.item()

        # Save the model if validation accuracy improves
        train_acc = total_train_acc / len(train_loader)
        train_f1 = total_train_f1 / len(train_loader)
        val_acc = total_val_acc / len(val_loader)
        val_f1 = total_val_f1 / len(val_loader)

        # After each epoch, compute the average explainability scores (if calculated)
        avg_sufficiency_score = sufficiency_score_sum / sufficiency_count if sufficiency_count > 0 else None
        avg_infid_score = infid_score_sum / infidelity_count if infidelity_count > 0 else None
        avg_sens_score = sens_score_sum / sensitivity_count if sensitivity_count > 0 else None


        dropout_fixed_images = scheduler.apply_dropout(fixed_images, explainability_score=explainability_score, epoch=epoch)
        log_fixed_images_with_dropout(fixed_images, dropout_fixed_images, epoch)
        log_attributions(attributions[:3], fixed_images[:3], epoch, method_name=args.explainer)
        

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
        print(f"Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": total_train_loss / len(train_loader),
            "train_accuracy": total_train_acc / len(train_loader),
            "train_f1_score": total_train_f1 / len(train_loader),
            "val_accuracy": total_val_acc / len(val_loader),
            "val_f1_score": total_val_f1 / len(val_loader),
            "learning_rate": args.learning_rate,
            "dropout_probability": dropout_probability,
            "avg_sufficiency": avg_sufficiency_score,
            "avg_infidelity": avg_infid_score,
            "avg_sensitivity": avg_sens_score
        })


    model.load_state_dict(torch.load(best_model_path))
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
            elif args.explainer == 'integrated_gradients':
                attributions = explainability.compute_integrated_gradients(test_images, target=test_labels)
                explainer_method = explainability.compute_integrated_gradients
            elif args.explainer == 'compute_input_x_gradient':
                attributions = explainability.compute_input_x_gradient(test_images, target=test_labels)
                explainer_method = explainability.compute_input_x_gradient

            # Visualize and log attribution maps for the first few test images
            if idx < 5:
                for i in range(min(3, test_images.size(0))):
                    experiment_name = wandb.run.name if wandb.run.name else "experiment"
                    visualize_attr_maps(test_images[i], attributions[i], args.explainer, idx=idx,
                                        experiment_name=experiment_name)
                    idx += 1

            if idx >= 15:
                break

            # Compute and accumulate explainability metrics
            if args.metric:
                if 'sufficiency' in args.metric:
                    # Calculate sufficiency scores for different retention proportions
                    sufficiency_scores = explainability.calculate_sufficiency(test_images, attributions,
                                                                              retention_proportions=[0.15, 0.30, 0.50])
                    for prop, score in sufficiency_scores.items():
                        sufficiency_scores_sum[prop] += score
                    sufficiency_count += len(test_images)

                if 'infidelity' in args.metric:
                    infid_score = explainability.calculate_infidelity(perturb_fn, test_images, attributions,
                                                                      target=test_labels)
                    infid_score_sum += infid_score.item()
                    infidelity_count += 1

                if 'sensitivity' in args.metric:
                    sens_score = explainability.calculate_sensitivity(explainer_method, test_images, target=test_labels)
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

        metrics_table.add_data(data_type, accuracy, f1_score,
                               avg_sufficiency_scores[0.15], avg_sufficiency_scores[0.30], avg_sufficiency_scores[0.50],
                               avg_infid_score, avg_sens_score)

        wandb.log({"Test Metrics Table": metrics_table})

        # Log the explainability metrics
        wandb.log({
            "test_accuracy": accuracy,
            "test_f1_score": f1_score,
            "avg_sufficiency_15%": avg_sufficiency_scores[0.15],
            "avg_sufficiency_30%": avg_sufficiency_scores[0.30],
            "avg_sufficiency_50%": avg_sufficiency_scores[0.50],
            "avg_infidelity_score_test": avg_infid_score,
            "avg_sensitivity_score_test": avg_sens_score,
        })




if __name__ == "__main__":
    main()
