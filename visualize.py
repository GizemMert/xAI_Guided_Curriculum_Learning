import wandb
import matplotlib.pyplot as plt
import torch
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Denormalization function for CIFAR-10
def denormalize(image, mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)):
    mean = torch.tensor(mean).view(3, 1, 1).to(device)
    std = torch.tensor(std).view(3, 1, 1).to(device)
    image = image * std + mean
    return image.clamp(0, 1)

def visualize_attr_maps(image, attribution, explainer_type, idx=0, experiment_name="experiment"):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))


    image_viz = denormalize(image).cpu().detach().numpy().transpose(1, 2, 0)

    if attribution.shape[0] == 3:
        attribution_viz = np.mean(attribution.cpu().detach().numpy(), axis=0)
    else:
        attribution_viz = attribution.cpu().detach().numpy()

    # Plot the original image
    ax[0].imshow(image_viz)
    ax[0].set_title(f'Original Image {idx}')
    ax[0].axis('off')

    # Plot the attribution heatmap
    ax[1].imshow(image_viz)
    ax[1].imshow(attribution_viz, cmap='jet', alpha=0.5)  # Overlay heatmap with transparency
    ax[1].set_title(f'{explainer_type} Attribution {idx}')
    ax[1].axis('off')

    wandb.log({f"{experiment_name}_attr_map_{idx}": wandb.Image(fig)})
    plt.close(fig)

