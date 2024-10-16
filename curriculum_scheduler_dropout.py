import torch
import numpy as np

class CurriculumScheduler:
    def __init__(self, initial_theta=1.0, final_theta=0.5, warmup_epochs=5, total_epochs=150, gamma=0.005, decay_type='exponential', metric_type='epoch'):

        self.initial_theta = initial_theta  # No dropout at the start
        self.final_theta = final_theta  # Final dropout retain probability
        self.warmup_epochs = warmup_epochs  # No dropout for the first few epochs
        self.total_epochs = total_epochs  # Total number of epochs
        self.gamma = gamma  # Decay rate for exponential decay
        self.decay_type = decay_type  # Choose between 'linear' or 'exponential' decay
        self.metric_type = metric_type  # Either 'epoch' or based on an explainability score

    def get_retain_probability(self, explainability_score=None, epoch=None):

        retain_prob = None

        # Epoch-based scheduling
        if self.metric_type == 'epoch':
            if epoch is not None:
                # No dropout during warmup phase (first few epochs)
                if epoch < self.warmup_epochs:
                    retain_prob = self.initial_theta  # θ = 1, no dropout
                else:
                    # Normalize epoch progression after warmup
                    normalized_epoch = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)

                    # Adjust based on either linear or exponential decay
                    if self.decay_type == 'linear':
                        retain_prob = self.initial_theta - normalized_epoch * (self.initial_theta - self.final_theta)
                    elif self.decay_type == 'exponential':
                        retain_prob = (self.initial_theta - self.final_theta) * np.exp(
                            -self.gamma * (epoch - self.warmup_epochs)) + self.final_theta
            else:
                raise ValueError("Epoch-based scheduling requires 'epoch' to be passed.")

        # Explainability-based scheduling
        elif self.metric_type in ['sufficiency', 'infidelity', 'sensitivity']:
            if explainability_score is not None:
                # Use torch.clamp() to ensure explainability_score is between 0 and 1
                score = torch.clamp(explainability_score, 0, 1)  # Ensure score is in [0, 1]
                retain_prob = self.final_theta + score * (self.initial_theta - self.final_theta)
            else:
                raise ValueError("Explainability-based scheduling requires 'explainability_score' to be passed.")

        if retain_prob is None:
            raise ValueError("retain_prob was not set. Check the metric_type and input parameters.")

        return retain_prob

    def apply_dropout(self, input_tensor, explainability_score=None, epoch=None):
        """
        Apply Bernoulli noise to the images based on either epoch or explainability score.
        """
        retain_prob = self.get_retain_probability(explainability_score, epoch)

        # Ensure retain_prob is a scalar
        if isinstance(retain_prob, torch.Tensor):
            retain_prob = retain_prob.item()

        device = input_tensor.device

        if input_tensor.dim() == 4:  # For a batch of images (B, C, H, W)
            B, C, H, W = input_tensor.shape

            binary_mask = torch.bernoulli(torch.full((B, C, H, W), retain_prob)).to(device)


            output = input_tensor * binary_mask

        else:  # For a single image (C, H, W)
            C, H, W = input_tensor.shape


            binary_mask = torch.bernoulli(torch.full((C, H, W), retain_prob)).to(device)


            output = input_tensor * binary_mask

        return output







