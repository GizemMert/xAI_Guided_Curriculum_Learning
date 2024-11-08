import torch
from captum.attr import IntegratedGradients, Saliency, InputXGradient, GuidedGradCam, DeepLift, GuidedBackprop
from captum.metrics import infidelity, sensitivity_max

class Explainability:
    def __init__(self, model):
        self.model = model


    def compute_saliency(self, inputs, target):
        """Compute Saliency Maps."""
        saliency = Saliency(self.model)
        attributions = saliency.attribute(inputs, target=target)
        return attributions


    def compute_input_x_gradient(self, inputs, target):
        """Compute Input X Gradient."""
        ixg = InputXGradient(self.model)
        attributions = ixg.attribute(inputs, target=target)
        return attributions


    def compute_guided_backprop(self, inputs, target):
        """Compute Guided Backpropagation Attributions."""
        gbp = GuidedBackprop(self.model)
        attributions = gbp.attribute(inputs, target=target)
        return attributions


    ### Explainability Metrics ###

    def calculate_sufficiency(self, inputs, attributions, retention_proportions=[0.15, 0.30, 0.50]):
        batch_size = inputs.shape[0]
        flattened_attributions = attributions.view(batch_size, -1)

        num_features = flattened_attributions.shape[1]

        sufficiency_scores = {}

        # Original output
        with torch.no_grad():
            original_output = self.model(inputs)
            _, original_predictions = torch.max(original_output, dim=1)

        for retention_proportion in retention_proportions:
            k = int(retention_proportion * num_features)

            # Extract top-k indices for each example in the batch
            _, topk_indices = torch.topk(flattened_attributions, k=k, dim=1, largest=True)

            perturbed_inputs = torch.zeros_like(inputs)
            flattened_inputs = inputs.view(batch_size, -1)

            for i in range(batch_size):
                perturbed_inputs.view(batch_size, -1)[i, topk_indices[i]] = flattened_inputs[i, topk_indices[i]]

            print(f"Retained features: {k}/{num_features}")

            # Compute output for perturbed input
            with torch.no_grad():
                perturbed_output = self.model(perturbed_inputs)
                _, perturbed_predictions = torch.max(perturbed_output, dim=1)  # Perturbed predictions

            # Class accuracy for perturbed predictions
            accuracy = (perturbed_predictions == original_predictions).float().mean().item()

            # Sufficiency score (accuracy) for this retention proportion
            sufficiency_scores[retention_proportion] = accuracy

        return sufficiency_scores

    def calculate_infidelity(self, perturb_fn, inputs, attributions, target):

        infidelity_score = infidelity(self.model, perturb_fn, inputs, attributions, target=target)
        
        return infidelity_score.mean()


    def calculate_sensitivity(self, explainer_method, inputs, target):

        sensitivity_score = sensitivity_max(explainer_method, inputs, target=target)
        return sensitivity_score.mean()
