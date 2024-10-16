import torch

class ExplainabilityEvaluator:
    def __init__(self, model):
        self.model = model

    def sufficiency(self, inputs, attributions, retention_proportion=0.15):
        batch_size, channels, height, width = inputs.shape
        num_features = channels * height * width
        flattened_attributions = attributions.view(batch_size, num_features)


        with torch.no_grad():
            original_output = self.model(inputs)
            _, original_predictions = torch.max(original_output, dim=1)  # Original predictions

        # Rank features by importance and retain top-k based on the 15% retention proportion
        k = int(retention_proportion * num_features)
        _, topk_indices = torch.topk(flattened_attributions, k=k, dim=1, largest=True)

        perturbed_inputs = torch.zeros_like(inputs)
        for i in range(batch_size):
            perturbed_inputs.view(batch_size, num_features)[i, topk_indices[i]] = inputs.view(batch_size, num_features)[i, topk_indices[i]]

        print(f"Retained features: {k}/{num_features}")

        with torch.no_grad():
            perturbed_output = self.model(perturbed_inputs)
            _, perturbed_predictions = torch.max(perturbed_output, dim=1)

        # class accuracy for perturbed predictions
        accuracy = (perturbed_predictions == original_predictions).float().mean().item()

        return accuracy
