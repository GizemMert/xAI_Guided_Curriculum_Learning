import timm
import torch
import torch.nn as nn

class CurriculumMLPMixer(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(CurriculumMLPMixer, self).__init__()

        if pretrained:
            print("Loading pre-trained MLP-Mixer model...")
            self.mixer = timm.create_model('mixer_b16_224', pretrained=True)
        else:
            print("Training MLP-Mixer model from scratch...")
            self.mixer = timm.create_model('mixer_b16_224', pretrained=False)

        # Replace the head to match the number of output classes
        self.mixer.head = nn.Linear(self.mixer.head.in_features, num_classes)

    def forward(self, x):
        """Forward pass through the MLP-Mixer."""
        return self.mixer(x)