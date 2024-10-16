import torch.nn as nn
import timm

class CurriculumVisionTransformer(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(CurriculumVisionTransformer, self).__init__()

        if pretrained:
            print("Loading pre-trained DeiT-Tiny model...")
            self.vit = timm.create_model('deit_tiny_patch16_224', pretrained=True)
        else:
            print("Training DeiT-Tiny model from scratch...")
            self.vit = timm.create_model('deit_tiny_patch16_224', pretrained=False)

        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

    def forward(self, x):
        """Forward pass through the Vision Transformer."""
        return self.vit(x)
