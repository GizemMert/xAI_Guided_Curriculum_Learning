import timm
import torch.nn as nn

class CurriculumConvNeXt(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(CurriculumConvNeXt, self).__init__()

        if pretrained:
            print("Loading pre-trained ConvNeXt model...")
            self.convnext = timm.create_model('convnext_tiny', pretrained=True)
        else:
            print("Training ConvNeXt model from scratch...")
            self.convnext = timm.create_model('convnext_tiny', pretrained=False)

        self.convnext.head.fc = nn.Linear(self.convnext.head.fc.in_features, num_classes)

    def forward(self, x):
        return self.convnext(x)

    