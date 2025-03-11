import torch

import torch.nn as nn

# import torchvision.models as models
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class ModifiedClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(ModifiedClassifier, self).__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT
        self.backbone = mobilenet_v3_small(weights=weights)
        # self.backbone = models.efficientnet_b0(
        #     pretrained=True
        # )  # Use EfficientNet as feature extractor
        self.backbone.classifier = nn.Sequential(
            nn.Linear(self.backbone.classifier[0].in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)
