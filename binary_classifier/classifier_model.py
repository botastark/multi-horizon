import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class ModifiedClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        """
        Args:
            pretrained: start from ImageNet weights (downloads them on first
                use). Set False when a fine-tuned state dict is loaded right
                after, e.g. in Predicter, so no download is needed.
        """
        super(ModifiedClassifier, self).__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights)
        self.backbone.classifier = nn.Sequential(
            nn.Linear(self.backbone.classifier[0].in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)
