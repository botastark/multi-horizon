import numpy as np
import torch

import torch.nn as nn
from torchvision import transforms
from PIL import Image
from model.wheat_model_classifier import InceptionResNetV2

# #!/usr/bin/env python3
# import sys
# import os

# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

# Now, import the model
from binary_classifier.classifier_model import ModifiedClassifier

import torch

model_path = "/home/bota/Desktop/active_sensing/binary_classifier/models/best_model_auc91_lr1_-05_bs128_wd_2.5-04.pth"


class Predicter(nn.Module):
    def __init__(
        self,
        num_classes=2,
        img_size=180,
        model_weights_path=None,
    ):
        super(Predicter, self).__init__()
        if num_classes == 3:
            # self.img_size= 150,
            self.img_size = 150

            self.model = InceptionResNetV2(num_classes)
            if model_weights_path is None:
                model_weights_path = "/home/bota/Desktop/active_sensing/src/model/model_resnet_single_image.p"

        elif num_classes == 2:
            self.img_size = img_size
            self.model = ModifiedClassifier(num_classes=num_classes)

        else:
            raise ValueError("Invalid number of classes. Expected 2 or 3.")

        if model_weights_path is not None:
            self.model.load_state_dict(
                torch.load(
                    model_weights_path,
                    weights_only=True,
                )
            )

        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device("cpu")
        self.model = self.model.to(self.device)

        # super(ModifiedClassifier, self).__init__()
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.num_classes = num_classes

    def predict(self, img_path):
        self.model.eval()
        if isinstance(img_path, str):
            img = Image.open(img_path)
        else:
            img = img_path
        data = self.transform(img)
        data = data.unsqueeze(0)

        with torch.no_grad():
            data = data.to(self.device)
            outputs = self.model(data)
            if self.num_classes == 3:
                pred = torch.argmax(outputs, dim=1)
                pred = pred.tolist()
                outputs = torch.nn.functional.softmax(outputs, dim=1)
                outputs = outputs.tolist()
                return pred

            else:
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                preds = preds.cpu().numpy()
                preds = np.argmax(preds)
                # pred = np.argmax(preds[0])
                # Convert probabilities to class predictions (0 or 1)
                # preds = torch.argmax(preds, dim=1).tolist()

                return preds

    # Update predict function to handle batch prediction with a fixed batch size
    def predict_batch(self, img_paths, batch_size=64):
        self.model.eval()
        # Initialize an empty list to hold all predictions
        all_predictions = []

        for i in range(0, len(img_paths), batch_size):
            batch_paths = img_paths[i : i + batch_size]
            if isinstance(batch_paths[0], str):
                images = [
                    self.transform(Image.open(img_path)) for img_path in batch_paths
                ]
            else:
                images = [self.transform(img) for img in batch_paths]

            images = torch.stack(images)

            images = images.to(self.device)

            with torch.no_grad():
                outputs = self.model(images)
                if self.num_classes == 3:
                    predictions = torch.argmax(outputs, dim=1).tolist()
                else:
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).float()

                    # Convert probabilities to class predictions (0 or 1)
                    # preds = torch.argmax(preds, dim=1).tolist()
                    predictions = preds.cpu().numpy()

            # Append batch predictions to the final list
            all_predictions.extend(predictions)

        return all_predictions
