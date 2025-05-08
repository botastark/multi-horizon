import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# #!/usr/bin/env python3
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from binary_classifier.classifier_model import ModifiedClassifier


class Predicter(nn.Module):
    def __init__(
        self,
        num_classes=2,
        img_size=180,
        model_weights_path=None,
    ):
        """
        Initialize the Predicter.

        Parameters:
            num_classes (int): Number of output classes (currently only 2 is supported).
            img_size (int): The target size for image resizing.
            model_weights_path (str): Path to the model weights file.
        """
        super(Predicter, self).__init__()
        if num_classes == 2:
            self.img_size = img_size
            self.model = ModifiedClassifier(num_classes=num_classes)

        else:
            raise ValueError("Invalid number of classes. Expected 2.")

        if model_weights_path is not None:
            self.model.load_state_dict(
                torch.load(
                    model_weights_path,
                    weights_only=True,
                    map_location=torch.device("cpu"),
                )
            )
        # Set device to CPU (update if GPU support is desired)
        self.device = torch.device("cpu")
        self.model = self.model.to(self.device)
        # Define the image transformation pipeline
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
        """
        Predict the class for a single image.

        Parameters:
            img_input (str, PIL.Image.Image, or list): A single image path, a PIL Image.

        Returns:
            For binary classification: Returns an integer (0 or 1) for a single image.
        """
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
            if self.num_classes == 2:
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                preds = preds.cpu().numpy()
                preds = np.argmax(preds)
                return preds
            # else:
            #     pred = torch.argmax(outputs, dim=1)
            #     pred = pred.tolist()
            #     outputs = torch.nn.functional.softmax(outputs, dim=1)
            #     outputs = outputs.tolist()
            #     return pred

    # Update predict function to handle batch prediction with a fixed batch size
    def predict_batch(self, img_paths, batch_size=64):
        """
        Predict classes for a batch of images/paths.

        Parameters:
            img_inputs (list): A list where each element is either a file path (str) or a PIL Image.
            batch_size (int): Batch size for prediction.

        Returns:
            list: A list of predictions for the input images.
        """
        self.model.eval()
        all_predictions = []

        for i in range(0, len(img_paths), batch_size):
            batch_paths = img_paths[i : i + batch_size]
            # Load images based on input type
            if isinstance(batch_paths[0], str):
                images = [
                    self.transform(Image.open(img_path)) for img_path in batch_paths
                ]
            else:
                images = [self.transform(img) for img in batch_paths]

            images = torch.stack(images).to(self.device)

            with torch.no_grad():
                outputs = self.model(images)
                if self.num_classes == 2:
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).float()
                    predictions = preds.cpu().numpy()
                else:
                    predictions = torch.argmax(outputs, dim=1).tolist()
            all_predictions.extend(predictions)
        return all_predictions
