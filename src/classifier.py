from model.wheat_model_classifier import InceptionResNetV2
import torch
from PIL import Image

import torch.nn as nn
import torch.optim as optim

from torchvision import transforms


num_classes = 3

model = InceptionResNetV2(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.00001)
model.load_state_dict(
    torch.load(
        "/home/bota/Desktop/active_sensing/src/model/model_resnet_single_image.p",
        weights_only=True,
    )
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

transform = transforms.Compose(
    [
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def predict(img_path):
    if isinstance(img_path, str):

        img = Image.open(img_path)
    else:
        img = img_path
    data = transform(img)
    data = data.unsqueeze(0)

    with torch.no_grad():
        data = data.to(device)
        outputs = model(data)
        pred = torch.argmax(outputs, dim=1)
        pred = pred.tolist()
        outputs = torch.nn.functional.softmax(outputs, dim=1)
        outputs = outputs.tolist()

    return pred


# dir_all_tiles = '/home/bota/Downloads/projtiles1/'
# img_path = dir_all_tiles+'DJI_20240607121633_0156_D_point19_tile15_15_crop.png'
# print(predict(img_path))


# Update predict function to handle batch prediction with a fixed batch size
def predict_batch(img_paths, batch_size=64):

    # Initialize an empty list to hold all predictions
    all_predictions = []

    # Process images in batches
    for i in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[i : i + batch_size]

        # Load and transform images in the current batch
        images = [transform(Image.open(img_path)) for img_path in batch_paths]
        images = torch.stack(images)

        # Move images to the same device as the model
        images = images.to(device)

        with torch.no_grad():
            # Make predictions for the current batch
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1).tolist()

        # Append batch predictions to the final list
        all_predictions.extend(predictions)

    return all_predictions
