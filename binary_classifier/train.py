from binary_model import LodgingClassifier
from dataloader import WheatOthomapDataset
from train_test import train
import torch
from torch.utils.data import DataLoader, random_split

annotation_path = "/home/bota/Desktop/active_sensing/src/annotation.txt"
dataset_path = "/media/bota/BOTA/wheat/example-run-001_20241014T1739_ortho_dsm.tif"
tile_ortomappixel_path = "/home/bota/Desktop/active_sensing/data/tomatotiles.txt"

import torch

torch.cuda.empty_cache()


class Arguments:

    imagenetpretrain = True

    bal = None
    lr = 2.5e-4

    n_epochs = 20
    freqm = 48
    timem = 48
    mixup = 0.6
    batch_size = 1
    fstride = 10
    tstride = 10
    audio_length = 128
    noise = True
    num_workers = 1
    exp_dir = "/content/drive/MyDrive/Thesis/resout"
    optimizer = "adam"
    metrics = "acc"
    loss = "BCE"

    lrscheduler_start = 5
    lrscheduler_step = 1
    lrscheduler_decay = 0.85

    warmup = False
    wa = False
    wa_start = 1
    wa_end = 5

    n_print_steps = 100
    n_class = 2
    lr_patience = 2
    save_model = True


args = Arguments()


model = LodgingClassifier(
    label_dim=args.n_class,
    fstride=args.fstride,
    tstride=args.tstride,
    input_fdim=128,
    input_tdim=args.audio_length,
    imagenet_pretrain=args.imagenetpretrain,
    model_size="base384",
)


# Define dataset
all_data = WheatOthomapDataset(dataset_path, annotation_path, tile_ortomappixel_path)

# Define split sizes
total_size = len(all_data)
train_size = int(0.7 * total_size)  # 70% for training
valid_size = int(0.15 * total_size)  # 15% for validation
test_size = total_size - train_size - valid_size  # Remaining 15% for testing
print(
    f"Train size: {train_size}, Validation size: {valid_size}, Test size: {test_size}"
)
# Split dataset
train_data, valid_data, test_data = random_split(
    all_data, [train_size, valid_size, test_size]
)

# Create DataLoaders
train_loader = DataLoader(
    train_data,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
)
valid_loader = DataLoader(
    valid_data,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True,
)
test_loader = DataLoader(
    test_data,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True,
)

# for x in train_loader:
#     img_tensor, label_tensor = x
#     print(img_tensor.shape)
#     print(label_tensor)

#     break
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
train(model, train_loader, valid_loader, args)
