import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from dataset import TissueDataset

IMG_SIZE = 256
BATCH_SIZE = 4

def load_dataloaders():
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)), # resize to 256x256
        transforms.RandomHorizontalFlip(), # random horizontal flipping
        transforms.ToTensor(), # scales to [0, 1]
        transforms.Lambda(lambda t: (t * 2) - 1) # scale between [-1, 1]
    ]

    data_transform = transforms.Compose(data_transforms)

    dataset = TissueDataset(
        csv_path="../age-diffusion/gtex_features_512.csv",
        transform=data_transform,
        tissue_filter=["Ovary"],
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    return dataloader
    # # Compute split sizes
    # total_len = len(dataset)
    # train_len = int(total_len * 0.8)
    # test_len = total_len - train_len

    # # Split dataset
    # train_dataset, test_dataset = random_split(dataset, [train_len, test_len])

    # # Create DataLoaders
    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    # test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # return train_loader, test_loader