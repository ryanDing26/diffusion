import torch
import numpy as np
from dataset import TissueDataset
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

GPU = ""
IMG_SIZE = 256
BATCH_SIZE = 4 if GPU == "3090" else 8

def load_dataloaders(distributed=False):
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

    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=(sampler is None), sampler=sampler, num_workers=0)
    return dataloader, sampler
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