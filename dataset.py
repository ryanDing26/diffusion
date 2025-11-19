import os
import h5py
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
import torch.nn.functional as F


class TissueDataset(Dataset):
    """
    Improved H5 tissue tile dataset with better error handling and metadata encoding.
    """
    
    def __init__(self, csv_path, transform=None, tissue_filter=None, verbose=True, conditional=False):
        """
        Args:
            csv_path (str): Path to CSV with metadata
            transform (callable): torchvision transform for images
            tissue_filter (list[str]): Only load tiles for these tissue types
            verbose (bool): Print loading information
        """
        self.transform = transform
        self.verbose = verbose
        self.conditional = conditional
        # Load and process CSV
        self.df = pd.read_csv(csv_path)
        
        # Apply tissue filter if provided
        if tissue_filter is not None:
            tissue_filter_lower = [t.strip().lower() for t in tissue_filter]
            mask = self.df["Tissue"].str.strip().str.lower().isin(tissue_filter_lower)
            self.df = self.df[mask].reset_index(drop=True)
            if self.verbose:
                print(f"Filtered to tissues: {tissue_filter}")
                print(f"Samples after filtering: {len(self.df)}")
        
        # Validate and filter H5 files
        self.valid_rows = []
        self.tiles = []
        
        for idx, row in self.df.iterrows():
            h5_path = row["Tissue Sample ID"]
            
            # Check if file exists
            if not os.path.exists(h5_path):
                if self.verbose:
                    print(f"⚠️ Missing file: {h5_path}")
                continue
            
            try:
                with h5py.File(h5_path, "r") as f:
                    if "tiles" in f:
                        num_tiles = len(f["tiles"])
                        if num_tiles > 0:
                            self.valid_rows.append(idx)
                            
                            # Add tiles to global index
                            for t in range(num_tiles):
                                self.tiles.append((idx, t, h5_path))
                    else:
                        if self.verbose:
                            print(f"⚠️ No 'tiles' key in {h5_path}")
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Error reading {h5_path}: {e}")
        
        if not self.tiles:
            raise ValueError("No valid tiles found! Check your data paths and CSV.")
        
        # Reset dataframe to only valid rows
        self.df = self.df.iloc[self.valid_rows].reset_index(drop=True)
                
        if self.verbose:
            print(f"✅ Dataset initialized:")
            print(f"   - Total tiles: {len(self.tiles)}")
            print(f"   - Valid slides: {len(self.valid_rows)}")    
    
    def __len__(self):
        return len(self.tiles)
    
    def __getitem__(self, idx):
        row_idx, tile_idx, h5_path = self.tiles[idx]
        
        # Load tile
        try:
            with h5py.File(h5_path, "r") as f:
                tile = f["tiles"][tile_idx][...]
            
            # Convert to PIL Image; is this needed?
            img = Image.fromarray(tile)

            # Apply transforms
            if self.transform:
                img = self.transform(img)
            
            return img
            
        except Exception as e:
            raise RuntimeError(f"Failed to load tile {idx}. Fix your data!") from e


if __name__ == "__main__":
    from torchvision import transforms
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    IMG_SIZE = 256
    # Define transforms
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)), # resize to 256x256
        transforms.RandomHorizontalFlip(), # random horizontal flipping
        transforms.ToTensor(), # scales to [0, 1]
        transforms.Lambda(lambda t: (t * 2) - 1) # scale between [-1, 1]
    ]

    data_transform = transforms.Compose(data_transforms)
    
    # Create dataset
    print("Loading dataset...")
    dataset = TissueDataset(
        csv_path="../age-diffusion/gtex_features.csv",
        transform=data_transform,
        tissue_filter=["Ovary"],
        verbose=True
    )
    
    print(f"\nDataset statistics:")
    print(f"Total tiles: {len(dataset)}")
    
    # Test single sample
    print("\n Testing single sample...")
    sample = dataset[0]
    print(f"Image shape: {sample.shape}")
    # Test dataloader
    print("\nTesting dataloader...")
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    print(f"Batch image shape: {batch.shape}")
    
    # Visualize some samples
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(8):
        sample = dataset[i].permute(1, 2, 0)
        axes[i].imshow(sample)
        axes[i].axis('off')
    
    plt.suptitle("Dataset Samples")
    plt.tight_layout()
    plt.savefig("dataset_preview.png")
    print("\nSaved preview to dataset_preview.png")