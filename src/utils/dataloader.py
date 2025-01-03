import os
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np
from typing import Callable

# computed on training set

MEAN = 0.0253
STD = 0.0243

def standardization(x: torch.Tensor,
              mean: float = MEAN,
              std: float = STD) -> torch.Tensor:
    return (x - mean)/std

def unstandardization(x: torch.Tensor,
                mean: float = MEAN,
                std: float = STD) -> torch.Tensor:
    return std * x + mean


# Create a custom dataset if there's no specific structure
class CTDataset(datasets.VisionDataset):
    def __init__(self, root_dir, transform=None):
        super().__init__(root_dir, transform=transform)
        self.root_dir = root_dir
        self.transform = transform
        self.ct_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]

    def __len__(self):
        return len(self.ct_paths)

    def __getitem__(self, index):

        ct_path = self.ct_paths[index]
        ct = np.rot90(np.load(ct_path), 2)
        ct = torch.from_numpy(ct.copy()).unsqueeze(0)
        
        if self.transform:
            ct = self.transform(ct)
        
        return ct 

# Usage example
root_dir_train = "./data/train_slices"
ct_train_dataset = CTDataset(root_dir=root_dir_train, 
                             transform=standardization)

# Adjust batch_size to fit in memory
ct_train_dataloader = DataLoader(ct_train_dataset,
                           batch_size=1, 
                           shuffle=True)