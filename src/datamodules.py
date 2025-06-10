"""
Data loading for 3-channel PNG inputs
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class SliceDataset(Dataset):
    def __init__(self, csv_path, img_dir, augment=False):
        self.df      = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.augment = augment

        base_tf = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ]
        if augment:
            aug_tf = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
            ]
            self.tf = transforms.Compose(aug_tf + base_tf)
        else:
            self.tf = transforms.Compose(base_tf)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        sid   = row.slice_id
        label = float(row["any"])

        img = Image.open(f"{self.img_dir}/{sid}.png").convert("RGB")
        img = self.tf(img)

        return img, torch.tensor(label, dtype=torch.float32)

def make_loaders(batch_size, img_dir, train, dev, test=None):
    loaders = {}
    loaders["train"] = DataLoader(
        SliceDataset(train, img_dir, augment=True),
        batch_size=batch_size, shuffle=True,  num_workers=4
    )
    loaders["dev"]   = DataLoader(
        SliceDataset(dev,   img_dir, augment=False),
        batch_size=batch_size, shuffle=False, num_workers=4
    )
    if test:
        loaders["test"] = DataLoader(
            SliceDataset(test, img_dir, augment=False),
            batch_size=batch_size, shuffle=False, num_workers=4
        )
    return loaders
