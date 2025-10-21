# File: xai_ct_project/scripts/dataset_loader.py

import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

# -----------------------------
# CUSTOM DATASET
# -----------------------------
class CovidCTDataset(Dataset):
    def __init__(self, json_file, transform=None):
        with open(json_file) as f:
            self.data = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item['filepath']
        label = item['label']

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


# -----------------------------
# FUNCTION TO CREATE DATALOADERS
# -----------------------------
def get_dataloaders(processed_path='data/preprocessed/SARS-COV-2',
                    batch_size=16, num_workers=2):
    
    # TRANSFORMS
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),  # Converts [0,255] -> [0,1]
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    
    # DATASETS
    train_dataset = CovidCTDataset(json_file=f'{processed_path}/train.json', transform=train_transform)
    val_dataset = CovidCTDataset(json_file=f'{processed_path}/val.json', transform=test_transform)
    test_dataset = CovidCTDataset(json_file=f'{processed_path}/test.json', transform=test_transform)
    
    # DATALOADERS
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader
