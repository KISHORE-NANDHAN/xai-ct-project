# ============================================================
# File: scripts/dataset_loader.py
# Description: Unified DataLoader for COVID-CT XAI Project
# ============================================================

import os
import json
from PIL import Image
from glob import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import random


# ============================================================
# 1️⃣ Dataset Class
# ============================================================
class COVIDCTDataset(Dataset):
    """
    Custom Dataset for preprocessed CT images.
    Reads JSON manifest files of the form:
    [
        {"image_path": "data/preprocessed/COVIDx/img_001.png", "label": 1},
        {"image_path": "data/preprocessed/SARS-CoV2/img_002.png", "label": 0},
        ...
    ]
    """

    def __init__(self, manifest_files, transform=None):
        self.samples = []
        self.transform = transform

        # Support multiple manifest JSONs (auto-detect datasets)
        if isinstance(manifest_files, list):
            for mf in manifest_files:
                if os.path.exists(mf):
                    with open(mf, "r") as f:
                        self.samples.extend(json.load(f))
        else:
            if os.path.exists(manifest_files):
                with open(manifest_files, "r") as f:
                    self.samples = json.load(f)

        # Safety check
        if len(self.samples) == 0:
            raise ValueError("❌ No samples found. Check your JSON paths or preprocessing output!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # Get the file path (supports multiple key names)
        img_path = item.get("image_path") or item.get("file") or item.get("file_path") or item.get("path")
        if img_path is None:
            raise KeyError(f"❌ No image path found in sample: {item}")

        # Load image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Map label string to integer
        label_map = {"COVID": 0, "NORMAL": 1, "PNEUMONIA": 2}  # adjust based on your dataset
        label_str = item.get("label")
        if label_str is None:
            # fallback: infer from folder name
            folder = os.path.basename(os.path.dirname(img_path))
            label = label_map.get(folder.upper(), 0)
        else:
            label = label_map.get(str(label_str).upper(), 0)

        return image, torch.tensor(label, dtype=torch.long)

# ============================================================
# 2️⃣ Helper Functions
# ============================================================

def get_transforms(split="train", input_size=224):
    """
    Define augmentations for training and normalization for val/test.
    """
    if split == "train":
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    return transform


def auto_discover_manifests(data_dir="data", split="train"):
    """
    Automatically find JSON manifests for a given split (train/val/test)
    across all datasets (COVIDx, SARS-CoV2, COVID-CT-MD, etc.).
    """
    pattern = os.path.join(data_dir, f"*_{split}.json")
    manifests = glob(pattern)

    # Also check root data folder if not in preprocessed/
    if len(manifests) == 0:
        manifests = glob(os.path.join(data_dir, f"{split}_*.json"))

    if len(manifests) == 0:
        raise FileNotFoundError(f"⚠️ No {split} manifest files found under {data_dir}")

    return manifests


# ============================================================
# 3️⃣ Main Function: Get Dataloaders
# ============================================================

def get_dataloaders(cfg):
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["device"]["num_workers"]

    active_dataset = cfg["dataset"]["active"]
    dataset_cfg = cfg["dataset"]["available"][active_dataset]

    input_size = dataset_cfg["input_size"]

    # JSON manifests directly from config
    train_jsons = [dataset_cfg["train_json"]]
    val_jsons   = [dataset_cfg["val_json"]]
    test_jsons  = [dataset_cfg["test_json"]]

    # Transforms
    train_tf = get_transforms("train", input_size)
    val_tf   = get_transforms("val", input_size)
    test_tf  = get_transforms("test", input_size)

    # Datasets
    train_ds = COVIDCTDataset(train_jsons, transform=train_tf)
    val_ds   = COVIDCTDataset(val_jsons, transform=val_tf)
    test_ds  = COVIDCTDataset(test_jsons, transform=test_tf)

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader



# ============================================================
# 4️⃣ Debug / Standalone Run
# ============================================================
if __name__ == "__main__":
    import yaml

    # Example config for testing
    dummy_cfg = yaml.safe_load("""
    model:
      input_size: 224
    training:
      batch_size: 4
    device:
      num_workers: 0
    """)

    train_loader, val_loader, test_loader = get_dataloaders(dummy_cfg)
    print(f"Train batches: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")

    images, labels = next(iter(train_loader))
    print(f"Image batch shape: {images.shape} | Labels: {labels}")
