# ============================================================
# File: scripts/train_utils.py
# Description: Training & validation helper functions
# ============================================================

import torch
from tqdm import tqdm
import os

# -----------------------------
# Train for one epoch
# -----------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, use_amp=False):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss, all_preds, all_labels


# -----------------------------
# Validate for one epoch
# -----------------------------
def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss, all_preds, all_labels


# -----------------------------
# Save checkpoint
# -----------------------------
def save_checkpoint(model, optimizer, epoch, path, best_metric=None, metric_name="val_auc"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_metric": best_metric
    }
    torch.save(state, path)
    print(f"💾 Checkpoint saved at: {path}")


# -----------------------------
# Load checkpoint
# -----------------------------
def load_checkpoint(model, optimizer, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    print(f"📥 Checkpoint loaded from: {path}")
    return checkpoint.get("epoch", 0), checkpoint.get("best_metric", None)
