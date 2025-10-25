# ============================================================
# File: main_train.py
# Description: Model training & validation for XAI-CT Project
# ============================================================

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
from tqdm import tqdm
from datetime import datetime
import pandas as pd

from scripts.dataset_loader import get_dataloaders
from models.resnet2d import build_model_from_config
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# ============================================================
# 1️⃣ Utility: Load YAML config
# ============================================================
def load_config(path="config/model.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ============================================================
# 2️⃣ Training & Validation Functions
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    running_loss = 0.0
    preds, targets = [], []

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with amp.autocast(enabled=scaler is not None):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average="weighted")
    return epoch_loss, acc, f1


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds, targets = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average="weighted")

    try:
        auc = roc_auc_score(
            targets,
            torch.nn.functional.one_hot(torch.tensor(preds), num_classes=len(set(targets))),
            multi_class="ovr"
        )
    except:
        auc = 0.0

    return epoch_loss, acc, f1, auc


# ============================================================
# 3️⃣ Checkpointing & Logging
# ============================================================
def save_checkpoint(model, optimizer, epoch, save_dir, filename="best_model.pth"):
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, filename)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict()
    }, ckpt_path)
    print(f"💾 Saved checkpoint: {ckpt_path}")


def log_metrics(metrics_dir, history):
    os.makedirs(metrics_dir, exist_ok=True)
    file_path = os.path.join(metrics_dir, "training_log.csv")
    df = pd.DataFrame(history)
    df.to_csv(file_path, index=False)
    print(f"📊 Metrics logged to: {file_path}")


# ============================================================
# 4️⃣ Device Selection
# ============================================================
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "xpu") and torch.backends.xpu.is_available():
        device = torch.device("xpu")
        print("🚀 Using Intel GPU (Xe)")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU. Training will be slower.")
    return device


# ============================================================
# 5️⃣ Main Training Engine
# ============================================================
def main():
    cfg = load_config()

    # Device
    device = get_device()

    # Determine active dataset & num_classes
    active_dataset = cfg["dataset"]["active"]
    num_classes = cfg["dataset"]["available"][active_dataset]["num_classes"]
    cfg["model"]["num_classes"] = num_classes

    # Dataloaders
    train_loader, val_loader, _ = get_dataloaders(cfg)

    # Model
    model = build_model_from_config(cfg).to(device)

    # Criterion
    criterion = getattr(nn, cfg["training"]["criterion"])()

    # Optimizer
    optimizer_type = cfg["training"]["optimizer"].lower()
    lr = float(cfg["training"]["learning_rate"])
    wd = float(cfg["training"]["weight_decay"])

    if optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=wd)

    # Scheduler
    scheduler = None
    if cfg["scheduler"]["type"] == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=cfg["scheduler"]["step_size"],
                                              gamma=cfg["scheduler"]["gamma"])

    # Mixed precision only on CUDA
    scaler = amp.GradScaler() if (cfg["training"]["use_amp"] and device.type == "cuda") else None

    # Training loop
    best_auc = 0.0
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": [], "val_auc": []}

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        print(f"\n🟢 Epoch {epoch}/{cfg['training']['epochs']}")
        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_acc, val_f1, val_auc = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)

        # Save best model
        if val_auc > best_auc and cfg["logging"]["save_best_only"]:
            best_auc = val_auc
            save_checkpoint(model, optimizer, epoch,
                            cfg["paths"]["save_root"], cfg["logging"]["checkpoint_name"])

        # Step scheduler
        if scheduler:
            scheduler.step()

    # Log metrics
    log_metrics(cfg["paths"]["metrics_root"], history)
    print("\n✅ Training complete. Best AUC:", best_auc)


# ============================================================
# 6️⃣ Entry Point
# ============================================================
if __name__ == "__main__":
    main()
