# File: xai_ct_project/main_train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from scripts.dataset_loader import get_dataloaders
from sklearn.metrics import accuracy_score, roc_auc_score
import os
from torch.utils.data import DataLoader

# -----------------------------
# CONFIG
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
CHECKPOINT_DIR = "models"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def main():
    # -----------------------------
    # LOAD DATA
    # -----------------------------
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)

    # -----------------------------
    # MODEL
    # -----------------------------
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # updated for new torchvision
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)  # Binary output
    model = model.to(DEVICE)

    # -----------------------------
    # LOSS & OPTIMIZER
    # -----------------------------
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # -----------------------------
    # TRAINING LOOP
    # -----------------------------
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = torch.sigmoid(outputs).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = accuracy_score(all_labels, [1 if p>0.5 else 0 for p in all_preds])
        try:
            train_auroc = roc_auc_score(all_labels, all_preds)
        except:
            train_auroc = 0.5

        print(f"Epoch [{epoch}/{NUM_EPOCHS}] | Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | AUROC: {train_auroc:.4f}")

        scheduler.step()

        # -----------------------------
        # VALIDATION
        # -----------------------------
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
                outputs = model(imgs)
                preds = torch.sigmoid(outputs).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, [1 if p>0.5 else 0 for p in val_preds])
        try:
            val_auroc = roc_auc_score(val_labels, val_preds)
        except:
            val_auroc = 0.5

        print(f"Validation | Acc: {val_acc:.4f} | AUROC: {val_auroc:.4f}")

        # -----------------------------
        # SAVE CHECKPOINT
        # -----------------------------
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"resnet18_epoch{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}\n")

    # -----------------------------
    # TEST EVALUATION
    # -----------------------------
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
            outputs = model(imgs)
            preds = torch.sigmoid(outputs).cpu().numpy()
            test_preds.extend(preds)
            test_labels.extend(labels.cpu().numpy())

    test_acc = accuracy_score(test_labels, [1 if p>0.5 else 0 for p in test_preds])
    try:
        test_auroc = roc_auc_score(test_labels, test_preds)
    except:
        test_auroc = 0.5

    print(f"\n✅ Test Results | Acc: {test_acc:.4f} | AUROC: {test_auroc:.4f}")


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # Required for Windows
    main()
