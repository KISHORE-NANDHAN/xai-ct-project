# ============================================================
# File: evaluate.py
# Description: Model evaluation & metrics for XAI-CT Project
# ============================================================

import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.dataset_loader import get_dataloaders
from models.resnet2d import build_model

# ============================================================
# 1️⃣ Load YAML config
# ============================================================
def load_config(path="config/model.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ============================================================
# 2️⃣ Evaluate Function
# ============================================================
def evaluate(model, loader, device, num_classes):
    model.eval()
    preds, targets, probs = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            prob = torch.nn.functional.softmax(outputs, dim=1)
            pred = torch.argmax(prob, dim=1)

            preds.extend(pred.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            probs.extend(prob.cpu().numpy())

    preds = np.array(preds)
    targets = np.array(targets)
    probs = np.array(probs)

    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average="weighted")

    try:
        auc_score = roc_auc_score(targets, probs, multi_class="ovr")
    except Exception:
        auc_score = 0.0

    cm = confusion_matrix(targets, preds)
    report = classification_report(targets, preds, target_names=[f"Class {i}" for i in range(num_classes)])

    return acc, f1, auc_score, cm, report, probs, targets


# ============================================================
# 3️⃣ Plot Confusion Matrix
# ============================================================
def plot_confusion_matrix(cm, save_dir, title="Confusion Matrix"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path)
    plt.close()
    print(f"💾 Confusion matrix saved to {path}")


# ============================================================
# 4️⃣ Plot ROC Curve
# ============================================================
def plot_roc(probs, targets, num_classes, save_dir, title="ROC Curve"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(6, 5))

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve((targets == i).astype(int), probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    path = os.path.join(save_dir, "roc_curve.png")
    plt.savefig(path)
    plt.close()
    print(f"💾 ROC curve saved to {path}")


# ============================================================
# 5️⃣ Main Evaluation Script
# ============================================================
def main():
    cfg = load_config()
    device = torch.device("cuda" if (cfg["device"]["use_gpu"] and torch.cuda.is_available()) else "cpu")
    print(f"🚀 Using device: {device}")

    # Load test loader
    _, _, test_loader = get_dataloaders(cfg)

    # Load model
    model = build_model(cfg["model"]).to(device)
    ckpt_path = os.path.join(cfg["paths"]["save_dir"], cfg["logging"]["checkpoint_name"])
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"📥 Loaded model checkpoint: {ckpt_path}")

    # Evaluate
    num_classes = cfg["model"]["num_classes"]
    acc, f1, auc_score, cm, report, probs, targets = evaluate(model, test_loader, device, num_classes)

    print(f"\n✅ Test Accuracy: {acc:.4f}")
    print(f"✅ Test F1 Score: {f1:.4f}")
    print(f"✅ Test AUC Score: {auc_score:.4f}")
    print("\nClassification Report:\n", report)

    # Save Confusion Matrix & ROC
    plot_confusion_matrix(cm, cfg["paths"]["plots_dir"])
    plot_roc(probs, targets, num_classes, cfg["paths"]["plots_dir"])


if __name__ == "__main__":
    main()
