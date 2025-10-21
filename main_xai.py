# File: xai_ct_project/main_xai.py

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import cv2
from captum.attr import LayerGradCam, LayerAttribution

from scripts.dataset_loader import CovidCTDataset

# -----------------------------
# CONFIG
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/resnet18_epoch10.pth"  # latest trained checkpoint
PROCESSED_PATH = "data/preprocessed/SARS-COV-2"
OUTPUT_DIR = "reports/gradcam_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.ioff()  # Disable interactive plotting for headless systems

# -----------------------------
# WRAPPER FOR BINARY MODEL
# -----------------------------
class BinaryToTwoClassWrapper(nn.Module):
    """
    Converts a binary (1-output) model into a pseudo 2-class model
    so Grad-CAM can compute class-wise gradients.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if out.ndim == 1:
            out = out.unsqueeze(1)
        prob = torch.sigmoid(out)
        two_class = torch.cat([1 - prob, prob], dim=1)  # [non-COVID, COVID]
        return two_class


# -----------------------------
# LOAD MODEL
# -----------------------------
print("🧠 Loading trained model...")

base_model = models.resnet18(weights=None)
num_ftrs = base_model.fc.in_features
base_model.fc = nn.Linear(num_ftrs, 1)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
if "model_state_dict" in checkpoint:
    base_model.load_state_dict(checkpoint["model_state_dict"])
else:
    base_model.load_state_dict(checkpoint)

model = BinaryToTwoClassWrapper(base_model).to(DEVICE)
model.eval()

print("✅ Model loaded successfully.")

# -----------------------------
# LOAD TEST DATASET
# -----------------------------
print("📁 Loading test dataset...")

test_dataset = CovidCTDataset(
    json_file=os.path.join(PROCESSED_PATH, "test.json"),
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print(f"📊 Test samples: {len(test_dataset)}")

# -----------------------------
# GRAD-CAM INITIALIZATION
# -----------------------------
target_layer = model.model.layer4[-1]
gradcam = LayerGradCam(model, target_layer)


# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def create_gradcam_overlay(img_tensor, cam, alpha=0.4, cmap="jet"):
    """Create overlay image from Grad-CAM."""
    img = img_tensor.detach().cpu().numpy()

    # Convert (C,H,W) -> (H,W,C)
    if img.ndim == 3:
        if img.shape[0] == 1:
            img = np.repeat(img, 3, axis=0)
        img = np.transpose(img, (1, 2, 0))

    # Normalize input image
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # Process CAM
    heatmap = cam.detach().cpu().numpy()
    if heatmap.ndim == 3:
        heatmap = heatmap[0]
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap = plt.get_cmap(cmap)(heatmap)[..., :3]

    # Overlay
    overlay = (1 - alpha) * img + alpha * heatmap
    return np.clip(overlay, 0, 1)


def save_overlay_image(img_tensor, cam, save_path):
    """Save Grad-CAM overlay image to disk."""
    overlay = create_gradcam_overlay(img_tensor, cam)
    overlay_bgr = cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, overlay_bgr)


# -----------------------------
# GENERATE GRAD-CAMS
# -----------------------------
print("🔍 Generating Grad-CAM heatmaps for test set...")

saved_count = 0
for i, (img, label) in enumerate(test_loader):
    img, label = img.to(DEVICE), label.to(DEVICE)

    try:
        # Compute Grad-CAM
        cam = gradcam.attribute(img, target=int(label.item()))
        cam = LayerAttribution.interpolate(cam, img.shape[2:])

        # Save overlay
        save_path = os.path.join(OUTPUT_DIR, f"gradcam_{i}_class{int(label.item())}.png")
        save_overlay_image(img[0], cam[0], save_path)
        saved_count += 1

        if i % 20 == 0:
            print(f"✅ Processed {i}/{len(test_loader)} images...")

    except Exception as e:
        print(f"⚠️ Skipping sample {i} due to error: {e}")

print(f"\n🎉 Completed! Saved {saved_count} Grad-CAM heatmaps to: {OUTPUT_DIR}")
