# ============================================================
# File: main_xai.py
# Description: Generate explainability maps (Grad-CAM / Attention)
# ============================================================

import torch
from pathlib import Path
import yaml
from scripts.dataset_loader import get_dataloaders
from models.resnet2d import build_model
from captum.attr import LayerGradCam, visualization as viz
import matplotlib.pyplot as plt

# -----------------------------
# Load config
# -----------------------------
CONFIG_PATH = Path("config/model.yaml")
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda" if (cfg["device"]["use_gpu"] and torch.cuda.is_available()) else "cpu")
save_dir = Path(cfg["paths"]["explain_dir"])
save_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load model
# -----------------------------
model = build_model(cfg["model"]).to(device)
ckpt_path = Path(cfg["paths"]["save_dir"]) / cfg["logging"]["checkpoint_name"]
checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# -----------------------------
# Load test data
# -----------------------------
_, _, test_loader = get_dataloaders(cfg)

# -----------------------------
# Grad-CAM for last conv layer
# -----------------------------
target_layer = model.layer4[-1]  # ResNet last conv block
gradcam = LayerGradCam(model, target_layer)

for idx, (images, labels) in enumerate(test_loader):
    images, labels = images.to(device), labels.to(device)
    for i in range(images.size(0)):
        input_img = images[i].unsqueeze(0)
        label = labels[i].item()
        attr = gradcam.attribute(input_img, target=label)
        attr = attr.cpu().detach().numpy()[0]

        plt.figure(figsize=(5, 5))
        viz.visualize_image_attr(
            attr,
            input_img[0].cpu().permute(1, 2, 0).numpy(),
            method="heat_map",
            sign="absolute",
            show_colorbar=True
        )
        plt.axis("off")
        plt.title(f"Label: {label}")
        plt.savefig(save_dir / f"gradcam_{idx}_{i}.png")
        plt.close()

print(f"✅ Explainability maps saved in: {save_dir}")
