# ============================================
# Dataset Card Generator - XAI-CT Project
# ============================================

import json
import os
import yaml
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
import numpy as np

# ---------------------------------------------------------
# Load configuration
# ---------------------------------------------------------
CONFIG_PATH = Path("config/preprocess.yaml")
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

DATASET_NAME = CONFIG["datasets"]["active"]
CURATED_DIR = Path(CONFIG["paths"]["curated_data"]) / DATASET_NAME
META_FILE = CURATED_DIR / CONFIG["metadata"]["output_file"]
CARD_OUT = Path(CONFIG["paths"]["dataset_card"])
FIGURE_DIR = Path(CONFIG["paths"]["figures"])
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Generate dataset card
# ---------------------------------------------------------
def generate_dataset_card():
    # Load metadata
    if not META_FILE.exists():
        print(f"❌ Metadata file not found: {META_FILE}")
        return
    meta = json.load(open(META_FILE))

    # Class counts
    labels = [m["label"] for m in meta]
    label_names = [m.get("label_name", str(m["label"])) for m in meta]
    counts = Counter(labels)
    total = len(meta)

    # Optional intensity stats (for CT datasets)
    intensities = []
    for m in meta:
        if "intensity_range" in m:
            intensities.append(m["intensity_range"])
    intensities = np.array(intensities)
    intensity_stats = {}
    if len(intensities) > 0:
        intensity_stats = {
            "min": float(np.min(intensities[:, 0])),
            "max": float(np.max(intensities[:, 1])),
            "mean": float(np.mean(intensities[:, 0])),
            "std": float(np.std(intensities[:, 1]))
        }

    # Build stats dict
    stats = {
        "project": CONFIG["project"]["name"],
        "version": CONFIG["project"]["version"],
        "dataset": DATASET_NAME,
        "total_images": total,
        "class_distribution": {str(k): v for k, v in counts.items()},
        "image_size": CONFIG["preprocess"]["image_size"],
        "augmentation": CONFIG["preprocess"]["augmentations"],
        "normalization": CONFIG["preprocess"]["normalization"],
        "source": CONFIG["datasets"]["available"][DATASET_NAME]["source"],
        "intensity_stats": intensity_stats
    }

    # Save dataset card JSON
    CARD_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(CARD_OUT, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"✅ Dataset card saved at: {CARD_OUT}")

    # ---------------------------------------------------------
    # Plot class distribution
    # ---------------------------------------------------------
    plt.figure(figsize=(6, 6))
    plt.pie(counts.values(), labels=[CONFIG["datasets"]["available"][DATASET_NAME]["classes"][k] if isinstance(k,int) else k for k in counts.keys()],
            autopct="%1.1f%%", startangle=90, colors=plt.cm.Set3.colors)
    plt.title(f"{DATASET_NAME} - Class Distribution")
    plt.tight_layout()
    pie_path = FIGURE_DIR / "class_distribution.png"
    plt.savefig(pie_path, dpi=200)
    plt.close()
    print(f"📊 Class distribution figure saved at: {pie_path}")

# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    generate_dataset_card()
