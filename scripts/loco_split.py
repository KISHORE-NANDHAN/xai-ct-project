# ============================================
# LOCO / Stratified Split Generator - XAI-CT
# ============================================

import json
import yaml
import random
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split, GroupShuffleSplit

# ---------------------------------------------------------
# Load config
# ---------------------------------------------------------
CONFIG_PATH = Path("config/preprocess.yaml")
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

DATASET_NAME = CONFIG["datasets"]["active"]
CURATED_DIR = Path(CONFIG["paths"]["curated_data"]) / DATASET_NAME
META_FILE = CURATED_DIR / CONFIG["metadata"]["output_file"]
OUT_DIR = CURATED_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Logging
LOG_FILE = Path(CONFIG["paths"]["logs"]) / "splits.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logging.info(f"📌 Starting split generation for dataset: {DATASET_NAME}")

# ---------------------------------------------------------
# Load metadata
# ---------------------------------------------------------
with open(META_FILE) as f:
    meta = json.load(f)

labels = [m["label"] for m in meta]

# ---------------------------------------------------------
# Choose splitting method
# ---------------------------------------------------------
if CONFIG["split"]["method"].lower() == "loco":
    logging.info("Using LOCO (Leave-One-Group-Out) split")
    # Expect metadata to contain a 'site' or 'scanner' field
    groups = [m.get("site", "site1") for m in meta]  # fallback site1
    splitter = GroupShuffleSplit(n_splits=1, test_size=CONFIG["split"]["test_ratio"], random_state=CONFIG["split"]["seed"])
    train_idx, test_idx = next(splitter.split(meta, groups=labels, groups=groups))
    train_temp = [meta[i] for i in train_idx]
    test = [meta[i] for i in test_idx]

    # Split train_temp into train/val
    val_size = CONFIG["split"]["val_ratio"] / (CONFIG["split"]["train_ratio"] + CONFIG["split"]["val_ratio"])
    train, val = train_test_split(
        train_temp,
        test_size=val_size,
        stratify=[m["label"] for m in train_temp],
        random_state=CONFIG["split"]["seed"]
    )

else:
    logging.info("Using Stratified split")
    # Classic stratified split
    train, temp = train_test_split(
        meta,
        test_size=(CONFIG["split"]["val_ratio"] + CONFIG["split"]["test_ratio"]),
        stratify=labels,
        random_state=CONFIG["split"]["seed"]
    )
    val_size = CONFIG["split"]["test_ratio"] / (CONFIG["split"]["val_ratio"] + CONFIG["split"]["test_ratio"])
    val, test = train_test_split(
        temp,
        test_size=val_size,
        stratify=[m["label"] for m in temp],
        random_state=CONFIG["split"]["seed"]
    )

# ---------------------------------------------------------
# Save manifests
# ---------------------------------------------------------
splits = {"train": train, "val": val, "test": test}
for split_name, split_data in splits.items():
    split_path = OUT_DIR / f"{split_name}_manifest.json"
    with open(split_path, "w") as f:
        json.dump(split_data, f, indent=2)
    logging.info(f"✅ {split_name.capitalize()} manifest created: {split_path}")
    print(f"✅ {split_name.capitalize()} manifest created: {split_path}")

logging.info("🏁 Split generation completed.")
