# File: xai_ct_project/scripts/data_preprocessing.py

import os
from PIL import Image
from tqdm import tqdm
import json
import random

# -----------------------------
# CONFIG
# -----------------------------
RAW_PATH = 'data/raw/SARS-COV-2'
PROCESSED_PATH = 'data/preprocessed/SARS-COV-2'
IMG_SIZE = (224, 224)  # Resize images to 224x224
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1  # Test will be remaining 0.1

# -----------------------------
# CREATE PROCESSED FOLDER
# -----------------------------
os.makedirs(PROCESSED_PATH, exist_ok=True)

# -----------------------------
# PROCESS IMAGES
# -----------------------------
metadata = []

for label in ['covid', 'non-covid']:
    in_dir = os.path.join(RAW_PATH, label)
    out_dir = os.path.join(PROCESSED_PATH, label)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nProcessing {label} images...")
    for img_file in tqdm(os.listdir(in_dir)):
        try:
            img_path = os.path.join(in_dir, img_file)
            img = Image.open(img_path).convert('RGB')  # Ensure RGB
            img = img.resize(IMG_SIZE)
            save_path = os.path.join(out_dir, img_file)
            img.save(save_path)

            metadata.append({
                "filepath": save_path,
                "label": 1 if label == 'covid' else 0,
                "label_name": label
            })
        except Exception as e:
            print(f"Error processing {img_file}: {e}")

# -----------------------------
# SHUFFLE AND SPLIT DATA
# -----------------------------
random.shuffle(metadata)
n = len(metadata)
train_end = int(TRAIN_SPLIT * n)
val_end = int((TRAIN_SPLIT + VAL_SPLIT) * n)

train_data = metadata[:train_end]
val_data = metadata[train_end:val_end]
test_data = metadata[val_end:]

# Save metadata JSON files
with open(os.path.join(PROCESSED_PATH, 'train.json'), 'w') as f:
    json.dump(train_data, f, indent=4)
with open(os.path.join(PROCESSED_PATH, 'val.json'), 'w') as f:
    json.dump(val_data, f, indent=4)
with open(os.path.join(PROCESSED_PATH, 'test.json'), 'w') as f:
    json.dump(test_data, f, indent=4)

print("\n✅ Preprocessing complete!")
print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
