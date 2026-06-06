"""
coin_augment.py  -  Data augmentation for SA coin dataset

Run AFTER coin_reorganise.py, from your project root:
    python coin_augment.py

Reads from data/coins/raw/, writes augmented images to data/coins/augmented/,
and generates coin_labels.csv with one row per output image.

Each original image produces 15 augmented versions:
    1 orig + 7 rotations + 2 contrast + 2 brightness + 2 blur + 1 noise = 15

This mirrors the banknote augment.py pipeline exactly so both datasets
feed into the same training script with no extra preprocessing differences.
"""

import os
import cv2
import csv
import numpy as np

# Config
INPUT_DIR  = os.path.join("data", "coins", "raw")
OUTPUT_DIR = os.path.join("data", "coins", "augmented")
CSV_PATH   = "coin_labels.csv"

ROTATION_ANGLES  = [45, 90, 135, 180, 225, 270, 315]
CONTRAST_FACTORS = [1.2, 1.5]
BRIGHTNESS_VALS  = [30, 60]
BLUR_KERNELS     = [(3, 3), (5, 5)]


def rotate_with_white_bg(image, angle):
    """Rotate image and fill exposed corners with white (not black).
    Canvas is expanded so the coin isn't clipped at the edges."""
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w / 2) - cx
    M[1, 2] += (new_h / 2) - cy

    return cv2.warpAffine(
        image, M, (new_w, new_h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )


def boost_contrast(image, factor):
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)


def boost_brightness(image, value):
    return cv2.convertScaleAbs(image, alpha=1.0, beta=value)


def add_gaussian_noise(image, mean=0, stddev=10):
    noise = np.random.normal(mean, stddev, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def parse_folder_name(folder_name):
    """Expects folders named: 10c_new_back, R1_old_front, etc.
    Order: denomination_era_side  (same convention as banknote folders)"""
    parts = folder_name.split("_")
    denomination = parts[0]   # 5c, 10c, 20c, 50c, R1, R2, R5
    era          = parts[1]   # old, new
    side         = parts[2]   # front, back
    return denomination, era, side


os.makedirs(OUTPUT_DIR, exist_ok=True)
csv_rows = []
total_saved = 0

for class_folder in sorted(os.listdir(INPUT_DIR)):
    class_path = os.path.join(INPUT_DIR, class_folder)
    if not os.path.isdir(class_path):
        continue

    out_class_path = os.path.join(OUTPUT_DIR, class_folder)
    os.makedirs(out_class_path, exist_ok=True)

    denomination, era, side = parse_folder_name(class_folder)

    image_files = [
        f for f in os.listdir(class_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    for fname in image_files:
        image = cv2.imread(os.path.join(class_path, fname))
        if image is None:
            print(f"  Could not load {fname}, skipping")
            continue

        base = os.path.splitext(fname)[0]

        def save(img, suffix):
            global total_saved
            out_name = f"{base}_{suffix}.jpg"
            cv2.imwrite(os.path.join(out_class_path, out_name), img)
            csv_rows.append({
                "filename":     os.path.join(class_folder, out_name),
                "denomination": denomination,
                "era":          era,
                "side":         side,
                "class":        class_folder,
            })
            total_saved += 1

        save(image, "orig")

        for angle in ROTATION_ANGLES:
            save(rotate_with_white_bg(image, angle), f"rot{angle}")

        for factor in CONTRAST_FACTORS:
            save(boost_contrast(image, factor), f"contrast{factor:.1f}")

        for value in BRIGHTNESS_VALS:
            save(boost_brightness(image, value), f"bright{value}")

        for k in BLUR_KERNELS:
            save(cv2.GaussianBlur(image, k, 0), f"blur{k[0]}x{k[1]}")

        save(add_gaussian_noise(image), "noise")

        print(f"  Done: {class_folder}/{fname}")

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["filename", "denomination", "era", "side", "class"]
    )
    writer.writeheader()
    writer.writerows(csv_rows)

print(f"\nDone! {total_saved} augmented images saved to '{OUTPUT_DIR}'")
print(f"Labels written to '{CSV_PATH}'")
print(f"\nPer original: 1 orig + 7 rotations + 2 contrast + 2 brightness + 2 blur + 1 noise = 15")
