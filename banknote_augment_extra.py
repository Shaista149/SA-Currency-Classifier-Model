"""
banknote_augment_extra.py - Extra augmentation for SA banknote dataset

Writes new augmented images to a SEPARATE folder (banknotes/augmented_extra/)
and creates a NEW CSV (banknote_labels_extra.csv).

Nothing in your existing augmented/ folder or banknote_labels.csv is touched.
When you're happy with the results, manually copy the images across and
append the CSV rows yourself.

Run from Currency_Classifier/data/:
    python banknote_augment_extra.py

New variants per original image:
    2 perspective warps
    2 shadow overlays
    2 motion blurs
    2 background composites
    2 combined (perspective warp + shadow)
    = 10 new variants per original image

Folder structure expected:
    Currency_Classifier/data/
        background_images/          <- your 70 background photos
        banknote_labels.csv         <- existing CSV (NOT touched)
        banknotes/
            raw/                    <- original flat note images (read from)
            augmented/              <- existing folder (NOT touched)
            augmented_extra/        <- new images written here (created automatically)
        banknote_labels_extra.csv   <- new CSV written here
"""

import os
import cv2
import csv
import random
import numpy as np

# Config
INPUT_DIR  = os.path.join("banknotes", "raw")
OUTPUT_DIR = os.path.join("banknotes", "augmented_extra")
CSV_PATH   = "banknote_labels_extra.csv"
BG_DIR     = "background_images"

def perspective_warp(image, strength=0.15):
    """Simulate the note being held at a slight angle to the camera."""
    h, w = image.shape[:2]
    margin = int(min(h, w) * strength)
    pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    pts2 = pts1 + np.random.uniform(-margin, margin, pts1.shape).astype(np.float32)
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(
        image, M, (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )


def add_shadow(image):
    """Cast a random gradient shadow across part of the image."""
    h, w = image.shape[:2]
    shadow_map = np.ones((h, w), dtype=np.float32)
    x1 = np.random.randint(0, w // 2)
    x2 = np.random.randint(w // 2, w)
    shadow_map[:, :x1] *= np.random.uniform(0.35, 0.65)
    shadow_map[:, x2:] *= np.random.uniform(0.35, 0.65)
    shadow_map = np.stack([shadow_map] * 3, axis=-1)
    return np.clip(image.astype(np.float32) * shadow_map, 0, 255).astype(np.uint8)


def motion_blur(image):
    """Simulate camera shake / a moving hand."""
    size = random.choice([9, 13, 17])
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = np.ones(size) / size
    angle = np.random.uniform(0, 180)
    M = cv2.getRotationMatrix2D((size // 2, size // 2), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (size, size))
    kernel /= kernel.sum() if kernel.sum() != 0 else 1
    return cv2.filter2D(image, -1, kernel)


def paste_on_background(note_img, bg_img):
    """
    Mask out the white background of the flat note and paste it
    onto a random real-world background image.
    """
    h, w = note_img.shape[:2]
    bg = cv2.resize(bg_img, (w, h))
    gray = cv2.cvtColor(note_img, cv2.COLOR_BGR2GRAY)
    _, mask_bg = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    mask_note = cv2.bitwise_not(mask_bg)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_note = cv2.morphologyEx(mask_note, cv2.MORPH_CLOSE, kernel)
    mask_bg = cv2.bitwise_not(mask_note)
    note_part = cv2.bitwise_and(note_img, note_img, mask=mask_note)
    bg_part = cv2.bitwise_and(bg, bg, mask=mask_bg)
    return cv2.add(note_part, bg_part)


def parse_folder_name(folder_name):
    # Expects: R10_old_front -> denomination, era, side
    parts = folder_name.split("_")
    return parts[0], parts[1], parts[2]


# Load background images once 
bg_images = []
if os.path.isdir(BG_DIR):
    for f in os.listdir(BG_DIR):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = cv2.imread(os.path.join(BG_DIR, f))
            if img is not None:
                bg_images.append(img)
print(f"Loaded {len(bg_images)} background images")

# Main loop 
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

        # 2 perspective warps
        save(perspective_warp(image, strength=0.12), "persp1")
        save(perspective_warp(image, strength=0.18), "persp2")

        # 2 shadow overlays
        save(add_shadow(image), "shadow1")
        save(add_shadow(image), "shadow2")

        # 2 motion blurs
        save(motion_blur(image), "mblur1")
        save(motion_blur(image), "mblur2")

        # 2 background composites
        if bg_images:
            save(paste_on_background(image, random.choice(bg_images)), "bg1")
            save(paste_on_background(image, random.choice(bg_images)), "bg2")

        # 2 combined: perspective warp + shadow
        save(add_shadow(perspective_warp(image, strength=0.15)), "persp_shadow1")
        save(add_shadow(perspective_warp(image, strength=0.20)), "persp_shadow2")

        print(f"  Done: {class_folder}/{fname}  (+10 variants)")

# Write new CSV (does NOT touch existing banknote_labels.csv) 
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["filename", "denomination", "era", "side", "class"]
    )
    writer.writeheader()
    writer.writerows(csv_rows)

print(f"\nDone! {total_saved} new images saved to '{OUTPUT_DIR}'")
print(f"New CSV written to '{CSV_PATH}'")
print(f"\nNext steps:")
print(f"  1. Check the images in '{OUTPUT_DIR}' look good")
print(f"  2. Copy the class folders into 'banknotes/augmented/'")
print(f"  3. Append '{CSV_PATH}' rows to 'banknote_labels.csv'")
print(f"\nPer original: 2 persp + 2 shadow + 2 mblur + 2 bg + 2 persp_shadow = 10")
