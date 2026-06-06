"""
banknote_bg_augment.py - Background composite augmentation ONLY for SA banknotes

Places the note as a smaller object on top of a real background image,
like it's sitting on a table. No masking needed.

Writes to:  banknotes/augmented_bg/
CSV:        banknote_labels_bg.csv

Run from Currency_Classifier/data/:
    python banknote_bg_augment.py

Generates 2 bg variants per original raw image.

Folder structure expected:
    Currency_Classifier/data/
        background_images/          <- your background photos
        banknotes/
            raw/                    <- original flat note images (read from)
            augmented_bg/           <- new images written here (created automatically)
        banknote_labels_bg.csv      <- new CSV written here
"""

import os
import cv2
import csv
import random
import numpy as np

INPUT_DIR  = os.path.join("banknotes", "raw")
OUTPUT_DIR = os.path.join("banknotes", "augmented_bg")
CSV_PATH   = "banknote_labels_bg.csv"
BG_DIR     = "background_images"

# How large the note appears relative to the background canvas (0.0 to 1.0)
NOTE_SCALE_MIN = 0.55
NOTE_SCALE_MAX = 0.75


def place_on_background(note_img, bg_img):
    """
    Resize the note to ~60-75% of the background canvas and paste it at a
    random position. No pixel masking - the note is placed as a whole object.
    Optionally applies a slight random rotation to make it look naturally placed.
    """
    bg_h, bg_w = bg_img.shape[:2]
    note_h, note_w = note_img.shape[:2]

    # Pick a random scale so the note fits within the canvas
    scale = random.uniform(NOTE_SCALE_MIN, NOTE_SCALE_MAX)
    new_w = int(bg_w * scale)
    new_h = int(new_w * (note_h / note_w))  # preserve aspect ratio

    # Safety: don't let the note be bigger than the background
    if new_h >= bg_h:
        new_h = bg_h - 2
        new_w = int(new_h * (note_w / note_h))

    note_resized = cv2.resize(note_img, (new_w, new_h))

    # Random slight rotation (-15 to +15 degrees) so it looks casually placed
    angle = random.uniform(-15, 15)
    cx, cy = new_w // 2, new_h // 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    note_rotated = cv2.warpAffine(
        note_resized, M, (new_w, new_h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )

    # Random placement so the note stays fully inside the background
    max_x = bg_w - new_w
    max_y = bg_h - new_h
    x_off = random.randint(0, max(0, max_x))
    y_off = random.randint(0, max(0, max_y))

    canvas = bg_img.copy()
    roi = canvas[y_off:y_off + new_h, x_off:x_off + new_w]

    # Blend: replace white pixels in the note with the background underneath
    gray = cv2.cvtColor(note_rotated, cv2.COLOR_BGR2GRAY)
    _, white_mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    note_mask = cv2.bitwise_not(white_mask)

    roi_bg = cv2.bitwise_and(roi, roi, mask=white_mask)
    note_fg = cv2.bitwise_and(note_rotated, note_rotated, mask=note_mask)
    blended = cv2.add(roi_bg, note_fg)
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = blended

    return canvas


def parse_folder_name(folder_name):
    # Expects: R10_new_front -> denomination, era, side
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

if not bg_images:
    print("ERROR: No background images found in background_images/. Exiting.")
    exit(1)

print(f"Loaded {len(bg_images)} background images")

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

        save(place_on_background(image, random.choice(bg_images)), "bg1")
        save(place_on_background(image, random.choice(bg_images)), "bg2")

        print(f"  Done: {class_folder}/{fname}  (+2 bg variants)")

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["filename", "denomination", "era", "side", "class"]
    )
    writer.writeheader()
    writer.writerows(csv_rows)

print(f"\nDone! {total_saved} new images saved to '{OUTPUT_DIR}'")
print(f"New CSV written to '{CSV_PATH}'")
print(f"\nNext steps:")
print(f"  1. Check images in '{OUTPUT_DIR}' look good")
print(f"  2. Upload augmented_bg/ folder to Drive")
print(f"  3. Send banknote_labels_bg.csv back so labels can be verified/merged")
