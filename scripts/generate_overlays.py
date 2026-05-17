import cv2
import numpy as np
import os
from pathlib import Path

# Set these paths to your actual folders
images_dir = Path("rfdetr_seg_eval/images")
labels_dir = Path("rfdetr_seg_eval/labels")
output_dir = Path("rfdetr_seg_eval/overlays")
output_dir.mkdir(exist_ok=True)

# Pick how many images to generate
MAX_IMAGES = 8

count = 0
for label_file in sorted(labels_dir.glob("*.txt")):
    if count >= MAX_IMAGES:
        break

    # Find matching image (try jpg and png)
    stem = label_file.stem
    img_path = None
    for ext in [".jpg", ".jpeg", ".png"]:
        candidate = images_dir / (stem + ext)
        if candidate.exists():
            img_path = candidate
            break

    if img_path is None:
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        continue

    h, w = img.shape[:2]
    overlay = img.copy()

    with open(label_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            # Skip class index, rest are x y pairs
            coords = list(map(float, parts[1:]))
            points = []
            for i in range(0, len(coords) - 1, 2):
                px = int(coords[i] * w)
                py = int(coords[i + 1] * h)
                points.append([px, py])
            points = np.array(points, dtype=np.int32)

            # Fill polygon with semi-transparent green
            cv2.fillPoly(overlay, [points], color=(0, 200, 80))
            # Draw polygon border
            cv2.polylines(
                overlay, [points], isClosed=True, color=(0, 255, 100), thickness=2
            )

    # Blend overlay with original
    result = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)
    out_path = output_dir / (stem + "_overlay.jpg")
    cv2.imwrite(str(out_path), result)
    print(f"Saved: {out_path}")
    count += 1

print("Done.")
