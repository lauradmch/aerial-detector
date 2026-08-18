"""Draw YOLO HBB boxes on a few prepared images to verify the conversion."""

import random
from pathlib import Path

import cv2

IMAGES = Path("data/prepared/images/train")
LABELS = Path("data/prepared/labels/train")
OUTPUT = Path("outputs/sanity_check")
OUTPUT.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {0: "airplane", 1: "ship", 2: "vehicle"}
COLORS = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 0, 255)}  # BGR

random.seed(0)
sample = random.sample(list(IMAGES.glob("*.jpg")), 5)

for img_path in sample:
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    label_path = LABELS / f"{img_path.stem}.txt"
    for line in label_path.read_text().strip().splitlines():
        cid, cx, cy, bw, bh = line.split()
        cid = int(cid)
        cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
        x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
        x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), COLORS[cid], 2)
        cv2.putText(
            img, CLASS_NAMES[cid], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[cid], 2
        )
    cv2.imwrite(str(OUTPUT / img_path.name), img)

print(f"Wrote {len(sample)} images to {OUTPUT}")
