"""Run best.pt on 10 val images and save annotated results."""

import random
from pathlib import Path

from ultralytics import YOLO

WEIGHTS = Path("runs/detect/baseline/weights/best.pt")
VAL_IMAGES = Path("data/prepared/images/val")
OUTPUT = Path("outputs/qualitative")
OUTPUT.mkdir(parents=True, exist_ok=True)

model = YOLO(str(WEIGHTS))

# choose 10 random images from validation set
random.seed(0)
sample = random.sample(list(VAL_IMAGES.glob("*.jpg")), 10)

# inference: boxes + classes + scores
for img_path in sample:
    results = model.predict(
        source=str(img_path),
        conf=0.25,  # seuil de confiance minimum
        save=True,  # draw predicted boxes
        project=str(OUTPUT.absolute()),
        name="",
        exist_ok=True,
        verbose=False,
    )

print(f"Wrote 10 annotated images to {OUTPUT}")
