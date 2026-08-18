"""Draw ground-truth boxes (green) and predictions (red) side by side."""

import random
from pathlib import Path

import cv2
from ultralytics import YOLO

WEIGHTS = Path("runs/detect/baseline/weights/best.pt")
VAL_IMAGES = Path("data/prepared/images/val")
VAL_LABELS = Path("data/prepared/labels/val")
OUTPUT = Path("outputs/gt_vs_pred")
OUTPUT.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {0: "airplane", 1: "ship", 2: "vehicle"}


def draw_yolo_labels(img, label_path, color, tag):
    """Draw YOLO HBB labels on the image with a color tag."""
    if not label_path.exists():
        return
    h, w = img.shape[:2]
    for line in label_path.read_text().strip().splitlines():
        cid, cx, cy, bw, bh = line.split()
        cid = int(cid)
        cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
        x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
        x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img, f"{tag} {CLASS_NAMES[cid]}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )


def visualize(n: int = 10) -> None:
    """Overlay GT (green) and predictions (red) on n random val images."""
    model = YOLO(str(WEIGHTS))
    random.seed(0)
    sample = random.sample(list(VAL_IMAGES.glob("*.jpg")), n)

    for img_path in sample:
        img = cv2.imread(str(img_path))

        # GT en vert (BGR)
        draw_yolo_labels(img, VAL_LABELS / f"{img_path.stem}.txt", (0, 255, 0), "GT")

        # Prédictions en rouge
        results = model.predict(source=str(img_path), conf=0.25, verbose=False)
        for box, cls in zip(
            results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.cls.cpu().numpy(), strict=True
        ):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                img,
                f"P {CLASS_NAMES[int(cls)]}",
                (x1, y2 + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

        cv2.imwrite(str(OUTPUT / img_path.name), img)

    print(f"Wrote {n} images to {OUTPUT}")


if __name__ == "__main__":
    visualize()
