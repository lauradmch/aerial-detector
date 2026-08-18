"""Evaluate the trained model on the val split and dump per-class metrics."""

import json
from pathlib import Path

from ultralytics import YOLO

WEIGHTS = Path("runs/detect/baseline/weights/best.pt")
DATA_YAML = Path("configs/dataset.yaml")
OUTPUT = Path("outputs/per_class_metrics.json")


def evaluate() -> None:
    """Run val on best checkpoint, save per-class metrics as JSON."""
    model = YOLO(str(WEIGHTS))
    metrics = model.val(data=str(DATA_YAML), verbose=False)

    names = list(metrics.names.values())
    payload = {
        "map50_global": float(metrics.box.map50),
        "map50_95_global": float(metrics.box.map),
        "per_class": {
            "mAP50": dict(zip(names, metrics.box.maps.tolist(), strict=True)),
            "precision": dict(zip(names, metrics.box.p.tolist(), strict=True)),
            "recall": dict(zip(names, metrics.box.r.tolist(), strict=True)),
        },
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2))
    print(f"mAP@0.5 = {payload['map50_global']:.4f}")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    evaluate()
