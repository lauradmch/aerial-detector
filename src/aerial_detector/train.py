"""Train YOLOv8n on the DIOR-R subset with MLflow tracking."""

import os
from pathlib import Path

import mlflow
from ultralytics import YOLO

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

CONFIG_PATH = Path("configs/dataset.yaml")

# Hyperparameters
HYPERPARAMS = {
    "model": "yolov8n.pt",
    "data": str(CONFIG_PATH),
    "epochs": 20,
    "imgsz": 640,
    "batch": 16,
    "lr0": 0.01,
    "patience": 5,
    "project": str(Path("runs/detect").absolute()),
    "name": "baseline",
}


def train() -> None:
    """Train YOLOv8n and log everything to MLflow."""
    mlflow.set_experiment("aerial-detector")

    with mlflow.start_run():
        mlflow.log_params(HYPERPARAMS)

        model = YOLO(HYPERPARAMS["model"])
        results = model.train(**{k: v for k, v in HYPERPARAMS.items() if k != "model"})

        # logging final metrics
        # results.box.map50 → mAP@0.5 global
        # results.box.map → mAP@0.5:0.95 global (stricter)
        mlflow.log_metric("mAP50", results.box.map50)
        mlflow.log_metric("mAP50_95", results.box.map)

        # logging best checkpoint as artifact
        best = Path("runs/detect/baseline/weights/best.pt")
        if best.exists():
            mlflow.log_artifact(str(best))

        print(f"mAP@0.5 = {results.box.map50:.4f}")


if __name__ == "__main__":
    train()
