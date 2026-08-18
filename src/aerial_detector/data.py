"""Data preparation for DIOR-R subset in YOLO HBB format."""
import shutil
import numpy as np

from sklearn.model_selection import train_test_split
from pathlib import Path


#----- CONSTANTS ------
TARGET_CLASSES = {0: 0, 13: 1, 18: 2}  # {orig_id: new_id}
# correspond to : airplane → 0, ship → 1, vehicle → 2
CLASS_NAMES = {0: "airplane", 1: "ship", 2: "vehicle"}

SRC_LABELS = Path("data/YOLODIOR-R/raw/labels")
SRC_IMAGES = Path("data/YOLODIOR-R/raw/images")
DST_ROOT = Path("data/prepared")

#------ HELPERS --------
def obb_to_hbb(points: np.ndarray) -> tuple[float, float, float, float]:
    """Convert an oriented bounding box (4 corner points) to an axis-aligned YOLO HBB.

    Args:
        points: shape (4, 2), normalized coordinates in [0, 1].

    Returns:
        (cx, cy, w, h) in normalized YOLO format.
    """
    xs, ys = points[:, 0], points[:, 1]
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min
    return float(cx), float(cy), float(w), float(h)

def parse_obb_label(label_path: Path) -> list[tuple[int, np.ndarray]]:
    """Parse a DIOR-R OBB label file.

    Args:
        label_path: path to a .txt file with one line per object.

    Returns:
        List of (class_id, points) tuples where points has shape (4, 2).
    """
    objects = []
    lines = label_path.read_text().strip().splitlines()
    for line in lines:
        parts = line.split()
        try:
            objects.append((int(parts[0]), np.array(parts[1:], dtype=float).reshape(4, 2)))
        except ValueError:
            continue
    return objects

def filter_and_remap(objects: list[tuple[int, np.ndarray]],
                     mapping: dict[int, int]) -> list[tuple[int, np.ndarray]]:
    """Keep only objects whose class is in mapping, remap their IDs.

    Args:
        objects: output of parse_obb_label.
        mapping: {original_class_id: new_class_id}.

    Returns:
        Filtered list with remapped class IDs.
    """
    target_objects = [(mapping[cid], points) for cid, points in objects  if cid in mapping]
    return target_objects

#------ PIPELINE ------
def prepare_dataset(seed: int = 42, val_ratio: float = 0.2) -> None:
    """Filter DIOR-R raw data (from Kaggle's test split) to 3 classes, 
    convert OBB→HBB, split train/val.
    """
    
    # 1. Preparing folder structure
    for split in ("train", "val", "all"): 
        (DST_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    # 2. looping on labels, filtering, converting, writing
    kept_stems = []  
    for label_path in SRC_LABELS.glob("*.txt"):
        objects = parse_obb_label(label_path)
        # parse, filter, skip if empty
        objects = filter_and_remap(objects, TARGET_CLASSES) #(new_cid, points)
        if not objects:
            continue
        # writing YOLO HBB label
        lines = []
        for cid, points in objects:
            cx, cy, w, h = obb_to_hbb(points)
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        dst_label = DST_ROOT / "labels" / "all" / f"{label_path.stem}.txt"
        dst_label.write_text("\n".join(lines) + "\n")

        # copy images from source to destination
        src_img = SRC_IMAGES / f"{label_path.stem}.jpg"
        dst_img = DST_ROOT / "images" / "all" / f"{label_path.stem}.jpg"
        shutil.copy(src_img, dst_img)

        kept_stems.append(label_path.stem)
    
    # 3. Split 80/20 and move files
    train_stems, val_stems = train_test_split(kept_stems, test_size=val_ratio, random_state=seed)
    for split_name, stems in [("train", train_stems), ("val", val_stems)]:
        for stem in stems:
            shutil.move(DST_ROOT / "images" / "all" / f"{stem}.jpg",
                        DST_ROOT / "images" / split_name / f"{stem}.jpg")
            shutil.move(DST_ROOT / "labels" / "all" / f"{stem}.txt",
                        DST_ROOT / "labels" / split_name / f"{stem}.txt")

    
    print(f"Prepared {len(kept_stems)} images")
    print(f"Prepared {len(train_stems)} train / {len(val_stems)} val images")


if __name__ == "__main__":
    prepare_dataset()