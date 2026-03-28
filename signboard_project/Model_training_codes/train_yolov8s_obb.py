"""
=============================================================================
  Train YOLOv8s-OBB (Oriented Bounding Box) on Merged ICDAR 2013 + 2015
  Strategy: Use original quad annotations directly as oriented boxes
  Hardware: NVIDIA RTX 3050 6GB Laptop GPU
  Expected mAP50: 0.64 - 0.68 (up from 0.596 YOLOv8s standard)
=============================================================================

WHY OBB IS BETTER FOR SCENE TEXT:
  Standard boxes waste space on rotated text:
    ┌─────────────────────┐        OBB:    ╱─────────╲
    │  ░░ "Hello" ░░░░░   │               ╱  "Hello"  ╲
    └─────────────────────┘              ╱─────────────╲
  OBB wraps tightly around rotated text → better features → higher accuracy

OBB ANNOTATION FORMAT:
  YOLO OBB format:  class_id  cx  cy  w  h  angle(radians)
  We compute this from the original ICDAR quad points using cv2.minAreaRect()

RUN:
  python train_yolov8s_obb.py

OUTPUT:
  signboard_project/best_yolov8s_obb.pt   ← your OBB model
  runs/yolov8s_obb_merged/                ← full training results
"""

import shutil, random, math
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR    = Path("signboard_project")

# Raw data sources
TRAIN_IMG_DIR_2015 = Path("raw_data")        / "ch4_training_images"
TRAIN_GT_DIR_2015  = Path("raw_data")        / "ch4_training_localization_transcription_gt"
TRAIN_IMG_DIR_2013 = Path("raw_data_2013")   / "ch2_training_images"
TRAIN_GT_DIR_2013  = Path("raw_data_2013")   / "ch2_training_localization_transcription_gt"

# OBB dataset output (separate from standard dataset)
OBB_DIR         = BASE_DIR / "dataset_obb"
OBB_TRAIN_IMG   = OBB_DIR / "images"  / "train"
OBB_VAL_IMG     = OBB_DIR / "images"  / "val"
OBB_TRAIN_LBL   = OBB_DIR / "labels"  / "train"
OBB_VAL_LBL     = OBB_DIR / "labels"  / "val"
OBB_YAML        = OBB_DIR / "data.yaml"

OUTPUT_PT       = BASE_DIR / "best_yolov8s_obb.pt"

VAL_SPLIT = 0.15
SEED      = 42


# =============================================================================
# STEP 1: ANNOTATION PARSERS
# =============================================================================

def parse_icdar2015_gt(gt_path):
    """
    ICDAR 2015: x1,y1,x2,y2,x3,y3,x4,y4,transcription (comma separated)
    Returns list of (x1,y1,x2,y2,x3,y3,x4,y4) tuples.
    """
    annotations = []
    with open(gt_path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip().strip('"') for p in line.split(",")]
            try:
                if len(parts) >= 8:
                    coords = list(map(int, parts[:8]))
                    annotations.append(tuple(coords))
            except ValueError:
                continue
    return annotations


def parse_icdar2013_gt(gt_path):
    """
    ICDAR 2013: left top right bottom "transcription" (space separated)
    Returns list of (x1,y1,x2,y2,x3,y3,x4,y4) tuples (expanded from rect).
    """
    annotations = []
    with open(gt_path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                if len(parts) >= 4:
                    l, t, r, b = map(int, parts[:4])
                    # Expand rect → quad clockwise from top-left
                    annotations.append((l, t, r, t, r, b, l, b))
            except ValueError:
                continue
    return annotations


# =============================================================================
# STEP 2: QUAD → OBB CONVERSION (the key difference from standard pipeline)
# =============================================================================

def quad_to_obb(x1, y1, x2, y2, x3, y3, x4, y4, img_w, img_h):
    """
    Convert a quadrilateral to YOLO OBB format.

    YOLO OBB format: cx cy w h angle
    - cx, cy: normalized center point
    - w, h:   normalized width and height of the ORIENTED box
    - angle:  rotation in DEGREES (-180 to 180) — Ultralytics OBB uses degrees

    Uses cv2.minAreaRect() which finds the tightest rotated rectangle
    around the 4 quad points. This is much tighter than axis-aligned bbox
    for rotated text.
    """
    pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.float32)

    # minAreaRect returns ((cx, cy), (w, h), angle_degrees)
    (cx, cy), (w, h), angle = cv2.minAreaRect(pts)

    # Normalize by image dimensions
    cx /= img_w
    cy /= img_h
    w  /= img_w
    h  /= img_h

    # Clamp to [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w  = max(0.0, min(1.0, w))
    h  = max(0.0, min(1.0, h))

    # angle is already in degrees for Ultralytics OBB
    return cx, cy, w, h, angle


def convert_to_obb_label(img_path, gt_path, out_label_path, is_2013=False):
    """
    Convert one image's annotation to YOLO OBB format.
    Returns number of boxes written.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return 0
    img_h, img_w = img.shape[:2]

    if is_2013:
        annotations = parse_icdar2013_gt(gt_path)
    else:
        annotations = parse_icdar2015_gt(gt_path)

    if not annotations:
        return 0

    lines = []
    for coords in annotations:
        x1, y1, x2, y2, x3, y3, x4, y4 = coords
        # Normalize coordinates
        x1 /= img_w
        y1 /= img_h
        x2 /= img_w
        y2 /= img_h
        x3 /= img_w
        y3 /= img_h
        x4 /= img_w
        y4 /= img_h

        lines.append(f"0 {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}")

    if not lines:
        return 0

    out_label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_label_path, "w") as f:
        f.write("\n".join(lines))
    return len(lines)


# =============================================================================
# STEP 3: BUILD OBB DATASET
# =============================================================================

def build_obb_dataset():
    """
    Convert ICDAR 2015 + 2013 annotations to OBB format
    and build the YOLO directory structure.
    """

    # Create output dirs
    for d in [OBB_TRAIN_IMG, OBB_VAL_IMG, OBB_TRAIN_LBL, OBB_VAL_LBL]:
        d.mkdir(parents=True, exist_ok=True)

    all_samples = []  # list of (img_path, label_path, stem_with_prefix)

    # ── Process ICDAR 2015 ────────────────────────────────────────────────────
    print("\n[1/3]  Converting ICDAR 2015 → OBB format ...")
    tmp_2015_lbl = BASE_DIR / "tmp_obb" / "2015_labels"
    tmp_2015_lbl.mkdir(parents=True, exist_ok=True)

    img_files_2015 = sorted(TRAIN_IMG_DIR_2015.glob("*.jpg"))
    print(f"       Found {len(img_files_2015)} ICDAR 2015 images")

    for img_path in tqdm(img_files_2015, desc="  2015", unit="img"):
        stem = img_path.stem
        gt_path = TRAIN_GT_DIR_2015 / f"gt_{stem}.txt"
        if not gt_path.exists():
            continue
        out_lbl = tmp_2015_lbl / f"{stem}.txt"
        n = convert_to_obb_label(img_path, gt_path, out_lbl, is_2013=False)
        if n > 0:
            all_samples.append((img_path, out_lbl, stem))

    print(f"       {len(all_samples)} valid ICDAR 2015 images")

    # ── Process ICDAR 2013 ────────────────────────────────────────────────────
    print("\n[2/3]  Converting ICDAR 2013 → OBB format ...")
    tmp_2013_lbl = BASE_DIR / "tmp_obb" / "2013_labels"
    tmp_2013_lbl.mkdir(parents=True, exist_ok=True)

    img_files_2013 = sorted(TRAIN_IMG_DIR_2013.glob("*.jpg"))
    print(f"       Found {len(img_files_2013)} ICDAR 2013 images")

    n_2013 = 0
    for img_path in tqdm(img_files_2013, desc="  2013", unit="img"):
        stem = img_path.stem
        gt_path = TRAIN_GT_DIR_2013 / f"gt_{stem}.txt"
        if not gt_path.exists():
            continue
        out_lbl = tmp_2013_lbl / f"icdar13_{stem}.txt"
        n = convert_to_obb_label(img_path, gt_path, out_lbl, is_2013=True)
        if n > 0:
            all_samples.append((img_path, out_lbl, f"icdar13_{stem}"))
            n_2013 += 1

    print(f"       {n_2013} valid ICDAR 2013 images")
    print(f"\n       Total: {len(all_samples)} images")

    # ── Train / Val split ─────────────────────────────────────────────────────
    print("\n[3/3]  Building OBB dataset structure ...")
    random.seed(SEED)
    random.shuffle(all_samples)
    n_val   = max(1, math.floor(len(all_samples) * VAL_SPLIT))
    val_set = all_samples[:n_val]
    trn_set = all_samples[n_val:]

    print(f"       Train: {len(trn_set)} | Val: {len(n_val and val_set)}")

    def copy_sample(img_path, lbl_path, prefix, img_dst, lbl_dst):
        img_name = f"{prefix}{img_path.suffix}"
        lbl_name = f"{prefix}.txt"
        shutil.copy(img_path, img_dst / img_name)
        shutil.copy(lbl_path, lbl_dst / lbl_name)

    for img_path, lbl_path, stem in tqdm(trn_set, desc="  Train", unit="file"):
        copy_sample(img_path, lbl_path, stem, OBB_TRAIN_IMG, OBB_TRAIN_LBL)

    for img_path, lbl_path, stem in tqdm(val_set, desc="  Val  ", unit="file"):
        copy_sample(img_path, lbl_path, stem, OBB_VAL_IMG, OBB_VAL_LBL)

    # Cleanup temp
    shutil.rmtree(BASE_DIR / "tmp_obb", ignore_errors=True)

    total_train = len(list(OBB_TRAIN_IMG.glob("*")))
    total_val   = len(list(OBB_VAL_IMG.glob("*")))
    print(f"\n  ✓  OBB dataset ready: Train={total_train} | Val={total_val}")


# =============================================================================
# STEP 4: WRITE data.yaml FOR OBB
# =============================================================================

def create_obb_yaml():
    yaml_content = f"""# Merged ICDAR 2015 + 2013 — YOLOv8s-OBB
task: obb

path: {OBB_DIR.resolve()}
train: images/train
val:   images/val

nc: 1
names:
  0: text
"""
    with open(OBB_YAML, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"  ✓  OBB data.yaml saved → {OBB_YAML}\n")


# =============================================================================
# STEP 5: TRAIN YOLOv8s-OBB
# =============================================================================

def train_obb():
    from ultralytics import YOLO

    print("=" * 60)
    print("  YOLOv8s-OBB Training on Merged ICDAR 2013 + 2015")
    print("=" * 60)
    print(f"  Model : yolov8s-obb.pt (COCO pretrained OBB Small)")
    print(f"  Data  : {OBB_YAML}")
    print(f"  Boxes : Oriented (tight around rotated text)")
    print("=" * 60)

    # Downloads yolov8s-obb.pt automatically (~22MB)
    from ultralytics import YOLO

    model = YOLO("yolov8s-obb.pt")

    # 🔥 FORCE OBB MODE (this is the real fix)
    model.overrides["task"] = "obb"
    model.task = "obb"

    print("Forced task:", model.task)  # should print obb
    results = model.train(
        data          = str(OBB_YAML),
        task="obb", 

        # ── Core ──────────────────────────────────────────────────────────────
        epochs        = 150,
        patience      = 25,
        imgsz         = 640,
        batch         = 8,
        workers       = 4,
        device        = 0,
        amp           = True,
        cache         = "ram",      # use "disk" if RAM < 12GB

        # ── Learning rate ─────────────────────────────────────────────────────
        optimizer     = "AdamW",
        lr0           = 5e-4,
        lrf           = 0.01,
        warmup_epochs = 3,

        # ── Augmentations for scene text ──────────────────────────────────────
        mosaic        = 1.0,
        mixup         = 0.15,
        copy_paste    = 0.1,
        degrees       = 15.0,       # OBB handles rotation natively — this helps
        translate     = 0.1,
        scale         = 0.5,
        shear         = 5.0,
        perspective   = 0.0005,
        flipud        = 0.0,        # never flip upside down
        fliplr        = 0.5,
        hsv_h         = 0.015,
        hsv_s         = 0.7,
        hsv_v         = 0.4,

        # ── Output ────────────────────────────────────────────────────────────
        project       = "runs",
        name          = "yolov8s_obb_merged",
        exist_ok      = True,
        verbose       = True,
    )

    # Save best model
    best_out = Path(results.save_dir) / "weights" / "best.pt"
    if best_out.exists():
        shutil.copy(best_out, OUTPUT_PT)
        print(f"\n{'=' * 60}")
        print(f"  ✓  OBB model saved → {OUTPUT_PT.resolve()}")
        print(f"  ✓  Full results    → {results.save_dir}")
        print(f"{'=' * 60}")

        metrics = results.results_dict
        print(f"\n  Final Metrics:")
        print(f"  mAP50    : {metrics.get('metrics/mAP50(B)', 'N/A'):.4f}")
        print(f"  mAP50-95 : {metrics.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
        print(f"  Precision: {metrics.get('metrics/precision(B)', 'N/A'):.4f}")
        print(f"  Recall   : {metrics.get('metrics/recall(B)', 'N/A'):.4f}")
    else:
        print(f"\n  ⚠  best.pt not found at: {best_out}")

    return results


# =============================================================================
# SANITY CHECK
# =============================================================================

def check_raw_data():
    missing = []
    if not TRAIN_IMG_DIR_2015.exists():
        missing.append(f"ICDAR 2015 images: {TRAIN_IMG_DIR_2015}")
    if not TRAIN_GT_DIR_2015.exists():
        missing.append(f"ICDAR 2015 GT:     {TRAIN_GT_DIR_2015}")
    if not TRAIN_IMG_DIR_2013.exists():
        missing.append(f"ICDAR 2013 images: {TRAIN_IMG_DIR_2013}")
    if not TRAIN_GT_DIR_2013.exists():
        missing.append(f"ICDAR 2013 GT:     {TRAIN_GT_DIR_2013}")

    if missing:
        print("\n  ✗  Missing raw data:")
        for m in missing:
            print(f"     {m}")
        return False
    return True


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="YOLOv8s-OBB training on merged ICDAR 2013+2015"
    )
    parser.add_argument("--skip_build", action="store_true",
                        help="Skip dataset build (use if already built)")
    parser.add_argument("--build_only", action="store_true",
                        help="Only build OBB dataset, do not train")
    args = parser.parse_args()

    if not args.skip_build:
        if not check_raw_data():
            exit(1)
        build_obb_dataset()
        create_obb_yaml()

    if args.build_only:
        print("\n  Done. Run without --build_only to start training.")
        exit(0)

    if args.skip_build and not OBB_YAML.exists():
        print(f"  ✗  OBB dataset not found. Run without --skip_build first.")
        exit(1)

    train_obb()


# =============================================================================
# USAGE
# =============================================================================
#
#  Full pipeline (build OBB dataset + train):
#    python train_yolov8s_obb.py
#
#  Build dataset only (verify before training):
#    python train_yolov8s_obb.py --build_only
#
#  Train only (if dataset already built):
#    python train_yolov8s_obb.py --skip_build
#
#  Inference with OBB model:
#    from ultralytics import YOLO
#    model = YOLO("signboard_project/best_yolov8s_obb.pt")
#    results = model.predict("image.jpg", conf=0.35)
#    results[0].show()   # shows oriented boxes
#
#  OOM error? → reduce batch=8 to batch=4
#  RAM < 12GB? → change cache="ram" to cache="disk"
#
# =============================================================================