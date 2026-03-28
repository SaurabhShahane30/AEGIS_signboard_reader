"""
=============================================================================
  Fine-tune YOLOv8 Nano (best.pt) + Merge ICDAR 2013 Data
  Strategy: Lower LR fine-tune + expanded dataset
  Hardware: NVIDIA RTX 3050 (6 GB VRAM)
=============================================================================

BEFORE RUNNING:
  1. Download ICDAR 2013 from: https://rrc.cvc.uab.es/?ch=2&com=downloads
       - ch2_training_images.zip          (~17 MB)
       - ch2_training_localization_transcription_gt.zip  (~1 MB)
  2. Unzip into:
       raw_data_2013/
           ch2_training_images/           ← 229 training images
           ch2_training_localization_transcription_gt/  ← GT txt files

  3. Your existing ICDAR 2015 dataset should already be at:
       signboard_project/dataset/images/train/
       signboard_project/dataset/images/val/
       signboard_project/dataset/labels/train/
       signboard_project/dataset/labels/val/

  4. Your best.pt should be at:
       signboard_project/best.pt

THEN RUN:
  python finetune_with_icdar2013.py

OUTPUT:
  runs/finetune_nano_v2/weights/best.pt   ← your improved model
"""

import os, shutil, random, math
from pathlib import Path

import cv2
from tqdm import tqdm

# =============================================================================
# PATHS — edit these if your folders are named differently
# =============================================================================

BASE_DIR       = Path("signboard_project")
BEST_PT        = BASE_DIR / "best.pt"             # your existing trained model

# ICDAR 2015 — already converted, just reference existing dataset
EXISTING_TRAIN_IMG = BASE_DIR / "dataset" / "images" / "train"
EXISTING_TRAIN_LBL = BASE_DIR / "dataset" / "labels" / "train"
EXISTING_VAL_IMG   = BASE_DIR / "dataset" / "images" / "val"
EXISTING_VAL_LBL   = BASE_DIR / "dataset" / "labels" / "val"

# ICDAR 2013 — raw download location
RAW_2013_DIR       = Path("raw_data_2013")
ICDAR2013_IMG_DIR  = RAW_2013_DIR / "ch2_training_images"
ICDAR2013_GT_DIR   = RAW_2013_DIR / "ch2_training_localization_transcription_gt"

# Merged dataset output
MERGED_DIR         = BASE_DIR / "dataset_merged"
MERGED_TRAIN_IMG   = MERGED_DIR / "images" / "train"
MERGED_TRAIN_LBL   = MERGED_DIR / "labels" / "train"
MERGED_VAL_IMG     = MERGED_DIR / "images" / "val"
MERGED_VAL_LBL     = MERGED_DIR / "labels" / "val"
MERGED_YAML        = MERGED_DIR / "data.yaml"

VAL_SPLIT_2013 = 0.15   # 15% of ICDAR 2013 goes to val
SEED           = 42


# =============================================================================
# STEP 1: ANNOTATION CONVERTER (same logic as your pipeline, works for both
#         ICDAR 2013 and 2015 — same quad format)
# =============================================================================

def quad_to_yolo(x1, y1, x2, y2, x3, y3, x4, y4, img_w, img_h):
    """Convert ICDAR quadrilateral → YOLO normalised bbox."""
    xs = [x1, x2, x3, x4]
    ys = [y1, y2, y3, y4]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    cx = (xmin + xmax) / 2.0 / img_w
    cy = (ymin + ymax) / 2.0 / img_h
    bw = (xmax - xmin) / img_w
    bh = (ymax - ymin) / img_h

    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    bw = max(0.0, min(1.0, bw))
    bh = max(0.0, min(1.0, bh))
    return cx, cy, bw, bh


def parse_icdar_gt(gt_path):
    """
    Handles BOTH formats automatically:
    - ICDAR 2015: x1,y1,x2,y2,x3,y3,x4,y4,transcription  (comma separated, 8 coords)
    - ICDAR 2013: left top right bottom "transcription"     (space separated, 4 coords)
    Always returns 8-coord tuples for quad_to_yolo compatibility.
    """
    annotations = []
    with open(gt_path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Detect format by checking if commas exist
            if "," in line:
                # ICDAR 2015 — comma separated 8 coords
                parts = [p.strip().strip('"') for p in line.split(",")]
                try:
                    if len(parts) >= 8:
                        coords = list(map(int, parts[:8]))
                        annotations.append(tuple(coords))
                except ValueError:
                    continue
            else:
                # ICDAR 2013 — space separated 4 coords
                parts = line.split()
                try:
                    if len(parts) >= 4:
                        l, t, r, b = map(int, parts[:4])
                        # Expand rect to quad (clockwise from top-left)
                        annotations.append((l, t, r, t, r, b, l, b))
                except ValueError:
                    continue
    return annotations


def convert_single_image(img_path, gt_path, out_label_path):
    """Convert one image's annotation. Returns number of boxes written."""
    img = cv2.imread(str(img_path))
    if img is None:
        return 0
    h, w = img.shape[:2]

    annotations = parse_icdar_gt(gt_path)
    if not annotations:
        return 0

    lines = []
    for coords in annotations:
        x1, y1, x2, y2, x3, y3, x4, y4 = coords
        cx, cy, bw, bh = quad_to_yolo(x1, y1, x2, y2, x3, y3, x4, y4, w, h)
        if bw < 1e-4 or bh < 1e-4:
            continue
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    if not lines:
        return 0

    out_label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_label_path, "w") as f:
        f.write("\n".join(lines))
    return len(lines)


# =============================================================================
# STEP 2: CONVERT + MERGE ICDAR 2013 INTO EXISTING DATASET
# =============================================================================

def check_icdar2013_available():
    """Check if ICDAR 2013 raw data exists."""
    if not ICDAR2013_IMG_DIR.exists():
        print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  ICDAR 2013 not found at: {ICDAR2013_IMG_DIR}
║                                                                      ║
║  Download instructions:                                              ║
║  1. Go to: https://rrc.cvc.uab.es/?ch=2&com=downloads               ║
║  2. Register (free) and download:                                    ║
║     • ch2_training_images.zip          (~17 MB)                      ║
║     • ch2_training_localization_                                     ║
║       transcription_gt.zip             (~1 MB)                       ║
║  3. Unzip both into:  raw_data_2013/                                 ║
║  4. Re-run this script                                               ║
╚══════════════════════════════════════════════════════════════════════╝
        """)
        return False
    return True


def build_merged_dataset():
    """
    1. Copy existing ICDAR 2015 train/val → merged dataset
    2. Convert ICDAR 2013, split 85/15, add to merged train/val
    3. Write new data.yaml pointing to merged dataset
    """

    # ── Create merged directory structure ─────────────────────────────────────
    for d in [MERGED_TRAIN_IMG, MERGED_TRAIN_LBL, MERGED_VAL_IMG, MERGED_VAL_LBL]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Copy existing ICDAR 2015 data ─────────────────────────────────────────
    print("\n[1/3]  Copying existing ICDAR 2015 data → merged dataset ...")

    def copy_dir(src, dst, desc):
        files = list(src.glob("*"))
        for f in tqdm(files, desc=desc, unit="file"):
            shutil.copy(f, dst / f.name)
        return len(files)

    n_train_img = copy_dir(EXISTING_TRAIN_IMG, MERGED_TRAIN_IMG, "  Train images")
    n_train_lbl = copy_dir(EXISTING_TRAIN_LBL, MERGED_TRAIN_LBL, "  Train labels")
    n_val_img   = copy_dir(EXISTING_VAL_IMG,   MERGED_VAL_IMG,   "  Val images  ")
    n_val_lbl   = copy_dir(EXISTING_VAL_LBL,   MERGED_VAL_LBL,   "  Val labels  ")

    print(f"       2015 → Train: {n_train_img} images | Val: {n_val_img} images")

    # ── Convert ICDAR 2013 ────────────────────────────────────────────────────
    print("\n[2/3]  Converting ICDAR 2013 annotations → YOLO format ...")

    img_files_2013 = sorted(ICDAR2013_IMG_DIR.glob("*.jpg"))
    if not img_files_2013:
        # also try .png
        img_files_2013 = sorted(ICDAR2013_IMG_DIR.glob("*.png"))

    print(f"       Found {len(img_files_2013)} ICDAR 2013 images")

    tmp_dir = BASE_DIR / "tmp_2013"
    tmp_img = tmp_dir / "images"
    tmp_lbl = tmp_dir / "labels"
    tmp_img.mkdir(parents=True, exist_ok=True)
    tmp_lbl.mkdir(parents=True, exist_ok=True)

    valid_stems = []
    for img_path in tqdm(img_files_2013, desc="  Converting", unit="img"):
        stem = img_path.stem

        # ICDAR 2013 GT filename format: gt_img_1.txt (matches img_1.jpg)
        gt_filename = "gt_" + stem + ".txt"
        gt_path = ICDAR2013_GT_DIR / gt_filename

        if not gt_path.exists():
            continue

        out_lbl = tmp_lbl / (f"icdar13_{stem}.txt")
        n = convert_single_image(img_path, gt_path, out_lbl)
        if n > 0:
            # Prefix with icdar13_ to avoid filename collision with 2015
            new_img_name = f"icdar13_{img_path.name}"
            shutil.copy(img_path, tmp_img / new_img_name)
            valid_stems.append(stem)

    print(f"       {len(valid_stems)} ICDAR 2013 images with valid annotations")

    # ── Split ICDAR 2013 into train/val ───────────────────────────────────────
    print("\n[3/3]  Merging ICDAR 2013 into merged dataset ...")

    random.seed(SEED)
    random.shuffle(valid_stems)
    n_val   = max(1, math.floor(len(valid_stems) * VAL_SPLIT_2013))
    val_set = set(valid_stems[:n_val])
    trn_set = set(valid_stems[n_val:])

    print(f"       ICDAR 2013 → Train: {len(trn_set)} | Val: {len(val_set)}")

    for stem in tqdm(valid_stems, desc="  Copying", unit="file"):
        img_name = f"icdar13_{stem}.jpg"
        lbl_name = f"icdar13_{stem}.txt"

        img_src = tmp_img / img_name
        lbl_src = tmp_lbl / lbl_name

        # fallback to .png if .jpg doesn't exist
        if not img_src.exists():
            img_name = f"icdar13_{stem}.png"
            img_src  = tmp_img / img_name

        if stem in trn_set:
            if img_src.exists(): shutil.copy(img_src, MERGED_TRAIN_IMG / img_name)
            if lbl_src.exists(): shutil.copy(lbl_src, MERGED_TRAIN_LBL / lbl_name)
        else:
            if img_src.exists(): shutil.copy(img_src, MERGED_VAL_IMG   / img_name)
            if lbl_src.exists(): shutil.copy(lbl_src, MERGED_VAL_LBL   / lbl_name)

    # Cleanup temp
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── Final counts ──────────────────────────────────────────────────────────
    total_train = len(list(MERGED_TRAIN_IMG.glob("*")))
    total_val   = len(list(MERGED_VAL_IMG.glob("*")))
    print(f"\n  ✓  Merged dataset ready:")
    print(f"     Train: {total_train} images  (was {n_train_img})")
    print(f"     Val:   {total_val} images  (was {n_val_img})")


# =============================================================================
# STEP 3: WRITE MERGED data.yaml
# =============================================================================

def create_merged_yaml():
    yaml_content = f"""# Merged ICDAR 2015 + ICDAR 2013 — YOLOv8 Single class: "text"
path: {MERGED_DIR.resolve()}
train: images/train
val:   images/val

nc: 1
names:
  0: text
"""
    with open(MERGED_YAML, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"\n  ✓  data.yaml saved → {MERGED_YAML}\n")


# =============================================================================
# STEP 4: FINE-TUNE best.pt ON MERGED DATASET
# =============================================================================

def finetune():
    from ultralytics import YOLO

    if not BEST_PT.exists():
        raise FileNotFoundError(
            f"best.pt not found at {BEST_PT}\n"
            "Make sure your trained model is at signboard_project/best.pt"
        )

    print("=" * 60)
    print("  Fine-tuning YOLOv8 Nano best.pt on merged dataset")
    print("=" * 60)
    print(f"  Model   : {BEST_PT}")
    print(f"  Data    : {MERGED_YAML}")
    print(f"  Strategy: Lower LR + freeze backbone + more data")
    print("=" * 60)

    model = YOLO(str(BEST_PT))

    results = model.train(
        data          = str(MERGED_YAML),

        # ── Core settings ─────────────────────────────────────────────────────
        epochs        = 80,
        patience      = 20,
        imgsz         = 640,
        batch         = 16,
        workers       = 4,
        device        = 0,
        amp           = True,
        cache         = "ram",       # set "disk" if RAM < 12 GB

        # ── Fine-tune LR (KEY CHANGE from original training) ──────────────────
        # Original was lr0=1e-3. We use 10x lower to avoid overwriting
        # the good weights already learned in best.pt
        optimizer     = "AdamW",
        lr0           = 1e-4,        # ← 10x lower than original
        lrf           = 0.01,        # cosine decay to 1e-6
        warmup_epochs = 2,           # gentle warmup before full training

        # ── Freeze backbone (fine-tuning spirit) ──────────────────────────────
        # Freezes first 10 layers of the CSPDarknet backbone.
        # The neck and detection head train freely.
        # Remove this line if you want full model training (slower but can be better).
        freeze        = 10,

        # ── Augmentations tuned for scene text ────────────────────────────────
        mosaic        = 1.0,         # combine 4 images — great for small text
        mixup         = 0.1,         # blend images for variety
        copy_paste    = 0.05,        # paste text boxes across images
        degrees       = 10.0,        # ±10° rotation — handles tilted signs
        translate     = 0.1,
        scale         = 0.5,
        shear         = 3.0,
        perspective   = 0.0003,
        flipud        = 0.0,         # never flip upside down (text is directional)
        fliplr        = 0.5,
        hsv_h         = 0.015,
        hsv_s         = 0.7,
        hsv_v         = 0.4,

        # ── Output ────────────────────────────────────────────────────────────
        project       = "runs",
        name          = "finetune_nano_v2",
        exist_ok      = True,
        verbose       = True,
    )

    # Save best.pt to root for easy access
    import shutil
    best_out = Path(results.save_dir) / "weights" / "best.pt"
    dest     = BASE_DIR / "best_v2.pt"
    if best_out.exists():
        shutil.copy(best_out, dest)
        print(f"\n  ✓  Fine-tuned model saved → {dest.resolve()}")
        print(f"  ✓  Full results         → {results.save_dir}")
    else:
        print(f"\n  ⚠  best.pt not found at {best_out}")

    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8 Nano best.pt with merged ICDAR 2013+2015"
    )
    parser.add_argument(
        "--skip_merge", action="store_true",
        help="Skip dataset merging (use if already merged). Goes straight to training."
    )
    parser.add_argument(
        "--merge_only", action="store_true",
        help="Only merge the dataset, do not train."
    )
    args = parser.parse_args()

    if not args.skip_merge:
        # Check ICDAR 2013 is downloaded
        if not check_icdar2013_available():
            print("\n  ✗  Please download ICDAR 2013 first (see instructions above).")
            exit(1)

        build_merged_dataset()
        create_merged_yaml()

    if args.merge_only:
        print("\n  Done. Run again without --merge_only to start training.")
        exit(0)

    if args.skip_merge and not MERGED_YAML.exists():
        print(f"\n  ✗  Merged data.yaml not found at {MERGED_YAML}")
        print("     Run without --skip_merge first to build the merged dataset.")
        exit(1)

    finetune()


# =============================================================================
# USAGE
# =============================================================================
#
#  Full pipeline (merge + train):
#    python finetune_with_icdar2013.py
#
#  Only merge datasets (check output before training):
#    python finetune_with_icdar2013.py --merge_only
#
#  Only train (if already merged):
#    python finetune_with_icdar2013.py --skip_merge
#
#  If RAM < 12 GB, change cache="ram" to cache="disk" in finetune()
#  If OOM error, reduce batch=16 to batch=8
#
# =============================================================================