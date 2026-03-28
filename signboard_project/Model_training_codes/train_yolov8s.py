"""
=============================================================================
  Train YOLOv8s on Merged ICDAR 2013 + 2015 Dataset
  Strategy: Full model training (no freeze), larger model, more data
  Hardware: NVIDIA RTX 3050 6GB Laptop GPU
  Expected mAP50: 0.58 - 0.65 (up from 0.52 Nano)
=============================================================================

REQUIREMENTS:
  - Merged dataset already built at: signboard_project/dataset_merged/
  - Run merge first if not done: python mew_model_finetune.py --merge_only
  - CUDA PyTorch installed (torch+cu121)

RUN:
  python train_yolov8s.py

OUTPUT:
  signboard_project/best_yolov8s.pt   ← your improved model
  runs/yolov8s_merged/                ← full training results
"""

import shutil
from pathlib import Path
from ultralytics import YOLO

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR    = Path("signboard_project")
MERGED_YAML = BASE_DIR / "dataset_merged" / "data.yaml"
OUTPUT_PT   = BASE_DIR / "best_yolov8s.pt"

# =============================================================================
# SANITY CHECKS
# =============================================================================

def check_ready():
    if not MERGED_YAML.exists():
        print(f"""
  ✗  Merged dataset not found at: {MERGED_YAML}
     Please run the merge step first:
       python mew_model_finetune.py --merge_only
        """)
        return False

    # Count images
    train_imgs = list((BASE_DIR / "dataset_merged" / "images" / "train").glob("*"))
    val_imgs   = list((BASE_DIR / "dataset_merged" / "images" / "val").glob("*"))
    print(f"  ✓  Dataset found: {len(train_imgs)} train | {len(val_imgs)} val images")

    if len(train_imgs) < 900:
        print("  ⚠  Train images seem low — did ICDAR 2013 merge correctly?")
        print("     Expected ~1045. Run: python mew_model_finetune.py --merge_only")
        return False

    return True


# =============================================================================
# TRAINING
# =============================================================================

def train():
    print("=" * 60)
    print("  YOLOv8s Training on Merged ICDAR 2013 + 2015")
    print("=" * 60)
    print(f"  Model   : yolov8s.pt (COCO pretrained Small)")
    print(f"  Data    : {MERGED_YAML}")
    print(f"  Train   : ~1045 images | Val: ~184 images")
    print(f"  Strategy: Full model (no freeze), lower LR, rich augmentation")
    print("=" * 60)

    # Load YOLOv8s from COCO pretrained — downloads automatically (~22 MB)
    model = YOLO("yolov8s.pt")

    results = model.train(
        data          = str(MERGED_YAML),

        # ── Core ──────────────────────────────────────────────────────────────
        epochs        = 150,          # more epochs — Small needs more time to converge
        patience      = 25,           # slightly more patience than before
        imgsz         = 640,
        batch         = 8,            # reduced from 16 — Small has more params, needs more VRAM
        workers       = 4,
        device        = 0,            # GPU
        amp           = True,         # FP16 mixed precision — saves ~1.5GB VRAM
        cache         = "ram",        # faster training; use "disk" if RAM < 12GB

        # ── Learning Rate ─────────────────────────────────────────────────────
        # Lower than default (1e-3) since we're fine-tuning COCO → scene text
        # High enough to actually learn new domain features
        optimizer     = "AdamW",
        lr0           = 5e-4,         # starting LR
        lrf           = 0.01,         # cosine decay to lr0 * lrf = 5e-6
        warmup_epochs = 3,            # gentle warmup for first 3 epochs
        warmup_bias_lr= 0.1,

        # ── NO freeze — full model training ───────────────────────────────────
        # Nano+freeze failed; Small is strong enough to train fully
        # freeze      = 0,            # this is the default, no need to set

        # ── Augmentations tuned for scene text ────────────────────────────────
        mosaic        = 1.0,          # combine 4 images — critical for small text
        mixup         = 0.15,         # blend images — adds lighting variety
        copy_paste    = 0.1,          # paste text boxes across images
        degrees       = 15.0,         # ±15° rotation — handles tilted/rotated signs
        translate     = 0.1,          # random shift
        scale         = 0.5,          # random scale
        shear         = 5.0,          # perspective-like shear
        perspective   = 0.0005,       # camera angle simulation
        flipud        = 0.0,          # NEVER flip upside down — text is directional
        fliplr        = 0.5,          # horizontal flip is fine
        hsv_h         = 0.015,        # hue jitter
        hsv_s         = 0.7,          # saturation jitter — handles overcast/sunny
        hsv_v         = 0.4,          # brightness jitter

        # ── Output ────────────────────────────────────────────────────────────
        project       = "runs",
        name          = "yolov8s_merged",
        exist_ok      = True,
        verbose       = True,
    )

    # ── Save best.pt to project root ──────────────────────────────────────────
    best_out = Path(results.save_dir) / "weights" / "best.pt"
    if best_out.exists():
        shutil.copy(best_out, OUTPUT_PT)
        print(f"\n{'=' * 60}")
        print(f"  ✓  Best model saved → {OUTPUT_PT.resolve()}")
        print(f"  ✓  Full results     → {results.save_dir}")
        print(f"{'=' * 60}")

        # Print final metrics
        metrics = results.results_dict
        print(f"\n  Final Metrics:")
        print(f"  mAP50    : {metrics.get('metrics/mAP50(B)', 'N/A'):.4f}")
        print(f"  mAP50-95 : {metrics.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
        print(f"  Precision: {metrics.get('metrics/precision(B)', 'N/A'):.4f}")
        print(f"  Recall   : {metrics.get('metrics/recall(B)', 'N/A'):.4f}")
    else:
        print(f"\n  ⚠  best.pt not found at expected path: {best_out}")

    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    if not check_ready():
        exit(1)
    train()


# =============================================================================
# TIPS
# =============================================================================
#
#  OOM (out of memory) error?
#    → Change batch=8 to batch=4
#    → Or change cache="ram" to cache="disk"
#
#  Want to run inference with the new model?
#    from ultralytics import YOLO
#    model = YOLO("signboard_project/best_yolov8s.pt")
#    results = model.predict("path/to/image.jpg", conf=0.35)
#    results[0].show()
#
#  Want to compare Nano vs Small on the same image?
#    nano  = YOLO("signboard_project/best.pt")
#    small = YOLO("signboard_project/best_yolov8s.pt")
#
#  Export to ONNX for deployment:
#    model = YOLO("signboard_project/best_yolov8s.pt")
#    model.export(format="onnx", imgsz=640, simplify=True)
#
# =============================================================================
