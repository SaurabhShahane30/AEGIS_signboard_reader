"""
=============================================================================
 Real-Time Signboard Detection Pipeline
 YOLOv8 + ICDAR 2015 Scene Text Dataset
 Optimised for NVIDIA RTX 3050 (6 GB VRAM)
=============================================================================

DEPENDENCIES  (install once):
    pip install ultralytics opencv-python-headless Pillow tqdm requests
    pip install pytesseract          # optional – for OCR demo
    # If you prefer EasyOCR instead of Tesseract:
    # pip install easyocr

ICDAR 2015 DATASET DOWNLOAD:
    The dataset requires registration at:
    https://rrc.cvc.uab.es/?ch=4&com=downloads
    After downloading, place the files as follows:
        raw_data/
            ch4_training_images/          ← 1000 training images
            ch4_training_localization_transcription_gt/  ← GT txt files
            ch4_test_images/              ← 500 test images
            Challenge4_Test_Task1_GT/     ← test GT (optional)
    OR run this script with --download_sample to get a small demo subset.
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  IMPORTS & GLOBAL PATHS
# ─────────────────────────────────────────────────────────────────────────────
import os, shutil, random, math, argparse, zipfile, urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# 1.  PATHS  –  edit RAW_DATA_DIR if your ICDAR folder is elsewhere
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR       = Path("signboard_project")
RAW_DATA_DIR   = Path("raw_data")          # put downloaded ICDAR data here

TRAIN_IMG_DIR  = RAW_DATA_DIR / "ch4_training_images"
TRAIN_GT_DIR   = RAW_DATA_DIR / "ch4_training_localization_transcription_gt"
TEST_IMG_DIR   = RAW_DATA_DIR / "ch4_test_images"

DATASET_DIR    = BASE_DIR / "dataset"
YOLO_TRAIN_IMG = DATASET_DIR / "images" / "train"
YOLO_VAL_IMG   = DATASET_DIR / "images" / "val"
YOLO_TRAIN_LBL = DATASET_DIR / "labels" / "train"
YOLO_VAL_LBL   = DATASET_DIR / "labels" / "val"
RUNS_DIR       = BASE_DIR / "runs"
CROPS_DIR      = BASE_DIR / "crops"

YAML_PATH      = DATASET_DIR / "data.yaml"

# Training hyper-parameters tuned for RTX 3050 6 GB
TRAIN_CFG = dict(
    model      = "yolov8n.pt",   # nano – smallest footprint; use yolov8s.pt for +accuracy
    imgsz      = 640,            # 640 fits comfortably in 6 GB; lower to 512 if OOM
    batch  = 16,   # was 8 — doubles throughput
    epochs     = 100,
    patience   = 20,             # early stopping
    workers    = 4,
    device     = 0,              # GPU 0
    optimizer  = "AdamW",
    lr0        = 1e-3,
    lrf        = 0.01,
    mosaic     = 1.0,
    augment    = True,
    amp        = True,           # automatic mixed precision → saves ~1.5 GB VRAM
    cache      = "ram",          # cache images in RAM for speed; set "disk" if RAM<16GB
    project    = str(RUNS_DIR),
    name       = "signboard_v1",
    exist_ok   = True,
)

VAL_SPLIT = 0.15   # 15 % of training images → validation
SEED      = 42


# ─────────────────────────────────────────────────────────────────────────────
# 2.  (OPTIONAL) DOWNLOAD A SMALL DEMO SUBSET FROM ICDAR SAMPLE MIRROR
#     Real training requires the full dataset from rrc.cvc.uab.es
# ─────────────────────────────────────────────────────────────────────────────
DEMO_URLS = {
    # ~5 MB sample hosted publicly for CI / quick-start testing
    "images": "https://www.kaggle.com/api/v1/datasets/download/nageshsingh/icdar-2015",
    # NOTE: Kaggle requires authentication; replace with your own mirror if needed.
}

def download_sample():
    """
    Prints instructions for obtaining ICDAR 2015.
    Automated download requires a free account at rrc.cvc.uab.es.
    """
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║  ICDAR 2015 Download Instructions                                    ║
╠══════════════════════════════════════════════════════════════════════╣
║  1. Register (free) at: https://rrc.cvc.uab.es/?ch=4&com=downloads  ║
║  2. Download:                                                        ║
║     • ch4_training_images.zip          (~89 MB)                     ║
║     • ch4_training_localization_        (~3 MB)                     ║
║       transcription_gt.zip                                          ║
║     • ch4_test_images.zip              (~45 MB)                     ║
║  3. Unzip all into  raw_data/                                        ║
║  4. Re-run this script (without --download_sample)                  ║
╚══════════════════════════════════════════════════════════════════════╝
    """)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  ANNOTATION CONVERTER  –  ICDAR quad → YOLO axis-aligned bounding box
# ─────────────────────────────────────────────────────────────────────────────
def quad_to_yolo(x1,y1,x2,y2,x3,y3,x4,y4, img_w, img_h):
    """
    Convert a quadrilateral (8 coords) to a YOLO normalised bbox.
    YOLO format:  class_id  cx  cy  w  h   (all 0-1 relative to image size)

    We take the axis-aligned bounding rectangle of the quad points so that
    rotated / perspective text boxes are safely enclosed.
    """
    xs = [x1, x2, x3, x4]
    ys = [y1, y2, y3, y4]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    cx = (xmin + xmax) / 2.0 / img_w
    cy = (ymin + ymax) / 2.0 / img_h
    bw = (xmax - xmin) / img_w
    bh = (ymax - ymin) / img_h

    # Clamp to [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    bw = max(0.0, min(1.0, bw))
    bh = max(0.0, min(1.0, bh))

    return cx, cy, bw, bh


def parse_icdar_gt(gt_path):
    """
    Parse one ICDAR 2015 ground-truth file.
    Each line: x1,y1,x2,y2,x3,y3,x4,y4,transcription
    Returns list of (x1,y1,...,x4,y4) tuples (ints).
    Lines with '###' (illegible) are included as hard negatives.
    """
    annotations = []
    with open(gt_path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 8)          # split on first 8 commas only
            if len(parts) < 9:
                continue
            try:
                coords = list(map(int, parts[:8]))
                annotations.append(tuple(coords))
            except ValueError:
                continue                         # skip malformed lines
    return annotations


def convert_annotations(img_path, gt_path, out_label_path):
    """
    Convert a single ICDAR GT file to YOLO label format.
    Writes  <out_label_path>.txt  with one bbox per line.
    Returns number of boxes written (0 means empty / skip).
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return 0
    h, w = img.shape[:2]

    annotations = parse_icdar_gt(gt_path)
    if not annotations:
        return 0

    lines = []
    for coords in annotations:
        x1,y1,x2,y2,x3,y3,x4,y4 = coords
        cx, cy, bw, bh = quad_to_yolo(x1,y1,x2,y2,x3,y3,x4,y4, w, h)
        if bw < 1e-4 or bh < 1e-4:
            continue                             # degenerate box – skip
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    if not lines:
        return 0

    out_label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_label_path, "w") as f:
        f.write("\n".join(lines))
    return len(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  DATASET BUILDER  –  copies images & labels, performs train/val split
# ─────────────────────────────────────────────────────────────────────────────
def build_dataset():
    """
    1. Collects all paired (image, GT) files from ICDAR training folder.
    2. Converts annotations to YOLO format in a temp directory.
    3. Randomly splits into train / val.
    4. Creates the final YOLO folder structure.
    """
    print("\n[1/4]  Scanning ICDAR 2015 training data …")
    img_files = sorted(TRAIN_IMG_DIR.glob("*.jpg"))
    if not img_files:
        raise FileNotFoundError(
            f"No .jpg images found in {TRAIN_IMG_DIR}\n"
            "Please download ICDAR 2015 and place it under raw_data/. "
            "Run with --download_sample for instructions."
        )
    print(f"       Found {len(img_files)} images.")

    # ── temporary conversion output ──────────────────────────────────────────
    tmp_img = BASE_DIR / "tmp" / "images"
    tmp_lbl = BASE_DIR / "tmp" / "labels"
    tmp_img.mkdir(parents=True, exist_ok=True)
    tmp_lbl.mkdir(parents=True, exist_ok=True)

    print("[2/4]  Converting ICDAR annotations → YOLO format …")
    valid_stems = []
    for img_path in tqdm(img_files, unit="img"):
        stem = img_path.stem
        gt_filename = "gt_" + stem + ".txt"
        gt_path = TRAIN_GT_DIR / gt_filename

        if not gt_path.exists():
            continue  # skip images without GT

        out_lbl = tmp_lbl / (stem + ".txt")
        n = convert_annotations(img_path, gt_path, out_lbl)
        if n > 0:
            shutil.copy(img_path, tmp_img / img_path.name)
            valid_stems.append(stem)

    print(f"       {len(valid_stems)} images with valid annotations.")

    # ── train / val split ────────────────────────────────────────────────────
    print("[3/4]  Splitting into train / val …")
    random.seed(SEED)
    random.shuffle(valid_stems)
    n_val   = max(1, math.floor(len(valid_stems) * VAL_SPLIT))
    val_set = set(valid_stems[:n_val])
    trn_set = set(valid_stems[n_val:])
    print(f"       Train: {len(trn_set)}  |  Val: {len(val_set)}")

    # ── copy to final YOLO structure ──────────────────────────────────────────
    print("[4/4]  Building YOLO directory structure …")
    for d in [YOLO_TRAIN_IMG, YOLO_VAL_IMG, YOLO_TRAIN_LBL, YOLO_VAL_LBL]:
        d.mkdir(parents=True, exist_ok=True)

    for stem in tqdm(valid_stems, unit="file"):
        img_src = tmp_img / (stem + ".jpg")
        lbl_src = tmp_lbl / (stem + ".txt")
        if stem in trn_set:
            shutil.copy(img_src, YOLO_TRAIN_IMG / img_src.name)
            shutil.copy(lbl_src, YOLO_TRAIN_LBL / lbl_src.name)
        else:
            shutil.copy(img_src, YOLO_VAL_IMG   / img_src.name)
            shutil.copy(lbl_src, YOLO_VAL_LBL   / lbl_src.name)

    # Clean temp dir
    shutil.rmtree(BASE_DIR / "tmp", ignore_errors=True)
    print("  ✓  Dataset ready.\n")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  DATA.YAML
# ─────────────────────────────────────────────────────────────────────────────
def create_yaml():
    """Write the data.yaml that Ultralytics expects."""
    yaml_content = f"""# ICDAR 2015 → YOLOv8 – Single class: "text"
path: {DATASET_DIR.resolve()}
train: images/train
val:   images/val

nc: 1
names:
  0: text
"""
    YAML_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(YAML_PATH, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"[YAML]  Saved → {YAML_PATH}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def train_model():
    """
    Train YOLOv8 with parameters optimised for RTX 3050 6 GB VRAM.

    Key RTX-3050 optimisations:
    ─────────────────────────────
    • amp=True          – FP16 mixed precision saves ~40 % VRAM
    • batch=8           – safe ceiling; bump to 16 only if GPU shows <80 % usage
    • imgsz=640         – standard YOLO input; drop to 512 if OOM errors occur
    • cache='ram'       – speeds up epochs 2+ by keeping images in memory
    • workers=4         – balanced for most CPU configurations
    • model='yolov8n'   – nano backbone is fastest; upgrade to 's' for accuracy
    """
    from ultralytics import YOLO

    print("=" * 60)
    print("  Starting YOLOv8 Training")
    print("=" * 60)

    model = YOLO(TRAIN_CFG["model"])   # downloads pretrained weights automatically

    results = model.train(
        data       = str(YAML_PATH),
        imgsz      = TRAIN_CFG["imgsz"],
        batch      = TRAIN_CFG["batch"],
        epochs     = TRAIN_CFG["epochs"],
        patience   = TRAIN_CFG["patience"],
        workers    = TRAIN_CFG["workers"],
        device     = TRAIN_CFG["device"],
        optimizer  = TRAIN_CFG["optimizer"],
        lr0        = TRAIN_CFG["lr0"],
        lrf        = TRAIN_CFG["lrf"],
        mosaic     = TRAIN_CFG["mosaic"],
        augment    = TRAIN_CFG["augment"],
        amp        = TRAIN_CFG["amp"],
        cache      = TRAIN_CFG["cache"],
        project    = TRAIN_CFG["project"],
        name       = TRAIN_CFG["name"],
        exist_ok   = TRAIN_CFG["exist_ok"],
        verbose    = True,
    )

    # ── locate best.pt ────────────────────────────────────────────────────────
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    dest_pt = BASE_DIR / "best.pt"
    if best_pt.exists():
        shutil.copy(best_pt, dest_pt)
        print(f"\n  ✓  best.pt saved → {dest_pt.resolve()}")
    else:
        print(f"\n  ⚠  best.pt not found at expected path: {best_pt}")

    return dest_pt if dest_pt.exists() else best_pt


# ─────────────────────────────────────────────────────────────────────────────
# 7.  INFERENCE  –  detect text regions on a test image
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(model_path, image_path, conf_thresh=0.35, iou_thresh=0.45):
    """
    Run YOLOv8 inference on a single image.
    Returns:
        annotated  – BGR image with drawn boxes
        detections – list of dicts {xyxy, conf, cls}
    """
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    results = model.predict(
        source  = str(image_path),
        conf    = conf_thresh,
        iou     = iou_thresh,
        imgsz   = TRAIN_CFG["imgsz"],
        device  = TRAIN_CFG["device"],
        verbose = False,
    )

    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf  = float(box.conf[0])
            cls   = int(box.cls[0])
            detections.append({"xyxy": (x1, y1, x2, y2), "conf": conf, "cls": cls})

    # Draw bounding boxes on a copy of the image
    img = cv2.imread(str(image_path))
    annotated = img.copy()
    for det in detections:
        x1, y1, x2, y2 = det["xyxy"]
        label = f"text {det['conf']:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, label, (x1, max(y1-6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    return annotated, detections


# ─────────────────────────────────────────────────────────────────────────────
# 8.  OCR CROPPING  –  cut detected regions, pass to Tesseract / EasyOCR
# ─────────────────────────────────────────────────────────────────────────────
def crop_and_ocr(image_path, detections, use_easyocr=False):
    """
    For each detected bounding box:
      1. Crop the region from the original image.
      2. Apply a light pre-processing pipeline (resize, sharpen, threshold).
      3. Pass to OCR engine and print results.

    Set use_easyocr=True if you have EasyOCR installed; otherwise uses Tesseract.
    """
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(image_path))

    if use_easyocr:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=True)
    else:
        try:
            import pytesseract
        except ImportError:
            print("  ⚠  pytesseract not installed. Install it or use --easyocr flag.")
            pytesseract = None

    results_ocr = []
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["xyxy"]
        # Add a small padding around the crop
        pad  = 4
        h, w = img.shape[:2]
        x1p  = max(0, x1 - pad)
        y1p  = max(0, y1 - pad)
        x2p  = min(w, x2 + pad)
        y2p  = min(h, y2 + pad)

        crop = img[y1p:y2p, x1p:x2p]
        if crop.size == 0:
            continue

        # ── Pre-processing for better OCR accuracy ────────────────────────────
        # 1. Upscale to at least 32 px tall (Tesseract performs best ≥32 px)
        target_h = max(crop.shape[0], 48)
        scale    = target_h / crop.shape[0]
        crop_up  = cv2.resize(crop, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)
        # 2. Convert to greyscale
        grey = cv2.cvtColor(crop_up, cv2.COLOR_BGR2GRAY)
        # 3. Sharpen
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharp  = cv2.filter2D(grey, -1, kernel)
        # 4. Otsu threshold (binarise)
        _, thresh = cv2.threshold(sharp, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        crop_path = CROPS_DIR / f"crop_{i:03d}.png"
        cv2.imwrite(str(crop_path), thresh)

        # ── OCR ───────────────────────────────────────────────────────────────
        text = ""
        if use_easyocr:
            ocr_result = reader.readtext(np.array(thresh))
            text = " ".join([r[1] for r in ocr_result])
        elif pytesseract:
            text = pytesseract.image_to_string(
                thresh,
                config="--psm 7 --oem 3"   # single-line mode
            ).strip()

        results_ocr.append({"box": det["xyxy"], "text": text, "crop": crop_path})
        print(f"  Crop {i:03d}  [{x1},{y1} → {x2},{y2}]  OCR: '{text}'")

    return results_ocr


# ─────────────────────────────────────────────────────────────────────────────
# 9.  MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="YOLOv8 Signboard Detection – ICDAR 2015 Pipeline"
    )
    parser.add_argument("--download_sample", action="store_true",
                        help="Print ICDAR 2015 download instructions and exit.")
    parser.add_argument("--skip_dataset",    action="store_true",
                        help="Skip dataset build (use if already converted).")
    parser.add_argument("--skip_train",      action="store_true",
                        help="Skip training (use existing best.pt).")
    parser.add_argument("--infer",           type=str, default=None,
                        help="Path to a test image for inference demo.")
    parser.add_argument("--model",           type=str,
                        default=str(BASE_DIR / "best.pt"),
                        help="Path to trained model weights (for --infer).")
    parser.add_argument("--easyocr",         action="store_true",
                        help="Use EasyOCR instead of Tesseract.")
    parser.add_argument("--conf",            type=float, default=0.35,
                        help="Detection confidence threshold (default 0.35).")
    args = parser.parse_args()

    # ── 0.  Print download instructions ──────────────────────────────────────
    if args.download_sample:
        download_sample()
        return

    # ── 1-4.  Build dataset ───────────────────────────────────────────────────
    if not args.skip_dataset:
        build_dataset()

    # ── 5.  Create data.yaml ──────────────────────────────────────────────────
    if not args.skip_dataset:
        create_yaml()

    # ── 6-8.  Train ───────────────────────────────────────────────────────────
    model_path = Path(args.model)
    if not args.skip_train:
        model_path = train_model()

    # ── 9-10.  Inference + OCR ────────────────────────────────────────────────
    if args.infer:
        image_path = Path(args.infer)
        if not image_path.exists():
            print(f"  ✗  Image not found: {image_path}")
            return

        if not model_path.exists():
            print(f"  ✗  Model not found: {model_path}\n"
                  "     Train first or pass --model <path>.")
            return

        print(f"\n[Inference]  Image : {image_path}")
        print(f"[Inference]  Model : {model_path}")
        annotated, detections = run_inference(
            model_path, image_path, conf_thresh=args.conf
        )

        # Save annotated image
        out_img = BASE_DIR / ("result_" + image_path.name)
        cv2.imwrite(str(out_img), annotated)
        print(f"  ✓  Annotated image → {out_img}")
        print(f"  ✓  Detections      : {len(detections)}")

        # Crop + OCR
        if detections:
            print("\n[OCR]  Cropping detected regions …")
            crop_and_ocr(image_path, detections, use_easyocr=args.easyocr)
            print(f"  ✓  Crops saved → {CROPS_DIR}/")
    else:
        print("\nTip: Run inference with:  python pipeline.py --infer <image.jpg>")


if __name__ == "__main__":
    main()


# =============================================================================
# QUICK-START GUIDE
# =============================================================================
#
#  Step 1 – Get dataset instructions:
#    python icdar2yolo_pipeline.py --download_sample
#
#  Step 2 – Full pipeline (build + train):
#    python icdar2yolo_pipeline.py
#
#  Step 3 – Inference on a test image:
#    python icdar2yolo_pipeline.py --skip_train --infer path/to/image.jpg
#
#  Step 4 – Inference + OCR (Tesseract):
#    python icdar2yolo_pipeline.py --skip_train --infer path/to/image.jpg
#
#  Step 5 – Inference + OCR (EasyOCR):
#    python icdar2yolo_pipeline.py --skip_train --infer path/to/image.jpg --easyocr
#
#  MEMORY TIPS (RTX 3050 6 GB):
#  ─────────────────────────────
#  • OOM during training?  → set imgsz=512 and/or batch=4 in TRAIN_CFG
#  • Want more accuracy?   → switch model to "yolov8s.pt" (small, ~2× slower)
#  • Very long training?   → set cache="disk" if RAM < 8 GB
#  • Exporting to ONNX / TensorRT for deployment:
#       from ultralytics import YOLO
#       model = YOLO("signboard_project/best.pt")
#       model.export(format="onnx", imgsz=640, simplify=True)
# =============================================================================
