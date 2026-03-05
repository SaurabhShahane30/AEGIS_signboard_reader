"""
=============================================================================
  Signboard Board-Level Cropper
  Merges nearby YOLOv8 text detections into ONE whole-board crop
  RAW pixels — zero colour change, ready for your second model
=============================================================================
  Usage:
    # Single image
    python crop_boards.py --infer raw_data\ch4_test_images\img_0001.jpg

    # Entire folder
    python crop_boards.py --folder raw_data\ch4_test_images

    # Tune merge aggressiveness (higher = merges more boxes together)
    python crop_boards.py --folder raw_data\ch4_test_images --merge_dist 120

    # Also save annotated overview to see what was merged
    python crop_boards.py --infer raw_data\ch4_test_images\img_0001.jpg --annotated
=============================================================================
"""

import cv2
import argparse
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  — tweak these to suit your signboard style
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH  = Path("signboard_project/best.pt")
OUTPUT_DIR  = Path("signboard_project/board_crops")
CONF        = 0.30    # lower than before → catch all text on the board
IOU         = 0.45
MERGE_DIST  = 80      # px — boxes within this distance get merged into one board
                      # increase to 150+ for large signboards with spaced-out text
PADDING     = 20      # px of padding around the merged board crop
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — GROUP nearby boxes using a simple proximity-based merge
# ─────────────────────────────────────────────────────────────────────────────
def expand_box(box, dist):
    """Expand a box outward by `dist` pixels on all sides for overlap testing."""
    x1, y1, x2, y2 = box
    return (x1 - dist, y1 - dist, x2 + dist, y2 + dist)


def boxes_overlap(a, b):
    """Return True if two boxes overlap (or touch)."""
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def merge_boxes(boxes, merge_dist):
    """
    Iteratively merge boxes that are within `merge_dist` pixels of each other.
    Returns a list of merged (x1, y1, x2, y2) bounding rectangles.

    Algorithm:
      1. Expand each box by merge_dist to create a "proximity zone"
      2. If two proximity zones overlap → they belong to the same board
      3. Merge all boxes in a group into their union bounding rectangle
      4. Repeat until no more merges happen (handles chain merging)
    """
    if not boxes:
        return []

    # Work with list of lists for mutability
    groups = [[b] for b in boxes]

    merged = True
    while merged:
        merged = False
        new_groups = []
        used = [False] * len(groups)

        for i in range(len(groups)):
            if used[i]:
                continue
            current = groups[i]
            # Union box of current group
            ux1 = min(b[0] for b in current)
            uy1 = min(b[1] for b in current)
            ux2 = max(b[2] for b in current)
            uy2 = max(b[3] for b in current)
            expanded_i = expand_box((ux1, uy1, ux2, uy2), merge_dist)

            for j in range(i + 1, len(groups)):
                if used[j]:
                    continue
                jx1 = min(b[0] for b in groups[j])
                jy1 = min(b[1] for b in groups[j])
                jx2 = max(b[2] for b in groups[j])
                jy2 = max(b[3] for b in groups[j])
                expanded_j = expand_box((jx1, jy1, jx2, jy2), merge_dist)

                if boxes_overlap(expanded_i, expanded_j):
                    current = current + groups[j]
                    used[j] = True
                    merged = True
                    # Update union for continued merging in this pass
                    ux1 = min(b[0] for b in current)
                    uy1 = min(b[1] for b in current)
                    ux2 = max(b[2] for b in current)
                    uy2 = max(b[3] for b in current)
                    expanded_i = expand_box((ux1, uy1, ux2, uy2), merge_dist)

            new_groups.append(current)
            used[i] = True

        groups = new_groups

    # Convert each group to its union bounding rectangle
    merged_boxes = []
    for group in groups:
        x1 = min(b[0] for b in group)
        y1 = min(b[1] for b in group)
        x2 = max(b[2] for b in group)
        y2 = max(b[3] for b in group)
        merged_boxes.append((x1, y1, x2, y2))

    return merged_boxes


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — DETECT → MERGE → CROP
# ─────────────────────────────────────────────────────────────────────────────
def process_image(model, image_path: Path, conf: float,
                  merge_dist: int, out_dir: Path, annotated: bool):
    """
    Full pipeline for one image:
      detect text boxes → merge into boards → crop raw pixels → save
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  ✗  Could not read: {image_path}")
        return 0

    h, w = img.shape[:2]

    # ── Detect ────────────────────────────────────────────────────────────────
    results = model.predict(
        source  = str(image_path),
        conf    = conf,
        iou     = IOU,
        imgsz   = 640,
        verbose = False,
    )

    raw_boxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            raw_boxes.append((x1, y1, x2, y2))

    if not raw_boxes:
        print(f"  –  [{image_path.name}]  no text detected (try --conf 0.2)")
        return 0

    # ── Merge nearby boxes into boards ───────────────────────────────────────
    board_boxes = merge_boxes(raw_boxes, merge_dist)

    print(f"  [{image_path.name}]  "
          f"{len(raw_boxes)} text regions → {len(board_boxes)} board(s)")

    # ── Crop each board — raw pixels ──────────────────────────────────────────
    saved = 0
    for i, (x1, y1, x2, y2) in enumerate(board_boxes):
        # Add padding, clamp to image size
        x1p = max(0, x1 - PADDING)
        y1p = max(0, y1 - PADDING)
        x2p = min(w, x2 + PADDING)
        y2p = min(h, y2 + PADDING)

        crop = img[y1p:y2p, x1p:x2p]
        if crop.size == 0:
            continue

        out_name = f"{image_path.stem}_board{i}.jpg"
        cv2.imwrite(str(out_dir / out_name), crop,
                    [cv2.IMWRITE_JPEG_QUALITY, 100])
        print(f"    ✓  board {i}  ({x2p-x1p}×{y2p-y1p}px)  →  {out_name}")
        saved += 1

    # ── Optional annotated overview ───────────────────────────────────────────
    if annotated:
        vis = img.copy()
        # Draw individual text boxes in blue
        for (x1, y1, x2, y2) in raw_boxes:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 150, 0), 1)
        # Draw merged board boxes in green (thick)
        for i, (x1, y1, x2, y2) in enumerate(board_boxes):
            x1p = max(0, x1 - PADDING)
            y1p = max(0, y1 - PADDING)
            x2p = min(w, x2 + PADDING)
            y2p = min(h, y2 + PADDING)
            cv2.rectangle(vis, (x1p, y1p), (x2p, y2p), (0, 255, 0), 3)
            cv2.putText(vis, f"board {i}", (x1p, max(y1p - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        ann_path = out_dir / f"annotated_{image_path.name}"
        cv2.imwrite(str(ann_path), vis)
        print(f"    📋  annotated → {ann_path.name}  "
              f"(blue=text boxes, green=merged boards)")

    return saved


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Crop whole signboards by merging nearby text detections"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--infer",  type=str, help="Single image path")
    group.add_argument("--folder", type=str, help="Folder of images")

    parser.add_argument("--conf",       type=float, default=CONF,
                        help=f"Detection confidence (default {CONF})")
    parser.add_argument("--merge_dist", type=int,   default=MERGE_DIST,
                        help=f"Max px gap between boxes to merge (default {MERGE_DIST})")
    parser.add_argument("--model",      type=str,   default=str(MODEL_PATH),
                        help="Path to best.pt")
    parser.add_argument("--annotated",  action="store_true",
                        help="Save annotated overview image for visual QA")
    args = parser.parse_args()

    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"✗  Model not found: {model_path}")
        return
    print(f"\nModel  : {model_path}")
    model = YOLO(str(model_path))

    # Collect images
    if args.infer:
        images = [Path(args.infer)]
    else:
        folder = Path(args.folder)
        images = [p for p in sorted(folder.iterdir())
                  if p.suffix.lower() in IMG_EXTS]
        print(f"Images : {len(images)} found in {folder}")

    # Output dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output : {OUTPUT_DIR.resolve()}")
    print(f"Merge distance : {args.merge_dist}px\n")

    # Process
    total = 0
    for img_path in images:
        total += process_image(
            model, img_path,
            conf       = args.conf,
            merge_dist = args.merge_dist,
            out_dir    = OUTPUT_DIR,
            annotated  = args.annotated,
        )

    print(f"\n{'='*50}")
    print(f"  Done.  {total} board crop(s) saved → {OUTPUT_DIR.resolve()}")
    print(f"{'='*50}\n")

    # ── Tuning guide ──────────────────────────────────────────────────────────
    if total == 0:
        print("  💡 No crops saved. Try:")
        print("     --conf 0.2          (detect more text)")
        print("     --merge_dist 150    (merge farther-apart boxes)")


if __name__ == "__main__":
    main()