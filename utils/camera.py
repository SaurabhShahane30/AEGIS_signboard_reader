"""
camera.py — Camera loop: YOLO detection → EasyOCR → priority routing.
KEY FIX: Nearby boxes are merged into one big region before OCR.
So instead of reading 'MOBILE', 'PHONE' separately,
it reads the whole signboard as 'DO NOT USE MOBILE PHONE'.
"""

import cv2
import easyocr
import numpy as np
from datetime import datetime
from ultralytics import YOLO

from . import state
print(state.__file__)
from .config import (
    MODEL_PATH, CAMERA_INDEX,
    CONFIDENCE_THRESHOLD, MIN_BOX_AREA,
    WAKE_WORD,
)
from .priority import is_duplicate, build_passive_alert, should_speak_priority_alert
from .tts import speak

import torch
from ultralytics.nn import tasks as _ul_tasks

def _patched_torch_safe_load(weight):
    ckpt = torch.load(weight, map_location="cpu", weights_only=False)
    return ckpt, weight

_ul_tasks.torch_safe_load = _patched_torch_safe_load

# ── Constants ────────────────────────────────────────────────────────
OCR_EVERY_N_FRAMES = 4
MIN_CONFIDENCE     = 0.35
MIN_TEXT_LEN       = 2
PAD                = 12

# How close two boxes must be (in pixels) to be grouped as one signboard.
# Increase this if boxes on the same sign are still being treated separately.
# Decrease if boxes from different signs are being wrongly merged.
MERGE_DISTANCE     = 80

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def preprocess_crop_for_ocr(crop: np.ndarray) -> np.ndarray:
    h, w = crop.shape[:2]
    if h < 120:
        scale = 120 / h
        crop = cv2.resize(crop, (max(int(w * scale), 1), 120),
                          interpolation=cv2.INTER_CUBIC)
    gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray  = _CLAHE.apply(gray)
    gray  = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)


def boxes_are_close(b1, b2, max_dist: int) -> bool:
    """
    Check if two boxes are close enough to belong to the same signboard.

    b1 and b2 are (x1, y1, x2, y2) tuples.

    Logic: find the gap between the two boxes in both X and Y directions.
    If both gaps are within max_dist, they are neighbours on the same sign.

    Gap calculation:
      - If boxes overlap in an axis, gap = 0 (they're already touching)
      - If boxes don't overlap, gap = distance between their nearest edges
    """
    x_gap = max(0, max(b1[0], b2[0]) - min(b1[2], b2[2]))
    y_gap = max(0, max(b1[1], b2[1]) - min(b1[3], b2[3]))
    return x_gap <= max_dist and y_gap <= max_dist


def merge_boxes(boxes: list[tuple]) -> list[tuple]:
    """
    Group nearby boxes and merge each group into one big bounding box.

    Returns a list of merged (x1, y1, x2, y2) boxes — one per signboard.

    Algorithm (Union-Find style grouping):
      1. Start — every box is its own group.
      2. For every pair of boxes, if they are close → merge their groups.
      3. Repeat until no more merges happen.
      4. For each group, compute the bounding box that covers all boxes in it.

    Why not just one pass?
      A chain like [box1 near box2, box2 near box3] needs multiple passes
      so box1 and box3 end up in the same group even if they aren't directly close.
    """
    if not boxes:
        return []

    # Each box starts in its own group (group id = index)
    groups = list(range(len(boxes)))

    def find(i):
        while groups[i] != i:
            groups[i] = groups[groups[i]]
            i = groups[i]
        return i

    def union(i, j):
        groups[find(i)] = find(j)

    # Merge groups for all close pairs
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes_are_close(boxes[i], boxes[j], MERGE_DISTANCE):
                union(i, j)

    # Collect all boxes that share the same root into groups
    group_map = {}
    for i, box in enumerate(boxes):
        root = find(i)
        if root not in group_map:
            group_map[root] = []
        group_map[root].append(box)

    # For each group, compute one big bounding box covering all boxes in it
    merged = []
    for group_boxes in group_map.values():
        x1 = min(b[0] for b in group_boxes)
        y1 = min(b[1] for b in group_boxes)
        x2 = max(b[2] for b in group_boxes)
        y2 = max(b[3] for b in group_boxes)
        merged.append((x1, y1, x2, y2))

    return merged


def camera_worker():
    import time

    print("[Camera] Loading YOLO model...")
    model = YOLO(MODEL_PATH)

    print("[OCR] Initialising EasyOCR...")
    reader = easyocr.Reader(["en"], gpu=True)

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {CAMERA_INDEX}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[Camera] Running. Press Q to quit.\n")
    speak("System ready. Say hey system to give a command.")

    frame_count = 0

    while state.running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame_count += 1
        run_ocr_this_frame = (frame_count % OCR_EVERY_N_FRAMES == 0)

        # ── YOLO inference ──────────────────────────────────────
        results   = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        annotated = results[0].plot()

        # ── Step 1: Collect all valid boxes from YOLO ───────────
        raw_boxes = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_area = (x2 - x1) * (y2 - y1)
            if box_area < MIN_BOX_AREA:
                continue
            # Apply padding to each individual box before merging
            x1 = max(0, x1 - PAD)
            y1 = max(0, y1 - PAD)
            x2 = min(frame.shape[1], x2 + PAD)
            y2 = min(frame.shape[0], y2 + PAD)
            raw_boxes.append((x1, y1, x2, y2))

        # ── Step 2: Merge nearby boxes into one region per sign ─
        merged_boxes = merge_boxes(raw_boxes)

        # ── Step 3: OCR each merged region as one signboard ─────
        all_words = []

        if run_ocr_this_frame and merged_boxes:
            for (mx1, my1, mx2, my2) in merged_boxes:
                crop = frame[my1:my2, mx1:mx2]
                if crop.size == 0:
                    continue

                try:
                    processed   = preprocess_crop_for_ocr(crop)
                    results_ocr = reader.readtext(processed, detail=1)

                    box_words = []
                    for (_, text, conf) in results_ocr:
                        text = text.strip()
                        if conf >= MIN_CONFIDENCE and len(text) >= MIN_TEXT_LEN:
                            box_words.append(text)

                    if box_words:
                        # Join words from this merged region in reading order
                        sign_text = " ".join(box_words)
                        print(f"[OCR Sign] '{sign_text}'")
                        all_words.append(sign_text)

                        # Draw merged box on screen in green
                        cv2.rectangle(annotated, (mx1, my1), (mx2, my2),
                                      (0, 255, 0), 2)
                        cv2.putText(
                            annotated, sign_text[:40],  # cap display length
                            (mx1, max(my1 - 8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (0, 255, 255), 2,
                        )

                except Exception as e:
                    print(f"[OCR Error] {e}")

        # ── Step 4: Combine all signs and route to priority/LLM ─
        if all_words:
            combined = " | ".join(all_words)
            print(f"[OCR Combined] '{combined}'")

            if not is_duplicate(combined):
                priority_key, passive_alert = build_passive_alert(combined)
                if passive_alert and priority_key and should_speak_priority_alert(priority_key):
                    speak(passive_alert)
                elif len(combined) > 3:
                    state.ocr_queue.put(combined)

        # ── HUD overlay ──────────────────────────────────────────
        ts = datetime.now().strftime("%H:%M:%S")
        cv2.putText(annotated, f"Vision Assistant  {ts}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated, f"Say: '{WAKE_WORD}' + command",
                    (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1)
        cv2.putText(annotated, f"Custom priorities: {len(state.user_priority_list)}  |  conf>{CONFIDENCE_THRESHOLD}",
                    (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 255), 1)

        cv2.imshow("Vision Assistant", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()