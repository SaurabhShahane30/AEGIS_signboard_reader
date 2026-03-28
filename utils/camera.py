"""
camera.py — Camera loop: YOLO detection → EasyOCR → priority routing.
KEY FIX: All crops batched together, combined into one string before
priority check — so "MOBILE PHONE" gets detected instead of just "MOBILE".
"""

import cv2
import queue
import time
import numpy as np
from datetime import datetime
from ultralytics import YOLO

# ── Pillow >= 10 compatibility (ANTIALIAS removed, replaced by LANCZOS) ──
import PIL.Image
if not hasattr(PIL.Image, "ANTIALIAS"):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

import easyocr

import torch
from ultralytics.nn import tasks as _ul_tasks

from . import state
print(state.__file__)
from .config import (
    MODEL_PATH, CAMERA_INDEX,
    CONFIDENCE_THRESHOLD, MIN_BOX_AREA,
    WAKE_WORD,
)
from .priority import (
    is_duplicate, build_passive_alert,
    should_speak_priority_alert,
    is_relevant_to_active_list,
    current_mode,
)
from .session import log_event
from .tts import speak


# ── Torch / YOLO patch ───────────────────────────────────────────────
def _patched_torch_safe_load(weight):
    ckpt = torch.load(weight, map_location="cpu", weights_only=False)
    return ckpt, weight

_ul_tasks.torch_safe_load = _patched_torch_safe_load


# ── Constants ────────────────────────────────────────────────────────
OCR_EVERY_N_FRAMES = 4
MIN_CONFIDENCE     = 0.35
MIN_TEXT_LEN       = 2
PAD                = 12

# CLAHE created once at startup
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


def camera_worker():
    """
    Main-thread camera loop — YOLO only, never calls EasyOCR directly.
    Pushes crops to state.crop_queue for ocr_worker() to consume.
    Drops the crop silently if the queue is full (keeps camera real-time).
    """
    print("[Camera] Loading YOLO model...")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {CAMERA_INDEX}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[Camera] Running. Press Q to quit.\n")
    speak("System ready. Say hey system to give a command.")

    while state.running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame_count += 1
        run_ocr_this_frame = (frame_count % OCR_EVERY_N_FRAMES == 0)

        # ── YOLO inference ──────────────────────────────────
        results   = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        annotated = results[0].plot()

        # Collect all crops this frame
        crops_this_frame = []

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            box_area = (x2 - x1) * (y2 - y1)
            if box_area < MIN_BOX_AREA:
                continue

            px1 = max(0, x1 - PAD)
            py1 = max(0, y1 - PAD)
            px2 = min(frame.shape[1], x2 + PAD)
            py2 = min(frame.shape[0], y2 + PAD)

            crop = frame[py1:py2, px1:px2]
            if crop.size == 0:
                continue

            crops_this_frame.append((crop, px1, py1, box_area))

        # ── OCR every N frames only ─────────────────────────
        all_words = []

        if run_ocr_this_frame and crops_this_frame:
            for (crop, px1, py1, box_area) in crops_this_frame:
                try:
                    processed = preprocess_crop_for_ocr(crop)
                    results_ocr = reader.readtext(processed, detail=1)

                    for (_, text, conf) in results_ocr:
                        text = text.strip()
                        if conf >= MIN_CONFIDENCE and len(text) >= MIN_TEXT_LEN:
                            all_words.append(text)
                            print(f"[OCR] '{text}'  (box: {box_area}px²)")

                            cv2.putText(
                                annotated, text,
                                (px1, max(py1 - 8, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                (0, 255, 255), 2,
                            )
                except Exception as e:
                    print(f"[OCR Error] {e}")

        # ── KEY FIX: combine ALL words into ONE string ──────
        # "MOBILE" + "PHONE" + "NOT" + "USE" → "MOBILE PHONE NOT USE"
        # Priority check now sees the full sentence, not individual words
        if all_words:
            combined = " ".join(all_words)
            print(f"[OCR Combined] '{combined}'")

            if not is_duplicate(combined):
                priority_key, passive_alert = build_passive_alert(combined)
                if passive_alert and priority_key and should_speak_priority_alert(priority_key):
                    speak(passive_alert)
                elif len(combined) > 3:
                    state.ocr_queue.put(combined)

        # ── HUD overlay ─────────────────────────────────────────
        ts = datetime.now().strftime("%H:%M:%S")
        cv2.putText(annotated, f"Vision Assistant  {ts}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated, f"Say: '{WAKE_WORD}' + command",
                    (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1)

        mode_label = current_mode()
        mode_color = (100, 255, 180) if state.user_priority_list else (100, 200, 255)
        cv2.putText(annotated, f"Mode: {mode_label}  |  conf>{CONFIDENCE_THRESHOLD}",
                    (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 1)

        cv2.imshow("Vision Assistant", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()