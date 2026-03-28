"""
camera.py — Camera loop: YOLO detection → EasyOCR → priority routing.
Optimised: OCR every 5th frame, per-box cache, strong deduplication.
"""

import cv2
import easyocr
import numpy as np
from datetime import datetime
from ultralytics import YOLO

import state
from config import (
    MODEL_PATH, CAMERA_INDEX,
    CONFIDENCE_THRESHOLD, MIN_BOX_AREA,
    WAKE_WORD,
)
from priority import is_duplicate, build_passive_alert, should_speak_priority_alert
from tts import speak

import torch
from ultralytics.nn import tasks as _ul_tasks

def _patched_torch_safe_load(weight):
    ckpt = torch.load(weight, map_location="cpu", weights_only=False)
    return ckpt, weight

_ul_tasks.torch_safe_load = _patched_torch_safe_load

# ── Constants ────────────────────────────────────────────────────────
OCR_EVERY_N_FRAMES = 5      # Run OCR only every 5th frame
MIN_CONFIDENCE     = 0.45   # Ignore low confidence garbage reads
MIN_TEXT_LEN       = 3      # Ignore single/double char noise
PAD                = 12     # Padding around YOLO box

# CLAHE created once at startup, reused every frame
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def preprocess_crop_for_ocr(crop: np.ndarray) -> np.ndarray:
    """
    Clean up the image before sending to EasyOCR.
    Upscale → grayscale → CLAHE → blur → Otsu threshold.
    """
    h, w = crop.shape[:2]

    # Upscale if crop is too small
    min_height = 120
    if h < min_height:
        scale = min_height / h
        crop = cv2.resize(crop, (max(int(w * scale), 1), min_height),
                          interpolation=cv2.INTER_CUBIC)

    gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray  = _CLAHE.apply(gray)
    gray  = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)


def run_ocr_on_crop(reader: easyocr.Reader, crop: np.ndarray) -> list[str]:
    """
    Run EasyOCR on a preprocessed crop.
    Filters out low confidence and short garbage text.
    """
    processed = preprocess_crop_for_ocr(crop)
    results   = reader.readtext(processed, detail=1)

    output = []
    for (_, text, conf) in results:
        text = text.strip()
        if conf >= MIN_CONFIDENCE and len(text) >= MIN_TEXT_LEN:
            output.append(text)

    return output


def camera_worker():
    """
    Main camera loop. Keep on main thread for OpenCV/Windows compatibility.
    Press Q to quit.
    """
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

    frame_count    = 0
    last_ocr_cache = {}  # key = (x1,y1,x2,y2), value = list of texts

    while state.running:
        ret, frame = cap.read()
        if not ret:
            import time; time.sleep(0.05)
            continue

        frame_count += 1
        run_ocr_this_frame = (frame_count % OCR_EVERY_N_FRAMES == 0)

        # ── YOLO inference ──────────────────────────────────
        results   = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        annotated = results[0].plot()

        all_ocr_text     = []
        frame_seen_texts = set()   # dedupe within this single frame
        current_keys     = set()   # track which boxes are visible this frame

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            box_area = (x2 - x1) * (y2 - y1)
            if box_area < MIN_BOX_AREA:
                continue

            # Padded coords
            px1 = max(0, x1 - PAD)
            py1 = max(0, y1 - PAD)
            px2 = min(frame.shape[1], x2 + PAD)
            py2 = min(frame.shape[0], y2 + PAD)

            box_key = (px1, py1, px2, py2)
            current_keys.add(box_key)

            crop = frame[py1:py2, px1:px2]
            if crop.size == 0:
                continue

            # ── OCR: run on OCR frames, reuse cache on skipped frames ──
            if run_ocr_this_frame:
                texts = run_ocr_on_crop(reader, crop)
                last_ocr_cache[box_key] = texts
            else:
                texts = last_ocr_cache.get(box_key, [])

            # ── Per-frame deduplication and noise filtering ──
            for text in texts:
                text_upper = text.upper().strip()

                if text_upper in frame_seen_texts:
                    continue  # already shown this frame

                frame_seen_texts.add(text_upper)
                all_ocr_text.append(text)
                print(f"[OCR] '{text}'  (box: {box_area}px²)")

                cv2.putText(
                    annotated, text,
                    (px1, max(py1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 255, 255), 2,
                )

        # ── Remove stale cache entries (boxes no longer visible) ──
        if run_ocr_this_frame:
            last_ocr_cache = {
                k: v for k, v in last_ocr_cache.items()
                if k in current_keys
            }

        # ── Route OCR text to priority or LLM ──────────────────
        if all_ocr_text:
            combined = " | ".join(all_ocr_text)
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
        cv2.putText(annotated, f"Custom priorities: {len(state.user_priority_list)}  |  conf>{CONFIDENCE_THRESHOLD}",
                    (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 255), 1)

        cv2.imshow("Vision Assistant", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()