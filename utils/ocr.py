"""
============================================================
  ocr_worker.py — Dedicated EasyOCR thread
============================================================
"""

import queue
import cv2
import numpy as np
import easyocr

from . import state
from .direction import get_direction
from .scene import record_signboard
from .priority import is_duplicate, build_passive_alert, should_speak_priority_alert
from .snapshot import save_alert_snapshot
from .tts import speak


# ─────────────────────────────────────────────────────────
#  Crop preprocessing
# ─────────────────────────────────────────────────────────

def preprocess_crop(crop: np.ndarray) -> np.ndarray:
    gray     = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
    return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)


# ─────────────────────────────────────────────────────────
#  OCR worker
# ─────────────────────────────────────────────────────────

def ocr_worker():
    print("[OCR] Initializing EasyOCR...")
    reader = easyocr.Reader(["en"], gpu=True)
    print("[OCR] Ready.")
 
    while state.running:
        try:
            item = state.crop_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        try:
            crop, box_coords, annotated_frame = item
            x1, y1, x2, y2 = box_coords

            frame_h, frame_w = annotated_frame.shape[:2]
            direction = get_direction(x1, x2, frame_w)

            ocr_results = reader.readtext(crop, detail=0, paragraph=True)

            for text in ocr_results:
                text = text.strip()
                if len(text) < 2:
                    continue

                print(f"[OCR] '{text}'  →  {direction}")

                if is_duplicate(text):
                    continue

                frame_h, frame_w = annotated_frame.shape[:2]
                record_signboard(text, direction, x1, y1, x2, y2, frame_w, frame_h)

                priority_key, passive_alert = build_passive_alert(text, direction)

                if passive_alert and priority_key:
                    if should_speak_priority_alert(priority_key):
                        save_alert_snapshot(
                            annotated_frame,
                            (x1, y1, x2, y2),
                            text,
                            direction,
                        )
                        speak(passive_alert)

                        # One-time detection: remove from list after speaking
                        if priority_key in state.user_priority_list:
                            state.user_priority_list.discard(priority_key)
                            print(f"[Priority] '{priority_key}' detected once — removed from list.")

                # non-priority text: drop silently, no LLM call

        except Exception as e:
            print(f"[OCR Error] {e}")
        finally:
            state.crop_queue.task_done()