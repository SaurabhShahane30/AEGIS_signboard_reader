"""
camera.py — Camera loop + dedicated OCR worker thread.

Architecture:
  camera_worker()  — YOLO inference → pushes crops into state.crop_queue.
                     Never blocks on OCR; real-time camera loop.
  ocr_worker()     — pops crops → EasyOCR (paragraph mode)
                     → relevance gate → dedup → passive alert or LLM queue.

Priority gate:
  CRITICAL-ONLY mode  (user_priority_list is empty)
      Only text fuzzy-matching CRITICAL_LIST is forwarded.
  PRIORITY+CRITICAL mode  (user_priority_list has >= 1 item)
      Text matching CRITICAL_LIST OR user_priority_list is forwarded.
  Mode switches automatically on ADD / CLEAR voice commands.
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
PAD          = 6    # pixels of padding around each YOLO box
MIN_TEXT_LEN = 2    # minimum OCR text length to consider

# Vote buffer: used by clear_vote_buffer_for_item() in priority.py
# to flush stale entries when the priority list changes.
_vote_buffer: dict = {}   # text_lower -> hit_count


# ── OCR worker ───────────────────────────────────────────────────────

def ocr_worker():
    """
    Dedicated OCR thread — receives (crop, (x1,y1)) from state.crop_queue.

    Pipeline per crop:
      1. EasyOCR readtext  (paragraph=True, detail=0)
      2. Relevance gate    — drop text not matching active list (fuzzy)
      3. Dedup check       — drop text seen within cooldown window
      4. Passive alert     — immediate speak for critical/priority exact matches
      5. LLM queue         — relevant non-critical text forwarded to llm_worker
      6. log_event         — every forwarded text logged to session JSON
    """
    print("[OCR] Initialising EasyOCR...")
    reader = easyocr.Reader(["en"], gpu=True)
    print("[OCR] Ready.")

    while state.running:
        try:
            crop, box_coords, ts = state.crop_queue.get(timeout=0.5)
            if time.time() - ts > 2:
                continue  # Skip old crops to reduce latency
        except queue.Empty:
            continue

        try:
            ocr_results = reader.readtext(crop, detail=0, paragraph=True)

            for text in ocr_results:
                text = text.strip()
                if len(text) < MIN_TEXT_LEN:
                    continue

                print(f"[OCR] '{text}'")

                # ── Relevance gate ────────────────────────────────────
                if not is_relevant_to_active_list(text):
                    continue

                # ── Dedup + routing ───────────────────────────────────
                if not is_duplicate(text):
                    priority_key, passive_alert = build_passive_alert(text)
                    if passive_alert and priority_key \
                            and should_speak_priority_alert(priority_key):
                        speak(passive_alert)
                        log_event("alert",
                                  f"passive alert='{passive_alert}'  ocr='{text}'")
                    elif len(text) > 3:
                        state.ocr_queue.put(text)
                        log_event("detection", f"ocr='{text}'")

        except Exception as e:
            print(f"[OCR Error] {e}")
        finally:
            state.crop_queue.task_done()


# ── Camera worker ─────────────────────────────────────────────────────

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

        # ── YOLO inference ──────────────────────────────────────────
        results   = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        annotated = results[0].plot()

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if (x2 - x1) * (y2 - y1) < MIN_BOX_AREA:
                continue

            x1p = max(0, x1 - PAD)
            y1p = max(0, y1 - PAD)
            x2p = min(frame.shape[1], x2 + PAD)
            y2p = min(frame.shape[0], y2 + PAD)

            crop = frame[y1p:y2p, x1p:x2p]
            if crop.size == 0:
                continue

            try:
                state.crop_queue.put_nowait((crop.copy(), (x1p, y1p), time.time()))
            except queue.Full:
                pass   # ocr_worker is behind — drop, next frame retries

        # ── HUD ──────────────────────────────────────────────────────
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