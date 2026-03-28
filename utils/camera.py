"""
============================================================
  camera.py — Camera capture, YOLO inference, and HUD overlay
============================================================

PyTorch 2.6 safe-load patch is applied here before YOLO import.
"""

import os
import time
import cv2
from datetime import datetime

from .config import (
    MODEL_PATH, CAMERA_INDEX, CONFIDENCE_THRESHOLD,
    MIN_BOX_AREA, WAKE_WORD, ALERT_SNAPSHOT_DIR,
)
from . import state
from .tts import speak

# ── PyTorch 2.6 fix (must precede YOLO import) ────────────
import torch
from ultralytics.nn import tasks as _ul_tasks

def _patched_torch_safe_load(weight):
    ckpt = torch.load(weight, map_location="cpu", weights_only=False)
    return ckpt, weight

_ul_tasks.torch_safe_load = _patched_torch_safe_load
# ──────────────────────────────────────────────────────────

from ultralytics import YOLO

import PIL.Image
if not hasattr(PIL.Image, "ANTIALIAS"):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

def camera_worker():
    """
    Camera thread: capture → YOLO → push crops to crop_queue.
    crop_queue items carry (crop, box_coords, annotated_frame).
    """
    print("[Camera] Loading YOLO model...")
    model = YOLO(MODEL_PATH)
 
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[Camera] Retrying with DirectShow backend...")
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
 
        # ── YOLO inference ──────────────────────────────────────
        results   = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        annotated = results[0].plot()
 
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
 
            if (x2 - x1) * (y2 - y1) < MIN_BOX_AREA:
                continue
 
            pad = 6
            x1p = max(0, x1 - pad)
            y1p = max(0, y1 - pad)
            x2p = min(frame.shape[1], x2 + pad)
            y2p = min(frame.shape[0], y2 + pad)
 
            crop = frame[y1p:y2p, x1p:x2p]
            if crop.size == 0:
                continue
 
            try:
                state.crop_queue.put_nowait(
                    (crop.copy(), (x1, y1, x2, y2), annotated.copy())
                )
            except Exception:
                pass  # queue full — drop and continue
 
        # ── HUD ────────────────────────────────────────────────
        ts = datetime.now().strftime("%H:%M:%S")
        cv2.putText(annotated, f"Vision Assistant  {ts}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated, f"Say: '{WAKE_WORD}' + command",
                    (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1)
        cv2.putText(
            annotated,
            f"Custom priorities: {len(state.user_priority_list)}  |  conf>{CONFIDENCE_THRESHOLD}",
            (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 255), 1,
        )
        snap_count = (
            len(os.listdir(ALERT_SNAPSHOT_DIR))
            if os.path.isdir(ALERT_SNAPSHOT_DIR) else 0
        )
        cv2.putText(annotated, f"Snapshots saved: {snap_count}",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 120), 1)
 
        cv2.imshow("Vision Assistant", annotated)
 
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
 
    cap.release()
    cv2.destroyAllWindows()