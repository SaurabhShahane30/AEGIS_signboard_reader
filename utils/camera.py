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
import base64
import queue as _queue

from . import state
from .config import (
    MODEL_PATH, CAMERA_INDEX, CONFIDENCE_THRESHOLD,
    MIN_BOX_AREA, WAKE_WORD, ALERT_SNAPSHOT_DIR, INPUT_SOURCE, STREAM_FPS, JPEG_QUALITY
)
from .tts import speak
from .websocket_server import broadcast_event

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

def _run_yolo_and_queue_crops(model, frame):
    """
    Run YOLO on a frame, push crops to crop_queue, return
    (annotated_frame, detections_list).
    """
    results   = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
    annotated = results[0].plot()
    detections = []
 
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if (x2 - x1) * (y2 - y1) < MIN_BOX_AREA:
            continue
 
        conf  = float(box.conf[0])
        cls   = int(box.cls[0])
        label = model.names.get(cls, str(cls))
        detections.append({
            "label":      label,
            "confidence": round(conf, 2),
            "box":        [x1, y1, x2, y2],
        })
 
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
            pass
 
    return annotated, detections
 
 
def _draw_hud(annotated):
    """Draw timestamp / info overlay on the annotated frame."""
    ts = datetime.now().strftime("%H:%M:%S")
    snap_count = (
        len(os.listdir(ALERT_SNAPSHOT_DIR))
        if os.path.isdir(ALERT_SNAPSHOT_DIR) else 0
    )
    cv2.putText(annotated, f"Vision Assistant  {ts}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(annotated, f"Say: '{WAKE_WORD}' + command",
                (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1)
    cv2.putText(
        annotated,
        f"Custom priorities: {len(state.user_priority_list)}  |  conf>{CONFIDENCE_THRESHOLD}",
        (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 255), 1,
    )
    cv2.putText(annotated, f"Snapshots saved: {snap_count}",
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 120), 1)
    return annotated
 
 
def _broadcast_frame(annotated, detections):
    """Encode annotated frame as JPEG and broadcast to Flutter."""
    _, buffer = cv2.imencode(
        ".jpg", annotated,
        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
    )
    frame_b64 = base64.b64encode(buffer).decode("utf-8")
    broadcast_event({
        "type":       "frame",
        "data":       frame_b64,
        "detections": detections,
    })
 
 
def camera_worker():
    """
    Camera thread: capture → YOLO → push crops to crop_queue.
 
    INPUT_SOURCE = "local"   → reads from laptop webcam
    INPUT_SOURCE = "flutter" → reads frames from ws_frame_queue
                               (filled by websocket_server when Flutter sends frames)
 
    In both cases the annotated frame is:
      - shown in a local cv2 window  (local source only)
      - broadcast to Flutter over WS (always)
    """
    print("[Camera] Loading YOLO model...")
    model = YOLO(MODEL_PATH)
 
    cap = None
    if INPUT_SOURCE == "local":
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            print("[Camera] Retrying with DirectShow backend...")
            cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open camera {CAMERA_INDEX}")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print("[Camera] Local webcam open.")
    else:
        print("[Camera] Flutter source — waiting for frames over WebSocket.")
 
    print("[Camera] Running. Press Q to quit (local mode only).\n")
    speak("System ready. Say hey system to give a command.")
 
    last_stream_time = 0
    stream_interval  = 1.0 / STREAM_FPS
 
    while state.running:
 
        # ── Get next frame ──────────────────────────────────────
        if INPUT_SOURCE == "local":
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue
        else:
            try:
                frame = state.ws_frame_queue.get(timeout=0.5)
            except _queue.Empty:
                continue
 
        # ── YOLO + crop queue ───────────────────────────────────
        annotated, detections = _run_yolo_and_queue_crops(model, frame)
 
        # ── HUD overlay ────────────────────────────────────────
        annotated = _draw_hud(annotated)
 
        # ── Broadcast to Flutter (rate-limited) ─────────────────
        now = time.time()
        if now - last_stream_time >= stream_interval:
            last_stream_time = now
            _broadcast_frame(annotated, detections)
 
        # ── Local window (only in local mode) ───────────────────
        if INPUT_SOURCE == "local":
            cv2.imshow("Vision Assistant", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
 
    if cap is not None:
        cap.release()
    if INPUT_SOURCE == "local":
        cv2.destroyAllWindows()
    print("[Camera] Stopped.")