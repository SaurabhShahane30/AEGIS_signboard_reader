"""
camera.py — Camera loop: YOLO detection → EasyOCR → priority routing.

Flow per frame:
  1. YOLO detects sign bounding boxes above MIN_BOX_AREA.
  2. EasyOCR reads text from each cropped box.
  3. Combined OCR text is deduplicated (is_duplicate).
  4. If it matches a priority term → immediate passive alert (build_passive_alert).
  5. Otherwise → pushed to ocr_queue for the LLM worker to evaluate.
  6. Annotated frame with HUD overlay is displayed via OpenCV.

Notes:
  - cv2.VideoCapture must stay on the main thread on Windows (CAP_DSHOW).
  - Press Q to quit.
"""

import cv2
import easyocr
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

# ── PyTorch 2.6 compatibility fix (must run before YOLO loads) ──
import torch
from ultralytics.nn import tasks as _ul_tasks

def _patched_torch_safe_load(weight):
    ckpt = torch.load(weight, map_location="cpu", weights_only=False)
    return ckpt, weight

_ul_tasks.torch_safe_load = _patched_torch_safe_load
# ────────────────────────────────────────────────────────────────


def camera_worker():
    """
    Main camera loop. Runs on the calling thread (keep on main thread for
    OpenCV/Windows compatibility). Exits cleanly on 'Q' keypress.
    """
    print("[Camera] Loading YOLO model...")
    model = YOLO(MODEL_PATH)

    print("[OCR] Initialising EasyOCR...")
    reader = easyocr.Reader(["en"], gpu=True)   # set gpu=False if no CUDA

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
            import time; time.sleep(0.05)
            continue

        # ── YOLO inference ──────────────────────────────────
        results  = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        annotated = results[0].plot()
        all_ocr_text = []

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Skip tiny / noisy boxes
            box_area = (x2 - x1) * (y2 - y1)
            if box_area < MIN_BOX_AREA:
                continue

            # Pad box slightly
            pad = 6
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(frame.shape[1], x2 + pad)
            y2 = min(frame.shape[0], y2 + pad)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # ── EasyOCR ──────────────────────────────────
            ocr_results = reader.readtext(crop, detail=0, paragraph=True)
            for text in ocr_results:
                text = text.strip()
                if len(text) < 2:
                    continue

                all_ocr_text.append(text)
                print(f"[OCR] '{text}'  (box: {box_area}px²)")

                cv2.putText(
                    annotated, text,
                    (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 255, 255), 2,
                )

        # ── Route OCR text ──────────────────────────────────
        if all_ocr_text:
            combined = " | ".join(all_ocr_text)
            if not is_duplicate(combined):
                priority_key, passive_alert = build_passive_alert(combined)
                if passive_alert and priority_key and should_speak_priority_alert(priority_key):
                    speak(passive_alert)
                elif len(combined) > 3:
                    # Non-priority text: let LLM decide
                    state.ocr_queue.put(combined)

        # ── HUD overlay ────────────────────────────────────
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
