"""
============================================================
  VISION ASSISTANT — Complete System
  Stack: YOLOv8 (best.pt) + EasyOCR + Ollama (llama3.1) + pyttsx3
  Author: Built for you
============================================================

SETUP:
    1. Make sure Ollama is running:   ollama serve
    2. Make sure model is pulled:     ollama pull llama3.1
    3. Install Python deps:
         pip install ultralytics easyocr opencv-python pyttsx3 \
                     SpeechRecognition pyaudio numpy requests
    4. Set MODEL_PATH below to your best.pt location.
    5. Run:  python vision_assistant.py

VOICE COMMANDS (say "Hey system" first):
    "add wet floor to priority"
    "remove stop from priority"
    "what are my priorities?"
    "clear user list"
============================================================
"""

import cv2
import threading
import time
import queue
import sys
import json
import requests
import numpy as np
import pyttsx3
import speech_recognition as sr
import easyocr
from collections import deque
from datetime import datetime

# ── PyTorch 2.6 fix (must be before YOLO import) ───────────
import torch
from ultralytics.nn import tasks as _ul_tasks

def _patched_torch_safe_load(weight):
    ckpt = torch.load(weight, map_location="cpu", weights_only=False)
    return ckpt, weight

_ul_tasks.torch_safe_load = _patched_torch_safe_load
# ────────────────────────────────────────────────────────────

from ultralytics import YOLO

# ─────────────────────────────────────────────
#  CONFIG — Edit these
# ─────────────────────────────────────────────
MODEL_PATH           = "signboard_project/best.pt"  # your YOLO model
CAMERA_INDEX         = 0               # 0 = default webcam
WAKE_WORD            = "hello system"    # voice trigger phrase
OCR_COOLDOWN_SEC     = 4              # seconds before same text re-triggers
LLM_RATE_LIMIT_SEC   = 2              # min gap between LLM calls
CONFIDENCE_THRESHOLD = 0.5            # raise this if too many false detections
MIN_BOX_AREA         = 3000           # pixels² — ignore tiny/noisy boxes
OLLAMA_URL           = "http://localhost:11434/api/generate"
OLLAMA_MODEL         = "llama3.1:8b"

# ─────────────────────────────────────────────
#  PRIORITY LISTS
# ─────────────────────────────────────────────
CRITICAL_LIST = {
    "stop", "danger", "no entry", "wet floor", "slippery road",
    "hazard", "warning", "emergency", "fire", "poison",
    "explosive", "high voltage", "do not enter", "keep out",
    "restricted", "caution", "alert", "no exit", "blocked"
}

user_priority_list = set()   # grows/shrinks via voice commands

# ─────────────────────────────────────────────
#  SHARED STATE
# ─────────────────────────────────────────────
ocr_queue    = queue.Queue()
speech_queue = queue.Queue()
tts_queue    = queue.Queue()
recent_texts = deque(maxlen=30)
last_llm_call = 0
lock         = threading.Lock()
running      = True

# ─────────────────────────────────────────────
#  OLLAMA CALLER
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are a vision assistant embedded in a real-time camera system.

ROLE 1 — ACTIVE WATCHER:
When given OCR text from a camera frame, decide if it needs an alert.
- If it matches or is close to the critical list or user priority list -> respond: ALERT: <short message>
- Otherwise -> respond: IGNORE
Keep alerts under 15 words. Be calm, not robotic.

ROLE 2 — COMMAND PARSER:
When given a voice command, respond with exactly one of:
  ADD: <item>
  REMOVE: <item>
  LIST:
  CLEAR:
  UNKNOWN: <helpful reply>

No markdown. No bullet points. Plain text only. One line response."""


def call_ollama(prompt: str) -> str:
    """Call local Ollama and return response text."""
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}",
            "stream": False,
            "options": {
                "temperature": 0.1,   # low temp = consistent structured responses
                "num_predict": 60     # short responses only
            }
        }
        resp = requests.post(OLLAMA_URL, json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json().get("response", "IGNORE").strip()
    except requests.exceptions.ConnectionError:
        print("[Ollama] ERROR: Cannot connect. Is 'ollama serve' running?")
        return "IGNORE"
    except Exception as e:
        print(f"[Ollama Error] {e}")
        return "IGNORE"


# ─────────────────────────────────────────────
#  TTS THREAD
# ─────────────────────────────────────────────
def tts_worker():
    engine = pyttsx3.init()
    engine.setProperty("rate", 160)
    engine.setProperty("volume", 1.0)

    voices = engine.getProperty("voices")
    for v in voices:
        if "english" in v.name.lower() or "en" in v.id.lower():
            engine.setProperty("voice", v.id)
            break

    while running:
        try:
            text = tts_queue.get(timeout=0.5)
            print(f"\n🔊 {text}")
            engine.say(text)
            engine.runAndWait()
            tts_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[TTS Error] {e}")


def speak(text: str):
    tts_queue.put(text)


# ─────────────────────────────────────────────
#  PRIORITY HELPERS
# ─────────────────────────────────────────────
def is_priority(text: str) -> bool:
    """Fast local check — avoids calling LLM for irrelevant text."""
    t = text.lower()
    for item in CRITICAL_LIST | user_priority_list:
        if item.lower() in t:
            return True
    return False


def priority_context() -> str:
    return (
        f"Critical list: {', '.join(sorted(CRITICAL_LIST))}\n"
        f"User priority list: {', '.join(sorted(user_priority_list)) or 'empty'}"
    )


def is_duplicate(text: str) -> bool:
    now = time.time()
    t = text.lower().strip()
    with lock:
        for (rt, ts) in recent_texts:
            if rt == t and (now - ts) < OCR_COOLDOWN_SEC:
                return True
        recent_texts.append((t, now))
    return False


# ─────────────────────────────────────────────
#  LLM BRAIN THREAD
# ─────────────────────────────────────────────
def llm_worker():
    global last_llm_call, user_priority_list

    while running:
        # Voice commands = higher priority
        try:
            command = speech_queue.get_nowait()
            print(f"\n🎤 Command: '{command}'")
            prompt = f'VOICE COMMAND from user: "{command}"\n{priority_context()}'
            response = call_ollama(prompt)
            print(f"[LLM -> Command] {response}")
            _handle_command_response(response)
            speech_queue.task_done()
            continue
        except queue.Empty:
            pass

        # OCR alerts
        try:
            ocr_text = ocr_queue.get(timeout=0.3)
            now = time.time()

            if (now - last_llm_call) < LLM_RATE_LIMIT_SEC:
                ocr_queue.task_done()
                continue

            last_llm_call = now
            prompt = (
                f'OCR TEXT from camera: "{ocr_text}"\n'
                f"{priority_context()}\n"
                f"Should I alert the user?"
            )
            response = call_ollama(prompt)
            print(f"[LLM -> OCR] {response}")

            if response.upper().startswith("ALERT:"):
                speak(response[6:].strip())

            ocr_queue.task_done()

        except queue.Empty:
            continue


def _handle_command_response(response: str):
    global user_priority_list
    upper = response.upper()

    if upper.startswith("ADD:"):
        item = response[4:].strip().lower()
        user_priority_list.add(item)
        speak(f"Added {item} to your priority list.")
        print(f"[Priority] User list: {user_priority_list}")

    elif upper.startswith("REMOVE:"):
        item = response[7:].strip().lower()
        if item in user_priority_list:
            user_priority_list.discard(item)
            speak(f"Removed {item} from your priority list.")
        else:
            speak(f"{item} was not in your list.")

    elif upper.startswith("LIST:"):
        if user_priority_list:
            speak(f"Your priorities are: {', '.join(sorted(user_priority_list))}")
        else:
            speak("Your custom priority list is empty.")

    elif upper.startswith("CLEAR:"):
        user_priority_list.clear()
        speak("Custom priority list cleared.")

    elif upper.startswith("UNKNOWN:"):
        speak(response[8:].strip())

    else:
        speak(response)


# ─────────────────────────────────────────────
#  VOICE INPUT THREAD
# ─────────────────────────────────────────────
def voice_worker():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    print(f"[Voice] Wake word: '{WAKE_WORD.upper()}'")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)

    while running:
        try:
            with mic as source:
                audio = recognizer.listen(source, timeout=3, phrase_time_limit=6)

            text = recognizer.recognize_google(audio).lower()
            print(f"[Voice] Heard: '{text}'")

            if WAKE_WORD in text:
                speak("Listening.")
                print("[Voice] Waiting for command...")
                with mic as source:
                    cmd_audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)
                command = recognizer.recognize_google(cmd_audio).lower()
                print(f"[Voice] Command: '{command}'")
                speech_queue.put(command)

        except sr.WaitTimeoutError:
            continue
        except sr.UnknownValueError:
            continue
        except sr.RequestError as e:
            print(f"[Voice Error] {e}")
            time.sleep(2)
        except Exception as e:
            print(f"[Voice Error] {e}")
            time.sleep(1)


# ─────────────────────────────────────────────
#  CAMERA + YOLO + OCR
# ─────────────────────────────────────────────
def camera_worker():
    print("[Camera] Loading YOLO model...")
    model = YOLO(MODEL_PATH)

    print("[OCR] Initializing EasyOCR...")
    reader = easyocr.Reader(["en"], gpu=True)  # set gpu=False if no CUDA

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {CAMERA_INDEX}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[Camera] Running. Press Q to quit.\n")
    speak("System ready. Say hello system to give a command.")

    while running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        # ── YOLO ──
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        annotated = results[0].plot()
        all_ocr_text = []

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # ── NOISE FILTER: skip small boxes ──
            box_area = (x2 - x1) * (y2 - y1)
            if box_area < MIN_BOX_AREA:
                continue

            # Padding
            pad = 6
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(frame.shape[1], x2 + pad)
            y2 = min(frame.shape[0], y2 + pad)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # ── EasyOCR ──
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
                    (0, 255, 255), 2
                )

        # Queue for LLM
        if all_ocr_text:
            combined = " | ".join(all_ocr_text)
            if not is_duplicate(combined):
                if is_priority(combined) or len(combined) > 3:
                    ocr_queue.put(combined)

        # ── HUD ──
        ts = datetime.now().strftime("%H:%M:%S")
        cv2.putText(annotated, f"Vision Assistant  {ts}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated, f"Say: '{WAKE_WORD}' + command",
                    (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1)
        cv2.putText(annotated, f"Custom priorities: {len(user_priority_list)}  |  conf>{CONFIDENCE_THRESHOLD}",
                    (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 255), 1)

        cv2.imshow("Vision Assistant", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    global running

    print("=" * 60)
    print("  VISION ASSISTANT  |  Ollama llama3.1  |  YOLOv8 + OCR")
    print("=" * 60)

    # Ollama connectivity check
    print("[Check] Pinging Ollama...")
    try:
        requests.get("http://localhost:11434", timeout=3)
        print("[Check] Ollama is running ✓")
    except Exception:
        print("\n[ERROR] Ollama is not running!")
        print("        Open a NEW terminal and run:  ollama serve")
        print("        Then re-run this script.\n")
        sys.exit(1)

    threads = []

    # TTS first
    t_tts = threading.Thread(target=tts_worker, daemon=True, name="TTS")
    t_tts.start()
    threads.append(t_tts)
    time.sleep(0.4)

    speak("Vision assistant starting up.")

    # LLM brain
    t_llm = threading.Thread(target=llm_worker, daemon=True, name="LLM")
    t_llm.start()
    threads.append(t_llm)

    # Voice
    t_voice = threading.Thread(target=voice_worker, daemon=True, name="Voice")
    t_voice.start()
    threads.append(t_voice)

    # Camera on main thread (required by OpenCV on Windows)
    try:
        camera_worker()
    except KeyboardInterrupt:
        print("\n[Main] Interrupted.")
    finally:
        running = False
        print("[Main] Shutting down...")
        time.sleep(1)
        print("[Main] Done.")


if __name__ == "__main__":
    main()