"""
============================================================
  VISION ASSISTANT — Main Entry Point
  Stack: YOLOv8 (best.pt) + EasyOCR + Ollama (llama3.1) + gTTS
============================================================

SETUP:
    1. Make sure Ollama is running:   ollama serve
    2. Make sure model is pulled:     ollama pull llama3.1
    3. Install Python deps:
         pip install ultralytics easyocr opencv-python pyttsx3 \
                     SpeechRecognition pyaudio numpy requests \
                     sounddevice soundfile gtts
    4. Set MODEL_PATH in config.py to your best.pt location.
    5. Run:  python main.py

VOICE COMMANDS (say "Hey system" first):
    "add wet floor to priority"
    "remove stop from priority"
    "what are my priorities?"
    "clear user list"
"""

import sys
import time
import threading
import os
import requests

from utils.config import ALERT_SNAPSHOT_DIR, OLLAMA_URL
from utils.state import running, user_priority_list
from utils.tts import tts_worker, speak
from utils.llm import llm_worker
from utils.voice import voice_worker
from utils.ocr import ocr_worker
from utils.camera import camera_worker


def main():
    import utils.state as state
    state.running = True
    state.user_priority_list = set()  # always start empty

    print("=" * 60)
    print("  VISION ASSISTANT  |  Ollama llama3.1  |  YOLOv8 + OCR")
    print("=" * 60)

    os.makedirs(ALERT_SNAPSHOT_DIR, exist_ok=True)
    print(f"[Init] Alert snapshots will be saved to: {os.path.abspath(ALERT_SNAPSHOT_DIR)}/")

    # ── Ollama connectivity check ──
    print("[Check] Pinging Ollama...")
    try:
        requests.get(OLLAMA_URL.replace("/api/generate", ""), timeout=3)
        print("[Check] Ollama is running ✓")
    except Exception:
        print("\n[ERROR] Ollama is not running!")
        print("        Open a NEW terminal and run:  ollama serve")
        print("        Then re-run this script.\n")
        sys.exit(1)

    # ── gTTS check ──
    print("[Check] Verifying gTTS is available...")
    try:
        from gtts import gTTS  # noqa: F401
        print("[Check] gTTS is available ✓")
    except ImportError:
        print("[ERROR] gTTS is not installed!")
        print("        Run:  pip install gtts")
        sys.exit(1)

    threads = []

    t_tts = threading.Thread(target=tts_worker, daemon=True, name="TTS")
    t_tts.start()
    threads.append(t_tts)
    time.sleep(0.4)

    speak("Vision assistant starting up.")

    t_llm = threading.Thread(target=llm_worker, daemon=True, name="LLM")
    t_llm.start()
    threads.append(t_llm)

    t_voice = threading.Thread(target=voice_worker, daemon=True, name="Voice")
    t_voice.start()
    threads.append(t_voice)

    t_ocr = threading.Thread(target=ocr_worker, daemon=True, name="OCR")
    t_ocr.start()
    threads.append(t_ocr)

    # Camera runs on the main thread (required by OpenCV on Windows)
    try:
        camera_worker()
    except KeyboardInterrupt:
        print("\n[Main] Interrupted.")
    finally:
        state.running = False
        print("[Main] Shutting down...")
        time.sleep(1)
        print("[Main] Done.")


if __name__ == "__main__":
    main()