"""
main.py — Entry point for Vision Assistant.

Thread startup order:
  1. TTS      — must be first so all other threads can call speak()
  2. LLM      — processes voice commands and OCR alerts
  3. Voice    — listens for wake word and commands
  4. OCR      — runs EasyOCR on crops from camera (separate thread)
  5. Camera   — YOLO inference + pushes crops (runs on main thread for OpenCV)

Session lifecycle:
  - start_session() called before threads start
  - end_session() called on exit (user command or KeyboardInterrupt)
"""

import sys
import time
import threading
import requests

from utils import state
from utils.config import USE_ELEVENLABS, ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID, ELEVENLABS_MODEL_ID
from utils.tts import tts_worker, speak
from utils.llm import llm_worker
from utils.voice import voice_worker
from utils.camera import camera_worker, ocr_worker
from utils.session import start_session, end_session


# ── Pre-flight checks ──────────────────────────────────────

def check_ollama():
    print("[Check] Pinging Ollama...")
    try:
        requests.get("http://localhost:11434", timeout=3)
        print("[Check] Ollama is running ✓")
    except Exception:
        print("\n[ERROR] Ollama is not running!")
        print("        Open a NEW terminal and run:  ollama serve")
        print("        Then re-run this script.\n")
        sys.exit(1)


def check_elevenlabs():
    print("[Check] Testing ElevenLabs API key...")
    if not ELEVENLABS_API_KEY:
        print("[ERROR] ELEVENLABS_API_KEY is not set in your .env file!")
        sys.exit(1)

    try:
        test_resp = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}",
            headers={
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": ELEVENLABS_API_KEY,
            },
            json={"text": "ok", "model_id": ELEVENLABS_MODEL_ID},
            timeout=10,
        )
        if test_resp.status_code == 200:
            print("[Check] ElevenLabs connected ✓")
        elif test_resp.status_code == 401:
            print(f"[ERROR] ElevenLabs rejected API key: {test_resp.text}")
            sys.exit(1)
        else:
            print(f"[WARN] ElevenLabs status {test_resp.status_code}: {test_resp.text}")

    except Exception as e:
        print(f"[ERROR] ElevenLabs unreachable: {e}")
        sys.exit(1)


# ── Entry point ────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  VISION ASSISTANT  |  Ollama llama3.1  |  YOLOv8 + OCR")
    print("=" * 60)

    check_ollama()

    if USE_ELEVENLABS:
        check_elevenlabs()

    start_session()

    threads = []

    # 1. TTS first — every other thread may call speak()
    t_tts = threading.Thread(target=tts_worker, daemon=True, name="TTS")
    t_tts.start()
    threads.append(t_tts)
    time.sleep(0.4)   # give TTS engine a moment to initialise

    speak("Vision assistant starting up.")

    # 2. LLM brain
    t_llm = threading.Thread(target=llm_worker, daemon=True, name="LLM")
    t_llm.start()
    threads.append(t_llm)

    # 3. Voice input
    t_voice = threading.Thread(target=voice_worker, daemon=True, name="Voice")
    t_voice.start()
    threads.append(t_voice)

    # 4. Dedicated OCR worker (receives crops from camera via state.crop_queue)
    t_ocr = threading.Thread(target=ocr_worker, daemon=True, name="OCR")
    t_ocr.start()
    threads.append(t_ocr)

    # 5. Camera on main thread (required by OpenCV on most platforms)
    try:
        camera_worker()
    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user.")
    finally:
        state.running = False
        end_session(reason="process_exit")
        print("[Main] Shutting down threads...")
        time.sleep(1)
        print("[Main] Done.")

if __name__ == "__main__":
    main()