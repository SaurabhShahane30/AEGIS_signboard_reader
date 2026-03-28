"""
main.py — Entry point for Vision Assistant.

Startup sequence:
  1. Verify Ollama is running.
  2. Verify ElevenLabs API key (if USE_ELEVENLABS is True).
  3. Launch daemon threads: TTS → LLM brain → Voice input.
  4. Run camera loop on the main thread (required by OpenCV on Windows).

Usage:
    python main.py

Prerequisites:
    ollama serve          (in a separate terminal)
    ollama pull llama3.1
    pip install ultralytics easyocr opencv-python pyttsx3 \\
                SpeechRecognition pyaudio numpy requests \\
                sounddevice soundfile python-dotenv googletrans==4.0.0rc1
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
from utils.camera import camera_worker
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

    # TTS thread — must start first so other threads can call speak()
    t_tts = threading.Thread(target=tts_worker, daemon=True, name="TTS")
    t_tts.start()
    threads.append(t_tts)
    time.sleep(0.4)   # give TTS engine time to initialise

    speak("Vision assistant starting up.")

    # LLM brain thread
    t_llm = threading.Thread(target=llm_worker, daemon=True, name="LLM")
    t_llm.start()
    threads.append(t_llm)

    # Voice input thread
    t_voice = threading.Thread(target=voice_worker, daemon=True, name="Voice")
    t_voice.start()
    threads.append(t_voice)

    # Camera loop — must run on the main thread (OpenCV / Windows requirement)
    try:
        camera_worker()
    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user.")
    finally:
        end_session(reason="process_exit")
        state.running = False
        print("[Main] Shutting down threads...")
        time.sleep(1)
        print("[Main] Done.")


if __name__ == "__main__":
    main()
