"""
tts.py — Text-to-speech output layer.

Features:
  - ElevenLabs /stream endpoint (no temp file written to disk)
  - pyttsx3 fallback when ElevenLabs is unavailable or disabled
  - Optional translation before speech (googletrans)
  - speak() helper enqueues text from any thread
  - tts_worker() runs on its own daemon thread
"""

import sys
import subprocess
import requests
import pyttsx3
from . import state
from .config import (
    USE_ELEVENLABS, ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID,
    ELEVENLABS_MODEL_ID, TTS_HTTP_TIMEOUT_SEC,
    TARGET_LANGUAGE, SOURCE_LANGUAGE,
)

try:
    from googletrans import Translator
except ImportError:
    Translator = None


# ── Translation ────────────────────────────────────────────

def _get_translator():
    """Lazily initialise the googletrans Translator (keeps startup fast)."""
    if state.translator_instance is not None:
        return state.translator_instance
    if Translator is None:
        return None
    try:
        state.translator_instance = Translator()
        return state.translator_instance
    except Exception as e:
        print(f"[Translate Error] init failed: {e}")
        return None


def translate_for_speech(text: str) -> str:
    """
    Translate `text` to TARGET_LANGUAGE only at the final output stage.
    Results are cached in state.translation_cache.
    """
    if not text:
        return text
    if TARGET_LANGUAGE.lower() in {"", "en"}:
        return text

    key = (text, TARGET_LANGUAGE.lower())
    with state.lock:
        if key in state.translation_cache:
            return state.translation_cache[key]

    translator = _get_translator()
    if translator is None:
        return text

    try:
        result = translator.translate(text, src=SOURCE_LANGUAGE, dest=TARGET_LANGUAGE.lower())
        translated = result.text if result and result.text else text
        with state.lock:
            state.translation_cache[key] = translated
        return translated
    except Exception as e:
        print(f"[Translate Error] {e}")
        return text


# ── ElevenLabs ─────────────────────────────────────────────

def speak_with_elevenlabs(text: str):
    """
    Synthesise speech via ElevenLabs /stream endpoint.
    Audio bytes are piped directly to the OS audio player — no temp file written.

    Raises RuntimeError or requests.HTTPError on failure (caller falls back to pyttsx3).
    """
    if not ELEVENLABS_API_KEY:
        raise RuntimeError("ELEVENLABS_API_KEY is not set")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY,
    }
    payload = {"text": text, "model_id": ELEVENLABS_MODEL_ID}

    resp = requests.post(
        url, headers=headers, json=payload,
        timeout=TTS_HTTP_TIMEOUT_SEC, stream=True,
    )
    resp.raise_for_status()

    audio_data = b"".join(resp.iter_content(chunk_size=4096))

    if sys.platform == "darwin":
        proc = subprocess.Popen(
            ["afplay", "-"],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        proc.communicate(input=audio_data)

    elif sys.platform.startswith("linux"):
        proc = subprocess.Popen(
            ["ffplay", "-nodisp", "-autoexit", "-i", "pipe:0"],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        proc.communicate(input=audio_data)

    elif sys.platform.startswith("win"):
        try:
            import io
            import sounddevice as sd
            import soundfile as sf
            audio_io = io.BytesIO(audio_data)
            data, samplerate = sf.read(audio_io)
            sd.play(data, samplerate)
            sd.wait()
        except ImportError:
            raise RuntimeError(
                "Windows in-memory playback requires: pip install sounddevice soundfile"
            )

    else:
        raise RuntimeError(f"Unsupported platform for audio playback: {sys.platform}")


# ── pyttsx3 ────────────────────────────────────────────────

def _build_local_tts_engine():
    """Create and configure a pyttsx3 engine. Returns None if unavailable."""
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)
        engine.setProperty("volume", 1.0)
        voices = engine.getProperty("voices")
        for v in voices:
            if "english" in v.name.lower() or "en" in v.id.lower():
                engine.setProperty("voice", v.id)
                break
        return engine
    except Exception as e:
        print(f"[TTS Local Error] engine init failed: {e}")
        return None


# ── Public API ─────────────────────────────────────────────

def speak(text: str):
    """Enqueue `text` for speech output from any thread."""
    state.tts_queue.put(text)


def tts_worker():
    """
    Daemon thread: drains tts_queue, translates if needed, and speaks.
    Tries ElevenLabs first; falls back to pyttsx3 on any error.
    """
    engine = None
    if not USE_ELEVENLABS:
        engine = _build_local_tts_engine()

    while state.running:
        try:
            text = state.tts_queue.get(timeout=0.5)
            final_text = translate_for_speech(text)
            print(f"\n🔊 {final_text}")

            spoken = False

            if USE_ELEVENLABS:
                try:
                    speak_with_elevenlabs(final_text)
                    spoken = True
                except Exception as e:
                    print(f"[TTS ElevenLabs Error] {e} — falling back to pyttsx3")

            if not spoken:
                if engine is None:
                    engine = _build_local_tts_engine()
                if engine is not None:
                    engine.say(final_text)
                    engine.runAndWait()
                    spoken = True

            if not spoken:
                print("[TTS] No available speech backend. Text was not spoken.")

            state.tts_queue.task_done()

        except Exception:
            # queue.Empty on timeout — normal, just loop
            continue
