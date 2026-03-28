"""
============================================================
  tts.py — Text-to-speech thread (gTTS primary, pyttsx3 fallback)
           + optional translation
============================================================
"""

import sys
import io
import queue
import subprocess
from datetime import datetime

from .config import TARGET_LANGUAGE, SOURCE_LANGUAGE
from . import state
from .websocket_server import broadcast_event   

try:
    from googletrans import Translator
except Exception:
    Translator = None

# ─────────────────────────────────────────────────────────
#  Translation
# ─────────────────────────────────────────────────────────

def get_translator():
    """Lazily initialize translator so startup stays fast and robust."""
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
    """Translate only at the final speech stage to avoid affecting detection logic."""
    if not text:
        return text
    if TARGET_LANGUAGE.lower() in {"", "en"}:
        return text

    key = (text, TARGET_LANGUAGE.lower())
    with state.lock:
        if key in state.translation_cache:
            return state.translation_cache[key]

    translator = get_translator()
    if translator is None:
        return text

    try:
        translated = translator.translate(
            text,
            src=SOURCE_LANGUAGE,
            dest=TARGET_LANGUAGE.lower(),
        ).text
        if translated:
            with state.lock:
                state.translation_cache[key] = translated
            return translated
        return text
    except Exception as e:
        print(f"[Translate Error] {e}")
        return text

# ─────────────────────────────────────────────────────────
#  pyttsx3 fallback engine
# ─────────────────────────────────────────────────────────

def build_local_tts_engine():
    """Create pyttsx3 engine; return None if unavailable."""
    try:
        import pyttsx3
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


# ─────────────────────────────────────────────────────────
#  gTTS playback (in-memory, no temp file)
# ─────────────────────────────────────────────────────────

def speak_with_gtts(text: str, lang: str = "en"):
    """
    Synthesize speech with gTTS and play it in memory.
    Audio bytes are piped directly to the platform audio player.
    """
    from gtts import gTTS

    tts = gTTS(text=text, lang=lang, slow=False)
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_data = audio_buffer.getvalue()

    if sys.platform == "darwin":
        proc = subprocess.Popen(
            ["afplay", "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        proc.communicate(input=audio_data)

    elif sys.platform.startswith("linux"):
        proc = subprocess.Popen(
            ["ffplay", "-nodisp", "-autoexit", "-i", "pipe:0"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        proc.communicate(input=audio_data)

    elif sys.platform.startswith("win"):
        try:
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

# ─────────────────────────────────────────────────────────
#  TTS thread
# ─────────────────────────────────────────────────────────

def tts_worker():
    engine = None

    while state.running:
        try:
            text = state.tts_queue.get(timeout=0.5)

            final_text = translate_for_speech(text)
            gtts_lang  = TARGET_LANGUAGE.lower() if TARGET_LANGUAGE else "en"
            print(f"\n🔊 {final_text}")

            # ── Broadcast alert text to Flutter ───────────────────
            try:
                broadcast_event({
                    "type":      "alert",
                    "text":      final_text,
                    "timestamp": datetime.now().isoformat(),
                })
            except Exception as e:
                print(f"[TTS WS Error] {e}")
 
            spoken = False
            
            # Primary: gTTS
            try:
                speak_with_gtts(final_text, lang=gtts_lang)
                spoken = True
            except Exception as e:
                print(f"[TTS gTTS Error] {e} | falling back to pyttsx3")

            # Fallback: pyttsx3
            if not spoken:
                if engine is None:
                    engine = build_local_tts_engine()
                if engine is not None:
                    engine.say(final_text)
                    engine.runAndWait()
                    spoken = True

            if not spoken:
                print("[TTS] No available speech backend. Text was not spoken.")

            state.tts_queue.task_done()

        except queue.Empty:
            continue
        except Exception as e:
            print(f"[TTS Error] {e}")


def speak(text: str):
    state.tts_queue.put(text)

