"""
============================================================
  tts.py — Text-to-speech thread (gTTS primary, pyttsx3 fallback)
           + optional translation via deep-translator
============================================================

Translation dependency:
    pip install deep-translator

deep-translator is used instead of googletrans because googletrans 4.x
switched to an async API (.translate() returns a coroutine), which
breaks synchronous code. deep-translator is synchronous, actively
maintained, and supports the same language codes.

Language is read from state at call-time (state.target_language /
state.source_language) so it can be changed at runtime without restart.
"""

import sys
import io
import queue
import subprocess
from datetime import datetime

from . import state
from .websocket_server import broadcast_event


# ─────────────────────────────────────────────────────────
#  Translation  (deep-translator — sync, no coroutine issues)
# ─────────────────────────────────────────────────────────

try:
    from deep_translator import GoogleTranslator as _GTrans
    _TRANSLATOR_AVAILABLE = True
except ImportError:
    _TRANSLATOR_AVAILABLE = False
    print(
        "[TTS] deep-translator not installed. Translation disabled.\n"
        "      Run:  pip install deep-translator"
    )


def translate_for_speech(text: str) -> str:
    """
    Translate text into the current target language.

    Language is read from state at call-time so runtime changes
    (e.g. a voice command switching language) take effect immediately
    without restarting the process.

    src is always 'en' because:
      - All LLM responses are in English.
      - All hard-coded system strings ("Listening.", "Added X...") are English.
      - state.source_language controls STT input, not TTS output.

    The region tag is stripped before use so "hi-IN" -> "hi", which is
    what both deep-translator and gTTS expect.
    """
    if not text:
        return text

    # Read from state at call-time for runtime language switching
    target = (getattr(state, "target_language", None) or "en").lower().strip()
    tgt_base = target.split("-")[0]   # "hi-IN" -> "hi"

    if tgt_base in {"", "en"}:
        return text                   # no translation needed

    # Cache lookup
    key = (text, tgt_base)
    with state.lock:
        if key in state.translation_cache:
            return state.translation_cache[key]

    if not _TRANSLATOR_AVAILABLE:
        return text

    try:
        translated = _GTrans(source="en", target=tgt_base).translate(text)
        if translated and translated.strip():
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

            # Read target_language from state at call-time so runtime
            # language changes apply to the very next utterance.
            tgt = (getattr(state, "target_language", None) or "en").lower().strip()
            gtts_lang = tgt.split("-")[0] or "en"   # strip region tag for gTTS

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