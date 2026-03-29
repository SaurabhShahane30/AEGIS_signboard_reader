"""
============================================================
  voice.py — Wake-word detection and voice command capture
============================================================
"""

import time
import speech_recognition as sr

from .config import WAKE_WORD, SOURCE_LANGUAGE
from . import state
from .tts import speak

# ── Direct-command trigger phrases ───────────────────────
# These bypass the wake word — if heard, the utterance goes
# straight to the LLM as a command.
#
# Two sets are maintained in parallel:
#   1. English originals
#   2. Hindi — both Devanagari (returned by hi-IN STT) and
#      phonetic/transliterated forms (returned when a Hindi
#      speaker uses en-US STT or mixes languages)
#
# To add another language, append its equivalents below.

COMMAND_TRIGGERS = (
    # ── English ──────────────────────────────────────────
    "add ",
    "remove ",
    "delete ",
    "clear ",
    "reset ",
    "what are my",
    "list my",
    "show my",
    "alert me",
    "notify me",
    "watch for",
    "remember",
    "stop tracking",
    "don't alert",
    "don't track",

    # ── Hindi: ADD intent ────────────────────────────────
    # Devanagari (hi-IN STT output)
    "जोड़ो",
    "जोड़",
    "शामिल करो",
    "ऐड करो",
    # Transliterated / phonetic (en-US STT of Hindi speech)
    "jodo ",
    "add karo",
    "shamil karo",
    "add kar",
    "yaad rakh",
    "dhyan rakh",
    "batao jab",
    "alert karo",
    "notify karo",
    "bata dena",
    "track karo",
    "watch karo",

    # ── Hindi: REMOVE intent ─────────────────────────────
    "हटाओ",
    "हटा दो",
    "रिमूव करो",
    "hatao ",
    "hata do",
    "remove karo",
    "delete karo",
    "band karo",
    "track mat karo",
    "alert mat karo",
    "mat batao",

    # ── Hindi: LIST intent ───────────────────────────────
    "लिस्ट दिखाओ",
    "लिस्ट बताओ",
    "मेरी लिस्ट",
    "list dikhao",
    "list batao",
    "kya hai meri",
    "meri list",
    "priority dikhao",
    "priorities dikhao",
    "kya track",
    "what are meri",
    "meri priorities",

    # ── Hindi: CLEAR intent ──────────────────────────────
    "सब हटाओ",
    "साफ करो",
    "क्लियर करो",
    "saaf karo",
    "clear karo",
    "sab hatao",
    "reset karo",
)


def _looks_like_command(text: str) -> bool:
    t = text.lower()
    return any(t.startswith(trigger) or trigger.strip() in t for trigger in COMMAND_TRIGGERS)


def voice_worker():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    recognizer.energy_threshold         = 300
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold          = 0.6

    print(f"[Voice] Wake word: '{WAKE_WORD.upper()}'")
    print(f"[Voice] STT language: {SOURCE_LANGUAGE}")
    print("[Voice] Calibrating microphone for ambient noise (2 seconds)...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=2)
    print(f"[Voice] Calibration done. Energy threshold: {recognizer.energy_threshold:.0f}")
    print("[Voice] List commands work at any time. Say 'hey system' for anything else.")

    while state.running:
        try:
            with mic as source:
                audio = recognizer.listen(source, timeout=3, phrase_time_limit=8)

            text = recognizer.recognize_google(audio, language=SOURCE_LANGUAGE).lower().strip()
            if not text:
                continue

            print(f"[Voice] Heard: '{text}'")

            # ── Direct command — no wake word needed ─────────────
            if _looks_like_command(text):
                print("[Voice] Direct command detected.")
                state.speech_queue.put(text)

            # ── Wake word — open explicit listening window ────────
            elif WAKE_WORD in text:
                speak("Listening.")
                print("[Voice] Waiting for command...")
                try:
                    with mic as source:
                        cmd_audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)
                    command = recognizer.recognize_google(
                        cmd_audio, language=SOURCE_LANGUAGE
                    ).lower().strip()
                    if command:
                        print(f"[Voice] Wake-word command: '{command}'")
                        state.speech_queue.put(command)
                except sr.WaitTimeoutError:
                    speak("I didn't catch that.")
                except sr.UnknownValueError:
                    speak("Sorry, I couldn't understand.")

        except sr.WaitTimeoutError:
            continue
        except sr.UnknownValueError:
            continue
        except sr.RequestError as e:
            print(f"[Voice Error] Google STT request failed: {e}")
            time.sleep(2)
        except Exception as e:
            print(f"[Voice Error] {e}")
            time.sleep(1)