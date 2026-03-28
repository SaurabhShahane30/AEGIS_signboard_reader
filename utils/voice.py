"""
============================================================
  voice.py — Wake-word detection and voice command capture
============================================================
"""

import time
import speech_recognition as sr

from .config import WAKE_WORD
from . import state
from .tts import speak

# Phrases that signal a direct command without needing the wake word
COMMAND_TRIGGERS = (
    "add ", "remove ", "delete ", "clear ", "reset ",
    "what are my", "list my", "show my",
    "alert me", "notify me", "watch for", "remember",
    "stop tracking", "don't alert", "don't track",
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
    print("[Voice] Calibrating microphone for ambient noise (2 seconds)...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=2)
    print(f"[Voice] Calibration done. Energy threshold: {recognizer.energy_threshold:.0f}")
    print("[Voice] List commands work at any time. Say 'hey system' for anything else.")
 
    while state.running:
        try:
            with mic as source:
                audio = recognizer.listen(source, timeout=3, phrase_time_limit=8)
 
            text = recognizer.recognize_google(audio).lower().strip()
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
                    command = recognizer.recognize_google(cmd_audio).lower().strip()
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