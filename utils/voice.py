"""
voice.py — Microphone input: wake word detection and command capture.

Flow:
  1. Continuously listens for WAKE_WORD ("hey system").
  2. On match → speaks "Listening." and captures the follow-up command.
  3. Pushes command string into state.speech_queue for llm_worker.

Recognizer tuning:
  - Lower energy threshold (300) catches quieter speech.
  - Dynamic threshold adapts to ambient noise in real time.
  - 0.6 s pause threshold for snappier end-of-speech detection.
  - 2-second ambient noise calibration on startup.
"""

import time
import speech_recognition as sr
from . import state
from .config import WAKE_WORD
from .tts import speak
from .session import end_session, log_event

END_SESSION_PHRASES = {"end session", "end the session", "stop session", "finish session"}

def voice_worker():
    """
    Daemon thread: listens for the wake word then captures a voice command.
    Pushes the command into state.speech_queue for the LLM worker to process.
    """
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    # ── Tuned settings for reliable wake-word detection ──
    recognizer.energy_threshold         = 300    # lower = catches quieter speech
    recognizer.dynamic_energy_threshold = True   # auto-adjusts to live ambient noise
    recognizer.pause_threshold          = 0.6    # faster end-of-speech detection

    print(f"[Voice] Wake word: '{WAKE_WORD.upper()}'")
    print("[Voice] Calibrating microphone for ambient noise (2 seconds)...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=2)
    print(f"[Voice] Calibration done. Energy threshold: {recognizer.energy_threshold:.0f}")

    while state.running:
        try:
            # ── Listen for wake word ──────────────────────
            with mic as source:
                audio = recognizer.listen(source, timeout=3, phrase_time_limit=6)

            heard = recognizer.recognize_google(audio).lower()
            print(f"[Voice] Heard: '{heard}'")

            if WAKE_WORD not in heard:
                continue

            # ── Wake word matched — capture command ───────
            speak("Listening.")
            print("[Voice] Waiting for command...")
            with mic as source:
                cmd_audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)

            command = recognizer.recognize_google(cmd_audio).lower()
            print(f"[Voice] Command: '{command}'")
            
            # ── Intercept "end session" — never send to LLM ──
            if any(phrase in command for phrase in END_SESSION_PHRASES):
                speak("Session ended. Starting a new session.")
                log_event("command", "end session")
                end_session(reason="user_command")
                # system keeps running; end_session() starts a fresh session
                continue
 
            state.speech_queue.put(command)

        except sr.WaitTimeoutError:
            continue                   # silence timeout — normal
        except sr.UnknownValueError:
            continue                   # speech not recognised — normal
        except sr.RequestError as e:
            print(f"[Voice Error] Google STT request failed: {e}")
            time.sleep(2)
        except Exception as e:
            print(f"[Voice Error] {e}")
            time.sleep(1)