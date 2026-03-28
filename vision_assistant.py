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
                     SpeechRecognition pyaudio numpy requests \
                     sounddevice soundfile
    4. Set MODEL_PATH below to your best.pt location.
    5. Run:  python vision_assistant.py

VOICE COMMANDS (say "Hey system" first):
    "add wet floor to priority"
    "remove stop from priority"
    "what are my priorities?"
    "clear user list"

CHANGES vs previous version:
    - ElevenLabs: uses /stream endpoint — no temp file written to disk
    - ElevenLabs: startup connectivity + API key check added
    - Wake word changed to "hey system" (more phonetically distinct)
    - Voice recognizer: lower energy threshold, dynamic threshold ON,
      shorter pause threshold, longer ambient noise calibration

OCR CHANGES (this version):
    - [OCR-1] EasyOCR moved to its own dedicated thread (ocr_worker)
              Camera thread now only pushes crops — never blocks on readtext()
    - [OCR-2] Fuzzy keyword matching added via difflib.SequenceMatcher
              Catches OCR noise like "Wet Flo0r" matching "wet floor"
    - [OCR-3] user_priority_list persisted to user_priorities.json
              Loaded on startup, saved on every ADD / REMOVE / CLEAR
    - [OCR-4] Crop preprocessing added (grayscale + CLAHE + denoise)
              Improves EasyOCR accuracy on uneven lighting and shadows

DIRECTION CHANGES (this version):
    - [DIR-1] get_direction() computes relative sign position
              (left / ahead-left / ahead / ahead-right / right)
              using box center vs frame width, with span-overlap blending
    - [DIR-2] alert_snapshot_dir created at startup (alert_snapshots/)
    - [DIR-3] save_alert_snapshot() saves the full annotated frame
              with the detected sign box highlighted and a label
              stamped on it; filename encodes timestamp + sign text
    - [DIR-4] crop_queue now carries (crop, (x1,y1,x2,y2), frame_ref)
              so ocr_worker has the full box and latest annotated frame
    - [DIR-5] Passive alerts and LLM alerts both append direction string
    - [DIR-6] LLM prompt extended with direction context so the model
              can include direction in its natural language response
============================================================
"""

import cv2
import threading
import time
import queue
import sys
import os
import json
from difflib import SequenceMatcher
from dotenv import load_dotenv
load_dotenv()
import subprocess
import requests
import numpy as np
import pyttsx3
import speech_recognition as sr
import easyocr
from collections import deque
from datetime import datetime

try:
    from googletrans import Translator
except Exception:
    Translator = None

import PIL.Image
if not hasattr(PIL.Image, "ANTIALIAS"):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

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
#  CONFIG
# ─────────────────────────────────────────────
MODEL_PATH           = "signboard_project/best_yolov8s.pt"  # your YOLO model
CAMERA_INDEX         = 1              # 0 = default webcam
WAKE_WORD            = "hey system"   # FIX: changed from "hello system" — more phonetically distinct
OCR_COOLDOWN_SEC     = 4              # seconds before same text re-triggers
LLM_RATE_LIMIT_SEC   = 2             # min gap between LLM calls
CONFIDENCE_THRESHOLD = 0.5            # raise this if too many false detections
MIN_BOX_AREA         = 3000           # pixels² — ignore tiny/noisy boxes
OLLAMA_URL           = "http://localhost:11434/api/generate"
OLLAMA_MODEL         = "llama3.1:8b"

# Speech output options
TARGET_LANGUAGE      = "en"           # e.g. "en", "hi", "es"
SOURCE_LANGUAGE      = "auto"
USE_ELEVENLABS       = False
ELEVENLABS_API_KEY   = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID  = os.getenv("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")
ELEVENLABS_MODEL_ID  = "eleven_multilingual_v2"
TTS_HTTP_TIMEOUT_SEC = 12
PRIORITY_REPEAT_COOLDOWN_SEC = 4    # suppress repeating same priority alert too often

# [DIR-2] Snapshot output directory
ALERT_SNAPSHOT_DIR   = "alert_snapshots"

# ─────────────────────────────────────────────
#  PRIORITY LISTS
# ─────────────────────────────────────────────
CRITICAL_LIST = {
    "stop", "danger", "no entry", "wet floor", "slippery road",
    "hazard", "warning", "emergency", "fire", "poison",
    "explosive", "high voltage", "do not enter", "keep out",
    "restricted", "caution", "alert", "no exit", "blocked"
}

# [OCR-3] Loaded from disk on startup; saved on every mutation
PRIORITY_FILE = "user_priorities.json"

def load_priorities() -> set:
    try:
        with open(PRIORITY_FILE, "r") as f:
            data = json.load(f)
            print(f"[Memory] Loaded {len(data)} saved priorities: {data}")
            return set(data)
    except FileNotFoundError:
        return set()
    except json.JSONDecodeError:
        print("[Memory] Priority file corrupted — starting fresh.")
        return set()

def save_priorities():
    try:
        with open(PRIORITY_FILE, "w") as f:
            json.dump(list(user_priority_list), f, indent=2)
    except Exception as e:
        print(f"[Memory] Save failed: {e}")

user_priority_list = set()   # populated in main() via load_priorities()

# ─────────────────────────────────────────────
#  SHARED STATE
# ─────────────────────────────────────────────
ocr_queue    = queue.Queue()
# [DIR-4] crop_queue items are now (crop, (x1,y1,x2,y2), annotated_frame)
crop_queue   = queue.Queue(maxsize=5)
speech_queue = queue.Queue()
tts_queue    = queue.Queue()
recent_texts = deque(maxlen=30)
last_llm_call = 0
lock         = threading.Lock()
running      = True
translation_cache = {}
translator_instance = None
last_priority_alert_times = {}

# [DIR-4] Rolling frame buffer — camera writes, OCR/LLM reads
# Stores (timestamp, annotated_frame) tuples; maxlen caps memory use
frame_buffer = deque(maxlen=30)

# ─────────────────────────────────────────────
#  [DIR-1] DIRECTION DETECTION
# ─────────────────────────────────────────────
# The frame is divided into three horizontal thirds:
#   left zone   : box_center_ratio < 0.33
#   ahead zone  : 0.33 <= box_center_ratio <= 0.66
#   right zone  : box_center_ratio > 0.66
#
# If the box *spans* two zones (e.g. center is right but left edge
# reaches into the ahead zone), we report a blended direction.

def get_direction(x1: int, x2: int, frame_width: int) -> str:
    """
    Compute the relative direction of a detected sign.

    Parameters
    ----------
    x1, x2      : bounding box left and right pixel x-coordinates
    frame_width : width of the full camera frame in pixels

    Returns
    -------
    One of: "to your left" | "ahead and to your left" | "straight ahead"
            | "ahead and to your right" | "to your right"
    """
    if frame_width <= 0:
        return "straight ahead"

    center_ratio = ((x1 + x2) / 2) / frame_width
    left_ratio   = x1 / frame_width
    right_ratio  = x2 / frame_width

    LEFT_THRESH  = 0.33
    RIGHT_THRESH = 0.66

    # Determine which zone(s) the box occupies
    in_left   = left_ratio  < LEFT_THRESH
    in_right  = right_ratio > RIGHT_THRESH
    in_center = center_ratio >= LEFT_THRESH and center_ratio <= RIGHT_THRESH

    if in_left and in_right:
        # Rare: box spans the whole frame
        return "straight ahead"

    if in_left and in_center:
        return "ahead and to your left"

    if in_right and in_center:
        return "ahead and to your right"

    if center_ratio < LEFT_THRESH:
        return "to your left"

    if center_ratio > RIGHT_THRESH:
        return "to your right"

    return "straight ahead"


# ─────────────────────────────────────────────
#  [DIR-3] ALERT SNAPSHOT SAVER
# ─────────────────────────────────────────────
def save_alert_snapshot(
    annotated_frame: np.ndarray,
    box: tuple,
    sign_text: str,
    direction: str,
) -> str | None:
    """
    Save a full annotated camera frame to disk when an alert fires.

    The sign bounding box is drawn with a bright highlight rect and a
    label strip so the snapshot is self-documenting.  The filename
    encodes the timestamp and a sanitised form of the sign text so
    alert_snapshots/ can be reviewed chronologically.

    Parameters
    ----------
    annotated_frame : full frame already annotated by YOLO (numpy BGR)
    box             : (x1, y1, x2, y2) of the detected sign
    sign_text       : OCR text that triggered the alert
    direction       : direction string from get_direction()

    Returns
    -------
    Saved filepath string, or None on failure.
    """
    try:
        os.makedirs(ALERT_SNAPSHOT_DIR, exist_ok=True)

        frame = annotated_frame.copy()
        x1, y1, x2, y2 = box
        h, w = frame.shape[:2]

        # ── Draw highlight box ──────────────────────────────────────
        # Thick cyan outer rect (visible over YOLO's default colours)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

        # Draw the frame's vertical midline as a thin reference line
        mid_x = w // 2
        cv2.line(frame, (mid_x, 0), (mid_x, h), (200, 200, 200), 1, cv2.LINE_AA)

        # ── Label strip above the box ───────────────────────────────
        label = f"{sign_text}  |  {direction}"
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.65
        thickness  = 2
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        strip_y1 = max(0, y1 - th - baseline - 8)
        strip_y2 = max(th + baseline + 8, y1)
        strip_x2 = min(w, x1 + tw + 10)

        # Dark semi-transparent background for the label
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, strip_y1), (strip_x2, strip_y2), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(
            frame, label,
            (x1 + 5, strip_y2 - baseline - 2),
            font, font_scale,
            (0, 255, 255), thickness, cv2.LINE_AA,
        )

        # ── Timestamp watermark (bottom-left) ──────────────────────
        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(
            frame, ts,
            (10, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (180, 180, 180), 1, cv2.LINE_AA,
        )

        # ── Save ────────────────────────────────────────────────────
        safe_text = "".join(c if c.isalnum() or c in "_ " else "_" for c in sign_text)
        safe_text = safe_text.replace(" ", "_")[:30]
        filename  = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{safe_text}.jpg"
        filepath  = os.path.join(ALERT_SNAPSHOT_DIR, filename)

        cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"[Snapshot] Saved: {filepath}")
        return filepath

    except Exception as e:
        print(f"[Snapshot Error] {e}")
        return None


# ─────────────────────────────────────────────
#  OLLAMA CALLER
# ─────────────────────────────────────────────
SYSTEM_PROMPT_WATCHER = """You are a vision assistant for a real-time camera system.

Given OCR text from a camera frame, decide whether it requires an alert.

If it matches or closely resembles a critical item or user priority → respond with a short, natural alert message.
Otherwise → respond: IGNORE

Rules:
- Calm, friendly, human-like phrasing
- No prefixes like "ALERT:" or "---" or role labels
- Under 15 words
- Include direction naturally if provided (e.g. "Pharmacy detected to your left.")
- Output ONLY the single alert sentence or the word IGNORE. Nothing else."""

SYSTEM_PROMPT_INTENT = """You are an intent parser for a priority memory system.

Map the user's voice command to exactly one structured action.

Valid outputs:
ADD: <item>
REMOVE: <item>
LIST:
CLEAR:
UNKNOWN: <short helpful reply>

Rules:
- Infer intent, don't require exact keywords
- "notify/alert/watch for/remember" → ADD
- "stop tracking/remove/don't alert" → REMOVE
- Normalize item: lowercase, singular, no filler words
- Output EXACTLY one line, no markdown, no explanation

Examples:
Input: let me know when we find a pharmacy
Output: ADD: pharmacy

Input: alert me if you see a hospital
Output: ADD: hospital

Input: don't track pharmacy anymore
Output: REMOVE: pharmacy

Input: what are my priorities
Output: LIST:

Input: clear my list
Output: CLEAR:

Input: hello how are you
Output: UNKNOWN: I can add, remove, list, or clear priority items.
Output constraints:

Exactly one line
No markdown
No explanations
No extra text beyond the required format
Plain text only. Single-line response."""


def call_ollama(prompt: str, system_prompt: str = SYSTEM_PROMPT_WATCHER) -> str:
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": f"{system_prompt}\n\n{prompt}",
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 40
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
def get_translator():
    """Lazily initialize translator so startup stays fast and robust."""
    global translator_instance
    if translator_instance is not None:
        return translator_instance
    if Translator is None:
        return None
    try:
        translator_instance = Translator()
        return translator_instance
    except Exception as e:
        print(f"[Translate Error] init failed: {e}")
        return None


def translate_for_speech(text: str) -> str:
    """Translate only at final speech stage to avoid affecting detection/logic."""
    if not text:
        return text
    if TARGET_LANGUAGE.lower() in {"", "en"}:
        return text

    key = (text, TARGET_LANGUAGE.lower())
    with lock:
        if key in translation_cache:
            return translation_cache[key]

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
            with lock:
                translation_cache[key] = translated
            return translated
        return text
    except Exception as e:
        print(f"[Translate Error] {e}")
        return text


def build_local_tts_engine():
    """Create pyttsx3 engine; return None if unavailable."""
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


def speak_with_elevenlabs(text: str):
    """
    FIX: Synthesize speech with ElevenLabs using the /stream endpoint.
    Audio bytes are piped directly to the player — no temp file is written to disk.
    """
    if not ELEVENLABS_API_KEY:
        raise RuntimeError("ELEVENLABS_API_KEY is not set")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY,
    }
    payload = {
        "text": text,
        "model_id": ELEVENLABS_MODEL_ID,
    }

    resp = requests.post(
        url, headers=headers, json=payload,
        timeout=TTS_HTTP_TIMEOUT_SEC, stream=True
    )
    resp.raise_for_status()

    audio_data = b"".join(resp.iter_content(chunk_size=4096))

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


def tts_worker():
    engine = None
    if not USE_ELEVENLABS:
        engine = build_local_tts_engine()

    while running:
        try:
            text = tts_queue.get(timeout=0.5)

            final_text = translate_for_speech(text)
            print(f"\n🔊 {final_text}")

            spoken = False

            if USE_ELEVENLABS:
                try:
                    speak_with_elevenlabs(final_text)
                    spoken = True
                except Exception as e:
                    print(f"[TTS ElevenLabs Error] {e} | falling back to pyttsx3")

            if not spoken:
                if engine is None:
                    engine = build_local_tts_engine()
                if engine is not None:
                    engine.say(final_text)
                    engine.runAndWait()
                    spoken = True

            if not spoken:
                print("[TTS] No available speech backend. Text was not spoken.")

            tts_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[TTS Error] {e}")


def speak(text: str):
    tts_queue.put(text)


# ─────────────────────────────────────────────
#  [OCR-2] FUZZY MATCHING
# ─────────────────────────────────────────────
def fuzzy_match(ocr_text: str, keyword: str, threshold: float = 0.82) -> bool:
    """
    Returns True if keyword appears in ocr_text with fuzzy tolerance.
    Handles OCR noise like 'Wet Flo0r' matching 'wet floor'.
    """
    t = ocr_text.lower()
    k = keyword.lower()

    if k in t:
        return True

    t_words = t.split()
    k_words = k.split()
    window_size = len(k_words)

    for i in range(max(1, len(t_words) - window_size + 1)):
        window = " ".join(t_words[i : i + window_size])
        ratio = SequenceMatcher(None, window, k).ratio()
        if ratio >= threshold:
            return True

    return False


# ─────────────────────────────────────────────
#  PRIORITY HELPERS
# ─────────────────────────────────────────────
def is_priority(text: str) -> bool:
    """Fast local check — avoids calling LLM for irrelevant text."""
    t = text.lower()
    for item in CRITICAL_LIST | user_priority_list:
        if fuzzy_match(t, item):
            return True
    return False


def build_passive_alert(text: str, direction: str = "straight ahead"):
    """
    Create immediate local alert for critical/custom priority matches.

    [DIR-5] direction is appended to every alert message so the user
    knows which way to look without waiting for the LLM.

    Returns a tuple: (priority_key, alert_message) or (None, None).
    """
    t = text.lower()

    for item in sorted(CRITICAL_LIST | user_priority_list, key=len, reverse=True):
        if not fuzzy_match(t, item):
            continue

        # Build the directional suffix
        # "straight ahead" is the neutral case — still worth saying
        dir_phrase = direction  # e.g. "to your left", "straight ahead"

        if item in {"wet floor", "slippery road"}:
            return item, f"Warning. Slippery surface {dir_phrase}."
        if item in {"fire", "explosive", "high voltage"}:
            return item, f"Danger. {item} {dir_phrase}."
        if item in {"no entry", "do not enter", "keep out", "restricted"}:
            return item, f"Warning. Restricted area {dir_phrase}."

        # Generic: e.g. "Pharmacy detected to your left."
        return item, f"{item.capitalize()} detected, {dir_phrase}."

    return None, None

def should_speak_priority_alert(priority_key: str) -> bool:
    """Rate-limit repeated passive alerts for the same priority item."""
    now = time.time()
    with lock:
        last_ts = last_priority_alert_times.get(priority_key)
        if last_ts is not None and (now - last_ts) < PRIORITY_REPEAT_COOLDOWN_SEC:
            return False
        last_priority_alert_times[priority_key] = now
    return True

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
        # ── Voice commands (highest priority, checked first) ──────────────
        try:
            command = speech_queue.get_nowait()
            print(f"\n🎤 Command: '{command}'")
            prompt = (
                f'USER_UTTERANCE: "{command}"\n'
                f"{priority_context()}"
            )
            response = call_ollama(prompt, system_prompt=SYSTEM_PROMPT_INTENT)
            print(f"[LLM -> Command] {response}")
            _handle_command_response(response)
            speech_queue.task_done()
            continue
        except queue.Empty:
            pass

        # ── OCR alerts ────────────────────────────────────────────────────
        try:
            item = ocr_queue.get(timeout=0.3)

            if isinstance(item, tuple):
                ocr_text, direction = item
            else:
                ocr_text, direction = str(item), "straight ahead"

            now = time.time()
            if (now - last_llm_call) < LLM_RATE_LIMIT_SEC:
                ocr_queue.task_done()
                continue

            last_llm_call = now

            prompt = (
                f'OCR TEXT from camera: "{ocr_text}"\n'
                f'Sign direction: {direction}\n'
                f"{priority_context()}\n"
                f"Should I alert the user? Include the direction naturally in your response."
            )
            response = call_ollama(prompt, system_prompt=SYSTEM_PROMPT_WATCHER)
            print(f"[LLM -> OCR] {response}")

            if response and response.upper().strip() != "IGNORE":
                speak(response.strip())

            ocr_queue.task_done()

        except queue.Empty:
            continue

def _handle_command_response(response: str):
    global user_priority_list
    upper = response.upper()

    if upper.startswith("ADD:"):
        item = response[4:].strip().lower()
        user_priority_list.add(item)
        save_priorities()
        speak(f"Added {item} to your priority list.")
        print(f"[Priority] User list: {user_priority_list}")

    elif upper.startswith("REMOVE:"):
        item = response[7:].strip().lower()
        if item in user_priority_list:
            user_priority_list.discard(item)
            save_priorities()
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
        save_priorities()
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

    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.6

    print(f"[Voice] Wake word: '{WAKE_WORD.upper()}'")
    print("[Voice] Calibrating microphone for ambient noise (2 seconds)...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=2)
    print(f"[Voice] Calibration done. Energy threshold: {recognizer.energy_threshold:.0f}")

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
#  [OCR-4] CROP PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_crop(crop: np.ndarray) -> np.ndarray:
    gray     = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
    return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)


# ─────────────────────────────────────────────
#  [OCR-1] DEDICATED OCR THREAD
# ─────────────────────────────────────────────

def ocr_worker():
    print("[OCR] Initializing EasyOCR...")
    reader = easyocr.Reader(["en"], gpu=True)
    print("[OCR] Ready.")

    while running:
        try:
            item = crop_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        try:
            crop, box_coords, annotated_frame = item
            x1, y1, x2, y2 = box_coords

            frame_h, frame_w = annotated_frame.shape[:2]
            direction = get_direction(x1, x2, frame_w)

            ocr_results = reader.readtext(crop, detail=0, paragraph=True)

            for text in ocr_results:
                text = text.strip()
                if len(text) < 2:
                    continue

                print(f"[OCR] '{text}'  →  {direction}")

                if is_duplicate(text):
                    continue

                priority_key, passive_alert = build_passive_alert(text, direction)

                if passive_alert and priority_key:
                    if should_speak_priority_alert(priority_key):
                        save_alert_snapshot(
                            annotated_frame,
                            (x1, y1, x2, y2),
                            text,
                            direction,
                        )
                        speak(passive_alert)

                elif len(text) > 3:
                    ocr_queue.put((text, direction))

        except Exception as e:
            print(f"[OCR Error] {e}")
        finally:
            crop_queue.task_done()

# ─────────────────────────────────────────────
#  CAMERA + YOLO
# ─────────────────────────────────────────────
def camera_worker():
    """
    Camera thread: capture → YOLO → push crops to crop_queue.

    [DIR-4] crop_queue items now carry the full annotated frame so
            ocr_worker can pass it to save_alert_snapshot().
    """
    print("[Camera] Loading YOLO model...")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {CAMERA_INDEX}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[Camera] Running. Press Q to quit.\n")
    speak("System ready. Say hey system to give a command.")

    while running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        # ── YOLO ──
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        annotated = results[0].plot()

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            box_area = (x2 - x1) * (y2 - y1)
            if box_area < MIN_BOX_AREA:
                continue

            pad = 6
            x1p = max(0, x1 - pad)
            y1p = max(0, y1 - pad)
            x2p = min(frame.shape[1], x2 + pad)
            y2p = min(frame.shape[0], y2 + pad)

            crop = frame[y1p:y2p, x1p:x2p]
            if crop.size == 0:
                continue

            # [DIR-4] Push (crop, full_box, annotated_frame) instead of (crop, origin)
            # We use the un-padded box coords for direction + snapshot highlighting
            try:
                crop_queue.put_nowait((crop.copy(), (x1, y1, x2, y2), annotated.copy()))
            except queue.Full:
                pass  # drop this crop; next frame will try again

        # ── HUD ──
        ts = datetime.now().strftime("%H:%M:%S")
        cv2.putText(annotated, f"Vision Assistant  {ts}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated, f"Say: '{WAKE_WORD}' + command",
                    (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1)
        cv2.putText(annotated, f"Custom priorities: {len(user_priority_list)}  |  conf>{CONFIDENCE_THRESHOLD}",
                    (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 255), 1)
        # [DIR-2] Show snapshot count on HUD
        snap_count = len(os.listdir(ALERT_SNAPSHOT_DIR)) if os.path.isdir(ALERT_SNAPSHOT_DIR) else 0
        cv2.putText(annotated, f"Snapshots saved: {snap_count}",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 120), 1)

        cv2.imshow("Vision Assistant", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    global running, user_priority_list

    print("=" * 60)
    print("  VISION ASSISTANT  |  Ollama llama3.1  |  YOLOv8 + OCR")
    print("=" * 60)

    # [OCR-3] Load saved priorities before any thread starts
    user_priority_list = load_priorities()

    # [DIR-2] Ensure snapshot directory exists at startup
    os.makedirs(ALERT_SNAPSHOT_DIR, exist_ok=True)
    print(f"[Init] Alert snapshots will be saved to: {os.path.abspath(ALERT_SNAPSHOT_DIR)}/")

    # ── Ollama connectivity check ──
    print("[Check] Pinging Ollama...")
    try:
        requests.get("http://localhost:11434", timeout=3)
        print("[Check] Ollama is running ✓")
    except Exception:
        print("\n[ERROR] Ollama is not running!")
        print("        Open a NEW terminal and run:  ollama serve")
        print("        Then re-run this script.\n")
        sys.exit(1)

    if USE_ELEVENLABS:
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
                json={
                    "text": "ok",
                    "model_id": ELEVENLABS_MODEL_ID,
                },
                timeout=10,
            )
            if test_resp.status_code == 200:
                print("[Check] ElevenLabs connected ✓")
            elif test_resp.status_code == 401:
                print(f"[ERROR] ElevenLabs rejected key: {test_resp.text}")
                sys.exit(1)
            else:
                print(f"[WARN] ElevenLabs status {test_resp.status_code}: {test_resp.text}")
        except Exception as e:
            print(f"[ERROR] ElevenLabs unreachable: {e}")
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

    # [OCR-1] Dedicated OCR thread
    t_ocr = threading.Thread(target=ocr_worker, daemon=True, name="OCR")
    t_ocr.start()
    threads.append(t_ocr)

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