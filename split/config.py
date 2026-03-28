"""
config.py — All configuration constants and environment variables.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Model & Camera ─────────────────────────────────────────
MODEL_PATH           = "../signboard_project/best_yolov8s.pt"
CAMERA_INDEX         = 0
CONFIDENCE_THRESHOLD = 0.5
MIN_BOX_AREA         = 3000          # px² — ignore tiny/noisy detections

# ── Wake Word & Voice ──────────────────────────────────────
WAKE_WORD            = "hey system"

# ── OCR / LLM Rate Limiting ────────────────────────────────
OCR_COOLDOWN_SEC     = 4
LLM_RATE_LIMIT_SEC   = 2

# ── Ollama ─────────────────────────────────────────────────
OLLAMA_URL           = "http://localhost:11434/api/generate"
OLLAMA_MODEL         = "llama3.1:8b"

# ── Speech / Translation ───────────────────────────────────
TARGET_LANGUAGE      = "en"          # e.g. "en", "hi", "es"
SOURCE_LANGUAGE      = "auto"

# ── ElevenLabs ─────────────────────────────────────────────
USE_ELEVENLABS        = True
ELEVENLABS_API_KEY    = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID   = os.getenv("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")
ELEVENLABS_MODEL_ID   = "eleven_multilingual_v2"
TTS_HTTP_TIMEOUT_SEC  = 12

# ── Alert Cooldown ─────────────────────────────────────────
PRIORITY_REPEAT_COOLDOWN_SEC = 12
