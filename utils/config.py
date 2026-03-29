"""
============================================================
  config.py — All tuneable constants in one place
============================================================
"""

# ── Input source ─────────────────────────────────────────
# "local"   = laptop webcam + laptop mic; output on local cv2 window
# "flutter" = video + audio from Flutter app over WebSocket; output back to app
# "esp32"   = video from ESP32-CAM HTTP stream + laptop mic; output on local cv2 window
INPUT_SOURCE = "local"

# ── Model & Camera ───────────────────────────────────────
MODEL_PATH           = "signboard_project/best_yolov8s.pt"
CAMERA_INDEX         = 0              # 0 = default webcam. Used if INPUT_SOURCE is "local".

# ── ESP32-CAM ────────────────────────────────────────────
# Used only when INPUT_SOURCE = "esp32"
ESP32_STREAM_URL     = "http://172.20.10.6:81/stream"
ESP32_RECONNECT_SEC  = 3              # seconds to wait before retrying a dropped stream

# ── WebSocket streaming (for external display) ─────────────
WS_HOST      = "0.0.0.0"
WS_PORT      = 8765
JPEG_QUALITY = 65
STREAM_FPS   = 15

# ── Detection thresholds ─────────────────────────────────
CONFIDENCE_THRESHOLD = 0.5            # raise if too many false detections
MIN_BOX_AREA         = 3000           # pixels² — ignore tiny/noisy boxes

# ── Timing ───────────────────────────────────────────────
OCR_COOLDOWN_SEC              = 4     # seconds before same text re-triggers
LLM_RATE_LIMIT_SEC            = 4     # min gap between LLM calls
PRIORITY_REPEAT_COOLDOWN_SEC  = 4     # suppress repeating same priority alert

# ── Wake word ────────────────────────────────────────────
WAKE_WORD = "hey system"

# ── Ollama ───────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"

# ── Speech / Translation ─────────────────────────────────
# TARGET_LANGUAGE : BCP-47 code for TTS output + translation destination
#   "en" → English (no translation performed)
#   "hi" → Hindi
#   "es" → Spanish
#   "mr" → Marathi   "ta" → Tamil   "bn" → Bengali   "fr" → French
TARGET_LANGUAGE      = "en"
 
# SOURCE_LANGUAGE : BCP-47 code passed to Google STT for voice input recognition
#   "en-US"  → English (United States)
#   "hi-IN"  → Hindi (India)   ← switch here for Hindi voice commands
#   "es-ES"  → Spanish (Spain)
#   Keep "en-US" if voice commands are in English regardless of TTS output language.
SOURCE_LANGUAGE      = "en-US"
TTS_HTTP_TIMEOUT_SEC = 12

# ── Persistence ──────────────────────────────────────────
PRIORITY_FILE      = "user_priorities.json"
ALERT_SNAPSHOT_DIR = "alert_snapshots"

# ── Priority lists ───────────────────────────────────────
CRITICAL_LIST = {
    "stop", "danger", "no entry", "wet floor", "slippery road",
    "hazard", "warning", "emergency", "fire", "poison",
    "explosive", "high voltage", "do not enter", "keep out",
    "restricted", "caution", "alert", "no exit", "blocked"
}