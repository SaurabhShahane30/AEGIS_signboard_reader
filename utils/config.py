"""
============================================================
  config.py — All tuneable constants in one place
============================================================
"""

# ── Input source ─────────────────────────────────────────
# "local"   = laptop camera + laptop microphone
# "flutter" = video frames + voice commands from Flutter over WebSocket
INPUT_SOURCE = "local"

# ── Model & Camera ───────────────────────────────────────
MODEL_PATH           = "signboard_project/best_yolov8s.pt"
CAMERA_INDEX         = 0              # 0 = default webcam. Used if input source is "local".

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
TARGET_LANGUAGE      = "en"           # e.g. "en", "hi", "es"
SOURCE_LANGUAGE      = "auto"
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