"""
============================================================
  state.py — Shared mutable state, queues, and priority I/O
============================================================
"""

import queue
import threading
from collections import deque

from .config import SOURCE_LANGUAGE, TARGET_LANGUAGE

# ── Runtime flag ─────────────────────────────────────────
running = True

# ── Queues ───────────────────────────────────────────────
crop_queue     = queue.Queue(maxsize=5)
speech_queue   = queue.Queue()
tts_queue      = queue.Queue()
ws_frame_queue = queue.Queue(maxsize=5)   # Flutter → camera_worker frames

# ── WebSocket state ───────────────────────────────────────
ws_clients      = set()
ws_clients_lock = threading.Lock()
ws_event_queue  = queue.Queue(maxsize=100)  # capped — drops oldest if no client consuming

# ── Shared data structures ───────────────────────────────
recent_texts              = deque(maxlen=30)
last_priority_alert_times = {}
translation_cache         = {}

# ── Synchronisation ──────────────────────────────────────
lock = threading.Lock()

# ── User priority list (populated by load_priorities) ────
user_priority_list: set = set()

# ── Translator singleton ─────────────────────────────────
translator_instance = None

# ── Runtime language settings (mutable, updated via WebSocket) ───────────
# Initialised from config.py defaults; Flutter can override at runtime
# via a {"type": "set_language", ...} WebSocket message.
source_language: str = SOURCE_LANGUAGE   # e.g. "en-US" or "hi-IN"
target_language: str = TARGET_LANGUAGE   # e.g. "en"   or "hi"