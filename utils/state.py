"""
============================================================
  state.py — Shared mutable state, queues, and priority I/O
============================================================
"""

import queue
import threading
from collections import deque

# ── Runtime flag ─────────────────────────────────────────
running = True

# ── Queues ───────────────────────────────────────────────
crop_queue   = queue.Queue(maxsize=5)
speech_queue = queue.Queue()
tts_queue    = queue.Queue()
ws_frame_queue = queue.Queue(maxsize=5)   # Flutter → camera_worker frames

# ── WebSocket state ───────────────────────────────────────
ws_clients      = set()
ws_clients_lock = threading.Lock()
ws_event_queue  = queue.Queue()

# ── Shared data structures ───────────────────────────────
recent_texts             = deque(maxlen=30)
last_priority_alert_times = {}
translation_cache        = {}

# ── Synchronisation ──────────────────────────────────────
lock = threading.Lock()

# ── User priority list (populated by load_priorities) ────
user_priority_list: set = set()

# ── Translator singleton ─────────────────────────────────
translator_instance = None