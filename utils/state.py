"""
state.py — Shared mutable state: queues, locks, flags, and caches
           used across all threads.
"""

import queue
import threading
from collections import deque

# ── Inter-thread queues ────────────────────────────────────
crop_queue   = queue.Queue(maxsize=4)  # camera  → ocr_worker  (bounded: drop if full)
ocr_queue    = queue.Queue()           # ocr_worker → llm_worker
speech_queue = queue.Queue()           # voice   → llm_worker
tts_queue    = queue.Queue()           # anyone  → tts_worker

# ── Deduplication buffer ───────────────────────────────────
recent_texts: deque = deque(maxlen=30)   # (text, timestamp) pairs

# ── LLM rate-limiting ──────────────────────────────────
last_llm_call: float = 0.0

# ── LLM alert dedup — prevents same alert firing repeatedly ──
# Maps normalised alert text -> timestamp last spoken
last_llm_alert_times: dict = {}

# ── Thread safety ──────────────────────────────────────────
lock = threading.Lock()

# ── App lifecycle ──────────────────────────────────────────
running: bool = True

# ── Translation cache ──────────────────────────────────────
translation_cache: dict = {}
translator_instance = None       # lazily initialised in tts.py

# ── Priority alert timestamps ──────────────────────────────
last_priority_alert_times: dict = {}

# ── User priority list (mutable at runtime via voice) ──────
user_priority_list: set = set()

# ── Session management ─────────────────────────────────────
session_active: bool       = False
current_session_id: str    = ""
_session_data: dict        = {}    # internal: owned by session.py