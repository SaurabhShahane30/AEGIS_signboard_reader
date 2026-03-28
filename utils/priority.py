"""
priority.py — Priority list management and alert helpers.

Responsibilities:
  - CRITICAL_LIST (hardcoded safety terms)
  - Fast local keyword check (avoids unnecessary LLM calls)
  - Passive rule-based alert builder
  - Per-item alert rate-limiter
  - OCR text deduplication
"""

import time
from . import state
from .config import OCR_COOLDOWN_SEC, PRIORITY_REPEAT_COOLDOWN_SEC

# ── Hardcoded safety terms ─────────────────────────────────
CRITICAL_LIST: set = {
    "stop", "danger", "no entry", "wet floor", "slippery road",
    "hazard", "warning", "emergency", "fire", "poison",
    "explosive", "high voltage", "do not enter", "keep out",
    "restricted", "caution", "alert", "no exit", "blocked",
}


def all_priorities() -> set:
    """Return union of CRITICAL_LIST and user_priority_list."""
    return CRITICAL_LIST | state.user_priority_list


def is_priority(text: str) -> bool:
    """
    Fast local keyword scan — avoids sending irrelevant OCR text to the LLM.
    Returns True if any priority term appears in `text`.
    """
    t = text.lower()
    return any(item.lower() in t for item in all_priorities())


def build_passive_alert(text: str):
    """
    Build an immediate rule-based alert for critical/custom matches
    (longer phrases are checked first for specificity).

    Returns:
        (priority_key: str, alert_message: str) — or (None, None) if no match.
    """
    t = text.lower()
    for item in sorted(all_priorities(), key=len, reverse=True):
        if item.lower() not in t:
            continue

        if item in {"wet floor", "slippery road"}:
            return item, "Warning. Slippery surface ahead."
        if item in {"fire", "explosive", "high voltage"}:
            return item, f"Danger. {item} nearby."
        if item in {"no entry", "do not enter", "keep out", "restricted"}:
            return item, "Warning. Restricted area ahead."

        return item, f"Warning. {item} ahead."

    return None, None


def should_speak_priority_alert(priority_key: str) -> bool:
    """
    Rate-limit repeated alerts for the same priority item.
    Returns True only if the cooldown for `priority_key` has elapsed.
    """
    now = time.time()
    with state.lock:
        last_ts = state.last_priority_alert_times.get(priority_key)
        if last_ts is not None and (now - last_ts) < PRIORITY_REPEAT_COOLDOWN_SEC:
            return False
        state.last_priority_alert_times[priority_key] = now
    return True


def is_duplicate(text: str) -> bool:
    """
    Check whether `text` was already processed within OCR_COOLDOWN_SEC.
    Appends (text, timestamp) to recent_texts if it's a new entry.
    """
    now = time.time()
    t = text.lower().strip()
    with state.lock:
        for (rt, ts) in state.recent_texts:
            if rt == t and (now - ts) < OCR_COOLDOWN_SEC:
                return True
        state.recent_texts.append((t, now))
    return False


def priority_context() -> str:
    """Return a human-readable summary of active priorities for LLM prompts."""
    return (
        f"Critical list: {', '.join(sorted(CRITICAL_LIST))}\n"
        f"User priority list: {', '.join(sorted(state.user_priority_list)) or 'empty'}"
    )
