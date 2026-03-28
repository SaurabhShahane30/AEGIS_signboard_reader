"""
============================================================
  priorities.py — Priority matching, alert building, dedup
============================================================
"""

import time
from difflib import SequenceMatcher

from .config import CRITICAL_LIST, OCR_COOLDOWN_SEC, PRIORITY_REPEAT_COOLDOWN_SEC
from . import state


# ─────────────────────────────────────────────────────────
#  Fuzzy matching
# ─────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────
#  Priority helpers
# ─────────────────────────────────────────────────────────

def is_priority(text: str) -> bool:
    """Fast local check — avoids calling LLM for irrelevant text."""
    t = text.lower()
    for item in CRITICAL_LIST | state.user_priority_list:
        if fuzzy_match(t, item):
            return True
    return False


def build_passive_alert(text: str, direction: str = "straight ahead"):
    """
    Create an immediate local alert for critical/custom priority matches.
    Returns (priority_key, alert_message) or (None, None).
    """
    t = text.lower()

    for item in sorted(CRITICAL_LIST | state.user_priority_list, key=len, reverse=True):
        if not fuzzy_match(t, item):
            continue

        dir_phrase = direction

        if item in {"wet floor", "slippery road"}:
            return item, f"Warning. Slippery surface {dir_phrase}."
        if item in {"fire", "explosive", "high voltage"}:
            return item, f"Danger. {item} {dir_phrase}."
        if item in {"no entry", "do not enter", "keep out", "restricted"}:
            return item, f"Warning. Restricted area {dir_phrase}."

        return item, f"{item.capitalize()} detected, {dir_phrase}."

    return None, None


def should_speak_priority_alert(priority_key: str) -> bool:
    """Rate-limit repeated passive alerts for the same priority item."""
    now = time.time()
    with state.lock:
        last_ts = state.last_priority_alert_times.get(priority_key)
        if last_ts is not None and (now - last_ts) < PRIORITY_REPEAT_COOLDOWN_SEC:
            return False
        state.last_priority_alert_times[priority_key] = now
    return True


def priority_context() -> str:
    return (
        f"Critical list: {', '.join(sorted(CRITICAL_LIST))}\n"
        f"User priority list: {', '.join(sorted(state.user_priority_list)) or 'empty'}"
    )


def is_duplicate(text: str) -> bool:
    now = time.time()
    t = text.lower().strip()
    with state.lock:
        for (rt, ts) in state.recent_texts:
            if rt == t and (now - ts) < OCR_COOLDOWN_SEC:
                return True
        state.recent_texts.append((t, now))
    return False