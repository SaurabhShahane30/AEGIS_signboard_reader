"""
priority.py — Priority list management and alert helpers.

Responsibilities:
  - CRITICAL_LIST (hardcoded safety terms)
  - Fast local keyword check (avoids unnecessary LLM calls)
  - Passive rule-based alert builder
  - Per-item alert rate-limiter
  - OCR text deduplication
  - is_relevant_to_active_list() — hard gate used by camera.py
    to drop OCR text that matches nothing in the active list.

Detection modes (determined by whether user_priority_list is empty):

  CRITICAL-ONLY mode  (user_priority_list is empty)
    Only OCR text that fuzzy-matches something in CRITICAL_LIST
    is forwarded. Everything else is silently dropped.

  PRIORITY+CRITICAL mode  (user_priority_list has >= 1 item)
    OCR text that fuzzy-matches anything in CRITICAL_LIST
    OR user_priority_list is forwarded. Everything else dropped.

The switch between modes is automatic — it happens the moment the
user says "add pharmacy" (user_priority_list gains an entry) and
reverts when they say "clear" (user_priority_list becomes empty again).
"""

import time
import difflib
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


def active_list() -> set:
    """
    Return the set of terms that should trigger detection right now.

    CRITICAL-ONLY mode : user_priority_list is empty -> only CRITICAL_LIST
    PRIORITY mode      : user_priority_list has items -> both lists combined
    """
    if state.user_priority_list:
        return CRITICAL_LIST | state.user_priority_list
    return CRITICAL_LIST


# ── Fuzzy relevance gate ───────────────────────────────────

def is_relevant_to_active_list(ocr_text: str) -> bool:
    """
    Returns True if `ocr_text` is relevant to at least one term
    in the currently active list (critical-only or priority+critical).

    Three-pass check:

      Pass 1 — exact substring  (clean reads: "PHARMACY", "WET FLOOR")

      Pass 2 — fuzzy word match  (near-correct reads: "PHARWACY", "EMERGANCY")
               Each OCR word is compared to each term word at cutoff 0.75.
               Also checks the reverse — each term word compared to OCR words —
               so a partially garbled word still matches.

      Pass 3 — short-fragment rescue  (heavily garbled short reads: "HRWCY").
               If an OCR word is >= 4 chars and shares >= 72% of its characters
               (as a set) with a term word of similar length (+-3), it passes.

    Similarity thresholds:
      "pharmacy" / "pharwacy" = 0.88  -> pass 2
      "pharmacy" / "phrmacy"  = 0.87  -> pass 2
      "pharmacy" / "hrwcy"    = 0.46  -> pass 3: shared chars h,r,c,y = 4/5 = 0.80
      "pharmacy" / "panzcy"   = 0.46  -> pass 3: shared chars a,c,y   = 3/6 = 0.50 (fail)
      "warning"  / "warnng"   = 0.93  -> pass 2
      "warning"  / "walking"  = 0.62  -> fail all passes
    """
    if not ocr_text:
        return False

    text_lower = ocr_text.lower()
    terms = active_list()

    # Pass 1: fast exact substring check
    for term in terms:
        if term.lower() in text_lower:
            return True

    ocr_words = text_lower.split()

    for term in terms:
        term_words = term.lower().split()
        for tw in term_words:

            # Pass 2: fuzzy similarity (handles 1-2 char OCR errors)
            # Only for term words >= 5 chars; shorter words (stop, fire, keep,
            # exit, alert) are too short for reliable fuzzy matching and stay
            # exact-only to prevent false hits like "shop" -> "stop".
            if len(tw) >= 5:
                if difflib.get_close_matches(tw, ocr_words, n=1, cutoff=0.75):
                    return True
                for ow in ocr_words:
                    if len(ow) >= 4:
                        if difflib.get_close_matches(ow, [tw], n=1, cutoff=0.75):
                            return True
            else:
                # Short term words (stop, fire, keep, exit, alert) — exact only
                if tw in ocr_words:
                    return True
                continue

            # Pass 3: character-set overlap for heavily garbled short fragments
            # Requires OCR word >= 5 chars to avoid false hits on short common
            # words (e.g. "open" vs "poison" would pass at lower thresholds).
            for ow in ocr_words:
                if len(ow) < 5:
                    continue
                if abs(len(ow) - len(tw)) > 3:
                    continue
                shared = len(set(ow) & set(tw))
                ratio  = shared / max(len(set(ow)), 1)
                if ratio >= 0.80:
                    return True

    return False


def clear_vote_buffer_for_item(item: str):
    """
    Called by llm.py after ADD/REMOVE so stale vote-buffer entries from
    before the priority change don't block or pollute new detections.
    """
    try:
        from . import camera as _cam
        keys_to_clear = [k for k in list(_cam._vote_buffer.keys())
                         if difflib.get_close_matches(
                             item.lower(), k.lower().split(), n=1, cutoff=0.70)]
        for k in keys_to_clear:
            _cam._vote_buffer.pop(k, None)
        if keys_to_clear:
            print(f"[Priority] Cleared {len(keys_to_clear)} stale vote-buffer "
                  f"entries for '{item}'")
    except Exception as e:
        print(f"[Priority] vote buffer clear skipped: {e}")


def current_mode() -> str:
    """Return a human-readable string of the current detection mode."""
    if state.user_priority_list:
        return f"priority+critical ({len(state.user_priority_list)} custom)"
    return "critical-only"


# ── Existing helpers ──────────────────────────────────────

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
    Uses fuzzy matching to handle OCR variations (e.g., "pharmacy" vs "pharmcy").
    Appends (text, timestamp) to recent_texts if it's a new entry.
    """
    import difflib
    now = time.time()
    t = text.lower().strip()
    with state.lock:
        for (rt, ts) in state.recent_texts:
            if (now - ts) < OCR_COOLDOWN_SEC:
                # Exact match
                if rt == t:
                    return True
                # Fuzzy match for OCR variations
                similarity = difflib.SequenceMatcher(None, rt, t).ratio()
                if similarity > 0.8:  # Adjust threshold as needed
                    return True
        state.recent_texts.append((t, now))
    return False


def priority_context() -> str:
    """Return a human-readable summary of active priorities for LLM prompts."""
    return (
        f"Detection mode: {current_mode()}\n"
        f"Critical list: {', '.join(sorted(CRITICAL_LIST))}\n"
        f"User priority list: {', '.join(sorted(state.user_priority_list)) or 'empty'}"
    )