"""
session.py — Local session management.

A session:
  - Starts when the system boots (main.py calls start_session()).
  - Ends when the user says "end session" or the process exits.
  - Is saved as a JSON file in sessions/<session_id>.json.
  - Contains its own snapshot of user_priority_list at the time it ended.
  - Does NOT carry priority list data forward to the next session.

Session file schema:
{
  "session_id":    "20250328_143201",
  "started_at":    "2025-03-28T14:32:01.123456",
  "ended_at":      "2025-03-28T14:58:44.001234",   # null if crashed
  "end_reason":    "user_command" | "process_exit" | "crashed",
  "priority_list": [],       # snapshot at session end
  "events": [
    {
      "time":  "2025-03-28T14:32:15.001234",
      "type":  "detection" | "alert" | "command",
      "data":  "..."
    },
    ...
  ]
}
"""

import json
import os
from datetime import datetime

from . import state

SESSIONS_DIR = "sessions"


# ── Internal helpers ───────────────────────────────────────

def _session_path(session_id: str) -> str:
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")


def _now_iso() -> str:
    return datetime.now().isoformat()


def _save(data: dict):
    """Write session data to disk atomically (write then rename)."""
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    path   = _session_path(data["session_id"])
    tmp    = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


# ── Public API ─────────────────────────────────────────────

def start_session():
    """
    Initialise a new session.
    - Resets user_priority_list to empty (no carry-over from previous sessions).
    - Sets state.session_active = True.
    - Writes an initial JSON file immediately (so partial data is safe on crash).
    """
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    data = {
        "session_id":  session_id,
        "started_at":  _now_iso(),
        "ended_at":    None,
        "end_reason":  None,
        "priority_list": [],
        "events":      [],
    }

    # ── Clean slate: no priority list bleed-over ──
    with state.lock:
        state.user_priority_list.clear()
        state.current_session_id = session_id
        state.session_active     = True
        state._session_data      = data

    _save(data)
    print(f"[Session] Started → sessions/{session_id}.json")


def end_session(reason: str = "user_command"):
    """
    Close the current session and immediately start a fresh one (unless shutting down).

    - Snapshots user_priority_list into the old session JSON.
    - Clears user_priority_list (no bleed-over).
    - When reason is "process_exit" a new session is NOT started (app is shutting down).

    reason: "user_command" | "process_exit" | "crashed"
    """
    with state.lock:
        if not state.session_active:
            return
        state.session_active = False
        data = state._session_data
        data["ended_at"]      = _now_iso()
        data["end_reason"]    = reason
        data["priority_list"] = sorted(state.user_priority_list)

    _save(data)
    print(f"[Session] Ended ({reason}) → sessions/{data['session_id']}.json")

    # Keep the system alive — start a new clean session unless we're shutting down
    if reason != "process_exit":
        start_session()


def log_event(event_type: str, detail: str):
    """
    Append a timestamped event to the current session.

    event_type: "detection" | "alert" | "command"
    detail:     human-readable string describing the event
    """
    with state.lock:
        if not state.session_active:
            return
        data = state._session_data
        data["events"].append({
            "time":  _now_iso(),
            "type":  event_type,
            "data":  detail,
        })

    # Flush to disk after every event (safe against crashes)
    _save(data)


def get_recent_events(n: int = 10) -> list:
    """
    Return the last `n` events from the current live session.
    Used by llm.py to inject session context into prompts.
    Returns an empty list if no session is active.
    """
    with state.lock:
        if not state.session_active:
            return []
        return list(state._session_data["events"][-n:])


def format_context_for_prompt(n: int = 10) -> str:
    """
    Return a compact, human-readable summary of the last `n` session events,
    formatted for injection into an Ollama prompt.

    Example output:
        Session context (last 4 events):
        [14:32:15] detection   — ocr='EXIT ONLY'
        [14:33:01] alert       — passive alert='Warning. wet floor ahead.'
        [14:35:44] command     — input='add pharmacy' → response='ADD: pharmacy'
        [14:36:10] detection   — ocr='Apollo Pharmacy | Medical Store'
    """
    events = get_recent_events(n)
    if not events:
        return "Session context: no events yet this session."

    lines = [f"Session context (last {len(events)} events):"]
    for e in events:
        try:
            ts = e["time"][11:19]   # extract HH:MM:SS from ISO string
        except Exception:
            ts = "??:??:??"
        lines.append(f"  [{ts}] {e['type']:<10} — {e['data']}")

    return "\n".join(lines)