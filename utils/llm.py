"""
llm.py — LLM brain: Ollama integration and command handling.

Responsibilities:
  - call_ollama()              : raw HTTP call to local Ollama instance
  - llm_worker()               : daemon thread draining ocr_queue & speech_queue
  - _handle_command_response() : mutates user_priority_list based on LLM parse
"""

import time
import requests

from . import state
from .config import OLLAMA_URL, OLLAMA_MODEL, LLM_RATE_LIMIT_SEC, LLM_ALERT_COOLDOWN_SEC
from .priority import priority_context, clear_vote_buffer_for_item
from .tts import speak
from .session import log_event, format_context_for_prompt

# ── System Prompt ──────────────────────────────────────────
SYSTEM_PROMPT = """You are a vision assistant operating within a real-time camera system.

════════════════════════════════════════
ROLE 1 — ACTIVE WATCHER (OCR alerts)
════════════════════════════════════════
You receive OCR text detected from a camera frame, the user's priority list,
a hardcoded critical safety list, and recent session context.

Your job: decide if the OCR text warrants alerting the user.

SEMANTIC MATCHING — this is critical:
Do NOT require exact word matches. Use your language understanding to judge meaning.
Examples of what this means in practice:
  - User priority "pharmacy" → also alert for: chemist, medical store, drugstore,
    medicine shop, Apollo, MedPlus, Wellness Forever, or any store clearly selling medicines.
  - User priority "hospital" → also alert for: clinic, nursing home, emergency, casualty,
    health centre, or any medical facility sign.
  - User priority "restaurant" → also alert for: café, dhaba, eatery, food court, bistro,
    or any establishment clearly serving food.
  - Critical item "wet floor" → also alert for: slippery surface, caution floor, mopped area.
The rule: if a reasonable person would consider the detected text to be the same category
as a priority item, treat it as a match and alert.

SESSION CONTEXT AWARENESS:
You will be given a log of recent events from this session.
Use this only as background — do NOT alert based on past events alone.
Only alert if the CURRENT OCR TEXT matches a priority item.
You may enrich the alert message with context (e.g. "again" or "still ahead"),
but only when the current text itself justifies an alert.

Alert format:
  - Respond with: ALERT: <your message>
  - Keep under 15 words
  - Calm, friendly, human-like tone
  - No robotic phrasing
  - If no match: respond exactly with: IGNORE

════════════════════════════════════════
ROLE 2 — INTENT PARSER (voice commands)
════════════════════════════════════════
You receive one user voice command and recent command history from this session.

Map the input to exactly one structured action.

Valid outputs:
  ADD: <item>
  REMOVE: <item>
  LIST:
  CLEAR:
  UNKNOWN: <short helpful reply>

Core behaviour:
  - Infer intent — do not rely on exact keywords.
  - "notify me / alert me / let me know / watch for / keep an eye out for / add ... to list" → ADD
  - "stop tracking / remove / don't alert for" → REMOVE
  - "what are my priorities / what am I tracking" → LIST
  - "reset / clear / delete all" → CLEAR
  - Normalise item: lowercase, singular, remove filler words, concise noun phrase.
  - Use session command history to resolve pronouns:
      "remove that" / "forget the last one" → infer from the most recent ADD in history.

Examples:
  Input: let me know when we find a pharmacy
  Output: ADD: pharmacy

  Input: alert me if you see a hospital sign
  Output: ADD: hospital

  Input: keep an eye out for bus stops
  Output: ADD: bus stop

  Input: keep an eye out for general stores
  Output: ADD: general store

  Input: add pharmacy to my list
  Output: ADD: pharmacy

  Input: add hospital to list
  Output: ADD: hospital

  Input: watch for pharmacy
  Output: ADD: pharmacy

  Input: don't track pharmacy anymore
  Output: REMOVE: pharmacy

  Input: remove that     (last command was ADD: pharmacy)
  Output: REMOVE: pharmacy

  Input: what are my priorities
  Output: LIST:

  Input: clear my list
  Output: CLEAR:

  Input: hello how are you
  Output: UNKNOWN: I can add, remove, list, or clear priority items.

Output constraints:
  Exactly one line. No markdown. No explanations. Plain text only."""


# ── Ollama caller ──────────────────────────────────────────

def call_ollama(prompt: str) -> str:
    """
    Send `prompt` to the local Ollama instance.
    Returns the model's response text, or "IGNORE" on any failure.
    """
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}",
            "stream": False,
            "options": {
                "temperature": 0.1,   # low = consistent structured responses
                "num_predict": 80,
            },
        }
        resp = requests.post(OLLAMA_URL, json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json().get("response", "IGNORE").strip()

    except requests.exceptions.ConnectionError:
        print("[Ollama] ERROR: Cannot connect. Is 'ollama serve' running?")
        return "IGNORE"
    except Exception as e:
        print(f"[Ollama Error] {e}")
        return "IGNORE"


# ── Command handler ────────────────────────────────────────

def _handle_command_response(response: str):
    """
    Parse an LLM command response and mutate user_priority_list accordingly.

    After ADD or REMOVE, clears any stale vote-buffer entries so the camera
    loop immediately starts fresh detection for the new priority item.
    """
    upper = response.upper()

    if upper.startswith("ADD:"):
        item = response[4:].strip().lower()
        state.user_priority_list.add(item)
        speak(f"Added {item} to your priority list.")
        print(f"[Priority] User list: {state.user_priority_list}")
        clear_vote_buffer_for_item(item)

    elif upper.startswith("REMOVE:"):
        item = response[7:].strip().lower()
        if item in state.user_priority_list:
            state.user_priority_list.discard(item)
            speak(f"Removed {item} from your priority list.")
            clear_vote_buffer_for_item(item)
        else:
            speak(f"{item} was not in your list.")

    elif upper.startswith("LIST:"):
        if state.user_priority_list:
            speak(f"Your priorities are: {', '.join(sorted(state.user_priority_list))}")
        else:
            speak("Your custom priority list is empty.")

    elif upper.startswith("CLEAR:"):
        state.user_priority_list.clear()
        speak("Custom priority list cleared.")

    elif upper.startswith("UNKNOWN:"):
        speak(response[8:].strip())

    else:
        speak(response)


# ── Worker thread ──────────────────────────────────────────

def llm_worker():
    """
    Daemon thread: processes voice commands (high priority) and OCR text.

    Flow:
      1. Drain speech_queue first (user voice commands are time-sensitive).
      2. Process one OCR item per LLM_RATE_LIMIT_SEC cycle.

    Note: relevance filtering is done upstream in ocr_worker(), so every
    item in ocr_queue has already passed the is_relevant_to_active_list() gate.
    """
    while state.running:
        # ── Voice commands (higher priority) ──────────────
        try:
            command = state.speech_queue.get_nowait()
            print(f"\n🎤 Command: '{command}'")
            session_ctx = format_context_for_prompt(n=8)
            prompt = (
                f"TASK: ROLE 2 — intent parsing for priority memory management.\n\n"
                f"{session_ctx}\n\n"
                f"{priority_context()}\n\n"
                f'USER_UTTERANCE: "{command}"\n'
                f"Map this to exactly one structured action."
            )
            response = call_ollama(prompt)
            print(f"[LLM → Command] {response}")
            _handle_command_response(response)
            log_event("command", f"input='{command}' → response='{response}'")
            state.speech_queue.task_done()
            continue
        except Exception:
            pass  # queue.Empty — normal

        # ── OCR alerts ────────────────────────────────────
        try:
            ocr_text = state.ocr_queue.get(timeout=0.3)
            now = time.time()

            if (now - state.last_llm_call) < LLM_RATE_LIMIT_SEC:
                # Rate limit active — drain the entire queue so stale crops
                # don't pile up and fire as new alerts later.
                state.ocr_queue.task_done()
                while not state.ocr_queue.empty():
                    try:
                        state.ocr_queue.get_nowait()
                        state.ocr_queue.task_done()
                    except Exception:
                        break
                continue

            state.last_llm_call = now
            session_ctx = format_context_for_prompt(n=10)
            prompt = (
                f"TASK: ROLE 1 — decide if this OCR text warrants an alert.\n\n"
                f"{session_ctx}\n\n"
                f"{priority_context()}\n\n"
                f'OCR TEXT from camera: "{ocr_text}"\n\n'
                f"Use semantic matching and session context. "
                f"Respond with ALERT: <message> or IGNORE."
            )
            response = call_ollama(prompt)
            print(f"[LLM → OCR] {response}")

            if response.upper().startswith("ALERT:"):
                alert_text = response[6:].strip()

                # ── Alert-level dedup ─────────────────────────────
                # Normalise: lowercase, strip punctuation for comparison
                alert_key = "".join(c for c in alert_text.lower() if c.isalnum() or c == " ").strip()
                now2 = time.time()
                last_spoken = state.last_llm_alert_times.get(alert_key, 0)
                if (now2 - last_spoken) < LLM_ALERT_COOLDOWN_SEC:
                    print(f"[LLM] Alert suppressed (cooldown): '{alert_text}'")
                else:
                    state.last_llm_alert_times[alert_key] = now2
                    speak(alert_text)
                    log_event("alert", f"ocr='{ocr_text}' → alert='{alert_text}'")

            state.ocr_queue.task_done()

        except Exception:
            continue  # queue.Empty on timeout — normal