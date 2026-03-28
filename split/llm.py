"""
llm.py — LLM brain: Ollama integration and command handling.

Responsibilities:
  - call_ollama()       : raw HTTP call to local Ollama instance
  - llm_worker()        : daemon thread draining ocr_queue & speech_queue
  - _handle_command_response() : mutates user_priority_list based on LLM parse
"""

import time
import requests
import state
from config import OLLAMA_URL, OLLAMA_MODEL, LLM_RATE_LIMIT_SEC
from priority import priority_context
from tts import speak

# ── System Prompt ──────────────────────────────────────────
SYSTEM_PROMPT = """You are a vision assistant operating within a real-time camera system.

ROLE 1 — ACTIVE WATCHER:
Given OCR text from a camera frame, decide whether it requires an alert.

If it matches or closely resembles a critical item or user priority → respond with a short, natural alert message.
Otherwise → respond: IGNORE

Alert style:
Use calm, friendly, human-like phrasing
Avoid rigid or robotic wording
Do not use prefixes like "ALERT:"
Keep under 15 words

ROLE 2 — INTENT PARSER FOR PRIORITY MEMORY:

You receive one user voice command at a time.
Map the natural language input to exactly one structured action.

Valid outputs only:
ADD: <item>
REMOVE: <item>
LIST:
CLEAR:
UNKNOWN: <short helpful reply>

Core behavior:
Infer intent, do not rely on exact keywords.
If user expresses "notify me / alert me / let me know / watch for / remember / keep an eye out for" an object or place, treat it as ADD.
If user says stop tracking / remove / don't alert for something, treat it as REMOVE.
If user asks what is being tracked, treat as LIST.
If user asks to reset/delete all custom tracked items, treat as CLEAR.
Normalize item: lowercase, singular when appropriate, remove filler words and articles, keep a concise noun phrase.

High-priority examples:
Input: let me know when we find a pharmacy
Output: ADD: pharmacy

Input: alert me if you see a hospital sign
Output: ADD: hospital

Input: keep an eye out for bus stops
Output: ADD: bus stop

Input: don't track pharmacy anymore
Output: REMOVE: pharmacy

Input: what are my priorities
Output: LIST:

Input: clear my list
Output: CLEAR:

Input: hello how are you
Output: UNKNOWN: I can add, remove, list, or clear priority items.

Output constraints:
Exactly one line
No markdown
No explanations
No extra text beyond the required format
Plain text only. Single-line response."""


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
                "num_predict": 60,    # short outputs only
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
    Also triggers a spoken confirmation.
    """
    upper = response.upper()

    if upper.startswith("ADD:"):
        item = response[4:].strip().lower()
        state.user_priority_list.add(item)
        speak(f"Added {item} to your priority list.")
        print(f"[Priority] User list: {state.user_priority_list}")

    elif upper.startswith("REMOVE:"):
        item = response[7:].strip().lower()
        if item in state.user_priority_list:
            state.user_priority_list.discard(item)
            speak(f"Removed {item} from your priority list.")
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
    """
    while state.running:
        # ── Voice commands (higher priority) ──────────────
        try:
            command = state.speech_queue.get_nowait()
            print(f"\n🎤 Command: '{command}'")
            prompt = (
                f'USER_UTTERANCE: "{command}"\n'
                "TASK: infer intent for priority memory management using ROLE 2 rules.\n"
                f"{priority_context()}"
            )
            response = call_ollama(prompt)
            print(f"[LLM → Command] {response}")
            _handle_command_response(response)
            state.speech_queue.task_done()
            continue
        except Exception:
            pass  # queue.Empty — normal

        # ── OCR alerts ────────────────────────────────────
        try:
            ocr_text = state.ocr_queue.get(timeout=0.3)
            now = time.time()

            if (now - state.last_llm_call) < LLM_RATE_LIMIT_SEC:
                state.ocr_queue.task_done()
                continue

            state.last_llm_call = now
            prompt = (
                f'OCR TEXT from camera: "{ocr_text}"\n'
                f"{priority_context()}\n"
                "Should I alert the user?"
            )
            response = call_ollama(prompt)
            print(f"[LLM → OCR] {response}")

            if response.upper().startswith("ALERT:"):
                speak(response[6:].strip())

            state.ocr_queue.task_done()

        except Exception:
            continue  # queue.Empty on timeout — normal
