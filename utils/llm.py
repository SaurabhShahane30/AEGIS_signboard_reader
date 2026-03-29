"""
============================================================
  llm.py — Ollama caller, system prompts, and LLM brain thread
============================================================
"""

import queue
import requests

from .config import OLLAMA_URL, OLLAMA_MODEL
from . import state
from .tts import speak
from .priority import priority_context
from .websocket_server import broadcast_event


# ─────────────────────────────────────────────────────────
#  System prompts
# ─────────────────────────────────────────────────────────

SYSTEM_PROMPT_WATCHER = """You are a vision assistant for a real-time camera system.

Given OCR text from a camera frame, decide whether it requires an alert.

If it matches or closely resembles a critical item or user priority → respond with a short, natural alert message.
Otherwise → respond: IGNORE

Rules:
- Calm, friendly, human-like phrasing
- No prefixes like "ALERT:" or "---" or role labels
- Under 15 words
- Include direction naturally if provided (e.g. "Pharmacy detected to your left.")
- Output ONLY the single alert sentence or the word IGNORE. Nothing else."""

SYSTEM_PROMPT_INTENT = """You are an intent parser for a priority memory system.

Map the user's voice command to exactly one structured action.

Valid outputs:
ADD: <item>
REMOVE: <item>
LIST:
CLEAR:
UNKNOWN: <short helpful reply>

Rules:
- Infer intent, don't require exact keywords
- "notify/alert/watch for/remember" → ADD
- "jodo/shamil karo/yaad rakh/track karo/जोड़ो/शामिल करो" → ADD
- "stop tracking/remove/don't alert" → REMOVE
- "hatao/hata do/band karo/हटाओ/हटा दो" → REMOVE
- "meri list/list dikhao/priorities batao/मेरी लिस्ट" → LIST
- "saaf karo/sab hatao/साफ करो/सब हटाओ" → CLEAR
- Normalize item: lowercase, singular, no filler words
- Extract the item noun — ignore Hindi grammar particles (ko, mein, se, ki, ka)
- Output EXACTLY one line, no markdown, no explanation

Examples:
Input: let me know when we find a pharmacy
Output: ADD: pharmacy

Input: alert me if you see a hospital
Output: ADD: hospital

Input: don't track pharmacy anymore
Output: REMOVE: pharmacy

Input: what are my priorities
Output: LIST:

Input: clear my list
Output: CLEAR:

Input: hello how are you
Output: UNKNOWN: I can add, remove, list, or clear priority items."""


# ─────────────────────────────────────────────────────────
#  Core Ollama call
# ─────────────────────────────────────────────────────────

# Keywords that mark a valid structured response line
_VALID_PREFIXES = ("ADD:", "REMOVE:", "LIST:", "CLEAR:", "UNKNOWN:", "IGNORE")


def _extract_structured_line(raw: str) -> str:
    """
    Pull the first line that starts with a known structured prefix.
    Guards against models that add preamble like 'Here is the parsed action:'
    before the actual output line.
    """
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        upper = stripped.upper()
        for prefix in _VALID_PREFIXES:
            if upper.startswith(prefix):
                return stripped
    # Nothing matched — return the raw response so the else branch in the
    # handler can speak it as a fallback rather than silently dropping it.
    return raw.strip()


def call_ollama(prompt: str, system_prompt: str = SYSTEM_PROMPT_WATCHER) -> str:
    try:
        payload = {
            "model":  OLLAMA_MODEL,
            "system": system_prompt,        # dedicated system field — keeps it out of prompt
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 60,          # slight headroom so long UNKNOWN replies aren't cut
            },
        }
        resp = requests.post(OLLAMA_URL, json=payload, timeout=15)
        resp.raise_for_status()
        raw = resp.json().get("response", "IGNORE").strip()
        return _extract_structured_line(raw)
    except requests.exceptions.ConnectionError:
        print("[Ollama] ERROR: Cannot connect. Is 'ollama serve' running?")
        return "IGNORE"
    except Exception as e:
        print(f"[Ollama Error] {e}")
        return "IGNORE"
 
 
# ─────────────────────────────────────────────────────────
#  Command response handler
# ─────────────────────────────────────────────────────────
 
def _handle_command_response(response: str):
    upper = response.upper()

    try:
        _broadcast = broadcast_event
    except Exception:
        _broadcast = lambda e: None  # noqa: E731
 
    if upper.startswith("ADD:"):
        item = response[4:].strip().lower()
        state.user_priority_list.add(item)
        speak(f"Added {item} to your priority list.")
        print(f"[Priority] User list: {state.user_priority_list}")
        _broadcast({"type": "priorities_updated", "priorities": list(state.user_priority_list)})
 
    elif upper.startswith("REMOVE:"):
        item = response[7:].strip().lower()
        if item in state.user_priority_list:
            state.user_priority_list.discard(item)
            speak(f"Removed {item} from your priority list.")
            _broadcast({"type": "priorities_updated", "priorities": list(state.user_priority_list)})
        else:
            speak(f"{item} was not in your list.")
 
    elif upper.startswith("LIST:"):
        priorities = sorted(state.user_priority_list)
        if priorities:
            speak(f"Your priorities are: {', '.join(priorities)}")
        else:
            speak("Your custom priority list is empty.")
        _broadcast({"type": "priorities_updated", "priorities": priorities})
 
    elif upper.startswith("CLEAR:"):
        state.user_priority_list.clear()
        speak("Custom priority list cleared.")
        _broadcast({"type": "priorities_updated", "priorities": []})
 
    elif upper.startswith("UNKNOWN:"):
        speak(response[8:].strip())
 
    else:
        speak(response)
 
 
# ─────────────────────────────────────────────────────────
#  LLM brain thread
# ─────────────────────────────────────────────────────────
 
def llm_worker():
    while state.running:
        try:
            command = state.speech_queue.get(timeout=0.3)
            print(f"\n🎤 Command: '{command}'")
            prompt = (
                f'USER_UTTERANCE: "{command}"\n'
                f"{priority_context()}"
            )
            response = call_ollama(prompt, system_prompt=SYSTEM_PROMPT_INTENT)
            print(f"[LLM -> Command] {response}")
            _handle_command_response(response)
            state.speech_queue.task_done()
        except queue.Empty:
            continue