"""
============================================================
  llm.py — Ollama caller, system prompts, and LLM brain thread
============================================================
"""

import time
import queue
import requests

from .config import OLLAMA_URL, OLLAMA_MODEL
from . import state
from .tts import speak
from .priority import priority_context


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
- "stop tracking/remove/don't alert" → REMOVE
- Normalize item: lowercase, singular, no filler words
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

def call_ollama(prompt: str, system_prompt: str = SYSTEM_PROMPT_WATCHER) -> str:
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": f"{system_prompt}\n\n{prompt}",
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 40,
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
 
 
# ─────────────────────────────────────────────────────────
#  Command response handler
# ─────────────────────────────────────────────────────────
 
def _handle_command_response(response: str):
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
 
            state.ocr_queue.task_done()
 
        except queue.Empty:
            continue