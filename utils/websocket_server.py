"""
============================================================
  websocket_server.py — WebSocket server for Flutter client
============================================================

Incoming messages from Flutter:
  {type: "frame",         data: <base64 JPEG>}   — video frame (flutter source)
  {type: "voice_command", text: <string>}         — voice command (flutter source)

Outgoing messages to Flutter:
  {type: "frame",              data: <base64 JPEG>, detections: [...]}
  {type: "alert",              text: <string>, timestamp: <ISO>}
  {type: "init",               priorities: [...], critical: [...]}
  {type: "priorities_updated", priorities: [...]}
"""

import json
import asyncio
import base64
import queue
import threading
import socket

import cv2
import numpy as np
import websockets

from .config import WS_HOST, WS_PORT, INPUT_SOURCE, CRITICAL_LIST
from . import state


# ─────────────────────────────────────────────────────────
#  broadcast_event — called from any thread
# ─────────────────────────────────────────────────────────

def broadcast_event(event: dict):
    """Push an event dict to all connected Flutter clients.

    Uses put_nowait so callers (camera thread, TTS thread, etc.) are never
    blocked if no Flutter client is connected and the queue fills up.
    The oldest event is silently dropped when the queue is at capacity.
    """
    try:
        state.ws_event_queue.put_nowait(event)
    except Exception:
        # Queue full (no client draining it) — drop the event silently.
        pass


# ─────────────────────────────────────────────────────────
#  WebSocket handler — one per connected Flutter client
# ─────────────────────────────────────────────────────────

async def ws_handler(websocket):
    print(f"[WS] Client connected: {websocket.remote_address}")
    with state.ws_clients_lock:
        state.ws_clients.add(websocket)

    # Send initial state so Flutter can populate its priorities tab immediately
    await websocket.send(json.dumps({
        "type":       "init",
        "priorities": list(state.user_priority_list),
        "critical":   list(CRITICAL_LIST),
    }))

    try:
        async for message in websocket:
            try:
                msg = json.loads(message)
                msg_type = msg.get("type")

                # ── Incoming video frame from Flutter ────────────
                if msg_type == "frame" and INPUT_SOURCE == "flutter":
                    frame_bytes = base64.b64decode(msg["data"])
                    nparr = np.frombuffer(frame_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        try:
                            state.ws_frame_queue.put_nowait(frame)
                        except Exception:
                            pass  # queue full — drop frame

                # ── Incoming voice command from Flutter ──────────
                elif msg_type == "voice_command":
                    text = msg.get("text", "").strip()
                    if text:
                        print(f"[WS] Voice command from Flutter: '{text}'")
                        state.speech_queue.put(text)

            except Exception as e:
                print(f"[WS] Message error: {e}")

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        with state.ws_clients_lock:
            state.ws_clients.discard(websocket)
        print(f"[WS] Client disconnected: {websocket.remote_address}")


# ─────────────────────────────────────────────────────────
#  Event broadcaster — drains ws_event_queue → all clients
# ─────────────────────────────────────────────────────────

async def ws_event_broadcaster():
    loop = asyncio.get_event_loop()

    while True:
        try:
            event = await loop.run_in_executor(
                None, state.ws_event_queue.get, True, 0.1
            )
        except queue.Empty:
            # Timeout with no events — expected, just loop again
            await asyncio.sleep(0.05)
            continue
        except Exception as e:
            print(f"[WS Broadcaster] Unexpected error on queue.get: {e}")
            await asyncio.sleep(0.05)
            continue

        with state.ws_clients_lock:
            clients = set(state.ws_clients)

        if not clients:
            state.ws_event_queue.task_done()
            continue

        msg  = json.dumps(event)
        dead = set()
        for ws in clients:
            try:
                await ws.send(msg)
            except Exception:
                dead.add(ws)

        if dead:
            with state.ws_clients_lock:
                state.ws_clients -= dead

        state.ws_event_queue.task_done()


# ─────────────────────────────────────────────────────────
#  Server entry point
# ─────────────────────────────────────────────────────────

async def ws_main():
    print(f"[WS] Server starting on ws://{WS_HOST}:{WS_PORT}")
    async with websockets.serve(ws_handler, WS_HOST, WS_PORT):
        await ws_event_broadcaster()


def start_ws_server():
    """Run in a daemon thread — creates its own event loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(ws_main())


# ─────────────────────────────────────────────────────────
#  Helper — print local IP for Flutter config
# ─────────────────────────────────────────────────────────

def print_connection_info():
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "YOUR_LAPTOP_IP"
    print(f"\n{'='*60}")
    print(f"  WebSocket ready  →  ws://{local_ip}:{WS_PORT}")
    print(f"  Set in Flutter:      SERVER_IP = '{local_ip}'")
    print(f"  Input source:        {INPUT_SOURCE.upper()}")
    print(f"{'='*60}\n")