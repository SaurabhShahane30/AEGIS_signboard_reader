"""
============================================================
  scene.py — Rolling signboard scene for active / conversational vision
============================================================
"""

import time
from collections import deque

from . import state


def _depth_hint(x1: int, y1: int, x2: int, y2: int, frame_w: int, frame_h: int) -> str:
    """
    No real depth sensor — infer coarse distance from how large the sign is in frame.
    Passed to the LLM as a hint only (no claim of metric accuracy).
    """
    box_area = max(0, (x2 - x1) * (y2 - y1))
    frame_area = max(1, frame_w * frame_h)
    cov = box_area / frame_area

    if cov > 0.12:
        return "appears very close (large in frame)"
    if cov > 0.045:
        return "at a moderate distance"
    return "farther away (small in frame)"


def record_signboard(
    text: str,
    direction: str,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    frame_w: int,
    frame_h: int,
) -> None:
    """
    Append one OCR line from a sign crop. Dedupe rapid repeats of same text+direction.
    """
    text = text.strip()
    if len(text) < 2:
        return

    depth = _depth_hint(x1, y1, x2, y2, frame_w, frame_h)
    now = time.time()
    key = (text.lower(), direction)

    with state.lock:
        if state.scene_signboards:
            last = state.scene_signboards[-1]
            if (
                (last["text"].lower(), last["direction"]) == key
                and (now - last["ts"]) < 2.0
            ):
                last["ts"] = now
                last["depth"] = depth
                return

        state.scene_signboards.append(
            {
                "text": text,
                "direction": direction,
                "depth": depth,
                "ts": now,
            }
        )


def get_scene_snapshot(max_items: int = 14):
    """
    Recent unique (text, direction) observations, newest first, for LLM context.
    """
    with state.lock:
        items = list(state.scene_signboards)

    items.sort(key=lambda x: -x["ts"])
    seen = set()
    out = []
    for it in items:
        k = (it["text"].lower().strip(), it["direction"])
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
        if len(out) >= max_items:
            break
    return out


def scene_context_lines() -> str:
    """Human-readable block for the vision LLM prompt."""
    snap = get_scene_snapshot()
    if not snap:
        return ""
    lines = []
    for it in snap:
        lines.append(
            f'- Text: "{it["text"]}" | direction: {it["direction"]} | distance_hint: {it["depth"]}'
        )
    return "\n".join(lines)
