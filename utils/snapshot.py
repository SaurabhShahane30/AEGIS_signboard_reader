"""
============================================================
  snapshot.py — Save annotated alert frames to disk
============================================================
"""

import os
import cv2
import numpy as np
from datetime import datetime

from .config import ALERT_SNAPSHOT_DIR


def save_alert_snapshot(
    annotated_frame: np.ndarray,
    box: tuple,
    sign_text: str,
    direction: str,
) -> str | None:
    """
    Save a full annotated camera frame to disk when an alert fires.

    The sign bounding box is highlighted with a cyan rect and a label
    strip. The filename encodes the timestamp and a sanitised form of
    the sign text so alert_snapshots/ can be reviewed chronologically.

    Parameters
    ----------
    annotated_frame : full frame already annotated by YOLO (numpy BGR)
    box             : (x1, y1, x2, y2) of the detected sign
    sign_text       : OCR text that triggered the alert
    direction       : direction string from get_direction()

    Returns
    -------
    Saved filepath string, or None on failure.
    """
    try:
        os.makedirs(ALERT_SNAPSHOT_DIR, exist_ok=True)

        frame = annotated_frame.copy()
        x1, y1, x2, y2 = box
        h, w = frame.shape[:2]

        # ── Highlight box ──────────────────────────────────────────
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

        # Vertical midline reference
        mid_x = w // 2
        cv2.line(frame, (mid_x, 0), (mid_x, h), (200, 200, 200), 1, cv2.LINE_AA)

        # ── Label strip ────────────────────────────────────────────
        label      = f"{sign_text}  |  {direction}"
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.65
        thickness  = 2
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        strip_y1 = max(0, y1 - th - baseline - 8)
        strip_y2 = max(th + baseline + 8, y1)
        strip_x2 = min(w, x1 + tw + 10)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, strip_y1), (strip_x2, strip_y2), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(
            frame, label,
            (x1 + 5, strip_y2 - baseline - 2),
            font, font_scale,
            (0, 255, 255), thickness, cv2.LINE_AA,
        )

        # ── Timestamp watermark ────────────────────────────────────
        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(
            frame, ts,
            (10, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (180, 180, 180), 1, cv2.LINE_AA,
        )

        # ── Save ───────────────────────────────────────────────────
        safe_text = "".join(
            c if c.isalnum() or c in "_ " else "_" for c in sign_text
        )
        safe_text = safe_text.replace(" ", "_")[:30]
        filename  = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{safe_text}.jpg"
        filepath  = os.path.join(ALERT_SNAPSHOT_DIR, filename)

        cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"[Snapshot] Saved: {filepath}")
        return filepath

    except Exception as e:
        print(f"[Snapshot Error] {e}")
        return None