"""
============================================================
  direction.py — Spatial direction of detected signs
============================================================

The frame is divided into three horizontal thirds:
  left zone   : box_center_ratio < 0.33
  ahead zone  : 0.33 <= box_center_ratio <= 0.66
  right zone  : box_center_ratio > 0.66

If the box spans two zones (e.g. center is right but left edge
reaches into the ahead zone), a blended direction is reported.
"""


def get_direction(x1: int, x2: int, frame_width: int) -> str:
    """
    Compute the relative direction of a detected sign.

    Parameters
    ----------
    x1, x2      : bounding box left and right pixel x-coordinates
    frame_width : width of the full camera frame in pixels

    Returns
    -------
    One of: "to your left" | "ahead and to your left" | "straight ahead"
            | "ahead and to your right" | "to your right"
    """
    if frame_width <= 0:
        return "straight ahead"

    center_ratio = ((x1 + x2) / 2) / frame_width
    left_ratio   = x1 / frame_width
    right_ratio  = x2 / frame_width

    LEFT_THRESH  = 0.33
    RIGHT_THRESH = 0.66

    in_left   = left_ratio  < LEFT_THRESH
    in_right  = right_ratio > RIGHT_THRESH
    in_center = LEFT_THRESH <= center_ratio <= RIGHT_THRESH

    if in_left and in_right:
        return "straight ahead"

    if in_left and in_center:
        return "ahead and to your left"

    if in_right and in_center:
        return "ahead and to your right"

    if center_ratio < LEFT_THRESH:
        return "to your left"

    if center_ratio > RIGHT_THRESH:
        return "to your right"

    return "straight ahead"