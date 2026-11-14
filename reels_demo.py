"""Live Step 6 check that opens Reels and clicks a matched mute icon."""

import importlib
import sys
import time
import webbrowser
from unittest.mock import patch

from config import MUTE_MATCH_THRESHOLD, REELS_INITIAL_LOAD_SECONDS, REELS_URL


def draw_result(cv2, np, screenshot, click_position, clicked, screen_size):
    """Show the searched screen region and where the actual click occurred."""
    frame = cv2.cvtColor(np.asarray(screenshot), cv2.COLOR_RGB2BGR)
    height, width = frame.shape[:2]
    screen_width, screen_height = screen_size
    left, right = round(width * 0.35), round(width * 0.66)
    top, bottom = round(height * 0.14), round(height * 0.86)
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 180, 50), 4)

    if click_position is not None:
        x = round(click_position[0] * width / screen_width)
        y = round(click_position[1] * height / screen_height)
        cv2.drawMarker(frame, (x, y), (0, 255, 0),
                       cv2.MARKER_CROSS, 42, 4, cv2.LINE_AA)

    scale = min(1000 / width, 600 / height)
    frame = cv2.resize(frame, (round(width * scale), round(height * scale)))
    panel = np.full((frame.shape[0], 350, 3), (30, 30, 30), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    white, green = (240, 240, 240), (80, 220, 90)
    cv2.putText(panel, "STEP 6: LIVE REELS", (18, 40), font, 0.64, white, 2, cv2.LINE_AA)
    cv2.putText(panel, "Blue: searched area", (20, 100), font, 0.52, white, 1, cv2.LINE_AA)
    cv2.putText(panel, "Green: click point", (20, 138), font, 0.52, white, 1, cv2.LINE_AA)
    cv2.putText(panel, f"Match threshold: {MUTE_MATCH_THRESHOLD:.2f}",
                (20, 190), font, 0.52, white, 1, cv2.LINE_AA)
    if clicked:
        cv2.putText(panel, "TASK COMPLETE!", (20, 255), font, 0.68, green, 2, cv2.LINE_AA)
        cv2.putText(panel, "Mute candidate clicked", (20, 290), font, 0.46, white, 1, cv2.LINE_AA)
        cv2.putText(panel, "Check Reel audio", (20, 325), font, 0.46, white, 1, cv2.LINE_AA)
    else:
        cv2.putText(panel, "ICON NOT FOUND", (20, 255), font, 0.68, (0, 130, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, "Check browser layout", (20, 290), font, 0.46, white, 1, cv2.LINE_AA)
    cv2.putText(panel, "Press Q to quit", (20, frame.shape[0] - 24),
                font, 0.46, white, 1, cv2.LINE_AA)
    return np.hstack((frame, panel))


def run():
    """Open Reels, perform the real icon click, and show what was searched."""
    import cv2
    import numpy as np
    import pyautogui

    # The preceding unit check imports reels with a fake desktop module.
    sys.modules.pop("reels", None)
    reels = importlib.import_module("reels")
    if not webbrowser.open(REELS_URL):
        print("Could not open Instagram Reels in a browser.")
        return False
    time.sleep(REELS_INITIAL_LOAD_SECONDS)

    last_screenshot = None
    click_position = None
    original_screenshot = pyautogui.screenshot
    original_click = pyautogui.click

    def record_screenshot(*args, **kwargs):
        nonlocal last_screenshot
        last_screenshot = original_screenshot(*args, **kwargs)
        return last_screenshot

    def record_click(*args, **kwargs):
        nonlocal click_position
        click_position = pyautogui.position()
        return original_click(*args, **kwargs)

    with patch.object(pyautogui, "screenshot", side_effect=record_screenshot), \
         patch.object(pyautogui, "click", side_effect=record_click):
        clicked = reels.find_and_click_mute_icon()

    if last_screenshot is None:
        print("No screenshot was captured. Check the mute icon file and screen permissions.")
        return False

    view = draw_result(cv2, np, last_screenshot, click_position, clicked,
                       pyautogui.size())
    try:
        while True:
            cv2.imshow("BACSA Step 6 Live Check", view)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
    return clicked
