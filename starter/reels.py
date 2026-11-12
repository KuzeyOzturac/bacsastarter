"""Browser and screen automation for Instagram Reels."""

import time
import webbrowser
from pathlib import Path

import cv2
import numpy as np
import pyautogui

from config import (
    MUTE_MATCH_THRESHOLD,
    MUTE_SEARCH_INTERVAL,
    MUTE_SEARCH_TIMEOUT,
    MUTE_TEMPLATE_SCALES,
    NEXT_REEL_LOAD_SECONDS,
    REELS_INITIAL_LOAD_SECONDS,
    REELS_URL,
    REEL_SCROLL_AMOUNT,
)

MUTE_ICON_PATH = Path(__file__).with_name("mute_icon.png")
WINDOW_TITLE = "Focus Detector"


def find_and_click_mute_icon():
    if not MUTE_ICON_PATH.exists():
        print(f"ERROR: mute icon was not found: {MUTE_ICON_PATH}")
        return False
    template = cv2.imread(str(MUTE_ICON_PATH), cv2.IMREAD_COLOR)
    if template is None:
        print(f"ERROR: OpenCV could not read {MUTE_ICON_PATH}")
        return False

    screen_width, screen_height = pyautogui.size()
    deadline = time.monotonic() + MUTE_SEARCH_TIMEOUT
    while time.monotonic() < deadline:
        screenshot = cv2.cvtColor(np.asarray(pyautogui.screenshot()), cv2.COLOR_RGB2BGR)
        image_height, image_width = screenshot.shape[:2]
        scale_x, scale_y = image_width / screen_width, image_height / screen_height
        left, right = int(image_width * 0.35), int(image_width * 0.66)
        top, bottom = int(image_height * 0.14), int(image_height * 0.86)
        region = screenshot[top:bottom, left:right]
        if region.size == 0:
            print("ERROR: Reel search region is empty.")
            return False

        best_score = -1.0
        best_match = None
        for scale in MUTE_TEMPLATE_SCALES:
            width = round(template.shape[1] * scale)
            height = round(template.shape[0] * scale)
            if min(width, height) < 5 or width >= region.shape[1] or height >= region.shape[0]:
                continue
            interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
            resized = cv2.resize(template, (width, height), interpolation=interpolation)
            result = cv2.matchTemplate(region, resized, cv2.TM_CCOEFF_NORMED)
            _, score, _, location = cv2.minMaxLoc(result)
            if score > best_score:
                best_score = score
                best_match = (location, width, height)

        if best_match is not None and best_score >= MUTE_MATCH_THRESHOLD:
            # STEP 6: Convert the match to screen coordinates and click its center.
            pass
        time.sleep(MUTE_SEARCH_INTERVAL)

    print("Mute icon was not found.")
    return False


def shrink_and_front_capture_window():
    screen_width, screen_height = pyautogui.size()
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, screen_width // 3, screen_height // 3)
    cv2.moveWindow(WINDOW_TITLE, 0, 0)
    try:
        cv2.setWindowProperty(WINDOW_TITLE, cv2.WND_PROP_TOPMOST, 1)
    except (cv2.error, AttributeError) as error:
        print(f"Could not set capture window always on top: {error}")
    cv2.waitKey(1)
    time.sleep(0.2)
    cv2.moveWindow(WINDOW_TITLE, 0, 0)


def open_reels():
    print("Opening Instagram Reels...")
    if not webbrowser.open(REELS_URL):
        print("Could not open Instagram Reels in a browser.")
        return False
    time.sleep(REELS_INITIAL_LOAD_SECONDS)
    if not find_and_click_mute_icon():
        return False
    time.sleep(0.3)
    shrink_and_front_capture_window()
    return True


def next_reel():
    print("Scrolling to next Reel...")
    # STEP 7: Scroll by REEL_SCROLL_AMOUNT.
    time.sleep(NEXT_REEL_LOAD_SECONDS)
