"""Run one workshop checkpoint: python checks.py 1 (through 7)."""

import argparse
import importlib
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / os.environ.get("BACSA_CHECK_SOURCE", "starter")))

from focus import FocusTracker
from vision import (
    calculate_gaze,
    eyes_are_closed,
    head_direction,
)


def face_points():
    points = [SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]
    for index, x, y in (
        (234, 0.2, 0.5), (454, 0.8, 0.5), (1, 0.5, 0.5),
        (33, 0.2, 0.5), (133, 0.4, 0.5),
        (362, 0.6, 0.5), (263, 0.8, 0.5),
        (159, 0.3, 0.4), (145, 0.3, 0.6),
        (386, 0.7, 0.4), (374, 0.7, 0.6),
    ):
        points[index] = SimpleNamespace(x=x, y=y)
    for index in range(468, 473):
        points[index] = SimpleNamespace(x=0.3, y=0.5)
    for index in range(473, 478):
        points[index] = SimpleNamespace(x=0.7, y=0.5)
    return points


def reels_with_fake_desktop():
    """Import Reel logic without needing a real desktop during checks."""
    fake_pyautogui = Mock()
    with patch.dict(sys.modules, {"pyautogui": fake_pyautogui}):
        sys.modules.pop("reels", None)
        return importlib.import_module("reels")


class Step1(unittest.TestCase):
    def test_head_turns(self):
        points = face_points()
        self.assertEqual(head_direction(points), "center")
        points[1].x = 0.3
        self.assertEqual(head_direction(points), "left")
        points[1].x = 0.7
        self.assertEqual(head_direction(points), "right")

    def test_head_sequence_requires_order_and_steady_pose(self):
        from head_demo import HeadSequence

        sequence = HeadSequence()
        sequence.update("left", 0)
        self.assertEqual(sequence.index, 0)
        sequence.update("center", 1)
        sequence.update("center", 1.7)
        self.assertEqual(sequence.index, 1)
        sequence.update("right", 2)
        self.assertEqual(sequence.index, 1)
        sequence.update("left", 3)
        sequence.update("left", 3.7)
        sequence.update("right", 4)
        sequence.update("right", 4.7)
        sequence.update("center", 5)
        sequence.update("center", 5.7)
        self.assertTrue(sequence.complete)


class Step2(unittest.TestCase):
    def test_eyes_closed_and_open(self):
        points = face_points()
        self.assertFalse(eyes_are_closed(points, 1000, 1000))
        for index in (159, 386):
            points[index].y = 0.49
        for index in (145, 374):
            points[index].y = 0.51
        self.assertTrue(eyes_are_closed(points, 1000, 1000))

    def test_three_blinks_require_closing_and_reopening(self):
        from blink_demo import BlinkSequence

        sequence = BlinkSequence()
        sequence.update(True, 0)
        self.assertEqual(sequence.count, 0)
        for start in (1.0, 2.0, 3.0):
            sequence.update(False, start)
            sequence.update(False, start + 0.15)
            sequence.update(True, start + 0.20)
            sequence.update(True, start + 0.28)
            self.assertEqual(sequence.count, int(start - 1))
            sequence.update(False, start + 0.35)
            self.assertEqual(sequence.count, int(start))
        self.assertTrue(sequence.complete)


class Step3(unittest.TestCase):
    def test_gaze_ratios_change_with_iris(self):
        points = face_points()
        horizontal, vertical, _, _ = calculate_gaze(points, 1000, 1000)
        self.assertAlmostEqual(horizontal, 0.5)
        self.assertAlmostEqual(vertical, 0.5)
        for index in range(468, 473):
            points[index].x = 0.35
        for index in range(473, 478):
            points[index].x = 0.75
        horizontal, _, _, _ = calculate_gaze(points, 1000, 1000)
        self.assertGreater(horizontal, 0.7)


class Step4(unittest.TestCase):
    def test_calibration_and_smoothing(self):
        tracker = FocusTracker()
        steady = ("center", False, 0.2, 0.3)
        tracker.update(0, steady)
        tracker.update(1, steady)
        calibrated = tracker.update(3, steady)
        self.assertAlmostEqual(tracker.center[0], 0.2)
        self.assertAlmostEqual(tracker.center[1], 0.3)
        self.assertAlmostEqual(calibrated.smooth_horizontal, 0.2)
        changed = tracker.update(4, ("center", False, 0.8, 0.3))
        self.assertAlmostEqual(changed.smooth_horizontal, 0.35)


class Step5(unittest.TestCase):
    def test_one_trigger_until_focus_returns(self):
        tracker = FocusTracker()
        steady = ("center", False, 0.5, 0.5)
        away = ("left", False, 0.5, 0.5)
        tracker.update(0, steady)
        tracker.update(3, steady)
        self.assertFalse(tracker.update(4, away).trigger)
        self.assertTrue(tracker.update(5, away).trigger)
        self.assertFalse(tracker.update(6, away).trigger)
        tracker.update(7, steady)
        tracker.update(8, away)
        self.assertTrue(tracker.update(9, away).trigger)


class Step6(unittest.TestCase):
    def test_mute_match_clicks_template_center(self):
        import cv2
        import numpy as np
        from PIL import Image
        reels = reels_with_fake_desktop()

        template = cv2.imread(str(reels.MUTE_ICON_PATH))
        self.assertIsNotNone(template, "starter/mute_icon.png is missing")
        height, width = template.shape[:2]
        self.assertLess(width, 300)
        self.assertLess(height, 300)
        screenshot = np.zeros((1000, 1000, 3), dtype=np.uint8)
        screenshot[450:450 + height, 450:450 + width] = template
        image = Image.fromarray(cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB))
        with patch.object(reels.pyautogui, "size", return_value=(1000, 1000)), \
             patch.object(reels.pyautogui, "screenshot", return_value=image), \
             patch.object(reels.pyautogui, "moveTo") as move, \
             patch.object(reels.pyautogui, "click") as click, \
             patch.object(reels.time, "sleep"):
            self.assertTrue(reels.find_and_click_mute_icon())
        click.assert_called_once()
        x, y = move.call_args.args[:2]
        self.assertAlmostEqual(x, 450 + width / 2, delta=2)
        self.assertAlmostEqual(y, 450 + height / 2, delta=2)


class Step7(unittest.TestCase):
    def test_next_reel_scrolls_once(self):
        reels = reels_with_fake_desktop()

        with patch.object(reels.pyautogui, "scroll") as scroll, \
             patch.object(reels.time, "sleep"):
            reels.next_reel()
        scroll.assert_called_once_with(reels.REEL_SCROLL_AMOUNT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check one BACSA workshop step")
    parser.add_argument("step", type=int, choices=range(1, 8))
    parser.add_argument("--webcam", action="store_true", help="run the live Step 1 or 2 webcam check")
    args = parser.parse_args()
    if args.webcam and args.step not in (1, 2):
        parser.error("--webcam is available for Steps 1 and 2 only")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(globals()[f"Step{args.step}"])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
    if args.webcam:
        if args.step == 1:
            from head_demo import run

            print("Follow Center, Left, Right, Center on screen. Press Q when finished.")
        else:
            from blink_demo import run

            print("Blink three times on screen. Press Q when finished.")
        completed = run()
        print(f"Step {args.step} webcam task {'complete' if completed else 'incomplete'}.")
        sys.exit(0 if completed else 1)
