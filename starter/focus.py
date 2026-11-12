"""Calibration and one-shot focus-loss event detection."""

from collections import deque
from dataclasses import dataclass

from config import CALIBRATION_SECONDS, DISTRACTION_SECONDS, GAZE_SMOOTHING_FRAMES
from vision import gaze_direction


@dataclass(frozen=True)
class FocusStatus:
    text: str
    color: tuple[int, int, int]
    gaze: str = "unknown"
    smooth_horizontal: float | None = None
    smooth_vertical: float | None = None
    trigger: bool = False


class FocusTracker:
    def __init__(self):
        self.horizontal = deque(maxlen=GAZE_SMOOTHING_FRAMES)
        self.vertical = deque(maxlen=GAZE_SMOOTHING_FRAMES)
        self.calibration_horizontal = []
        self.calibration_vertical = []
        self.calibration_start = None
        self.center = None
        self.distracted_since = None
        self.triggered = False

    def update(self, now, measurement=None):
        """measurement is (head, closed, horizontal, vertical), or None if no face."""
        if measurement is None:
            self.horizontal.clear()
            self.vertical.clear()
            if self.center is None:
                self.calibration_start = None
                self.calibration_horizontal.clear()
                self.calibration_vertical.clear()
        else:
            head, closed, horizontal, vertical = measurement
            self.horizontal.append(horizontal)
            self.vertical.append(vertical)
            # STEP 4: Replace each reading with the mean of its recent values.

        if self.center is None:
            self.distracted_since = None
            self.triggered = False
            if measurement is not None:
                if self.calibration_start is None:
                    self.calibration_start = now
                self.calibration_horizontal.append(horizontal)
                self.calibration_vertical.append(vertical)
                elapsed = now - self.calibration_start
                if elapsed >= CALIBRATION_SECONDS:
                    # STEP 4: Save the average calibration readings as self.center.
                    self.center = (0.5, 0.5)
            else:
                elapsed = 0
            if self.center is None:
                remaining = max(0, CALIBRATION_SECONDS - elapsed)
                h = horizontal if measurement is not None else None
                v = vertical if measurement is not None else None
                return FocusStatus(f"CALIBRATING - LOOK AT SCREEN ({remaining:.1f}s)", (255, 255, 0), "calibrating", h, v)

        gaze = "unknown" if measurement is None else gaze_direction(horizontal, vertical, *self.center)
        if measurement is None:
            reason = "Face not visible"
        elif head != "center":
            reason = f"Head turned {head}"
        elif closed:
            reason = "Eyes closed"
        else:
            reason = {"side": "Eyes looking away", "up": "Looking up", "down": "Looking down"}.get(gaze)

        h = horizontal if measurement is not None else None
        v = vertical if measurement is not None else None
        if reason is None:
            self.distracted_since = None
            self.triggered = False
            return FocusStatus("FOCUSED", (0, 200, 0), gaze, h, v)

        if self.distracted_since is None:
            self.distracted_since = now
        elapsed = now - self.distracted_since
        if elapsed < DISTRACTION_SECONDS:
            return FocusStatus(f"Possible distraction: {reason} ({DISTRACTION_SECONDS - elapsed:.1f}s)", (0, 165, 255), gaze, h, v)
        # STEP 5: Trigger once per continuous focus loss.
        trigger = False
        return FocusStatus(f"FOCUS LOST: {reason}", (0, 0, 255), gaze, h, v, trigger)
