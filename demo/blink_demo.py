"""Live webcam checkpoint for the Step 2 eye-closure exercise."""

import time

from config import EYE_CLOSED_THRESHOLD
from vision import (
    LEFT_EYE_BOTTOM,
    LEFT_EYE_INNER,
    LEFT_EYE_OUTER,
    LEFT_EYE_TOP,
    RIGHT_EYE_BOTTOM,
    RIGHT_EYE_INNER,
    RIGHT_EYE_OUTER,
    RIGHT_EYE_TOP,
    eye_aspect_ratio,
    eyes_are_closed,
    landmark_xy
)


BLINKS_REQUIRED = 3
OPEN_CONFIRM_SECONDS = 0.10
MIN_CLOSED_SECONDS = 0.04
MAX_CLOSED_SECONDS = 1.5


class BlinkSequence:
    """Count separate open-closed-open cycles while a face stays visible."""

    def __init__(self):
        self.count = 0
        self.phase = "confirm_open"
        self.open_since = None
        self.closed_since = None

    @property
    def complete(self):
        return self.count == BLINKS_REQUIRED

    def update(self, closed, now):
        if self.complete:
            return
        if closed is None:
            self.phase = "confirm_open"
            self.open_since = None
            self.closed_since = None
            return

        if self.phase == "confirm_open":
            if closed:
                self.open_since = None
            elif self.open_since is None:
                self.open_since = now
            elif now - self.open_since >= OPEN_CONFIRM_SECONDS:
                self.phase = "await_closed"
        elif self.phase == "await_closed":
            if closed:
                self.closed_since = now
                self.phase = "await_reopen"
        elif self.phase == "await_reopen" and not closed:
            duration = now - self.closed_since
            if MIN_CLOSED_SECONDS <= duration <= MAX_CLOSED_SECONDS:
                self.count += 1
            self.phase = "confirm_open"
            self.open_since = now
            self.closed_since = None


def draw_eye_landmarks(cv2, frame, landmarks):
    """Mark the four points used to measure each eye's height and width."""
    height, width = frame.shape[:2]
    eyes = (
        ((LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_EYE_OUTER, LEFT_EYE_INNER), (255, 255, 0)),
        ((RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_EYE_OUTER, RIGHT_EYE_INNER), (255, 0, 255))
    )
    for (top, bottom, outer, inner), color in eyes:
        positions = [landmark_xy(landmarks, index, width, height)
                     for index in (top, bottom, outer, inner)]
        cv2.line(frame, positions[0], positions[1], color, 2, cv2.LINE_AA)
        cv2.line(frame, positions[2], positions[3], color, 2, cv2.LINE_AA)
        for point in positions:
            cv2.circle(frame, point, 5, color, -1, cv2.LINE_AA)


def draw_panel(cv2, panel, sequence, closed, left_ratio, right_ratio):
    """Show blink progress, eye ratios, and the closure threshold."""
    white = (240, 240, 240)
    green = (80, 220, 90)
    amber = (40, 200, 255)
    gray = (130, 130, 130)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(panel, "STEP 2: BLINKS", (18, 38), font, 0.68, white, 2, cv2.LINE_AA)
    for index in range(BLINKS_REQUIRED):
        if index < sequence.count:
            color, marker = green, "DONE"
        elif index == sequence.count:
            color, marker = amber, "DO THIS"
        else:
            color, marker = gray, "WAIT"
        cv2.putText(panel, f"{index + 1}. BLINK  {marker}", (20, 90 + index * 48),
                    font, 0.57, color, 2, cv2.LINE_AA)

    state = "NO FACE" if closed is None else "CLOSED" if closed else "OPEN"
    left_text = f"{left_ratio:.3f}" if left_ratio is not None else "N/A"
    right_text = f"{right_ratio:.3f}" if right_ratio is not None else "N/A"
    average_text = (f"{(left_ratio + right_ratio) / 2:.3f}"
                    if left_ratio is not None and right_ratio is not None else "N/A")
    cv2.putText(panel, f"Eyes: {state}", (20, 270), font, 0.55, white, 1, cv2.LINE_AA)
    cv2.putText(panel, f"Left ratio: {left_text}", (20, 305), font, 0.55, white, 1, cv2.LINE_AA)
    cv2.putText(panel, f"Right ratio: {right_text}", (20, 340), font, 0.55, white, 1, cv2.LINE_AA)
    cv2.putText(panel, f"Average: {average_text}", (20, 375), font, 0.55, white, 1, cv2.LINE_AA)
    cv2.putText(panel, f"Threshold: {EYE_CLOSED_THRESHOLD:.3f}",
                (20, 410), font, 0.55, white, 1, cv2.LINE_AA)

    if sequence.complete:
        cv2.putText(panel, "TASK COMPLETE!", (20, 465), font, 0.75, green, 2, cv2.LINE_AA)
    else:
        cv2.putText(panel, "Open, close, then reopen", (20, 465), font, 0.45, white, 1, cv2.LINE_AA)
    cv2.putText(panel, "Cyan: left  Magenta: right", (20, 495),
                font, 0.42, gray, 1, cv2.LINE_AA)
    cv2.putText(panel, "Press Q to quit", (20, 525), font, 0.42, gray, 1, cv2.LINE_AA)


def run():
    """Open the camera and guide the tester through three blinks."""
    import cv2
    import mediapipe as mp
    import numpy as np

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError("Could not open webcam. Check camera permission and close other camera apps.")

    sequence = BlinkSequence()
    try:
        with mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:
            while True:
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("Could not read a webcam frame.")
                frame = cv2.resize(cv2.flip(frame, 1), (960, 540))
                height, width = frame.shape[:2]
                result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                closed = left_ratio = right_ratio = None
                if result.multi_face_landmarks:
                    landmarks = result.multi_face_landmarks[0].landmark
                    left_ratio = eye_aspect_ratio(
                        landmarks, LEFT_EYE_TOP, LEFT_EYE_BOTTOM,
                        LEFT_EYE_OUTER, LEFT_EYE_INNER, width, height
                    )
                    right_ratio = eye_aspect_ratio(
                        landmarks, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM,
                        RIGHT_EYE_OUTER, RIGHT_EYE_INNER, width, height
                    )
                    closed = eyes_are_closed(landmarks, width, height)
                    draw_eye_landmarks(cv2, frame, landmarks)

                sequence.update(closed, time.monotonic())
                panel = np.full((height, 350, 3), (30, 30, 30), dtype=np.uint8)
                draw_panel(cv2, panel, sequence, closed, left_ratio, right_ratio)
                cv2.imshow("BACSA Step 2 Webcam Check", np.hstack((frame, panel)))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return sequence.complete
