"""Live webcam checkpoint for the Step 3 iris-gaze exercise."""

import time

from blink_demo import draw_eye_landmarks
from config import GAZE_HORIZONTAL_MARGIN
from vision import calculate_gaze, eyes_are_closed, head_direction


DIRECTIONS = ("center", "left", "right", "center")
CENTER_SECONDS = 0.8
HOLD_SECONDS = 0.4


class GazeSequence:
    """Use a short center reference to check eye movements in order."""

    def __init__(self):
        self.index = 0
        self.center = None
        self.capture_start = None
        self.capture_values = []
        self.hold_start = None

    @property
    def complete(self):
        return self.index == len(DIRECTIONS)

    def direction(self, horizontal, vertical):
        """Classify a reading relative to this check's center reference."""
        if self.center is None or horizontal is None or vertical is None:
            return None
        delta_h = horizontal - self.center[0]
        if delta_h < -GAZE_HORIZONTAL_MARGIN:
            return "left"
        if delta_h > GAZE_HORIZONTAL_MARGIN:
            return "right"
        return "center"

    def update(self, horizontal, vertical, head_centered, eyes_open, now):
        """Advance after valid center capture or a held target direction."""
        if self.complete:
            return
        valid = (horizontal is not None and vertical is not None
                 and head_centered and eyes_open)
        if not valid:
            self.hold_start = None
            if self.center is None:
                self.capture_start = None
                self.capture_values.clear()
            return

        if self.center is None:
            if self.capture_start is None:
                self.capture_start = now
            self.capture_values.append((horizontal, vertical))
            if now - self.capture_start >= CENTER_SECONDS:
                count = len(self.capture_values)
                self.center = (
                    sum(value[0] for value in self.capture_values) / count,
                    sum(value[1] for value in self.capture_values) / count
                )
                self.index = 1
            return

        if self.direction(horizontal, vertical) != DIRECTIONS[self.index]:
            self.hold_start = None
        elif self.hold_start is None:
            self.hold_start = now
        elif now - self.hold_start >= HOLD_SECONDS:
            self.index += 1
            self.hold_start = None


def draw_panel(cv2, panel, sequence, horizontal, vertical, head_centered, eyes_open):
    """Show task progress, live gaze ratios, reference, and margins."""
    white = (240, 240, 240)
    green = (80, 220, 90)
    amber = (40, 200, 255)
    gray = (130, 130, 130)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(panel, "STEP 3: EYE GAZE", (18, 35), font, 0.65, white, 2, cv2.LINE_AA)
    for index, requested in enumerate(DIRECTIONS):
        if index < sequence.index:
            color, marker = green, "DONE"
        elif index == sequence.index:
            color, marker = amber, "DO THIS"
        else:
            color, marker = gray, "WAIT"
        cv2.putText(panel, f"{index + 1}. {requested.upper()}  {marker}",
                    (20, 75 + index * 42), font, 0.53, color, 2, cv2.LINE_AA)

    if sequence.complete:
        message, color = "TASK COMPLETE!", green
    elif horizontal is None:
        message, color = "FACE NOT VISIBLE", amber
    elif not head_centered or not eyes_open:
        message, color = "CENTER HEAD / OPEN EYES", amber
    elif sequence.center is None:
        message, color = "LOOK AT CENTER", amber
    else:
        message, color = f"Detected: {sequence.direction(horizontal, vertical)}", white
    cv2.putText(panel, message, (20, 340), font, 0.54, color, 2, cv2.LINE_AA)

    h_text = f"{horizontal:.3f}" if horizontal is not None else "N/A"
    v_text = f"{vertical:.3f}" if vertical is not None else "N/A"
    center_text = (f"{sequence.center[0]:.3f}, {sequence.center[1]:.3f}"
                   if sequence.center else "not captured")
    cv2.putText(panel, f"Horizontal: {h_text}", (20, 380), font, 0.53, white, 1, cv2.LINE_AA)
    cv2.putText(panel, f"Vertical: {v_text}", (20, 410), font, 0.53, white, 1, cv2.LINE_AA)
    cv2.putText(panel, f"Center H,V: {center_text}", (20, 440), font, 0.49, white, 1, cv2.LINE_AA)
    cv2.putText(panel, f"Horizontal margin: {GAZE_HORIZONTAL_MARGIN:.2f}",
                (20, 470), font, 0.48, white, 1, cv2.LINE_AA)
    cv2.putText(panel, "R: recenter   Q: quit", (20, 520), font, 0.48, gray, 1, cv2.LINE_AA)


def run():
    """Open the camera and guide the tester through four gaze positions."""
    import cv2
    import mediapipe as mp
    import numpy as np

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError("Could not open webcam. Check camera permission and close other camera apps.")

    sequence = GazeSequence()
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
                horizontal = vertical = None
                head_centered = eyes_open = False
                if result.multi_face_landmarks:
                    landmarks = result.multi_face_landmarks[0].landmark
                    horizontal, vertical, left_iris, right_iris = calculate_gaze(
                        landmarks, width, height
                    )
                    head_centered = head_direction(landmarks) == "center"
                    eyes_open = not eyes_are_closed(landmarks, width, height)
                    draw_eye_landmarks(cv2, frame, landmarks)
                    for point, label in ((left_iris, "L IRIS"), (right_iris, "R IRIS")):
                        x, y = int(point[0]), int(point[1])
                        cv2.circle(frame, (x, y), 7, (0, 255, 255), -1, cv2.LINE_AA)
                        cv2.putText(frame, label, (x + 10, y + 18),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 255), 2, cv2.LINE_AA)

                sequence.update(horizontal, vertical, head_centered, eyes_open,
                                time.monotonic())
                panel = np.full((height, 380, 3), (30, 30, 30), dtype=np.uint8)
                draw_panel(cv2, panel, sequence, horizontal, vertical,
                           head_centered, eyes_open)
                cv2.imshow("BACSA Step 3 Webcam Check", np.hstack((frame, panel)))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    sequence = GazeSequence()
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return sequence.complete
