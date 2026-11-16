"""Live webcam checkpoint for the Step 1 head-direction exercise."""

import time

from config import HEAD_TURN_THRESHOLD
from vision import LEFT_FACE, NOSE_TIP, RIGHT_FACE, head_direction, landmark_xy


DIRECTIONS = ("center", "left", "right", "center")
HOLD_SECONDS = 0.6


class HeadSequence:
    """Advance only after each requested direction is held steadily."""

    def __init__(self):
        self.index = 0
        self.hold_start = None

    @property
    def complete(self):
        return self.index == len(DIRECTIONS)

    def update(self, direction, now):
        if self.complete:
            return
        if direction != DIRECTIONS[self.index]:
            self.hold_start = None
            return
        if self.hold_start is None:
            self.hold_start = now
        elif now - self.hold_start >= HOLD_SECONDS:
            self.index += 1
            self.hold_start = None


def draw_panel(cv2, panel, sequence, direction, offset):
    """Show the ordered tasks and live head-position measurements."""
    white = (240, 240, 240)
    green = (80, 220, 90)
    amber = (40, 200, 255)
    gray = (130, 130, 130)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(panel, "STEP 1: HEAD TURNS", (18, 38), font, 0.65, white, 2, cv2.LINE_AA)
    for index, requested in enumerate(DIRECTIONS):
        if index < sequence.index:
            color, marker = green, "DONE"
        elif index == sequence.index:
            color, marker = amber, "DO THIS"
        else:
            color, marker = gray, "WAIT"
        label = f"{index + 1}. {requested.upper()}  {marker}"
        cv2.putText(panel, label, (20, 88 + index * 48), font, 0.57, color, 2, cv2.LINE_AA)

    observed = direction.upper() if direction else "NO FACE"
    offset_text = f"{offset:+.3f}" if offset is not None else "N/A"
    cv2.putText(panel, f"Detected: {observed}", (20, 315), font, 0.55, white, 1, cv2.LINE_AA)
    cv2.putText(panel, f"Offset: {offset_text}", (20, 350), font, 0.55, white, 1, cv2.LINE_AA)
    cv2.putText(panel, f"Threshold: +/-{HEAD_TURN_THRESHOLD:.3f}", (20, 385), font, 0.55, white, 1, cv2.LINE_AA)

    if sequence.complete:
        cv2.putText(panel, "TASK COMPLETE!", (20, 455), font, 0.75, green, 2, cv2.LINE_AA)
    else:
        cv2.putText(panel, "Hold each pose briefly", (20, 455), font, 0.48, white, 1, cv2.LINE_AA)
    cv2.putText(panel, "Press Q to quit", (20, 510), font, 0.48, gray, 1, cv2.LINE_AA)


def run():
    """Open the camera and guide Center, Left, Right, Center in order."""
    import cv2
    import mediapipe as mp
    import numpy as np

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError("Could not open webcam. Check camera permission and close other camera apps.")

    sequence = HeadSequence()
    colors = {
        LEFT_FACE: ((255, 255, 0), "LEFT"),
        RIGHT_FACE: ((255, 0, 255), "RIGHT"),
        NOSE_TIP: ((0, 255, 255), "NOSE")
    }
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
                direction = None
                offset = None
                if result.multi_face_landmarks:
                    landmarks = result.multi_face_landmarks[0].landmark
                    left_x = landmarks[LEFT_FACE].x
                    right_x = landmarks[RIGHT_FACE].x
                    face_width = right_x - left_x
                    if abs(face_width) >= 1e-6:
                        offset = (landmarks[NOSE_TIP].x - left_x) / face_width - 0.5
                        direction = head_direction(landmarks)
                    for index, (color, label) in colors.items():
                        x, y = landmark_xy(landmarks, index, width, height)
                        cv2.circle(frame, (x, y), 7, color, -1, cv2.LINE_AA)
                        cv2.putText(frame, label, (x + 10, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

                sequence.update(direction, time.monotonic())
                panel = np.full((height, 350, 3), (30, 30, 30), dtype=np.uint8)
                draw_panel(cv2, panel, sequence, direction, offset)
                cv2.imshow("BACSA Step 1 Webcam Check", np.hstack((frame, panel)))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return sequence.complete
