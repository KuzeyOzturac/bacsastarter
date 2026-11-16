"""Live webcam checkpoints for smoothing, calibration, and focus loss."""

import time
from collections import deque

from config import CALIBRATION_SECONDS, DISTRACTION_SECONDS, GAZE_SMOOTHING_FRAMES
from focus import FocusTracker
from vision import calculate_gaze, eyes_are_closed, head_direction


class FocusChallenge:
    """Guide two focus losses with a sustained first loss and a reset."""

    TASKS = (
        "CALIBRATE",
        "FIRST FOCUS LOSS",
        "STAY AWAY: NO REPEAT",
        "RETURN TO FOCUS",
        "SECOND FOCUS LOSS"
    )

    def __init__(self):
        self.index = 0
        self.trigger_count = 0
        self.repeated_trigger = False
        self.hold_start = None

    @property
    def complete(self):
        return self.index == len(self.TASKS) and not self.repeated_trigger

    def update(self, status, calibrated, now):
        """Advance through calibration, two events, and a focus reset."""
        if self.complete or self.repeated_trigger:
            return
        if status.trigger:
            self.trigger_count += 1

        if self.index == 0 and calibrated:
            self.index = 1
        elif self.index == 1 and status.trigger:
            self.index = 2
            self.hold_start = now
        elif self.index == 2:
            if status.trigger:
                self.repeated_trigger = True
            elif status.text.startswith("FOCUS LOST") and now - self.hold_start >= 1.0:
                self.index = 3
                self.hold_start = None
        elif self.index == 3:
            if status.text == "FOCUSED":
                if self.hold_start is None:
                    self.hold_start = now
                elif now - self.hold_start >= 0.4:
                    self.index = 4
                    self.hold_start = None
            else:
                self.hold_start = None
        elif self.index == 4 and status.trigger:
            self.index = 5


def read_frame(cv2, face_mesh, camera):
    """Read a frame and build the measurement consumed by FocusTracker."""
    ok, frame = camera.read()
    if not ok:
        raise RuntimeError("Could not read a webcam frame.")
    frame = cv2.resize(cv2.flip(frame, 1), (960, 540))
    height, width = frame.shape[:2]
    result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    measurement = None
    raw = None
    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark
        horizontal, vertical, left_iris, right_iris = calculate_gaze(
            landmarks, width, height
        )
        measurement = (head_direction(landmarks), eyes_are_closed(landmarks, width, height),
                       horizontal, vertical)
        raw = (horizontal, vertical)
        for point in (left_iris, right_iris):
            cv2.circle(frame, (int(point[0]), int(point[1])), 6,
                       (0, 255, 255), -1, cv2.LINE_AA)
    return frame, measurement, raw


def draw_graph(cv2, np, panel, raw_history, smooth_history):
    """Draw recent raw and smoothed horizontal gaze on the same axes."""
    left, top, width, height = 20, 330, 390, 110
    cv2.rectangle(panel, (left, top), (left + width, top + height),
                  (90, 90, 90), 1)
    for history, color in ((raw_history, (255, 180, 60)),
                           (smooth_history, (80, 220, 90))):
        if len(history) < 2:
            continue
        points = np.array([
            (left + round(i * (width - 1) / (history.maxlen - 1)),
             top + height - round(max(0.0, min(1.0, value)) * height))
            for i, value in enumerate(history)
        ], dtype=np.int32)
        cv2.polylines(panel, [points], False, color, 2, cv2.LINE_AA)


def draw_step4_panel(cv2, np, panel, tracker, status, raw, samples, raw_history, smooth_history, now):
    """Show calibration progress, gaze values, and the smoothing graph."""
    white = (240, 240, 240)
    green = (80, 220, 90)
    amber = (40, 200, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    calibrated = tracker.center is not None
    smooth_ready = calibrated and samples >= GAZE_SMOOTHING_FRAMES

    cv2.putText(panel, "STEP 4: CALIBRATE + SMOOTH", (18, 35),
                font, 0.58, white, 2, cv2.LINE_AA)
    cv2.putText(panel, f"1. LOOK CENTER  {'DONE' if calibrated else 'DO THIS'}",
                (20, 85), font, 0.56, green if calibrated else amber, 2, cv2.LINE_AA)
    cv2.putText(panel, f"2. SMOOTH 5 FRAMES  {'DONE' if smooth_ready else 'WAIT'}",
                (20, 125), font, 0.53, green if smooth_ready else amber, 2, cv2.LINE_AA)
    remaining = (max(0.0, CALIBRATION_SECONDS - (now - tracker.calibration_start))
                 if tracker.calibration_start is not None and not calibrated
                 else 0.0)
    cv2.putText(panel, f"Calibration remaining: {remaining:.1f}s", (20, 170),
                font, 0.5, white, 1, cv2.LINE_AA)
    raw_text = f"{raw[0]:.3f}, {raw[1]:.3f}" if raw else "N/A"
    smooth_text = (f"{status.smooth_horizontal:.3f}, {status.smooth_vertical:.3f}"
                   if status.smooth_horizontal is not None else "N/A")
    center_text = (f"{tracker.center[0]:.3f}, {tracker.center[1]:.3f}"
                   if calibrated else "not learned")
    cv2.putText(panel, f"Raw H,V: {raw_text}", (20, 205), font, 0.5, white, 1, cv2.LINE_AA)
    cv2.putText(panel, f"Smooth H,V: {smooth_text}", (20, 235), font, 0.5, white, 1, cv2.LINE_AA)
    cv2.putText(panel, f"Center H,V: {center_text}", (20, 265), font, 0.5, white, 1, cv2.LINE_AA)
    cv2.putText(panel, "Horizontal history: raw blue / smooth green", (20, 315),
                font, 0.42, white, 1, cv2.LINE_AA)
    draw_graph(cv2, np, panel, raw_history, smooth_history)
    if smooth_ready:
        cv2.putText(panel, "TASK COMPLETE!", (20, 475), font, 0.68, green, 2, cv2.LINE_AA)
    else:
        cv2.putText(panel, "Keep looking at the screen", (20, 475),
                    font, 0.49, white, 1, cv2.LINE_AA)
    cv2.putText(panel, "R: restart   Q: quit", (20, 520),
                font, 0.48, white, 1, cv2.LINE_AA)
    return smooth_ready


def draw_step5_panel(cv2, panel, tracker, status, challenge, now):
    """Show the timed distraction and one-shot trigger challenge."""
    white = (240, 240, 240)
    green = (80, 220, 90)
    amber = (40, 200, 255)
    gray = (130, 130, 130)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(panel, "STEP 5: ONE-SHOT TRIGGER", (18, 35),
                font, 0.61, white, 2, cv2.LINE_AA)
    for index, task in enumerate(challenge.TASKS):
        if index < challenge.index:
            color, marker = green, "DONE"
        elif index == challenge.index:
            color, marker = amber, "DO THIS"
        else:
            color, marker = gray, "WAIT"
        cv2.putText(panel, f"{index + 1}. {task}  {marker}",
                    (20, 75 + index * 43), font, 0.46, color, 2, cv2.LINE_AA)
    elapsed = now - tracker.distracted_since if tracker.distracted_since is not None else 0.0
    cv2.putText(panel, f"Status: {status.text[:34]}", (20, 330),
                font, 0.46, white, 1, cv2.LINE_AA)
    cv2.putText(panel, f"Distracted: {elapsed:.1f}s / {DISTRACTION_SECONDS:.1f}s",
                (20, 365), font, 0.52, white, 1, cv2.LINE_AA)
    cv2.putText(panel, f"Trigger count: {challenge.trigger_count}",
                (20, 400), font, 0.54, white, 1, cv2.LINE_AA)
    if challenge.repeated_trigger:
        cv2.putText(panel, "REPEATED TRIGGER - R TO RETRY", (20, 465),
                    font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
    elif challenge.complete:
        cv2.putText(panel, "TASK COMPLETE!", (20, 465),
                    font, 0.7, green, 2, cv2.LINE_AA)
    else:
        cv2.putText(panel, "Turn head away, then look back", (20, 465),
                    font, 0.48, white, 1, cv2.LINE_AA)
    cv2.putText(panel, "R: restart   Q: quit", (20, 520),
                font, 0.48, gray, 1, cv2.LINE_AA)


def run(step):
    """Run the selected focus webcam checkpoint without Reel automation."""
    import cv2
    import mediapipe as mp
    import numpy as np

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError("Could not open webcam. Check camera permission and close other camera apps.")

    tracker = FocusTracker()
    challenge = FocusChallenge()
    raw_history = deque(maxlen=80)
    smooth_history = deque(maxlen=80)
    samples_after_calibration = 0
    completed = False
    try:
        with mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:
            while True:
                frame, measurement, raw = read_frame(cv2, face_mesh, camera)
                now = time.monotonic()
                status = tracker.update(now, measurement)
                if raw is not None and status.smooth_horizontal is not None:
                    raw_history.append(raw[0])
                    smooth_history.append(status.smooth_horizontal)
                    if tracker.center is not None:
                        samples_after_calibration += 1

                panel = np.full((540, 430, 3), (30, 30, 30), dtype=np.uint8)
                if step == 4:
                    completed = draw_step4_panel(
                        cv2, np, panel, tracker, status, raw,
                        samples_after_calibration, raw_history, smooth_history, now
                    ) or completed
                else:
                    challenge.update(status, tracker.center is not None, now)
                    draw_step5_panel(cv2, panel, tracker, status, challenge, now)
                    completed = challenge.complete

                cv2.imshow(f"BACSA Step {step} Webcam Check", np.hstack((frame, panel)))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    tracker = FocusTracker()
                    challenge = FocusChallenge()
                    raw_history.clear()
                    smooth_history.clear()
                    samples_after_calibration = 0
                    completed = False
    finally:
        camera.release()
        cv2.destroyAllWindows()
    return completed
