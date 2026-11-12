"""Run the webcam focus detector and Reel actions."""

import time

import cv2
import mediapipe as mp

from config import CAMERA_INDEX
from display import draw_iris, draw_status
from focus import FocusTracker
from reels import MUTE_ICON_PATH, next_reel, open_reels
from vision import calculate_gaze, eyes_are_closed, head_direction


def main():
    print(f"Mute icon template: {MUTE_ICON_PATH}")
    if not MUTE_ICON_PATH.exists():
        print("WARNING: mute_icon.png is missing.")

    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError("Could not open webcam.")

    tracker = FocusTracker()
    reels_started = False
    try:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        with mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:
            while True:
                success, frame = camera.read()
                if not success:
                    print("Could not read camera frame.")
                    break
                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                measurement = None
                if results.multi_face_landmarks:
                    face = results.multi_face_landmarks[0].landmark
                    horizontal, vertical, left_iris, right_iris = calculate_gaze(face, width, height)
                    measurement = (head_direction(face), eyes_are_closed(face, width, height), horizontal, vertical)
                    draw_iris(frame, left_iris)
                    draw_iris(frame, right_iris)

                status = tracker.update(time.monotonic(), measurement)
                if status.trigger:
                    if reels_started:
                        next_reel()
                    else:
                        reels_started = open_reels()

                draw_status(frame, status.text, status.color)
                if measurement is not None:
                    cv2.putText(frame, f"Gaze H: {status.smooth_horizontal:.3f} V: {status.smooth_vertical:.3f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(frame, f"Head: {measurement[0]} | Eyes: {status.gaze}", (20, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                    action = "scroll Reel" if reels_started else "open Reels"
                    cv2.putText(frame, f"Next loss: {action}", (20, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, "Press Q to quit", (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow("Focus Detector", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
