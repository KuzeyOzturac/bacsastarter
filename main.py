"""BACSA code-along: build a tiny webcam focus detector.

Run this starter first. Then fill in the four TODOs using README.md.
"""

import time
import webbrowser

import cv2


WAIT_SECONDS = 1.0
REELS_URL = "https://www.instagram.com/reels/"


def main():
    if not hasattr(cv2, "CascadeClassifier") or not hasattr(cv2, "data"):
        raise RuntimeError(
            "This Python loaded an incomplete cv2 package. Create and activate "
            "a fresh .venv, then run: python -m pip install -r requirements.txt"
        )

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open the webcam. Check camera permissions.")

    # OpenCV includes this face detector, so no model download is needed.
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if detector.empty():
        camera.release()
        raise RuntimeError("Could not load the face detector.")

    # TODO 1: Remember when the face first disappears and whether Reels opened.

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read a webcam frame.")
                break

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # TODO 2: Ask the detector to find faces in gray.
            # Set face_visible to True when at least one face is found.
            face_visible = True

            # TODO 3: Start/reset the timer based on face_visible.
            # Set status_text to a message you want to show on screen.
            status_text = "Camera ready - press Q to quit"

            # TODO 4: After WAIT_SECONDS without a face, open REELS_URL once.

            cv2.putText(
                frame,
                status_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("BACSA Focus Demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
