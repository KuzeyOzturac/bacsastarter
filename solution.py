"""Completed version of the BACSA webcam code-along."""

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

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if detector.empty():
        camera.release()
        raise RuntimeError("Could not load the face detector.")

    missing_since = None
    reels_opened = False

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read a webcam frame.")
                break

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            face_visible = len(faces) > 0

            if face_visible:
                missing_since = None
                status_text = "Face visible"
            else:
                if missing_since is None:
                    missing_since = time.monotonic()
                elapsed = time.monotonic() - missing_since
                status_text = f"Face missing: {elapsed:.1f}s"

                if elapsed >= WAIT_SECONDS and not reels_opened:
                    reels_opened = True
                    opened = webbrowser.open(REELS_URL)
                    status_text = "Opening Reels" if opened else "Could not open browser"

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
