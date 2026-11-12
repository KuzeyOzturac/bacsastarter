# BACSA: build a tiny focus detector

This is a **code-along starter**, not the full Dopaminehub app. We will use a webcam to spot a face. If the face is missing for one second, the program opens Instagram Reels once. This version does **not** track eyes, judge attention, unmute videos, or scroll Reels. A face leaving the camera is only a simple signal for this exercise; it does not prove someone stopped concentrating.

## Before the session

Use Python 3.11 or newer and a webcam. In a terminal in this folder:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Then install and run:

```bash
python -m pip install -r requirements.txt
python main.py
```

Allow camera access if your computer asks. A webcam window should appear. Press **Q** while that window is active to quit. If `python` is unavailable, try `python3` (and `py` on Windows). The face detector is included with OpenCV; no separate model download is needed.

## Code along (about 25 minutes)

Open `main.py`. The starter runs before you change anything. Fill in each TODO, run the program, and predict what should happen before testing it.

### 1. Give the program a memory (5 minutes)

Under **TODO 1**, add:

```python
missing_since = None
reels_opened = False
```

`None` means the timer has not started. `reels_opened` stops the browser from opening over and over.

### 2. Find a face (5 minutes)

Under **TODO 2**, replace `face_visible = True` with:

```python
faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
face_visible = len(faces) > 0
```

The detector returns a list of face locations. We only need to know whether that list is empty. **Try it:** cover the camera. What value should `face_visible` have?

### 3. Wait before deciding (5 minutes)

Under **TODO 3**, replace the starter `status_text` line with:

```python
if face_visible:
    missing_since = None
    status_text = "Face visible"
else:
    if missing_since is None:
        missing_since = time.monotonic()
    elapsed = time.monotonic() - missing_since
    status_text = f"Face missing: {elapsed:.1f}s"
```

`time.monotonic()` is a clock for measuring elapsed time. Seeing a face resets the timer. **Try it:** step out of view, return, then step out again. Does the count restart?

### 4. Take one action (5 minutes)

Under **TODO 4**, inside the `else` block from step 3, after `status_text`, add:

```python
    if elapsed >= WAIT_SECONDS and not reels_opened:
        reels_opened = True
        opened = webbrowser.open(REELS_URL)
        status_text = "Opening Reels" if opened else "Could not open browser"
```

Those lines should line up with `elapsed = ...` inside the `else` block. Remove the standalone TODO 4 comment after moving the code. **Try it:** leave the camera view for less than a second, then longer than a second. Which one opens the browser? Return and leave again: does it open a second tab?

The complete result is in `solution.py` if you get stuck. Run it with `python solution.py`.

## Presenter notes

- Ask the room to predict each result before running a step.
- Use **covering the camera** to make the missing-face test reliable.
- Mention that a webcam or permissions problem can stop the live run; keep a recording or screenshots ready.
- Explain that the full [Dopaminehub](https://github.com/KuzeyOzturac/dopaminehub) adds gaze, head direction, eye closure, smoothing, calibration, and Reel controls. This workshop teaches the smaller detect–wait–act pattern first.
- The starter opens a browser on the presenter's computer. Participants can comment out `webbrowser.open(REELS_URL)` and use `print("Focus loss detected")` if they prefer a console-only action.
