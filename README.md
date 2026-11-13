# BACSA Dopaminehub code-along

Build the original feature path together: **webcam landmarks → head and eye measurements → gaze → calibration and smoothing → timed focus loss → open and unmute Reels → scroll on later focus losses**.

[starter/](starter/) has seven marked gaps. Fill one gap, run its check, then move on. [solution/](solution/) has the completed project. Allow about 75–90 minutes. The checks cover logic; the final camera and browser test needs a Mac with permissions.

## Setup before the session

Use **Python 3.11**, a webcam, and macOS camera, screen recording, and accessibility permissions. From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python checks.py 1
```

The last command should **fail** initially: Step 1 is unfinished. If `python3.11` is unavailable, install Python 3.11 or configure it in PyCharm. Run commands from the repository root with the virtual environment active.

## Step 1 — head direction (10 minutes)

In `starter/vision.py`, find `STEP 1` in `head_direction`. The code measures where the nose sits between the two sides of the face. Compare `offset` with the threshold:

```python
if offset < -HEAD_TURN_THRESHOLD:
    return "left"
if offset > HEAD_TURN_THRESHOLD:
    return "right"
return "center"
```

**Logic check:** `python checks.py 1` should pass. Why are there negative and positive thresholds?

**Webcam check:** Run `python checks.py 1 --webcam`. Follow **Center → Left → Right → Center** with your head. Hold each position briefly. Each finished direction turns green, and the window says **TASK COMPLETE!** after the fourth. The left-side, right-side, and nose landmarks are drawn on your face; the current offset and `HEAD_TURN_THRESHOLD` appear beside the video. Press **Q** to close the window. This check uses the `head_direction()` you just completed in `starter/vision.py`.

## Step 2 — eye closure (8 minutes)

In `starter/vision.py`, find `STEP 2`. The code divides each eye's height by its width and averages the two ratios. Replace `return False` with:

```python
return average_ratio < EYE_CLOSED_THRESHOLD
```

**Logic check:** `python checks.py 2` should pass for both open and closed sample eyes.

**Webcam check:** Run `python checks.py 2 --webcam`. Blink three times. Each blink turns green only after your eyes close and reopen; then the window says **TASK COMPLETE!** The top, bottom, inner, and outer landmarks for both eyes are marked on your face. The left and right eye aspect ratios, their average, and `EYE_CLOSED_THRESHOLD` appear beside the video. Press **Q** to close the window. This check uses the `eyes_are_closed()` you just completed in `starter/vision.py`.

## Step 3 — iris gaze (10 minutes)

In `starter/vision.py`, find `STEP 3`. The helpers already measure iris position in each eye. Replace the two placeholders:

```python
horizontal = (left_horizontal + right_horizontal) / 2
vertical = (left_vertical + right_vertical) / 2
```

**Logic check:** `python checks.py 3` moves the sample irises and checks that the result changes.

**Webcam check:** Run `python checks.py 3 --webcam`. Keep your head still and follow **Center → Left → Right → Up → Down → Center** with your eyes. Look at the screen during the first Center step so the check can capture a short reference. Hold each later gaze briefly. Completed directions turn green; the final one displays **TASK COMPLETE!** Eye landmarks and iris centers appear on your face. The panel shows horizontal and vertical gaze ratios, the center reference, and the margins. Press **R** to start over with a new center reference or **Q** to quit. This short reference belongs only to the check; Step 4 builds the app's full calibration and smoothing.

## Step 4 — smoothing and calibration (12 minutes)

In `starter/focus.py`, fill both `STEP 4` gaps. After appending each reading to its `deque`, average the recent values:

```python
horizontal = sum(self.horizontal) / len(self.horizontal)
vertical = sum(self.vertical) / len(self.vertical)
```

Replace `self.center = (0.5, 0.5)` with the average of the readings collected during the three-second calibration:

```python
self.center = (
    sum(self.calibration_horizontal) / len(self.calibration_horizontal),
    sum(self.calibration_vertical) / len(self.calibration_vertical),
)
```

**Check:** `python checks.py 4` should pass. Why learn *your* center instead of assuming everyone's is `0.5`?

## Step 5 — timed, one-shot focus loss (10 minutes)

The rest of `FocusTracker.update` already identifies a distraction and waits `DISTRACTION_SECONDS`. At `STEP 5`, replace `trigger = False`:

```python
trigger = not self.triggered
self.triggered = True
```

The existing focused branch resets `self.triggered`. **Check:** `python checks.py 5` should pass. What if the flag were never reset?

## Step 6 — find and click the mute icon (15 minutes)

In `starter/reels.py`, the code searches part of the screen for `mute_icon.png` at several sizes. At `STEP 6`, replace `pass`:

```python
(x, y), width, height = best_match
click_x = round((left + x + width / 2) / scale_x)
click_y = round((top + y + height / 2) / scale_y)
pyautogui.moveTo(click_x, click_y, duration=0.3)
time.sleep(0.2)
pyautogui.click()
print(f"Reels unmuted (match score {best_score:.3f}).")
return True
```

The scale divisions turn screenshot pixels into screen coordinates. **Check:** `python checks.py 6` uses a fake screenshot and verifies the click location. It does not click your screen.

## Step 7 — connect browser actions (10 minutes)

In `starter/reels.py`, fill `STEP 7` in `next_reel`:

```python
pyautogui.scroll(REEL_SCROLL_AMOUNT)
```

Then fill `STEP 7` in `starter/main.py`, inside `if status.trigger:`:

```python
if reels_started:
    next_reel()
else:
    reels_started = open_reels()
```

**Check:** `python checks.py 7` verifies scrolling. Then run `python starter/main.py`. Look at the screen for the three-second calibration; the window should say `FOCUSED`. Turn your head away for over one second: Reels should open and the mute icon should be clicked. Look back to reset focus, then turn away again: it should scroll. Press **Q** in the Focus Detector window to quit.

## Presenter notes

- Run all seven checks and the final camera/browser flow on your presentation Mac before the session.
- Camera permissions, Instagram layout, and display scaling can affect the live demo. Keep a short recording as a backup.
- Face and eye readings estimate visible behavior; they cannot tell what someone is thinking.
- The first action counts as started only if the browser opens **and** the mute icon is found. A later focus loss retries if icon matching fails.
- Run a check against the completed code with `BACSA_CHECK_SOURCE=solution python checks.py 1` (change the number).
