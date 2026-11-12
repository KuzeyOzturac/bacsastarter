from config import HEAD_TURN_THRESHOLD, EYE_CLOSED_THRESHOLD, GAZE_HORIZONTAL_MARGIN, GAZE_VERTICAL_MARGIN
import math

LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133

RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263

LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145

RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

NOSE_TIP = 1
LEFT_FACE = 234
RIGHT_FACE = 454
def clamp(value, minimum, maximum):
    """Keep a number within an inclusive range.

    Args:
        value: Number to limit.
        minimum: Lowest allowed value.
        maximum: Highest allowed value.

    Returns:
        The value limited to the given range.
    """
    return max(
        minimum,
        min(value, maximum)
    )


def landmark_xy(
    landmarks,
    index,
    width,
    height
):
    """Convert a normalized face landmark to image pixel coordinates.

    Args:
        landmarks: Sequence of MediaPipe face landmarks.
        index: Landmark index to read.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        The landmark's integer (x, y) pixel coordinates.

    Raises:
        IndexError: If the landmark index is unavailable.
    """
    point = landmarks[index]

    return (
        int(point.x * width),
        int(point.y * height)
    )


def euclidean(p1, p2):
    """Measure the straight-line distance between two image points.

    Args:
        p1: First (x, y) point.
        p2: Second (x, y) point.

    Returns:
        Distance between the points in pixels.
    """
    return math.sqrt(
        (p1[0] - p2[0]) ** 2
        + (p1[1] - p2[1]) ** 2
    )


def average_point(
    landmarks,
    indices,
    width,
    height
):
    """Find the center of several face landmarks in image pixels.

    Args:
        landmarks: Sequence of MediaPipe face landmarks.
        indices: Landmark indices to average.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Mean (x, y) position of the selected landmarks.

    Raises:
        IndexError: If a requested landmark is unavailable.
        ZeroDivisionError: If indices is empty.
    """
    x = 0
    y = 0

    for index in indices:
        px, py = landmark_xy(
            landmarks,
            index,
            width,
            height
        )

        x += px
        y += py

    return (
        x / len(indices),
        y / len(indices)
    )


# ============================================================
# HEAD TRACKING
# ============================================================

def head_direction(landmarks):
    """Estimate whether the head points left, right, or center.

    Args:
        landmarks: MediaPipe face landmarks containing the nose and face sides.

    Returns:
        "left", "right", or "center" based on nose position.
    """
    left_x = landmarks[LEFT_FACE].x
    right_x = landmarks[RIGHT_FACE].x
    nose_x = landmarks[NOSE_TIP].x

    face_width = (
        right_x - left_x
    )

    if abs(face_width) < 1e-6:
        return "center"

    nose_position = (
        nose_x - left_x
    ) / face_width

    offset = (
        nose_position - 0.5
    )

    # STEP 1: Compare offset with HEAD_TURN_THRESHOLD.
    # Return "left" or "right" when the head turns far enough.
    return "center"


# ============================================================
# EYE CLOSURE
# ============================================================

def eye_aspect_ratio(
    landmarks,
    top_index,
    bottom_index,
    outer_index,
    inner_index,
    width,
    height
):
    """Measure one eye's height relative to its width.

    Args:
        landmarks: Sequence of MediaPipe face landmarks.
        top_index: Landmark at the top of the eye.
        bottom_index: Landmark at the bottom of the eye.
        outer_index: Landmark at the outer eye corner.
        inner_index: Landmark at the inner eye corner.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Eye height divided by eye width, or 1.0 for zero width.
    """
    top = landmark_xy(
        landmarks,
        top_index,
        width,
        height
    )

    bottom = landmark_xy(
        landmarks,
        bottom_index,
        width,
        height
    )

    outer = landmark_xy(
        landmarks,
        outer_index,
        width,
        height
    )

    inner = landmark_xy(
        landmarks,
        inner_index,
        width,
        height
    )

    vertical = euclidean(
        top,
        bottom
    )

    horizontal = euclidean(
        outer,
        inner
    )

    if horizontal == 0:
        return 1.0

    return vertical / horizontal


def eyes_are_closed(
    landmarks,
    width,
    height
):
    """Decide whether the average eye opening is below the threshold.

    Args:
        landmarks: MediaPipe face landmarks containing both eyes.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        True when the eyes appear closed; otherwise False.
    """
    left_ratio = eye_aspect_ratio(
        landmarks,
        LEFT_EYE_TOP,
        LEFT_EYE_BOTTOM,
        LEFT_EYE_OUTER,
        LEFT_EYE_INNER,
        width,
        height
    )

    right_ratio = eye_aspect_ratio(
        landmarks,
        RIGHT_EYE_TOP,
        RIGHT_EYE_BOTTOM,
        RIGHT_EYE_OUTER,
        RIGHT_EYE_INNER,
        width,
        height
    )

    average_ratio = (
        left_ratio + right_ratio
    ) / 2

    # STEP 2: Compare average_ratio with EYE_CLOSED_THRESHOLD.
    return False


# ============================================================
# IRIS / GAZE TRACKING
# ============================================================

def horizontal_gaze_ratio(
    iris_center,
    outer_corner,
    inner_corner
):
    """Locate an iris between the left and right eye corners.

    Args:
        iris_center: Iris center as an (x, y) pixel point.
        outer_corner: Outer eye corner as an (x, y) pixel point.
        inner_corner: Inner eye corner as an (x, y) pixel point.

    Returns:
        Horizontal position from 0.0 to 1.0, or 0.5 for zero width.
    """
    left_bound = min(
        outer_corner[0],
        inner_corner[0]
    )

    right_bound = max(
        outer_corner[0],
        inner_corner[0]
    )

    eye_width = (
        right_bound - left_bound
    )

    if eye_width <= 0:
        return 0.5

    ratio = (
        iris_center[0] - left_bound
    ) / eye_width

    return clamp(
        ratio,
        0.0,
        1.0
    )


def vertical_gaze_ratio(
    iris_center,
    eye_top,
    eye_bottom
):
    """Locate an iris between the top and bottom of an eye.

    Args:
        iris_center: Iris center as an (x, y) pixel point.
        eye_top: Top of the eye as an (x, y) pixel point.
        eye_bottom: Bottom of the eye as an (x, y) pixel point.

    Returns:
        Vertical position from 0.0 to 1.0, or 0.5 for zero height.
    """
    top_bound = min(
        eye_top[1],
        eye_bottom[1]
    )

    bottom_bound = max(
        eye_top[1],
        eye_bottom[1]
    )

    eye_height = (
        bottom_bound - top_bound
    )

    if eye_height <= 0:
        return 0.5

    ratio = (
        iris_center[1] - top_bound
    ) / eye_height

    return clamp(
        ratio,
        0.0,
        1.0
    )


def calculate_gaze(
    landmarks,
    width,
    height
):
    """Measure gaze in both eyes and return their average positions.

    Args:
        landmarks: MediaPipe face landmarks containing both eyes and irises.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Horizontal ratio, vertical ratio, left iris center, and right iris
        center. Iris centers are (x, y) pixel points.
    """
    left_iris_center = average_point(
        landmarks,
        LEFT_IRIS,
        width,
        height
    )

    right_iris_center = average_point(
        landmarks,
        RIGHT_IRIS,
        width,
        height
    )

    left_outer = landmark_xy(
        landmarks,
        LEFT_EYE_OUTER,
        width,
        height
    )

    left_inner = landmark_xy(
        landmarks,
        LEFT_EYE_INNER,
        width,
        height
    )

    right_inner = landmark_xy(
        landmarks,
        RIGHT_EYE_INNER,
        width,
        height
    )

    right_outer = landmark_xy(
        landmarks,
        RIGHT_EYE_OUTER,
        width,
        height
    )

    left_top = landmark_xy(
        landmarks,
        LEFT_EYE_TOP,
        width,
        height
    )

    left_bottom = landmark_xy(
        landmarks,
        LEFT_EYE_BOTTOM,
        width,
        height
    )

    right_top = landmark_xy(
        landmarks,
        RIGHT_EYE_TOP,
        width,
        height
    )

    right_bottom = landmark_xy(
        landmarks,
        RIGHT_EYE_BOTTOM,
        width,
        height
    )

    left_horizontal = horizontal_gaze_ratio(
        left_iris_center,
        left_outer,
        left_inner
    )

    right_horizontal = horizontal_gaze_ratio(
        right_iris_center,
        right_inner,
        right_outer
    )

    left_vertical = vertical_gaze_ratio(
        left_iris_center,
        left_top,
        left_bottom
    )

    right_vertical = vertical_gaze_ratio(
        right_iris_center,
        right_top,
        right_bottom
    )

    # STEP 3: Average the left and right eye ratios for each direction.
    horizontal = 0.5
    vertical = 0.5

    return (
        horizontal,
        vertical,
        left_iris_center,
        right_iris_center
    )


def gaze_direction(
    horizontal,
    vertical,
    center_horizontal,
    center_vertical
):
    """Compare current gaze with the calibrated screen-facing position.

    Args:
        horizontal: Current horizontal gaze ratio.
        vertical: Current vertical gaze ratio.
        center_horizontal: Calibrated horizontal gaze ratio.
        center_vertical: Calibrated vertical gaze ratio.

    Returns:
        "side", "up", "down", or "center" after applying gaze margins.
    """
    if (
        horizontal
        < center_horizontal
        - GAZE_HORIZONTAL_MARGIN
    ):
        return "side"

    if (
        horizontal
        > center_horizontal
        + GAZE_HORIZONTAL_MARGIN
    ):
        return "side"

    if (
        vertical
        < center_vertical
        - GAZE_VERTICAL_MARGIN
    ):
        return "up"

    if (
        vertical
        > center_vertical
        + GAZE_VERTICAL_MARGIN
    ):
        return "down"

    return "center"


# ============================================================
# INSTAGRAM / MUTE ICON DETECTION
# ============================================================

