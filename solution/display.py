import cv2

def draw_iris(
    frame,
    point,
):
    cv2.circle(
        frame,
        (
            int(point[0]),
            int(point[1]),
        ),
        3,
        (255, 255, 0),
        -1,
    )


def draw_status(
    frame,
    text,
    color,
):
    cv2.putText(
        frame,
        text,
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
        cv2.LINE_AA,
    )


# ============================================================
