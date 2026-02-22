import cv2
import numpy as np
from collections import deque

CAMERA_INDEX  = 0
OUTPUT_SIZE   = 500


# HELPER: Derive the 4 bounding-box corners
# from a detected circle (x, y, radius)
def circle_to_corners(x, y, r):
    """
    Returns the 4 corners of the square that tightly
    wraps around the detected circle, as a float32 array
    ready for getPerspectiveTransform().
    """
    return np.float32([
        [x - r, y - r],   # top-left
        [x + r, y - r],   # top-right
        [x + r, y + r],   # bottom-right
        [x - r, y + r],   # bottom-left
    ])



# HELPER: Warp the frame so the dartboard fills
# a flat OUTPUT_SIZE x OUTPUT_SIZE square
def warp_board(frame, src_corners):
    """
    Applies a perspective transform so the dartboard
    appears as a flat, centered circle.
    """
    dst_corners = np.float32([
        [0,           0          ],  # top-left
        [OUTPUT_SIZE, 0          ],  # top-right
        [OUTPUT_SIZE, OUTPUT_SIZE],  # bottom-right
        [0,           OUTPUT_SIZE],  # bottom-left
    ])

    # Compute the 3x3 transformation matrix
    M = cv2.getPerspectiveTransform(src_corners, dst_corners)

    # stretches/squishes the image so the board looks flat
    warped = cv2.warpPerspective(frame, M, (OUTPUT_SIZE, OUTPUT_SIZE))
    return warped, M



# HELPER: Draw the detected circle + center
# on the live feed frame
def draw_detection(frame, x, y, r):
    cv2.circle(frame, (x, y), r, (0, 255, 0), 2)        # Outer ring (green)
    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)        # Center dot (red)
    cv2.putText(frame, f"Board: ({x},{y}) r={r}",
                (x - r, y - r - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)



# SETUP: Create windows and trackbars
def create_windows():
    cv2.namedWindow("Live Feed")
    cv2.namedWindow("Warped Board")
    cv2.namedWindow("Tuner")

    # These trackbars let you tune HoughCircles in real time
    # without restarting the script
    cv2.createTrackbar("param1",  "Tuner", 50,  200, lambda x: None)  # Edge sensitivity
    cv2.createTrackbar("param2",  "Tuner", 30,  100, lambda x: None)  # Circle threshold
    cv2.createTrackbar("minR",    "Tuner", 50,  400, lambda x: None)  # Min circle radius
    cv2.createTrackbar("maxR",    "Tuner", 200, 600, lambda x: None)  # Max circle radius
    cv2.createTrackbar("blur",    "Tuner", 17,  51,  lambda x: None)  # Gaussian blur size


def get_trackbar_values():
    p1   = max(1, cv2.getTrackbarPos("param1", "Tuner"))
    p2   = max(1, cv2.getTrackbarPos("param2", "Tuner"))
    minR = max(1, cv2.getTrackbarPos("minR",   "Tuner"))
    maxR = max(1, cv2.getTrackbarPos("maxR",   "Tuner"))
    blur = cv2.getTrackbarPos("blur", "Tuner")

    # Blur kernel must be odd and at least 1
    if blur % 2 == 0:
        blur += 1
    blur = max(1, blur)

    return p1, p2, minR, maxR, blur



# CORE: Detect the dartboard circle in a frame
def detect_board(frame, p1, p2, minR, maxR, blur_size):
    """
    Converts the frame to grayscale, blurs it to reduce noise,
    then runs HoughCircles to find the dartboard.

    Returns (x, y, r) if found, or None if not detected.
    """
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=150,    # Only detect one board — ignore detections too close together
        param1=p1,
        param2=p2,
        minRadius=minR,
        maxRadius=maxR
    )

    if circles is not None:
        # Take only the strongest detection
        x, y, r = np.round(circles[0, 0]).astype("int")
        return x, y, r

    return None


# MAIN LOOP
def main():
    detection_history = deque(maxlen=10)  # Store last 10 detections
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"ERROR: Could not open camera {CAMERA_INDEX}.")
        print("Try changing CAMERA_INDEX to 1 or 2 at the top of this file.")
        return

    create_windows()

    last_corners = None

    print("Controls:")
    print("  Q       — quit")
    print("  S       — save current frame to disk")
    print("  Tuner   — drag sliders to adjust circle detection")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame. Is the camera connected?")
            break

        # Read current tuning values from sliders
        p1, p2, minR, maxR, blur_size = get_trackbar_values()

        # Detection
        result = detect_board(frame, p1, p2, minR, maxR, blur_size)

        display = frame.copy()

        if result is not None:
            detection_history.append(result)

        if len(detection_history) > 0:
            # Average all recent detections
            x = int(np.mean([d[0] for d in detection_history]))
            y = int(np.mean([d[1] for d in detection_history]))
            r = int(np.mean([d[2] for d in detection_history]))
            draw_detection(display, x, y, r)
            last_corners = circle_to_corners(x, y, r)
        
        '''if result is not None:
            x, y, r = result
            draw_detection(display, x, y, r)

            # Compute and store the 4 bounding corners for warping
            last_corners = circle_to_corners(x, y, r)

            # Status text
            cv2.putText(display, "Board DETECTED", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(display, "Board NOT FOUND — adjust Tuner sliders", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)'''

        # Warped View
        if last_corners is not None:
            h, w = frame.shape[:2]
            clamped = last_corners.copy()
            clamped[:, 0] = np.clip(clamped[:, 0], 0, w - 1)
            clamped[:, 1] = np.clip(clamped[:, 1], 0, h - 1)

            warped, M = warp_board(frame, clamped)

            # Draw crosshairs at the center of the warped view
            cx, cy = OUTPUT_SIZE // 2, OUTPUT_SIZE // 2
            cv2.line(warped, (cx - 20, cy), (cx + 20, cy), (0, 255, 255), 1)
            cv2.line(warped, (cx, cy - 20), (cx, cy + 20), (0, 255, 255), 1)
            cv2.circle(warped, (cx, cy), OUTPUT_SIZE // 2 - 2, (0, 255, 0), 1)

            cv2.imshow("Warped Board", warped)
        else:
            # Show a blank placeholder until the board is first detected
            blank = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE, 3), dtype=np.uint8)
            cv2.putText(blank, "Waiting for detection...", (30, OUTPUT_SIZE // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
            cv2.imshow("Warped Board", blank)

        # Show Live Feed
        cv2.imshow("Live Feed", display)

        # Keyboard Controls
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Quitting.")
            break

        elif key == ord('s'):
            cv2.imwrite("saved_frame.jpg", frame)
            if last_corners is not None:
                warped_save, _ = warp_board(frame, last_corners)
                cv2.imwrite("saved_warped.jpg", warped_save)
                print("Saved: saved_frame.jpg and saved_warped.jpg")
            else:
                print("Saved: saved_frame.jpg (no warp — board not detected yet)")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()