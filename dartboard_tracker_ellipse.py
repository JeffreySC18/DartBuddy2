import cv2
import numpy as np
from collections import deque


# CONFIGURATION
CAMERA_INDEX   = 0
OUTPUT_SIZE    = 500    # Size (px) of the warped top-down output window
SMOOTH_FRAMES  = 23     # How many frames to average (higher = smoother but slower to lock)

# Ellipse detection tuning
CANNY_LOW      = 34
CANNY_HIGH     = 100
BLUR_SIZE      = 17     # Must be odd

# Board size filter (pixels) — the contour must fall within this range to be considered
MIN_BOARD_PX   = 132
MAX_BOARD_PX   = 168

MIN_CIRCULARITY = 0.82

# How many pixels the bullseye center must move before we consider
# the board to have shifted and trigger a re-detection
REDETECT_THRESHOLD = 15   # pixels

# Radius padding — multiplier on top of the detected radius
# 1.0 = exact outer wire, 1.05 = 5% larger to fully enclose board surround
RADIUS_PADDING  = 1.05

# Stage 2 — Bullseye
BULLSEYE_SEARCH_RADIUS = 0.25
RED_LOW_1  = np.array([0,   120, 70])
RED_HIGH_1 = np.array([10,  255, 255])
RED_LOW_2  = np.array([170, 120, 70])
RED_HIGH_2 = np.array([180, 255, 255])


# ─────────────────────────────────────────────
# DETECTION STATE
# Tracks whether we have a confirmed lock and
# how long since we last ran full detection
# ─────────────────────────────────────────────
class DetectionState:
    def __init__(self):
        self.locked        = False      # True once we have a stable detection
        self.locked_radius = None       # Frozen radius once locked
        self.locked_cx     = None       # Frozen bullseye center x
        self.locked_cy     = None       # Frozen bullseye center y
        self.frames_since_recheck = 0
        self.recheck_interval     = 30  # Re-verify every N frames even when locked

    def lock(self, cx, cy, radius):
        self.locked        = True
        self.locked_cx     = cx
        self.locked_cy     = cy
        self.locked_radius = radius
        print(f"LOCKED — center=({cx},{cy})  radius={radius:.1f}px")

    def unlock(self):
        self.locked = False
        self.locked_radius = None
        self.locked_cx = None
        self.locked_cy = None
        print("UNLOCKED — re-detecting...")

    def center_has_moved(self, new_cx, new_cy):
        if self.locked_cx is None:
            return True
        dist = np.hypot(new_cx - self.locked_cx, new_cy - self.locked_cy)
        return dist > REDETECT_THRESHOLD


# ─────────────────────────────────────────────
# STAGE 1: Ellipse detection
# ─────────────────────────────────────────────
def detect_board_ellipse(frame):
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (BLUR_SIZE, BLUR_SIZE), 0)
    edges   = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, edges

    candidates = []
    for c in contours:
        _, _, w, h = cv2.boundingRect(c)
        size = max(w, h)
        if MIN_BOARD_PX <= size <= MAX_BOARD_PX and len(c) >= 5:
            candidates.append(c)

    if not candidates:
        return None, edges

    candidates.sort(key=cv2.contourArea, reverse=True)
    candidates = candidates[:3]

    best_ellipse = None
    best_score   = -1

    for contour in candidates:
        hull = cv2.convexHull(contour)
        if len(hull) < 5:
            continue

        ellipse = cv2.fitEllipse(hull)
        (cx, cy), (major, minor), angle = ellipse

        if minor < 1 or major < 1:
            continue

        circularity = min(major, minor) / max(major, minor)
        if circularity < MIN_CIRCULARITY:
            continue

        area  = np.pi * (major / 2) * (minor / 2)
        score = area * circularity

        if score > best_score:
            best_score   = score
            best_ellipse = ellipse

    return best_ellipse, edges


# ─────────────────────────────────────────────
# STAGE 2: Bullseye center correction
# ─────────────────────────────────────────────
def find_bullseye_center(frame, ellipse_cx, ellipse_cy, board_radius):
    h, w = frame.shape[:2]

    search_r = max(10, int(board_radius * BULLSEYE_SEARCH_RADIUS))
    x1 = max(0, ellipse_cx - search_r)
    y1 = max(0, ellipse_cy - search_r)
    x2 = min(w, ellipse_cx + search_r)
    y2 = min(h, ellipse_cy + search_r)

    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return ellipse_cx, ellipse_cy, None, (x1, y1, x2, y2)

    hsv      = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mask1    = cv2.inRange(hsv, RED_LOW_1,  RED_HIGH_1)
    mask2    = cv2.inRange(hsv, RED_LOW_2,  RED_HIGH_2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN,  kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    moments = cv2.moments(red_mask)
    if moments["m00"] > 100:
        local_cx     = int(moments["m10"] / moments["m00"])
        local_cy     = int(moments["m01"] / moments["m00"])
        corrected_cx = x1 + local_cx
        corrected_cy = y1 + local_cy
        return corrected_cx, corrected_cy, red_mask, (x1, y1, x2, y2)

    return ellipse_cx, ellipse_cy, red_mask, (x1, y1, x2, y2)


# ─────────────────────────────────────────────
# HELPERS: Corners and warp
# ─────────────────────────────────────────────
def build_corners_with_radius(cx, cy, radius):
    r = radius * RADIUS_PADDING     # Apply padding to fully enclose board
    return np.float32([
        [cx - r, cy - r],
        [cx + r, cy - r],
        [cx + r, cy + r],
        [cx - r, cy + r],
    ])


def warp_board(frame, src_corners):
    dst_corners = np.float32([
        [0,           0          ],
        [OUTPUT_SIZE, 0          ],
        [OUTPUT_SIZE, OUTPUT_SIZE],
        [0,           OUTPUT_SIZE],
    ])
    M = cv2.getPerspectiveTransform(src_corners, dst_corners)
    warped = cv2.warpPerspective(frame, M, (OUTPUT_SIZE, OUTPUT_SIZE))
    return warped, M


# ─────────────────────────────────────────────
# DRAW: Overlay on live feed
# ─────────────────────────────────────────────
def draw_detection(frame, state, ellipse, ellipse_cx, ellipse_cy,
                   corrected_cx, corrected_cy, search_bounds):
    (_, _), (major, minor), _ = ellipse
    raw_radius   = int(max(major, minor) / 2)
    padded_radius = int(raw_radius * RADIUS_PADDING)

    # Green ellipse from detector
    cv2.ellipse(frame, ellipse, (0, 200, 0), 1)

    # White circle = padded locked radius (the actual warp boundary)
    if state.locked:
        cv2.circle(frame, (corrected_cx, corrected_cy),
                   int(state.locked_radius * RADIUS_PADDING), (255, 255, 255), 2)

    # Yellow = geometric ellipse center
    cv2.circle(frame, (ellipse_cx, ellipse_cy), 5, (0, 255, 255), -1)
    cv2.putText(frame, "Geom", (ellipse_cx + 8, ellipse_cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # Red = corrected bullseye center
    cv2.circle(frame, (corrected_cx, corrected_cy), 6, (0, 0, 255), -1)
    cv2.putText(frame, "Bull", (corrected_cx + 8, corrected_cy + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # White search box
    x1, y1, x2, y2 = search_bounds
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

    # Status banner
    lock_str = f"LOCKED  r={int(state.locked_radius * RADIUS_PADDING)}px" \
               if state.locked else "Accumulating..."
    color    = (0, 255, 0) if state.locked else (0, 165, 255)
    cv2.putText(frame, lock_str, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.putText(frame, f"Raw radius : {raw_radius}px  Padded: {padded_radius}px",
                (10, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    cv2.putText(frame, f"Geom center: ({ellipse_cx}, {ellipse_cy})",
                (10, 85),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (0, 255, 255), 1)
    cv2.putText(frame, f"Bull center: ({corrected_cx}, {corrected_cy})",
                (10, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (0, 0, 255), 1)


# ─────────────────────────────────────────────
# SMOOTHING
# ─────────────────────────────────────────────
def smooth_ellipse(history):
    if not history:
        return None
    n       = len(history)
    weights = np.linspace(0.5, 1.0, n)
    weights /= weights.sum()
    return (
        (np.average([e[0][0] for e in history], weights=weights),
         np.average([e[0][1] for e in history], weights=weights)),
        (np.average([e[1][0] for e in history], weights=weights),
         np.average([e[1][1] for e in history], weights=weights)),
        np.average([e[2] for e in history], weights=weights)
    )


def smooth_point(history):
    if not history:
        return None
    n       = len(history)
    weights = np.linspace(0.5, 1.0, n)
    weights /= weights.sum()
    return (
        int(np.average([p[0] for p in history], weights=weights)),
        int(np.average([p[1] for p in history], weights=weights))
    )


# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
def create_windows():
    for name in ["Live Feed", "Warped Board", "Edge Debug", "Bullseye Debug", "Tuner"]:
        cv2.namedWindow(name)

    cv2.createTrackbar("Canny Low",    "Tuner", CANNY_LOW,        200, lambda x: None)
    cv2.createTrackbar("Canny High",   "Tuner", CANNY_HIGH,       300, lambda x: None)
    cv2.createTrackbar("Blur",         "Tuner", BLUR_SIZE,        51,  lambda x: None)
    cv2.createTrackbar("Min Size",     "Tuner", MIN_BOARD_PX,     600, lambda x: None)
    cv2.createTrackbar("Max Size",     "Tuner", MAX_BOARD_PX,     800, lambda x: None)
    cv2.createTrackbar("Circularity%", "Tuner", int(MIN_CIRCULARITY * 100), 100, lambda x: None)
    cv2.createTrackbar("Red Sat Min",  "Tuner", 120,              255, lambda x: None)
    cv2.createTrackbar("Red Val Min",  "Tuner", 70,               255, lambda x: None)
    cv2.createTrackbar("Search R%",    "Tuner", int(BULLSEYE_SEARCH_RADIUS * 100), 50, lambda x: None)
    cv2.createTrackbar("Pad %",        "Tuner", int(RADIUS_PADDING * 100 - 100), 30, lambda x: None)


def get_trackbar_values():
    low     = max(1,  cv2.getTrackbarPos("Canny Low",    "Tuner"))
    high    = max(2,  cv2.getTrackbarPos("Canny High",   "Tuner"))
    blur    = max(1,  cv2.getTrackbarPos("Blur",         "Tuner"))
    minS    = max(10, cv2.getTrackbarPos("Min Size",     "Tuner"))
    maxS    = max(11, cv2.getTrackbarPos("Max Size",     "Tuner"))
    circ    = max(1,  cv2.getTrackbarPos("Circularity%", "Tuner")) / 100.0
    sat     = max(1,  cv2.getTrackbarPos("Red Sat Min",  "Tuner"))
    val     = max(1,  cv2.getTrackbarPos("Red Val Min",  "Tuner"))
    srch    = max(1,  cv2.getTrackbarPos("Search R%",    "Tuner")) / 100.0
    pad     = 1.0 + cv2.getTrackbarPos("Pad %",          "Tuner") / 100.0
    if blur % 2 == 0:
        blur += 1
    return low, high, blur, minS, maxS, circ, sat, val, srch, pad


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera {CAMERA_INDEX}.")
        return

    create_windows()

    ellipse_history  = deque(maxlen=SMOOTH_FRAMES)
    bullseye_history = deque(maxlen=SMOOTH_FRAMES)
    last_corners     = None
    state            = DetectionState()

    print("Controls:  Q=quit  S=save  E=print values  L=lock  U=unlock")
    print()
    print("  White circle = locked warp boundary (what the scoring will use)")
    print("  Green ellipse = raw detector output")
    print("  Yellow dot = geometric center,  Red dot = bullseye center")
    print()
    print("WORKFLOW:")
    print("  1. Tune until the green ellipse sits on the OUTER wire reliably")
    print("  2. Increase Pad% until the white circle encloses the full board")
    print("  3. Once stable, press L to lock — circle stops moving entirely")
    print("  4. Press E to print final values for hardcoding")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        # Read tuner
        (canny_low, canny_high, blur, min_size, max_size,
         circularity, sat_min, val_min, search_r, pad) = get_trackbar_values()

        global CANNY_LOW, CANNY_HIGH, BLUR_SIZE, MIN_BOARD_PX, MAX_BOARD_PX
        global MIN_CIRCULARITY, RED_LOW_1, RED_HIGH_1, RED_LOW_2, RED_HIGH_2
        global BULLSEYE_SEARCH_RADIUS, RADIUS_PADDING

        CANNY_LOW, CANNY_HIGH, BLUR_SIZE = canny_low, canny_high, blur
        MIN_BOARD_PX, MAX_BOARD_PX       = min_size, max_size
        MIN_CIRCULARITY                  = circularity
        BULLSEYE_SEARCH_RADIUS           = search_r
        RADIUS_PADDING                   = pad
        RED_LOW_1  = np.array([0,   sat_min, val_min])
        RED_HIGH_1 = np.array([10,  255, 255])
        RED_LOW_2  = np.array([170, sat_min, val_min])
        RED_HIGH_2 = np.array([180, 255, 255])

        display = frame.copy()

        # ── Detection (always runs to accumulate history) ──
        ellipse, edges = detect_board_ellipse(frame)
        if ellipse is not None:
            ellipse_history.append(ellipse)

        smoothed_ellipse = smooth_ellipse(ellipse_history)
        bull_debug_img   = None

        if smoothed_ellipse is not None:
            ecx          = int(smoothed_ellipse[0][0])
            ecy          = int(smoothed_ellipse[0][1])
            board_radius = max(smoothed_ellipse[1]) / 2

            corrected_cx, corrected_cy, red_mask, search_bounds = \
                find_bullseye_center(frame, ecx, ecy, board_radius)

            bullseye_history.append((corrected_cx, corrected_cy))
            smoothed_bull = smooth_point(bullseye_history)
            if smoothed_bull:
                corrected_cx, corrected_cy = smoothed_bull

            # ── Auto-lock once history is full and stable ──
            # Only lock automatically if not manually overridden
            if (not state.locked
                    and len(ellipse_history) == SMOOTH_FRAMES
                    and len(bullseye_history) == SMOOTH_FRAMES):
                state.lock(corrected_cx, corrected_cy, board_radius)

            # ── If locked, check if board has moved significantly ──
            if state.locked and state.center_has_moved(corrected_cx, corrected_cy):
                print(f"Board moved — re-locking...")
                state.lock(corrected_cx, corrected_cy, board_radius)

            # Use locked values if available, otherwise use smoothed
            final_cx     = state.locked_cx     if state.locked else corrected_cx
            final_cy     = state.locked_cy     if state.locked else corrected_cy
            final_radius = state.locked_radius if state.locked else board_radius

            last_corners = build_corners_with_radius(final_cx, final_cy, final_radius)

            draw_ellipse = (
                (int(smoothed_ellipse[0][0]), int(smoothed_ellipse[0][1])),
                (int(smoothed_ellipse[1][0]), int(smoothed_ellipse[1][1])),
                smoothed_ellipse[2]
            )
            draw_detection(display, state, draw_ellipse,
                           ecx, ecy, final_cx, final_cy, search_bounds)

            if red_mask is not None:
                bull_debug_img = cv2.resize(red_mask, (200, 200),
                                            interpolation=cv2.INTER_NEAREST)
                bull_debug_img = cv2.cvtColor(bull_debug_img, cv2.COLOR_GRAY2BGR)
                cv2.putText(bull_debug_img, "Red mask", (5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(display, "Board NOT FOUND — adjust Tuner sliders", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # ── Debug windows ──────────────────────────
        if bull_debug_img is not None:
            cv2.imshow("Bullseye Debug", bull_debug_img)
        else:
            blank = np.zeros((200, 200, 3), dtype=np.uint8)
            cv2.putText(blank, "No detection", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            cv2.imshow("Bullseye Debug", blank)

        edge_display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        if smoothed_ellipse is not None:
            cv2.ellipse(edge_display,
                        ((int(smoothed_ellipse[0][0]), int(smoothed_ellipse[0][1])),
                         (int(smoothed_ellipse[1][0]), int(smoothed_ellipse[1][1])),
                         smoothed_ellipse[2]),
                        (0, 255, 0), 2)
        cv2.imshow("Edge Debug", edge_display)

        # ── Warped view ────────────────────────────
        if last_corners is not None:
            h, w = frame.shape[:2]
            clamped = last_corners.copy()
            clamped[:, 0] = np.clip(clamped[:, 0], 0, w - 1)
            clamped[:, 1] = np.clip(clamped[:, 1], 0, h - 1)
            warped, M = warp_board(frame, clamped)

            cx, cy = OUTPUT_SIZE // 2, OUTPUT_SIZE // 2
            cv2.line(warped,   (cx - 20, cy), (cx + 20, cy), (0, 255, 255), 1)
            cv2.line(warped,   (cx, cy - 20), (cx, cy + 20), (0, 255, 255), 1)
            cv2.circle(warped, (cx, cy), OUTPUT_SIZE // 2 - 2, (0, 255, 0), 1)
            cv2.imshow("Warped Board", warped)
        else:
            blank = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE, 3), dtype=np.uint8)
            cv2.putText(blank, "Waiting for detection...", (30, OUTPUT_SIZE // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
            cv2.imshow("Warped Board", blank)

        cv2.imshow("Live Feed", display)

        # ── Keys ───────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('l'):
            if smoothed_ellipse is not None:
                state.lock(corrected_cx, corrected_cy, board_radius)

        elif key == ord('u'):
            state.unlock()
            ellipse_history.clear()
            bullseye_history.clear()

        elif key == ord('s'):
            cv2.imwrite("saved_frame.jpg", frame)
            cv2.imwrite("saved_edges.jpg", edges)
            if last_corners is not None:
                warped_save, _ = warp_board(frame, last_corners)
                cv2.imwrite("saved_warped.jpg", warped_save)
                print("Saved: saved_frame.jpg, saved_edges.jpg, saved_warped.jpg")
            else:
                print("Saved: saved_frame.jpg, saved_edges.jpg")

        elif key == ord('e'):
            if smoothed_ellipse is not None:
                r = state.locked_radius if state.locked else board_radius
                print("\n── Hardcode these into CONFIGURATION ──")
                print(f"  CANNY_LOW              = {canny_low}")
                print(f"  CANNY_HIGH             = {canny_high}")
                print(f"  BLUR_SIZE              = {blur}")
                print(f"  MIN_BOARD_PX           = {min_size}")
                print(f"  MAX_BOARD_PX           = {max_size}")
                print(f"  MIN_CIRCULARITY        = {circularity:.2f}")
                print(f"  BULLSEYE_SEARCH_RADIUS = {search_r:.2f}")
                print(f"  RADIUS_PADDING         = {pad:.2f}")
                print(f"  RED_LOW_1  = np.array([0,   {sat_min}, {val_min}])")
                print(f"  RED_HIGH_1 = np.array([10,  255, 255])")
                print(f"  RED_LOW_2  = np.array([170, {sat_min}, {val_min}])")
                print(f"  RED_HIGH_2 = np.array([180, 255, 255])")
                print(f"  # Locked values (for fully hardcoded mode):")
                print(f"  # center=({state.locked_cx},{state.locked_cy})  radius={r:.1f}px")
            else:
                print("No detection yet.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
