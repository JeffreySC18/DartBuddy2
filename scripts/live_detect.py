"""
Live dartboard detection and scoring using a trained YOLOv8-Pose model.

Displays:
  - Board bounding box + 4 calibration keypoints
  - Dart bounding boxes + tip keypoints
  - Score label per dart (e.g. T20, D5, Bull)
  - Running score total for up to 3 darts
  - Per-dart confidence
  - FPS counter
  - Inference time (ms)
  - Board detection confidence
  - Mini scoreboard panel (right side)
"""

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit("pip install ultralytics")


# Dartboard geometry

SECTORS = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17,
           3, 19, 7, 16, 8, 11, 14, 9, 12, 5]

# Ring radii as fractions of board radius
RINGS = {
    'bull':         0.035,
    'outer_bull':   0.095,
    'triple_inner': 0.600,
    'triple_outer': 0.650,
    'double_inner': 0.950,
    'double_outer': 1.000,
}

# Calibration point positions in normalised board space
# Order: top, right, bottom, left  (at the double ring radius)
CAL_NORM = np.array([
    [ 0.0, -1.0],
    [ 1.0,  0.0],
    [ 0.0,  1.0],
    [-1.0,  0.0],
], dtype=np.float32)


def score_value(label: str) -> int:
    """Convert a score label like 'T20', 'D5', 'Bull' to its integer value."""
    if label == 'Bull (50)':        return 50
    if label == 'Outer Bull (25)':  return 25
    if label == 'Miss':             return 0
    if label.startswith('T'):
        try: return int(label[1:]) * 3
        except: return 0
    if label.startswith('D'):
        try: return int(label[1:]) * 2
        except: return 0
    try:   return int(label)
    except: return 0


def dart_score(nx: float, ny: float) -> str:
    dist = np.hypot(nx, ny)
    if dist <= RINGS['bull']:        return 'Bull (50)'
    if dist <= RINGS['outer_bull']:  return 'Outer Bull (25)'
    angle = (np.degrees(np.arctan2(nx, -ny)) + 360 + 9) % 360
    sector = SECTORS[int(angle / 18) % 20]
    if dist > RINGS['double_outer']:  return 'Miss'
    if dist >= RINGS['double_inner']: return f'D{sector}'
    if dist >= RINGS['triple_outer']: return f'T{sector}'
    return str(sector)


# Perspective correction

def estimate_homography(cal_pts_px: np.ndarray, board_radius: float):
    """
    Compute homography from pixel space to normalised board space.
    cal_pts_px: (4,2) detected calibration keypoints in pixels.
    board_radius: radius of board in pixels.
    Returns H (3x3) or None.
    """
    if cal_pts_px.shape[0] < 4:
        return None
    dst = CAL_NORM * board_radius
    try:
        H, mask = cv2.findHomography(
            cal_pts_px.astype(np.float32),
            dst.astype(np.float32),
            cv2.RANSAC, 5.0
        )
        return H
    except Exception:
        return None


def pixel_to_board_norm(px, py, board_center, board_radius, H):
    """Map a pixel coordinate to normalised board space (centre=0, radius=1)."""
    if H is not None:
        pt = np.array([[[float(px), float(py)]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, np.linalg.inv(H))[0][0]
        return transformed[0] / board_radius, transformed[1] / board_radius
    # Fallback: assume camera is perpendicular
    cx, cy = board_center
    return (px - cx) / board_radius, (py - cy) / board_radius


# Colours and fonts

C_BOARD    = (0,   210, 255)   # amber
C_DART     = (60,  60,  255)   # red
C_KP_BOARD = (0,   255, 120)   # green
C_KP_DART  = (0,   140, 255)   # orange
C_PANEL    = (20,  20,  20)    # dark panel background
C_TEXT     = (240, 240, 240)   # white text
C_ACCENT   = (0,   200, 255)   # amber accent
C_GOOD     = (80,  220, 80)    # green for good scores
C_WARN     = (0,   160, 255)   # orange for misses

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD  = cv2.FONT_HERSHEY_DUPLEX


def put_text(img, text, pos, scale=0.6, color=C_TEXT, thickness=1, font=FONT):
    cv2.putText(img, text, pos, font, scale, color, thickness, cv2.LINE_AA)


# HUD panel

PANEL_W = 260

def draw_panel(canvas: np.ndarray, dart_scores: list, fps: float,
               inf_ms: float, board_conf: float, frozen: bool):
    """
    Draw the right-side scoreboard panel onto canvas in-place.
    dart_scores: list of (label_str, int_value, conf) tuples, up to 3.
    """
    h, w = canvas.shape[:2]
    px = w - PANEL_W

    # Panel background
    cv2.rectangle(canvas, (px, 0), (w, h), C_PANEL, -1)
    cv2.line(canvas, (px, 0), (px, h), C_ACCENT, 2)

    y = 30

    # Title
    put_text(canvas, 'DartBuddy', (px + 14, y), 0.85, C_ACCENT, 2, FONT_BOLD)
    y += 34
    cv2.line(canvas, (px + 10, y), (w - 10, y), C_ACCENT, 1)
    y += 18

    # System stats
    put_text(canvas, 'SYSTEM', (px + 14, y), 0.5, C_ACCENT, 1)
    y += 22
    put_text(canvas, f'FPS        {fps:5.1f}', (px + 14, y), 0.55, C_TEXT, 1)
    y += 22
    put_text(canvas, f'Inference  {inf_ms:4.1f} ms', (px + 14, y), 0.55, C_TEXT, 1)
    y += 22

    board_color = C_GOOD if board_conf > 0.8 else C_WARN
    board_label = f'{board_conf:.0%}' if board_conf > 0 else 'NOT FOUND'
    put_text(canvas, f'Board conf {board_label}', (px + 14, y), 0.55, board_color, 1)
    y += 22

    if frozen:
        put_text(canvas, '[ FROZEN ]', (px + 14, y), 0.55, (0, 200, 255), 2)
    y += 28

    cv2.line(canvas, (px + 10, y), (w - 10, y), C_ACCENT, 1)
    y += 18

    # Dart scores
    put_text(canvas, 'DARTS', (px + 14, y), 0.5, C_ACCENT, 1)
    y += 22

    total = 0
    for i in range(3):
        label = f'Dart {i+1}'
        score_str = '---'
        val = 0
        conf_str = ''
        color = (100, 100, 100)

        if i < len(dart_scores):
            s_label, s_val, s_conf = dart_scores[i]
            score_str = s_label
            val = s_val
            total += val
            conf_str = f'  ({s_conf:.0%})'
            color = C_GOOD if s_val > 0 else C_WARN

        put_text(canvas, f'Dart {i+1}', (px + 14, y), 0.55, C_TEXT, 1)
        put_text(canvas, score_str, (px + 90, y), 0.6, color, 2)
        put_text(canvas, conf_str, (px + 90, y + 16), 0.42, (150, 150, 150), 1)
        y += 48

    cv2.line(canvas, (px + 10, y), (w - 10, y), C_ACCENT, 1)
    y += 18

    # Total
    put_text(canvas, 'TOTAL', (px + 14, y), 0.5, C_ACCENT, 1)
    y += 30
    total_color = C_GOOD if total > 0 else C_TEXT
    put_text(canvas, str(total), (px + 14, y), 1.4, total_color, 3, FONT_BOLD)
    y += 52

    cv2.line(canvas, (px + 10, y), (w - 10, y), C_ACCENT, 1)
    y += 18

    # Controls
    put_text(canvas, 'CONTROLS', (px + 14, y), 0.5, C_ACCENT, 1)
    y += 22
    for key, desc in [('Q', 'Quit'), ('S', 'Screenshot'),
                      ('R', 'Reset scores'), ('SPC', 'Freeze')]:
        put_text(canvas, f'{key:<4} {desc}', (px + 14, y), 0.48, (160, 160, 160), 1)
        y += 20


# Main detection + drawing

def process_frame(frame: np.ndarray, result, fps: float,
                  inf_ms: float, frozen: bool) -> tuple:
    """
    Draw all detections and the HUD panel onto frame.
    Returns (annotated_frame, dart_scores_list).
    dart_scores_list: [(label, value, conf), ...]
    """
    h, w = frame.shape[:2]
    # Make canvas wider to fit panel
    canvas = np.zeros((h, w + PANEL_W, 3), dtype=np.uint8)
    canvas[:, :w] = frame

    names = result.names  # {0: 'board', 1: 'dart'}

    board_center = None
    board_radius = None
    board_conf   = 0.0
    cal_pts      = []
    H            = None
    dart_scores  = []

    # Pass 1: board
    for i, box in enumerate(result.boxes):
        if names[int(box.cls[0])] != 'board':
            continue

        board_conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        board_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        board_radius = max(x2 - x1, y2 - y1) / 2

        # Board bounding box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), C_BOARD, 2)
        put_text(canvas, f'Board  {board_conf:.0%}',
                 (x1, y1 - 8), 0.6, C_BOARD, 2)

        # Board center cross
        cx, cy = board_center
        cv2.drawMarker(canvas, (cx, cy), C_BOARD,
                       cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)

        # Calibration keypoints
        if result.keypoints is not None and i < len(result.keypoints):
            kps = result.keypoints[i].xy[0].cpu().numpy()  # (4,2)
            labels = ['T', 'R', 'B', 'L']
            for j, kp in enumerate(kps):
                if kp[0] == 0 and kp[1] == 0:
                    continue
                kx, ky = int(kp[0]), int(kp[1])
                cv2.circle(canvas, (kx, ky), 7, C_KP_BOARD, -1)
                cv2.circle(canvas, (kx, ky), 7, (255,255,255), 1)
                put_text(canvas, labels[j], (kx + 9, ky + 5),
                         0.45, C_KP_BOARD, 1)
                cal_pts.append(kp)

        # Compute homography from calibration points
        if len(cal_pts) >= 4:
            H = estimate_homography(np.array(cal_pts[:4]), board_radius)

        break

    # Pass 2: darts
    dart_idx = 0
    for i, box in enumerate(result.boxes):
        if names[int(box.cls[0])] != 'dart':
            continue
        if dart_idx >= 3:
            break

        dart_conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Tip keypoint
        tip_px, tip_py = (x1 + x2) / 2, float(y2)
        if result.keypoints is not None and i < len(result.keypoints):
            kp = result.keypoints[i].xy[0].cpu().numpy()
            if kp.shape[0] > 0 and not (kp[0,0] == 0 and kp[0,1] == 0):
                tip_px, tip_py = float(kp[0,0]), float(kp[0,1])

        # Score
        label = '?'
        val   = 0
        if board_center and board_radius:
            nx, ny = pixel_to_board_norm(
                tip_px, tip_py, board_center, board_radius, H)
            label = dart_score(nx, ny)
            val   = score_value(label)

        dart_scores.append((label, val, dart_conf))

        # Draw dart box
        dart_color = C_GOOD if val > 0 else C_WARN
        cv2.rectangle(canvas, (x1, y1), (x2, y2), C_DART, 2)

        # Tip dot
        tx, ty = int(tip_px), int(tip_py)
        cv2.circle(canvas, (tx, ty), 6, C_KP_DART, -1)
        cv2.circle(canvas, (tx, ty), 6, (255,255,255), 1)

        # Score label above tip
        score_display = label.replace(' (50)', '').replace(' (25)', '')
        (tw, th), _ = cv2.getTextSize(score_display, FONT_BOLD, 0.8, 2)
        lx = max(0, tx - tw // 2)
        ly = max(th + 4, ty - 12)
        cv2.rectangle(canvas, (lx - 4, ly - th - 4),
                      (lx + tw + 4, ly + 4), (0,0,0), -1)
        put_text(canvas, score_display, (lx, ly), 0.8, dart_color, 2, FONT_BOLD)

        # Dart number tag
        put_text(canvas, f'#{dart_idx+1} {dart_conf:.0%}',
                 (x1, y1 - 8), 0.5, C_DART, 1)

        dart_idx += 1

    # FPS overlay 
    cv2.rectangle(canvas, (0, 0), (190, 36), (0,0,0), -1)
    put_text(canvas, f'FPS: {fps:5.1f}   {inf_ms:.0f}ms',
             (8, 24), 0.65, C_ACCENT, 2)

    # Right panel
    draw_panel(canvas, dart_scores, fps, inf_ms, board_conf, frozen)

    return canvas, dart_scores


# Entry points

def run_camera(model_path: str, cam_idx: int = 0,
               conf: float = 0.45, device: str = 'cuda'):
    model = YOLO(model_path)
    cap   = cv2.VideoCapture(cam_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {cam_idx}")

    fps_buf  = deque(maxlen=30)
    frozen   = False
    frozen_frame = None
    shot_n   = 0

    print("DartBuddy live detection started.")
    print("  Q=quit  S=screenshot  R=reset  SPACE=freeze")

    while True:
        if not frozen:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed.")
                break

            t0      = time.perf_counter()
            results = model.predict(frame, conf=conf, device=device,
                                    verbose=False, imgsz=800)
            inf_ms  = (time.perf_counter() - t0) * 1000
            fps_buf.append(inf_ms)
            fps     = 1000 / (sum(fps_buf) / len(fps_buf))

            canvas, dart_scores = process_frame(
                frame, results[0], fps, inf_ms, frozen)
            frozen_frame = canvas
        else:
            canvas = frozen_frame
            fps    = 0.0

        cv2.imshow('DartBuddy', canvas)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('s'):
            shot_n += 1
            fname = f'dartbuddy_shot_{shot_n:03d}.jpg'
            cv2.imwrite(fname, canvas)
            print(f'Screenshot saved: {fname}')

        elif key == ord('r'):
            print('Scores reset.')

        elif key == ord(' '):
            frozen = not frozen
            print('Frozen.' if frozen else 'Resumed.')

    cap.release()
    cv2.destroyAllWindows()


def run_image(model_path: str, image_path: str,
              conf: float = 0.45, device: str = 'cuda'):
    model  = YOLO(model_path)
    frame  = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    t0      = time.perf_counter()
    results = model.predict(frame, conf=conf, device=device,
                            verbose=False, imgsz=800)
    inf_ms  = (time.perf_counter() - t0) * 1000

    canvas, dart_scores = process_frame(frame, results[0], 0.0, inf_ms, False)

    print("\nDetected scores:")
    total = 0
    for i, (label, val, dart_conf) in enumerate(dart_scores):
        print(f"  Dart {i+1}: {label}  ({val} pts)  conf={dart_conf:.0%}")
        total += val
    print(f"  Total: {total}")

    cv2.imshow('DartBuddy', canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Auto-save result
    out = Path(image_path).stem + '_result.jpg'
    cv2.imwrite(out, canvas)
    print(f"Result saved to {out}")


# CLI

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='DartBuddy live detector')
    p.add_argument('--model',  required=True,
                   help='Path to best.pt  e.g. runs/pose/runs/dartboard_pose/train/weights/best.pt')
    p.add_argument('--source', default='0',
                   help='Camera index (0, 1, ...) or path to an image file')
    p.add_argument('--conf',   type=float, default=0.45,
                   help='Detection confidence threshold (default 0.45)')
    p.add_argument('--device', default='cuda',
                   choices=['cuda', 'cpu', 'mps'],
                   help='Inference device (default cuda)')
    args = p.parse_args()

    src = args.source
    if src.isdigit():
        run_camera(args.model, int(src), args.conf, args.device)
    else:
        run_image(args.model, src, args.conf, args.device)