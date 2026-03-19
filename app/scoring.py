import cv2
import numpy as np

SECTORS = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17,
           3, 19, 7, 16, 8, 11, 14, 9, 12, 5]

RINGS = {
    'bull':         0.035,
    'outer_bull':   0.095,
    'triple_inner': 0.600,
    'triple_outer': 0.650,
    'double_inner': 0.950,
    'double_outer': 1.000,
}

CAL_NORM = np.array([
    [ 0.0, -1.0],   # top
    [ 0.0,  1.0],   # bottom
    [-1.0,  0.0],   # left
    [ 1.0,  0.0],   # right
], dtype=np.float32)


def dart_score(nx: float, ny: float) -> str:
    dist = np.hypot(nx, ny)
    if dist <= RINGS['bull']:       return 'Bull (50)'
    if dist <= RINGS['outer_bull']: return 'Outer Bull (25)'
    angle = (np.degrees(np.arctan2(nx, -ny)) + 360 + 9) % 360
    sector = SECTORS[int(angle / 18) % 20]
    if dist > RINGS['double_outer']:  return 'Miss'
    if dist >= RINGS['double_inner']: return f'D{sector}'
    if dist >= RINGS['triple_outer']: return f'T{sector}'
    return str(sector)


def score_value(label: str) -> int:
    if label == 'Bull (50)':       return 50
    if label == 'Outer Bull (25)': return 25
    if label == 'Miss':            return 0
    if label.startswith('T'):
        try: return int(label[1:]) * 3
        except: return 0
    if label.startswith('D'):
        try: return int(label[1:]) * 2
        except: return 0
    try:   return int(label)
    except: return 0


def estimate_homography(cal_pts_px: np.ndarray, board_radius: float):
    if cal_pts_px.shape[0] < 4:
        return None
    dst = CAL_NORM * board_radius
    try:
        H, _ = cv2.findHomography(
            cal_pts_px.astype(np.float32),
            dst.astype(np.float32),
            cv2.RANSAC, 5.0,
        )
        return H
    except Exception:
        return None


def pixel_to_board_norm(px, py, board_center, board_radius, H):
    if H is not None:
        pt = np.array([[[float(px), float(py)]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, np.linalg.inv(H))[0][0]
        return transformed[0] / board_radius, transformed[1] / board_radius
    cx, cy = board_center
    return (px - cx) / board_radius, (py - cy) / board_radius