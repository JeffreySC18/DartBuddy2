import logging
import numpy as np
import cv2
from ultralytics import YOLO

from app.scoring import (
    dart_score, score_value,
    estimate_homography, pixel_to_board_norm,
    CAL_NORM, RINGS
)

log = logging.getLogger(__name__)


class Detector:
    def __init__(self, model_path: str, conf: float = 0.45, device: str = "cuda"):
        self.model  = YOLO(model_path)
        self.conf   = conf
        self.device = device

    def detect(self, frame: np.ndarray) -> dict:
        results = self.model.predict(
            frame,
            conf=self.conf,
            device=self.device,
            verbose=False,
            imgsz=800,
        )
        return self._parse(results[0])

    def _parse(self, result) -> dict:
        names = result.names
        board  = {"detected": False, "confidence": 0.0, "bbox": [], "keypoints": []}
        darts  = []

        board_center = None
        board_radius = None
        H            = None

        # ── Pass 1: board ────────────────────────────────────────────────
        for i, box in enumerate(result.boxes):
            if names[int(box.cls[0])] != "board":
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            board_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            board_radius = max(x2 - x1, y2 - y1) / 2

            log.info("=== BOARD DETECTED ===")
            log.info(f"  bbox:         ({x1}, {y1}) -> ({x2}, {y2})")
            log.info(f"  board_center: {board_center}")
            log.info(f"  board_radius: {board_radius:.1f}px")
            log.info(f"  confidence:   {float(box.conf[0]):.3f}")

            # Keypoints
            cal_pts = []
            if result.keypoints is not None and i < len(result.keypoints):
                kps = result.keypoints[i].xy[0].cpu().numpy()
                kp_labels = ['top', 'right', 'bottom', 'left']
                log.info(f"  raw keypoints from model (should be top/right/bottom/left):")
                for j, kp in enumerate(kps):
                    label = kp_labels[j] if j < len(kp_labels) else f'kp{j}'
                    if kp[0] == 0 and kp[1] == 0:
                        log.info(f"    [{j}] {label}: MISSING (0,0)")
                        cal_pts.append(None)
                    else:
                        log.info(f"    [{j}] {label}: ({kp[0]:.1f}, {kp[1]:.1f})")
                        cal_pts.append([round(float(kp[0]), 2), round(float(kp[1]), 2)])

            valid_cal = np.array([p for p in cal_pts if p is not None], dtype=np.float32)
            log.info(f"  valid calibration points: {len(valid_cal)}/4")

            # Log CAL_NORM being used
            log.info(f"  CAL_NORM in use:")
            cal_labels = ['top', 'right', 'bottom', 'left']
            for j, row in enumerate(CAL_NORM):
                log.info(f"    [{j}] {cal_labels[j]}: {row}")

            if len(valid_cal) >= 4:
                # Log what dst points we're solving to
                dst = CAL_NORM * board_radius
                log.info(f"  homography dst points (CAL_NORM * {board_radius:.1f}):")
                for j, (src_pt, dst_pt) in enumerate(zip(valid_cal, dst)):
                    log.info(f"    [{j}] src=({src_pt[0]:.1f},{src_pt[1]:.1f}) -> dst=({dst_pt[0]:.1f},{dst_pt[1]:.1f})")

                H = estimate_homography(valid_cal, board_radius)
                if H is None:
                    log.warning("  estimate_homography returned None — check if points are collinear")
                else:
                    log.info(f"  H matrix:\n{H}")

                    # Sanity check: transform each cal point back through H
                    # They should land near their CAL_NORM * board_radius positions
                    log.info("  Sanity check — transforming cal keypoints through H:")
                    for j, pt in enumerate(valid_cal):
                        p = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
                        t = cv2.perspectiveTransform(p, H)[0][0]
                        nx_check = t[0] / board_radius
                        ny_check = t[1] / board_radius
                        dist_check = np.hypot(nx_check, ny_check)
                        expected = CAL_NORM[j] if j < len(CAL_NORM) else ['?','?']
                        log.info(f"    [{j}] pixel ({pt[0]:.0f},{pt[1]:.0f}) -> norm ({nx_check:.3f}, {ny_check:.3f}) dist={dist_check:.3f}  [expected norm ~{expected}]")

            board = {
                "detected":   True,
                "confidence": round(float(box.conf[0]), 3),
                "bbox":       [x1, y1, x2, y2],
                "keypoints":  cal_pts,
            }
            break

        # ── Pass 2: darts ────────────────────────────────────────────────
        dart_idx = 0
        for i, box in enumerate(result.boxes):
            if names[int(box.cls[0])] != "dart":
                continue
            if dart_idx >= 3:
                break

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            tip_px, tip_py = (x1 + x2) / 2, float(y2)
            if result.keypoints is not None and i < len(result.keypoints):
                kp = result.keypoints[i].xy[0].cpu().numpy()
                if kp.shape[0] > 0 and not (kp[0, 0] == 0 and kp[0, 1] == 0):
                    tip_px, tip_py = float(kp[0, 0]), float(kp[0, 1])

            log.info(f"--- DART {dart_idx+1} ---")
            log.info(f"  tip pixel:  ({tip_px:.1f}, {tip_py:.1f})")
            log.info(f"  bbox:       ({x1},{y1}) -> ({x2},{y2})")
            log.info(f"  confidence: {float(box.conf[0]):.3f}")

            label, value = "?", 0
            if board_center and board_radius:
                nx, ny = pixel_to_board_norm(tip_px, tip_py, board_center, board_radius, H)
                dist   = np.hypot(nx, ny)

                # Show which ring it falls in
                ring = "unknown"
                for ring_name, ring_max in [
                    ("bull",           RINGS['bull']),
                    ("outer_bull",     RINGS['outer_bull']),
                    ("single_inner",   RINGS['triple_inner']),
                    ("triple",         RINGS['triple_outer']),
                    ("single_outer",   RINGS['double_inner']),
                    ("double",         RINGS['double_outer']),
                ]:
                    if dist <= ring_max:
                        ring = ring_name
                        break
                else:
                    ring = "MISS (outside double)"

                log.info(f"  norm:       ({nx:.4f}, {ny:.4f})")
                log.info(f"  dist:       {dist:.4f}  ->  ring: {ring}")
                log.info(f"  Ring boundaries for reference:")
                log.info(f"    bull         <= 0.035")
                log.info(f"    outer_bull   <= 0.095")
                log.info(f"    triple_inner <= 0.600")
                log.info(f"    triple_outer <= 0.650")
                log.info(f"    double_inner <= 0.950")
                log.info(f"    double_outer <= 1.000")

                label  = dart_score(nx, ny)
                value  = score_value(label)
                log.info(f"  SCORE: {label} ({value} pts)")

            darts.append({
                "tip":        [round(tip_px, 2), round(tip_py, 2)],
                "bbox":       [x1, y1, x2, y2],
                "confidence": round(float(box.conf[0]), 3),
                "score":      label,
                "value":      value,
            })
            dart_idx += 1

        total = sum(d["value"] for d in darts)
        log.info(f"=== TOTAL: {total} ===")

        return {
            "board": board,
            "darts": darts,
            "total": total,
        }