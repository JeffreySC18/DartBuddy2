import numpy as np
from ultralytics import YOLO

from app.scoring import (
    dart_score, score_value,
    estimate_homography, pixel_to_board_norm,
)


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

        # Pass 1: board
        for i, box in enumerate(result.boxes):
            if names[int(box.cls[0])] != "board":
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            board_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            board_radius = max(x2 - x1, y2 - y1) / 2

            cal_pts = []
            if result.keypoints is not None and i < len(result.keypoints):
                kps = result.keypoints[i].xy[0].cpu().numpy()
                for kp in kps:
                    if kp[0] == 0 and kp[1] == 0:
                        cal_pts.append(None)
                    else:
                        cal_pts.append([round(float(kp[0]), 2), round(float(kp[1]), 2)])

            valid_cal = np.array([p for p in cal_pts if p is not None], dtype=np.float32)
            if len(valid_cal) >= 4:
                H = estimate_homography(valid_cal, board_radius)

            board = {
                "detected":   True,
                "confidence": round(float(box.conf[0]), 3),
                "bbox":       [x1, y1, x2, y2],
                "keypoints":  cal_pts,
            }
            break

        # Pass 2: darts (max 3)
        dart_idx = 0
        for i, box in enumerate(result.boxes):
            if names[int(box.cls[0])] != "dart":
                continue
            if dart_idx >= 3:
                break

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Tip keypoint — fall back to box bottom-centre
            tip_px, tip_py = (x1 + x2) / 2, float(y2)
            if result.keypoints is not None and i < len(result.keypoints):
                kp = result.keypoints[i].xy[0].cpu().numpy()
                if kp.shape[0] > 0 and not (kp[0, 0] == 0 and kp[0, 1] == 0):
                    tip_px, tip_py = float(kp[0, 0]), float(kp[0, 1])

            label, value = "?", 0
            if board_center and board_radius:
                nx, ny = pixel_to_board_norm(tip_px, tip_py, board_center, board_radius, H)
                label  = dart_score(nx, ny)
                value  = score_value(label)

            darts.append({
                "tip":        [round(tip_px, 2), round(tip_py, 2)],
                "bbox":       [x1, y1, x2, y2],
                "confidence": round(float(box.conf[0]), 3),
                "score":      label,
                "value":      value,
            })
            dart_idx += 1

        return {
            "board": board,
            "darts": darts,
            "total": sum(d["value"] for d in darts),
        }