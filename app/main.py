import base64
import time
import logging

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.detector import Detector
from app.scoring import RINGS, SECTORS, estimate_homography

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="DartBuddy", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

MODEL_PATH = "model/best.pt"


@app.on_event("startup")
async def startup() -> None:
    log.info("Loading model from %s ...", MODEL_PATH)
    try:
        app.state.detector = Detector(MODEL_PATH)
        log.info("Model loaded.")
    except Exception as exc:
        log.error("Failed to load model: %s", exc)
        app.state.detector = None


def _decode_frame(raw: bytes) -> np.ndarray:
    arr   = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=422, detail="Could not decode image.")
    return frame


def _norm_to_pixel(nx, ny, board_radius, H_inv):
    pt = np.array([[[nx * board_radius, ny * board_radius]]], dtype=np.float32)
    px = cv2.perspectiveTransform(pt, H_inv)[0][0]
    return int(round(px[0])), int(round(px[1]))


def _draw_ring_overlay(frame: np.ndarray, cal_pts: list, board_radius: float) -> np.ndarray:
    overlay   = frame.copy()
    valid_cal = np.array([p for p in cal_pts if p is not None], dtype=np.float32)
    if len(valid_cal) < 4:
        return overlay

    H = estimate_homography(valid_cal, board_radius)
    if H is None:
        return overlay

    H_inv = np.linalg.inv(H)
    N     = 180

    ring_styles = [
        (RINGS['bull'],         (0,   0,   255), 2),
        (RINGS['outer_bull'],   (0,   140, 255), 2),
        (RINGS['triple_inner'], (255, 200, 0),   2),
        (RINGS['triple_outer'], (0,   255, 255), 3),
        (RINGS['double_inner'], (255, 200, 0),   2),
        (RINGS['double_outer'], (0,   255, 0),   3),
    ]

    for r, color, thickness in ring_styles:
        pts = []
        for k in range(N):
            a = 2 * np.pi * k / N
            pts.append(_norm_to_pixel(r * np.sin(a), -r * np.cos(a), board_radius, H_inv))
        cv2.polylines(overlay, [np.array(pts, dtype=np.int32)],
                      isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    for s in range(20):
        a  = np.radians(s * 18 - 9)
        p1 = _norm_to_pixel(RINGS['outer_bull']   * np.sin(a), -RINGS['outer_bull']   * np.cos(a), board_radius, H_inv)
        p2 = _norm_to_pixel(RINGS['double_outer'] * np.sin(a), -RINGS['double_outer'] * np.cos(a), board_radius, H_inv)
        cv2.line(overlay, p1, p2, (180, 180, 180), 1, cv2.LINE_AA)

    for s in range(20):
        a       = np.radians(s * 18)
        label_r = (RINGS['double_inner'] + RINGS['double_outer']) / 2 + 0.02
        px, py  = _norm_to_pixel(label_r * np.sin(a), -label_r * np.cos(a), board_radius, H_inv)
        cv2.putText(overlay, str(SECTORS[s]), (px - 8, py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    return cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)


@app.get("/health")
async def health(request: Request) -> dict:
    return {
        "status":       "ok",
        "model_loaded": request.app.state.detector is not None,
    }


@app.post("/detect")
async def detect(request: Request, file: UploadFile = File(...)) -> dict:
    detector: Detector | None = request.app.state.detector
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    frame = _decode_frame(await file.read())

    t0 = time.perf_counter()
    try:
        result = detector.detect(frame)
    except Exception as exc:
        log.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(exc))

    result["inference_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return result


@app.post("/debug")
async def debug(request: Request, file: UploadFile = File(...)) -> dict:
    detector: Detector | None = request.app.state.detector
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    frame = _decode_frame(await file.read())

    t0 = time.perf_counter()
    try:
        result = detector.detect(frame)
    except Exception as exc:
        log.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(exc))

    result["inference_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    if result["board"]["detected"]:
        cal_pts         = result["board"]["keypoints"]
        x1, y1, x2, y2 = result["board"]["bbox"]
        board_radius    = max(x2 - x1, y2 - y1) / 2
        annotated       = _draw_ring_overlay(frame, cal_pts, board_radius)

        for dart in result["darts"]:
            tx, ty = int(dart["tip"][0]), int(dart["tip"][1])
            color  = (0, 255, 100) if dart["value"] > 0 else (0, 100, 255)
            cv2.circle(annotated, (tx, ty), 8, color, -1)
            cv2.circle(annotated, (tx, ty), 8, (255, 255, 255), 1)
            cv2.putText(annotated, dart["score"], (tx - 16, ty - 14),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2, cv2.LINE_AA)
    else:
        annotated = frame

    _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
    result["debug_image"] = base64.b64encode(buf).decode('utf-8')
    return result


app.mount("/", StaticFiles(directory=".", html=True), name="static")