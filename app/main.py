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


def _ring_polygon(r, board_radius, H_inv, N=180):
    pts = []
    for k in range(N):
        a = 2 * np.pi * k / N
        pts.append(_norm_to_pixel(r * np.sin(a), -r * np.cos(a), board_radius, H_inv))
    return np.array(pts, dtype=np.int32)


def _draw_band_stripes(canvas, outer_poly, inner_poly, color, stripe_gap=6):
    """Fill the band between two ring polygons with diagonal stripes."""
    h, w = canvas.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [outer_poly], 255)
    cv2.fillPoly(mask, [inner_poly], 0)

    stripe_layer = np.zeros_like(canvas)
    for x in range(-h, w + h, stripe_gap):
        cv2.line(stripe_layer,
                 (x, 0), (x + h, h),
                 color, 1, cv2.LINE_AA)

    canvas[mask == 255] = (
        canvas[mask == 255] * 0.4 +
        stripe_layer[mask == 255] * 0.6
    ).astype(np.uint8)


def _draw_ring_overlay(frame: np.ndarray, cal_pts: list, board_radius: float) -> np.ndarray:
    valid_cal = np.array([p for p in cal_pts if p is not None], dtype=np.float32)
    if len(valid_cal) < 4:
        return frame.copy()

    H = estimate_homography(valid_cal, board_radius)
    if H is None:
        return frame.copy()

    H_inv    = np.linalg.inv(H)
    annotated = frame.copy()

    # Build ring polygons
    poly = {name: _ring_polygon(r, board_radius, H_inv) for name, r in RINGS.items()}

    # Stripe-fill triple band (triple_inner -> triple_outer) in cyan
    _draw_band_stripes(annotated, poly['triple_outer'], poly['triple_inner'], (0, 255, 255), stripe_gap=5)

    # Stripe-fill double band (double_inner -> double_outer) in green
    _draw_band_stripes(annotated, poly['double_outer'], poly['double_inner'], (0, 255, 80), stripe_gap=5)

    # Draw ring boundary lines on top
    ring_lines = [
        ('bull',         (0,   0,   255), 2),
        ('outer_bull',   (0,   140, 255), 2),
        ('triple_inner', (0,   255, 255), 1),
        ('triple_outer', (0,   255, 255), 2),
        ('double_inner', (0,   255, 80),  1),
        ('double_outer', (0,   255, 80),  2),
    ]
    for name, color, thickness in ring_lines:
        cv2.polylines(annotated, [poly[name]], isClosed=True,
                      color=color, thickness=thickness, lineType=cv2.LINE_AA)

    # Sector dividers
    for s in range(20):
        a  = np.radians(s * 18 - 9)
        p1 = _norm_to_pixel(RINGS['outer_bull']   * np.sin(a), -RINGS['outer_bull']   * np.cos(a), board_radius, H_inv)
        p2 = _norm_to_pixel(RINGS['double_outer'] * np.sin(a), -RINGS['double_outer'] * np.cos(a), board_radius, H_inv)
        cv2.line(annotated, p1, p2, (160, 160, 160), 1, cv2.LINE_AA)

    # Sector number labels
    for s in range(20):
        a       = np.radians(s * 18)
        label_r = (RINGS['double_inner'] + RINGS['double_outer']) / 2 + 0.02
        px, py  = _norm_to_pixel(label_r * np.sin(a), -label_r * np.cos(a), board_radius, H_inv)
        cv2.putText(annotated, str(SECTORS[s]), (px - 8, py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    return annotated


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
            cv2.circle(annotated, (tx, ty), 8, (255, 255, 255), 2)
            cv2.putText(annotated, dart["score"], (tx - 16, ty - 14),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2, cv2.LINE_AA)
    else:
        annotated = frame

    _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
    result["debug_image"] = base64.b64encode(buf).decode('utf-8')
    return result


app.mount("/", StaticFiles(directory=".", html=True), name="static")