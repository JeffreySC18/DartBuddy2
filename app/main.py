import time
import logging

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.detector import Detector

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


@app.get("/health")
async def health(request: Request) -> dict:
    return {
        "status": "ok",
        "model_loaded": request.app.state.detector is not None,
    }


@app.post("/detect")
async def detect(request: Request, file: UploadFile = File(...)) -> dict:
    detector: Detector | None = request.app.state.detector
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    raw = await file.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=422, detail="Could not decode image.")

    t0 = time.perf_counter()
    try:
        result = detector.detect(frame)
    except Exception as exc:
        log.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(exc))

    result["inference_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return result