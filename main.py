# main.py

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware

from app.models.text_model import TextToxicityModel
from app.models.image_model import ImageNSFWModel
from app.models.video_model import VideoAnalysisModel
from app.schemas import TextRequest, TextResponse, ImageResponse, VideoResponse
from app.services.text_service import classify_text
from app.services.image_service import classify_image
from app.services.video_service import classify_video

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)

# ── Load all models ONCE at startup ────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.text_model = TextToxicityModel()
    app.state.image_model = ImageNSFWModel()
    app.state.video_model = VideoAnalysisModel()
    yield
    del app.state.text_model
    del app.state.image_model
    del app.state.video_model

# ── App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SafeMedia Moderation API",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS — allows Flutter to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Health ─────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}

# ── Text endpoint ──────────────────────────────────────────────────────
@app.post("/predict/text", response_model=TextResponse)
def predict_text(payload: TextRequest, request: Request):
    try:
        return classify_text(payload.text, request.app.state.text_model)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Text inference failed.")

# ── Image endpoint ─────────────────────────────────────────────────────
@app.post("/predict/image", response_model=ImageResponse)
async def predict_image(request: Request, file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        return classify_image(
            image_bytes=image_bytes,
            filename=file.filename or "upload",
            content_type=file.content_type or "image/jpeg",
            model=request.app.state.image_model,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Image inference failed.")

# ── Video endpoint ─────────────────────────────────────────────────────
@app.post("/predict/video", response_model=VideoResponse)
async def predict_video(request: Request, file: UploadFile = File(...)):
    try:
        video_bytes = await file.read()
        return classify_video(
            video_bytes=video_bytes,
            filename=file.filename or "upload",
            content_type=file.content_type or "video/mp4",
            video_model=request.app.state.video_model,
            image_model=request.app.state.image_model,
            text_model=request.app.state.text_model,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Video inference failed.")