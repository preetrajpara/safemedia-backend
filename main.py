import logging
import sys

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.models.text_model import TextToxicityModel
from app.models.image_model import ImageNSFWModel
from app.models.video_model import VideoAnalysisModel

from app.schemas import TextRequest, TextResponse, ImageResponse, VideoResponse

from app.services.text_service import classify_text
from app.services.image_service import classify_image
from app.services.video_service import classify_video

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)

# ─────────────────────────────────────────────
# App Init
# ─────────────────────────────────────────────
app = FastAPI(
    title="SafeMedia Moderation API",
    version="2.0.0",
)

# ─────────────────────────────────────────────
# CORS (for Flutter / frontend)
# ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Lazy-loaded models (IMPORTANT for Render)
# ─────────────────────────────────────────────
text_model = None
image_model = None
video_model = None

# ─────────────────────────────────────────────
# Root (fix 404)
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "SafeMedia API is running 🚀"}

# ─────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}

# ─────────────────────────────────────────────
# TEXT
# ─────────────────────────────────────────────
@app.post("/predict/text", response_model=TextResponse)
def predict_text(payload: TextRequest):
    global text_model

    try:
        if text_model is None:
            print("Loading text model...")
            text_model = TextToxicityModel()

        return classify_text(payload.text, text_model)

    except Exception as e:
        print("TEXT ERROR:", str(e))
        raise HTTPException(status_code=500, detail="Text processing failed")

# ─────────────────────────────────────────────
# IMAGE
# ─────────────────────────────────────────────
@app.post("/predict/image", response_model=ImageResponse)
async def predict_image(file: UploadFile = File(...)):
    global image_model

    try:
        if image_model is None:
            print("Loading image model...")
            image_model = ImageNSFWModel()

        image_bytes = await file.read()

        return classify_image(
            image_bytes=image_bytes,
            filename=file.filename or "upload",
            content_type=file.content_type or "image/jpeg",
            model=image_model,
        )

    except Exception as e:
        print("IMAGE ERROR:", str(e))
        raise HTTPException(status_code=500, detail="Image processing failed")

# ─────────────────────────────────────────────
# VIDEO (LIGHT MODE)
# ─────────────────────────────────────────────
@app.post("/predict/video", response_model=VideoResponse)
async def predict_video(file: UploadFile = File(...)):
    global video_model, image_model, text_model

    try:
        if video_model is None:
            print("Loading video model...")
            video_model = VideoAnalysisModel()

        if image_model is None:
            image_model = ImageNSFWModel()

        if text_model is None:
            text_model = TextToxicityModel()

        video_bytes = await file.read()

        return classify_video(
            video_bytes=video_bytes,
            filename=file.filename or "upload",
            content_type=file.content_type or "video/mp4",
            video_model=video_model,
            image_model=image_model,
            text_model=text_model,
        )

    except Exception as e:
        print("VIDEO ERROR:", str(e))
        raise HTTPException(status_code=500, detail="Video processing failed")