# app/services/video_service.py

import os
import uuid
import tempfile
import logging

from app.models.video_model import VideoAnalysisModel
from app.models.image_model import ImageNSFWModel
from app.models.text_model import TextToxicityModel

logger = logging.getLogger(__name__)

# Strict mode
IMAGE_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.3
MAX_BYTES = 100 * 1024 * 1024  # 100 MB

ALLOWED_CONTENT_TYPES = {
    "video/mp4",
    "video/mpeg",
    "video/quicktime",
    "video/x-msvideo",  # .avi
    "video/webm",
}


def classify_video(
    video_bytes: bytes,
    filename: str,
    content_type: str,
    video_model: VideoAnalysisModel,
    image_model: ImageNSFWModel,
    text_model: TextToxicityModel,
) -> dict:

    # Guard: file size
    if len(video_bytes) > MAX_BYTES:
        raise ValueError("Video exceeds 100 MB limit.")

    # Guard: MIME type
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise ValueError(f"Unsupported video type '{content_type}'.")

    # Save video bytes to a temp file so OpenCV can read it
    tmp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.mp4")

    try:
        with open(tmp_path, "wb") as f:
            f.write(video_bytes)

        # ── Step 1: Extract and check frames ──────────────────────────
        frames = video_model.extract_frames(tmp_path)
        total_frames = len(frames)
        unsafe_frames = 0
        highest_image_score = 0.0

        for frame in frames:
            import io
            buf = io.BytesIO()
            frame.save(buf, format="JPEG")
            frame_bytes = buf.getvalue()

            result = image_model.predict(frame_bytes)
            if result["raw_label"] == "nsfw":
                score = result["score"]
                highest_image_score = max(highest_image_score, score)
                if score >= IMAGE_THRESHOLD:
                    unsafe_frames += 1

        # ── Step 2: Extract and check audio transcript ─────────────────
        transcript = video_model.extract_audio_transcript(tmp_path)
        transcript_toxic = False
        transcript_confidence = 0.0

        if transcript:
            text_result = text_model.predict(transcript)
            if (
                text_result["raw_label"] == "toxic"
                and text_result["score"] >= TEXT_THRESHOLD
            ):
                transcript_toxic = True
                transcript_confidence = text_result["score"]

        # ── Step 3: Final decision ─────────────────────────────────────
        # Flag as unsafe if ANY frame is unsafe OR transcript is toxic
        unsafe = unsafe_frames > 0 or transcript_toxic
        confidence = round(max(highest_image_score, transcript_confidence), 4)

        return {
            "filename": filename,
            "unsafe": unsafe,
            "confidence": confidence,
            "unsafe_frames": unsafe_frames,
            "total_frames": total_frames,
            "transcript": transcript,
            "transcript_toxic": transcript_toxic,
        }

    finally:
        # Always clean up temp video file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)