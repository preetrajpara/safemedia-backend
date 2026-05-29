# app/schemas.py

from pydantic import BaseModel, Field


# ── Text ──────────────────────────────────────────────────────────────
class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)

class TextResponse(BaseModel):
    input: str
    label: str
    confidence: float


# ── Image ─────────────────────────────────────────────────────────────
class ImageResponse(BaseModel):
    filename: str
    unsafe: bool
    confidence: float


# ── Video ─────────────────────────────────────────────────────────────
class VideoResponse(BaseModel):
    filename: str
    unsafe: bool
    confidence: float
    unsafe_frames: int
    total_frames: int
    transcript: str
    transcript_toxic: bool

