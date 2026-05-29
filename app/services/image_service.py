# app/services/image_service.py

from app.models.image_model import ImageNSFWModel

# Strict mode — flag anything above 30% confidence
CONFIDENCE_THRESHOLD = 0.3
MAX_BYTES = 10 * 1024 * 1024  # 10 MB

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
}

def classify_image(
    image_bytes: bytes,
    filename: str,
    content_type: str,
    model: ImageNSFWModel,
) -> dict:
    # Guard: file size
    if len(image_bytes) > MAX_BYTES:
        raise ValueError("Image exceeds 10 MB limit.")

    # Guard: MIME type
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise ValueError(f"Unsupported file type '{content_type}'.")

    result = model.predict(image_bytes)

    raw_label = result["raw_label"]
    score = result["score"]

    is_unsafe = raw_label == "nsfw" and score >= CONFIDENCE_THRESHOLD
    confidence = score if raw_label == "nsfw" else round(1.0 - score, 4)

    return {
        "filename": filename,
        "unsafe": is_unsafe,
        "confidence": confidence,
    }