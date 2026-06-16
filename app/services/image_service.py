def classify_image(
    image_bytes: bytes,
    filename: str,
    content_type: str,
    model,
) -> dict:

    # 🔥 HARD LIMIT for free tier
    MAX_SIZE = 2 * 1024 * 1024  # 2MB

    if len(image_bytes) > MAX_SIZE:
        raise ValueError("Image too large (max 2MB)")

    result = model.predict(image_bytes)

    unsafe = result["raw_label"] == "nsfw"
    confidence = result["score"]

    return {
        "filename": filename,
        "unsafe": unsafe,
        "confidence": confidence,
    }