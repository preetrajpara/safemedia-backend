import logging
import io
from PIL import Image
from transformers import pipeline

logger = logging.getLogger(__name__)

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}

class ImageNSFWModel:
    def __init__(self):
        logger.info("Loading image NSFW model...")
        self._pipe = pipeline(
            task="image-classification",
            model="Falconsai/nsfw_image_detection",
        )
        logger.info("Image model ready.")

    def predict(self, image_bytes: bytes) -> dict:
        image = self._decode(image_bytes)

        results = self._pipe(image)

        # 🔥 pick highest score safely
        best = max(results, key=lambda x: x["score"])

        label = best["label"].lower()
        score = float(best["score"])

        # 🔥 normalize output
        category = "nsfw" if label == "nsfw" else "safe"

        # 🔥 confidence logic
        if score > 0.85:
            confidence = "high"
        elif score > 0.65:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "label": category,
            "raw_label": label,
            "score": round(score, 4),
            "confidence": confidence,
        }

    @staticmethod
    def _decode(image_bytes: bytes) -> Image.Image:
        try:
            img = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            raise ValueError(f"Invalid image: {e}")

        if img.format not in ALLOWED_FORMATS:
            raise ValueError(
                f"Unsupported format '{img.format}'. Use JPEG, PNG, WEBP or BMP."
            )

        return img.convert("RGB")