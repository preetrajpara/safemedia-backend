import logging
import io
from PIL import Image
from transformers import pipeline

logger = logging.getLogger(__name__)

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}

class ImageNSFWModel:
    def __init__(self):
        logger.info("Loading lightweight NSFW model...")
        self._pipe = pipeline(
            task="image-classification",
            model="Falconsai/nsfw_image_detection",
            device=-1  # force CPU (important for Render)
        )
        logger.info("Image model ready.")

    def predict(self, image_bytes: bytes) -> dict:
        image = self._decode(image_bytes)
        result = self._pipe(image)[0]

        return {
            "raw_label": result["label"].lower(),
            "score": round(float(result["score"]), 4),
        }

    @staticmethod
    def _decode(image_bytes: bytes) -> Image.Image:
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img.verify()
            img = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            raise ValueError(f"Invalid image: {e}")

        if img.format not in ALLOWED_FORMATS:
            raise ValueError(f"Unsupported format '{img.format}'.")

        return img.convert("RGB")