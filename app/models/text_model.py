import logging
from transformers import pipeline

logger = logging.getLogger(__name__)

class TextToxicityModel:
    def __init__(self):
        logger.info("Loading text toxicity model...")
        self._pipe = pipeline(
            task="text-classification",
            model="unitary/toxic-bert",
            truncation=True,
            max_length=512,
        )
        logger.info("Text model ready.")

    def predict(self, text: str) -> dict:
        if not text.strip():
            raise ValueError("Text must not be empty.")

        result = self._pipe(text)[0]

        label = result["label"].lower()
        score = float(result["score"])

        # 🔥 Normalize categories
        category_map = {
            "toxic": "toxic",
            "insult": "toxic",
            "obscene": "toxic",
            "identity_hate": "toxic",
            "threat": "toxic",
            "severe_toxic": "toxic",
        }

        category = category_map.get(label, "neutral")

        # 🔥 Confidence thresholds
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