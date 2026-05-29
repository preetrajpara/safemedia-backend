# app/model.py

import logging
from transformers import pipeline

logger = logging.getLogger(__name__)

class TextToxicityModel:
    def __init__(self):
        logger.info("Loading toxic-bert model...")
        self._pipe = pipeline(
            task="text-classification",
            model="unitary/toxic-bert",
            truncation=True,
            max_length=512,
        )
        logger.info("Model ready.")

    def predict(self, text: str) -> dict:
        if not text.strip():
            raise ValueError("Text must not be empty.")

        result = self._pipe(text)[0]
        return {
            "raw_label": result["label"].lower(),
            "score": round(float(result["score"]), 4),
        }