# app/service.py

from app.model import TextToxicityModel

CONFIDENCE_THRESHOLD = 0.5

def classify_text(text: str, model: TextToxicityModel) -> dict:
    result = model.predict(text)

    raw_label = result["raw_label"]
    score = result["score"]

    if raw_label == "toxic" and score >= CONFIDENCE_THRESHOLD:
        label = "toxic"
        confidence = score
    else:
        label = "non-toxic"
        confidence = score if raw_label != "toxic" else round(1.0 - score, 4)

    return {
        "input": text,
        "label": label,
        "confidence": confidence,
    }