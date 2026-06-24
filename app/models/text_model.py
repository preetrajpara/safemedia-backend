# =========================
# 🔥 TEXT MODEL
# =========================
from transformers import pipeline

text_classifier = pipeline(
    "text-classification",
    model="unitary/toxic-bert"
)

def analyze_text(text):
    result = text_classifier(text)[0]

    score = result["score"] * 100
    label = result["label"].lower()

    if "toxic" in label:
        toxic = score
    else:
        toxic = 100 - score

    safe = 100 - toxic

    return round(toxic, 2), round(safe, 2)