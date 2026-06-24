# =========================
# 🔥 IMAGE MODEL
# =========================
image_classifier = pipeline(
    "image-classification",
    model="Falconsai/nsfw_image_detection"
)

def analyze_image(path):
    results = image_classifier(path)

    toxic = 0

    for r in results:
        label = r["label"].lower()

        if "nsfw" in label:
            toxic = r["score"] * 100

    safe = 100 - toxic

    return round(toxic, 2), round(safe, 2)