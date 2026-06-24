from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import cv2

from transformers import pipeline

app = FastAPI()

# =========================
# 🔥 CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# 🔥 REQUEST MODEL
# =========================
class TextRequest(BaseModel):
    text: str

# =========================
# 🔥 LOAD MODELS
# =========================
text_classifier = pipeline("text-classification", model="unitary/toxic-bert")
image_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection")

# =========================
# 🔥 DECISION
# =========================
def get_decision(safe):
    if safe >= 70:
        return "Safe to Upload ✅"
    elif safe >= 30:
        return "Risky to Upload ⚠️"
    else:
        return "Not Allowed to Upload 🚫"

# =========================
# 🔥 TEXT API
# =========================
@app.post("/predict/text")
async def predict_text(data: TextRequest):
    result = text_classifier(data.text)[0]

    score = result["score"] * 100
    label = result["label"].lower()

    toxic = score if "toxic" in label else 100 - score
    safe = 100 - toxic

    return {
        "type": "text",
        "toxic": round(toxic, 2),
        "safe": round(safe, 2),
        "decision": get_decision(safe)
    }

# =========================
# 🔥 IMAGE API
# =========================
@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    path = f"temp_{file.filename}"

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = image_classifier(path)

    toxic = 0
    for r in results:
        if "nsfw" in r["label"].lower():
            toxic = r["score"] * 100

    safe = 100 - toxic

    os.remove(path)

    return {
        "type": "image",
        "toxic": round(toxic, 2),
        "safe": round(safe, 2),
        "decision": get_decision(safe)
    }

# =========================
# 🔥 VIDEO API
# =========================
@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    path = f"temp_{file.filename}"

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    cap = cv2.VideoCapture(path)

    scores = []
    count = 0

    while cap.isOpened() and count < 3:
        ret, frame = cap.read()
        if not ret:
            break

        temp = f"frame_{count}.jpg"
        cv2.imwrite(temp, frame)

        results = image_classifier(temp)

        toxic = 0
        for r in results:
            if "nsfw" in r["label"].lower():
                toxic = r["score"] * 100

        scores.append(toxic)

        os.remove(temp)
        count += 1

    cap.release()
    os.remove(path)

    if not scores:
        return {"error": "Video failed"}

    avg_toxic = sum(scores) / len(scores)
    avg_safe = 100 - avg_toxic

    return {
        "type": "video",
        "toxic": round(avg_toxic, 2),
        "safe": round(avg_safe, 2),
        "decision": get_decision(avg_safe)
    }

# =========================
# 🔥 ROOT
# =========================
@app.get("/")
def home():
    return {"message": "SafeMedia API running 🚀"}