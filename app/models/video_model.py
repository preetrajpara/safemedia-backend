# =========================
# 🔥 VIDEO MODEL
# =========================
import cv2
import os

def analyze_video(path):
    cap = cv2.VideoCapture(path)

    scores = []
    count = 0

    while cap.isOpened() and count < 3:
        ret, frame = cap.read()
        if not ret:
            break

        temp = f"frame_{count}.jpg"
        cv2.imwrite(temp, frame)

        toxic, _ = analyze_image(temp)
        scores.append(toxic)

        os.remove(temp)
        count += 1
 
    cap.release()

    if not scores:
        return 0, 100

    avg_toxic = sum(scores) / len(scores)
    avg_safe = 100 - avg_toxic

    return round(avg_toxic, 2), round(avg_safe, 2)