# 🛡️ SafeMedia Backend (AI Content Moderation API)

SafeMedia Backend is a FastAPI-based AI system that analyzes **text, images, and videos** to detect harmful or unsafe content.

This backend powers the SafeMedia mobile app and provides real-time moderation results using machine learning models.

---

## 🚀 Features

* 🔤 **Text Moderation**

  * Detects toxic or harmful text
  * Returns toxicity and safety percentage

* 🖼️ **Image Moderation**

  * Analyzes uploaded images
  * Detects unsafe or offensive content

* 🎥 **Video Moderation**

  * Extracts frames from video
  * Performs AI-based content analysis

* 📊 **Smart Decision System**

  * ✅ Safe to Upload (70–100%)
  * ⚠️ Risky Content (30–70%)
  * 🚫 Not Allowed (0–30%)

---

## 🧠 Tech Stack

* **Backend Framework:** FastAPI
* **ML Models:** Transformers / Custom Logic
* **Libraries:**

  * `torch`
  * `transformers`
  * `opencv-python`
  * `pillow`
  * `numpy`
* **Deployment:** Hugging Face Spaces

---

## 📦 API Endpoints

### 🔤 Text Moderation

POST `/predict/text`

#### Request:

```json
{
  "text": "I hate you"
}
```

#### Response:

```json
{
  "toxic": 82.3,
  "safe": 17.7,
  "decision": "not_allowed"
}
```

---

### 🖼️ Image Moderation

POST `/predict/image`

* Upload image file

#### Response:

```json
{
  "toxic": 45.2,
  "safe": 54.8,
  "decision": "risky_content"
}
```

---

### 🎥 Video Moderation

POST `/predict/video`

* Upload video file

#### Response:

```json
{
  "toxic": 20.5,
  "safe": 79.5,
  "decision": "safe_to_upload"
}
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/safemedia-backend.git
cd safemedia-backend

pip install -r requirements.txt
```

---

## ▶️ Run Locally

```bash
uvicorn main:app --reload
```

Server will run at:

```
http://127.0.0.1:8000
```

---

## 📱 Integration

This backend is integrated with a Flutter mobile app that allows users to:

* Upload text, image, or video
* Get real-time moderation results
* View safety scores visually

---

## 🎯 Project Goal

To build a real-world AI moderation system that can:

* Detect harmful content
* Assist in safer online interactions
* Demonstrate full-stack AI application development

---

## 👨‍💻 Author

**Preet Rajpara**
