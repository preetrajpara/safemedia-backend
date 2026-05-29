# app/models/video_model.py

import logging
import os
import uuid
import tempfile
import cv2
import whisper
import ffmpeg
from PIL import Image

logger = logging.getLogger(__name__)

# Extract 1 frame every N seconds
FRAME_INTERVAL_SECONDS = 2

class VideoAnalysisModel:
    def __init__(self):
        logger.info("Loading Whisper audio model...")
        self._whisper = whisper.load_model("base")
        logger.info("Video model ready.")

    def extract_frames(self, video_path: str) -> list[Image.Image]:
        """Extract frames every FRAME_INTERVAL_SECONDS seconds."""
        frames = []
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError("Could not open video file.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        interval = max(1, int(fps * FRAME_INTERVAL_SECONDS))
        frame_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index % interval == 0:
                # Convert BGR (OpenCV) → RGB (PIL)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
            frame_index += 1

        cap.release()
        logger.info(f"Extracted {len(frames)} frames from video.")
        return frames

    def extract_audio_transcript(self, video_path: str) -> str:
        """Extract audio from video and transcribe using Whisper."""
        audio_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.wav")

        try:
            # Extract audio track using FFmpeg
            (
                ffmpeg
                .input(video_path)
                .output(audio_path, ac=1, ar="16000", format="wav")
                .overwrite_output()
                .run(quiet=True)
            )

            # Transcribe with Whisper
            result = self._whisper.transcribe(audio_path)
            transcript = result.get("text", "").strip()
            logger.info(f"Transcript: {transcript[:100]}...")
            return transcript

        except Exception as e:
            logger.warning(f"Audio transcription failed: {e}")
            return ""

        finally:
            # Always clean up temp audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)