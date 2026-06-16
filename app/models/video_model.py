import logging
import os
import uuid
import tempfile
import cv2
import ffmpeg
from PIL import Image

logger = logging.getLogger(__name__)

# 🔥 reduce frame load
MAX_FRAMES = 5

class VideoAnalysisModel:
    def __init__(self):
        logger.info("Video model ready (light mode).")

    def extract_frames(self, video_path: str) -> list[Image.Image]:
        frames = []
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError("Could not open video file.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total_frames // MAX_FRAMES)

        frame_index = 0

        while len(frames) < MAX_FRAMES:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))

            frame_index += step

        cap.release()
        logger.info(f"Extracted {len(frames)} frames.")
        return frames

    def extract_audio_transcript(self, video_path: str) -> str:
        """⚠️ OPTIONAL — disable for now"""
        return ""