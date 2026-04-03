import cv2
from pathlib import Path

def open_video_reader(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Impossible d'ouvrir la vidéo: {path}")
    return cap
