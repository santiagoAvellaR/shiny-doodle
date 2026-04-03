import cv2
import numpy as np
from pathlib import Path

def load_overlay_image(path: Path) -> np.ndarray:
    overlay = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if overlay is None:
        raise FileNotFoundError(f"Impossible de charger l'image d'overlay: {path}")
    return overlay
