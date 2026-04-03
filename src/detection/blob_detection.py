import cv2
import numpy as np
from src.detection.color_segmentation import build_mask_from_hsv_ranges

def calculate_centroid_from_mask(mask: np.ndarray, min_area: int) -> list[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_centers = []
    for c in contours:
        area = cv2.contourArea(c)
        if min_area <= area <= 5000:
            perimeter = cv2.arcLength(c, True)
            if perimeter > 0:
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if circularity > 0.5:
                    M = cv2.moments(c)
                    if M["m00"] > 0:
                        cx = M["m10"] / M["m00"]
                        cy = M["m01"] / M["m00"]
                        (bx, by), radius = cv2.minEnclosingCircle(c)
                        final_x = (cx + bx) / 2.0
                        final_y = (cy + by) / 2.0
                        valid_centers.append(np.array([final_x, final_y], dtype=np.float32))

    return valid_centers


def detect_markers(frame_bgr: np.ndarray, cfg: dict) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    centers: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    candidates_dict: dict[str, list[np.ndarray]] = {}

    for color_name, ranges in cfg["color_ranges_hsv"].items():
        mask = build_mask_from_hsv_ranges(hsv, ranges)
        masks[color_name] = mask
        candidates = calculate_centroid_from_mask(mask, cfg["min_blob_area"])
        if candidates:
            candidates_dict[color_name] = candidates
            
    for color in ["yellow", "red", "blue"]:
        if color in candidates_dict and candidates_dict[color]:
            centers[color] = candidates_dict[color][0]

    return centers, masks
