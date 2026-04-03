import cv2
import numpy as np

def undistort_frame(frame_bgr: np.ndarray, cfg: dict) -> np.ndarray:
    if not cfg["use_undistort"]:
        return frame_bgr

    K = cfg["camera_matrix"]
    dist = cfg["dist_coeffs"]
    return cv2.undistort(frame_bgr, K, dist)
