import cv2
import numpy as np
from src.detection.color_segmentation import build_mask_from_hsv_ranges

def calculate_refined_center(roi_mask: np.ndarray, min_area: int) -> tuple[np.ndarray, float, float] | None:
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < min_area / 2:
        return None
    perimeter = cv2.arcLength(c, True)
    circularity = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter > 0 else 0
    (x, y), radius = cv2.minEnclosingCircle(c)
    M = cv2.moments(c)
    if M["m00"] > 0:
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        final_x, final_y = (x + cx) / 2.0, (y + cy) / 2.0
        return np.array([final_x, final_y], dtype=np.float32), area, circularity
    return np.array([x, y], dtype=np.float32), area, circularity


def score_measurement(m: np.ndarray, pred: np.ndarray, area: float, circularity: float, cfg: dict) -> float:
    dist = np.linalg.norm(m - pred)
    conf_dist = np.exp(-dist / 30.0)
    conf_circ = np.clip(circularity, 0, 1)
    target_area = cfg["min_blob_area"] * 2.0
    conf_area = np.exp(-abs(area - target_area) / target_area)
    w = cfg["confidence_weights"]
    total = (w["dist"] * conf_dist + w["circularity"] * conf_circ + w["area"] * conf_area) / sum(w.values())
    return float(total)


def refine_green_two_stage(frame_bgr: np.ndarray, p0: np.ndarray, cfg: dict) -> tuple[np.ndarray, float] | None:
    h_img, w_img = frame_bgr.shape[:2]
    S_c = cfg["green_roi_size_coarse"] // 2
    x0, y0 = int(round(p0[0])), int(round(p0[1]))
    x_min, x_max = max(0, x0 - S_c), min(w_img, x0 + S_c)
    y_min, y_max = max(0, y0 - S_c), min(h_img, y0 + S_c)
    if x_max <= x_min or y_max <= y_min: return None
    roi_c = frame_bgr[y_min:y_max, x_min:x_max]
    hsv_c = cv2.cvtColor(roi_c, cv2.COLOR_BGR2HSV)
    mask_c_color = build_mask_from_hsv_ranges(hsv_c, cfg["green_hsv_ranges"])
    _, mask_c_v = cv2.threshold(hsv_c[:, :, 2], cfg["green_v_max"], 255, cv2.THRESH_BINARY_INV)
    mask_c = cv2.bitwise_and(mask_c_color, mask_c_v)
    res_c = calculate_refined_center(mask_c, cfg["min_blob_area"])
    if not res_c: return None
    raw_pt, area, circ = res_c
    abs_pt = raw_pt + np.array([x_min, y_min])
    
    S_f = cfg["green_roi_size_fine"] // 2
    xf, yf = int(round(abs_pt[0])), int(round(abs_pt[1]))
    xf_min, xf_max = max(0, xf - S_f), min(w_img, xf + S_f)
    yf_min, yf_max = max(0, yf - S_f), min(h_img, yf + S_f)
    roi_f = frame_bgr[yf_min:yf_max, xf_min:xf_max]
    hsv_f = cv2.cvtColor(roi_f, cv2.COLOR_BGR2HSV)
    mask_f_color = build_mask_from_hsv_ranges(hsv_f, cfg["green_hsv_ranges"])
    _, mask_f_v = cv2.threshold(hsv_f[:, :, 2], cfg["green_v_max"] - 10, 255, cv2.THRESH_BINARY_INV)
    mask_f = cv2.bitwise_and(mask_f_color, mask_f_v)
    res_f = calculate_refined_center(mask_f, cfg["min_blob_area"])
    if res_f:
        final_pt, final_area, final_circ = res_f
        conf = score_measurement(final_pt + np.array([xf_min, yf_min]), p0, final_area, final_circ, cfg)
        return final_pt + np.array([xf_min, yf_min]), conf
    conf = score_measurement(abs_pt, p0, area, circ, cfg) * 0.7
    return abs_pt, conf
