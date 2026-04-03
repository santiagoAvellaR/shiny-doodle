from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.geometry.homography import (
    compute_homography_from_overlay_to_plane,
    warp_overlay_to_frame,
    composite_overlay,
)
from src.geometry.order_corners import is_reasonable_quadrilateral


def default_seq1_config() -> dict:
    """
    Config de base pour la séquence 1.
    """
    return {
        "use_undistort": True,
        "camera_matrix": np.array(
            [
                [533.75781056, 0.0, 386.78762246],
                [0.0, 534.74578856, 275.71106165],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        "dist_coeffs": np.array(
            [-3.33535276e-01, 1.65338810e-01, -2.90030682e-04, -3.97059918e-04, -4.70631813e-02],
            dtype=np.float32,
        ),
        "expected_corner_order": ["yellow", "red", "green", "blue"],
        "min_blob_area": 80,
        "draw_debug": True,

        # --- PARÁMETROS DE FILTRADO Y ESTABILIDAD ---
        "filter_alpha": 0.6,   # Ganancia de posición
        "filter_beta": 0.1,    # Ganancia de velocidad
        "max_measurement_jump_px": 50.0, # Umbral de Gate para rechazar outliers
        "quad_consistency_area_tol": 0.3, # Tolerancia cambio area
        "quad_consistency_aspect_tol": 0.4, # Tolerancia cambio aspect ratio

        # --- PARÁMETROS DEL VERDE ROBUSTO ---
        "green_roi_size_coarse": 80,
        "green_roi_size_fine": 40,
        "green_v_max": 110,
        "green_min_circularity": 0.35,
        "green_hsv_ranges": [
            ((35, 20, 20), (105, 255, 120)),
        ],
        "prediction_weights": {
            "flow": 0.6,
            "geom": 0.4
        },
        "confidence_weights": {
            "dist": 1.0,
            "area": 0.3,
            "circularity": 0.5
        },

        "color_ranges_hsv": {
            "red": [
                ((0, 120, 60), (10, 255, 255)),
                ((170, 120, 60), (179, 255, 255)),
            ],
            "blue": [
                ((90, 80, 40), (135, 255, 255)),
            ],
            "yellow": [
                ((15, 80, 80), (38, 255, 255)),
            ],
        },
        "draw_colors_bgr": {
            "red": (0, 0, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
            "yellow": (0, 255, 255),
            "prediction": (255, 0, 255), # Magenta
            "measurement": (0, 255, 255), # Cyan (Detección cruda)
            "roi": (120, 120, 120),       # Gris
            "status_rej": (0, 0, 255),    # Rojo si rechazado
        },
    }


class MarkerTracker:
    def __init__(self, alpha: float, beta: float):
        self.alpha = alpha
        self.beta = beta
        self.pos: np.ndarray | None = None
        self.vel = np.array([0.0, 0.0], dtype=np.float32)
        self.last_valid_pos: np.ndarray | None = None
        self.missed_frames = 0

    def predict(self) -> np.ndarray | None:
        if self.pos is None:
            return None
        return self.pos + self.vel

    def update(self, measurement: np.ndarray | None, confidence: float = 1.0):
        if measurement is None:
            # Predicción pura con decaimiento de velocidad
            if self.pos is not None:
                self.pos = self.pos + self.vel
                self.vel *= 0.95
            self.missed_frames += 1
            return

        if self.pos is None:
            self.pos = measurement
            self.vel = np.array([0.0, 0.0], dtype=np.float32)
            self.missed_frames = 0
            return

        # Alpha-Beta Update
        pred = self.pos + self.vel
        res = measurement - pred

        # Adaptación de ganancia según confianza
        a = self.alpha * confidence
        b = self.beta * confidence

        self.pos = pred + a * res
        self.vel = self.vel + b * res
        self.missed_frames = 0
        self.last_valid_pos = self.pos.copy()


def undistort_frame(frame_bgr: np.ndarray, cfg: dict) -> np.ndarray:
    if not cfg["use_undistort"]:
        return frame_bgr

    K = cfg["camera_matrix"]
    dist = cfg["dist_coeffs"]
    return cv2.undistort(frame_bgr, K, dist)


def build_mask_from_hsv_ranges(hsv: np.ndarray, ranges: list[tuple[tuple[int, int, int], tuple[int, int, int]]]) -> np.ndarray:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for lower, upper in ranges:
        lower_np = np.array(lower, dtype=np.uint8)
        upper_np = np.array(upper, dtype=np.uint8)
        mask_part = cv2.inRange(hsv, lower_np, upper_np)
        mask = cv2.bitwise_or(mask, mask_part)

    # Nettoyage morphologique léger
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


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


def complete_with_previous(
    current_centers: dict[str, np.ndarray],
    previous_centers: dict[str, np.ndarray] | None,
    expected_colors: list[str],
) -> dict[str, np.ndarray]:
    completed: dict[str, np.ndarray] = {}
    for color in expected_colors:
        if color in current_centers:
            completed[color] = current_centers[color]
        elif previous_centers is not None and color in previous_centers:
            completed[color] = previous_centers[color]
    return completed


def smooth_centers(
    current_centers: dict[str, np.ndarray],
    previous_centers: dict[str, np.ndarray] | None,
    alpha: float,
) -> dict[str, np.ndarray]:
    if previous_centers is None:
        return current_centers
    smoothed: dict[str, np.ndarray] = {}
    for color, cur in current_centers.items():
        if color in previous_centers:
            prev = previous_centers[color]
            smoothed[color] = alpha * prev + (1.0 - alpha) * cur
        else:
            smoothed[color] = cur
    return smoothed


def centers_to_ordered_points(
    centers: dict[str, np.ndarray],
    expected_corner_order: list[str],
) -> np.ndarray:
    pts = np.array([centers[color] for color in expected_corner_order], dtype=np.float32)
    return pts


def draw_debug_info(
    image_bgr: np.ndarray,
    centers: dict[str, np.ndarray],
    ordered_pts: np.ndarray | None,
    cfg: dict,
    frame_idx: int,
    predictions: dict[str, np.ndarray] | None = None,
    measurements: dict[str, np.ndarray] | None = None,
    rej_status: dict[str, bool] | None = None,
    rois: dict[str, tuple[int, int, int, int]] | None = None,
) -> np.ndarray:
    debug = image_bgr.copy()

    if rois:
        for name, box in rois.items():
            cv2.rectangle(debug, (box[0], box[1]), (box[2], box[3]), cfg["draw_colors_bgr"]["roi"], 1)

    if predictions:
        for name, p in predictions.items():
            if p is not None:
                px, py = int(round(p[0])), int(round(p[1]))
                cv2.drawMarker(debug, (px, py), cfg["draw_colors_bgr"]["prediction"], cv2.MARKER_CROSS, 8, 1)

    if measurements:
        for name, m in measurements.items():
            if m is not None:
                mx, my = int(round(m[0])), int(round(m[1]))
                cv2.circle(debug, (mx, my), 3, cfg["draw_colors_bgr"]["measurement"], -1)

    for color_name, pt in centers.items():
        color_bgr = cfg["draw_colors_bgr"].get(color_name, (255, 255, 255))
        if rej_status and rej_status.get(color_name):
            color_bgr = cfg["draw_colors_bgr"]["status_rej"]
            
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(debug, (x, y), 8, color_bgr, 2)
        cv2.drawMarker(debug, (x, y), color_bgr, cv2.MARKER_TILTED_CROSS, 5, 2)
        cv2.putText(debug, color_name, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1, cv2.LINE_AA)

    if ordered_pts is not None:
        quad = ordered_pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(debug, [quad], isClosed=True, color=(255, 255, 255), thickness=1)

    cv2.putText(debug, f"frame: {frame_idx}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return debug


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


def load_overlay_image(path: Path) -> np.ndarray:
    overlay = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if overlay is None:
        raise FileNotFoundError(f"Impossible de charger l'image d'overlay: {path}")
    return overlay


def open_video_reader(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Impossible d'ouvrir la vidéo: {path}")
    return cap


def open_video_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Impossible de créer la vidéo de sortie")
    return writer


def run_seq1(
    input_video: Path,
    overlay_image: Path,
    output_video: Path,
    display: bool = False,
    max_frames: int | None = None,
) -> None:
    cfg = default_seq1_config()
    overlay_bgr = load_overlay_image(overlay_image)
    cap = open_video_reader(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = open_video_writer(output_video, fps, width, height)

    trackers = {name: MarkerTracker(cfg["filter_alpha"], cfg["filter_beta"]) for name in cfg["expected_corner_order"]}
    frame_idx = 0
    prev_quad_area = None
    prev_quad_aspect = None

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok: break
            if max_frames is not None and frame_idx >= max_frames: break
            frame_bgr = undistort_frame(frame_bgr, cfg)
            
            preds = {name: t.predict() for name, t in trackers.items()}
            raw_detections, _ = detect_markers(frame_bgr, cfg)
            measurements = {}
            rej_status = {name: False for name in trackers}
            
            for name in ["red", "blue", "yellow"]:
                if name in raw_detections:
                    m = raw_detections[name]
                    if preds[name] is not None:
                        dist = np.linalg.norm(m - preds[name])
                        if dist < cfg["max_measurement_jump_px"]:
                            measurements[name] = m
                        else:
                            rej_status[name] = True
                    else:
                        measurements[name] = m

            p0_green = preds["green"]
            if p0_green is None:
                if "yellow" in measurements and "red" in measurements and "blue" in measurements:
                    p0_green = measurements["red"] + measurements["blue"] - measurements["yellow"]
            
            if p0_green is not None:
                res_green = refine_green_two_stage(frame_bgr, p0_green, cfg)
                if res_green:
                    m_green, _ = res_green
                    measurements["green"] = m_green
            
            for name, t in trackers.items():
                t.update(measurements.get(name))

            current_centers = {name: t.pos for name, t in trackers.items() if t.pos is not None}
            ordered_pts = None
            if len(current_centers) == 4:
                ordered_pts = centers_to_ordered_points(current_centers, cfg["expected_corner_order"])
                from src.geometry.order_corners import polygon_area
                area = polygon_area(ordered_pts)
                side_w = np.linalg.norm(ordered_pts[0] - ordered_pts[1])
                side_h = np.linalg.norm(ordered_pts[0] - ordered_pts[3])
                aspect = side_w / side_h if side_h > 0 else 0
                
                is_consistent = True
                if prev_quad_area is not None:
                    if abs(area - prev_quad_area) / prev_quad_area > cfg["quad_consistency_area_tol"]: is_consistent = False
                if prev_quad_aspect is not None:
                    if abs(aspect - prev_quad_aspect) / prev_quad_aspect > cfg["quad_consistency_aspect_tol"]: is_consistent = False
                
                if is_consistent:
                    prev_quad_area = area
                    prev_quad_aspect = aspect

            result_bgr = frame_bgr.copy()
            if ordered_pts is not None and is_reasonable_quadrilateral(ordered_pts):
                H = compute_homography_from_overlay_to_plane(overlay_bgr, ordered_pts)
                warped_overlay, warped_mask = warp_overlay_to_frame(overlay_bgr, frame_bgr.shape, H)
                result_bgr = composite_overlay(frame_bgr, warped_overlay, warped_mask)

            if cfg["draw_debug"]:
                result_bgr = draw_debug_info(result_bgr, current_centers, ordered_pts, cfg, frame_idx, preds, measurements, rej_status)

            writer.write(result_bgr)
            if display:
                cv2.imshow("SY32", result_bgr)
                if cv2.waitKey(1) & 0xFF == 27: break
            frame_idx += 1

    finally:
        cap.release()
        writer.release()
        if display: cv2.destroyAllWindows()
    print(f"[OK] Vidéo enregistrée: {output_video}")