import numpy as np
import cv2
from src.detection.blob_detection import detect_markers
from src.detection.marker_refinement import estimate_green_from_yrb, detect_green_local, build_mask_from_hsv_ranges
from src.pipelines.pipeline_state import PipelineState
from src.geometry.missing_corner import complete_quad_with_temporal_affine

def acquire_strong_measurements(
    frame_bgr: np.ndarray,
    preds: dict[str, np.ndarray | None],
    state: PipelineState,
    cfg: dict
) -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, bool]]:
    """Handles raw optical detection of HSV markers and constructs strong inputs."""
    raw_detections, _ = detect_markers(frame_bgr, cfg)
    
    measurements = {}
    measurement_sources = {}
    rej_status = {name: False for name in state.trackers}
    
    # 1. RBY Checks
    for name in ["red", "blue", "yellow"]:
        if name in raw_detections:
            m = raw_detections[name]
            if preds[name] is not None:
                dist = np.linalg.norm(m - preds[name])
                
                # Check Reacquisition Gates
                if state.marker_miss_counts[name] >= cfg.get("hard_reacquire_after_miss", 10):
                    measurements[name] = m
                    measurement_sources[name] = "hsv"
                else:
                    if state.marker_miss_counts[name] >= cfg.get("reacquire_after_miss", 3):
                        gate = cfg.get("reacquire_gate_px", 180.0)
                    else:
                        gate = cfg.get("max_measurement_jump_px", 70.0)

                    if dist < gate:
                        measurements[name] = m
                        measurement_sources[name] = "hsv"
                    else:
                        rej_status[name] = True
            else:
                measurements[name] = m
                measurement_sources[name] = "hsv"

    # 2. Green local recovery
    pred_green = preds.get("green")
    green_meas = None
    is_bootstrap = False
    
    if "green" in state.trackers:
        is_bootstrap = (state.trackers["green"].pos is None) or (state.marker_miss_counts.get("green", 0) >= cfg.get("green_miss_limit", 15))
    
    if is_bootstrap:
        green_seed = estimate_green_from_yrb(
            measurements.get("yellow"),
            measurements.get("red"),
            measurements.get("blue")
        )
        if green_seed is not None:
            green_meas = detect_green_local(
                frame_bgr, green_seed, cfg.get("green_roi_bootstrap", 150),
                cfg["green_hsv_bootstrap"], cfg.get("green_min_area", 200), cfg.get("green_v_max_bootstrap", 255)
            )
            if green_meas is not None and np.linalg.norm(green_meas - green_seed) > cfg.get("green_gate_bootstrap", 100):
                green_meas = None
    elif pred_green is not None:
        green_meas = detect_green_local(
            frame_bgr, pred_green, cfg.get("green_roi_normal", 60),
            cfg["green_hsv_normal"], cfg.get("green_min_area", 200), cfg.get("green_v_max_normal", 180)
        )
        if green_meas is not None and np.linalg.norm(green_meas - pred_green) > cfg.get("green_gate_normal", 45.0):
            green_meas = None

    if green_meas is not None:
        measurements["green"] = green_meas
        measurement_sources["green"] = "green_local"
            
    return measurements, measurement_sources, rej_status


def get_geometric_marker_seed(
    name: str,
    predicted_ordered_pts: np.ndarray | None,
    state: PipelineState,
    visible_centers: dict[str, np.ndarray],
    cfg: dict,
) -> np.ndarray | None:
    expected_order = cfg["expected_corner_order"]
    if name not in expected_order:
        return None
    idx = expected_order.index(name)

    if predicted_ordered_pts is not None:
        pt = predicted_ordered_pts[idx]
        if not np.any(np.isnan(pt)) and not np.any(np.isinf(pt)):
            return pt.copy()

    if state.last_render_ordered_pts is not None:
        pt = state.last_render_ordered_pts[idx]
        if not np.any(np.isnan(pt)) and not np.any(np.isinf(pt)):
            return pt.copy()

    if state.last_full_real_ordered_pts is not None:
        pt = state.last_full_real_ordered_pts[idx]
        if not np.any(np.isnan(pt)) and not np.any(np.isinf(pt)):
            return pt.copy()

    if state.previous_complete_centers is not None and name in state.previous_complete_centers:
        pt = state.previous_complete_centers[name]
        if not np.any(np.isnan(pt)) and not np.any(np.isinf(pt)):
            return pt.copy()

    if len(visible_centers) == 3:
        comp_centers, status_temp, missing_name_temp = complete_quad_with_temporal_affine(
            visible_centers, state.previous_complete_centers, expected_order, cfg
        )
        if comp_centers is not None and missing_name_temp == name:
            pt = comp_centers[name]
            if not np.any(np.isnan(pt)) and not np.any(np.isinf(pt)):
                return pt.copy()

    return None

def reacquire_marker_local_by_geometry(
    frame_bgr: np.ndarray,
    name: str,
    seed: np.ndarray,
    cfg: dict,
) -> np.ndarray | None:
    hsv_ranges_dict = cfg.get("local_reacquire_hsv", {})
    if name not in hsv_ranges_dict:
        return None
    
    ranges = hsv_ranges_dict[name]
    radius = cfg.get("local_reacquire_radius", 90)
    
    h_img, w_img = frame_bgr.shape[:2]
    x0, y0 = int(round(seed[0])), int(round(seed[1]))
    
    x_min, x_max = max(0, x0 - radius), min(w_img, x0 + radius)
    y_min, y_max = max(0, y0 - radius), min(h_img, y0 + radius)
    if x_max <= x_min or y_max <= y_min:
        return None
        
    roi = frame_bgr[y_min:y_max, x_min:x_max]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = build_mask_from_hsv_ranges(hsv_roi, ranges)
    
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_area = cfg.get("local_reacquire_min_area", 25)
    min_circ = cfg.get("local_reacquire_min_circularity", 0.15)
    max_dist = cfg.get("local_reacquire_gate_px", 260.0)
    
    best_pt = None
    best_dist = float('inf')
    
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area: continue
            
        perimeter = cv2.arcLength(c, True)
        circ = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter > 0 else 0
        if circ < min_circ: continue
            
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            (cx, cy), _ = cv2.minEnclosingCircle(c)
            
        abs_pt = np.array([cx + x_min, cy + y_min], dtype=np.float32)
        dist = np.linalg.norm(abs_pt - seed)
        if dist < max_dist and dist < best_dist:
            best_dist = dist
            best_pt = abs_pt
            
    return best_pt

def run_local_reacquire(
    frame_bgr: np.ndarray,
    predicted_ordered_pts: np.ndarray | None,
    measurements: dict[str, np.ndarray],
    measurement_sources: dict[str, str],
    state: PipelineState,
    cfg: dict
) -> list[str]:
    local_reacquired_names = []
    if not cfg.get("local_reacquire_enabled", False):
        return local_reacquired_names
        
    source_name = cfg.get("local_reacquire_source_name", "hsv_local_reacquire")
    
    # 7. Foreground mask for hand occlusion check
    fg_mask = None
    if cfg.get("disable_local_reacquire_inside_foreground", True) and len(state.mask_buffer) > 0:
        fg_mask = state.mask_buffer[0] # Using last frame mask
        
    for name in cfg.get("expected_corner_order", ["yellow", "red", "green", "blue"]):
        if name not in measurements:
            if state.marker_miss_counts.get(name, 0) >= cfg.get("local_reacquire_after_miss", 3):
                seed = get_geometric_marker_seed(
                    name=name,
                    predicted_ordered_pts=predicted_ordered_pts,
                    state=state,
                    visible_centers=measurements,
                    cfg=cfg,
                )
                if seed is not None:
                    # ROI Foreground Check
                    if fg_mask is not None:
                        radius = cfg.get("local_reacquire_radius", 55)
                        h_img, w_img = fg_mask.shape[:2]
                        x0, y0 = int(round(seed[0])), int(round(seed[1]))
                        x_min, x_max = max(0, x0 - radius), min(w_img, x0 + radius)
                        y_min, y_max = max(0, y0 - radius), min(h_img, y0 + radius)
                        
                        if x_max > x_min and y_max > y_min:
                            roi_mask = fg_mask[y_min:y_max, x_min:x_max]
                            fg_ratio = np.count_nonzero(roi_mask) / roi_mask.size
                            if fg_ratio > cfg.get("local_reacquire_foreground_max_ratio", 0.4):
                                continue # Too much hand/foreground, likely occlusion still active

                    center = reacquire_marker_local_by_geometry(
                        frame_bgr=frame_bgr,
                        name=name,
                        seed=seed,
                        cfg=cfg,
                    )
                    if center is not None:
                        # 2. LOCAL_REACQUIRE MUST BE CANDIDATE INITIALLY
                        measurements[name] = center.astype(np.float32)
                        measurement_sources[name] = source_name + "_candidate"
                        local_reacquired_names.append(name)
                        
    return local_reacquired_names
