import cv2
import numpy as np

# A local dependency for pipeline state typing
from src.pipelines.pipeline_state import PipelineState


def extract_marker_template(frame_bgr: np.ndarray, center: np.ndarray, radius: int = 18) -> np.ndarray | None:
    """
    Extracts a square patch around a marker center.
    Returns patch or None if outside image.
    """
    x, y = int(center[0]), int(center[1])
    h, w = frame_bgr.shape[:2]

    # Ensure bounds
    if x - radius < 0 or x + radius >= w or y - radius < 0 or y + radius >= h:
        return None

    return frame_bgr[y - radius : y + radius + 1, x - radius : x + radius + 1].copy()


def update_marker_templates(
    frame_bgr: np.ndarray, 
    measurements: dict[str, np.ndarray], 
    templates: dict[str, np.ndarray | None], 
    cfg: dict
) -> dict[str, np.ndarray | None]:
    """
    Updates templates when markers are confidently visible.
    Should only update from real measurements.
    Uses cfg["template_update_alpha"] for slow adaptation.
    """
    new_templates = templates.copy()
    radius = cfg.get("template_patch_radius", 18)
    alpha = cfg.get("template_update_alpha", 0.05)
    
    for name, meas in measurements.items():
        patch = extract_marker_template(frame_bgr, meas, radius)
        if patch is not None:
            if new_templates.get(name) is None:
                new_templates[name] = patch.astype(np.float32)
            else:
                # Alpha blend templates
                current_t = new_templates[name]
                if patch.shape == current_t.shape:
                    updated_t = (1.0 - alpha) * current_t + alpha * patch.astype(np.float32)
                    new_templates[name] = updated_t
                    
    return new_templates


def detect_marker_by_template(
    frame_bgr: np.ndarray, 
    name: str, 
    prediction: np.ndarray, 
    template: np.ndarray, 
    cfg: dict
) -> tuple[np.ndarray | None, float]:
    """
    Search marker template in a local ROI around prediction.
    Returns (center, score) or (None, score).
    Uses cv2.matchTemplate with TM_CCOEFF_NORMED.
    """
    if template is None:
        return None, 0.0

    search_radius = cfg.get("template_search_radius", 65)
    h, w = frame_bgr.shape[:2]
    
    px, py = int(prediction[0]), int(prediction[1])
    
    x0 = max(0, px - search_radius)
    y0 = max(0, py - search_radius)
    x1 = min(w, px + search_radius + 1)
    y1 = min(h, py + search_radius + 1)
    
    if x1 - x0 < template.shape[1] or y1 - y0 < template.shape[0]:
        return None, 0.0
        
    roi = frame_bgr[y0:y1, x0:x1]
    
    # Template has shape (2*r+1, 2*r+1, 3). Convert both to float32 for correlation.
    # Optionally use GRAY
    if cfg.get("template_use_gray", False):
        roi_process = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        template_process = cv2.cvtColor(template.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    else:
        roi_process = roi
        template_process = template.astype(np.uint8)

    res = cv2.matchTemplate(roi_process, template_process, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    if max_val < cfg.get("template_match_min_score", 0.55):
        return None, max_val
        
    # max_loc is (x, y) of the top-left corner of the matched string
    th, tw = template.shape[:2]
    matched_center_x = x0 + max_loc[0] + tw / 2.0
    matched_center_y = y0 + max_loc[1] + th / 2.0
    
    return np.array([matched_center_x, matched_center_y], dtype=np.float32), max_val


def run_template_rescue(
    frame_bgr: np.ndarray,
    preds: dict[str, np.ndarray | None],
    measurements: dict[str, np.ndarray],
    measurement_sources: dict[str, str],
    state: PipelineState,
    cfg: dict
) -> list[str]:
    """Tries to find obscured markers using localized visual matching."""
    tm_recovered_names = []
    
    if not cfg.get("template_matching_enabled", False):
        return tm_recovered_names

    for name in cfg.get("expected_corner_order", ["yellow", "red", "green", "blue"]):
        if name not in measurements:
            pred = preds.get(name)
            template = state.marker_templates.get(name)
            
            if pred is not None and template is not None:
                # Dynamically stiffen acceptance threshold if marker hasn't been lost for long
                min_score = cfg.get("template_match_min_score", 0.82)
                if state.marker_miss_counts.get(name, 0) < cfg.get("reacquire_after_miss", 3):
                    min_score += 0.05
                    
                tm_center, tm_score = detect_marker_by_template(
                    frame_bgr=frame_bgr,
                    name=name,
                    prediction=pred,
                    template=template,
                    cfg=cfg,
                )
                
                if tm_center is not None and tm_score >= min_score:
                    dist = np.linalg.norm(tm_center - pred)
                    if dist <= cfg.get("template_max_accept_dist", 45.0):
                        # Geometric containment check preventing drifting matches
                        is_inside_val = True
                        if state.last_render_ordered_pts is not None:
                            dist_poly = cv2.pointPolygonTest(
                                state.last_render_ordered_pts.astype(np.float32), 
                                tuple(tm_center), 
                                True
                            )
                            is_inside_val = dist_poly >= -cfg.get("template_outside_quad_margin", 35)
                            
                        if is_inside_val:
                            measurements[name] = tm_center.astype(np.float32)
                            measurement_sources[name] = "template"
                            tm_recovered_names.append(name)
                            
    return tm_recovered_names
