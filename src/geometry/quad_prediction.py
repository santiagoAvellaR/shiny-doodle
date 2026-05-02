import numpy as np
import math
from src.pipelines.pipeline_state import PipelineState


def predict_quad_from_temporal_velocity(
    last_valid_ordered_pts: np.ndarray | None,
    prev_valid_ordered_pts: np.ndarray | None,
    velocity_factor: float = 0.7,
) -> np.ndarray | None:
    """
    Predicts the next quadrilateral using constant velocity extrapolation.

    Inputs:
    - last_valid_ordered_pts: np.ndarray shape (4, 2)
    - prev_valid_ordered_pts: np.ndarray shape (4, 2)
    - velocity_factor: float

    Returns:
    - predicted_pts: np.ndarray shape (4, 2), or None if not enough history.
    """
    if last_valid_ordered_pts is None:
        return None

    if prev_valid_ordered_pts is None:
        return last_valid_ordered_pts.copy()

    velocity = last_valid_ordered_pts - prev_valid_ordered_pts
    predicted_pts = last_valid_ordered_pts + velocity_factor * velocity
    return predicted_pts.astype(np.float32)


def ordered_pts_to_centers_dict(
    ordered_pts: np.ndarray,
    expected_order: list[str],
) -> dict[str, np.ndarray]:
    """
    Converts ordered points shape (4, 2) to a dict color -> point.
    expected_order is usually ["yellow", "red", "green", "blue"].
    """
    return {name: ordered_pts[i].copy() for i, name in enumerate(expected_order)}


def fuse_missing_corner_with_temporal_prediction(
    completed_centers: dict[str, np.ndarray],
    missing_name: str,
    predicted_ordered_pts: np.ndarray,
    expected_order: list[str],
    fusion_weight: float,
    max_jump_px: float | None = None,
) -> dict[str, np.ndarray]:
    """
    Fuses the geometrically estimated missing corner with the temporal prediction.

    completed_centers contains the geometric estimate.
    predicted_ordered_pts contains the velocity-based predicted quadrilateral.

    Only the missing corner should be modified.
    The 3 real visible corners must remain unchanged.

    Formula:
    fused = (1 - fusion_weight) * geometric_point + fusion_weight * temporal_point

    If max_jump_px is not None, reject the temporal fusion if temporal_point is too far
    from geometric_point.

    Return a new completed_centers dict.
    """
    if missing_name not in expected_order:
        return completed_centers

    missing_idx = expected_order.index(missing_name)
    temporal_point = predicted_ordered_pts[missing_idx]
    geometric_point = completed_centers[missing_name]

    if max_jump_px is not None:
        dist = np.linalg.norm(temporal_point - geometric_point)
        if dist > max_jump_px:
            return completed_centers.copy()

    fused_point = (1.0 - fusion_weight) * geometric_point + fusion_weight * temporal_point
    
    new_centers = completed_centers.copy()
    new_centers[missing_name] = fused_point.astype(np.float32)
    return new_centers

def validate_predictive_velocity(
    state: PipelineState,
    strong_visible_centers: dict[str, np.ndarray],
    cfg: dict
) -> np.ndarray | None:
    """Generates a predictive temporal vector strictly if it adheres to current known physical boundaries."""
    if not cfg.get("quad_velocity_enabled", False):
        return None

    # Prefer pure history for velocities
    predicted_ordered_pts = predict_quad_from_temporal_velocity(
        state.last_full_real_ordered_pts,
        state.prev_full_real_ordered_pts,
        cfg.get("quad_velocity_factor", 0.55),
    )
    if predicted_ordered_pts is None:
        predicted_ordered_pts = predict_quad_from_temporal_velocity(
            state.last_render_ordered_pts,
            state.prev_render_ordered_pts,
            cfg.get("quad_velocity_factor", 0.55),
        )
        
    if predicted_ordered_pts is None:
        return None
        
    # Validation against strong real points
    errors = []
    max_jump = 0.0
    expected_order = cfg["expected_corner_order"]
    
    for name, current_pt in strong_visible_centers.items():
        idx = expected_order.index(name)
        pred_pt = predicted_ordered_pts[idx]
        d = np.linalg.norm(current_pt - pred_pt)
        errors.append(d)
        if d > max_jump:
            max_jump = d
            
    if len(errors) > 0:
        rms = math.sqrt(sum(e*e for e in errors) / len(errors))
        if rms > cfg.get("quad_velocity_visible_rms_gate", 35.0):
            return None
        if max_jump > cfg.get("quad_velocity_max_corner_jump", 70.0):
            return None
            
    # Only keep last_predicted_ordered_pts if it actually passes validation
    state.last_predicted_ordered_pts = predicted_ordered_pts.copy()
    return predicted_ordered_pts
