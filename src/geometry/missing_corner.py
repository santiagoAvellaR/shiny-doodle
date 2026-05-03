from __future__ import annotations

import cv2
import numpy as np

from src.geometry.order_corners import centers_to_ordered_points, is_reasonable_quadrilateral, polygon_area, is_convex_ordered_quad
from src.pipelines.pipeline_state import PipelineState
from src.geometry.quad_prediction import fuse_missing_corner_with_temporal_prediction
from src.geometry.paper_edges import refine_missing_corner_with_paper_edges
from src.geometry.missing_corner_optimization import optimize_missing_corner_by_reference_alignment


def apply_affine_to_point(M: np.ndarray, point: np.ndarray) -> np.ndarray:
    """
    Applies a 2x3 affine transform to a 2D point.
    Returns np.ndarray shape (2,).
    """
    pt_h = np.array([point[0], point[1], 1.0], dtype=np.float64)
    transformed_pt = M @ pt_h
    return transformed_pt


def complete_quad_with_temporal_affine(
    current_centers: dict[str, np.ndarray],
    previous_complete_centers: dict[str, np.ndarray] | None,
    expected_order: list[str],
    cfg: dict,
) -> tuple[dict[str, np.ndarray] | None, str, str | None]:
    visible_names = list(current_centers.keys())

    if len(visible_names) == 4:
        return current_centers.copy(), "all_visible", None

    if len(visible_names) != 3:
        return None, "failed", None

    missing_name = next((color for color in expected_order if color not in visible_names), None)

    if previous_complete_centers is None:
        return None, "failed", missing_name

    if not all(name in previous_complete_centers for name in visible_names):
        return None, "failed", missing_name

    if missing_name not in previous_complete_centers:
        return None, "failed", missing_name

    src = np.array([previous_complete_centers[name] for name in visible_names], dtype=np.float32)
    dst = np.array([current_centers[name] for name in visible_names], dtype=np.float32)

    M = cv2.getAffineTransform(src, dst)
    estimated_pt = apply_affine_to_point(M, previous_complete_centers[missing_name])

    completed_centers = current_centers.copy()
    completed_centers[missing_name] = estimated_pt

    return completed_centers, "affine_completed", missing_name


def complete_quad_with_parallelogram_fallback(
    current_centers: dict[str, np.ndarray],
    expected_order: list[str],
) -> tuple[dict[str, np.ndarray] | None, str | None]:
    visible_names = list(current_centers.keys())

    if len(visible_names) != 3:
        return None, None
        
    missing_name = next((color for color in expected_order if color not in visible_names), None)
    if not missing_name:
        return None, None

    completed_centers = current_centers.copy()

    tl_n, tr_n, br_n, bl_n = expected_order
    
    try:
        if missing_name == tl_n:
            tr = current_centers[tr_n]
            br = current_centers[br_n]
            bl = current_centers[bl_n]
            completed_centers[tl_n] = tr + bl - br
        elif missing_name == tr_n:
            tl = current_centers[tl_n]
            br = current_centers[br_n]
            bl = current_centers[bl_n]
            completed_centers[tr_n] = tl + br - bl
        elif missing_name == br_n:
            tl = current_centers[tl_n]
            tr = current_centers[tr_n]
            bl = current_centers[bl_n]
            completed_centers[br_n] = tr + bl - tl
        elif missing_name == bl_n:
            tl = current_centers[tl_n]
            tr = current_centers[tr_n]
            br = current_centers[br_n]
            completed_centers[bl_n] = tl + br - tr
        else:
            return None, missing_name
    except KeyError:
        return None, missing_name
        
    return completed_centers, missing_name


def validate_completed_quad(
    completed_centers: dict[str, np.ndarray],
    expected_order: list[str],
    prev_area: float | None,
    prev_aspect: float | None,
    cfg: dict,
) -> tuple[bool, float | None, float | None]:
    try:
        ordered_pts = centers_to_ordered_points(completed_centers, expected_order)
    except KeyError:
        return False, None, None

    if not is_reasonable_quadrilateral(ordered_pts):
        return False, None, None

    area = polygon_area(ordered_pts)
    side_w = np.linalg.norm(ordered_pts[0] - ordered_pts[1])
    side_h = np.linalg.norm(ordered_pts[0] - ordered_pts[3])
    aspect = side_w / side_h if side_h > 0 else 0

    if prev_area is not None:
        if abs(area - prev_area) / prev_area > cfg.get("missing_corner_area_tol", 0.45):
            return False, None, None

    if prev_aspect is not None:
        if abs(aspect - prev_aspect) / prev_aspect > cfg.get("missing_corner_aspect_tol", 0.45):
            return False, None, None

    return True, area, aspect


def _internal_refine_centers(
    frame_bgr: np.ndarray,
    clean_reference: np.ndarray | None,
    completed_centers: dict[str, np.ndarray],
    missing_name: str,
    predicted_ordered_pts: np.ndarray | None,
    cfg: dict,
) -> tuple[dict[str, np.ndarray], bool, bool]:
    current_centers = completed_centers.copy()
    ref_optimized = False
    edge_optimized = False

    if cfg.get("quad_velocity_enabled", False) and predicted_ordered_pts is not None:
        current_centers = fuse_missing_corner_with_temporal_prediction(
            completed_centers=current_centers,
            missing_name=missing_name,
            predicted_ordered_pts=predicted_ordered_pts,
            expected_order=cfg["expected_corner_order"],
            fusion_weight=cfg.get("quad_velocity_fusion_weight", 0.20),
            max_jump_px=cfg.get("missing_corner_max_jump_px", 120.0),
        )

    if cfg.get("missing_corner_refine_enabled", False) and clean_reference is not None:
        opt_point = optimize_missing_corner_by_reference_alignment(
            frame_bgr=frame_bgr,
            clean_reference=clean_reference,
            completed_centers=current_centers,
            missing_name=missing_name,
            expected_order=cfg["expected_corner_order"],
            cfg=cfg,
        )
        if opt_point is not None:
            current_centers = current_centers.copy()
            current_centers[missing_name] = opt_point.astype(np.float32)
            ref_optimized = True

    if cfg.get("paper_edge_refine_enabled", False):
        edge_point = refine_missing_corner_with_paper_edges(
            frame_bgr=frame_bgr,
            completed_centers=current_centers,
            missing_name=missing_name,
            expected_order=cfg["expected_corner_order"],
            cfg=cfg,
        )
        if edge_point is not None:
            geometric_point = current_centers[missing_name]
            w_edge = cfg.get("paper_edge_fusion_weight", 0.25)
            fused_point = (1.0 - w_edge) * geometric_point + w_edge * edge_point
            current_centers[missing_name] = fused_point.astype(np.float32)
            edge_optimized = True
            
    return current_centers, ref_optimized, edge_optimized


def _check_completed_quad_validity(centers: dict, cfg: dict, prev_area: float | None, prev_aspect: float | None) -> bool:
    is_valid, _, _ = validate_completed_quad(centers, cfg["expected_corner_order"], prev_area, prev_aspect, cfg)
    if is_valid:
        pts = centers_to_ordered_points(centers, cfg["expected_corner_order"])
        if not is_reasonable_quadrilateral(pts) or not is_convex_ordered_quad(pts):
            return False
    return is_valid


def estimate_robust_geometry(
    frame_bgr: np.ndarray,
    strong_visible_centers: dict[str, np.ndarray],
    intermediate_visible_centers: dict[str, np.ndarray],
    visible_centers: dict[str, np.ndarray],
    predicted_ordered_pts: np.ndarray | None,
    state: PipelineState,
    cfg: dict
) -> tuple[np.ndarray | None, str, dict[str, np.ndarray] | None, str | None, bool, bool]:
    """Determines how to build the final valid point projection."""
    geometry_centers = None
    completion_status = "failed"
    missing_name = None
    
    was_ref_optimized = False
    was_edge_optimized = False

    reconstruction_centers = None
    reconstruction_is_weak = False
    
    combined_strong_and_inter = strong_visible_centers.copy()
    combined_strong_and_inter.update(intermediate_visible_centers)

    if len(strong_visible_centers) == 4:
        geometry_centers = strong_visible_centers.copy()
        completion_status = "all_visible"
        state.active_missing_name = None
    elif len(combined_strong_and_inter) == 4:
        geometry_centers = combined_strong_and_inter.copy()
        completion_status = "all_visible_candidate"
        state.missing_corner_hold_count = 0
        state.active_missing_name = None

    elif cfg.get("allow_missing_corner", True):
        if len(strong_visible_centers) == 3:
            reconstruction_centers = strong_visible_centers
            reconstruction_is_weak = False
        elif len(combined_strong_and_inter) == 3:
            reconstruction_centers = combined_strong_and_inter
            reconstruction_is_weak = False
            completion_status_tag = "intermediate"
        elif len(strong_visible_centers) == 2 and len(visible_centers) == 3:
            reconstruction_centers = visible_centers
            reconstruction_is_weak = True

        if reconstruction_centers is not None:
            # 6. PERSISTENCY LOGIC
            # Decide which method to try first based on history
            methods_to_try = []
            if state.active_geometry_method == "affine":
                methods_to_try = ["affine", "parallelogram"]
            elif state.active_geometry_method == "parallelogram":
                methods_to_try = ["parallelogram", "affine"]
            else:
                methods_to_try = ["affine", "parallelogram"]

            for method in methods_to_try:
                temp_centers = None
                temp_missing = None
                temp_tag = ""
                
                if method == "affine" and cfg.get("missing_corner_use_temporal_affine", True):
                    temp_centers, _, temp_missing = complete_quad_with_temporal_affine(
                        reconstruction_centers, state.previous_complete_centers, cfg["expected_corner_order"], cfg
                    )
                    temp_tag = "affine"
                elif method == "parallelogram" and cfg.get("missing_corner_use_parallelogram_fallback", True):
                    temp_centers, temp_missing = complete_quad_with_parallelogram_fallback(
                        reconstruction_centers, cfg["expected_corner_order"]
                    )
                    temp_tag = "parallelogram"

                if temp_centers is not None:
                    refined_centers, ref_op, edge_op = _internal_refine_centers(
                        frame_bgr, state.clean_reference, temp_centers, temp_missing, predicted_ordered_pts, cfg
                    )
                    is_valid = _check_completed_quad_validity(refined_centers, cfg, state.prev_quad_area, state.prev_quad_aspect)
                    
                    if is_valid:
                        # Success with this method
                        if state.active_geometry_method == temp_tag and temp_missing == state.active_missing_name:
                            state.geometry_method_fail_count = 0
                        else:
                            state.active_geometry_method = temp_tag
                            state.active_missing_name = temp_missing
                            state.geometry_method_fail_count = 0
                            state.missing_corner_hold_count = 0
                            
                        geometry_centers = refined_centers
                        missing_name = temp_missing
                        state.missing_corner_hold_count += 1
                        was_ref_optimized = ref_op
                        was_edge_optimized = edge_op
                        
                        prefix = temp_tag
                        if reconstruction_is_weak:
                            completion_status = f"{prefix}_completed_weak"
                        elif "completion_status_tag" in locals() and completion_status_tag == "intermediate":
                            completion_status = f"{prefix}_completed_intermediate"
                        else:
                            completion_status = f"{prefix}_completed"
                        break # Found a valid reconstruction
                    else:
                        # Method failed for this frame
                        if state.active_geometry_method == temp_tag:
                            state.geometry_method_fail_count += 1
                            if state.geometry_method_fail_count < cfg.get("geometry_method_switch_grace_frames", 3):
                                # Don't try other methods yet, keep the "failed" status for this frame to encourage persistence or freeze
                                break 

    if geometry_centers is not None and missing_name is not None and state.missing_corner_hold_count > cfg.get("missing_corner_max_hold_frames", 80):
        completion_status = completion_status + "_long_hold"

    raw_ordered_pts = None
    is_valid_quad = False
    is_full_real_geometry = (completion_status == "all_visible" and len(strong_visible_centers) == 4)

    if geometry_centers is not None:
        proposal_pts = centers_to_ordered_points(geometry_centers, cfg["expected_corner_order"])
        if is_reasonable_quadrilateral(proposal_pts) and is_convex_ordered_quad(proposal_pts):
            area = polygon_area(proposal_pts)
            side_w = np.linalg.norm(proposal_pts[0] - proposal_pts[1])
            side_h = np.linalg.norm(proposal_pts[0] - proposal_pts[3])
            aspect = side_w / side_h if side_h > 0 else 0
            
            if is_full_real_geometry:
                area_tol = cfg["quad_consistency_area_tol"]
                aspect_tol = cfg["quad_consistency_aspect_tol"]
            else:
                area_tol = cfg["missing_corner_area_tol"]
                aspect_tol = cfg["missing_corner_aspect_tol"]
                
            is_consistent = True
            if state.prev_quad_area is not None:
                if abs(area - state.prev_quad_area) / state.prev_quad_area > area_tol:
                    is_consistent = False
            if state.prev_quad_aspect is not None:
                if abs(aspect - state.prev_quad_aspect) / state.prev_quad_aspect > aspect_tol:
                    is_consistent = False
            
            if is_consistent:
                raw_ordered_pts = proposal_pts.copy()

                # 5. LIMIT MOVEMENT (Step Clamp)
                if missing_name is not None and state.last_render_final_pts is not None:
                    idx = cfg["expected_corner_order"].index(missing_name)
                    prev_pt = state.last_render_final_pts[idx]
                    new_pt = raw_ordered_pts[idx]
                    
                    dist_jump = np.linalg.norm(new_pt - prev_pt)
                    max_step = cfg.get("estimated_corner_max_step_px", 25.0)
                    
                    if dist_jump > max_step:
                        direction = (new_pt - prev_pt) / dist_jump
                        raw_ordered_pts[idx] = prev_pt + direction * max_step

                state.prev_quad_area, state.prev_quad_aspect = area, aspect
                is_valid_quad = True
                
                if state.last_render_ordered_pts is not None:
                    state.prev_render_ordered_pts = state.last_render_ordered_pts.copy()
                state.last_render_ordered_pts = raw_ordered_pts.copy()
                state.last_accepted_quad = raw_ordered_pts.copy()

                if is_full_real_geometry:
                    if state.last_full_real_ordered_pts is not None:
                        state.prev_full_real_ordered_pts = state.last_full_real_ordered_pts.copy()
                    state.last_full_real_ordered_pts = raw_ordered_pts.copy()
                    state.previous_complete_centers = geometry_centers.copy()
                    
                    # Reset persistency when all strong visible
                    state.active_geometry_method = None
                    state.active_missing_name = None
                    state.geometry_method_fail_count = 0

    if raw_ordered_pts is not None:
        state.lost_geometry_count = 0
        pts_to_process = raw_ordered_pts
    elif state.last_render_ordered_pts is not None and state.lost_geometry_count <= cfg.get("lost_geometry_max_freeze_frames", 5):
        state.lost_geometry_count += 1
        pts_to_process = state.last_render_ordered_pts
        completion_status = "frozen_last_valid"
    else:
        state.lost_geometry_count += 1
        pts_to_process = None
        completion_status = "lost_geometry"

    if pts_to_process is not None:
        # 4. ADAPTIVE SMOOTHING
        if completion_status == "all_visible":
            alpha = cfg.get("quad_ema_alpha_all_visible", 0.8)
        elif "completed" in completion_status and "intermediate" not in completion_status and "weak" not in completion_status:
            alpha = cfg.get("quad_ema_alpha_reconstructed", 0.45)
        elif "intermediate" in completion_status or "weak" in completion_status:
            alpha = cfg.get("quad_ema_alpha_intermediate", 0.30)
        elif completion_status == "frozen_last_valid":
            alpha = cfg.get("quad_ema_alpha_frozen", 0.0)
        else:
            alpha = cfg.get("quad_ema_alpha", 0.6)

        if state.rendered_pts_prev is None:
            state.rendered_pts_prev = pts_to_process.copy()
        else:
            state.rendered_pts_prev = alpha * pts_to_process + (1.0 - alpha) * state.rendered_pts_prev
        
        final_pts = state.rendered_pts_prev
        state.last_render_final_pts = final_pts.copy()
    else:
        final_pts = None
        state.last_render_final_pts = None

    return final_pts, completion_status, geometry_centers, missing_name, was_ref_optimized, was_edge_optimized
