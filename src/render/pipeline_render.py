import cv2
import numpy as np
from src.pipelines.pipeline_state import PipelineState
from src.geometry.order_corners import is_reasonable_quadrilateral
from src.geometry.plane_rectification import warp_plane_to_canonical, warp_mask_to_frame
from src.segmentation.foreground_on_plane import segment_foreground_on_rectified
from src.geometry.homography import compute_homography_from_overlay_to_plane, warp_overlay_to_frame
from src.render.occlusion_compositor import composite_overlay_under_foreground
from src.render.debug_view import draw_debug_info


def render_and_occlusion(
    frame_idx: int,
    frame_bgr: np.ndarray,
    overlay_bgr: np.ndarray,
    final_pts: np.ndarray | None,
    completion_status: str,
    strong_visible_names: set[str],
    state: PipelineState,
    inner_paper_mask: np.ndarray,
    cfg: dict
) -> np.ndarray:
    """Warps overlays beneath dynamically masked planar occlusion boundaries."""
    result_bgr = frame_bgr.copy()
    if final_pts is not None and is_reasonable_quadrilateral(final_pts):
        rect_view, H_frame_to_plane = warp_plane_to_canonical(frame_bgr, final_pts, cfg["canonical_plane_size"])
        
        is_full_real = (completion_status == "all_visible" and len(strong_visible_names) == 4)
        
        if state.clean_reference is None:
            if is_full_real:
                state.warmup_count += 1
                if state.warmup_count > cfg.get("ref_warmup_frames", 0):
                    rect_blur = cv2.GaussianBlur(rect_view, (3, 3), 0)
                    state.ref_buffer.append(rect_blur.astype(np.float32))
                    if len(state.ref_buffer) >= cfg.get("ref_init_frames", 5):
                        state.clean_reference = np.median(np.stack(state.ref_buffer), axis=0).astype(np.uint8)
                        print(f"[SEQ4] Reference built at frame {frame_idx}")
        else:
            fg_mask_raw = segment_foreground_on_rectified(rect_view, state.clean_reference, cfg)
            fg_mask_raw = cv2.bitwise_and(fg_mask_raw, inner_paper_mask)
            
            state.mask_buffer.append(fg_mask_raw)
            combined_mask = np.mean(np.stack(list(state.mask_buffer)), axis=0)
            fg_mask_stable = (combined_mask > 128).astype(np.uint8) * 255
            
            d_size = cfg.get("fg_mask_dilation", 9)
            k_dilated = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d_size, d_size))
            fg_mask_final = cv2.dilate(fg_mask_stable, k_dilated)
            
            if is_full_real:
                bg_mask = (fg_mask_final == 0)
                up_a = cfg.get("ref_update_alpha", 0.005)
                state.clean_reference[bg_mask] = cv2.addWeighted(
                    state.clean_reference[bg_mask], 1.0 - up_a, rect_view[bg_mask], up_a, 0
                )
            
            fg_mask_frame = warp_mask_to_frame(fg_mask_final, H_frame_to_plane, frame_bgr.shape)
            
            H_overlay = compute_homography_from_overlay_to_plane(overlay_bgr, final_pts)
            warped_overlay, warped_mask = warp_overlay_to_frame(overlay_bgr, frame_bgr.shape, H_overlay)
            
            e_size = cfg.get("render_overlay_erosion", 2)
            if e_size > 0:
                k_erosion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (e_size * 2 + 1, e_size * 2 + 1))
                warped_mask = cv2.erode(warped_mask, k_erosion)
            
            result_bgr = composite_overlay_under_foreground(frame_bgr, warped_overlay, warped_mask, fg_mask_frame, cfg)
            
    return result_bgr


def draw_advanced_debug_overlay(
    result_bgr: np.ndarray,
    frame_idx: int,
    preds: dict,
    measurements: dict,
    measurement_sources: dict,
    rej_status: dict,
    strong_visible_names: set[str],
    weak_visible_names: set[str],
    strong_visible_centers: dict,
    weak_visible_centers: dict,
    visible_centers: dict,
    final_pts: np.ndarray | None,
    completion_status: str,
    geometry_centers: dict | None,
    missing_name: str | None,
    tm_recovered_names: list[str],
    was_ref_optimized: bool,
    was_edge_optimized: bool,
    state: PipelineState,
    cfg: dict
):
    """Draws rich metadata showing optical bounds, fallback modes, and track estimates."""
    debug_centers = visible_centers.copy()
    result_bgr = draw_debug_info(result_bgr, debug_centers, final_pts, cfg, frame_idx, preds, measurements, rej_status)
    
    for tmn in tm_recovered_names:
        if tmn in debug_centers:
            tmc = debug_centers[tmn]
            cv2.putText(result_bgr, "TM", (int(tmc[0])-15, int(tmc[1])-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
    if cfg.get("draw_template_debug", False) and len(tm_recovered_names) > 0:
        cv2.putText(result_bgr, f"TM Recovered: {tm_recovered_names}", (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if cfg.get("draw_missing_corner_debug", True):
        y_offset = 65
        val_color = (0, 255, 255)
        cv2.putText(result_bgr, f"Strong Visible: {len(strong_visible_centers)} {sorted(strong_visible_names)}", (20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, val_color, 2)
        y_offset += 25
        cv2.putText(result_bgr, f"Weak Visible: {len(weak_visible_centers)} {sorted(weak_visible_names)}", (20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, val_color, 2)
        y_offset += 25
        cv2.putText(result_bgr, f"Geometry: {completion_status}", (20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, val_color, 2)
        y_offset += 25
        cv2.putText(result_bgr, f"Lost geometry: {state.lost_geometry_count}", (20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, val_color, 2)
        y_offset += 25
        
        if missing_name:
            cv2.putText(result_bgr, f"Missing: {missing_name} (Hold: {state.missing_corner_hold_count})", (20, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, val_color, 2)
            
            if geometry_centers and missing_name in geometry_centers and (
                "affine_completed" in completion_status or "parallelogram_completed" in completion_status
            ):
                est_pt = geometry_centers[missing_name]
                cv2.circle(result_bgr, (int(est_pt[0]), int(est_pt[1])), 8, (255, 0, 255), -1)
                cv2.circle(result_bgr, (int(est_pt[0]), int(est_pt[1])), 12, (255, 0, 255), 2)
                cv2.putText(result_bgr, f"est {missing_name}", (int(est_pt[0])+10, int(est_pt[1])-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    if cfg.get("draw_quad_prediction_debug", False) and state.last_predicted_ordered_pts is not None:
        pred_pts = state.last_predicted_ordered_pts.astype(int)
        cv2.polylines(result_bgr, [pred_pts], True, (0, 165, 255), 2)
        cv2.putText(result_bgr, "velocity predicted quad", tuple(pred_pts[0]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
                    
    if cfg.get("draw_refinement_debug", False):
        y_offset = 200
        if was_ref_optimized:
            cv2.putText(result_bgr, "optimized by REF", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 25
        if was_edge_optimized:
            cv2.putText(result_bgr, "edge refined", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)

    return result_bgr
