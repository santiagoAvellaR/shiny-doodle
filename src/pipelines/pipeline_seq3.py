from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from collections import deque

# Geometría y Homografía
from src.geometry.homography import (
    compute_homography_from_overlay_to_plane,
    warp_overlay_to_frame,
    composite_overlay,
)
from src.geometry.order_corners import (
    is_reasonable_quadrilateral,
    centers_to_ordered_points,
    polygon_area
)
from src.geometry.plane_rectification import (
    warp_plane_to_canonical,
    warp_mask_to_frame
)

# I/O
from src.io.video_reader import open_video_reader
from src.io.video_writer import open_video_writer
from src.io.image_loader import load_overlay_image

# Tracking y Filtrado
from src.tracking.marker_tracker import MarkerTracker

# Detección y Refinamiento
from src.detection.blob_detection import detect_markers
from src.detection.marker_refinement import estimate_green_from_yrb, detect_green_local

# Segmentación y Composición de Oclusión
from src.segmentation.foreground_on_plane import (
    segment_foreground_on_rectified,
    build_inner_paper_mask
)
from src.render.occlusion_compositor import composite_overlay_under_foreground

# Calibración
from src.calibration.undistort import undistort_frame

# Visualización
from src.render.debug_view import draw_debug_info

from src.pipelines.pipeline_seq1 import default_seq1_config


def default_seq3_config() -> dict:
    """
    Advanced configuration for Sequence 3.
    Optimized for responsiveness and robust hand-over-overlay occlusion.
    """
    cfg = default_seq1_config()
    cfg.update({
        # --- STABILITY PARAMETERS (Optimized) ---
        "quad_ema_alpha": 0.8,          # Increased for high responsiveness (reactive)
        
        # --- REFERENCE PARAMETERS (Fast Init) ---
        "canonical_plane_size": (600, 400),
        "ref_warmup_frames": 2,         # Start building reference almost immediately
        "ref_init_frames": 8,           # Quicker median calculation
        "ref_update_alpha": 0.005,       # Extremely slow adaptation for high robustness
        
        # --- SEGMENTATION PARAMETERS (Responsive) ---
        "fg_diff_thresh": 25,
        "fg_lab_weights": [1.0, 2.5, 2.5], 
        "fg_min_blob_area": 1200,        
        "fg_mask_history": 1,            # Instant hand response (no lag)
        "paper_inner_margin": 15,
        
        # --- RENDER PARAMETERS ---
        "render_erosion_size": 3,        
        "render_soft_blur_size": 7,      
        "draw_debug": True
    })
    return cfg


def run_seq3(
    input_video: Path,
    overlay_image: Path,
    output_video: Path,
    display: bool = False,
    max_frames: int | None = None,
) -> None:
    cfg = default_seq3_config()
    overlay_bgr = load_overlay_image(overlay_image)
    cap = open_video_reader(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = open_video_writer(output_video, fps, width, height)

    # Re-initialize trackers (Alpha-Beta)
    trackers = {name: MarkerTracker(cfg["filter_alpha"], cfg["filter_beta"]) for name in cfg["expected_corner_order"]}
    
    frame_idx = 0
    green_miss_count = 0
    prev_quad_area = None
    prev_quad_aspect = None
    last_valid_ordered_pts = None
    rendered_pts_prev = None

    # Reference model variables
    ref_buffer = []
    clean_reference = None
    warmup_count = 0
    inner_paper_mask = build_inner_paper_mask(cfg["canonical_plane_size"], cfg["paper_inner_margin"])
    
    # Mask smoothing buffer
    mask_buffer = deque(maxlen=cfg["fg_mask_history"])

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok: break
            if max_frames is not None and frame_idx >= max_frames: break
            
            # 1. Undistort
            frame_bgr = undistort_frame(frame_bgr, cfg)
            
            # --- 2) PREDICTION ---
            preds = {name: t.predict() for name, t in trackers.items()}
            
            # --- 3) MEASUREMENT (Seq 1 robust logic) ---
            raw_detections, _ = detect_markers(frame_bgr, cfg)
            measurements = {}
            rej_status = {name: False for name in trackers}
            
            # Standard markers (Red, Blue, Yellow)
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

            # Robust Green Logic (Seed from YRBy measurement)
            pred_green = preds["green"]
            green_meas = None
            is_bootstrap = (trackers["green"].pos is None) or (green_miss_count >= cfg["green_miss_limit"])
            
            if is_bootstrap:
                green_seed = estimate_green_from_yrb(measurements.get("yellow"), measurements.get("red"), measurements.get("blue"))
                if green_seed is not None:
                    green_meas = detect_green_local(frame_bgr, green_seed, cfg["green_roi_bootstrap"], cfg["green_hsv_bootstrap"], cfg["green_min_area"], cfg["green_v_max_bootstrap"])
                    if green_meas is not None and np.linalg.norm(green_meas - green_seed) > cfg["green_gate_bootstrap"]:
                        green_meas = None
            else:
                green_meas = detect_green_local(frame_bgr, pred_green, cfg["green_roi_normal"], cfg["green_hsv_normal"], cfg["green_min_area"], cfg["green_v_max_normal"])
                if green_meas is not None and np.linalg.norm(green_meas - pred_green) > cfg["green_gate_normal"]:
                    green_meas = None

            if green_meas is not None:
                measurements["green"] = green_meas
                green_miss_count = 0 
            else:
                green_miss_count += 1
            
            # --- 4) UPDATE trackers ---
            for name, t in trackers.items():
                t.update(measurements.get(name))

            # --- 5) GEOMETRIC CONSISTENCY & SMOOTHING (High Responiveness) ---
            current_centers = {name: t.pos for name, t in trackers.items() if t.pos is not None}
            raw_ordered_pts = None
            is_valid_quad = False
            
            if len(current_centers) == 4:
                proposal_pts = centers_to_ordered_points(current_centers, cfg["expected_corner_order"])
                # Non-blocking render: allow immediate display if geometric shape is reasonable
                if is_reasonable_quadrilateral(proposal_pts):
                    raw_ordered_pts = proposal_pts
                    
                    area = polygon_area(proposal_pts)
                    side_w = np.linalg.norm(proposal_pts[0] - proposal_pts[1])
                    side_h = np.linalg.norm(proposal_pts[0] - proposal_pts[3])
                    aspect = side_w / side_h if side_h > 0 else 0
                    
                    # Consistency check mainly for history update
                    is_consistent = True
                    if prev_quad_area is not None:
                        if abs(area - prev_quad_area) / prev_quad_area > cfg["quad_consistency_area_tol"]:
                            is_consistent = False
                    if prev_quad_aspect is not None:
                        if abs(aspect - prev_quad_aspect) / prev_quad_aspect > cfg["quad_consistency_aspect_tol"]:
                            is_consistent = False
                    
                    if is_consistent:
                        prev_quad_area, prev_quad_aspect = area, aspect
                        last_valid_ordered_pts = raw_ordered_pts.copy()
                        is_valid_quad = True
            
            # Use raw points or last valid
            pts_to_process = raw_ordered_pts if raw_ordered_pts is not None else last_valid_ordered_pts

            # Temporal Smoothing (Applied ONLY to rendering/rectification)
            if pts_to_process is not None:
                # If reference is not ready, alpha is 1.0 (no lag). Otherwise, moderate smoothing.
                alpha = 1.0 if clean_reference is None else cfg["quad_ema_alpha"]
                if rendered_pts_prev is None:
                    rendered_pts_prev = pts_to_process.copy()
                else:
                    rendered_pts_prev = alpha * pts_to_process + (1.0 - alpha) * rendered_pts_prev
                final_pts = rendered_pts_prev
            else:
                final_pts = None

            # --- 6) SEQ3 SPECIFIC: RECTIFICATION & OCCLUSION ---
            result_bgr = frame_bgr.copy()
            if final_pts is not None and is_reasonable_quadrilateral(final_pts):
                # A) Rectify current view using smoothed points for render
                rect_view, H_frame_to_plane = warp_plane_to_canonical(frame_bgr, final_pts, cfg["canonical_plane_size"])
                # B) Rectify using raw points for reference building (less "dragging" blur)
                rect_view_raw, _ = warp_plane_to_canonical(frame_bgr, pts_to_process, cfg["canonical_plane_size"])
                
                # Background Reference Management
                if clean_reference is None:
                    if is_valid_quad:
                        warmup_count += 1
                        if warmup_count > cfg["ref_warmup_frames"]:
                            rect_blur = cv2.GaussianBlur(rect_view_raw, (3, 3), 0)
                            ref_buffer.append(rect_blur.astype(np.float32))
                            if len(ref_buffer) >= cfg["ref_init_frames"]:
                                clean_reference = np.median(np.stack(ref_buffer), axis=0).astype(np.uint8)
                                print(f"[SEQ3] Reference built at frame {frame_idx}")
                    
                    # Policy: NO Rendering before reference is ready (prevents overlay on hand)
                    pass
                else:
                    # C) Segmentation (Lab + Blob cleaning)
                    fg_mask_raw = segment_foreground_on_rectified(rect_view, clean_reference, cfg)
                    fg_mask_raw = cv2.bitwise_and(fg_mask_raw, inner_paper_mask)
                    
                    # D) Temporal Mask Consensus (Smoothing ocluder edges)
                    mask_buffer.append(fg_mask_raw)
                    combined_mask = np.mean(np.stack(list(mask_buffer)), axis=0)
                    fg_mask_stable = (combined_mask > 128).astype(np.uint8) * 255
                    
                    # Dilatación para ser conservadores (Mano delante) para que tape bien los bordes
                    k_dilated = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                    fg_mask_final = cv2.dilate(fg_mask_stable, k_dilated)
                    
                    # E) Background Adaptation (Very slow)
                    bg_mask = (fg_mask_stable == 0)
                    up_a = cfg["ref_update_alpha"]
                    clean_reference[bg_mask] = cv2.addWeighted(clean_reference[bg_mask], 1.0 - up_a, rect_view[bg_mask], up_a, 0)
                    
                    # F) Render with soft alpha blending and occlusion awareness
                    fg_mask_frame = warp_mask_to_frame(fg_mask_final, H_frame_to_plane, frame_bgr.shape)
                    H_overlay = compute_homography_from_overlay_to_plane(overlay_bgr, final_pts)
                    warped_overlay, warped_mask = warp_overlay_to_frame(overlay_bgr, frame_bgr.shape, H_overlay)
                    result_bgr = composite_overlay_under_foreground(frame_bgr, warped_overlay, warped_mask, fg_mask_frame, cfg)

            # Debug Visualization
            if cfg["draw_debug"]:
                result_bgr = draw_debug_info(result_bgr, current_centers, final_pts, cfg, frame_idx, preds, measurements, rej_status)

            writer.write(result_bgr)
            if display:
                cv2.imshow("SY32 - Sequence 3 (Optimized)", result_bgr)
                if cv2.waitKey(1) & 0xFF == 27: break
            frame_idx += 1

    finally:
        cap.release()
        writer.release()
        if display: cv2.destroyAllWindows()
        
    print(f"[OK] Sequence 3 video completed (Polished Logic): {output_video}")
