from pathlib import Path
import cv2

# Detecciones y Tracking
from src.pipelines.pipeline_state import PipelineState
from src.tracking.marker_tracker import MarkerTracker
from src.detection.measurement_acquisition import acquire_strong_measurements, run_local_reacquire
from src.detection.template_matching import run_template_rescue
from src.tracking.tracker_management import update_trackers_and_templates, update_marker_visibility_state

# Geometría Funcional
from src.geometry.quad_prediction import validate_predictive_velocity
from src.geometry.missing_corner import estimate_robust_geometry

# Render y Oclusión
from src.segmentation.foreground_on_plane import build_inner_paper_mask
from src.render.pipeline_render import render_and_occlusion, draw_advanced_debug_overlay

# I/O y Calibración
from src.io.video_reader import open_video_reader
from src.io.video_writer import open_video_writer
from src.io.image_loader import load_overlay_image
from src.calibration.undistort import undistort_frame
from src.pipelines.pipeline_seq1 import default_seq1_config


def default_seq4_config() -> dict:
    cfg = default_seq1_config()
    cfg.update({
        "quad_ema_alpha": 0.8,
        "canonical_plane_size": (600, 400),
        "ref_warmup_frames": 0,         
        "ref_init_frames": 5,           
        "ref_update_alpha": 0.005,      
        
        "fg_diff_thresh": 25,
        "fg_lab_weights": [1.0, 2.5, 2.5], 
        "fg_min_blob_area": 1200,        
        "fg_mask_history": 1,            
        "paper_inner_margin": 8,         
        "fg_mask_dilation": 9,           
        
        "render_erosion_size": 3,        
        "render_overlay_erosion": 2,     
        "render_soft_blur_size": 7,      
        "draw_debug": True,
        
        "allow_missing_corner": True,
        "min_visible_markers_seq4": 3,
        "missing_corner_max_hold_frames": 80,
        "missing_corner_max_jump_px": 120.0,
        "missing_corner_area_tol": 0.60,
        "missing_corner_aspect_tol": 0.60,
        "missing_corner_use_temporal_affine": True,
        "missing_corner_use_parallelogram_fallback": True,
        "draw_missing_corner_debug": True,

        "reacquire_after_miss": 3,
        "reacquire_gate_px": 180.0,
        "hard_reacquire_after_miss": 10,
        "lost_geometry_max_freeze_frames": 5,

        "template_matching_enabled": True,
        "template_patch_radius": 18,
        "template_search_radius": 45,
        "template_match_min_score": 0.82,
        "template_update_alpha": 0.02,
        "template_max_accept_dist": 45.0,
        "template_use_gray": False,
        "template_outside_quad_margin": 35,
        "template_updates_tracker": False,
        "template_resets_miss_count": False,
        "draw_template_debug": True,

        "quad_velocity_enabled": True,
        "quad_velocity_factor": 0.55,
        "quad_velocity_fusion_weight": 0.20,
        "quad_velocity_visible_rms_gate": 35.0,
        "quad_velocity_max_corner_jump": 70.0,

        "paper_edge_refine_enabled": False,
        "paper_edge_search_margin": 45,
        "paper_edge_canny_low": 40,
        "paper_edge_canny_high": 120,
        "paper_edge_hough_threshold": 35,
        "paper_edge_min_line_length": 50,
        "paper_edge_max_line_gap": 15,
        "paper_edge_fusion_weight": 0.25,
        "paper_edge_max_correction_px": 40.0,
        "paper_edge_use_white_mask": True,
        "paper_white_s_max": 80,
        "paper_white_v_min": 120,
        "paper_white_min_area": 8000,

        "missing_corner_refine_enabled": False,
        "missing_corner_refine_coarse_radius": 35,
        "missing_corner_refine_coarse_step": 7,
        "missing_corner_refine_fine_radius": 8,
        "missing_corner_refine_fine_step": 2,
        "missing_corner_refine_max_improvement": 0.02,
        "missing_corner_refine_max_shift_px": 45.0,
        "missing_corner_refine_use_lab": True,
        "missing_corner_refine_downscale": 2,
        "draw_refinement_debug": True,
        "draw_quad_prediction_debug": True,

        "estimated_corner_max_step_px": 25.0,
        "quad_ema_alpha_all_visible": 0.82,
        "quad_ema_alpha_reconstructed": 0.45,
        "quad_ema_alpha_intermediate": 0.28,
        "quad_ema_alpha_frozen": 0.0,

        "local_reacquire_enabled": True,
        "local_reacquire_after_miss": 3,
        "local_reacquire_radius": 75,
        "local_reacquire_min_area": 35,
        "local_reacquire_min_circularity": 0.12,
        "local_reacquire_gate_px": 220.0,
        "local_reacquire_source_name": "hsv_local_reacquire",
        "local_reacquire_confirm_frames": 2,
        "local_reacquire_hard_after_miss": 5,
        "disable_local_reacquire_inside_foreground": True,
        "local_reacquire_foreground_max_ratio": 0.4,
        "local_reacquire_hsv": {
            "yellow": [((20, 60, 60), (40, 255, 255))],
            "red": [((0, 60, 60), (12, 255, 255)), ((165, 60, 60), (180, 255, 255))],
            "green": [((42, 50, 45), (92, 255, 255))], # Relaxed for reacquire
            "blue": [((100, 65, 55), (135, 255, 255))]
        },
        "geometry_method_switch_grace_frames": 3,
    })
    return cfg


def run_seq4(
    input_video: Path,
    overlay_image: Path,
    output_video: Path,
    display: bool = False,
    max_frames: int | None = None,
    debug: bool = False,
) -> None:
    cfg = default_seq4_config()
    overlay_bgr = load_overlay_image(overlay_image)
    cap = open_video_reader(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = open_video_writer(output_video, fps, width, height)

    state = PipelineState()
    state.trackers = {name: MarkerTracker(cfg["filter_alpha"], cfg["filter_beta"]) for name in cfg["expected_corner_order"]}
    
    frame_idx = 0
    inner_paper_mask = build_inner_paper_mask(cfg["canonical_plane_size"], cfg["paper_inner_margin"])

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok or (max_frames is not None and frame_idx >= max_frames):
                break

            if debug:
                cfg["draw_debug"] = True
            else:
                cfg["draw_debug"] = False
            
            frame_bgr = undistort_frame(frame_bgr, cfg)
            preds = {name: t.predict() for name, t in state.trackers.items()}
            
            # 1. Acquire Optical Information
            measurements, measurement_sources, rej_status = acquire_strong_measurements(frame_bgr, preds, state, cfg)
            
            # 2. Geometry-seeded local reacquisition (foreground aware)
            lr_recovered_names = run_local_reacquire(frame_bgr, state.last_predicted_ordered_pts, measurements, measurement_sources, state, cfg)
            
            # 3. Rescue via Template Matching
            tm_recovered_names = run_template_rescue(frame_bgr, preds, measurements, measurement_sources, state, cfg)
            
            # 4. Finalize Visibility States (Candidate Promotion Logic)
            update_marker_visibility_state(measurements, measurement_sources, state, cfg)
            
            # 5. Maintain Temporal Filtering Models
            update_trackers_and_templates(frame_bgr, measurements, measurement_sources, state, cfg)

            # 3. Contextualize Visibility Logic
            lr_source = cfg.get("local_reacquire_source_name", "hsv_local_reacquire")
            strong_visible_names = {n for n, src in measurement_sources.items() if src in {"hsv", "green_local", lr_source + "_confirmed"}}
            intermediate_visible_names = {n for n, src in measurement_sources.items() if src == lr_source + "_candidate"}
            weak_visible_names = {n for n, src in measurement_sources.items() if src == "template"}
            
            visible_centers = {n: measurements[n] for n in set(measurements.keys())}
            strong_visible_centers = {n: visible_centers[n] for n in strong_visible_names}
            intermediate_visible_centers = {n: visible_centers[n] for n in intermediate_visible_names}
            weak_visible_centers = {n: visible_centers[n] for n in weak_visible_names}

            # 4. Synthesize Geometry & Temporality
            state.last_predicted_ordered_pts = validate_predictive_velocity(state, strong_visible_centers, cfg)
            final_pts, struct_status, geometry_centers, missing_name, was_ref_op, was_edge_op = estimate_robust_geometry(
                frame_bgr, strong_visible_centers, intermediate_visible_centers, visible_centers, state.last_predicted_ordered_pts, state, cfg
            )

            # 5. Composite Output
            result_bgr = render_and_occlusion(
                frame_idx, frame_bgr, overlay_bgr, final_pts, struct_status, strong_visible_names, intermediate_visible_names, state, inner_paper_mask, cfg
            )

            # 6. Metadata Layout
            if cfg["draw_debug"]:
                # Calculate Alpha for debug
                if struct_status == "all_visible":
                    ema_alpha = cfg.get("quad_ema_alpha_all_visible", 0.82)
                elif "completed" in struct_status and "intermediate" not in struct_status and "weak" not in struct_status:
                    ema_alpha = cfg.get("quad_ema_alpha_reconstructed", 0.45)
                elif "intermediate" in struct_status or "weak" in struct_status:
                    ema_alpha = cfg.get("quad_ema_alpha_intermediate", 0.28)
                else:
                    ema_alpha = 1.0 # default

                result_bgr = draw_advanced_debug_overlay(
                    result_bgr, frame_idx, preds, measurements, measurement_sources, rej_status,
                    strong_visible_names, intermediate_visible_names, weak_visible_names,
                    strong_visible_centers, intermediate_visible_centers, weak_visible_centers,
                    visible_centers, final_pts, struct_status, geometry_centers, missing_name,
                    tm_recovered_names, lr_recovered_names, was_ref_op, was_edge_op, state, cfg, ema_alpha
                )

            writer.write(result_bgr)
            if display:
                cv2.imshow("SY32 - Sequence 4 (Modular Refactor)", result_bgr)
                if cv2.waitKey(1) & 0xFF == 27: break
            frame_idx += 1

    finally:
        cap.release()
        writer.release()
        if display: cv2.destroyAllWindows()
        
    print(f"[OK] Sequence 4 Modular video completed: {output_video}")
