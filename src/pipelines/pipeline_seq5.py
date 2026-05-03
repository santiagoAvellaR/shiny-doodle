from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np

# Geometría y Homografía
from src.geometry.homography import (
    compute_homography_from_overlay_to_plane,
    warp_overlay_to_frame,
    composite_overlay,
)
from src.geometry.order_corners import (
    centers_to_ordered_points,
    polygon_area
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

# Calibración
from src.calibration.undistort import undistort_frame

# Visualización
from src.render.debug_view import draw_debug_info


def default_seq5_config() -> dict:
    return {
        "use_undistort": True,
        "camera_matrix": np.array(
            [[533.75781056, 0.0, 386.78762246],
             [0.0, 534.74578856, 275.71106165],
             [0.0, 0.0, 1.0]], dtype=np.float32
        ),
        "dist_coeffs": np.array(
            [-3.33535276e-01, 1.65338810e-01, -2.90030682e-04, -3.97059918e-04, -4.70631813e-02],
            dtype=np.float32
        ),
        
        "expected_corner_order": ["yellow", "red", "green", "blue"],
        "min_blob_area": 40,
        "draw_debug": True,

        "filter_alpha": 0.8, 
        "filter_beta": 0.4,
        "max_measurement_jump_px": 150.0,
        
        "quad_consistency_area_tol": 0.8,
        "quad_consistency_aspect_tol": 0.8,

        "lk_params": dict(
            winSize=(45, 45),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        ),

        "green_miss_limit": 5,
        "green_roi_bootstrap": 120,
        "green_roi_normal": 60,
        "green_gate_bootstrap": 100.0,
        "green_gate_normal": 40.0,
        "green_hsv_bootstrap": [((30, 20, 20), (120, 255, 160))],
        "green_hsv_normal": [((35, 30, 30), (105, 255, 115))],
        "green_v_max_bootstrap": 150,
        "green_v_max_normal": 110,
        "green_min_area": 30,
        "green_min_circularity": 0.3,

        "color_ranges_hsv": {
            "red": [((0, 80, 40), (10, 255, 255)), ((170, 80, 40), (179, 255, 255))],
            "blue": [((90, 60, 30), (135, 255, 255))],
            "yellow": [((15, 60, 60), (38, 255, 255))],
        },
        "draw_colors_bgr": {
            "red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0), "yellow": (0, 255, 255),
            "prediction": (255, 0, 255), "measurement": (0, 255, 255), "roi": (120, 120, 120), "status_rej": (0, 0, 255),
        },
    }


def run_seq5(
    input_video: Path,
    overlay_image: Path,
    output_video: Path,
    display: bool = False,
    max_frames: int | None = None,
    debug: bool = False,
) -> None:
    cfg = default_seq5_config()
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
    green_miss_count = 0
    prev_quad_area = None
    prev_quad_aspect = None

    prev_gray = None
    prev_good_pts = {name: None for name in cfg["expected_corner_order"]}

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok: break
            if max_frames is not None and frame_idx >= max_frames: break
            
            if cfg["use_undistort"]:
                frame_bgr = undistort_frame(frame_bgr, cfg)
            
            curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            preds = {name: t.predict() for name, t in trackers.items()}
            
            raw_detections, _ = detect_markers(frame_bgr, cfg)
            temp_measurements = {}
            rej_status = {name: False for name in trackers}
            
            for name in ["red", "blue", "yellow"]:
                if name in raw_detections:
                    temp_measurements[name] = raw_detections[name]

            pred_green = preds["green"]
            is_bootstrap = (trackers["green"].pos is None) or (green_miss_count >= cfg["green_miss_limit"])
            green_meas = None
            
            if is_bootstrap:
                green_seed = estimate_green_from_yrb(
                    temp_measurements.get("yellow"), temp_measurements.get("red"), temp_measurements.get("blue")
                )
                if green_seed is not None:
                    green_meas = detect_green_local(
                        frame_bgr, center=green_seed, radius=cfg["green_roi_bootstrap"],
                        hsv_ranges=cfg["green_hsv_bootstrap"], v_max=cfg["green_v_max_bootstrap"],
                        min_area=cfg["green_min_area"]
                    )
                    if green_meas is not None and np.linalg.norm(green_meas - green_seed) > cfg["green_gate_bootstrap"]:
                        green_meas = None
            else:
                green_meas = detect_green_local(
                    frame_bgr, center=pred_green, radius=cfg["green_roi_normal"],
                    hsv_ranges=cfg["green_hsv_normal"], v_max=cfg["green_v_max_normal"],
                    min_area=cfg["green_min_area"]
                )
                if green_meas is not None and np.linalg.norm(green_meas - pred_green) > cfg["green_gate_normal"]:
                    green_meas = None

            if green_meas is not None:
                temp_measurements["green"] = green_meas
                green_miss_count = 0
            else:
                green_miss_count += 1

            measurements = {}
            for name in cfg["expected_corner_order"]:
                m = temp_measurements.get(name)
                
                if m is None and prev_gray is not None and prev_good_pts[name] is not None:
                    p0 = np.array([[prev_good_pts[name]]], dtype=np.float32)
                    p1, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **cfg["lk_params"])
                    if status[0][0] == 1:
                        m = p1[0][0]

                if m is not None:
                    if preds[name] is not None:
                        dist = np.linalg.norm(m - preds[name])
                        if dist < cfg["max_measurement_jump_px"]:
                            measurements[name] = m
                        else:
                            rej_status[name] = True
                    else:
                        measurements[name] = m

            for name, t in trackers.items():
                t.update(measurements.get(name))
                
            current_centers = {}
            for name, t in trackers.items():
                if t.pos is not None:
                    current_centers[name] = t.pos
                    prev_good_pts[name] = t.pos
            prev_gray = curr_gray.copy()

            ordered_pts = None
            if len(current_centers) == 4 and sum(1 for v in measurements.values() if v is not None) == 4:
                ordered_pts = centers_to_ordered_points(current_centers, cfg["expected_corner_order"])
                
                area = polygon_area(ordered_pts)
                side_w = np.linalg.norm(ordered_pts[0] - ordered_pts[1])
                side_h = np.linalg.norm(ordered_pts[0] - ordered_pts[3])
                aspect = side_w / side_h if side_h > 0 else 0
                
                is_consistent = True
                if prev_quad_area is not None and abs(area - prev_quad_area) / prev_quad_area > cfg["quad_consistency_area_tol"]:
                    is_consistent = False
                if prev_quad_aspect is not None and abs(aspect - prev_quad_aspect) / prev_quad_aspect > cfg["quad_consistency_aspect_tol"]:
                    is_consistent = False
                
                if is_consistent:
                    prev_quad_area, prev_quad_aspect = area, aspect
                else:
                    ordered_pts = None

            result_bgr = frame_bgr.copy()
            if ordered_pts is not None:
                H = compute_homography_from_overlay_to_plane(overlay_bgr, ordered_pts)
                warped_overlay, warped_mask = warp_overlay_to_frame(overlay_bgr, frame_bgr.shape, H)
                result_bgr = composite_overlay(frame_bgr, warped_overlay, warped_mask)

            if cfg["draw_debug"]:
                result_bgr = draw_debug_info(
                    image_bgr=result_bgr, centers=current_centers, ordered_pts=ordered_pts,
                    cfg=cfg, frame_idx=frame_idx, predictions=preds, measurements=measurements, rej_status=rej_status
                )

            writer.write(result_bgr)
            if display:
                cv2.imshow("SY32 - Seq5 Simple Core", result_bgr)
                if cv2.waitKey(1) & 0xFF == 27: break
            frame_idx += 1

    finally:
        cap.release()
        writer.release()
        if display: cv2.destroyAllWindows()
        
    print(f"[OK] Vidéo enregistrée: {output_video}")