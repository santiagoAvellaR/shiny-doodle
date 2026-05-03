from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment  # 新增：引入匈牙利算法

# Geometría y Homografía
from src.geometry.homography import (
    compute_homography_from_overlay_to_plane,
    warp_overlay_to_frame,
    composite_overlay,
)
from src.geometry.order_corners import (
    is_reasonable_quadrilateral,
    polygon_area
)

# I/O
from src.io.video_reader import open_video_reader
from src.io.video_writer import open_video_writer
from src.io.image_loader import load_overlay_image

# Tracking 
from src.tracking.marker_tracker import MarkerTracker
from src.tracking.same_point_tracker import detect_same_color_markers, order_points_geometric

# Calibración
from src.calibration.undistort import undistort_frame

# Visualización
from src.render.debug_view import draw_debug_info

def default_seq2_config() -> dict:
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
        
        "expected_corner_order": ["TL", "TR", "BR", "BL"], # 左上，右上，右下，左下
        "min_blob_area": 80,
        "draw_debug": True,

        "filter_alpha": 0.6,
        "filter_beta": 0.1,
        "max_measurement_jump_px": 50.0,
        "quad_consistency_area_tol": 0.3,
        "quad_consistency_aspect_tol": 0.4,

        "target_color_hsv": [
            ((0, 120, 60), (10, 255, 255)),    # Red low section
            ((170, 120, 60), (179, 255, 255)), # Red high section
        ],
        
        "draw_colors_bgr": {
            "TL": (0, 0, 255),    # red
            "TR": (0, 255, 0),    # green
            "BR": (255, 0, 0),    # blue
            "BL": (0, 255, 255),  # yellow
            "prediction": (255, 0, 255),
            "measurement": (0, 255, 255),
            "roi": (120, 120, 120),
            "status_rej": (0, 0, 255),
        },
    }

def run_seq2(
    input_video: Path,
    overlay_image: Path,
    output_video: Path,
    display: bool = False,
    max_frames: int | None = None,
    debug: bool = False,
) -> None:
    cfg = default_seq2_config()
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
            
            if cfg["use_undistort"]:
                frame_bgr = undistort_frame(frame_bgr, cfg)
            
            # --- 1) PREDICCIÓN ---
            preds = {name: t.predict() for name, t in trackers.items()}
            
            # --- 2) MEDICIÓN  ---
            measurements = {name: None for name in cfg["expected_corner_order"]}
            rej_status = {name: False for name in trackers}
            
            raw_centroids = detect_same_color_markers(frame_bgr, cfg["target_color_hsv"], cfg["min_blob_area"])
            
            is_tracking_stable = all(preds[name] is not None for name in cfg["expected_corner_order"])
            
            if len(raw_centroids) >= 4:
                if not is_tracking_stable:
                    best_4_points = np.array(raw_centroids[:4], dtype=np.float32)
                    ordered_pts = order_points_geometric(best_4_points)
                    measurements["TL"] = ordered_pts[0]
                    measurements["TR"] = ordered_pts[1]
                    measurements["BR"] = ordered_pts[2]
                    measurements["BL"] = ordered_pts[3]
                else:
                    pred_pts = np.array([preds[name] for name in cfg["expected_corner_order"]], dtype=np.float32)
                    curr_pts = np.array(raw_centroids, dtype=np.float32)
                    
                    cost_matrix = np.zeros((4, len(curr_pts)))
                    for i in range(4):
                        for j in range(len(curr_pts)):
                            cost_matrix[i, j] = np.linalg.norm(pred_pts[i] - curr_pts[j])
                    
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    
                    for i, j in zip(row_ind, col_ind):
                        name = cfg["expected_corner_order"][i]
                        dist = cost_matrix[i, j]
                        
                        if dist < cfg["max_measurement_jump_px"]:
                            measurements[name] = curr_pts[j]
                        else:
                            rej_status[name] = True
            
            # --- 3) ACTUALIZACIÓN DE TRACKERS ---
            for name, t in trackers.items():
                t.update(measurements.get(name))

            # --- 4) CONSISTENCIA GEOMÉTRICA ---
            current_centers = {name: t.pos for name, t in trackers.items() if t.pos is not None}
            final_ordered_pts = None
            
            if len(current_centers) == 4:
                final_ordered_pts = np.array([current_centers[n] for n in cfg["expected_corner_order"]], dtype=np.float32)
                
                area = polygon_area(final_ordered_pts)
                side_w = np.linalg.norm(final_ordered_pts[0] - final_ordered_pts[1])
                side_h = np.linalg.norm(final_ordered_pts[0] - final_ordered_pts[3])
                aspect = side_w / side_h if side_h > 0 else 0
                
                is_consistent = True
                if prev_quad_area is not None:
                    if abs(area - prev_quad_area) / prev_quad_area > cfg["quad_consistency_area_tol"]: is_consistent = False
                if prev_quad_aspect is not None:
                    if abs(aspect - prev_quad_aspect) / prev_quad_aspect > cfg["quad_consistency_aspect_tol"]: is_consistent = False
                
                if is_consistent:
                    prev_quad_area, prev_quad_aspect = area, aspect
                else:
                    final_ordered_pts = None

            # --- 5) COMPOSICIÓN Y RENDER ---
            result_bgr = frame_bgr.copy()
            if final_ordered_pts is not None and is_reasonable_quadrilateral(final_ordered_pts):
                H = compute_homography_from_overlay_to_plane(overlay_bgr, final_ordered_pts)
                warped_overlay, warped_mask = warp_overlay_to_frame(overlay_bgr, frame_bgr.shape, H)
                result_bgr = composite_overlay(frame_bgr, warped_overlay, warped_mask)

            if cfg["draw_debug"]:
                result_bgr = draw_debug_info(
                    image_bgr=result_bgr, centers=current_centers, ordered_pts=final_ordered_pts,
                    cfg=cfg, frame_idx=frame_idx, predictions=preds, measurements=measurements, rej_status=rej_status
                )

            writer.write(result_bgr)
            if display:
                cv2.imshow("SY32 - Sequence 2", result_bgr)
                if cv2.waitKey(1) & 0xFF == 27: break
            frame_idx += 1

    finally:
        cap.release()
        writer.release()
        if display: cv2.destroyAllWindows()
        
    print(f"[OK] Vidéo enregistrée: {output_video}")