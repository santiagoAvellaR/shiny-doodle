import cv2
import numpy as np
from pathlib import Path
from scipy.optimize import linear_sum_assignment

# Imports
from src.geometry.homography import compute_homography_from_overlay_to_plane, warp_overlay_to_frame, composite_overlay
from src.geometry.order_corners import is_reasonable_quadrilateral, polygon_area
from src.io.video_reader import open_video_reader
from src.io.video_writer import open_video_writer
from src.io.image_loader import load_overlay_image
from src.tracking.marker_tracker import MarkerTracker
from src.tracking.same_point_tracker import detect_same_color_markers, order_points_geometric
from src.calibration.undistort import undistort_frame

# Configuration globale
CONFIG = {
    "use_undistort": True,
    
    # Paramètres caméra (arrondis)
    "camera_matrix": np.array([
        [534.0, 0.0, 387.0],
        [0.0, 535.0, 276.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32),
    
    # Coeffs de distorsion
    "dist_coeffs": np.array([-0.33, 0.16, 0.0, 0.0, -0.05], dtype=np.float32),
    
    "expected_corner_order": ["TL", "TR", "BR", "BL"],
    "min_blob_area": 80,
    "draw_debug": False,

    "filter_alpha": 0.6,
    "filter_beta": 0.1,
    "max_measurement_jump_px": 50.0,
    "quad_consistency_area_tol": 0.3,
    "quad_consistency_aspect_tol": 0.4,

    # Plages de couleurs (rouge)
    "target_color_hsv": [
        ((0, 120, 60), (10, 255, 255)),
        ((170, 120, 60), (179, 255, 255)),
    ],
    
    "draw_colors_bgr": {
        "TL": (0, 0, 255),
        "TR": (0, 255, 0),
        "BR": (255, 0, 0),
        "BL": (0, 255, 255),
    }
}

def run_seq2(input_video, overlay_image, output_video, display=False, max_frames=None, debug=False):
    overlay_bgr = load_overlay_image(overlay_image)
    cap = open_video_reader(input_video)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    Path(output_video).parent.mkdir(parents=True, exist_ok=True)
    writer = open_video_writer(output_video, fps, width, height)

    trackers = {name: MarkerTracker(CONFIG["filter_alpha"], CONFIG["filter_beta"]) for name in CONFIG["expected_corner_order"]}
    
    frame_idx = 0
    prev_quad_area = None
    prev_quad_aspect = None

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok or (max_frames and frame_idx >= max_frames): 
                break
            
            if CONFIG["use_undistort"]:
                frame_bgr = undistort_frame(frame_bgr, CONFIG)
            
            # 1. Prediction
            preds = {name: t.predict() for name, t in trackers.items()}
            
            # 2. Mesure et detection
            measurements = {name: None for name in CONFIG["expected_corner_order"]}
            
            raw_centroids = detect_same_color_markers(frame_bgr, CONFIG["target_color_hsv"], CONFIG["min_blob_area"])
            is_tracking_stable = all(preds[name] is not None for name in CONFIG["expected_corner_order"])
            
            if len(raw_centroids) >= 4:
                if not is_tracking_stable:
                    # Initialisation géométrique
                    best_4_points = np.array(raw_centroids[:4], dtype=np.float32)
                    ordered_pts = order_points_geometric(best_4_points)
                    measurements["TL"] = ordered_pts[0]
                    measurements["TR"] = ordered_pts[1]
                    measurements["BR"] = ordered_pts[2]
                    measurements["BL"] = ordered_pts[3]
                else:
                    # Tracking normal (Hungarian algorithm pour le matching)
                    pred_pts = np.array([preds[name] for name in CONFIG["expected_corner_order"]], dtype=np.float32)
                    curr_pts = np.array(raw_centroids, dtype=np.float32)
                    
                    cost_matrix = np.zeros((4, len(curr_pts)))
                    for i in range(4):
                        for j in range(len(curr_pts)):
                            cost_matrix[i, j] = np.linalg.norm(pred_pts[i] - curr_pts[j])
                    
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    
                    for i, j in zip(row_ind, col_ind):
                        name = CONFIG["expected_corner_order"][i]
                        dist = cost_matrix[i, j]
                        
                        if dist < CONFIG["max_measurement_jump_px"]:
                            measurements[name] = curr_pts[j]
            
            # 3. Update des trackers
            for name, t in trackers.items():
                t.update(measurements.get(name))

            # 4. Vérification de la consistance géométrique (aire et aspect ratio)
            current_centers = {name: t.pos for name, t in trackers.items() if t.pos is not None}
            final_ordered_pts = None
            
            if len(current_centers) == 4:
                final_ordered_pts = np.array([current_centers[n] for n in CONFIG["expected_corner_order"]], dtype=np.float32)
                
                area = polygon_area(final_ordered_pts)
                side_w = np.linalg.norm(final_ordered_pts[0] - final_ordered_pts[1])
                side_h = np.linalg.norm(final_ordered_pts[0] - final_ordered_pts[3])
                aspect = side_w / side_h if side_h > 0 else 0
                
                is_consistent = True
                if prev_quad_area is not None:
                    if abs(area - prev_quad_area) / prev_quad_area > CONFIG["quad_consistency_area_tol"]: 
                        is_consistent = False
                if prev_quad_aspect is not None:
                    if abs(aspect - prev_quad_aspect) / prev_quad_aspect > CONFIG["quad_consistency_aspect_tol"]: 
                        is_consistent = False
                
                if is_consistent:
                    prev_quad_area, prev_quad_aspect = area, aspect
                else:
                    final_ordered_pts = None

            # 5. Affichage et homographie
            result_bgr = frame_bgr.copy()
            if final_ordered_pts is not None and is_reasonable_quadrilateral(final_ordered_pts):
                H = compute_homography_from_overlay_to_plane(overlay_bgr, final_ordered_pts)
                warped_overlay, warped_mask = warp_overlay_to_frame(overlay_bgr, frame_bgr.shape, H)
                result_bgr = composite_overlay(frame_bgr, warped_overlay, warped_mask)

            writer.write(result_bgr)
            if display:
                cv2.imshow("SY32 - Sequence 2", result_bgr)
                if cv2.waitKey(1) & 0xFF == 27: 
                    break
            frame_idx += 1

    finally:
        cap.release()
        writer.release()
        if display: 
            cv2.destroyAllWindows()
        
    print(f"[OK] Vidéo enregistrée: {output_video}")