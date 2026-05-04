import cv2
import numpy as np
from pathlib import Path

from src.geometry.homography import compute_homography_from_overlay_to_plane, warp_overlay_to_frame, composite_overlay
from src.geometry.order_corners import centers_to_ordered_points
from src.io.video_reader import open_video_reader
from src.io.video_writer import open_video_writer
from src.io.image_loader import load_overlay_image
from src.tracking.marker_tracker import MarkerTracker
from src.detection.blob_detection import detect_markers
from src.detection.marker_refinement import estimate_green_from_yrb, detect_green_local
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
    
    "expected_corner_order": ["yellow", "red", "green", "blue"],
    "min_blob_area": 40,
    "draw_debug": False,

    "filter_alpha": 0.8, 
    "filter_beta": 0.4,
    "max_measurement_jump_px": 150.0,

    "lk_params": dict(
        winSize=(45, 45),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    ),

    # Tracking du vert simplifié
    "green_roi": 80, 
    "green_gate": 60.0,
    "green_hsv": [((30, 25, 25), (110, 255, 140))], 
    "green_v_max": 130,
    "green_min_area": 30,
    "green_min_circularity": 0.3,

    "color_ranges_hsv": {
        "red": [((0, 80, 40), (10, 255, 255)), ((170, 80, 40), (179, 255, 255))],
        "blue": [((90, 60, 30), (135, 255, 255))],
        "yellow": [((15, 60, 60), (38, 255, 255))],
    }
}

def run_seq5(input_video, overlay_image, output_video, display=False, max_frames=None):
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
    prev_gray = None
    prev_good_pts = {name: None for name in CONFIG["expected_corner_order"]}

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok or (max_frames and frame_idx >= max_frames): 
                break
            
            if CONFIG["use_undistort"]:
                frame_bgr = undistort_frame(frame_bgr, CONFIG)
            
            curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            preds = {name: t.predict() for name, t in trackers.items()}
            
            # 1. Détection des 3 couleurs de base (Rouge, Bleu, Jaune)
            raw_detections, _ = detect_markers(frame_bgr, CONFIG)
            temp_measurements = {}
            for name in ["red", "blue", "yellow"]:
                if name in raw_detections:
                    temp_measurements[name] = raw_detections[name]

            # 2. Recherche du point vert
            pred_green = preds["green"]
            green_meas = None
            
            if pred_green is None:
                # Perte du point vert -> estimation via les autres couleurs
                green_seed = estimate_green_from_yrb(
                    temp_measurements.get("yellow"), 
                    temp_measurements.get("red"), 
                    temp_measurements.get("blue")
                )
                if green_seed is not None:
                    green_meas = detect_green_local(
                        frame_bgr, center=green_seed, radius=CONFIG["green_roi"],
                        hsv_ranges=CONFIG["green_hsv"], v_max=CONFIG["green_v_max"],
                        min_area=CONFIG["green_min_area"]
                    )
            else:
                # Tracking normal
                green_meas = detect_green_local(
                    frame_bgr, center=pred_green, radius=CONFIG["green_roi"],
                    hsv_ranges=CONFIG["green_hsv"], v_max=CONFIG["green_v_max"],
                    min_area=CONFIG["green_min_area"]
                )
                # Vérification du seuil de distance
                if green_meas is not None and np.linalg.norm(green_meas - pred_green) > CONFIG["green_gate"]:
                    green_meas = None

            if green_meas is not None:
                temp_measurements["green"] = green_meas

            # 3. Optical flow (secours) & filtrage des sauts
            measurements = {}
            for name in CONFIG["expected_corner_order"]:
                m = temp_measurements.get(name)
                
                # Utilisation du flux optique si le marqueur n'est pas détecté
                if m is None and prev_gray is not None and prev_good_pts[name] is not None:
                    p0 = np.array([[prev_good_pts[name]]], dtype=np.float32)
                    p1, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **CONFIG["lk_params"])
                    if status[0][0] == 1:
                        m = p1[0][0]

                # Validation de la mesure (éviter les sauts brusques)
                if m is not None:
                    if preds[name] is not None:
                        dist = np.linalg.norm(m - preds[name])
                        if dist < CONFIG["max_measurement_jump_px"]:
                            measurements[name] = m
                    else:
                        measurements[name] = m

            # 4. Mise à jour des trackers
            current_centers = {}
            for name, t in trackers.items():
                t.update(measurements.get(name))
                if t.pos is not None:
                    current_centers[name] = t.pos
                    prev_good_pts[name] = t.pos
                    
            prev_gray = curr_gray.copy()

            # 5. Affichage & Homographie
            result_bgr = frame_bgr.copy()
            if len(current_centers) == 4:
                ordered_pts = centers_to_ordered_points(current_centers, CONFIG["expected_corner_order"])
                H = compute_homography_from_overlay_to_plane(overlay_bgr, ordered_pts)
                warped_overlay, warped_mask = warp_overlay_to_frame(overlay_bgr, frame_bgr.shape, H)
                result_bgr = composite_overlay(frame_bgr, warped_overlay, warped_mask)

            writer.write(result_bgr)
            if display:
                cv2.imshow("SY32 - Seq5 Output", result_bgr)
                if cv2.waitKey(1) & 0xFF == 27: 
                    break
            frame_idx += 1

    finally:
        cap.release()
        writer.release()
        if display: 
            cv2.destroyAllWindows()
        
    print(f"[OK] Vidéo enregistrée: {output_video}")