import cv2
import numpy as np
import sys
import os

# Añadir el directorio actual al path para importar src
sys.path.append(os.getcwd())

from src.pipelines.pipeline_seq1 import (
    default_seq1_config, 
    detect_markers, 
    calculate_centroid_from_mask, 
    build_mask_from_hsv_ranges
)

def test_on_frame(frame_idx):
    K = np.array([[533.75781056, 0.0, 386.78772246], [0.0, 534.74578856, 275.71106165], [0.0, 0.0, 1.0]], dtype=np.float32)
    dist = np.array([-3.33535276e-01, 1.65338810e-01, -2.90030682e-04, -3.97059918e-04, -4.70631813e-02], dtype=np.float32)

    cap = cv2.VideoCapture('data/raw/seq1.mp4')
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame_bgr = cap.read()
    cap.release()

    if not ok:
        print(f"Error loading frame {frame_idx}")
        return

    frame = cv2.undistort(frame_bgr, K, dist)
    cfg = default_seq1_config()
    
    # 1. Detect Y, R, B
    centers, masks = detect_markers(frame, cfg)
    print(f"\n--- Frame {frame_idx} ---")
    for c in ["yellow", "red", "blue"]:
        if c in centers:
            print(f"{c}: {centers[c]}")
        else:
            print(f"{c}: MISSING")

    # 2. Predict Green (BR)
    if all(c in centers for c in ["yellow", "red", "blue"]):
        tl, tr, bl = centers['yellow'], centers['red'], centers['blue']
        geom_pt = tr + bl - tl
        print(f"Geometric prediction for green: {geom_pt}")
        
        # 3. ROI search for green
        x, y = int(geom_pt[0]), int(geom_pt[1])
        S = 40
        h_min, h_max = max(0, y-S), min(frame.shape[0], y+S)
        w_min, w_max = max(0, x-S), min(frame.shape[1], x+S)
        
        roi_bgr = frame[h_min:h_max, w_min:w_max]
        roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        
        # Color ranges adapted to the green marker in seq1
        green_ranges = [((30, 40, 40), (90, 255, 255))]
        mask_roi = build_mask_from_hsv_ranges(roi_hsv, green_ranges)
        
        roi_centers = calculate_centroid_from_mask(mask_roi, cfg["min_blob_area"] // 2)
        
        if roi_centers:
            # Pick closest to geometric prediction
            best_roi_pt = None
            min_d = float("inf")
            for rpt in roi_centers:
                abs_rpt = rpt + np.array([w_min, h_min])
                d = np.linalg.norm(abs_rpt - geom_pt)
                if d < min_d:
                    min_d = d
                    best_roi_pt = abs_rpt
            print(f"Green Refined: {best_roi_pt} (Found {len(roi_centers)} blobs in ROI)")
            
            # Debug save ROI
            debug_roi = roi_bgr.copy()
            for rpt in roi_centers:
                cv2.circle(debug_roi, (int(rpt[0]), int(rpt[1])), 5, (0, 255, 0), -1)
            cv2.imwrite(f"temp_debug/roi_green_f{frame_idx}.jpg", debug_roi)
            cv2.imwrite(f"temp_debug/mask_green_f{frame_idx}.jpg", mask_roi)
        else:
            print("Green NOT found in ROI")
    else:
        print("Cannot predict green, missing other markers")

if __name__ == "__main__":
    import os
    os.makedirs("temp_debug", exist_ok=True)
    test_on_frame(120)
    test_on_frame(150)
