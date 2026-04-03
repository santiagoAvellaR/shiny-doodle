import cv2
import numpy as np

K = np.array([
    [533.75781056, 0.0, 386.78762246],
    [0.0, 534.74578856, 275.71106165],
    [0.0, 0.0, 1.0],
], dtype=np.float32)
dist = np.array([-3.33535276e-01, 1.65338810e-01, -2.90030682e-04, -3.97059918e-04, -4.70631813e-02], dtype=np.float32)

cap = cv2.VideoCapture("data/raw/seq1.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
ok, frame = cap.read()
cap.release()

frame = cv2.undistort(frame, K, dist)
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Looking at 4 specific regions:
# TL ~ (230, 325)
# TR ~ (430, 250)
# BL ~ (295, 450)
# BR ~ (470, 380)

regions = {
    "TL": (230, 325),
    "TR": (430, 250),
    "BL": (295, 450),
    "BR": (470, 380),
    "CENTER": (360, 350)
}

for name, (cx_est, cy_est) in regions.items():
    # extract 40x40 region
    x1, x2 = max(0, cx_est-20), min(frame.shape[1], cx_est+20)
    y1, y2 = max(0, cy_est-20), min(frame.shape[0], cy_est+20)
    roi_bgr = frame[y1:y2, x1:x2]
    roi_hsv = hsv[y1:y2, x1:x2]
    
    # Calculate properties of the most saturated part of this ROI
    # Or just the darkest part if it's a dark color
    S = roi_hsv[:,:,1]
    # find the peak saturation pixel
    sv, max_s, min_loc, max_loc = cv2.minMaxLoc(S)
    px, py = max_loc
    
    peak_hsv = roi_hsv[py, px]
    peak_bgr = roi_bgr[py, px]
    print(f"Region {name}: Peak Saturation at ({x1+px}, {y1+py}) -> S={max_s}, HSV={peak_hsv}, BGR={peak_bgr}")
    
    # Mean of the highly saturated pixels (>30)
    mask_s = cv2.inRange(S, max(max_s - 30, 10), 255)
    mean_hsv = cv2.mean(roi_hsv, mask=mask_s)
    print(f"  Mean of saturated pixels: H={mean_hsv[0]:.1f}, S={mean_hsv[1]:.1f}, V={mean_hsv[2]:.1f}")

