import cv2
import numpy as np

K = np.array([
    [533.75781056, 0.0, 386.78762246],
    [0.0, 534.74578856, 275.71106165],
    [0.0, 0.0, 1.0],
], dtype=np.float32)
dist = np.array([-3.33535276e-01, 1.65338810e-01, -2.90030682e-04, -3.97059918e-04, -4.70631813e-02], dtype=np.float32)

cap = cv2.VideoCapture("data/raw/seq1.mp4")
ok, frame_bgr = cap.read()
cap.release()

frame = cv2.undistort(frame_bgr, K, dist)

# Let's get yellow, red, blue using the existing functions to see where they are
from src.pipelines.pipeline_seq1 import default_seq1_config, detect_markers
cfg = default_seq1_config()
centers, _ = detect_markers(frame, cfg)

print("Centers found:", centers.keys())
if "yellow" in centers and "red" in centers and "blue" in centers:
    tl = centers["yellow"]
    tr = centers["red"]
    bl = centers["blue"]
    predict_br = tr + bl - tl
    print("Predicted BR:", predict_br)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    x, y = int(predict_br[0]), int(predict_br[1])
    
    # search window 60x60 around predicted point
    S = 30
    roi = gray[max(0, y-S):min(gray.shape[0], y+S), max(0, x-S):min(gray.shape[1], x+S)]
    
    # goodFeaturesToTrack
    corners = cv2.goodFeaturesToTrack(roi, maxCorners=1, qualityLevel=0.1, minDistance=10)
    if corners is not None:
        cx = corners[0][0][0] + max(0, x-S)
        cy = corners[0][0][1] + max(0, y-S)
        print("Good Feature found at:", cx, cy)
    else:
        print("No good features found in ROI")
else:
    print("Missing some of Y, R, B")
