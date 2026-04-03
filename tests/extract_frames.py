import cv2
import numpy as np
import os

K = np.array([
    [533.75781056, 0.0, 386.78762246],
    [0.0, 534.74578856, 275.71106165],
    [0.0, 0.0, 1.0],
], dtype=np.float32)
dist = np.array([-3.33535276e-01, 1.65338810e-01, -2.90030682e-04, -3.97059918e-04, -4.70631813e-02], dtype=np.float32)

cap = cv2.VideoCapture("data/raw/seq1.mp4")

os.makedirs("results/debug", exist_ok=True)
for i in [0, 50, 100, 150]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, frame = cap.read()
    if not ok: break
    frame = cv2.undistort(frame, K, dist)
    cv2.imwrite(f"results/debug/frame_{i}.jpg", frame)

cap.release()
print("Saved frames")
