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

cx, cy = 475, 378
print("HSV around (475, 378)")
hsv_patch = hsv[cy-2:cy+3, cx-2:cx+3]
print("H:")
print(hsv_patch[:,:,0])
print("S:")
print(hsv_patch[:,:,1])
print("V:")
print(hsv_patch[:,:,2])

# Find the bounding box of pixels near (475, 378) that look "green"
roi_hsv = hsv[cy-15:cy+15, cx-15:cx+15]
h_mean = np.mean(roi_hsv[:,:,0])
s_mean = np.mean(roi_hsv[:,:,1])
v_mean = np.mean(roi_hsv[:,:,2])

print(f"ROI Mean - H: {h_mean:.1f}, S: {s_mean:.1f}, V: {v_mean:.1f}")
print("Min values in 15x15 ROI:")
print(f"H: {np.min(roi_hsv[:,:,0])}, S: {np.min(roi_hsv[:,:,1])}, V: {np.min(roi_hsv[:,:,2])}")
