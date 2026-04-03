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

lower, upper = ((22, 5, 40), (90, 255, 255))
mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
kernel = np.ones((3,3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    area = cv2.contourArea(c)
    if area < 20 or area > 5000:
        continue
        
    perimeter = cv2.arcLength(c, True)
    if perimeter == 0:
        circularity = 0
    else:
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        
    x, y, w, h = cv2.boundingRect(c)
    extent = area / (w * h)
    
    M = cv2.moments(c)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    print(f"Pos: ({cx}, {cy}), Area: {area:.1f}, Circ: {circularity:.2f}, Extent: {extent:.2f}")

