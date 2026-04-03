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

masks_to_test = [
    ("Old user mask", ((22, 5, 40), (90, 255, 255))),
    ("Strict H, strict S", ((40, 15, 40), (85, 255, 255))),
    ("Medium H, loose S", ((35, 10, 40), (85, 255, 255))),
    ("Medium H, strict S", ((30, 15, 40), (85, 255, 255))),
    ("Very strict H, loose S", ((45, 10, 40), (85, 255, 255))),
]

for name, (lower, upper) in masks_to_test:
    mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if 80 <= cv2.contourArea(c) <= 5000]
    
    if not valid_contours:
        print(f"Mask '{name}': NO valid contours")
        continue
        
    largest = max(valid_contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    M = cv2.moments(largest)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    print(f"Mask '{name}': Largest Blob at ({cx}, {cy}), Area={area:.1f}")
    if len(valid_contours) > 1:
        valid_contours.remove(largest)
        second = max(valid_contours, key=cv2.contourArea)
        area2 = cv2.contourArea(second)
        M2 = cv2.moments(second)
        cx2 = int(M2["m10"] / M2["m00"])
        cy2 = int(M2["m01"] / M2["m00"])
        print(f"  -> Second largest at ({cx2}, {cy2}), Area={area2:.1f}")

