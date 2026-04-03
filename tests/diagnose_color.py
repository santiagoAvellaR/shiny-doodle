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

# Save the frame
cv2.imwrite("results/debug_frame_50.jpg", frame)

# Let's use the exact color ranges defined in pipeline_seq1.py
ranges_hsv = {
    "red": [
        ((0, 120, 70), (10, 255, 255)),
        ((170, 120, 70), (179, 255, 255)),
    ],
    "green": [
        ((35, 40, 40), (90, 255, 255)), # The modified one
    ],
    "blue": [
        ((90, 80, 40), (130, 255, 255)),
    ],
    "yellow": [
        ((15, 80, 80), (35, 255, 255)),
    ]
}

for color, rngs in ranges_hsv.items():
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in rngs:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8)))
    
    kernel = np.ones((5,5), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Color: {color}")
    for c in contours:
        area = cv2.contourArea(c)
        if area > 40:
            M = cv2.moments(c)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            bgr_val = frame[cy, cx]
            hsv_val = hsv[cy, cx]
            print(f"  Pos: ({cx},{cy}), Area: {area:.1f}, BGR: {bgr_val}, HSV: {hsv_val}")

