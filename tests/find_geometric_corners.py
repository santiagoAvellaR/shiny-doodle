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
ok, frame_bgr = cap.read()
cap.release()

frame = cv2.undistort(frame_bgr, K, dist)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# The markers are dark relative to white paper, or brightly colored. 
# Let's threshold the image to find the paper, then find circles on it.
# Actually, let's just use edge detection and find closed contours
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)
kernel = np.ones((3,3), np.uint8)
edges = cv2.dilate(edges, kernel, iterations=1)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

corners = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if 40 < area < 2000:
        # Check circularity
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0: continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity > 0.5: # moderately circular
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Check HSV at center
                h, s, v = hsv[cy, cx]
                b, g, r = frame[cy, cx]
                corners.append({
                    "pos": (cx, cy),
                    "area": area,
                    "circ": circularity,
                    "hsv": (h, s, v),
                    "bgr": (b, g, r)
                })

print(f"Found {len(corners)} circular blobs")
corners.sort(key=lambda x: x["pos"][0] + x["pos"][1]) # roughly sort top-left to bottom-right
for c in corners:
    print(f"Pos {c['pos']}, Area {c['area']:.0f}, Circ {c['circ']:.2f}, HSV: {c['hsv']}, BGR: {c['bgr']}")
