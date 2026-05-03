import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

def order_points_geometric(pts: np.ndarray) -> np.ndarray:
   
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # TL
    rect[2] = pts[np.argmax(s)]  # BR

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # TR
    rect[3] = pts[np.argmax(diff)]  # BL

    return rect

# Bernoulli algorithm
def detect_same_color_markers(frame_bgr: np.ndarray, hsv_ranges: list, min_area: int) -> list[np.ndarray]:

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    
    for (lower, upper) in hsv_ranges:
        lower_bound = np.array(lower, dtype=np.uint8)
        upper_bound = np.array(upper, dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower_bound, upper_bound))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    centroids = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append(np.array([cx, cy], dtype=np.float32))
                
    return centroids


class SamePointTracker:

    def __init__(self):
        self.prev_pts = None 

    def _order_points_init(self, pts):

        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def update(self, current_centers):
        curr_pts = np.array(current_centers, dtype="float32")
        
        if self.prev_pts is None:
            self.prev_pts = self._order_points_init(curr_pts)
            return self.prev_pts

        cost_matrix = np.zeros((4, len(curr_pts)))
        for i in range(4):
            for j in range(len(curr_pts)):
                cost_matrix[i, j] = np.linalg.norm(self.prev_pts[i] - curr_pts[j])

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        ordered_pts = curr_pts[col_ind]
        
        self.prev_pts = ordered_pts
        
        return ordered_pts