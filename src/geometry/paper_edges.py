import cv2
import numpy as np
from src.geometry.order_corners import centers_to_ordered_points


def line_from_points(p1: np.ndarray, p2: np.ndarray) -> tuple[float, float, float]:
    x1, y1 = p1
    x2, y2 = p2
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    return a, b, c


def intersect_lines(l1: tuple[float, float, float], l2: tuple[float, float, float]) -> np.ndarray | None:
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        return None
    x = (b1 * c2 - b2 * c1) / det
    y = (a2 * c1 - a1 * c2) / det
    return np.array([x, y], dtype=np.float32)


def point_line_distance(point: np.ndarray, line: tuple[float, float, float]) -> float:
    a, b, c = line
    x, y = point
    return abs(a * x + b * y + c) / np.sqrt(a**2 + b**2 + 1e-12)


def clip_roi_around_quad(
    frame_shape: tuple[int, ...], 
    ordered_pts: np.ndarray, 
    margin: int
) -> tuple[int, int, int, int]:
    h, w = frame_shape[:2]
    
    xs = ordered_pts[:, 0]
    ys = ordered_pts[:, 1]
    
    x_min = int(np.floor(np.min(xs))) - margin
    x_max = int(np.ceil(np.max(xs))) + margin
    y_min = int(np.floor(np.min(ys))) - margin
    y_max = int(np.ceil(np.max(ys))) + margin
    
    x0 = max(0, x_min)
    x1 = min(w - 1, x_max)
    y0 = max(0, y_min)
    y1 = min(h - 1, y_max)
    
    return x0, y0, x1, y1


def refine_missing_corner_with_paper_edges(
    frame_bgr: np.ndarray,
    completed_centers: dict[str, np.ndarray],
    missing_name: str,
    expected_order: list[str],
    cfg: dict,
) -> np.ndarray | None:
    try:
        ordered_pts = centers_to_ordered_points(completed_centers, expected_order)
    except KeyError:
        return None

    margin = cfg.get("paper_edge_search_margin", 45)
    x0, y0, x1, y1 = clip_roi_around_quad(frame_bgr.shape, ordered_pts, margin)
    
    if x1 - x0 < 10 or y1 - y0 < 10:
        return None
        
    roi_bgr = frame_bgr[y0:y1, x0:x1]
    
    use_white_mask = cfg.get("paper_edge_use_white_mask", True)
    
    if use_white_mask:
        hsv_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(
            hsv_roi,
            np.array([0, 0, cfg.get("paper_white_v_min", 120)], dtype=np.uint8),
            np.array([179, cfg.get("paper_white_s_max", 80), 255], dtype=np.uint8)
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        
        white_area = cv2.countNonZero(white_mask)
        if white_area < cfg.get("paper_white_min_area", 8000):
            return None
            
        roi_gray = cv2.bitwise_and(white_mask, white_mask) # basically we run edges directly on the mask
        # To run robust canny we can use the blurred mask
        roi_gray = cv2.GaussianBlur(roi_gray, (3, 3), 0)
    else:
        roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        roi_gray = cv2.GaussianBlur(roi_gray, (3, 3), 0)
    
    edges = cv2.Canny(
        roi_gray, 
        cfg.get("paper_edge_canny_low", 40), 
        cfg.get("paper_edge_canny_high", 120)
    )
    
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=cfg.get("paper_edge_hough_threshold", 35),
        minLineLength=cfg.get("paper_edge_min_line_length", 50),
        maxLineGap=cfg.get("paper_edge_max_line_gap", 15)
    )
    
    if lines is None:
        return None
        
    tl, tr, br, bl = expected_order
    
    l1_pts = None
    l2_pts = None
    
    if missing_name == tl:
        l1_pts = (completed_centers[tl], completed_centers[tr])
        l2_pts = (completed_centers[tl], completed_centers[bl])
    elif missing_name == tr:
        l1_pts = (completed_centers[tl], completed_centers[tr])
        l2_pts = (completed_centers[tr], completed_centers[br])
    elif missing_name == br:
        l1_pts = (completed_centers[tr], completed_centers[br])
        l2_pts = (completed_centers[bl], completed_centers[br])
    elif missing_name == bl:
        l1_pts = (completed_centers[tl], completed_centers[bl])
        l2_pts = (completed_centers[bl], completed_centers[br])
    else:
        return None

    baseline_l1 = line_from_points(*l1_pts)
    baseline_l2 = line_from_points(*l2_pts)
    
    best_l1 = None
    best_l2 = None
    min_dist_l1 = float('inf')
    min_dist_l2 = float('inf')
    
    for line in lines:
        lx1, ly1, lx2, ly2 = line[0]
        pt1 = np.array([lx1 + x0, ly1 + y0], dtype=np.float32)
        pt2 = np.array([lx2 + x0, ly2 + y0], dtype=np.float32)
        
        l_curr = line_from_points(pt1, pt2)
        
        dist_1 = (point_line_distance(pt1, baseline_l1) + point_line_distance(pt2, baseline_l1)) / 2.0
        dist_2 = (point_line_distance(pt1, baseline_l2) + point_line_distance(pt2, baseline_l2)) / 2.0
        
        if dist_1 < margin and dist_1 < min_dist_l1:
            min_dist_l1 = dist_1
            best_l1 = l_curr
            
        if dist_2 < margin and dist_2 < min_dist_l2:
            min_dist_l2 = dist_2
            best_l2 = l_curr
            
    if best_l1 is None or best_l2 is None:
        return None
        
    intersect = intersect_lines(best_l1, best_l2)
    if intersect is None:
        return None
        
    jump = np.linalg.norm(intersect - completed_centers[missing_name])
    if jump > cfg.get("paper_edge_max_correction_px", 40.0):
        return None
        
    return intersect
