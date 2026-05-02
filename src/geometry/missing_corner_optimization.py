import cv2
import numpy as np

from src.geometry.order_corners import centers_to_ordered_points, is_reasonable_quadrilateral, is_convex_ordered_quad
from src.geometry.plane_rectification import warp_plane_to_canonical


def optimize_missing_corner_by_reference_alignment(
    frame_bgr: np.ndarray,
    clean_reference: np.ndarray,
    completed_centers: dict[str, np.ndarray],
    missing_name: str,
    expected_order: list[str],
    cfg: dict,
) -> np.ndarray | None:
    """
    Refines the missing corner by local search.
    It tries candidate positions around the current estimated missing point.
    For each candidate:
        - build candidate quad,
        - warp current frame to canonical plane,
        - compare rectified view with clean_reference,
        - choose the candidate with lowest photometric error.
    Returns refined_point or None.
    """
    if clean_reference is None:
        return None

    try:
        ordered_pts = centers_to_ordered_points(completed_centers, expected_order)
    except KeyError:
        return None

    if missing_name not in expected_order:
        return None
        
    missing_idx = expected_order.index(missing_name)
    current_point = ordered_pts[missing_idx].copy()
    
    canon_size = cfg.get("canonical_plane_size", (600, 400))
    use_lab = cfg.get("missing_corner_refine_use_lab", True)
    downscale = cfg.get("missing_corner_refine_downscale", 2)
    
    # Pre-process reference
    if use_lab:
        ref_proc = cv2.cvtColor(clean_reference, cv2.COLOR_BGR2LAB).astype(np.float32)
    else:
        ref_proc = cv2.cvtColor(clean_reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
    if downscale > 1:
        new_size = (canon_size[0] // downscale, canon_size[1] // downscale)
        ref_proc = cv2.resize(ref_proc, new_size, interpolation=cv2.INTER_AREA)

    def evaluate_point(pt: np.ndarray) -> tuple[float, np.ndarray]:
        cand_pts = ordered_pts.copy()
        cand_pts[missing_idx] = pt
        
        if not is_reasonable_quadrilateral(cand_pts):
            return float('inf'), cand_pts
            
        if not is_convex_ordered_quad(cand_pts):
            return float('inf'), cand_pts
            
        rect_view, _ = warp_plane_to_canonical(frame_bgr, cand_pts, canon_size)
        
        if use_lab:
            view_proc = cv2.cvtColor(rect_view, cv2.COLOR_BGR2LAB).astype(np.float32)
        else:
            view_proc = cv2.cvtColor(rect_view, cv2.COLOR_BGR2GRAY).astype(np.float32)
            
        if downscale > 1:
            view_proc = cv2.resize(view_proc, new_size, interpolation=cv2.INTER_AREA)

        # Photometric error difference
        diff = cv2.absdiff(view_proc, ref_proc)
        
        # Robust metric: ignore pixels with excessive differences (e.g. hand occlusions)
        # We can take median or sum of clipped pixels below 85th percentile
        if len(diff.shape) > 2:
            diff = np.mean(diff, axis=2)
            
        diff_sorted = np.sort(diff.flatten())
        p85_idx = int(len(diff_sorted) * 0.85)
        score = np.mean(diff_sorted[:p85_idx])
        return score, cand_pts

    # 1. Evaluate baseline
    best_score, _ = evaluate_point(current_point)
    best_point = current_point.copy()
    baseline_score = best_score
    
    if baseline_score == float('inf'):
        return None

    # Helper function for grid search
    def run_grid_search(center_pt, r_limit, step):
        nonlocal best_score, best_point
        found_better = False
        
        dxs = np.arange(-r_limit, r_limit + 1, step)
        dys = np.arange(-r_limit, r_limit + 1, step)
        
        for dy in dys:
            for dx in dxs:
                if dx == 0 and dy == 0:
                    continue
                cand_pt = center_pt + np.array([dx, dy], dtype=np.float32)
                score, _ = evaluate_point(cand_pt)
                if score < best_score:
                    best_score = score
                    best_point = cand_pt
                    found_better = True
        return found_better

    # Coarse Search
    coarse_radius = cfg.get("missing_corner_refine_coarse_radius", 35)
    coarse_step = cfg.get("missing_corner_refine_coarse_step", 7)
    run_grid_search(best_point, coarse_radius, coarse_step)
    
    # Fine Search
    fine_radius = cfg.get("missing_corner_refine_fine_radius", 8)
    fine_step = cfg.get("missing_corner_refine_fine_step", 2)
    run_grid_search(best_point, fine_radius, fine_step)

    # Validate shifts and improvements
    shift_px = np.linalg.norm(best_point - current_point)
    max_shift = cfg.get("missing_corner_refine_max_shift_px", 45.0)
    if shift_px > max_shift:
        return None
        
    improvement = (baseline_score - best_score) / (baseline_score + 1e-6)
    if improvement < cfg.get("missing_corner_refine_max_improvement", 0.02):
        return None

    return best_point
