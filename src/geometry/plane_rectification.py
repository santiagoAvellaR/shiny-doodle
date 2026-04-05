import cv2
import numpy as np

def warp_plane_to_canonical(frame_bgr: np.ndarray, ordered_pts: np.ndarray, out_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Warps the detected quadrilateral to a fixed-size fronto-parallel view.
    """
    w, h = out_size
    dst_pts = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1],
    ], dtype=np.float32)
    
    H = cv2.getPerspectiveTransform(ordered_pts.astype(np.float32), dst_pts)
    rectified = cv2.warpPerspective(frame_bgr, H, (w, h))
    return rectified, H


def warp_mask_to_frame(mask_rectified: np.ndarray, H_frame_to_plane: np.ndarray, frame_shape: tuple) -> np.ndarray:
    """
    Reprojects a mask from the rectified plane back to the original image coordinates.
    """
    h, w = frame_shape[:2]
    H_plane_to_frame = np.linalg.inv(H_frame_to_plane)
    mask_frame = cv2.warpPerspective(mask_rectified, H_plane_to_frame, (w, h), flags=cv2.INTER_NEAREST)
    return mask_frame
