from __future__ import annotations

import cv2
import numpy as np


def get_overlay_source_corners(overlay_bgr: np.ndarray) -> np.ndarray:
    """
    Retourne les 4 coins de l'image source dans l'ordre:
    top-left, top-right, bottom-right, bottom-left.
    """
    h, w = overlay_bgr.shape[:2]
    return np.array(
        [
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1],
        ],
        dtype=np.float32,
    )


def compute_homography_from_overlay_to_plane(
    overlay_bgr: np.ndarray,
    dst_pts: np.ndarray,
) -> np.ndarray:
    """
    Calcule l'homographie qui projette l'image source sur le quadrilatère dst_pts.

    Parameters
    ----------
    overlay_bgr : np.ndarray
        Image à incruster.
    dst_pts : np.ndarray
        Tableau (4,2) dans l'ordre:
        [top-left, top-right, bottom-right, bottom-left]
    """
    src_pts = get_overlay_source_corners(overlay_bgr)
    dst_pts = np.asarray(dst_pts, dtype=np.float32)

    if dst_pts.shape != (4, 2):
        raise ValueError(f"dst_pts doit être de shape (4,2), reçu {dst_pts.shape}")

    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return H


def warp_overlay_to_frame(
    overlay_bgr: np.ndarray,
    frame_shape: tuple[int, int, int],
    H: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Applique l'homographie à l'image à incruster et retourne:
    - l'image warpée
    - un masque warpé (uint8, 0..255)
    """
    frame_h, frame_w = frame_shape[:2]

    warped_overlay = cv2.warpPerspective(
        overlay_bgr,
        H,
        (frame_w, frame_h),
        flags=cv2.INTER_LINEAR,
    )

    src_mask = np.full(overlay_bgr.shape[:2], 255, dtype=np.uint8)
    warped_mask = cv2.warpPerspective(
        src_mask,
        H,
        (frame_w, frame_h),
        flags=cv2.INTER_NEAREST,
    )

    return warped_overlay, warped_mask


def composite_overlay(
    frame_bgr: np.ndarray,
    warped_overlay_bgr: np.ndarray,
    warped_mask: np.ndarray,
) -> np.ndarray:
    """
    Composition simple:
    - là où le masque vaut 255, on prend l'overlay warpé
    - sinon, on garde l'image d'origine
    """
    if warped_mask.ndim != 2:
        raise ValueError("warped_mask doit être une image 1 canal")

    result = frame_bgr.copy()
    mask_bool = warped_mask > 0
    result[mask_bool] = warped_overlay_bgr[mask_bool]
    return result