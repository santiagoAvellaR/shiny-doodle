from __future__ import annotations

import numpy as np


def order_points_clockwise(pts: np.ndarray) -> np.ndarray:
    """
    Ordonne 4 points 2D dans l'ordre:
    top-left, top-right, bottom-right, bottom-left

    Méthode classique basée sur:
    - somme x+y
    - différence x-y
    """
    pts = np.asarray(pts, dtype=np.float32)

    if pts.shape != (4, 2):
        raise ValueError(f"pts doit être de shape (4,2), reçu {pts.shape}")

    ordered = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    ordered[0] = pts[np.argmin(s)]  # top-left
    ordered[2] = pts[np.argmax(s)]  # bottom-right
    ordered[1] = pts[np.argmin(d)]  # top-right
    ordered[3] = pts[np.argmax(d)]  # bottom-left

    return ordered


def polygon_area(pts: np.ndarray) -> float:
    """
    Aire signée d'un polygone 2D.
    """
    pts = np.asarray(pts, dtype=np.float32)
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def is_reasonable_quadrilateral(
    pts: np.ndarray,
    min_area: float = 500.0,
) -> bool:
    """
    Vérification géométrique simple.
    """
    pts = np.asarray(pts, dtype=np.float32)

    if pts.shape != (4, 2):
        return False

    area = polygon_area(pts)
    return area >= min_area


def centers_to_ordered_points(
    centers: dict[str, np.ndarray],
    expected_corner_order: list[str],
) -> np.ndarray:
    """
    Construit [top-left, top-right, bottom-right, bottom-left]
    en utilisant directement l'identité des couleurs.
    """
    pts = np.array([centers[color] for color in expected_corner_order], dtype=np.float32)
    return pts


def is_convex_ordered_quad(pts: np.ndarray) -> bool:
    pts = np.asarray(pts, dtype=np.float32)

    if pts.shape != (4, 2):
        return False

    crosses = []
    for i in range(4):
        p0 = pts[i]
        p1 = pts[(i + 1) % 4]
        p2 = pts[(i + 2) % 4]

        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        crosses.append(cross)

    crosses = np.array(crosses)

    if np.any(np.abs(crosses) < 1e-3):
        return False

    return np.all(crosses > 0) or np.all(crosses < 0)