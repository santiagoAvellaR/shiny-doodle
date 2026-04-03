from __future__ import annotations
import numpy as np

def complete_with_previous(
    current_centers: dict[str, np.ndarray],
    previous_centers: dict[str, np.ndarray] | None,
    expected_colors: list[str],
) -> dict[str, np.ndarray]:
    completed: dict[str, np.ndarray] = {}
    for color in expected_colors:
        if color in current_centers:
            completed[color] = current_centers[color]
        elif previous_centers is not None and color in previous_centers:
            completed[color] = previous_centers[color]
    return completed


def smooth_centers(
    current_centers: dict[str, np.ndarray],
    previous_centers: dict[str, np.ndarray] | None,
    alpha: float,
) -> dict[str, np.ndarray]:
    if previous_centers is None:
        return current_centers
    smoothed: dict[str, np.ndarray] = {}
    for color, cur in current_centers.items():
        if color in previous_centers:
            prev = previous_centers[color]
            smoothed[color] = alpha * prev + (1.0 - alpha) * cur
        else:
            smoothed[color] = cur
    return smoothed
