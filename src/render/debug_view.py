import cv2
import numpy as np

def draw_debug_info(
    image_bgr: np.ndarray,
    centers: dict[str, np.ndarray],
    ordered_pts: np.ndarray | None,
    cfg: dict,
    frame_idx: int,
    predictions: dict[str, np.ndarray] | None = None,
    measurements: dict[str, np.ndarray] | None = None,
    rej_status: dict[str, bool] | None = None,
    rois: dict[str, tuple[int, int, int, int]] | None = None,
) -> np.ndarray:
    debug = image_bgr.copy()

    if rois:
        for name, box in rois.items():
            cv2.rectangle(debug, (box[0], box[1]), (box[2], box[3]), cfg["draw_colors_bgr"]["roi"], 1)

    if predictions:
        for name, p in predictions.items():
            if p is not None:
                px, py = int(round(p[0])), int(round(p[1]))
                cv2.drawMarker(debug, (px, py), cfg["draw_colors_bgr"]["prediction"], cv2.MARKER_CROSS, 8, 1)

    if measurements:
        for name, m in measurements.items():
            if m is not None:
                mx, my = int(round(m[0])), int(round(m[1]))
                cv2.circle(debug, (mx, my), 3, cfg["draw_colors_bgr"]["measurement"], -1)

    for color_name, pt in centers.items():
        color_bgr = cfg["draw_colors_bgr"].get(color_name, (255, 255, 255))
        if rej_status and rej_status.get(color_name):
            color_bgr = cfg["draw_colors_bgr"]["status_rej"]
            
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(debug, (x, y), 8, color_bgr, 2)
        cv2.drawMarker(debug, (x, y), color_bgr, cv2.MARKER_TILTED_CROSS, 5, 2)
        cv2.putText(debug, color_name, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1, cv2.LINE_AA)

    if ordered_pts is not None:
        quad = ordered_pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(debug, [quad], isClosed=True, color=(255, 255, 255), thickness=1)

    cv2.putText(debug, f"frame: {frame_idx}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return debug
