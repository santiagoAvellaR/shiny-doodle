from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.geometry.homography import (
    compute_homography_from_overlay_to_plane,
    warp_overlay_to_frame,
    composite_overlay,
)
from src.geometry.order_corners import is_reasonable_quadrilateral


def default_seq1_config() -> dict:
    """
    Config de base pour la séquence 1.
    IMPORTANT:
    - les bornes HSV sont un point de départ
    - il faudra probablement les ajuster avec vos vraies vidéos
    - expected_corner_order doit refléter l'ordre réel des couleurs:
      [top-left, top-right, bottom-right, bottom-left]
    """
    return {
        "use_undistort": True,
        "camera_matrix": np.array(
            [
                [533.75781056, 0.0, 386.78762246],
                [0.0, 534.74578856, 275.71106165],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        "dist_coeffs": np.array(
            [-3.33535276e-01, 1.65338810e-01, -2.90030682e-04, -3.97059918e-04, -4.70631813e-02],
            dtype=np.float32,
        ),
        # À ADAPTER selon la vraie disposition dans votre séquence
        "expected_corner_order": ["yellow", "red", "green", "blue"],
        "min_blob_area": 80,
        "smooth_alpha": 0.70,   # plus grand = plus lisse / plus de mémoire
        "draw_debug": True,
        "color_ranges_hsv": {
            "red": [
                ((0, 120, 70), (10, 255, 255)),
                ((170, 120, 70), (179, 255, 255)),
            ],
            # El verde se extrae geométricamente
            # "green": [
            #     ((22, 5, 40), (90, 255, 255)),
            # ],
            "blue": [
                ((90, 80, 40), (130, 255, 255)),
            ],
            "yellow": [
                ((15, 80, 80), (35, 255, 255)),
            ],
        },
        "draw_colors_bgr": {
            "red": (0, 0, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
            "yellow": (0, 255, 255),
        },
    }


def undistort_frame(frame_bgr: np.ndarray, cfg: dict) -> np.ndarray:
    if not cfg["use_undistort"]:
        return frame_bgr

    K = cfg["camera_matrix"]
    dist = cfg["dist_coeffs"]
    return cv2.undistort(frame_bgr, K, dist)


def build_mask_from_hsv_ranges(hsv: np.ndarray, ranges: list[tuple[tuple[int, int, int], tuple[int, int, int]]]) -> np.ndarray:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for lower, upper in ranges:
        lower_np = np.array(lower, dtype=np.uint8)
        upper_np = np.array(upper, dtype=np.uint8)
        mask_part = cv2.inRange(hsv, lower_np, upper_np)
        mask = cv2.bitwise_or(mask, mask_part)

    # Nettoyage morphologique léger
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def calculate_centroid_from_mask(mask: np.ndarray, min_area: int) -> list[np.ndarray]:
    """
    Trouve les centres des blobs circulaires dans un masque binaire.
    Retourne une liste de centres [x, y].
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_centers = []
    for c in contours:
        area = cv2.contourArea(c)
        if min_area <= area <= 5000:
            perimeter = cv2.arcLength(c, True)
            if perimeter > 0:
                # La circularité est 4*pi*A / P^2 (1.0 pour un cercle parfait)
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if circularity > 0.5: # On est assez strict
                    M = cv2.moments(c)
                    if M["m00"] > 0:
                        cx = M["m10"] / M["m00"]
                        cy = M["m01"] / M["m00"]
                        # On raffine encore un peu avec le cercle englobant
                        (bx, by), radius = cv2.minEnclosingCircle(c)
                        # On fait une moyenne ou on prend le centre du cercle englobant
                        # Le centre du cercle englobant est souvent plus stable si le blob est bruité
                        final_x = (cx + bx) / 2.0
                        final_y = (cy + by) / 2.0
                        valid_centers.append(np.array([final_x, final_y], dtype=np.float32))

    return valid_centers


def detect_markers(frame_bgr: np.ndarray, cfg: dict) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    centers: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    candidates_dict: dict[str, list[np.ndarray]] = {}

    for color_name, ranges in cfg["color_ranges_hsv"].items():
        mask = build_mask_from_hsv_ranges(hsv, ranges)
        masks[color_name] = mask
        
        candidates = calculate_centroid_from_mask(mask, cfg["min_blob_area"])
        if candidates:
            candidates_dict[color_name] = candidates
            
    for color in ["yellow", "red", "blue"]:
        if color in candidates_dict and candidates_dict[color]:
            # On prend le premier car ce sont des couleurs uniques
            centers[color] = candidates_dict[color][0]

    return centers, masks


def complete_with_previous(
    current_centers: dict[str, np.ndarray],
    previous_centers: dict[str, np.ndarray] | None,
    expected_colors: list[str],
) -> dict[str, np.ndarray]:
    """
    Si une couleur manque, on peut réutiliser sa position précédente.
    Pour la seq1 ce n'est normalement pas nécessaire, mais c'est utile
    pour éviter de casser tout le pipeline à la moindre détection ratée.
    """
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


def draw_debug_info(
    image_bgr: np.ndarray,
    centers: dict[str, np.ndarray],
    ordered_pts: np.ndarray | None,
    cfg: dict,
    frame_idx: int,
) -> np.ndarray:
    debug = image_bgr.copy()

    for color_name, pt in centers.items():
        color_bgr = cfg["draw_colors_bgr"].get(color_name, (255, 255, 255))
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(debug, (x, y), 8, color_bgr, -1)
        cv2.putText(
            debug,
            color_name,
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color_bgr,
            2,
            cv2.LINE_AA,
        )

    if ordered_pts is not None:
        quad = ordered_pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(debug, [quad], isClosed=True, color=(255, 255, 255), thickness=2)

        labels = ["TL", "TR", "BR", "BL"]
        for label, pt in zip(labels, ordered_pts):
            x, y = int(round(pt[0])), int(round(pt[1]))
            cv2.putText(
                debug,
                label,
                (x + 10, y + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    cv2.putText(
        debug,
        f"frame: {frame_idx}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return debug


def load_overlay_image(path: Path) -> np.ndarray:
    overlay = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if overlay is None:
        raise FileNotFoundError(f"Impossible de charger l'image d'overlay: {path}")
    return overlay


def open_video_reader(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Impossible d'ouvrir la vidéo: {path}")
    return cap


def open_video_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Impossible de créer la vidéo de sortie: {path}")
    return writer


def run_seq1(
    input_video: Path,
    overlay_image: Path,
    output_video: Path,
    display: bool = False,
    max_frames: int | None = None,
) -> None:
    cfg = default_seq1_config()

    overlay_bgr = load_overlay_image(overlay_image)
    cap = open_video_reader(input_video)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = open_video_writer(output_video, fps, width, height)

    prev_centers: dict[str, np.ndarray] | None = None
    expected_colors = cfg["expected_corner_order"]

    frame_idx = 0
    
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    green_tracker_pt = None
    prev_gray = None

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if max_frames is not None and frame_idx >= max_frames:
                break

            frame_bgr = undistort_frame(frame_bgr, cfg)
            frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            current_centers, _ = detect_markers(frame_bgr, cfg)
            
            # --- OPTICAL FLOW Y REFINAMIENTO COLOR/GEOMETRÍA ---
            # Predicción por Optical Flow (LK)
            lk_ok = False
            if green_tracker_pt is not None and prev_gray is not None:
                p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, green_tracker_pt, None, **lk_params)
                if st[0][0] == 1:
                    green_tracker_pt = p1
                    lk_ok = True
            
            # Predicción geométrica (segunda opinión)
            geom_pt = None
            if "yellow" in current_centers and "red" in current_centers and "blue" in current_centers:
                geom_pt = current_centers["red"] + current_centers["blue"] - current_centers["yellow"]

            # Refinamiento final del VERDE usando color en ROI (más robusto que GFTT)
            # Definimos el punto de partida para la búsqueda
            search_base = None
            if lk_ok:
                search_base = green_tracker_pt[0].flatten()
            elif geom_pt is not None:
                search_base = geom_pt

            if search_base is not None:
                x, y = int(search_base[0]), int(search_base[1])
                S = 40  # Ventana de búsqueda algo más amplia si falla LK
                h_min, h_max = max(0, y-S), min(frame_bgr.shape[0], y+S)
                w_min, w_max = max(0, x-S), min(frame_bgr.shape[1], x+S)
                
                roi_bgr = frame_bgr[h_min:h_max, w_min:w_max]
                roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
                
                # Mascaras de "verde" específicamente ajustadas para seq1 en esa zona
                # Si el usuario dice que el H de la mesa es similar, jugamos con Saturación y Valor
                green_ranges = [((30, 40, 40), (90, 255, 255))]
                mask_roi = build_mask_from_hsv_ranges(roi_hsv, green_ranges)
                
                # Buscamos el centroide del círculo más redondo en la ROI
                roi_centers = calculate_centroid_from_mask(mask_roi, cfg["min_blob_area"] // 2)
                
                if roi_centers:
                    # Si hay varios, cogemos el más cercano a search_base
                    best_roi_pt = None
                    min_d = float("inf")
                    for rpt in roi_centers:
                        # rpt es relativo a la ROI
                        abs_rpt = rpt + np.array([w_min, h_min])
                        d = np.linalg.norm(abs_rpt - search_base)
                        if d < min_d:
                            min_d = d
                            best_roi_pt = abs_rpt
                    
                    current_centers["green"] = best_roi_pt
                    green_tracker_pt = np.array([[best_roi_pt[0], best_roi_pt[1]]], dtype=np.float32)
                elif lk_ok:
                    # Si no hay blob verde pero LK funcionó, confiamos en LK
                    current_centers["green"] = search_base
                elif geom_pt is not None:
                    # Último recurso: geometría pura sin refinamiento si falla todo
                    current_centers["green"] = geom_pt
                
            prev_gray = frame_gray
            completed_centers = complete_with_previous(
                current_centers=current_centers,
                previous_centers=prev_centers,
                expected_colors=expected_colors,
            )

            ordered_pts = None
            result_bgr = frame_bgr.copy()

            if len(completed_centers) == 4:
                completed_centers = smooth_centers(
                    current_centers=completed_centers,
                    previous_centers=prev_centers,
                    alpha=cfg["smooth_alpha"],
                )

                ordered_pts = centers_to_ordered_points(
                    centers=completed_centers,
                    expected_corner_order=expected_colors,
                )

                if is_reasonable_quadrilateral(ordered_pts, min_area=500.0):
                    H = compute_homography_from_overlay_to_plane(
                        overlay_bgr=overlay_bgr,
                        dst_pts=ordered_pts,
                    )
                    warped_overlay, warped_mask = warp_overlay_to_frame(
                        overlay_bgr=overlay_bgr,
                        frame_shape=frame_bgr.shape,
                        H=H,
                    )
                    result_bgr = composite_overlay(
                        frame_bgr=frame_bgr,
                        warped_overlay_bgr=warped_overlay,
                        warped_mask=warped_mask,
                    )
                    prev_centers = completed_centers

            if cfg["draw_debug"]:
                result_bgr = draw_debug_info(
                    image_bgr=result_bgr,
                    centers=completed_centers,
                    ordered_pts=ordered_pts,
                    cfg=cfg,
                    frame_idx=frame_idx,
                )

            writer.write(result_bgr)

            if display:
                cv2.imshow("SY32 - Seq1", result_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break

            frame_idx += 1

    finally:
        cap.release()
        writer.release()
        if display:
            cv2.destroyAllWindows()

    print(f"[OK] Vidéo enregistrée dans: {output_video}")