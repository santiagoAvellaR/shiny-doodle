from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np

# Geometría y Homografía
from src.geometry.homography import (
    compute_homography_from_overlay_to_plane,
    warp_overlay_to_frame,
    composite_overlay,
)
from src.geometry.order_corners import (
    is_reasonable_quadrilateral,
    centers_to_ordered_points,
    polygon_area
)

# I/O
from src.io.video_reader import open_video_reader
from src.io.video_writer import open_video_writer
from src.io.image_loader import load_overlay_image

# Tracking y Filtrado
from src.tracking.marker_tracker import MarkerTracker

# Detección y Refinamiento
from src.detection.blob_detection import detect_markers
from src.detection.marker_refinement import estimate_green_from_yrb, detect_green_local

# Calibración
from src.calibration.undistort import undistort_frame

# Visualización
from src.render.debug_view import draw_debug_info


def default_seq1_config() -> dict:
    """
    Config de base pour la séquence 1.
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
        "expected_corner_order": ["yellow", "red", "green", "blue"],
        "min_blob_area": 80,
        "draw_debug": True,

        # --- PARÁMETROS DE FILTRADO Y ESTABILIDAD ---
        "filter_alpha": 0.6,   # Ganancia de posición
        "filter_beta": 0.1,    # Ganancia de velocidad
        "max_measurement_jump_px": 50.0, # Umbral de Gate para rechazar outliers
        "quad_consistency_area_tol": 0.3, # Tolerancia cambio area
        "quad_consistency_aspect_tol": 0.4, # Tolerancia cambio aspect ratio

        # --- PARÁMETROS DEL VERDE ROBUSTO (Bootstrap & Normal) ---
        "green_miss_limit": 5,           # Frames antes de volver a bootstrap
        "green_roi_bootstrap": 120,      # ROI amplia para búsqueda inicial
        "green_roi_normal": 60,          # ROI pequeña para tracking estable
        "green_gate_bootstrap": 100.0,   # Gating laxo
        "green_gate_normal": 40.0,       # Gating estricto
        "green_hsv_bootstrap": [
            ((30, 20, 20), (120, 255, 160)), # Más permisivo
        ],
        "green_hsv_normal": [
            ((35, 30, 30), (105, 255, 115)), # Más estricto
        ],
        "green_v_max_bootstrap": 150,
        "green_v_max_normal": 110,
        "green_min_area": 50,
        "green_min_circularity": 0.3,

        "confidence_weights": {
            "dist": 1.0,
            "area": 0.3,
            "circularity": 0.5
        },

        "color_ranges_hsv": {
            "red": [
                ((0, 120, 60), (10, 255, 255)),
                ((170, 120, 60), (179, 255, 255)),
            ],
            "blue": [
                ((90, 80, 40), (135, 255, 255)),
            ],
            "yellow": [
                ((15, 80, 80), (38, 255, 255)),
            ],
        },
        "draw_colors_bgr": {
            "red": (0, 0, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
            "yellow": (0, 255, 255),
            "prediction": (255, 0, 255), # Magenta
            "measurement": (0, 255, 255), # Cyan (Detección cruda)
            "roi": (120, 120, 120),       # Gris
            "status_rej": (0, 0, 255),    # Rojo si rechazado
        },
    }


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
    if fps <= 0: fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = open_video_writer(output_video, fps, width, height)

    # Inicializar trackers con Alpha-Beta
    trackers = {name: MarkerTracker(cfg["filter_alpha"], cfg["filter_beta"]) for name in cfg["expected_corner_order"]}
    
    frame_idx = 0
    green_miss_count = 0
    prev_quad_area = None
    prev_quad_aspect = None

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok: break
            if max_frames is not None and frame_idx >= max_frames: break
            
            # Calibración
            frame_bgr = undistort_frame(frame_bgr, cfg)
            
            # --- 1) PREDICCIÓN ---
            preds = {name: t.predict() for name, t in trackers.items()}
            
            # --- 2) MEDICIÓN ---
            raw_detections, _ = detect_markers(frame_bgr, cfg)
            measurements = {}
            rej_status = {name: False for name in trackers}
            
            # 2a) Detección de marcadores estándar con Gating
            for name in ["red", "blue", "yellow"]:
                if name in raw_detections:
                    m = raw_detections[name]
                    if preds[name] is not None:
                        dist = np.linalg.norm(m - preds[name])
                        if dist < cfg["max_measurement_jump_px"]:
                            measurements[name] = m
                        else:
                            rej_status[name] = True
                    else:
                        measurements[name] = m

            # 2b) Refinamiento especial para el Verde (Bootstrap vs Normal)
            # Esto corrige el retardo en Seq3 al usar una búsqueda geométrica inicial
            pred_green = preds["green"]
            green_meas = None
            
            # Decidir fase: bootstrap (si no inicializado o perdido) o normal (si ya lo seguimos)
            is_bootstrap = (trackers["green"].pos is None) or (green_miss_count >= cfg["green_miss_limit"])
            
            if is_bootstrap:
                # Semilla geométrica desde Y, R, B (Bootstrapping)
                green_seed = estimate_green_from_yrb(
                    measurements.get("yellow"), 
                    measurements.get("red"), 
                    measurements.get("blue")
                )
                if green_seed is not None:
                    green_meas = detect_green_local(
                        frame_bgr,
                        center=green_seed,
                        radius=cfg["green_roi_bootstrap"],
                        hsv_ranges=cfg["green_hsv_bootstrap"],
                        v_max=cfg["green_v_max_bootstrap"],
                        min_area=cfg["green_min_area"]
                    )
                    # Gating laxo para permitir la primera detección
                    if green_meas is not None:
                        dist = np.linalg.norm(green_meas - green_seed)
                        if dist > cfg["green_gate_bootstrap"]:
                            green_meas = None
            else:
                # Tracking normal usando la predicción del tracker Alpha-Beta
                green_meas = detect_green_local(
                    frame_bgr,
                    center=pred_green,
                    radius=cfg["green_roi_normal"],
                    hsv_ranges=cfg["green_hsv_normal"],
                    v_max=cfg["green_v_max_normal"],
                    min_area=cfg["green_min_area"]
                )
                # Gating estricto para rechazar outliers
                if green_meas is not None:
                    dist = np.linalg.norm(green_meas - pred_green)
                    if dist > cfg["green_gate_normal"]:
                        green_meas = None
            
            # Actualizar mediciones y contador de fallos
            if green_meas is not None:
                measurements["green"] = green_meas
                green_miss_count = 0
            else:
                green_miss_count += 1
            
            # --- 3) ACTUALIZACIÓN DE TRACKERS ---
            for name, t in trackers.items():
                t.update(measurements.get(name))

            # --- 4) CONSISTENCIA GEOMÉTRICA ---
            current_centers = {name: t.pos for name, t in trackers.items() if t.pos is not None}
            ordered_pts = None
            if len(current_centers) == 4:
                ordered_pts = centers_to_ordered_points(current_centers, cfg["expected_corner_order"])
                
                area = polygon_area(ordered_pts)
                side_w = np.linalg.norm(ordered_pts[0] - ordered_pts[1])
                side_h = np.linalg.norm(ordered_pts[0] - ordered_pts[3])
                aspect = side_w / side_h if side_h > 0 else 0
                
                is_consistent = True
                if prev_quad_area is not None:
                    if abs(area - prev_quad_area) / prev_quad_area > cfg["quad_consistency_area_tol"]:
                        is_consistent = False
                if prev_quad_aspect is not None:
                    if abs(aspect - prev_quad_aspect) / prev_quad_aspect > cfg["quad_consistency_aspect_tol"]:
                        is_consistent = False
                
                if is_consistent:
                    prev_quad_area = area
                    prev_quad_aspect = aspect

            # --- 5) COMPOSICIÓN Y RENDER ---
            result_bgr = frame_bgr.copy()
            if ordered_pts is not None and is_reasonable_quadrilateral(ordered_pts):
                H = compute_homography_from_overlay_to_plane(overlay_bgr, ordered_pts)
                warped_overlay, warped_mask = warp_overlay_to_frame(overlay_bgr, frame_bgr.shape, H)
                result_bgr = composite_overlay(frame_bgr, warped_overlay, warped_mask)

            # Visualización de Debug
            if cfg["draw_debug"]:
                result_bgr = draw_debug_info(
                    image_bgr=result_bgr,
                    centers=current_centers,
                    ordered_pts=ordered_pts,
                    cfg=cfg,
                    frame_idx=frame_idx,
                    predictions=preds,
                    measurements=measurements,
                    rej_status=rej_status
                )

            writer.write(result_bgr)
            if display:
                cv2.imshow("SY32 - Seq1 Modular", result_bgr)
                if cv2.waitKey(1) & 0xFF == 27: break
            frame_idx += 1

    finally:
        cap.release()
        writer.release()
        if display: cv2.destroyAllWindows()
        
    print(f"[OK] Vidéo enregistrée modularisée: {output_video}")