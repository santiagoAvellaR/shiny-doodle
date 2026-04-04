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
from src.detection.marker_refinement import refine_green_two_stage

# Calibración
from src.calibration.undistort import undistort_frame

# Visualización
from src.render.debug_view import draw_debug_info


def default_seq3_config() -> dict:
    """
    Configuración base extensible para la secuencia 3 (con oclusión).
    """
    return {
        "use_undistort": True,
        "camera_matrix": np.array(
            [[533.75781056, 0.0, 386.78762246], [0.0, 534.74578856, 275.71106165], [0.0, 0.0, 1.0]],
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
        "filter_alpha": 0.6,
        "filter_beta": 0.1,
        "max_measurement_jump_px": 50.0,
        "quad_consistency_area_tol": 0.3,
        "quad_consistency_aspect_tol": 0.4,

        # --- PARÁMETROS DEL VERDE ROBUSTO ---
        "green_roi_size_coarse": 80,
        "green_roi_size_fine": 40,
        "green_v_max": 110,
        "green_min_circularity": 0.35,
        "green_hsv_ranges": [((35, 20, 20), (105, 255, 120))],
        "prediction_weights": {"flow": 0.6, "geom": 0.4},
        "confidence_weights": {"dist": 1.0, "area": 0.3, "circularity": 0.5},

        # --- PARÁMETROS DE OCLUSIÓN (SEQ 3) ---
        "plane_size": (700, 500),         # (ancho, alto) de la hoja rectificada
        "occ_diff_threshold": 35,         # Umbral de diferencia para detectar mano (0-255)
        "occ_min_area": 500,              # Área mínima de oclusión para evitar ruido
        "occ_ref_max_frames": 15,         # Cantidad de frames para el buffer de referencia
        "occ_marker_excl_radius": 25,     # Radio para ignorar los stickers en el plano
        
        # --- COLORES PARA DETECCIÓN ESTÁNDAR ---
        "color_ranges_hsv": {
            "red": [((0, 120, 60), (10, 255, 255)), ((170, 120, 60), (179, 255, 255))],
            "blue": [((90, 80, 40), (135, 255, 255))],
            "yellow": [((15, 80, 80), (38, 255, 255))],
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



# ==============================================================================
# FUNCIONES ESPECÍFICAS DE OCLUSIÓN (SECUENCIA 3)
# ==============================================================================

def build_plane_mask(frame_shape: tuple, quad_pts: np.ndarray) -> np.ndarray:
    """Crea una máscara binaria (uint8) del cuadrilátero en la imagen original."""
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, quad_pts.astype(np.int32), 255)
    return mask


def rectify_plane_view(frame: np.ndarray, quad_pts: np.ndarray, plane_size: tuple) -> np.ndarray:
    """Warp del frame a una vista frontal (700x500 por defecto) del plano de la hoja."""
    w, h = plane_size
    dst_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    H_rect = cv2.getPerspectiveTransform(quad_pts.astype(np.float32), dst_pts)
    rectified = cv2.warpPerspective(frame, H_rect, (w, h))
    return rectified, H_rect


def update_reference_plane_model(buffer: list[np.ndarray], current_view: np.ndarray, max_size: int = 15) -> np.ndarray | None:
    """Acumula frames en un buffer y devuelve la mediana temporal como referencia limpia."""
    buffer.append(current_view.astype(np.float32))
    if len(buffer) > max_size:
        buffer.pop(0)
    
    if len(buffer) < 5: # Mínimo necesario para una mediana decente
        return None
    
    # Calculamos la mediana de los frames acumulados
    median_ref = np.median(np.stack(buffer), axis=0).astype(np.uint8)
    return median_ref


def create_marker_exclusion_mask_in_plane(plane_size: tuple, radius: int = 25) -> np.ndarray:
    """Mascara para ignorar los 4 stickers en el plano rectificado."""
    w, h = plane_size
    mask = np.ones((h, w), dtype=np.uint8) * 255
    corners = [(0, 0), (w, 0), (w, h), (0, h)]
    for c in corners:
        cv2.circle(mask, c, radius, 0, -1) # 0 = ignorar
    return mask


def detect_occlusion_on_plane(plane_view: np.ndarray, reference_plane: np.ndarray, threshold: int = 35) -> np.ndarray:
    """Detecta la mano por diferencia absoluta vs referencia limpia."""
    # Diferencia en escala de grises es rápida y efectiva para papel blanco
    gray_curr = cv2.cvtColor(plane_view, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(reference_plane, cv2.COLOR_BGR2GRAY)
    
    diff = cv2.absdiff(gray_curr, gray_ref)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    return mask


def smooth_occlusion_mask(mask: np.ndarray) -> np.ndarray:
    """Limpia ruido con operaciones morfológicas."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # Eliminar puntos sueltos
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # Cerrar huecos
    return mask


def project_plane_mask_to_frame(mask_plane: np.ndarray, H_rect: np.ndarray, frame_shape: tuple) -> np.ndarray:
    """Reproyecta la máscara de oclusión del plano rectificado al frame original."""
    h, w = frame_shape[:2]
    # Necesitamos la inversa de la homografía de rectificación
    H_inv = np.linalg.inv(H_rect)
    mask_frame = cv2.warpPerspective(mask_plane, H_inv, (w, h), flags=cv2.INTER_NEAREST)
    return mask_frame


def compose_overlay_under_occluder(
    frame: np.ndarray,
    warped_overlay: np.ndarray,
    warped_mask: np.ndarray,
    occlusion_mask_frame: np.ndarray
) -> np.ndarray:
    """
    Composición final: El overlay solo aparece donde está la hoja Y NO hay mano.
    """
    result = frame.copy()
    
    # 1. El overlay debe estar dentro de la hoja (warped_mask)
    # 2. El overlay NO debe estar donde hay oclusión (occlusion_mask_frame == 0)
    # Nota: occlusion_mask_frame es uint8 (255 mano, 0 fondo)
    
    visible_overlay_mask = (warped_mask > 0) & (occlusion_mask_frame == 0)
    
    result[visible_overlay_mask] = warped_overlay[visible_overlay_mask]
    return result


# ==============================================================================
# PIPELINE PRINCIPAL (RUN_SEQ3)
# ==============================================================================

def run_seq3(
    input_video: Path,
    overlay_image: Path,
    output_video: Path,
    display: bool = False,
    max_frames: int | None = None,
) -> None:
    cfg = default_seq3_config()
    overlay_bgr = load_overlay_image(overlay_image)
    cap = open_video_reader(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = open_video_writer(output_video, fps, width, height)

    trackers = {name: MarkerTracker(cfg["filter_alpha"], cfg["filter_beta"]) for name in cfg["expected_corner_order"]}
    
    # BUFFER PARA MODELO DE REFERENCIA (SEQ 3)
    ref_buffer = []
    clean_reference = None
    marker_excl_mask = create_marker_exclusion_mask_in_plane(cfg["plane_size"], cfg["occ_marker_excl_radius"])
    
    frame_idx = 0
    prev_quad_area = None
    prev_quad_aspect = None

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok or (max_frames and frame_idx >= max_frames):
                break
            
            frame_raw = undistort_frame(frame_bgr, cfg)
            
            # --- 1) PREDICCIÓN ---
            preds = {name: t.predict() for name, t in trackers.items()}
            
            # --- 2) MEDICIÓN ---
            raw_detections, _ = detect_markers(frame_raw, cfg)
            measurements = {}
            rej_status = {name: False for name in trackers}
            
            # Stickers estándar (R, B, Y)
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

            # Sticker Verde (Refinamiento local)
            p0_green = preds["green"]
            if p0_green is None and "yellow" in measurements and "red" in measurements and "blue" in measurements:
                p0_green = measurements["red"] + measurements["blue"] - measurements["yellow"]
            
            if p0_green is not None:
                res_green = refine_green_two_stage(frame_raw, p0_green, cfg)
                if res_green:
                    measurements["green"] = res_green[0]
            
            # --- 3) ACTUALIZACIÓN ---
            for name, t in trackers.items():
                t.update(measurements.get(name))

            # --- 4) GEOMETRÍA ---
            curr_centers = {name: t.pos for name, t in trackers.items() if t.pos is not None}
            ordered_pts = None
            if len(curr_centers) == 4:
                ordered_pts = centers_to_ordered_points(curr_centers, cfg["expected_corner_order"])
                area = polygon_area(ordered_pts)
                side_w = np.linalg.norm(ordered_pts[0] - ordered_pts[1])
                side_h = np.linalg.norm(ordered_pts[0] - ordered_pts[3])
                aspect = side_w / side_h if side_h > 0 else 0
                
                is_consistent = True
                if prev_quad_area and abs(area - prev_quad_area) / prev_quad_area > cfg["quad_consistency_area_tol"]:
                    is_consistent = False
                if prev_quad_aspect and abs(aspect - prev_quad_aspect) / prev_quad_aspect > cfg["quad_consistency_aspect_tol"]:
                    is_consistent = False
                
                if is_consistent:
                    prev_quad_area, prev_quad_aspect = area, aspect
                else:
                    ordered_pts = None # Descartar cuadrilátero inconsistente

            # --- 5) OCLUSIÓN Y COMPOSICIÓN (BLOQUE SEQ 3) ---
            result_bgr = frame_raw.copy()
            if ordered_pts is not None and is_reasonable_quadrilateral(ordered_pts):
                # Calcular Homografía de incrustación (Overlay -> Frame)
                H_overlay = compute_homography_from_overlay_to_plane(overlay_bgr, ordered_pts)
                warped_overlay, warped_mask = warp_overlay_to_frame(overlay_bgr, frame_raw.shape, H_overlay)
                
                # --- LOGICA DE OCLUSION ---
                # A) Rectificación a vista frontal
                rect_view, H_rect = rectify_plane_view(frame_raw, ordered_pts, cfg["plane_size"])
                
                # B) Actualizar modelo de referencia (buscando hojas sin mano)
                # Opcional: Solo añadir al buffer si la oclusión es pequeña o en los primeros frames
                clean_reference = update_reference_plane_model(ref_buffer, rect_view, cfg["occ_ref_max_frames"])
                
                if clean_reference is not None:
                    # C) Detectar oclusión
                    occ_mask_plane = detect_occlusion_on_plane(rect_view, clean_reference, cfg["occ_diff_threshold"])
                    # Excluir stickers
                    occ_mask_plane = cv2.bitwise_and(occ_mask_plane, marker_excl_mask)
                    # D) Suavizar
                    occ_mask_plane = smooth_occlusion_mask(occ_mask_plane)
                    # E) Re-proyectar al frame
                    hand_mask_frame = project_plane_mask_to_frame(occ_mask_plane, H_rect, frame_raw.shape)
                    
                    # F) Composición: Mano sobre Overlay
                    result_bgr = compose_overlay_under_occluder(frame_raw, warped_overlay, warped_mask, hand_mask_frame)
                else:
                    # Si no hay referencia aún, usar composición normal
                    result_bgr = composite_overlay(frame_raw, warped_overlay, warped_mask)
            
            # Debug visual
            if cfg["draw_debug"]:
                # Mostramos si la referencia está lista en una esquina
                if clean_reference is not None:
                    ref_small = cv2.resize(clean_reference, (140, 100))
                    result_bgr[10:110, 10:150] = ref_small
                    cv2.putText(result_bgr, "Ref Model", (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

                result_bgr = draw_debug_info(result_bgr, curr_centers, ordered_pts, cfg, frame_idx, preds, measurements, rej_status)

            writer.write(result_bgr)
            if display:
                cv2.imshow("SY32 - Seq3 Occlusion", result_bgr)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            frame_idx += 1

    finally:
        cap.release()
        writer.release()
        if display: cv2.destroyAllWindows()
        
    print(f"[OK] Vidéo Seq3 complétée: {output_video}")