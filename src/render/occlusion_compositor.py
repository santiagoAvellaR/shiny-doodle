import cv2
import numpy as np

def composite_overlay_under_foreground(
    frame_bgr: np.ndarray,
    warped_overlay: np.ndarray,
    overlay_mask: np.ndarray,
    foreground_mask: np.ndarray,
    cfg: dict
) -> np.ndarray:
    """
    Composición final optimizada con suavizado de bordes (Alpha Blending).
    Muestra el overlay solo donde está la hoja y NO hay oclusión (mano).
    """
    # 1. Erosionar ligeramente la máscara del overlay para evitar bordes ruidosos en el papel
    erosion_size = cfg.get("render_erosion_size", 3)
    if erosion_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))
        overlay_mask = cv2.erode(overlay_mask, kernel)

    # 2. Definir la visibilidad binaria (Papel Y NO Mano)
    # foreground_mask es 255 donde hay mano, bitwise_not es 255 donde NO hay mano.
    visible_mask_binary = cv2.bitwise_and(overlay_mask, cv2.bitwise_not(foreground_mask))
    
    # 3. Crear máscara Alpha suavizada (0.0 a 1.0)
    blur_size = cfg.get("render_soft_blur_size", 5)
    if blur_size > 0:
        if blur_size % 2 == 0: blur_size += 1
        alpha = cv2.GaussianBlur(visible_mask_binary.astype(np.float32), (blur_size, blur_size), 0) / 255.0
    else:
        alpha = visible_mask_binary.astype(np.float32) / 255.0
    
    alpha = np.clip(alpha, 0, 1)
    alpha_3ch = cv2.merge([alpha, alpha, alpha])
    
    # 4. Mezcla ponderada (Alpha Blending)
    # Out = Frame * (1 - Alpha) + Overlay * Alpha
    result = (frame_bgr.astype(np.float32) * (1.0 - alpha_3ch) + 
              warped_overlay.astype(np.float32) * alpha_3ch)
    
    return result.astype(np.uint8)
