import cv2
import numpy as np

def build_inner_paper_mask(size: tuple[int, int], margin: int = 20) -> np.ndarray:
    """
    Creates a mask with a margin to ignore boundary errors.
    """
    w, h = size
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (margin, margin), (w - margin, h - margin), 255, -1)
    return mask


def keep_largest_component(mask: np.ndarray, min_area: int) -> np.ndarray:
    """
    Retiene solo las componentes conexas que superen min_area.
    Útil para eliminar ruido tipo 'sal y pimienta'.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask)
    
    res = np.zeros_like(mask)
    # Empezamos desde 1 para ignorar el fondo (label 0)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            res[labels == i] = 255
    return res


def segment_foreground_on_rectified(current_bgr: np.ndarray, reference_bgr: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Detecta el primer plano (objeto oclusor) comparando el plano actual con la referencia limpia.
    Usa el espacio de color Lab para ser más robusto a sombras y variaciones de brillo.
    """
    # Suavizado previo para reducir ruido de alta frecuencia
    cur_blur = cv2.GaussianBlur(current_bgr, (5, 5), 0)
    ref_blur = cv2.GaussianBlur(reference_bgr, (5, 5), 0)
    
    # Cambio a Lab
    cur_lab = cv2.cvtColor(cur_blur, cv2.COLOR_BGR2Lab)
    ref_lab = cv2.cvtColor(ref_blur, cv2.COLOR_BGR2Lab)
    
    diff = cv2.absdiff(cur_lab, ref_lab)
    
    # Combinación ponderada de canales Lab
    # Damos más peso a 'a' y 'b' (cromáticos) para ignorar cambios de L (luminosidad/sombras)
    weights = cfg.get("fg_lab_weights", [1.0, 2.5, 2.5])
    diff_w = (diff[:,:,0].astype(np.float32) * weights[0] + 
              diff[:,:,1].astype(np.float32) * weights[1] + 
              diff[:,:,2].astype(np.float32) * weights[2]) / sum(weights)
    diff_w = np.clip(diff_w, 0, 255).astype(np.uint8)
    
    # Umbralización
    _, mask = cv2.threshold(diff_w, cfg["fg_diff_thresh"], 255, cv2.THRESH_BINARY)
    
    # Operaciones morfológicas para limpiar la máscara
    k_size = cfg.get("fg_morph_kernel_size", 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Eliminar componentes pequeñas
    mask = keep_largest_component(mask, cfg.get("fg_min_blob_area", 800))
    
    return mask
