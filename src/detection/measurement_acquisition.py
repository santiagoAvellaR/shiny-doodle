import numpy as np
from src.detection.blob_detection import detect_markers
from src.detection.marker_refinement import estimate_green_from_yrb, detect_green_local
from src.pipelines.pipeline_state import PipelineState

def acquire_strong_measurements(
    frame_bgr: np.ndarray,
    preds: dict[str, np.ndarray | None],
    state: PipelineState,
    cfg: dict
) -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, bool]]:
    """Handles raw optical detection of HSV markers and constructs strong inputs."""
    raw_detections, _ = detect_markers(frame_bgr, cfg)
    
    measurements = {}
    measurement_sources = {}
    rej_status = {name: False for name in state.trackers}
    
    # 1. RBY Checks
    for name in ["red", "blue", "yellow"]:
        if name in raw_detections:
            m = raw_detections[name]
            if preds[name] is not None:
                dist = np.linalg.norm(m - preds[name])
                
                # Check Reacquisition Gates
                if state.marker_miss_counts[name] >= cfg.get("hard_reacquire_after_miss", 10):
                    measurements[name] = m
                    measurement_sources[name] = "hsv"
                else:
                    if state.marker_miss_counts[name] >= cfg.get("reacquire_after_miss", 3):
                        gate = cfg.get("reacquire_gate_px", 180.0)
                    else:
                        gate = cfg.get("max_measurement_jump_px", 70.0)

                    if dist < gate:
                        measurements[name] = m
                        measurement_sources[name] = "hsv"
                    else:
                        rej_status[name] = True
            else:
                measurements[name] = m
                measurement_sources[name] = "hsv"

    # 2. Green local recovery
    pred_green = preds.get("green")
    green_meas = None
    is_bootstrap = False
    
    if "green" in state.trackers:
        is_bootstrap = (state.trackers["green"].pos is None) or (state.marker_miss_counts.get("green", 0) >= cfg.get("green_miss_limit", 15))
    
    if is_bootstrap:
        green_seed = estimate_green_from_yrb(
            measurements.get("yellow"),
            measurements.get("red"),
            measurements.get("blue")
        )
        if green_seed is not None:
            green_meas = detect_green_local(
                frame_bgr, green_seed, cfg.get("green_roi_bootstrap", 150),
                cfg["green_hsv_bootstrap"], cfg.get("green_min_area", 200), cfg.get("green_v_max_bootstrap", 255)
            )
            if green_meas is not None and np.linalg.norm(green_meas - green_seed) > cfg.get("green_gate_bootstrap", 100):
                green_meas = None
    elif pred_green is not None:
        green_meas = detect_green_local(
            frame_bgr, pred_green, cfg.get("green_roi_normal", 60),
            cfg["green_hsv_normal"], cfg.get("green_min_area", 200), cfg.get("green_v_max_normal", 180)
        )
        if green_meas is not None and np.linalg.norm(green_meas - pred_green) > cfg.get("green_gate_normal", 45.0):
            green_meas = None

    if green_meas is not None:
        measurements["green"] = green_meas
        measurement_sources["green"] = "green_local"
            
    return measurements, measurement_sources, rej_status
