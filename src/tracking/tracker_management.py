import numpy as np
from src.pipelines.pipeline_state import PipelineState
from src.detection.template_matching import update_marker_templates


def update_marker_visibility_state(
    measurements: dict[str, np.ndarray],
    measurement_sources: dict[str, str],
    state: PipelineState,
    cfg: dict
) -> None:
    """Centralized responsibility for miss counts, reacquire confirmation, and source promotion."""
    confirm_thresh = cfg.get("local_reacquire_confirm_frames", 2)
    lr_source = cfg.get("local_reacquire_source_name", "hsv_local_reacquire")
    
    for name in cfg.get("expected_corner_order", ["yellow", "red", "green", "blue"]):
        src = measurement_sources.get(name)
        
        # 1. Strong Sources
        if src in {"hsv", "green_local"}:
            state.reacquire_confirm_counts[name] = confirm_thresh
            state.marker_miss_counts[name] = 0
            
        # 2. Local Reacquire Sources (Candidate vs Confirmed)
        elif src == lr_source or src == lr_source + "_candidate":
            state.reacquire_confirm_counts[name] += 1
            state.marker_miss_counts[name] = 0
            
            # Promotion logic
            if state.reacquire_confirm_counts[name] >= confirm_thresh:
                measurement_sources[name] = lr_source + "_confirmed"
            else:
                # Force candidate name if not yet confirmed
                measurement_sources[name] = lr_source + "_candidate"
                
        # 3. Confirmed Local Reacquire (already promoted in logic above or previous frame)
        elif src == lr_source + "_confirmed":
            state.reacquire_confirm_counts[name] = confirm_thresh # Cap
            state.marker_miss_counts[name] = 0
            
        # 4. Template Matching
        elif src == "template":
            if cfg.get("template_resets_miss_count", False):
                state.marker_miss_counts[name] = 0
            else:
                state.marker_miss_counts[name] += 1
            state.reacquire_confirm_counts[name] = 0
            
        # 5. Miss
        else:
            state.marker_miss_counts[name] += 1
            state.reacquire_confirm_counts[name] = 0


def update_trackers_and_templates(
    frame_bgr: np.ndarray,
    measurements: dict[str, np.ndarray],
    measurement_sources: dict[str, str],
    state: PipelineState,
    cfg: dict
) -> None:
    lr_source = cfg.get("local_reacquire_source_name", "hsv_local_reacquire")
    
    # We only update trackers with strong or confirmed measurement sources
    tracker_update_measurements = {}
    for name, pt in measurements.items():
        src = measurement_sources.get(name)
        if src in {"hsv", "green_local", lr_source + "_confirmed"}:
            tracker_update_measurements[name] = pt
        elif src == "template" and cfg.get("template_updates_tracker", False):
            tracker_update_measurements[name] = pt
            
    for name, t in state.trackers.items():
        src = measurement_sources.get(name)
        pt = tracker_update_measurements.get(name)
        
        # Snap if it was a long miss (hard reacquire)
        if src == lr_source + "_confirmed" and state.marker_miss_counts.get(name, 0) >= cfg.get("local_reacquire_hard_after_miss", 5):
            t.reset(pt)
        else:
            t.update(pt)
            
    # Update Graphic Templates STRICTLY with real/strong observations
    template_update_measurements = {
        name: pt for name, pt in measurements.items()
        if measurement_sources.get(name) in {"hsv", "green_local", lr_source + "_confirmed"}
    }
    state.marker_templates = update_marker_templates(
        frame_bgr,
        template_update_measurements,
        state.marker_templates,
        cfg,
    )
