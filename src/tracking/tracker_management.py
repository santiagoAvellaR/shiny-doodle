import numpy as np
from src.pipelines.pipeline_state import PipelineState
from src.detection.template_matching import update_marker_templates


def update_trackers_and_templates(
    frame_bgr: np.ndarray,
    measurements: dict[str, np.ndarray],
    measurement_sources: dict[str, str],
    state: PipelineState,
    cfg: dict
) -> None:
    """Safe state update locking weak templates from degrading filters or history."""
    tracker_update_measurements = {}
    for name, pt in measurements.items():
        src = measurement_sources.get(name)
        if src in {"hsv", "green_local"}:
            tracker_update_measurements[name] = pt
        elif src == "template" and cfg.get("template_updates_tracker", False):
            tracker_update_measurements[name] = pt
            
    for name, t in state.trackers.items():
        t.update(tracker_update_measurements.get(name))
        
    for name in state.trackers:
        src = measurement_sources.get(name)
        if src in {"hsv", "green_local"} or (src == "template" and cfg.get("template_resets_miss_count", False)):
            state.marker_miss_counts[name] = 0
        else:
            state.marker_miss_counts[name] += 1
            
    # Update Graphic Templates STRICTLY with real/strong observations
    template_update_measurements = {
        name: pt for name, pt in measurements.items()
        if measurement_sources.get(name) in {"hsv", "green_local"}
    }
    state.marker_templates = update_marker_templates(
        frame_bgr,
        template_update_measurements,
        state.marker_templates,
        cfg,
    )
