from dataclasses import dataclass, field
import numpy as np
from collections import deque
from src.tracking.marker_tracker import MarkerTracker

@dataclass
class PipelineState:
    """Centralizes persistent tracking histories to reduce function parameter bloat."""
    trackers: dict[str, MarkerTracker] = field(default_factory=dict)
    
    # Histories
    last_render_ordered_pts: np.ndarray | None = None
    prev_render_ordered_pts: np.ndarray | None = None
    last_full_real_ordered_pts: np.ndarray | None = None
    prev_full_real_ordered_pts: np.ndarray | None = None
    
    last_predicted_ordered_pts: np.ndarray | None = None
    rendered_pts_prev: np.ndarray | None = None
    last_accepted_quad: np.ndarray | None = None
    last_render_final_pts: np.ndarray | None = None
    previous_complete_centers: dict[str, np.ndarray] | None = None

    # Counts
    marker_miss_counts: dict[str, int] = field(default_factory=dict)
    reacquire_confirm_counts: dict[str, int] = field(default_factory=dict)
    lost_geometry_count: int = 0
    missing_corner_hold_count: int = 0
    active_missing_name: str | None = None
    active_geometry_method: str | None = None # "affine" or "parallelogram"
    geometry_method_fail_count: int = 0
    
    # Area/Aspect History
    prev_quad_area: float | None = None
    prev_quad_aspect: float | None = None

    # Mask & Background Reference
    marker_templates: dict[str, np.ndarray | None] = field(default_factory=dict)
    clean_reference: np.ndarray | None = None
    ref_buffer: list[np.ndarray] = field(default_factory=list)
    mask_buffer: deque = field(default_factory=lambda: deque(maxlen=1))
    warmup_count: int = 0
