from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RRTrackConfig:
    """Runtime policy for the paper-derived RRTrack state machine.

    The paper publishes alpha, buffer sizes and bank capacities, but not all
    decision thresholds.  Fields marked as project defaults are therefore kept
    configurable and are adapted online after enough stable frames.
    """

    # Published implementation details.
    ema_alpha: float = 0.05
    mad_window: int = 100
    retrieval_top_k: int = 3
    online_bank_capacity: int = 64
    dino_input_size: int = 224
    dino_context_padding: float = 0.15

    # Project defaults for thresholds not reported by the paper.
    precision_track: float = 0.62
    precision_floor: float = 0.25
    precision_stagnation: float = 0.48
    precision_short_memory: float = 0.66
    precision_long_memory: float = 0.78
    support_long_memory: float = 0.68
    precision_bank: float = 0.82
    support_bank: float = 0.74
    entropy_max: float = 0.48
    similarity_offline: float = 0.45
    similarity_online: float = 0.55

    min_snap_area_px: int = 96
    min_track_area_px: int = 48
    min_valid_depth_px: int = 24
    min_depth_m: float = 0.05
    max_depth_m: float = 2.0
    stagnation_window: int = 8
    stagnation_rotation_deg: float = 1.0
    stagnation_translation_m: float = 0.004
    lost_patience_frames: int = 2
    recovery_retry_interval: int = 1
    global_register_after_retrieval_failures: int = 3
    sam3_recovery_after_frames: int = 4
    long_memory_interval: int = 20
    max_long_anchors: int = 6
    online_bank_interval: int = 15
    adaptive_min_samples: int = 8
    adaptive_mad_scale: float = 2.5

    def __post_init__(self) -> None:
        probability_fields = (
            "precision_track",
            "precision_floor",
            "precision_stagnation",
            "precision_short_memory",
            "precision_long_memory",
            "support_long_memory",
            "precision_bank",
            "support_bank",
            "entropy_max",
            "similarity_offline",
            "similarity_online",
        )
        for name in probability_fields:
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        for name in (
            "mad_window",
            "retrieval_top_k",
            "online_bank_capacity",
            "min_snap_area_px",
            "min_track_area_px",
            "stagnation_window",
            "lost_patience_frames",
            "global_register_after_retrieval_failures",
            "max_long_anchors",
            "sam3_recovery_after_frames",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
