"""RRTrack-inspired recoverable 2D--6D object tracking.

The implementation follows the public RRTrack paper.  It is intentionally
adapter based: CUTIE, FoundationPose/SAM6D and DINOv2 can be replaced without
changing the state machine.
"""

from .config import RRTrackConfig
from .models import FrameObservation, RRTrackOutput, TrackerState
from .tracker import RRTracker

__all__ = ["FrameObservation", "RRTrackConfig", "RRTrackOutput", "RRTracker", "TrackerState"]
