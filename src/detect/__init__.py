"""
Person Detection and Tracking
"""
from .person_detector import Detection, PersonDetector, iter_video_frames, video_metadata

__all__ = [
    "Detection", 
    "PersonDetector", 
    "iter_video_frames", 
    "video_metadata"
]

__version__ = "1.0.0"
