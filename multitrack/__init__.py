# Multi-Camera Surveillance System
"""
Système de surveillance multi-caméras avec synchronisation temporelle,
gestion de flux RTSP/ONVIF, stockage organisé et monitoring.
"""

from .video_metadata import VideoMetadata, extract_all_metadata
from .video_sync import VideoSynchronizer, SyncMethod
from .stream_manager import StreamManager, CameraStream
from .data_storage import DataStorage
from .health_monitor import HealthMonitor
from .sync_merger import SyncMerger

__version__ = "1.0.0"
__all__ = [
    "VideoMetadata",
    "extract_all_metadata", 
    "VideoSynchronizer",
    "SyncMethod",
    "StreamManager",
    "CameraStream",
    "DataStorage",
    "HealthMonitor",
    "SyncMerger",
]
