"""
Gestionnaire de flux RTSP/ONVIF pour caméras réseau.
Gère les connexions multi-flux parallèles avec buffers circulaires
et reconnexion automatique.
"""

import cv2
import numpy as np
import threading
import queue
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable, Any
from enum import Enum
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamStatus(Enum):
    """États possibles d'un flux."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    STREAMING = "streaming"
    ERROR = "error"
    RECONNECTING = "reconnecting"


@dataclass
class CameraConfig:
    """Configuration d'une caméra."""
    camera_id: str
    name: str
    location: str
    source: str              # URL RTSP, chemin fichier, ou device ID
    
    # Paramètres RTSP/ONVIF
    username: Optional[str] = None
    password: Optional[str] = None
    transport: str = "tcp"   # tcp ou udp
    
    # Paramètres de capture
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    
    # Transformation
    rotation_degrees: int = 0  # 0, 90, 180, 270
    
    # Buffer
    buffer_size: int = 30
    
    # Reconnexion
    auto_reconnect: bool = True
    reconnect_delay: float = 5.0
    max_reconnect_attempts: int = 10


@dataclass
class CameraStream:
    """
    Représente un flux de caméra actif.
    Gère le buffer et les statistiques.
    """
    config: CameraConfig
    status: StreamStatus = StreamStatus.DISCONNECTED
    
    # Capture
    capture: Optional[cv2.VideoCapture] = None
    
    # Buffer circulaire
    frame_buffer: deque = field(default_factory=lambda: deque(maxlen=30))
    
    # Statistiques
    frames_received: int = 0
    frames_dropped: int = 0
    last_frame_time: Optional[float] = None
    fps_actual: float = 0.0
    
    # Timestamps
    connected_at: Optional[datetime] = None
    last_error: Optional[str] = None
    reconnect_attempts: int = 0
    
    # Thread
    _thread: Optional[threading.Thread] = None
    _stop_event: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def __post_init__(self):
        self.frame_buffer = deque(maxlen=self.config.buffer_size)


class StreamManager:
    """
    Gestionnaire de flux multi-caméras.
    Gère les connexions, buffers et synchronisation.
    """
    
    def __init__(
        self,
        max_workers: int = 8,
        on_frame_callback: Optional[Callable[[str, np.ndarray, float], None]] = None,
        on_status_change: Optional[Callable[[str, StreamStatus], None]] = None,
        on_error: Optional[Callable[[str, str], None]] = None,
    ):
        """
        Initialise le gestionnaire.
        
        Args:
            max_workers: Nombre max de threads de capture
            on_frame_callback: Callback appelé pour chaque frame (camera_id, frame, timestamp)
            on_status_change: Callback pour les changements de statut
            on_error: Callback pour les erreurs
        """
        self.streams: Dict[str, CameraStream] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        self.on_frame_callback = on_frame_callback
        self.on_status_change = on_status_change
        self.on_error = on_error
        
        self._running = False
        self._global_lock = threading.Lock()
        
    def add_camera(self, config: CameraConfig) -> CameraStream:
        """Ajoute une caméra au gestionnaire."""
        stream = CameraStream(config=config)
        
        with self._global_lock:
            self.streams[config.camera_id] = stream
            
        logger.info(f"Caméra ajoutée: {config.camera_id} ({config.location})")
        return stream
    
    def remove_camera(self, camera_id: str):
        """Retire une caméra du gestionnaire."""
        with self._global_lock:
            if camera_id in self.streams:
                stream = self.streams[camera_id]
                self._disconnect_stream(stream)
                del self.streams[camera_id]
                logger.info(f"Caméra retirée: {camera_id}")
    
    def _build_rtsp_url(self, config: CameraConfig) -> str:
        """Construit l'URL RTSP avec authentification."""
        source = config.source
        
        # Si c'est déjà une URL RTSP avec auth, la retourner
        if source.startswith("rtsp://") and "@" in source:
            return source
        
        # Ajouter l'authentification si nécessaire
        if config.username and config.password:
            if source.startswith("rtsp://"):
                source = source.replace(
                    "rtsp://",
                    f"rtsp://{config.username}:{config.password}@"
                )
        
        return source
    
    def _connect_stream(self, stream: CameraStream) -> bool:
        """Établit la connexion à une caméra."""
        config = stream.config
        
        self._update_status(stream, StreamStatus.CONNECTING)
        
        try:
            source = config.source
            
            # Configuration pour RTSP
            if source.startswith("rtsp://"):
                source = self._build_rtsp_url(config)
                
                # Options GStreamer pour une meilleure latence
                gst_pipeline = (
                    f"rtspsrc location={source} latency=0 ! "
                    "rtph264depay ! h264parse ! avdec_h264 ! "
                    "videoconvert ! appsink"
                )
                
                # Essayer d'abord avec GStreamer
                cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
                
                if not cap.isOpened():
                    # Fallback vers FFmpeg/OpenCV direct
                    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            else:
                # Fichier local ou device
                cap = cv2.VideoCapture(source)
            
            if not cap.isOpened():
                raise ConnectionError(f"Impossible d'ouvrir: {source}")
            
            # Configurer les propriétés si spécifiées
            if config.width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
            if config.height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
            if config.fps:
                cap.set(cv2.CAP_PROP_FPS, config.fps)
            
            # Buffer minimal pour réduire la latence
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            stream.capture = cap
            stream.connected_at = datetime.now()
            stream.reconnect_attempts = 0
            
            self._update_status(stream, StreamStatus.CONNECTED)
            logger.info(f"✓ Connecté: {config.camera_id}")
            
            return True
            
        except Exception as e:
            stream.last_error = str(e)
            self._update_status(stream, StreamStatus.ERROR)
            
            if self.on_error:
                self.on_error(config.camera_id, str(e))
            
            logger.error(f"✗ Erreur connexion {config.camera_id}: {e}")
            return False
    
    def _disconnect_stream(self, stream: CameraStream):
        """Déconnecte un flux."""
        stream._stop_event.set()
        
        if stream._thread and stream._thread.is_alive():
            stream._thread.join(timeout=2.0)
        
        if stream.capture:
            stream.capture.release()
            stream.capture = None
        
        self._update_status(stream, StreamStatus.DISCONNECTED)
    
    def _update_status(self, stream: CameraStream, status: StreamStatus):
        """Met à jour le statut d'un flux."""
        old_status = stream.status
        stream.status = status
        
        if self.on_status_change and old_status != status:
            self.on_status_change(stream.config.camera_id, status)
    
    def _capture_loop(self, stream: CameraStream):
        """Boucle de capture pour un flux."""
        config = stream.config
        fps_counter = 0
        fps_start_time = time.time()
        
        self._update_status(stream, StreamStatus.STREAMING)
        
        while not stream._stop_event.is_set():
            try:
                if stream.capture is None:
                    break
                
                ret, frame = stream.capture.read()
                current_time = time.time()
                
                if not ret or frame is None:
                    stream.frames_dropped += 1
                    
                    # Tentative de reconnexion si configurée
                    if config.auto_reconnect:
                        self._handle_reconnect(stream)
                    else:
                        break
                    continue
                
                # Mettre à jour les statistiques
                stream.frames_received += 1
                stream.last_frame_time = current_time
                fps_counter += 1
                
                # Calculer le FPS réel
                if current_time - fps_start_time >= 1.0:
                    stream.fps_actual = fps_counter / (current_time - fps_start_time)
                    fps_counter = 0
                    fps_start_time = current_time
                
                # Appliquer la rotation si configurée
                if config.rotation_degrees != 0:
                    if config.rotation_degrees == 90:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    elif config.rotation_degrees == 180:
                        frame = cv2.rotate(frame, cv2.ROTATE_180)
                    elif config.rotation_degrees == 270:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                # Ajouter au buffer
                with stream._lock:
                    stream.frame_buffer.append((frame, current_time))
                
                # Callback
                if self.on_frame_callback:
                    self.on_frame_callback(config.camera_id, frame, current_time)
                
            except Exception as e:
                stream.last_error = str(e)
                logger.error(f"Erreur capture {config.camera_id}: {e}")
                
                if config.auto_reconnect:
                    self._handle_reconnect(stream)
                else:
                    break
        
        self._update_status(stream, StreamStatus.DISCONNECTED)
    
    def _handle_reconnect(self, stream: CameraStream):
        """Gère la reconnexion automatique."""
        config = stream.config
        
        if stream.reconnect_attempts >= config.max_reconnect_attempts:
            logger.error(f"Max reconnexion atteint pour {config.camera_id}")
            self._update_status(stream, StreamStatus.ERROR)
            stream._stop_event.set()
            return
        
        self._update_status(stream, StreamStatus.RECONNECTING)
        stream.reconnect_attempts += 1
        
        logger.info(f"Reconnexion {config.camera_id} ({stream.reconnect_attempts}/{config.max_reconnect_attempts})")
        
        # Fermer l'ancienne connexion
        if stream.capture:
            stream.capture.release()
            stream.capture = None
        
        # Attendre avant de reconnecter
        time.sleep(config.reconnect_delay)
        
        # Réessayer la connexion
        if not stream._stop_event.is_set():
            self._connect_stream(stream)
    
    def start_all(self):
        """Démarre tous les flux."""
        self._running = True
        
        for camera_id, stream in self.streams.items():
            if stream.status == StreamStatus.DISCONNECTED:
                if self._connect_stream(stream):
                    stream._stop_event.clear()
                    stream._thread = threading.Thread(
                        target=self._capture_loop,
                        args=(stream,),
                        daemon=True,
                        name=f"capture_{camera_id}"
                    )
                    stream._thread.start()
    
    def stop_all(self):
        """Arrête tous les flux."""
        self._running = False
        
        for stream in self.streams.values():
            self._disconnect_stream(stream)
    
    def start_camera(self, camera_id: str):
        """Démarre un flux spécifique."""
        if camera_id not in self.streams:
            raise ValueError(f"Caméra inconnue: {camera_id}")
        
        stream = self.streams[camera_id]
        
        if stream.status == StreamStatus.DISCONNECTED:
            if self._connect_stream(stream):
                stream._stop_event.clear()
                stream._thread = threading.Thread(
                    target=self._capture_loop,
                    args=(stream,),
                    daemon=True
                )
                stream._thread.start()
    
    def stop_camera(self, camera_id: str):
        """Arrête un flux spécifique."""
        if camera_id in self.streams:
            self._disconnect_stream(self.streams[camera_id])
    
    def get_frame(self, camera_id: str) -> Optional[tuple]:
        """
        Récupère la dernière frame d'une caméra.
        
        Returns:
            Tuple (frame, timestamp) ou None
        """
        if camera_id not in self.streams:
            return None
        
        stream = self.streams[camera_id]
        
        with stream._lock:
            if stream.frame_buffer:
                return stream.frame_buffer[-1]
        
        return None
    
    def get_all_frames(self) -> Dict[str, Optional[tuple]]:
        """
        Récupère les dernières frames de toutes les caméras.
        
        Returns:
            Dict camera_id -> (frame, timestamp)
        """
        frames = {}
        
        for camera_id in self.streams:
            frames[camera_id] = self.get_frame(camera_id)
        
        return frames
    
    def get_synchronized_frames(
        self,
        max_time_diff_ms: float = 100.0
    ) -> Optional[Dict[str, tuple]]:
        """
        Récupère des frames synchronisées de toutes les caméras.
        
        Args:
            max_time_diff_ms: Différence de temps max acceptable
            
        Returns:
            Dict camera_id -> (frame, timestamp) si synchronisé, None sinon
        """
        frames = self.get_all_frames()
        
        # Filtrer les caméras sans frame
        valid_frames = {k: v for k, v in frames.items() if v is not None}
        
        if len(valid_frames) < len(self.streams):
            return None
        
        # Vérifier la synchronisation
        timestamps = [v[1] for v in valid_frames.values()]
        time_spread = (max(timestamps) - min(timestamps)) * 1000  # en ms
        
        if time_spread > max_time_diff_ms:
            logger.warning(f"Désynchronisation détectée: {time_spread:.1f}ms")
            return None
        
        return valid_frames
    
    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """Retourne le statut de toutes les caméras."""
        status = {}
        
        for camera_id, stream in self.streams.items():
            status[camera_id] = {
                "status": stream.status.value,
                "location": stream.config.location,
                "frames_received": stream.frames_received,
                "frames_dropped": stream.frames_dropped,
                "fps_actual": f"{stream.fps_actual:.1f}",
                "last_error": stream.last_error,
                "reconnect_attempts": stream.reconnect_attempts,
            }
        
        return status
    
    def __enter__(self):
        self.start_all()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_all()


def create_onvif_url(
    ip: str,
    port: int = 554,
    username: str = "admin",
    password: str = "",
    stream_path: str = "/Streaming/Channels/101"
) -> str:
    """
    Crée une URL RTSP compatible ONVIF.
    
    Args:
        ip: Adresse IP de la caméra
        port: Port RTSP (généralement 554)
        username: Nom d'utilisateur
        password: Mot de passe
        stream_path: Chemin du stream (varie selon le fabricant)
        
    Returns:
        URL RTSP formatée
    """
    if password:
        return f"rtsp://{username}:{password}@{ip}:{port}{stream_path}"
    return f"rtsp://{ip}:{port}{stream_path}"


# Chemins de stream courants par fabricant
ONVIF_STREAM_PATHS = {
    "hikvision": "/Streaming/Channels/101",
    "dahua": "/cam/realmonitor?channel=1&subtype=0",
    "axis": "/axis-media/media.amp",
    "vivotek": "/live.sdp",
    "generic": "/stream1",
}


if __name__ == "__main__":
    import time
    
    # Exemple avec des fichiers locaux
    dataset_path = "/home/epl/DATA/S03/MTH2329/APEKE/PROJET/Dataset"
    
    manager = StreamManager()
    
    # Ajouter quelques caméras (fichiers vidéo)
    cameras = [
        CameraConfig(
            camera_id="cam_devanture",
            name="Caméra Devanture",
            location="Devanture - Porte Entrée",
            source=f"{dataset_path}/CAMERA_DEVANTURE_PORTE_ENTREE.mp4",
            buffer_size=30,
        ),
        CameraConfig(
            camera_id="cam_hall",
            name="Caméra Hall",
            location="Hall - Porte Entrée",
            source=f"{dataset_path}/CAMERA_HALL_PORTE_ENTREE.mp4",
            buffer_size=30,
        ),
    ]
    
    for cam in cameras:
        manager.add_camera(cam)
    
    print("\n🎥 Démarrage des flux...\n")
    
    with manager:
        for i in range(50):  # Lire 50 frames
            frames = manager.get_all_frames()
            
            status = manager.get_status()
            for cam_id, s in status.items():
                print(f"  {cam_id}: {s['status']} - {s['frames_received']} frames @ {s['fps_actual']} FPS")
            
            time.sleep(0.04)  # ~25 FPS
    
    print("\n✓ Test terminé")
