"""
Module de synchronisation vidéo.
Implémente plusieurs méthodes d'alignement temporel:
- Synchronisation par timestamps metadata
- Cross-corrélation audio
- Détection d'événements visuels
- Gestion des framerates différents
"""

import cv2
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Tuple, Any
import logging
from concurrent.futures import ThreadPoolExecutor
import threading

try:
    from scipy import signal
    from scipy.io import wavfile
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from .video_metadata import VideoMetadata, extract_video_metadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SyncMethod(Enum):
    """Méthodes de synchronisation disponibles."""
    TIMESTAMP = "timestamp"      # Via metadata timestamps
    AUDIO = "audio"              # Cross-corrélation audio
    VISUAL = "visual"            # Événements visuels (flash, mouvement)
    MANUAL = "manual"            # Offsets manuels


@dataclass
class SyncResult:
    """Résultat de synchronisation pour une vidéo."""
    video_path: str
    camera_id: str
    offset_seconds: float      # Décalage par rapport à la référence
    offset_frames: int         # Décalage en frames
    confidence: float          # Confiance de la synchronisation (0-1)
    method_used: SyncMethod
    original_fps: float
    target_fps: float
    
    def get_start_frame(self) -> int:
        """Retourne le frame de départ après synchronisation."""
        return max(0, self.offset_frames)


class VideoSynchronizer:
    """
    Synchroniseur multi-vidéos.
    Aligne temporellement plusieurs flux vidéo.
    """
    
    def __init__(self, target_fps: float = 25.0, max_drift_ms: float = 100.0):
        """
        Initialise le synchroniseur.
        
        Args:
            target_fps: FPS cible pour uniformiser les videos
            max_drift_ms: Dérive maximale acceptable en millisecondes
        """
        self.target_fps = target_fps
        self.max_drift_ms = max_drift_ms
        self.sync_results: Dict[str, SyncResult] = {}
        
    def synchronize_by_timestamp(
        self, 
        metadata_list: List[VideoMetadata]
    ) -> List[SyncResult]:
        """
        Synchronise les vidéos par leurs timestamps de création.
        
        Cette méthode utilise les metadata de création/modification
        pour calculer les offsets relatifs.
        """
        results = []
        
        # Trouver les timestamps disponibles
        timestamps = []
        for meta in metadata_list:
            if meta.creation_time:
                from datetime import datetime
                try:
                    # Parser ISO format
                    ts = datetime.fromisoformat(meta.creation_time.replace("Z", "+00:00"))
                    timestamps.append((meta, ts))
                except ValueError:
                    logger.warning(f"Timestamp invalide pour {meta.filename}")
                    
        if len(timestamps) < 2:
            logger.warning("Timestamps insuffisants, utilisation des temps de modification")
            # Fallback sur les temps de modification
            timestamps = []
            for meta in metadata_list:
                if meta.modification_time:
                    from datetime import datetime
                    ts = datetime.fromisoformat(meta.modification_time)
                    timestamps.append((meta, ts))
        
        if not timestamps:
            raise ValueError("Aucun timestamp disponible pour la synchronisation")
        
        # Trouver le timestamp de référence (le plus récent comme point de départ)
        reference_meta, reference_ts = max(timestamps, key=lambda x: x[1])
        logger.info(f"Référence: {reference_meta.filename} @ {reference_ts}")
        
        for meta, ts in timestamps:
            offset_seconds = (reference_ts - ts).total_seconds()
            offset_frames = int(offset_seconds * meta.fps)
            
            result = SyncResult(
                video_path=meta.filepath,
                camera_id=meta.camera_id,
                offset_seconds=offset_seconds,
                offset_frames=offset_frames,
                confidence=0.8 if meta.creation_time else 0.5,
                method_used=SyncMethod.TIMESTAMP,
                original_fps=meta.fps,
                target_fps=self.target_fps,
            )
            results.append(result)
            self.sync_results[meta.camera_id] = result
            
        return results
    
    def synchronize_by_audio(
        self,
        video_paths: List[str],
        reference_index: int = 0
    ) -> List[SyncResult]:
        """
        Synchronise les vidéos par cross-corrélation audio.
        
        Cette méthode extrait les pistes audio et calcule le décalage
        par corrélation croisée du signal.
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy requis pour la synchronisation audio")
        
        import subprocess
        import tempfile
        import os
        
        def extract_audio(video_path: str) -> Optional[np.ndarray]:
            """Extrait l'audio en numpy array."""
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            
            try:
                cmd = [
                    "ffmpeg", "-y", "-i", video_path,
                    "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                    tmp_path
                ]
                result = subprocess.run(cmd, capture_output=True, timeout=60)
                
                if result.returncode == 0 and os.path.exists(tmp_path):
                    sample_rate, audio = wavfile.read(tmp_path)
                    return audio.astype(np.float32)
                return None
            except Exception as e:
                logger.error(f"Erreur extraction audio: {e}")
                return None
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        
        # Extraire audio de référence
        logger.info(f"Extraction audio de référence: {video_paths[reference_index]}")
        ref_audio = extract_audio(video_paths[reference_index])
        
        if ref_audio is None:
            raise ValueError("Impossible d'extraire l'audio de référence")
        
        results = []
        
        for i, video_path in enumerate(video_paths):
            meta = extract_video_metadata(video_path)
            
            if i == reference_index:
                # Vidéo de référence, offset = 0
                result = SyncResult(
                    video_path=video_path,
                    camera_id=meta.camera_id,
                    offset_seconds=0.0,
                    offset_frames=0,
                    confidence=1.0,
                    method_used=SyncMethod.AUDIO,
                    original_fps=meta.fps,
                    target_fps=self.target_fps,
                )
            else:
                logger.info(f"Cross-corrélation avec: {meta.filename}")
                audio = extract_audio(video_path)
                
                if audio is None:
                    logger.warning(f"Pas d'audio pour {meta.filename}, skip")
                    continue
                
                # Cross-corrélation
                correlation = signal.correlate(ref_audio, audio, mode='full')
                lag = np.argmax(np.abs(correlation)) - len(audio) + 1
                
                # Convertir lag (samples @ 16kHz) en secondes
                offset_seconds = lag / 16000.0
                offset_frames = int(offset_seconds * meta.fps)
                
                # Confiance basée sur la netteté du pic
                peak_value = np.max(np.abs(correlation))
                mean_value = np.mean(np.abs(correlation))
                confidence = min(1.0, (peak_value / mean_value) / 10.0)
                
                result = SyncResult(
                    video_path=video_path,
                    camera_id=meta.camera_id,
                    offset_seconds=offset_seconds,
                    offset_frames=offset_frames,
                    confidence=confidence,
                    method_used=SyncMethod.AUDIO,
                    original_fps=meta.fps,
                    target_fps=self.target_fps,
                )
            
            results.append(result)
            self.sync_results[meta.camera_id] = result
        
        return results
    
    def synchronize_by_visual_event(
        self,
        video_paths: List[str],
        event_type: str = "flash",
        search_window_seconds: float = 10.0
    ) -> List[SyncResult]:
        """
        Synchronise par détection d'événements visuels communs.
        
        Args:
            event_type: Type d'événement ("flash", "motion", "clap")
            search_window_seconds: Fenêtre de recherche initiale
        """
        def detect_flash(video_path: str, max_frames: int = 300) -> Optional[int]:
            """Détecte un changement brusque de luminosité (flash)."""
            cap = cv2.VideoCapture(video_path)
            prev_brightness = None
            flash_frame = None
            max_diff = 0
            
            for frame_idx in range(max_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray)
                
                if prev_brightness is not None:
                    diff = abs(brightness - prev_brightness)
                    if diff > max_diff and diff > 30:  # Seuil de détection
                        max_diff = diff
                        flash_frame = frame_idx
                
                prev_brightness = brightness
            
            cap.release()
            return flash_frame
        
        def detect_motion_peak(video_path: str, max_frames: int = 300) -> Optional[int]:
            """Détecte un pic de mouvement."""
            cap = cv2.VideoCapture(video_path)
            prev_gray = None
            motion_peak_frame = None
            max_motion = 0
            
            for frame_idx in range(max_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)
                
                if prev_gray is not None:
                    delta = cv2.absdiff(prev_gray, gray)
                    motion = np.sum(delta > 25)
                    
                    if motion > max_motion:
                        max_motion = motion
                        motion_peak_frame = frame_idx
                
                prev_gray = gray
            
            cap.release()
            return motion_peak_frame
        
        # Sélectionner la fonction de détection
        detect_func = detect_flash if event_type == "flash" else detect_motion_peak
        
        results = []
        reference_frame = None
        reference_fps = None
        
        for i, video_path in enumerate(video_paths):
            meta = extract_video_metadata(video_path)
            max_frames = int(search_window_seconds * meta.fps)
            
            event_frame = detect_func(video_path, max_frames)
            
            if event_frame is None:
                logger.warning(f"Événement non détecté dans {meta.filename}")
                continue
            
            logger.info(f"Événement détecté frame {event_frame} dans {meta.filename}")
            
            if i == 0:
                reference_frame = event_frame
                reference_fps = meta.fps
            
            # Calculer l'offset par rapport à la référence
            if reference_frame is not None:
                ref_time = reference_frame / reference_fps
                this_time = event_frame / meta.fps
                offset_seconds = ref_time - this_time
                offset_frames = int(offset_seconds * meta.fps)
            else:
                offset_seconds = 0.0
                offset_frames = 0
            
            result = SyncResult(
                video_path=video_path,
                camera_id=meta.camera_id,
                offset_seconds=offset_seconds,
                offset_frames=offset_frames,
                confidence=0.7,  # Confiance moyenne pour la méthode visuelle
                method_used=SyncMethod.VISUAL,
                original_fps=meta.fps,
                target_fps=self.target_fps,
            )
            results.append(result)
            self.sync_results[meta.camera_id] = result
        
        return results
    
    def set_manual_offsets(
        self,
        offsets: Dict[str, float],
        metadata_list: List[VideoMetadata]
    ) -> List[SyncResult]:
        """
        Définit des offsets manuels pour chaque caméra.
        
        Args:
            offsets: Dict camera_id -> offset en secondes
            metadata_list: Liste des métadonnées vidéo
        """
        results = []
        
        for meta in metadata_list:
            offset_seconds = offsets.get(meta.camera_id, 0.0)
            offset_frames = int(offset_seconds * meta.fps)
            
            result = SyncResult(
                video_path=meta.filepath,
                camera_id=meta.camera_id,
                offset_seconds=offset_seconds,
                offset_frames=offset_frames,
                confidence=1.0,
                method_used=SyncMethod.MANUAL,
                original_fps=meta.fps,
                target_fps=self.target_fps,
            )
            results.append(result)
            self.sync_results[meta.camera_id] = result
        
        return results


class FramerateNormalizer:
    """
    Normalise les framerates pour uniformiser les flux vidéo.
    Gère l'interpolation et le sous-échantillonnage.
    """
    
    def __init__(self, target_fps: float = 25.0):
        self.target_fps = target_fps
        
    def get_frame_mapping(
        self,
        original_fps: float,
        frame_count: int
    ) -> List[Tuple[int, float]]:
        """
        Crée une mapping des frames source vers les frames cible.
        
        Returns:
            Liste de tuples (source_frame_index, blend_weight)
            Pour l'interpolation, blend_weight indique le poids de la frame suivante
        """
        if original_fps == self.target_fps:
            return [(i, 0.0) for i in range(frame_count)]
        
        # Calculer le ratio
        ratio = original_fps / self.target_fps
        
        # Calculer le nombre de frames en sortie
        output_frame_count = int(frame_count / ratio)
        
        mapping = []
        for out_idx in range(output_frame_count):
            # Position dans la vidéo source
            src_pos = out_idx * ratio
            src_frame = int(src_pos)
            blend = src_pos - src_frame
            
            if src_frame < frame_count:
                mapping.append((src_frame, blend))
        
        return mapping
    
    def interpolate_frames(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        weight: float
    ) -> np.ndarray:
        """
        Interpole deux frames avec le poids donné.
        
        Args:
            frame1: Première frame
            frame2: Deuxième frame  
            weight: Poids de frame2 (0.0 = frame1 pure, 1.0 = frame2 pure)
        """
        if weight == 0.0:
            return frame1
        if weight == 1.0:
            return frame2
            
        return cv2.addWeighted(frame1, 1.0 - weight, frame2, weight, 0)


class SynchronizedReader:
    """
    Lecteur synchronisé multi-vidéos.
    Lit les frames de plusieurs vidéos de manière synchronisée.
    """
    
    def __init__(
        self,
        sync_results: List[SyncResult],
        target_fps: float = 25.0
    ):
        self.sync_results = {r.camera_id: r for r in sync_results}
        self.target_fps = target_fps
        self.normalizer = FramerateNormalizer(target_fps)
        
        self.captures: Dict[str, cv2.VideoCapture] = {}
        self.frame_mappings: Dict[str, List[Tuple[int, float]]] = {}
        self.current_frame = 0
        self.lock = threading.Lock()
        
    def open(self):
        """Ouvre tous les flux vidéo."""
        for camera_id, result in self.sync_results.items():
            cap = cv2.VideoCapture(result.video_path)
            if not cap.isOpened():
                raise ValueError(f"Impossible d'ouvrir: {result.video_path}")
            
            self.captures[camera_id] = cap
            
            # Appliquer l'offset initial
            if result.offset_frames > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, result.offset_frames)
            
            # Créer le mapping de frames
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_mappings[camera_id] = self.normalizer.get_frame_mapping(
                result.original_fps,
                frame_count - result.offset_frames
            )
    
    def read_synchronized(self) -> Dict[str, Optional[np.ndarray]]:
        """
        Lit un ensemble de frames synchronisées de toutes les caméras.
        
        Returns:
            Dict camera_id -> frame (ou None si fin de vidéo)
        """
        with self.lock:
            frames = {}
            
            for camera_id, cap in self.captures.items():
                mapping = self.frame_mappings[camera_id]
                
                if self.current_frame >= len(mapping):
                    frames[camera_id] = None
                    continue
                
                src_frame, blend = mapping[self.current_frame]
                
                # Positionner et lire
                cap.set(cv2.CAP_PROP_POS_FRAMES, src_frame)
                ret, frame = cap.read()
                
                if not ret:
                    frames[camera_id] = None
                    continue
                
                # Interpolation si nécessaire
                if blend > 0.01:
                    ret2, frame2 = cap.read()
                    if ret2:
                        frame = self.normalizer.interpolate_frames(frame, frame2, blend)
                
                frames[camera_id] = frame
            
            self.current_frame += 1
            return frames
    
    def get_total_frames(self) -> int:
        """Retourne le nombre total de frames synchronisées."""
        if not self.frame_mappings:
            return 0
        return min(len(m) for m in self.frame_mappings.values())
    
    def reset(self):
        """Remet la lecture au début."""
        with self.lock:
            self.current_frame = 0
            for camera_id, result in self.sync_results.items():
                if camera_id in self.captures:
                    offset = max(0, result.offset_frames)
                    self.captures[camera_id].set(cv2.CAP_PROP_POS_FRAMES, offset)
    
    def close(self):
        """Ferme tous les flux."""
        for cap in self.captures.values():
            cap.release()
        self.captures.clear()
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    import glob
    import os
    
    # Exemple d'utilisation
    dataset_path = "/home/epl/DATA/S03/MTH2329/APEKE/PROJET/Dataset"
    videos = glob.glob(os.path.join(dataset_path, "*.mp4"))[:3]  # Test avec 3 vidéos
    
    print(f"\n🔄 Test de synchronisation avec {len(videos)} vidéos...\n")
    
    # Extraire métadonnées
    from video_metadata import extract_all_metadata
    metadata_list = extract_all_metadata(videos)
    
    # Créer synchroniseur
    synchronizer = VideoSynchronizer(target_fps=25.0)
    
    # Test synchronisation par timestamp
    print("\n📅 Synchronisation par timestamps:")
    try:
        results = synchronizer.synchronize_by_timestamp(metadata_list)
        for r in results:
            print(f"   {r.camera_id}: offset = {r.offset_seconds:.2f}s ({r.offset_frames} frames)")
    except Exception as e:
        print(f"   Erreur: {e}")
    
    # Test synchronisation visuelle
    print("\n👁️ Synchronisation par événement visuel (flash):")
    try:
        results = synchronizer.synchronize_by_visual_event(videos[:2], event_type="motion")
        for r in results:
            print(f"   {r.camera_id}: offset = {r.offset_seconds:.2f}s")
    except Exception as e:
        print(f"   Erreur: {e}")
