"""
Module de lecture synchronisée basée sur les timestamps d'enregistrement.
Affiche les vidéos en temps réel selon leurs heures de début originales.
"""

import cv2
import numpy as np
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging

from .video_metadata import extract_video_metadata, VideoMetadata
from .stream_manager import CameraConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SyncedVideoSource:
    """Représente une source vidéo avec ses informations de synchronisation."""
    camera_id: str
    filepath: str
    name: str
    metadata: VideoMetadata
    
    # Timestamps
    start_time: datetime
    end_time: datetime
    
    # Offset relatif (en secondes par rapport au début global)
    offset_seconds: float = 0.0
    
    # État
    capture: Optional[cv2.VideoCapture] = None
    is_active: bool = False
    current_frame: int = 0
    
    # Transformation
    rotation_degrees: int = 0


class TimeSyncedPlayer:
    """
    Lecteur multi-vidéo synchronisé par timestamps d'enregistrement.
    Les vidéos démarrent et s'arrêtent selon leurs heures d'enregistrement originales.
    """
    
    def __init__(
        self,
        target_fps: float = 25.0,
        playback_speed: float = 1.0,
        grid_cell_size: Tuple[int, int] = (480, 360),
        align_start: bool = False,
        enable_tracking: bool = False,
        enable_mcmot: bool = False,
        enable_reid: bool = True,
        reid_threshold: float = 0.6,
        tracker_model: str = "yolov8n.pt",
    ):
        """
        Initialise le lecteur synchronisé.
        
        Args:
            target_fps: FPS d'affichage cible
            playback_speed: Vitesse de lecture (1.0 = temps réel)
            grid_cell_size: Taille des cellules de la grille
            align_start: Si True, toutes les vidéos démarrent en même temps
            enable_tracking: Activer le tracking simple
            enable_mcmot: Activer le tracking cross-camera (MCMOT)
            enable_reid: Activer ReID pour MCMOT
            reid_threshold: Seuil de similarité ReID
            tracker_model: Modèle YOLO pour le tracking
        """
        self.target_fps = target_fps
        self.playback_speed = playback_speed
        self.grid_cell_size = grid_cell_size
        self.align_start = align_start
        self.enable_tracking = enable_tracking
        self.enable_mcmot = enable_mcmot
        
        self.sources: Dict[str, SyncedVideoSource] = {}
        
        # Timing global
        self.global_start_time: Optional[datetime] = None
        self.global_end_time: Optional[datetime] = None
        self.current_time: Optional[datetime] = None
        
        # État
        self._running = False
        self._paused = False
        self._lock = threading.Lock()
        
        # Timestamps manuels (nom_partiel -> datetime ou str "HH:MM:SS")
        self.manual_timestamps: Dict[str, str] = {}
        self.reference_date: str = "2025-12-11"  # Date de référence par défaut
        
        # Tracker pour chaque caméra
        self._trackers: Dict[str, Any] = {}
        self._cached_detections: Dict[str, list] = {}  # Dernières détections par caméra
        self._frame_counter: int = 0  # Compteur pour skip frames
        self._track_every_n: int = 5  # Tracker 1 frame sur 5 pour fluidité
        if enable_tracking:
            try:
                from .object_tracker import ObjectTracker
                self._tracker_class = ObjectTracker
                self._tracker_model = tracker_model
                logger.info(f"🎯 Tracking activé (modèle: {tracker_model}, 1/{self._track_every_n} frames)")
            except ImportError:
                logger.warning("Module object_tracker non disponible")
                self.enable_tracking = False
        
        # MCMOT Tracker Avancé (IDs globaux cross-camera avec galerie adaptative)
        self._mcmot_tracker = None
        self._mcmot_timestamp = 0.0
        self._mcmot_stats_cache = {}  # Cache pour statistiques
        if enable_mcmot:
            try:
                from .mcmot_advanced import MCMOTAdvancedTracker
                self._mcmot_tracker = MCMOTAdvancedTracker(
                    model_path=tracker_model,
                    enable_reid=enable_reid,
                    reid_threshold=reid_threshold,
                )
                reid_status = "activé" if enable_reid else "désactivé"
                logger.info(f"🌐 MCMOT Avancé activé (ReID: {reid_status}, seuil: {reid_threshold})")
                logger.info(f"   Galerie adaptative: activée")
            except ImportError as e:
                logger.warning(f"Module mcmot_advanced non disponible: {e}")
                self.enable_mcmot = False
        
    def set_manual_timestamps(self, timestamps: Dict[str, str], reference_date: str = None):
        """
        Définit les timestamps manuels pour les vidéos.
        
        Args:
            timestamps: Dict avec nom partiel -> heure "HH:MM:SS"
            reference_date: Date de référence "YYYY-MM-DD"
        """
        self.manual_timestamps = timestamps
        if reference_date:
            self.reference_date = reference_date
        logger.info(f"📋 {len(timestamps)} timestamps manuels configurés")
        
    def _get_manual_start_time(self, filename: str) -> Optional[datetime]:
        """Cherche un timestamp manuel pour ce fichier."""
        filename_upper = filename.upper()
        
        for pattern, time_str in self.manual_timestamps.items():
            if pattern.upper() in filename_upper:
                try:
                    # Parser l'heure
                    time_obj = datetime.strptime(time_str, "%H:%M:%S").time()
                    # Combiner avec la date de référence
                    date_obj = datetime.strptime(self.reference_date, "%Y-%m-%d").date()
                    return datetime.combine(date_obj, time_obj)
                except ValueError as e:
                    logger.warning(f"Format invalide pour {pattern}: {time_str}")
        return None
        
    def add_video(
        self,
        filepath: str,
        camera_id: Optional[str] = None,
        name: Optional[str] = None,
        rotation_degrees: int = 0
    ) -> SyncedVideoSource:
        """Ajoute une vidéo au lecteur synchronisé."""
        # Extraire les métadonnées
        metadata = extract_video_metadata(filepath)
        
        if camera_id is None:
            camera_id = metadata.camera_id
        if name is None:
            name = Path(filepath).stem
        
        # Chercher d'abord un timestamp manuel
        start_time = self._get_manual_start_time(name)
        
        if start_time:
            logger.info(f"  → Timestamp manuel utilisé pour {name}")
        else:
            # Sinon utiliser les métadonnées
            start_time = self._parse_start_time(metadata)
            if start_time is None:
                logger.warning(f"Pas de timestamp pour {name}, utilisation du temps de modification")
                start_time = datetime.fromisoformat(metadata.modification_time)
        
        # Calculer le temps de fin
        end_time = start_time + timedelta(seconds=metadata.duration_seconds)
        
        source = SyncedVideoSource(
            camera_id=camera_id,
            filepath=filepath,
            name=name,
            metadata=metadata,
            start_time=start_time,
            end_time=end_time,
            rotation_degrees=rotation_degrees,
        )
        
        self.sources[camera_id] = source
        logger.info(f"✓ Ajouté: {name} | Début: {start_time.strftime('%H:%M:%S')} | Durée: {metadata.duration_seconds:.0f}s")
        
        return source
    
    def _parse_start_time(self, metadata: VideoMetadata) -> Optional[datetime]:
        """Parse le timestamp de début depuis les métadonnées."""
        if metadata.creation_time:
            try:
                # Format ISO avec timezone
                ts = metadata.creation_time.replace("Z", "+00:00")
                return datetime.fromisoformat(ts).replace(tzinfo=None)
            except ValueError:
                pass
        
        if metadata.modification_time:
            try:
                return datetime.fromisoformat(metadata.modification_time)
            except ValueError:
                pass
        
        return None
    
    def prepare(self):
        """Prépare la lecture en calculant les offsets."""
        if not self.sources:
            raise ValueError("Aucune vidéo ajoutée")
        
        if self.align_start:
            # Mode démarrage simultané: toutes les vidéos commencent en même temps
            logger.info("\n🔄 Mode: Démarrage simultané (toutes les vidéos commencent ensemble)")
            
            # Trouver la durée max
            max_duration = max(s.metadata.duration_seconds for s in self.sources.values())
            
            # Définir un temps de référence commun
            reference_time = datetime.now().replace(microsecond=0)
            
            for source in self.sources.values():
                # Toutes les vidéos commencent au même moment
                source.start_time = reference_time
                source.end_time = reference_time + timedelta(seconds=source.metadata.duration_seconds)
                source.offset_seconds = 0.0
            
            self.global_start_time = reference_time
            self.global_end_time = reference_time + timedelta(seconds=max_duration)
            self.current_time = self.global_start_time
            
            logger.info(f"   Durée max: {max_duration:.0f}s")
            logger.info(f"   Toutes les {len(self.sources)} vidéos démarrent ensemble")
        else:
            # Mode synchronisation par timestamp original
            start_times = [s.start_time for s in self.sources.values()]
            end_times = [s.end_time for s in self.sources.values()]
            
            self.global_start_time = min(start_times)
            self.global_end_time = max(end_times)
            self.current_time = self.global_start_time
            
            # Calculer les offsets relatifs
            for source in self.sources.values():
                source.offset_seconds = (source.start_time - self.global_start_time).total_seconds()
            
            # Trier par heure de début
            sorted_sources = sorted(self.sources.values(), key=lambda s: s.start_time)
            
            logger.info(f"\n📅 Timeline de synchronisation:")
            logger.info(f"   Début global: {self.global_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"   Fin globale:  {self.global_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"   Durée totale: {(self.global_end_time - self.global_start_time).total_seconds():.0f}s")
            logger.info(f"\n   Ordre de démarrage:")
            
            for i, source in enumerate(sorted_sources):
                delay = source.offset_seconds
                logger.info(f"   {i+1}. {source.name}: +{delay:.1f}s")
    
    def _open_capture(self, source: SyncedVideoSource) -> bool:
        """Ouvre la capture vidéo pour une source."""
        if source.capture is not None:
            return True
        
        cap = cv2.VideoCapture(source.filepath)
        if not cap.isOpened():
            logger.error(f"Impossible d'ouvrir: {source.filepath}")
            return False
        
        source.capture = cap
        return True
    
    def _get_frame_at_time(self, source: SyncedVideoSource, current_time: datetime) -> Optional[np.ndarray]:
        """Récupère la frame correspondant au temps courant (optimisé pour fluidité)."""
        # Vérifier si la vidéo est active à ce moment
        if current_time < source.start_time or current_time > source.end_time:
            source.is_active = False
            return None
        
        source.is_active = True
        
        # Ouvrir la capture si nécessaire
        if source.capture is None:
            if not self._open_capture(source):
                return None
        
        # Calculer le frame correspondant
        elapsed = (current_time - source.start_time).total_seconds()
        target_frame = int(elapsed * source.metadata.fps)
        
        # Si on a déjà la bonne frame en cache, la retourner
        if hasattr(source, '_cached_frame') and source._cached_frame is not None:
            if source.current_frame == target_frame:
                return source._cached_frame
        
        # Position actuelle
        current_pos = int(source.capture.get(cv2.CAP_PROP_POS_FRAMES))
        
        # Seek seulement si nécessaire (grand écart)
        # Sinon lire séquentiellement pour fluidité
        if target_frame < current_pos or target_frame > current_pos + 5:
            source.capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        else:
            # Avancer séquentiellement jusqu'à la frame cible
            while current_pos < target_frame:
                source.capture.grab()  # grab est plus rapide que read
                current_pos += 1
        
        ret, frame = source.capture.read()
        if not ret:
            return getattr(source, '_cached_frame', None)  # Retourner le cache si erreur
        
        # Appliquer la rotation
        if source.rotation_degrees != 0:
            if source.rotation_degrees == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif source.rotation_degrees == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif source.rotation_degrees == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Appliquer le tracking si activé
        if self.enable_tracking and frame is not None:
            frame = self._apply_tracking(source.camera_id, frame)
        
        # Appliquer le tracking MCMOT si activé (priorité sur tracking simple)
        if self.enable_mcmot and frame is not None:
            frame = self._apply_mcmot(source.camera_id, frame)
        
        # Cacher la frame
        source._cached_frame = frame
        source.current_frame = target_frame
        return frame
    
    def _apply_tracking(self, camera_id: str, frame: np.ndarray) -> np.ndarray:
        """Applique le tracking sur une frame avec frame skipping pour fluidité."""
        h, w = frame.shape[:2]
        
        # Créer le tracker pour cette caméra si nécessaire
        if camera_id not in self._trackers:
            surveillance_classes = [0, 1, 2, 3, 24, 26, 28, 63, 67]
            self._trackers[camera_id] = self._tracker_class(
                model_path=self._tracker_model,
                tracker="bytetrack",
                confidence_threshold=0.5,
                classes_filter=surveillance_classes,
            )
            self._cached_detections[camera_id] = []
        
        # Ne tracker que 1 frame sur N
        should_track = (self._frame_counter % self._track_every_n) == 0
        
        if should_track:
            # Résolution réduite pour le tracking (320x240 pour vitesse max)
            target_w, target_h = 320, 240
            scale_x, scale_y = w / target_w, h / target_h
            
            small_frame = cv2.resize(frame, (target_w, target_h))
            tracker = self._trackers[camera_id]
            result = tracker.track(small_frame)
            
            # Sauvegarder les détections mises à l'échelle
            self._cached_detections[camera_id] = [
                (int(obj.bbox[0] * scale_x), int(obj.bbox[1] * scale_y),
                 int(obj.bbox[2] * scale_x), int(obj.bbox[3] * scale_y))
                for obj in result.tracked_objects
            ]
        
        # Dessiner les détections (même anciennes)
        for bbox in self._cached_detections.get(camera_id, []):
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        return frame
    
    def _apply_mcmot(self, camera_id: str, frame: np.ndarray) -> np.ndarray:
        """Applique le tracking MCMOT Avancé avec IDs globaux et galerie adaptative."""
        if self._mcmot_tracker is None:
            return frame
        
        h, w = frame.shape[:2]
        
        # Initialiser le cache si nécessaire
        if camera_id not in self._cached_detections:
            self._cached_detections[camera_id] = {'detections': [], 'timestamp': 0}
        
        # Ne tracker que 1 frame sur N pour fluidité maximale
        should_track = (self._frame_counter % self._track_every_n) == 0
        
        current_time = time.time()
        
        if should_track:
            # Tracking sync pour meilleure précision (pas async)
            try:
                # Résolution augmentée pour meilleure détection (640x480 au lieu de 320x240)
                target_w, target_h = 640, 480
                small_frame = cv2.resize(frame, (target_w, target_h))
                
                self._mcmot_timestamp += 1.0 / self.target_fps
                tracks = self._mcmot_tracker.process_frame(camera_id, small_frame, self._mcmot_timestamp)
                self._mcmot_tracker.associate_cross_camera(self._mcmot_timestamp)
                
                scale_x, scale_y = w / target_w, h / target_h
                detections = []
                for track in tracks:
                    # Filtrer les détections avec confiance trop basse
                    if track.confidence < 0.5:
                        continue
                        
                    x1, y1, x2, y2 = track.bbox
                    global_id = self._mcmot_tracker.get_global_id(camera_id, track.track_id)
                    
                    # Vérifier si multi-caméra
                    is_multi_cam = False
                    is_confirmed = False
                    if global_id and global_id in self._mcmot_tracker.global_tracks:
                        gt = self._mcmot_tracker.global_tracks[global_id]
                        is_multi_cam = len(gt.local_tracks) > 1
                        is_confirmed = gt.is_confirmed
                    
                    detections.append({
                        'bbox': (int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)),
                        'global_id': global_id,
                        'class_name': track.class_name,
                        'is_multi_cam': is_multi_cam,
                        'is_confirmed': is_confirmed,
                    })
                
                # Mettre à jour le cache avec timestamp
                self._cached_detections[camera_id] = {
                    'detections': detections,
                    'timestamp': current_time
                }
                
                # Mettre à jour les stats en cache
                self._mcmot_stats_cache = self._mcmot_tracker.get_statistics()
                
                # Nettoyage mémoire périodique (toutes les 100 frames)
                if self._frame_counter % 100 == 0:
                    self._mcmot_tracker.cleanup_memory(self._mcmot_timestamp)
            except Exception as e:
                pass
        
        # Vérifier si le cache est trop vieux (expiration après 0.5 seconde)
        cache_data = self._cached_detections.get(camera_id, {'detections': [], 'timestamp': 0})
        cache_age = current_time - cache_data.get('timestamp', 0)
        
        if cache_age > 0.5:
            # Cache expiré - ne pas dessiner
            return frame
        
        # Dessiner les détections en cache
        import colorsys
        for det in cache_data.get('detections', []):
            if isinstance(det, dict):
                x1, y1, x2, y2 = det['bbox']
                global_id = det.get('global_id')
                class_name = det.get('class_name', '')
                is_multi_cam = det.get('is_multi_cam', False)
                is_confirmed = det.get('is_confirmed', False)
                
                if global_id:
                    hue = (global_id * 0.618) % 1.0
                    rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.95)
                    color = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
                    
                    # Label avec étoile si multi-caméra
                    label = f"G{global_id}"
                    if class_name:
                        label += f" {class_name}"
                    if is_multi_cam:
                        label += " ★"  # Étoile pour objets vus sur plusieurs caméras
                else:
                    color = (128, 128, 128)
                    label = class_name or "?"
                
                # Dessiner bbox
                thickness = 3 if is_confirmed else 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                # Dessiner label
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - lh - 6), (x1 + lw + 4, y1), color, -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def _create_grid(self, frames: Dict[str, Optional[np.ndarray]]) -> np.ndarray:
        """Crée une grille d'affichage avec les frames."""
        n = len(self.sources)
        cols = min(4, n) if n > 1 else 1
        rows = (n + cols - 1) // cols
        
        cell_w, cell_h = self.grid_cell_size
        grid = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
        grid[:] = (30, 30, 30)  # Fond gris foncé
        
        sorted_sources = sorted(self.sources.values(), key=lambda s: s.start_time)
        
        for i, source in enumerate(sorted_sources):
            row, col = i // cols, i % cols
            x, y = col * cell_w, row * cell_h
            
            frame = frames.get(source.camera_id)
            
            if frame is not None and source.is_active:
                # Redimensionner
                cell = cv2.resize(frame, (cell_w, cell_h))
                
                # Overlay avec nom
                cv2.rectangle(cell, (0, 0), (cell_w, 28), (0, 0, 0), -1)
                cv2.putText(
                    cell, source.name[:25],
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1
                )
                
                # Indicateur actif (vert)
                cv2.circle(cell, (cell_w - 15, 14), 6, (0, 255, 0), -1)
            else:
                # Placeholder pour vidéo inactive
                cell = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
                cell[:] = (40, 40, 40)
                
                # Nom
                cv2.putText(
                    cell, source.name[:25],
                    (10, cell_h // 2 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1
                )
                
                # Status
                if self.current_time < source.start_time:
                    wait_time = (source.start_time - self.current_time).total_seconds()
                    status = f"Debut dans {wait_time:.0f}s"
                else:
                    status = "Termine"
                
                cv2.putText(
                    cell, status,
                    (10, cell_h // 2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1
                )
            
            grid[y:y+cell_h, x:x+cell_w] = cell
        
        return grid
    
    def _add_timeline_overlay(self, grid: np.ndarray) -> np.ndarray:
        """Ajoute une barre de timeline en bas de la grille avec stats MCMOT."""
        h, w = grid.shape[:2]
        bar_height = 40 if not self.enable_mcmot else 60  # Plus de hauteur pour MCMOT
        
        # Créer une nouvelle image avec la barre
        output = np.zeros((h + bar_height, w, 3), dtype=np.uint8)
        output[:h] = grid
        output[h:] = (50, 50, 50)
        
        # Dessiner la timeline
        if self.global_start_time and self.global_end_time and self.current_time:
            total_duration = (self.global_end_time - self.global_start_time).total_seconds()
            elapsed = (self.current_time - self.global_start_time).total_seconds()
            progress = elapsed / total_duration if total_duration > 0 else 0
            
            # Barre de progression
            bar_width = int((w - 20) * progress)
            cv2.rectangle(output, (10, h + 10), (w - 10, h + 25), (80, 80, 80), -1)
            cv2.rectangle(output, (10, h + 10), (10 + bar_width, h + 25), (0, 200, 0), -1)
            
            # Temps actuel
            time_str = self.current_time.strftime("%H:%M:%S")
            cv2.putText(
                output, time_str,
                (10, h + 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1
            )
            
            # Temps restant
            remaining = total_duration - elapsed
            remaining_str = f"-{int(remaining // 60)}:{int(remaining % 60):02d}"
            cv2.putText(
                output, remaining_str,
                (w - 70, h + 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1
            )
            
            # Nombre de caméras actives
            active_count = sum(1 for s in self.sources.values() if s.is_active)
            cv2.putText(
                output, f"Cameras: {active_count}/{len(self.sources)}",
                (w // 2 - 50, h + 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1
            )
            
            # Stats MCMOT si activé
            if self.enable_mcmot and self._mcmot_stats_cache:
                stats = self._mcmot_stats_cache
                total_tracks = stats.get('total_global_tracks', 0)
                cross_cam = stats.get('cross_camera_tracks', 0)
                confirmed = stats.get('confirmed_tracks', 0)
                
                # Ligne MCMOT
                mcmot_text = f"MCMOT: {total_tracks} objets | {confirmed} confirmés | {cross_cam} cross-cam ★"
                cv2.putText(
                    output, mcmot_text,
                    (10, h + 55), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 200, 255), 1
                )
                
                # Galerie
                gallery = stats.get('gallery', {})
                if gallery:
                    gallery_text = f"Galerie: {gallery.get('total_features', 0)} features"
                    cv2.putText(
                        output, gallery_text,
                        (w - 180, h + 55), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (150, 150, 150), 1
                    )
        
        return output
    
    def play(self, window_name: str = "Synchronized Multi-Camera Playback", record_output: str = None, headless: bool = False, max_frames: int = None):
        """
        Lance la lecture synchronisée.
        
        Args:
            window_name: Nom de la fenêtre d'affichage
            record_output: Chemin du fichier vidéo de sortie (None = pas d'enregistrement)
            headless: Si True, pas d'affichage GUI (enregistrement seulement)
            max_frames: Nombre max de frames à traiter (None = tout)
        """
        self.prepare()
        
        self._running = True
        frame_interval = 1.0 / self.target_fps
        
        # En mode headless, on n'ouvre pas de fenêtre
        if not headless:
            try:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            except cv2.error as e:
                logger.warning(f"Impossible d'ouvrir la fenêtre GUI: {e}")
                logger.info("Passage en mode headless automatique")
                headless = True
        
        # Initialiser le VideoWriter si enregistrement demandé
        writer = None
        if record_output:
            # Calculer la taille de la grille
            n = len(self.sources)
            cols = min(4, n) if n > 1 else 1
            rows = (n + cols - 1) // cols
            cell_w, cell_h = self.grid_cell_size
            grid_width = cols * cell_w
            bar_height = 40 if not self.enable_mcmot else 60
            grid_height = rows * cell_h + bar_height
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(record_output, fourcc, self.target_fps, (grid_width, grid_height))
            logger.info(f"🔴 Enregistrement: {record_output}")
        elif headless:
            logger.warning("Mode headless sans enregistrement - rien ne sera sauvegardé!")
        
        if headless:
            logger.info(f"\n▶️ Traitement headless démarré (vitesse max)")
        else:
            logger.info(f"\n▶️ Lecture démarrée (vitesse: {self.playback_speed}x)")
            logger.info("   Contrôles: Q=Quitter, ESPACE=Pause, +/-=Vitesse")
        
        playback_start = time.time()
        frame_count = 0
        last_progress = 0
        
        try:
            while self._running:
                loop_start = time.time()
                
                if not self._paused:
                    if headless:
                        # Mode headless: avancer frame par frame
                        elapsed_frames = frame_count / self.target_fps
                        self.current_time = self.global_start_time + timedelta(seconds=elapsed_frames)
                    else:
                        # Mode normal: temps réel
                        real_elapsed = (time.time() - playback_start) * self.playback_speed
                        self.current_time = self.global_start_time + timedelta(seconds=real_elapsed)
                    
                    # Vérifier si on a atteint la fin
                    if self.current_time >= self.global_end_time:
                        logger.info("\n✓ Fin de la lecture")
                        break
                    
                    # Vérifier max_frames
                    if max_frames and frame_count >= max_frames:
                        logger.info(f"\n✓ Limite de {max_frames} frames atteinte")
                        break
                
                # Récupérer les frames de chaque source
                frames = {}
                for camera_id, source in self.sources.items():
                    frames[camera_id] = self._get_frame_at_time(source, self.current_time)
                
                # Créer la grille
                grid = self._create_grid(frames)
                grid = self._add_timeline_overlay(grid)
                
                # Incrémenter le compteur de frames (pour frame skipping du tracking)
                if self.enable_tracking or self.enable_mcmot:
                    self._frame_counter += 1
                
                # Enregistrer si writer actif
                if writer and not self._paused:
                    writer.write(grid)
                
                frame_count += 1
                
                # Affichage
                if not headless:
                    cv2.imshow(window_name, grid)
                    
                    # Contrôle du framerate
                    elapsed = time.time() - loop_start
                    wait_time = max(1, int((frame_interval - elapsed) * 1000))
                    
                    key = cv2.waitKey(wait_time) & 0xFF
                    
                    if key == ord('q') or key == 27:
                        break
                    elif key == ord(' '):
                        self._paused = not self._paused
                        logger.info("⏸️ Pause" if self._paused else "▶️ Reprise")
                        if not self._paused:
                            elapsed_so_far = (self.current_time - self.global_start_time).total_seconds()
                            playback_start = time.time() - (elapsed_so_far / self.playback_speed)
                    elif key == ord('+') or key == ord('='):
                        self.playback_speed = min(10.0, self.playback_speed * 1.5)
                        elapsed_so_far = (self.current_time - self.global_start_time).total_seconds()
                        playback_start = time.time() - (elapsed_so_far / self.playback_speed)
                        logger.info(f"⏩ Vitesse: {self.playback_speed:.1f}x")
                    elif key == ord('-'):
                        self.playback_speed = max(0.1, self.playback_speed / 1.5)
                        elapsed_so_far = (self.current_time - self.global_start_time).total_seconds()
                        playback_start = time.time() - (elapsed_so_far / self.playback_speed)
                        logger.info(f"⏪ Vitesse: {self.playback_speed:.1f}x")
                    elif key == ord('s'):
                        filename = f"sync_screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                        cv2.imwrite(filename, grid)
                        logger.info(f"📷 Screenshot: {filename}")
                else:
                    # Mode headless: afficher progression
                    total_duration = (self.global_end_time - self.global_start_time).total_seconds()
                    elapsed = (self.current_time - self.global_start_time).total_seconds()
                    progress = int(elapsed / total_duration * 100)
                    
                    if progress >= last_progress + 5:  # Update every 5%
                        elapsed_time = time.time() - playback_start
                        fps_actual = frame_count / elapsed_time if elapsed_time > 0 else 0
                        logger.info(f"   Progression: {progress}% ({frame_count} frames, {fps_actual:.1f} fps)")
                        last_progress = progress
        
        finally:
            self._running = False
            if not headless:
                try:
                    cv2.destroyWindow(window_name)
                except:
                    pass
            
            # Fermer le writer si actif
            if writer:
                writer.release()
                logger.info(f"✓ Enregistrement terminé: {record_output}")
            
            # Fermer les captures
            for source in self.sources.values():
                if source.capture:
                    source.capture.release()
                    source.capture = None
            
            # Stats finales en mode headless
            if headless:
                elapsed_time = time.time() - playback_start
                logger.info(f"\n📊 Statistiques finales:")
                logger.info(f"   Frames traités: {frame_count}")
                logger.info(f"   Temps réel: {elapsed_time:.1f}s")
                logger.info(f"   FPS moyen: {frame_count / elapsed_time:.1f}")
                
                if self.enable_mcmot and self._mcmot_stats_cache:
                    stats = self._mcmot_stats_cache
                    logger.info(f"   MCMOT: {stats.get('total_global_tracks', 0)} objets, {stats.get('cross_camera_tracks', 0)} cross-cam")
    
    def stop(self):
        """Arrête la lecture."""
        self._running = False


def get_rotation_for_video(filepath: str) -> int:
    """Retourne la rotation à appliquer selon le fichier vidéo."""
    basename = Path(filepath).name.upper()
    if "DEVANTURE_PORTE_ENTREE" in basename:
        return 270  # -90°
    return 0


if __name__ == "__main__":
    import glob
    
    # Test avec les vidéos du dataset
    dataset_path = "/home/epl/DATA/S03/MTH2329/APEKE/PROJET/Dataset"
    videos = glob.glob(f"{dataset_path}/*.mp4") + glob.glob(f"{dataset_path}/*.MP4")
    
    if videos:
        player = TimeSyncedPlayer(
            target_fps=25.0,
            playback_speed=10.0,  # 10x pour le test
        )
        
        for video in videos:
            rotation = get_rotation_for_video(video)
            player.add_video(video, rotation_degrees=rotation)
        
        player.play()
