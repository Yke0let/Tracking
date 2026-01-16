"""
Module MCMOT: Multi-Camera Multi-Object Tracking avec Ré-identification.
Permet de maintenir des IDs cohérents pour les mêmes objets vus sur différentes caméras.
"""

import cv2
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
import logging
import time
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    import torch
    import torchvision.transforms as T
    from torchvision.models import resnet50, ResNet50_Weights
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch/Ultralytics non disponibles")


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class LocalTrack:
    """Track local à une caméra."""
    track_id: int
    camera_id: str
    class_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    confidence: float
    feature: Optional[np.ndarray] = None  # ReID feature vector
    timestamp: float = 0.0
    
    # Position dans le plan commun (Bird's Eye View)
    bev_position: Optional[Tuple[float, float]] = None


@dataclass
class GlobalTrack:
    """Track global unifié à travers les caméras."""
    global_id: int
    class_name: str
    
    # Tracks locaux associés {camera_id: local_track_id}
    local_tracks: Dict[str, int] = field(default_factory=dict)
    
    # Feature moyenne pour ReID
    mean_feature: Optional[np.ndarray] = None
    
    # Historique des positions BEV
    bev_history: List[Tuple[float, float, float]] = field(default_factory=list)  # x, y, timestamp
    
    # Dernière position connue par caméra
    last_seen: Dict[str, float] = field(default_factory=dict)
    
    first_seen: float = 0.0
    total_detections: int = 0


# =============================================================================
# CAMERA CALIBRATION
# =============================================================================

class CameraCalibration:
    """Gère la calibration des caméras pour projection BEV."""
    
    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self.homography: Optional[np.ndarray] = None  # 3x3 homography matrix
        self.intrinsic: Optional[np.ndarray] = None   # 3x3 camera matrix
        self.distortion: Optional[np.ndarray] = None  # Distortion coefficients
        self.extrinsic: Optional[np.ndarray] = None   # 4x4 [R|t] matrix
    
    def set_homography_from_points(
        self,
        image_points: List[Tuple[int, int]],
        ground_points: List[Tuple[float, float]],
    ):
        """
        Calcule l'homographie à partir de correspondances image-sol.
        
        Args:
            image_points: Points dans l'image [(x1,y1), (x2,y2), ...]
            ground_points: Points correspondants au sol [(X1,Y1), (X2,Y2), ...]
        """
        src = np.array(image_points, dtype=np.float32)
        dst = np.array(ground_points, dtype=np.float32)
        
        self.homography, _ = cv2.findHomography(src, dst)
        logger.info(f"✓ Homographie calculée pour {self.camera_id}")
    
    def project_to_ground(self, image_point: Tuple[int, int]) -> Tuple[float, float]:
        """Projette un point image vers le plan du sol."""
        if self.homography is None:
            return (float(image_point[0]), float(image_point[1]))
        
        pt = np.array([[[image_point[0], image_point[1]]]], dtype=np.float32)
        projected = cv2.perspectiveTransform(pt, self.homography)
        return (float(projected[0, 0, 0]), float(projected[0, 0, 1]))
    
    def project_bbox_center_to_ground(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """Projette le centre bas du bbox (pieds) vers le sol."""
        x1, y1, x2, y2 = bbox
        foot_point = ((x1 + x2) // 2, y2)  # Centre bas du bbox
        return self.project_to_ground(foot_point)


# =============================================================================
# ReID FEATURE EXTRACTOR
# =============================================================================

class ReIDFeatureExtractor:
    """Extracteur de features pour la ré-identification."""
    
    def __init__(self, model_name: str = "resnet50", device: str = "auto"):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch requis pour ReID")
        
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"
        )
        
        # Charger le modèle ResNet50 comme extracteur de features
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])  # Remove last FC
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Transformations
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.feature_dim = 2048
        logger.info(f"✓ ReID Feature Extractor initialisé ({self.device})")
    
    def extract(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Extrait la feature ReID pour un crop."""
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(image.shape[1], x2), min(image.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros(self.feature_dim)
        
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(self.feature_dim)
        
        # BGR to RGB
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # Transform
        tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
        
        # Extract
        with torch.no_grad():
            feature = self.model(tensor)
            feature = feature.squeeze().cpu().numpy()
        
        # Normalize
        feature = feature / (np.linalg.norm(feature) + 1e-8)
        
        return feature
    
    def compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Calcule la similarité cosinus entre deux features."""
        if feat1 is None or feat2 is None:
            return 0.0
        return float(1 - cosine(feat1, feat2))


# =============================================================================
# MCMOT TRACKER
# =============================================================================

class MCMOTTracker:
    """
    Multi-Camera Multi-Object Tracker.
    Maintient des IDs globaux cohérents à travers les caméras.
    """
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        reid_threshold: float = 0.6,
        max_time_gap: float = 5.0,
        max_distance: float = 200.0,
        enable_reid: bool = True,
    ):
        """
        Args:
            model_path: Modèle YOLO
            reid_threshold: Seuil de similarité ReID pour matching
            max_time_gap: Gap temporel max (secondes) pour réassocier
            max_distance: Distance BEV max pour matching
            enable_reid: Activer l'extraction ReID
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch/Ultralytics requis")
        
        self.model = YOLO(model_path)
        self.reid_threshold = reid_threshold
        self.max_time_gap = max_time_gap
        self.max_distance = max_distance
        self.enable_reid = enable_reid
        
        # Extracteur ReID
        self.reid_extractor = None
        if enable_reid:
            try:
                self.reid_extractor = ReIDFeatureExtractor()
            except Exception as e:
                logger.warning(f"ReID désactivé: {e}")
                self.enable_reid = False
        
        # Calibrations par caméra
        self.calibrations: Dict[str, CameraCalibration] = {}
        
        # Tracks locaux par caméra {camera_id: {local_id: LocalTrack}}
        self.local_tracks: Dict[str, Dict[int, LocalTrack]] = defaultdict(dict)
        
        # Tracks globaux {global_id: GlobalTrack}
        self.global_tracks: Dict[int, GlobalTrack] = {}
        self.next_global_id = 1
        
        # Mapping local -> global {(camera_id, local_id): global_id}
        self.local_to_global: Dict[Tuple[str, int], int] = {}
        
        # Classes de surveillance
        self.surveillance_classes = [0, 1, 2, 3, 24, 26, 28, 63, 67]
        
        logger.info(f"✓ MCMOT Tracker initialisé")
        logger.info(f"   ReID: {'activé' if enable_reid else 'désactivé'}")
        logger.info(f"   Seuil similarité: {reid_threshold}")
    
    def add_camera_calibration(
        self,
        camera_id: str,
        image_points: List[Tuple[int, int]],
        ground_points: List[Tuple[float, float]],
    ):
        """Ajoute la calibration pour une caméra."""
        calib = CameraCalibration(camera_id)
        calib.set_homography_from_points(image_points, ground_points)
        self.calibrations[camera_id] = calib
    
    def process_frame(
        self,
        camera_id: str,
        frame: np.ndarray,
        timestamp: float,
    ) -> List[LocalTrack]:
        """
        Traite une frame d'une caméra.
        
        Args:
            camera_id: ID de la caméra
            frame: Image BGR
            timestamp: Timestamp actuel
            
        Returns:
            Liste des LocalTrack détectés
        """
        # Détection + tracking local avec seuil bas pour meilleure persistance
        results = self.model.track(
            frame,
            conf=0.35,  # Seuil équilibré
            classes=self.surveillance_classes,
            tracker="bytetrack.yaml",  # ByteTrack plus rapide
            persist=True,
            verbose=False,
        )
        
        local_tracks = []
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                local_id = int(boxes.id[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                confidence = float(boxes.conf[i].cpu().numpy())
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                bbox = (x1, y1, x2, y2)
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                
                # Extraire feature ReID
                feature = None
                if self.enable_reid and self.reid_extractor:
                    feature = self.reid_extractor.extract(frame, bbox)
                
                # Projeter vers BEV
                bev_pos = None
                if camera_id in self.calibrations:
                    bev_pos = self.calibrations[camera_id].project_bbox_center_to_ground(bbox)
                
                track = LocalTrack(
                    track_id=local_id,
                    camera_id=camera_id,
                    class_id=class_id,
                    class_name=self.model.names.get(class_id, f"class_{class_id}"),
                    bbox=bbox,
                    center=center,
                    confidence=confidence,
                    feature=feature,
                    timestamp=timestamp,
                    bev_position=bev_pos,
                )
                
                local_tracks.append(track)
                self.local_tracks[camera_id][local_id] = track
        
        return local_tracks
    
    def associate_cross_camera(self, timestamp: float):
        """
        Associe les tracks entre caméras pour attribuer des IDs globaux.
        Utilise ReID features + distance BEV + contraintes temporelles.
        """
        # Collecter tous les tracks locaux récents
        recent_tracks: List[LocalTrack] = []
        for camera_id, tracks in self.local_tracks.items():
            for local_id, track in tracks.items():
                if timestamp - track.timestamp < self.max_time_gap:
                    recent_tracks.append(track)
        
        if not recent_tracks:
            return
        
        # Pour chaque track local, chercher correspondance globale
        for track in recent_tracks:
            key = (track.camera_id, track.track_id)
            
            if key in self.local_to_global:
                # Déjà associé
                global_id = self.local_to_global[key]
                self._update_global_track(global_id, track)
            else:
                # Chercher correspondance par ReID
                best_global = self._find_best_global_match(track)
                
                if best_global is not None:
                    # Associer au track global existant
                    self.local_to_global[key] = best_global
                    self._update_global_track(best_global, track)
                else:
                    # Créer nouveau track global
                    self._create_global_track(track)
    
    def _find_best_global_match(self, local_track: LocalTrack) -> Optional[int]:
        """Trouve le meilleur match global pour un track local."""
        best_global_id = None
        best_score = 0.0
        
        for global_id, global_track in self.global_tracks.items():
            # Vérifier si pas déjà associé à cette caméra
            if local_track.camera_id in global_track.local_tracks:
                continue
            
            # Même classe
            if global_track.class_name != local_track.class_name:
                continue
            
            score = self._compute_match_score(local_track, global_track)
            
            if score > best_score and score > self.reid_threshold:
                best_score = score
                best_global_id = global_id
        
        return best_global_id
    
    def _compute_match_score(self, local_track: LocalTrack, global_track: GlobalTrack) -> float:
        """Calcule un score de correspondance entre track local et global."""
        score = 0.0
        n_factors = 0
        
        # Score ReID (similarité features)
        if self.enable_reid and local_track.feature is not None and global_track.mean_feature is not None:
            reid_score = self.reid_extractor.compute_similarity(
                local_track.feature, global_track.mean_feature
            )
            score += reid_score * 2.0  # Poids double pour ReID
            n_factors += 2
        
        # Score distance BEV (si calibration disponible)
        if local_track.bev_position and global_track.bev_history:
            last_bev = global_track.bev_history[-1][:2]
            dist = np.sqrt(
                (local_track.bev_position[0] - last_bev[0])**2 +
                (local_track.bev_position[1] - last_bev[1])**2
            )
            dist_score = max(0, 1 - dist / self.max_distance)
            score += dist_score
            n_factors += 1
        
        return score / n_factors if n_factors > 0 else 0.0
    
    def _create_global_track(self, local_track: LocalTrack):
        """Crée un nouveau track global."""
        global_id = self.next_global_id
        self.next_global_id += 1
        
        global_track = GlobalTrack(
            global_id=global_id,
            class_name=local_track.class_name,
            local_tracks={local_track.camera_id: local_track.track_id},
            mean_feature=local_track.feature.copy() if local_track.feature is not None else None,
            first_seen=local_track.timestamp,
            total_detections=1,
        )
        
        if local_track.bev_position:
            global_track.bev_history.append(
                (local_track.bev_position[0], local_track.bev_position[1], local_track.timestamp)
            )
        
        global_track.last_seen[local_track.camera_id] = local_track.timestamp
        
        self.global_tracks[global_id] = global_track
        self.local_to_global[(local_track.camera_id, local_track.track_id)] = global_id
    
    def _update_global_track(self, global_id: int, local_track: LocalTrack):
        """Met à jour un track global avec un nouveau track local."""
        global_track = self.global_tracks[global_id]
        
        global_track.local_tracks[local_track.camera_id] = local_track.track_id
        global_track.last_seen[local_track.camera_id] = local_track.timestamp
        global_track.total_detections += 1
        
        # Mettre à jour la feature moyenne (moving average)
        if local_track.feature is not None:
            if global_track.mean_feature is None:
                global_track.mean_feature = local_track.feature.copy()
            else:
                alpha = 0.1  # Taux de mise à jour
                global_track.mean_feature = (
                    (1 - alpha) * global_track.mean_feature + alpha * local_track.feature
                )
                global_track.mean_feature /= np.linalg.norm(global_track.mean_feature) + 1e-8
        
        # Ajouter position BEV
        if local_track.bev_position:
            global_track.bev_history.append(
                (local_track.bev_position[0], local_track.bev_position[1], local_track.timestamp)
            )
            # Garder seulement les 100 dernières positions
            if len(global_track.bev_history) > 100:
                global_track.bev_history = global_track.bev_history[-100:]
    
    def get_global_id(self, camera_id: str, local_id: int) -> Optional[int]:
        """Retourne l'ID global pour un track local."""
        return self.local_to_global.get((camera_id, local_id))
    
    def draw_with_global_ids(
        self,
        camera_id: str,
        frame: np.ndarray,
        local_tracks: List[LocalTrack],
    ) -> np.ndarray:
        """Dessine les bboxes avec les IDs globaux."""
        annotated = frame.copy()
        
        for track in local_tracks:
            global_id = self.get_global_id(camera_id, track.track_id)
            
            x1, y1, x2, y2 = track.bbox
            
            # Couleur basée sur l'ID global
            if global_id:
                hue = (global_id * 0.618) % 1.0
                import colorsys
                rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.95)
                color = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
                label = f"G{global_id} ({track.class_name})"
            else:
                color = (128, 128, 128)
                label = f"L{track.track_id}"
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - label_h - 6), (x1 + label_w + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques du tracker."""
        return {
            "total_global_tracks": len(self.global_tracks),
            "active_cameras": len(self.local_tracks),
            "local_tracks_per_camera": {
                cam: len(tracks) for cam, tracks in self.local_tracks.items()
            },
            "cross_camera_tracks": sum(
                1 for gt in self.global_tracks.values()
                if len(gt.local_tracks) > 1
            ),
        }


# =============================================================================
# BIRD'S EYE VIEW VISUALIZER
# =============================================================================

class BEVVisualizer:
    """Visualise les tracks sur un plan vue de dessus (Bird's Eye View)."""
    
    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        scale: float = 1.0,
        offset: Tuple[float, float] = (400, 300),
    ):
        self.width = width
        self.height = height
        self.scale = scale
        self.offset = offset
    
    def draw_bev(self, global_tracks: Dict[int, GlobalTrack]) -> np.ndarray:
        """Dessine la vue BEV des tracks globaux."""
        bev = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        bev[:] = (50, 50, 50)  # Fond gris
        
        for global_id, track in global_tracks.items():
            if not track.bev_history:
                continue
            
            # Couleur du track
            import colorsys
            hue = (global_id * 0.618) % 1.0
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.95)
            color = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            
            # Dessiner la trajectoire
            points = []
            for x, y, _ in track.bev_history:
                px = int(x * self.scale + self.offset[0])
                py = int(y * self.scale + self.offset[1])
                if 0 <= px < self.width and 0 <= py < self.height:
                    points.append((px, py))
            
            if len(points) >= 2:
                for i in range(1, len(points)):
                    cv2.line(bev, points[i-1], points[i], color, 2)
            
            # Position actuelle
            if points:
                cv2.circle(bev, points[-1], 8, color, -1)
                cv2.putText(bev, f"G{global_id}", (points[-1][0] + 10, points[-1][1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return bev


if __name__ == "__main__":
    # Test MCMOT
    tracker = MCMOTTracker(enable_reid=False)  # Désactiver ReID pour test rapide
    
    print("MCMOT Tracker créé avec succès")
    print(f"Stats: {tracker.get_statistics()}")
