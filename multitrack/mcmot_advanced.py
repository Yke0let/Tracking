"""
Module MCMOT Avancé: Multi-Camera Multi-Object Tracking avec Ré-identification
et Apprentissage Adaptatif.

Ce module étend mcmot.py avec:
- Calibration 3D complète (cv2.calibrateCamera)
- Galerie adaptative pour reconnaissance cross-camera
- Features ReID améliorées (support TorchReID)
- Matching Hungarian optimisé
- Gestion avancée des occlusions
"""

import cv2
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Set
from pathlib import Path
from datetime import datetime
import logging
import time
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine
import threading

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
# CALIBRATION AVANCÉE
# =============================================================================

class AdvancedCameraCalibration:
    """Calibration avancée avec homographie et/ou calibration 3D complète."""
    
    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        
        # Homographie (projection planaire)
        self.homography: Optional[np.ndarray] = None
        
        # Calibration intrinsèque
        self.intrinsic: Optional[np.ndarray] = None  # Matrice K 3x3
        self.distortion: Optional[np.ndarray] = None  # Coefficients distorsion
        
        # Calibration extrinsèque  
        self.rotation: Optional[np.ndarray] = None   # Vecteur rotation 3x1
        self.translation: Optional[np.ndarray] = None  # Vecteur translation 3x1
        self.extrinsic: Optional[np.ndarray] = None  # Matrice [R|t] 3x4
        
        # Erreur de reprojection
        self.reprojection_error: float = 0.0
        
    def set_homography_from_points(
        self,
        image_points: List[Tuple[int, int]],
        ground_points: List[Tuple[float, float]],
        method: int = cv2.RANSAC,
    ):
        """
        Calcule l'homographie image -> sol à partir de correspondances.
        
        Args:
            image_points: Points dans l'image [(x1,y1), ...]
            ground_points: Points au sol en mètres [(X1,Y1), ...]
            method: cv2.RANSAC, cv2.LMEDS, ou 0 pour tous les points
        """
        if len(image_points) < 4:
            raise ValueError("Au moins 4 points requis pour l'homographie")
        
        src = np.array(image_points, dtype=np.float32)
        dst = np.array(ground_points, dtype=np.float32)
        
        self.homography, mask = cv2.findHomography(src, dst, method)
        
        # Calculer l'erreur
        inliers = np.sum(mask) if mask is not None else len(image_points)
        logger.info(f"✓ Homographie calculée pour {self.camera_id} ({inliers}/{len(image_points)} inliers)")
        
    def calibrate_camera_3d(
        self,
        object_points: List[np.ndarray],
        image_points: List[np.ndarray],
        image_size: Tuple[int, int],
        flags: int = 0,
    ):
        """
        Calibration 3D complète avec cv2.calibrateCamera.
        
        Args:
            object_points: Liste de tableaux de points 3D (damier)
            image_points: Liste de tableaux de points 2D détectés
            image_size: (width, height) de l'image
            flags: Flags OpenCV (ex: cv2.CALIB_FIX_ASPECT_RATIO)
        """
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            object_points, image_points, image_size, None, None, flags=flags
        )
        
        self.intrinsic = K
        self.distortion = dist
        self.reprojection_error = ret
        
        # Utiliser la première pose pour l'extrinsèque
        if rvecs and tvecs:
            self.rotation = rvecs[0]
            self.translation = tvecs[0]
            
            # Construire matrice extrinsèque [R|t]
            R, _ = cv2.Rodrigues(rvecs[0])
            self.extrinsic = np.hstack([R, tvecs[0]])
        
        logger.info(f"✓ Calibration 3D pour {self.camera_id} (erreur: {ret:.4f} px)")
        
    def undistort_point(self, point: Tuple[int, int]) -> Tuple[float, float]:
        """Corrige la distorsion d'un point."""
        if self.intrinsic is None or self.distortion is None:
            return (float(point[0]), float(point[1]))
        
        pt = np.array([[[point[0], point[1]]]], dtype=np.float32)
        undistorted = cv2.undistortPoints(pt, self.intrinsic, self.distortion, P=self.intrinsic)
        return (float(undistorted[0, 0, 0]), float(undistorted[0, 0, 1]))
    
    def project_to_ground(self, image_point: Tuple[int, int]) -> Tuple[float, float]:
        """Projette un point image vers le plan du sol."""
        # Corriger distorsion si disponible
        if self.intrinsic is not None and self.distortion is not None:
            image_point = self.undistort_point(image_point)
        
        if self.homography is not None:
            pt = np.array([[[image_point[0], image_point[1]]]], dtype=np.float32)
            projected = cv2.perspectiveTransform(pt, self.homography)
            return (float(projected[0, 0, 0]), float(projected[0, 0, 1]))
        
        return (float(image_point[0]), float(image_point[1]))
    
    def project_bbox_to_ground(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """Projette le centre bas du bbox (pieds) vers le sol."""
        x1, y1, x2, y2 = bbox
        foot_point = ((x1 + x2) // 2, y2)
        return self.project_to_ground(foot_point)


# =============================================================================
# GALERIE ADAPTATIVE POUR APPRENTISSAGE CROSS-CAMERA
# =============================================================================

@dataclass
class ObjectGallery:
    """Galerie de features pour un objet spécifique."""
    global_id: int
    class_name: str
    features: List[np.ndarray] = field(default_factory=list)
    cameras_seen: Set[str] = field(default_factory=set)
    first_seen: float = 0.0
    last_updated: float = 0.0
    total_observations: int = 0


class AdaptiveGallery:
    """
    Galerie adaptative pour apprentissage et reconnaissance cross-camera.
    
    Lorsqu'une caméra identifie un objet pour la première fois:
    1. Crée une galerie de features pour cet objet
    2. Met à jour la galerie à chaque observation
    3. Utilise la galerie pour reconnaissance dans d'autres caméras
    """
    
    def __init__(
        self,
        max_features_per_object: int = 20,
        similarity_threshold: float = 0.7,
        duplicate_threshold: float = 0.95,
    ):
        self.max_features = max_features_per_object
        self.similarity_threshold = similarity_threshold
        self.duplicate_threshold = duplicate_threshold
        
        self.galleries: Dict[int, ObjectGallery] = {}
        self._lock = threading.Lock()
        
    def add_observation(
        self,
        global_id: int,
        feature: np.ndarray,
        class_name: str,
        camera_id: str,
        timestamp: float,
    ):
        """Ajoute une observation à la galerie d'un objet."""
        with self._lock:
            if global_id not in self.galleries:
                self.galleries[global_id] = ObjectGallery(
                    global_id=global_id,
                    class_name=class_name,
                    first_seen=timestamp,
                )
            
            gallery = self.galleries[global_id]
            gallery.cameras_seen.add(camera_id)
            gallery.last_updated = timestamp
            gallery.total_observations += 1
            
            # Vérifier si pas duplicate
            if not self._is_duplicate(gallery, feature):
                gallery.features.append(feature.copy())
                
                # Pruning si trop de features
                if len(gallery.features) > self.max_features:
                    self._prune_gallery(gallery)
    
    def _is_duplicate(self, gallery: ObjectGallery, feature: np.ndarray) -> bool:
        """Vérifie si la feature est trop similaire à une existante."""
        for existing in gallery.features:
            sim = 1 - cosine(feature, existing)
            if sim > self.duplicate_threshold:
                return True
        return False
    
    def _prune_gallery(self, gallery: ObjectGallery):
        """Réduit la galerie en gardant les features les plus diversifiées."""
        if len(gallery.features) <= self.max_features:
            return
        
        # Garder les features les plus récentes et diversifiées
        # Stratégie: clustering k-medoids simplifié
        features = np.array(gallery.features)
        n = len(features)
        
        # Calculer matrice de distance
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                d = cosine(features[i], features[j])
                distances[i, j] = d
                distances[j, i] = d
        
        # Sélectionner les plus diversifiés (max distance aux autres)
        diversity_scores = np.sum(distances, axis=1)
        indices = np.argsort(-diversity_scores)[:self.max_features]
        
        gallery.features = [gallery.features[i] for i in sorted(indices)]
    
    def find_match(
        self,
        feature: np.ndarray,
        class_name: str,
        exclude_camera: Optional[str] = None,
    ) -> Tuple[Optional[int], float]:
        """
        Cherche un match dans toutes les galeries de même classe.
        
        Args:
            feature: Feature à matcher
            class_name: Classe de l'objet
            exclude_camera: Caméra à exclure (optionnel)
            
        Returns:
            (global_id, score) ou (None, 0.0) si pas de match
        """
        best_match = None
        best_score = 0.0
        
        with self._lock:
            for global_id, gallery in self.galleries.items():
                if gallery.class_name != class_name:
                    continue
                
                # Calculer la meilleure similarité avec la galerie
                for gal_feat in gallery.features:
                    sim = 1 - cosine(feature, gal_feat)
                    if sim > best_score and sim > self.similarity_threshold:
                        best_score = sim
                        best_match = global_id
        
        return best_match, best_score
    
    def get_mean_feature(self, global_id: int) -> Optional[np.ndarray]:
        """Retourne la feature moyenne d'un objet."""
        with self._lock:
            if global_id not in self.galleries:
                return None
            
            features = self.galleries[global_id].features
            if not features:
                return None
            
            mean = np.mean(features, axis=0)
            mean /= np.linalg.norm(mean) + 1e-8
            return mean
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques de la galerie."""
        with self._lock:
            return {
                "total_objects": len(self.galleries),
                "cross_camera_objects": sum(
                    1 for g in self.galleries.values()
                    if len(g.cameras_seen) > 1
                ),
                "total_features": sum(
                    len(g.features) for g in self.galleries.values()
                ),
                "avg_features_per_object": (
                    np.mean([len(g.features) for g in self.galleries.values()])
                    if self.galleries else 0
                ),
            }


# =============================================================================
# EXTRACTEUR REID AVANCÉ
# =============================================================================

class AdvancedReIDExtractor:
    """
    Extracteur de features ReID avec support multi-modèles.
    
    Supporte:
    - ResNet50 (ImageNet) - par défaut
    - OSNet (TorchReID) - si disponible
    - TransReID - si disponible
    """
    
    def __init__(
        self,
        model_name: str = "resnet50",
        device: str = "auto",
        batch_size: int = 8,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch requis pour ReID")
        
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"
        )
        self.batch_size = batch_size
        self.model_name = model_name
        
        # Initialiser le modèle
        self.model, self.feature_dim = self._load_model(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Transformations
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),  # Standard ReID size
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Cache pour éviter re-calcul
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_max_size = 1000
        
        logger.info(f"✓ ReID Extractor initialisé: {model_name} ({self.device})")
        
    def _load_model(self, model_name: str) -> Tuple[torch.nn.Module, int]:
        """Charge le modèle ReID."""
        if model_name == "resnet50":
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            model = torch.nn.Sequential(*list(model.children())[:-1])
            return model, 2048
        
        # Essayer TorchReID (OSNet)
        try:
            import torchreid
            model = torchreid.models.build_model(
                name="osnet_ain_x1_0",
                num_classes=1,
                pretrained=True,
            )
            return model, 512
        except ImportError:
            logger.warning("TorchReID non disponible, utilisation de ResNet50")
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            model = torch.nn.Sequential(*list(model.children())[:-1])
            return model, 2048
    
    def extract(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        use_cache: bool = True,
    ) -> np.ndarray:
        """Extrait la feature ReID pour un crop."""
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        
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
        
        # Normalisation L2
        feature = feature / (np.linalg.norm(feature) + 1e-8)
        
        return feature
    
    def extract_batch(
        self,
        image: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
    ) -> List[np.ndarray]:
        """Extrait les features pour plusieurs bboxes en batch."""
        if not bboxes:
            return []
        
        crops = []
        valid_indices = []
        
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
            
            if x2 > x1 and y2 > y1:
                crop = image[y1:y2, x1:x2]
                if crop.size > 0:
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    crops.append(self.transform(crop_rgb))
                    valid_indices.append(i)
        
        # Résultats par défaut
        results = [np.zeros(self.feature_dim) for _ in bboxes]
        
        if not crops:
            return results
        
        # Batch processing
        batch = torch.stack(crops).to(self.device)
        
        with torch.no_grad():
            features = self.model(batch)
            features = features.squeeze(-1).squeeze(-1).cpu().numpy()
        
        # Normalisation et assignation
        for i, idx in enumerate(valid_indices):
            feat = features[i] if len(features.shape) > 1 else features
            results[idx] = feat / (np.linalg.norm(feat) + 1e-8)
        
        return results
    
    @staticmethod
    def compute_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Calcule la similarité cosinus entre deux features."""
        if feat1 is None or feat2 is None:
            return 0.0
        return float(1 - cosine(feat1, feat2))


# =============================================================================
# DATA CLASSES AMÉLIORÉES
# =============================================================================

@dataclass
class LocalTrackAdvanced:
    """Track local avec informations étendues."""
    track_id: int
    camera_id: str
    class_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    confidence: float
    timestamp: float
    
    # ReID
    feature: Optional[np.ndarray] = None
    
    # Position BEV
    bev_position: Optional[Tuple[float, float]] = None
    
    # Métriques
    velocity: Optional[Tuple[float, float]] = None  # Vitesse en BEV
    area: int = 0  # Surface du bbox


@dataclass
class GlobalTrackAdvanced:
    """Track global avec galerie et historique complet."""
    global_id: int
    class_name: str
    
    # Associations locales {camera_id: local_track_id}
    local_tracks: Dict[str, int] = field(default_factory=dict)
    
    # Feature moyenne (EMA)
    mean_feature: Optional[np.ndarray] = None
    
    # Historique BEV [(x, y, timestamp), ...]
    bev_history: List[Tuple[float, float, float]] = field(default_factory=list)
    
    # Timestamps par caméra
    last_seen: Dict[str, float] = field(default_factory=dict)
    first_seen: float = 0.0
    total_detections: int = 0
    
    # Toutes les caméras qui ont vu cet objet (pour ré-identification)
    cameras_seen: Set[str] = field(default_factory=set)
    
    # Confiance du track
    confidence_score: float = 1.0
    is_confirmed: bool = False  # Confirmé après N détections
    
    # Vélocité estimée
    velocity: Optional[Tuple[float, float]] = None


# =============================================================================
# MCMOT TRACKER AVANCÉ
# =============================================================================

class MCMOTAdvancedTracker:
    """
    Multi-Camera Multi-Object Tracker Avancé.
    
    Fonctionnalités:
    - Galerie adaptative pour apprentissage cross-camera
    - Calibration 3D avec correction distorsion
    - Matching Hungarian optimisé
    - Gestion avancée des occlusions
    - Filtrage des faux positifs
    """
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        reid_threshold: float = 0.6,
        max_time_gap: float = 5.0,
        max_distance: float = 200.0,
        enable_reid: bool = True,
        min_detections_confirm: int = 3,
        feature_update_rate: float = 0.1,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch/Ultralytics requis")
        
        self.model = YOLO(model_path)
        self.reid_threshold = reid_threshold
        self.max_time_gap = max_time_gap
        self.max_distance = max_distance
        self.enable_reid = enable_reid
        self.min_detections_confirm = min_detections_confirm
        self.feature_update_rate = feature_update_rate
        
        # Extracteur ReID
        self.reid_extractor: Optional[AdvancedReIDExtractor] = None
        if enable_reid:
            try:
                self.reid_extractor = AdvancedReIDExtractor()
            except Exception as e:
                logger.warning(f"ReID désactivé: {e}")
                self.enable_reid = False
        
        # Galerie adaptative
        self.gallery = AdaptiveGallery(
            similarity_threshold=reid_threshold,
        )
        
        # Calibrations par caméra
        self.calibrations: Dict[str, AdvancedCameraCalibration] = {}
        
        # Tracks
        self.local_tracks: Dict[str, Dict[int, LocalTrackAdvanced]] = defaultdict(dict)
        self.global_tracks: Dict[int, GlobalTrackAdvanced] = {}
        self.next_global_id = 1
        self.local_to_global: Dict[Tuple[str, int], int] = {}
        
        # Classes de surveillance
        self.surveillance_classes = [0, 1, 2, 3, 24, 26, 28, 63, 67]
        
        # Trackers ByteTrack persistants par caméra
        self._bytetrack_states: Dict[str, Any] = {}
        
        logger.info(f"✓ MCMOT Advanced Tracker initialisé")
        logger.info(f"   ReID: {'activé' if enable_reid else 'désactivé'}")
        logger.info(f"   Seuil similarité: {reid_threshold}")
        logger.info(f"   Galerie adaptative: activée")
    
    def add_camera_calibration(
        self,
        camera_id: str,
        image_points: List[Tuple[int, int]],
        ground_points: List[Tuple[float, float]],
    ):
        """Ajoute la calibration homographie pour une caméra."""
        calib = AdvancedCameraCalibration(camera_id)
        calib.set_homography_from_points(image_points, ground_points)
        self.calibrations[camera_id] = calib
    
    def add_camera_calibration_3d(
        self,
        camera_id: str,
        object_points: List[np.ndarray],
        image_points: List[np.ndarray],
        image_size: Tuple[int, int],
    ):
        """Ajoute la calibration 3D complète pour une caméra."""
        if camera_id not in self.calibrations:
            self.calibrations[camera_id] = AdvancedCameraCalibration(camera_id)
        
        self.calibrations[camera_id].calibrate_camera_3d(
            object_points, image_points, image_size
        )
    
    def process_frame(
        self,
        camera_id: str,
        frame: np.ndarray,
        timestamp: float,
    ) -> List[LocalTrackAdvanced]:
        """Traite une frame d'une caméra."""
        # Détection + tracking local avec seuil de confiance élevé
        results = self.model.track(
            frame,
            conf=0.5,  # Augmenté de 0.35 pour réduire les faux positifs
            classes=self.surveillance_classes,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
        )
        
        local_tracks = []
        bboxes_for_reid = []
        track_indices = []
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                local_id = int(boxes.id[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                confidence = float(boxes.conf[i].cpu().numpy())
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                bbox = (x1, y1, x2, y2)
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                
                # Projeter vers BEV
                bev_pos = None
                if camera_id in self.calibrations:
                    bev_pos = self.calibrations[camera_id].project_bbox_to_ground(bbox)
                
                track = LocalTrackAdvanced(
                    track_id=local_id,
                    camera_id=camera_id,
                    class_id=class_id,
                    class_name=self.model.names.get(class_id, f"class_{class_id}"),
                    bbox=bbox,
                    center=center,
                    confidence=confidence,
                    timestamp=timestamp,
                    bev_position=bev_pos,
                    area=(x2 - x1) * (y2 - y1),
                )
                
                local_tracks.append(track)
                bboxes_for_reid.append(bbox)
                track_indices.append(len(local_tracks) - 1)
        
        # Extraction batch des features ReID
        if self.enable_reid and self.reid_extractor and bboxes_for_reid:
            features = self.reid_extractor.extract_batch(frame, bboxes_for_reid)
            for i, idx in enumerate(track_indices):
                local_tracks[idx].feature = features[i]
        
        # Sauvegarder les tracks locaux
        for track in local_tracks:
            self.local_tracks[camera_id][track.track_id] = track
        
        return local_tracks
    
    def associate_cross_camera(self, timestamp: float):
        """
        Associe les tracks entre caméras avec Hungarian algorithm et ré-identification.
        
        Gère:
        1. Association initiale: nouveau track -> nouveau global ID
        2. Cross-camera: même personne dans différentes caméras -> même ID
        3. Ré-entrée: personne qui revient dans une caméra -> retrouve son ID
        """
        # Collecter tous les tracks locaux récents non-associés
        unassociated_tracks: List[LocalTrackAdvanced] = []
        
        for camera_id, tracks in self.local_tracks.items():
            for local_id, track in tracks.items():
                if timestamp - track.timestamp > self.max_time_gap:
                    continue
                
                key = (track.camera_id, track.track_id)
                if key in self.local_to_global:
                    # Déjà associé - mettre à jour
                    global_id = self.local_to_global[key]
                    if global_id in self.global_tracks:
                        self._update_global_track(global_id, track, timestamp)
                else:
                    unassociated_tracks.append(track)
        
        if not unassociated_tracks:
            return
        
        # Pour chaque track non-associé, chercher une correspondance
        for track in unassociated_tracks:
            key = (track.camera_id, track.track_id)
            matched_global_id = None
            best_score = 0.0
            
            # 1. D'abord chercher dans la GALERIE (ré-identification)
            if self.enable_reid and track.feature is not None and track.class_name == "person":
                gallery_match, gallery_score = self.gallery.find_match(
                    track.feature, track.class_name
                )
                if gallery_match is not None and gallery_score > self.reid_threshold:
                    matched_global_id = gallery_match
                    best_score = gallery_score
                    logger.debug(f"ReID: {track.camera_id}:L{track.track_id} -> G{matched_global_id} (score={gallery_score:.2f})")
            
            # 2. Si pas de match galerie, chercher dans les tracks globaux actifs
            if matched_global_id is None:
                for global_id, gt in self.global_tracks.items():
                    if gt.class_name != track.class_name:
                        continue
                    
                    # Skip si cette caméra a déjà un autre track associé à ce global
                    if track.camera_id in gt.local_tracks:
                        existing_local = gt.local_tracks[track.camera_id]
                        if existing_local != track.track_id:
                            # Vérifier si l'ancien track est toujours actif
                            old_track = self.local_tracks.get(track.camera_id, {}).get(existing_local)
                            if old_track and timestamp - old_track.timestamp < 2.0:
                                continue  # L'ancien track est encore actif
                    
                    # Calculer le score de matching
                    score = self._compute_match_score(track, gt)
                    if score > best_score and score > 0.5:
                        best_score = score
                        matched_global_id = global_id
            
            # 3. Associer ou créer
            if matched_global_id is not None:
                self.local_to_global[key] = matched_global_id
                self._update_global_track(matched_global_id, track, timestamp)
                
                # Log pour debug
                gt = self.global_tracks[matched_global_id]
                if len(gt.local_tracks) > 1 or track.camera_id in gt.cameras_seen:
                    pass  # Ré-identification réussie
            else:
                # Nouveau track global
                self._create_global_track(track, timestamp)
    
    def _compute_match_score(
        self,
        local: LocalTrackAdvanced,
        global_track: GlobalTrackAdvanced,
    ) -> float:
        """
        Calcule un score de matching (0-1) entre un track local et global.
        Plus le score est élevé, meilleur est le match.
        """
        scores = []
        weights = []
        
        # 1. Score ReID (le plus important pour les personnes)
        if self.enable_reid and local.feature is not None and global_track.mean_feature is not None:
            reid_sim = AdvancedReIDExtractor.compute_similarity(
                local.feature, global_track.mean_feature
            )
            scores.append(reid_sim)
            weights.append(3.0 if local.class_name == "person" else 1.0)
        
        # 2. Score de distance BEV (si calibration disponible)
        if local.bev_position and global_track.bev_history:
            last_bev = global_track.bev_history[-1][:2]
            dist = np.sqrt((local.bev_position[0] - last_bev[0])**2 + 
                          (local.bev_position[1] - last_bev[1])**2)
            # Normaliser: 0 = loin, 1 = proche
            dist_score = max(0, 1 - dist / self.max_distance)
            scores.append(dist_score)
            weights.append(1.0)
        
        # 3. Score temporel (récemment vu = meilleur)
        if global_track.last_seen:
            last_time = max(global_track.last_seen.values())
            time_gap = abs(local.timestamp - last_time)
            time_score = max(0, 1 - time_gap / self.max_time_gap)
            scores.append(time_score)
            weights.append(0.5)
        
        if not scores:
            return 0.0
        
        # Moyenne pondérée
        return sum(s * w for s, w in zip(scores, weights)) / sum(weights)
    
    def _compute_cost_matrix(
        self,
        local_tracks: List[LocalTrackAdvanced],
        global_ids: List[int],
    ) -> np.ndarray:
        """Calcule la matrice de coût pour Hungarian algorithm."""
        n_local = len(local_tracks)
        n_global = len(global_ids)
        cost_matrix = np.ones((n_local, n_global)) * 1000
        
        for i, local in enumerate(local_tracks):
            for j, gid in enumerate(global_ids):
                global_track = self.global_tracks[gid]
                
                # Skip si même caméra déjà associée
                if local.camera_id in global_track.local_tracks:
                    continue
                
                # Skip si classes différentes
                if local.class_name != global_track.class_name:
                    continue
                
                cost = self._compute_match_cost(local, global_track)
                cost_matrix[i, j] = cost
        
        return cost_matrix
    
    def _compute_match_cost(
        self,
        local: LocalTrackAdvanced,
        global_track: GlobalTrackAdvanced,
    ) -> float:
        """Calcule le coût de matching entre track local et global."""
        cost = 0.0
        n_factors = 0
        
        # 1. Coût ReID (similarité features)
        if self.enable_reid and local.feature is not None:
            # Chercher dans la galerie
            gallery_match, gallery_score = self.gallery.find_match(
                local.feature, local.class_name
            )
            
            if gallery_match == global_track.global_id:
                # Match avec la galerie
                reid_cost = 1 - gallery_score
            elif global_track.mean_feature is not None:
                # Match avec feature moyenne
                sim = AdvancedReIDExtractor.compute_similarity(
                    local.feature, global_track.mean_feature
                )
                reid_cost = 1 - sim
            else:
                reid_cost = 1.0
            
            cost += reid_cost * 2.0  # Poids double pour ReID
            n_factors += 2
        
        # 2. Coût distance BEV
        if local.bev_position and global_track.bev_history:
            last_bev = global_track.bev_history[-1][:2]
            dist = np.sqrt(
                (local.bev_position[0] - last_bev[0])**2 +
                (local.bev_position[1] - last_bev[1])**2
            )
            dist_cost = min(1.0, dist / self.max_distance)
            cost += dist_cost
            n_factors += 1
        
        return cost / n_factors if n_factors > 0 else 1000
    
    def _create_global_track(self, local: LocalTrackAdvanced, timestamp: float):
        """Crée un nouveau track global."""
        global_id = self.next_global_id
        self.next_global_id += 1
        
        global_track = GlobalTrackAdvanced(
            global_id=global_id,
            class_name=local.class_name,
            local_tracks={local.camera_id: local.track_id},
            mean_feature=local.feature.copy() if local.feature is not None else None,
            first_seen=timestamp,
            total_detections=1,
            cameras_seen={local.camera_id},  # Ajouter la première caméra
        )
        
        if local.bev_position:
            global_track.bev_history.append(
                (local.bev_position[0], local.bev_position[1], timestamp)
            )
        
        global_track.last_seen[local.camera_id] = timestamp
        
        self.global_tracks[global_id] = global_track
        self.local_to_global[(local.camera_id, local.track_id)] = global_id
        
        # Ajouter à la galerie adaptative
        if local.feature is not None:
            self.gallery.add_observation(
                global_id, local.feature, local.class_name,
                local.camera_id, timestamp
            )
    
    def _update_global_track(
        self,
        global_id: int,
        local: LocalTrackAdvanced,
        timestamp: float,
    ):
        """Met à jour un track global."""
        gt = self.global_tracks[global_id]
        
        gt.local_tracks[local.camera_id] = local.track_id
        gt.last_seen[local.camera_id] = timestamp
        gt.cameras_seen.add(local.camera_id)  # Marquer cette caméra comme ayant vu l'objet
        gt.total_detections += 1
        
        # Confirmer après N détections
        if gt.total_detections >= self.min_detections_confirm:
            gt.is_confirmed = True
        
        # Mettre à jour feature (EMA)
        if local.feature is not None:
            if gt.mean_feature is None:
                gt.mean_feature = local.feature.copy()
            else:
                alpha = self.feature_update_rate
                gt.mean_feature = (1 - alpha) * gt.mean_feature + alpha * local.feature
                gt.mean_feature /= np.linalg.norm(gt.mean_feature) + 1e-8
            
            # Ajouter à la galerie
            self.gallery.add_observation(
                global_id, local.feature, local.class_name,
                local.camera_id, timestamp
            )
        
        # Ajouter position BEV
        if local.bev_position:
            gt.bev_history.append(
                (local.bev_position[0], local.bev_position[1], timestamp)
            )
            gt.bev_history = gt.bev_history[-30:]  # Réduit pour économiser la mémoire
            
            # Calculer vélocité
            if len(gt.bev_history) >= 2:
                p1 = gt.bev_history[-2]
                p2 = gt.bev_history[-1]
                dt = p2[2] - p1[2]
                if dt > 0:
                    gt.velocity = ((p2[0] - p1[0]) / dt, (p2[1] - p1[1]) / dt)
    
    def get_global_id(self, camera_id: str, local_id: int) -> Optional[int]:
        """Retourne l'ID global pour un track local."""
        return self.local_to_global.get((camera_id, local_id))
    
    def filter_false_positives(
        self,
        tracks: List[LocalTrackAdvanced],
        min_confidence: float = 0.5,
    ) -> List[LocalTrackAdvanced]:
        """Filtre les faux positifs."""
        filtered = []
        for track in tracks:
            key = (track.camera_id, track.track_id)
            global_id = self.local_to_global.get(key)
            
            if global_id:
                gt = self.global_tracks[global_id]
                if gt.is_confirmed:
                    filtered.append(track)
            elif track.confidence >= min_confidence:
                filtered.append(track)
        
        return filtered
    
    def draw_with_global_ids(
        self,
        camera_id: str,
        frame: np.ndarray,
        local_tracks: List[LocalTrackAdvanced],
    ) -> np.ndarray:
        """Dessine les bboxes avec les IDs globaux."""
        import colorsys
        annotated = frame.copy()
        
        for track in local_tracks:
            global_id = self.get_global_id(camera_id, track.track_id)
            x1, y1, x2, y2 = track.bbox
            
            if global_id:
                gt = self.global_tracks.get(global_id)
                hue = (global_id * 0.618) % 1.0
                rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.95)
                color = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
                
                # Indicateur multi-caméra
                multi_cam = len(gt.local_tracks) > 1 if gt else False
                label = f"G{global_id} ({track.class_name})"
                if multi_cam:
                    label += " ★"  # Étoile pour multi-camera
            else:
                color = (128, 128, 128)
                label = f"L{track.track_id}"
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - h - 6), (x1 + w + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques du tracker."""
        gallery_stats = self.gallery.get_statistics()
        
        return {
            "total_global_tracks": len(self.global_tracks),
            "confirmed_tracks": sum(1 for gt in self.global_tracks.values() if gt.is_confirmed),
            "active_cameras": len(self.local_tracks),
            "cross_camera_tracks": sum(
                1 for gt in self.global_tracks.values()
                if len(gt.local_tracks) > 1
            ),
            "gallery": gallery_stats,
        }
    
    def cleanup_memory(self, current_timestamp: float, max_age: float = 30.0):
        """
        Nettoie la mémoire en supprimant les tracks et données obsolètes.
        
        Args:
            current_timestamp: Timestamp actuel
            max_age: Âge max en secondes pour garder les données
        """
        import gc
        
        # 1. Nettoyer les tracks locaux trop vieux
        stale_local_keys = []
        for camera_id, tracks in self.local_tracks.items():
            for local_id, track in list(tracks.items()):
                if current_timestamp - track.timestamp > max_age:
                    del tracks[local_id]
                    stale_local_keys.append((camera_id, local_id))
        
        # 2. Nettoyer les mappings local->global obsolètes
        for key in stale_local_keys:
            if key in self.local_to_global:
                del self.local_to_global[key]
        
        # 3. Nettoyer les tracks globaux non vus depuis longtemps
        stale_global_ids = []
        for global_id, gt in list(self.global_tracks.items()):
            # Vérifier si vu récemment dans au moins une caméra
            last_seen_any = max(gt.last_seen.values()) if gt.last_seen else 0
            if current_timestamp - last_seen_any > max_age * 2:
                stale_global_ids.append(global_id)
        
        for gid in stale_global_ids:
            del self.global_tracks[gid]
            # Nettoyer la galerie aussi
            if gid in self.gallery.galleries:
                del self.gallery.galleries[gid]
        
        # 4. Réduire historique BEV des tracks restants
        for gt in self.global_tracks.values():
            if len(gt.bev_history) > 30:  # Réduit de 100 à 30
                gt.bev_history = gt.bev_history[-30:]
        
        # 5. Limiter les features dans les galeries
        for gallery in self.gallery.galleries.values():
            if len(gallery.features) > 10:  # Réduit de 20 à 10
                gallery.features = gallery.features[-10:]
        
        # 6. Forcer le garbage collector
        gc.collect()
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Retourne une estimation de l'utilisation mémoire."""
        import sys
        
        local_tracks_count = sum(len(t) for t in self.local_tracks.values())
        global_tracks_count = len(self.global_tracks)
        gallery_features = sum(len(g.features) for g in self.gallery.galleries.values())
        bev_history_total = sum(len(gt.bev_history) for gt in self.global_tracks.values())
        
        return {
            "local_tracks": local_tracks_count,
            "global_tracks": global_tracks_count,
            "gallery_features": gallery_features,
            "bev_history_points": bev_history_total,
            "local_to_global_mappings": len(self.local_to_global),
        }


# =============================================================================
# VISUALISEUR BEV AMÉLIORÉ
# =============================================================================

class AdvancedBEVVisualizer:
    """Visualiseur BEV avec trajectoires et informations étendues."""
    
    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        scale: float = 1.0,
        offset: Tuple[float, float] = (400, 300),
        background_image: Optional[np.ndarray] = None,
    ):
        self.width = width
        self.height = height
        self.scale = scale
        self.offset = offset
        self.background = background_image
        
    def world_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """Convertit coordonnées monde en pixels."""
        px = int(x * self.scale + self.offset[0])
        py = int(y * self.scale + self.offset[1])
        return (px, py)
    
    def draw_bev(
        self,
        global_tracks: Dict[int, GlobalTrackAdvanced],
        show_trajectories: bool = True,
        show_velocities: bool = True,
    ) -> np.ndarray:
        """Dessine la vue BEV."""
        import colorsys
        
        if self.background is not None:
            bev = cv2.resize(self.background, (self.width, self.height))
        else:
            bev = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            bev[:] = (50, 50, 50)
        
        for global_id, track in global_tracks.items():
            if not track.bev_history:
                continue
            
            # Couleur du track
            hue = (global_id * 0.618) % 1.0
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.95)
            color = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            
            # Trajectoire
            if show_trajectories and len(track.bev_history) >= 2:
                points = [self.world_to_pixel(x, y) for x, y, _ in track.bev_history]
                points = [(p[0], p[1]) for p in points 
                         if 0 <= p[0] < self.width and 0 <= p[1] < self.height]
                
                for i in range(1, len(points)):
                    cv2.line(bev, points[i-1], points[i], color, 2)
            
            # Position actuelle
            last_pos = track.bev_history[-1]
            px, py = self.world_to_pixel(last_pos[0], last_pos[1])
            
            if 0 <= px < self.width and 0 <= py < self.height:
                # Cercle
                radius = 10 if track.is_confirmed else 6
                cv2.circle(bev, (px, py), radius, color, -1)
                
                # Label
                label = f"G{global_id}"
                if len(track.local_tracks) > 1:
                    label += f" ({len(track.local_tracks)}cam)"
                cv2.putText(bev, label, (px + 12, py),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Vélocité
                if show_velocities and track.velocity:
                    vx, vy = track.velocity
                    end_x = px + int(vx * 20)
                    end_y = py + int(vy * 20)
                    cv2.arrowedLine(bev, (px, py), (end_x, end_y), color, 2)
        
        return bev


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Test du module
    print("=== Test MCMOT Advanced ===\n")
    
    try:
        tracker = MCMOTAdvancedTracker(enable_reid=False)
        print(f"✓ Tracker créé")
        print(f"  Stats: {tracker.get_statistics()}")
        
        gallery = AdaptiveGallery()
        print(f"\n✓ Galerie adaptative créée")
        print(f"  Stats: {gallery.get_statistics()}")
        
        print("\n=== Test réussi ===")
    except Exception as e:
        print(f"✗ Erreur: {e}")
