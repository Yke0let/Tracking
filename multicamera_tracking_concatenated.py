#!/usr/bin/env python3
"""
=============================================================================
SYSTÈME DE TRACKING MULTI-CAMÉRAS À PARTIR D'UNE VIDÉO CONCATÉNÉE
=============================================================================

Ce script traite une vidéo contenant plusieurs vues de caméras de surveillance
concaténées horizontalement et effectue:

1. Détection YOLOv8 des classes: personne, sac à dos, sac à main, casque, moto, voiture
2. Tracking multi-objets avec ByteTrack/BoT-SORT pour IDs persistants
3. Ré-identification (ReID) cross-caméra avec ResNet50/OSNet
4. Association objets-personnes via IoU et/ou pose estimation MediaPipe
5. Visualisation temps réel et export vidéo annotée

Installation des dépendances:
    pip install ultralytics>=8.0.0 opencv-python>=4.8.0 torch>=2.0.0 torchvision>=0.15.0
    pip install mediapipe>=0.10.0  # Optionnel: pour pose estimation
    pip install torchreid  # Optionnel: pour OSNet (meilleur ReID)
    
    # GPU CUDA (si pas déjà installé):
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

Usage:
    python multicamera_tracking_concatenated.py --input reseau_final.mp4 --output output_tracked.mp4 --show
    python multicamera_tracking_concatenated.py --input reseau_final.mp4 --no-reid --max-frames 500

Auteur: Système de surveillance multi-caméras
Date: Janvier 2026
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import time
import argparse
import logging
import os
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from pathlib import Path

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# IMPORTS CONDITIONNELS
# =============================================================================

# YOLOv8 (Ultralytics)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.error("Ultralytics non installé. pip install ultralytics")

# TorchVision pour ReID
try:
    import torchvision.transforms as T
    from torchvision.models import resnet50, ResNet50_Weights
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    logger.warning("TorchVision non installé. pip install torchvision")

# MediaPipe pour pose estimation (optionnel)
# Note: MediaPipe >= 0.10.x peut utiliser l'API tasks au lieu de solutions
try:
    import mediapipe as mp
    # Vérifier si l'ancienne API solutions est disponible
    if hasattr(mp, 'solutions'):
        MEDIAPIPE_AVAILABLE = True
    else:
        MEDIAPIPE_AVAILABLE = False
        logger.info("MediaPipe: API 'solutions' non disponible (version >= 0.10.x). Association par IoU uniquement.")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None
    logger.info("MediaPipe non disponible. Association par IoU uniquement.")

# TorchReID pour OSNet (optionnel, meilleur que ResNet50)
try:
    import torchreid
    TORCHREID_AVAILABLE = True
except ImportError:
    TORCHREID_AVAILABLE = False
    logger.info("TorchReID non disponible. Utilisation de ResNet50 pour ReID.")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Configuration globale du système de tracking."""
    
    # Classes COCO à détecter (id: nom)
    # person=0, bicycle=1, car=2, motorcycle=3, backpack=24, handbag=26, suitcase=28
    TARGET_CLASSES: Dict[int, str] = field(default_factory=lambda: {
        0: 'person',       # personne
        2: 'car',          # voiture
        3: 'motorcycle',   # moto
        24: 'backpack',    # sac à dos
        26: 'handbag',     # sac à main
        28: 'suitcase',    # valise/sac
        # Note: 'casque/helmet' n'est pas dans COCO standard
        # On peut le mapper si un modèle custom est utilisé
    })
    
    # Classes considérées comme "objets portables" (associables aux personnes)
    PORTABLE_CLASSES: Set[int] = field(default_factory=lambda: {24, 26, 28})  # backpack, handbag, suitcase
    
    # Seuils
    CONFIDENCE_THRESHOLD: float = 0.25  # Abaissé de 0.4 à 0.25 pour mieux détecter les sacs
    IOU_THRESHOLD: float = 0.5           # Pour NMS
    ASSOCIATION_IOU_THRESHOLD: float = 0.3  # Pour associer objets aux personnes
    REID_THRESHOLD: float = 0.6          # Similarité minimale pour ReID
    
    # Tracking
    MAX_AGE: int = 30                    # Frames avant de perdre un track
    FEATURE_HISTORY_SIZE: int = 20       # Nombre de features à garder par personne
    
    # Caméras
    NUM_CAMERAS: int = 4                 # Nombre de vues horizontales
    
    # Couleurs pour visualisation (BGR)
    COLORS: Dict[str, Tuple[int, int, int]] = field(default_factory=lambda: {
        'person': (0, 255, 0),       # Vert
        'car': (255, 0, 0),          # Bleu
        'motorcycle': (0, 165, 255), # Orange
        'backpack': (255, 255, 0),   # Cyan
        'handbag': (255, 0, 255),    # Magenta
        'helmet': (0, 255, 255),     # Jaune
        'default': (128, 128, 128),  # Gris
    })


# Instance globale de configuration
CONFIG = Config()


# =============================================================================
# EXTRACTEUR DE FEATURES POUR REID
# =============================================================================

class ReIDFeatureExtractor:
    """
    Extracteur de features d'apparence pour la ré-identification.
    
    Utilise OSNet (via torchreid) si disponible, sinon ResNet50.
    Les features sont normalisées en L2 pour permettre le calcul de similarité cosinus.
    """
    
    def __init__(self, device: str = 'auto', use_osnet: bool = True):
        """
        Initialise l'extracteur de features.
        
        Args:
            device: 'cuda', 'cpu', ou 'auto' pour détection automatique
            use_osnet: Utiliser OSNet si disponible (recommandé)
        """
        # Déterminer le device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"ReID: Utilisation du device {self.device}")
        
        # Charger le modèle
        self.model = None
        self.model_type = None
        
        if use_osnet and TORCHREID_AVAILABLE:
            self._init_osnet()
        elif TORCHVISION_AVAILABLE:
            self._init_resnet50()
        else:
            logger.error("Aucun modèle ReID disponible!")
        
        # Transformations pour prétraitement
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),  # Taille standard pour ReID
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _init_osnet(self):
        """Initialise OSNet depuis torchreid."""
        try:
            self.model = torchreid.models.build_model(
                name='osnet_x1_0',
                num_classes=1,  # Pour extraction de features uniquement
                loss='softmax',
                pretrained=True
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            self.model_type = 'osnet'
            logger.info("ReID: OSNet x1.0 chargé avec succès")
        except Exception as e:
            logger.warning(f"Échec chargement OSNet: {e}")
            self._init_resnet50()
    
    def _init_resnet50(self):
        """Initialise ResNet50 comme extracteur de features."""
        try:
            # Charger ResNet50 pré-entraîné sur ImageNet
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            
            # Retirer la couche de classification finale pour obtenir les features
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            
            self.model = self.model.to(self.device)
            self.model.eval()
            self.model_type = 'resnet50'
            logger.info("ReID: ResNet50 chargé avec succès (features 2048-dim)")
        except Exception as e:
            logger.error(f"Échec chargement ResNet50: {e}")
    
    @torch.no_grad()
    def extract(self, image_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extrait les features d'apparence d'un crop d'image.
        
        Args:
            image_crop: Image BGR (numpy array) du crop de la personne
            
        Returns:
            Features normalisées (numpy array 1D) ou None si échec
        """
        if self.model is None:
            return None
        
        if image_crop is None or image_crop.size == 0:
            return None
        
        try:
            # Convertir BGR -> RGB
            image_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
            
            # Appliquer les transformations
            tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
            
            # Extraire les features
            if self.model_type == 'osnet':
                features = self.model(tensor)
            else:  # resnet50
                features = self.model(tensor)
                features = features.squeeze()
            
            # Normaliser en L2
            features = F.normalize(features.flatten(), p=2, dim=0)
            
            return features.cpu().numpy()
            
        except Exception as e:
            logger.debug(f"Erreur extraction features: {e}")
            return None
    
    @torch.no_grad()
    def extract_batch(self, crops: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """
        Extrait les features pour un batch de crops.
        
        Args:
            crops: Liste d'images BGR
            
        Returns:
            Liste de features (ou None pour les crops invalides)
        """
        if self.model is None or not crops:
            return [None] * len(crops)
        
        valid_indices = []
        tensors = []
        
        for i, crop in enumerate(crops):
            if crop is not None and crop.size > 0:
                try:
                    image_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    tensor = self.transform(image_rgb)
                    tensors.append(tensor)
                    valid_indices.append(i)
                except Exception:
                    pass
        
        if not tensors:
            return [None] * len(crops)
        
        try:
            batch = torch.stack(tensors).to(self.device)
            
            if self.model_type == 'osnet':
                features = self.model(batch)
            else:
                features = self.model(batch)
                features = features.squeeze(-1).squeeze(-1)
            
            features = F.normalize(features, p=2, dim=1)
            features_np = features.cpu().numpy()
            
            results = [None] * len(crops)
            for idx, feat_idx in enumerate(valid_indices):
                results[feat_idx] = features_np[idx]
            
            return results
            
        except Exception as e:
            logger.debug(f"Erreur batch extraction: {e}")
            return [None] * len(crops)
    
    @staticmethod
    def compute_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Calcule la similarité cosinus entre deux features."""
        if feat1 is None or feat2 is None:
            return 0.0
        return float(np.dot(feat1, feat2))


# =============================================================================
# STRUCTURES DE DONNÉES POUR LE TRACKING
# =============================================================================

@dataclass
class Detection:
    """Représente une détection d'objet dans une frame."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    track_id: Optional[int] = None   # ID local du tracker
    camera_id: Optional[int] = None
    feature: Optional[np.ndarray] = None


@dataclass 
class PersonTrack:
    """
    Représente une personne trackée à travers les caméras.
    
    Stocke l'historique des features pour le matching ReID
    et les objets associés (sacs, casques).
    """
    global_id: int
    
    # Features d'apparence (galerie)
    features: deque = field(default_factory=lambda: deque(maxlen=CONFIG.FEATURE_HISTORY_SIZE))
    
    # Dernière position connue par caméra
    last_seen_camera: Optional[int] = None
    last_bbox: Optional[Tuple[int, int, int, int]] = None
    last_frame: int = 0
    
    # Caméras où cette personne a été vue
    cameras_seen: Set[int] = field(default_factory=set)
    
    # Objets associés (class_id, track_id local)
    associated_objects: List[Tuple[str, int]] = field(default_factory=list)
    
    # Compteurs
    total_frames_seen: int = 0
    
    def add_feature(self, feature: np.ndarray):
        """Ajoute une nouvelle feature à la galerie."""
        if feature is not None:
            self.features.append(feature)
    
    def get_mean_feature(self) -> Optional[np.ndarray]:
        """Retourne la feature moyenne de la galerie."""
        if not self.features:
            return None
        return np.mean(list(self.features), axis=0)
    
    def compute_similarity(self, feature: np.ndarray) -> float:
        """Calcule la similarité maximale avec les features de la galerie."""
        if feature is None or not self.features:
            return 0.0
        
        max_sim = 0.0
        for stored_feature in self.features:
            sim = float(np.dot(feature, stored_feature))
            max_sim = max(max_sim, sim)
        
        return max_sim


# =============================================================================
# GESTIONNAIRE DE TRACKING CROSS-CAMÉRA
# =============================================================================

class CrossCameraTracker:
    """
    Gestionnaire de tracking cross-caméra avec ré-identification AMÉLIORÉ.
    
    Améliorations vs version précédente:
    - Seuil ReID plus bas (0.4) pour meilleur matching
    - Re-matching dans la même caméra si track_id local change
    - Garde les tracks plus longtemps (max_age * 10)
    - Utilise moyenne des features pour matching plus stable
    """
    
    def __init__(self, reid_extractor: ReIDFeatureExtractor, 
                 reid_threshold: float = 0.4,  # Abaissé de 0.6 à 0.4
                 max_age: int = 50):  # Augmenté de 30 à 50
        """
        Initialise le tracker cross-caméra.
        
        Args:
            reid_extractor: Extracteur de features ReID
            reid_threshold: Seuil de similarité pour associer une détection (abaissé à 0.4)
            max_age: Frames avant de considérer un track comme perdu
        """
        self.reid = reid_extractor
        self.reid_threshold = reid_threshold
        self.max_age = max_age
        
        # Registre global des personnes
        self.global_tracks: Dict[int, PersonTrack] = {}
        self.next_global_id = 1
        
        # Mapping: (camera_id, local_track_id) -> global_id
        self.local_to_global: Dict[Tuple[int, int], int] = {}
        
        # Historique des global_ids par caméra
        self.camera_active_ids: Dict[int, Set[int]] = defaultdict(set)
        
        # Cache des dernières features par global_id pour matching rapide
        self.feature_cache: Dict[int, np.ndarray] = {}
    
    def update(self, camera_id: int, detections: List[Detection], 
               frame_number: int) -> Dict[int, int]:
        """
        Met à jour le tracking global avec les détections d'une caméra.
        
        Algorithme amélioré:
        1. Pour chaque détection, vérifier d'abord le mapping existant
        2. Sinon, chercher un match ReID dans TOUS les tracks (même caméra inclue)
        3. Créer un nouveau global_id seulement si aucun match trouvé
        """
        mapping = {}
        current_camera_ids = set()
        
        # Collecter toutes les personnes à traiter
        person_detections = [d for d in detections if d.class_id == 0 and d.track_id is not None]
        
        for det in person_detections:
            local_key = (camera_id, det.track_id)
            
            # 1. Vérifier si on a déjà un mapping pour ce track local
            if local_key in self.local_to_global:
                global_id = self.local_to_global[local_key]
                
                # Mettre à jour le track existant
                if global_id in self.global_tracks:
                    track = self.global_tracks[global_id]
                    self._update_track(track, det, camera_id, frame_number)
                    mapping[det.track_id] = global_id
                    current_camera_ids.add(global_id)
                    continue
            
            # 2. Nouveau track local - chercher un match ReID
            global_id, similarity = self._find_best_match(det, camera_id, frame_number)
            
            if global_id is not None:
                # Match trouvé - associer au global existant
                self.local_to_global[local_key] = global_id
                track = self.global_tracks[global_id]
                self._update_track(track, det, camera_id, frame_number)
                
                logger.debug(f"ReID Match: local({camera_id},{det.track_id}) -> G{global_id} (sim={similarity:.2f})")
            else:
                # Aucun match - créer un nouveau global
                global_id = self._create_new_track(det, camera_id, frame_number)
                self.local_to_global[local_key] = global_id
                
                logger.debug(f"New Track: local({camera_id},{det.track_id}) -> G{global_id}")
            
            mapping[det.track_id] = global_id
            current_camera_ids.add(global_id)
        
        # Mettre à jour les IDs actifs de cette caméra
        self.camera_active_ids[camera_id] = current_camera_ids
        
        # Nettoyer les vieux tracks (beaucoup plus permissif)
        self._cleanup_old_tracks(frame_number)
        
        return mapping
    
    def _update_track(self, track: PersonTrack, det: Detection, 
                      camera_id: int, frame_number: int):
        """Met à jour un track existant avec une nouvelle détection."""
        track.last_seen_camera = camera_id
        track.last_bbox = det.bbox
        track.last_frame = frame_number
        track.cameras_seen.add(camera_id)
        track.total_frames_seen += 1
        
        if det.feature is not None:
            track.add_feature(det.feature)
            # Mettre à jour le cache avec la feature moyenne
            mean_feat = track.get_mean_feature()
            if mean_feat is not None:
                self.feature_cache[track.global_id] = mean_feat
    
    def _create_new_track(self, det: Detection, camera_id: int, 
                          frame_number: int) -> int:
        """Crée un nouveau track global."""
        global_id = self.next_global_id
        self.next_global_id += 1
        
        track = PersonTrack(
            global_id=global_id,
            last_seen_camera=camera_id,
            last_bbox=det.bbox,
            last_frame=frame_number,
            cameras_seen={camera_id},
            total_frames_seen=1
        )
        
        if det.feature is not None:
            track.add_feature(det.feature)
            self.feature_cache[global_id] = det.feature
        
        self.global_tracks[global_id] = track
        return global_id
    
    def _find_best_match(self, detection: Detection, camera_id: int, 
                         frame_number: int) -> Tuple[Optional[int], float]:
        """
        Cherche le meilleur match ReID pour une nouvelle détection.
        
        AMÉLIORÉ: 
        - Cherche dans TOUS les tracks (même caméra incluse)
        - Utilise la feature moyenne pour plus de stabilité
        - Retourne aussi le score de similarité
        """
        if detection.feature is None:
            return None, 0.0
        
        best_match_id = None
        best_similarity = self.reid_threshold
        
        for global_id, track in self.global_tracks.items():
            # Vérifier si ce track n'est pas trop vieux
            age = frame_number - track.last_frame
            if age > self.max_age * 10:  # Beaucoup plus permissif: 10x max_age
                continue
            
            # Utiliser le cache de features moyennes si disponible
            if global_id in self.feature_cache:
                stored_feature = self.feature_cache[global_id]
                similarity = float(np.dot(detection.feature, stored_feature))
            else:
                # Sinon calculer avec la galerie
                similarity = track.compute_similarity(detection.feature)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = global_id
        
        return best_match_id, best_similarity
    
    def _cleanup_old_tracks(self, current_frame: int):
        """
        Supprime les tracks très anciens.
        AMÉLIORÉ: Beaucoup plus permissif pour garder les IDs longtemps.
        """
        to_remove = []
        
        for global_id, track in self.global_tracks.items():
            age = current_frame - track.last_frame
            # Garder les tracks pendant max_age * 20 frames (très long)
            if age > self.max_age * 20:
                to_remove.append(global_id)
        
        for gid in to_remove:
            del self.global_tracks[gid]
            if gid in self.feature_cache:
                del self.feature_cache[gid]
            # Nettoyer aussi le mapping local->global
            keys_to_remove = [k for k, v in self.local_to_global.items() if v == gid]
            for k in keys_to_remove:
                del self.local_to_global[k]
    
    def get_track(self, global_id: int) -> Optional[PersonTrack]:
        """Retourne le track global par ID."""
        return self.global_tracks.get(global_id)
    
    def get_stats(self) -> Dict:
        """Retourne les statistiques du tracker."""
        cross_camera = sum(1 for t in self.global_tracks.values() 
                          if len(t.cameras_seen) > 1)
        return {
            'total_global_tracks': len(self.global_tracks),
            'cross_camera_tracks': cross_camera,
            'active_mappings': len(self.local_to_global),
        }


# =============================================================================
# DÉTECTION D'OBJETS ABANDONNÉS
# =============================================================================

@dataclass
class TrackedObject:
    """Représente un objet suivi pour détection d'abandon."""
    track_id: int
    class_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    camera_id: int
    first_seen_frame: int
    last_seen_frame: int
    last_owner_id: Optional[int] = None  # Dernier global_id de personne associée
    frames_stationary: int = 0
    frames_without_owner: int = 0
    is_abandoned: bool = False
    picked_up_by: Optional[int] = None  # global_id du suspect


class AbandonedObjectTracker:
    """
    Détecte les objets abandonnés (stationnaires sans propriétaire).
    
    Critères d'abandon:
    - Objet stationnaire (mouvement < seuil pixels)
    - Pas de personne à proximité pendant N frames
    """
    
    def __init__(self, 
                 abandon_threshold_frames: int = 25,  # ~1 sec @ 25fps
                 movement_threshold: float = 15.0,    # Pixels
                 proximity_threshold: float = 150.0): # Pixels (augmenté de 100 à 150)
        """
        Args:
            abandon_threshold_frames: Frames avant qu'un objet soit abandonné
            movement_threshold: Mouvement max en pixels pour être "stationnaire"
            proximity_threshold: Distance max pour qu'une personne soit "proche"
        """
        self.abandon_threshold = abandon_threshold_frames
        self.movement_threshold = movement_threshold
        self.proximity_threshold = proximity_threshold
        
        # Objets suivis par caméra: {camera_id: {track_id: TrackedObject}}
        self.tracked_objects: Dict[int, Dict[int, TrackedObject]] = defaultdict(dict)
        
        # Objets abandonnés actifs
        self.abandoned_objects: Dict[Tuple[int, int], TrackedObject] = {}  # (cam, track_id)
    
    def update(self, camera_id: int, detections: List[Detection], 
               person_global_mapping: Dict[int, int],
               frame_number: int) -> List[TrackedObject]:
        """
        Met à jour le tracking des objets et détecte les abandons ET vols.
        
        LOGIQUE AMÉLIORÉE:
        1. ABANDON: Objet stationnaire sans propriétaire pendant N frames
        2. VOL: Objet change de propriétaire (personne A → personne B)
        
        Args:
            camera_id: ID de la caméra
            detections: Toutes les détections (personnes + objets)
            person_global_mapping: Mapping local_id -> global_id pour personnes
            frame_number: Numéro de frame
            
        Returns:
            Liste des objets nouvellement abandonnés
        """
        # Séparer personnes et objets portables
        persons = [d for d in detections if d.class_id == 0]
        objects = [d for d in detections if d.class_id in CONFIG.PORTABLE_CLASSES and d.track_id is not None]
        
        newly_abandoned = []
        ownership_changes = []  # Liste de (nouveau_owner_id, objet, ancien_owner_id)
        current_object_ids = set()
        
        for obj in objects:
            current_object_ids.add(obj.track_id)
            
            # Trouver la personne la plus proche
            current_owner = self._find_nearest_person(obj, persons, person_global_mapping)
            
            if obj.track_id in self.tracked_objects[camera_id]:
                # Objet existant - mettre à jour
                tracked = self.tracked_objects[camera_id][obj.track_id]
                prev_center = tracked.center
                
                # Vérifier mouvement
                movement = np.sqrt((obj.center[0] - prev_center[0])**2 + 
                                  (obj.center[1] - prev_center[1])**2)
                
                if movement < self.movement_threshold:
                    tracked.frames_stationary += 1
                else:
                    tracked.frames_stationary = 0
                
                # === DÉTECTION DE CHANGEMENT DE PROPRIÉTAIRE (VOL) ===
                if current_owner is not None:
                    # Il y a quelqu'un près de l'objet
                    if tracked.last_owner_id is not None and tracked.last_owner_id != current_owner:
                        # Le propriétaire a CHANGÉ! C'est un potentiel vol!
                        if not hasattr(tracked, 'original_owner'):
                            tracked.original_owner = tracked.last_owner_id
                        
                        # Si c'est le propriétaire original qui reprend son objet, OK
                        if hasattr(tracked, 'original_owner') and current_owner != tracked.original_owner:
                            # UNE AUTRE personne prend l'objet!
                            ownership_changes.append((current_owner, tracked, tracked.original_owner))
                            logger.warning(f"⚠️ CHANGEMENT PROPRIÉTAIRE: {tracked.class_name} - G{tracked.last_owner_id} → G{current_owner}")
                    
                    # Mettre à jour le propriétaire
                    if tracked.last_owner_id is None:
                        # Premier propriétaire détecté
                        if not hasattr(tracked, 'original_owner'):
                            tracked.original_owner = current_owner
                    
                    tracked.last_owner_id = current_owner
                    tracked.frames_without_owner = 0
                else:
                    tracked.frames_without_owner += 1
                
                # Mettre à jour position
                tracked.bbox = obj.bbox
                tracked.center = obj.center
                tracked.last_seen_frame = frame_number
                
                # Vérifier si abandonné (stationnaire + sans propriétaire)
                if (tracked.frames_stationary >= self.abandon_threshold and 
                    tracked.frames_without_owner >= self.abandon_threshold and
                    not tracked.is_abandoned):
                    tracked.is_abandoned = True
                    self.abandoned_objects[(camera_id, obj.track_id)] = tracked
                    newly_abandoned.append(tracked)
                    logger.info(f"🚨 OBJET ABANDONNÉ: {tracked.class_name} (CAM{camera_id+1}, track={obj.track_id})")
            else:
                # Nouvel objet
                tracked = TrackedObject(
                    track_id=obj.track_id,
                    class_id=obj.class_id,
                    class_name=obj.class_name,
                    bbox=obj.bbox,
                    center=obj.center,
                    camera_id=camera_id,
                    first_seen_frame=frame_number,
                    last_seen_frame=frame_number,
                    last_owner_id=self._find_nearest_person(obj, persons, person_global_mapping)
                )
                self.tracked_objects[camera_id][obj.track_id] = tracked
        
        # Nettoyer les objets disparus
        to_remove = [tid for tid in self.tracked_objects[camera_id] if tid not in current_object_ids]
        for tid in to_remove:
            del self.tracked_objects[camera_id][tid]
        
        return newly_abandoned, ownership_changes
    
    def _find_nearest_person(self, obj: Detection, persons: List[Detection],
                             mapping: Dict[int, int]) -> Optional[int]:
        """Trouve la personne la plus proche (global_id) si dans le seuil."""
        min_dist = float('inf')
        nearest_global_id = None
        
        for person in persons:
            if person.track_id is None:
                continue
            
            # Distance centre à centre
            dist = np.sqrt((obj.center[0] - person.center[0])**2 + 
                          (obj.center[1] - person.center[1])**2)
            
            if dist < min_dist and dist < self.proximity_threshold:
                min_dist = dist
                nearest_global_id = mapping.get(person.track_id)
        
        return nearest_global_id
    
    def check_pickup(self, camera_id: int, detections: List[Detection],
                     person_global_mapping: Dict[int, int],
                     frame_number: int) -> List[Tuple[int, TrackedObject]]:
        """
        Vérifie si une personne ramasse un objet abandonné.
        
        Returns:
            Liste de tuples (global_id du suspect, objet ramassé)
        """
        pickups = []
        persons = [d for d in detections if d.class_id == 0]
        current_object_ids = {d.track_id for d in detections 
                             if d.class_id in CONFIG.PORTABLE_CLASSES and d.track_id is not None}
        
        # Vérifier chaque objet abandonné
        abandoned_to_remove = []
        for key, abandoned in self.abandoned_objects.items():
            if key[0] != camera_id:
                continue
            
            # L'objet a-t-il disparu?
            if abandoned.track_id not in current_object_ids:
                # Chercher qui était proche
                for person in persons:
                    if person.track_id is None:
                        continue
                    
                    # Vérifier si la personne était dans la zone de l'objet
                    iou = self._compute_iou(person.bbox, abandoned.bbox)
                    dist = np.sqrt((person.center[0] - abandoned.center[0])**2 + 
                                  (person.center[1] - abandoned.center[1])**2)
                    
                    if iou > 0.1 or dist < self.proximity_threshold:
                        global_id = person_global_mapping.get(person.track_id)
                        if global_id is not None:
                            # SUSPECT IDENTIFIÉ!
                            abandoned.picked_up_by = global_id
                            pickups.append((global_id, abandoned))
                            logger.warning(f"⚠️ SUSPECT IDENTIFIÉ: G{global_id} a ramassé {abandoned.class_name}!")
                            abandoned_to_remove.append(key)
                            break
        
        # Nettoyer les objets ramassés
        for key in abandoned_to_remove:
            if key in self.abandoned_objects:
                del self.abandoned_objects[key]
        
        return pickups
    
    def _compute_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Calcule l'IoU entre deux bboxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def get_abandoned_objects(self, camera_id: int = None) -> List[TrackedObject]:
        """Retourne les objets abandonnés actifs."""
        if camera_id is not None:
            return [obj for (cam, _), obj in self.abandoned_objects.items() if cam == camera_id]
        return list(self.abandoned_objects.values())


# =============================================================================
# SYSTÈME DE TRACKING DES SUSPECTS
# =============================================================================

@dataclass
class SuspectInfo:
    """Information sur un suspect."""
    global_id: int
    reason: str  # Ex: "A ramassé backpack"
    stolen_object: Optional[TrackedObject] = None
    first_identified_frame: int = 0
    first_identified_camera: int = 0
    cameras_seen: Set[int] = field(default_factory=set)
    alert_count: int = 0


class SuspectTracker:
    """
    Gère le tracking des suspects à travers les caméras.
    """
    
    def __init__(self):
        self.suspects: Dict[int, SuspectInfo] = {}  # global_id -> SuspectInfo
    
    def add_suspect(self, global_id: int, reason: str, 
                    stolen_object: TrackedObject, frame_number: int,
                    camera_id: int) -> SuspectInfo:
        """Ajoute un nouveau suspect."""
        if global_id in self.suspects:
            return self.suspects[global_id]
        
        suspect = SuspectInfo(
            global_id=global_id,
            reason=reason,
            stolen_object=stolen_object,
            first_identified_frame=frame_number,
            first_identified_camera=camera_id,
            cameras_seen={camera_id}
        )
        self.suspects[global_id] = suspect
        return suspect
    
    def is_suspect(self, global_id: int) -> bool:
        """Vérifie si une personne est un suspect."""
        return global_id in self.suspects
    
    def get_suspect(self, global_id: int) -> Optional[SuspectInfo]:
        """Retourne les infos du suspect."""
        return self.suspects.get(global_id)
    
    def update_sighting(self, global_id: int, camera_id: int) -> bool:
        """
        Met à jour quand un suspect est vu.
        
        Returns:
            True si c'est une nouvelle caméra (alerte à générer)
        """
        if global_id not in self.suspects:
            return False
        
        suspect = self.suspects[global_id]
        is_new_camera = camera_id not in suspect.cameras_seen
        suspect.cameras_seen.add(camera_id)
        
        return is_new_camera
    
    def get_all_suspects(self) -> List[SuspectInfo]:
        """Retourne tous les suspects."""
        return list(self.suspects.values())


# =============================================================================
# SYSTÈME D'ALERTES
# =============================================================================

class AlertSystem:
    """
    Gère les alertes visuelles et le logging des événements.
    """
    
    def __init__(self, log_file: str = None):
        self.alerts: List[Dict] = []
        self.active_alerts: List[Dict] = []  # Alertes à afficher
        self.alert_duration = 75  # Frames d'affichage (3 sec @ 25fps)
        
        self.log_file = log_file
        if log_file:
            with open(log_file, 'w') as f:
                f.write("=== SYSTÈME D'ALERTES MULTI-CAMÉRAS ===\n\n")
    
    def add_alert(self, alert_type: str, message: str, 
                  frame_number: int, camera_id: int = None,
                  global_id: int = None):
        """Ajoute une nouvelle alerte."""
        alert = {
            'type': alert_type,
            'message': message,
            'frame': frame_number,
            'camera_id': camera_id,
            'global_id': global_id,
            'remaining_frames': self.alert_duration
        }
        
        self.alerts.append(alert)
        self.active_alerts.append(alert)
        
        # Log
        log_msg = f"[Frame {frame_number}] {alert_type}: {message}"
        logger.warning(log_msg)
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(f"{log_msg}\n")
    
    def add_abandoned_alert(self, obj: TrackedObject, frame_number: int):
        """Alerte pour objet abandonné."""
        self.add_alert(
            'OBJET_ABANDONNÉ',
            f"{obj.class_name} abandonné dans CAM{obj.camera_id + 1}",
            frame_number,
            camera_id=obj.camera_id
        )
    
    def add_suspect_alert(self, global_id: int, obj: TrackedObject, 
                          frame_number: int, camera_id: int):
        """Alerte pour nouveau suspect."""
        self.add_alert(
            'SUSPECT_IDENTIFIÉ',
            f"G{global_id} a ramassé {obj.class_name} dans CAM{camera_id + 1}",
            frame_number,
            camera_id=camera_id,
            global_id=global_id
        )
    
    def add_suspect_sighting(self, global_id: int, camera_id: int, 
                             frame_number: int):
        """Alerte quand suspect vu dans nouvelle caméra."""
        self.add_alert(
            'SUSPECT_DÉTECTÉ',
            f"⚠️ SUSPECT G{global_id} détecté dans CAM{camera_id + 1}!",
            frame_number,
            camera_id=camera_id,
            global_id=global_id
        )
    
    def update(self):
        """Met à jour les alertes actives (décrémente durée)."""
        self.active_alerts = [a for a in self.active_alerts if a['remaining_frames'] > 0]
        for alert in self.active_alerts:
            alert['remaining_frames'] -= 1
    
    def get_active_alerts(self) -> List[Dict]:
        """Retourne les alertes à afficher."""
        return self.active_alerts
    
    def draw_alerts(self, frame: np.ndarray) -> np.ndarray:
        """Dessine les alertes sur la frame."""
        if not self.active_alerts:
            return frame
        
        h, w = frame.shape[:2]
        
        # Bannière en haut
        for i, alert in enumerate(self.active_alerts[:3]):  # Max 3 alertes
            y = 50 + i * 35
            
            # Couleur selon type
            if 'SUSPECT' in alert['type']:
                color = (0, 0, 255)  # Rouge
            else:
                color = (0, 165, 255)  # Orange
            
            # Fond semi-transparent
            cv2.rectangle(frame, (10, y - 25), (w - 10, y + 5), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, y - 25), (w - 10, y + 5), color, 2)
            
            # Texte
            cv2.putText(frame, alert['message'], (20, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return frame


# =============================================================================
# ASSOCIATEUR OBJETS-PERSONNES
# =============================================================================

class ObjectPersonAssociator:
    """
    Associe les objets détectés (sacs, casques) aux personnes.
    
    Utilise:
    - L'intersection over union (IoU) des bounding boxes
    - (Optionnel) Les keypoints MediaPipe pour association basée sur la pose
    """
    
    def __init__(self, iou_threshold: float = 0.3, use_pose: bool = True):
        """
        Initialise l'associateur.
        
        Args:
            iou_threshold: Seuil IoU minimum pour association
            use_pose: Utiliser MediaPipe pose si disponible
        """
        self.iou_threshold = iou_threshold
        self.use_pose = use_pose and MEDIAPIPE_AVAILABLE
        
        # MediaPipe Pose
        if self.use_pose:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                min_detection_confidence=0.5
            )
            logger.info("Associateur: MediaPipe Pose activé")
        else:
            self.pose = None
    
    def associate(self, persons: List[Detection], objects: List[Detection],
                  frame: Optional[np.ndarray] = None) -> Dict[int, List[Detection]]:
        """
        Associe les objets aux personnes.
        
        Args:
            persons: Liste des détections de personnes
            objects: Liste des détections d'objets (sacs, casques, etc.)
            frame: Image pour pose estimation (optionnel)
            
        Returns:
            Dictionnaire person_track_id -> [objets associés]
        """
        associations = defaultdict(list)
        
        if not persons or not objects:
            return associations
        
        # Méthode principale: IoU
        for obj in objects:
            best_person_id = None
            best_score = 0.0
            
            for person in persons:
                if person.track_id is None:
                    continue
                
                # Calculer l'IoU
                iou = self._compute_iou(obj.bbox, person.bbox)
                
                # Vérifier si l'objet est dans la zone de la personne
                overlap_score = self._compute_overlap_score(obj.bbox, person.bbox)
                
                # Score combiné
                score = max(iou, overlap_score * 0.8)
                
                if score > best_score and score >= self.iou_threshold:
                    best_score = score
                    best_person_id = person.track_id
            
            if best_person_id is not None:
                associations[best_person_id].append(obj)
        
        # Méthode optionnelle: Pose estimation
        if self.use_pose and frame is not None:
            pose_associations = self._associate_by_pose(persons, objects, frame)
            
            # Fusionner (pose prend priorité si conflit)
            for person_id, objs in pose_associations.items():
                for obj in objs:
                    if obj not in associations[person_id]:
                        associations[person_id].append(obj)
        
        return dict(associations)
    
    def _compute_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Calcule l'IoU entre deux bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_overlap_score(self, obj_box: Tuple, person_box: Tuple) -> float:
        """
        Calcule le pourcentage de l'objet qui est dans la bbox de la personne.
        Utile pour les petits objets portés.
        """
        x1 = max(obj_box[0], person_box[0])
        y1 = max(obj_box[1], person_box[1])
        x2 = min(obj_box[2], person_box[2])
        y2 = min(obj_box[3], person_box[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        obj_area = (obj_box[2] - obj_box[0]) * (obj_box[3] - obj_box[1])
        
        return intersection / obj_area if obj_area > 0 else 0.0
    
    def _associate_by_pose(self, persons: List[Detection], objects: List[Detection],
                           frame: np.ndarray) -> Dict[int, List[Detection]]:
        """
        Associe les objets aux personnes via les keypoints de pose.
        
        Pour les sacs à dos: proche des épaules
        Pour les sacs à main: proche des mains/poignets
        """
        associations = defaultdict(list)
        
        if not self.pose:
            return associations
        
        for person in persons:
            if person.track_id is None:
                continue
            
            # Extraire le crop de la personne
            x1, y1, x2, y2 = person.bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            person_crop = frame[y1:y2, x1:x2]
            
            try:
                # Détecter la pose
                results = self.pose.process(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
                
                if results.pose_landmarks:
                    # Keypoints utiles
                    landmarks = results.pose_landmarks.landmark
                    
                    # Épaules (pour sac à dos)
                    left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                    right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    
                    # Poignets (pour sac à main)
                    left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
                    right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
                    
                    for obj in objects:
                        obj_center = obj.center
                        
                        # Convertir keypoints en coordonnées image globale
                        crop_h, crop_w = person_crop.shape[:2]
                        
                        # Vérifier la proximité selon le type d'objet
                        if obj.class_id == 24:  # backpack
                            # Proche des épaules
                            shoulder_x = (left_shoulder.x + right_shoulder.x) / 2 * crop_w + x1
                            shoulder_y = (left_shoulder.y + right_shoulder.y) / 2 * crop_h + y1
                            
                            dist = np.sqrt((obj_center[0] - shoulder_x)**2 + 
                                         (obj_center[1] - shoulder_y)**2)
                            
                            if dist < (y2 - y1) * 0.5:  # 50% de la hauteur
                                associations[person.track_id].append(obj)
                        
                        elif obj.class_id == 26:  # handbag
                            # Proche des poignets
                            for wrist in [left_wrist, right_wrist]:
                                if wrist.visibility > 0.5:
                                    wrist_x = wrist.x * crop_w + x1
                                    wrist_y = wrist.y * crop_h + y1
                                    
                                    dist = np.sqrt((obj_center[0] - wrist_x)**2 + 
                                                 (obj_center[1] - wrist_y)**2)
                                    
                                    if dist < (y2 - y1) * 0.4:
                                        if obj not in associations[person.track_id]:
                                            associations[person.track_id].append(obj)
                                        break
                                        
            except Exception as e:
                logger.debug(f"Erreur pose estimation: {e}")
        
        return dict(associations)


# =============================================================================
# PROCESSEUR VIDÉO PRINCIPAL
# =============================================================================

class MultiCameraVideoProcessor:
    """
    Processeur principal pour le tracking multi-caméras.
    
    Traite une vidéo concaténée contenant plusieurs vues de caméras,
    applique la détection/tracking par région, et fusionne les tracks
    avec ré-identification cross-caméra.
    """
    
    def __init__(self, 
                 model_path: str = "yolov8m.pt",
                 num_cameras: int = 4,
                 tracker_type: str = "bytetrack",
                 use_reid: bool = True,
                 reid_threshold: float = 0.6,
                 use_pose: bool = True,
                 device: str = "auto",
                 # Nouveaux paramètres pour détection d'abandon et suspects
                 detect_abandoned: bool = True,
                 abandon_threshold: int = 75,  # Frames (~3 sec @ 25fps)
                 alert_log_file: str = None):
        """
        Initialise le processeur.
        
        Args:
            model_path: Chemin vers le modèle YOLO
            num_cameras: Nombre de vues caméra dans la vidéo
            tracker_type: 'bytetrack' ou 'botsort'
            use_reid: Activer la ré-identification cross-caméra
            reid_threshold: Seuil de similarité pour ReID
            use_pose: Utiliser MediaPipe pour association objets
            device: Device pour l'inférence ('auto', 'cuda', 'cpu')
            detect_abandoned: Activer détection d'objets abandonnés
            abandon_threshold: Frames avant qu'un objet soit abandonné
            alert_log_file: Fichier pour logger les alertes
        """
        self.num_cameras = num_cameras
        self.use_reid = use_reid
        self.use_pose = use_pose
        self.detect_abandoned = detect_abandoned
        
        # Device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"Device: {self.device}")
        
        # Charger le modèle YOLO
        if not YOLO_AVAILABLE:
            raise ImportError("Ultralytics n'est pas installé. pip install ultralytics")
        
        self.model = YOLO(model_path)
        logger.info(f"Modèle YOLO chargé: {model_path}")
        
        # Configuration du tracker
        self.tracker_type = tracker_type
        self.tracker_config = f"{tracker_type}.yaml"
        
        # Classes à détecter
        self.target_class_ids = list(CONFIG.TARGET_CLASSES.keys())
        
        # ReID
        if use_reid:
            self.reid_extractor = ReIDFeatureExtractor(device=self.device)
            self.cross_camera = CrossCameraTracker(
                self.reid_extractor, 
                reid_threshold=reid_threshold
            )
        else:
            self.reid_extractor = None
            self.cross_camera = None
        
        # Associateur objets-personnes
        self.associator = ObjectPersonAssociator(use_pose=use_pose)
        
        # Mapping local -> global par caméra (fallback sans ReID)
        self.local_global_mapping: Dict[Tuple[int, int], int] = {}
        self.next_fallback_id = 1
        
        # Couleurs pour les IDs
        self.id_colors: Dict[int, Tuple[int, int, int]] = {}
        
        # === NOUVEAUX COMPOSANTS ===
        # Détection d'objets abandonnés
        if detect_abandoned:
            self.abandoned_tracker = AbandonedObjectTracker(
                abandon_threshold_frames=abandon_threshold
            )
            logger.info(f"Détection d'objets abandonnés: activée (seuil={abandon_threshold} frames)")
        else:
            self.abandoned_tracker = None
        
        # Tracking des suspects
        self.suspect_tracker = SuspectTracker()
        
        # Système d'alertes
        self.alert_system = AlertSystem(log_file=alert_log_file)
        if alert_log_file:
            logger.info(f"Alertes logguées dans: {alert_log_file}")
    
    def split_frame(self, frame: np.ndarray) -> List[np.ndarray]:
        """
        Divise la frame en régions par caméra (horizontalement).
        
        Args:
            frame: Image complète
            
        Returns:
            Liste des régions par caméra
        """
        h, w = frame.shape[:2]
        region_width = w // self.num_cameras
        
        regions = []
        for i in range(self.num_cameras):
            x1 = i * region_width
            x2 = (i + 1) * region_width if i < self.num_cameras - 1 else w
            regions.append(frame[:, x1:x2].copy())
        
        return regions
    
    def process_region(self, region: np.ndarray, camera_id: int
                      ) -> Tuple[List[Detection], np.ndarray]:
        """
        Traite une région de caméra: détection + tracking.
        
        Args:
            region: Image de la région
            camera_id: ID de la caméra
            
        Returns:
            Tuple (liste de détections, frame annotée)
        """
        # Exécuter YOLO avec tracking
        results = self.model.track(
            region,
            persist=True,
            tracker=self.tracker_config,
            conf=CONFIG.CONFIDENCE_THRESHOLD,
            iou=CONFIG.IOU_THRESHOLD,
            classes=self.target_class_ids,
            device=self.device,
            verbose=False
        )
        
        detections = []
        
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy()
                
                # Track IDs (peuvent être None)
                if result.boxes.id is not None:
                    track_ids = result.boxes.id.cpu().numpy().astype(int)
                else:
                    track_ids = [None] * len(boxes)
                
                for box, cls_id, conf, track_id in zip(boxes, classes, confs, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    
                    class_name = CONFIG.TARGET_CLASSES.get(cls_id, f"class_{cls_id}")
                    
                    det = Detection(
                        class_id=cls_id,
                        class_name=class_name,
                        confidence=float(conf),
                        bbox=(x1, y1, x2, y2),
                        center=center,
                        track_id=int(track_id) if track_id is not None else None,
                        camera_id=camera_id
                    )
                    
                    detections.append(det)
        
        return detections
    
    def extract_reid_features(self, frame: np.ndarray, 
                               detections: List[Detection]) -> None:
        """
        Extrait les features ReID pour les personnes détectées.
        
        Args:
            frame: Image source
            detections: Liste des détections (modifiée in-place)
        """
        if self.reid_extractor is None:
            return
        
        # Collecter les crops de personnes
        person_indices = []
        crops = []
        
        for i, det in enumerate(detections):
            if det.class_id == 0:  # person
                x1, y1, x2, y2 = det.bbox
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2]
                    crops.append(crop)
                    person_indices.append(i)
        
        if not crops:
            return
        
        # Extraction batch
        features = self.reid_extractor.extract_batch(crops)
        
        for idx, feat in zip(person_indices, features):
            detections[idx].feature = feat
    
    def get_color_for_id(self, global_id: int) -> Tuple[int, int, int]:
        """Génère une couleur unique pour un ID global."""
        if global_id not in self.id_colors:
            np.random.seed(global_id * 42)
            self.id_colors[global_id] = tuple(np.random.randint(100, 255, 3).tolist())
        return self.id_colors[global_id]
    
    def visualize_region(self, region: np.ndarray, detections: List[Detection],
                         global_mapping: Dict[int, int],
                         associations: Dict[int, List[Detection]],
                         camera_id: int,
                         frame_number: int = 0) -> np.ndarray:
        """
        Dessine les annotations sur une région.
        
        Args:
            region: Image de la région
            detections: Détections dans cette région
            global_mapping: Mapping local_id -> global_id
            associations: Objets associés par personne
            camera_id: ID de la caméra
            frame_number: Numéro de frame (pour animations)
            
        Returns:
            Région annotée
        """
        vis = region.copy()
        
        # Dessiner les objets abandonnés en premier (arrière-plan)
        if self.abandoned_tracker:
            for abandoned in self.abandoned_tracker.get_abandoned_objects(camera_id):
                x1, y1, x2, y2 = abandoned.bbox
                
                # Animation clignotante (alternance toutes les 10 frames)
                if (frame_number // 10) % 2 == 0:
                    color = (0, 0, 255)  # Rouge
                else:
                    color = (0, 165, 255)  # Orange
                
                cv2.rectangle(vis, (x1-3, y1-3), (x2+3, y2+3), color, 3)
                
                # Label ABANDONNÉ
                label = f"⚠ ABANDONNÉ: {abandoned.class_name}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(vis, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0] + 10, y1), color, -1)
                cv2.putText(vis, label, (x1 + 5, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # Couleur selon la classe
            if det.class_id == 0 and det.track_id is not None:  # Person
                global_id = global_mapping.get(det.track_id, det.track_id)
                
                # Vérifier si c'est un suspect
                if self.suspect_tracker.is_suspect(global_id):
                    # SUSPECT - style spécial rouge
                    color = (0, 0, 255)  # Rouge vif
                    suspect_info = self.suspect_tracker.get_suspect(global_id)
                    label = f"⚠ SUSPECT G{global_id}"
                    
                    # Dessiner bbox épais
                    cv2.rectangle(vis, (x1-2, y1-2), (x2+2, y2+2), color, 4)
                    
                    # Fond rouge pour label
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(vis, (x1, y1 - label_size[1] - 10), 
                                 (x1 + label_size[0] + 10, y1), color, -1)
                    cv2.putText(vis, label, (x1 + 5, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                else:
                    # Personne normale
                    color = self.get_color_for_id(global_id)
                    label = f"G{global_id}"
                    
                    # Ajouter les objets associés
                    if det.track_id in associations:
                        obj_names = [obj.class_name for obj in associations[det.track_id]]
                        if obj_names:
                            label += f" [{','.join(obj_names)}]"
                    
                    # Dessiner bbox
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                    
                    # Dessiner label
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(vis, (x1, y1 - label_size[1] - 5), 
                                 (x1 + label_size[0], y1), color, -1)
                    cv2.putText(vis, label, (x1, y1 - 3), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                color = CONFIG.COLORS.get(det.class_name, CONFIG.COLORS['default'])
                label = f"{det.class_name}"
                
                # Dessiner bbox
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                
                # Dessiner label
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(vis, (x1, y1 - label_size[1] - 5), 
                             (x1 + label_size[0], y1), color, -1)
                cv2.putText(vis, label, (x1, y1 - 3), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Label de caméra
        cv2.putText(vis, f"CAM {camera_id + 1}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return vis
    
    def process_video(self, input_path: str, output_path: str = None,
                      show: bool = True, max_frames: int = None,
                      headless: bool = False) -> None:
        """
        Traite une vidéo complète.
        
        Args:
            input_path: Chemin de la vidéo d'entrée
            output_path: Chemin de la vidéo de sortie (optionnel)
            show: Afficher en temps réel
            max_frames: Limiter le nombre de frames
            headless: Mode sans affichage (pour serveurs)
        """
        # Ouvrir la vidéo
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            logger.error(f"Impossible d'ouvrir la vidéo: {input_path}")
            return
        
        # Propriétés vidéo
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Vidéo: {width}x{height} @ {fps}fps, {total_frames} frames")
        logger.info(f"Régions caméra: {self.num_cameras} x {width // self.num_cameras}x{height}")
        
        # Writer de sortie avec haute qualité
        writer = None
        temp_output_path = None
        use_ffmpeg_encode = False
        
        if output_path:
            # Essayer d'utiliser H.264 pour une meilleure qualité/compression
            # Priorité: avc1 (H.264) > X264 > mp4v (MPEG-4)
            codecs_to_try = [
                ('avc1', 'H.264/AVC'),
                ('X264', 'x264'),
                ('H264', 'H.264'),
                ('mp4v', 'MPEG-4'),
            ]
            
            writer = None
            for codec, codec_name in codecs_to_try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                test_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                if test_writer.isOpened():
                    writer = test_writer
                    logger.info(f"Vidéo de sortie: {output_path} (codec: {codec_name})")
                    break
                test_writer.release()
            
            if writer is None:
                # Fallback: écrire en brut puis encoder avec FFmpeg
                import tempfile
                temp_output_path = output_path.replace('.mp4', '_temp.avi')
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                writer = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
                use_ffmpeg_encode = True
                logger.info(f"Vidéo temporaire: {temp_output_path} (sera encodée en H.264)")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                if max_frames and frame_count > max_frames:
                    break
                
                frame_start = time.time()
                
                # Diviser en régions
                regions = self.split_frame(frame)
                
                all_detections = []
                all_mappings = []
                all_associations = []
                annotated_regions = []
                
                # Traiter chaque région
                for cam_id, region in enumerate(regions):
                    # Détection + tracking
                    detections = self.process_region(region, cam_id)
                    
                    # Extraire features ReID pour les personnes
                    if self.use_reid:
                        self.extract_reid_features(region, detections)
                    
                    # Séparer personnes et objets
                    persons = [d for d in detections if d.class_id == 0]
                    objects = [d for d in detections if d.class_id != 0]
                    
                    # Associer objets aux personnes
                    associations = self.associator.associate(persons, objects, region)
                    
                    # Matching cross-camera
                    if self.cross_camera:
                        global_mapping = self.cross_camera.update(
                            cam_id, detections, frame_count
                        )
                    else:
                        # Fallback: utiliser les track_ids locaux comme globaux
                        global_mapping = {}
                        for det in detections:
                            if det.track_id is not None:
                                key = (cam_id, det.track_id)
                                if key not in self.local_global_mapping:
                                    self.local_global_mapping[key] = self.next_fallback_id
                                    self.next_fallback_id += 1
                                global_mapping[det.track_id] = self.local_global_mapping[key]
                    
                    # === DÉTECTION D'OBJETS ABANDONNÉS ET SUSPECTS ===
                    if self.abandoned_tracker:
                        # Mettre à jour le tracking d'objets abandonnés ET détecter les vols
                        newly_abandoned, ownership_changes = self.abandoned_tracker.update(
                            cam_id, detections, global_mapping, frame_count
                        )
                        
                        # Alertes pour nouveaux objets abandonnés
                        for abandoned in newly_abandoned:
                            self.alert_system.add_abandoned_alert(abandoned, frame_count)
                        
                        # === DÉTECTION DE VOL (changement de propriétaire) ===
                        for new_owner_id, stolen_obj, original_owner_id in ownership_changes:
                            # Marquer le nouveau propriétaire comme SUSPECT
                            self.suspect_tracker.add_suspect(
                                new_owner_id,
                                f"A pris {stolen_obj.class_name} de G{original_owner_id}",
                                stolen_obj,
                                frame_count,
                                cam_id
                            )
                            self.alert_system.add_alert(
                                'VOL_DÉTECTÉ',
                                f"⚠️ G{new_owner_id} a pris {stolen_obj.class_name} (propriétaire: G{original_owner_id})",
                                frame_count,
                                camera_id=cam_id,
                                global_id=new_owner_id
                            )
                        
                        # Vérifier si quelqu'un ramasse un objet abandonné
                        pickups = self.abandoned_tracker.check_pickup(
                            cam_id, detections, global_mapping, frame_count
                        )
                        
                        # Marquer comme suspects et générer alertes
                        for suspect_id, stolen_obj in pickups:
                            self.suspect_tracker.add_suspect(
                                suspect_id,
                                f"A ramassé {stolen_obj.class_name} abandonné",
                                stolen_obj,
                                frame_count,
                                cam_id
                            )
                            self.alert_system.add_suspect_alert(
                                suspect_id, stolen_obj, frame_count, cam_id
                            )
                    
                    # Vérifier si des suspects sont visibles dans cette caméra
                    for det in detections:
                        if det.class_id == 0 and det.track_id is not None:
                            global_id = global_mapping.get(det.track_id)
                            if global_id and self.suspect_tracker.is_suspect(global_id):
                                # Suspect vu - vérifier si nouvelle caméra
                                is_new_cam = self.suspect_tracker.update_sighting(global_id, cam_id)
                                if is_new_cam:
                                    self.alert_system.add_suspect_sighting(
                                        global_id, cam_id, frame_count
                                    )
                    
                    all_detections.append(detections)
                    all_mappings.append(global_mapping)
                    all_associations.append(associations)
                    
                    # Visualiser la région
                    annotated = self.visualize_region(
                        region, detections, global_mapping, associations, cam_id, frame_count
                    )
                    annotated_regions.append(annotated)
                
                # Recomposer la frame complète
                merged_frame = np.hstack(annotated_regions)
                
                # Dessiner les alertes actives
                merged_frame = self.alert_system.draw_alerts(merged_frame)
                self.alert_system.update()  # Décrémenter les durées
                
                # Ajouter info FPS et compteurs
                frame_time = time.time() - frame_start
                current_fps = 1.0 / frame_time if frame_time > 0 else 0
                
                # Stats
                info_text = f"Frame: {frame_count}/{total_frames} | FPS: {current_fps:.1f}"
                if self.cross_camera:
                    stats = self.cross_camera.get_stats()
                    info_text += f" | Tracks: {stats['total_global_tracks']} (Cross: {stats['cross_camera_tracks']})"
                
                # Ajouter stats suspects
                suspects = self.suspect_tracker.get_all_suspects()
                if suspects:
                    info_text += f" | ⚠ Suspects: {len(suspects)}"
                
                cv2.putText(merged_frame, info_text, (10, height - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Écrire la frame de sortie
                if writer:
                    writer.write(merged_frame)
                
                # Afficher
                if show and not headless:
                    # Redimensionner pour affichage si trop grand
                    display_frame = merged_frame
                    if width > 1920:
                        scale = 1920 / width
                        display_frame = cv2.resize(merged_frame, None, fx=scale, fy=scale)
                    
                    cv2.imshow('Multi-Camera Tracking', display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("Arrêt demandé par l'utilisateur")
                        break
                    elif key == ord(' '):
                        # Pause
                        cv2.waitKey(0)
                
                # Log progression
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    avg_fps = frame_count / elapsed
                    logger.info(f"Frame {frame_count}/{total_frames} - FPS moyen: {avg_fps:.1f}")
        
        except KeyboardInterrupt:
            logger.info("Interruption clavier")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if show and not headless:
                cv2.destroyAllWindows()
        
        # Si on a utilisé un fichier temporaire, encoder avec FFmpeg en H.264 haute qualité
        if use_ffmpeg_encode and temp_output_path and output_path:
            import subprocess
            import shutil
            
            logger.info("Encodage final avec FFmpeg (H.264 haute qualité)...")
            
            # Commande FFmpeg pour encoder en H.264 avec haute qualité
            # -crf 18 = qualité quasi lossless, -preset slow = meilleure compression
            # -b:v 12M = bitrate cible de 12 Mbps pour vidéo fluide
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-i', temp_output_path,
                '-c:v', 'libx264',
                '-crf', '18',
                '-preset', 'medium',
                '-b:v', '12M',
                '-maxrate', '15M',
                '-bufsize', '24M',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                output_path
            ]
            
            try:
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info(f"Vidéo encodée avec succès: {output_path}")
                    # Supprimer le fichier temporaire
                    if os.path.exists(temp_output_path):
                        os.remove(temp_output_path)
                else:
                    logger.warning(f"Erreur FFmpeg: {result.stderr}")
                    # Garder le fichier temporaire comme backup
                    if os.path.exists(temp_output_path):
                        shutil.move(temp_output_path, output_path)
            except FileNotFoundError:
                logger.warning("FFmpeg non trouvé, conservation du fichier temporaire")
                if os.path.exists(temp_output_path):
                    shutil.move(temp_output_path, output_path)
        
        # Stats finales
        elapsed = time.time() - start_time
        logger.info(f"Traitement terminé: {frame_count} frames en {elapsed:.1f}s ({frame_count/elapsed:.1f} FPS)")
        
        if self.cross_camera:
            stats = self.cross_camera.get_stats()
            logger.info(f"Statistiques ReID:")
            logger.info(f"  - IDs globaux: {stats['total_global_tracks']}")
            logger.info(f"  - Cross-caméra: {stats['cross_camera_tracks']}")


# =============================================================================
# POINT D'ENTRÉE CLI
# =============================================================================

def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Système de tracking multi-caméras avec ReID",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Traitement avec affichage temps réel
  python multicamera_tracking_concatenated.py --input reseau_final.mp4 --show
  
  # Export vidéo sans affichage
  python multicamera_tracking_concatenated.py --input reseau_final.mp4 --output tracked.mp4 --headless
  
  # Sans ReID (plus rapide)
  python multicamera_tracking_concatenated.py --input reseau_final.mp4 --no-reid
  
  # Test rapide (100 frames)
  python multicamera_tracking_concatenated.py --input reseau_final.mp4 --max-frames 100 --show
        """
    )
    
    # Arguments
    parser.add_argument('--input', '-i', required=True,
                       help='Chemin de la vidéo d\'entrée')
    parser.add_argument('--output', '-o', default=None,
                       help='Chemin de la vidéo de sortie')
    parser.add_argument('--model', '-m', default='yolov8m.pt',
                       help='Modèle YOLO (yolov8n/s/m/l/x.pt)')
    parser.add_argument('--cameras', '-c', type=int, default=4,
                       help='Nombre de vues caméra dans la vidéo')
    parser.add_argument('--tracker', '-t', default='bytetrack',
                       choices=['bytetrack', 'botsort'],
                       help='Type de tracker')
    parser.add_argument('--no-reid', action='store_true',
                       help='Désactiver la ré-identification cross-caméra')
    parser.add_argument('--reid-threshold', type=float, default=0.6,
                       help='Seuil de similarité ReID (0.0-1.0)')
    parser.add_argument('--no-pose', action='store_true',
                       help='Désactiver MediaPipe pose pour association objets')
    parser.add_argument('--show', '-s', action='store_true',
                       help='Afficher en temps réel')
    parser.add_argument('--headless', action='store_true',
                       help='Mode sans affichage (pour serveurs)')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Limiter le nombre de frames à traiter')
    parser.add_argument('--device', default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device pour l\'inférence')
    
    # === NOUVEAUX ARGUMENTS POUR DÉTECTION D'ABANDON ET SUSPECTS ===
    parser.add_argument('--detect-abandoned', action='store_true', default=True,
                       help='Activer la détection d\'objets abandonnés (par défaut: activé)')
    parser.add_argument('--no-abandoned', action='store_true',
                       help='Désactiver la détection d\'objets abandonnés')
    parser.add_argument('--abandon-threshold', type=int, default=25,
                       help='Frames avant qu\'un objet soit abandonné (défaut: 25 = 1 sec @ 25fps)')
    parser.add_argument('--alert-log', type=str, default=None,
                       help='Fichier pour logger les alertes (ex: alerts.log)')
    
    args = parser.parse_args()
    
    # Vérifier le fichier d'entrée
    if not Path(args.input).exists():
        logger.error(f"Fichier non trouvé: {args.input}")
        return 1
    
    # Créer le processeur
    try:
        processor = MultiCameraVideoProcessor(
            model_path=args.model,
            num_cameras=args.cameras,
            tracker_type=args.tracker,
            use_reid=not args.no_reid,
            reid_threshold=args.reid_threshold,
            use_pose=not args.no_pose,
            device=args.device,
            # Nouveaux paramètres
            detect_abandoned=not args.no_abandoned,
            abandon_threshold=args.abandon_threshold,
            alert_log_file=args.alert_log
        )
    except Exception as e:
        logger.error(f"Erreur initialisation: {e}")
        return 1
    
    # Traiter la vidéo
    try:
        processor.process_video(
            input_path=args.input,
            output_path=args.output,
            show=args.show,
            max_frames=args.max_frames,
            headless=args.headless
        )
    except Exception as e:
        logger.error(f"Erreur traitement: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
