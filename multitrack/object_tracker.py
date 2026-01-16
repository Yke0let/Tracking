"""
Module de tracking multi-objets avec visualisation des trajectoires.
Intègre YOLOv8 + ByteTrack/BoT-SORT pour le suivi persistant.
"""

import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
import logging
import time
import colorsys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("Ultralytics non installé. pip install ultralytics")


@dataclass
class TrackedObject:
    """Représente un objet suivi avec son historique."""
    track_id: int
    class_id: int
    class_name: str
    
    # Position actuelle
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    confidence: float
    
    # Historique des positions (pour trajectoire)
    trajectory: deque = field(default_factory=lambda: deque(maxlen=50))
    
    # Métadonnées
    first_seen: int = 0  # Frame number
    last_seen: int = 0
    total_frames: int = 0
    
    # État
    is_active: bool = True
    frames_since_seen: int = 0
    
    def update(self, bbox: Tuple[int, int, int, int], confidence: float, frame_number: int):
        """Met à jour la position de l'objet."""
        self.bbox = bbox
        self.confidence = confidence
        x1, y1, x2, y2 = bbox
        self.center = ((x1 + x2) // 2, (y1 + y2) // 2)
        self.trajectory.append(self.center)
        self.last_seen = frame_number
        self.total_frames += 1
        self.frames_since_seen = 0
        self.is_active = True


@dataclass
class TrackingResult:
    """Résultat du tracking pour une frame."""
    frame: np.ndarray
    tracked_objects: List[TrackedObject]
    frame_number: int
    inference_time_ms: float
    active_tracks: int
    total_tracks: int


class ObjectTracker:
    """
    Tracker multi-objets avec YOLOv8 + ByteTrack/BoT-SORT.
    Inclut visualisation des trajectoires.
    """
    
    # Trackers disponibles dans Ultralytics
    AVAILABLE_TRACKERS = ["bytetrack", "botsort"]
    
    def __init__(
        self,
        model_path: str = "yolov8m.pt",
        tracker: str = "bytetrack",
        confidence_threshold: float = 0.5,
        classes_filter: Optional[List[int]] = None,
        trajectory_length: int = 50,
        max_age: int = 30,  # Frames avant de considérer perdu
    ):
        """
        Initialise le tracker.
        
        Args:
            model_path: Chemin du modèle YOLO
            tracker: Type de tracker (bytetrack, botsort)
            confidence_threshold: Seuil de confiance
            classes_filter: IDs des classes à tracker
            trajectory_length: Longueur max de la trajectoire
            max_age: Frames max sans détection avant suppression
        """
        if not YOLO_AVAILABLE:
            raise ImportError("Ultralytics non installé. pip install ultralytics")
        
        self.model = YOLO(model_path)
        self.tracker_type = tracker
        self.confidence = confidence_threshold
        self.classes_filter = classes_filter
        self.trajectory_length = trajectory_length
        self.max_age = max_age
        
        # Dictionnaire des objets suivis
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.frame_count = 0
        self.total_tracks_created = 0
        
        # Couleurs par ID
        self._colors: Dict[int, Tuple[int, int, int]] = {}
        
        logger.info(f"✓ Tracker initialisé: {tracker}")
        logger.info(f"   Modèle: {model_path}")
        logger.info(f"   Classes: {classes_filter if classes_filter else 'toutes'}")
    
    def _get_color(self, track_id: int) -> Tuple[int, int, int]:
        """Génère une couleur unique pour chaque ID."""
        if track_id not in self._colors:
            hue = (track_id * 0.618033988749895) % 1.0
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.95)
            self._colors[track_id] = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        return self._colors[track_id]
    
    def track(self, frame: np.ndarray) -> TrackingResult:
        """
        Effectue le tracking sur une frame.
        
        Args:
            frame: Image BGR
            
        Returns:
            TrackingResult avec les objets suivis
        """
        start_time = time.time()
        
        # Tracking avec YOLO
        results = self.model.track(
            frame,
            conf=self.confidence,
            classes=self.classes_filter,
            tracker=f"{self.tracker_type}.yaml",
            persist=True,
            verbose=False,
        )
        
        # Parser les résultats
        current_ids = set()
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                track_id = int(boxes.id[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                confidence = float(boxes.conf[i].cpu().numpy())
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                bbox = (x1, y1, x2, y2)
                
                current_ids.add(track_id)
                
                if track_id in self.tracked_objects:
                    # Mettre à jour l'objet existant
                    self.tracked_objects[track_id].update(bbox, confidence, self.frame_count)
                else:
                    # Nouvel objet
                    class_name = self.model.names.get(class_id, f"class_{class_id}")
                    obj = TrackedObject(
                        track_id=track_id,
                        class_id=class_id,
                        class_name=class_name,
                        bbox=bbox,
                        center=((x1 + x2) // 2, (y1 + y2) // 2),
                        confidence=confidence,
                        trajectory=deque(maxlen=self.trajectory_length),
                        first_seen=self.frame_count,
                        last_seen=self.frame_count,
                    )
                    obj.trajectory.append(obj.center)
                    self.tracked_objects[track_id] = obj
                    self.total_tracks_created += 1
        
        # Marquer les objets non vus
        for track_id, obj in list(self.tracked_objects.items()):
            if track_id not in current_ids:
                obj.frames_since_seen += 1
                if obj.frames_since_seen > self.max_age:
                    obj.is_active = False
        
        self.frame_count += 1
        inference_time = (time.time() - start_time) * 1000
        
        # Liste des objets actuellement visibles (détectés sur cette frame)
        current_objects = [obj for obj in self.tracked_objects.values() 
                          if obj.is_active and obj.frames_since_seen == 0]
        
        return TrackingResult(
            frame=frame,
            tracked_objects=current_objects,
            frame_number=self.frame_count,
            inference_time_ms=inference_time,
            active_tracks=len(current_objects),
            total_tracks=self.total_tracks_created,
        )
    
    def draw_tracking(
        self,
        frame: np.ndarray,
        result: TrackingResult,
        show_trajectory: bool = True,
        show_bbox: bool = True,
        show_id: bool = True,
        trajectory_thickness: int = 2,
    ) -> np.ndarray:
        """
        Dessine les résultats du tracking sur la frame.
        
        Args:
            frame: Image BGR
            result: Résultat du tracking
            show_trajectory: Afficher les trajectoires
            show_bbox: Afficher les bounding boxes
            show_id: Afficher les IDs
            trajectory_thickness: Épaisseur des trajectoires
            
        Returns:
            Image annotée
        """
        annotated = frame.copy()
        
        for obj in result.tracked_objects:
            color = self._get_color(obj.track_id)
            
            # Trajectoire
            if show_trajectory and len(obj.trajectory) > 1:
                points = list(obj.trajectory)
                for i in range(1, len(points)):
                    # Gradient d'opacité
                    alpha = i / len(points)
                    thickness = max(1, int(trajectory_thickness * alpha))
                    cv2.line(annotated, points[i-1], points[i], color, thickness)
            
            # Bounding box
            if show_bbox:
                x1, y1, x2, y2 = obj.bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # Centre
                cv2.circle(annotated, obj.center, 4, color, -1)
            
            # Label avec ID
            if show_id:
                x1, y1, x2, y2 = obj.bbox
                label = f"#{obj.track_id} {obj.class_name} {obj.confidence:.2f}"
                
                (label_w, label_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    annotated, (x1, y1 - label_h - 6), 
                    (x1 + label_w + 4, y1), color, -1
                )
                cv2.putText(
                    annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                )
        
        # Info overlay
        info_text = f"Tracks: {result.active_tracks} | Total: {result.total_tracks} | {result.inference_time_ms:.1f}ms"
        cv2.putText(annotated, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return annotated
    
    def track_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        show: bool = False,
        max_frames: Optional[int] = None,
        show_trajectory: bool = True,
    ) -> List[TrackingResult]:
        """
        Effectue le tracking sur une vidéo complète.
        
        Args:
            video_path: Chemin de la vidéo
            output_path: Chemin de sortie (None = pas de sauvegarde)
            show: Afficher en temps réel
            max_frames: Nombre max de frames
            show_trajectory: Afficher les trajectoires
            
        Returns:
            Liste des TrackingResult
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Impossible d'ouvrir: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        results = []
        
        logger.info(f"🎥 Tracking: {video_path}")
        logger.info(f"   Frames: {total_frames}, FPS: {fps:.1f}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if max_frames and self.frame_count >= max_frames:
                    break
                
                # Tracking
                result = self.track(frame)
                results.append(result)
                
                # Visualisation
                annotated = self.draw_tracking(
                    frame, result, show_trajectory=show_trajectory
                )
                
                if writer:
                    writer.write(annotated)
                
                if show:
                    cv2.imshow("Tracking", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Progress
                if self.frame_count % 100 == 0:
                    progress = self.frame_count / total_frames * 100
                    logger.info(f"   {progress:.1f}% - {result.active_tracks} tracks actifs")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if show:
                cv2.destroyAllWindows()
        
        logger.info(f"✓ Terminé: {len(results)} frames, {self.total_tracks_created} tracks créés")
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques du tracking."""
        active = [o for o in self.tracked_objects.values() if o.is_active]
        inactive = [o for o in self.tracked_objects.values() if not o.is_active]
        
        class_counts = {}
        for obj in self.tracked_objects.values():
            class_counts[obj.class_name] = class_counts.get(obj.class_name, 0) + 1
        
        return {
            "total_frames": self.frame_count,
            "total_tracks": self.total_tracks_created,
            "active_tracks": len(active),
            "lost_tracks": len(inactive),
            "class_distribution": class_counts,
            "avg_track_length": np.mean([o.total_frames for o in self.tracked_objects.values()]) if self.tracked_objects else 0,
        }
    
    def reset(self):
        """Réinitialise le tracker."""
        self.tracked_objects.clear()
        self.frame_count = 0
        self.total_tracks_created = 0
        self._colors.clear()


def create_surveillance_tracker(
    model_size: str = "m",
    tracker: str = "bytetrack",
    confidence: float = 0.5,
) -> ObjectTracker:
    """
    Crée un tracker optimisé pour la surveillance.
    
    Args:
        model_size: n, s, m, l, x
        tracker: bytetrack ou botsort
        confidence: Seuil de confiance
        
    Returns:
        ObjectTracker configuré
    """
    model_path = f"yolov8{model_size}.pt"
    
    # Classes surveillance
    surveillance_classes = [0, 1, 2, 3, 24, 26, 28, 63, 67]
    
    return ObjectTracker(
        model_path=model_path,
        tracker=tracker,
        confidence_threshold=confidence,
        classes_filter=surveillance_classes,
    )


# ============================================================================
# MÉTRIQUES D'ÉVALUATION
# ============================================================================

class TrackingMetrics:
    """
    Calcul des métriques de tracking: MOTA, IDF1, HOTA.
    
    Pour utiliser ces métriques, vous avez besoin d'annotations ground truth
    au format MOTChallenge.
    """
    
    @staticmethod
    def compute_iou(box1: Tuple, box2: Tuple) -> float:
        """Calcule l'IoU entre deux bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0
    
    @staticmethod
    def compute_mota(
        gt_tracks: Dict[int, List[Tuple]],
        pred_tracks: Dict[int, List[Tuple]],
        iou_threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Calcule MOTA (Multiple Object Tracking Accuracy).
        
        MOTA = 1 - (FN + FP + IDSW) / GT
        
        Args:
            gt_tracks: Ground truth {frame: [(id, x1, y1, x2, y2), ...]}
            pred_tracks: Prédictions {frame: [(id, x1, y1, x2, y2), ...]}
            iou_threshold: Seuil IoU pour matching
            
        Returns:
            Dict avec FN, FP, IDSW, MOTA
        """
        fn = 0  # False Negatives (missed)
        fp = 0  # False Positives
        idsw = 0  # ID Switches
        gt_total = 0
        
        last_match = {}  # gt_id -> pred_id
        
        for frame in sorted(set(gt_tracks.keys()) | set(pred_tracks.keys())):
            gt = gt_tracks.get(frame, [])
            pred = pred_tracks.get(frame, [])
            
            gt_total += len(gt)
            
            # Matching par IoU
            matched_gt = set()
            matched_pred = set()
            
            for gt_obj in gt:
                gt_id, *gt_box = gt_obj
                best_iou = 0
                best_pred = None
                
                for pred_obj in pred:
                    pred_id, *pred_box = pred_obj
                    if pred_id in matched_pred:
                        continue
                    
                    iou = TrackingMetrics.compute_iou(gt_box, pred_box)
                    if iou > best_iou and iou >= iou_threshold:
                        best_iou = iou
                        best_pred = pred_obj
                
                if best_pred:
                    pred_id = best_pred[0]
                    matched_gt.add(gt_id)
                    matched_pred.add(pred_id)
                    
                    # Check ID switch
                    if gt_id in last_match and last_match[gt_id] != pred_id:
                        idsw += 1
                    last_match[gt_id] = pred_id
            
            fn += len(gt) - len(matched_gt)
            fp += len(pred) - len(matched_pred)
        
        mota = 1 - (fn + fp + idsw) / gt_total if gt_total > 0 else 0
        
        return {
            "MOTA": mota,
            "FN": fn,
            "FP": fp,
            "IDSW": idsw,
            "GT": gt_total,
        }


if __name__ == "__main__":
    # Test du tracker
    tracker = create_surveillance_tracker(model_size="m", tracker="bytetrack")
    
    test_video = "/home/epl/DATA/S03/MTH2329/APEKE/PROJET/Dataset/CAMERA_HALL_PORTE_ENTREE.mp4"
    
    import os
    if os.path.exists(test_video):
        results = tracker.track_video(
            test_video,
            output_path="tracking_output.mp4",
            show=True,
            max_frames=500,
            show_trajectory=True,
        )
        
        stats = tracker.get_statistics()
        print(f"\nStatistiques: {stats}")
