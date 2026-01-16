"""
Module de détection d'objets pour surveillance multi-caméras.
Supporte YOLOv8, YOLOv8-seg, et le tracking d'objets.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Callable
from pathlib import Path
from datetime import datetime
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import conditionnel d'ultralytics
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("Ultralytics non installé. pip install ultralytics")


@dataclass
class Detection:
    """Représente une détection d'objet."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    
    # Optionnel
    track_id: Optional[int] = None
    mask: Optional[np.ndarray] = None
    keypoints: Optional[np.ndarray] = None
    
    # Métadonnées
    camera_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    frame_number: Optional[int] = None


@dataclass
class DetectionResult:
    """Résultat de détection pour une frame."""
    frame: np.ndarray
    detections: List[Detection]
    inference_time_ms: float
    frame_number: int
    timestamp: Optional[datetime] = None


# Classes COCO par défaut
COCO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
    35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket",
    39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife",
    44: "spoon", 45: "bowl", 46: "banana", 47: "apple", 48: "sandwich",
    49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
    54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted plant",
    59: "bed", 60: "dining table", 61: "toilet", 62: "tv", 63: "laptop",
    64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier",
    79: "toothbrush"
}

# Classes de surveillance pertinentes
SURVEILLANCE_CLASSES = {
    "person": 0,       # Personne
    "bicycle": 1,      # Vélo
    "car": 2,          # Voiture
    "motorcycle": 3,   # Moto
    "backpack": 24,    # Sac à dos
    "handbag": 26,     # Sac à main
    "suitcase": 28,    # Valise
    "laptop": 63,      # Ordinateur portable
    "cell phone": 67,  # Téléphone
}


class ObjectDetector:
    """
    Détecteur d'objets basé sur YOLOv8.
    Supporte la détection, la segmentation et le tracking.
    """
    
    def __init__(
        self,
        model_path: str = "yolov8m.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        classes_filter: Optional[List[int]] = None,
        device: str = "auto",
        enable_tracking: bool = False,
    ):
        """
        Initialise le détecteur.
        
        Args:
            model_path: Chemin vers le modèle YOLO (ou nom: yolov8n/s/m/l/x)
            confidence_threshold: Seuil de confiance minimum
            iou_threshold: Seuil IoU pour NMS
            classes_filter: Liste des classes à détecter (None = toutes)
            device: Device (auto, cpu, cuda, 0, 1, etc.)
            enable_tracking: Activer le suivi d'objets
        """
        if not YOLO_AVAILABLE:
            raise ImportError("Ultralytics non installé. pip install ultralytics")
        
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.classes_filter = classes_filter
        self.device = device
        self.enable_tracking = enable_tracking
        
        # Charger le modèle
        logger.info(f"Chargement du modèle: {model_path}")
        self.model = YOLO(model_path)
        
        # Récupérer les noms des classes
        self.class_names = self.model.names
        
        logger.info(f"✓ Modèle chargé ({len(self.class_names)} classes)")
    
    def detect(
        self,
        frame: np.ndarray,
        camera_id: Optional[str] = None,
        frame_number: Optional[int] = None,
    ) -> DetectionResult:
        """
        Détecte les objets dans une frame.
        
        Args:
            frame: Image BGR (numpy array)
            camera_id: ID de la caméra source
            frame_number: Numéro de la frame
            
        Returns:
            DetectionResult avec les détections
        """
        start_time = time.time()
        
        # Détection ou tracking
        if self.enable_tracking:
            results = self.model.track(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                classes=self.classes_filter,
                persist=True,
                verbose=False,
            )
        else:
            results = self.model.predict(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                classes=self.classes_filter,
                verbose=False,
            )
        
        inference_time = (time.time() - start_time) * 1000
        
        # Parser les résultats
        detections = self._parse_results(results[0], camera_id, frame_number)
        
        return DetectionResult(
            frame=frame,
            detections=detections,
            inference_time_ms=inference_time,
            frame_number=frame_number or 0,
            timestamp=datetime.now(),
        )
    
    def _parse_results(
        self,
        result,
        camera_id: Optional[str] = None,
        frame_number: Optional[int] = None,
    ) -> List[Detection]:
        """Parse les résultats YOLO en objets Detection."""
        detections = []
        
        if result.boxes is None:
            return detections
        
        boxes = result.boxes
        
        for i in range(len(boxes)):
            # Bounding box
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
            
            # Classe et confiance
            class_id = int(boxes.cls[i].cpu().numpy())
            confidence = float(boxes.conf[i].cpu().numpy())
            class_name = self.class_names.get(class_id, f"class_{class_id}")
            
            # Centre
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Track ID (si tracking activé)
            track_id = None
            if boxes.id is not None:
                track_id = int(boxes.id[i].cpu().numpy())
            
            # Mask (si segmentation)
            mask = None
            if result.masks is not None:
                mask = result.masks.data[i].cpu().numpy()
            
            detection = Detection(
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                bbox=(x1, y1, x2, y2),
                center=(cx, cy),
                track_id=track_id,
                mask=mask,
                camera_id=camera_id,
                frame_number=frame_number,
                timestamp=datetime.now(),
            )
            detections.append(detection)
        
        return detections
    
    def detect_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        show: bool = False,
        max_frames: Optional[int] = None,
        callback: Optional[Callable[[DetectionResult], None]] = None,
    ) -> List[DetectionResult]:
        """
        Détecte les objets dans une vidéo.
        
        Args:
            video_path: Chemin de la vidéo
            output_path: Chemin de sortie (None = pas de sauvegarde)
            show: Afficher en temps réel
            max_frames: Nombre max de frames à traiter
            callback: Fonction appelée pour chaque frame
            
        Returns:
            Liste des DetectionResult
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Impossible d'ouvrir: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        camera_id = Path(video_path).stem
        
        # Writer vidéo si output spécifié
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        results = []
        frame_number = 0
        
        logger.info(f"🎥 Traitement: {video_path}")
        logger.info(f"   Frames: {total_frames}, FPS: {fps:.1f}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if max_frames and frame_number >= max_frames:
                    break
                
                # Détection
                result = self.detect(frame, camera_id, frame_number)
                results.append(result)
                
                # Callback
                if callback:
                    callback(result)
                
                # Dessiner les détections
                annotated = self.draw_detections(frame, result.detections)
                
                # Sauvegarder
                if writer:
                    writer.write(annotated)
                
                # Afficher
                if show:
                    cv2.imshow("Detection", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_number += 1
                
                # Progress
                if frame_number % 100 == 0:
                    progress = frame_number / total_frames * 100
                    avg_time = np.mean([r.inference_time_ms for r in results[-100:]])
                    logger.info(f"   {progress:.1f}% - {avg_time:.1f}ms/frame")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if show:
                cv2.destroyAllWindows()
        
        logger.info(f"✓ Terminé: {len(results)} frames traitées")
        
        return results
    
    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
    ) -> np.ndarray:
        """Dessine les détections sur une frame."""
        annotated = frame.copy()
        
        # Couleurs par défaut
        default_colors = {
            0: (0, 255, 0),    # person - vert
            2: (255, 0, 0),    # car - bleu
            3: (0, 0, 255),    # motorcycle - rouge
            24: (255, 255, 0), # backpack - cyan
            26: (255, 0, 255), # handbag - magenta
            63: (0, 165, 255), # laptop - orange
            67: (203, 192, 255), # cell phone - rose
        }
        colors = colors or default_colors
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = colors.get(det.class_id, (0, 255, 255))
            
            # Rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Label
            label = f"{det.class_name}"
            if det.track_id is not None:
                label += f" #{det.track_id}"
            label += f" {det.confidence:.2f}"
            
            # Fond du label
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - label_h - 5), (x1 + label_w, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Mask (si disponible)
            if det.mask is not None:
                mask_resized = cv2.resize(det.mask, (frame.shape[1], frame.shape[0]))
                mask_bool = mask_resized > 0.5
                overlay = annotated.copy()
                overlay[mask_bool] = color
                annotated = cv2.addWeighted(annotated, 0.7, overlay, 0.3, 0)
        
        return annotated
    
    def get_statistics(self, results: List[DetectionResult]) -> Dict[str, Any]:
        """Calcule les statistiques de détection."""
        if not results:
            return {}
        
        all_detections = [d for r in results for d in r.detections]
        
        # Comptage par classe
        class_counts = {}
        for det in all_detections:
            class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
        
        # Temps d'inférence
        times = [r.inference_time_ms for r in results]
        
        return {
            "total_frames": len(results),
            "total_detections": len(all_detections),
            "avg_detections_per_frame": len(all_detections) / len(results),
            "class_counts": class_counts,
            "avg_inference_ms": np.mean(times),
            "min_inference_ms": np.min(times),
            "max_inference_ms": np.max(times),
        }


class SegmentationDetector(ObjectDetector):
    """Détecteur avec segmentation d'instance (YOLOv8-seg)."""
    
    def __init__(
        self,
        model_path: str = "yolov8m-seg.pt",
        **kwargs
    ):
        super().__init__(model_path=model_path, **kwargs)


class MultiCameraDetector:
    """
    Détecteur multi-caméras synchronisé.
    Gère plusieurs flux vidéo en parallèle.
    """
    
    def __init__(
        self,
        model_path: str = "yolov8m.pt",
        confidence_threshold: float = 0.5,
        enable_tracking: bool = True,
    ):
        self.detector = ObjectDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            enable_tracking=enable_tracking,
        )
        
        self.cameras: Dict[str, cv2.VideoCapture] = {}
        self.results: Dict[str, List[DetectionResult]] = {}
    
    def add_camera(self, camera_id: str, source: str):
        """Ajoute une source vidéo."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Impossible d'ouvrir: {source}")
        
        self.cameras[camera_id] = cap
        self.results[camera_id] = []
        logger.info(f"✓ Caméra ajoutée: {camera_id}")
    
    def detect_all(self) -> Dict[str, DetectionResult]:
        """Détecte sur toutes les caméras (une frame)."""
        frame_results = {}
        
        for camera_id, cap in self.cameras.items():
            ret, frame = cap.read()
            if not ret:
                continue
            
            result = self.detector.detect(frame, camera_id)
            frame_results[camera_id] = result
            self.results[camera_id].append(result)
        
        return frame_results
    
    def release(self):
        """Libère les ressources."""
        for cap in self.cameras.values():
            cap.release()


def create_surveillance_detector(
    model_size: str = "m",
    enable_tracking: bool = True,
    confidence: float = 0.5,
) -> ObjectDetector:
    """
    Crée un détecteur optimisé pour la surveillance.
    
    Args:
        model_size: n (nano), s (small), m (medium), l (large), x (xlarge)
        enable_tracking: Activer le suivi
        confidence: Seuil de confiance
        
    Returns:
        ObjectDetector configuré
    """
    model_path = f"yolov8{model_size}.pt"
    
    # Classes pertinentes pour la surveillance (IDs COCO)
    surveillance_classes = [0, 1, 2, 3, 24, 26, 28, 63, 67]
    
    return ObjectDetector(
        model_path=model_path,
        confidence_threshold=confidence,
        classes_filter=surveillance_classes,
        enable_tracking=enable_tracking,
    )


if __name__ == "__main__":
    # Test du détecteur
    detector = create_surveillance_detector(model_size="m", enable_tracking=True)
    
    # Tester sur une vidéo
    test_video = "/home/epl/DATA/S03/MTH2329/APEKE/PROJET/Dataset/CAMERA_HALL_PORTE_ENTREE.mp4"
    
    import os
    if os.path.exists(test_video):
        results = detector.detect_video(
            test_video,
            output_path="detection_output.mp4",
            show=True,
            max_frames=500,
        )
        
        stats = detector.get_statistics(results)
        print(f"\nStatistiques: {stats}")
