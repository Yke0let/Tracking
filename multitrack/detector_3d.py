"""
Module de détection 3D multi-vues pour surveillance.
Utilise la triangulation depuis plusieurs caméras calibrées.
"""

import cv2
import numpy as np
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import logging

from .object_detector import Detection, DetectionResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CameraCalibration:
    """Paramètres de calibration d'une caméra."""
    camera_id: str
    
    # Paramètres intrinsèques
    camera_matrix: np.ndarray  # 3x3 matrice K
    dist_coeffs: np.ndarray    # Coefficients de distorsion
    
    # Paramètres extrinsèques
    rotation: np.ndarray       # 3x3 ou 3x1 (Rodrigues)
    translation: np.ndarray    # 3x1 vecteur T
    
    # Taille de l'image
    image_size: Tuple[int, int] = (1920, 1080)
    
    @property
    def projection_matrix(self) -> np.ndarray:
        """Calcule la matrice de projection P = K[R|t]."""
        if self.rotation.shape == (3, 1) or self.rotation.shape == (1, 3):
            R, _ = cv2.Rodrigues(self.rotation)
        else:
            R = self.rotation
        
        Rt = np.hstack([R, self.translation.reshape(3, 1)])
        return self.camera_matrix @ Rt


@dataclass
class Detection3D:
    """Représente une détection 3D triangulée."""
    position_3d: np.ndarray  # [X, Y, Z] en coordonnées monde
    
    # Détections 2D sources
    detections_2d: Dict[str, Detection]  # camera_id -> Detection
    
    # Métadonnées
    class_id: int
    class_name: str
    confidence: float  # Confiance combinée
    
    # Erreur de reprojection
    reprojection_error: float = 0.0
    
    # Track ID (si tracking multi-vue)
    track_id: Optional[int] = None


class MultiViewTriangulator:
    """
    Triangule les détections 2D de plusieurs caméras en positions 3D.
    """
    
    def __init__(self):
        self.calibrations: Dict[str, CameraCalibration] = {}
    
    def add_camera(self, calibration: CameraCalibration):
        """Ajoute une caméra calibrée."""
        self.calibrations[calibration.camera_id] = calibration
        logger.info(f"✓ Caméra calibrée ajoutée: {calibration.camera_id}")
    
    def load_calibration(self, calibration_path: str):
        """Charge les calibrations depuis un fichier JSON."""
        with open(calibration_path, 'r') as f:
            data = json.load(f)
        
        for cam_data in data.get("cameras", []):
            calib = CameraCalibration(
                camera_id=cam_data["camera_id"],
                camera_matrix=np.array(cam_data["camera_matrix"]),
                dist_coeffs=np.array(cam_data["dist_coeffs"]),
                rotation=np.array(cam_data["rotation"]),
                translation=np.array(cam_data["translation"]),
                image_size=tuple(cam_data.get("image_size", [1920, 1080])),
            )
            self.add_camera(calib)
        
        logger.info(f"✓ Calibration chargée: {len(self.calibrations)} caméras")
    
    def save_calibration(self, output_path: str):
        """Sauvegarde les calibrations dans un fichier JSON."""
        data = {"cameras": []}
        
        for camera_id, calib in self.calibrations.items():
            cam_data = {
                "camera_id": camera_id,
                "camera_matrix": calib.camera_matrix.tolist(),
                "dist_coeffs": calib.dist_coeffs.tolist(),
                "rotation": calib.rotation.tolist(),
                "translation": calib.translation.tolist(),
                "image_size": list(calib.image_size),
            }
            data["cameras"].append(cam_data)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"✓ Calibration sauvegardée: {output_path}")
    
    def triangulate_point(
        self,
        points_2d: Dict[str, Tuple[float, float]],
    ) -> Tuple[np.ndarray, float]:
        """
        Triangule un point 3D depuis plusieurs vues 2D.
        
        Args:
            points_2d: Dict camera_id -> (x, y) en pixels
            
        Returns:
            (point_3d, reprojection_error)
        """
        if len(points_2d) < 2:
            raise ValueError("Au moins 2 vues nécessaires pour la triangulation")
        
        # Collecter les matrices de projection et les points
        proj_matrices = []
        points = []
        camera_ids = []
        
        for camera_id, (x, y) in points_2d.items():
            if camera_id not in self.calibrations:
                continue
            
            calib = self.calibrations[camera_id]
            
            # Point non-distordu
            point_undist = cv2.undistortPoints(
                np.array([[[x, y]]], dtype=np.float32),
                calib.camera_matrix,
                calib.dist_coeffs,
                P=calib.camera_matrix
            )[0][0]
            
            proj_matrices.append(calib.projection_matrix)
            points.append(point_undist)
            camera_ids.append(camera_id)
        
        if len(proj_matrices) < 2:
            raise ValueError("Pas assez de caméras calibrées")
        
        # Triangulation DLT
        point_3d = self._triangulate_dlt(proj_matrices, points)
        
        # Calcul de l'erreur de reprojection
        reprojection_error = self._compute_reprojection_error(
            point_3d, proj_matrices, points
        )
        
        return point_3d, reprojection_error
    
    def _triangulate_dlt(
        self,
        proj_matrices: List[np.ndarray],
        points_2d: List[np.ndarray]
    ) -> np.ndarray:
        """Triangulation par DLT (Direct Linear Transform)."""
        n_views = len(proj_matrices)
        A = np.zeros((2 * n_views, 4))
        
        for i, (P, pt) in enumerate(zip(proj_matrices, points_2d)):
            x, y = pt[0], pt[1]
            A[2*i] = x * P[2] - P[0]
            A[2*i + 1] = y * P[2] - P[1]
        
        # SVD pour résoudre AX = 0
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        
        # Homogène vers euclidien
        return X[:3] / X[3]
    
    def _compute_reprojection_error(
        self,
        point_3d: np.ndarray,
        proj_matrices: List[np.ndarray],
        points_2d: List[np.ndarray]
    ) -> float:
        """Calcule l'erreur de reprojection moyenne."""
        errors = []
        
        X = np.append(point_3d, 1)  # Coordonnées homogènes
        
        for P, pt in zip(proj_matrices, points_2d):
            # Reprojection
            x_proj = P @ X
            x_proj = x_proj[:2] / x_proj[2]
            
            # Erreur
            error = np.linalg.norm(x_proj - pt)
            errors.append(error)
        
        return np.mean(errors)
    
    def match_detections(
        self,
        detections_per_camera: Dict[str, List[Detection]],
        max_distance: float = 100.0,
    ) -> List[Dict[str, Detection]]:
        """
        Associe les détections entre caméras par classe et proximité épipolaire.
        
        Args:
            detections_per_camera: Dict camera_id -> list of detections
            max_distance: Distance max pour le matching
            
        Returns:
            Liste de groupes de détections matchées
        """
        # Grouper par classe
        detections_by_class: Dict[int, Dict[str, List[Detection]]] = {}
        
        for camera_id, detections in detections_per_camera.items():
            for det in detections:
                if det.class_id not in detections_by_class:
                    detections_by_class[det.class_id] = {}
                if camera_id not in detections_by_class[det.class_id]:
                    detections_by_class[det.class_id][camera_id] = []
                detections_by_class[det.class_id][camera_id].append(det)
        
        matched_groups = []
        
        # Pour chaque classe
        for class_id, cameras_dets in detections_by_class.items():
            if len(cameras_dets) < 2:
                continue
            
            # Simple matching par ordre (à améliorer avec contrainte épipolaire)
            camera_ids = list(cameras_dets.keys())
            
            # Prendre la première caméra comme référence
            ref_cam = camera_ids[0]
            ref_dets = cameras_dets[ref_cam]
            
            for ref_det in ref_dets:
                group = {ref_cam: ref_det}
                
                # Chercher le match le plus proche dans les autres caméras
                for other_cam in camera_ids[1:]:
                    other_dets = cameras_dets[other_cam]
                    
                    best_match = None
                    best_dist = float('inf')
                    
                    for other_det in other_dets:
                        # Distance euclidienne simple (à améliorer)
                        dist = np.linalg.norm(
                            np.array(ref_det.center) - np.array(other_det.center)
                        )
                        if dist < best_dist and dist < max_distance:
                            best_dist = dist
                            best_match = other_det
                    
                    if best_match:
                        group[other_cam] = best_match
                
                if len(group) >= 2:
                    matched_groups.append(group)
        
        return matched_groups
    
    def triangulate_detections(
        self,
        detections_per_camera: Dict[str, List[Detection]],
        max_reprojection_error: float = 10.0,
    ) -> List[Detection3D]:
        """
        Triangule les détections matchées en positions 3D.
        
        Args:
            detections_per_camera: Dict camera_id -> list of detections
            max_reprojection_error: Erreur max acceptable
            
        Returns:
            Liste des détections 3D
        """
        # Matcher les détections
        matched_groups = self.match_detections(detections_per_camera)
        
        detections_3d = []
        
        for group in matched_groups:
            # Extraire les centres
            points_2d = {
                cam_id: (det.center[0], det.center[1])
                for cam_id, det in group.items()
            }
            
            try:
                point_3d, error = self.triangulate_point(points_2d)
                
                if error > max_reprojection_error:
                    continue
                
                # Détection de référence pour les métadonnées
                ref_det = list(group.values())[0]
                
                det_3d = Detection3D(
                    position_3d=point_3d,
                    detections_2d=group,
                    class_id=ref_det.class_id,
                    class_name=ref_det.class_name,
                    confidence=np.mean([d.confidence for d in group.values()]),
                    reprojection_error=error,
                    track_id=ref_det.track_id,
                )
                detections_3d.append(det_3d)
                
            except Exception as e:
                logger.warning(f"Triangulation échouée: {e}")
                continue
        
        return detections_3d


class Detector3D:
    """
    Détecteur 3D complet combinant détection 2D et triangulation.
    """
    
    def __init__(
        self,
        model_path: str = "yolov8m.pt",
        calibration_path: Optional[str] = None,
    ):
        from .object_detector import ObjectDetector
        
        self.detector = ObjectDetector(
            model_path=model_path,
            enable_tracking=True,
        )
        
        self.triangulator = MultiViewTriangulator()
        
        if calibration_path:
            self.triangulator.load_calibration(calibration_path)
    
    def detect_multi_view(
        self,
        frames: Dict[str, np.ndarray],
        frame_number: int = 0,
    ) -> Tuple[Dict[str, DetectionResult], List[Detection3D]]:
        """
        Détecte et triangule depuis plusieurs vues.
        
        Args:
            frames: Dict camera_id -> frame
            frame_number: Numéro de la frame
            
        Returns:
            (results_2d, detections_3d)
        """
        # Détection 2D sur chaque caméra
        results_2d = {}
        detections_per_camera = {}
        
        for camera_id, frame in frames.items():
            result = self.detector.detect(frame, camera_id, frame_number)
            results_2d[camera_id] = result
            detections_per_camera[camera_id] = result.detections
        
        # Triangulation
        detections_3d = self.triangulator.triangulate_detections(
            detections_per_camera
        )
        
        return results_2d, detections_3d
    
    def draw_3d_detections(
        self,
        detections_3d: List[Detection3D],
        world_size: Tuple[float, float] = (20.0, 20.0),
        output_size: Tuple[int, int] = (800, 600),
    ) -> np.ndarray:
        """
        Dessine une vue "bird's eye" des détections 3D.
        
        Args:
            detections_3d: Liste des détections 3D
            world_size: Taille du monde en mètres (largeur, profondeur)
            output_size: Taille de l'image de sortie
            
        Returns:
            Image BGR avec vue du dessus
        """
        img = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
        img[:] = (40, 40, 40)  # Fond gris foncé
        
        # Conversion monde -> pixels
        scale_x = output_size[0] / world_size[0]
        scale_y = output_size[1] / world_size[1]
        
        # Grille
        for i in range(int(world_size[0]) + 1):
            x = int(i * scale_x)
            cv2.line(img, (x, 0), (x, output_size[1]), (60, 60, 60), 1)
        for i in range(int(world_size[1]) + 1):
            y = int(i * scale_y)
            cv2.line(img, (0, y), (output_size[0], y), (60, 60, 60), 1)
        
        # Dessiner les détections
        colors = {
            0: (0, 255, 0),    # person
            2: (255, 0, 0),    # car
            3: (0, 0, 255),    # motorcycle
        }
        
        for det in detections_3d:
            X, Y, Z = det.position_3d
            
            # Centrer et convertir
            px = int((X + world_size[0] / 2) * scale_x)
            py = int((Z + world_size[1] / 2) * scale_y)
            
            if 0 <= px < output_size[0] and 0 <= py < output_size[1]:
                color = colors.get(det.class_id, (255, 255, 0))
                
                # Cercle
                cv2.circle(img, (px, py), 10, color, -1)
                
                # Label
                label = f"{det.class_name}"
                if det.track_id:
                    label += f" #{det.track_id}"
                
                cv2.putText(img, label, (px + 15, py), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Légende
        cv2.putText(img, f"Detections 3D: {len(detections_3d)}", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return img


def create_example_calibration(output_path: str):
    """
    Crée un exemple de fichier de calibration.
    Les valeurs sont fictives et doivent être remplacées par une vraie calibration.
    """
    # Matrices intrinsèques fictives (focale ~1000px pour 1080p)
    K = np.array([
        [1000, 0, 960],
        [0, 1000, 540],
        [0, 0, 1]
    ])
    
    dist = np.zeros(5)
    
    data = {
        "cameras": [
            {
                "camera_id": "cam_0",
                "camera_matrix": K.tolist(),
                "dist_coeffs": dist.tolist(),
                "rotation": np.eye(3).tolist(),
                "translation": [0, 0, 0],
                "image_size": [1920, 1080],
            },
            {
                "camera_id": "cam_1",
                "camera_matrix": K.tolist(),
                "dist_coeffs": dist.tolist(),
                "rotation": np.eye(3).tolist(),
                "translation": [2.0, 0, 0],  # 2m à droite
                "image_size": [1920, 1080],
            },
            {
                "camera_id": "cam_2",
                "camera_matrix": K.tolist(),
                "dist_coeffs": dist.tolist(),
                "rotation": np.eye(3).tolist(),
                "translation": [0, 0, 5.0],  # 5m devant
                "image_size": [1920, 1080],
            },
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"✓ Calibration exemple créée: {output_path}")


if __name__ == "__main__":
    # Créer un exemple de calibration
    create_example_calibration("calibration_example.json")
    
    # Test du triangulateur
    triangulator = MultiViewTriangulator()
    triangulator.load_calibration("calibration_example.json")
    
    # Test triangulation
    points = {
        "cam_0": (960, 540),
        "cam_1": (800, 540),
    }
    
    try:
        point_3d, error = triangulator.triangulate_point(points)
        print(f"Point 3D: {point_3d}")
        print(f"Erreur reprojection: {error:.2f}px")
    except Exception as e:
        print(f"Erreur: {e}")
