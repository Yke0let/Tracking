"""
Pipeline de prétraitement vidéo pour créer un dataset cohérent.
Inclut: conversion, normalisation, correction distorsion, stabilisation,
extraction métadonnées, découpage, et augmentation de données.
"""

import cv2
import numpy as np
import subprocess
import json
import os
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Callable, Any
from datetime import datetime
from enum import Enum
import logging
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OutputFormat(Enum):
    """Formats de sortie supportés."""
    MP4_H264 = "mp4_h264"
    MP4_H265 = "mp4_h265"
    AVI_MJPEG = "avi_mjpeg"
    WEBM_VP9 = "webm_vp9"


@dataclass
class PreprocessingConfig:
    """Configuration du pipeline de prétraitement."""
    # Format de sortie
    output_format: OutputFormat = OutputFormat.MP4_H264
    
    # Résolution
    target_width: int = 1280
    target_height: int = 720
    preserve_aspect_ratio: bool = True
    padding_color: Tuple[int, int, int] = (0, 0, 0)
    
    # FPS
    target_fps: float = 25.0
    
    # Qualité (CRF pour H.264/H.265, 0-51, plus bas = meilleur)
    quality_crf: int = 23
    
    # Stabilisation
    enable_stabilization: bool = False
    stabilization_smoothing: int = 30
    
    # Correction de distorsion
    enable_distortion_correction: bool = False
    camera_matrix: Optional[np.ndarray] = None
    dist_coeffs: Optional[np.ndarray] = None
    
    # Découpage
    extract_frames: bool = False
    frame_interval: int = 1  # Extraire toutes les N frames
    clip_duration: Optional[float] = None  # Durée des clips en secondes
    
    # Augmentation
    enable_augmentation: bool = False
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    
    # Sortie
    output_dir: str = "preprocessed"
    overwrite: bool = False


class VideoConverter:
    """Convertit les vidéos vers un format commun."""
    
    CODEC_MAP = {
        OutputFormat.MP4_H264: {"ext": ".mp4", "vcodec": "libx264", "acodec": "aac"},
        OutputFormat.MP4_H265: {"ext": ".mp4", "vcodec": "libx265", "acodec": "aac"},
        OutputFormat.AVI_MJPEG: {"ext": ".avi", "vcodec": "mjpeg", "acodec": "pcm_s16le"},
        OutputFormat.WEBM_VP9: {"ext": ".webm", "vcodec": "libvpx-vp9", "acodec": "libopus"},
    }
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
    
    def convert(self, input_path: str, output_path: str) -> bool:
        """Convertit une vidéo vers le format cible."""
        codec_info = self.CODEC_MAP[self.config.output_format]
        
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-c:v", codec_info["vcodec"],
            "-preset", "medium",
            "-crf", str(self.config.quality_crf),
            "-r", str(self.config.target_fps),
        ]
        
        # Filtre de redimensionnement avec préservation ratio
        if self.config.preserve_aspect_ratio:
            scale_filter = (
                f"scale={self.config.target_width}:{self.config.target_height}:"
                f"force_original_aspect_ratio=decrease,"
                f"pad={self.config.target_width}:{self.config.target_height}:(ow-iw)/2:(oh-ih)/2:"
                f"color=black"
            )
        else:
            scale_filter = f"scale={self.config.target_width}:{self.config.target_height}"
        
        cmd.extend(["-vf", scale_filter])
        
        # Audio (optionnel)
        cmd.extend(["-c:a", codec_info["acodec"], "-b:a", "128k"])
        
        cmd.append(output_path)
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                return False
            return True
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout converting {input_path}")
            return False
        except Exception as e:
            logger.error(f"Error converting {input_path}: {e}")
            return False


class VideoNormalizer:
    """Normalise la résolution et le FPS des vidéos."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
    
    def normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Normalise une frame à la résolution cible."""
        h, w = frame.shape[:2]
        target_w, target_h = self.config.target_width, self.config.target_height
        
        if self.config.preserve_aspect_ratio:
            # Calculer le ratio de redimensionnement
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            # Redimensionner
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Créer l'image avec padding
            output = np.full((target_h, target_w, 3), self.config.padding_color, dtype=np.uint8)
            
            # Centrer l'image
            x_offset = (target_w - new_w) // 2
            y_offset = (target_h - new_h) // 2
            output[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            return output
        else:
            return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    
    def get_output_fps(self, input_fps: float) -> float:
        """Retourne le FPS de sortie."""
        return self.config.target_fps


class DistortionCorrector:
    """Corrige la distorsion de lentille (fisheye, grand-angle)."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self._map1 = None
        self._map2 = None
    
    def calibrate_from_checkerboard(
        self,
        images: List[np.ndarray],
        pattern_size: Tuple[int, int] = (9, 6),
        square_size: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calibre la caméra à partir d'images d'un damier.
        
        Args:
            images: Liste d'images contenant le damier
            pattern_size: Nombre de coins intérieurs (colonnes, lignes)
            square_size: Taille d'un carré en unités arbitraires
            
        Returns:
            camera_matrix, dist_coeffs
        """
        # Points 3D du damier
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        objp *= square_size
        
        obj_points = []  # Points 3D
        img_points = []  # Points 2D
        
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            
            ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
            
            if ret:
                obj_points.append(objp)
                
                # Affiner les coins
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                img_points.append(corners_refined)
        
        if len(obj_points) < 3:
            raise ValueError("Pas assez d'images valides pour la calibration (minimum 3)")
        
        # Calibration
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, gray.shape[::-1], None, None
        )
        
        logger.info(f"Calibration terminée. Erreur RMS: {ret:.4f}")
        
        return camera_matrix, dist_coeffs
    
    def set_calibration(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
        """Définit les paramètres de calibration."""
        self.config.camera_matrix = camera_matrix
        self.config.dist_coeffs = dist_coeffs
        self._map1 = None
        self._map2 = None
    
    def correct_frame(self, frame: np.ndarray) -> np.ndarray:
        """Corrige la distorsion d'une frame."""
        if self.config.camera_matrix is None or self.config.dist_coeffs is None:
            return frame
        
        h, w = frame.shape[:2]
        
        # Calculer les maps une seule fois
        if self._map1 is None or self._map2 is None:
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                self.config.camera_matrix, self.config.dist_coeffs, (w, h), 1, (w, h)
            )
            self._map1, self._map2 = cv2.initUndistortRectifyMap(
                self.config.camera_matrix, self.config.dist_coeffs,
                None, new_camera_matrix, (w, h), cv2.CV_16SC2
            )
        
        return cv2.remap(frame, self._map1, self._map2, cv2.INTER_LINEAR)


class VideoStabilizer:
    """Stabilise les vidéos tremblantes."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.prev_gray = None
        self.transforms = []
    
    def reset(self):
        """Réinitialise le stabilisateur."""
        self.prev_gray = None
        self.transforms = []
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Traite une frame pour la stabilisation."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return frame
        
        # Détecter les features
        prev_pts = cv2.goodFeaturesToTrack(
            self.prev_gray, maxCorners=200, qualityLevel=0.01,
            minDistance=30, blockSize=3
        )
        
        if prev_pts is None or len(prev_pts) < 10:
            self.prev_gray = gray
            return frame
        
        # Suivre les features
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, prev_pts, None)
        
        # Filtrer les bons points
        idx = np.where(status == 1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]
        
        if len(prev_pts) < 10:
            self.prev_gray = gray
            return frame
        
        # Estimer la transformation
        m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
        
        if m is None:
            self.prev_gray = gray
            return frame
        
        # Extraire les paramètres
        dx = m[0, 2]
        dy = m[1, 2]
        da = np.arctan2(m[1, 0], m[0, 0])
        
        self.transforms.append([dx, dy, da])
        
        # Lisser les transformations
        if len(self.transforms) >= self.config.stabilization_smoothing:
            smoothed = self._smooth_transforms()
            
            # Appliquer la correction
            h, w = frame.shape[:2]
            m_smooth = np.zeros((2, 3), np.float64)
            m_smooth[0, 0] = np.cos(smoothed[2])
            m_smooth[0, 1] = -np.sin(smoothed[2])
            m_smooth[1, 0] = np.sin(smoothed[2])
            m_smooth[1, 1] = np.cos(smoothed[2])
            m_smooth[0, 2] = smoothed[0]
            m_smooth[1, 2] = smoothed[1]
            
            frame = cv2.warpAffine(frame, m_smooth, (w, h))
        
        self.prev_gray = gray
        return frame
    
    def _smooth_transforms(self) -> List[float]:
        """Lisse les transformations cumulées."""
        n = len(self.transforms)
        trajectory = np.cumsum(self.transforms, axis=0)
        
        # Lissage par moyenne mobile
        window = self.config.stabilization_smoothing
        kernel = np.ones(window) / window
        
        smoothed_trajectory = np.copy(trajectory)
        for i in range(3):
            smoothed_trajectory[:, i] = np.convolve(trajectory[:, i], kernel, mode='same')
        
        # Différence pour obtenir les transformations lissées
        diff = smoothed_trajectory - trajectory
        
        return self.transforms[-1] + diff[-1].tolist()


class FrameExtractor:
    """Extrait des frames ou des clips depuis les vidéos."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
    
    def extract_frames(
        self,
        video_path: str,
        output_dir: str,
        frame_interval: int = 1,
        format: str = "png"
    ) -> int:
        """
        Extrait des frames d'une vidéo.
        
        Returns:
            Nombre de frames extraites
        """
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return 0
        
        frame_count = 0
        extracted = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                filename = os.path.join(output_dir, f"frame_{extracted:06d}.{format}")
                cv2.imwrite(filename, frame)
                extracted += 1
            
            frame_count += 1
        
        cap.release()
        logger.info(f"Extracted {extracted} frames from {video_path}")
        return extracted
    
    def extract_clips(
        self,
        video_path: str,
        output_dir: str,
        clip_duration: float = 10.0
    ) -> int:
        """
        Découpe une vidéo en clips de durée fixe.
        
        Returns:
            Nombre de clips créés
        """
        os.makedirs(output_dir, exist_ok=True)
        
        basename = Path(video_path).stem
        output_pattern = os.path.join(output_dir, f"{basename}_clip_%03d.mp4")
        
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-c", "copy",
            "-map", "0",
            "-segment_time", str(clip_duration),
            "-f", "segment",
            "-reset_timestamps", "1",
            output_pattern
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            
            # Compter les clips créés
            clips = list(Path(output_dir).glob(f"{basename}_clip_*.mp4"))
            logger.info(f"Created {len(clips)} clips from {video_path}")
            return len(clips)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error creating clips: {e}")
            return 0


class DataAugmenter:
    """Applique des augmentations de données aux frames."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self._albumentations_available = False
        
        try:
            import albumentations as A
            self._albumentations_available = True
            
            self.transform = A.Compose([
                A.RandomBrightnessContrast(
                    brightness_limit=(config.brightness_range[0] - 1, config.brightness_range[1] - 1),
                    contrast_limit=(config.contrast_range[0] - 1, config.contrast_range[1] - 1),
                    p=0.5
                ),
                A.GaussNoise(var_limit=(10, 50), p=0.3),
                A.MotionBlur(blur_limit=3, p=0.2),
            ])
        except ImportError:
            logger.warning("Albumentations not installed. Using basic augmentation.")
    
    def augment_frame(self, frame: np.ndarray) -> np.ndarray:
        """Applique des augmentations à une frame."""
        if self._albumentations_available:
            return self.transform(image=frame)["image"]
        else:
            # Augmentation basique avec OpenCV
            return self._basic_augment(frame)
    
    def _basic_augment(self, frame: np.ndarray) -> np.ndarray:
        """Augmentation basique sans Albumentations."""
        # Ajustement aléatoire de luminosité/contraste
        alpha = np.random.uniform(*self.config.contrast_range)  # Contraste
        beta = np.random.uniform(-30, 30)  # Luminosité
        
        adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        return adjusted
    
    def create_day_night_variants(
        self,
        frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crée des variantes jour/nuit d'une frame.
        
        Returns:
            (day_frame, night_frame)
        """
        # Version jour (plus lumineux, contraste plus élevé)
        day = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)
        
        # Version nuit (plus sombre, moins de contraste)
        night = cv2.convertScaleAbs(frame, alpha=0.6, beta=-40)
        # Ajouter une teinte bleutée pour la nuit
        night = cv2.addWeighted(night, 0.8, np.full_like(night, (40, 20, 0)), 0.2, 0)
        
        return day, night


class MetadataExtractor:
    """Extrait et gère les métadonnées des vidéos."""
    
    @staticmethod
    def extract_with_ffprobe(video_path: str) -> Dict[str, Any]:
        """Extrait les métadonnées avec FFprobe."""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            video_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)
        except Exception as e:
            logger.error(f"FFprobe error for {video_path}: {e}")
            return {}
    
    @staticmethod
    def extract_gps(video_path: str) -> Optional[Dict[str, float]]:
        """Extrait les données GPS si disponibles."""
        try:
            # Utiliser exiftool si disponible
            cmd = ["exiftool", "-json", "-GPS*", video_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                if data and len(data) > 0:
                    gps_data = data[0]
                    if "GPSLatitude" in gps_data and "GPSLongitude" in gps_data:
                        return {
                            "latitude": gps_data.get("GPSLatitude"),
                            "longitude": gps_data.get("GPSLongitude"),
                            "altitude": gps_data.get("GPSAltitude"),
                        }
        except Exception:
            pass
        
        return None
    
    @staticmethod
    def save_calibration(
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        output_path: str
    ):
        """Sauvegarde les paramètres de calibration."""
        data = {
            "camera_matrix": camera_matrix.tolist(),
            "dist_coeffs": dist_coeffs.tolist(),
            "calibration_date": datetime.now().isoformat(),
        }
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def load_calibration(calibration_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Charge les paramètres de calibration."""
        with open(calibration_path, "r") as f:
            data = json.load(f)
        
        camera_matrix = np.array(data["camera_matrix"])
        dist_coeffs = np.array(data["dist_coeffs"])
        
        return camera_matrix, dist_coeffs


class PreprocessingPipeline:
    """Pipeline complet de prétraitement vidéo."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        
        self.converter = VideoConverter(config)
        self.normalizer = VideoNormalizer(config)
        self.distortion_corrector = DistortionCorrector(config)
        self.stabilizer = VideoStabilizer(config)
        self.frame_extractor = FrameExtractor(config)
        self.augmenter = DataAugmenter(config)
        self.metadata_extractor = MetadataExtractor()
    
    def process_video(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Dict[str, Any]:
        """
        Traite une vidéo complète à travers le pipeline.
        
        Returns:
            Dict avec les résultats et métadonnées
        """
        input_path = str(Path(input_path).resolve())
        input_name = Path(input_path).stem
        
        # Créer le dossier de sortie
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        if output_path is None:
            ext = VideoConverter.CODEC_MAP[self.config.output_format]["ext"]
            output_path = os.path.join(self.config.output_dir, f"{input_name}_processed{ext}")
        
        # Vérifier si le fichier existe déjà
        if os.path.exists(output_path) and not self.config.overwrite:
            logger.info(f"Skipping {input_name} (already exists)")
            return {"status": "skipped", "output": output_path}
        
        logger.info(f"Processing: {input_name}")
        
        # Extraire les métadonnées
        metadata = self.metadata_extractor.extract_with_ffprobe(input_path)
        gps_data = self.metadata_extractor.extract_gps(input_path)
        
        # Si on a besoin de traitements frame par frame
        needs_frame_processing = (
            self.config.enable_stabilization or
            self.config.enable_distortion_correction or
            self.config.enable_augmentation
        )
        
        if needs_frame_processing:
            result = self._process_with_frames(input_path, output_path, progress_callback)
        else:
            # Utiliser FFmpeg directement (plus rapide)
            success = self.converter.convert(input_path, output_path)
            result = {"status": "success" if success else "error", "output": output_path}
        
        # Extraire les frames si demandé
        if self.config.extract_frames:
            frames_dir = os.path.join(self.config.output_dir, f"{input_name}_frames")
            self.frame_extractor.extract_frames(
                output_path, frames_dir, self.config.frame_interval
            )
            result["frames_dir"] = frames_dir
        
        # Créer les clips si demandé
        if self.config.clip_duration:
            clips_dir = os.path.join(self.config.output_dir, f"{input_name}_clips")
            self.frame_extractor.extract_clips(
                output_path, clips_dir, self.config.clip_duration
            )
            result["clips_dir"] = clips_dir
        
        result["metadata"] = metadata
        result["gps"] = gps_data
        
        return result
    
    def _process_with_frames(
        self,
        input_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Dict[str, Any]:
        """Traite la vidéo frame par frame."""
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            return {"status": "error", "message": f"Cannot open {input_path}"}
        
        # Récupérer les propriétés
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Créer le writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_path, fourcc, self.config.target_fps,
            (self.config.target_width, self.config.target_height)
        )
        
        # Réinitialiser le stabilisateur
        if self.config.enable_stabilization:
            self.stabilizer.reset()
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Correction de distorsion
            if self.config.enable_distortion_correction:
                frame = self.distortion_corrector.correct_frame(frame)
            
            # Stabilisation
            if self.config.enable_stabilization:
                frame = self.stabilizer.process_frame(frame)
            
            # Normalisation (résolution)
            frame = self.normalizer.normalize_frame(frame)
            
            # Augmentation
            if self.config.enable_augmentation:
                frame = self.augmenter.augment_frame(frame)
            
            out.write(frame)
            frame_count += 1
            
            if progress_callback and frame_count % 100 == 0:
                progress_callback(frame_count / total_frames)
        
        cap.release()
        out.release()
        
        # Réencoder avec FFmpeg pour meilleure compression
        temp_path = output_path + ".temp.mp4"
        shutil.move(output_path, temp_path)
        
        cmd = [
            "ffmpeg", "-y", "-i", temp_path,
            "-c:v", "libx264", "-crf", str(self.config.quality_crf),
            "-preset", "medium", output_path
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            os.remove(temp_path)
        except Exception:
            shutil.move(temp_path, output_path)
        
        return {"status": "success", "output": output_path, "frames_processed": frame_count}


if __name__ == "__main__":
    # Test du pipeline
    config = PreprocessingConfig(
        target_width=1280,
        target_height=720,
        target_fps=25.0,
        output_dir="preprocessed",
    )
    
    pipeline = PreprocessingPipeline(config)
    
    # Tester avec une vidéo du dataset
    test_video = "/home/epl/DATA/S03/MTH2329/APEKE/PROJET/Dataset/CAMERA_HALL_PORTE_ENTREE.mp4"
    
    if os.path.exists(test_video):
        result = pipeline.process_video(test_video)
        print(f"Result: {result}")
