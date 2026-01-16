"""
Module de fusion et affichage de flux synchronisés.
Crée des grilles multi-caméras, exporte des vidéos composites,
et gère l'affichage en temps réel.
"""

import cv2
import numpy as np
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple, Any
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GridLayout:
    """Configuration de la grille d'affichage."""
    rows: int
    cols: int
    cell_width: int
    cell_height: int
    padding: int = 5
    background_color: Tuple[int, int, int] = (30, 30, 30)
    
    @property
    def total_width(self) -> int:
        return self.cols * (self.cell_width + self.padding) + self.padding
    
    @property
    def total_height(self) -> int:
        return self.rows * (self.cell_height + self.padding) + self.padding
    
    def get_cell_position(self, index: int) -> Tuple[int, int]:
        """Retourne la position (x, y) du coin supérieur gauche d'une cellule."""
        row = index // self.cols
        col = index % self.cols
        x = self.padding + col * (self.cell_width + self.padding)
        y = self.padding + row * (self.cell_height + self.padding)
        return x, y


class SyncMerger:
    """
    Fusionne et affiche des flux vidéo synchronisés.
    Supporte l'affichage en grille et l'export vidéo.
    """
    
    def __init__(
        self,
        layout: Optional[GridLayout] = None,
        target_fps: float = 25.0,
        show_timestamps: bool = True,
        show_camera_names: bool = True,
    ):
        """
        Initialise le merger.
        
        Args:
            layout: Configuration de la grille
            target_fps: FPS cible pour l'affichage/export
            show_timestamps: Afficher les timestamps sur chaque vue
            show_camera_names: Afficher les noms de caméras
        """
        self.layout = layout
        self.target_fps = target_fps
        self.show_timestamps = show_timestamps
        self.show_camera_names = show_camera_names
        
        self.camera_order: List[str] = []
        self.camera_names: Dict[str, str] = {}
        
        # État
        self._running = False
        self._lock = threading.Lock()
        
        # Writer pour l'export
        self._writer: Optional[cv2.VideoWriter] = None
        
    def configure_cameras(
        self,
        camera_ids: List[str],
        names: Optional[Dict[str, str]] = None
    ):
        """
        Configure les caméras à afficher.
        
        Args:
            camera_ids: Liste ordonnée des IDs de caméras
            names: Dict camera_id -> nom d'affichage
        """
        self.camera_order = camera_ids
        self.camera_names = names or {cid: cid for cid in camera_ids}
        
        # Auto-calculer le layout si non spécifié
        if self.layout is None:
            n = len(camera_ids)
            if n <= 1:
                rows, cols = 1, 1
            elif n <= 2:
                rows, cols = 1, 2
            elif n <= 4:
                rows, cols = 2, 2
            elif n <= 6:
                rows, cols = 2, 3
            elif n <= 9:
                rows, cols = 3, 3
            elif n <= 12:
                rows, cols = 3, 4
            else:
                rows, cols = 4, 4
            
            self.layout = GridLayout(
                rows=rows,
                cols=cols,
                cell_width=640,
                cell_height=480,
            )
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Redimensionne une frame à la taille de cellule."""
        if self.layout is None:
            return frame
        
        return cv2.resize(
            frame,
            (self.layout.cell_width, self.layout.cell_height),
            interpolation=cv2.INTER_LINEAR
        )
    
    def _add_overlay(
        self,
        frame: np.ndarray,
        camera_id: str,
        timestamp: Optional[float] = None,
        status: str = "OK"
    ) -> np.ndarray:
        """Ajoute les overlays (nom, timestamp, status) sur une frame."""
        h, w = frame.shape[:2]
        
        # Fond semi-transparent pour le texte
        overlay = frame.copy()
        
        # Bandeau en haut
        cv2.rectangle(overlay, (0, 0), (w, 30), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Nom de la caméra
        if self.show_camera_names:
            name = self.camera_names.get(camera_id, camera_id)
            cv2.putText(
                frame, name,
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
            )
        
        # Timestamp
        if self.show_timestamps and timestamp:
            ts_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S.%f")[:-4]
            cv2.putText(
                frame, ts_str,
                (w - 120, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
            )
        
        # Indicateur de statut
        if status != "OK":
            color = (0, 0, 255) if status == "ERROR" else (0, 165, 255)
            cv2.circle(frame, (w - 15, 15), 8, color, -1)
        
        return frame
    
    def _create_placeholder(self, camera_id: str, message: str = "No Signal") -> np.ndarray:
        """Crée une image placeholder pour une caméra sans signal."""
        if self.layout is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        frame = np.zeros(
            (self.layout.cell_height, self.layout.cell_width, 3),
            dtype=np.uint8
        )
        
        # Fond gris foncé
        frame[:] = (40, 40, 40)
        
        # Texte centré
        name = self.camera_names.get(camera_id, camera_id)
        
        # Nom de la caméra
        text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        x = (self.layout.cell_width - text_size[0]) // 2
        cv2.putText(
            frame, name,
            (x, self.layout.cell_height // 2 - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2
        )
        
        # Message d'erreur
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        x = (self.layout.cell_width - text_size[0]) // 2
        cv2.putText(
            frame, message,
            (x, self.layout.cell_height // 2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1
        )
        
        return frame
    
    def create_grid_frame(
        self,
        frames: Dict[str, Optional[Tuple[np.ndarray, float]]],
        status: Optional[Dict[str, str]] = None
    ) -> np.ndarray:
        """
        Crée une frame de grille à partir des frames individuelles.
        
        Args:
            frames: Dict camera_id -> (frame, timestamp) ou None
            status: Dict camera_id -> status string
            
        Returns:
            Frame composite de la grille
        """
        if self.layout is None:
            raise ValueError("Layout non configuré. Appelez configure_cameras() d'abord.")
        
        # Créer le canvas
        grid = np.zeros(
            (self.layout.total_height, self.layout.total_width, 3),
            dtype=np.uint8
        )
        grid[:] = self.layout.background_color
        
        # Placer chaque frame
        for i, camera_id in enumerate(self.camera_order):
            if i >= self.layout.rows * self.layout.cols:
                break
            
            x, y = self.layout.get_cell_position(i)
            
            frame_data = frames.get(camera_id)
            cam_status = status.get(camera_id, "OK") if status else "OK"
            
            if frame_data is not None:
                frame, timestamp = frame_data
                cell = self._resize_frame(frame)
                cell = self._add_overlay(cell, camera_id, timestamp, cam_status)
            else:
                cell = self._create_placeholder(camera_id, "No Signal")
            
            # Placer dans la grille
            grid[y:y+self.layout.cell_height, x:x+self.layout.cell_width] = cell
        
        return grid
    
    def start_recording(
        self,
        output_path: str,
        codec: str = "mp4v"
    ):
        """
        Démarre l'enregistrement de la grille.
        
        Args:
            output_path: Chemin du fichier de sortie
            codec: Codec vidéo (mp4v, XVID, H264)
        """
        if self.layout is None:
            raise ValueError("Layout non configuré")
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = cv2.VideoWriter(
            output_path,
            fourcc,
            self.target_fps,
            (self.layout.total_width, self.layout.total_height)
        )
        
        logger.info(f"Enregistrement démarré: {output_path}")
    
    def write_frame(self, grid_frame: np.ndarray):
        """Écrit une frame dans le fichier de sortie."""
        if self._writer is not None:
            self._writer.write(grid_frame)
    
    def stop_recording(self):
        """Arrête l'enregistrement."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            logger.info("Enregistrement terminé")
    
    def display_live(
        self,
        frame_source: callable,
        window_name: str = "Multi-Camera View",
        status_source: Optional[callable] = None,
        record_path: Optional[str] = None
    ):
        """
        Affiche les flux en temps réel.
        
        Args:
            frame_source: Fonction retournant Dict[camera_id, (frame, timestamp)]
            window_name: Nom de la fenêtre
            status_source: Fonction retournant Dict[camera_id, status]
            record_path: Chemin pour enregistrement simultané (optionnel)
        """
        if record_path:
            self.start_recording(record_path)
        
        self._running = True
        frame_interval = 1.0 / self.target_fps
        
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        try:
            while self._running:
                start_time = time.time()
                
                # Récupérer les frames
                frames = frame_source()
                status = status_source() if status_source else None
                
                # Créer la grille
                grid = self.create_grid_frame(frames, status)
                
                # Afficher
                cv2.imshow(window_name, grid)
                
                # Enregistrer si actif
                if self._writer:
                    self.write_frame(grid)
                
                # Contrôle du framerate
                elapsed = time.time() - start_time
                wait_time = max(1, int((frame_interval - elapsed) * 1000))
                
                key = cv2.waitKey(wait_time) & 0xFF
                if key == ord('q') or key == 27:  # q ou Escape
                    break
                elif key == ord('r'):  # Toggle recording
                    if self._writer:
                        self.stop_recording()
                    else:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        self.start_recording(f"recording_{ts}.mp4")
                elif key == ord('s'):  # Screenshot
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"screenshot_{ts}.png", grid)
                    logger.info(f"Screenshot sauvegardé")
        
        finally:
            self._running = False
            self.stop_recording()
            cv2.destroyWindow(window_name)
    
    def stop(self):
        """Arrête l'affichage."""
        self._running = False


def merge_videos_to_grid(
    video_paths: Dict[str, str],
    output_path: str,
    target_fps: float = 25.0,
    max_duration: Optional[float] = None,
    show_progress: bool = True
) -> str:
    """
    Fusionne plusieurs vidéos en une grille synchronisée.
    
    Args:
        video_paths: Dict camera_id -> chemin vidéo
        output_path: Chemin de sortie
        target_fps: FPS de sortie
        max_duration: Durée max en secondes
        show_progress: Afficher la progression
        
    Returns:
        Chemin du fichier créé
    """
    from .video_sync import SynchronizedReader, SyncResult, SyncMethod
    from .video_metadata import extract_video_metadata
    
    # Créer les résultats de sync (pas d'offset pour un merge simple)
    sync_results = []
    for camera_id, path in video_paths.items():
        meta = extract_video_metadata(path)
        sync_results.append(SyncResult(
            video_path=path,
            camera_id=camera_id,
            offset_seconds=0.0,
            offset_frames=0,
            confidence=1.0,
            method_used=SyncMethod.MANUAL,
            original_fps=meta.fps,
            target_fps=target_fps,
        ))
    
    # Configurer le merger
    merger = SyncMerger(target_fps=target_fps)
    merger.configure_cameras(
        list(video_paths.keys()),
        {cid: cid.replace("_", " ").title() for cid in video_paths.keys()}
    )
    
    # Ouvrir le reader synchronisé
    with SynchronizedReader(sync_results, target_fps) as reader:
        total_frames = reader.get_total_frames()
        
        if max_duration:
            total_frames = min(total_frames, int(max_duration * target_fps))
        
        # Démarrer l'enregistrement
        merger.start_recording(output_path)
        
        try:
            for frame_idx in range(total_frames):
                # Lire les frames synchronisées
                frames_dict = reader.read_synchronized()
                
                # Convertir au format attendu
                frames = {}
                for camera_id, frame in frames_dict.items():
                    if frame is not None:
                        frames[camera_id] = (frame, time.time())
                    else:
                        frames[camera_id] = None
                
                # Créer et écrire la grille
                grid = merger.create_grid_frame(frames)
                merger.write_frame(grid)
                
                # Progression
                if show_progress and frame_idx % 100 == 0:
                    progress = (frame_idx / total_frames) * 100
                    print(f"\r⏳ Progression: {progress:.1f}%", end="", flush=True)
        
        finally:
            merger.stop_recording()
    
    if show_progress:
        print(f"\r✓ Export terminé: {output_path}")
    
    return output_path


if __name__ == "__main__":
    import glob
    import os
    
    # Test avec les vidéos du dataset
    dataset_path = "/home/epl/DATA/S03/MTH2329/APEKE/PROJET/Dataset"
    videos = sorted(glob.glob(os.path.join(dataset_path, "*.mp4")))[:4]
    
    if videos:
        print(f"\n🎬 Test de fusion avec {len(videos)} vidéos...\n")
        
        video_paths = {
            f"cam_{i}": path
            for i, path in enumerate(videos)
        }
        
        output = os.path.join(dataset_path, "grid_output.mp4")
        
        merge_videos_to_grid(
            video_paths,
            output,
            target_fps=25.0,
            max_duration=10.0  # 10 secondes pour le test
        )
    else:
        print("Aucune vidéo trouvée pour le test")
