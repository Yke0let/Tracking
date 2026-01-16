#!/usr/bin/env python3
"""
Démonstration MCMOT: Multi-Camera Multi-Object Tracking avec Ré-identification.

Ce script démontre l'utilisation du système MCMOT avancé avec:
- 2-4 caméras synchronisées
- IDs globaux cohérents cross-camera
- Galerie adaptative pour apprentissage
- Visualisation BEV temps réel

Usage:
    python -m multicam.mcmot_demo --videos video1.mp4 video2.mp4
    python -m multicam.mcmot_demo --videos video1.mp4 video2.mp4 --enable-reid
    python -m multicam.mcmot_demo --videos *.mp4 --calibrate
"""

import argparse
import cv2
import numpy as np
import time
import glob
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import des modules MCMOT
try:
    from .mcmot_advanced import (
        MCMOTAdvancedTracker,
        AdvancedBEVVisualizer,
        AdvancedCameraCalibration,
        AdaptiveGallery,
    )
    MCMOT_AVAILABLE = True
except ImportError:
    try:
        from mcmot_advanced import (
            MCMOTAdvancedTracker,
            AdvancedBEVVisualizer,
            AdvancedCameraCalibration,
            AdaptiveGallery,
        )
        MCMOT_AVAILABLE = True
    except ImportError:
        MCMOT_AVAILABLE = False
        logger.warning("Module mcmot_advanced non disponible")


class CalibrationHelper:
    """Assistant pour la calibration interactive des caméras."""
    
    def __init__(self, camera_id: str, window_name: str):
        self.camera_id = camera_id
        self.window_name = window_name
        self.image_points: List[Tuple[int, int]] = []
        self.ground_points: List[Tuple[float, float]] = []
        self.current_frame: Optional[np.ndarray] = None
        
    def mouse_callback(self, event, x, y, flags, param):
        """Callback pour les clics souris."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.image_points.append((x, y))
            # Demander les coordonnées sol
            print(f"Point image {len(self.image_points)}: ({x}, {y})")
            print("Entrez les coordonnées sol (X Y en mètres): ", end="")
    
    def calibrate_interactive(self, frame: np.ndarray) -> Optional[AdvancedCameraCalibration]:
        """Calibration interactive avec interface graphique."""
        self.current_frame = frame.copy()
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print(f"\n=== Calibration caméra {self.camera_id} ===")
        print("Cliquez sur au moins 4 points correspondant au sol")
        print("Après chaque clic, entrez les coordonnées sol (X Y)")
        print("Appuyez sur 'c' pour calculer, 'q' pour annuler\n")
        
        while True:
            display = self.current_frame.copy()
            
            # Dessiner les points déjà cliqués
            for i, pt in enumerate(self.image_points):
                cv2.circle(display, pt, 5, (0, 255, 0), -1)
                cv2.putText(display, str(i+1), (pt[0]+5, pt[1]-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Instructions
            text = f"Points: {len(self.image_points)}/4+ | C=Calculer | Q=Annuler"
            cv2.putText(display, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(100) & 0xFF
            
            if key == ord('q'):
                cv2.destroyWindow(self.window_name)
                return None
            elif key == ord('c') and len(self.image_points) >= 4:
                break
        
        cv2.destroyWindow(self.window_name)
        
        # Créer la calibration
        if len(self.image_points) == len(self.ground_points) >= 4:
            calib = AdvancedCameraCalibration(self.camera_id)
            calib.set_homography_from_points(self.image_points, self.ground_points)
            return calib
        
        return None


class MCMOTDemo:
    """Démonstration complète du système MCMOT."""
    
    def __init__(
        self,
        video_paths: List[str],
        enable_reid: bool = True,
        reid_threshold: float = 0.6,
        target_fps: float = 25.0,
        grid_cell_size: Tuple[int, int] = (480, 360),
        show_bev: bool = True,
    ):
        self.video_paths = video_paths
        self.target_fps = target_fps
        self.grid_cell_size = grid_cell_size
        self.show_bev = show_bev
        
        # Initialiser le tracker MCMOT
        self.tracker = MCMOTAdvancedTracker(
            enable_reid=enable_reid,
            reid_threshold=reid_threshold,
        )
        
        # Captures vidéo
        self.captures: Dict[str, cv2.VideoCapture] = {}
        self.camera_ids: List[str] = []
        
        # BEV visualizer
        self.bev_visualizer = AdvancedBEVVisualizer(
            width=400,
            height=300,
        ) if show_bev else None
        
        # État
        self._running = False
        self._paused = False
        self._timestamp = 0.0
        
        logger.info(f"✓ MCMOT Demo initialisée avec {len(video_paths)} vidéos")
        
    def setup(self):
        """Configure les captures vidéo."""
        for i, path in enumerate(self.video_paths):
            camera_id = f"cam{i+1}"
            cap = cv2.VideoCapture(path)
            
            if not cap.isOpened():
                logger.error(f"Impossible d'ouvrir: {path}")
                continue
            
            self.captures[camera_id] = cap
            self.camera_ids.append(camera_id)
            
            # Métadonnées
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frames / fps if fps > 0 else 0
            
            logger.info(f"  {camera_id}: {Path(path).name} ({duration:.1f}s, {fps:.1f}fps)")
        
        if not self.captures:
            raise ValueError("Aucune vidéo chargée")
    
    def add_default_calibrations(self):
        """Ajoute des calibrations par défaut (identité)."""
        for camera_id in self.camera_ids:
            # Calibration identité (pas de transformation)
            calib = AdvancedCameraCalibration(camera_id)
            # Points exemple (à personnaliser)
            image_points = [(0, 480), (640, 480), (640, 0), (0, 0)]
            ground_points = [(0, 10), (10, 10), (10, 0), (0, 0)]
            calib.set_homography_from_points(image_points, ground_points)
            self.tracker.calibrations[camera_id] = calib
    
    def create_grid(
        self,
        frames: Dict[str, np.ndarray],
        tracks: Dict[str, list],
    ) -> np.ndarray:
        """Crée la grille d'affichage."""
        n = len(self.camera_ids)
        cols = min(4, n) if n > 1 else 1
        rows = (n + cols - 1) // cols
        
        cell_w, cell_h = self.grid_cell_size
        grid = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
        grid[:] = (30, 30, 30)
        
        for i, camera_id in enumerate(self.camera_ids):
            row, col = i // cols, i % cols
            x, y = col * cell_w, row * cell_h
            
            frame = frames.get(camera_id)
            if frame is not None:
                # Dessiner les tracks
                local_tracks = tracks.get(camera_id, [])
                annotated = self.tracker.draw_with_global_ids(
                    camera_id, frame, local_tracks
                )
                
                # Redimensionner
                cell = cv2.resize(annotated, (cell_w, cell_h))
                
                # Overlay nom
                cv2.rectangle(cell, (0, 0), (cell_w, 28), (0, 0, 0), -1)
                cv2.putText(cell, f"{camera_id}", (5, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Nombre d'objets
                n_tracks = len(local_tracks)
                cv2.putText(cell, f"{n_tracks} objets", (cell_w - 80, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            else:
                cell = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
                cell[:] = (40, 40, 40)
                cv2.putText(cell, camera_id, (10, cell_h // 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
            
            grid[y:y+cell_h, x:x+cell_w] = cell
        
        return grid
    
    def add_info_panel(self, grid: np.ndarray) -> np.ndarray:
        """Ajoute un panneau d'informations."""
        h, w = grid.shape[:2]
        panel_height = 80
        
        output = np.zeros((h + panel_height, w, 3), dtype=np.uint8)
        output[:h] = grid
        output[h:] = (40, 40, 40)
        
        # Statistiques
        stats = self.tracker.get_statistics()
        
        y = h + 20
        cv2.putText(output, f"Tracks Globaux: {stats['total_global_tracks']}",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(output, f"Confirmés: {stats['confirmed_tracks']}",
                   (180, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.putText(output, f"Cross-Camera: {stats['cross_camera_tracks']}",
                   (330, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
        
        # Galerie
        gallery_stats = stats.get('gallery', {})
        y += 25
        cv2.putText(output, f"Galerie: {gallery_stats.get('total_objects', 0)} objets, "
                   f"{gallery_stats.get('total_features', 0)} features",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Timestamp
        y += 25
        cv2.putText(output, f"Time: {self._timestamp:.2f}s",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Contrôles
        cv2.putText(output, "Q=Quit | SPACE=Pause | S=Screenshot",
                   (w - 280, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        return output
    
    def run(self, record_output: Optional[str] = None):
        """Lance la démonstration."""
        self.setup()
        self.add_default_calibrations()
        
        self._running = True
        frame_interval = 1.0 / self.target_fps
        
        window_name = "MCMOT Demo - Multi-Camera Tracking"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Writer pour enregistrement
        writer = None
        if record_output:
            n = len(self.camera_ids)
            cols = min(4, n) if n > 1 else 1
            rows = (n + cols - 1) // cols
            cell_w, cell_h = self.grid_cell_size
            
            if self.show_bev:
                out_w = cols * cell_w + 400  # +BEV
            else:
                out_w = cols * cell_w
            out_h = rows * cell_h + 80  # +panel
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(record_output, fourcc, self.target_fps, (out_w, out_h))
            logger.info(f"🔴 Enregistrement: {record_output}")
        
        logger.info(f"\n▶️ Démonstration démarrée")
        logger.info("   Contrôles: Q=Quitter, ESPACE=Pause, S=Screenshot\n")
        
        try:
            while self._running:
                loop_start = time.time()
                
                if not self._paused:
                    self._timestamp += frame_interval
                
                # Lire les frames
                frames: Dict[str, np.ndarray] = {}
                all_done = True
                
                for camera_id, cap in self.captures.items():
                    ret, frame = cap.read()
                    if ret:
                        frames[camera_id] = frame
                        all_done = False
                    else:
                        # Reboucler
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                        if ret:
                            frames[camera_id] = frame
                            all_done = False
                
                if all_done and not self._paused:
                    logger.info("✓ Fin des vidéos")
                    break
                
                # Tracker chaque frame
                all_tracks: Dict[str, list] = {}
                
                if not self._paused:
                    for camera_id, frame in frames.items():
                        tracks = self.tracker.process_frame(
                            camera_id, frame, self._timestamp
                        )
                        all_tracks[camera_id] = tracks
                    
                    # Association cross-camera
                    self.tracker.associate_cross_camera(self._timestamp)
                
                # Créer l'affichage
                grid = self.create_grid(frames, all_tracks)
                output = self.add_info_panel(grid)
                
                # Ajouter BEV si activé
                if self.show_bev and self.bev_visualizer:
                    bev = self.bev_visualizer.draw_bev(self.tracker.global_tracks)
                    bev = cv2.resize(bev, (400, output.shape[0]))
                    output = np.hstack([output, bev])
                
                # Enregistrer
                if writer and not self._paused:
                    writer.write(output)
                
                # Afficher
                cv2.imshow(window_name, output)
                
                # Timing
                elapsed = time.time() - loop_start
                wait_time = max(1, int((frame_interval - elapsed) * 1000))
                
                key = cv2.waitKey(wait_time) & 0xFF
                
                if key == ord('q') or key == 27:
                    break
                elif key == ord(' '):
                    self._paused = not self._paused
                    logger.info("⏸️ Pause" if self._paused else "▶️ Reprise")
                elif key == ord('s'):
                    filename = f"mcmot_{datetime.now().strftime('%H%M%S')}.png"
                    cv2.imwrite(filename, output)
                    logger.info(f"📷 Screenshot: {filename}")
                elif key == ord('i'):
                    # Afficher stats
                    stats = self.tracker.get_statistics()
                    print(f"\n=== Statistiques ===")
                    for k, v in stats.items():
                        print(f"  {k}: {v}")
                    print()
        
        finally:
            self._running = False
            cv2.destroyAllWindows()
            
            if writer:
                writer.release()
                logger.info(f"✓ Enregistrement terminé: {record_output}")
            
            for cap in self.captures.values():
                cap.release()
            
            # Stats finales
            print("\n=== Statistiques Finales ===")
            stats = self.tracker.get_statistics()
            for k, v in stats.items():
                print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser(
        description="Démonstration MCMOT Multi-Camera Tracking"
    )
    parser.add_argument(
        "--videos", "-v",
        nargs="+",
        required=True,
        help="Chemins vers les fichiers vidéo (2-4 caméras)"
    )
    parser.add_argument(
        "--enable-reid",
        action="store_true",
        default=False,
        help="Activer l'extraction ReID (plus lent mais plus précis)"
    )
    parser.add_argument(
        "--reid-threshold",
        type=float,
        default=0.6,
        help="Seuil de similarité ReID (défaut: 0.6)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="FPS cible (défaut: 25)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Fichier de sortie pour enregistrement"
    )
    parser.add_argument(
        "--no-bev",
        action="store_true",
        default=False,
        help="Désactiver la vue Bird's Eye View"
    )
    
    args = parser.parse_args()
    
    # Résoudre les globs
    video_paths = []
    for pattern in args.videos:
        matched = glob.glob(pattern)
        if matched:
            video_paths.extend(matched)
        elif Path(pattern).exists():
            video_paths.append(pattern)
        else:
            logger.warning(f"Fichier non trouvé: {pattern}")
    
    if len(video_paths) < 2:
        logger.error("Au moins 2 vidéos requises pour MCMOT")
        return 1
    
    if len(video_paths) > 4:
        logger.warning(f"Limitation à 4 vidéos (reçu: {len(video_paths)})")
        video_paths = video_paths[:4]
    
    # Lancer la démo
    demo = MCMOTDemo(
        video_paths=video_paths,
        enable_reid=args.enable_reid,
        reid_threshold=args.reid_threshold,
        target_fps=args.fps,
        show_bev=not args.no_bev,
    )
    
    demo.run(record_output=args.output)
    return 0


if __name__ == "__main__":
    exit(main())
