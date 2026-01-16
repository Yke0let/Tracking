#!/usr/bin/env python3
"""
Exemple de lecture et synchronisation de flux RTSP en parallèle.
Démontre l'utilisation avec des caméras réseau réelles.
"""

import cv2
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np


@dataclass
class RTSPCamera:
    """Configuration d'une caméra RTSP."""
    camera_id: str
    url: str
    name: str = ""
    username: str = ""
    password: str = ""


class MultiRTSPReader:
    """
    Lecteur multi-flux RTSP avec synchronisation.
    Gère les buffers, la reconnexion et l'uniformisation des framerates.
    """
    
    def __init__(
        self,
        cameras: list,
        target_fps: float = 25.0,
        buffer_size: int = 30,
        max_sync_drift_ms: float = 100.0
    ):
        self.cameras = {cam.camera_id: cam for cam in cameras}
        self.target_fps = target_fps
        self.buffer_size = buffer_size
        self.max_sync_drift_ms = max_sync_drift_ms
        
        # Buffers circulaires par caméra
        self.buffers: Dict[str, deque] = {
            cam.camera_id: deque(maxlen=buffer_size) 
            for cam in cameras
        }
        
        # Captures OpenCV
        self.captures: Dict[str, cv2.VideoCapture] = {}
        
        # État
        self.running = False
        self.threads: Dict[str, threading.Thread] = {}
        self.locks: Dict[str, threading.Lock] = {
            cam.camera_id: threading.Lock() 
            for cam in cameras
        }
        
        # Statistiques
        self.stats = {
            cam.camera_id: {
                "frames_received": 0,
                "frames_dropped": 0,
                "fps_actual": 0.0,
                "last_frame_time": None,
                "reconnections": 0,
            }
            for cam in cameras
        }
    
    def _build_url(self, camera: RTSPCamera) -> str:
        """Construit l'URL RTSP avec authentification."""
        url = camera.url
        if camera.username and camera.password:
            # Insérer les credentials dans l'URL
            if "://" in url:
                protocol, rest = url.split("://", 1)
                url = f"{protocol}://{camera.username}:{camera.password}@{rest}"
        return url
    
    def _connect(self, camera_id: str) -> bool:
        """Établit la connexion à une caméra."""
        camera = self.cameras[camera_id]
        url = self._build_url(camera)
        
        # Options pour réduire la latence
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        
        # Paramètres pour flux RTSP
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print(f"❌ Impossible de connecter: {camera_id}")
            return False
        
        self.captures[camera_id] = cap
        print(f"✓ Connecté: {camera_id}")
        return True
    
    def _capture_loop(self, camera_id: str):
        """Boucle de capture pour une caméra."""
        camera = self.cameras[camera_id]
        fps_counter = 0
        fps_start_time = time.time()
        
        while self.running:
            try:
                cap = self.captures.get(camera_id)
                
                if cap is None or not cap.isOpened():
                    # Tentative de reconnexion
                    print(f"🔄 Reconnexion: {camera_id}")
                    self.stats[camera_id]["reconnections"] += 1
                    
                    if self._connect(camera_id):
                        cap = self.captures.get(camera_id)
                    else:
                        time.sleep(2.0)
                        continue
                
                ret, frame = cap.read()
                current_time = time.time()
                
                if not ret or frame is None:
                    self.stats[camera_id]["frames_dropped"] += 1
                    continue
                
                # Ajouter au buffer
                with self.locks[camera_id]:
                    self.buffers[camera_id].append((frame, current_time))
                
                # Mettre à jour les stats
                self.stats[camera_id]["frames_received"] += 1
                self.stats[camera_id]["last_frame_time"] = current_time
                fps_counter += 1
                
                # Calculer le FPS réel
                if current_time - fps_start_time >= 1.0:
                    self.stats[camera_id]["fps_actual"] = fps_counter
                    fps_counter = 0
                    fps_start_time = current_time
                
            except Exception as e:
                print(f"Erreur {camera_id}: {e}")
                time.sleep(0.1)
    
    def start(self):
        """Démarre tous les flux."""
        self.running = True
        
        for camera_id in self.cameras:
            if self._connect(camera_id):
                thread = threading.Thread(
                    target=self._capture_loop,
                    args=(camera_id,),
                    daemon=True
                )
                thread.start()
                self.threads[camera_id] = thread
    
    def stop(self):
        """Arrête tous les flux."""
        self.running = False
        
        for thread in self.threads.values():
            thread.join(timeout=2.0)
        
        for cap in self.captures.values():
            cap.release()
        
        self.captures.clear()
        self.threads.clear()
    
    def get_latest_frames(self) -> Dict[str, Optional[Tuple[np.ndarray, float]]]:
        """Récupère les dernières frames de toutes les caméras."""
        frames = {}
        
        for camera_id in self.cameras:
            with self.locks[camera_id]:
                if self.buffers[camera_id]:
                    frames[camera_id] = self.buffers[camera_id][-1]
                else:
                    frames[camera_id] = None
        
        return frames
    
    def get_synchronized_frames(self) -> Optional[Dict[str, Tuple[np.ndarray, float]]]:
        """
        Récupère des frames synchronisées (dans la fenêtre de tolérance).
        Retourne None si la synchronisation échoue.
        """
        frames = self.get_latest_frames()
        
        # Filtrer les caméras sans frame
        valid_frames = {k: v for k, v in frames.items() if v is not None}
        
        if len(valid_frames) < len(self.cameras):
            return None
        
        # Vérifier la synchronisation temporelle
        timestamps = [v[1] for v in valid_frames.values()]
        spread_ms = (max(timestamps) - min(timestamps)) * 1000
        
        if spread_ms > self.max_sync_drift_ms:
            return None
        
        return valid_frames
    
    def create_grid(
        self, 
        frames: Dict[str, Optional[Tuple[np.ndarray, float]]],
        cell_size: Tuple[int, int] = (640, 480)
    ) -> np.ndarray:
        """Crée une grille à partir des frames."""
        n = len(self.cameras)
        cols = 2 if n > 1 else 1
        rows = (n + cols - 1) // cols
        
        cell_w, cell_h = cell_size
        grid = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
        
        for i, camera_id in enumerate(self.cameras):
            row, col = i // cols, i % cols
            x, y = col * cell_w, row * cell_h
            
            frame_data = frames.get(camera_id)
            
            if frame_data is not None:
                frame, ts = frame_data
                cell = cv2.resize(frame, cell_size)
                
                # Overlay avec nom et timestamp
                cv2.putText(
                    cell, camera_id,
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 255, 255), 2
                )
            else:
                cell = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
                cv2.putText(
                    cell, f"{camera_id}: No Signal",
                    (20, cell_h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (100, 100, 100), 2
                )
            
            grid[y:y+cell_h, x:x+cell_w] = cell
        
        return grid
    
    def print_stats(self):
        """Affiche les statistiques."""
        print("\n📊 Statistiques:")
        for camera_id, stats in self.stats.items():
            print(f"   {camera_id}:")
            print(f"      Frames: {stats['frames_received']}")
            print(f"      Drops: {stats['frames_dropped']}")
            print(f"      FPS: {stats['fps_actual']}")
            print(f"      Reconnexions: {stats['reconnections']}")
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()


# ============================================================
# EXEMPLES D'UTILISATION
# ============================================================

def example_with_files():
    """Exemple avec des fichiers vidéo locaux (pour test)."""
    print("\n" + "="*60)
    print("📺 Exemple: Lecture de fichiers vidéo en parallèle")
    print("="*60)
    
    import glob
    
    # Utiliser des fichiers locaux comme source
    videos = glob.glob("Dataset/*.mp4")[:4]
    
    if not videos:
        print("❌ Aucune vidéo trouvée dans Dataset/")
        return
    
    cameras = [
        RTSPCamera(
            camera_id=f"cam_{i}",
            url=path,  # Fichier local au lieu de RTSP
            name=f"Camera {i+1}"
        )
        for i, path in enumerate(videos)
    ]
    
    reader = MultiRTSPReader(cameras, target_fps=25.0)
    
    with reader:
        print("\n▶️ Lecture pendant 5 secondes...")
        print("   Appuyez sur 'Q' pour quitter\n")
        
        start_time = time.time()
        
        while time.time() - start_time < 5.0:
            frames = reader.get_latest_frames()
            grid = reader.create_grid(frames)
            
            cv2.imshow("Multi-Camera View", grid)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
        reader.print_stats()


def example_with_rtsp():
    """Exemple avec des flux RTSP réels."""
    print("\n" + "="*60)
    print("📺 Exemple: Lecture de flux RTSP en parallèle")
    print("="*60)
    
    # Configuration des caméras RTSP
    cameras = [
        RTSPCamera(
            camera_id="cam_entree",
            url="rtsp://192.168.1.100:554/stream1",
            name="Entrée principale",
            username="admin",
            password="password"
        ),
        RTSPCamera(
            camera_id="cam_parking",
            url="rtsp://192.168.1.101:554/stream1",
            name="Parking",
            username="admin",
            password="password"
        ),
        RTSPCamera(
            camera_id="cam_hall",
            url="rtsp://192.168.1.102:554/stream1",
            name="Hall",
            username="admin",
            password="password"
        ),
    ]
    
    reader = MultiRTSPReader(
        cameras,
        target_fps=25.0,
        buffer_size=30,
        max_sync_drift_ms=100.0
    )
    
    with reader:
        print("\n▶️ Flux en direct...")
        print("   Appuyez sur 'Q' pour quitter")
        print("   Appuyez sur 'S' pour screenshot\n")
        
        while True:
            # Essayer d'obtenir des frames synchronisées
            frames = reader.get_synchronized_frames()
            
            if frames is None:
                # Fallback sur les dernières frames disponibles
                frames = reader.get_latest_frames()
            
            grid = reader.create_grid(frames)
            
            # Indicateur de synchronisation
            sync_status = "SYNC" if frames else "ASYNC"
            cv2.putText(
                grid, sync_status,
                (grid.shape[1] - 80, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0) if sync_status == "SYNC" else (0, 165, 255), 2
            )
            
            cv2.imshow("Multi-RTSP View", grid)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(filename, grid)
                print(f"📷 Screenshot: {filename}")
        
        cv2.destroyAllWindows()
        reader.print_stats()


def example_gstreamer():
    """
    Exemple avec GStreamer pour une latence minimale.
    GStreamer offre de meilleures performances que FFmpeg pour le RTSP.
    """
    print("\n" + "="*60)
    print("📺 Exemple: RTSP avec GStreamer (faible latence)")
    print("="*60)
    
    # Pipeline GStreamer pour RTSP avec décodage hardware
    rtsp_url = "rtsp://192.168.1.100:554/stream1"
    
    gst_pipeline = (
        f"rtspsrc location={rtsp_url} latency=0 ! "
        "rtph264depay ! h264parse ! "
        "avdec_h264 ! videoconvert ! "
        "video/x-raw,format=BGR ! appsink drop=1"
    )
    
    print(f"\n🔧 Pipeline: {gst_pipeline}\n")
    
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("❌ Échec ouverture GStreamer. FFmpeg sera utilisé en fallback.")
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    
    if not cap.isOpened():
        print("❌ Impossible d'ouvrir le flux")
        return
    
    print("✓ Flux ouvert. Appuyez sur 'Q' pour quitter.\n")
    
    fps_counter = 0
    fps_start = time.time()
    fps_display = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("⚠️ Frame perdue, reconnexion...")
            time.sleep(0.1)
            continue
        
        fps_counter += 1
        if time.time() - fps_start >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_start = time.time()
        
        # Afficher FPS
        cv2.putText(
            frame, f"FPS: {fps_display}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 0), 2
        )
        
        cv2.imshow("GStreamer RTSP", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    
    print("\n" + "#"*60)
    print("#  EXEMPLES RTSP MULTI-CAMÉRAS")
    print("#"*60)
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "files":
            example_with_files()
        elif mode == "rtsp":
            example_with_rtsp()
        elif mode == "gstreamer":
            example_gstreamer()
        else:
            print(f"Mode inconnu: {mode}")
            print("Modes disponibles: files, rtsp, gstreamer")
    else:
        # Mode par défaut: fichiers locaux
        example_with_files()
