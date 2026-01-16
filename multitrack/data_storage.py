"""
Module de stockage et organisation des données vidéo.
Gère la structure de dossiers, la base de métadonnées SQLite,
et le logging des événements.
"""

import os
import json
import sqlite3
import shutil
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VideoRecord:
    """Enregistrement vidéo dans la base de données."""
    id: Optional[int]
    camera_id: str
    filename: str
    filepath: str
    
    # Timestamps
    start_time: str
    end_time: Optional[str]
    duration_seconds: float
    
    # Propriétés vidéo
    width: int
    height: int
    fps: float
    frame_count: int
    codec: str
    file_size_bytes: int
    
    # Métadonnées
    location: str
    has_audio: bool
    bitrate_kbps: Optional[float]
    
    # État
    is_synchronized: bool = False
    sync_offset_seconds: Optional[float] = None
    
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class EventLog:
    """Log d'événement système."""
    id: Optional[int]
    timestamp: str
    camera_id: Optional[str]
    event_type: str        # stream_start, stream_stop, desync, error, reconnect
    message: str
    severity: str          # info, warning, error, critical
    metadata: Optional[str]  # JSON


class DataStorage:
    """
    Système de stockage pour les données de surveillance.
    Gère les fichiers vidéo et la base de métadonnées.
    """
    
    def __init__(
        self,
        base_path: str = "output",
        db_name: str = "metadata.db"
    ):
        """
        Initialise le système de stockage.
        
        Args:
            base_path: Chemin racine pour le stockage
            db_name: Nom du fichier de base de données
        """
        self.base_path = Path(base_path).resolve()
        self.db_path = self.base_path / db_name
        
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        
        # Créer la structure
        self._setup_directories()
        self._setup_database()
        
    def _setup_directories(self):
        """Crée la structure de dossiers."""
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Dossiers standards
        (self.base_path / "clips").mkdir(exist_ok=True)
        (self.base_path / "exports").mkdir(exist_ok=True)
        (self.base_path / "logs").mkdir(exist_ok=True)
        
        logger.info(f"Structure de dossiers créée: {self.base_path}")
    
    def _setup_database(self):
        """Initialise la base de données SQLite."""
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        
        cursor = self._conn.cursor()
        
        # Table des enregistrements vidéo
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL UNIQUE,
                
                start_time TEXT,
                end_time TEXT,
                duration_seconds REAL,
                
                width INTEGER,
                height INTEGER,
                fps REAL,
                frame_count INTEGER,
                codec TEXT,
                file_size_bytes INTEGER,
                
                location TEXT,
                has_audio INTEGER,
                bitrate_kbps REAL,
                
                is_synchronized INTEGER DEFAULT 0,
                sync_offset_seconds REAL,
                
                created_at TEXT NOT NULL
            )
        """)
        
        # Table des événements
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                camera_id TEXT,
                event_type TEXT NOT NULL,
                message TEXT,
                severity TEXT,
                metadata TEXT
            )
        """)
        
        # Table des caméras
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cameras (
                camera_id TEXT PRIMARY KEY,
                name TEXT,
                location TEXT,
                source TEXT,
                is_active INTEGER DEFAULT 1,
                last_seen TEXT,
                config TEXT
            )
        """)
        
        # Index pour les requêtes fréquentes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_videos_camera ON videos(camera_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_videos_time ON videos(start_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_time ON events(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_camera ON events(camera_id)")
        
        self._conn.commit()
        logger.info(f"Base de données initialisée: {self.db_path}")
    
    def get_camera_dir(self, camera_id: str, date: Optional[datetime] = None) -> Path:
        """
        Retourne le chemin du dossier pour une caméra.
        
        Structure: base_path/camera_id/YYYY-MM-DD/
        """
        if date is None:
            date = datetime.now()
        
        date_str = date.strftime("%Y-%m-%d")
        camera_dir = self.base_path / camera_id / date_str
        camera_dir.mkdir(parents=True, exist_ok=True)
        
        return camera_dir
    
    def generate_filename(
        self,
        camera_id: str,
        timestamp: Optional[datetime] = None,
        extension: str = "mp4"
    ) -> str:
        """Génère un nom de fichier unique."""
        if timestamp is None:
            timestamp = datetime.now()
        
        return f"{camera_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.{extension}"
    
    def add_video_record(self, record: VideoRecord) -> int:
        """
        Ajoute un enregistrement vidéo à la base.
        
        Returns:
            ID de l'enregistrement créé
        """
        with self._lock:
            cursor = self._conn.cursor()
            
            cursor.execute("""
                INSERT INTO videos (
                    camera_id, filename, filepath,
                    start_time, end_time, duration_seconds,
                    width, height, fps, frame_count, codec, file_size_bytes,
                    location, has_audio, bitrate_kbps,
                    is_synchronized, sync_offset_seconds,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.camera_id, record.filename, record.filepath,
                record.start_time, record.end_time, record.duration_seconds,
                record.width, record.height, record.fps, record.frame_count,
                record.codec, record.file_size_bytes,
                record.location, int(record.has_audio), record.bitrate_kbps,
                int(record.is_synchronized), record.sync_offset_seconds,
                record.created_at
            ))
            
            self._conn.commit()
            return cursor.lastrowid
    
    def get_videos_by_camera(
        self,
        camera_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Récupère les vidéos d'une caméra."""
        with self._lock:
            cursor = self._conn.cursor()
            
            query = "SELECT * FROM videos WHERE camera_id = ?"
            params = [camera_id]
            
            if start_date:
                query += " AND start_time >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND start_time <= ?"
                params.append(end_date)
            
            query += " ORDER BY start_time DESC"
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_videos_in_timerange(
        self,
        start_time: str,
        end_time: str
    ) -> List[Dict[str, Any]]:
        """Récupère toutes les vidéos dans une plage horaire."""
        with self._lock:
            cursor = self._conn.cursor()
            
            cursor.execute("""
                SELECT * FROM videos
                WHERE start_time <= ? AND (end_time >= ? OR end_time IS NULL)
                ORDER BY camera_id, start_time
            """, (end_time, start_time))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def log_event(
        self,
        event_type: str,
        message: str,
        camera_id: Optional[str] = None,
        severity: str = "info",
        metadata: Optional[Dict] = None
    ):
        """Enregistre un événement système."""
        with self._lock:
            cursor = self._conn.cursor()
            
            cursor.execute("""
                INSERT INTO events (timestamp, camera_id, event_type, message, severity, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                camera_id,
                event_type,
                message,
                severity,
                json.dumps(metadata) if metadata else None
            ))
            
            self._conn.commit()
        
        # Logger aussi dans le fichier
        log_func = getattr(logger, severity, logger.info)
        log_func(f"[{event_type}] {camera_id or 'SYSTEM'}: {message}")
    
    def get_events(
        self,
        camera_id: Optional[str] = None,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Récupère les événements filtrés."""
        with self._lock:
            cursor = self._conn.cursor()
            
            query = "SELECT * FROM events WHERE 1=1"
            params = []
            
            if camera_id:
                query += " AND camera_id = ?"
                params.append(camera_id)
            
            if event_type:
                query += " AND event_type = ?"
                params.append(event_type)
            
            if severity:
                query += " AND severity = ?"
                params.append(severity)
            
            query += f" ORDER BY timestamp DESC LIMIT {limit}"
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def register_camera(
        self,
        camera_id: str,
        name: str,
        location: str,
        source: str,
        config: Optional[Dict] = None
    ):
        """Enregistre ou met à jour une caméra."""
        with self._lock:
            cursor = self._conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO cameras
                (camera_id, name, location, source, is_active, last_seen, config)
                VALUES (?, ?, ?, ?, 1, ?, ?)
            """, (
                camera_id, name, location, source,
                datetime.now().isoformat(),
                json.dumps(config) if config else None
            ))
            
            self._conn.commit()
    
    def update_camera_last_seen(self, camera_id: str):
        """Met à jour le timestamp de dernière activité."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                "UPDATE cameras SET last_seen = ? WHERE camera_id = ?",
                (datetime.now().isoformat(), camera_id)
            )
            self._conn.commit()
    
    def get_cameras(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """Liste les caméras enregistrées."""
        with self._lock:
            cursor = self._conn.cursor()
            
            query = "SELECT * FROM cameras"
            if active_only:
                query += " WHERE is_active = 1"
            
            cursor.execute(query)
            return [dict(row) for row in cursor.fetchall()]
    
    def export_metadata_json(self, output_path: Optional[str] = None) -> str:
        """Exporte toutes les métadonnées en JSON."""
        if output_path is None:
            output_path = str(self.base_path / "exports" / f"metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        with self._lock:
            cursor = self._conn.cursor()
            
            data = {
                "exported_at": datetime.now().isoformat(),
                "cameras": [dict(row) for row in cursor.execute("SELECT * FROM cameras").fetchall()],
                "videos": [dict(row) for row in cursor.execute("SELECT * FROM videos").fetchall()],
                "recent_events": [dict(row) for row in cursor.execute(
                    "SELECT * FROM events ORDER BY timestamp DESC LIMIT 1000"
                ).fetchall()],
            }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Métadonnées exportées: {output_path}")
        return output_path
    
    def get_storage_statistics(self) -> Dict[str, Any]:
        """Calcule les statistiques de stockage."""
        with self._lock:
            cursor = self._conn.cursor()
            
            # Stats des vidéos
            cursor.execute("""
                SELECT 
                    COUNT(*) as video_count,
                    SUM(file_size_bytes) as total_size,
                    SUM(duration_seconds) as total_duration,
                    SUM(frame_count) as total_frames,
                    COUNT(DISTINCT camera_id) as camera_count
                FROM videos
            """)
            video_stats = dict(cursor.fetchone())
            
            # Stats par caméra
            cursor.execute("""
                SELECT camera_id, COUNT(*) as count, SUM(duration_seconds) as duration
                FROM videos
                GROUP BY camera_id
            """)
            per_camera = [dict(row) for row in cursor.fetchall()]
            
            return {
                "total_videos": video_stats["video_count"],
                "total_size_gb": (video_stats["total_size"] or 0) / (1024**3),
                "total_duration_hours": (video_stats["total_duration"] or 0) / 3600,
                "total_frames": video_stats["total_frames"],
                "camera_count": video_stats["camera_count"],
                "per_camera": per_camera,
            }
    
    def cleanup_old_files(self, days_to_keep: int = 30) -> int:
        """
        Supprime les fichiers plus anciens que le nombre de jours spécifié.
        
        Returns:
            Nombre de fichiers supprimés
        """
        from datetime import timedelta
        
        cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
        deleted_count = 0
        
        with self._lock:
            cursor = self._conn.cursor()
            
            # Trouver les anciens fichiers
            cursor.execute(
                "SELECT id, filepath FROM videos WHERE start_time < ?",
                (cutoff_date,)
            )
            
            for row in cursor.fetchall():
                filepath = row["filepath"]
                
                if os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                        deleted_count += 1
                    except Exception as e:
                        logger.error(f"Erreur suppression {filepath}: {e}")
                
                # Supprimer de la base
                cursor.execute("DELETE FROM videos WHERE id = ?", (row["id"],))
            
            self._conn.commit()
        
        logger.info(f"Nettoyage: {deleted_count} fichiers supprimés")
        return deleted_count
    
    def close(self):
        """Ferme la connexion à la base de données."""
        if self._conn:
            self._conn.close()
            self._conn = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    # Test du module
    storage = DataStorage(base_path="/tmp/multicam_test")
    
    # Enregistrer des caméras
    storage.register_camera(
        camera_id="cam_devanture",
        name="Caméra Devanture",
        location="Devanture - Porte Entrée",
        source="/path/to/video.mp4"
    )
    
    # Logger des événements
    storage.log_event("stream_start", "Flux démarré", camera_id="cam_devanture")
    storage.log_event("desync", "Désynchronisation détectée: 150ms", camera_id="cam_devanture", severity="warning")
    
    # Afficher les stats
    stats = storage.get_storage_statistics()
    print(f"\n📊 Statistiques:")
    print(f"   Vidéos: {stats['total_videos']}")
    print(f"   Taille: {stats['total_size_gb']:.2f} GB")
    
    # Exporter
    storage.export_metadata_json()
    
    storage.close()
