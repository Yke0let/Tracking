"""
Module d'extraction de métadonnées vidéo.
Extraction de résolution, FPS, durée, codec, timestamps via OpenCV et subprocess (ffprobe).
"""

import cv2
import json
import subprocess
import os
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    """Structure de métadonnées pour une vidéo."""
    filepath: str
    filename: str
    camera_id: str
    location: str
    
    # Propriétés vidéo
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    codec: str
    
    # Timestamps
    creation_time: Optional[str] = None
    modification_time: Optional[str] = None
    
    # Audio
    has_audio: bool = False
    audio_codec: Optional[str] = None
    audio_sample_rate: Optional[int] = None
    
    # Métadonnées additionnelles
    file_size_bytes: int = 0
    bitrate_kbps: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convertit en JSON."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VideoMetadata':
        """Crée une instance depuis un dictionnaire."""
        return cls(**data)


def extract_camera_info_from_filename(filename: str) -> tuple:
    """
    Extrait l'ID caméra et l'emplacement depuis le nom de fichier.
    Ex: CAMERA_DEVANTURE_PORTE_ENTREE.mp4 -> (cam_devanture_porte_entree, Devanture - Porte Entrée)
    """
    base = Path(filename).stem.upper()
    
    # Mapping des emplacements connus
    location_mapping = {
        "DEVANTURE_PORTE_ENTREE": "Devanture - Porte Entrée",
        "DEVANTURE_SOUS_ARBRE": "Devanture - Sous Arbre",
        "ESCALIER_DEBUT_COULOIR_GAUCHE": "Escalier - Début Couloir Gauche",
        "FIN_COULOIR_DROIT": "Fin Couloir Droit",
        "FIN_COULOIR_GAUCHE_ETAGE1": "Fin Couloir Gauche - Étage 1",
        "FIN_COULOIR_GAUCHE_REZ": "Fin Couloir Gauche - Rez-de-chaussée",
        "HALL_PORTE_DROITE": "Hall - Porte Droite",
        "HALL_PORTE_ENTREE": "Hall - Porte Entrée",
        "HALL_PORTE_GAUCHE": "Hall - Porte Gauche",
    }
    
    # Nettoyer le nom
    camera_part = base.replace("CAMERA_", "").replace("(1)", "").replace("_PARTIE_1", "_P1").replace("_PARTIE_2", "_P2")
    camera_id = f"cam_{camera_part.lower()}"
    
    # Trouver l'emplacement correspondant
    location = "Emplacement inconnu"
    for key, value in location_mapping.items():
        if key in base:
            location = value
            break
    
    return camera_id, location


def extract_metadata_opencv(filepath: str) -> Dict[str, Any]:
    """
    Extrait les métadonnées basiques via OpenCV.
    """
    cap = cv2.VideoCapture(filepath)
    
    if not cap.isOpened():
        raise ValueError(f"Impossible d'ouvrir la vidéo: {filepath}")
    
    metadata = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
    }
    
    # Convertir le codec en string
    fourcc = metadata["codec"]
    metadata["codec"] = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    
    # Calculer la durée
    if metadata["fps"] > 0:
        metadata["duration_seconds"] = metadata["frame_count"] / metadata["fps"]
    else:
        metadata["duration_seconds"] = 0.0
    
    cap.release()
    return metadata


def extract_metadata_ffprobe(filepath: str) -> Dict[str, Any]:
    """
    Extrait les métadonnées avancées via ffprobe (si disponible).
    Inclut les informations audio et les timestamps de création.
    """
    try:
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            filepath
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            logger.warning(f"ffprobe a échoué pour {filepath}")
            return {}
        
        data = json.loads(result.stdout)
        metadata = {}
        
        # Informations du format
        if "format" in data:
            fmt = data["format"]
            metadata["bitrate_kbps"] = float(fmt.get("bit_rate", 0)) / 1000
            
            # Timestamps de création
            if "tags" in fmt:
                metadata["creation_time"] = fmt["tags"].get("creation_time")
        
        # Informations des streams
        for stream in data.get("streams", []):
            if stream["codec_type"] == "video":
                # FPS plus précis depuis ffprobe
                if "r_frame_rate" in stream:
                    num, den = map(int, stream["r_frame_rate"].split("/"))
                    if den > 0:
                        metadata["fps_precise"] = num / den
            
            elif stream["codec_type"] == "audio":
                metadata["has_audio"] = True
                metadata["audio_codec"] = stream.get("codec_name")
                metadata["audio_sample_rate"] = int(stream.get("sample_rate", 0))
        
        return metadata
        
    except FileNotFoundError:
        logger.warning("ffprobe non installé. Utilisez 'sudo apt install ffmpeg'")
        return {}
    except subprocess.TimeoutExpired:
        logger.warning(f"Timeout ffprobe pour {filepath}")
        return {}
    except Exception as e:
        logger.warning(f"Erreur ffprobe: {e}")
        return {}


def extract_video_metadata(filepath: str) -> VideoMetadata:
    """
    Extrait toutes les métadonnées d'une vidéo.
    Combine OpenCV et ffprobe pour des informations complètes.
    """
    filepath = str(Path(filepath).resolve())
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Fichier non trouvé: {filepath}")
    
    filename = os.path.basename(filepath)
    camera_id, location = extract_camera_info_from_filename(filename)
    
    # Métadonnées OpenCV (base)
    opencv_meta = extract_metadata_opencv(filepath)
    
    # Métadonnées ffprobe (avancées)
    ffprobe_meta = extract_metadata_ffprobe(filepath)
    
    # Informations fichier
    file_stat = os.stat(filepath)
    modification_time = datetime.fromtimestamp(file_stat.st_mtime).isoformat()
    
    # Fusionner les métadonnées
    fps = ffprobe_meta.get("fps_precise", opencv_meta["fps"])
    
    return VideoMetadata(
        filepath=filepath,
        filename=filename,
        camera_id=camera_id,
        location=location,
        width=opencv_meta["width"],
        height=opencv_meta["height"],
        fps=fps,
        frame_count=opencv_meta["frame_count"],
        duration_seconds=opencv_meta["duration_seconds"],
        codec=opencv_meta["codec"],
        creation_time=ffprobe_meta.get("creation_time"),
        modification_time=modification_time,
        has_audio=ffprobe_meta.get("has_audio", False),
        audio_codec=ffprobe_meta.get("audio_codec"),
        audio_sample_rate=ffprobe_meta.get("audio_sample_rate"),
        file_size_bytes=file_stat.st_size,
        bitrate_kbps=ffprobe_meta.get("bitrate_kbps"),
    )


def extract_all_metadata(video_paths: List[str]) -> List[VideoMetadata]:
    """
    Extrait les métadonnées de plusieurs vidéos.
    
    Args:
        video_paths: Liste des chemins vers les fichiers vidéo
        
    Returns:
        Liste des métadonnées pour chaque vidéo
    """
    metadata_list = []
    
    for path in video_paths:
        try:
            meta = extract_video_metadata(path)
            metadata_list.append(meta)
            logger.info(f"✓ {meta.filename}: {meta.width}x{meta.height} @ {meta.fps:.2f} FPS, {meta.duration_seconds:.1f}s")
        except Exception as e:
            logger.error(f"✗ Erreur pour {path}: {e}")
    
    return metadata_list


def save_metadata_report(metadata_list: List[VideoMetadata], output_path: str):
    """
    Sauvegarde un rapport JSON des métadonnées.
    """
    report = {
        "generated_at": datetime.now().isoformat(),
        "video_count": len(metadata_list),
        "videos": [m.to_dict() for m in metadata_list],
        "summary": {
            "total_duration_seconds": sum(m.duration_seconds for m in metadata_list),
            "total_frames": sum(m.frame_count for m in metadata_list),
            "resolutions": list(set(f"{m.width}x{m.height}" for m in metadata_list)),
            "framerates": list(set(f"{m.fps:.2f}" for m in metadata_list)),
        }
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Rapport sauvegardé: {output_path}")


if __name__ == "__main__":
    import glob
    
    # Exemple d'utilisation
    dataset_path = "/home/epl/DATA/S03/MTH2329/APEKE/PROJET/Dataset"
    videos = glob.glob(os.path.join(dataset_path, "*.mp4")) + glob.glob(os.path.join(dataset_path, "*.MP4"))
    
    print(f"\n📹 Analyse de {len(videos)} vidéos...\n")
    
    metadata_list = extract_all_metadata(videos)
    
    print(f"\n📊 Résumé:")
    print(f"   - Vidéos analysées: {len(metadata_list)}")
    print(f"   - Durée totale: {sum(m.duration_seconds for m in metadata_list) / 60:.1f} minutes")
    print(f"   - Résolutions: {set(f'{m.width}x{m.height}' for m in metadata_list)}")
    print(f"   - FPS: {set(f'{m.fps:.2f}' for m in metadata_list)}")
    
    # Sauvegarder le rapport
    save_metadata_report(metadata_list, os.path.join(dataset_path, "metadata_report.json"))
