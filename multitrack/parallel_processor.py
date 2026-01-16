"""
Module de traitement parallèle pour le prétraitement de vidéos.
Utilise ProcessPoolExecutor pour paralléliser les traitements CPU-intensifs.
"""

import os
import time
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable, Any
from queue import Queue
from threading import Thread
import multiprocessing

from .preprocessing import PreprocessingConfig, PreprocessingPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingTask:
    """Représente une tâche de traitement."""
    video_path: str
    output_path: Optional[str] = None
    priority: int = 0
    
    def __lt__(self, other):
        return self.priority < other.priority


@dataclass 
class ProcessingResult:
    """Résultat d'une tâche de traitement."""
    video_path: str
    output_path: Optional[str]
    status: str  # success, error, skipped
    duration_seconds: float
    message: Optional[str] = None
    metadata: Optional[Dict] = None


def _process_single_video(args: tuple) -> ProcessingResult:
    """
    Fonction worker pour traiter une seule vidéo.
    Doit être au niveau module pour être picklable.
    """
    video_path, config_dict, output_path = args
    
    start_time = time.time()
    
    try:
        # Recréer la config depuis le dict
        config = PreprocessingConfig(**config_dict)
        pipeline = PreprocessingPipeline(config)
        
        result = pipeline.process_video(video_path, output_path)
        
        duration = time.time() - start_time
        
        return ProcessingResult(
            video_path=video_path,
            output_path=result.get("output"),
            status=result.get("status", "success"),
            duration_seconds=duration,
            metadata=result.get("metadata")
        )
        
    except Exception as e:
        duration = time.time() - start_time
        return ProcessingResult(
            video_path=video_path,
            output_path=output_path,
            status="error",
            duration_seconds=duration,
            message=str(e)
        )


class ParallelProcessor:
    """
    Gestionnaire de traitement parallèle pour vidéos.
    Utilise ProcessPoolExecutor pour CPU-bound tasks.
    """
    
    def __init__(
        self,
        config: PreprocessingConfig,
        max_workers: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int, ProcessingResult], None]] = None
    ):
        """
        Initialise le processeur parallèle.
        
        Args:
            config: Configuration de prétraitement
            max_workers: Nombre max de workers (défaut: CPU cores - 1)
            progress_callback: Callback appelé pour chaque vidéo terminée
        """
        self.config = config
        self.max_workers = max_workers or max(1, multiprocessing.cpu_count() - 1)
        self.progress_callback = progress_callback
        
        self._results: List[ProcessingResult] = []
        self._running = False
    
    def _config_to_dict(self) -> Dict:
        """Convertit la config en dict pour pickling."""
        return {
            "output_format": self.config.output_format,
            "target_width": self.config.target_width,
            "target_height": self.config.target_height,
            "preserve_aspect_ratio": self.config.preserve_aspect_ratio,
            "padding_color": self.config.padding_color,
            "target_fps": self.config.target_fps,
            "quality_crf": self.config.quality_crf,
            "enable_stabilization": self.config.enable_stabilization,
            "stabilization_smoothing": self.config.stabilization_smoothing,
            "enable_distortion_correction": self.config.enable_distortion_correction,
            "extract_frames": self.config.extract_frames,
            "frame_interval": self.config.frame_interval,
            "clip_duration": self.config.clip_duration,
            "enable_augmentation": self.config.enable_augmentation,
            "brightness_range": self.config.brightness_range,
            "contrast_range": self.config.contrast_range,
            "output_dir": self.config.output_dir,
            "overwrite": self.config.overwrite,
        }
    
    def process_videos(
        self,
        video_paths: List[str],
        output_paths: Optional[List[str]] = None
    ) -> List[ProcessingResult]:
        """
        Traite plusieurs vidéos en parallèle.
        
        Args:
            video_paths: Liste des chemins vidéo
            output_paths: Liste optionnelle des chemins de sortie
            
        Returns:
            Liste des résultats
        """
        if not video_paths:
            return []
        
        self._running = True
        self._results = []
        
        total = len(video_paths)
        config_dict = self._config_to_dict()
        
        # Préparer les arguments
        if output_paths is None:
            output_paths = [None] * total
        
        args_list = [
            (video_paths[i], config_dict, output_paths[i])
            for i in range(total)
        ]
        
        logger.info(f"🚀 Démarrage du traitement parallèle de {total} vidéos")
        logger.info(f"   Workers: {self.max_workers}")
        
        start_time = time.time()
        completed = 0
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Soumettre toutes les tâches
            future_to_path = {
                executor.submit(_process_single_video, args): args[0]
                for args in args_list
            }
            
            # Collecter les résultats
            for future in as_completed(future_to_path):
                video_path = future_to_path[future]
                
                try:
                    result = future.result()
                    self._results.append(result)
                    
                    completed += 1
                    
                    # Log le résultat
                    status_icon = "✓" if result.status == "success" else "✗" if result.status == "error" else "→"
                    logger.info(
                        f"   [{completed}/{total}] {status_icon} {Path(video_path).name} "
                        f"({result.duration_seconds:.1f}s)"
                    )
                    
                    # Callback
                    if self.progress_callback:
                        self.progress_callback(completed, total, result)
                        
                except Exception as e:
                    logger.error(f"   ✗ {Path(video_path).name}: {e}")
                    self._results.append(ProcessingResult(
                        video_path=video_path,
                        output_path=None,
                        status="error",
                        duration_seconds=0,
                        message=str(e)
                    ))
                    completed += 1
        
        self._running = False
        
        total_time = time.time() - start_time
        success_count = sum(1 for r in self._results if r.status == "success")
        error_count = sum(1 for r in self._results if r.status == "error")
        
        logger.info(f"\n📊 Traitement terminé en {total_time:.1f}s")
        logger.info(f"   Succès: {success_count}/{total}")
        logger.info(f"   Erreurs: {error_count}/{total}")
        logger.info(f"   Temps moyen: {total_time/total:.1f}s/vidéo")
        
        return self._results
    
    def process_directory(
        self,
        input_dir: str,
        extensions: List[str] = None
    ) -> List[ProcessingResult]:
        """
        Traite toutes les vidéos d'un dossier.
        
        Args:
            input_dir: Dossier contenant les vidéos
            extensions: Extensions à traiter (défaut: mp4, MP4, avi, mov)
            
        Returns:
            Liste des résultats
        """
        if extensions is None:
            extensions = ["mp4", "MP4", "avi", "AVI", "mov", "MOV", "mkv", "MKV"]
        
        video_paths = []
        
        for ext in extensions:
            video_paths.extend(Path(input_dir).glob(f"*.{ext}"))
        
        video_paths = [str(p) for p in sorted(video_paths)]
        
        if not video_paths:
            logger.warning(f"Aucune vidéo trouvée dans {input_dir}")
            return []
        
        logger.info(f"📁 Trouvé {len(video_paths)} vidéos dans {input_dir}")
        
        return self.process_videos(video_paths)
    
    def get_summary(self) -> Dict[str, Any]:
        """Retourne un résumé du traitement."""
        if not self._results:
            return {"status": "no_results"}
        
        success = [r for r in self._results if r.status == "success"]
        errors = [r for r in self._results if r.status == "error"]
        skipped = [r for r in self._results if r.status == "skipped"]
        
        total_duration = sum(r.duration_seconds for r in self._results)
        
        return {
            "total": len(self._results),
            "success": len(success),
            "errors": len(errors),
            "skipped": len(skipped),
            "total_duration_seconds": total_duration,
            "average_duration_seconds": total_duration / len(self._results) if self._results else 0,
            "error_details": [
                {"path": r.video_path, "message": r.message}
                for r in errors
            ]
        }


class BatchProcessor:
    """
    Traitement par lots avec gestion de la mémoire.
    Utile pour des centaines de vidéos longues.
    """
    
    def __init__(
        self,
        config: PreprocessingConfig,
        batch_size: int = 10,
        max_workers: int = None
    ):
        """
        Initialise le processeur par lots.
        
        Args:
            config: Configuration de prétraitement
            batch_size: Nombre de vidéos par lot
            max_workers: Workers par lot
        """
        self.config = config
        self.batch_size = batch_size
        self.max_workers = max_workers
        
        self._all_results: List[ProcessingResult] = []
    
    def process_all(
        self,
        video_paths: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[ProcessingResult]:
        """
        Traite toutes les vidéos par lots.
        
        Args:
            video_paths: Liste complète des vidéos
            progress_callback: Callback (videos_done, total)
            
        Returns:
            Liste de tous les résultats
        """
        total = len(video_paths)
        batches = [
            video_paths[i:i + self.batch_size]
            for i in range(0, total, self.batch_size)
        ]
        
        logger.info(f"📦 Traitement de {total} vidéos en {len(batches)} lots")
        
        self._all_results = []
        completed = 0
        
        for batch_idx, batch in enumerate(batches):
            logger.info(f"\n🔄 Lot {batch_idx + 1}/{len(batches)} ({len(batch)} vidéos)")
            
            processor = ParallelProcessor(
                self.config,
                max_workers=self.max_workers
            )
            
            results = processor.process_videos(batch)
            self._all_results.extend(results)
            
            completed += len(batch)
            
            if progress_callback:
                progress_callback(completed, total)
        
        return self._all_results


def estimate_processing_time(
    video_paths: List[str],
    config: PreprocessingConfig,
    sample_size: int = 2
) -> float:
    """
    Estime le temps de traitement total.
    
    Args:
        video_paths: Liste des vidéos
        config: Configuration
        sample_size: Nombre de vidéos pour l'estimation
        
    Returns:
        Temps estimé en secondes
    """
    if len(video_paths) < sample_size:
        sample_size = len(video_paths)
    
    sample_paths = video_paths[:sample_size]
    
    processor = ParallelProcessor(config, max_workers=1)
    results = processor.process_videos(sample_paths)
    
    avg_time = sum(r.duration_seconds for r in results) / len(results)
    
    # Estimer avec parallélisme
    workers = max(1, multiprocessing.cpu_count() - 1)
    estimated = (len(video_paths) * avg_time) / workers
    
    return estimated


if __name__ == "__main__":
    import glob
    
    # Test du traitement parallèle
    config = PreprocessingConfig(
        target_width=1280,
        target_height=720,
        target_fps=25.0,
        output_dir="preprocessed",
    )
    
    # Trouver les vidéos
    dataset_path = "/home/epl/DATA/S03/MTH2329/APEKE/PROJET/Dataset"
    videos = glob.glob(f"{dataset_path}/*.mp4") + glob.glob(f"{dataset_path}/*.MP4")
    
    if videos:
        processor = ParallelProcessor(config, max_workers=4)
        results = processor.process_videos(videos[:3])  # Test avec 3 vidéos
        
        summary = processor.get_summary()
        print(f"\nSummary: {summary}")
