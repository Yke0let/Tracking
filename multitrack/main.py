"""
Point d'entrée principal du système de surveillance multi-caméras.
Interface CLI pour l'analyse, la synchronisation et l'affichage des flux.
"""

import argparse
import glob
import os
import sys
import json
import time
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict

# Import des modules
from .video_metadata import extract_all_metadata, save_metadata_report, VideoMetadata
from .video_sync import VideoSynchronizer, SyncMethod, SynchronizedReader
from .stream_manager import StreamManager, CameraConfig
from .data_storage import DataStorage, VideoRecord
from .health_monitor import HealthMonitor, AlertLevel
from .sync_merger import SyncMerger, merge_videos_to_grid
from .time_sync_player import TimeSyncedPlayer, get_rotation_for_video
from .preprocessing import PreprocessingConfig, PreprocessingPipeline, OutputFormat
from .parallel_processor import ParallelProcessor, BatchProcessor
from .object_detector import ObjectDetector, create_surveillance_detector
from .object_tracker import ObjectTracker, create_surveillance_tracker
from .mcmot import MCMOTTracker, BEVVisualizer


def expand_video_patterns(patterns):
    """
    Étend les patterns pour inclure les extensions majuscules/minuscules.
    Ex: Dataset/*.mp4 -> inclut aussi Dataset/*.MP4
    """
    video_paths = []
    for pattern in patterns:
        video_paths.extend(glob.glob(pattern))
        # Ajouter aussi la version majuscule/minuscule
        if pattern.endswith('.mp4'):
            video_paths.extend(glob.glob(pattern[:-4] + '.MP4'))
        elif pattern.endswith('.MP4'):
            video_paths.extend(glob.glob(pattern[:-4] + '.mp4'))
    # Supprimer les doublons
    return list(dict.fromkeys(video_paths))


def analyze_videos(args):
    """Analyse les métadonnées des vidéos existantes."""
    print("\n📹 Analyse des vidéos...\n")
    
    # Trouver les vidéos (inclut .mp4 et .MP4)
    video_paths = expand_video_patterns(args.videos)
    
    if not video_paths:
        print("❌ Aucune vidéo trouvée")
        return
    
    print(f"Vidéos trouvées: {len(video_paths)}\n")
    
    # Extraire les métadonnées
    metadata_list = extract_all_metadata(video_paths)
    
    # Afficher le résumé
    print(f"\n{'='*60}")
    print("📊 RÉSUMÉ")
    print(f"{'='*60}")
    
    total_duration = sum(m.duration_seconds for m in metadata_list)
    total_frames = sum(m.frame_count for m in metadata_list)
    resolutions = set(f"{m.width}x{m.height}" for m in metadata_list)
    framerates = set(f"{m.fps:.2f}" for m in metadata_list)
    
    print(f"   Vidéos analysées: {len(metadata_list)}")
    print(f"   Durée totale: {total_duration/60:.1f} minutes")
    print(f"   Frames totales: {total_frames:,}")
    print(f"   Résolutions: {', '.join(resolutions)}")
    print(f"   Framerates: {', '.join(framerates)} FPS")
    
    # Sauvegarder le rapport si demandé
    if args.output:
        save_metadata_report(metadata_list, args.output)
        print(f"\n✓ Rapport sauvegardé: {args.output}")
    
    return metadata_list


def synchronize_videos(args):
    """Synchronise les vidéos existantes."""
    print("\n🔄 Synchronisation des vidéos...\n")
    
    # Trouver les vidéos (inclut .mp4 et .MP4)
    video_paths = expand_video_patterns(args.videos)
    
    if len(video_paths) < 2:
        print("❌ Au moins 2 vidéos requises pour la synchronisation")
        return
    
    # Extraire les métadonnées
    metadata_list = extract_all_metadata(video_paths)
    
    # Créer le synchroniseur
    synchronizer = VideoSynchronizer(
        target_fps=args.target_fps,
        max_drift_ms=args.max_drift_ms
    )
    
    # Choisir la méthode
    method = SyncMethod(args.method)
    print(f"Méthode: {method.value}\n")
    
    try:
        if method == SyncMethod.TIMESTAMP:
            results = synchronizer.synchronize_by_timestamp(metadata_list)
        elif method == SyncMethod.AUDIO:
            results = synchronizer.synchronize_by_audio(video_paths)
        elif method == SyncMethod.VISUAL:
            results = synchronizer.synchronize_by_visual_event(
                video_paths,
                event_type=args.event_type
            )
        else:
            print("❌ Méthode non supportée")
            return
        
        # Afficher les résultats
        print(f"\n{'='*60}")
        print("📊 RÉSULTATS DE SYNCHRONISATION")
        print(f"{'='*60}")
        
        for r in results:
            print(f"\n   {r.camera_id}:")
            print(f"      Offset: {r.offset_seconds:.3f}s ({r.offset_frames} frames)")
            print(f"      Confiance: {r.confidence*100:.0f}%")
            print(f"      FPS: {r.original_fps:.2f} → {r.target_fps:.2f}")
        
        # Sauvegarder les résultats
        if args.output:
            sync_data = {
                "method": method.value,
                "target_fps": args.target_fps,
                "results": [
                    {
                        "camera_id": r.camera_id,
                        "video_path": r.video_path,
                        "offset_seconds": r.offset_seconds,
                        "offset_frames": r.offset_frames,
                        "confidence": r.confidence,
                    }
                    for r in results
                ]
            }
            with open(args.output, "w") as f:
                json.dump(sync_data, f, indent=2)
            print(f"\n✓ Résultats sauvegardés: {args.output}")
        
        return results
        
    except Exception as e:
        print(f"❌ Erreur de synchronisation: {e}")
        return None


def merge_videos(args):
    """Fusionne les vidéos en une grille."""
    print("\n🎬 Fusion des vidéos...\n")
    
    # Trouver les vidéos (inclut .mp4 et .MP4)
    video_paths = expand_video_patterns(args.videos)
    
    if not video_paths:
        print("❌ Aucune vidéo trouvée")
        return
    
    # Créer le dict camera_id -> path
    video_dict = {}
    for i, path in enumerate(video_paths[:args.max_cameras]):
        basename = Path(path).stem
        camera_id = f"cam_{basename.lower().replace(' ', '_')[:20]}"
        video_dict[camera_id] = path
    
    print(f"Fusion de {len(video_dict)} vidéos...\n")
    
    output = args.output or "merged_output.mp4"
    
    merge_videos_to_grid(
        video_dict,
        output,
        target_fps=args.target_fps,
        max_duration=args.duration,
        show_progress=True
    )
    
    print(f"\n✓ Vidéo créée: {output}")


def stream_live(args):
    """Affiche les flux en temps réel."""
    print("\n📺 Mode flux en temps réel...\n")
    
    # Charger la configuration
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
    else:
        # Configuration par défaut pour les fichiers locaux
        video_paths = expand_video_patterns(args.videos)
        
        def get_rotation_for_video(path):
            """Retourne la rotation à appliquer selon le fichier vidéo."""
            basename = Path(path).name.upper()
            # Rotation de -90° (270°) pour CAMERA_DEVANTURE_PORTE_ENTREE
            if "DEVANTURE_PORTE_ENTREE" in basename:
                return 270
            return 0
        
        config = {
            "cameras": [
                {
                    "camera_id": f"cam_{i}",
                    "name": Path(p).stem,
                    "location": f"Camera {i}",
                    "source": p,
                    "rotation": get_rotation_for_video(p),
                }
                for i, p in enumerate(video_paths[:args.max_cameras])
            ]
        }
    
    if not config.get("cameras"):
        print("❌ Aucune caméra configurée")
        return
    
    # Créer le gestionnaire
    manager = StreamManager()
    
    # Créer le moniteur de santé
    def on_alert(alert):
        level_icon = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "❌",
            AlertLevel.CRITICAL: "🚨",
        }
        print(f"{level_icon.get(alert.level, '•')} {alert.camera_id}: {alert.message}")
    
    monitor = HealthMonitor(on_alert=on_alert)
    
    # Ajouter les caméras
    for cam_config in config["cameras"]:
        camera = CameraConfig(
            camera_id=cam_config["camera_id"],
            name=cam_config.get("name", cam_config["camera_id"]),
            location=cam_config.get("location", ""),
            source=cam_config["source"],
            username=cam_config.get("username"),
            password=cam_config.get("password"),
            rotation_degrees=cam_config.get("rotation", 0),
        )
        manager.add_camera(camera)
        monitor.add_camera(camera.camera_id)
    
    # Configurer le merger
    merger = SyncMerger(target_fps=args.target_fps)
    merger.configure_cameras(
        [c["camera_id"] for c in config["cameras"]],
        {c["camera_id"]: c.get("name", c["camera_id"]) for c in config["cameras"]}
    )
    
    print(f"Démarrage de {len(config['cameras'])} flux...\n")
    print("Contrôles: Q=Quitter, R=Enregistrer, S=Screenshot\n")
    
    # Fonction pour récupérer les frames
    def get_frames():
        frames = manager.get_all_frames()
        # Rapporter au moniteur
        for cam_id, frame_data in frames.items():
            if frame_data:
                stream = manager.streams.get(cam_id)
                if stream:
                    monitor.report_frame(cam_id, stream.fps_actual)
        return frames
    
    def get_status():
        return {
            cam_id: health.status.value
            for cam_id, health in monitor.cameras.items()
        }
    
    # Démarrer
    try:
        manager.start_all()
        monitor.start()
        
        # Petit délai pour laisser les flux démarrer
        time.sleep(1.0)
        
        # Affichage live
        record_path = args.output if args.record else None
        merger.display_live(
            get_frames,
            window_name="Multi-Camera Surveillance",
            status_source=get_status,
            record_path=record_path
        )
        
    finally:
        merger.stop()
        monitor.stop()
        manager.stop_all()
        
    print("\n✓ Arrêté")


def sync_live(args):
    """Lecture synchronisée basée sur les timestamps d'enregistrement."""
    print("\n🔄 Mode lecture synchronisée par timestamp...\n")
    
    # Trouver les vidéos
    video_paths = expand_video_patterns(args.videos)
    
    if not video_paths:
        print("❌ Aucune vidéo trouvée")
        return
    
    print(f"Vidéos trouvées: {len(video_paths)}\n")
    
    # Créer le lecteur synchronisé
    enable_track = getattr(args, 'track', False)
    enable_mcmot = getattr(args, 'mcmot', False)
    enable_reid = not getattr(args, 'no_reid', False)
    reid_threshold = getattr(args, 'reid_threshold', 0.6)
    player = TimeSyncedPlayer(
        target_fps=args.target_fps,
        playback_speed=args.speed,
        grid_cell_size=(args.cell_width, args.cell_height),
        align_start=args.align_start,
        enable_tracking=enable_track and not enable_mcmot,
        enable_mcmot=enable_mcmot,
        enable_reid=enable_reid,
        reid_threshold=reid_threshold,
        tracker_model="yolov8n.pt",
    )
    
    # Charger les timestamps manuels si --manual-sync est activé
    if args.manual_sync:
        # Timestamps manuels avec corrections de synchronisation
        manual_timestamps = {
            "HALL_PORTE_DROITE": "12:10:10",      # +1s retard (était 12:10:09)
            "HALL_PORTE_GAUCHE": "12:09:55",      # -2s avancé
            "DEBUT_COULOIR_DROIT": "12:10:00",    # Retardé de 5s pour sync avec HALL_PORTE_GAUCHE
            "DEVANTURE_SOUS_ARBRE": "12:10:00",   # -2s avancé
            "FIN_COULOIR_GAUCHE_REZ_PARTIE_2": "12:10:03",  # DEBUT
            "FIN_COULOIR_DROIT": "12:10:08",
            "HALL_PORTE_ENTREE": "12:10:27",      # -1s avancé (était 12:10:28)
            "DEVANTURE_PORTE_ENTREE": "12:10:34", # -3s avancé (était 12:10:37)
            "FIN_COULOIR_GAUCHE_REZ_PARTIE_1": "12:17:33",  # FIN
        }
        
        # Exclusions
        exclude_patterns = ["PARTIE_1(1)", "ESCALIER", "ETAGE1", "FIN_COULOIR_GAUCHE_REZ_FIN"]
        
        player.set_manual_timestamps(manual_timestamps, reference_date="2025-12-11")
        
        # Filtrer les vidéos à exclure
        filtered_paths = []
        for vp in video_paths:
            vp_upper = Path(vp).stem.upper()
            excluded = any(pat.upper() in vp_upper for pat in exclude_patterns)
            if not excluded:
                filtered_paths.append(vp)
            else:
                print(f"  ⏭️ Exclu: {Path(vp).name}")
        video_paths = filtered_paths
        print()
    
    # Ajouter les vidéos avec rotation appropriée
    for video_path in video_paths[:args.max_cameras]:
        rotation = get_rotation_for_video(video_path)
        try:
            player.add_video(video_path, rotation_degrees=rotation)
        except Exception as e:
            print(f"⚠️ Erreur pour {video_path}: {e}")
    
    if not player.sources:
        print("❌ Aucune vidéo valide")
        return
    
    # Lancer la lecture
    try:
        record_path = getattr(args, 'record', None)
        headless = getattr(args, 'headless', False)
        max_frames = getattr(args, 'max_frames', None)
        player.play(record_output=record_path, headless=headless, max_frames=max_frames)
    except KeyboardInterrupt:
        player.stop()
        print("\n⚠️ Interrompu")


def preprocess_videos(args):
    """Prétraite les vidéos pour créer un dataset cohérent."""
    print("\n🔧 Pipeline de prétraitement vidéo...\n")
    
    # Trouver les vidéos
    video_paths = expand_video_patterns(args.videos)
    
    if not video_paths:
        print("❌ Aucune vidéo trouvée")
        return
    
    print(f"Vidéos trouvées: {len(video_paths)}\n")
    
    # Créer la configuration
    config = PreprocessingConfig(
        output_format=OutputFormat.MP4_H264,
        target_width=args.width,
        target_height=args.height,
        target_fps=args.fps,
        quality_crf=args.quality,
        enable_stabilization=args.stabilize,
        extract_frames=args.extract_frames,
        frame_interval=args.frame_interval,
        clip_duration=args.clip_duration,
        enable_augmentation=args.augment,
        output_dir=args.output,
        overwrite=args.overwrite,
    )
    
    # Traitement parallèle ou séquentiel
    if args.parallel:
        processor = ParallelProcessor(config, max_workers=args.workers)
        results = processor.process_videos(video_paths)
        summary = processor.get_summary()
        
        print(f"\n{'='*60}")
        print("📊 RÉSUMÉ")
        print(f"{'='*60}")
        print(f"   Total: {summary['total']}")
        print(f"   Succès: {summary['success']}")
        print(f"   Erreurs: {summary['errors']}")
        print(f"   Temps total: {summary['total_duration_seconds']:.1f}s")
    else:
        pipeline = PreprocessingPipeline(config)
        
        for i, video_path in enumerate(video_paths):
            print(f"[{i+1}/{len(video_paths)}] {Path(video_path).name}")
            result = pipeline.process_video(video_path)
            status = "✓" if result.get("status") == "success" else "✗"
            print(f"   {status} -> {result.get('output', 'error')}")
    
    print(f"\n✓ Prétraitement terminé. Sortie: {args.output}/")


def detect_objects(args):
    """Détecte les objets dans les vidéos."""
    print("\n🔍 Détection d'objets...\n")
    
    # Trouver les vidéos
    video_paths = expand_video_patterns(args.videos)
    
    if not video_paths:
        print("❌ Aucune vidéo trouvée")
        return
    
    print(f"Vidéos trouvées: {len(video_paths)}\n")
    
    # Créer le détecteur
    detector = create_surveillance_detector(
        model_size=args.model,
        enable_tracking=args.tracking,
        confidence=args.confidence,
    )
    
    # Créer le dossier de sortie
    os.makedirs(args.output, exist_ok=True)
    
    for video_path in video_paths:
        video_name = Path(video_path).stem
        output_path = os.path.join(args.output, f"{video_name}_detected.mp4")
        
        print(f"📹 {video_name}")
        
        try:
            results = detector.detect_video(
                video_path,
                output_path=output_path,
                show=args.show,
                max_frames=args.max_frames,
            )
            
            stats = detector.get_statistics(results)
            
            print(f"   ✓ {stats['total_frames']} frames, {stats['total_detections']} détections")
            print(f"   ✓ {stats['avg_inference_ms']:.1f}ms/frame")
            print(f"   ✓ Classes: {stats['class_counts']}")
            print(f"   → {output_path}\n")
            
        except Exception as e:
            print(f"   ✗ Erreur: {e}\n")
    
    print(f"✓ Détection terminée. Sortie: {args.output}/")


def track_objects(args):
    """Effectue le tracking multi-objets avec trajectoires."""
    print("\n🔍 Tracking multi-objets...\n")
    
    # Trouver les vidéos
    video_paths = expand_video_patterns(args.videos)
    
    if not video_paths:
        print("❌ Aucune vidéo trouvée")
        return
    
    print(f"Vidéos trouvées: {len(video_paths)}\n")
    
    # Créer le tracker
    tracker = create_surveillance_tracker(
        model_size=args.model,
        tracker=args.tracker,
        confidence=args.confidence,
    )
    
    # Créer le dossier de sortie
    os.makedirs(args.output, exist_ok=True)
    
    for video_path in video_paths:
        video_name = Path(video_path).stem
        output_path = os.path.join(args.output, f"{video_name}_tracked.mp4")
        
        print(f"📹 {video_name}")
        
        try:
            # Reset pour chaque vidéo
            tracker.reset()
            
            results = tracker.track_video(
                video_path,
                output_path=output_path,
                show=args.show,
                max_frames=args.max_frames,
                show_trajectory=not args.no_trajectory,
            )
            
            stats = tracker.get_statistics()
            
            print(f"   ✓ {stats['total_frames']} frames, {stats['total_tracks']} tracks")
            print(f"   ✓ Actifs: {stats['active_tracks']}, Perdus: {stats['lost_tracks']}")
            print(f"   ✓ Classes: {stats['class_distribution']}")
            print(f"   → {output_path}\n")
            
        except Exception as e:
            print(f"   ✗ Erreur: {e}\n")
    
    print(f"✓ Tracking terminé. Sortie: {args.output}/")


def mcmot_track(args):
    """Multi-Camera Multi-Object Tracking avec IDs globaux."""
    print("\n🎯 MCMOT - Multi-Camera Tracking...\n")
    
    video_paths = expand_video_patterns(args.videos)
    
    if not video_paths:
        print("❌ Aucune vidéo trouvée")
        return
    
    print(f"Caméras: {len(video_paths)}\n")
    
    # Créer le tracker MCMOT
    tracker = MCMOTTracker(
        model_path=f"yolov8{args.model}.pt",
        reid_threshold=args.reid_threshold,
        enable_reid=not args.no_reid,
    )
    
    # Ouvrir les captures
    captures = {}
    for path in video_paths[:4]:  # Max 4 caméras
        camera_id = Path(path).stem
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            captures[camera_id] = cap
            print(f"  ✓ {camera_id}")
    
    if not captures:
        print("❌ Aucune vidéo valide")
        return
    
    # Fenêtre d'affichage
    cv2.namedWindow("MCMOT", cv2.WINDOW_NORMAL)
    
    frame_count = 0
    start_time = time.time()
    
    print(f"\n▶️ Démarrage du tracking multi-caméra...")
    print("   Contrôles: Q=Quitter\n")
    
    try:
        while True:
            timestamp = time.time() - start_time
            frames = {}
            
            # Lire une frame de chaque caméra
            for camera_id, cap in list(captures.items()):
                ret, frame = cap.read()
                if not ret:
                    continue
                frames[camera_id] = frame
            
            if not frames:
                break
            
            if args.max_frames and frame_count >= args.max_frames:
                break
            
            # Tracker chaque caméra
            all_tracks = {}
            for camera_id, frame in frames.items():
                tracks = tracker.process_frame(camera_id, frame, timestamp)
                all_tracks[camera_id] = tracks
            
            # Association cross-camera
            tracker.associate_cross_camera(timestamp)
            
            # Dessiner avec IDs globaux
            annotated_frames = {}
            for camera_id, frame in frames.items():
                annotated = tracker.draw_with_global_ids(
                    camera_id, frame, all_tracks.get(camera_id, [])
                )
                annotated_frames[camera_id] = annotated
            
            # Créer une grille d'affichage
            n = len(annotated_frames)
            cols = 2 if n > 1 else 1
            rows = (n + cols - 1) // cols
            cell_h, cell_w = 360, 480
            grid = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
            
            for i, (cam_id, frame) in enumerate(annotated_frames.items()):
                row, col = i // cols, i % cols
                x, y = col * cell_w, row * cell_h
                resized = cv2.resize(frame, (cell_w, cell_h))
                grid[y:y+cell_h, x:x+cell_w] = resized
            
            # Afficher stats
            stats = tracker.get_statistics()
            info = f"Global: {stats['total_global_tracks']} | Cross-cam: {stats['cross_camera_tracks']}"
            cv2.putText(grid, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("MCMOT", grid)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
    
    finally:
        for cap in captures.values():
            cap.release()
        cv2.destroyAllWindows()
    
    stats = tracker.get_statistics()
    print(f"\n✓ Terminé: {frame_count} frames")
    print(f"   Tracks globaux: {stats['total_global_tracks']}")
    print(f"   Tracks cross-camera: {stats['cross_camera_tracks']}")


def create_database(args):
    """Crée et initialise la base de données."""
    print("\n💾 Initialisation du stockage...\n")
    
    storage = DataStorage(base_path=args.output)
    
    # Si des vidéos sont spécifiées, les importer
    if args.videos:
        video_paths = expand_video_patterns(args.videos)
        
        if video_paths:
            print(f"Import de {len(video_paths)} vidéos...")
            metadata_list = extract_all_metadata(video_paths)
            
            for meta in metadata_list:
                # Enregistrer la caméra
                storage.register_camera(
                    camera_id=meta.camera_id,
                    name=meta.camera_id,
                    location=meta.location,
                    source=meta.filepath
                )
                
                # Enregistrer la vidéo
                record = VideoRecord(
                    id=None,
                    camera_id=meta.camera_id,
                    filename=meta.filename,
                    filepath=meta.filepath,
                    start_time=meta.creation_time or meta.modification_time or "",
                    end_time=None,
                    duration_seconds=meta.duration_seconds,
                    width=meta.width,
                    height=meta.height,
                    fps=meta.fps,
                    frame_count=meta.frame_count,
                    codec=meta.codec,
                    file_size_bytes=meta.file_size_bytes,
                    location=meta.location,
                    has_audio=meta.has_audio,
                    bitrate_kbps=meta.bitrate_kbps,
                )
                storage.add_video_record(record)
            
            print(f"✓ {len(metadata_list)} vidéos importées")
    
    # Afficher les stats
    stats = storage.get_storage_statistics()
    
    print(f"\n{'='*60}")
    print("📊 STATISTIQUES")
    print(f"{'='*60}")
    print(f"   Vidéos: {stats['total_videos']}")
    print(f"   Taille: {stats['total_size_gb']:.2f} GB")
    print(f"   Durée: {stats['total_duration_hours']:.1f} heures")
    print(f"   Caméras: {stats['camera_count']}")
    
    # Export
    export_path = storage.export_metadata_json()
    print(f"\n✓ Base initialisée: {storage.db_path}")
    print(f"✓ Export JSON: {export_path}")
    
    storage.close()


def main():
    """Point d'entrée CLI."""
    parser = argparse.ArgumentParser(
        description="Système de surveillance multi-caméras",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Analyser les vidéos
  python -m multicam analyze Dataset/*.mp4 -o metadata.json
  
  # Synchroniser les vidéos
  python -m multicam sync Dataset/*.mp4 --method timestamp -o sync_results.json
  
  # Fusionner en grille
  python -m multicam merge Dataset/*.mp4 -o grid.mp4 --duration 60
  
  # Affichage en temps réel
  python -m multicam live Dataset/*.mp4
  
  # Avec flux RTSP
  python -m multicam live --config cameras.json
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commande")
    
    # Commande: analyze
    p_analyze = subparsers.add_parser("analyze", help="Analyser les métadonnées vidéo")
    p_analyze.add_argument("videos", nargs="+", help="Fichiers vidéo (glob patterns acceptés)")
    p_analyze.add_argument("-o", "--output", help="Fichier de sortie JSON")
    p_analyze.set_defaults(func=analyze_videos)
    
    # Commande: sync
    p_sync = subparsers.add_parser("sync", help="Synchroniser les vidéos")
    p_sync.add_argument("videos", nargs="+", help="Fichiers vidéo")
    p_sync.add_argument("-m", "--method", default="timestamp",
                        choices=["timestamp", "audio", "visual"],
                        help="Méthode de synchronisation")
    p_sync.add_argument("--event-type", default="motion",
                        choices=["flash", "motion"],
                        help="Type d'événement pour la méthode visuelle")
    p_sync.add_argument("--target-fps", type=float, default=25.0,
                        help="FPS cible")
    p_sync.add_argument("--max-drift-ms", type=float, default=100.0,
                        help="Dérive max acceptable en ms")
    p_sync.add_argument("-o", "--output", help="Fichier de sortie JSON")
    p_sync.set_defaults(func=synchronize_videos)
    
    # Commande: merge
    p_merge = subparsers.add_parser("merge", help="Fusionner les vidéos en grille")
    p_merge.add_argument("videos", nargs="+", help="Fichiers vidéo")
    p_merge.add_argument("-o", "--output", default="merged.mp4",
                         help="Fichier de sortie")
    p_merge.add_argument("--target-fps", type=float, default=25.0,
                         help="FPS de sortie")
    p_merge.add_argument("--duration", type=float,
                         help="Durée max en secondes")
    p_merge.add_argument("--max-cameras", type=int, default=16,
                         help="Nombre max de caméras")
    p_merge.set_defaults(func=merge_videos)
    
    # Commande: live
    p_live = subparsers.add_parser("live", help="Affichage en temps réel")
    p_live.add_argument("videos", nargs="*", help="Fichiers vidéo ou sources")
    p_live.add_argument("-c", "--config", help="Fichier de configuration JSON")
    p_live.add_argument("--target-fps", type=float, default=25.0,
                        help="FPS cible")
    p_live.add_argument("--max-cameras", type=int, default=16,
                        help="Nombre max de caméras")
    p_live.add_argument("-r", "--record", action="store_true",
                        help="Enregistrer automatiquement")
    p_live.add_argument("-o", "--output", default="recording.mp4",
                        help="Fichier d'enregistrement")
    p_live.set_defaults(func=stream_live)
    
    # Commande: sync-live (lecture synchronisée par timestamp)
    p_sync_live = subparsers.add_parser("sync-live", 
        help="Lecture synchronisée basée sur les timestamps d'enregistrement")
    p_sync_live.add_argument("videos", nargs="+", help="Fichiers vidéo")
    p_sync_live.add_argument("--target-fps", type=float, default=25.0,
                             help="FPS cible")
    p_sync_live.add_argument("--speed", type=float, default=1.0,
                             help="Vitesse de lecture (1.0 = temps réel)")
    p_sync_live.add_argument("--max-cameras", type=int, default=16,
                             help="Nombre max de caméras")
    p_sync_live.add_argument("--cell-width", type=int, default=480,
                             help="Largeur des cellules de grille")
    p_sync_live.add_argument("--cell-height", type=int, default=360,
                             help="Hauteur des cellules de grille")
    p_sync_live.add_argument("--align-start", action="store_true",
                             help="Toutes les vidéos démarrent en même temps (ignorer les timestamps)")
    p_sync_live.add_argument("--manual-sync", action="store_true",
                             help="Utiliser les timestamps manuels configurés (8 caméras principales)")
    p_sync_live.add_argument("--track", action="store_true",
                             help="Activer le tracking d'objets avec trajectoires")
    p_sync_live.add_argument("--mcmot", action="store_true",
                             help="Activer le tracking cross-camera avec IDs globaux")
    p_sync_live.add_argument("--no-reid", action="store_true",
                             help="Désactiver ReID (plus rapide, utilise position)")
    p_sync_live.add_argument("--reid-threshold", type=float, default=0.6,
                             help="Seuil de similarité ReID (0-1)")
    p_sync_live.add_argument("--record", type=str, metavar="OUTPUT.mp4",
                             help="Enregistrer le flux synchronisé dans un fichier vidéo")
    p_sync_live.add_argument("--headless", action="store_true",
                             help="Mode sans affichage (enregistrement uniquement)")
    p_sync_live.add_argument("--max-frames", type=int,
                             help="Nombre max de frames à traiter")
    p_sync_live.set_defaults(func=sync_live)
    
    # Commande: preprocess (prétraitement vidéo)
    p_prep = subparsers.add_parser("preprocess", 
        help="Prétraiter les vidéos pour créer un dataset cohérent")
    p_prep.add_argument("videos", nargs="+", help="Fichiers vidéo")
    p_prep.add_argument("-o", "--output", default="preprocessed",
                        help="Dossier de sortie")
    p_prep.add_argument("--width", type=int, default=1280,
                        help="Largeur cible")
    p_prep.add_argument("--height", type=int, default=720,
                        help="Hauteur cible")
    p_prep.add_argument("--fps", type=float, default=25.0,
                        help="FPS cible")
    p_prep.add_argument("--quality", type=int, default=23,
                        help="Qualité CRF (0-51, plus bas = meilleur)")
    p_prep.add_argument("--stabilize", action="store_true",
                        help="Activer la stabilisation d'image")
    p_prep.add_argument("--extract-frames", action="store_true",
                        help="Extraire les frames en images")
    p_prep.add_argument("--frame-interval", type=int, default=1,
                        help="Intervalle d'extraction (1 = toutes les frames)")
    p_prep.add_argument("--clip-duration", type=float,
                        help="Découper en clips de N secondes")
    p_prep.add_argument("--augment", action="store_true",
                        help="Appliquer des augmentations de données")
    p_prep.add_argument("--parallel", action="store_true",
                        help="Traitement parallèle (multi-process)")
    p_prep.add_argument("--workers", type=int, default=4,
                        help="Nombre de workers pour le traitement parallèle")
    p_prep.add_argument("--overwrite", action="store_true",
                        help="Écraser les fichiers existants")
    p_prep.set_defaults(func=preprocess_videos)
    
    # Commande: detect (détection d'objets)
    p_detect = subparsers.add_parser("detect", 
        help="Détecter les objets dans les vidéos (YOLOv8)")
    p_detect.add_argument("videos", nargs="+", help="Fichiers vidéo")
    p_detect.add_argument("-o", "--output", default="detections",
                          help="Dossier de sortie")
    p_detect.add_argument("-m", "--model", default="m",
                          choices=["n", "s", "m", "l", "x"],
                          help="Taille du modèle (n=nano, s=small, m=medium, l=large, x=xlarge)")
    p_detect.add_argument("-c", "--confidence", type=float, default=0.5,
                          help="Seuil de confiance (0-1)")
    p_detect.add_argument("--tracking", action="store_true",
                          help="Activer le suivi d'objets")
    p_detect.add_argument("--show", action="store_true",
                          help="Afficher en temps réel")
    p_detect.add_argument("--max-frames", type=int,
                          help="Nombre max de frames à traiter")
    p_detect.set_defaults(func=detect_objects)
    
    # Commande: track (tracking multi-objets avec trajectoires)
    p_track = subparsers.add_parser("track",
        help="Tracking multi-objets avec visualisation des trajectoires")
    p_track.add_argument("videos", nargs="+", help="Fichiers vidéo")
    p_track.add_argument("-o", "--output", default="tracking",
                         help="Dossier de sortie")
    p_track.add_argument("-m", "--model", default="m",
                         choices=["n", "s", "m", "l", "x"],
                         help="Taille du modèle YOLO")
    p_track.add_argument("-t", "--tracker", default="bytetrack",
                         choices=["bytetrack", "botsort"],
                         help="Algorithme de tracking")
    p_track.add_argument("-c", "--confidence", type=float, default=0.5,
                         help="Seuil de confiance")
    p_track.add_argument("--show", action="store_true",
                         help="Afficher en temps réel")
    p_track.add_argument("--max-frames", type=int,
                         help="Nombre max de frames")
    p_track.add_argument("--no-trajectory", action="store_true",
                         help="Désactiver l'affichage des trajectoires")
    p_track.set_defaults(func=track_objects)
    
    # Commande: mcmot (Multi-Camera Multi-Object Tracking)
    p_mcmot = subparsers.add_parser("mcmot",
        help="Multi-Camera Tracking avec IDs globaux cross-camera")
    p_mcmot.add_argument("videos", nargs="+", help="Fichiers vidéo (max 4)")
    p_mcmot.add_argument("-m", "--model", default="n",
                         choices=["n", "s", "m", "l", "x"],
                         help="Taille du modèle YOLO")
    p_mcmot.add_argument("--reid-threshold", type=float, default=0.6,
                         help="Seuil de similarité ReID (0-1)")
    p_mcmot.add_argument("--no-reid", action="store_true",
                         help="Désactiver ReID (utiliser seulement position)")
    p_mcmot.add_argument("--max-frames", type=int,
                         help="Nombre max de frames")
    p_mcmot.set_defaults(func=mcmot_track)
    
    # Commande: init-db
    p_db = subparsers.add_parser("init-db", help="Initialiser la base de données")
    p_db.add_argument("-o", "--output", default="output",
                      help="Dossier de sortie")
    p_db.add_argument("videos", nargs="*", help="Vidéos à importer")
    p_db.set_defaults(func=create_database)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Exécuter la commande
    args.func(args)


if __name__ == "__main__":
    main()
