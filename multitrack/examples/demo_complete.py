#!/usr/bin/env python3
"""
Exemple complet d'utilisation du système multi-caméras.
Démontre toutes les fonctionnalités principales.
"""

import os
import sys
import time
import glob

# Ajouter le parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multitrack.video_metadata import extract_all_metadata, save_metadata_report
from multitrack.video_sync import VideoSynchronizer, SyncMethod, SynchronizedReader
from multitrack.stream_manager import StreamManager, CameraConfig
from multitrack.data_storage import DataStorage
from multitrack.health_monitor import HealthMonitor, AlertLevel
from multitrack.sync_merger import SyncMerger, GridLayout


def demo_metadata_extraction():
    """Démo: Extraction de métadonnées."""
    print("\n" + "="*60)
    print("📹 DÉMO 1: Extraction de Métadonnées")
    print("="*60)
    
    dataset_path = "Dataset"
    videos = glob.glob(os.path.join(dataset_path, "*.mp4")) + \
             glob.glob(os.path.join(dataset_path, "*.MP4"))
    
    if not videos:
        print("❌ Aucune vidéo trouvée dans Dataset/")
        return None
    
    print(f"\n🔍 Analyse de {len(videos)} vidéos...\n")
    
    metadata_list = extract_all_metadata(videos)
    
    # Résumé
    print(f"\n📊 Résumé:")
    for meta in metadata_list:
        print(f"   • {meta.filename}")
        print(f"     Résolution: {meta.width}x{meta.height}")
        print(f"     FPS: {meta.fps:.2f}")
        print(f"     Durée: {meta.duration_seconds:.1f}s")
        print(f"     Emplacement: {meta.location}")
        print()
    
    # Sauvegarder
    save_metadata_report(metadata_list, "output/metadata_report.json")
    
    return metadata_list


def demo_synchronization(metadata_list):
    """Démo: Synchronisation des vidéos."""
    print("\n" + "="*60)
    print("🔄 DÉMO 2: Synchronisation des Vidéos")
    print("="*60)
    
    if not metadata_list or len(metadata_list) < 2:
        print("❌ Au moins 2 vidéos nécessaires")
        return None
    
    synchronizer = VideoSynchronizer(target_fps=25.0, max_drift_ms=100.0)
    
    # Méthode 1: Par timestamps
    print("\n📅 Méthode 1: Synchronisation par Timestamps")
    try:
        results = synchronizer.synchronize_by_timestamp(metadata_list)
        for r in results:
            print(f"   {r.camera_id}: offset = {r.offset_seconds:.3f}s")
    except Exception as e:
        print(f"   ⚠️ {e}")
    
    # Méthode 2: Par événement visuel
    print("\n👁️ Méthode 2: Synchronisation par Détection de Mouvement")
    try:
        video_paths = [m.filepath for m in metadata_list[:2]]
        results = synchronizer.synchronize_by_visual_event(
            video_paths, 
            event_type="motion",
            search_window_seconds=5.0
        )
        for r in results:
            print(f"   {r.camera_id}: offset = {r.offset_seconds:.3f}s (confiance: {r.confidence*100:.0f}%)")
    except Exception as e:
        print(f"   ⚠️ {e}")
    
    return results


def demo_stream_manager():
    """Démo: Gestionnaire de flux."""
    print("\n" + "="*60)
    print("📺 DÉMO 3: Gestionnaire de Flux Multi-Caméras")
    print("="*60)
    
    dataset_path = "Dataset"
    videos = glob.glob(os.path.join(dataset_path, "*.mp4"))[:3]
    
    if not videos:
        print("❌ Aucune vidéo trouvée")
        return
    
    # Créer le gestionnaire
    manager = StreamManager()
    
    # Callback pour les frames
    frame_counts = {}
    
    def on_frame(camera_id, frame, timestamp):
        frame_counts[camera_id] = frame_counts.get(camera_id, 0) + 1
    
    manager.on_frame_callback = on_frame
    
    # Ajouter les caméras
    for i, path in enumerate(videos):
        config = CameraConfig(
            camera_id=f"cam_{i}",
            name=f"Caméra {i+1}",
            location=f"Emplacement {i+1}",
            source=path,
            buffer_size=30,
        )
        manager.add_camera(config)
    
    print(f"\n🎬 Lecture de {len(videos)} vidéos pendant 3 secondes...")
    
    with manager:
        time.sleep(3.0)
        
        # Afficher les stats
        status = manager.get_status()
        print(f"\n📊 Statistiques:")
        for cam_id, s in status.items():
            print(f"   {cam_id}: {s['frames_received']} frames @ {s['fps_actual']} FPS")


def demo_health_monitoring():
    """Démo: Monitoring de santé."""
    print("\n" + "="*60)
    print("🔍 DÉMO 4: Monitoring de Santé des Flux")
    print("="*60)
    
    alerts_received = []
    
    def on_alert(alert):
        alerts_received.append(alert)
        level_icon = {"info": "ℹ️", "warning": "⚠️", "error": "❌", "critical": "🚨"}
        print(f"   {level_icon.get(alert.level.value, '•')} {alert.camera_id}: {alert.message}")
    
    monitor = HealthMonitor(check_interval=0.5, on_alert=on_alert)
    
    # Ajouter des caméras
    for i in range(3):
        monitor.add_camera(f"cam_{i}", fps_expected=25.0)
    
    print("\n📡 Simulation de flux pendant 2 secondes...")
    
    import random
    
    with monitor:
        for _ in range(20):
            for cam_id in ["cam_0", "cam_1", "cam_2"]:
                # Simuler des variations
                fps = 25.0 + random.uniform(-8, 2)
                latency = random.uniform(10, 200)
                
                if random.random() < 0.1:
                    monitor.report_drop(cam_id)
                else:
                    monitor.report_frame(cam_id, fps, latency)
                
                monitor.report_sync_offset(cam_id, random.uniform(-80, 80))
            
            time.sleep(0.1)
        
        # Statut final
        print(f"\n📊 Statut final:")
        for cam_id, status in monitor.get_status().items():
            print(f"   {cam_id}: {status['status']} (score: {status['health_score']}%)")
        
        print(f"\n   Alertes générées: {len(alerts_received)}")


def demo_data_storage():
    """Démo: Stockage et organisation."""
    print("\n" + "="*60)
    print("💾 DÉMO 5: Stockage et Organisation des Données")
    print("="*60)
    
    storage = DataStorage(base_path="output")
    
    # Enregistrer des caméras
    cameras = [
        ("cam_devanture", "Caméra Devanture", "Devanture - Porte"),
        ("cam_hall", "Caméra Hall", "Hall - Entrée"),
        ("cam_couloir", "Caméra Couloir", "Couloir - Étage 1"),
    ]
    
    print("\n📝 Enregistrement des caméras...")
    for cam_id, name, location in cameras:
        storage.register_camera(cam_id, name, location, f"rtsp://example/{cam_id}")
        print(f"   ✓ {name}")
    
    # Logger des événements
    print("\n📋 Logging d'événements...")
    storage.log_event("stream_start", "Flux démarré", "cam_devanture")
    storage.log_event("desync", "Désynchronisation: 150ms", "cam_hall", severity="warning")
    storage.log_event("reconnect", "Reconnexion réussie", "cam_couloir")
    
    # Statistiques
    stats = storage.get_storage_statistics()
    print(f"\n📊 Statistiques:")
    print(f"   Vidéos: {stats['total_videos']}")
    print(f"   Caméras: {stats['camera_count']}")
    
    # Export
    export_path = storage.export_metadata_json()
    print(f"\n✓ Export JSON: {export_path}")
    
    storage.close()


def demo_grid_display():
    """Démo: Affichage en grille."""
    print("\n" + "="*60)
    print("🖥️ DÉMO 6: Affichage en Grille Multi-Caméras")
    print("="*60)
    
    dataset_path = "Dataset"
    videos = glob.glob(os.path.join(dataset_path, "*.mp4"))[:4]
    
    if not videos:
        print("❌ Aucune vidéo trouvée")
        return
    
    print(f"\n🎬 Création d'un aperçu de 5 secondes avec {len(videos)} vidéos...")
    
    from multitrack.sync_merger import merge_videos_to_grid
    
    video_dict = {
        f"cam_{i}": path
        for i, path in enumerate(videos)
    }
    
    output = "output/demo_grid.mp4"
    
    merge_videos_to_grid(
        video_dict,
        output,
        target_fps=25.0,
        max_duration=5.0,
        show_progress=True
    )
    
    print(f"\n✓ Vidéo créée: {output}")


def main():
    """Exécute toutes les démos."""
    print("\n" + "#"*60)
    print("#  SYSTÈME DE SURVEILLANCE MULTI-CAMÉRAS - DÉMONSTRATION  #")
    print("#"*60)
    
    # Créer le dossier output
    os.makedirs("output", exist_ok=True)
    
    try:
        # Démo 1: Métadonnées
        metadata_list = demo_metadata_extraction()
        
        # Démo 2: Synchronisation
        if metadata_list:
            demo_synchronization(metadata_list)
        
        # Démo 3: Stream Manager
        demo_stream_manager()
        
        # Démo 4: Health Monitoring
        demo_health_monitoring()
        
        # Démo 5: Data Storage
        demo_data_storage()
        
        # Démo 6: Grid Display
        demo_grid_display()
        
        print("\n" + "="*60)
        print("✅ TOUTES LES DÉMOS TERMINÉES AVEC SUCCÈS")
        print("="*60)
        
        print("""
📁 Fichiers générés:
   • output/metadata_report.json - Rapport des métadonnées
   • output/metadata.db - Base de données SQLite
   • output/demo_grid.mp4 - Vidéo grille de démonstration
   • output/exports/*.json - Exports de métadonnées

🚀 Prochaines étapes:
   • Lancez: python -m multicam live Dataset/*.mp4
   • Ou: python -m multicam --help pour voir toutes les options
        """)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
