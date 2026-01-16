"""
Module de monitoring de santé des flux vidéo.
Détecte les pertes de frames, désynchronisations,
et génère des alertes en temps réel.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, List, Any
from enum import Enum
from datetime import datetime, timedelta
from collections import deque
import logging
import statistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Niveaux d'alerte."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class HealthStatus(Enum):
    """États de santé."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


@dataclass
class Alert:
    """Représente une alerte système."""
    timestamp: str
    camera_id: Optional[str]
    level: AlertLevel
    message: str
    metric_name: str
    metric_value: float
    threshold: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "camera_id": self.camera_id,
            "level": self.level.value,
            "message": self.message,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
        }


@dataclass
class CameraHealth:
    """État de santé d'une caméra."""
    camera_id: str
    status: HealthStatus = HealthStatus.OFFLINE
    
    # Métriques de performance
    fps_actual: float = 0.0
    fps_expected: float = 25.0
    fps_history: deque = field(default_factory=lambda: deque(maxlen=60))
    
    # Compteurs
    frames_received: int = 0
    frames_dropped: int = 0
    reconnections: int = 0
    
    # Timestamps
    last_frame_time: Optional[float] = None
    last_healthy_time: Optional[float] = None
    
    # Latence
    latency_ms: float = 0.0
    latency_history: deque = field(default_factory=lambda: deque(maxlen=60))
    
    # Synchronisation
    sync_offset_ms: float = 0.0
    sync_drift_history: deque = field(default_factory=lambda: deque(maxlen=60))
    
    # Scores
    health_score: float = 100.0
    
    def update_fps(self, fps: float):
        """Met à jour le FPS et l'historique."""
        self.fps_actual = fps
        self.fps_history.append(fps)
    
    def get_avg_fps(self) -> float:
        """Retourne le FPS moyen sur la fenêtre."""
        if not self.fps_history:
            return 0.0
        return statistics.mean(self.fps_history)
    
    def get_drop_rate(self) -> float:
        """Retourne le taux de perte de frames."""
        total = self.frames_received + self.frames_dropped
        if total == 0:
            return 0.0
        return self.frames_dropped / total


class HealthMonitor:
    """
    Moniteur de santé pour les flux multi-caméras.
    Détecte les anomalies et génère des alertes.
    """
    
    def __init__(
        self,
        check_interval: float = 1.0,
        on_alert: Optional[Callable[[Alert], None]] = None,
    ):
        """
        Initialise le moniteur.
        
        Args:
            check_interval: Intervalle de vérification en secondes
            on_alert: Callback appelé pour chaque alerte
        """
        self.check_interval = check_interval
        self.on_alert = on_alert
        
        self.cameras: Dict[str, CameraHealth] = {}
        self.alerts: List[Alert] = []
        
        # Seuils configurables
        self.thresholds = {
            "fps_min_ratio": 0.8,           # FPS min = 80% du FPS attendu
            "max_latency_ms": 500.0,        # Latence max
            "max_sync_drift_ms": 100.0,     # Dérive de sync max
            "max_drop_rate": 0.05,          # Taux de perte max (5%)
            "offline_timeout_s": 5.0,       # Timeout pour considérer offline
        }
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
    def add_camera(self, camera_id: str, fps_expected: float = 25.0):
        """Ajoute une caméra au monitoring."""
        with self._lock:
            self.cameras[camera_id] = CameraHealth(
                camera_id=camera_id,
                fps_expected=fps_expected,
            )
        logger.info(f"Monitoring ajouté: {camera_id}")
    
    def remove_camera(self, camera_id: str):
        """Retire une caméra du monitoring."""
        with self._lock:
            if camera_id in self.cameras:
                del self.cameras[camera_id]
    
    def report_frame(
        self,
        camera_id: str,
        fps: float,
        latency_ms: float = 0.0,
        timestamp: Optional[float] = None
    ):
        """
        Rapporte la réception d'une frame.
        Appelé par le stream manager pour chaque frame.
        """
        if timestamp is None:
            timestamp = time.time()
        
        with self._lock:
            if camera_id not in self.cameras:
                self.add_camera(camera_id)
            
            health = self.cameras[camera_id]
            health.frames_received += 1
            health.last_frame_time = timestamp
            health.update_fps(fps)
            
            health.latency_ms = latency_ms
            health.latency_history.append(latency_ms)
            
            # Mettre à jour le statut
            if health.status == HealthStatus.OFFLINE:
                health.status = HealthStatus.HEALTHY
                self._create_alert(
                    camera_id, AlertLevel.INFO,
                    "Flux reconnecté",
                    "status", 1.0, 0.0
                )
    
    def report_drop(self, camera_id: str, count: int = 1):
        """Rapporte des frames perdues."""
        with self._lock:
            if camera_id in self.cameras:
                self.cameras[camera_id].frames_dropped += count
    
    def report_sync_offset(self, camera_id: str, offset_ms: float):
        """Rapporte l'offset de synchronisation."""
        with self._lock:
            if camera_id in self.cameras:
                health = self.cameras[camera_id]
                health.sync_offset_ms = offset_ms
                health.sync_drift_history.append(offset_ms)
    
    def report_reconnection(self, camera_id: str):
        """Rapporte une reconnexion."""
        with self._lock:
            if camera_id in self.cameras:
                self.cameras[camera_id].reconnections += 1
    
    def _calculate_health_score(self, health: CameraHealth) -> float:
        """
        Calcule le score de santé (0-100).
        
        Facteurs:
        - FPS ratio (40%)
        - Drop rate (30%)
        - Latency (15%)
        - Sync drift (15%)
        """
        score = 100.0
        
        # FPS ratio
        if health.fps_expected > 0:
            fps_ratio = health.fps_actual / health.fps_expected
            if fps_ratio < self.thresholds["fps_min_ratio"]:
                penalty = (self.thresholds["fps_min_ratio"] - fps_ratio) * 100
                score -= penalty * 0.4
        
        # Drop rate
        drop_rate = health.get_drop_rate()
        if drop_rate > self.thresholds["max_drop_rate"]:
            penalty = (drop_rate - self.thresholds["max_drop_rate"]) * 200
            score -= min(30, penalty * 0.3)
        
        # Latency
        if health.latency_ms > self.thresholds["max_latency_ms"]:
            penalty = (health.latency_ms - self.thresholds["max_latency_ms"]) / 10
            score -= min(15, penalty * 0.15)
        
        # Sync drift
        if abs(health.sync_offset_ms) > self.thresholds["max_sync_drift_ms"]:
            penalty = (abs(health.sync_offset_ms) - self.thresholds["max_sync_drift_ms"]) / 5
            score -= min(15, penalty * 0.15)
        
        return max(0, min(100, score))
    
    def _determine_status(self, health: CameraHealth) -> HealthStatus:
        """Détermine le statut de santé basé sur le score."""
        if health.last_frame_time is None:
            return HealthStatus.OFFLINE
        
        # Vérifier le timeout
        time_since_frame = time.time() - health.last_frame_time
        if time_since_frame > self.thresholds["offline_timeout_s"]:
            return HealthStatus.OFFLINE
        
        score = health.health_score
        
        if score >= 80:
            return HealthStatus.HEALTHY
        elif score >= 50:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.UNHEALTHY
    
    def _create_alert(
        self,
        camera_id: Optional[str],
        level: AlertLevel,
        message: str,
        metric_name: str,
        metric_value: float,
        threshold: float
    ):
        """Crée et enregistre une alerte."""
        alert = Alert(
            timestamp=datetime.now().isoformat(),
            camera_id=camera_id,
            level=level,
            message=message,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold,
        )
        
        self.alerts.append(alert)
        
        # Garder seulement les 1000 dernières alertes
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]
        
        # Callback
        if self.on_alert:
            self.on_alert(alert)
        
        # Logger
        log_func = getattr(logger, level.value, logger.info)
        log_func(f"[ALERT] {camera_id or 'SYSTEM'}: {message}")
    
    def _check_camera_health(self, camera_id: str, health: CameraHealth):
        """Vérifie la santé d'une caméra et génère les alertes."""
        old_status = health.status
        
        # Calculer le score
        health.health_score = self._calculate_health_score(health)
        health.status = self._determine_status(health)
        
        # Alertes sur changement de statut
        if old_status != health.status:
            if health.status == HealthStatus.OFFLINE:
                self._create_alert(
                    camera_id, AlertLevel.ERROR,
                    "Flux hors ligne - aucune frame reçue",
                    "offline_duration",
                    time.time() - (health.last_frame_time or 0),
                    self.thresholds["offline_timeout_s"]
                )
            elif health.status == HealthStatus.UNHEALTHY:
                self._create_alert(
                    camera_id, AlertLevel.WARNING,
                    f"Flux dégradé - score: {health.health_score:.0f}%",
                    "health_score",
                    health.health_score,
                    50.0
                )
            elif health.status == HealthStatus.HEALTHY and old_status != HealthStatus.HEALTHY:
                health.last_healthy_time = time.time()
        
        # Alertes sur métriques spécifiques
        # FPS trop bas
        fps_ratio = health.fps_actual / health.fps_expected if health.fps_expected > 0 else 0
        if fps_ratio < self.thresholds["fps_min_ratio"] and health.status != HealthStatus.OFFLINE:
            self._create_alert(
                camera_id, AlertLevel.WARNING,
                f"FPS bas: {health.fps_actual:.1f}/{health.fps_expected:.1f}",
                "fps_ratio",
                fps_ratio,
                self.thresholds["fps_min_ratio"]
            )
        
        # Sync drift
        if abs(health.sync_offset_ms) > self.thresholds["max_sync_drift_ms"]:
            self._create_alert(
                camera_id, AlertLevel.WARNING,
                f"Désynchronisation: {health.sync_offset_ms:.1f}ms",
                "sync_drift_ms",
                abs(health.sync_offset_ms),
                self.thresholds["max_sync_drift_ms"]
            )
    
    def _check_global_sync(self):
        """Vérifie la synchronisation globale entre caméras."""
        if len(self.cameras) < 2:
            return
        
        # Calculer la dispersion des offsets
        offsets = [h.sync_offset_ms for h in self.cameras.values() if h.status != HealthStatus.OFFLINE]
        
        if len(offsets) >= 2:
            spread = max(offsets) - min(offsets)
            
            if spread > self.thresholds["max_sync_drift_ms"] * 2:
                self._create_alert(
                    None, AlertLevel.WARNING,
                    f"Désynchronisation globale: {spread:.1f}ms entre caméras",
                    "global_sync_spread",
                    spread,
                    self.thresholds["max_sync_drift_ms"] * 2
                )
    
    def _monitor_loop(self):
        """Boucle principale de monitoring."""
        while self._running:
            with self._lock:
                for camera_id, health in self.cameras.items():
                    self._check_camera_health(camera_id, health)
                
                self._check_global_sync()
            
            time.sleep(self.check_interval)
    
    def start(self):
        """Démarre le monitoring."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("Monitoring démarré")
    
    def stop(self):
        """Arrête le monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("Monitoring arrêté")
    
    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """Retourne le statut de toutes les caméras."""
        with self._lock:
            return {
                camera_id: {
                    "status": health.status.value,
                    "health_score": round(health.health_score, 1),
                    "fps_actual": round(health.fps_actual, 1),
                    "fps_expected": health.fps_expected,
                    "drop_rate": round(health.get_drop_rate() * 100, 2),
                    "latency_ms": round(health.latency_ms, 1),
                    "sync_offset_ms": round(health.sync_offset_ms, 1),
                    "frames_received": health.frames_received,
                    "frames_dropped": health.frames_dropped,
                    "reconnections": health.reconnections,
                }
                for camera_id, health in self.cameras.items()
            }
    
    def get_global_status(self) -> Dict[str, Any]:
        """Retourne le statut global du système."""
        with self._lock:
            statuses = [h.status for h in self.cameras.values()]
            scores = [h.health_score for h in self.cameras.values()]
            
            if not statuses:
                return {"status": "no_cameras", "health_score": 0}
            
            # Le statut global est le pire statut
            if HealthStatus.OFFLINE in statuses:
                global_status = HealthStatus.OFFLINE
            elif HealthStatus.UNHEALTHY in statuses:
                global_status = HealthStatus.UNHEALTHY
            elif HealthStatus.DEGRADED in statuses:
                global_status = HealthStatus.DEGRADED
            else:
                global_status = HealthStatus.HEALTHY
            
            return {
                "status": global_status.value,
                "health_score": round(statistics.mean(scores), 1) if scores else 0,
                "cameras_online": sum(1 for s in statuses if s != HealthStatus.OFFLINE),
                "cameras_total": len(statuses),
                "recent_alerts": len([a for a in self.alerts[-100:] if a.level in [AlertLevel.WARNING, AlertLevel.ERROR]]),
            }
    
    def get_alerts(
        self,
        camera_id: Optional[str] = None,
        level: Optional[AlertLevel] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Récupère les alertes récentes."""
        alerts = self.alerts[-limit:]
        
        if camera_id:
            alerts = [a for a in alerts if a.camera_id == camera_id]
        
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        return [a.to_dict() for a in alerts]
    
    def set_threshold(self, name: str, value: float):
        """Modifie un seuil."""
        if name in self.thresholds:
            self.thresholds[name] = value
            logger.info(f"Seuil modifié: {name} = {value}")
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


if __name__ == "__main__":
    import random
    
    # Callback d'alerte
    def on_alert(alert: Alert):
        print(f"🚨 [{alert.level.value.upper()}] {alert.camera_id}: {alert.message}")
    
    # Créer le moniteur
    monitor = HealthMonitor(check_interval=0.5, on_alert=on_alert)
    
    # Ajouter des caméras
    for i in range(3):
        monitor.add_camera(f"cam_{i}", fps_expected=25.0)
    
    print("\n🔍 Test du monitoring...\n")
    
    with monitor:
        # Simuler des frames
        for _ in range(20):
            for cam_id in ["cam_0", "cam_1", "cam_2"]:
                # Simuler des variations
                fps = 25.0 + random.uniform(-5, 2)
                latency = random.uniform(10, 100)
                
                # Simuler une perte occasionnelle
                if random.random() < 0.1:
                    monitor.report_drop(cam_id)
                else:
                    monitor.report_frame(cam_id, fps, latency)
                
                # Simuler un offset de sync
                monitor.report_sync_offset(cam_id, random.uniform(-50, 50))
            
            time.sleep(0.1)
        
        # Afficher le statut
        print("\n📊 Statut final:")
        for cam_id, status in monitor.get_status().items():
            print(f"  {cam_id}: {status['status']} ({status['health_score']}%)")
        
        global_status = monitor.get_global_status()
        print(f"\n🌐 Global: {global_status['status']} - {global_status['cameras_online']}/{global_status['cameras_total']} en ligne")
