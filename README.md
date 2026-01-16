# Projet Multi-Camera Tracking System

Système complet de surveillance et tracking multi-caméras avec ré-identification cross-caméra, détection d'objets, et tracking de suspects.

---

## Structure du Projet

```
PROJET/
├── multiTracking/                    # Module principal de tracking multi-caméras
│   ├── main.py                  # Point d'entrée CLI
│   ├── object_detector.py       # Détection YOLOv8
│   ├── object_tracker.py        # Tracking ByteTrack/BoT-SORT
│   ├── mcmot.py                 # Multi-Camera Multi-Object Tracking
│   ├── mcmot_advanced.py        # MCMOT avec galerie adaptative
│   ├── video_sync.py            # Synchronisation temporelle des vidéos
│   ├── time_sync_player.py      # Lecteur synchronisé temps réel
│   ├── sync_merger.py           # Fusion des flux synchronisés
│   ├── stream_manager.py        # Gestion des flux vidéo
│   ├── parallel_processor.py    # Traitement parallèle GPU
│   ├── preprocessing.py         # Prétraitement des frames
│   ├── detector_3d.py           # Détection 3D avec calibration
│   ├── health_monitor.py        # Monitoring système
│   └── data_storage.py          # Stockage des données
│
├── multicamera_tracking_concatenated.py  # Script standalone pour vidéo concaténée
│                                         # avec détection de suspects
│
├── Dataset/                     # Vidéos sources
└── Rendu_video/                 # Vidéos de sortie
```

---

## Installation

```bash
# Créer environnement virtuel
python -m venv .venv
source .venv/bin/activate

# Installer dépendances
pip install -r multicam/requirements.txt
pip install -r requirements_tracking.txt

# Installer FFmpeg (recommandé)
sudo apt install ffmpeg
```

---

## Commandes Principales

### 1. Analyse des Vidéos

```bash
# Analyser les métadonnées des vidéos
python -m multicam analyze Dataset/*.mp4

# Affichage temps réel
python -m multicam live Dataset/*.mp4
```

### 2. Détection d'Objets

```bash
# Détection basique
python -m multicam detect Dataset/CAMERA_HALL_*.mp4

# Avec tracking
python -m multicam detect Dataset/*.mp4 --tracking --show

# Modèle léger (plus rapide)
python -m multicam detect Dataset/*.mp4 -m n --confidence 0.4
```

### 3. Lecture Synchronisée

```bash
# Lecture temps réel synchronisée
python -m multicam sync-live Dataset/*.mp4

# Lecture accélérée (10x)
python -m multicam sync-live Dataset/*.mp4 --speed 10

# Synchronisation manuelle
python -m multicam sync-live Dataset/*.mp4 --manual-sync
```

### 4. MCMOT (Multi-Camera Multi-Object Tracking)

```bash
# Avec ReID (précis mais lent)
python -m multicam mcmot Dataset/*.mp4

# Sans ReID (rapide)
python -m multicam mcmot Dataset/*.mp4 --no-reid

# Seuil ReID personnalisé
python -m multicam mcmot Dataset/*.mp4 --reid-threshold 0.7
```

### 5. MCMOT Synchronisé

```bash
# Mode complet avec ReID
python -m multicam sync-live Dataset/*.mp4 --manual-sync --mcmot

# Sans ReID (plus fluide)
python -m multicam sync-live Dataset/*.mp4 --manual-sync --mcmot --no-reid

# Enregistrement en mode headless
python -m multicam sync-live Dataset/*.mp4 --manual-sync --mcmot \
    --headless --record output.mp4
```

---

## Tracking sur Vidéo Concaténée avec Détection de Suspects

Le script `multicamera_tracking_concatenated.py` permet le tracking sur une vidéo avec 4 caméras concaténées horizontalement, avec **détection de vols**.

### Fonctionnalités

- Détection YOLOv8 (personnes, sacs, voitures, motos)
- ReID cross-caméra avec OSNet
- **Détection d'objets abandonnés**
- **Détection de vols** (changement de propriétaire d'objet)
- **Tracking de suspects** avec alertes

### Usage

```bash
# Commande complète
python multicamera_tracking_concatenated.py \
    --input reseau_final.mp4 \
    --output output.mp4 \
    --headless \
    --reid-threshold 0.35 \
    --alert-log alerts.log

# Sans ReID (plus rapide)
python multicamera_tracking_concatenated.py \
    --input reseau_final.mp4 --output out.mp4 --no-reid

# Test rapide
python multicamera_tracking_concatenated.py \
    --input reseau_final.mp4 --max-frames 500 --show
```

### Options

| Option | Description | Défaut |
|--------|-------------|--------|
| `--input, -i` | Vidéo d'entrée | **requis** |
| `--output, -o` | Vidéo de sortie | None |
| `--cameras, -c` | Nombre de caméras | 4 |
| `--model, -m` | Modèle YOLO | yolov8m.pt |
| `--no-reid` | Désactiver ReID | False |
| `--reid-threshold` | Seuil similarité | 0.6 |
| `--no-abandoned` | Désactiver détection abandon | False |
| `--abandon-threshold` | Frames avant abandon | 25 |
| `--alert-log` | Fichier alertes | None |
| `--show` | Affichage temps réel | False |
| `--headless` | Mode serveur | False |
| `--max-frames` | Limite frames | None |

---

## Architecture MCMOT

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  Camera 1    │   │  Camera 2    │   │  Camera N    │
│  ByteTrack   │   │  ByteTrack   │   │  ByteTrack   │
│  Local IDs   │   │  Local IDs   │   │  Local IDs   │
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘
       │                  │                  │
       └────────┬─────────┴──────────────────┘
                │
       ┌────────▼────────┐
       │  ReID Features  │ (OSNet/ResNet50)
       │  + BEV Position │ (Homography)
       └────────┬────────┘
                │
       ┌────────▼────────┐
       │ Cross-Camera    │ (Cosine + Hungarian)
       │   Matching      │
       └────────┬────────┘
                │
       ┌────────▼────────┐
       │   Global IDs    │ G1, G2, G3...
       └─────────────────┘
```

---

## Phase 2: Abandoned Object & Suspect Detection System

Le système détecte automatiquement les objets abandonnés et identifie les suspects qui les ramassent.

### Composants Implémentés

| Classe | Description |
|--------|-------------|
| `AbandonedObjectTracker` | Suit les objets portables et détecte quand ils sont abandonnés |
| `SuspectTracker` | Gère la liste des suspects et leur suivi cross-caméra |
| `AlertSystem` | Génère alertes visuelles + log fichier |
| `TrackedObject` | Structure de données pour objets suivis |
| `SuspectInfo` | Information sur un suspect (raison, objet volé, caméras vues) |

### Logique de Détection

#### 1. Détection d'Objet Abandonné

```
Objet stationnaire (< 15px mouvement) 
+ Pas de personne à proximité (< 150px)
+ Pendant 25 frames (1 seconde)
→ OBJET ABANDONNÉ
```

#### 2. Détection de Vol (Changement de Propriétaire)

```
Objet associé à Personne A (propriétaire original)
→ Personne B s'approche et prend l'objet
→ B ≠ A → VOL DÉTECTÉ → B = SUSPECT
```

#### 3. Tracking de Suspect Cross-Caméra

```
Suspect identifié dans CAM1
→ ReID le retrouve dans CAM3
→ ALERTE: "Suspect G5 détecté dans CAM3!"
```

### Événements et Alertes

| Type | Condition | Alerte |
|------|-----------|--------|
| `OBJET_ABANDONNÉ` | Objet seul > 1 sec | "backpack abandonné dans CAM2" |
| `VOL_DÉTECTÉ` | Objet change de propriétaire | "G3 a pris backpack (prop: G1)" |
| `SUSPECT_IDENTIFIÉ` | Personne ramasse objet abandonné | "G3 a ramassé backpack" |
| `SUSPECT_DÉTECTÉ` | Suspect vu dans nouvelle caméra | "⚠ SUSPECT G3 dans CAM4!" |

### Affichage Visuel

- **Personne normale**: Cadre coloré + "G{id}"
- **Suspect**: Cadre **rouge épais** + "⚠ SUSPECT G{id}"
- **Objet abandonné**: Cadre **clignotant rouge/orange** + "⚠ ABANDONNÉ"
- **Bannière d'alerte**: En haut de l'écran pour événements récents

### Paramètres Configurables

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `abandon_threshold` | 25 frames | Temps avant qu'un objet soit considéré abandonné |
| `proximity_threshold` | 150 px | Distance max pour associer objet à personne |
| `movement_threshold` | 15 px | Mouvement max pour être "stationnaire" |

### Classes d'Objets Surveillées

- `backpack` (24) - Sac à dos
- `handbag` (26) - Sac à main  
- `suitcase` (28) - Valise

---

## Dépendances

Voir `multicam/requirements.txt` et `requirements_tracking.txt`:

- OpenCV, NumPy, SciPy
- PyTorch, TorchVision
- Ultralytics (YOLOv8)
- TorchReID (OSNet)
- MediaPipe (optionnel)
- FFmpeg (recommandé)

---

## Auteur

TCHANGANI Teleky Franck Emmanuel
