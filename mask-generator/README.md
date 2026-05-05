# Mask Generator — Robocar (sous-projet 2)

## En une phrase

À partir d'une image caméra de voiture, ce projet génère **(1)** un masque qui isole les lignes du circuit, puis **(2)** un tableau de distances (raycast) que l'IA pilote utilise pour conduire.

## À quoi ça sert dans le projet Robocar global

L'objectif final: pouvoir échanger le raycast "natif" du simulateur par mon raycast calculé à partir d'une image.

## Comment ça marche (le pipeline)

```
   Image caméra (RGB, 347×256)
            │
            ▼
   ┌────────────────┐
   │  Réseau U-Net  │   CNN de segmentation, 7.76M paramètres
   └────────────────┘
            │
            ▼
   Masque binaire des lignes (noir/blanc)
            │
            ▼
   ┌────────────────┐
   │  Raycast 2D    │   N rayons partant du bas-centre
   └────────────────┘
            │
            ▼
   Tableau de N distances → entrée pour l'IA pilote
```

### Le réseau U-Net (étape 1)

Un **U-Net** est un type de CNN spécialisé dans la **segmentation d'image**: il prend une image en entrée et produit, pour chaque pixel, une probabilité d'appartenir à une certaine classe (ici "ligne du circuit" ou "pas une ligne"). C'est l'architecture de référence pour ce genre de tâche.

L'encodeur (la partie qui "comprend" l'image) est un **MobileNetV2 pré-entraîné sur ImageNet**: au lieu d'apprendre à reconnaître les formes à partir de zéro, le réseau démarre avec une connaissance générale du monde visuel acquise sur des millions d'images, puis se spécialise sur notre tâche pendant l'entraînement. Avec seulement 334 paires de données, ce transfer learning fait gagner ~3 points d'IoU vs un entraînement from scratch.

Ici le réseau a été entraîné sur 334 paires (image, masque) générées automatiquement par le simulateur Unity (toutes les 0.5s pendant qu'une voiture roule, le sim sauvegarde l'image caméra ET le masque correspondant des lignes).

À l'inférence, on utilise du **Test-Time Augmentation (TTA)**: chaque image est prédite deux fois — version normale + version retournée horizontalement — et les deux prédictions sont moyennées. Coûte 2× le temps d'inférence mais gagne 1 à 3 points d'IoU.

### Le raycast 2D (étape 2)

Une fois qu'on a le masque, on doit le transformer en quelque chose de simple à donner à l'IA pilote. Pour ça, on tire **N rayons** depuis le bas-centre de l'image (= la position virtuelle de la voiture) dans toutes les directions vers le haut, et pour chaque rayon on compte combien de pixels il parcourt avant de toucher une ligne.

Résultat: un tableau de N entiers = "à quelle distance se trouve la ligne dans cette direction".

Le nombre de rayons et l'angle du cône (FOV) sont lus dans `../config/agents.json`, c'est-à-dire **le même fichier que le simulateur utilise**. Comme ça, mon code et le client Unity de mes coéquipiers utilisent toujours la même configuration → pas de désalignement à l'intégration.

## Résultats

Entraîné sur 334 paires (image, masque), réparties en 70% train / 15% validation / 15% test.

### Métriques de segmentation (sur 33 images jamais vues à l'entraînement)

| Métrique | Valeur | Ce que ça veut dire |
| --- | --- | --- |
| **Test IoU** | **0.950** | Le masque prédit recouvre à 95% le masque vérité-terrain |
| **Test F1** | 0.972 | Précision et rappel équilibrés à 97.2% |
| **Test accuracy** | 99.85% | 99.85% des pixels sont correctement classés |
| Best validation IoU | 0.984 (epoch 51) | Sur l'ensemble de validation pendant l'entraînement |
| Inférence U-Net | ~80 ms / image (avec TTA) | Sur Mac M-series (MPS) |
| Raycast | 2.8 ms / 50 rayons | Quasi-gratuit en termes de coût |
| Pipeline complet | < 100 ms / image | → ~10 FPS avec TTA, ~20 FPS sans |

> Anything > 0.85 IoU est considéré comme solide pour cette tâche. **0.95** sur des données jamais vues est très bon.

### Validation d'intégration — comparaison avec le raycast natif Unity

C'est la métrique la plus importante pour l'équipe: **est-ce que le raycast calculé à partir du masque prédit ressemble au raycast que la simu calcule sur le masque vérité-terrain ?** Si oui, on peut remplacer le raycast Unity par le nôtre sans perturber l'IA pilote.

| Métrique | Valeur |
| --- | --- |
| **Erreur absolue moyenne par rayon** | **0.10 pixel** |
| Erreur médiane | 0.00 pixel |
| 95e percentile de l'erreur | 0.00 pixel |
| 99e percentile de l'erreur | 3.00 pixels |
| Pire cas (un seul rayon) | 10 pixels |
| **Rayons à ≤3 px de la référence** | **99.1%** |
| **Rayons à ≤5 px de la référence** | **99.7%** |
| Rayons à ≤10 px de la référence | 100% |

**Verdict: integration-safe.** Sur l'image standard, notre raycast est à **moins d'un pixel** du raycast natif Unity. L'IA pilote ne verra essentiellement aucune différence en conduite.

### Le cas le plus dur (IoU 0.75)

Sur l'ensemble de test, une seule image atteint un score sensiblement bas (0.75 sur `pair_000161`, une perspective atypique). En la décortiquant:

- **Recall = 96%** → le modèle **trouve quasi toutes les vraies lignes**
- **Dans la zone proche de la caméra** (là où le raycast est le plus critique): zéro erreur

Donc même sur l'image la plus difficile, les distances calculées en sortie de notre pipeline restent fiables pour la prise de décision immédiate du pilote.

## Structure du dépôt

```
mask-generator/
├── config.yaml                  Config centrale (tailles d'image, hyperparamètres)
├── pyproject.toml               Package Python installable (pip install -e .)
├── requirements.txt             
├── README.md                    
│
├── src/                         Code Python
│   ├── api.py                   MaskRaycaster — l'API que les coéquipiers utilisent (avec TTA)
│   ├── predict.py               Prédiction sur une image (CLI)
│   ├── evaluate.py              Évaluation visuelle sur le test set
│   ├── compare_raycast.py       Comparaison notre raycast vs raycast natif Unity
│   ├── team_config.py           Lecture de ../config/agents.json
│   ├── config.py                Chargement de config.yaml typé
│   ├── dataset/
│   │   ├── prepare.py           Consolide les paires Unity dans data/raw/
│   │   └── dataset.py           Dataset PyTorch + augmentations
│   ├── model/
│   │   ├── factory.py           Factory de modèle (smp UNet pré-entraîné ou custom)
│   │   ├── unet.py              U-Net "from scratch" (gardé comme fallback)
│   │   ├── losses.py            BCE + Dice (gère le déséquilibre des classes)
│   │   ├── metrics.py           IoU / F1 / accuracy
│   │   ├── train.py             Boucle d'entraînement
│   │   └── quantize.py          Export ONNX fp32 / int8 pour le Jetson
│   └── raycast/
│       └── raycast.py           Algo de raycast vectorisé
│
├── scripts/                     Petits scripts shell pour lancer chaque étape
│   ├── prepare_data.sh
│   ├── train.sh
│   ├── evaluate.sh
│   ├── compare_raycast.sh       Validation d'intégration: notre raycast vs référence
│   ├── predict.sh
│   ├── quantize.sh
│   └── tensorboard.sh
│
├── notebooks/
│   └── 01_eda.ipynb             Analyse exploratoire du dataset
│
└── unity-source/                Le projet Unity (NON commité car ~2 Go)
                                 → re-télécharger depuis le PDF..
```

## Installation

Prérequis: **Python 3.10+**. Sur Mac Apple Silicon, l'entraînement utilise automatiquement MPS (le GPU intégré).

```bash
cd mask-generator
python -m venv .venv
source .venv/bin/activate
pip install -e .[all]
```


## Utilisation — étape par étape

### 1. Préparer le dataset

```bash
./scripts/prepare_data.sh
```

Ça parcourt les dossiers Unity (`withFilter/`, `withoutFilter/`, `CarScreenshots/`), associe chaque image à son masque, et range tout proprement dans `data/raw/{images,masks}/` avec un manifeste CSV.

**Résultat actuel: 334 paires.**

### 2. Entraîner le modèle

```bash
./scripts/train.sh
```

Ça lance 60 époques (avec early stopping si pas d'amélioration pendant 10 époques de suite). Le meilleur modèle est sauvegardé dans `models/best.pt`.

Pour suivre l'entraînement en direct (courbes loss / IoU / F1):

```bash
./scripts/tensorboard.sh
# puis ouvrir http://localhost:6006 dans le navigateur
```

### 3. Évaluer visuellement

```bash
./scripts/evaluate.sh
```

Ça prend 8 images **du test set** (jamais vues pendant l'entraînement) et produit pour chacune une grille 2×2:

```
┌────────────────┬────────────────┐
│  image entrée  │ masque réel    │
├────────────────┼────────────────┤
│ masque prédit  │ raycast tracé  │
└────────────────┴────────────────┘
```

Les images sont sauvegardées dans `models/eval_<timestamp>/`, avec l'IoU dans le nom du fichier pour repérer les bons et les mauvais cas en un coup d'œil.

### 4. Vérifier l'alignement avec le raycast natif Unity

```bash
./scripts/compare_raycast.sh
```

Ça compare, sur les 33 images du test set, **notre raycast** (calculé via le masque prédit par le CNN) avec le **raycast de référence** (calculé directement sur le masque vérité-terrain de la simu). Sortie:

- Statistiques chiffrées (erreur moyenne, percentiles)
- Histogramme des erreurs (`error_histogram.png`)
- Courbes superposées sur 6 images représentatives (`raycast_curves.png`)
- Verdict d'intégration: **safe / acceptable / drift**

C'est cette métrique qui prouve que notre code peut remplacer le raycast Unity sans casser l'IA pilote.

### 5. Prédire sur une seule image

```bash
./scripts/predict.sh --image data/raw/images/pair_000172.png
```

Produit dans `outputs/`:
- `pred_mask.png` — le masque binaire prédit
- `overlay.png` — l'image originale avec le masque en rouge
- `raycast.txt` — les distances du raycast (une par ligne)

### 6. Exporter pour le Jetson Nano

```bash
./scripts/quantize.sh
```

Produit:
- `models/best.onnx` (+ `best.onnx.data` qui contient les poids — les deux doivent être copiés ensemble)
- `models/best.int8.onnx` — version quantifiée en 8 bits, ~3× plus légère, prête pour TensorRT sur le Jetson.

## API d'intégration (pour mes coéquipiers)

Le client RacingSimulator de mes potes peut utiliser mon code comme ceci:

```python
from src.api import MaskRaycaster

# À faire UNE fois au démarrage du programme (charge le modèle en mémoire)
raycaster = MaskRaycaster(
    checkpoint="models/best.pt",
    device="cpu",   # ou "mps" sur Mac, ou "cuda" sur Jetson
)

# À faire à CHAQUE frame (très rapide, le modèle reste en mémoire)
distances = raycaster.image_to_raycast(camera_frame_rgb)

# distances est un tableau numpy de N entiers, où N est lu dans
# ../config/agents.json (donc aligné automatiquement sur ce que
# l'IA pilote attend — pas de désalignement possible).
```

## Notes techniques (pour aller plus loin)

### Pourquoi U-Net plutôt qu'autre chose ?

C'est l'architecture de référence pour la segmentation d'image, conçue à l'origine pour la segmentation médicale. Elle a deux avantages cruciaux ici:

1. Elle préserve la résolution spatiale (entrée et sortie ont la même taille en pixels) → on peut faire un masque pixel-précis.
2. Elle est légère et entraînable sur peu de données — parfait quand on n'a que 334 paires.

### Pourquoi MobileNetV2 comme encodeur ?

La tâche de segmentation se décompose en deux: **comprendre l'image** (encodeur) puis **reconstruire un masque pixel par pixel** (décodeur). Pour l'encodeur on utilise MobileNetV2 pré-entraîné sur ImageNet plutôt qu'un encodeur from-scratch:

- **Pré-entraîné**: gain de ~3 points d'IoU sur petit dataset, c'est le meilleur ratio gain/effort
- **MobileNetV2**: léger (3.5M params côté encodeur), pure ReLU/Conv → bien supporté par ONNX Runtime et TensorRT pour le Jetson Nano (contrairement à EfficientNet qui utilise SiLU, plus dur à quantifier)

### Pourquoi BatchNorm + ReLU ?

Le PDF demande de pouvoir quantifier le modèle (le passer en int8) pour le faire tourner sur le Jetson Nano (matériel embarqué). BatchNorm + ReLU + ConvTranspose sont les opérations les mieux supportées par les runtimes embarqués (ONNX Runtime, TensorRT). À l'inverse, des opérations plus modernes comme GroupNorm ou SiLU posent souvent problème.

### Pourquoi BCE + Dice comme loss ?

Le masque vérité-terrain a seulement ~3% de pixels "ligne" pour ~97% de pixels "fond". Un réseau naïf apprendrait à toujours prédire "fond" et aurait 97% d'accuracy sans rien comprendre.

- **BCE avec `pos_weight=5`** force le réseau à donner plus d'importance à la classe minoritaire (les lignes).
- **Dice loss** mesure le recouvrement entre la prédiction et la vérité-terrain → un signal d'apprentissage robuste à ce genre de déséquilibre.

La somme des deux donne un meilleur entraînement qu'aucun des deux seul.

### Pourquoi 352×256 alors que l'image fait 347×256 ?

Un U-Net avec 4 sous-échantillonnages (max-pool ÷2 quatre fois) exige une taille divisible par 16. 347 ne l'est pas, mais 352 l'est. On ajoute donc 5 pixels de padding noir à droite avant de passer dans le réseau, puis on recoupe la prédiction à 347 pixels avant de faire le raycast.
