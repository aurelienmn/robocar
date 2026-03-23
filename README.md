# 🏎️ RoboCar - Apprentissage par IA pour conduite autonome

Projet d'apprentissage automatique pour entraîner une IA à conduire de manière autonome dans un simulateur de course Unity.

## 📋 Description

Ce projet utilise **Unity ML-Agents** et des techniques d'apprentissage supervisé pour créer une IA capable de piloter une voiture de course. Le système utilise 50 raycasters pour détecter l'environnement et prédit les actions de direction (steering) et d'accélération (throttle).

## 🛠️ Technologies utilisées

- **Python 3.x** avec environnement virtuel
- **Unity ML-Agents** (mlagents-envs 1.1.0)
- **Scikit-learn** pour l'entraînement du modèle (MLPRegressor)
- **Pandas** et **NumPy** pour la manipulation des données
- **Matplotlib** pour la visualisation
- **Keyboard** pour la collecte de données manuelle

## 📁 Structure du projet

```
robocar/
├── client/                      # Scripts Python
│   ├── supervise1.py           # Collecte de données en conduisant manuellement
│   ├── check_data.py           # Analyse des données collectées
│   ├── train_model.py          # Entraînement du modèle ML
│   ├── run_ai.py               # Exécution de l'IA entraînée
│   ├── control_avec_keyboard.py # Contrôle manuel sans collecte
│   ├── test_connection.py      # Test de connexion au simulateur
│   └── requirements.txt        # Dépendances Python
├── simulator/                   # Exécutable Unity
│   └── RacingSimulator.exe
├── config/
│   └── agents.json             # Configuration ML-Agents
└── data/
    └── driving_data.csv        # Données d'entraînement collectées
```

## 🚀 Installation

### 1. Cloner le projet

```bash
git clone <url-du-repo>
cd robocar
```

### 2. Créer un environnement virtuel

```bash
python -m venv .venv
```

### 3. Activer l'environnement virtuel

**Windows (PowerShell):**

```bash
.venv\Scripts\Activate.ps1
```

**Windows (CMD):**

```bash
.venv\Scripts\activate.bat
```

**Linux/Mac:**

```bash
source .venv/bin/activate
```

### 4. Installer les dépendances

```bash
cd client
pip install -r requirements.txt
```

## 📖 Guide d'utilisation

### Étape 1 : Collecte des données

Collectez des données en conduisant manuellement dans le simulateur :

```bash
python supervise1.py
```

**Contrôles:**

- `Z` : Avancer
- `S` : Reculer
- `Q` : Tourner à gauche
- `D` : Tourner à droite
- `ESC` : Quitter

💡 **Conseil:** Collectez au moins 1000-2000 échantillons variés pour un bon entraînement.

### Étape 2 : Vérifier les données

Analysez les données collectées :

```bash
python check_data.py
```

Cette commande affiche :

- Aperçu des données
- Nombre d'échantillons
- Statistiques sur throttle et steering
- Valeurs manquantes

### Étape 3 : Entraîner le modèle

Entraînez le réseau de neurones :

```bash
python train_model.py
```

Le script :

- Charge les données de `data/driving_data.csv`
- Entraîne un réseau de neurones multicouche (MLPRegressor)
- Affiche le MSE (Mean Squared Error)
- Sauvegarde le modèle dans `steering_model.pkl`

### Étape 4 : Tester l'IA

Lancez l'IA pour conduire de manière autonome :

```bash
python run_ai.py
```

L'IA utilise le modèle entraîné pour prédire la direction basée sur les données des raycasters.

### Bonus : Contrôle manuel

Pour tester le simulateur sans collecter de données :

```bash
python control_avec_keyboard.py
```

## 🧠 Architecture du modèle

- **Entrées:** 50 valeurs de raycasters (distances aux obstacles)
- **Architecture:** MLPRegressor avec couches cachées (64, 32)
- **Sortie:** Valeur de steering (-1 à 1)
- **Throttle:** Fixé à 0.6 pendant l'exécution de l'IA

## 📊 Données

Le fichier `driving_data.csv` contient :

- 50 colonnes `ray_0` à `ray_49` : Distances détectées par les raycasters
- `throttle` : Accélération (-1 à 1)
- `steering` : Direction (-1 = gauche, 1 = droite)

## ⚙️ Configuration

### Ports et chemins

Par défaut, le projet utilise :

- **Port Unity:** 5004
- **Chemin simulateur:** `C:\Projet\robocar\simulator\RacingSimulator.exe`
- **Config ML-Agents:** `C:\Projet\robocar\config\agents.json`
- **Données:** `C:\Projet\robocar\data\driving_data.csv`

Modifiez ces chemins dans les scripts si nécessaire.

## 🐛 Dépannage

### Erreur de connexion au simulateur

- Vérifiez que le chemin vers `RacingSimulator.exe` est correct
- Assurez-vous qu'aucun autre processus n'utilise le port 5004
- Vérifiez que le fichier `config/agents.json` existe

### Erreur lors de l'entraînement

- Vérifiez que `driving_data.csv` existe et contient des données
- Assurez-vous d'avoir collecté suffisamment d'échantillons (>100)

### Le modèle conduit mal

- Collectez plus de données variées (virages, lignes droites)
- Augmentez le nombre d'époques dans `train_model.py`
- Ajustez l'architecture du réseau de neurones

## 📝 Améliorations futures

- [ ] Prédire également le throttle avec l'IA
- [ ] Ajouter des techniques d'augmentation de données
- [ ] Implémenter un apprentissage par renforcement (PPO, SAC)
- [ ] Ajouter une visualisation en temps réel des prédictions
- [ ] Optimiser l'architecture du réseau de neurones

## 📄 Licence

Projet personnel - Libre d'utilisation pour l'apprentissage.

## 👤 Auteur

Projet RoboCar - 2026
