from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = str(ROOT / "data" / "training_data.csv")
MODEL_PATH = str(ROOT / "client" / "models" / "driving_model.pkl")

RAY_COUNT = 50

# 1. Charger les données
df = pd.read_csv(DATA_PATH)
print("Lignes chargées :", len(df))

if df.empty:
    raise ValueError("Le CSV est vide.")

# 2. Vérifier les colonnes
required_columns = [f"ray_{i}" for i in range(RAY_COUNT)] + ["throttle", "steering"]

missing = [col for col in required_columns if col not in df.columns]
if missing:
    raise ValueError(f"Colonnes manquantes : {missing}")

# 3. Nettoyage propre
df = df.dropna()

# Garder uniquement les valeurs cohérentes
df = df[
    (df["throttle"].between(-1, 1)) &
    (df["steering"].between(-1, 1))
]

# Ne PAS supprimer les faibles actions :
# throttle faible / steering faible = important pour apprendre à rouler droit
print("Lignes après nettoyage :", len(df))

if len(df) < 100:
    raise ValueError(
        "Pas assez de données pour un bon entraînement. "
        "Il faut idéalement plusieurs centaines ou milliers de lignes propres."
    )

# 4. Entrées / sorties
X = df[[f"ray_{i}" for i in range(RAY_COUNT)]]
y = df[["throttle", "steering"]]

# 5. Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# 6. Pipeline : normalisation + réseau
model = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=0.0007,
        max_iter=1500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=42,
        verbose=True
    ))
])

# 7. Entraînement
print("Entraînement...")
model.fit(X_train, y_train)

# 8. Évaluation
preds = model.predict(X_test)

mse = mean_squared_error(y_test, preds)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("\n--- Résultats globaux ---")
print(f"MSE global : {mse:.6f}")
print(f"MAE global : {mae:.6f}")
print(f"R2 global  : {r2:.6f}")

true_throttle = y_test["throttle"].values
true_steering = y_test["steering"].values

pred_throttle = preds[:, 0]
pred_steering = preds[:, 1]

print("\n--- Détail throttle ---")
print(f"MSE : {mean_squared_error(true_throttle, pred_throttle):.6f}")
print(f"MAE : {mean_absolute_error(true_throttle, pred_throttle):.6f}")

print("\n--- Détail steering ---")
print(f"MSE : {mean_squared_error(true_steering, pred_steering):.6f}")
print(f"MAE : {mean_absolute_error(true_steering, pred_steering):.6f}")

# 9. Quelques prédictions pour vérifier
print("\n--- Exemples prédictions ---")
for i in range(min(10, len(preds))):
    print(
        f"Réel throttle={true_throttle[i]:.3f}, steering={true_steering[i]:.3f} "
        f"=> Prédit throttle={pred_throttle[i]:.3f}, steering={pred_steering[i]:.3f}"
    )

# 10. Sauvegarde
joblib.dump(model, MODEL_PATH)
print(f"\nModèle sauvegardé : {MODEL_PATH}")