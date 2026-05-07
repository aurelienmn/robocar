from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = str(ROOT / "data" / "driving_data.csv")
MODEL_PATH = str(ROOT / "client" / "driving_model.pkl")

# 1. Charger les données
df = pd.read_csv(DATA_PATH)
print("Lignes chargées :", len(df))

if df.empty:
    raise ValueError("Le CSV est vide.")

required_columns = [f"ray_{i}" for i in range(50)] + ["throttle", "steering"]
missing = [col for col in required_columns if col not in df.columns]
if missing:
    raise ValueError(f"Colonnes manquantes : {missing}")

# 2. Nettoyage minimal
df = df.dropna()

# Enlever les lignes où il ne se passe presque rien
df = df[(df["throttle"].abs() > 0.03) | (df["steering"].abs() > 0.03)]

print("Lignes après nettoyage :", len(df))

if len(df) < 100:
    raise ValueError("Pas assez de données pour un bon entraînement. Il faut idéalement plusieurs centaines de lignes.")

# 3. Entrées / sorties
X = df[[f"ray_{i}" for i in range(50)]]
y = df[["throttle", "steering"]]

# 4. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Pipeline : scaler + réseau
model = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=0.0007,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42
    ))
])

# 6. Entraînement
print("Entraînement...")
model.fit(X_train, y_train)

# 7. Évaluation
preds = model.predict(X_test)

mse = mean_squared_error(y_test, preds)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"MSE global : {mse:.6f}")
print(f"MAE global : {mae:.6f}")
print(f"R2 global  : {r2:.6f}")

# Évaluation séparée
true_throttle = y_test["throttle"].values
true_steering = y_test["steering"].values

pred_throttle = preds[:, 0]
pred_steering = preds[:, 1]

print("\n--- Détail ---")
print(f"Throttle MSE : {mean_squared_error(true_throttle, pred_throttle):.6f}")
print(f"Throttle MAE : {mean_absolute_error(true_throttle, pred_throttle):.6f}")
print(f"Steering MSE : {mean_squared_error(true_steering, pred_steering):.6f}")
print(f"Steering MAE : {mean_absolute_error(true_steering, pred_steering):.6f}")

# 8. Sauvegarde
joblib.dump(model, MODEL_PATH)
print(f"Modèle sauvegardé : {MODEL_PATH}")