import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import joblib

DATA_PATH = r"C:\Projet\robocar\data\driving_data.csv"

# 1. Charger les données
df = pd.read_csv(DATA_PATH)

# 2. Séparer X / y
X = df[[f"ray_{i}" for i in range(50)]]
y = df["steering"]  #  on commence avec steering uniquement

# 3. Split train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Modèle simple
model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    max_iter=300,
    learning_rate_init=0.001,
)

# 5. Entraînement
print("Training...")
model.fit(X_train, y_train)

# 6. Évaluation
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)

print("MSE:", mse)

# 7. Sauvegarde
joblib.dump(model, "steering_model.pkl")
print("Modèle sauvegardé !")