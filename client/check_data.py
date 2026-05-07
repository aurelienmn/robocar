from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = str(ROOT / "data" / "driving_data.csv")

df = pd.read_csv(DATA_PATH)

print("Aperçu :")
print(df.head())

print("\nShape :", df.shape)
print("\nColonnes :")
print(df.columns.tolist())

print("\nValeurs manquantes :")
print(df.isnull().sum())

print("\nStats throttle :")
print(df["throttle"].describe())

print("\nStats steering :")
print(df["steering"].describe())

print("\nRépartition throttle :")
print(df["throttle"].value_counts().sort_index())

print("\nRépartition steering :")
print(df["steering"].value_counts().sort_index())