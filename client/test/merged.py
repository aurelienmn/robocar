import pandas as pd
import glob
import os

# Dossier où sont tes fichiers lap
DATA_DIR = r"C:\Projet\robocar\data"

# Fichier de sortie
OUTPUT_FILE = os.path.join(DATA_DIR, "driving_data.csv")

# Récupérer tous les fichiers lap_*.csv
files = glob.glob(os.path.join(DATA_DIR, "lap_*.csv"))

print(f"{len(files)} fichiers trouvés")

if not files:
    raise ValueError("Aucun fichier lap trouvé")

# Lire et concaténer
df_list = []
for file in files:
    print(f"Lecture : {file}")
    df = pd.read_csv(file)
    df_list.append(df)

# Fusion
merged_df = pd.concat(df_list, ignore_index=True)

print(f"Total lignes : {len(merged_df)}")

# Sauvegarde
merged_df.to_csv(OUTPUT_FILE, index=False)

print(f"Fichier final créé : {OUTPUT_FILE}")