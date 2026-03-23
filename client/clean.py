import pandas as pd

INPUT = r"C:\Projet\robocar\data\driving_data.csv"
OUTPUT = r"C:\Projet\robocar\data\driving_data_clean.csv"

df = pd.read_csv(INPUT)

ray_cols = [f"ray_{i}" for i in range(50)]

# Exemple :
# imaginons que ray_0 à ray_9 regardent à gauche
# et ray_40 à ray_49 regardent à droite
left_rays = [f"ray_{i}" for i in range(0, 10)]
right_rays = [f"ray_{i}" for i in range(40, 50)]

# Distance minimale acceptable
SAFE_DISTANCE = 0.15

# Garde seulement les lignes où on n'est pas trop proche des bords
df = df[
    (df[left_rays].min(axis=1) > SAFE_DISTANCE) &
    (df[right_rays].min(axis=1) > SAFE_DISTANCE)
]

df.to_csv(OUTPUT, index=False)
print("CSV nettoyé : lignes trop proches des bords supprimées ")