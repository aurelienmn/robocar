import argparse
import glob
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"

parser = argparse.ArgumentParser()
parser.add_argument("--piste", choices=["1", "2", "3", "all"], default="all",
                    help="Piste(s) à fusionner. 'all' = les 3 pistes.")
parser.add_argument("--output", type=str, default=None,
                    help="Chemin du CSV de sortie. Par défaut : data/training_data[_pisteN].csv")
args = parser.parse_args()

if args.piste == "all":
    pistes = ["Piste1", "Piste2", "Piste3"]
    default_out = DATA_DIR / "training_data.csv"
else:
    pistes = [f"Piste{args.piste}"]
    default_out = DATA_DIR / f"training_data_piste{args.piste}.csv"

OUTPUT = Path(args.output) if args.output else default_out

files = []
for piste in pistes:
    run_manuel = DATA_DIR / piste / "run_manuel"
    if not run_manuel.exists():
        print(f"⚠️  {run_manuel} introuvable, ignoré")
        continue
    found = sorted(glob.glob(str(run_manuel / "*.csv")))
    print(f"{piste}: {len(found)} fichiers")
    files.extend(found)

if not files:
    raise ValueError("Aucun fichier trouvé. Vérifie data/Piste*/run_manuel/")

print(f"\n{len(files)} fichiers à fusionner")

df_list = []
for f in files:
    df_list.append(pd.read_csv(f))

merged = pd.concat(df_list, ignore_index=True)
print(f"Total lignes : {len(merged)}")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(OUTPUT, index=False)
print(f"Fichier final : {OUTPUT}")
