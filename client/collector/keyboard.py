import argparse
import csv
import glob
import os
import sys
from pathlib import Path

import numpy as np
import keyboard
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

ROOT = Path(__file__).resolve().parent.parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("--piste", type=int, choices=[1, 2, 3], required=True,
                    help="Numéro de la piste (1, 2 ou 3)")
args = parser.parse_args()

if sys.platform == "win32":
    SIMULATOR_PATH = str(ROOT / "simulator" / "BuildWindows" / "RacingSimulator.exe")
else:
    SIMULATOR_PATH = str(ROOT / "simulator" / "BuildMac" / "RacingSimulator.app")

CONFIG_PATH = str(ROOT / "config" / "agents.json")
DATA_DIR = ROOT / "data" / f"Piste{args.piste}" / "run_manuel"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_next_lap_path():
    existing = glob.glob(str(DATA_DIR / "lap_*.csv"))
    nums = []
    for f in existing:
        name = os.path.basename(f).replace("lap_", "").replace(".csv", "")
        try:
            nums.append(int(name.split("_")[0]))
        except ValueError:
            continue
    next_num = max(nums) + 1 if nums else 1
    return DATA_DIR / f"lap_{next_num:04d}.csv"


LAP_PATH = get_next_lap_path()

env = UnityEnvironment(
    file_name=SIMULATOR_PATH,
    base_port=5004,
    additional_args=["--config-path", CONFIG_PATH],
    no_graphics=False
)

header = [f"ray_{i}" for i in range(50)] + ["throttle", "steering"]

try:
    env.reset()
    print(f"Connexion OK — enregistrement dans {LAP_PATH}")
    print("Z = avancer | S = reculer | Q = gauche | D = droite | ESC = quitter")

    behavior_name = list(env.behavior_specs.keys())[0]

    with open(LAP_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        while True:
            decision_steps, _ = env.get_steps(behavior_name)

            if len(decision_steps) == 0:
                env.step()
                continue

            obs = decision_steps.obs[0][0]

            throttle = 0.0
            steering = 0.0

            if keyboard.is_pressed("z"):
                throttle = 1.0
            elif keyboard.is_pressed("s"):
                throttle = -1.0

            if keyboard.is_pressed("q"):
                steering = -1.0
            elif keyboard.is_pressed("d"):
                steering = 1.0

            if keyboard.is_pressed("esc"):
                break

            actions = np.array([[throttle, steering]], dtype=np.float32)
            env.set_actions(behavior_name, ActionTuple(continuous=actions))

            row = list(obs) + [throttle, steering]
            writer.writerow(row)

            env.step()

finally:
    env.close()
    print("Environnement fermé")
