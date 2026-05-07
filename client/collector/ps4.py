import argparse
import csv
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pygame
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

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    raise Exception("Aucune manette détectée. Branche ta manette PS4 avant de lancer le script.")

joystick = pygame.joystick.Joystick(0)
joystick.init()

print(f"Manette détectée : {joystick.get_name()}")

env = UnityEnvironment(
    file_name=SIMULATOR_PATH,
    base_port=5004,
    additional_args=["--config-path", CONFIG_PATH],
    no_graphics=False
)

header = [f"ray_{i}" for i in range(50)] + ["throttle", "steering"]


def clamp(value, min_value=-1.0, max_value=1.0):
    return max(min_value, min(max_value, value))


try:
    env.reset()
    print(f"Connexion OK — enregistrement dans {LAP_PATH}")
    print("Contrôles manette PS4 :")
    print("- Stick gauche : direction")
    print("- R2 : avancer")
    print("- L2 : reculer")
    print("- Bouton Options/Start : quitter")

    behavior_name = list(env.behavior_specs.keys())[0]

    with open(LAP_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        while True:
            pygame.event.pump()

            decision_steps, _ = env.get_steps(behavior_name)

            if len(decision_steps) == 0:
                env.step()
                continue

            obs = decision_steps.obs[0][0]

            steering = joystick.get_axis(0)

            l2_raw = joystick.get_axis(4)
            r2_raw = joystick.get_axis(5)

            l2 = (l2_raw + 1) / 2
            r2 = (r2_raw + 1) / 2

            throttle = r2 - l2

            if abs(steering) < 0.08:
                steering = 0.0
            if abs(throttle) < 0.08:
                throttle = 0.0

            steering = clamp(steering)
            throttle = clamp(throttle)

            quit_pressed = False
            for btn in [7, 9]:
                if btn < joystick.get_numbuttons() and joystick.get_button(btn):
                    quit_pressed = True
                    break

            if quit_pressed:
                break

            actions = np.array([[throttle, steering]], dtype=np.float32)
            env.set_actions(behavior_name, ActionTuple(continuous=actions))

            row = list(obs) + [throttle, steering]
            writer.writerow(row)

            env.step()

finally:
    env.close()
    pygame.quit()
    print("Environnement fermé")
