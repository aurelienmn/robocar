import csv
import os
import numpy as np
import keyboard
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

SIMULATOR_PATH = r"C:\Projet\robocar\simulator\RacingSimulator.exe"
CONFIG_PATH = r"C:\Projet\robocar\config\agents.json"
DATA_PATH = r"C:\Projet\robocar\data\driving_data.csv"

os.makedirs(r"C:\Projet\robocar\data", exist_ok=True)

env = UnityEnvironment(
    file_name=SIMULATOR_PATH,
    base_port=5004,
    additional_args=["--config-path", CONFIG_PATH],
    no_graphics=False
)

header = [f"ray_{i}" for i in range(50)] + ["throttle", "steering"]

file_exists = os.path.exists(DATA_PATH)

try:
    env.reset()
    print("Connexion OK")
    print("Z = avancer | S = reculer | Q = gauche | D = droite | ESC = quitter")

    behavior_name = list(env.behavior_specs.keys())[0]

    with open(DATA_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(header)

        while True:
            decision_steps, _ = env.get_steps(behavior_name)

            if len(decision_steps) == 0:
                env.step()
                continue

            obs = decision_steps.obs[0][0]  # shape (50,)

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