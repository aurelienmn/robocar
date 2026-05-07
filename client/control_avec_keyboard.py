import sys
from pathlib import Path

import numpy as np
import keyboard
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

ROOT = Path(__file__).resolve().parent.parent

if sys.platform == "win32":
    SIMULATOR_PATH = str(ROOT / "simulator" / "BuildWindows" / "RacingSimulator.exe")
else:
    SIMULATOR_PATH = str(ROOT / "simulator" / "BuildMac" / "RacingSimulator.app")

CONFIG_PATH = str(ROOT / "config" / "agents.json")

env = UnityEnvironment(
    file_name=SIMULATOR_PATH,
    base_port=5004,
    additional_args=["--config-path", CONFIG_PATH],
    no_graphics=False
)

try:
    env.reset()
    print("Connexion OK")
    print("Z = avancer | S = reculer | Q = gauche | D = droite | ESC = quitter")

    behavior_name = list(env.behavior_specs.keys())[0]

    while True:
        decision_steps, _ = env.get_steps(behavior_name)

        if len(decision_steps) == 0:
            env.step()
            continue

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
        env.step()

finally:
    env.close()
    print("Environnement fermé")