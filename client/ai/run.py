import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

ROOT = Path(__file__).resolve().parent.parent.parent

if sys.platform == "win32":
    SIMULATOR_PATH = str(ROOT / "simulator" / "BuildWindows" / "RacingSimulator.exe")
else:
    SIMULATOR_PATH = str(ROOT / "simulator" / "BuildMac" / "RacingSimulator.app")

CONFIG_PATH = str(ROOT / "config" / "agents.json")
MODEL_PATH = str(ROOT / "client" / "models" / "driving_model.pkl")

model = joblib.load(MODEL_PATH)

env = UnityEnvironment(
    file_name=SIMULATOR_PATH,
    base_port=5004,
    additional_args=["--config-path", CONFIG_PATH],
    no_graphics=False
)

def clamp(value, min_value=-1.0, max_value=1.0):
    return max(min_value, min(max_value, value))

try:
    env.reset()
    print("IA connectée")

    behavior_name = list(env.behavior_specs.keys())[0]

    while True:
        decision_steps, _ = env.get_steps(behavior_name)

        if len(decision_steps) == 0:
            env.step()
            continue

        obs = decision_steps.obs[0][0]

        obs_df = pd.DataFrame([obs], columns=[f"ray_{i}" for i in range(50)])
        pred = model.predict(obs_df)[0]

        throttle = clamp(float(pred[0]), -1.0, 1.0)
        steering = clamp(float(pred[1]), -1.0, 1.0)

        actions = np.array([[throttle, steering]], dtype=np.float32)
        env.set_actions(behavior_name, ActionTuple(continuous=actions))

        env.step()

finally:
    env.close()