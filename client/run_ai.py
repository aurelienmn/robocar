import numpy as np
import pandas as pd
import joblib
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

SIMULATOR_PATH = r"C:\Projet\robocar\simulator\RacingSimulator.exe"
CONFIG_PATH = r"C:\Projet\robocar\config\agents.json"

model = joblib.load("steering_model.pkl")

env = UnityEnvironment(
    file_name=SIMULATOR_PATH,
    base_port=5004,
    additional_args=["--config-path", CONFIG_PATH],
    no_graphics=False
)

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

        #  prédiction
        obs_df = pd.DataFrame([obs], columns=[f"ray_{i}" for i in range(50)])
        steering = model.predict(obs_df)[0]

        # throttle fixe
        throttle = 0.6

        actions = np.array([[throttle, steering]], dtype=np.float32)
        env.set_actions(behavior_name, ActionTuple(continuous=actions))

        env.step()

finally:
    env.close()