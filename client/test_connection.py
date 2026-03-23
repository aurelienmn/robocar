from mlagents_envs.environment import UnityEnvironment

SIMULATOR_PATH = r"C:\Projet\robocar\simulator\RacingSimulator.exe"
CONFIG_PATH = r"C:\Projet\robocar\config\agents.json"

env = UnityEnvironment(
    file_name=SIMULATOR_PATH,
    base_port=5004,
    additional_args=["--config-path", CONFIG_PATH],
    no_graphics=False
)

env.reset()
print("Connexion OK")
print(env.behavior_specs)
env.close()