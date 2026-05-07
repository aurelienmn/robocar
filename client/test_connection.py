import sys
from pathlib import Path

from mlagents_envs.environment import UnityEnvironment

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

env.reset()
print("Connexion OK")
print(env.behavior_specs)
env.close()