"""Read the team's agents.json — the source of truth for raycast parameters.

The team's RacingSimulator uses /robocar/config/agents.json to configure
agent perception. To stay in sync at integration time, our raycast should
default to the same n_rays / fov values defined there.

Format:
    {
      "agents": [
        { "fov": 180, "nbRay": 10 }
      ]
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


# mask-generator/ → robocar/ → robocar/config/agents.json
DEFAULT_AGENTS_JSON = Path(__file__).resolve().parents[2] / "config" / "agents.json"


@dataclass(frozen=True)
class AgentConfig:
    n_rays: int
    fov: float


def load_agent_config(
    agent_index: int = 0,
    agents_json_path: Path | str = DEFAULT_AGENTS_JSON,
) -> AgentConfig:
    """Load the (n_rays, fov) for a given agent from the team's agents.json.

    Parameters
    ----------
    agent_index : int
        Which agent's config to use when multiple are defined. Default 0.
    agents_json_path : Path | str
        Override the agents.json location (mostly for tests).

    Raises
    ------
    FileNotFoundError if the file is missing.
    IndexError       if the agent_index is out of range.
    """
    path = Path(agents_json_path)
    with open(path) as f:
        data = json.load(f)

    agents = data.get("agents", [])
    if not agents:
        raise ValueError(f"No agents defined in {path}")
    if agent_index >= len(agents):
        raise IndexError(f"agent_index={agent_index} but only {len(agents)} agent(s) in {path}")

    cfg = agents[agent_index]
    return AgentConfig(n_rays=int(cfg["nbRay"]), fov=float(cfg["fov"]))
