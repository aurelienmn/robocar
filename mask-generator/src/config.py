from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PathsConfig:
    unity_root: Path
    data_raw: Path
    data_processed: Path
    models: Path
    logs: Path


@dataclass(frozen=True)
class ImageConfig:
    unity_width: int
    unity_height: int
    model_width: int
    model_height: int


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int
    num_workers: int
    epochs: int
    lr: float
    weight_decay: float
    val_split: float
    test_split: float
    seed: int
    early_stop_patience: int
    device: str


@dataclass(frozen=True)
class ModelConfig:
    type: str
    in_channels: int
    out_channels: int
    base_filters: int                 # used only when type == "custom_unet"
    encoder_name: str | None          # used only when type == "smp_unet"
    encoder_weights: str | None       # used only when type == "smp_unet"

    def to_factory_dict(self) -> dict:
        """Translate to the dict expected by src.model.factory.build_model()."""
        d: dict = {
            "type": self.type,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
        }
        if self.type == "smp_unet":
            d["encoder_name"] = self.encoder_name
            d["encoder_weights"] = self.encoder_weights
        elif self.type == "custom_unet":
            d["base_filters"] = self.base_filters
        return d


@dataclass(frozen=True)
class RaycastConfig:
    default_n_rays: int
    default_fov: float


@dataclass(frozen=True)
class Config:
    paths: PathsConfig
    image: ImageConfig
    train: TrainConfig
    model: ModelConfig
    raycast: RaycastConfig
    mask_threshold: int


def load_config(path: Path | str = PROJECT_ROOT / "config.yaml") -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)

    paths = PathsConfig(**{k: PROJECT_ROOT / v for k, v in raw["paths"].items()})

    raw_model = raw["model"]
    encoder_weights = raw_model.get("encoder_weights", "imagenet")
    if encoder_weights in (None, "null", "None", ""):
        encoder_weights = None

    return Config(
        paths=paths,
        image=ImageConfig(**raw["image"]),
        train=TrainConfig(**raw["train"]),
        model=ModelConfig(
            type=raw_model.get("type", "smp_unet"),
            in_channels=raw_model.get("in_channels", 3),
            out_channels=raw_model.get("out_channels", 1),
            base_filters=raw_model.get("base_filters", 32),
            encoder_name=raw_model.get("encoder_name", "mobilenet_v2"),
            encoder_weights=encoder_weights,
        ),
        raycast=RaycastConfig(**raw["raycast"]),
        mask_threshold=raw["mask_threshold"],
    )
