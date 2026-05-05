"""Model factory — choose between custom UNet (from scratch) and smp UNet (pretrained encoder).

The pretrained variant uses a MobileNetV2 encoder pre-trained on ImageNet, which
typically gives +3 to +5 IoU points on small datasets (we have only 334 pairs).
MobileNetV2 was picked over EfficientNet because it uses pure ReLU/Conv ops, which
are well supported by ONNX Runtime and TensorRT for Jetson deployment.
"""

from __future__ import annotations

import torch
from torch import nn

from src.model.unet import UNet as CustomUNet


def build_model(model_config: dict) -> nn.Module:
    """Instantiate a model from its config dict.

    Expected keys:
        type           — "smp_unet" (default) or "custom_unet"
        in_channels    — int (default 3)
        out_channels   — int (default 1)
        For "smp_unet":
            encoder_name     (default "mobilenet_v2")
            encoder_weights  (default "imagenet" or None)
        For "custom_unet":
            base_filters    (default 32)
    """
    model_type = model_config.get("type", "smp_unet")
    in_channels = model_config.get("in_channels", 3)
    out_channels = model_config.get("out_channels", 1)

    if model_type == "smp_unet":
        import segmentation_models_pytorch as smp
        return smp.Unet(
            encoder_name=model_config.get("encoder_name", "mobilenet_v2"),
            encoder_weights=model_config.get("encoder_weights", "imagenet"),
            in_channels=in_channels,
            classes=out_channels,
        )

    if model_type == "custom_unet":
        return CustomUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_filters=model_config.get("base_filters", 32),
        )

    raise ValueError(f"Unknown model type: {model_type!r}")


def model_config_from_yaml(yaml_model: dict) -> dict:
    """Translate the `model:` block of config.yaml into a factory-compatible dict."""
    cfg: dict = {
        "type": yaml_model.get("type", "smp_unet"),
        "in_channels": yaml_model.get("in_channels", 3),
        "out_channels": yaml_model.get("out_channels", 1),
    }
    if cfg["type"] == "smp_unet":
        cfg["encoder_name"] = yaml_model.get("encoder_name", "mobilenet_v2")
        # encoder_weights: pass `None` (Python) when YAML says null/None to skip pretrained
        ew = yaml_model.get("encoder_weights", "imagenet")
        cfg["encoder_weights"] = None if ew in (None, "null", "None", "") else ew
    elif cfg["type"] == "custom_unet":
        cfg["base_filters"] = yaml_model.get("base_filters", 32)
    return cfg
