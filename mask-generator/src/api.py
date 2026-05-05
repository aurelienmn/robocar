"""End-to-end inference API: camera image → 1D raycast distances.

This is the public surface of the Mask Generator sub-project. The teammates'
RacingSimulator client imports this function and uses the returned array as
input to the driving AI, replacing the simulator's built-in raycast.

    from src.api import MaskRaycaster
    raycaster = MaskRaycaster(checkpoint="models/best.pt", device="mps")
    distances = raycaster.image_to_raycast(image, n_rays=50, fov=180)

The class is preferred over a free function because the model load is
expensive (~100ms) and should happen once at process startup, not per frame.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from src.model.factory import build_model
from src.raycast.raycast import cast_rays
from src.team_config import load_agent_config


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class MaskRaycaster:
    """Stateful inference engine. Construct once, call per frame."""

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cpu",
        mask_threshold: float = 0.5,
        use_tta: bool = True,
    ) -> None:
        """Load a trained model and prepare for inference.

        Parameters
        ----------
        checkpoint : path to a .pt file produced by src.model.train.
        device : "cpu" | "mps" | "cuda".
        mask_threshold : sigmoid output > threshold becomes a foreground pixel.
        use_tta : if True, runs each frame twice (original + horizontal flip)
                  and averages predictions. Costs 2x inference time but
                  typically gains +1 to +3 IoU points.
        """
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        self.model_w = cfg["model_width"]
        self.model_h = cfg["model_height"]
        self.device = torch.device(device)
        self.mask_threshold = mask_threshold
        self.use_tta = use_tta

        # Backwards compatibility: old checkpoints stored model fields at the top
        # level; new checkpoints have a nested "model" dict for the factory.
        if "model" in cfg:
            model_cfg = cfg["model"]
        else:
            model_cfg = {
                "type": "custom_unet",
                "in_channels": cfg["in_channels"],
                "out_channels": cfg["out_channels"],
                "base_filters": cfg["base_filters"],
            }

        self.model = build_model(model_cfg)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device).eval()

    def predict_mask(self, image: np.ndarray) -> np.ndarray:
        """Run the U-Net on an RGB image, return a binary mask of the original size.

        Parameters
        ----------
        image : ndarray, shape (H, W, 3), dtype uint8, RGB
            Camera frame as produced by Unity (typically 256 x 347).

        Returns
        -------
        mask : ndarray, shape (H, W), dtype bool
            Binary line mask aligned to the input image.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected RGB image (H,W,3), got shape {image.shape}")

        h_in, w_in = image.shape[:2]
        # Pad on the right + bottom to model dims (constant 0).
        pad_h = max(0, self.model_h - h_in)
        pad_w = max(0, self.model_w - w_in)
        padded = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        # If the input was already larger than model dims, center-crop. Rare in practice.
        padded = padded[: self.model_h, : self.model_w]

        normed = (padded.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        tensor = torch.from_numpy(normed).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits)

            if self.use_tta:
                # Test-time augmentation: predict on the horizontally-flipped image,
                # then flip the prediction back and average. Gains +1-3 IoU at the
                # cost of one extra forward pass.
                tensor_flip = torch.flip(tensor, dims=[3])
                logits_flip = self.model(tensor_flip)
                probs_flip = torch.flip(torch.sigmoid(logits_flip), dims=[3])
                probs = (probs + probs_flip) / 2.0

            probs = probs[0, 0].cpu().numpy()

        binary = probs > self.mask_threshold
        # Crop back to the original image size.
        return binary[:h_in, :w_in]

    def image_to_raycast(
        self,
        image: np.ndarray,
        n_rays: int | None = None,
        fov: float | None = None,
        agent_index: int = 0,
    ) -> np.ndarray:
        """Run the full pipeline: image → predicted mask → raycast distances.

        This is the function the RacingSimulator client should call each frame.

        If `n_rays` or `fov` is None, the value is loaded from the team's
        ../config/agents.json (the same file the simulator uses), guaranteeing
        the output dimension matches what the driving AI expects.
        """
        if n_rays is None or fov is None:
            agent = load_agent_config(agent_index=agent_index)
            n_rays = agent.n_rays if n_rays is None else n_rays
            fov = agent.fov if fov is None else fov

        mask = self.predict_mask(image)
        return cast_rays(mask, n_rays=n_rays, fov=fov)
