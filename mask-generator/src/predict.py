"""Single-image inference CLI — useful for debugging and demos.

Mirrors the team's expected interface (predict.py + raycast.py from teammates):

    python -m src.predict --image data/raw/images/pair_000172.png

Outputs (in outputs/ unless --out-dir):
    pred_mask.png    — binary mask predicted by the U-Net
    overlay.png      — input image with the mask tinted in red
    raycast.txt      — raycast distances (one int per line), nb_rays + fov
                       defaulted from ../config/agents.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from src.api import MaskRaycaster
from src.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict mask + raycast from a single image.")
    parser.add_argument("--image", required=True, help="Path to the input camera image (PNG).")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path (default: models/best.pt).")
    parser.add_argument("--out-dir", default="outputs", help="Where to write outputs (default: outputs/).")
    parser.add_argument("--device", default=None, help="Override device (mps/cuda/cpu). Default: from config.")
    parser.add_argument("--n-rays", type=int, default=None, help="Override n_rays. Default: from agents.json.")
    parser.add_argument("--fov", type=float, default=None, help="Override fov. Default: from agents.json.")
    parser.add_argument("--agent-index", type=int, default=0, help="Which agent in agents.json to use.")
    args = parser.parse_args()

    cfg = load_config()
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    checkpoint = Path(args.checkpoint) if args.checkpoint else (cfg.paths.models / "best.pt")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}. Train first with scripts/train.sh.")

    device = args.device or cfg.train.device
    print(f"Loading model from {checkpoint} on {device}")
    raycaster = MaskRaycaster(checkpoint=str(checkpoint), device=device)

    print(f"Reading {image_path}")
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Could not decode image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    pred_mask = raycaster.predict_mask(image_rgb)
    distances = raycaster.image_to_raycast(
        image_rgb, n_rays=args.n_rays, fov=args.fov, agent_index=args.agent_index
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Binary mask as PNG
    mask_png = (pred_mask.astype(np.uint8) * 255)
    cv2.imwrite(str(out_dir / "pred_mask.png"), mask_png)

    # 2. Overlay (red tint where the model predicted "line")
    overlay = image_rgb.copy()
    red = np.zeros_like(overlay)
    red[pred_mask] = (255, 0, 0)
    overlay = cv2.addWeighted(overlay, 1.0, red, 0.5, 0)
    cv2.imwrite(str(out_dir / "overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # 3. Raycast distances as plain text
    raycast_path = out_dir / "raycast.txt"
    with raycast_path.open("w") as f:
        f.write(f"# n_rays={len(distances)} fov={args.fov if args.fov is not None else '(from agents.json)'}\n")
        for d in distances:
            f.write(f"{int(d)}\n")

    print(f"  pred_mask  → {out_dir / 'pred_mask.png'}")
    print(f"  overlay    → {out_dir / 'overlay.png'}")
    print(f"  raycast    → {raycast_path} ({len(distances)} distances, range [{distances.min()}, {distances.max()}])")


if __name__ == "__main__":
    main()
