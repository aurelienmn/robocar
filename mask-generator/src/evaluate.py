"""End-to-end visual evaluation of the trained pipeline.

Loads the best checkpoint, runs the full chain (image → predicted mask →
raycast distances), and saves side-by-side visualizations to
models/eval_<timestamp>/. Use this to:

    - sanity-check the model is producing sensible masks
    - compare predicted mask vs ground-truth mask
    - inspect raycast hits visually

Run as: python -m src.evaluate
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from src.api import MaskRaycaster
from src.config import load_config
from src.dataset.dataset import split_pair_ids
from src.raycast.raycast import draw_rays


def overlay_mask(image: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha=0.45) -> np.ndarray:
    """Blend a binary mask over an RGB image."""
    out = image.copy()
    overlay = np.zeros_like(image)
    overlay[mask] = color
    return cv2.addWeighted(out, 1.0, overlay, alpha, 0)


def make_visualization(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    distances: np.ndarray,
    n_rays: int,
    fov: float,
) -> np.ndarray:
    """Compose a 2x2 grid: input | GT mask overlay | predicted overlay | raycast viz."""
    h, w = image.shape[:2]
    panel_image = image.copy()
    panel_gt = overlay_mask(image, gt_mask, color=(0, 255, 0))
    panel_pred = overlay_mask(image, pred_mask, color=(255, 0, 0))
    panel_rays = draw_rays(pred_mask, distances, n_rays=n_rays, fov=fov)

    # Add labels at the top of each panel.
    def label(img: np.ndarray, text: str) -> np.ndarray:
        out = img.copy()
        cv2.rectangle(out, (0, 0), (w, 22), (0, 0, 0), -1)
        cv2.putText(out, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    top = np.concatenate([label(panel_image, "input camera"), label(panel_gt, "ground-truth mask")], axis=1)
    bot = np.concatenate([label(panel_pred, "predicted mask"), label(panel_rays, "raycast on prediction")], axis=1)
    return np.concatenate([top, bot], axis=0)


def main() -> None:
    cfg = load_config()
    best = cfg.paths.models / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"No trained checkpoint at {best}. Run scripts/train.sh first.")

    raycaster = MaskRaycaster(checkpoint=str(best), device=cfg.train.device)

    # CRITICAL: sample only from the TEST split (data the model never saw during training).
    # Reusing the same seed/split as training keeps splits consistent across runs.
    images_dir = cfg.paths.data_raw / "images"
    masks_dir = cfg.paths.data_raw / "masks"
    all_ids = sorted(p.stem for p in images_dir.glob("*.png"))
    _, _, test_ids = split_pair_ids(all_ids, cfg.train.val_split, cfg.train.test_split, cfg.train.seed)
    print(f"Test set has {len(test_ids)} held-out samples (model never saw these).")

    rng = np.random.default_rng(0)
    sample_ids = list(rng.choice(test_ids, size=min(8, len(test_ids)), replace=False))

    out_dir = cfg.paths.models / f"eval_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_rays = cfg.raycast.default_n_rays
    fov = cfg.raycast.default_fov

    print(f"Evaluating on {len(sample_ids)} samples → {out_dir}")
    for pair_id in sample_ids:
        image = cv2.cvtColor(cv2.imread(str(images_dir / f"{pair_id}.png")), cv2.COLOR_BGR2RGB)
        gt_rgb = cv2.imread(str(masks_dir / f"{pair_id}.png"))
        gt_mask = gt_rgb.min(axis=-1) >= cfg.mask_threshold

        pred_mask = raycaster.predict_mask(image)
        distances = raycaster.image_to_raycast(image, n_rays=n_rays, fov=fov)

        # Per-sample IoU as a quick sanity check.
        intersect = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        iou = intersect / max(1, union)

        viz = make_visualization(image, gt_mask, pred_mask, distances, n_rays, fov)
        out_path = out_dir / f"{pair_id}_iou{iou:.2f}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
        print(f"  {pair_id}  IoU={iou:.3f}  raycast min={distances.min()} max={distances.max()} → {out_path.name}")

    print(f"\n✓ Visualizations saved to {out_dir}")


if __name__ == "__main__":
    main()
