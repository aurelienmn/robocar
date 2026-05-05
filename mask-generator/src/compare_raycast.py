"""Compare two raycasts on the same images:
    A = our pipeline (image → predicted mask → cast_rays)  ← what our code outputs
    B = reference   (ground-truth mask → cast_rays)         ← what the sim would output

If A ≈ B, our code is integration-safe: replacing the simulator's native raycast
by ours wouldn't meaningfully change what the driving AI sees.

Run as: python -m src.compare_raycast
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.api import MaskRaycaster
from src.config import load_config
from src.dataset.dataset import split_pair_ids
from src.raycast.raycast import cast_rays
from src.team_config import load_agent_config


def main() -> None:
    cfg = load_config()
    best = cfg.paths.models / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"No checkpoint at {best}. Train first.")

    # n_rays / fov: same source the team uses (config/agents.json)
    agent = load_agent_config()
    n_rays = agent.n_rays
    fov = agent.fov
    print(f"Comparing on {n_rays} rays at fov={fov}° (from agents.json)")

    raycaster = MaskRaycaster(checkpoint=str(best), device=cfg.train.device)

    images_dir = cfg.paths.data_raw / "images"
    masks_dir = cfg.paths.data_raw / "masks"
    all_ids = sorted(p.stem for p in images_dir.glob("*.png"))
    _, _, test_ids = split_pair_ids(all_ids, cfg.train.val_split, cfg.train.test_split, cfg.train.seed)
    print(f"Test set: {len(test_ids)} images (never seen during training)")

    # For each test image, compare the two raycasts.
    all_errors = []          # absolute differences per ray, flattened across all images
    per_image_mae = []       # mean absolute error per image (one number per image)
    per_image_max_err = []   # worst-ray error per image

    for pair_id in test_ids:
        image_rgb = cv2.cvtColor(cv2.imread(str(images_dir / f"{pair_id}.png")), cv2.COLOR_BGR2RGB)
        gt_mask_rgb = cv2.imread(str(masks_dir / f"{pair_id}.png"))
        gt_mask = gt_mask_rgb.min(axis=-1) >= cfg.mask_threshold

        dist_ours = raycaster.image_to_raycast(image_rgb, n_rays=n_rays, fov=fov)
        dist_ref = cast_rays(gt_mask, n_rays=n_rays, fov=fov)

        diff = np.abs(dist_ours.astype(np.int64) - dist_ref.astype(np.int64))
        all_errors.append(diff)
        per_image_mae.append(diff.mean())
        per_image_max_err.append(diff.max())

    all_errors_arr = np.concatenate(all_errors)
    per_image_mae = np.array(per_image_mae)
    per_image_max_err = np.array(per_image_max_err)

    # ===== Aggregate stats =====
    print("\n" + "=" * 60)
    print("RESULTS — our raycast vs reference raycast")
    print("=" * 60)
    print(f"Total ray comparisons:        {all_errors_arr.size}")
    print(f"Mean absolute error:          {all_errors_arr.mean():.2f} pixels")
    print(f"Median absolute error:        {np.median(all_errors_arr):.2f} pixels")
    print(f"95th percentile error:        {np.percentile(all_errors_arr, 95):.2f} pixels")
    print(f"99th percentile error:        {np.percentile(all_errors_arr, 99):.2f} pixels")
    print(f"Max error (single worst ray): {all_errors_arr.max()} pixels")
    print()
    print(f"Rays within  3 px of reference: {100 * (all_errors_arr <=  3).mean():.1f}%")
    print(f"Rays within  5 px of reference: {100 * (all_errors_arr <=  5).mean():.1f}%")
    print(f"Rays within 10 px of reference: {100 * (all_errors_arr <= 10).mean():.1f}%")
    print()
    print(f"Per-image mean error — best:  {per_image_mae.min():.2f} px ({test_ids[per_image_mae.argmin()]})")
    print(f"Per-image mean error — worst: {per_image_mae.max():.2f} px ({test_ids[per_image_mae.argmax()]})")
    print(f"Per-image mean error — avg:   {per_image_mae.mean():.2f} px ± {per_image_mae.std():.2f}")

    # ===== Visualizations =====
    out_dir = cfg.paths.models / f"compare_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Histogram of all errors
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.arange(0, min(50, all_errors_arr.max()) + 1, 1)
    ax.hist(all_errors_arr, bins=bins, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Absolute error (pixels)")
    ax.set_ylabel("Number of rays")
    ax.set_title(f"Distribution of per-ray errors (n={all_errors_arr.size}, mean={all_errors_arr.mean():.2f} px)")
    ax.axvline(all_errors_arr.mean(), color="red", linestyle="--", label=f"mean={all_errors_arr.mean():.2f}")
    ax.axvline(np.median(all_errors_arr), color="green", linestyle="--", label=f"median={np.median(all_errors_arr):.0f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "error_histogram.png", dpi=120)
    plt.close(fig)

    # 2. Side-by-side raycast curves for 6 representative samples (best, median, worst, + 3 random)
    rng = np.random.default_rng(0)
    interest = [
        ("best",   int(per_image_mae.argmin())),
        ("median", int(np.argsort(per_image_mae)[len(per_image_mae) // 2])),
        ("worst",  int(per_image_mae.argmax())),
    ]
    for k, idx in enumerate(rng.choice(len(test_ids), size=3, replace=False)):
        interest.append((f"random_{k+1}", int(idx)))

    fig, axes = plt.subplots(len(interest), 1, figsize=(10, 2.4 * len(interest)))
    if len(interest) == 1:
        axes = [axes]
    for ax, (label, idx) in zip(axes, interest):
        pair_id = test_ids[idx]
        image_rgb = cv2.cvtColor(cv2.imread(str(images_dir / f"{pair_id}.png")), cv2.COLOR_BGR2RGB)
        gt_mask = cv2.imread(str(masks_dir / f"{pair_id}.png")).min(axis=-1) >= cfg.mask_threshold

        dist_ours = raycaster.image_to_raycast(image_rgb, n_rays=n_rays, fov=fov)
        dist_ref = cast_rays(gt_mask, n_rays=n_rays, fov=fov)

        ax.plot(dist_ref, label="reference (sim raycast on GT mask)", color="black", linewidth=1.6)
        ax.plot(dist_ours, label="ours (CNN mask + our raycast)", color="red", linewidth=1.2, alpha=0.85)
        ax.fill_between(range(n_rays), dist_ref, dist_ours, color="red", alpha=0.15)
        ax.set_title(f"{label}: {pair_id} — mean abs error = {per_image_mae[idx]:.2f} px")
        ax.set_xlabel("ray index")
        ax.set_ylabel("distance (pixels)")
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "raycast_curves.png", dpi=120)
    plt.close(fig)

    print(f"\n→ Histogram   : {out_dir / 'error_histogram.png'}")
    print(f"→ Curves comp.: {out_dir / 'raycast_curves.png'}")

    # ===== Verdict =====
    mae = all_errors_arr.mean()
    pct_within_5 = 100 * (all_errors_arr <= 5).mean()
    print("\n" + "=" * 60)
    if mae < 5 and pct_within_5 > 90:
        print("✓ INTEGRATION-SAFE")
        print(f"  Our raycast tracks the reference within {mae:.1f} px on average.")
        print(f"  {pct_within_5:.0f}% of rays are within 5 px of the reference.")
        print(f"  → Replacing the sim's native raycast by ours should not destabilize the driving AI.")
    elif mae < 15:
        print("△ ACCEPTABLE — review")
        print(f"  Mean error {mae:.1f} px. Most rays are close but some diverge.")
        print(f"  → Coordinate with teammate before final integration.")
    else:
        print("✗ TOO MUCH DRIFT")
        print(f"  Mean error {mae:.1f} px. The two raycasts disagree significantly.")
        print(f"  → Investigate (probably a model issue or a mask preprocessing mismatch).")
    print("=" * 60)


if __name__ == "__main__":
    main()
