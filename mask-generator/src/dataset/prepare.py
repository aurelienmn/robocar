"""Consolidate Unity-generated (image, mask) pairs into a clean dataset folder.

The Unity sim writes pairs to two locations using two different naming conventions:

    unity-source/unitySimulator/withFilter/Car{i}_{t}.png       (mask)
    unity-source/unitySimulator/withoutFilter/Car{i}_{t}.png    (image)

    unity-source/unitySimulator/CarScreenshots/Car{i}_Vision_{t}.png    (mask)
    unity-source/unitySimulator/CarScreenshots/Car{i}_NoVision_{t}.png  (image)

Pairs are matched by the {i}_{t} key. Output:

    data/raw/images/pair_000000.png
    data/raw/masks/pair_000000.png
    data/raw/manifest.csv
"""

from __future__ import annotations

import csv
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

# Allow running as `python -m src.dataset.prepare` or `python src/dataset/prepare.py`.
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import load_config


@dataclass
class Pair:
    key: str           # e.g. "Car0_0,1016504" — the {agent}_{timer} identifier
    image: Path        # source camera image
    mask: Path         # source line mask
    source: str        # "withFilter" or "CarScreenshots" — for traceability


def collect_pairs(unity_root: Path) -> list[Pair]:
    """Walk the two known dump locations and pair files by their key."""
    pairs: list[Pair] = []

    # withFilter (masks) and withoutFilter (images) — matched by identical filename.
    wf_dir = unity_root / "withFilter"
    wof_dir = unity_root / "withoutFilter"
    if wf_dir.is_dir() and wof_dir.is_dir():
        for mask_path in sorted(wf_dir.glob("*.png")):
            image_path = wof_dir / mask_path.name
            if image_path.exists():
                key = mask_path.stem  # e.g. "Car0_0,1016504"
                pairs.append(Pair(key=key, image=image_path, mask=mask_path, source="withFilter"))

    # CarScreenshots: Vision_* (masks) and NoVision_* (images) — matched by replacing the tag.
    cs_dir = unity_root / "CarScreenshots"
    if cs_dir.is_dir():
        for mask_path in sorted(cs_dir.glob("*_Vision_*.png")):
            image_path = mask_path.with_name(mask_path.name.replace("_Vision_", "_NoVision_"))
            if image_path.exists():
                # Strip the "_Vision" tag so the key matches the timer-based identity.
                key = mask_path.stem.replace("_Vision", "")
                pairs.append(Pair(key=key, image=image_path, mask=mask_path, source="CarScreenshots"))

    return pairs


def deduplicate(pairs: list[Pair]) -> list[Pair]:
    """Drop pairs that share the same key, keeping the first occurrence."""
    seen: set[str] = set()
    out: list[Pair] = []
    for p in pairs:
        if p.key in seen:
            continue
        seen.add(p.key)
        out.append(p)
    return out


def materialize(pairs: list[Pair], out_root: Path) -> Path:
    """Copy pairs into out_root with stable sequential names. Returns manifest path."""
    images_dir = out_root / "images"
    masks_dir = out_root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_root / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_id", "source", "original_key", "image_src", "mask_src"])
        for idx, p in enumerate(pairs):
            pair_id = f"pair_{idx:06d}"
            shutil.copy2(p.image, images_dir / f"{pair_id}.png")
            shutil.copy2(p.mask, masks_dir / f"{pair_id}.png")
            writer.writerow([pair_id, p.source, p.key, str(p.image), str(p.mask)])

    return manifest_path


def main() -> None:
    cfg = load_config()
    pairs = collect_pairs(cfg.paths.unity_root)
    print(f"Found {len(pairs)} raw pairs across all source folders.")

    deduped = deduplicate(pairs)
    print(f"After de-duplication by key: {len(deduped)} pairs.")
    by_source: dict[str, int] = {}
    for p in deduped:
        by_source[p.source] = by_source.get(p.source, 0) + 1
    for src, n in by_source.items():
        print(f"  {src:20s} {n}")

    if not deduped:
        print("No pairs found. Did you run the Unity sim to dump screenshots?", file=sys.stderr)
        sys.exit(1)

    if cfg.paths.data_raw.exists():
        print(f"Wiping existing {cfg.paths.data_raw} for a clean rebuild.")
        shutil.rmtree(cfg.paths.data_raw)
    manifest = materialize(deduped, cfg.paths.data_raw)
    print(f"Wrote {len(deduped)} pairs to {cfg.paths.data_raw}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
