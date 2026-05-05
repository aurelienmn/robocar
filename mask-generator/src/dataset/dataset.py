"""PyTorch Dataset for the line-mask segmentation task."""

from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


def build_train_transform(width: int, height: int) -> A.Compose:
    """Augmentations applied to (image, mask) pairs during training.

    Geometric augmentations are kept conservative — heavy distortion would teach
    the model wrong spatial priors that don't match what the real camera sees.

    Perspective + CoarseDropout were added to specifically address the worst
    failure modes observed in the test set (atypical camera angles, partial
    occlusions). Normalization uses ImageNet stats because the smp encoder is
    pretrained on ImageNet.
    """
    return A.Compose([
        A.PadIfNeeded(min_height=height, min_width=width, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.6),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=10, p=0.4),
        A.GaussNoise(std_range=(0.02, 0.08), p=0.3),
        A.MotionBlur(blur_limit=5, p=0.15),
        A.HorizontalFlip(p=0.5),
        A.Affine(translate_percent={"x": (-0.05, 0.05), "y": (-0.03, 0.03)}, scale=(0.95, 1.05), rotate=(-5, 5), p=0.5),
        # Perspective: simulates different camera mount angles — directly targets
        # the "atypical perspective" failure mode (the IoU 0.69 outlier).
        A.Perspective(scale=(0.04, 0.10), p=0.3),
        # CoarseDropout: simulates partial occlusions (debris, shadows, sensor noise).
        A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(8, 24), hole_width_range=(8, 24), p=0.25),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def build_eval_transform(width: int, height: int) -> A.Compose:
    return A.Compose([
        A.PadIfNeeded(min_height=height, min_width=width, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class LineMaskDataset(Dataset):
    """Lazily loads (image, mask) pairs and applies an Albumentations transform.

    Masks are binarized: any pixel with all RGB channels above mask_threshold
    becomes a foreground (line) pixel. The output mask is a single-channel
    float tensor with values in {0, 1}.
    """

    def __init__(
        self,
        pair_ids: list[str],
        images_dir: Path,
        masks_dir: Path,
        transform: A.Compose,
        mask_threshold: int,
    ) -> None:
        self.pair_ids = pair_ids
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.mask_threshold = mask_threshold

    def __len__(self) -> int:
        return len(self.pair_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pair_id = self.pair_ids[idx]

        image = cv2.imread(str(self.images_dir / f"{pair_id}.png"), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_rgb = cv2.imread(str(self.masks_dir / f"{pair_id}.png"), cv2.IMREAD_COLOR)
        mask_bin = (mask_rgb.min(axis=-1) >= self.mask_threshold).astype(np.uint8)

        transformed = self.transform(image=image, mask=mask_bin)
        x = transformed["image"].float()
        y = transformed["mask"].float().unsqueeze(0)
        return x, y


def split_pair_ids(
    pair_ids: list[str], val_split: float, test_split: float, seed: int
) -> tuple[list[str], list[str], list[str]]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(pair_ids))
    n = len(pair_ids)
    n_test = int(round(n * test_split))
    n_val = int(round(n * val_split))
    test = [pair_ids[i] for i in perm[:n_test]]
    val = [pair_ids[i] for i in perm[n_test : n_test + n_val]]
    train = [pair_ids[i] for i in perm[n_test + n_val :]]
    return train, val, test
