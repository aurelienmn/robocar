"""Loss functions for binary segmentation.

BCEDiceLoss combines BCE-with-logits (per-pixel classification signal) with a
soft Dice loss (set-overlap signal). The mix is robust to the heavy class
imbalance typical of line masks (~3% foreground).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1.0) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = (2, 3)
        intersect = (probs * target).sum(dim=dims)
        union = probs.sum(dim=dims) + target.sum(dim=dims)
        dice = (2 * intersect + self.eps) / (union + self.eps)
        return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, pos_weight: float | None = 5.0) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        # pos_weight up-weights the foreground (line) class to fight the ~30:1 imbalance.
        self.pos_weight = torch.tensor(pos_weight) if pos_weight is not None else None
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pos_weight = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
        dice = self.dice(logits, target)
        return self.bce_weight * bce + (1 - self.bce_weight) * dice
