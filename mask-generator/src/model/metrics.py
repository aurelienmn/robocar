"""Segmentation metrics computed on binarized predictions."""

from __future__ import annotations

import torch


def _binarize(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (torch.sigmoid(logits) > threshold).float()


def iou(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = _binarize(logits)
    dims = (2, 3)
    intersect = (pred * target).sum(dim=dims)
    union = pred.sum(dim=dims) + target.sum(dim=dims) - intersect
    return ((intersect + eps) / (union + eps)).mean()


def f1(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = _binarize(logits)
    dims = (2, 3)
    tp = (pred * target).sum(dim=dims)
    fp = (pred * (1 - target)).sum(dim=dims)
    fn = ((1 - pred) * target).sum(dim=dims)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    return (2 * precision * recall / (precision + recall + eps)).mean()


def pixel_accuracy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = _binarize(logits)
    return (pred == target).float().mean()
