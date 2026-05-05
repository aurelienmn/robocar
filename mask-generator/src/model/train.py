"""Training loop for the line-mask U-Net.

Run as: python -m src.model.train

Saves the best checkpoint (highest val IoU) to models/best.pt and a tensorboard
log to logs/ run-named by start time.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

# Allow direct execution from the project root.
sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config import load_config
from src.dataset.dataset import (
    LineMaskDataset,
    build_eval_transform,
    build_train_transform,
    split_pair_ids,
)
from src.model.factory import build_model
from src.model.losses import BCEDiceLoss
from src.model.metrics import f1, iou, pixel_accuracy


def pick_device(preferred: str) -> torch.device:
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_loaders(cfg) -> tuple[DataLoader, DataLoader, DataLoader]:
    images_dir = cfg.paths.data_raw / "images"
    masks_dir = cfg.paths.data_raw / "masks"
    pair_ids = sorted(p.stem for p in images_dir.glob("*.png"))
    if not pair_ids:
        raise RuntimeError(
            f"No pairs found in {images_dir}. Run `python -m src.dataset.prepare` first."
        )

    train_ids, val_ids, test_ids = split_pair_ids(
        pair_ids, cfg.train.val_split, cfg.train.test_split, cfg.train.seed
    )
    train_tf = build_train_transform(cfg.image.model_width, cfg.image.model_height)
    eval_tf = build_eval_transform(cfg.image.model_width, cfg.image.model_height)

    def make(ids: list[str], tf, shuffle: bool) -> DataLoader:
        ds = LineMaskDataset(ids, images_dir, masks_dir, tf, cfg.mask_threshold)
        return DataLoader(
            ds,
            batch_size=cfg.train.batch_size,
            shuffle=shuffle,
            num_workers=cfg.train.num_workers,
            pin_memory=False,  # MPS doesn't benefit from pin_memory
            drop_last=shuffle,
        )

    return make(train_ids, train_tf, True), make(val_ids, eval_tf, False), make(test_ids, eval_tf, False)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, loss_fn, device: torch.device) -> dict:
    model.eval()
    totals = {"loss": 0.0, "iou": 0.0, "f1": 0.0, "acc": 0.0}
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        bs = x.size(0)
        totals["loss"] += loss_fn(logits, y).item() * bs
        totals["iou"] += iou(logits, y).item() * bs
        totals["f1"] += f1(logits, y).item() * bs
        totals["acc"] += pixel_accuracy(logits, y).item() * bs
        n += bs
    return {k: v / n for k, v in totals.items()}


def train_one_epoch(model, loader, loss_fn, optimizer, device, writer, epoch) -> float:
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc=f"epoch {epoch:02d}", leave=False)
    for step, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        global_step = epoch * len(loader) + step
        writer.add_scalar("train/loss_step", loss.item(), global_step)
    return running_loss / max(1, len(loader))


def main() -> None:
    cfg = load_config()
    torch.manual_seed(cfg.train.seed)

    device = pick_device(cfg.train.device)
    print(f"Device: {device}")

    train_loader, val_loader, test_loader = make_loaders(cfg)
    print(f"Loaders — train:{len(train_loader.dataset)} val:{len(val_loader.dataset)} test:{len(test_loader.dataset)}")

    model_factory_cfg = cfg.model.to_factory_dict()
    model = build_model(model_factory_cfg).to(device)
    print(f"Model: {cfg.model.type}", end="")
    if cfg.model.type == "smp_unet":
        print(f" (encoder={cfg.model.encoder_name}, pretrained={cfg.model.encoder_weights})")
    else:
        print(f" (base_filters={cfg.model.base_filters})")
    print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
    loss_fn = BCEDiceLoss()

    cfg.paths.models.mkdir(parents=True, exist_ok=True)
    cfg.paths.logs.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=str(cfg.paths.logs / run_name))
    best_path = cfg.paths.models / "best.pt"

    best_val_iou = -1.0
    epochs_without_improvement = 0

    t0 = time.time()
    for epoch in range(cfg.train.epochs):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device, writer, epoch)
        val = evaluate(model, val_loader, loss_fn, device)
        scheduler.step()

        writer.add_scalar("train/loss_epoch", train_loss, epoch)
        for k, v in val.items():
            writer.add_scalar(f"val/{k}", v, epoch)
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)

        improved = val["iou"] > best_val_iou
        marker = " ✓ best" if improved else ""
        print(
            f"[{epoch:02d}] train_loss={train_loss:.4f} | "
            f"val_loss={val['loss']:.4f} iou={val['iou']:.4f} f1={val['f1']:.4f} acc={val['acc']:.4f}{marker}"
        )

        if improved:
            best_val_iou = val["iou"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_iou": val["iou"],
                    "config": {
                        "model": model_factory_cfg,
                        "model_width": cfg.image.model_width,
                        "model_height": cfg.image.model_height,
                    },
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.train.early_stop_patience:
                print(f"Early stopping after {cfg.train.early_stop_patience} epochs without improvement.")
                break

    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed/60:.1f} min. Best val IoU: {best_val_iou:.4f}")

    # Final test on the held-out set using the best checkpoint
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        test = evaluate(model, test_loader, loss_fn, device)
        print(f"Test (best ckpt) — loss={test['loss']:.4f} iou={test['iou']:.4f} f1={test['f1']:.4f} acc={test['acc']:.4f}")
        writer.add_scalar("test/iou", test["iou"], 0)
        writer.add_scalar("test/f1", test["f1"], 0)

    writer.close()


if __name__ == "__main__":
    main()
