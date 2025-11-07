"""Training script for the VAE / tumor evolution model.

This script uses the provided `TumorEvolutionModel` and `GaussianNLLLoss` from
`vae.model` and the dataset loader `VAFDataset` from `vae.datasets`.

Behavior:
- Attempts to load a packed dataset if provided or falls back to a small random
  dummy dataset when none exists.
- Uses DataLoader with batch_size=1 to handle variable-length sequence inputs.
- Runs a small training loop and saves a checkpoint `vae_checkpoint.pth`.
"""
from __future__ import annotations

import argparse
import os
from typing import Optional, Sized

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from model import TumorEvolutionModel, GaussianNLLLoss
import datasets
import eval_utils
from utils import seed_all

import wandb

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--packed-file", type=str, default='/root/data/wja/project/CHESS.cpp/data/chess/train/packed_train_data.npz',
                   help="Path to packed_train_data.npz (optional). If omitted, tries default or falls back to a tiny random dataset.")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-eta-min", type=float, default=0.0,
                   help="Minimum learning rate for cosine annealing schedule.")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Use 1 to avoid padding variable-length sequences.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-steps-per-epoch", type=int, default=-1,
                   help="Limit number of steps per epoch to keep quick runs manageable.")
    p.add_argument("--val-size", type=float, default=0.2,
                   help="validation samples to hold out (default 0.2). If less than 1, interpreted as fraction of dataset.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for initialization.")
    return p.parse_args()


def build_dataloader(packed_file: Optional[str], batch_size: int):
    if packed_file is not None and os.path.exists(packed_file):
        ds = datasets.VAFDataset(packed_file=packed_file, to_torch=True)
    else:
        # try default path used by datasets module
        default_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'chess', 'train', 'packed_train_data.npz')
        if os.path.exists(default_path):
            ds = datasets.VAFDataset(packed_file=default_path, to_torch=True)
        else:
            print("No packed dataset found; using small synthetic dataset for a smoke run.")
            # create tiny synthetic dataset (list-like) using VAFDataset semantics
            import numpy as np

            x_list = [np.random.randn(4, 7).astype(np.float32) for _ in range(16)]
            y_list = np.stack([np.random.randn(100).astype(np.float32) for _ in range(16)], axis=0)
            tmp_path = os.path.join(os.getcwd(), "tmp_train_data.npz")
            np.savez_compressed(tmp_path, x=np.array(x_list, dtype=object), y=y_list, names=np.array([str(i) for i in range(len(x_list))], dtype=object))
            ds = datasets.VAFDataset(packed_file=tmp_path, to_torch=True)

    # Use batch_size=1 to avoid padding; VAFDataset returns (x: (m,7), y: (100,)) tensors
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    return loader


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, opt: torch.optim.Optimizer,
                    loader: DataLoader, device: torch.device, max_steps: int):
    model.train()
    total_loss = 0.0
    steps = 0
    for batch in tqdm(loader,desc="Training in epoch", leave=False):
        # batch is (x_batch, y_batch)
        x_batch, y_batch = batch
        # We expect batch_size to be 1 (variable-length sequences). If batch_size>1,
        # the default collate will fail for variable lengths — this script uses batch_size=1.
        # Unwrap the single sample from the batch dimension.
        if isinstance(x_batch, torch.Tensor) and x_batch.dim() >= 2 and x_batch.size(0) == 1:
            x = x_batch[0].to(device)
        else:
            x = x_batch.to(device)  # fallback

        if isinstance(y_batch, torch.Tensor) and y_batch.dim() >= 2 and y_batch.size(0) == 1:
            y = y_batch[0].to(device)
        else:
            y = y_batch.to(device)

        opt.zero_grad()

        # Forward: model expects a single-sample tensor of shape (m,7) and returns (100,2)
        y_pred = model(x)

        # Criterion expects shapes (batch_size, 100, 2) and (batch_size, 100)
        y_pred_b = y_pred.unsqueeze(0)
        y_b = y.unsqueeze(0)

        loss = criterion(y_pred_b, y_b)
        loss.backward()
        opt.step()

        total_loss += loss.item()
        steps += 1
        if max_steps!=-1 and steps >= max_steps:
            break

    avg_loss = total_loss / max(1, steps)
    return avg_loss


def evaluate_model_on_loader(model: torch.nn.Module, val_loader: Optional[DataLoader], device: torch.device, max_samples: int = 200):
    """Evaluate model by sampling from predicted Gaussian and computing mean L2 distance to ground truth.

    Returns average L2 distance over up to `max_samples` validation samples. If val_loader is None, returns None.
    """
    if val_loader is None:
        return None
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            x_batch, y_batch = batch
            # unwrap single-sample batch
            if isinstance(x_batch, torch.Tensor) and x_batch.size(0) == 1:
                x = x_batch[0].to(device)
            else:
                x = x_batch.to(device)

            if isinstance(y_batch, torch.Tensor) and y_batch.size(0) == 1:
                y = y_batch[0].to(device)
            else:
                y = y_batch.to(device)

            y_pred = model(x)  # (100,2)
            # sample from predicted Gaussian using eval_utils helper
            sampled = eval_utils.sample(y_pred).to(device)

            # compute L2 (Euclidean) distance
            if isinstance(sampled, torch.Tensor) and isinstance(y, torch.Tensor):
                metric = torch.norm(sampled - y).item()
            else:
                # fallback: move both to numpy
                a = sampled.cpu().numpy() if hasattr(sampled, 'cpu') else np.asarray(sampled)
                b = y.cpu().numpy() if hasattr(y, 'cpu') else np.asarray(y)
                metric = float(np.linalg.norm(a - b))

            total += metric
            count += 1
            if count >= max_samples:
                break
    return (total / max(1, count)) if count > 0 else None


def main():
    args = parse_args()
    seed_all(args.seed)
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="zhu_ye-chinese-academy-of-sciences",
        # Set the wandb project where this run will be logged.
        project="cancer-vae",
        # Track hyperparameters and run metadata.
        config={
            "dataset": "build_based_chess_cpp",
            "epochs": args.epochs,
            "seed": args.seed,
        },
    )


    device = torch.device(args.device)

    loader = build_dataloader(args.packed_file, args.batch_size)

    model = TumorEvolutionModel(input_dim=7).to(device)
    criterion = GaussianNLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
        eta_min=args.lr_eta_min,
    )

    # Build train/validation split (simple hold-out from the dataset)
    dataset = loader.dataset
    total_samples = len(dataset) if isinstance(dataset, Sized) else 0

    if args.val_size < 1.0:
        args.val_size = int(args.val_size * total_samples)
    print(f"Using {total_samples - args.val_size} training samples. Using {args.val_size} validation samples.")

    baseline_val = None
    val_indices = []
    if total_samples and args.val_size > 0 and total_samples > 1:
        val_n = min(args.val_size, max(1, int(total_samples // 10))) if total_samples >= args.val_size * 2 else min(args.val_size, total_samples // 2)
        val_n = max(1, val_n)
        val_indices = list(range(0, val_n))
        train_indices = list(range(val_n, total_samples))
        if len(train_indices) == 0:
            # if dataset too small, skip splitting
            train_loader = loader
            val_loader = None
        else:
            train_ds = Subset(dataset, train_indices)
            val_ds = Subset(dataset, val_indices)
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    else:
        train_loader = loader
        val_loader = None

    # If we have a validation subset and the dataset contains the packed y_list,
    # compute a baseline pairwise L2 distance across the validation samples
    if val_loader is not None and hasattr(dataset, 'y_list'):
        try:
            y_all = getattr(dataset, 'y_list')
            # val_indices was created above when splitting; fall back to first N if not
            val_idx = val_indices if val_indices else list(range(min(len(y_all), args.val_size if args.val_size > 0 else len(y_all))))
            if isinstance(y_all, np.ndarray) and len(val_idx) > 1:
                dists = []
                for i in range(len(val_idx)):
                    for j in range(i + 1, len(val_idx)):
                        a = y_all[val_idx[i]]
                        b = y_all[val_idx[j]]
                        try:
                            dists.append(float(np.linalg.norm(a - b)))
                        except Exception:
                            # ensure numeric
                            dists.append(float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))
                baseline_val = float(np.mean(dists)) if dists else 0.0
                print(f"Validation baseline pairwise L2 (mean) over {len(val_idx)} samples: {baseline_val:.6f}")
            else:
                print("Not enough validation samples (or missing y_list) to compute baseline pairwise distance.")
        except Exception as e:
            print(f"Could not compute validation baseline: {e}")

    n_samples = len(dataset) if isinstance(dataset, Sized) else 'unknown'
    train_loader_size = len(train_loader.dataset) if hasattr(train_loader, 'dataset') and isinstance(train_loader.dataset, Sized) else 'unknown'
    print(f"Starting training on device={device} with {n_samples} samples (train loader size: {train_loader_size})")
    baseline_val = baseline_val if baseline_val is not None else 0.0
    for epoch in tqdm(range(1, args.epochs + 1), desc="Training"):
        avg_loss = train_one_epoch(model, criterion, optimizer, train_loader, device, args.max_steps_per_epoch)
        val_metric = evaluate_model_on_loader(model, val_loader, device)
        current_lr = scheduler.get_last_lr()[0]
        if val_metric is None:
            print(f"Epoch {epoch}/{args.epochs}: avg_loss={avg_loss:.6f}  lr={current_lr:.6e}")
        else:
            print(f"Epoch {epoch}/{args.epochs}: avg_loss={avg_loss:.6f}  val_l2={val_metric:.6f}  baseline_val={baseline_val:.6f}  lr={current_lr:.6e}")
            run.log({"epoch": epoch, "lr": current_lr, "avg_loss": avg_loss, "val_l2": val_metric, "baseline_val": baseline_val})
        scheduler.step()
    run.finish()
    ckpt_dir = '/root/data/wja/project/CHESS.cpp/vae/ckpts'
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "vae_checkpoint.pth")
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        },
        ckpt_path,
    )
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    
    main()
