"""Training script for Multi-Task 1D-CNN Infrasonic Clog Detector.

Trains simultaneous 4-class severity classification and distance regression.
Loss: Total Loss = CrossEntropyLoss + 0.05 * MSELoss.
Saves checkpoint to saved_models/clog_1dcnn.pth.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.clog_cnn1d import InfrasonicClogDetector1DCNN, SyntheticAcousticDataset


def train_model(
    num_epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    train_size: int = 1200,
    val_size: int = 300,
    pipe_length_m: float = 100.0,
    fs: int = 2000,
    save_path: str = "saved_models/clog_1dcnn.pth",
    device: Optional[torch.device] = None,
    seed: int = 42,
) -> Tuple[InfrasonicClogDetector1DCNN, List[Dict[str, float]]]:
    """Train the InfrasonicClogDetector1DCNN on synthetic acoustic reflection data.

    Args:
        num_epochs: Number of training epochs (default 10).
        batch_size: Mini-batch size (default 32).
        lr: Adam optimizer learning rate (default 1e-3).
        train_size: Number of training samples synthesized.
        val_size: Number of validation samples synthesized.
        pipe_length_m: Simulated sewer pipe segment length in meters.
        fs: Sampling rate in Hz.
        save_path: Destination path for model weights.
        device: Torch device (defaults to cuda if available else cpu).
        seed: Random seed for dataset reproducibility.

    Returns:
        Tuple of (trained_model, history_records).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Initializing training on device: {device}")

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Prepare datasets and loaders
    print(f"[Train] Generating synthetic acoustic datasets (Train: {train_size}, Val: {val_size})...")
    train_dataset = SyntheticAcousticDataset(
        size=train_size,
        pipe_length_m=pipe_length_m,
        fs=fs,
        total_time=1.0,
        seed=seed,
    )
    val_dataset = SyntheticAcousticDataset(
        size=val_size,
        pipe_length_m=pipe_length_m,
        fs=fs,
        total_time=1.0,
        seed=seed + 9999,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss functions, optimizer
    model = InfrasonicClogDetector1DCNN(in_channels=1, num_classes=4).to(device)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    history: List[Dict[str, float]] = []
    print(f"[Train] Architecture parameters: {model.get_num_parameters():,} trainable weights")
    print(f"[Train] Loss objective: Total Loss = CrossEntropyLoss + 0.05 * MSELoss")
    print("-" * 75)

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_total_loss = 0.0
        train_cls_loss = 0.0
        train_reg_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            waveforms = batch["waveform"].to(device)
            target_classes = batch["class_label"].to(device)
            target_dists = batch["distance_m"].to(device)

            optimizer.zero_grad()
            logits, distances = model(waveforms)

            loss_cls = criterion_cls(logits, target_classes)
            # Evaluate regression loss on samples with true blockage (classes > 0)
            clog_mask = target_classes > 0
            if clog_mask.any():
                loss_reg = criterion_reg(distances[clog_mask], target_dists[clog_mask])
            else:
                loss_reg = torch.tensor(0.0, device=device)

            total_loss = loss_cls + 0.05 * loss_reg
            total_loss.backward()
            optimizer.step()

            train_total_loss += total_loss.item() * len(waveforms)
            train_cls_loss += loss_cls.item() * len(waveforms)
            train_reg_loss += loss_reg.item() * len(waveforms)

            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == target_classes).sum().item()
            train_total += len(waveforms)

        epoch_loss = train_total_loss / train_total
        epoch_cls_loss = train_cls_loss / train_total
        epoch_reg_loss = train_reg_loss / train_total
        epoch_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_total_loss = 0.0
        val_correct = 0
        val_total = 0
        val_abs_errors: List[float] = []

        with torch.no_grad():
            for batch in val_loader:
                waveforms = batch["waveform"].to(device)
                target_classes = batch["class_label"].to(device)
                target_dists = batch["distance_m"].to(device)

                logits, distances = model(waveforms)
                loss_cls = criterion_cls(logits, target_classes)
                clog_mask = target_classes > 0
                if clog_mask.any():
                    loss_reg = criterion_reg(distances[clog_mask], target_dists[clog_mask])
                    abs_err = torch.abs(distances[clog_mask] - target_dists[clog_mask])
                    val_abs_errors.extend(abs_err.cpu().tolist())
                else:
                    loss_reg = torch.tensor(0.0, device=device)

                total_loss = loss_cls + 0.05 * loss_reg
                val_total_loss += total_loss.item() * len(waveforms)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == target_classes).sum().item()
                val_total += len(waveforms)

        val_loss = val_total_loss / val_total
        val_acc = val_correct / val_total
        val_mae = float(np.mean(val_abs_errors)) if val_abs_errors else 0.0

        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss,
                "train_cls_loss": epoch_cls_loss,
                "train_reg_loss": epoch_reg_loss,
                "train_acc": epoch_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_mae": val_mae,
            }
        )

        print(
            f"Epoch [{epoch:2d}/{num_epochs:2d}] | "
            f"Train Loss: {epoch_loss:.4f} (Cls: {epoch_cls_loss:.3f}, Reg: {epoch_reg_loss:.3f}) | "
            f"Train Acc: {epoch_acc * 100:.1f}% | "
            f"Val Acc: {val_acc * 100:.1f}% | "
            f"Val Dist MAE: {val_mae:.3f}m"
        )

    # Save model weights
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print("-" * 75)
    print(f"[Train] Model weights saved successfully to: {save_path}")

    return model, history


if __name__ == "__main__":
    train_model()
