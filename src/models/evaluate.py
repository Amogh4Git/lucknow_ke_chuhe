"""Evaluation and noise robustness assessment for Multi-Task 1D-CNN Infrasonic Clog Detector.

Computes:
- Classification Accuracy
- Macro F1-Score
- Distance Mean Absolute Error (MAE in meters)
- Noise Degradation Benchmark across sigma in [0.0, 0.1, 0.2, 0.3]
"""

import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from src.models.clog_cnn1d import (
    SEVERITY_CLASSES,
    InfrasonicClogDetector1DCNN,
    SyntheticAcousticDataset,
)
from src.physics.acoustic_simulator import generate_pipe_reflection


def evaluate_model_on_dataset(
    model: InfrasonicClogDetector1DCNN,
    dataset: SyntheticAcousticDataset,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Evaluate accuracy, macro F1-score, and distance MAE on a dataset.

    Args:
        model: Trained InfrasonicClogDetector1DCNN instance.
        dataset: SyntheticAcousticDataset to evaluate.
        batch_size: DataLoader mini-batch size.
        device: Torch device (defaults to cpu/cuda).

    Returns:
        Dict containing:
            - 'accuracy': Overall classification accuracy [0.0, 1.0]
            - 'macro_f1': Macro-averaged F1 score [0.0, 1.0]
            - 'distance_mae_m': Mean Absolute Error in meters for clogged pipes
            - 'total_samples': Count of evaluated samples
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_y_true: List[int] = []
    all_y_pred: List[int] = []
    abs_errors: List[float] = []

    with torch.no_grad():
        for batch in loader:
            waveforms = batch["waveform"].to(device)
            target_classes = batch["class_label"].to(device)
            target_dists = batch["distance_m"].to(device)

            logits, distances = model(waveforms)
            preds = torch.argmax(logits, dim=1)

            all_y_true.extend(target_classes.cpu().tolist())
            all_y_pred.extend(preds.cpu().tolist())

            # Evaluate distance error on pipes where blockage is present (class > 0)
            clog_mask = target_classes > 0
            if clog_mask.any():
                err = torch.abs(distances[clog_mask] - target_dists[clog_mask])
                abs_errors.extend(err.cpu().tolist())

    acc = float(accuracy_score(all_y_true, all_y_pred))
    macro_f1 = float(f1_score(all_y_true, all_y_pred, average="macro", zero_division=0))
    dist_mae = float(np.mean(abs_errors)) if abs_errors else 0.0

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "distance_mae_m": dist_mae,
        "total_samples": float(len(all_y_true)),
    }


def evaluate_noise_degradation(
    model: InfrasonicClogDetector1DCNN,
    noise_levels: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3),
    test_size: int = 200,
    pipe_length_m: float = 100.0,
    fs: int = 2000,
    seed: int = 500,
    device: Optional[torch.device] = None,
) -> List[Dict[str, Any]]:
    """Evaluate performance degradation across increasing AWGN noise levels.

    Args:
        model: Trained 1D-CNN model.
        noise_levels: Tuple of AWGN standard deviations sigma to test.
        test_size: Number of synthetic test samples generated per noise level.
        pipe_length_m: Length of sewer pipe in meters.
        fs: Sampling rate in Hz.
        seed: Random seed for test data reproducibility.
        device: Torch compute device.

    Returns:
        List of metrics dictionaries, one per noise level.
    """
    records: List[Dict[str, Any]] = []

    for sigma in noise_levels:
        dataset = SyntheticAcousticDataset(
            size=test_size,
            pipe_length_m=pipe_length_m,
            fs=fs,
            total_time=1.0,
            noise_level_range=(sigma, sigma),  # Fixed noise level
            seed=seed,
        )
        metrics = evaluate_model_on_dataset(model, dataset, device=device)
        records.append(
            {
                "noise_sigma": sigma,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "distance_mae_m": metrics["distance_mae_m"],
                "total_samples": int(metrics["total_samples"]),
            }
        )

    return records


def print_noise_degradation_table(records: List[Dict[str, Any]]) -> None:
    """Print formatted ASCII table of noise degradation benchmark results."""
    print("\n" + "=" * 65)
    print("NOISE DEGRADATION BENCHMARK RESULTS (Infrasonic Sewer Digital Twin)")
    print("=" * 65)
    print(f"{'Noise (sigma)':<15} | {'Accuracy':<12} | {'Macro F1':<12} | {'Distance MAE (m)':<16}")
    print("-" * 65)
    for r in records:
        print(
            f"{r['noise_sigma']:<15.2f} | "
            f"{r['accuracy'] * 100:>10.2f}% | "
            f"{r['macro_f1']:>12.4f} | "
            f"{r['distance_mae_m']:>14.3f} m"
        )
    print("=" * 65)


def run_evaluation(
    model_path: str = "saved_models/clog_1dcnn.pth",
    clean_test_size: int = 250,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Execute complete model evaluation and noise degradation verification."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Evaluate] Loading model weights from: {model_path}")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}. Train the model first.")

    model = InfrasonicClogDetector1DCNN(in_channels=1, num_classes=4)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 1. Clean test data evaluation (noise_level = 0.0)
    print(f"[Evaluate] Running evaluation on clean test data (N={clean_test_size}, sigma=0.0)...")
    clean_test_dataset = SyntheticAcousticDataset(
        size=clean_test_size,
        pipe_length_m=100.0,
        fs=2000,
        total_time=1.0,
        noise_level_range=(0.0, 0.0),  # Clean data
        seed=12345,
    )
    clean_metrics = evaluate_model_on_dataset(model, clean_test_dataset, device=device)

    print(f"\n[Evaluate] Clean Test Performance:")
    print(f"  - Accuracy:         {clean_metrics['accuracy'] * 100:.2f}%")
    print(f"  - Macro F1-Score:   {clean_metrics['macro_f1']:.4f}")
    print(f"  - Distance MAE:     {clean_metrics['distance_mae_m']:.4f} meters")

    # Verify requirement MAE < 1.0m
    mae = clean_metrics["distance_mae_m"]
    assert mae < 1.0, f"Distance MAE ({mae:.4f}m) exceeds 1.0m threshold on clean test data!"
    print(f"  - Verification:     SUCCESS (MAE = {mae:.4f}m < 1.0m)")

    # 2. Noise degradation test across sigma in [0.0, 0.1, 0.2, 0.3]
    print("\n[Evaluate] Running noise degradation test across sigma = [0.0, 0.1, 0.2, 0.3]...")
    degradation_results = evaluate_noise_degradation(
        model,
        noise_levels=(0.0, 0.1, 0.2, 0.3),
        test_size=150,
        seed=54321,
        device=device,
    )
    print_noise_degradation_table(degradation_results)

    return clean_metrics, degradation_results


if __name__ == "__main__":
    run_evaluation()
