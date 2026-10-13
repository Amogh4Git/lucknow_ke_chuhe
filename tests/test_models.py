"""Unit tests for the Multi-Task 1D-CNN and Synthetic Acoustic Dataset."""

import torch
from torch.utils.data import DataLoader
import pytest

from src.models.clog_cnn1d import (
    SEVERITY_CLASSES,
    InfrasonicClogDetector1DCNN,
    SyntheticAcousticDataset,
    severity_to_class,
)


def test_dummy_forward_pass_shapes() -> None:
    """Verify dummy forward pass with shape (8, 1, 2000) yields (8, 4) and (8,)."""
    batch_size = 8
    channels = 1
    seq_len = 2000

    model = InfrasonicClogDetector1DCNN(in_channels=channels, num_classes=4)
    x = torch.randn(batch_size, channels, seq_len)

    logits, distances = model(x)

    assert logits.shape == (8, 4), f"Expected logits shape (8, 4), got {logits.shape}"
    assert distances.shape == (8,), f"Expected distances shape (8,), got {distances.shape}"
    assert not torch.isnan(logits).any(), "Logits contain NaN values."
    assert not torch.isnan(distances).any(), "Distances contain NaN values."


def test_variable_input_length_compatibility() -> None:
    """Verify that AdaptiveAvgPool1d accommodates varying sequence lengths."""
    model = InfrasonicClogDetector1DCNN()

    for seq_len in [1200, 2000, 3500]:
        x = torch.randn(4, 1, seq_len)
        logits, distances = model(x)
        assert logits.shape == (4, 4)
        assert distances.shape == (4,)


def test_2d_input_automatic_channel_expansion() -> None:
    """Verify that 2D tensor (batch_size, seq_len) is handled cleanly."""
    model = InfrasonicClogDetector1DCNN()
    x = torch.randn(4, 2000)  # No channel dimension explicitly provided

    logits, distances = model(x)
    assert logits.shape == (4, 4)
    assert distances.shape == (4,)


def test_gradient_flow_multi_task() -> None:
    """Verify that gradients propagate to both heads during backpropagation."""
    model = InfrasonicClogDetector1DCNN()
    x = torch.randn(4, 1, 2000)
    target_classes = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    target_dists = torch.tensor([100.0, 25.0, 45.0, 70.0], dtype=torch.float32)

    logits, distances = model(x)

    loss_cls = torch.nn.functional.cross_entropy(logits, target_classes)
    loss_reg = torch.nn.functional.mse_loss(distances, target_dists)
    total_loss = loss_cls + 0.01 * loss_reg

    total_loss.backward()

    # Check gradients exist in both heads
    for p in model.classification_head.parameters():
        assert p.grad is not None and not torch.isnan(p.grad).any()
    for p in model.regression_head.parameters():
        assert p.grad is not None and not torch.isnan(p.grad).any()
    for p in model.conv_blocks.parameters():
        assert p.grad is not None and not torch.isnan(p.grad).any()


def test_synthetic_acoustic_dataset_and_dataloader() -> None:
    """Verify SyntheticAcousticDataset generates valid batches and supports tuple unpacking."""
    dataset = SyntheticAcousticDataset(size=8, pipe_length_m=80.0, total_time=1.0, seed=123)
    assert len(dataset) == 8

    # Single item checks
    item = dataset[0]
    assert "waveform" in item
    assert "class_label" in item
    assert "distance_m" in item
    assert item["waveform"].shape == (1, 2000)
    assert item["class_label"].item() in [0, 1, 2, 3]

    # Test tuple return mode
    tuple_dataset = SyntheticAcousticDataset(size=4, return_tuple=True, seed=123)
    waveform, class_label, distance_m = tuple_dataset[0]
    assert waveform.shape == (1, 2000)
    assert class_label.item() in [0, 1, 2, 3]
    assert 0.0 <= distance_m.item() <= 100.0

    # Test DataLoader batching
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    assert batch["waveform"].shape == (4, 1, 2000)
    assert batch["class_label"].shape == (4,)
    assert batch["distance_m"].shape == (4,)


def test_severity_to_class_mapping() -> None:
    """Verify severity continuous-to-discrete mapping."""
    assert severity_to_class(0.0) == 0
    assert severity_to_class(0.04) == 0
    assert severity_to_class(0.15) == 1
    assert severity_to_class(0.35) == 1
    assert severity_to_class(0.50) == 2
    assert severity_to_class(0.70) == 2
    assert severity_to_class(0.85) == 3
    assert severity_to_class(1.0) == 3
    assert len(SEVERITY_CLASSES) == 4
