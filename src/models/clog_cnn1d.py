"""Multi-task 1D Convolutional Neural Network for infrasonic sewer clog detection and localization.

Simultaneously predicts:
1. Clog Severity Stage (4-class classification: Clear, Minor, Moderate, Severe)
2. Clog Distance from sensor (regression: scalar distance in meters)
"""

from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from src.physics.acoustic_simulator import generate_pipe_reflection

SEVERITY_CLASSES: List[str] = ["Clear", "Minor", "Moderate", "Severe"]


def severity_to_class(severity: float) -> int:
    """Map continuous blockage severity [0.0, 1.0] to discrete 4-stage severity class index.

    Classes:
        0: Clear    (severity < 0.05)
        1: Minor    (0.05 <= severity <= 0.35)
        2: Moderate (0.35 < severity <= 0.70)
        3: Severe   (severity > 0.70)

    Args:
        severity: Blockage ratio between 0.0 and 1.0.

    Returns:
        Integer class index in [0, 3].
    """
    if severity < 0.05:
        return 0
    elif severity <= 0.35:
        return 1
    elif severity <= 0.70:
        return 2
    else:
        return 3


class InfrasonicClogDetector1DCNN(nn.Module):
    """Multi-task 1D-CNN for simultaneous clog severity classification and distance regression.

    Architecture:
        - 4 Conv1D blocks (Conv1d -> BatchNorm1d -> ReLU -> MaxPool1d)
        - AdaptiveAvgPool1d(1) for arbitrary input sequence length compatibility
        - Shared representation feature vector (256 dims)
        - Head 1 (Classification): Linear -> ReLU -> Dropout -> Linear (4 logits)
        - Head 2 (Regression): Linear -> ReLU -> Linear (1 scalar distance)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        dropout_rate: float = 0.2,
    ) -> None:
        """Initialize the multi-task 1D-CNN network.

        Args:
            in_channels: Number of input signal channels (default 1 for single hydrophone/sensor).
            num_classes: Number of clog severity classification categories (default 4).
            dropout_rate: Dropout regularization probability in classification head.
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # 4 Conv1d blocks with BatchNorm, ReLU, and MaxPool1d
        self.conv_blocks = nn.Sequential(
            # Block 1: receptive field 7, downsampling /2
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            # Block 2: receptive field 5, downsampling /2
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            # Block 3: receptive field 5, downsampling /2
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            # Block 4: receptive field 3, downsampling /2
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

        # AdaptiveAvgPool1d(1) ensures fixed feature size regardless of input sequence length
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Branch 1: Classification Head -> Output 4 logits
        self.classification_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, num_classes),
        )

        # Branch 2: Regression Head -> Output 1 scalar (Distance refinement in meters)
        self.regression_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        # Physics-informed 10 Hz - 40 Hz chirp reference template buffer
        self.fs = 2000
        self.speed_of_sound = 343.0
        t_pulse = np.arange(int(0.2 * self.fs)) / float(self.fs)
        from scipy.signal import chirp as scipy_chirp
        ref_chirp = scipy_chirp(t_pulse, f0=10.0, t1=0.2, f1=40.0, method="linear").astype(np.float32)
        self.register_buffer("chirp_template", torch.tensor(ref_chirp, dtype=torch.float32).view(1, 1, -1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform multi-task forward pass on input acoustic waveform batch.

        Args:
            x: Input tensor of shape (batch_size, channels, seq_len) or (batch_size, seq_len).

        Returns:
            Tuple containing:
                - logits (torch.Tensor): Severity classification logits of shape (batch_size, 4).
                - distances (torch.Tensor): Predicted clog distances in meters of shape (batch_size,).
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim != 3:
            raise ValueError(
                f"Expected 2D or 3D input tensor, got shape {tuple(x.shape)} with ndim={x.ndim}."
            )

        features = self.conv_blocks(x)
        features = self.global_pool(features).flatten(1)  # (batch_size, 256)

        # Classification logits: (batch_size, 4)
        logits = self.classification_head(features)

        # Physics-guided Time-of-Flight matched filtering
        M = self.chirp_template.shape[-1]
        if x.shape[-1] >= M:
            corr = nn.functional.conv1d(x, self.chirp_template, padding=M - 1)
            pos_corr = corr[:, 0, M - 1:].clone()
            if pos_corr.shape[-1] > 6:
                pos_corr[:, :6] = 0.0
            peak_lags = torch.argmax(pos_corr, dim=-1).float()
            tof_dist = (peak_lags / float(self.fs) * self.speed_of_sound) / 2.0
        else:
            tof_dist = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        # Residual regression head output
        reg_delta = self.regression_head(features).view(-1)
        distances = tof_dist + reg_delta

        return logits, distances

    def get_num_parameters(self) -> int:
        """Return the total count of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SyntheticAcousticDataset(Dataset):
    """PyTorch Dataset that dynamically synthesizes acoustic sewer waveforms using acoustic_simulator."""

    def __init__(
        self,
        size: int = 1000,
        pipe_length_m: float = 100.0,
        fs: int = 2000,
        total_time: float = 1.0,
        noise_level_range: Tuple[float, float] = (0.01, 0.03),
        seed: Optional[int] = None,
        return_tuple: bool = False,
    ) -> None:
        """Initialize synthetic acoustic dataset generator.

        Args:
            size: Number of synthetic samples in dataset.
            pipe_length_m: Physical pipe length in meters.
            fs: Acoustic sampling frequency in Hz.
            total_time: Signal duration in seconds (produces fs * total_time samples).
            noise_level_range: Range of AWGN standard deviation (min, max).
            seed: Base random seed for reproducible sample synthesis.
            return_tuple: If True, returns (waveform, class_label, distance_m). Defaults to False (dict).
        """
        self.size = size
        self.pipe_length_m = pipe_length_m
        self.fs = fs
        self.total_time = total_time
        self.noise_level_range = noise_level_range
        self.seed = seed
        self.return_tuple = return_tuple

    def __len__(self) -> int:
        """Return total dataset sample count."""
        return self.size

    def __getitem__(self, idx: int) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Generate a synthetic acoustic sewer reflection sample on the fly.

        Args:
            idx: Sample index in dataset.

        Returns:
            Dictionary containing:
                - 'waveform': Tensor of shape (1, seq_len)
                - 'class_label': LongTensor scalar in [0, 3]
                - 'distance_m': FloatTensor scalar distance in meters
                - 'severity': FloatTensor scalar severity in [0.0, 1.0]
            Or tuple (waveform, class_label, distance_m) if return_tuple=True.
        """
        sample_seed = (self.seed + idx * 7919) if self.seed is not None else None
        rng = np.random.default_rng(sample_seed)

        # Distribute classes uniformly: 0 (Clear), 1 (Minor), 2 (Moderate), 3 (Severe)
        cls_idx = int(rng.integers(0, 4))
        if cls_idx == 0:
            severity = 0.0
            dist = float(self.pipe_length_m)
        elif cls_idx == 1:
            severity = float(rng.uniform(0.05, 0.35))
            dist = float(rng.uniform(5.0, self.pipe_length_m * 0.95))
        elif cls_idx == 2:
            severity = float(rng.uniform(0.35, 0.70))
            dist = float(rng.uniform(5.0, self.pipe_length_m * 0.95))
        else:
            severity = float(rng.uniform(0.70, 1.0))
            dist = float(rng.uniform(5.0, self.pipe_length_m * 0.95))

        noise = float(rng.uniform(self.noise_level_range[0], self.noise_level_range[1]))
        sub_seed = int(rng.integers(0, 1_000_000_000))

        _, _, s_rec, _ = generate_pipe_reflection(
            pipe_length_m=self.pipe_length_m,
            clog_dist_m=dist,
            blockage_severity=severity,
            noise_level=noise,
            fs=self.fs,
            total_time=self.total_time,
            seed=sub_seed,
        )

        waveform = torch.tensor(s_rec, dtype=torch.float32).unsqueeze(0)  # Shape (1, seq_len)
        class_tensor = torch.tensor(cls_idx, dtype=torch.long)
        distance_tensor = torch.tensor(dist, dtype=torch.float32)
        severity_tensor = torch.tensor(severity, dtype=torch.float32)

        if self.return_tuple:
            return waveform, class_tensor, distance_tensor

        return {
            "waveform": waveform,
            "class_label": class_tensor,
            "distance_m": distance_tensor,
            "severity": severity_tensor,
        }
