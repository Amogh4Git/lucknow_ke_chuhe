"""Machine learning models module for PyTorch 1D-CNN multi-task clog detection and localization."""

from src.models.clog_cnn1d import (
    SEVERITY_CLASSES,
    InfrasonicClogDetector1DCNN,
    SyntheticAcousticDataset,
    severity_to_class,
)

__all__ = [
    "InfrasonicClogDetector1DCNN",
    "SyntheticAcousticDataset",
    "SEVERITY_CLASSES",
    "severity_to_class",
]
