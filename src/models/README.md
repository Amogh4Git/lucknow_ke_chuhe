# Machine Learning Models Module (`src/models/`)

## Purpose
This module defines the Multi-Task 1D-CNN neural network architecture, synthetic dataset generator, training loop, and evaluation pipeline.

## Components
- `clog_cnn1d.py`:
  - `InfrasonicClogDetector1DCNN`: 4-block 1D-CNN with shared temporal features, a 4-class classification head, and a distance regression head.
  - `SyntheticAcousticDataset`: PyTorch `Dataset` that generates dynamic acoustic reflections on the fly.
  - `severity_to_class(severity)`: Categorizes continuous severity into `Clear`, `Minor`, `Moderate`, or `Severe`.
- `train.py`:
  - `train_model()`: Multi-task training loop optimizing $\mathcal{L} = \mathcal{L}_{\text{CE}} + 0.05 \cdot \mathcal{L}_{\text{MSE}}$.
  - Saves trained model weights to `saved_models/clog_1dcnn.pth`.
- `evaluate.py`:
  - `evaluate_model_on_dataset()`: Computes classification accuracy, macro F1-score, and distance MAE.
  - `evaluate_noise_degradation()`: Tests model degradation across AWGN noise levels ($\sigma = [0.0, 0.1, 0.2, 0.3]$).

## Inputs and Outputs
- **Inputs**: Acoustic time-series waveforms of shape `(batch_size, 1, seq_len)`.
- **Outputs**: Classification logits of shape `(batch_size, 4)` and scalar distances of shape `(batch_size,)`.
