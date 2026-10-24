# Signal Processing Module (`src/signal/`)

## Purpose
This module provides signal conditioning, cross-correlation, and feature extraction for sewer acoustic measurements. It isolates reflection echoes and computes physical Time-of-Flight (ToF) distance estimates.

## Components
- `processor.py`:
  - `butter_bandpass_filter(data, lowcut, highcut, fs, order)`: 4th-order zero-phase Butterworth filter isolating 10 Hz to 40 Hz infrasonic band.
  - `estimate_clog_distance(s_rec, s_tx, fs, speed_of_sound)`: Performs matched filtering, removes zero-lag direct spillover, and locates the echo delay $\tau$.
  - `compute_xcorr_profile(s_rec, s_tx, fs, speed_of_sound)`: Computes full cross-correlation profile with aligned distance and time axes.
  - `extract_acoustic_features(s_rec, s_tx, fs, speed_of_sound)`: Extracts statistical and acoustic features including energy, RMS, and SNR.

## Inputs and Outputs
- **Inputs**: Raw received acoustic signal `s_rec`, reference chirp `s_tx`, sampling frequency `fs`, sound speed `c`.
- **Outputs**: Estimated blockage distance (m), cross-correlation power array, and feature dictionary.

## Import Convention
Always import this module using its full package path:
```python
from src.signal.processor import estimate_clog_distance
```
This prevents naming conflicts with the Python standard-library `signal` module.
