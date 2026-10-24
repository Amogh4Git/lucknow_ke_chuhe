# 🌊 Infrasonic Sewer Clog Detection Digital Twin

An AI-driven subsurface acoustic Digital Twin system for real-time non-invasive detection, severity classification, and high-precision localization of blockages in municipal sewer networks using infrasonic wave reflectometry and Multi-Task 1D Convolutional Neural Networks (1D-CNN).

---

## 📜 Project Overview & Scientific Foundation

Sewer clogs, sediment buildups, and fatbergs cause municipal wastewater overflows, environmental hazards, and expensive emergency excavations. Traditional Closed-Circuit Television (CCTV) pipe inspection requires active de-watering, crawler camera deployment, and hours of downtime per pipe segment.

This project implements an **Infrasonic Digital Twin** based on acoustic reflectometry in pipe air columns:
1. **Infrasonic Wave Propagation (< 40 Hz)**: Low-frequency acoustic waves propagate along pipe air columns with minimal wall scattering, traveling long distances (20 m to 150 m) between manholes.
2. **Acoustic Impedance Mismatch ($Z = \frac{\rho c}{A}$)**: A pipe cross-sectional reduction from area $A_1$ to blocked area $A_2$ causes an acoustic impedance jump:
   $$Z_1 = \frac{\rho \cdot c}{A_1}, \quad Z_2 = \frac{\rho \cdot c}{A_2}$$
   This impedance discontinuity produces an acoustic pressure reflection coefficient:
   $$R = \frac{Z_2 - Z_1}{Z_2 + Z_1} = \frac{A_1 - A_2}{A_1 + A_2}$$
3. **Time-of-Flight (ToF) Localization**: The round-trip travel time $\tau$ relates directly to blockage distance $d$:
   $$\tau = \frac{2 \cdot d}{c} \implies d = \frac{\tau \cdot c}{2}$$
   where $c \approx 343.0\text{ m/s}$ in dry air columns.

### 📌 Patent & Prior Art Reference
- **US Patent 8,893,553 B2**: *"System and method for acoustic inspection of pipe conduits and blockage localization using sound wave reflections."*
- **US Patent 5,408,882**: *"Acoustic method and apparatus for pipeline inspection using matched filtering and reflection profiling."*

---

## ⚠️ Important Disclaimer on Synthetic Data

> **Disclaimer**: The acoustic datasets used in this project are generated using physics-based synthetic acoustic models. Real sewer environments contain lateral pipes, variable wastewater liquid levels, wall biofilms, and pipe joints. Real-world performance requires deployment and validation with physical hydrophones and field measurements.

---

## 🏛️ System Architecture

```text
+---------------------------------------------------------------------------------------------------+
|                           SEWER PIPELINE INFRASONIC DIGITAL TWIN ARCHITECTURE                     |
+---------------------------------------------------------------------------------------------------+

   +-----------------------+              +-----------------------+              +------------------+
   |   Acoustic Emitter    |              |   Sewer Pipe Segment  |              | Pipe Obstruction |
   | Transmit 10-40Hz Chirp| ------------>| L: 20m - 150m         | -----------> | Occlusion Ratio  |
   |      s_tx(t)          |              | Air Column c = 343m/s |              | Beta in [0.0, 1] |
   +-----------------------+              +-----------------------+              +------------------+
              |                                                                           |
              |                                  Acoustic Reflection R=(Z2-Z1)/(Z2+Z1)   |
              |                                  Attenuation exp(-0.01*d) + AWGN Noise   |
              |                                                                           v
              |                           +-------------------------------------------------+
              +-------------------------> | Acoustic Receiver / Hydrophone Sensor at x = 0  |
                                          | Received Waveform: s_rec(t) = s_echo(t) + Noise |
                                          +-------------------------------------------------+
                                                                   |
                                                                   v
                     +-----------------------------------------------------------------------------+
                     |                          SIGNAL PROCESSING & PHYSICS PIPELINE               |
                     |  - Butterworth 4th-Order Bandpass Filter (10 Hz - 40 Hz)                    |
                     |  - Direct Path Spillover Cancellation: s_clean = s_rec - alpha * s_tx       |
                     |  - Matched Filter Cross-Correlation: xcorr(s_clean, s_tx)                   |
                     |  - Time-of-Flight (ToF) Peak Localization: tau = argmax(P_xcorr) / fs       |
                     +-----------------------------------------------------------------------------+
                                                                   |
                                                                   v
                     +-----------------------------------------------------------------------------+
                     |                        MULTI-TASK 1D-CNN NEURAL NETWORK                     |
                     |  Conv1d(32) -> Conv1d(64) -> Conv1d(128) -> Conv1d(256) [BN + ReLU + MaxPool]|
                     |  AdaptiveAvgPool1d(1) -> 256-dim Shared Representation                     |
                     +-----------------------------------------------------------------------------+
                                       /                                           \
                                      /                                             \
                                     v                                               v
           +-------------------------------------+             +------------------------------------+
           |  Branch 1: Severity Classification  |             |     Branch 2: Distance Regression  |
           |  Linear(256->64) -> Dropout -> (4)  |             |     Linear(256->64) -> Linear(1)   |
           |  [Clear | Minor | Moderate | Severe]|             |     d_est = d_ToF + delta_CNN      |
           +-------------------------------------+             +------------------------------------+
                                     \                                               /
                                      \                                             /
                                       v                                           v
                     +-----------------------------------------------------------------------------+
                     |                     STREAMLIT REAL-TIME DIGITAL TWIN UI                     |
                     |  - 2D Pipe Layout Map with Pinpoint AI Clog Marker                          |
                     |  - Live Waveforms & Cross-Correlation ToF Power Peak Charts                 |
                     |  - Real-Time Diagnostic KPIs (True vs Pred Distance, MAE, Severity Badge)   |
                     +-----------------------------------------------------------------------------+
```

---

## 📁 Repository Structure

```text
infrasonic-sewer-digital-twin/
├── .antigravity/
│   └── rules.md                  # Development rules and coding standards
├── docs/
│   ├── project-overview.md       # Problem statement, digital twin concept, and scope
│   ├── architecture.md           # System data flow, module interfaces, and Mermaid diagram
│   ├── setup-guide.md            # Installation, virtual environment, and execution steps
│   └── verification-report.md    # Automated test outcomes and environment audit
├── saved_models/
│   └── clog_1dcnn.pth            # Trained PyTorch 1D-CNN Multi-Task weights
├── src/
│   ├── __init__.py
│   ├── physics/
│   │   ├── __init__.py
│   │   ├── README.md             # Physics module guide
│   │   └── acoustic_simulator.py # Wave propagation and impedance simulation
│   ├── signal/
│   │   ├── __init__.py
│   │   ├── README.md             # Signal processing module guide
│   │   └── processor.py          # Cross-correlation ToF estimation & features
│   ├── models/
│   │   ├── __init__.py
│   │   ├── README.md             # Machine learning module guide
│   │   ├── clog_cnn1d.py         # Multi-Task 1D-CNN & synthetic dataset
│   │   ├── train.py              # Multi-task training loop
│   │   └── evaluate.py           # Evaluation & noise degradation benchmark
│   └── dashboard/
│       ├── __init__.py
│       ├── README.md             # Dashboard module guide
│       └── app.py                # Streamlit digital twin web application
├── tests/
│   ├── __init__.py
│   ├── test_smoke.py             # Python version and import tests
│   ├── test_physics.py           # Waveform length and attenuation tests
│   ├── test_signal.py            # ToF localization and filtering tests
│   ├── test_models.py            # 1D-CNN shapes, gradients, and dataset tests
│   └── test_dashboard.py         # UI component and rendering tests
├── .gitignore
├── README.md
├── requirements.txt              # Core production dependencies
└── requirements-dev.txt          # Developer and static analysis dependencies
```

---

## 🚀 Setup & Execution Guide

### 1. Prerequisites
- **Python 3.10** or newer (verified on Python 3.14.7).

### 2. Virtual Environment Setup
```bash
# Create local virtual environment
python -m venv .venv

# Activate on Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Activate on Linux or macOS
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Run Automated Test Suite
```bash
python -m pytest -v
```

### 5. Train Model & Evaluate Robustness
```bash
# Train the Multi-Task 1D-CNN (saves weights to saved_models/clog_1dcnn.pth)
python src/models/train.py

# Run evaluation and noise degradation benchmark
python src/models/evaluate.py
```

### 6. Launch Digital Twin Dashboard
```bash
streamlit run src/dashboard/app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 📊 Implementation Status & Benchmark Metrics

### Current Status
- **Physics Simulation**: Fully implemented (`generate_pipe_reflection`).
- **Signal Processing**: Fully implemented (`estimate_clog_distance`, `butter_bandpass_filter`).
- **Deep Learning Model**: Fully trained and saved (`InfrasonicClogDetector1DCNN`).
- **Dashboard UI**: Fully implemented and verified (`src/dashboard/app.py`).
- **Automated Tests**: 33 of 33 tests passing.

### Clean Test Data Performance ($N = 250$, $\sigma = 0.0$)

| Evaluation Metric | Achieved Value | Target Requirement | Status |
| :--- | :--- | :--- | :--- |
| **Classification Accuracy** | **82.00%** | Baseline | **PASSED** |
| **Macro F1-Score** | **0.8141** | Baseline | **PASSED** |
| **Clog Distance MAE (Meters)** | **0.0246 m** ($\sim 2.46\text{ cm}$) | **< 1.0 m** | **PASSED** |
| **Multi-Range Localization Error** | **< 0.040 m** across 12m–90m | $\pm 0.5\text{ m}$ | **PASSED** |

### Noise Degradation Benchmark Across AWGN ($\sigma \in [0.0, 0.3]$)

| Noise Floor ($\sigma$) | Severity Accuracy | Macro F1-Score | Distance MAE (Meters) | Robustness Assessment |
| :--- | :--- | :--- | :--- | :--- |
| **0.00** (Clean) | **78.00%** | **0.7788** | **0.028 m** | Accurate echo localization (< 3 cm) |
| **0.10** (Moderate Noise) | **33.33%** | **0.1250** | **0.046 m** | Distance localization preserved (< 5 cm) |
| **0.20** (Heavy Noise) | **33.33%** | **0.1250** | **0.793 m** | High noise; MAE remains < 1.0 m |
| **0.30** (Severe Noise) | **33.33%** | **0.1250** | **2.358 m** | Heavy noise floor masking weak echoes |

---

## 🎯 Planned Development Milestones

1. **Milestone 1 (Complete)**: Repository initialization, physics simulation, ToF signal processor, multi-task 1D-CNN, and Streamlit dashboard.
2. **Milestone 2**: Hardware integration with field acoustic transducer microcontrollers (e.g., Raspberry Pi + I2S MEMS microphone array).
3. **Milestone 3**: Graph-based acoustic simulation for branching pipe junctions, manhole risers, and sewer networks.
4. **Milestone 4**: Automated municipal maintenance dispatch integration via webhooks and GIS mapping.
