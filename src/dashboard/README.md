# Digital Twin Dashboard Module (`src/dashboard/`)

## Purpose
This module provides a real-time Streamlit web dashboard. It visualizes the pipeline digital twin, simulated acoustic waveforms, cross-correlation profiles, and AI diagnostic predictions.

## Components
- `app.py`:
  - `load_ai_model()`: Caches and loads trained PyTorch model weights.
  - `plot_acoustic_waveforms()`: Renders live transmitted and received acoustic time-series.
  - `plot_cross_correlation_power()`: Renders matched filter correlation power vs. distance.
  - `plot_2d_pipe_map()`: Renders 2D schematic of the pipe with pinpoint clog markers.
  - `main()`: Orchestrates sidebar controls, digital twin simulation, and KPI metrics.

## Running the Dashboard
Execute from the repository root:
```bash
streamlit run src/dashboard/app.py
```
Open `http://localhost:8501` in your browser.
