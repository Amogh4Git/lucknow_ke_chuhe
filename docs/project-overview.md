# Project Overview: AI-Based Infrasonic Sewer Clog Detection Digital Twin

## 1. Problem Statement
Municipal sewer networks suffer from frequent obstructions caused by fatbergs, sediment accumulation, and collapsed infrastructure. Undetected blockages cause wastewater overflows, public health hazards, and costly emergency repairs.

Current sewer inspection relies primarily on Closed-Circuit Television (CCTV) crawler cameras. CCTV inspection requires:
- Active bypass pumping and pipe cleaning.
- Manual crawler deployment through manholes.
- Significant labor, high operational costs, and hours of operational downtime.

CCTV cannot provide continuous or rapid preliminary screening of long pipeline segments.

## 2. Infrasonic Acoustic Monitoring
Infrasonic acoustic waves (frequencies below 40 Hz) propagate efficiently through pipe air columns:
- Low-frequency waves suffer negligible high-frequency scattering from rough pipe walls.
- Waves travel long distances between manholes (20 m to 150 m).
- Obstructions reduce the pipe cross-sectional area and cause an acoustic impedance discontinuity:
  $$Z = \frac{\rho \cdot c}{A}$$
- This impedance mismatch reflects a portion of the wave back toward the sensor.
- The round-trip travel time of the reflected echo directly indicates the distance to the obstruction.

## 3. The Digital Twin Concept
A Digital Twin is a computational model that mirrors the state of a physical asset in real time:
- The sewer digital twin simulates acoustic wave propagation in the pipe air column.
- The system processes acoustic sensor signals to extract reflection signatures.
- AI models classify blockage severity and predict clog location.
- The interactive digital twin dashboard visualizes physical pipe state and alerts operators before overflows occur.

## 4. Expected System Inputs and Outputs

### Inputs
- Transmitted acoustic excitation signal (10 Hz – 40 Hz linear chirp).
- Received acoustic sensor time-series pressure data.
- Pipe structural metadata (segment length, nominal diameter, fluid level).

### Outputs
- Obstruction location (distance in meters from access manhole).
- Blockage severity stage (`Clear`, `Minor`, `Moderate`, `Severe`).
- Acoustic reflection profile and cross-correlation power.
- Confidence metrics and visual pipe state.

## 5. Intended Users and Hackathon Demonstration Scenario

### Intended Users
- Municipal wastewater utility engineers.
- Sewer maintenance teams and field dispatchers.
- Smart city infrastructure monitoring operators.

### Hackathon Demonstration Scenario
1. An operator selects a sewer pipe segment between two manholes.
2. The operator configures pipe length, obstruction distance, severity, and noise levels.
3. The digital twin simulates infrasonic wave propagation.
4. Signal processing isolates the reflection echo.
5. The PyTorch Multi-Task 1D-CNN predicts blockage stage and distance.
6. The dashboard displays the live acoustic waveforms, correlation peak, and a 2D pipe layout with a pinpoint marker.

## 6. Project Scope and Limitations
- **Current Scope**: Focuses on synthetic acoustic data generation, signal processing, multi-task neural network modeling, and an interactive Streamlit digital twin dashboard.
- **Physical Assumptions**: Assumes a straight cylindrical pipe segment with dry air column above the effluent line and acoustic propagation speed $c \approx 343\text{ m/s}$.
- **Limitations**: Real sewer pipes contain bends, junctions, lateral connections, and varying sediment profiles that require future field validation with physical hydrophones.
