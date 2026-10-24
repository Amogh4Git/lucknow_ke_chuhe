# Physics Simulation Module (`src/physics/`)

## Purpose
This module generates synthetic acoustic time-series waveforms for sewer pipe networks. It models infrasonic wave propagation and reflection off physical blockages.

## Physical Foundation
1. **Acoustic Impedance**:
   $$Z = \frac{\rho \cdot c}{A}$$
   - $\rho$: Air density ($1.225\text{ kg/m}^3$).
   - $c$: Speed of sound in dry air ($343.0\text{ m/s}$).
   - $A$: Pipe cross-sectional area ($\text{m}^2$).

2. **Reflection Coefficient**:
   $$R = \frac{Z_2 - Z_1}{Z_2 + Z_1} = \frac{A_1 - A_2}{A_1 + A_2}$$

3. **Time-of-Flight (ToF)**:
   $$\tau = \frac{2 \cdot d}{c}$$
   - $d$: Distance to obstruction ($\text{m}$).
   - $\tau$: Round-trip delay ($\text{s}$).

## Components
- `acoustic_simulator.py`:
  - `compute_acoustic_impedance(area_m2, rho, c)`: Calculates column impedance.
  - `compute_reflection_coefficient(z1, z2)`: Calculates boundary reflection.
  - `generate_pipe_reflection(pipe_length_m, clog_dist_m, blockage_severity, noise_level, fs)`: Synthesizes transmitted chirp $s_{\text{tx}}(t)$ and reflected echo $s_{\text{rec}}(t)$.

## Inputs and Outputs
- **Inputs**: Pipe length (m), clog distance (m), blockage severity (0.0 to 1.0), noise level, sampling rate (Hz).
- **Outputs**: Time array `t`, transmitted wave `s_tx`, received echo `s_rec`, and ground-truth metadata dictionary.
