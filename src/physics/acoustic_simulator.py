"""Acoustic simulation module for sewer pipe infrasonic wave propagation and reflection.

Infrasonic waves (< 40 Hz) propagate along sewer pipe air columns and reflect off
blockages due to acoustic impedance mismatches:
    Z = (rho * c) / A
where:
    rho: Air density (kg/m^3)
    c: Speed of sound in air (~343 m/s)
    A: Cross-sectional area of the pipe (m^2)
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
from scipy.signal import chirp

# Physical constants
SPEED_OF_SOUND_AIR: float = 343.0  # Speed of sound in dry air at ~20°C (m/s)
AIR_DENSITY: float = 1.225  # Air density at standard sea level (kg/m^3)
ATTENUATION_COEFF: float = 0.01  # Infrasonic sewer pipe distance attenuation coefficient (1/m)
DEFAULT_F_MIN: float = 10.0  # Infrasonic lower frequency limit (Hz)
DEFAULT_F_MAX: float = 40.0  # Infrasonic upper frequency limit (Hz)


def compute_acoustic_impedance(
    area_m2: float,
    rho: float = AIR_DENSITY,
    c: float = SPEED_OF_SOUND_AIR,
) -> float:
    """Calculate characteristic acoustic impedance of a pipe air column: Z = (rho * c) / A.

    Args:
        area_m2: Cross-sectional area of the pipe column (m^2).
        rho: Air density (kg/m^3). Defaults to 1.225 kg/m^3.
        c: Speed of sound in air (m/s). Defaults to 343.0 m/s.

    Returns:
        Acoustic impedance Z in Pa*s/m^3 (acoustic ohms).

    Raises:
        ValueError: If area_m2 <= 0.
    """
    if area_m2 <= 0.0:
        raise ValueError(f"Cross-sectional area must be positive, got {area_m2}.")
    return (rho * c) / area_m2


def compute_reflection_coefficient(z1: float, z2: float) -> float:
    """Compute the acoustic pressure reflection coefficient R = (Z2 - Z1) / (Z2 + Z1).

    Args:
        z1: Acoustic impedance before junction/blockage (Pa*s/m^3).
        z2: Acoustic impedance at junction/blockage (Pa*s/m^3).

    Returns:
        Reflection coefficient R in [-1.0, 1.0].
    """
    if z1 + z2 == 0.0:
        return 0.0
    return (z2 - z1) / (z2 + z1)


def generate_pipe_reflection(
    pipe_length_m: float,
    clog_dist_m: float,
    blockage_severity: float,
    noise_level: float,
    fs: int = 2000,
    pulse_duration: float = 0.2,
    total_time: Optional[float] = None,
    f_min: float = DEFAULT_F_MIN,
    f_max: float = DEFAULT_F_MAX,
    include_direct_pulse: bool = False,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate synthetic acoustic time-series waveforms for sewer pipe clog reflection.

    Models a 10 Hz - 40 Hz linear chirp pulse sent down a sewer pipe, propagating,
    reflecting off a blockage located at `clog_dist_m`, and returning to the acoustic
    transceiver at the pipe entrance.

    Args:
        pipe_length_m: Total physical length of the sewer pipe segment (meters).
        clog_dist_m: Distance from the acoustic transceiver to the clog (meters).
        blockage_severity: Degree of pipe blockage in range [0.0, 1.0]
            (0.0 = completely clear, 1.0 = 100% full clog).
        noise_level: Standard deviation of Additive White Gaussian Noise (AWGN).
        fs: Sampling frequency in Hz. Defaults to 2000 Hz.
        pulse_duration: Duration of the transmitted chirp pulse in seconds. Defaults to 0.2s.
        total_time: Total simulation duration in seconds. If None, automatically
            calculated to include full round-trip travel time through the pipe plus pulse duration.
        f_min: Chirp start frequency in Hz. Defaults to 10.0 Hz.
        f_max: Chirp end frequency in Hz. Defaults to 40.0 Hz.
        include_direct_pulse: Whether to include the direct transmitted pulse in s_rec. Defaults to False.
        seed: Random seed for reproducible AWGN noise generation. Defaults to None.

    Returns:
        Tuple containing:
            - t (np.ndarray): Time vector of length int(round(fs * total_time)).
            - s_tx (np.ndarray): Transmitted 10 Hz - 40 Hz linear chirp signal.
            - s_rec (np.ndarray): Received echo signal with attenuation and AWGN.
            - metadata (Dict[str, Any]): Ground-truth simulation parameters.

    Raises:
        ValueError: If pipe dimensions, severity, noise, or frequencies are invalid.
    """
    if pipe_length_m <= 0.0:
        raise ValueError(f"pipe_length_m must be positive, got {pipe_length_m}.")
    if clog_dist_m < 0.0 or clog_dist_m > pipe_length_m:
        raise ValueError(
            f"clog_dist_m ({clog_dist_m}) must be in range [0.0, pipe_length_m ({pipe_length_m})]."
        )
    if not (0.0 <= blockage_severity <= 1.0):
        raise ValueError(
            f"blockage_severity must be in [0.0, 1.0], got {blockage_severity}."
        )
    if noise_level < 0.0:
        raise ValueError(f"noise_level must be non-negative, got {noise_level}.")
    if fs <= 0:
        raise ValueError(f"Sampling frequency fs must be positive, got {fs}.")
    if pulse_duration <= 0.0:
        raise ValueError(f"pulse_duration must be positive, got {pulse_duration}.")
    if f_min <= 0.0 or f_max <= f_min:
        raise ValueError(f"Invalid frequency bounds: f_min={f_min}, f_max={f_max}.")

    # Calculate propagation delay: tau = (2 * clog_dist_m) / 343.0
    c = SPEED_OF_SOUND_AIR
    tau = (2.0 * clog_dist_m) / c

    # Determine total simulation time
    if total_time is None:
        # Round trip to end of pipe plus pulse duration and buffer
        round_trip_pipe = (2.0 * pipe_length_m) / c
        total_time = float(round_trip_pipe + pulse_duration + 0.1)
    elif total_time <= 0.0:
        raise ValueError(f"total_time must be positive, got {total_time}.")

    num_samples = int(np.round(fs * total_time))
    t = np.arange(num_samples, dtype=np.float64) / float(fs)

    # 1. Transmitted wave s_tx(t): 10 Hz - 40 Hz linear chirp
    s_tx = np.zeros(num_samples, dtype=np.float64)
    tx_mask = (t >= 0.0) & (t < pulse_duration)
    if np.any(tx_mask):
        s_tx[tx_mask] = chirp(
            t[tx_mask],
            f0=f_min,
            t1=pulse_duration,
            f1=f_max,
            method="linear",
        )

    # 2. Reflected echo wave s_echo(t)
    # Attenuation: exp(-0.01 * clog_dist_m)
    attenuation = float(np.exp(-ATTENUATION_COEFF * clog_dist_m))
    reflection_amplitude = float(blockage_severity * attenuation)

    s_echo = np.zeros(num_samples, dtype=np.float64)
    if reflection_amplitude > 0.0:
        echo_mask = (t >= tau) & (t < tau + pulse_duration)
        if np.any(echo_mask):
            t_rel = t[echo_mask] - tau
            s_echo[echo_mask] = reflection_amplitude * chirp(
                t_rel,
                f0=f_min,
                t1=pulse_duration,
                f1=f_max,
                method="linear",
            )

    # 3. Additive White Gaussian Noise (AWGN)
    rng = np.random.default_rng(seed)
    if noise_level > 0.0:
        noise = rng.normal(0.0, noise_level, size=num_samples)
    else:
        noise = np.zeros(num_samples, dtype=np.float64)

    # Received signal
    s_rec = s_echo + noise
    if include_direct_pulse:
        s_rec = s_rec + s_tx

    metadata: Dict[str, Any] = {
        "pipe_length_m": float(pipe_length_m),
        "clog_dist_m": float(clog_dist_m),
        "blockage_severity": float(blockage_severity),
        "noise_level": float(noise_level),
        "fs": int(fs),
        "c": float(c),
        "tau_true_s": float(tau),
        "attenuation": float(attenuation),
        "reflection_amplitude": float(reflection_amplitude),
        "pulse_duration_s": float(pulse_duration),
        "total_time": float(total_time),
        "total_time_s": float(total_time),
        "f_min_hz": float(f_min),
        "f_max_hz": float(f_max),
        "num_samples": int(num_samples),
        "seed": seed,
    }

    return t, s_tx, s_rec, metadata
