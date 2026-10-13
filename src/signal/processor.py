"""Signal processing module for sewer pipe acoustic cross-correlation and feature extraction.

Implements Time-of-Flight (ToF) estimation via matched filtering / cross-correlation
to localize pipe blockages and estimate acoustic reflection profiles.
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from scipy.signal import butter, correlate, correlation_lags, sosfiltfilt


def butter_bandpass_filter(
    data: np.ndarray,
    lowcut: float = 10.0,
    highcut: float = 40.0,
    fs: int = 2000,
    order: int = 4,
) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filter to extract infrasonic band.

    Args:
        data: 1D input acoustic time-series signal.
        lowcut: Lower passband cutoff frequency in Hz. Defaults to 10.0 Hz.
        highcut: Upper passband cutoff frequency in Hz. Defaults to 40.0 Hz.
        fs: Sampling frequency in Hz. Defaults to 2000 Hz.
        order: Filter order. Defaults to 4.

    Returns:
        Filtered 1D numpy array with same length as input.

    Raises:
        ValueError: If cutoff frequencies or sampling rates are invalid.
    """
    if lowcut <= 0.0 or highcut <= lowcut or highcut >= fs / 2.0:
        raise ValueError(
            f"Invalid filter frequencies: lowcut={lowcut}, highcut={highcut}, Nyquist={fs / 2.0}."
        )
    sos = butter(order, [lowcut, highcut], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, data)


def estimate_clog_distance(
    s_rec: np.ndarray,
    s_tx: np.ndarray,
    fs: int = 2000,
    speed_of_sound: float = 343.0,
    min_distance_m: float = 0.5,
    cancel_direct_pulse: bool = True,
) -> Tuple[float, np.ndarray]:
    """Estimate blockage distance from acoustic signals using Time-of-Flight (ToF) cross-correlation.

    Performs matched filtering between received echo `s_rec` and transmitted chirp `s_tx`,
    filters out zero-lag initial pulse spillover, locates the reflection peak, and
    calculates the round-trip distance:
        d = (tau * speed_of_sound) / 2.0

    Args:
        s_rec: 1D array of the received acoustic signal.
        s_tx: 1D array of the transmitted acoustic signal pulse.
        fs: Sampling frequency in Hz. Defaults to 2000 Hz.
        speed_of_sound: Acoustic propagation velocity in air column (m/s). Defaults to 343.0 m/s.
        min_distance_m: Minimum physical distance (meters) to consider for reflection peak
            search to suppress near-field direct spillover. Defaults to 0.5 meters.
        cancel_direct_pulse: Whether to cancel direct transmission pulse bleed-through at lag 0
            via orthogonal projection. Defaults to True.

    Returns:
        Tuple containing:
            - estimated_distance (float): Calculated blockage distance in meters.
            - xcorr_power (np.ndarray): 1D array of cross-correlation power (with zero-lag
              spillover masked to 0.0).

    Raises:
        ValueError: If input arrays are empty, dimensional mismatch occurs, or parameters are invalid.
    """
    s_rec = np.asarray(s_rec, dtype=np.float64)
    s_tx = np.asarray(s_tx, dtype=np.float64)

    if s_rec.ndim != 1 or s_tx.ndim != 1:
        raise ValueError("s_rec and s_tx must be 1-dimensional arrays.")
    if len(s_rec) == 0 or len(s_tx) == 0:
        raise ValueError("Input signals must not be empty.")
    if fs <= 0:
        raise ValueError(f"Sampling frequency fs must be positive, got {fs}.")
    if speed_of_sound <= 0.0:
        raise ValueError(f"Speed of sound must be positive, got {speed_of_sound}.")
    if min_distance_m < 0.0:
        raise ValueError(f"min_distance_m must be non-negative, got {min_distance_m}.")

    # Direct transmission interference cancellation (lag 0 direct path spillover)
    s_rec_proc = s_rec.copy()
    if cancel_direct_pulse:
        energy_tx = float(np.dot(s_tx, s_tx))
        if energy_tx > 1e-12:
            alpha = float(np.dot(s_rec, s_tx) / energy_tx)
            # Subtract direct pulse component if significant correlation is detected at lag 0
            if alpha > 0.05:
                s_rec_proc = s_rec_proc - alpha * s_tx

    # 1. Perform cross-correlation between s_rec and s_tx
    corr = correlate(s_rec_proc, s_tx, mode="full")
    lags = correlation_lags(len(s_rec_proc), len(s_tx), mode="full")

    # Cross-correlation power
    xcorr_power = np.square(corr)

    # 2. Filter out zero-lag initial pulse spillover and negative non-causal lags
    # Minimum lag corresponding to min_distance_m:
    # tau_min = (2 * min_distance_m) / speed_of_sound
    min_lag = max(1, int(np.ceil((2.0 * min_distance_m / speed_of_sound) * fs)))

    # Zero out non-causal and zero-lag spillover regions in the power profile
    spillover_mask = lags < min_lag
    xcorr_power[spillover_mask] = 0.0

    # 3. Locate peak index among valid positive lags
    peak_idx = int(np.argmax(xcorr_power))
    best_lag = int(lags[peak_idx])

    # Convert lag to time delay tau
    tau = float(best_lag) / float(fs)

    # 4. Calculate estimated distance: d = (tau * speed_of_sound) / 2.0
    estimated_distance = float((tau * speed_of_sound) / 2.0)

    return estimated_distance, xcorr_power


def compute_xcorr_profile(
    s_rec: np.ndarray,
    s_tx: np.ndarray,
    fs: int = 2000,
    speed_of_sound: float = 343.0,
) -> Dict[str, np.ndarray]:
    """Compute detailed cross-correlation profile including time delay and distance axes.

    Args:
        s_rec: 1D received signal array.
        s_tx: 1D transmitted signal array.
        fs: Sampling frequency in Hz. Defaults to 2000 Hz.
        speed_of_sound: Acoustic propagation velocity (m/s). Defaults to 343.0 m/s.

    Returns:
        Dict containing lags, time_delays_s, distance_axis_m, correlation, and power.
    """
    corr = correlate(s_rec, s_tx, mode="full")
    lags = correlation_lags(len(s_rec), len(s_tx), mode="full")
    time_delays = lags.astype(np.float64) / float(fs)
    # Distance corresponding to positive lags (two-way travel)
    distances = (time_delays * speed_of_sound) / 2.0
    power = np.square(corr)

    return {
        "lags": lags,
        "time_delays_s": time_delays,
        "distance_axis_m": distances,
        "correlation": corr,
        "power": power,
    }


def extract_acoustic_features(
    s_rec: np.ndarray,
    s_tx: np.ndarray,
    fs: int = 2000,
    speed_of_sound: float = 343.0,
    min_distance_m: float = 0.5,
) -> Dict[str, float]:
    """Extract acoustic digital twin features from raw time-series signals.

    Args:
        s_rec: Received signal array.
        s_tx: Transmitted pulse array.
        fs: Sampling frequency in Hz.
        speed_of_sound: Sound speed in m/s.
        min_distance_m: Minimum search distance in meters.

    Returns:
        Dictionary of extracted physical and statistical features:
            - estimated_distance_m
            - tau_delay_s
            - peak_xcorr_power
            - signal_energy
            - rms_amplitude
            - snr_estimate_db
    """
    d_est, power = estimate_clog_distance(
        s_rec=s_rec,
        s_tx=s_tx,
        fs=fs,
        speed_of_sound=speed_of_sound,
        min_distance_m=min_distance_m,
    )

    peak_power = float(np.max(power))
    tau = float((2.0 * d_est) / speed_of_sound)
    signal_energy = float(np.sum(np.square(s_rec)))
    rms = float(np.sqrt(np.mean(np.square(s_rec))))

    # Estimate SNR from peak vs noise floor
    median_power = float(np.median(power[power > 0])) if np.any(power > 0) else 1e-12
    snr_db = float(10.0 * np.log10(max(1e-12, peak_power / max(1e-12, median_power))))

    return {
        "estimated_distance_m": d_est,
        "tau_delay_s": tau,
        "peak_xcorr_power": peak_power,
        "signal_energy": signal_energy,
        "rms_amplitude": rms,
        "snr_estimate_db": snr_db,
    }
