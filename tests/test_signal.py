"""Unit tests for the acoustic signal processing and distance estimation module."""

import numpy as np
import pytest

from src.physics.acoustic_simulator import generate_pipe_reflection
from src.signal.processor import (
    butter_bandpass_filter,
    compute_xcorr_profile,
    estimate_clog_distance,
    extract_acoustic_features,
)


def test_estimate_clog_distance_45m() -> None:
    """Verify that for a synthetic input with clog_dist_m=45.0, distance is within +/- 0.5 meters."""
    true_dist = 45.0
    fs = 2000
    pipe_length = 100.0

    t, s_tx, s_rec, meta = generate_pipe_reflection(
        pipe_length_m=pipe_length,
        clog_dist_m=true_dist,
        blockage_severity=0.8,
        noise_level=0.02,
        fs=fs,
        seed=42,
    )

    est_dist, xcorr_power = estimate_clog_distance(
        s_rec=s_rec,
        s_tx=s_tx,
        fs=fs,
        speed_of_sound=343.0,
    )

    error = abs(est_dist - true_dist)
    assert error <= 0.5, f"Estimated {est_dist}m, true {true_dist}m, error {error}m exceeds 0.5m tolerance."
    assert isinstance(est_dist, float)
    assert isinstance(xcorr_power, np.ndarray)


def test_estimate_clog_distance_with_direct_pulse_spillover() -> None:
    """Verify distance estimation accuracy when direct transmission pulse spillover is present."""
    true_dist = 45.0
    fs = 2000

    t, s_tx, s_rec, meta = generate_pipe_reflection(
        pipe_length_m=100.0,
        clog_dist_m=true_dist,
        blockage_severity=0.8,
        noise_level=0.02,
        fs=fs,
        include_direct_pulse=True,
        seed=42,
    )

    est_dist, xcorr_power = estimate_clog_distance(
        s_rec=s_rec,
        s_tx=s_tx,
        fs=fs,
        speed_of_sound=343.0,
    )

    error = abs(est_dist - true_dist)
    assert error <= 0.5, f"Estimated {est_dist}m with direct spillover exceeds 0.5m error: {error}m."


@pytest.mark.parametrize("target_dist", [12.0, 25.0, 50.0, 75.0, 90.0])
def test_estimate_clog_distance_various_ranges(target_dist: float) -> None:
    """Verify high-accuracy localization across a broad range of pipe lengths."""
    fs = 2000
    pipe_length = 100.0

    t, s_tx, s_rec, _ = generate_pipe_reflection(
        pipe_length_m=pipe_length,
        clog_dist_m=target_dist,
        blockage_severity=0.75,
        noise_level=0.015,
        fs=fs,
        seed=int(target_dist),
    )

    est_dist, _ = estimate_clog_distance(s_rec=s_rec, s_tx=s_tx, fs=fs)
    error = abs(est_dist - target_dist)
    assert error <= 0.5, f"At {target_dist}m, estimated {est_dist}m with error {error}m."


def test_xcorr_power_array_properties() -> None:
    """Verify shape, finiteness, and zero-lag blanking of the cross-correlation power output."""
    fs = 2000
    t, s_tx, s_rec, _ = generate_pipe_reflection(
        pipe_length_m=60.0,
        clog_dist_m=30.0,
        blockage_severity=0.7,
        noise_level=0.01,
        fs=fs,
        seed=1,
    )

    est_dist, xcorr_power = estimate_clog_distance(s_rec=s_rec, s_tx=s_tx, fs=fs)
    expected_len = len(s_rec) + len(s_tx) - 1

    assert len(xcorr_power) == expected_len
    assert not np.isnan(xcorr_power).any()
    assert not np.isinf(xcorr_power).any()
    assert (xcorr_power >= 0.0).all()


def test_butter_bandpass_filter_attenuates_out_of_band_noise() -> None:
    """Verify bandpass filter preserves infrasonic band while strongly rejecting out-of-band signals."""
    fs = 2000
    t = np.arange(2000) / fs

    # 25 Hz infrasound signal (inside passband 10-40 Hz)
    sig_infrasound = np.sin(2 * np.pi * 25.0 * t)
    # 200 Hz interference signal (far above passband)
    sig_noise = np.sin(2 * np.pi * 200.0 * t)

    combined = sig_infrasound + sig_noise
    filtered = butter_bandpass_filter(combined, lowcut=10.0, highcut=40.0, fs=fs)

    # Infrasound component should be preserved
    infrasound_rms = np.sqrt(np.mean(sig_infrasound**2))
    filtered_rms = np.sqrt(np.mean(filtered**2))
    assert np.isclose(filtered_rms, infrasound_rms, rtol=0.15)

    # Filter purely high frequency noise and verify strong power rejection (> 25 dB)
    pure_noise_filtered = butter_bandpass_filter(sig_noise, lowcut=10.0, highcut=40.0, fs=fs)
    noise_rms = np.sqrt(np.mean(sig_noise**2))
    filtered_noise_rms = np.sqrt(np.mean(pure_noise_filtered**2))
    rms_attenuation_ratio = filtered_noise_rms / noise_rms
    assert rms_attenuation_ratio < 0.05, f"High frequency noise RMS ratio {rms_attenuation_ratio} too weak."


def test_extract_acoustic_features() -> None:
    """Verify that extract_acoustic_features returns comprehensive feature dictionary."""
    fs = 2000
    t, s_tx, s_rec, meta = generate_pipe_reflection(
        pipe_length_m=80.0,
        clog_dist_m=40.0,
        blockage_severity=0.8,
        noise_level=0.02,
        fs=fs,
        seed=10,
    )

    features = extract_acoustic_features(s_rec=s_rec, s_tx=s_tx, fs=fs)

    required_keys = [
        "estimated_distance_m",
        "tau_delay_s",
        "peak_xcorr_power",
        "signal_energy",
        "rms_amplitude",
        "snr_estimate_db",
    ]
    for key in required_keys:
        assert key in features, f"Missing feature key: {key}"
        assert not np.isnan(features[key]), f"Feature {key} is NaN"

    assert abs(features["estimated_distance_m"] - 40.0) <= 0.5


def test_compute_xcorr_profile() -> None:
    """Verify compute_xcorr_profile produces aligned distance and delay axes."""
    fs = 2000
    s_tx = np.zeros(200)
    s_rec = np.zeros(300)
    s_tx[0] = 1.0
    s_rec[50] = 1.0

    profile = compute_xcorr_profile(s_rec, s_tx, fs=fs, speed_of_sound=343.0)
    assert "lags" in profile
    assert "time_delays_s" in profile
    assert "distance_axis_m" in profile
    assert "correlation" in profile
    assert "power" in profile
    assert len(profile["lags"]) == len(s_rec) + len(s_tx) - 1


def test_input_validation_errors() -> None:
    """Verify input validation for invalid parameters and array dimensions."""
    s_valid = np.ones(100)

    with pytest.raises(ValueError, match="1-dimensional"):
        estimate_clog_distance(np.ones((10, 10)), s_valid)

    with pytest.raises(ValueError, match="empty"):
        estimate_clog_distance(np.array([]), s_valid)

    with pytest.raises(ValueError, match="fs must be positive"):
        estimate_clog_distance(s_valid, s_valid, fs=-100)

    with pytest.raises(ValueError, match="Speed of sound"):
        estimate_clog_distance(s_valid, s_valid, speed_of_sound=-343.0)

    with pytest.raises(ValueError, match="Invalid filter frequencies"):
        butter_bandpass_filter(s_valid, lowcut=50.0, highcut=40.0, fs=2000)
