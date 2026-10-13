"""Unit tests for the acoustic simulator physics module."""

import numpy as np
import pytest

from src.physics.acoustic_simulator import (
    SPEED_OF_SOUND_AIR,
    compute_acoustic_impedance,
    compute_reflection_coefficient,
    generate_pipe_reflection,
)


def test_output_array_lengths_and_no_nan_explicit_duration() -> None:
    """Verify output array lengths match fs * total_time and contain no NaN/Inf values."""
    fs = 2000
    total_time = 1.5
    pipe_length_m = 100.0
    clog_dist_m = 35.0
    blockage_severity = 0.6
    noise_level = 0.02

    t, s_tx, s_rec, metadata = generate_pipe_reflection(
        pipe_length_m=pipe_length_m,
        clog_dist_m=clog_dist_m,
        blockage_severity=blockage_severity,
        noise_level=noise_level,
        fs=fs,
        total_time=total_time,
        seed=42,
    )

    expected_len = int(fs * total_time)
    assert len(t) == expected_len, f"Expected length {expected_len}, got {len(t)}"
    assert len(s_tx) == expected_len, f"Expected length {expected_len}, got {len(s_tx)}"
    assert len(s_rec) == expected_len, f"Expected length {expected_len}, got {len(s_rec)}"

    # Check for no NaNs or Infs
    assert not np.isnan(t).any(), "Time vector t contains NaN values."
    assert not np.isinf(t).any(), "Time vector t contains Inf values."
    assert not np.isnan(s_tx).any(), "Transmitted signal s_tx contains NaN values."
    assert not np.isinf(s_tx).any(), "Transmitted signal s_tx contains Inf values."
    assert not np.isnan(s_rec).any(), "Received signal s_rec contains NaN values."
    assert not np.isinf(s_rec).any(), "Received signal s_rec contains Inf values."


def test_output_array_lengths_auto_total_time() -> None:
    """Verify output array length matches fs * total_time when total_time is calculated automatically."""
    fs = 2000
    pipe_length_m = 60.0
    clog_dist_m = 20.0

    t, s_tx, s_rec, metadata = generate_pipe_reflection(
        pipe_length_m=pipe_length_m,
        clog_dist_m=clog_dist_m,
        blockage_severity=0.5,
        noise_level=0.01,
        fs=fs,
    )

    total_time = metadata["total_time"]
    expected_len = int(np.round(fs * total_time))
    assert len(t) == expected_len
    assert len(s_tx) == expected_len
    assert len(s_rec) == expected_len
    assert not np.isnan(s_rec).any()


def test_time_delay_calculation() -> None:
    """Verify that delay tau is calculated accurately as (2 * clog_dist_m) / 343.0."""
    clog_dist_m = 51.45
    expected_tau = (2.0 * clog_dist_m) / SPEED_OF_SOUND_AIR

    _, _, _, metadata = generate_pipe_reflection(
        pipe_length_m=100.0,
        clog_dist_m=clog_dist_m,
        blockage_severity=0.8,
        noise_level=0.0,
        fs=2000,
    )

    assert np.isclose(metadata["tau_true_s"], expected_tau, atol=1e-6)
    assert np.isclose(metadata["c"], 343.0, atol=1e-6)


def test_clear_pipe_no_reflection() -> None:
    """Verify that when blockage_severity=0.0 and noise_level=0.0, no echo is present."""
    t, s_tx, s_rec, metadata = generate_pipe_reflection(
        pipe_length_m=80.0,
        clog_dist_m=40.0,
        blockage_severity=0.0,
        noise_level=0.0,
        fs=2000,
        total_time=1.0,
    )

    assert metadata["reflection_amplitude"] == 0.0
    assert np.all(s_rec == 0.0), "Clear pipe with zero noise should have zero echo."
    assert np.max(np.abs(s_tx)) > 0.5, "Transmitted pulse should be present."


def test_blockage_severity_and_attenuation_scaling() -> None:
    """Verify reflection amplitude scales with severity and exponential distance attenuation."""
    dist = 50.0
    attenuation = np.exp(-0.01 * dist)

    # 100% clog
    _, _, s_rec_full, meta_full = generate_pipe_reflection(
        pipe_length_m=100.0,
        clog_dist_m=dist,
        blockage_severity=1.0,
        noise_level=0.0,
        fs=2000,
        total_time=1.0,
    )

    # 50% clog
    _, _, s_rec_half, meta_half = generate_pipe_reflection(
        pipe_length_m=100.0,
        clog_dist_m=dist,
        blockage_severity=0.5,
        noise_level=0.0,
        fs=2000,
        total_time=1.0,
    )

    assert np.isclose(meta_full["reflection_amplitude"], 1.0 * attenuation)
    assert np.isclose(meta_half["reflection_amplitude"], 0.5 * attenuation)

    # Peak amplitude of half clog should be half of full clog
    peak_full = np.max(np.abs(s_rec_full))
    peak_half = np.max(np.abs(s_rec_half))
    assert np.isclose(peak_half, 0.5 * peak_full, rtol=1e-3)


def test_distance_attenuation_decay() -> None:
    """Verify that further clogs yield smaller reflected amplitudes."""
    _, _, s_near, _ = generate_pipe_reflection(
        pipe_length_m=100.0,
        clog_dist_m=10.0,
        blockage_severity=1.0,
        noise_level=0.0,
        fs=2000,
        total_time=1.0,
    )

    _, _, s_far, _ = generate_pipe_reflection(
        pipe_length_m=100.0,
        clog_dist_m=80.0,
        blockage_severity=1.0,
        noise_level=0.0,
        fs=2000,
        total_time=1.0,
    )

    assert np.max(np.abs(s_near)) > np.max(np.abs(s_far))


def test_noise_level_reproducibility() -> None:
    """Verify deterministic noise generation when seed is provided."""
    _, _, s1, _ = generate_pipe_reflection(
        pipe_length_m=50.0,
        clog_dist_m=20.0,
        blockage_severity=0.5,
        noise_level=0.05,
        fs=2000,
        total_time=1.0,
        seed=123,
    )
    _, _, s2, _ = generate_pipe_reflection(
        pipe_length_m=50.0,
        clog_dist_m=20.0,
        blockage_severity=0.5,
        noise_level=0.05,
        fs=2000,
        total_time=1.0,
        seed=123,
    )
    assert np.array_equal(s1, s2)


def test_input_validation_errors() -> None:
    """Verify that invalid parameters raise descriptive ValueError exceptions."""
    with pytest.raises(ValueError, match="clog_dist_m"):
        generate_pipe_reflection(pipe_length_m=50.0, clog_dist_m=60.0, blockage_severity=0.5, noise_level=0.01)

    with pytest.raises(ValueError, match="blockage_severity"):
        generate_pipe_reflection(pipe_length_m=50.0, clog_dist_m=20.0, blockage_severity=1.5, noise_level=0.01)

    with pytest.raises(ValueError, match="noise_level"):
        generate_pipe_reflection(pipe_length_m=50.0, clog_dist_m=20.0, blockage_severity=0.5, noise_level=-0.1)

    with pytest.raises(ValueError, match="pipe_length_m"):
        generate_pipe_reflection(pipe_length_m=-10.0, clog_dist_m=5.0, blockage_severity=0.5, noise_level=0.01)


def test_acoustic_impedance_functions() -> None:
    """Verify impedance calculation Z = (rho * c) / A and reflection coefficient."""
    area = 0.5  # m^2
    z = compute_acoustic_impedance(area_m2=area)
    expected_z = (1.225 * 343.0) / 0.5
    assert np.isclose(z, expected_z)

    # Infinite restriction (z2 -> inf) => R = 1
    r_full = compute_reflection_coefficient(z1=100.0, z2=1000000.0)
    assert np.isclose(r_full, 1.0, atol=1e-3)

    # Identical impedance => R = 0
    r_zero = compute_reflection_coefficient(z1=100.0, z2=100.0)
    assert np.isclose(r_zero, 0.0)
