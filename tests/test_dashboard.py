"""Unit tests for the Streamlit dashboard components and plotting functions."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.dashboard.app import (
    load_ai_model,
    plot_2d_pipe_map,
    plot_acoustic_waveforms,
    plot_cross_correlation_power,
)


def test_load_ai_model() -> None:
    """Verify load_ai_model retrieves the trained Multi-Task 1D-CNN."""
    model = load_ai_model("saved_models/clog_1dcnn.pth")
    assert model is not None
    assert not model.training


def test_plot_acoustic_waveforms() -> None:
    """Verify plot_acoustic_waveforms produces a valid Matplotlib figure."""
    t = np.linspace(0, 1.0, 2000)
    s_tx = np.sin(2 * np.pi * 25.0 * t)
    s_rec = 0.5 * s_tx + 0.01 * np.random.randn(2000)

    fig = plot_acoustic_waveforms(t, s_tx, s_rec, tau_true=0.25)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_cross_correlation_power() -> None:
    """Verify plot_cross_correlation_power generates a valid figure."""
    dist_axis = np.linspace(0, 100.0, 1000)
    power = np.exp(-((dist_axis - 45.0) ** 2) / 2.0)

    fig = plot_cross_correlation_power(
        dist_axis=dist_axis,
        power=power,
        pred_dist=45.02,
        true_dist=45.0,
        pipe_length=100.0,
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_2d_pipe_map_clogged_and_clear() -> None:
    """Verify plot_2d_pipe_map handles both clogged and clear pipe states."""
    fig_clogged = plot_2d_pipe_map(
        pipe_length_m=100.0,
        pred_dist_m=35.0,
        true_dist_m=35.0,
        severity=0.7,
        is_clear=False,
    )
    assert isinstance(fig_clogged, plt.Figure)
    plt.close(fig_clogged)

    fig_clear = plot_2d_pipe_map(
        pipe_length_m=100.0,
        pred_dist_m=100.0,
        true_dist_m=100.0,
        severity=0.0,
        is_clear=True,
    )
    assert isinstance(fig_clear, plt.Figure)
    plt.close(fig_clear)
