"""Streamlit Dashboard for Infrasonic Sewer Clog Detection Digital Twin.

Real-time interactive dashboard visualizing:
- Pipe acoustic wave simulation and chirp reflection
- Cross-correlation Time-of-Flight (ToF) matched filtering
- PyTorch Multi-Task 1D-CNN AI predictions (Stage Classification & Distance Regression)
- 2D interactive sewer pipe digital twin schematic with pinpoint clog marker
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Optional, Tuple
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

from src.models.clog_cnn1d import SEVERITY_CLASSES, InfrasonicClogDetector1DCNN
from src.physics.acoustic_simulator import (
    AIR_DENSITY,
    SPEED_OF_SOUND_AIR,
    compute_acoustic_impedance,
    compute_reflection_coefficient,
    generate_pipe_reflection,
)
from src.signal.processor import compute_xcorr_profile, estimate_clog_distance

# Streamlit Page Configuration
st.set_page_config(
    page_title="Infrasonic Sewer Digital Twin",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_ai_model(model_path: str = "saved_models/clog_1dcnn.pth") -> Optional[InfrasonicClogDetector1DCNN]:
    """Load and cache the trained 1D-CNN Multi-Task model."""
    full_path = project_root / model_path
    if not full_path.exists():
        st.warning(f"Saved model not found at {full_path}. Running with physics ToF baseline.")
        return None
    model = InfrasonicClogDetector1DCNN(in_channels=1, num_classes=4)
    state = torch.load(full_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def plot_acoustic_waveforms(t: np.ndarray, s_tx: np.ndarray, s_rec: np.ndarray, tau_true: float) -> plt.Figure:
    """Generate Matplotlib line chart for transmitted and received acoustic signals."""
    fig, ax = plt.subplots(figsize=(12, 3.8), dpi=100)
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#161b22")

    ax.plot(t, s_tx, color="#58a6ff", linewidth=1.4, alpha=0.9, label="Transmitted Chirp s_tx(t) [10-40 Hz]")
    ax.plot(t, s_rec, color="#3fb950", linewidth=1.2, alpha=0.85, label="Received Acoustic Signal s_rec(t)")

    # Echo arrival marker
    if tau_true < t[-1]:
        ax.axvline(tau_true, color="#f85149", linestyle="--", linewidth=1.5, alpha=0.8, label=f"Echo Arrival (τ = {tau_true:.3f}s)")

    ax.set_title("Acoustic Time-Series Waveforms (Infrasonic Sewer Monitoring)", color="#e6edf3", fontsize=12, pad=10)
    ax.set_xlabel("Time (seconds)", color="#8b949e", fontsize=10)
    ax.set_ylabel("Normalized Pressure Amplitude", color="#8b949e", fontsize=10)
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.grid(True, color="#21262d", linestyle="--", alpha=0.7)
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3", loc="upper right", fontsize=9)
    plt.tight_layout()
    return fig


def plot_cross_correlation_power(
    dist_axis: np.ndarray,
    power: np.ndarray,
    pred_dist: float,
    true_dist: float,
    pipe_length: float,
) -> plt.Figure:
    """Generate Matplotlib line chart for cross-correlation ToF power profile."""
    fig, ax = plt.subplots(figsize=(12, 3.8), dpi=100)
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#161b22")

    # Filter to display within pipe length bounds
    mask = (dist_axis >= 0.0) & (dist_axis <= pipe_length)
    x_plot = dist_axis[mask]
    y_plot = power[mask]
    norm_y = y_plot / (np.max(y_plot) + 1e-12)

    ax.plot(x_plot, norm_y, color="#bc8cff", linewidth=1.8, label="Cross-Correlation ToF Power Profile")
    ax.fill_between(x_plot, 0, norm_y, color="#bc8cff", alpha=0.25)

    # Highlight predicted and real blockage distance
    ax.axvline(pred_dist, color="#d29922", linestyle="-", linewidth=2.0, label=f"AI Estimated Clog: {pred_dist:.2f} m")
    ax.axvline(true_dist, color="#f85149", linestyle=":", linewidth=2.0, label=f"Ground Truth Clog: {true_dist:.2f} m")

    ax.set_title("Matched Filter Cross-Correlation ToF Power (Distance Profile)", color="#e6edf3", fontsize=12, pad=10)
    ax.set_xlabel("Distance from Acoustic Transceiver (meters)", color="#8b949e", fontsize=10)
    ax.set_ylabel("Normalized Correlation Power", color="#8b949e", fontsize=10)
    ax.set_xlim(0, pipe_length)
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.grid(True, color="#21262d", linestyle="--", alpha=0.7)
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3", loc="upper right", fontsize=9)
    plt.tight_layout()
    return fig


def plot_2d_pipe_map(
    pipe_length_m: float,
    pred_dist_m: float,
    true_dist_m: float,
    severity: float,
    is_clear: bool,
) -> plt.Figure:
    """Render a 2D schematic layout bar of the sewer pipe with pinpoint clog markers."""
    fig, ax = plt.subplots(figsize=(12, 2.6), dpi=100)
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#161b22")

    # Draw Concrete Pipe Walls
    pipe_height = 2.0
    wall_thickness = 0.25
    ax.add_patch(patches.Rectangle((0, pipe_height), pipe_length_m, wall_thickness, color="#484f58", label="Pipe Casing"))
    ax.add_patch(patches.Rectangle((0, -wall_thickness), pipe_length_m, wall_thickness, color="#484f58"))

    # Pipe Interior (Air column / effluent flow)
    ax.add_patch(patches.Rectangle((0, 0), pipe_length_m, pipe_height, color="#1f242c"))
    # Effluent liquid level in lower 25% of pipe
    liquid_height = 0.4
    ax.add_patch(patches.Rectangle((0, 0), pipe_length_m, liquid_height, color="#238636", alpha=0.35, label="Wastewater Baseflow"))

    # Transceiver Icon at entrance x=0
    ax.add_patch(patches.Rectangle((-2.0, -0.2), 2.0, pipe_height + 0.4, color="#1f6feb", alpha=0.9, label="Acoustic Transceiver"))
    ax.text(-1.0, pipe_height / 2, "SENSOR", color="white", fontsize=8, fontweight="bold", ha="center", va="center")

    if not is_clear:
        # Draw physical blockage geometry at ground truth location
        block_width = max(1.5, pipe_length_m * 0.02)
        block_height = pipe_height * severity
        block_y = pipe_height - block_height
        ax.add_patch(
            patches.Rectangle(
                (true_dist_m - block_width / 2, block_y),
                block_width,
                block_height,
                color="#da3633",
                alpha=0.85,
                label=f"Blockage ({severity*100:.0f}%)",
            )
        )

        # Pinpoint Marker at AI Predicted Clog Location
        marker_x = min(pred_dist_m, pipe_length_m)
        ax.plot(marker_x, pipe_height + 0.5, marker="v", markersize=14, color="#d29922", label=f"AI Pinpoint ({marker_x:.1f}m)")
        ax.axvline(marker_x, color="#d29922", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.text(marker_x, pipe_height + 0.9, f"AI: {marker_x:.1f}m", color="#d29922", fontsize=9, fontweight="bold", ha="center")

        # Ground Truth Marker
        ax.plot(true_dist_m, -0.6, marker="^", markersize=12, color="#f85149", label=f"True Clog ({true_dist_m:.1f}m)")
        ax.text(true_dist_m, -1.0, f"True: {true_dist_m:.1f}m", color="#f85149", fontsize=9, ha="center")
    else:
        ax.text(pipe_length_m / 2, pipe_height / 2, "PIPE 100% CLEAR - NO BLOCKAGES DETECTED", color="#3fb950", fontsize=11, fontweight="bold", ha="center", va="center")

    ax.set_xlim(-4.0, pipe_length_m + 4.0)
    ax.set_ylim(-1.4, pipe_height + 1.4)
    ax.set_title("2D Sewer Pipe Digital Twin Layout Map", color="#e6edf3", fontsize=12, pad=10)
    ax.set_xlabel("Pipe Chainage (meters from access manhole)", color="#8b949e", fontsize=10)
    ax.get_yaxis().set_visible(False)
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3", loc="upper right", fontsize=8, ncol=4)
    plt.tight_layout()
    return fig


def main() -> None:
    """Streamlit Application Entrypoint."""
    st.title("🌊 Infrasonic Sewer Clog Detection Digital Twin")
    st.markdown(
        "AI-powered subsurface acoustic pipeline digital twin. Infrasonic chirp waves (10–40 Hz) reflect off sewer blockages "
        "due to acoustic impedance mismatches ($Z = \\frac{\\rho c}{A}$). Matched filtering and multi-task 1D-CNNs localize and classify blockages in real-time."
    )

    # 1. Sidebar Inputs
    st.sidebar.header("⚙️ Sewer Pipeline Digital Twin Controls")

    pipe_length_m = st.sidebar.slider(
        "Pipe Segment Length (m)",
        min_value=20.0,
        max_value=150.0,
        value=100.0,
        step=5.0,
        help="Total physical length of sewer pipe segment between consecutive manholes.",
    )

    max_clog_slider = min(140.0, pipe_length_m - 2.0)
    clog_dist_m = st.sidebar.slider(
        "Clog Distance from Transceiver (m)",
        min_value=5.0,
        max_value=float(max_clog_slider),
        value=min(45.0, max_clog_slider),
        step=1.0,
        help="Simulated ground-truth distance from the sensor transceiver to the obstruction.",
    )

    blockage_severity_pct = st.sidebar.slider(
        "Blockage Severity (%)",
        min_value=0,
        max_value=100,
        value=75,
        step=5,
        help="Percentage of pipe cross-sectional area occluded (0% = Clear, 100% = Full blockage).",
    )

    noise_level = st.sidebar.slider(
        "Environmental Noise Level (AWGN σ)",
        min_value=0.00,
        max_value=0.20,
        value=0.02,
        step=0.01,
        help="Additive White Gaussian Noise simulating turbulent sewer flow and ambient urban acoustic noise.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Transceiver Parameters")
    fs = 2000
    st.sidebar.write(f"• **Sampling Rate ($f_s$):** {fs} Hz")
    st.sidebar.write("• **Band:** 10 Hz – 40 Hz Linear Chirp")
    st.sidebar.write(f"• **Speed of Sound ($c$):** {SPEED_OF_SOUND_AIR} m/s")

    # 2. Synthetic Acoustic Physics Simulation
    severity_ratio = float(blockage_severity_pct) / 100.0
    t, s_tx, s_rec, metadata = generate_pipe_reflection(
        pipe_length_m=pipe_length_m,
        clog_dist_m=clog_dist_m,
        blockage_severity=severity_ratio,
        noise_level=noise_level,
        fs=fs,
        total_time=1.0,
        seed=42,
    )

    # 3. Time-of-Flight Cross-Correlation Signal Processing
    tof_est_dist, xcorr_power = estimate_clog_distance(
        s_rec=s_rec,
        s_tx=s_tx,
        fs=fs,
        speed_of_sound=SPEED_OF_SOUND_AIR,
    )
    profile = compute_xcorr_profile(s_rec, s_tx, fs=fs, speed_of_sound=SPEED_OF_SOUND_AIR)

    # 4. Multi-Task 1D-CNN AI Inference
    model = load_ai_model()
    if model is not None:
        input_tensor = torch.tensor(s_rec, dtype=torch.float32).view(1, 1, -1)
        with torch.no_grad():
            logits, ai_dist_tensor = model(input_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            predicted_class_idx = int(torch.argmax(logits, dim=1).item())
            ai_predicted_distance = float(ai_dist_tensor.item())

        ai_severity_label = SEVERITY_CLASSES[predicted_class_idx]
    else:
        # Fallback if checkpoint not present
        if severity_ratio < 0.05:
            predicted_class_idx = 0
        elif severity_ratio <= 0.35:
            predicted_class_idx = 1
        elif severity_ratio <= 0.70:
            predicted_class_idx = 2
        else:
            predicted_class_idx = 3
        ai_severity_label = SEVERITY_CLASSES[predicted_class_idx]
        ai_predicted_distance = tof_est_dist
        probs = np.zeros(4)
        probs[predicted_class_idx] = 1.0

    is_clear = (blockage_severity_pct == 0) or (ai_severity_label == "Clear" and blockage_severity_pct < 5)
    final_pred_distance = ai_predicted_distance if not is_clear else pipe_length_m
    abs_error_m = abs(final_pred_distance - clog_dist_m) if not is_clear else 0.0

    # Severity Badge Color Tag
    badge_colors = {
        "Clear": "#238636",      # Green
        "Minor": "#388bfd",      # Blue
        "Moderate": "#d29922",   # Orange
        "Severe": "#da3633",     # Red
    }
    badge_color = badge_colors.get(ai_severity_label, "#8b949e")

    # 5. Main View: KPI Metrics Banner
    st.markdown("### 📊 Real-Time Diagnostic KPIs")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Ground Truth Clog Distance", f"{clog_dist_m:.2f} m" if not is_clear else "None (Clear)")
    with col2:
        st.metric("AI Predicted Distance", f"{final_pred_distance:.2f} m" if not is_clear else "Pipe Clear")
    with col3:
        st.metric("Localization MAE Error", f"{abs_error_m:.3f} m" if not is_clear else "0.000 m")
    with col4:
        st.markdown(
            f"""
            <div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 10px 14px; text-align: center;">
                <span style="color: #8b949e; font-size: 13px;">AI Severity Classification</span><br/>
                <span style="background-color: {badge_color}; color: white; padding: 4px 14px; border-radius: 12px; font-weight: bold; font-size: 16px; display: inline-block; margin-top: 4px;">
                    {ai_severity_label.upper()}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")

    # 6. 2D Pipe Map Visualization
    st.markdown("### 🗺️ Pipe Digital Twin 2D Layout")
    pipe_map_fig = plot_2d_pipe_map(
        pipe_length_m=pipe_length_m,
        pred_dist_m=final_pred_distance,
        true_dist_m=clog_dist_m,
        severity=severity_ratio,
        is_clear=is_clear,
    )
    st.pyplot(pipe_map_fig)
    plt.close(pipe_map_fig)

    # 7. Line Charts Panel
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("### 📈 Live Acoustic Signals")
        wave_fig = plot_acoustic_waveforms(t, s_tx, s_rec, metadata["tau_true_s"])
        st.pyplot(wave_fig)
        plt.close(wave_fig)

    with col_chart2:
        st.markdown("### 🔍 Cross-Correlation ToF Power Peak")
        corr_fig = plot_cross_correlation_power(
            dist_axis=profile["distance_axis_m"],
            power=profile["power"],
            pred_dist=final_pred_distance,
            true_dist=clog_dist_m,
            pipe_length=pipe_length_m,
        )
        st.pyplot(corr_fig)
        plt.close(corr_fig)

    # 8. Digital Twin Physical Parameters & AI Class Confidence
    with st.expander("🔬 Digital Twin Physics & AI Confidence Breakdown", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Acoustic Impedance & Wave Parameters")
            pipe_radius_m = 0.5  # Typical 1.0m diameter main sewer
            area_m2 = float(np.pi * (pipe_radius_m**2))
            z_pipe = compute_acoustic_impedance(area_m2)
            blocked_area_m2 = max(1e-4, area_m2 * (1.0 - severity_ratio))
            z_blocked = compute_acoustic_impedance(blocked_area_m2)
            r_coeff = compute_reflection_coefficient(z_pipe, z_blocked)
            attenuation = float(np.exp(-0.01 * clog_dist_m))

            st.write(f"• **Pipe Free Cross-Section Area ($A_1$):** `{area_m2:.3f} m²`")
            st.write(f"• **Blocked Area ($A_2$):** `{blocked_area_m2:.3f} m²`")
            st.write(f"• **Air Acoustic Impedance ($Z_1$):** `{z_pipe:.1f} Pa·s/m³`")
            st.write(f"• **Blockage Impedance ($Z_2$):** `{z_blocked:.1f} Pa·s/m³`")
            st.write(f"• **Pressure Reflection Coefficient ($R$):** `{r_coeff:+.3f}`")
            st.write(f"• **Round-Trip Delay ($\\tau$):** `{metadata['tau_true_s']:.4f} s`")
            st.write(f"• **Attenuation Decay ($e^{{-0.01 d}}$):** `{attenuation:.4f}`")

        with c2:
            st.markdown("#### 1D-CNN Stage Probability Distribution")
            for idx, (label, prob) in enumerate(zip(SEVERITY_CLASSES, probs)):
                st.write(f"**{label}** ({prob*100:.1f}%)")
                st.progress(float(prob))


if __name__ == "__main__":
    main()
