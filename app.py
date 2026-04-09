"""Streamlit app for uploading light curves and screening for transit-like patterns."""

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from exoplanet_research.config import build_config
from exoplanet_research.inference import analyze_uploaded_light_curve, load_uploaded_csv
from exoplanet_research.model_registry import MODEL_VERSIONS
from exoplanet_research.training_service import prepare_screening_model


st.set_page_config(
    page_title="Shepherd.Ai",
    page_icon="🪐",
    layout="wide",
)


st.markdown(
    """
    <style>
        html, body {
            height: 100%;
            overflow: hidden;
        }
        [data-testid="stAppViewContainer"] {
            height: 100vh;
            overflow-x: hidden !important;
            overflow-y: auto !important;
        }
        [data-testid="stMain"] {
            overflow: visible !important;
        }
        [data-testid="stMainBlockContainer"] {
            overflow: visible !important;
            padding-bottom: 3rem;
        }
        .stApp {
            background:
                radial-gradient(circle at 15% 18%, rgba(118, 171, 255, 0.24), transparent 18%),
                radial-gradient(circle at 82% 12%, rgba(255, 202, 122, 0.18), transparent 16%),
                radial-gradient(circle at 50% 0%, rgba(77, 115, 191, 0.18), transparent 30%),
            linear-gradient(180deg, #040914 0%, #08101d 42%, #0b1422 100%);
            color: #edf4ff;
            position: relative;
            overflow-x: hidden;
        }
        .stApp > header,
        .stApp > div,
        .stApp [data-testid="stAppViewContainer"],
        .stApp [data-testid="stMain"],
        .stApp [data-testid="stMainBlockContainer"] {
            position: relative;
            z-index: 2;
        }
        .stApp::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            opacity: 0.65;
            z-index: 0;
            background-image:
                radial-gradient(circle at 8% 14%, rgba(255,255,255,0.95) 0 1.2px, transparent 1.7px),
                radial-gradient(circle at 18% 72%, rgba(173,210,255,0.95) 0 1px, transparent 1.6px),
                radial-gradient(circle at 32% 28%, rgba(255,255,255,0.88) 0 1.1px, transparent 1.7px),
                radial-gradient(circle at 44% 84%, rgba(255,220,173,0.85) 0 1.2px, transparent 1.8px),
                radial-gradient(circle at 57% 19%, rgba(190,223,255,0.92) 0 1.3px, transparent 1.9px),
                radial-gradient(circle at 68% 63%, rgba(255,255,255,0.84) 0 1px, transparent 1.6px),
                radial-gradient(circle at 79% 34%, rgba(173,210,255,0.9) 0 1.15px, transparent 1.8px),
                radial-gradient(circle at 90% 78%, rgba(255,232,189,0.9) 0 1.2px, transparent 1.9px),
                radial-gradient(circle at 24% 46%, rgba(255,255,255,0.55) 0 0.8px, transparent 1.3px),
                radial-gradient(circle at 40% 60%, rgba(255,255,255,0.55) 0 0.8px, transparent 1.3px),
                radial-gradient(circle at 61% 48%, rgba(255,255,255,0.52) 0 0.8px, transparent 1.3px),
                radial-gradient(circle at 84% 22%, rgba(255,255,255,0.52) 0 0.8px, transparent 1.3px);
        }
        .stApp::after {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            opacity: 0.4;
            z-index: 0;
            background:
                linear-gradient(115deg, transparent 0 20%, rgba(136,184,255,0.06) 20% 20.4%, transparent 20.4% 100%),
                linear-gradient(35deg, transparent 0 68%, rgba(255,190,123,0.05) 68% 68.3%, transparent 68.3% 100%);
        }
        .stApp h1, .stApp h2, .stApp h3 {
            color: #f5f8ff;
        }
        .stApp p, .stApp li, .stApp label, .stApp div {
            color: #d8e6ff;
        }
        .hero-card {
            position: relative;
            padding: 2rem 2rem 1.8rem 2rem;
            border-radius: 28px;
            background:
                radial-gradient(circle at 82% 20%, rgba(255, 194, 112, 0.18), transparent 18%),
                radial-gradient(circle at 10% 10%, rgba(129, 180, 255, 0.20), transparent 20%),
                linear-gradient(135deg, rgba(9, 20, 37, 0.96), rgba(8, 14, 28, 0.92));
            border: 1px solid rgba(125, 174, 255, 0.24);
            box-shadow: 0 24px 70px rgba(0, 0, 0, 0.36);
            margin-bottom: 1.2rem;
            overflow: hidden;
        }
        .hero-card::before {
            content: "";
            position: absolute;
            inset: -2px;
            background:
                radial-gradient(circle at 12% 20%, rgba(117, 169, 255, 0.18), transparent 14%),
                radial-gradient(circle at 86% 24%, rgba(255, 193, 111, 0.16), transparent 12%);
            pointer-events: none;
        }
        .hero-grid {
            position: absolute;
            inset: 0;
            background-image:
                linear-gradient(rgba(116, 164, 237, 0.07) 1px, transparent 1px),
                linear-gradient(90deg, rgba(116, 164, 237, 0.07) 1px, transparent 1px);
            background-size: 36px 36px;
            mask-image: linear-gradient(180deg, rgba(0,0,0,0.55), transparent 92%);
            pointer-events: none;
        }
        .hero-orbit {
            position: absolute;
            right: -60px;
            top: -70px;
            width: 260px;
            height: 260px;
            border-radius: 50%;
            border: 1px solid rgba(129, 180, 255, 0.18);
            box-shadow:
                0 0 0 28px rgba(129, 180, 255, 0.05),
                0 0 0 56px rgba(129, 180, 255, 0.03);
            pointer-events: none;
        }
        .hero-kicker {
            text-transform: uppercase;
            letter-spacing: 0.18em;
            font-size: 0.78rem;
            color: #96bbff;
            margin-bottom: 0.6rem;
            position: relative;
            z-index: 1;
        }
        .hero-title {
            font-size: 2.8rem;
            font-weight: 700;
            line-height: 1.05;
            margin-bottom: 0.45rem;
            max-width: 14ch;
            position: relative;
            z-index: 1;
        }
        .hero-product {
            color: #f7fbff;
        }
        .hero-product-dot {
            color: #8ec5ff;
        }
        .hero-subhead {
            font-size: 1rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #8db7ff;
            margin-bottom: 0.7rem;
            position: relative;
            z-index: 1;
        }
        .hero-subtitle {
            font-size: 1rem;
            max-width: 42rem;
            color: #d2e4ff;
            position: relative;
            z-index: 1;
        }
        .info-chip-row {
            display: flex;
            gap: 0.75rem;
            flex-wrap: wrap;
            margin: 1.1rem 0 0.25rem 0;
            position: relative;
            z-index: 1;
        }
        .info-chip {
            padding: 0.5rem 0.8rem;
            border-radius: 999px;
            background: rgba(93, 140, 214, 0.14);
            border: 1px solid rgba(141, 183, 255, 0.20);
            color: #e8f2ff;
            font-size: 0.92rem;
            backdrop-filter: blur(6px);
        }
        .section-card {
            padding: 1rem 1.1rem;
            border-radius: 22px;
            background:
                linear-gradient(180deg, rgba(12, 23, 39, 0.86), rgba(8, 17, 31, 0.8));
            border: 1px solid rgba(125, 174, 255, 0.15);
            box-shadow: 0 18px 45px rgba(0, 0, 0, 0.20);
            margin-bottom: 1rem;
            backdrop-filter: blur(8px);
        }
        .result-good {
            padding: 1.1rem 1.2rem;
            border-radius: 22px;
            background:
                radial-gradient(circle at 85% 20%, rgba(147, 213, 255, 0.16), transparent 18%),
                linear-gradient(135deg, rgba(26, 66, 107, 0.96), rgba(13, 31, 58, 0.90));
            border: 1px solid rgba(118, 181, 255, 0.34);
            box-shadow: 0 18px 45px rgba(9, 18, 31, 0.30);
            margin-bottom: 1rem;
        }
        .result-calm {
            padding: 1.1rem 1.2rem;
            border-radius: 22px;
            background:
                linear-gradient(135deg, rgba(31, 49, 70, 0.94), rgba(19, 32, 49, 0.90));
            border: 1px solid rgba(153, 184, 224, 0.24);
            box-shadow: 0 18px 45px rgba(9, 18, 31, 0.24);
            margin-bottom: 1rem;
        }
        .result-title {
            font-size: 1.2rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .result-text {
            color: #d8e6ff;
            margin: 0;
        }
        [data-testid="stSidebar"] {
            background:
                radial-gradient(circle at top, rgba(92, 138, 214, 0.16), transparent 20%),
                linear-gradient(180deg, rgba(6, 14, 27, 0.98), rgba(10, 18, 33, 0.98));
            border-right: 1px solid rgba(125, 174, 255, 0.15);
            z-index: 3;
        }
        [data-testid="stMetric"] {
            background: rgba(11, 24, 40, 0.84);
            border: 1px solid rgba(125, 174, 255, 0.18);
            border-radius: 18px;
            padding: 0.9rem;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
        }
        .stButton > button {
            border-radius: 999px;
            border: 1px solid rgba(141, 183, 255, 0.30);
            background: linear-gradient(135deg, #7eaefb, #a9d3ff);
            color: #07111f;
            font-weight: 700;
            box-shadow: 0 10px 30px rgba(103, 158, 255, 0.28);
        }
        .stFileUploader {
            background: rgba(12, 23, 39, 0.74);
            border-radius: 18px;
            padding: 0.5rem 0.8rem 0.8rem 0.8rem;
            border: 1px solid rgba(125, 174, 255, 0.14);
            box-shadow: 0 12px 30px rgba(0,0,0,0.16);
        }
        .tiny-note {
            color: #9db6d7;
            font-size: 0.88rem;
            margin-top: -0.15rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model_bundle(version_id: str):
    """Train or load one Shepherd model version once per app session."""
    return prepare_screening_model(version_id)


def plot_uploaded_light_curve(dataframe: pd.DataFrame):
    """Build a point-by-point plot for the uploaded light curve."""
    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#06363f")
    ax.set_facecolor("#06363f")
    ax.scatter(
        dataframe["time"],
        dataframe["flux"],
        s=7,
        color="#d8fff8",
        alpha=0.78,
        edgecolors="none",
    )
    ax.set_title("Uploaded light curve points", color="#f2f7ff")
    ax.set_xlabel("Time", color="#d8e6ff")
    ax.set_ylabel("Flux", color="#d8e6ff")
    ax.grid(alpha=0.12, color="#b7fff4")
    ax.tick_params(colors="#d8e6ff")
    for spine in ax.spines.values():
        spine.set_color("#66c6d0")
    fig.tight_layout()
    return fig


def build_upload_signature(dataframe: pd.DataFrame, uploaded_name: str | None) -> tuple:
    """Create a lightweight signature so results persist only for the current upload."""
    return (
        uploaded_name or "uploaded_light_curve",
        int(len(dataframe)),
        round(float(dataframe["time"].min()), 6),
        round(float(dataframe["time"].max()), 6),
        round(float(dataframe["flux"].mean()), 6),
    )


def plot_size_comparison(result):
    """Draw a simple star-versus-object size comparison."""
    fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0d1828")
    ax.set_facecolor("#0d1828")
    ax.set_aspect("equal")

    star_radius = 1.0
    object_radius = max(0.03, min(result.estimated_radius_ratio, 0.95))

    star = plt.Circle((0.0, 0.0), star_radius, color="#f0c27a", alpha=0.95)
    object_circle = plt.Circle((2.35, 0.0), object_radius, color="#8ec5ff", alpha=0.95)
    glow = plt.Circle((0.0, 0.0), star_radius * 1.06, color="#ffdca3", alpha=0.12)

    ax.add_patch(glow)
    ax.add_patch(star)
    ax.add_patch(object_circle)

    ax.text(0.0, -1.28, "Host Star", ha="center", va="top", color="#edf4ff", fontsize=11)
    ax.text(
        2.35,
        -1.28,
        "Estimated Object",
        ha="center",
        va="top",
        color="#edf4ff",
        fontsize=11,
    )

    ax.text(
        2.35,
        1.22,
        f"Radius ratio approx. {result.estimated_radius_ratio:.3f}",
        ha="center",
        va="bottom",
        color="#bfd6f7",
        fontsize=10,
    )

    if result.estimated_object_radius_in_stellar_radii is not None:
        ax.text(
            2.35,
            1.38,
            f"approx. {result.estimated_object_radius_in_stellar_radii:.3f} stellar radii",
            ha="center",
            va="bottom",
            color="#bfd6f7",
            fontsize=10,
        )

    ax.set_xlim(-1.5, 3.6)
    ax.set_ylim(-1.55, 1.65)
    ax.axis("off")
    fig.tight_layout()
    return fig


def _conceptual_transit_profile(phases: np.ndarray, radius_ratio: float, depth: float) -> np.ndarray:
    """Build a simple conceptual transit curve for the animation panel."""
    object_radius = max(0.03, min(radius_ratio, 0.55))
    track_limit = 1.15 + object_radius
    x_positions = -track_limit + (2.0 * track_limit * phases)
    overlap = np.clip(1.0 - (np.abs(x_positions) / track_limit), 0.0, 1.0)
    return 1.0 - (max(depth, 0.002) * overlap**1.8)


def plot_transit_animation_frame(result, phase: float):
    """Draw a conceptual frame of the object crossing the star and the matching flux dip."""
    radius_ratio = max(0.03, min(result.estimated_radius_ratio, 0.55))
    depth = max(result.estimated_transit_depth, 0.002)
    track_limit = 1.15 + radius_ratio
    object_x = -track_limit + (2.0 * track_limit * phase)

    phases = np.linspace(0.0, 1.0, 160)
    curve_flux = _conceptual_transit_profile(phases, radius_ratio, depth)
    current_flux = float(_conceptual_transit_profile(np.array([phase]), radius_ratio, depth)[0])

    fig = plt.figure(figsize=(10, 5.2), facecolor="#0d1828")
    grid = fig.add_gridspec(2, 1, height_ratios=[1.35, 1], hspace=0.25)

    star_ax = fig.add_subplot(grid[0])
    curve_ax = fig.add_subplot(grid[1])

    for axis in (star_ax, curve_ax):
        axis.set_facecolor("#0d1828")

    star_glow = plt.Circle((0.0, 0.0), 1.08, color="#ffdca3", alpha=0.12)
    star_body = plt.Circle((0.0, 0.0), 1.0, color="#f0c27a", alpha=0.96)
    object_body = plt.Circle((object_x, 0.0), radius_ratio, color="#8ec5ff", alpha=0.98)

    star_ax.add_patch(star_glow)
    star_ax.add_patch(star_body)
    star_ax.add_patch(object_body)
    star_ax.plot(
        [-track_limit, track_limit],
        [0.0, 0.0],
        linestyle="--",
        linewidth=1.0,
        color="#6c85aa",
        alpha=0.45,
    )
    star_ax.text(
        0.0,
        -1.34,
        "Conceptual transit view",
        ha="center",
        va="top",
        color="#edf4ff",
        fontsize=11,
    )
    star_ax.set_xlim(-1.65, 1.65)
    star_ax.set_ylim(-1.42, 1.42)
    star_ax.set_aspect("equal")
    star_ax.axis("off")

    curve_ax.plot(phases, curve_flux, color="#88b8ff", linewidth=2.1)
    curve_ax.scatter([phase], [current_flux], color="#ffbf78", s=48, zorder=3)
    curve_ax.fill_between(phases, curve_flux, 1.0, color="#5b89c8", alpha=0.14)
    curve_ax.set_title("Conceptual flux response during the crossing", color="#f2f7ff")
    curve_ax.set_xlabel("Transit progress", color="#d8e6ff")
    curve_ax.set_ylabel("Relative flux", color="#d8e6ff")
    curve_ax.grid(alpha=0.15, color="#8eb8ff")
    curve_ax.tick_params(colors="#d8e6ff")
    for spine in curve_ax.spines.values():
        spine.set_color("#49688f")

    fig.tight_layout()
    return fig


def render_data_explainer(dataframe: pd.DataFrame) -> None:
    """Explain the uploaded light-curve data in beginner-friendly language."""
    time_start = float(dataframe["time"].min())
    time_end = float(dataframe["time"].max())
    flux_min = float(dataframe["flux"].min())
    flux_max = float(dataframe["flux"].max())
    flux_mean = float(dataframe["flux"].mean())

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("What This Data Means")
    st.write(
        "This file is a light curve. A light curve is a record of how a star's brightness changes over time."
    )
    st.write(
        "Each row in the table has two main values:"
    )
    st.write("- `time`: when the telescope recorded the measurement")
    st.write("- `flux`: how bright the star looked at that moment")
    st.write(
        "When these points are plotted as a line, small dips can sometimes suggest that something passed in front of the star, such as a possible planet transit."
    )
    st.write(
        f"In this uploaded file, the measurements run from `{time_start:.3f}` to `{time_end:.3f}`, "
        f"and the brightness values range from `{flux_min:.6f}` to `{flux_max:.6f}` with an average of `{flux_mean:.6f}`."
    )
    st.info(
        "Important: the graph does not prove that a planet is present. It only shows how the brightness changed. Shepherd.Ai looks for patterns in those changes that resemble transit signals."
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_result_explainer(result) -> None:
    """Explain the screening output in beginner-friendly language."""
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("How To Read This Result")
    st.write(
        f"Shepherd.Ai split the uploaded light curve into `{result.num_windows}` smaller sections called windows and scored each one for how transit-like it looks."
    )
    st.write(
        f"A window is called **flagged** when its transit score is at or above the current threshold of `{result.threshold:.3f}`."
    )
    st.write(
        "Flagged does **not** mean a planet was confirmed. It means that section of the light curve looks interesting enough to review more closely."
    )
    st.write(
        f"In this run, `{result.num_positive_windows}` out of `{result.num_windows}` windows were flagged, and the highest single-window score was `{result.max_window_probability:.3f}`."
    )
    st.write(
        f"Those flagged windows form about `{result.flagged_event_groups}` separate suspicious region(s) in the light curve."
    )
    st.write(
        f"The overall transit-likeness score of `{result.overall_probability:.3f}` is a broad summary of how suspicious the full uploaded signal looks on average."
    )
    st.info(
        "A higher score means the model thinks the pattern looks more like the transit examples it learned from. It does not mean the result is certain, and it should be treated as a screening clue rather than a final scientific answer."
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_candidate_snapshot(result) -> None:
    """Summarize the result like a short screening note."""
    if result.estimated_period_time is not None and result.flagged_event_groups >= 3:
        repeat_label = "Multiple repeat-like event groups were found in this upload."
    elif result.estimated_period_time is not None:
        repeat_label = "There is some repeat-like spacing, but the repeat evidence is still limited."
    else:
        repeat_label = "This upload does not yet show a clear repeating transit pattern."

    if result.estimated_baseline_variability > 0:
        depth_to_variability = (
            result.estimated_transit_depth / result.estimated_baseline_variability
        )
    else:
        depth_to_variability = 0.0

    if result.estimated_symmetry_score >= 0.75 and depth_to_variability >= 2.0:
        shape_label = "The strongest dip looks fairly clean and symmetric."
    elif result.estimated_symmetry_score >= 0.55 and depth_to_variability >= 1.2:
        shape_label = "The strongest dip looks somewhat transit-like, but not especially clean."
    else:
        shape_label = "The strongest dip is noisy or uneven, so it needs cautious review."

    if result.estimated_radius_ratio < 0.03:
        size_label = "If the signal is real, the object would be small relative to the star."
    elif result.estimated_radius_ratio < 0.08:
        size_label = "If the signal is real, the object would be modest in size relative to the star."
    elif result.estimated_radius_ratio < 0.15:
        size_label = "If the signal is real, the object would be fairly large relative to the star."
    else:
        size_label = "If the signal is real, the object would be very large relative to the star."

    if (
        result.max_window_probability >= result.threshold
        and result.estimated_period_time is not None
        and result.estimated_symmetry_score >= 0.6
    ):
        takeaway = "This looks like a stronger transit candidate worth closer follow-up review."
    elif result.max_window_probability >= result.threshold:
        takeaway = "This is an interesting screening hit, but it still needs more vetting."
    else:
        takeaway = "This upload does not currently look like a strong transit candidate."

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Candidate Review Snapshot")
    st.write(repeat_label)
    st.write(shape_label)
    st.write(size_label)
    st.info(takeaway)
    st.markdown("</div>", unsafe_allow_html=True)


def render_astrophysics_metrics(result) -> None:
    """Show simple estimated signal metrics in astrophysics-style language."""
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Estimated Signal Metrics")
    st.write(
        "These are approximate screening metrics from the strongest-looking part of the light curve."
    )
    st.write(
        "They are useful for interpretation, but they are not final scientific measurements."
    )
    left_col, right_col = st.columns(2)
    with left_col:
        st.metric("Estimated transit depth", f"{result.estimated_transit_depth:.6f}")
        st.metric("Estimated duration (points)", f"{result.estimated_duration_points}")
        st.metric("Half-depth width (time)", f"{result.estimated_half_depth_width_time:.3f}")
        st.metric("Estimated radius ratio", f"{result.estimated_radius_ratio:.3f}")
        st.metric("Symmetry score", f"{result.estimated_symmetry_score:.3f}")
    with right_col:
        st.metric("Estimated duration (time)", f"{result.estimated_duration_time:.3f}")
        st.metric("Estimated SNR", f"{result.estimated_snr:.2f}")
        st.metric("Baseline variability", f"{result.estimated_baseline_variability:.5f}")
        if result.estimated_object_radius_in_stellar_radii is not None:
            st.metric(
                "Estimated object radius",
                f"{result.estimated_object_radius_in_stellar_radii:.3f} stellar radii",
            )
        else:
            st.metric("Estimated object radius", "Add star radius")
    period_col, center_col = st.columns(2)
    with period_col:
        if result.estimated_period_time is not None:
            st.metric("Estimated period", f"{result.estimated_period_time:.3f}")
        else:
            st.metric("Estimated period", "Need repeats")
    with center_col:
        if result.estimated_transit_center_time is not None:
            st.metric("Transit center time", f"{result.estimated_transit_center_time:.3f}")
        else:
            st.metric("Transit center time", "Unavailable")
    st.caption(
        "Transit depth describes how far the brightness dips. Duration and half-depth width describe how wide the event appears. SNR estimates how strong that dip is compared with nearby noise. Radius ratio estimates the object's size compared to the star. Symmetry and baseline variability help judge how planet-like the event shape looks."
    )
    if result.stellar_radius_input is not None:
        st.write(
            f"Using the entered stellar radius of `{result.stellar_radius_input:.3f}` stellar radii, Shepherd.Ai estimates the transiting object could be about `{result.estimated_object_radius_in_stellar_radii:.3f}` stellar radii."
        )
    else:
        st.write(
            "If you know the star's radius, enter it below the model panel and Shepherd.Ai will estimate the object's size in stellar-radius units."
        )
    if result.estimated_period_time is not None:
        st.write(
            f"Shepherd.Ai found repeat-like flagged regions spaced by about `{result.estimated_period_time:.3f}` time units, which is the current rough period estimate."
        )
    else:
        st.write(
            "A period estimate needs at least two separate flagged event regions in the uploaded light curve, so some uploads will not produce one yet."
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_size_comparison(result) -> None:
    """Show a simple side-by-side star and object comparison graphic."""
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Estimated Size Comparison")
    st.write(
        "This is a simple illustration of the estimated object size compared with the host star."
    )
    st.write(
        "It is based on the transit depth and, if provided, your star-radius input."
    )
    st.pyplot(plot_size_comparison(result))
    st.caption(
        "This is a conceptual comparison image, not a real photograph of the system."
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_transit_animation(result) -> None:
    """Show a conceptual transit animation with a synchronized flux response."""
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Transit Animation")
    st.write(
        "This conceptual animation shows how a transiting object could cross the star while the observed flux dips at the same time."
    )
    st.write(
        "It is based on the estimated radius ratio and transit depth from this screening result, so it should be treated as an illustration rather than a literal reconstruction."
    )
    phase = st.slider(
        "Transit progress",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        key=f"transit_phase_{result.best_window_index}_{result.num_windows}",
        help="Move the slider to watch the estimated object cross the star and see the matching conceptual flux dip.",
    )
    st.pyplot(plot_transit_animation_frame(result, phase))
    st.caption(
        "The top panel shows a conceptual crossing. The lower panel shows the corresponding conceptual brightness change over that crossing."
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_transit_scale(result) -> None:
    """Show a visual scale for the overall transit-likeness score."""
    score_percent = max(0.0, min(100.0, result.overall_probability * 100.0))

    if result.overall_probability < 0.35:
        zone_label = "Low transit-like signal"
    elif result.overall_probability < 0.65:
        zone_label = "Moderate transit-like signal"
    else:
        zone_label = "High transit-like signal"

    st.markdown(
        f"""
        <div class="section-card">
            <div class="result-title">Transit-Likeness Scale</div>
            <p class="result-text">
                This scale shows where the uploaded light curve falls from low to high transit-likeness based on the model's overall score.
            </p>
            <div style="margin-top: 1rem;">
                <div style="
                    position: relative;
                    height: 18px;
                    border-radius: 999px;
                    background: linear-gradient(90deg, #35506e 0%, #7e9cc3 45%, #f0c27a 72%, #ff925c 100%);
                    border: 1px solid rgba(255,255,255,0.10);
                    overflow: visible;
                ">
                    <div style="
                        position: absolute;
                        left: calc({score_percent:.2f}% - 10px);
                        top: -8px;
                        width: 20px;
                        height: 34px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        color: #ffffff;
                        font-size: 1.1rem;
                        font-weight: 700;
                        text-shadow: 0 0 10px rgba(0,0,0,0.35);
                    ">▲</div>
                </div>
                <div style="
                    display: flex;
                    justify-content: space-between;
                    margin-top: 0.55rem;
                    font-size: 0.88rem;
                    color: #bfd6f7;
                ">
                    <span>Low</span>
                    <span>Moderate</span>
                    <span>High</span>
                </div>
            </div>
            <div style="margin-top: 0.95rem; color: #e9f3ff;">
                <strong>Current position:</strong> {score_percent:.1f}%<br/>
                <strong>Interpretation:</strong> {zone_label}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_transit_scale(result) -> None:
    """Show a visual scale for the overall transit-likeness score."""
    score_percent = max(0.0, min(100.0, result.overall_probability * 100.0))

    if result.overall_probability < 0.35:
        zone_label = "Low transit-like signal"
    elif result.overall_probability < 0.65:
        zone_label = "Moderate transit-like signal"
    else:
        zone_label = "High transit-like signal"

    st.markdown(
        f"""
        <div class="section-card">
            <div class="result-title">Transit-Likeness Scale</div>
            <p class="result-text">
                This scale shows where the uploaded light curve falls from low to high transit-likeness based on the model's overall score.
            </p>
            <div style="margin-top: 1rem;">
                <div style="
                    position: relative;
                    height: 18px;
                    border-radius: 999px;
                    background: linear-gradient(90deg, #35506e 0%, #7e9cc3 45%, #f0c27a 72%, #ff925c 100%);
                    border: 1px solid rgba(255,255,255,0.10);
                    overflow: visible;
                ">
                    <div style="
                        position: absolute;
                        left: calc({score_percent:.2f}% - 10px);
                        top: -8px;
                        width: 20px;
                        height: 34px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        color: #ffffff;
                        font-size: 1.1rem;
                        font-weight: 700;
                        text-shadow: 0 0 10px rgba(0,0,0,0.35);
                    ">^</div>
                </div>
                <div style="
                    display: flex;
                    justify-content: space-between;
                    margin-top: 0.55rem;
                    font-size: 0.88rem;
                    color: #bfd6f7;
                ">
                    <span>Low</span>
                    <span>Moderate</span>
                    <span>High</span>
                </div>
            </div>
            <div style="margin-top: 0.95rem; color: #e9f3ff;">
                <strong>Current position:</strong> {score_percent:.1f}%<br/>
                <strong>Interpretation:</strong> {zone_label}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_screening_results(result) -> None:
    """Render the stored screening results block."""
    result_class = "result-good" if "detected" in result.decision_label.lower() else "result-calm"
    st.markdown(
        f"""
        <div class="{result_class}">
            <div class="result-title">{result.decision_label}</div>
            <p class="result-text">{result.explanation}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_transit_scale(result)
    render_candidate_snapshot(result)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Result at a Glance")
    quick_cols = st.columns(4)
    with quick_cols[0]:
        st.metric("Overall transit-likeness", f"{result.overall_probability:.3f}")
    with quick_cols[1]:
        st.metric("Flagged windows", f"{result.num_positive_windows} / {result.num_windows}")
    with quick_cols[2]:
        st.metric("Peak window probability", f"{result.max_window_probability:.3f}")
    with quick_cols[3]:
        st.metric("Decision threshold", f"{result.threshold:.2f}")
    st.markdown("</div>", unsafe_allow_html=True)

    signal_tab, metrics_tab, animation_tab, window_tab, explanation_tab = st.tabs(
        [
            "Signal View",
            "Candidate Metrics",
            "Animation",
            "Window Details",
            "Explanation",
        ]
    )

    with signal_tab:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Window Probability Plot")
        fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0d1828")
        ax.set_facecolor("#0d1828")
        colors = np.where(
            result.window_summary["prediction"] == 1,
            "#8fd3ff",
            "#6b7f9c",
        )
        ax.scatter(
            result.window_summary["window_index"],
            result.window_summary["probability"],
            c=colors,
            s=34,
        )
        ax.plot(
            result.window_summary["window_index"],
            result.window_summary["probability"],
            color="#88b8ff",
            alpha=0.45,
        )
        ax.axhline(result.threshold, linestyle="--", color="#ffb56b", label="Decision threshold")
        ax.set_xlabel("Window index", color="#d8e6ff")
        ax.set_ylabel("Transit probability", color="#d8e6ff")
        ax.set_title("Transit-like score by window", color="#f2f7ff")
        ax.grid(alpha=0.14, color="#8eb8ff")
        ax.tick_params(colors="#d8e6ff")
        for spine in ax.spines.values():
            spine.set_color("#49688f")
        legend = ax.legend(frameon=False)
        for text in legend.get_texts():
            text.set_color("#d8e6ff")
        fig.tight_layout()
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    with metrics_tab:
        render_astrophysics_metrics(result)

    with animation_tab:
        render_transit_animation(result)

    with window_tab:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Window Results")
        display_table = result.window_summary.copy()
        display_table["probability"] = display_table["probability"].map(lambda value: round(value, 3))
        display_table["prediction"] = display_table["prediction"].map(
            lambda value: "Flagged" if value == 1 else "Clear"
        )
        st.write(
            "Showing the first 120 windows here so the page stays on one continuous scroll."
        )
        st.download_button(
            "Download Full Window Results",
            data=display_table.to_csv(index=False).encode("utf-8"),
            file_name="shepherd_window_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.table(display_table.head(120))
        st.markdown("</div>", unsafe_allow_html=True)

    with explanation_tab:
        render_result_explainer(result)


st.markdown(
    """
    <div class="hero-card">
        <div class="hero-grid"></div>
        <div class="hero-orbit"></div>
        <div class="hero-kicker">Summer Research Prototype</div>
        <div class="hero-title">
            <span class="hero-product">Shepherd</span><span class="hero-product-dot">.Ai</span>
        </div>
        <div class="hero-subhead">Exoplanet Transit Screener</div>
        <div class="hero-subtitle">
            Upload a light curve, preview its brightness signal, and scan for
            transit-like dips using the current Kepler-trained baseline model.
        </div>
        <div class="info-chip-row">
            <div class="info-chip">CSV upload</div>
            <div class="info-chip">Transit signal scan</div>
            <div class="info-chip">Kepler baseline model</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="tiny-note">This app is a screening tool, not an exoplanet confirmation system.</div>',
    unsafe_allow_html=True,
)

version_lookup = {version.display_name: version for version in MODEL_VERSIONS}
default_version_name = "SHEP 1.2"

model_status_placeholder = st.empty()
main_notice_placeholder = st.empty()

with model_status_placeholder.container():
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Initializing Mission Systems")
    st.write(
        "The interface is ready. The app is now loading the saved model and preparing the screening tools."
    )
    st.info("If this is the first launch, the model setup may take a moment.")
    st.markdown("</div>", unsafe_allow_html=True)

with st.spinner("Loading saved model and preparing the screening engine..."):
    selected_version_name = st.sidebar.selectbox(
        "Choose Shepherd model",
        options=list(version_lookup.keys()),
        index=list(version_lookup.keys()).index(default_version_name),
        help="Each Shepherd version can use a different trained model and mission focus.",
    )
    selected_version = version_lookup[selected_version_name]
    try:
        model_bundle = get_model_bundle(selected_version.version_id)
    except FileNotFoundError as exc:
        st.error(
            "This Shepherd.Ai deployment is missing one of its packaged model files. "
            "Train and save the model locally first, then include it in deployment."
        )
        st.code(str(exc))
        st.stop()

main_notice_placeholder.success(
    f"{model_bundle.version_name} ready. Source: {'saved file' if model_bundle.loaded_from_disk else 'fresh training run'}"
)

with st.sidebar:
    st.header("Mission Panel")
    st.write("1. Prepare a CSV file with `time` and `flux` columns.")
    st.write("2. Upload the file.")
    st.write("3. Click analyze to screen the data.")
    st.write("4. Review the probability scores and flagged windows.")

    st.header("Model Status")
    st.write(f"Version: `{model_bundle.version_name}`")
    st.write(f"Focus: `{model_bundle.version_description}`")
    st.write(f"Model source: `{'saved file' if model_bundle.loaded_from_disk else 'fresh training run'}`")
    st.write(f"Training stage: `{model_bundle.stage_name}`")
    st.write(f"Training data mode: `{model_bundle.data_message}`")
    st.write(f"Training windows: `{model_bundle.num_examples}`")
    st.write(f"Baseline test F1: `{model_bundle.metrics['f1_score']:.3f}`")
    st.write(f"Recommended threshold: `{model_bundle.recommended_threshold:.3f}`")

model_status_placeholder.empty()

if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_result_signature" not in st.session_state:
    st.session_state.last_result_signature = None
if "last_result_version" not in st.session_state:
    st.session_state.last_result_version = None
if "last_result_stellar_radius" not in st.session_state:
    st.session_state.last_result_stellar_radius = None

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Optional Star Information")
stellar_radius_input = st.number_input(
    "Host star radius (in stellar-radius units)",
    min_value=0.0,
    value=0.0,
    step=0.1,
    help="If you know the host star's radius, enter it here so Shepherd.Ai can estimate the transiting object's size. Leave it at 0 if unknown.",
)
st.caption(
    "This is optional. Without it, the app can still estimate the object's size relative to the star using the transit depth."
)
st.markdown("</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload a light-curve CSV",
    type=["csv"],
    help="The file should contain columns named `time` and `flux`.",
)

if uploaded_file is None:
    st.session_state.last_result = None
    st.session_state.last_result_signature = None
    st.session_state.last_result_version = None
    st.session_state.last_result_stellar_radius = None
    st.info(
        "No file uploaded yet. Upload a CSV with `time` and `flux` columns to begin screening."
    )
else:
    dataframe = load_uploaded_csv(uploaded_file)
    upload_signature = build_upload_signature(dataframe, getattr(uploaded_file, "name", None))
    render_data_explainer(dataframe)
    preview_col, plot_col = st.columns([1, 1.4], gap="large")

    with preview_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Signal Snapshot")
        st.write(f"Rows loaded: `{len(dataframe)}`")
        st.write(f"Time span: `{dataframe['time'].min():.3f}` to `{dataframe['time'].max():.3f}`")
        st.write(
            f"Flux range: `{dataframe['flux'].min():.6f}` to `{dataframe['flux'].max():.6f}`"
        )
        st.table(dataframe.head(10))
        st.markdown("</div>", unsafe_allow_html=True)

    with plot_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Light Curve View")
        st.pyplot(plot_uploaded_light_curve(dataframe))
        st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Analyze Light Curve", type="primary"):
        st.session_state.last_result = analyze_uploaded_light_curve(
            dataframe=dataframe,
            model=model_bundle.model,
            config=model_bundle.config,
            threshold=model_bundle.recommended_threshold,
            stellar_radius=stellar_radius_input if stellar_radius_input > 0 else None,
        )
        st.session_state.last_result_signature = upload_signature
        st.session_state.last_result_version = selected_version.version_id
        st.session_state.last_result_stellar_radius = (
            stellar_radius_input if stellar_radius_input > 0 else None
        )

    show_stored_result = (
        st.session_state.last_result is not None
        and st.session_state.last_result_signature == upload_signature
        and st.session_state.last_result_version == selected_version.version_id
        and st.session_state.last_result_stellar_radius
        == (stellar_radius_input if stellar_radius_input > 0 else None)
    )

    if show_stored_result:
        render_screening_results(st.session_state.last_result)
