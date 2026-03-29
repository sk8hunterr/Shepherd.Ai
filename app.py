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
    """Build a simple plot for the uploaded light curve."""
    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0d1828")
    ax.set_facecolor("#0d1828")
    ax.plot(dataframe["time"], dataframe["flux"], linewidth=1.1, color="#88b8ff")
    ax.set_title("Uploaded light curve", color="#f2f7ff")
    ax.set_xlabel("Time", color="#d8e6ff")
    ax.set_ylabel("Flux", color="#d8e6ff")
    ax.grid(alpha=0.18, color="#8eb8ff")
    ax.tick_params(colors="#d8e6ff")
    for spine in ax.spines.values():
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
    with right_col:
        st.metric("Estimated duration (time)", f"{result.estimated_duration_time:.3f}")
        st.metric("Estimated SNR", f"{result.estimated_snr:.2f}")
    st.caption(
        "Transit depth describes how far the brightness dips. Duration describes how long the dip-like region lasts. SNR estimates how strong that dip is compared with nearby noise."
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
default_version_name = "SHEP 1.1"

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

uploaded_file = st.file_uploader(
    "Upload a light-curve CSV",
    type=["csv"],
    help="The file should contain columns named `time` and `flux`.",
)

if uploaded_file is None:
    st.info(
        "No file uploaded yet. Upload a CSV with `time` and `flux` columns to begin screening."
    )
else:
    dataframe = load_uploaded_csv(uploaded_file)
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
        st.dataframe(dataframe.head(10), use_container_width=True, height=280)
        st.markdown("</div>", unsafe_allow_html=True)

    with plot_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Light Curve View")
        st.pyplot(plot_uploaded_light_curve(dataframe))
        st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Analyze Light Curve", type="primary"):
        result = analyze_uploaded_light_curve(
            dataframe=dataframe,
            model=model_bundle.model,
            config=model_bundle.config,
            threshold=model_bundle.recommended_threshold,
        )

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
        render_result_explainer(result)
        render_transit_scale(result)
        render_astrophysics_metrics(result)

        left_col, right_col = st.columns(2)
        with left_col:
            st.metric("Overall transit-likeness", f"{result.overall_probability:.3f}")
            st.metric("Flagged windows", f"{result.num_positive_windows} / {result.num_windows}")
        with right_col:
            st.metric("Peak window probability", f"{result.max_window_probability:.3f}")
            st.metric("Decision threshold", f"{result.threshold:.2f}")

        chart_col, table_col = st.columns([1.25, 1], gap="large")
        with chart_col:
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

        with table_col:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Window Results")
            display_table = result.window_summary.copy()
            display_table["probability"] = display_table["probability"].map(lambda value: round(value, 3))
            display_table["prediction"] = display_table["prediction"].map(
                lambda value: "Flagged" if value == 1 else "Clear"
            )
            st.dataframe(display_table, use_container_width=True, height=390)
            st.markdown("</div>", unsafe_allow_html=True)
