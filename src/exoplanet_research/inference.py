"""Inference helpers for uploaded light-curve files."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ProjectConfig
from .data_loader import LoadedLightCurve
from .modeling import extract_transit_features
from .preprocessing import make_windows


@dataclass
class ScreeningResult:
    """Summary returned by the GUI after screening one uploaded light curve."""

    overall_probability: float
    max_window_probability: float
    num_windows: int
    num_positive_windows: int
    decision_label: str
    explanation: str
    threshold: float
    window_summary: pd.DataFrame
    estimated_transit_depth: float
    estimated_duration_time: float
    estimated_duration_points: int
    estimated_snr: float
    flagged_event_groups: int
    best_window_index: int


def load_uploaded_csv(file_like) -> pd.DataFrame:
    """Read an uploaded CSV and validate the required columns."""
    dataframe = pd.read_csv(file_like)
    expected_columns = {"time", "flux"}
    if not expected_columns.issubset(dataframe.columns):
        raise ValueError(
            "The uploaded CSV must contain columns named `time` and `flux`."
        )

    cleaned = dataframe.loc[:, ["time", "flux"]].copy()
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan).dropna()
    if cleaned.empty:
        raise ValueError("The uploaded file did not contain any valid rows after cleaning.")

    return cleaned


def _build_uploaded_lightcurve(dataframe: pd.DataFrame) -> LoadedLightCurve:
    """Convert the uploaded data into the same internal format used by the pipeline."""
    return LoadedLightCurve(
        target_name="uploaded_light_curve",
        time=dataframe["time"].to_numpy(dtype=float),
        flux=dataframe["flux"].to_numpy(dtype=float),
        source="uploaded_csv",
    )


def _largest_true_run(mask: np.ndarray) -> tuple[int, int]:
    """Return the longest contiguous True run as start/end indices."""
    best_start = 0
    best_end = 0
    current_start = None

    for index, is_true in enumerate(mask):
        if is_true and current_start is None:
            current_start = index
        elif not is_true and current_start is not None:
            if index - current_start > best_end - best_start:
                best_start = current_start
                best_end = index
            current_start = None

    if current_start is not None and len(mask) - current_start > best_end - best_start:
        best_start = current_start
        best_end = len(mask)

    return best_start, best_end


def _count_flagged_groups(predictions: np.ndarray) -> int:
    """Count distinct groups of flagged windows."""
    groups = 0
    previous_flag = 0
    for flag in predictions.astype(int):
        if flag == 1 and previous_flag == 0:
            groups += 1
        previous_flag = int(flag)
    return groups


def _estimate_transit_metrics(best_window, probability: float) -> tuple[float, float, int, float]:
    """Estimate depth, duration, and SNR from the strongest window."""
    flux_window = np.asarray(best_window.flux_window, dtype=float)
    time_window = np.asarray(best_window.time_window, dtype=float)

    median_flux = float(np.median(flux_window))
    local_noise = float(np.std(flux_window))
    low_flux_threshold = float(np.percentile(flux_window, 5))
    low_flux_mask = flux_window <= low_flux_threshold
    run_start, run_end = _largest_true_run(low_flux_mask)
    duration_points = max(1, run_end - run_start)

    if len(time_window) > 1:
        cadence = float(np.median(np.diff(time_window)))
    else:
        cadence = 0.0
    duration_time = duration_points * cadence

    min_flux = float(np.min(flux_window))
    depth = max(0.0, median_flux - min_flux)
    snr = depth / max(local_noise, 1e-12)

    if probability < 0.05:
        return 0.0, 0.0, 0, 0.0

    return depth, duration_time, duration_points, float(snr)


def analyze_uploaded_light_curve(
    dataframe: pd.DataFrame,
    model,
    config: ProjectConfig,
    threshold: float = 0.6,
) -> ScreeningResult:
    """Screen an uploaded light curve by applying the baseline model to each window."""
    lightcurve = _build_uploaded_lightcurve(dataframe)
    windows = make_windows(lightcurve, config)

    if not windows:
        raise ValueError(
            f"The uploaded light curve is too short. It needs at least {config.window_length} rows."
        )

    raw_windows = np.vstack([window.flux_window for window in windows])
    features = extract_transit_features(raw_windows)
    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    window_summary = pd.DataFrame(
        {
            "window_index": np.arange(len(windows)),
            "probability": probabilities,
            "prediction": predictions,
        }
    )

    overall_probability = float(np.mean(probabilities))
    max_probability = float(np.max(probabilities))
    num_positive_windows = int(np.sum(predictions))
    best_window_index = int(np.argmax(probabilities))
    flagged_event_groups = _count_flagged_groups(predictions)
    depth, duration_time, duration_points, snr = _estimate_transit_metrics(
        windows[best_window_index],
        max_probability,
    )

    if max_probability >= threshold:
        decision_label = "Transit-like pattern detected"
        explanation = (
            "At least one window crossed the transit-likeness threshold, so the app flagged "
            "this light curve for review."
        )
    else:
        decision_label = "No strong transit-like pattern detected"
        explanation = (
            "None of the windows crossed the current threshold, so the app did not find a "
            "strong transit-like dip."
        )

    return ScreeningResult(
        overall_probability=overall_probability,
        max_window_probability=max_probability,
        num_windows=len(windows),
        num_positive_windows=num_positive_windows,
        decision_label=decision_label,
        explanation=explanation,
        threshold=threshold,
        window_summary=window_summary,
        estimated_transit_depth=depth,
        estimated_duration_time=duration_time,
        estimated_duration_points=duration_points,
        estimated_snr=snr,
        flagged_event_groups=flagged_event_groups,
        best_window_index=best_window_index,
    )
