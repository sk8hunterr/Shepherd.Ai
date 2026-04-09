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
    estimated_radius_ratio: float
    stellar_radius_input: float | None
    estimated_object_radius_in_stellar_radii: float | None
    estimated_period_time: float | None
    estimated_transit_center_time: float | None
    estimated_symmetry_score: float
    estimated_half_depth_width_time: float
    estimated_baseline_variability: float


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


def _safe_divide(numerator: float, denominator: float) -> float:
    """Avoid divide-by-zero in app-side estimates."""
    if abs(denominator) < 1e-12:
        return 0.0
    return float(numerator / denominator)


def _flagged_event_centers(
    windows,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> list[float]:
    """Estimate event-center times from contiguous groups of flagged windows."""
    centers: list[float] = []
    group_times: list[float] = []
    group_weights: list[float] = []

    for window, probability, prediction in zip(windows, probabilities, predictions, strict=False):
        window_center = float(np.mean(window.time_window))
        if int(prediction) == 1:
            group_times.append(window_center)
            group_weights.append(float(probability))
        elif group_times:
            centers.append(float(np.average(group_times, weights=group_weights)))
            group_times = []
            group_weights = []

    if group_times:
        centers.append(float(np.average(group_times, weights=group_weights)))

    return centers


def _estimate_period_from_centers(centers: list[float]) -> float | None:
    """Estimate a repeat period from multiple flagged-event centers."""
    if len(centers) < 2:
        return None
    diffs = np.diff(np.asarray(centers, dtype=float))
    valid_diffs = diffs[diffs > 0]
    if len(valid_diffs) == 0:
        return None
    return float(np.median(valid_diffs))


def _estimate_transit_metrics(best_window, probability: float) -> tuple[float, float, int, float]:
    """Estimate depth, duration, and SNR from the strongest window."""
    flux_window = np.asarray(best_window.flux_window, dtype=float)
    time_window = np.asarray(best_window.time_window, dtype=float)
    smoothing_width = min(5, len(flux_window))
    if smoothing_width % 2 == 0:
        smoothing_width = max(1, smoothing_width - 1)

    if smoothing_width > 1:
        smoothing_kernel = np.ones(smoothing_width, dtype=float) / smoothing_width
        smoothed_flux = np.convolve(flux_window, smoothing_kernel, mode="same")
    else:
        smoothed_flux = flux_window

    median_flux = float(np.median(flux_window))
    local_noise = float(np.std(flux_window))
    low_flux_threshold = float(np.percentile(smoothed_flux, 8))
    low_flux_mask = smoothed_flux <= low_flux_threshold
    run_start, run_end = _largest_true_run(low_flux_mask)
    duration_points = max(1, run_end - run_start)

    if len(time_window) > 1:
        cadence = float(np.median(np.diff(time_window)))
    else:
        cadence = 0.0
    duration_time = duration_points * cadence

    min_flux = float(np.min(smoothed_flux))
    depth = max(0.0, median_flux - min_flux)
    snr = depth / max(local_noise, 1e-12)

    if probability < 0.05:
        return 0.0, 0.0, 0, 0.0

    return depth, duration_time, duration_points, float(snr)


def _estimate_shape_metrics(best_window, depth: float) -> tuple[float | None, float, float, float]:
    """Estimate transit-center, symmetry, half-depth width, and baseline variability."""
    flux_window = np.asarray(best_window.flux_window, dtype=float)
    time_window = np.asarray(best_window.time_window, dtype=float)

    min_index = int(np.argmin(flux_window))
    transit_center = float(time_window[min_index])

    left_region = flux_window[:min_index] if min_index > 0 else flux_window[:1]
    right_region = flux_window[min_index + 1 :] if min_index + 1 < len(flux_window) else flux_window[-1:]
    left_depth = float(np.mean(left_region) - np.min(flux_window))
    right_depth = float(np.mean(right_region) - np.min(flux_window))
    symmetry_score = _safe_divide(min(left_depth, right_depth), max(left_depth, right_depth))

    baseline_level = float(np.median(flux_window))
    half_depth_threshold = baseline_level - (depth * 0.5)
    half_depth_mask = flux_window <= half_depth_threshold
    run_start, run_end = _largest_true_run(half_depth_mask)
    if len(time_window) > 1:
        cadence = float(np.median(np.diff(time_window)))
    else:
        cadence = 0.0
    half_depth_width_time = max(0, run_end - run_start) * cadence

    local_radius = max(3, min(12, len(flux_window) // 12))
    local_start = max(0, min_index - local_radius)
    local_end = min(len(flux_window), min_index + local_radius + 1)
    baseline_excluding_dip = np.concatenate([flux_window[:local_start], flux_window[local_end:]])
    if len(baseline_excluding_dip) == 0:
        baseline_excluding_dip = flux_window
    baseline_variability = float(np.std(baseline_excluding_dip))

    return transit_center, float(symmetry_score), float(half_depth_width_time), baseline_variability


def analyze_uploaded_light_curve(
    dataframe: pd.DataFrame,
    model,
    config: ProjectConfig,
    threshold: float = 0.6,
    stellar_radius: float | None = None,
) -> ScreeningResult:
    """Screen an uploaded light curve by applying the baseline model to each window."""
    lightcurve = _build_uploaded_lightcurve(dataframe)
    windows = make_windows(lightcurve, config)

    if not windows:
        raise ValueError(
            f"The uploaded light curve is too short. It needs at least {config.window_length} rows."
        )

    raw_windows = np.vstack([window.flux_window for window in windows])
    expected_feature_count = None
    if hasattr(model, "n_features_in_"):
        expected_feature_count = int(model.n_features_in_)
    elif hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            if hasattr(step, "n_features_in_"):
                expected_feature_count = int(step.n_features_in_)
                break
    time_windows = np.vstack([window.time_window for window in windows])
    target_names = np.array([window.target_name for window in windows], dtype=object)
    features = extract_transit_features(
        raw_windows,
        expected_feature_count=expected_feature_count,
        time_windows=time_windows,
        target_names=target_names,
    )
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
    flagged_centers = _flagged_event_centers(windows, probabilities, predictions)
    estimated_period = _estimate_period_from_centers(flagged_centers)
    depth, duration_time, duration_points, snr = _estimate_transit_metrics(
        windows[best_window_index],
        max_probability,
    )
    transit_center_time, symmetry_score, half_depth_width_time, baseline_variability = _estimate_shape_metrics(
        windows[best_window_index],
        depth,
    )
    radius_ratio = float(np.sqrt(max(depth, 0.0)))
    estimated_object_radius = None
    if stellar_radius is not None and stellar_radius > 0:
        estimated_object_radius = float(stellar_radius * radius_ratio)

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
        estimated_radius_ratio=radius_ratio,
        stellar_radius_input=stellar_radius,
        estimated_object_radius_in_stellar_radii=estimated_object_radius,
        estimated_period_time=estimated_period,
        estimated_transit_center_time=transit_center_time,
        estimated_symmetry_score=symmetry_score,
        estimated_half_depth_width_time=half_depth_width_time,
        estimated_baseline_variability=baseline_variability,
    )
