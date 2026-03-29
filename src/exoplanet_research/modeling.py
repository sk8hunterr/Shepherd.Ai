"""Baseline machine learning model for transit classification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import ProjectConfig


@dataclass
class ModelArtifacts:
    """Objects produced during training."""

    model: Pipeline
    model_name: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray


def _safe_divide(numerator: float, denominator: float) -> float:
    """Avoid divide-by-zero in feature calculations."""
    if abs(denominator) < 1e-12:
        return 0.0
    return float(numerator / denominator)


def _longest_low_flux_run(flux_window: np.ndarray, threshold: float) -> float:
    """Measure how many consecutive points stay below a low-flux threshold."""
    below_threshold = flux_window < threshold
    longest_run = 0
    current_run = 0
    for is_low in below_threshold:
        if is_low:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0
    return float(longest_run)


def _largest_contiguous_region(mask: np.ndarray) -> tuple[int, int]:
    """Return the start and end indices of the longest True run."""
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


def _window_slope(values: np.ndarray) -> float:
    """Estimate a simple left-to-right slope for a small window segment."""
    if len(values) < 2:
        return 0.0
    x_axis = np.arange(len(values), dtype=float)
    slope, _ = np.polyfit(x_axis, values, deg=1)
    return float(slope)


def extract_transit_features(X: np.ndarray) -> np.ndarray:
    """Convert each raw window into richer transit-shape summary features."""
    feature_rows = []
    for flux_window in X:
        flux_window = np.asarray(flux_window, dtype=float)
        min_flux = float(np.min(flux_window))
        max_flux = float(np.max(flux_window))
        mean_flux = float(np.mean(flux_window))
        median_flux = float(np.median(flux_window))
        std_flux = float(np.std(flux_window))
        mad_flux = float(np.median(np.abs(flux_window - median_flux)))
        percentile_5 = float(np.percentile(flux_window, 5))
        percentile_25 = float(np.percentile(flux_window, 25))
        percentile_75 = float(np.percentile(flux_window, 75))
        lowest_points_mean = float(np.mean(np.sort(flux_window)[:10]))
        dip_strength = float(median_flux - min_flux)
        threshold = mean_flux - std_flux
        longest_run = _longest_low_flux_run(flux_window, threshold)
        iqr_flux = float(percentile_75 - percentile_25)

        min_index = int(np.argmin(flux_window))
        center_offset = abs(min_index - ((len(flux_window) - 1) / 2.0))
        normalized_center_offset = _safe_divide(center_offset, len(flux_window))

        low_flux_mask = flux_window <= percentile_5
        low_flux_fraction = float(np.mean(low_flux_mask))
        dip_start, dip_end = _largest_contiguous_region(low_flux_mask)
        dip_width = float(max(0, dip_end - dip_start))
        dip_width_fraction = _safe_divide(dip_width, len(flux_window))

        left_region = flux_window[:min_index] if min_index > 0 else flux_window[:1]
        right_region = flux_window[min_index + 1 :] if min_index + 1 < len(flux_window) else flux_window[-1:]
        left_mean = float(np.mean(left_region))
        right_mean = float(np.mean(right_region))
        symmetry_gap = abs(left_mean - right_mean)
        left_depth = left_mean - min_flux
        right_depth = right_mean - min_flux
        depth_balance = _safe_divide(min(left_depth, right_depth), max(left_depth, right_depth))

        local_radius = max(3, min(12, len(flux_window) // 12))
        local_start = max(0, min_index - local_radius)
        local_end = min(len(flux_window), min_index + local_radius + 1)
        local_segment = flux_window[local_start:local_end]
        local_noise = float(np.std(local_segment))
        depth_to_noise = _safe_divide(dip_strength, max(local_noise, std_flux, 1e-12))

        edge_window = max(2, min(10, len(flux_window) // 20))
        left_slope = _window_slope(flux_window[max(0, min_index - edge_window): min_index + 1])
        right_slope = _window_slope(flux_window[min_index: min(len(flux_window), min_index + edge_window + 1)])
        slope_contrast = abs(left_slope - right_slope)

        baseline_excluding_dip = np.concatenate(
            [flux_window[:local_start], flux_window[local_end:]]
        )
        if len(baseline_excluding_dip) == 0:
            baseline_excluding_dip = flux_window
        baseline_mean = float(np.mean(baseline_excluding_dip))
        baseline_std = float(np.std(baseline_excluding_dip))
        baseline_depth = baseline_mean - min_flux
        baseline_depth_to_noise = _safe_divide(baseline_depth, max(baseline_std, 1e-12))

        feature_rows.append(
            [
                min_flux,
                max_flux,
                mean_flux,
                median_flux,
                std_flux,
                mad_flux,
                percentile_5,
                percentile_25,
                percentile_75,
                iqr_flux,
                lowest_points_mean,
                dip_strength,
                longest_run,
                low_flux_fraction,
                dip_width,
                dip_width_fraction,
                normalized_center_offset,
                symmetry_gap,
                depth_balance,
                left_slope,
                right_slope,
                slope_contrast,
                local_noise,
                depth_to_noise,
                baseline_depth,
                baseline_depth_to_noise,
            ]
        )

    return np.asarray(feature_rows, dtype=float)


def _split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    config: ProjectConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the dataset once so multiple models can be compared fairly."""
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        raise ValueError(
            "The prepared dataset contains only one class, so the model cannot train. "
            "This usually means the target selection created only transit windows or only "
            "non-transit windows."
        )
    engineered_X = extract_transit_features(X)
    return train_test_split(
        engineered_X,
        y,
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=y,
    )


def _build_logistic_model(config: ProjectConfig) -> Pipeline:
    """Construct the logistic regression baseline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=config.logistic_max_iter,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def _build_random_forest_model(config: ProjectConfig) -> Pipeline:
    """Construct a simple random forest comparison model."""
    return Pipeline(
        steps=[
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=400,
                    random_state=config.random_seed,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )


def _fit_and_package(
    model: Pipeline,
    model_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> ModelArtifacts:
    """Train one model and package the outputs in a shared format."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return ModelArtifacts(
        model=model,
        model_name=model_name,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
    )


def train_baseline_model(X: np.ndarray, y: np.ndarray, config: ProjectConfig) -> ModelArtifacts:
    """Train the baseline logistic regression classifier on explainable transit features."""
    X_train, X_test, y_train, y_test = _split_dataset(X, y, config)
    model = _build_logistic_model(config)
    return _fit_and_package(model, "logistic_regression", X_train, X_test, y_train, y_test)


def train_model_comparison(
    X: np.ndarray,
    y: np.ndarray,
    config: ProjectConfig,
) -> list[ModelArtifacts]:
    """Train multiple simple models on the same split for fair comparison."""
    X_train, X_test, y_train, y_test = _split_dataset(X, y, config)
    models = [
        ("logistic_regression", _build_logistic_model(config)),
        ("random_forest", _build_random_forest_model(config)),
    ]
    return [
        _fit_and_package(model, model_name, X_train, X_test, y_train, y_test)
        for model_name, model in models
    ]
