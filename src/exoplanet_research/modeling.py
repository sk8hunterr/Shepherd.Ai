"""Baseline machine learning model for transit classification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
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
    train_target_names: np.ndarray
    test_target_names: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray


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
    regions = _true_regions(mask)
    if not regions:
        return 0, 0
    return max(regions, key=lambda region: region[1] - region[0])


def _true_regions(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return all contiguous True runs as (start, end) index pairs."""
    regions: list[tuple[int, int]] = []
    current_start = None

    for index, is_true in enumerate(mask):
        if is_true and current_start is None:
            current_start = index
        elif not is_true and current_start is not None:
            regions.append((current_start, index))
            current_start = None

    if current_start is not None:
        regions.append((current_start, len(mask)))

    return regions


def _window_slope(values: np.ndarray) -> float:
    """Estimate a simple left-to-right slope for a small window segment."""
    if len(values) < 2:
        return 0.0
    x_axis = np.arange(len(values), dtype=float)
    slope, _ = np.polyfit(x_axis, values, deg=1)
    return float(slope)


def _count_true_runs(mask: np.ndarray) -> float:
    """Count the number of contiguous True regions in a mask."""
    runs = 0
    previous = False
    for value in mask:
        is_true = bool(value)
        if is_true and not previous:
            runs += 1
        previous = is_true
    return float(runs)


def _safe_skew(values: np.ndarray) -> float:
    """Compute a simple skewness estimate without external dependencies."""
    std = float(np.std(values))
    if std < 1e-12:
        return 0.0
    centered = values - float(np.mean(values))
    return float(np.mean((centered / std) ** 3))


def _safe_kurtosis(values: np.ndarray) -> float:
    """Compute excess kurtosis without external dependencies."""
    std = float(np.std(values))
    if std < 1e-12:
        return 0.0
    centered = values - float(np.mean(values))
    return float(np.mean((centered / std) ** 4) - 3.0)


def _low_flux_group_centers(flux_window: np.ndarray, threshold: float) -> list[float]:
    """Find the center positions of low-flux groups inside a window."""
    mask = flux_window <= threshold
    centers: list[float] = []
    current_start = None
    for index, is_low in enumerate(mask):
        if is_low and current_start is None:
            current_start = index
        elif not is_low and current_start is not None:
            centers.append((current_start + index - 1) / 2.0)
            current_start = None
    if current_start is not None:
        centers.append((current_start + len(mask) - 1) / 2.0)
    return centers


def _periodicity_hint(flux_window: np.ndarray) -> tuple[float, float]:
    """Estimate a simple repeat-pattern hint from autocorrelation peaks."""
    centered = np.asarray(flux_window, dtype=float) - float(np.mean(flux_window))
    variance = float(np.var(centered))
    if variance < 1e-12 or len(centered) < 20:
        return 0.0, 0.0

    autocorr = np.correlate(centered, centered, mode="full")[len(centered) - 1 :]
    autocorr = autocorr / max(autocorr[0], 1e-12)
    min_lag = max(5, len(centered) // 20)
    max_lag = max(min_lag + 1, len(centered) // 2)
    candidate = autocorr[min_lag:max_lag]
    if len(candidate) == 0:
        return 0.0, 0.0

    best_index = int(np.argmax(candidate))
    best_value = float(candidate[best_index])
    best_lag = float(min_lag + best_index)
    return best_value, _safe_divide(best_lag, len(centered))


def _prepare_time_windows(X: np.ndarray, time_windows: np.ndarray | None) -> np.ndarray:
    """Return usable time windows, falling back to point indices when needed."""
    if time_windows is not None:
        prepared = np.asarray(time_windows, dtype=float)
        if prepared.shape == X.shape:
            return prepared

    point_axis = np.arange(X.shape[1], dtype=float)
    return np.tile(point_axis, (X.shape[0], 1))


def _percentile_ranks(values: np.ndarray) -> np.ndarray:
    """Convert values into stable 0-1 percentile ranks within one target."""
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return np.zeros(len(values), dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks / float(len(values) - 1)


def _extract_event_context_features(
    X: np.ndarray,
    time_windows: np.ndarray | None = None,
    target_names: np.ndarray | None = None,
) -> np.ndarray:
    """Add time-aware and target-level repeat-event context features.

    These features are intentionally more candidate-like than the early window-only
    summaries. They ask whether the strongest dip in this window belongs to a
    repeatable pattern across the same target's observed windows.
    """
    time_data = _prepare_time_windows(X, time_windows)
    target_values = (
        np.asarray(target_names, dtype=object)
        if target_names is not None and len(target_names) == len(X)
        else np.full(len(X), "single_target_context", dtype=object)
    )

    depths = np.zeros(len(X), dtype=float)
    snrs = np.zeros(len(X), dtype=float)
    dip_times = np.zeros(len(X), dtype=float)
    window_spans = np.zeros(len(X), dtype=float)
    cadence_medians = np.zeros(len(X), dtype=float)
    cadence_jitters = np.zeros(len(X), dtype=float)
    dip_center_fractions = np.zeros(len(X), dtype=float)

    for index, (flux_window, time_window) in enumerate(zip(X, time_data, strict=False)):
        flux_window = np.asarray(flux_window, dtype=float)
        time_window = np.asarray(time_window, dtype=float)
        min_index = int(np.argmin(flux_window))
        baseline = float(np.median(flux_window))
        local_depth = max(0.0, baseline - float(flux_window[min_index]))
        noise = float(np.std(flux_window - baseline))

        if len(time_window) > 1:
            diffs = np.diff(time_window)
            positive_diffs = diffs[diffs > 0]
            cadence = float(np.median(positive_diffs)) if len(positive_diffs) else 1.0
            cadence_jitter = _safe_divide(float(np.std(positive_diffs)), max(cadence, 1e-12))
            span = max(float(time_window[-1] - time_window[0]), cadence)
        else:
            cadence = 1.0
            cadence_jitter = 0.0
            span = 1.0

        depths[index] = local_depth
        snrs[index] = _safe_divide(local_depth, max(noise, 1e-12))
        dip_times[index] = float(time_window[min_index])
        window_spans[index] = span
        cadence_medians[index] = cadence
        cadence_jitters[index] = cadence_jitter
        dip_center_fractions[index] = _safe_divide(float(min_index), max(len(flux_window) - 1, 1))

    context_rows = np.zeros((len(X), 17), dtype=float)

    for target in sorted(set(target_values.tolist())):
        target_indices = np.where(target_values == target)[0]
        target_depths = depths[target_indices]
        target_snrs = snrs[target_indices]
        target_dip_times = dip_times[target_indices]
        target_time_span = max(float(np.max(target_dip_times) - np.min(target_dip_times)), 1e-12)

        depth_rank = _percentile_ranks(target_depths)
        snr_rank = _percentile_ranks(target_snrs)
        depth_median = float(np.median(target_depths))
        depth_mad = float(np.median(np.abs(target_depths - depth_median)))
        depth_threshold = float(np.percentile(target_depths, 75))
        snr_threshold = max(0.75, float(np.percentile(target_snrs, 55)))
        candidate_mask = (target_depths >= depth_threshold) & (target_snrs >= snr_threshold)

        if int(np.sum(candidate_mask)) < 2 and len(target_indices) >= 2:
            strongest = np.argsort(target_depths)[-2:]
            candidate_mask[strongest] = True

        candidate_depths = target_depths[candidate_mask]
        candidate_snrs = target_snrs[candidate_mask]
        candidate_times = np.sort(target_dip_times[candidate_mask])
        candidate_count = int(len(candidate_times))
        candidate_fraction = _safe_divide(candidate_count, len(target_indices))

        if candidate_count >= 2:
            candidate_diffs = np.diff(candidate_times)
            candidate_diffs = candidate_diffs[candidate_diffs > 0]
        else:
            candidate_diffs = np.array([], dtype=float)

        if len(candidate_diffs):
            median_period = float(np.median(candidate_diffs))
            period_regularity = 1.0 / (
                1.0 + _safe_divide(float(np.std(candidate_diffs)), max(median_period, 1e-12))
            )
            median_period_fraction = _safe_divide(median_period, target_time_span)
        else:
            median_period = 0.0
            period_regularity = 0.0
            median_period_fraction = 0.0

        if len(candidate_depths):
            candidate_depth_median = float(np.median(candidate_depths))
            candidate_snr_median = float(np.median(candidate_snrs))
            depth_consistency = 1.0 / (
                1.0
                + _safe_divide(
                    float(np.std(candidate_depths)),
                    max(candidate_depth_median, 1e-12),
                )
            )
        else:
            candidate_depth_median = 0.0
            candidate_snr_median = 0.0
            depth_consistency = 0.0

        for local_position, global_index in enumerate(target_indices):
            own_time = target_dip_times[local_position]
            if candidate_count >= 2:
                gaps = np.abs(candidate_times - own_time)
                positive_gaps = gaps[gaps > 1e-12]
                nearest_gap = float(np.min(positive_gaps)) if len(positive_gaps) else median_period
            else:
                nearest_gap = 0.0

            depth_separation = _safe_divide(
                target_depths[local_position] - depth_median,
                max(depth_mad, 1e-12),
            )

            context_rows[global_index] = [
                window_spans[global_index],
                cadence_medians[global_index],
                cadence_jitters[global_index],
                dip_center_fractions[global_index],
                depth_rank[local_position],
                snr_rank[local_position],
                float(np.log1p(candidate_count)),
                candidate_fraction,
                float(np.log1p(target_time_span)),
                median_period_fraction,
                period_regularity,
                depth_consistency,
                _safe_divide(nearest_gap, target_time_span),
                1.0 if candidate_count >= 2 else 0.0,
                depth_separation,
                candidate_snr_median,
                candidate_depth_median,
            ]

    return context_rows


def _merge_candidate_events(
    event_times: np.ndarray,
    event_depths: np.ndarray,
    event_snrs: np.ndarray,
    merge_gap: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge nearby high-dip windows into one candidate event."""
    if len(event_times) == 0:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    order = np.argsort(event_times)
    sorted_times = np.asarray(event_times, dtype=float)[order]
    sorted_depths = np.asarray(event_depths, dtype=float)[order]
    sorted_snrs = np.asarray(event_snrs, dtype=float)[order]

    merged_times: list[float] = []
    merged_depths: list[float] = []
    merged_snrs: list[float] = []
    group_times = [float(sorted_times[0])]
    group_depths = [float(sorted_depths[0])]
    group_snrs = [float(sorted_snrs[0])]

    for time_value, depth_value, snr_value in zip(
        sorted_times[1:],
        sorted_depths[1:],
        sorted_snrs[1:],
        strict=False,
    ):
        if float(time_value) - group_times[-1] <= merge_gap:
            group_times.append(float(time_value))
            group_depths.append(float(depth_value))
            group_snrs.append(float(snr_value))
            continue

        weights = np.asarray(group_depths, dtype=float) * np.maximum(group_snrs, 1e-6)
        if float(np.sum(weights)) <= 0:
            weights = np.ones(len(group_times), dtype=float)
        merged_times.append(float(np.average(group_times, weights=weights)))
        merged_depths.append(float(np.max(group_depths)))
        merged_snrs.append(float(np.max(group_snrs)))
        group_times = [float(time_value)]
        group_depths = [float(depth_value)]
        group_snrs = [float(snr_value)]

    weights = np.asarray(group_depths, dtype=float) * np.maximum(group_snrs, 1e-6)
    if float(np.sum(weights)) <= 0:
        weights = np.ones(len(group_times), dtype=float)
    merged_times.append(float(np.average(group_times, weights=weights)))
    merged_depths.append(float(np.max(group_depths)))
    merged_snrs.append(float(np.max(group_snrs)))

    return (
        np.asarray(merged_times, dtype=float),
        np.asarray(merged_depths, dtype=float),
        np.asarray(merged_snrs, dtype=float),
    )


def _phase_distance(phase_values: np.ndarray, target_phase: float) -> np.ndarray:
    """Return circular phase distance from one target phase."""
    raw_distance = np.abs(np.asarray(phase_values, dtype=float) - target_phase)
    return np.minimum(raw_distance, 1.0 - raw_distance)


def _score_period_candidate(
    event_times: np.ndarray,
    period: float,
    reference_time: float,
    tolerance_fraction: float = 0.08,
) -> tuple[float, float, float]:
    """Score how well event times align to a candidate period."""
    if period <= 0 or len(event_times) < 3:
        return 0.0, 0.0, 1.0

    phases = ((event_times - reference_time) / period) % 1.0
    residuals = _phase_distance(phases, 0.0)
    inlier_mask = residuals <= tolerance_fraction
    inlier_fraction = float(np.mean(inlier_mask))
    residual_std_fraction = float(np.std(residuals[inlier_mask])) if np.any(inlier_mask) else 1.0
    alignment_score = inlier_fraction / (1.0 + residual_std_fraction / max(tolerance_fraction, 1e-12))
    return float(alignment_score), inlier_fraction, residual_std_fraction


def _estimate_candidate_period(event_times: np.ndarray) -> tuple[float, float, float, float]:
    """Estimate a repeat period from merged candidate-event centers."""
    event_times = np.sort(np.asarray(event_times, dtype=float))
    if len(event_times) < 3:
        return 0.0, 0.0, 0.0, 1.0

    time_span = max(float(event_times[-1] - event_times[0]), 1e-12)
    pairwise_periods: list[float] = []
    for start_index in range(len(event_times)):
        for end_index in range(start_index + 1, len(event_times)):
            gap = float(event_times[end_index] - event_times[start_index])
            if gap <= 0:
                continue
            # Include harmonic candidates because missed transits can make a true
            # period appear as a multiple of the observed event spacing.
            for harmonic in (1, 2, 3):
                candidate_period = gap / harmonic
                if candidate_period >= max(time_span / 80.0, 1e-6):
                    pairwise_periods.append(candidate_period)

    if not pairwise_periods:
        return 0.0, 0.0, 0.0, 1.0

    candidate_periods = np.unique(np.round(pairwise_periods, 6))
    reference_time = float(event_times[0])
    best_period = 0.0
    best_score = -1.0
    best_inlier_fraction = 0.0
    best_residual_std_fraction = 1.0

    for period in candidate_periods:
        score, inlier_fraction, residual_std_fraction = _score_period_candidate(
            event_times,
            float(period),
            reference_time,
        )
        # Prefer shorter periods only when the alignment score is meaningfully tied.
        tie_breaker = -_safe_divide(float(period), time_span) * 0.005
        score_with_tie_break = score + tie_breaker
        if score_with_tie_break > best_score:
            best_period = float(period)
            best_score = float(score_with_tie_break)
            best_inlier_fraction = inlier_fraction
            best_residual_std_fraction = residual_std_fraction

    return best_period, max(0.0, best_score), best_inlier_fraction, best_residual_std_fraction


def _extract_candidate_event_features(
    X: np.ndarray,
    time_windows: np.ndarray | None = None,
    target_names: np.ndarray | None = None,
) -> np.ndarray:
    """Build target-level candidate-event features for harder transit vetting."""
    time_data = _prepare_time_windows(X, time_windows)
    target_values = (
        np.asarray(target_names, dtype=object)
        if target_names is not None and len(target_names) == len(X)
        else np.full(len(X), "single_target_context", dtype=object)
    )

    depths = np.zeros(len(X), dtype=float)
    snrs = np.zeros(len(X), dtype=float)
    dip_times = np.zeros(len(X), dtype=float)
    window_spans = np.zeros(len(X), dtype=float)

    for index, (flux_window, time_window) in enumerate(zip(X, time_data, strict=False)):
        flux_window = np.asarray(flux_window, dtype=float)
        time_window = np.asarray(time_window, dtype=float)
        min_index = int(np.argmin(flux_window))
        baseline = float(np.median(flux_window))
        depth = max(0.0, baseline - float(flux_window[min_index]))
        robust_noise = float(
            np.median(np.abs(flux_window - baseline)) * 1.4826
        )
        standard_noise = float(np.std(flux_window - baseline))

        if len(time_window) > 1:
            span = max(float(time_window[-1] - time_window[0]), 1e-12)
        else:
            span = 1.0

        depths[index] = depth
        snrs[index] = _safe_divide(depth, max(robust_noise, standard_noise, 1e-12))
        dip_times[index] = float(time_window[min_index])
        window_spans[index] = span

    event_rows = np.zeros((len(X), 25), dtype=float)

    for target in sorted(set(target_values.tolist())):
        target_indices = np.where(target_values == target)[0]
        target_depths = depths[target_indices]
        target_snrs = snrs[target_indices]
        target_dip_times = dip_times[target_indices]
        target_window_spans = window_spans[target_indices]
        target_time_span = max(float(np.max(target_dip_times) - np.min(target_dip_times)), 1e-12)

        depth_threshold = max(float(np.percentile(target_depths, 70)), float(np.median(target_depths)))
        snr_threshold = max(0.75, float(np.percentile(target_snrs, 50)))
        candidate_mask = (target_depths >= depth_threshold) & (target_snrs >= snr_threshold)
        if int(np.sum(candidate_mask)) < 3 and len(target_indices) >= 3:
            strongest = np.argsort(target_depths)[-3:]
            candidate_mask[strongest] = True

        merge_gap = max(float(np.median(target_window_spans)) * 0.8, target_time_span / 500.0)
        event_times, event_depths, event_snrs = _merge_candidate_events(
            target_dip_times[candidate_mask],
            target_depths[candidate_mask],
            target_snrs[candidate_mask],
            merge_gap=merge_gap,
        )

        event_count = int(len(event_times))
        event_fraction = _safe_divide(event_count, len(target_indices))
        period, period_score, inlier_fraction, residual_std_fraction = _estimate_candidate_period(
            event_times,
        )
        period_fraction = _safe_divide(period, target_time_span)
        cycles_observed = _safe_divide(target_time_span, max(period, 1e-12)) if period > 0 else 0.0

        if event_count:
            event_depth_median = float(np.median(event_depths))
            event_depth_mad = float(np.median(np.abs(event_depths - event_depth_median)))
            event_snr_median = float(np.median(event_snrs))
            event_snr_mad = float(np.median(np.abs(event_snrs - event_snr_median)))
            event_depth_consistency = 1.0 / (
                1.0 + _safe_divide(float(np.std(event_depths)), max(event_depth_median, 1e-12))
            )
            event_snr_consistency = 1.0 / (
                1.0 + _safe_divide(float(np.std(event_snrs)), max(event_snr_median, 1e-12))
            )
        else:
            event_depth_median = 0.0
            event_depth_mad = 0.0
            event_snr_median = 0.0
            event_snr_mad = 0.0
            event_depth_consistency = 0.0
            event_snr_consistency = 0.0

        if event_count >= 2:
            odd_depths = event_depths[::2]
            even_depths = event_depths[1::2]
            odd_mean = float(np.mean(odd_depths))
            even_mean = float(np.mean(even_depths)) if len(even_depths) else odd_mean
            odd_even_depth_balance = _safe_divide(
                min(odd_mean, even_mean),
                max(odd_mean, even_mean, 1e-12),
            )
            event_depth_trend = abs(_window_slope(event_depths))
        else:
            odd_even_depth_balance = 1.0
            event_depth_trend = 0.0

        if period > 0:
            reference_time = float(event_times[0]) if event_count else float(np.min(target_dip_times))
            target_phases = ((target_dip_times - reference_time) / period) % 1.0
            primary_distance = _phase_distance(target_phases, 0.0)
            secondary_distance = _phase_distance(target_phases, 0.5)
            phase_tolerance = 0.08
            primary_mask = primary_distance <= phase_tolerance
            secondary_mask = secondary_distance <= phase_tolerance
            primary_depth = (
                float(np.percentile(target_depths[primary_mask], 90))
                if np.any(primary_mask)
                else event_depth_median
            )
            secondary_depth = (
                float(np.percentile(target_depths[secondary_mask], 90))
                if np.any(secondary_mask)
                else 0.0
            )
            secondary_to_primary_depth = _safe_divide(secondary_depth, max(primary_depth, 1e-12))
            secondary_event_fraction = float(np.mean(secondary_mask))
        else:
            primary_distance = np.ones(len(target_indices), dtype=float)
            secondary_distance = np.ones(len(target_indices), dtype=float)
            secondary_to_primary_depth = 0.0
            secondary_event_fraction = 0.0

        event_depth_ranks = _percentile_ranks(target_depths)
        event_snr_ranks = _percentile_ranks(target_snrs)

        for local_position, global_index in enumerate(target_indices):
            own_time = target_dip_times[local_position]
            if event_count:
                nearest_gap = float(np.min(np.abs(event_times - own_time)))
            else:
                nearest_gap = target_time_span

            event_rows[global_index] = [
                float(np.log1p(event_count)),
                event_fraction,
                float(np.log1p(target_time_span)),
                period_fraction,
                period_score,
                inlier_fraction,
                residual_std_fraction,
                event_depth_consistency,
                event_snr_consistency,
                odd_even_depth_balance,
                event_depth_trend,
                secondary_to_primary_depth,
                secondary_event_fraction,
                _safe_divide(nearest_gap, target_time_span),
                float(primary_distance[local_position]),
                float(secondary_distance[local_position]),
                1.0 if float(primary_distance[local_position]) <= 0.08 else 0.0,
                1.0 if float(secondary_distance[local_position]) <= 0.08 else 0.0,
                _safe_divide(
                    target_depths[local_position] - event_depth_median,
                    max(event_depth_mad, 1e-12),
                ),
                _safe_divide(
                    target_snrs[local_position] - event_snr_median,
                    max(event_snr_mad, 1e-12),
                ),
                float(event_depth_ranks[local_position]),
                float(event_snr_ranks[local_position]),
                inlier_fraction if float(primary_distance[local_position]) <= 0.08 else 0.0,
                cycles_observed,
                _safe_divide(event_depth_mad, max(event_depth_median, 1e-12)),
            ]

    return event_rows


def _extract_context_aware_transit_features(
    X: np.ndarray,
    time_windows: np.ndarray | None = None,
    target_names: np.ndarray | None = None,
) -> np.ndarray:
    """Return the 71-feature context-aware detector vector from SHEP 1.1.5."""
    base_features = _extract_eclipse_aware_transit_features(X)
    context_features = _extract_event_context_features(
        X,
        time_windows=time_windows,
        target_names=target_names,
    )
    return np.hstack([base_features, context_features])


def _extract_legacy_transit_features(X: np.ndarray) -> np.ndarray:
    """Return the original 26-feature transit summary vector for saved-model compatibility."""
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


def _extract_enhanced_transit_features(X: np.ndarray) -> np.ndarray:
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
        skew_flux = _safe_skew(flux_window)
        kurt_flux = _safe_kurtosis(flux_window)

        min_index = int(np.argmin(flux_window))
        center_offset = abs(min_index - ((len(flux_window) - 1) / 2.0))
        normalized_center_offset = _safe_divide(center_offset, len(flux_window))

        low_flux_mask = flux_window <= percentile_5
        low_flux_fraction = float(np.mean(low_flux_mask))
        low_flux_run_count = _count_true_runs(low_flux_mask)
        low_flux_group_centers = _low_flux_group_centers(flux_window, percentile_5)
        dip_start, dip_end = _largest_contiguous_region(low_flux_mask)
        dip_width = float(max(0, dip_end - dip_start))
        dip_width_fraction = _safe_divide(dip_width, len(flux_window))
        if len(low_flux_group_centers) >= 2:
            first_spacing = float(low_flux_group_centers[1] - low_flux_group_centers[0])
        else:
            first_spacing = 0.0
        normalized_group_spacing = _safe_divide(first_spacing, len(flux_window))

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
        baseline_mad = float(np.median(np.abs(baseline_excluding_dip - np.median(baseline_excluding_dip))))

        half_depth_threshold = baseline_mean - (baseline_depth * 0.5)
        half_depth_mask = flux_window <= half_depth_threshold
        half_depth_start, half_depth_end = _largest_contiguous_region(half_depth_mask)
        half_depth_width = float(max(0, half_depth_end - half_depth_start))
        half_depth_width_fraction = _safe_divide(half_depth_width, len(flux_window))

        dip_segment = flux_window[dip_start:dip_end] if dip_end > dip_start else flux_window[min_index:min_index + 1]
        dip_floor_std = float(np.std(dip_segment))
        dip_floor_mean = float(np.mean(dip_segment))
        dip_floor_to_noise = _safe_divide(baseline_mean - dip_floor_mean, max(baseline_std, 1e-12))

        left_baseline = flux_window[:max(dip_start, 1)]
        right_baseline = flux_window[min(dip_end, len(flux_window) - 1):]
        left_baseline_mean = float(np.mean(left_baseline)) if len(left_baseline) else baseline_mean
        right_baseline_mean = float(np.mean(right_baseline)) if len(right_baseline) else baseline_mean
        baseline_asymmetry = abs(left_baseline_mean - right_baseline_mean)

        central_slice_radius = max(2, len(flux_window) // 10)
        central_start = max(0, (len(flux_window) // 2) - central_slice_radius)
        central_end = min(len(flux_window), (len(flux_window) // 2) + central_slice_radius)
        central_mean = float(np.mean(flux_window[central_start:central_end]))
        central_depth = baseline_mean - central_mean
        periodicity_strength, periodicity_lag_fraction = _periodicity_hint(flux_window)

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
                skew_flux,
                kurt_flux,
                lowest_points_mean,
                dip_strength,
                longest_run,
                low_flux_fraction,
                low_flux_run_count,
                normalized_group_spacing,
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
                baseline_mad,
                half_depth_width,
                half_depth_width_fraction,
                dip_floor_std,
                dip_floor_to_noise,
                baseline_asymmetry,
                central_depth,
                periodicity_strength,
                periodicity_lag_fraction,
            ]
        )

    return np.asarray(feature_rows, dtype=float)


def _extract_pre_repeat_transit_features(X: np.ndarray) -> np.ndarray:
    """Return the feature set used before repeat-pattern features were added."""
    full_features = _extract_enhanced_transit_features(X)
    return full_features[:, :-3]


def _extract_false_positive_aware_v1_transit_features(X: np.ndarray) -> np.ndarray:
    """Add the first false-positive separation layer on top of the 39-feature set."""
    base_features = _extract_enhanced_transit_features(X)
    extra_feature_rows = []

    for flux_window in X:
        flux_window = np.asarray(flux_window, dtype=float)
        min_flux = float(np.min(flux_window))
        mean_flux = float(np.mean(flux_window))
        std_flux = float(np.std(flux_window))
        min_index = int(np.argmin(flux_window))
        percentile_5 = float(np.percentile(flux_window, 5))

        low_flux_mask = flux_window <= percentile_5
        dip_start, dip_end = _largest_contiguous_region(low_flux_mask)

        local_radius = max(3, min(12, len(flux_window) // 12))
        local_start = max(0, min_index - local_radius)
        local_end = min(len(flux_window), min_index + local_radius + 1)
        baseline_excluding_dip = np.concatenate(
            [flux_window[:local_start], flux_window[local_end:]]
        )
        if len(baseline_excluding_dip) == 0:
            baseline_excluding_dip = flux_window

        baseline_mean = float(np.mean(baseline_excluding_dip))
        baseline_std = float(np.std(baseline_excluding_dip))
        baseline_median = float(np.median(baseline_excluding_dip))
        baseline_depth = baseline_mean - min_flux

        half_depth_threshold = baseline_mean - (baseline_depth * 0.5)
        half_depth_mask = flux_window <= half_depth_threshold
        half_depth_start, half_depth_end = _largest_contiguous_region(half_depth_mask)
        half_depth_width = max(0, half_depth_end - half_depth_start)

        significant_threshold = baseline_mean - max(baseline_depth * 0.35, baseline_std)
        significant_regions = _true_regions(flux_window <= significant_threshold)
        significant_dip_count = float(len(significant_regions))

        region_depths = [
            max(0.0, baseline_mean - float(np.min(flux_window[start:end])))
            for start, end in significant_regions
            if end > start
        ]
        region_depths.sort(reverse=True)
        primary_region_depth = region_depths[0] if region_depths else max(0.0, baseline_depth)
        secondary_region_depth = region_depths[1] if len(region_depths) > 1 else 0.0
        secondary_depth_fraction = _safe_divide(
            secondary_region_depth,
            max(primary_region_depth, 1e-12),
        )
        primary_depth_dominance = _safe_divide(
            primary_region_depth,
            max(sum(region_depths), 1e-12),
        )

        significant_centers = [
            (start + end - 1) / 2.0
            for start, end in significant_regions
            if end > start
        ]
        if len(significant_centers) >= 3:
            spacing_std_fraction = _safe_divide(
                float(np.std(np.diff(np.asarray(significant_centers, dtype=float)))),
                len(flux_window),
            )
        else:
            spacing_std_fraction = 0.0

        full_window_slope = _window_slope(flux_window)
        left_baseline_end = max(dip_start, 2)
        right_baseline_start = min(max(dip_end, 0), max(len(flux_window) - 2, 0))
        left_baseline_slope = _window_slope(flux_window[:left_baseline_end])
        right_baseline_slope = _window_slope(flux_window[right_baseline_start:])
        baseline_slope_gap = abs(left_baseline_slope - right_baseline_slope)
        trend_depth_ratio = _safe_divide(
            abs(full_window_slope) * len(flux_window),
            max(baseline_depth, baseline_std, 1e-12),
        )

        floor_threshold = min_flux + (max(baseline_depth, 0.0) * 0.2)
        floor_mask = flux_window <= floor_threshold
        floor_start, floor_end = _largest_contiguous_region(floor_mask)
        flat_floor_fraction = _safe_divide(
            max(0, floor_end - floor_start),
            max(half_depth_width, 1),
        )

        ingress_width = max(1, min_index - half_depth_start)
        egress_width = max(1, half_depth_end - min_index)
        ingress_egress_balance = _safe_divide(
            min(ingress_width, egress_width),
            max(ingress_width, egress_width),
        )

        if baseline_std < 1e-12:
            baseline_outlier_fraction = 0.0
        else:
            standardized_baseline = np.abs(baseline_excluding_dip - baseline_median) / baseline_std
            baseline_outlier_fraction = float(np.mean(standardized_baseline >= 2.5))

        extra_feature_rows.append(
            [
                significant_dip_count,
                secondary_depth_fraction,
                primary_depth_dominance,
                spacing_std_fraction,
                full_window_slope,
                left_baseline_slope,
                right_baseline_slope,
                baseline_slope_gap,
                trend_depth_ratio,
                flat_floor_fraction,
                ingress_egress_balance,
                baseline_outlier_fraction,
            ]
        )

    extra_features = np.asarray(extra_feature_rows, dtype=float)
    return np.hstack([base_features, extra_features])


def _extract_eclipse_aware_transit_features(X: np.ndarray) -> np.ndarray:
    """Add alternating-dip and secondary-event clues on top of the 51-feature set."""
    base_features = _extract_false_positive_aware_v1_transit_features(X)
    extra_feature_rows = []

    for flux_window in X:
        flux_window = np.asarray(flux_window, dtype=float)
        min_flux = float(np.min(flux_window))
        baseline_mean = float(np.mean(flux_window))
        baseline_std = float(np.std(flux_window))
        baseline_depth = max(0.0, baseline_mean - min_flux)
        significant_threshold = baseline_mean - max(baseline_depth * 0.35, baseline_std)
        significant_regions = _true_regions(flux_window <= significant_threshold)

        region_depths = [
            max(0.0, baseline_mean - float(np.min(flux_window[start:end])))
            for start, end in significant_regions
            if end > start
        ]
        region_centers = [
            (start + end - 1) / 2.0
            for start, end in significant_regions
            if end > start
        ]

        if len(region_depths) >= 2:
            odd_depths = region_depths[::2]
            even_depths = region_depths[1::2]
            odd_mean = float(np.mean(odd_depths))
            even_mean = float(np.mean(even_depths)) if even_depths else odd_mean
            alternating_depth_balance = _safe_divide(
                min(odd_mean, even_mean),
                max(odd_mean, even_mean, 1e-12),
            )
            depth_variation_fraction = _safe_divide(
                float(np.std(region_depths)),
                max(float(np.mean(region_depths)), 1e-12),
            )
        else:
            alternating_depth_balance = 1.0
            depth_variation_fraction = 0.0

        if len(region_depths) >= 2:
            primary_index = int(np.argmax(region_depths))
            secondary_candidates = [
                (index, depth_value)
                for index, depth_value in enumerate(region_depths)
                if index != primary_index
            ]
            secondary_index, _ = max(secondary_candidates, key=lambda item: item[1])
            secondary_center_offset_fraction = _safe_divide(
                abs(region_centers[secondary_index] - region_centers[primary_index]),
                len(flux_window),
            )
        else:
            secondary_center_offset_fraction = 0.0

        extra_feature_rows.append(
            [
                alternating_depth_balance,
                depth_variation_fraction,
                secondary_center_offset_fraction,
            ]
        )

    extra_features = np.asarray(extra_feature_rows, dtype=float)
    return np.hstack([base_features, extra_features])


def extract_transit_features(
    X: np.ndarray,
    expected_feature_count: int | None = None,
    time_windows: np.ndarray | None = None,
    target_names: np.ndarray | None = None,
) -> np.ndarray:
    """Extract the feature set that matches the saved model or the newest training path."""
    if expected_feature_count == 26:
        return _extract_legacy_transit_features(X)
    if expected_feature_count == 36:
        return _extract_pre_repeat_transit_features(X)
    if expected_feature_count == 39:
        return _extract_enhanced_transit_features(X)
    if expected_feature_count == 51:
        return _extract_false_positive_aware_v1_transit_features(X)
    if expected_feature_count == 54:
        return _extract_eclipse_aware_transit_features(X)
    if expected_feature_count == 71:
        return _extract_context_aware_transit_features(
            X,
            time_windows=time_windows,
            target_names=target_names,
        )

    context_aware_features = _extract_context_aware_transit_features(
        X,
        time_windows=time_windows,
        target_names=target_names,
    )
    candidate_event_features = _extract_candidate_event_features(
        X,
        time_windows=time_windows,
        target_names=target_names,
    )
    return np.hstack([context_aware_features, candidate_event_features])


def _split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    config: ProjectConfig,
    groups: np.ndarray | None = None,
    time_windows: np.ndarray | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Split the dataset once so multiple models can be compared fairly."""
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        raise ValueError(
            "The prepared dataset contains only one class, so the model cannot train. "
            "This usually means the target selection created only transit windows or only "
            "non-transit windows."
        )
    group_values = (
        np.asarray(groups, dtype=object)
        if groups is not None
        else np.full(len(y), "unknown", dtype=object)
    )
    engineered_X = extract_transit_features(
        X,
        time_windows=time_windows,
        target_names=group_values,
    )
    sample_indices = np.arange(len(y))
    if groups is not None:
        splitter = StratifiedGroupKFold(
            n_splits=4,
            shuffle=True,
            random_state=config.random_seed,
        )
        train_index, test_index = next(splitter.split(engineered_X, y, group_values))
        return (
            engineered_X[train_index],
            engineered_X[test_index],
            y[train_index],
            y[test_index],
            group_values[train_index],
            group_values[test_index],
            sample_indices[train_index],
            sample_indices[test_index],
        )
    return train_test_split(
        engineered_X,
        y,
        group_values,
        sample_indices,
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


def _build_hist_gradient_model(config: ProjectConfig) -> Pipeline:
    """Construct a boosted-tree comparison model for harder nonlinear cases."""
    return Pipeline(
        steps=[
            (
                "classifier",
                HistGradientBoostingClassifier(
                    max_depth=7,
                    learning_rate=0.05,
                    max_iter=350,
                    min_samples_leaf=10,
                    l2_regularization=0.15,
                    random_state=config.random_seed,
                ),
            ),
        ]
    )


def _fit_model(
    model: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> Pipeline:
    """Fit a pipeline, passing weights only to the final estimator when available."""
    fit_kwargs: dict[str, np.ndarray] = {}
    if sample_weight is not None:
        final_step_name = model.steps[-1][0]
        fit_kwargs[f"{final_step_name}__sample_weight"] = sample_weight
    model.fit(X_train, y_train, **fit_kwargs)
    return model


def _normalize_sample_weights(sample_weight: np.ndarray) -> np.ndarray:
    """Keep average training weight near 1 so weighting stays stable across runs."""
    mean_weight = float(np.mean(sample_weight))
    if mean_weight <= 0:
        return sample_weight
    return sample_weight / mean_weight


def _build_role_aware_sample_weights(
    y_train: np.ndarray,
    train_target_roles: np.ndarray | None,
    train_example_roles: np.ndarray | None,
    config: ProjectConfig,
) -> np.ndarray | None:
    """Emphasize the hardest known negative families before any mined reweighting."""
    if train_target_roles is None or train_example_roles is None:
        return None

    sample_weight = np.ones(len(y_train), dtype=float)
    negative_mask = y_train == 0

    koi_false_positive_mask = negative_mask & (train_target_roles == "koi_false_positive_host")
    quiet_control_mask = negative_mask & (train_target_roles == "quiet_control_target")
    false_positive_event_mask = negative_mask & (
        train_example_roles == "catalog_false_positive_event_window"
    )

    sample_weight[koi_false_positive_mask] *= config.koi_false_positive_host_weight
    sample_weight[quiet_control_mask] *= config.quiet_control_target_weight
    sample_weight[false_positive_event_mask] *= config.catalog_false_positive_event_weight

    return _normalize_sample_weights(sample_weight)


def _mine_hard_negative_weights(
    model: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray | None,
    train_target_roles: np.ndarray | None,
    train_example_roles: np.ndarray | None,
    config: ProjectConfig,
) -> np.ndarray | None:
    """Up-weight the hardest negatives and missed positives after a scout fit."""
    if sample_weight is None or train_target_roles is None or train_example_roles is None:
        return sample_weight

    hard_negative_source_mask = (y_train == 0) & (
        (train_target_roles == "koi_false_positive_host")
        | (train_example_roles == "catalog_false_positive_event_window")
    )
    hard_positive_source_mask = (y_train == 1) & (
        (train_target_roles == "positive_koi_host")
        | (train_example_roles == "positive_transit_window")
    )
    if not np.any(hard_negative_source_mask) and not np.any(hard_positive_source_mask):
        return sample_weight

    scout_model = clone(model)
    _fit_model(scout_model, X_train, y_train, sample_weight=sample_weight)
    train_prob = scout_model.predict_proba(X_train)[:, 1]

    hard_negative_mask = np.zeros(len(y_train), dtype=bool)
    if np.any(hard_negative_source_mask):
        negative_probabilities = train_prob[hard_negative_source_mask]
        percentile_threshold = float(
            np.quantile(negative_probabilities, config.hard_negative_top_percentile)
        )
        probability_threshold = max(config.hard_negative_probability_threshold, percentile_threshold)
        hard_negative_mask = hard_negative_source_mask & (train_prob >= probability_threshold)

        if not np.any(hard_negative_mask):
            highest_index = np.where(hard_negative_source_mask)[0][
                int(np.argmax(negative_probabilities))
            ]
            hard_negative_mask[highest_index] = True

    hard_positive_mask = np.zeros(len(y_train), dtype=bool)
    if np.any(hard_positive_source_mask):
        positive_probabilities = train_prob[hard_positive_source_mask]
        percentile_ceiling = float(
            np.quantile(positive_probabilities, config.hard_positive_bottom_percentile)
        )
        probability_ceiling = min(config.hard_positive_probability_threshold, percentile_ceiling)
        hard_positive_mask = hard_positive_source_mask & (train_prob <= probability_ceiling)

        if not np.any(hard_positive_mask):
            lowest_index = np.where(hard_positive_source_mask)[0][
                int(np.argmin(positive_probabilities))
            ]
            hard_positive_mask[lowest_index] = True

    refined_weight = sample_weight.copy()
    refined_weight[hard_negative_mask] *= config.hard_negative_boost
    refined_weight[hard_positive_mask] *= config.hard_positive_boost
    return _normalize_sample_weights(refined_weight)


def _fit_and_package(
    model: Pipeline,
    model_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    train_target_names: np.ndarray,
    test_target_names: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> ModelArtifacts:
    """Train one model and package the outputs in a shared format."""
    _fit_model(model, X_train, y_train, sample_weight=sample_weight)
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
        train_target_names=train_target_names,
        test_target_names=test_target_names,
        train_indices=train_indices,
        test_indices=test_indices,
    )


def train_baseline_model(
    X: np.ndarray,
    y: np.ndarray,
    config: ProjectConfig,
    groups: np.ndarray | None = None,
    time_windows: np.ndarray | None = None,
    target_roles: np.ndarray | None = None,
    example_roles: np.ndarray | None = None,
) -> ModelArtifacts:
    """Train the baseline logistic regression classifier on explainable transit features."""
    (
        X_train,
        X_test,
        y_train,
        y_test,
        train_targets,
        test_targets,
        train_indices,
        test_indices,
    ) = _split_dataset(
        X, y, config, groups=groups, time_windows=time_windows
    )
    model = _build_logistic_model(config)
    train_target_roles = (
        np.asarray(target_roles, dtype=object)[train_indices]
        if target_roles is not None and len(target_roles) == len(y)
        else None
    )
    train_example_roles = (
        np.asarray(example_roles, dtype=object)[train_indices]
        if example_roles is not None and len(example_roles) == len(y)
        else None
    )
    base_weight = _build_role_aware_sample_weights(
        y_train,
        train_target_roles,
        train_example_roles,
        config,
    )
    final_weight = _mine_hard_negative_weights(
        model,
        X_train,
        y_train,
        base_weight,
        train_target_roles,
        train_example_roles,
        config,
    )
    return _fit_and_package(
        model,
        "logistic_regression",
        X_train,
        X_test,
        y_train,
        y_test,
        train_targets,
        test_targets,
        train_indices,
        test_indices,
        sample_weight=final_weight,
    )


def train_model_comparison(
    X: np.ndarray,
    y: np.ndarray,
    config: ProjectConfig,
    groups: np.ndarray | None = None,
    time_windows: np.ndarray | None = None,
    target_roles: np.ndarray | None = None,
    example_roles: np.ndarray | None = None,
) -> list[ModelArtifacts]:
    """Train multiple simple models on the same split for fair comparison."""
    (
        X_train,
        X_test,
        y_train,
        y_test,
        train_targets,
        test_targets,
        train_indices,
        test_indices,
    ) = _split_dataset(
        X, y, config, groups=groups, time_windows=time_windows
    )
    train_target_roles = (
        np.asarray(target_roles, dtype=object)[train_indices]
        if target_roles is not None and len(target_roles) == len(y)
        else None
    )
    train_example_roles = (
        np.asarray(example_roles, dtype=object)[train_indices]
        if example_roles is not None and len(example_roles) == len(y)
        else None
    )
    models = [
        ("logistic_regression", _build_logistic_model(config)),
        ("random_forest", _build_random_forest_model(config)),
        ("hist_gradient_boosting", _build_hist_gradient_model(config)),
    ]
    packaged_models: list[ModelArtifacts] = []
    for model_name, model in models:
        base_weight = _build_role_aware_sample_weights(
            y_train,
            train_target_roles,
            train_example_roles,
            config,
        )
        final_weight = _mine_hard_negative_weights(
            model,
            X_train,
            y_train,
            base_weight,
            train_target_roles,
            train_example_roles,
            config,
        )
        packaged_models.append(
            _fit_and_package(
                model,
                model_name,
                X_train,
                X_test,
                y_train,
                y_test,
                train_targets,
                test_targets,
                train_indices,
                test_indices,
                sample_weight=final_weight,
            )
        )
    return packaged_models
