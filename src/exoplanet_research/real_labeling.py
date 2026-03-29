"""Real-label helpers using Kepler KOI or TESS TOI catalogs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import re

from .config import ProjectConfig
from .labeling import LabeledExample
from .preprocessing import WindowedLightCurve


KEPLER_POSITIVE_DISPOSITIONS = {"CONFIRMED", "CANDIDATE"}
TESS_POSITIVE_DISPOSITIONS = {"CP", "KP", "PC", "APC"}


@dataclass
class RealTransitLabelingReport:
    """Summary of how Stage 3 labels were created."""

    matched_targets: int
    unmatched_targets: int
    positive_examples: int
    negative_examples: int
    skipped_examples: int


def _normalize_target_name(value: str) -> str:
    cleaned = str(value).strip().lower()
    cleaned = re.sub(r"\s+[a-z]$", "", cleaned)
    return cleaned


def _prepare_koi_catalog(koi_catalog: pd.DataFrame) -> pd.DataFrame:
    """Keep only the KOI rows and columns needed for Stage 3."""
    catalog = koi_catalog.copy()
    catalog["kepler_name"] = catalog["kepler_name"].fillna("")
    catalog["target_key"] = catalog["kepler_name"].map(_normalize_target_name)
    catalog["koi_disposition"] = catalog["koi_disposition"].fillna("")
    catalog = catalog[catalog["koi_disposition"].isin(KEPLER_POSITIVE_DISPOSITIONS)]
    catalog["period_days"] = catalog["koi_period"].astype(float)
    catalog["epoch_reference"] = catalog["koi_time0bk"].astype(float)
    catalog["duration_days"] = catalog["koi_duration"].astype(float) / 24.0
    return catalog


def _prepare_toi_catalog(toi_catalog: pd.DataFrame) -> pd.DataFrame:
    """Keep only the TOI rows and columns needed for Stage 4."""
    catalog = toi_catalog.copy()
    catalog["tfopwg_disp"] = catalog["tfopwg_disp"].fillna("")
    catalog = catalog[catalog["tfopwg_disp"].isin(TESS_POSITIVE_DISPOSITIONS)]
    catalog = catalog[catalog["tid"].notna()]
    catalog["target_key"] = catalog["tid"].map(lambda value: _normalize_target_name(f"TIC {int(value)}"))
    catalog["period_days"] = catalog["pl_orbper"].astype(float)
    catalog["epoch_reference"] = catalog["pl_tranmid"].astype(float)
    catalog.loc[catalog["epoch_reference"] > 1_000_000, "epoch_reference"] -= 2457000.0
    catalog["duration_days"] = catalog["pl_trandurh"].astype(float) / 24.0
    return catalog


def _prepare_catalog(catalog_df: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    """Dispatch to the correct catalog prep for the current real-label stage."""
    if config.labeling_mode == "real_toi":
        return _prepare_toi_catalog(catalog_df)
    return _prepare_koi_catalog(catalog_df)


def _transit_overlap_fraction(
    time_window: np.ndarray,
    transit_center: float,
    duration_days: float,
) -> float:
    """Measure how much of one transit event overlaps a window."""
    half_duration = duration_days / 2.0
    transit_start = transit_center - half_duration
    transit_end = transit_center + half_duration
    window_start = float(time_window[0])
    window_end = float(time_window[-1])
    overlap = max(0.0, min(window_end, transit_end) - max(window_start, transit_start))
    if duration_days <= 0:
        return 0.0
    return overlap / duration_days


def _compute_transit_centers(
    period_days: float,
    epoch_bkjd: float,
    time_min: float,
    time_max: float,
) -> np.ndarray:
    """Compute expected transit centers inside the observed time range."""
    start_index = int(np.floor((time_min - epoch_bkjd) / period_days)) - 1
    end_index = int(np.ceil((time_max - epoch_bkjd) / period_days)) + 1
    indices = np.arange(start_index, end_index + 1)
    centers = epoch_bkjd + indices * period_days
    mask = (centers >= time_min - period_days) & (centers <= time_max + period_days)
    return centers[mask]


def create_real_labeled_examples(
    windows: list[WindowedLightCurve],
    catalog_df: pd.DataFrame,
    config: ProjectConfig,
) -> tuple[list[LabeledExample], RealTransitLabelingReport]:
    """Label windows using real KOI/TOI ephemerides instead of injected transits."""
    prepared_catalog = _prepare_catalog(catalog_df, config)
    catalog_by_target = {
        target_name: rows.copy()
        for target_name, rows in prepared_catalog.groupby("target_key")
    }

    examples: list[LabeledExample] = []
    matched_targets: set[str] = set()
    seen_targets: set[str] = set()
    skipped_examples = 0

    for window in windows:
        target_key = _normalize_target_name(window.target_name)
        seen_targets.add(target_key)
        target_rows = catalog_by_target.get(target_key)

        label = 0
        ambiguous_window = False
        if target_rows is not None and not target_rows.empty:
            matched_targets.add(target_key)
            for _, row in target_rows.iterrows():
                transit_centers = _compute_transit_centers(
                    period_days=float(row["period_days"]),
                    epoch_bkjd=float(row["epoch_reference"]),
                    time_min=float(window.time_window[0]),
                    time_max=float(window.time_window[-1]),
                )
                duration_days = float(row["duration_days"])
                for center in transit_centers:
                    overlap_fraction = _transit_overlap_fraction(
                        time_window=window.time_window,
                        transit_center=float(center),
                        duration_days=duration_days,
                    )
                    if overlap_fraction >= 0.5:
                        label = 1
                        break
                    if 0.15 <= overlap_fraction < 0.5:
                        ambiguous_window = True
                if label == 1 or ambiguous_window:
                    break

        if ambiguous_window:
            skipped_examples += 1
            continue

        examples.append(
            LabeledExample(
                target_name=window.target_name,
                flux=window.flux_window,
                label=label,
                injected=False,
                source=window.source,
            )
        )

    positive_examples = sum(example.label for example in examples)
    negative_examples = len(examples) - positive_examples

    report = RealTransitLabelingReport(
        matched_targets=len(matched_targets),
        unmatched_targets=len(seen_targets - matched_targets),
        positive_examples=positive_examples,
        negative_examples=negative_examples,
        skipped_examples=skipped_examples,
    )
    return examples, report
