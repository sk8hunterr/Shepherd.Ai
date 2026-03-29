"""Data loading helpers for Kepler and TESS light curves."""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import pandas as pd

from .config import ProjectConfig

try:
    import lightkurve as lk
except ImportError:  # pragma: no cover - handled at runtime for user guidance
    lk = None


@dataclass
class LoadedLightCurve:
    """Container for a single light curve."""

    target_name: str
    time: np.ndarray
    flux: np.ndarray
    source: str


def _clean_series(time: np.ndarray, flux: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove NaNs and infinities before downstream processing."""
    mask = np.isfinite(time) & np.isfinite(flux)
    return time[mask], flux[mask]


def _to_arrays(lightcurve) -> tuple[np.ndarray, np.ndarray]:
    """Convert a Lightkurve object to plain numpy arrays."""
    time = np.asarray(lightcurve.time.value, dtype=float)
    flux = np.asarray(lightcurve.flux.value, dtype=float)
    return _clean_series(time, flux)


def _load_with_lightkurve(config: ProjectConfig) -> list[LoadedLightCurve]:
    """Download a configurable sample of mission light curves."""
    if lk is None:
        raise ImportError(
            "lightkurve is not installed. Run `pip install -r requirements.txt` first."
        )

    loaded_curves: list[LoadedLightCurve] = []

    for target in config.kepler_targets:
        search_kwargs = {
            "mission": config.mission,
            "cadence": config.cadence,
        }
        if config.author:
            search_kwargs["author"] = config.author

        search_result = lk.search_lightcurve(target, **search_kwargs)

        if len(search_result) == 0:
            continue

        kept_count = 0
        for row_index in range(len(search_result)):
            if not _matches_stage_filters(search_result.table[row_index], config):
                continue

            lightcurve = search_result[row_index].download(download_dir=str(config.cache_dir))
            if lightcurve is None:
                continue

            time, flux = _to_arrays(lightcurve)
            if len(flux) < config.window_length:
                continue

            loaded_curves.append(
                LoadedLightCurve(
                    target_name=target,
                    time=time,
                    flux=flux,
                    source=str(config.mission).lower(),
                )
            )

            kept_count += 1
            if kept_count >= config.max_lightcurves_per_target:
                break

    if len(loaded_curves) < config.minimum_total_lightcurves:
        raise RuntimeError(
            f"Not enough {config.mission} light curves were downloaded for the requested stage. "
            f"Expected at least {config.minimum_total_lightcurves}, got {len(loaded_curves)}."
        )

    return loaded_curves


def _extract_sector_value(row) -> int | None:
    """Best-effort sector extraction from a Lightkurve search row."""
    for key in ("sequence_number", "sequence", "sequence_number_s"):
        if key in row.colnames:
            value = row[key]
            try:
                return int(value)
            except (TypeError, ValueError):
                continue

    for key in ("mission", "description", "productFilename"):
        if key not in row.colnames:
            continue
        text = str(row[key])
        match = re.search(r"(?:sector|s)(\d{1,4})", text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
        digits = re.findall(r"\d{1,4}", text)
        if digits:
            try:
                return int(digits[0])
            except ValueError:
                continue
    return None


def _matches_stage_filters(row, config: ProjectConfig) -> bool:
    """Apply optional mission-specific filters to search results."""
    if config.min_tess_sector is None:
        return True

    sector = _extract_sector_value(row)
    if sector is None:
        return False
    return sector >= config.min_tess_sector


def _build_synthetic_lightcurve(
    target_name: str,
    length: int,
    rng: np.random.Generator,
) -> LoadedLightCurve:
    """Fallback light curve used when internet access is unavailable."""
    time = np.linspace(0.0, 30.0, num=length)
    trend = 1.0 + 0.002 * np.sin(time / 2.5)
    stellar_variation = 0.0015 * np.sin(time * 1.3) + 0.001 * np.cos(time * 0.7)
    noise = rng.normal(0.0, 0.0015, size=length)
    flux = trend + stellar_variation + noise
    return LoadedLightCurve(target_name=target_name, time=time, flux=flux, source="synthetic")


def _load_fallback_data(config: ProjectConfig) -> list[LoadedLightCurve]:
    """Create a small local dataset so the project still runs offline."""
    rng = np.random.default_rng(config.random_seed)
    fallback_curves = []
    for index, target in enumerate(config.kepler_targets, start=1):
        fallback_curves.append(
            _build_synthetic_lightcurve(
                target_name=f"{target}-fallback-{index}",
                length=2500,
                rng=rng,
            )
        )
    return fallback_curves


def load_light_curves(
    config: ProjectConfig,
    allow_fallback: bool = True,
) -> tuple[list[LoadedLightCurve], str]:
    """Load real mission curves when possible, otherwise use a synthetic fallback."""
    try:
        curves = _load_with_lightkurve(config)
        message = (
            f"Loaded real {config.mission} data for {len(curves)} light curves "
            f"across {len(config.kepler_targets)} requested targets."
        )
        return curves, message
    except Exception as exc:
        if not allow_fallback:
            raise
        curves = _load_fallback_data(config)
        message = (
            f"Could not download {config.mission} data in this environment, so the pipeline used "
            f"a synthetic fallback instead. Reason: {exc}"
        )
        return curves, message


def summarize_light_curves(curves: list[LoadedLightCurve]) -> pd.DataFrame:
    """Create a compact summary table for reporting."""
    records = []
    for curve in curves:
        records.append(
            {
                "target_name": curve.target_name,
                "source": curve.source,
                "num_points": len(curve.flux),
                "time_start": float(curve.time[0]),
                "time_end": float(curve.time[-1]),
            }
        )
    return pd.DataFrame.from_records(records)
