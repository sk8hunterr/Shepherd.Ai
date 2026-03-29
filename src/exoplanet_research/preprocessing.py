"""Preprocessing steps for light curve windows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ProjectConfig
from .data_loader import LoadedLightCurve


@dataclass
class WindowedLightCurve:
    """A fixed-length slice of a larger light curve."""

    target_name: str
    time_window: np.ndarray
    flux_window: np.ndarray
    source: str


def robust_normalize(flux: np.ndarray) -> np.ndarray:
    """Normalize around the median so small dips are easier to compare."""
    median = np.median(flux)
    centered = flux / median
    return centered - 1.0


def make_windows(
    lightcurve: LoadedLightCurve,
    config: ProjectConfig,
) -> list[WindowedLightCurve]:
    """Split one light curve into overlapping windows with a shared length."""
    normalized_flux = robust_normalize(lightcurve.flux)
    windows: list[WindowedLightCurve] = []

    start_positions = range(
        0,
        max(len(normalized_flux) - config.window_length + 1, 0),
        config.stride,
    )

    for count, start in enumerate(start_positions):
        end = start + config.window_length
        if end > len(normalized_flux):
            break
        windows.append(
            WindowedLightCurve(
                target_name=lightcurve.target_name,
                time_window=lightcurve.time[start:end],
                flux_window=normalized_flux[start:end],
                source=lightcurve.source,
            )
        )
        if count + 1 >= config.max_windows_per_lightcurve:
            break

    return windows


def build_window_dataset(
    lightcurves: list[LoadedLightCurve],
    config: ProjectConfig,
) -> list[WindowedLightCurve]:
    """Create the full list of fixed-size windows."""
    dataset: list[WindowedLightCurve] = []
    for curve in lightcurves:
        dataset.extend(make_windows(curve, config))
    return dataset
