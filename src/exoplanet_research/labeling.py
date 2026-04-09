"""Label creation using simple injected transit signals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ProjectConfig
from .preprocessing import WindowedLightCurve


@dataclass
class LabeledExample:
    """One supervised learning example."""

    target_name: str
    time: np.ndarray
    flux: np.ndarray
    label: int
    injected: bool
    source: str
    example_role: str = "unclassified"


def inject_box_transit(
    flux_window: np.ndarray,
    rng: np.random.Generator,
    depth_range: tuple[float, float],
    width_range: tuple[int, int],
) -> np.ndarray:
    """Inject a simple box-shaped transit into a flux window."""
    injected_flux = flux_window.copy()
    width = int(rng.integers(width_range[0], width_range[1] + 1))
    max_start = len(injected_flux) - width
    start = int(rng.integers(0, max_start + 1))
    depth = float(rng.uniform(depth_range[0], depth_range[1]))
    injected_flux[start : start + width] -= depth
    return injected_flux


def create_labeled_examples(
    windows: list[WindowedLightCurve],
    config: ProjectConfig,
) -> list[LabeledExample]:
    """Create binary labels by injecting transits into part of the dataset."""
    rng = np.random.default_rng(config.random_seed)
    examples: list[LabeledExample] = []

    for window in windows:
        make_transit = rng.random() < config.transit_fraction
        flux = window.flux_window
        if make_transit:
            flux = inject_box_transit(
                flux_window=flux,
                rng=rng,
                depth_range=config.transit_depth_range,
                width_range=config.transit_width_range,
            )

        examples.append(
            LabeledExample(
                target_name=window.target_name,
                time=window.time_window,
                flux=flux,
                label=int(make_transit),
                injected=make_transit,
                source=window.source,
                example_role=(
                    "synthetic_transit_window" if make_transit else "synthetic_background_window"
                ),
            )
        )

    return examples


def examples_to_arrays(examples: list[LabeledExample]) -> tuple[np.ndarray, np.ndarray]:
    """Convert the dataset into model-ready numpy arrays."""
    X = np.vstack([example.flux for example in examples])
    y = np.array([example.label for example in examples], dtype=int)
    return X, y


def examples_to_time_array(examples: list[LabeledExample]) -> np.ndarray:
    """Convert example time windows into a model-side context array."""
    return np.vstack([example.time for example in examples])


def examples_to_arrays_with_targets(
    examples: list[LabeledExample],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert examples into arrays and preserve target names for group-aware splitting."""
    X, y = examples_to_arrays(examples)
    target_names = np.array([example.target_name for example in examples], dtype=object)
    return X, y, target_names


def examples_to_arrays_with_metadata(
    examples: list[LabeledExample],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert examples into arrays and preserve target and example-role metadata."""
    X, y, target_names = examples_to_arrays_with_targets(examples)
    time_windows = examples_to_time_array(examples)
    example_roles = np.array([example.example_role for example in examples], dtype=object)
    return X, y, target_names, time_windows, example_roles
