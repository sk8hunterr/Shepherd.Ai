"""Dataset preparation helpers for reusable training datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np

from .catalog_loader import auto_select_targets, load_real_catalog
from .config import ProjectConfig
from .data_loader import load_light_curves, summarize_light_curves
from .labeling import create_labeled_examples, examples_to_arrays
from .preprocessing import build_window_dataset
from .real_labeling import create_real_labeled_examples, RealTransitLabelingReport


@dataclass
class PreparedDataset:
    """Prepared dataset plus metadata for training and reporting."""

    X: np.ndarray
    y: np.ndarray
    data_message: str
    summary_df_text: str
    num_lightcurves: int
    num_windows: int
    num_examples: int
    catalog_message: str | None
    real_label_report: RealTransitLabelingReport | None
    dataset_path: Path


def get_prepared_dataset_path(config: ProjectConfig) -> Path:
    """Return the stage-specific prepared dataset file path."""
    return config.prepared_data_dir / f"{config.stage_name}_dataset.npz"


def get_prepared_metadata_path(config: ProjectConfig) -> Path:
    """Return the metadata path for the prepared dataset."""
    return config.prepared_data_dir / f"{config.stage_name}_dataset_metadata.json"


def prepare_training_dataset(
    config: ProjectConfig,
    allow_fallback: bool,
) -> PreparedDataset:
    """Build and save a reusable training dataset for the requested stage."""
    catalog_message = None
    real_label_report = None
    catalog_df = None
    if config.labeling_mode.startswith("real_"):
        catalog_df, catalog_message = load_real_catalog(config, allow_download=True)
        selected_targets = auto_select_targets(config, catalog_df)
        if selected_targets:
            config.kepler_targets = selected_targets
            catalog_message = (
                f"{catalog_message}; auto-selected {len(selected_targets)} {config.mission} targets "
                "from the real-label catalog"
            )

    lightcurves, data_message = load_light_curves(config, allow_fallback=allow_fallback)
    summary_df = summarize_light_curves(lightcurves)
    windows = build_window_dataset(lightcurves, config)

    if config.labeling_mode.startswith("real_"):
        labeled_examples, real_label_report = create_real_labeled_examples(
            windows=windows,
            catalog_df=catalog_df,
            config=config,
        )
    else:
        labeled_examples = create_labeled_examples(windows, config)

    X, y = examples_to_arrays(labeled_examples)

    dataset_path = get_prepared_dataset_path(config)
    metadata_path = get_prepared_metadata_path(config)
    np.savez_compressed(dataset_path, X=X, y=y)

    metadata = {
        "stage_name": config.stage_name,
        "labeling_mode": config.labeling_mode,
        "mission": config.mission,
        "requested_targets": config.kepler_targets,
        "data_message": data_message,
        "summary_df_text": summary_df.to_string(index=False),
        "num_lightcurves": len(lightcurves),
        "num_windows": len(windows),
        "num_examples": len(labeled_examples),
        "catalog_message": catalog_message,
        "real_label_report": (
            {
                "matched_targets": real_label_report.matched_targets,
                "unmatched_targets": real_label_report.unmatched_targets,
                "positive_examples": real_label_report.positive_examples,
                "negative_examples": real_label_report.negative_examples,
                "skipped_examples": real_label_report.skipped_examples,
            }
            if real_label_report is not None
            else None
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return PreparedDataset(
        X=X,
        y=y,
        data_message=data_message,
        summary_df_text=summary_df.to_string(index=False),
        num_lightcurves=len(lightcurves),
        num_windows=len(windows),
        num_examples=len(labeled_examples),
        catalog_message=catalog_message,
        real_label_report=real_label_report,
        dataset_path=dataset_path,
    )
