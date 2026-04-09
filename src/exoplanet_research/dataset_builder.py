"""Dataset preparation helpers for reusable training datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np

from .catalog_loader import (
    auto_select_targets,
    build_target_role_lookup,
    classify_target_role,
    load_real_catalog,
)
from .config import ProjectConfig
from .data_loader import load_light_curves, summarize_light_curves
from .labeling import create_labeled_examples, examples_to_arrays_with_metadata
from .preprocessing import build_window_dataset
from .real_labeling import create_real_labeled_examples, RealTransitLabelingReport


@dataclass
class PreparedDataset:
    """Prepared dataset plus metadata for training and reporting."""

    X: np.ndarray
    y: np.ndarray
    target_names: np.ndarray
    time_windows: np.ndarray
    target_roles: np.ndarray
    example_roles: np.ndarray
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
    target_role_lookup: dict[str, str] = {}
    if config.labeling_mode.startswith("real_"):
        catalog_df, catalog_message = load_real_catalog(config, allow_download=True)
        target_role_lookup = build_target_role_lookup(config, catalog_df)
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

    labeled_examples = _rebalance_examples(labeled_examples, config)
    X, y, target_names, time_windows, example_roles = examples_to_arrays_with_metadata(labeled_examples)
    target_roles = np.array(
        [classify_target_role(target_name, target_role_lookup) for target_name in target_names],
        dtype=object,
    )

    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        raise ValueError(
            "The prepared dataset contains only one class after labeling. "
            "For real-label stages, this usually means the mission download fell back "
            "to synthetic data or the selected targets did not produce matched transit windows. "
            "Run the stage with --real-only on a machine that can reach the mission archive."
        )

    dataset_path = get_prepared_dataset_path(config)
    metadata_path = get_prepared_metadata_path(config)
    np.savez_compressed(
        dataset_path,
        X=X,
        y=y,
        target_names=target_names,
        time_windows=time_windows,
        target_roles=target_roles,
        example_roles=example_roles,
    )

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
        "time_window_shape": list(time_windows.shape),
        "class_counts": {
            "negative": int(np.sum(y == 0)),
            "positive": int(np.sum(y == 1)),
        },
        "target_role_counts": {
            str(role): int(np.sum(target_roles == role))
            for role in sorted(set(target_roles.tolist()))
        },
        "example_role_counts": {
            str(role): int(np.sum(example_roles == role))
            for role in sorted(set(example_roles.tolist()))
        },
        "catalog_message": catalog_message,
        "real_label_report": (
            {
                "matched_targets": real_label_report.matched_targets,
                "unmatched_targets": real_label_report.unmatched_targets,
                "positive_examples": real_label_report.positive_examples,
                "negative_examples": real_label_report.negative_examples,
                "skipped_examples": real_label_report.skipped_examples,
                "hard_negative_examples": real_label_report.hard_negative_examples,
                "example_role_counts": real_label_report.example_role_counts,
            }
            if real_label_report is not None
            else None
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return PreparedDataset(
        X=X,
        y=y,
        target_names=target_names,
        time_windows=time_windows,
        target_roles=target_roles,
        example_roles=example_roles,
        data_message=data_message,
        summary_df_text=summary_df.to_string(index=False),
        num_lightcurves=len(lightcurves),
        num_windows=len(windows),
        num_examples=len(labeled_examples),
        catalog_message=catalog_message,
        real_label_report=real_label_report,
        dataset_path=dataset_path,
    )


def _rebalance_examples(
    examples,
    config: ProjectConfig,
):
    """Reduce extreme class imbalance by downsampling the majority class."""
    max_ratio = config.max_positive_negative_ratio
    if max_ratio is None:
        return examples

    positive_examples = [example for example in examples if example.label == 1]
    negative_examples = [example for example in examples if example.label == 0]

    if not positive_examples or not negative_examples:
        return examples

    allowed_positive_count = int(round(len(negative_examples) * max_ratio))
    if len(positive_examples) <= allowed_positive_count:
        return examples

    rng = np.random.default_rng(config.random_seed)
    kept_positive_indices = rng.choice(
        len(positive_examples),
        size=allowed_positive_count,
        replace=False,
    )
    kept_positive_indices = set(int(index) for index in kept_positive_indices)
    kept_positive_examples = [
        example for index, example in enumerate(positive_examples) if index in kept_positive_indices
    ]
    balanced_examples = negative_examples + kept_positive_examples
    rng.shuffle(balanced_examples)
    return balanced_examples
