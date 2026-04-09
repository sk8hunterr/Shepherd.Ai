"""Utilities for saving and loading trained models."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import joblib

from .config import ProjectConfig
from .model_registry import (
    format_experiment_display_name,
    get_checkpoint_version_for_stage,
    get_numeric_version,
    slugify_shepherd_version,
    ModelVersion,
)


def get_model_path(config: ProjectConfig) -> Path:
    """Return the checkpoint-style baseline file path for the deployed model."""
    checkpoint_version = get_checkpoint_version_for_stage(config.stage_name)
    return config.model_dir / checkpoint_version.model_filename


def get_model_path_for_version(config: ProjectConfig, model_version: ModelVersion) -> Path:
    """Return the saved checkpoint path for one registered Shepherd version."""
    return config.model_dir / model_version.model_filename


def get_legacy_model_path(config: ProjectConfig) -> Path:
    """Return the older stage-based baseline filename for compatibility."""
    return config.model_dir / f"baseline_{config.stage_name}_model.joblib"


def get_experiment_model_dir(config: ProjectConfig) -> Path:
    """Return the directory used for non-baseline experiment models."""
    return config.model_dir / "experiments" / config.stage_name


def get_experiment_manifest_path(config: ProjectConfig) -> Path:
    """Return the manifest file that tracks patch-style experiment versions."""
    return get_experiment_model_dir(config) / "experiment_versions.json"


def _infer_model_name_from_experiment_path(path: Path) -> str:
    """Infer the model name from a saved experiment filename when possible."""
    parts = path.stem.split("_")
    if len(parts) >= 8 and parts[0] == "shep":
        return "_".join(parts[4:-2])
    if len(parts) >= 4 and parts[0].startswith("stage"):
        return "_".join(parts[1:-2])
    return "experiment_model"


def _load_experiment_manifest(config: ProjectConfig) -> dict:
    """Load or initialize the experiment-version manifest for one stage."""
    experiment_dir = get_experiment_model_dir(config)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = get_experiment_manifest_path(config)
    checkpoint_version = get_numeric_version(
        get_checkpoint_version_for_stage(config.stage_name).display_name
    )

    if manifest_path.exists():
        # Accept BOM-prefixed UTF-8 manifests written by some Windows tools.
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    else:
        manifest = {
            "checkpoint_version": checkpoint_version,
            "experiments": [],
        }

    manifest.setdefault("checkpoint_version", checkpoint_version)
    manifest.setdefault("experiments", [])

    if not manifest["experiments"]:
        existing_models = sorted(
            experiment_dir.glob("*.joblib"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if existing_models:
            seeded_version = format_experiment_display_name(checkpoint_version, 1)
            manifest["experiments"].append(
                {
                    "display_name": seeded_version,
                    "numeric_version": f"{checkpoint_version}.1",
                    "path": existing_models[0].name,
                    "model_name": _infer_model_name_from_experiment_path(existing_models[0]),
                    "backfilled": True,
                }
            )

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _next_experiment_version(config: ProjectConfig) -> tuple[str, str]:
    """Return the next patch-style Shepherd version for a new experiment model."""
    manifest = _load_experiment_manifest(config)
    checkpoint_version = str(manifest["checkpoint_version"])
    patch_numbers: list[int] = []
    version_prefix = f"{checkpoint_version}."

    for entry in manifest["experiments"]:
        numeric_version = str(entry.get("numeric_version", ""))
        if numeric_version.startswith(version_prefix):
            patch_suffix = numeric_version[len(version_prefix):]
            if patch_suffix.isdigit():
                patch_numbers.append(int(patch_suffix))

    next_patch_number = max(patch_numbers, default=0) + 1
    return (
        format_experiment_display_name(checkpoint_version, next_patch_number),
        f"{checkpoint_version}.{next_patch_number}",
    )


def _register_experiment_version(
    config: ProjectConfig,
    model_path: Path,
    display_name: str,
    numeric_version: str,
    model_name: str | None,
    recommended_threshold: float | None,
) -> None:
    """Append one saved experiment model to the stage manifest."""
    manifest = _load_experiment_manifest(config)
    manifest["experiments"].append(
        {
            "display_name": display_name,
            "numeric_version": numeric_version,
            "path": model_path.name,
            "model_name": model_name,
            "recommended_threshold": recommended_threshold,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    get_experiment_manifest_path(config).write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def get_experiment_model_path(
    config: ProjectConfig,
    model_name: str | None = None,
    shepherd_version: str | None = None,
) -> Path:
    """Create a versioned experiment-model path without touching the main baseline."""
    experiment_dir = get_experiment_model_dir(config)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_suffix = model_name or "model"
    safe_model_suffix = str(model_suffix).replace(" ", "_")
    version_slug = slugify_shepherd_version(shepherd_version or "SHEP experiment")
    return experiment_dir / f"{version_slug}_{safe_model_suffix}_{timestamp}.joblib"


def save_trained_model(
    model,
    config: ProjectConfig,
    metrics: dict[str, float],
    data_message: str,
    num_examples: int,
    model_name: str | None = None,
    recommended_threshold: float | None = None,
    save_mode: str = "baseline",
) -> Path:
    """Save the trained model and a small amount of metadata to disk."""
    shepherd_version = get_checkpoint_version_for_stage(config.stage_name).display_name
    numeric_version = get_numeric_version(shepherd_version)
    if save_mode == "experiment":
        shepherd_version, numeric_version = _next_experiment_version(config)
        model_path = get_experiment_model_path(
            config,
            model_name=model_name,
            shepherd_version=shepherd_version,
        )
    else:
        model_path = get_model_path(config)
    payload = {
        "model": model,
        "stage_name": config.stage_name,
        "metrics": metrics,
        "data_message": data_message,
        "num_examples": num_examples,
        "model_name": model_name,
        "recommended_threshold": recommended_threshold,
        "shepherd_version": shepherd_version,
        "numeric_version": numeric_version,
        "model_path": str(model_path),
    }
    joblib.dump(payload, model_path)
    if save_mode == "experiment":
        _register_experiment_version(
            config,
            model_path=model_path,
            display_name=shepherd_version,
            numeric_version=numeric_version,
            model_name=model_name,
            recommended_threshold=recommended_threshold,
        )
    return model_path


def load_trained_model(config: ProjectConfig, model_path: Path | None = None) -> dict | None:
    """Load a previously saved model if it exists."""
    candidate_paths = [model_path] if model_path is not None else [get_model_path(config)]
    candidate_paths.append(get_legacy_model_path(config))
    for model_path in candidate_paths:
        if model_path is None:
            continue
        if model_path.exists():
            payload = joblib.load(model_path)
            payload.setdefault("model_path", str(model_path))
            payload.setdefault(
                "shepherd_version",
                get_checkpoint_version_for_stage(config.stage_name).display_name,
            )
            payload.setdefault(
                "numeric_version",
                get_numeric_version(
                    get_checkpoint_version_for_stage(config.stage_name).display_name
                ),
            )
            return payload
    return None
