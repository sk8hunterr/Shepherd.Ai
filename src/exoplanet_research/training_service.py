"""Helpers for preparing a baseline model for the GUI."""

from __future__ import annotations

from dataclasses import dataclass
import os

from .config import ProjectConfig, ensure_directories
from .data_loader import load_light_curves
from .evaluation import compute_metrics
from .labeling import create_labeled_examples, examples_to_arrays_with_targets
from .modeling import train_baseline_model
from .model_io import get_model_path, get_model_path_for_version, load_trained_model, save_trained_model
from .model_registry import ModelVersion, get_model_version
from .preprocessing import build_window_dataset


@dataclass
class ScreeningModelBundle:
    """Trained baseline model and summary details for the app."""

    model: object
    config: ProjectConfig
    stage_name: str
    data_message: str
    num_examples: int
    metrics: dict[str, float]
    model_path: str
    loaded_from_disk: bool
    version_name: str
    version_description: str
    recommended_threshold: float


def _bundle_from_saved_payload(
    saved_payload: dict,
    config: ProjectConfig,
    model_version: ModelVersion,
) -> ScreeningModelBundle:
    """Convert a saved model payload into the app bundle format."""
    return ScreeningModelBundle(
        model=saved_payload["model"],
        config=config,
        stage_name=str(saved_payload.get("stage_name", config.stage_name)),
        data_message=str(saved_payload.get("data_message", "Saved model loaded from disk.")),
        num_examples=int(saved_payload.get("num_examples", 0)),
        metrics=dict(saved_payload.get("metrics", {})),
        model_path=str(saved_payload.get("model_path", get_model_path(config))),
        loaded_from_disk=True,
        version_name=model_version.display_name,
        version_description=model_version.description,
        recommended_threshold=float(
            saved_payload.get("recommended_threshold", model_version.recommended_threshold)
        ),
    )


def prepare_screening_model(version_id: str) -> ScreeningModelBundle:
    """Load an existing Shepherd.Ai version or train a fallback one for the app."""
    model_version = get_model_version(version_id)
    config = ProjectConfig(stage_name=model_version.stage_name)
    ensure_directories(config)

    saved_payload = load_trained_model(
        config,
        model_path=get_model_path_for_version(config, model_version),
    )
    if saved_payload is not None:
        return _bundle_from_saved_payload(
            saved_payload=saved_payload,
            config=config,
            model_version=model_version,
        )

    allow_runtime_training = os.getenv("SHEPHERD_ALLOW_RUNTIME_TRAINING", "0") == "1"
    if not allow_runtime_training:
        raise FileNotFoundError(
            "No saved model was found for this Shepherd version. "
            "For deployment, include the packaged model files in the repository, or set "
            "SHEPHERD_ALLOW_RUNTIME_TRAINING=1 for a local fallback."
        )

    lightcurves, data_message = load_light_curves(config, allow_fallback=True)
    windows = build_window_dataset(lightcurves, config)
    examples = create_labeled_examples(windows, config)
    X, y, target_names = examples_to_arrays_with_targets(examples)
    artifacts = train_baseline_model(X, y, config, groups=target_names)
    metrics = compute_metrics(artifacts.y_test, artifacts.y_pred)
    model_path = save_trained_model(
        model=artifacts.model,
        config=config,
        metrics=metrics,
        data_message=data_message,
        num_examples=len(examples),
        model_name=artifacts.model_name,
        recommended_threshold=model_version.recommended_threshold,
    )

    return ScreeningModelBundle(
        model=artifacts.model,
        config=config,
        stage_name=config.stage_name,
        data_message=data_message,
        num_examples=len(examples),
        metrics=metrics,
        model_path=str(model_path),
        loaded_from_disk=False,
        version_name=model_version.display_name,
        version_description=model_version.description,
        recommended_threshold=model_version.recommended_threshold,
    )
