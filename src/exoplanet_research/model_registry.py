"""Friendly model-version registry for the Shepherd.Ai app."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelVersion:
    """Metadata shown in the app for a selectable saved model."""

    version_id: str
    display_name: str
    description: str
    stage_name: str
    recommended_threshold: float


MODEL_VERSIONS = [
    ModelVersion(
        version_id="shep-1-0",
        display_name="SHEP 1.0",
        description="Original Kepler baseline screener",
        stage_name="stage2",
        recommended_threshold=0.60,
    ),
    ModelVersion(
        version_id="shep-1-1",
        display_name="SHEP 1.1",
        description="Current Kepler specialist with real-label training",
        stage_name="stage3",
        recommended_threshold=0.4242,
    ),
    ModelVersion(
        version_id="shep-1-2",
        display_name="SHEP 1.2",
        description="Experimental TESS screener",
        stage_name="stage4",
        recommended_threshold=0.60,
    ),
]


def get_model_version(version_id: str) -> ModelVersion:
    """Return the metadata for one registered Shepherd.Ai version."""
    for version in MODEL_VERSIONS:
        if version.version_id == version_id:
            return version
    raise ValueError(f"Unknown model version: {version_id}")
