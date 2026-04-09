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
    model_filename: str
    checkpoint_for_stage: bool = False


MODEL_VERSIONS = [
    ModelVersion(
        version_id="shep-1-0",
        display_name="SHEP 1.0",
        description="Original Kepler baseline screener",
        stage_name="stage2",
        recommended_threshold=0.60,
        model_filename="shep_1_0_checkpoint.joblib",
        checkpoint_for_stage=True,
    ),
    ModelVersion(
        version_id="shep-1-1",
        display_name="SHEP 1.1",
        description="Previous Kepler checkpoint kept for comparison",
        stage_name="stage3",
        recommended_threshold=0.4242,
        model_filename="shep_1_1_checkpoint.joblib",
    ),
    ModelVersion(
        version_id="shep-1-2",
        display_name="SHEP 1.2",
        description="Promoted Kepler checkpoint with event-aware false-positive rejection",
        stage_name="stage3",
        recommended_threshold=0.5201,
        model_filename="shep_1_2_checkpoint.joblib",
        checkpoint_for_stage=True,
    ),
    ModelVersion(
        version_id="shep-tess-beta",
        display_name="SHEP TESS Beta",
        description="Experimental TESS screener",
        stage_name="stage4",
        recommended_threshold=0.60,
        model_filename="shep_tess_beta_checkpoint.joblib",
        checkpoint_for_stage=True,
    ),
]


def get_model_version(version_id: str) -> ModelVersion:
    """Return the metadata for one registered Shepherd.Ai version."""
    for version in MODEL_VERSIONS:
        if version.version_id == version_id:
            return version
    raise ValueError(f"Unknown model version: {version_id}")


def get_checkpoint_version_for_stage(stage_name: str) -> ModelVersion:
    """Return the checkpoint-style Shepherd version associated with one stage."""
    for version in MODEL_VERSIONS:
        if version.stage_name == stage_name and version.checkpoint_for_stage:
            return version
    for version in MODEL_VERSIONS:
        if version.stage_name == stage_name:
            return version
    raise ValueError(f"No checkpoint Shepherd version is registered for stage: {stage_name}")


def get_numeric_version(display_name: str) -> str:
    """Extract the numeric portion from a Shepherd display name like 'SHEP 1.1'."""
    prefix = "SHEP "
    if not display_name.startswith(prefix):
        raise ValueError(f"Unexpected Shepherd display name: {display_name}")
    return display_name[len(prefix):].strip()


def format_experiment_display_name(checkpoint_numeric_version: str, patch_number: int) -> str:
    """Create a patch-style experiment name such as 'SHEP 1.1.1'."""
    return f"SHEP {checkpoint_numeric_version}.{patch_number}"


def slugify_shepherd_version(display_name: str) -> str:
    """Convert a Shepherd display name into a filename-safe slug."""
    return display_name.lower().replace(" ", "_").replace(".", "_")
