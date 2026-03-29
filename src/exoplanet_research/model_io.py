"""Utilities for saving and loading trained models."""

from __future__ import annotations

from pathlib import Path

import joblib

from .config import ProjectConfig


def get_model_path(config: ProjectConfig) -> Path:
    """Return the default file path for the trained model of a given stage."""
    return config.model_dir / f"baseline_{config.stage_name}_model.joblib"


def save_trained_model(
    model,
    config: ProjectConfig,
    metrics: dict[str, float],
    data_message: str,
    num_examples: int,
) -> Path:
    """Save the trained model and a small amount of metadata to disk."""
    model_path = get_model_path(config)
    payload = {
        "model": model,
        "stage_name": config.stage_name,
        "metrics": metrics,
        "data_message": data_message,
        "num_examples": num_examples,
    }
    joblib.dump(payload, model_path)
    return model_path


def load_trained_model(config: ProjectConfig) -> dict | None:
    """Load a previously saved model if it exists."""
    model_path = get_model_path(config)
    if not model_path.exists():
        return None
    return joblib.load(model_path)
