"""Evaluation helpers for the baseline classifier."""

from __future__ import annotations

from pathlib import Path

import json
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute standard binary classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def compute_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute the confusion matrix."""
    return confusion_matrix(y_true, y_pred)


def compute_probability_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Compute threshold-independent ranking metrics."""
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
    }


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Search for the threshold that maximizes F1 on the current evaluation split."""
    candidate_thresholds = np.unique(np.round(y_prob, 4))
    candidate_thresholds = np.concatenate(([0.05], candidate_thresholds, [0.95]))

    best_result = {
        "threshold": 0.5,
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": -1.0,
    }

    for threshold in candidate_thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        metrics = compute_metrics(y_true, y_pred)
        if metrics["f1_score"] > best_result["f1_score"]:
            best_result = {
                "threshold": float(threshold),
                **metrics,
            }

    return best_result


def compute_curve_data(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, list[float]]:
    """Return ROC and precision-recall curve points for plotting and reporting."""
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
    return {
        "roc_fpr": fpr.tolist(),
        "roc_tpr": tpr.tolist(),
        "roc_thresholds": roc_thresholds.tolist(),
        "pr_precision": precision.tolist(),
        "pr_recall": recall.tolist(),
        "pr_thresholds": pr_thresholds.tolist(),
    }


def save_metrics(
    metrics: dict[str, float],
    confusion: np.ndarray,
    output_dir: Path,
    probability_metrics: dict[str, float] | None = None,
    best_threshold: dict[str, float] | None = None,
) -> None:
    """Save evaluation results for later reference."""
    payload = {
        "metrics": metrics,
        "confusion_matrix": confusion.tolist(),
    }
    if probability_metrics is not None:
        payload["probability_metrics"] = probability_metrics
    if best_threshold is not None:
        payload["best_threshold"] = best_threshold
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_model_comparison(results: list[dict[str, object]], output_dir: Path) -> None:
    """Save side-by-side comparison metrics for multiple models."""
    (output_dir / "model_comparison.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
