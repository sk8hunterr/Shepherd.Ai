"""Plotting utilities for light curves and model outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from .data_loader import LoadedLightCurve
from .labeling import LabeledExample


def plot_light_curve_examples(lightcurves: list[LoadedLightCurve], output_dir: Path) -> None:
    """Plot a few raw light curves to show the input data."""
    num_curves = min(3, len(lightcurves))
    fig, axes = plt.subplots(num_curves, 1, figsize=(10, 3 * num_curves), sharex=False)
    axes = np.atleast_1d(axes)

    for axis, curve in zip(axes, lightcurves[:num_curves]):
        axis.plot(curve.time, curve.flux, linewidth=0.8)
        axis.set_title(f"Raw light curve: {curve.target_name} ({curve.source})")
        axis.set_xlabel("Time")
        axis.set_ylabel("Flux")

    fig.tight_layout()
    fig.savefig(output_dir / "light_curve_examples.png", dpi=150)
    plt.close(fig)


def plot_labeled_windows(examples: list[LabeledExample], output_dir: Path) -> None:
    """Plot one non-transit and one transit example."""
    negatives = [example for example in examples if example.label == 0]
    positives = [example for example in examples if example.label == 1]
    selected = []
    if negatives:
        selected.append(("Non-transit example", negatives[0]))
    if positives:
        selected.append(("Transit example", positives[0]))

    fig, axes = plt.subplots(len(selected), 1, figsize=(10, 3 * len(selected)), sharex=True)
    axes = np.atleast_1d(axes)

    for axis, (title, example) in zip(axes, selected):
        axis.plot(example.flux, linewidth=1.0)
        axis.set_title(title)
        axis.set_xlabel("Window index")
        axis.set_ylabel("Normalized flux")

    fig.tight_layout()
    fig.savefig(output_dir / "window_examples.png", dpi=150)
    plt.close(fig)


def plot_confusion(confusion: np.ndarray, output_dir: Path) -> None:
    """Plot the confusion matrix."""
    fig, ax = plt.subplots(figsize=(5, 5))
    display = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=["No transit", "Transit"])
    display.plot(ax=ax, colorbar=False)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_prediction_scores(y_prob: np.ndarray, y_true: np.ndarray, output_dir: Path) -> None:
    """Plot predicted transit probabilities for the test set."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(range(len(y_prob)), y_prob, c=y_true, cmap="coolwarm", alpha=0.7)
    ax.set_title("Predicted transit probability on test windows")
    ax.set_xlabel("Test example index")
    ax.set_ylabel("Predicted probability")
    fig.tight_layout()
    fig.savefig(output_dir / "prediction_scores.png", dpi=150)
    plt.close(fig)


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    output_dir: Path,
) -> None:
    """Plot ROC curve for the evaluation split."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=150)
    plt.close(fig)


def plot_precision_recall_curve(
    recall: np.ndarray,
    precision: np.ndarray,
    average_precision: float,
    output_dir: Path,
) -> None:
    """Plot precision-recall curve for the evaluation split."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(recall, precision, label=f"AP = {average_precision:.3f}", linewidth=2)
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(output_dir / "precision_recall_curve.png", dpi=150)
    plt.close(fig)
