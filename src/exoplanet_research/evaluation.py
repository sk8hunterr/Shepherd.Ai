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


def _target_family(target_name: object) -> str:
    """Group targets into broad families for easier diagnostic reporting."""
    normalized = str(target_name).strip().lower()
    if normalized.startswith("kepler-"):
        return "kepler_named_koi_host"
    if normalized.startswith("kic "):
        return "kic_false_positive_or_control"
    if normalized.startswith("tic "):
        return "tic_tess_target"
    return "other_target"


def _binary_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    """Return named confusion-matrix counts for binary predictions."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "errors": int(fp + fn),
    }


def _diagnostic_metrics(
    group_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    target_family: str | None = None,
) -> dict[str, object]:
    """Build one compact diagnostic row for a target or target family."""
    counts = _binary_counts(y_true, y_pred)
    metrics = compute_metrics(y_true, y_pred)
    row: dict[str, object] = {
        "group": group_name,
        "examples": int(len(y_true)),
        "actual_positive": int(np.sum(y_true == 1)),
        "actual_negative": int(np.sum(y_true == 0)),
        "predicted_positive": int(np.sum(y_pred == 1)),
        "predicted_positive_rate": float(np.mean(y_pred == 1)),
        "mean_probability": float(np.mean(y_prob)),
        **counts,
        **metrics,
    }
    if target_family is not None:
        row["target_family"] = target_family
    return row


def _group_diagnostics(
    group_values: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    include_target_family: bool = False,
    target_roles: np.ndarray | None = None,
) -> list[dict[str, object]]:
    """Compute diagnostics for each distinct group value."""
    rows: list[dict[str, object]] = []
    for group_value in sorted({str(value) for value in group_values}):
        mask = np.asarray([str(value) == group_value for value in group_values])
        family = _target_family(group_value) if include_target_family else None
        if include_target_family and target_roles is not None:
            matching_roles = sorted({str(value) for value in target_roles[mask]})
            family = matching_roles[0] if len(matching_roles) == 1 else "mixed_target_roles"
        rows.append(
            _diagnostic_metrics(
                group_name=group_value,
                y_true=y_true[mask],
                y_pred=y_pred[mask],
                y_prob=y_prob[mask],
                target_family=family,
            )
        )
    return rows


def _target_family_diagnostics(
    target_names: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    target_roles: np.ndarray | None = None,
) -> list[dict[str, object]]:
    """Compute diagnostics by broad target family."""
    family_values = (
        np.asarray(target_roles, dtype=object)
        if target_roles is not None
        else np.asarray([_target_family(value) for value in target_names], dtype=object)
    )
    return _group_diagnostics(family_values, y_true, y_pred, y_prob)


def _worst_target_rows(
    target_rows: list[dict[str, object]],
    top_n: int,
) -> list[dict[str, object]]:
    """Return the target rows with the most incorrect predictions."""
    return sorted(
        target_rows,
        key=lambda row: (
            int(row["errors"]),
            int(row["false_positives"]),
            int(row["false_negatives"]),
            int(row["examples"]),
        ),
        reverse=True,
    )[:top_n]


def _target_event_level_diagnostics(
    target_names: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    target_roles: np.ndarray | None = None,
    top_n: int = 15,
) -> dict[str, object]:
    """Summarize detector behavior at the target/candidate-event level."""
    target_rows: list[dict[str, object]] = []
    target_actual: list[int] = []
    target_predicted: list[int] = []
    target_probability: list[float] = []
    target_role_values: list[str] = []

    for target_name in sorted({str(value) for value in target_names}):
        mask = np.asarray([str(value) == target_name for value in target_names])
        actual_value = int(np.any(y_true[mask] == 1))
        predicted_value = int(np.any(y_pred[mask] == 1))
        probability_value = float(np.max(y_prob[mask]))
        if target_roles is not None:
            matching_roles = sorted({str(value) for value in target_roles[mask]})
            target_role = matching_roles[0] if len(matching_roles) == 1 else "mixed_target_roles"
        else:
            target_role = _target_family(target_name)

        row = _diagnostic_metrics(
            group_name=target_name,
            y_true=np.asarray([actual_value], dtype=int),
            y_pred=np.asarray([predicted_value], dtype=int),
            y_prob=np.asarray([probability_value], dtype=float),
            target_family=target_role,
        )
        row["window_examples"] = int(np.sum(mask))
        row["positive_windows"] = int(np.sum(y_true[mask] == 1))
        row["flagged_windows"] = int(np.sum(y_pred[mask] == 1))
        target_rows.append(row)
        target_actual.append(actual_value)
        target_predicted.append(predicted_value)
        target_probability.append(probability_value)
        target_role_values.append(target_role)

    target_actual_array = np.asarray(target_actual, dtype=int)
    target_predicted_array = np.asarray(target_predicted, dtype=int)
    target_probability_array = np.asarray(target_probability, dtype=float)
    target_role_array = np.asarray(target_role_values, dtype=object)

    return {
        "overall": _diagnostic_metrics(
            "target_event_overall",
            target_actual_array,
            target_predicted_array,
            target_probability_array,
        ),
        "by_target_family": _group_diagnostics(
            target_role_array,
            target_actual_array,
            target_predicted_array,
            target_probability_array,
        ),
        "worst_targets": _worst_target_rows(target_rows, top_n),
        "all_targets": target_rows,
    }


def build_diagnostic_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    target_names: np.ndarray,
    target_roles: np.ndarray | None = None,
    example_roles: np.ndarray | None = None,
    best_threshold: dict[str, float] | None = None,
    top_n: int = 15,
) -> dict[str, object]:
    """Build target-level and target-family diagnostics for the evaluation split."""
    target_names = np.asarray(target_names, dtype=object)
    if target_roles is not None:
        target_roles = np.asarray(target_roles, dtype=object)
    if example_roles is not None:
        example_roles = np.asarray(example_roles, dtype=object)
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    target_rows = _group_diagnostics(
        target_names,
        y_true,
        y_pred,
        y_prob,
        include_target_family=True,
        target_roles=target_roles,
    )
    report: dict[str, object] = {
        "default_threshold": {
            "overall": _diagnostic_metrics("overall", y_true, y_pred, y_prob),
            "by_target_family": _target_family_diagnostics(
                target_names,
                y_true,
                y_pred,
                y_prob,
                target_roles=target_roles,
            ),
            "by_example_role": (
                _group_diagnostics(example_roles, y_true, y_pred, y_prob)
                if example_roles is not None
                else []
            ),
            "worst_targets": _worst_target_rows(target_rows, top_n),
            "all_targets": target_rows,
            "event_level": _target_event_level_diagnostics(
                target_names,
                y_true,
                y_pred,
                y_prob,
                target_roles=target_roles,
                top_n=top_n,
            ),
        }
    }

    if best_threshold is not None:
        threshold = float(best_threshold["threshold"])
        tuned_pred = (y_prob >= threshold).astype(int)
        tuned_target_rows = _group_diagnostics(
            target_names,
            y_true,
            tuned_pred,
            y_prob,
            include_target_family=True,
            target_roles=target_roles,
        )
        report["tuned_threshold"] = {
            "threshold": threshold,
            "overall": _diagnostic_metrics("overall", y_true, tuned_pred, y_prob),
            "by_target_family": _target_family_diagnostics(
                target_names,
                y_true,
                tuned_pred,
                y_prob,
                target_roles=target_roles,
            ),
            "by_example_role": (
                _group_diagnostics(example_roles, y_true, tuned_pred, y_prob)
                if example_roles is not None
                else []
            ),
            "worst_targets": _worst_target_rows(tuned_target_rows, top_n),
            "all_targets": tuned_target_rows,
            "event_level": _target_event_level_diagnostics(
                target_names,
                y_true,
                tuned_pred,
                y_prob,
                target_roles=target_roles,
                top_n=top_n,
            ),
        }

    return report


def _format_diagnostic_section(section_name: str, section: dict[str, object]) -> list[str]:
    """Format one diagnostic section into readable summary lines."""
    overall = section["overall"]
    lines = [
        section_name,
        "-" * len(section_name),
        (
            "Overall: "
            f"examples={overall['examples']}, "
            f"accuracy={overall['accuracy']:.4f}, "
            f"precision={overall['precision']:.4f}, "
            f"recall={overall['recall']:.4f}, "
            f"f1={overall['f1_score']:.4f}, "
            f"false_positives={overall['false_positives']}, "
            f"false_negatives={overall['false_negatives']}"
        ),
        "",
        "By target role/family:",
    ]
    for row in section["by_target_family"]:
        lines.append(
            f"- {row['group']}: "
            f"examples={row['examples']}, "
            f"pos={row['actual_positive']}, "
            f"neg={row['actual_negative']}, "
            f"f1={row['f1_score']:.4f}, "
            f"fp={row['false_positives']}, "
            f"fn={row['false_negatives']}, "
            f"mean_prob={row['mean_probability']:.4f}"
        )

    if section.get("by_example_role"):
        lines.extend(["", "By example role:"])
        for row in section["by_example_role"]:
            lines.append(
                f"- {row['group']}: "
                f"examples={row['examples']}, "
                f"pos={row['actual_positive']}, "
                f"neg={row['actual_negative']}, "
                f"f1={row['f1_score']:.4f}, "
                f"fp={row['false_positives']}, "
                f"fn={row['false_negatives']}, "
                f"mean_prob={row['mean_probability']:.4f}"
            )

    lines.extend(["", "Worst targets by error count:"])
    for row in section["worst_targets"]:
        lines.append(
            f"- {row['group']} ({row['target_family']}): "
            f"examples={row['examples']}, "
            f"pos={row['actual_positive']}, "
            f"neg={row['actual_negative']}, "
            f"f1={row['f1_score']:.4f}, "
            f"fp={row['false_positives']}, "
            f"fn={row['false_negatives']}, "
            f"mean_prob={row['mean_probability']:.4f}"
        )

    if section.get("event_level"):
        event_level = section["event_level"]
        event_overall = event_level["overall"]
        lines.extend(
            [
                "",
                "Event-level target summary:",
                (
                    "- overall: "
                    f"targets={event_overall['examples']}, "
                    f"accuracy={event_overall['accuracy']:.4f}, "
                    f"precision={event_overall['precision']:.4f}, "
                    f"recall={event_overall['recall']:.4f}, "
                    f"f1={event_overall['f1_score']:.4f}, "
                    f"false_positive_targets={event_overall['false_positives']}, "
                    f"missed_positive_targets={event_overall['false_negatives']}"
                ),
                "- by target role:",
            ]
        )
        for row in event_level["by_target_family"]:
            lines.append(
                f"  - {row['group']}: "
                f"targets={row['examples']}, "
                f"actual_pos={row['actual_positive']}, "
                f"actual_neg={row['actual_negative']}, "
                f"f1={row['f1_score']:.4f}, "
                f"fp_targets={row['false_positives']}, "
                f"missed_targets={row['false_negatives']}, "
                f"mean_prob={row['mean_probability']:.4f}"
            )
    return lines


def save_diagnostic_report(report: dict[str, object], output_dir: Path) -> None:
    """Save model diagnostics as JSON and readable text."""
    (output_dir / "diagnostic_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    lines = [
        "Training Diagnostic Report",
        "==========================",
        "",
        "This report breaks the evaluation split down by target role/family and target.",
        "It helps identify whether the model is failing on positives, hard negatives, or specific stars.",
        "",
    ]
    lines.extend(_format_diagnostic_section("Default threshold", report["default_threshold"]))

    if "tuned_threshold" in report:
        tuned_threshold = report["tuned_threshold"]["threshold"]
        lines.extend(["", "", f"Tuned threshold ({tuned_threshold:.4f})"])
        lines.extend(_format_diagnostic_section("Tuned threshold results", report["tuned_threshold"]))

    (output_dir / "diagnostic_report.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
