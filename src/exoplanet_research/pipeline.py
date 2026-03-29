"""End-to-end pipeline for the exoplanet transit research project."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import numpy as np

from .config import build_config, ensure_directories
from .dataset_builder import prepare_training_dataset
from .evaluation import (
    compute_confusion,
    compute_curve_data,
    compute_metrics,
    compute_probability_metrics,
    find_best_threshold,
    save_metrics,
    save_model_comparison,
)
from .modeling import train_model_comparison
from .model_io import save_trained_model
from .output_manager import append_experiment_log, prepare_output_directory
from .visualization import (
    plot_confusion,
    plot_labeled_windows,
    plot_light_curve_examples,
    plot_precision_recall_curve,
    plot_prediction_scores,
    plot_roc_curve,
)


def _save_summary_text(summary_text: str, output_path: Path) -> None:
    """Write a human-readable run summary."""
    output_path.write_text(summary_text, encoding="utf-8")


def run_pipeline(stage: str = "stage1", allow_fallback: bool = True) -> None:
    """Run the full baseline project from data loading to evaluation."""
    config = build_config(stage)
    ensure_directories(config)
    run_output_dir = prepare_output_directory(config.output_dir)

    # Step 1: Prepare and save the dataset so it can be reused in future training runs.
    prepared_dataset = prepare_training_dataset(config, allow_fallback=allow_fallback)
    X, y = prepared_dataset.X, prepared_dataset.y

    # Step 2: Train and compare multiple simple baseline models on the prepared dataset.
    model_artifacts_list = train_model_comparison(X, y, config)
    comparison_rows: list[dict[str, object]] = []
    best_artifacts = None
    best_metrics = None
    best_confusion = None
    best_probability_metrics = None
    best_threshold = None
    best_curve_data = None
    best_f1 = -1.0

    for artifacts_candidate in model_artifacts_list:
        candidate_metrics = compute_metrics(artifacts_candidate.y_test, artifacts_candidate.y_pred)
        candidate_confusion = compute_confusion(artifacts_candidate.y_test, artifacts_candidate.y_pred)
        candidate_probability_metrics = compute_probability_metrics(
            artifacts_candidate.y_test,
            artifacts_candidate.y_prob,
        )
        candidate_best_threshold = find_best_threshold(
            artifacts_candidate.y_test,
            artifacts_candidate.y_prob,
        )
        comparison_rows.append(
            {
                "model_name": artifacts_candidate.model_name,
                "metrics": candidate_metrics,
                "probability_metrics": candidate_probability_metrics,
                "best_threshold": candidate_best_threshold,
                "confusion_matrix": candidate_confusion.tolist(),
            }
        )
        if candidate_metrics["f1_score"] > best_f1:
            best_f1 = candidate_metrics["f1_score"]
            best_artifacts = artifacts_candidate
            best_metrics = candidate_metrics
            best_confusion = candidate_confusion
            best_probability_metrics = candidate_probability_metrics
            best_threshold = candidate_best_threshold
            best_curve_data = compute_curve_data(
                artifacts_candidate.y_test,
                artifacts_candidate.y_prob,
            )

    artifacts = best_artifacts
    metrics = best_metrics
    confusion = best_confusion

    # Step 3: Save the best model and the comparison results.
    save_metrics(
        metrics,
        confusion,
        run_output_dir,
        probability_metrics=best_probability_metrics,
        best_threshold=best_threshold,
    )
    save_model_comparison(comparison_rows, run_output_dir)
    model_path = save_trained_model(
        model=artifacts.model,
        config=config,
        metrics=metrics,
        data_message=prepared_dataset.data_message,
        num_examples=prepared_dataset.num_examples,
    )

    # Step 4: Save figures and summaries for the latest run only.
    plot_confusion(confusion, run_output_dir)
    plot_prediction_scores(artifacts.y_prob, artifacts.y_test, run_output_dir)
    plot_roc_curve(
        np.asarray(best_curve_data["roc_fpr"], dtype=float),
        np.asarray(best_curve_data["roc_tpr"], dtype=float),
        best_probability_metrics["roc_auc"],
        run_output_dir,
    )
    plot_precision_recall_curve(
        np.asarray(best_curve_data["pr_recall"], dtype=float),
        np.asarray(best_curve_data["pr_precision"], dtype=float),
        best_probability_metrics["average_precision"],
        run_output_dir,
    )

    summary_lines = [
        "Exoplanet Transit Baseline Run",
        "================================",
        f"Stage: {config.stage_name}",
        f"Data status: {prepared_dataset.data_message}",
        "",
        "Loaded light curves:",
        prepared_dataset.summary_df_text,
        "",
        f"Requested targets: {len(config.kepler_targets)}",
        f"Loaded light curves: {prepared_dataset.num_lightcurves}",
        f"Number of windows: {prepared_dataset.num_windows}",
        f"Number of labeled examples: {prepared_dataset.num_examples}",
        f"Training examples: {len(artifacts.X_train)}",
        f"Test examples: {len(artifacts.X_test)}",
        f"Labeling mode: {config.labeling_mode}",
        f"Prepared dataset path: {prepared_dataset.dataset_path}",
        "",
        "Metrics:",
    ]
    if prepared_dataset.catalog_message:
        summary_lines.append(f"Catalog status: {prepared_dataset.catalog_message}")
    if prepared_dataset.real_label_report is not None:
        real_label_report = prepared_dataset.real_label_report
        summary_lines.append(
            "Real-label summary: "
            f"matched targets={real_label_report.matched_targets}, "
            f"unmatched targets={real_label_report.unmatched_targets}, "
            f"positive windows={real_label_report.positive_examples}, "
            f"negative windows={real_label_report.negative_examples}, "
            f"skipped ambiguous windows={real_label_report.skipped_examples}"
        )
    if prepared_dataset.catalog_message or prepared_dataset.real_label_report is not None:
        summary_lines.append("")
    summary_lines.append("Model comparison:")
    for row in comparison_rows:
        row_metrics = row["metrics"]
        summary_lines.append(
            f"- {row['model_name']}: "
            f"accuracy={row_metrics['accuracy']:.4f}, "
            f"precision={row_metrics['precision']:.4f}, "
            f"recall={row_metrics['recall']:.4f}, "
            f"f1={row_metrics['f1_score']:.4f}"
        )
    summary_lines.append("")
    summary_lines.append(f"Best model: {artifacts.model_name}")
    summary_lines.append("")
    summary_lines.extend([f"- {name}: {value:.4f}" for name, value in metrics.items()])
    summary_lines.append(f"- roc_auc: {best_probability_metrics['roc_auc']:.4f}")
    summary_lines.append(f"- average_precision: {best_probability_metrics['average_precision']:.4f}")
    summary_lines.append("")
    summary_lines.append("Best threshold on evaluation split:")
    summary_lines.append(f"- threshold: {best_threshold['threshold']:.4f}")
    summary_lines.append(f"- accuracy: {best_threshold['accuracy']:.4f}")
    summary_lines.append(f"- precision: {best_threshold['precision']:.4f}")
    summary_lines.append(f"- recall: {best_threshold['recall']:.4f}")
    summary_lines.append(f"- f1_score: {best_threshold['f1_score']:.4f}")
    summary_lines.append("")
    summary_lines.append(f"Saved model path: {model_path}")
    summary_lines.append("")
    summary_lines.append(f"Confusion matrix:\n{confusion}")

    run_summary = "\n".join(summary_lines)
    _save_summary_text(run_summary, run_output_dir / "run_summary.txt")

    experiment_log_path = append_experiment_log(
        config.output_dir,
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "stage": config.stage_name,
            "labeling_mode": config.labeling_mode,
            "num_lightcurves": prepared_dataset.num_lightcurves,
            "num_windows": prepared_dataset.num_windows,
            "num_examples": prepared_dataset.num_examples,
            "best_model": artifacts.model_name,
            "accuracy": f"{metrics['accuracy']:.4f}",
            "precision": f"{metrics['precision']:.4f}",
            "recall": f"{metrics['recall']:.4f}",
            "f1_score": f"{metrics['f1_score']:.4f}",
            "roc_auc": f"{best_probability_metrics['roc_auc']:.4f}",
            "average_precision": f"{best_probability_metrics['average_precision']:.4f}",
            "best_threshold": f"{best_threshold['threshold']:.4f}",
            "dataset_path": str(prepared_dataset.dataset_path),
            "model_path": str(model_path),
        },
    )
    print(f"Experiment log: {experiment_log_path}")
    print(run_summary)
