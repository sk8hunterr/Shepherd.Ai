"""Helpers for simpler output organization and experiment logging."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import csv


LATEST_RUN_DIRNAME = "latest_run"
ARCHIVE_DIRNAME = "archive"
EXPERIMENT_LOG_FILENAME = "experiment_log.csv"


def prepare_output_directory(output_root: Path) -> Path:
    """Keep a single latest_run folder and archive the previous contents when needed."""
    latest_run = output_root / LATEST_RUN_DIRNAME
    archive_dir = output_root / ARCHIVE_DIRNAME
    archive_dir.mkdir(parents=True, exist_ok=True)

    if latest_run.exists() and any(latest_run.iterdir()):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archived_run = archive_dir / f"run_{timestamp}"
        try:
            latest_run.rename(archived_run)
        except PermissionError:
            # If latest_run is open in another process (for example the live app),
            # keep it in place and write the new training outputs to a timestamped folder.
            archived_run.mkdir(parents=True, exist_ok=True)
            return archived_run

    latest_run.mkdir(parents=True, exist_ok=True)
    return latest_run


def append_experiment_log(output_root: Path, row: dict[str, object]) -> Path:
    """Append one run summary row to a simple experiment log CSV."""
    log_path = output_root / EXPERIMENT_LOG_FILENAME
    fieldnames = list(row.keys())
    write_header = not log_path.exists()

    with log_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return log_path
