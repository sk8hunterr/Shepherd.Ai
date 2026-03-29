"""Entry point for the exoplanet transit baseline project."""

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from exoplanet_research.pipeline import run_pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the exoplanet transit baseline pipeline."
    )
    parser.add_argument(
        "--stage",
        choices=["stage1", "stage2", "stage3", "stage4"],
        default="stage1",
        help="Choose a project stage to train and evaluate.",
    )
    parser.add_argument(
        "--real-only",
        action="store_true",
        help="Fail instead of using fallback synthetic data when real mission downloads are unavailable.",
    )
    args = parser.parse_args()

    run_pipeline(stage=args.stage, allow_fallback=not args.real_only)
