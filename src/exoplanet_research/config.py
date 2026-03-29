"""Project configuration for the exoplanet transit research prototype."""

from dataclasses import dataclass, field
from pathlib import Path


STAGE_TARGETS = {
    "stage1": ["Kepler-10", "Kepler-22", "Kepler-452"],
    "stage2": [
        "Kepler-10",
        "Kepler-11",
        "Kepler-18",
        "Kepler-20",
        "Kepler-22",
        "Kepler-37",
        "Kepler-62",
        "Kepler-69",
        "Kepler-90",
        "Kepler-186",
        "Kepler-296",
        "Kepler-438",
        "Kepler-442",
        "Kepler-452",
        "Kepler-440",
        "Kepler-444",
        "Kepler-1649",
        "Kepler-1544",
        "Kepler-1638",
        "Kepler-1652",
        "Kepler-5",
        "Kepler-7",
        "Kepler-8",
        "Kepler-9",
        "Kepler-12",
        "Kepler-13",
        "Kepler-14",
        "Kepler-15",
        "Kepler-16",
        "Kepler-17",
        "Kepler-19",
        "Kepler-21",
        "Kepler-68",
        "Kepler-70",
        "Kepler-130",
    ],
    "stage3": [
        "Kepler-10",
        "Kepler-11",
        "Kepler-18",
        "Kepler-20",
        "Kepler-22",
        "Kepler-37",
        "Kepler-62",
        "Kepler-69",
        "Kepler-90",
        "Kepler-186",
        "Kepler-296",
        "Kepler-438",
        "Kepler-442",
        "Kepler-452",
        "Kepler-440",
        "Kepler-444",
        "Kepler-1649",
        "Kepler-1544",
        "Kepler-1638",
        "Kepler-1652",
    ],
    "stage4": [],
}


@dataclass
class ProjectConfig:
    """Settings used across the project."""

    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = field(init=False)
    cache_dir: Path = field(init=False)
    prepared_data_dir: Path = field(init=False)
    output_dir: Path = field(init=False)
    model_dir: Path = field(init=False)

    stage_name: str = "stage1"
    kepler_targets: list[str] = field(default_factory=list)
    mission: str = "Kepler"
    author: str | None = None
    cadence: str = "long"
    max_lightcurves_per_target: int = 1
    minimum_total_lightcurves: int = 3
    auto_target_count: int | None = None
    min_tess_sector: int | None = None

    # Windowing keeps each training example the same shape.
    window_length: int = 200
    stride: int = 100
    max_windows_per_lightcurve: int = 80

    # Transit injection parameters create clear labels for the MVP.
    transit_fraction: float = 0.5
    transit_depth_range: tuple[float, float] = (0.002, 0.02)
    transit_width_range: tuple[int, int] = (6, 20)
    random_seed: int = 42
    labeling_mode: str = "synthetic"

    # Baseline model settings.
    test_size: float = 0.25
    logistic_max_iter: int = 1000

    def __post_init__(self) -> None:
        self.data_dir = self.project_root / "data"
        self.cache_dir = self.data_dir / "cache"
        self.prepared_data_dir = self.data_dir / "prepared"
        self.output_dir = self.project_root / "outputs"
        self.model_dir = self.project_root / "models"

        if not self.kepler_targets:
            self.kepler_targets = list(STAGE_TARGETS[self.stage_name])

        if self.stage_name == "stage2":
            self.max_lightcurves_per_target = 3
            self.max_windows_per_lightcurve = 180
            self.minimum_total_lightcurves = 12

        if self.stage_name == "stage3":
            self.max_lightcurves_per_target = 3
            self.max_windows_per_lightcurve = 220
            self.minimum_total_lightcurves = 18
            self.labeling_mode = "real_koi"
            self.auto_target_count = 60

        if self.stage_name == "stage4":
            self.mission = "TESS"
            self.author = "QLP"
            self.max_lightcurves_per_target = 3
            self.max_windows_per_lightcurve = 180
            self.minimum_total_lightcurves = 18
            self.labeling_mode = "real_toi"
            self.auto_target_count = 60
            self.min_tess_sector = 94


def build_config(stage: str) -> ProjectConfig:
    """Create a config preset for the requested project stage."""
    if stage not in STAGE_TARGETS:
        raise ValueError(f"Unknown stage: {stage}")
    return ProjectConfig(stage_name=stage)


def ensure_directories(config: ProjectConfig) -> None:
    """Create the folders used by the pipeline."""
    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    config.prepared_data_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.model_dir.mkdir(parents=True, exist_ok=True)
