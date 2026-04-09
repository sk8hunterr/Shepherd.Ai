"""Helpers for downloading, caching, and selecting real Kepler/TESS catalogs."""

from __future__ import annotations

from pathlib import Path
import re
from urllib.parse import quote

import pandas as pd
import requests

from .config import ProjectConfig


KOI_CACHE_FILENAME = "kepler_koi_cumulative_stage3.csv"
TOI_CACHE_FILENAME = "tess_toi_catalog_stage4.csv"
KOI_API_BASE = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    "?query={query}&format=csv"
)
TOI_API_BASE = KOI_API_BASE
KOI_POSITIVE_DISPOSITIONS = {"CONFIRMED", "CANDIDATE"}
KOI_NEGATIVE_DISPOSITIONS = {"FALSE POSITIVE"}
TOI_POSITIVE_DISPOSITIONS = {"CP", "KP", "PC", "APC"}
TOI_NEGATIVE_DISPOSITIONS = {"FP", "FA"}
KEPLER_CONTROL_TARGETS_FILENAME = "kepler_stage3_control_targets.txt"


def _normalize_host_name(value: str) -> str:
    """Normalize a host-star name by removing trailing planet letters."""
    return re.sub(r"\s+[a-z]$", "", str(value).strip(), flags=re.IGNORECASE)


def normalize_target_identifier(value: str) -> str:
    """Return a stable lowercase key for comparing target identifiers."""
    normalized = _normalize_host_name(str(value))
    return re.sub(r"\s+", " ", normalized).strip().lower()


def _read_kepler_control_target_file(config: ProjectConfig) -> list[str]:
    """Load manually curated Stage 3 control targets from disk when available."""
    target_path = config.data_dir / KEPLER_CONTROL_TARGETS_FILENAME
    if not target_path.exists():
        return []

    targets: list[str] = []
    for line in target_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        targets.append(stripped)
    return targets


def _discover_cached_kepler_control_targets(
    config: ProjectConfig,
    koi_catalog: pd.DataFrame,
) -> list[str]:
    """Find cached Kepler KIC targets that do not appear in the KOI catalog."""
    cache_dir = config.cache_dir / "mastDownload" / "Kepler"
    if not cache_dir.exists():
        return []

    koi_ids = {
        int(value)
        for value in koi_catalog["kepid"].dropna().astype(int).tolist()
    }
    cached_ids: set[int] = set()
    for path in cache_dir.glob("kplr*_lc_*"):
        match = re.match(r"kplr0*(\d+)_lc_", path.name)
        if match:
            cached_ids.add(int(match.group(1)))

    control_ids = sorted(cached_ids - koi_ids)
    return [f"KIC {control_id}" for control_id in control_ids]


def _load_kepler_control_targets(
    config: ProjectConfig,
    koi_catalog: pd.DataFrame,
) -> list[str]:
    """Combine curated and discovered Stage 3 control targets."""
    targets = _read_kepler_control_target_file(config)
    targets.extend(_discover_cached_kepler_control_targets(config, koi_catalog))

    seen: set[str] = set()
    deduped_targets: list[str] = []
    for target in targets:
        if target not in seen:
            deduped_targets.append(target)
            seen.add(target)
    return deduped_targets


def build_target_role_lookup(
    config: ProjectConfig,
    catalog: pd.DataFrame | None,
) -> dict[str, str]:
    """Build target-role metadata for diagnostics and future training analysis."""
    if catalog is None:
        return {}
    if config.labeling_mode == "real_toi":
        return _build_tess_target_role_lookup(catalog)
    return _build_kepler_target_role_lookup(config, catalog)


def _build_kepler_target_role_lookup(
    config: ProjectConfig,
    koi_catalog: pd.DataFrame,
) -> dict[str, str]:
    """Classify Kepler targets as positives, false positives, mixed systems, or controls."""
    catalog = koi_catalog.copy()
    catalog["kepler_name"] = catalog["kepler_name"].fillna("")
    catalog["koi_disposition"] = catalog["koi_disposition"].fillna("")
    catalog["host_name"] = catalog["kepler_name"].map(_normalize_host_name)
    catalog["kepid_numeric"] = pd.to_numeric(catalog["kepid"], errors="coerce")

    role_lookup: dict[str, str] = {}
    positive_kepids = set(
        catalog[
            catalog["koi_disposition"].isin(KOI_POSITIVE_DISPOSITIONS)
            & catalog["kepid_numeric"].notna()
        ]["kepid_numeric"]
        .dropna()
        .astype(int)
        .tolist()
    )
    false_positive_kepids = set(
        catalog[
            catalog["koi_disposition"].isin(KOI_NEGATIVE_DISPOSITIONS)
            & catalog["kepid_numeric"].notna()
        ]["kepid_numeric"]
        .dropna()
        .astype(int)
        .tolist()
    )
    mixed_kepids = positive_kepids & false_positive_kepids

    positive_hosts = catalog[
        catalog["koi_disposition"].isin(KOI_POSITIVE_DISPOSITIONS)
        & (catalog["host_name"].str.strip() != "")
    ]["host_name"].tolist()
    for host_name in positive_hosts:
        role_lookup[normalize_target_identifier(host_name)] = "positive_koi_host"

    for kepid in sorted(mixed_kepids):
        role_lookup[normalize_target_identifier(f"KIC {kepid}")] = "mixed_koi_host"

    for kepid in sorted(positive_kepids - mixed_kepids):
        role_lookup.setdefault(normalize_target_identifier(f"KIC {kepid}"), "positive_koi_host")

    for kepid in sorted(false_positive_kepids - mixed_kepids):
        role_lookup.setdefault(
            normalize_target_identifier(f"KIC {kepid}"),
            "koi_false_positive_host",
        )

    for target_name in _load_kepler_control_targets(config, koi_catalog):
        role_lookup.setdefault(normalize_target_identifier(target_name), "quiet_control_target")

    return role_lookup


def _build_tess_target_role_lookup(toi_catalog: pd.DataFrame) -> dict[str, str]:
    """Classify TESS targets as TOI positives or false positives for diagnostics."""
    catalog = toi_catalog.copy()
    catalog["tfopwg_disp"] = catalog["tfopwg_disp"].fillna("")
    catalog = catalog[catalog["tid"].notna()]

    role_lookup: dict[str, str] = {}
    positive_targets = catalog[catalog["tfopwg_disp"].isin(TOI_POSITIVE_DISPOSITIONS)]["tid"]
    for tic_id in positive_targets.dropna().astype(int).tolist():
        role_lookup[normalize_target_identifier(f"TIC {tic_id}")] = "positive_toi_host"

    negative_targets = catalog[catalog["tfopwg_disp"].isin(TOI_NEGATIVE_DISPOSITIONS)]["tid"]
    for tic_id in negative_targets.dropna().astype(int).tolist():
        role_lookup.setdefault(normalize_target_identifier(f"TIC {tic_id}"), "toi_false_positive_host")

    return role_lookup


def classify_target_role(target_name: str, target_role_lookup: dict[str, str]) -> str:
    """Classify one target using catalog metadata with safe fallback labels."""
    normalized = normalize_target_identifier(target_name)
    if normalized in target_role_lookup:
        return target_role_lookup[normalized]
    if normalized.startswith("kepler-"):
        return "kepler_named_unknown_host"
    if normalized.startswith("kic "):
        return "uncataloged_kic_target"
    if normalized.startswith("tic "):
        return "uncataloged_tic_target"
    return "unknown_target"


def _build_koi_query() -> str:
    """Return a compact query with the columns needed for Stage 3 labeling."""
    return """
        select
            kepid,
            kepler_name,
            kepoi_name,
            koi_disposition,
            koi_period,
            koi_time0bk,
            koi_duration
        from cumulative
        where koi_period is not null
          and koi_time0bk is not null
          and koi_duration is not null
    """


def _build_toi_query() -> str:
    """Return the columns needed for Stage 4 TESS/TOI labeling."""
    return """
        select
            tid,
            toi,
            tfopwg_disp,
            pl_orbper,
            pl_tranmid,
            pl_trandurh
        from toi
        where pl_orbper is not null
          and pl_tranmid is not null
          and pl_trandurh is not null
    """


def get_catalog_cache_path(config: ProjectConfig) -> Path:
    """Return the stage-appropriate cache path for the real-label catalog."""
    if config.labeling_mode == "real_toi":
        return config.data_dir / TOI_CACHE_FILENAME
    return config.data_dir / KOI_CACHE_FILENAME


def download_koi_catalog(config: ProjectConfig, timeout_seconds: int = 60) -> pd.DataFrame:
    """Download the official KOI cumulative table and cache it locally."""
    cache_path = get_catalog_cache_path(config)
    query = quote(" ".join(_build_koi_query().split()))
    url = KOI_API_BASE.format(query=query)

    response = requests.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    cache_path.write_bytes(response.content)

    return pd.read_csv(cache_path)


def download_toi_catalog(config: ProjectConfig, timeout_seconds: int = 60) -> pd.DataFrame:
    """Download the official TOI table and cache it locally."""
    cache_path = get_catalog_cache_path(config)
    query = quote(" ".join(_build_toi_query().split()))
    url = TOI_API_BASE.format(query=query)

    response = requests.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    cache_path.write_bytes(response.content)

    return pd.read_csv(cache_path)


def load_koi_catalog(config: ProjectConfig, allow_download: bool = True) -> tuple[pd.DataFrame, str]:
    """Load the cached KOI table or download it from the official archive."""
    cache_path = get_catalog_cache_path(config)

    if cache_path.exists():
        return pd.read_csv(cache_path), f"Loaded cached KOI catalog from {cache_path}"

    if not allow_download:
        raise FileNotFoundError(
            f"KOI catalog cache not found at {cache_path} and downloading was disabled."
        )

    dataframe = download_koi_catalog(config)
    return dataframe, f"Downloaded KOI catalog to {cache_path}"


def load_toi_catalog(config: ProjectConfig, allow_download: bool = True) -> tuple[pd.DataFrame, str]:
    """Load the cached TOI table or download it from the official archive."""
    cache_path = get_catalog_cache_path(config)

    if cache_path.exists():
        return pd.read_csv(cache_path), f"Loaded cached TOI catalog from {cache_path}"

    if not allow_download:
        raise FileNotFoundError(
            f"TOI catalog cache not found at {cache_path} and downloading was disabled."
        )

    dataframe = download_toi_catalog(config)
    return dataframe, f"Downloaded TOI catalog to {cache_path}"


def load_real_catalog(config: ProjectConfig, allow_download: bool = True) -> tuple[pd.DataFrame, str]:
    """Load the appropriate real-label catalog for the current stage."""
    if config.labeling_mode == "real_toi":
        return load_toi_catalog(config, allow_download=allow_download)
    return load_koi_catalog(config, allow_download=allow_download)


def _select_kepler_targets(
    config: ProjectConfig,
    koi_catalog: pd.DataFrame,
    target_count: int,
) -> list[str]:
    catalog = koi_catalog.copy()
    catalog["kepler_name"] = catalog["kepler_name"].fillna("")
    catalog["koi_disposition"] = catalog["koi_disposition"].fillna("")
    catalog["host_name"] = catalog["kepler_name"].map(_normalize_host_name)
    catalog["kepid_numeric"] = pd.to_numeric(catalog["kepid"], errors="coerce")
    positive_kepids = set(
        catalog[
            catalog["koi_disposition"].isin(KOI_POSITIVE_DISPOSITIONS)
            & catalog["kepid_numeric"].notna()
        ]["kepid_numeric"]
        .dropna()
        .astype(int)
        .tolist()
    )
    false_positive_kepids = set(
        catalog[
            catalog["koi_disposition"].isin(KOI_NEGATIVE_DISPOSITIONS)
            & catalog["kepid_numeric"].notna()
        ]["kepid_numeric"]
        .dropna()
        .astype(int)
        .tolist()
    )
    mixed_kepids = positive_kepids & false_positive_kepids

    positive_hosts = (
        catalog[
            catalog["koi_disposition"].isin(KOI_POSITIVE_DISPOSITIONS)
            & (catalog["host_name"].str.strip() != "")
        ]
        .groupby("host_name")
        .agg(
            candidate_count=("kepoi_name", "count"),
            min_period=("koi_period", "min"),
            median_period=("koi_period", "median"),
            max_period=("koi_period", "max"),
        )
        .reset_index()
        .sort_values(
            ["candidate_count", "median_period", "max_period", "host_name"],
            ascending=[False, False, False, True],
        )
    )

    negative_hosts = (
        catalog[
            catalog["koi_disposition"].isin(KOI_NEGATIVE_DISPOSITIONS)
            & catalog["kepid_numeric"].notna()
            & ~catalog["kepid_numeric"].astype(int).isin(mixed_kepids)
        ]
        .assign(
            target_name=lambda frame: frame["kepid_numeric"].map(
                lambda value: f"KIC {int(value)}"
            )
        )
        .groupby("target_name")
        .agg(
            candidate_count=("kepoi_name", "count"),
            min_period=("koi_period", "min"),
            median_period=("koi_period", "median"),
            max_period=("koi_period", "max"),
        )
        .reset_index()
        .sort_values(
            ["candidate_count", "median_period", "max_period", "target_name"],
            ascending=[False, False, False, True],
        )
    )

    control_targets = _load_kepler_control_targets(config, koi_catalog)

    positive_count = max(1, int(round(target_count * 0.40)))
    false_positive_count = max(1, int(round(target_count * 0.35)))
    control_count = max(1, target_count - positive_count - false_positive_count)

    selected: list[str] = []
    selected.extend(positive_hosts["host_name"].head(positive_count).tolist())
    selected.extend(negative_hosts["target_name"].head(false_positive_count).tolist())
    selected.extend(control_targets[:control_count])

    seen: set[str] = set()
    deduped_selected: list[str] = []
    for target in selected:
        if target not in seen:
            deduped_selected.append(target)
            seen.add(target)

    if len(deduped_selected) < target_count:
        fallback_pools = [
            negative_hosts["target_name"].tolist(),
            positive_hosts["host_name"].tolist(),
            control_targets,
        ]
        for pool in fallback_pools:
            for target in pool:
                if target not in seen:
                    deduped_selected.append(target)
                    seen.add(target)
                if len(deduped_selected) >= target_count:
                    break
            if len(deduped_selected) >= target_count:
                break

    return deduped_selected[:target_count]


def _select_tess_targets(toi_catalog: pd.DataFrame, target_count: int) -> list[str]:
    catalog = toi_catalog.copy()
    catalog["tfopwg_disp"] = catalog["tfopwg_disp"].fillna("")
    catalog = catalog[catalog["tid"].notna()]

    positive_targets = (
        catalog[catalog["tfopwg_disp"].isin(TOI_POSITIVE_DISPOSITIONS)]
        .groupby("tid")
        .agg(
            candidate_count=("toi", "count"),
            min_period=("pl_orbper", "min"),
            median_period=("pl_orbper", "median"),
        )
        .reset_index()
        .sort_values(["candidate_count", "median_period", "tid"], ascending=[False, False, True])
    )

    negative_targets = (
        catalog[catalog["tfopwg_disp"].isin(TOI_NEGATIVE_DISPOSITIONS)]
        .groupby("tid")
        .agg(
            candidate_count=("toi", "count"),
            min_period=("pl_orbper", "min"),
            median_period=("pl_orbper", "median"),
        )
        .reset_index()
        .sort_values(["candidate_count", "median_period", "tid"], ascending=[False, False, True])
    )

    positive_count = max(1, int(round(target_count * 0.6)))
    negative_count = max(1, target_count - positive_count)

    selected = positive_targets["tid"].head(positive_count).tolist()
    selected.extend(negative_targets["tid"].head(negative_count).tolist())
    return [f"TIC {int(tic_id)}" for tic_id in selected[:target_count]]


def auto_select_targets(config: ProjectConfig, catalog: pd.DataFrame) -> list[str]:
    """Choose a larger catalog-backed target list for real-label stages."""
    if config.auto_target_count is None:
        return list(config.kepler_targets)

    if config.labeling_mode == "real_toi":
        return _select_tess_targets(catalog, config.auto_target_count)

    return _select_kepler_targets(config, catalog, config.auto_target_count)
