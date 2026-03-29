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


def _normalize_host_name(value: str) -> str:
    """Normalize a host-star name by removing trailing planet letters."""
    return re.sub(r"\s+[a-z]$", "", str(value).strip(), flags=re.IGNORECASE)


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


def _select_kepler_targets(koi_catalog: pd.DataFrame, target_count: int) -> list[str]:
    catalog = koi_catalog.copy()
    catalog["kepler_name"] = catalog["kepler_name"].fillna("")
    catalog["koi_disposition"] = catalog["koi_disposition"].fillna("")
    catalog = catalog[catalog["kepler_name"].str.strip() != ""]
    catalog["host_name"] = catalog["kepler_name"].map(_normalize_host_name)

    positive_hosts = (
        catalog[catalog["koi_disposition"].isin(KOI_POSITIVE_DISPOSITIONS)]
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
        catalog[catalog["koi_disposition"].isin(KOI_NEGATIVE_DISPOSITIONS)]
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

    positive_count = max(1, int(round(target_count * 0.6)))
    negative_count = max(1, target_count - positive_count)

    selected = positive_hosts["host_name"].head(positive_count).tolist()
    selected.extend(negative_hosts["host_name"].head(negative_count).tolist())
    return selected[:target_count]


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

    return _select_kepler_targets(catalog, config.auto_target_count)
