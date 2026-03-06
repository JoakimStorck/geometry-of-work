from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from .db import OnetDB, open_db
from .registry import (
    list_versions_dot,
    latest_version_us,
    normalize_version,
    format_version_dot,
)

def default_data_onet_dir() -> Path:
    """Resolve data/onet using infra.PROJECT_ROOT when available (strict), else cwd/data/onet."""
    try:
        import infra  # type: ignore
        root = Path(infra.PROJECT_ROOT)
    except Exception:
        root = Path.cwd()
    return root / "data" / "onet"

def list_versions(data_onet_dir: Optional[Path] = None) -> list[str]:
    data_dir = Path(data_onet_dir) if data_onet_dir else default_data_onet_dir()
    return list_versions_dot(data_dir)

def get_db(
    *,
    version: Optional[str] = None,
    data_onet_dir: Optional[Path] = None,
) -> OnetDB:
    data_dir = Path(data_onet_dir) if data_onet_dir else default_data_onet_dir()
    if version is None:
        v_us = latest_version_us(data_dir)
        version = format_version_dot(v_us)
    return open_db(data_dir, version)

# Convenience re-exports (domain transforms)
from .transforms import (
    build_df_tasks,
    load_occ_meta,
    load_rle,
    load_job_families,
    load_crosswalk_2010_2019,
)
