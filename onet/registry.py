from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Tuple

_VERSION_DOT_RE = re.compile(r"^\s*(\d+)(?:[\._](\d+))?\s*$")

def normalize_version(version: str) -> str:
    """Normalize incoming version to underscore form used on disk, e.g. '30.0' -> '30_0'."""
    if version is None:
        raise ValueError("version is None")
    s = str(version).strip()
    m = _VERSION_DOT_RE.match(s)
    if not m:
        raise ValueError(f"Invalid O*NET version: {version!r}")
    major = int(m.group(1))
    minor = int(m.group(2) or 0)
    return f"{major}_{minor}"

def format_version_dot(version_us: str) -> str:
    """Format underscore version to dot form for display, e.g. '30_0' -> '30.0'."""
    v = normalize_version(version_us)
    major, minor = v.split("_")
    return f"{int(major)}.{int(minor)}"

def resolve_db_dir(data_onet_dir: Path, version_us: str) -> Path:
    v = normalize_version(version_us)
    db_dir = Path(data_onet_dir) / f"db_{v}"
    if not db_dir.exists():
        raise FileNotFoundError(f"Missing O*NET db folder: {db_dir}")
    return db_dir

def _parse_version_us_from_db_dirname(name: str) -> str | None:
    # expects 'db_30_0'
    if not name.startswith("db_"):
        return None
    tail = name[len("db_"):]
    try:
        return normalize_version(tail)
    except Exception:
        return None

def list_versions_us(data_onet_dir: Path) -> List[str]:
    out: List[str] = []
    for p in sorted(Path(data_onet_dir).glob("db_*")):
        if not p.is_dir():
            continue
        v = _parse_version_us_from_db_dirname(p.name)
        if v:
            out.append(v)
    # unique + sort numerically
    def key(v: str) -> Tuple[int,int]:
        a,b = v.split("_")
        return (int(a), int(b))
    out = sorted(set(out), key=key)
    return out

def list_versions_dot(data_onet_dir: Path) -> List[str]:
    return [format_version_dot(v) for v in list_versions_us(data_onet_dir)]

def latest_version_us(data_onet_dir: Path) -> str:
    vs = list_versions_us(data_onet_dir)
    if not vs:
        raise FileNotFoundError(f"No O*NET db_* directories under: {data_onet_dir}")
    return vs[-1]
