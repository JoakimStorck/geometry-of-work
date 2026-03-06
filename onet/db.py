from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd

from . import constants
from .io import read_tsv
from .registry import format_version_dot, normalize_version, resolve_db_dir

def _strip_ext(filename: str) -> str:
    for ext in constants.DEFAULT_TABLE_EXTS:
        if filename.endswith(ext):
            return filename[: -len(ext)]
    return filename

def build_table_registry(db_dir: Path) -> Dict[str, Path]:
    """Map logical table name (no extension) -> file path."""
    reg: Dict[str, Path] = {}
    for ext in constants.DEFAULT_TABLE_EXTS:
        for fp in sorted(Path(db_dir).glob(f"*{ext}")):
            base = _strip_ext(fp.name)
            if base in constants.IGNORED_TABLES:
                continue
            reg[base] = fp
    return reg

@dataclass(frozen=True)
class OnetDB:
    data_onet_dir: Path
    version_us: str
    db_dir: Path
    table_registry: Dict[str, Path]

    @property
    def version(self) -> str:
        return format_version_dot(self.version_us)

    def tables(self) -> List[str]:
        return sorted(self.table_registry.keys())

    def path(self, table: str) -> Path:
        name = str(table).strip()
        if name not in self.table_registry:
            raise KeyError(f"Unknown table {name!r}. Available: {self.tables()[:10]} ...")
        return self.table_registry[name]

    def read(self, table: str, **kwargs) -> pd.DataFrame:
        fp = self.path(table)
        return read_tsv(fp, **kwargs)

def open_db(data_onet_dir: Path, version: str) -> OnetDB:
    v_us = normalize_version(version)
    db_dir = resolve_db_dir(data_onet_dir, v_us)
    reg = build_table_registry(db_dir)
    return OnetDB(
        data_onet_dir=Path(data_onet_dir),
        version_us=v_us,
        db_dir=db_dir,
        table_registry=reg,
    )
