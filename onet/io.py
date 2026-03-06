from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

def read_tsv(fp: Path, **kwargs) -> pd.DataFrame:
    """Read an O*NET .txt table (TSV) with encoding fallback."""
    fp = Path(fp)
    try:
        return pd.read_csv(fp, sep="\t", encoding="utf-8", low_memory=False, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(fp, sep="\t", encoding="latin-1", low_memory=False, **kwargs)

def read_csv(fp: Path, **kwargs) -> pd.DataFrame:
    fp = Path(fp)
    return pd.read_csv(fp, **kwargs)
