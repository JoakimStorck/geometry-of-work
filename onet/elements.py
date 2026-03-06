from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import pandas as pd

from .db import OnetDB

def pick_scale(df_raw: pd.DataFrame, preferred: Sequence[str]) -> str:
    """Pick first available Scale ID from preferred list; fallback to first available."""
    if "Scale ID" not in df_raw.columns:
        raise ValueError("df_raw missing 'Scale ID' column")
    avail = [str(x).strip() for x in df_raw["Scale ID"].dropna().unique().tolist()]
    for s in preferred:
        if s in avail:
            return s
    if not avail:
        raise ValueError("No Scale ID values found")
    return avail[0]

def pivot_elements(
    df_raw: pd.DataFrame,
    *,
    scale_id: str,
    code_col: str = "O*NET-SOC Code",
    element_col: str = "Element Name",
    value_col: str = "Data Value",
    aggfunc: str = "mean",
) -> pd.DataFrame:
    """Pivot an element table to wide form: onet_code × element -> value."""
    for c in (code_col, element_col, value_col, "Scale ID"):
        if c not in df_raw.columns:
            raise ValueError(f"df_raw missing column: {c!r}")
    wide = (
        df_raw.loc[df_raw["Scale ID"] == scale_id]
        .pivot_table(index=code_col, columns=element_col, values=value_col, aggfunc=aggfunc)
        .reset_index()
        .rename(columns={code_col: "onet_code"})
    )
    wide["onet_code"] = wide["onet_code"].astype("string").str.strip()
    return wide

def wide(
    db: OnetDB,
    table: str,
    *,
    scale_preference: Sequence[str],
    aggfunc: str = "mean",
) -> tuple[pd.DataFrame, str]:
    df_raw = db.read(table)
    scale = pick_scale(df_raw, scale_preference)
    return pivot_elements(df_raw, scale_id=scale, aggfunc=aggfunc), scale

def long(
    db: OnetDB,
    table: str,
    *,
    scale_preference: Sequence[str],
    element_name: str = "element",
    value_name: str = "value",
    aggfunc: str = "mean",
) -> tuple[pd.DataFrame, str]:
    w, scale = wide(db, table, scale_preference=scale_preference, aggfunc=aggfunc)
    value_cols = [c for c in w.columns if c != "onet_code"]
    out = w.melt(id_vars=["onet_code"], value_vars=value_cols, var_name=element_name, value_name=value_name)
    return out, scale
