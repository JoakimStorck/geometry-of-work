# onet/tasks.py
from __future__ import annotations

import pandas as pd

from .db import OnetDB

_RATINGS_RENAME = {
    "O*NET-SOC Code": "onet_code",
    "Task ID": "task_id",
    "Scale ID": "scale_id",
    "Category": "category",
    "Data Value": "data_value",
    "N": "n",
    "Date": "date",
}

def statements(db: OnetDB) -> pd.DataFrame:
    return db.read("Task Statements")

def ratings(
    db: OnetDB,
    *,
    scale_id: str | None = None,
    normalize: bool = False,
) -> pd.DataFrame:
    """
    Task Ratings.

    normalize=False (default):
      - returnerar rå tabell (som idag)
      - scale_id filtrerar på "Scale ID" om kolumnen finns

    normalize=True:
      - returnerar normaliserat schema:
          onet_code, task_id, scale_id, category, data_value, n, date
      - typer: task_id -> Int64, data_value/n -> numeric
      - scale_id filtrerar på 'scale_id'
    """
    df = db.read("Task Ratings")

    if not normalize:
        if scale_id is None:
            return df
        if "Scale ID" not in df.columns:
            return df
        return df.loc[df["Scale ID"] == scale_id].copy()

    # --- normalize ---
    cols = [c for c in _RATINGS_RENAME.keys() if c in df.columns]
    out = df[cols].rename(columns=_RATINGS_RENAME).copy()

    # obligatoriska fält om de saknas i en version
    for c in ["category", "n", "date"]:
        if c not in out.columns:
            out[c] = pd.NA

    out["onet_code"] = out["onet_code"].astype(str).str.strip()
    out["scale_id"]  = out["scale_id"].astype(str).str.strip()
    out["category"]  = out["category"].astype(str).str.strip()
    out["date"]      = out["date"].astype(str).str.strip()

    out["task_id"]   = pd.to_numeric(out["task_id"], errors="coerce").astype("Int64")
    out["n"]         = pd.to_numeric(out["n"], errors="coerce")
    out["data_value"]= pd.to_numeric(out["data_value"], errors="coerce")

    if scale_id is not None:
        out = out.loc[out["scale_id"] == scale_id].copy()

    return out


def categories(db: OnetDB) -> pd.DataFrame:
    # Not always present in older versions; will raise if missing
    return db.read("Task Categories")

def build_task_frame(
    db: OnetDB,
    *,
    include_supplemental: bool = True,
    require_rle: bool = True,
    rt_scale_id: str = "RT",
) -> pd.DataFrame:
    """Convenience wrapper for the standard task dataframe used in embeddings."""
    return build_df_tasks(db, include_supplemental=include_supplemental, require_rle=require_rle, rt_scale_id=rt_scale_id)
