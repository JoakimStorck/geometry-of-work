from __future__ import annotations

import numpy as np
import pandas as pd

from .db import OnetDB

# O*NET table name and element filter
_TABLE   = "Education, Training, and Experience"
_ELEMENT = "Required Level of Education"


def data(db: OnetDB) -> pd.DataFrame:
    """Raw Education, Training, and Experience table."""
    return db.read(_TABLE)


def rle_by_occupation(db: OnetDB) -> pd.DataFrame:
    """
    Frequency-weighted mean Required Level of Education per occupation.

    Filters the Education, Training, and Experience table to the
    'Required Level of Education' element and computes the weighted mean
    over the ordinal scale (1–12).

    Returns
    -------
    DataFrame with columns:
        onet_code       : str    O*NET-SOC code
        rle_mean        : float  frequency-weighted mean (scale 1–12)
        rle_weight_sum  : float  sum of frequency weights
    """
    df = data(db).copy()

    # Normalise column names to lowercase for robust matching
    df.columns = df.columns.str.strip()
    col_code = next((c for c in df.columns if c.lower() in
                     ["o*net-soc code", "onet-soc code", "onet_soc_code"]), None)
    col_elem = next((c for c in df.columns if c.lower() in
                     ["element name", "element_name"]), None)
    col_lvl  = next((c for c in df.columns if c.lower() in
                     ["category", "scale value", "scale_value"]), None)
    col_w    = next((c for c in df.columns if c.lower() in
                     ["data value", "data_value"]), None)

    missing = [name for name, val in
               [("code", col_code), ("element", col_elem),
                ("level", col_lvl), ("weight", col_w)]
               if val is None]
    if missing:
        raise ValueError(
            f"rle_by_occupation: could not detect columns {missing}. "
            f"Available: {df.columns.tolist()}")

    df[col_elem] = df[col_elem].astype(str).str.strip()
    df = df[df[col_elem] == _ELEMENT].copy()

    df[col_lvl] = pd.to_numeric(df[col_lvl], errors="coerce")
    df[col_w]   = pd.to_numeric(df[col_w],   errors="coerce")
    df = df.dropna(subset=[col_code, col_lvl, col_w])
    df = df[(df[col_lvl] >= 1) & (df[col_lvl] <= 12) & (df[col_w] > 0)]

    result = (
        df.groupby(col_code, as_index=False)
          .apply(lambda g: pd.Series({
              "rle_mean":       float(np.average(g[col_lvl], weights=g[col_w])),
              "rle_weight_sum": float(g[col_w].sum()),
          }), include_groups=False)
          .reset_index(drop=True)
          .rename(columns={col_code: "onet_code"})
    )
    return result


def rle_by_job_family(db: OnetDB, occ_meta: pd.DataFrame,
                      weight_col: str | None = "TOT_EMP") -> pd.DataFrame:
    """
    Employment-weighted mean RLE per job family.

    Parameters
    ----------
    db        : OnetDB
    occ_meta  : DataFrame with columns onet_code and 'Job Family'.
                Typically from onet.load_occ_meta(db) merged with BLS employment.
    weight_col: Column in occ_meta to use as employment weight.
                Pass None for equal weighting.

    Returns
    -------
    DataFrame with columns:
        Job Family       : str
        rle_mean_family  : float  employment-weighted mean RLE
        rle_std_family   : float  employment-weighted std
        n_occupations    : int
    """
    rle_occ = rle_by_occupation(db)

    merged = occ_meta.merge(rle_occ, on="onet_code", how="left")
    merged = merged.dropna(subset=["Job Family", "rle_mean"])

    if weight_col and weight_col in merged.columns:
        merged["_w"] = pd.to_numeric(merged[weight_col], errors="coerce").fillna(0.0)
        merged.loc[merged["_w"] <= 0, "_w"] = 1.0
    else:
        merged["_w"] = 1.0

    def _agg(g):
        mu  = float(np.average(g["rle_mean"], weights=g["_w"]))
        std = float(np.sqrt(np.average((g["rle_mean"] - mu) ** 2, weights=g["_w"])))
        return pd.Series({
            "rle_mean_family": mu,
            "rle_std_family":  std,
            "n_occupations":   int(len(g)),
        })

    result = (
        merged.groupby("Job Family", sort=True)
              .apply(_agg, include_groups=False)
              .reset_index()
              .sort_values("rle_mean_family", ascending=False)
              .reset_index(drop=True)
    )
    return result
