"""
overlays.py
-----------
O*NET overlay computation for Skills/Abilities (and other element tables).

Computes, for each O*NET descriptor (e.g. each skill or ability):
  - the dominant angular direction theta_max in occupation space
  - the resultant length R (concentration measure)
  - per-occupation values aligned with the polar geometry

Two CSV exports are written per call (paper-relevant only):
  - {label}s_overlay__{table}__{scale}__angle_rank.csv
  - {label}s_overlay__{table}__{scale}__long.csv

Other diagnostics from the previous notebook (KDE profile, norm_scales,
intensity scales, intensity_area) have been removed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

import infra
from infra import log
import onet


# ─────────────────────────────────────────────────────────────
# Defaults (can be overridden per call)
# ─────────────────────────────────────────────────────────────

SCALE_PREFERENCE: tuple[str, ...] = ("LV", "IM")
K_AREA: int = 8
ROBUST_LO: float = 5.0
ROBUST_HI: float = 95.0


# ─────────────────────────────────────────────────────────────
# Math helpers
# ─────────────────────────────────────────────────────────────

def robust_minmax_params(x, lo: float = ROBUST_LO, hi: float = ROBUST_HI) -> tuple[float, float]:
    x = np.asarray(x, float)
    a, b = np.nanpercentile(x, [lo, hi])
    return float(a), float(b)


def apply_minmax_with_params(x, a: float, b: float) -> np.ndarray:
    den = max(b - a, 1e-12)
    y = (np.asarray(x, float) - a) / den
    return np.clip(y, 0.0, 1.0)


def robust_minmax(x, lo: float = ROBUST_LO, hi: float = ROBUST_HI) -> np.ndarray:
    a, b = robust_minmax_params(x, lo=lo, hi=hi)
    return apply_minmax_with_params(x, a, b)


def resultant_direction(theta: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Weighted circular mean direction and resultant length.

    Returns
    -------
    theta : float in [0, 2*pi)
    R     : float in [0, 1]
    """
    W = np.sum(w)
    if not np.isfinite(W) or W <= 0:
        return 0.0, 0.0
    X = np.sum(w * np.cos(theta))
    Y = np.sum(w * np.sin(theta))
    th = float(np.mod(np.arctan2(Y, X), 2 * np.pi))
    R = float(np.hypot(X, Y) / max(W, 1e-12))
    return th, R


# ─────────────────────────────────────────────────────────────
# Occupation geometry (polar + local kNN density)
# ─────────────────────────────────────────────────────────────

def prepare_occ_geometry(df_occ: pd.DataFrame, k_area: int = K_AREA) -> pd.DataFrame:
    """Return df with onet_code, xi, chi, area_kNN, density_kNN."""
    occ = df_occ[["onet_code", "xi", "chi"]].copy()
    xi = np.mod(occ["xi"].to_numpy(float), 2 * np.pi)
    chi = occ["chi"].to_numpy(float)
    x = chi * np.cos(xi)
    y = chi * np.sin(xi)
    coords = np.column_stack([x, y])
    nbrs = NearestNeighbors(n_neighbors=k_area + 1).fit(coords)
    dists, _ = nbrs.kneighbors(coords)
    r_k = dists[:, -1]
    area = (np.pi * np.maximum(r_k, 1e-12) ** 2) / k_area
    density = 1.0 / np.maximum(area, 1e-12)
    occ["xi"] = xi
    occ["area_kNN"] = area
    occ["density_kNN"] = density
    return occ


# ─────────────────────────────────────────────────────────────
# O*NET wide table loading
# ─────────────────────────────────────────────────────────────

def load_onet_wide(
    db,
    table_basename: str,
    *,
    scale_preference: Sequence[str] = SCALE_PREFERENCE,
) -> tuple[pd.DataFrame, str]:
    """Load a wide pivot of an O*NET element table, with onet_code as join key."""
    base = table_basename.lower()
    if base == "skills":
        wide_df, scale = onet.skills.wide(db, scale_preference=scale_preference)
    elif base == "abilities":
        wide_df, scale = onet.abilities.wide(db, scale_preference=scale_preference)
    else:
        wide_df, scale = onet.elements.wide(db, table_basename, scale_preference=scale_preference)

    if "onet_code" not in wide_df.columns and "O*NET-SOC Code" in wide_df.columns:
        wide_df = wide_df.rename(columns={"O*NET-SOC Code": "onet_code"})
    return wide_df, scale


# ─────────────────────────────────────────────────────────────
# Overlay result
# ─────────────────────────────────────────────────────────────

@dataclass
class OverlayResult:
    table: str            # "Skills" | "Abilities" | other
    scale: str            # e.g. "LV"
    label: str            # "skill" | "ability"
    chosen: list[str]     # descriptor names, sorted by theta_max
    occ: pd.DataFrame     # occupation rows (onet_code, xi, chi, area_kNN, density_kNN, value cols)
    angles_df: pd.DataFrame
    long_df: pd.DataFrame
    prefix: str           # filename prefix for exports
    angle_rank_fp: "object"  # Path to angle_rank csv
    long_fp: "object"        # Path to long csv


# ─────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────

def compute_overlay(
    db,
    df_occ: pd.DataFrame,
    table_basename: str,
    label: str,
    *,
    occ_geom: pd.DataFrame | None = None,
    names_include: Sequence[str] | None = None,
    scale_preference: Sequence[str] = SCALE_PREFERENCE,
) -> OverlayResult:
    """Compute angular direction (theta_max, R) per descriptor and write paper-relevant CSVs.

    Writes to infra.RP.exports:
      - {prefix}__angle_rank.csv  (one row per descriptor: theta_max, R, n_used)
      - {prefix}__long.csv        (one row per (occupation, descriptor): xi, chi, value, value_norm01_global)

    where prefix = "{label}s_overlay__{table}__{scale}".
    """
    if occ_geom is None:
        occ_geom = prepare_occ_geometry(df_occ)

    wide_df, scale = load_onet_wide(db, table_basename, scale_preference=scale_preference)
    value_cols = [c for c in wide_df.columns if c != "onet_code"]

    occ = occ_geom.merge(wide_df, on="onet_code", how="left")
    occ[value_cols] = occ[value_cols].apply(pd.to_numeric, errors="coerce")

    non_empty = [c for c in value_cols if occ[c].notna().any()]
    if names_include:
        chosen = [c for c in names_include if c in occ.columns]
    else:
        chosen = sorted(non_empty)
    if not chosen:
        raise RuntimeError(f"No {label}s selected from {table_basename} ({scale}).")

    xi_all = occ["xi"].to_numpy(float)

    # 1) Angular direction per descriptor
    peak_info: list[tuple[float, str, float, int]] = []
    vals_by: dict[str, np.ndarray] = {}
    for name in chosen:
        vals = occ[name].to_numpy(float)
        vals_by[name] = vals
        med = np.nanmedian(vals)
        w = np.clip(vals - med, 0.0, None)
        if not np.isfinite(w).any() or np.nanmax(w) <= 0:
            w = robust_minmax(vals)
        mask = np.isfinite(xi_all) & np.isfinite(w)
        th = xi_all[mask]
        ww = w[mask]
        theta_max, R = resultant_direction(th, ww)
        peak_info.append((theta_max, name, R, int(mask.sum())))

    peak_info.sort(key=lambda t: t[0])
    chosen_sorted = [n for _, n, *_ in peak_info]

    angles_df = pd.DataFrame(
        peak_info,
        columns=["theta_max_rad", label, "resultant_R", "n_used"],
    )
    angles_df["theta_max_deg"] = np.degrees(angles_df["theta_max_rad"]) % 360.0
    angles_df = angles_df[[label, "theta_max_rad", "theta_max_deg", "resultant_R", "n_used"]]

    # 2) Long format: per (occupation, descriptor) with value_norm01_global
    all_vals = np.concatenate([vals_by[n].astype(float) for n in chosen_sorted])
    glob_lo, glob_hi = robust_minmax_params(all_vals)

    long_rows: list[tuple] = []
    for n in chosen_sorted:
        vals = vals_by[n].astype(float)
        v_glob = apply_minmax_with_params(vals, glob_lo, glob_hi)
        for (oc, xi, chi, v, vg) in zip(
            occ["onet_code"],
            occ["xi"],
            occ["chi"],
            vals,
            v_glob,
        ):
            long_rows.append((oc, xi, chi, n, v, vg))
    long_df = pd.DataFrame(
        long_rows,
        columns=["onet_code", "xi", "chi", label, "value", "value_norm01_global"],
    )

    # 3) Exports
    prefix = f"{label}s_overlay__{table_basename}__{scale}"
    angle_fp = infra.RP.export_fp(f"{prefix}__angle_rank.csv")
    long_fp = infra.RP.export_fp(f"{prefix}__long.csv")
    angles_df.to_csv(angle_fp, index=False)
    long_df.to_csv(long_fp, index=False)

    log(f"Overlay: {table_basename} ({scale}) → {len(chosen_sorted)} descriptors")

    return OverlayResult(
        table=table_basename,
        scale=scale,
        label=label,
        chosen=chosen_sorted,
        occ=occ,
        angles_df=angles_df,
        long_df=long_df,
        prefix=prefix,
        angle_rank_fp=angle_fp,
        long_fp=long_fp,
    )
