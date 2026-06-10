"""
pair_level_chi_by_sector.py
---------------------------
Sector-wise pair-level test of the radial component of Proposition P3.

The pooled pair-level correlation between radial distance and absolute
outcome differences is near zero (Table pair-wage-edu). The framework
implies this: the return to chi changes sign with direction, so radial
separation should raise outcome differences within directions where
|beta_chi| is large and contribute nothing within directions where
beta_chi is near zero. Pooling across directions dilutes the local
effects below the residual level.

This module tests the implied conditional prediction. Pairs are
restricted to occupations within the same direction and the Spearman
correlation rho(dchi, |dlnw|) and rho(dchi, |dE|) is computed per
angular sector. Two pair samples are used:

  same_sector  Both occupations lie in the same 45-degree sector
               (the sample of the sectoral wage regression).
  dxi30        Pairs with circular angular distance below 30 degrees,
               assigned to the sector of their circular mean angle.

Prediction: within-sector rho is positive where the sectoral wage
regression gives large |beta_chi| (N, NE, S, SE, SW) and near zero
where beta_chi is near zero (E, NW, W). The per-sector pair rho should
track |beta_chi| across sectors.

Sectors follow the convention of the sectoral wage regression: eight
45-degree sectors centered on E = 0, NE = 45, ..., SE = 315 degrees,
sector k spanning [center_k - 22.5, center_k + 22.5) degrees.

For education the module additionally estimates the within-sector OLS
slope of E_bar on chi (HC3 standard errors), the education analog of
the sectoral wage regression, which the manuscript does not report.
If the slope changes sign across sectors in the same pattern as for
wages, one mechanism accounts for both pooled near-zero results.

Significance is assessed by permutation: chi is shuffled across
occupations within each sector (xi held fixed), preserving the pair
set, the sector assignment, and the sector-wise chi marginals, and all
statistics are recomputed per draw. Pairs share occupations, so
asymptotic pair-level p-values would be anticonservative. P-values are
two-sided with the Phipson-Smyth correction, (hits + 1) / (draws + 1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
import statsmodels.api as sm

import infra
from infra import log

from pair_level_wage_edu import load_wages, load_education


# ─────────────────────────────────────────────────────────────
# Sector convention (identical to the sectoral wage regression)
# ─────────────────────────────────────────────────────────────

SECTOR_WIDTH = np.pi / 4
SECTOR_LABELS = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
SECTOR_CENTERS_DEG = np.array([0, 45, 90, 135, 180, 225, 270, 315])


def assign_sector(xi: np.ndarray) -> np.ndarray:
    """Sector k spans [center_k - pi/8, center_k + pi/8) mod 2pi."""
    shifted = np.mod(np.asarray(xi, float) + SECTOR_WIDTH / 2, 2 * np.pi)
    return (shifted // SECTOR_WIDTH).astype(int)


def circular_mean_pair(xi_i: np.ndarray, xi_j: np.ndarray) -> np.ndarray:
    """Circular mean angle of each pair."""
    c = np.cos(xi_i) + np.cos(xi_j)
    s = np.sin(xi_i) + np.sin(xi_j)
    return np.mod(np.arctan2(s, c), 2 * np.pi)


# ─────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────

def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def _rank_residualize(v: np.ndarray, ctrl: np.ndarray) -> np.ndarray:
    return v - np.polyval(np.polyfit(ctrl, v, 1), ctrl)


def _spearman_partial(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Spearman correlation of a and b after rank residualization on c,
    matching the partial correlations of the pooled pair-level table."""
    if len(a) < 3:
        return float("nan")
    ra, rb, rc = rankdata(a), rankdata(b), rankdata(c)
    return float(np.corrcoef(_rank_residualize(ra, rc),
                             _rank_residualize(rb, rc))[0, 1])


def _within_sector_ols(y: np.ndarray, chi: np.ndarray):
    """OLS of y on chi within a sector, HC3 errors. Returns (beta, se, p)."""
    if len(y) < 15:
        return float("nan"), float("nan"), float("nan")
    X = sm.add_constant(chi)
    fit = sm.OLS(y, X).fit(cov_type="HC3")
    return float(fit.params[1]), float(fit.bse[1]), float(fit.pvalues[1])


# ─────────────────────────────────────────────────────────────
# Pair construction
# ─────────────────────────────────────────────────────────────

def _build_pairs(xi: np.ndarray, chi: np.ndarray,
                 lnw: np.ndarray, E_bar: np.ndarray) -> pd.DataFrame:
    n = len(xi)
    iu = np.triu_indices(n, k=1)
    dd = np.mod(xi[iu[0]] - xi[iu[1]], 2 * np.pi)
    dxi = np.minimum(dd, 2 * np.pi - dd)
    occ_sector = assign_sector(xi)
    return pd.DataFrame({
        "i": iu[0], "j": iu[1],
        "dxi_deg": np.degrees(dxi),
        "dchi": np.abs(chi[iu[0]] - chi[iu[1]]),
        "abs_dlnw": np.abs(lnw[iu[0]] - lnw[iu[1]]),
        "abs_dE": np.abs(E_bar[iu[0]] - E_bar[iu[1]]),
        "sector_i": occ_sector[iu[0]],
        "sector_j": occ_sector[iu[1]],
        "sector_mean": assign_sector(circular_mean_pair(xi[iu[0]], xi[iu[1]])),
    })


def _sample_masks(pairs: pd.DataFrame, dxi_max_deg: float
                  ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """For each sample: (inclusion mask, per-pair sector index)."""
    same = (pairs["sector_i"] == pairs["sector_j"]).to_numpy()
    near = (pairs["dxi_deg"] < dxi_max_deg).to_numpy()
    return {
        "same_sector": (same, pairs["sector_i"].to_numpy()),
        f"dxi{int(dxi_max_deg)}": (near, pairs["sector_mean"].to_numpy()),
    }


def _per_sector_rho(pairs: pd.DataFrame, mask: np.ndarray,
                    sector: np.ndarray) -> pd.DataFrame:
    rows = []
    for s in range(8):
        sub = pairs.loc[mask & (sector == s)]
        dchi = sub["dchi"].to_numpy()
        dxi = sub["dxi_deg"].to_numpy()
        w = sub["abs_dlnw"].to_numpy()
        E = sub["abs_dE"].to_numpy()
        rows.append({
            "sector": SECTOR_LABELS[s],
            "center_deg": int(SECTOR_CENTERS_DEG[s]),
            "n_pairs": len(sub),
            "rho_dchi_w": _spearman(dchi, w),
            "rho_dchi_E": _spearman(dchi, E),
            "rho_dchi_w_given_dxi": _spearman_partial(dchi, w, dxi),
            "rho_dchi_E_given_dxi": _spearman_partial(dchi, E, dxi),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# Main computation
# ─────────────────────────────────────────────────────────────

@dataclass
class SectorPairResult:
    n_occ: int
    occ_df: pd.DataFrame
    sector_df: pd.DataFrame      # per sample x sector: rho, p_perm, beta_chi
    summary_df: pd.DataFrame     # cross-sector tracking statistics


def compute_chi_by_sector(
    occ_df: pd.DataFrame,
    wages: pd.DataFrame,
    education: pd.Series,
    *,
    dxi_max_deg: float = 30.0,
    n_permutations: int = 1000,
    seed: int = 0,
    write_csv: bool = True,
) -> SectorPairResult:
    """Sector-wise pair-level correlation between radial distance and
    absolute wage/education differences.

    occ_df needs columns onet_code, xi, chi. Wages are merged on the
    7-char SOC prefix of onet_code; education on onet_code, matching
    compute_pair_level in pair_level_wage_edu.py.
    """
    m = occ_df.copy()
    m["soc"] = m["onet_code"].str[:7]
    m = m.merge(wages, left_on="soc", right_on="OCC_CODE")
    m = m.merge(education, left_on="onet_code", right_index=True)
    n = len(m)

    xi = np.mod(m["xi"].to_numpy(float), 2 * np.pi)
    chi = m["chi"].to_numpy(float)
    lnw = np.log(m["H_MEDIAN"].to_numpy(float))
    E_bar = m["E_bar"].to_numpy(float)
    occ_sector = assign_sector(xi)
    m["sector"] = [SECTOR_LABELS[s] for s in occ_sector]

    pairs = _build_pairs(xi, chi, lnw, E_bar)
    samples = _sample_masks(pairs, dxi_max_deg)

    # Observed per-sector statistics
    obs: dict[str, pd.DataFrame] = {
        name: _per_sector_rho(pairs, mask, sector)
        for name, (mask, sector) in samples.items()
    }

    # Reference: within-sector OLS slopes at the occupation level.
    # beta_chi for wages reproduces the sectoral wage regression
    # (unconditional specification); the education slope is new.
    occ_rows = []
    for s in range(8):
        sel = occ_sector == s
        bw, sew, pw = _within_sector_ols(lnw[sel], chi[sel])
        bE, seE, pE = _within_sector_ols(E_bar[sel], chi[sel])
        occ_rows.append({
            "sector": SECTOR_LABELS[s],
            "n_occupations": int(sel.sum()),
            "beta_chi_w": bw, "se_chi_w": sew, "p_chi_w": pw,
            "beta_chi_E": bE, "se_chi_E": seE, "p_chi_E": pE,
        })
    occ_slopes = pd.DataFrame(occ_rows)

    # Permutation test: chi shuffled within sector, xi fixed.
    # Pair set, dxi, sector assignment, and outcomes are invariant;
    # only dchi changes per draw.
    rng = np.random.default_rng(seed)
    hits = {name: np.zeros((8, 2), dtype=int) for name in samples}
    obs_mat = {
        name: df[["rho_dchi_w_given_dxi", "rho_dchi_E_given_dxi"]].to_numpy()
        for name, df in obs.items()
    }
    if n_permutations > 0:
        i_idx = pairs["i"].to_numpy()
        j_idx = pairs["j"].to_numpy()
        for _ in range(n_permutations):
            chi_p = chi.copy()
            for s in range(8):
                sel = np.where(occ_sector == s)[0]
                chi_p[sel] = chi_p[rng.permutation(sel)]
            dchi_p = np.abs(chi_p[i_idx] - chi_p[j_idx])
            for name, (mask, sector) in samples.items():
                for s in range(8):
                    pm = mask & (sector == s)
                    if pm.sum() < 3:
                        continue
                    dxi_s = pairs.loc[pm, "dxi_deg"].to_numpy()
                    rw = _spearman_partial(
                        dchi_p[pm], pairs.loc[pm, "abs_dlnw"].to_numpy(), dxi_s)
                    rE = _spearman_partial(
                        dchi_p[pm], pairs.loc[pm, "abs_dE"].to_numpy(), dxi_s)
                    if abs(rw) >= abs(obs_mat[name][s, 0]):
                        hits[name][s, 0] += 1
                    if abs(rE) >= abs(obs_mat[name][s, 1]):
                        hits[name][s, 1] += 1

    # Assemble per-sector table
    sector_frames = []
    for name, df in obs.items():
        df = df.copy()
        df["sample"] = name
        if n_permutations > 0:
            df["p_perm_w"] = (hits[name][:, 0] + 1) / (n_permutations + 1)
            df["p_perm_E"] = (hits[name][:, 1] + 1) / (n_permutations + 1)
        else:
            df["p_perm_w"] = np.nan
            df["p_perm_E"] = np.nan
        sector_frames.append(df)
    sector_df = pd.concat(sector_frames, ignore_index=True)
    sector_df = sector_df.merge(occ_slopes, on="sector")

    # Cross-sector tracking: does per-sector pair rho follow |beta_chi|
    # and does signed structure follow beta_chi^2 ordering? Eight points,
    # descriptive.
    summary_rows = []
    for name in samples:
        sub = sector_df[sector_df["sample"] == name]
        for outcome, rho_col, beta_col in [
            ("wage", "rho_dchi_w_given_dxi", "beta_chi_w"),
            ("education", "rho_dchi_E_given_dxi", "beta_chi_E"),
        ]:
            r_abs, p_abs = spearmanr(np.abs(sub[beta_col]), sub[rho_col])
            summary_rows.append({
                "sample": name, "outcome": outcome,
                "rho_track_absbeta": float(r_abs),
                "p_track_absbeta": float(p_abs),
                "n_sectors": int(len(sub)),
            })
    summary_df = pd.DataFrame(summary_rows)

    if write_csv:
        sector_df.to_csv(
            infra.RP.export_fp("pair_level_chi_by_sector.csv"), index=False)
        summary_df.to_csv(
            infra.RP.export_fp("pair_level_chi_by_sector__summary.csv"),
            index=False)

    log(f"[P3 sector-wise dchi] {n} occs | samples: "
        + ", ".join(f"{k}={int(v[0].sum())} pairs" for k, v in samples.items())
        + f" | permutations={n_permutations}")

    return SectorPairResult(
        n_occ=n, occ_df=m, sector_df=sector_df, summary_df=summary_df)
