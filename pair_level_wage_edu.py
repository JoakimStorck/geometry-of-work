"""
pair_level_wage_edu.py
----------------------
Test of Proposition P3 at the pair level.

    For all pairs of occupations with wage and education data: absolute
    differences in log median hourly wage and weighted-average required
    education, related to geometric distances (circular angular, radial,
    Euclidean).

Binned curves report the mean and the 20th-80th percentile range per
angular-distance bin. Each outcome is additionally decomposed via the
geometry-only specification (y ~ cos xi + sin xi + chi, the same form as
the Mincer geometry regression): pair differences split into a
geometry-predicted component and a residual component. The signed
components sum to the total pair difference; their mean absolute values
combine in quadrature. On the squared scale the decomposition is exact
and the geometry share of squared pair differences equals the regression
R^2 globally.

Significance is assessed with an occupation-level permutation test:
positions (xi, chi) are shuffled jointly across occupations and all
distance-outcome statistics are recomputed per draw. Pairs share
occupations and are not independent, so asymptotic pair-level p-values
would be anticonservative. Decomposition statistics are descriptive and
carry no permutation p-value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata

import infra
from infra import log


# ─────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────

def load_wages(path) -> pd.DataFrame:
    """BLS OEWS national file -> detailed SOC with numeric H_MEDIAN, TOT_EMP."""
    w = pd.read_excel(path)
    w = w[w["O_GROUP"] == "detailed"].copy()
    w["H_MEDIAN"] = pd.to_numeric(w["H_MEDIAN"], errors="coerce")
    w["TOT_EMP"] = pd.to_numeric(w["TOT_EMP"], errors="coerce")
    return w.dropna(subset=["H_MEDIAN"])[["OCC_CODE", "H_MEDIAN", "TOT_EMP"]]


def load_education(path) -> pd.Series:
    """O*NET Education, Training, and Experience data -> weighted-average
    required education per onet_code.

    Accepts the canonical O*NET database text file (tab-separated, with
    'n/a' for non-applicable fields) or an Excel export with the same
    columns.
    """
    p = str(path)
    if p.endswith(".txt"):
        ed = pd.read_csv(path, sep="\t", na_values=["n/a"])
    else:
        ed = pd.read_excel(path)
    ed = ed[ed["Scale ID"] == "RL"].copy()
    ed["Category"] = pd.to_numeric(ed["Category"], errors="coerce")
    ed["Data Value"] = pd.to_numeric(ed["Data Value"], errors="coerce")
    ed = ed.dropna(subset=["Category", "Data Value"])
    return (ed["Category"] * ed["Data Value"] / 100).groupby(
        ed["O*NET-SOC Code"]).sum().rename("E_bar")

# ─────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────

def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def _wpearson(a: np.ndarray, b: np.ndarray, wt: np.ndarray) -> float:
    aw = np.average(a, weights=wt)
    bw = np.average(b, weights=wt)
    cov = np.average((a - aw) * (b - bw), weights=wt)
    va = np.average((a - aw) ** 2, weights=wt)
    vb = np.average((b - bw) ** 2, weights=wt)
    return float(cov / np.sqrt(va * vb))


def _residualize(v: np.ndarray, ctrl: np.ndarray) -> np.ndarray:
    return v - np.polyval(np.polyfit(ctrl, v, 1), ctrl)


def _all_stats(rdxi, rdchi, rdEuc, radw, radE) -> dict[str, float]:
    """All unweighted statistics on rank-transformed inputs."""
    out = {
        "d_w": _pearson(rdEuc, radw),
        "d_E": _pearson(rdEuc, radE),
        "dxi_w": _pearson(rdxi, radw),
        "dxi_E": _pearson(rdxi, radE),
        "dchi_w": _pearson(rdchi, radw),
        "dchi_E": _pearson(rdchi, radE),
    }
    rdxi_res = _residualize(rdxi, rdchi)
    rdchi_res = _residualize(rdchi, rdxi)
    out["dxi_given_dchi_w"] = _pearson(rdxi_res, _residualize(radw, rdchi))
    out["dxi_given_dchi_E"] = _pearson(rdxi_res, _residualize(radE, rdchi))
    out["dchi_given_dxi_w"] = _pearson(rdchi_res, _residualize(radw, rdxi))
    out["dchi_given_dxi_E"] = _pearson(rdchi_res, _residualize(radE, rdxi))
    return out


# ─────────────────────────────────────────────────────────────
# Main computation
# ─────────────────────────────────────────────────────────────

@dataclass
class PairLevelResult:
    n_occ: int
    n_pairs: int
    occ_df: pd.DataFrame      # merged occupation-level data incl. fit/residual
    pairs_df: pd.DataFrame    # distances, total/fit/residual differences, weight
    binned_df: pd.DataFrame   # per-bin mean/p20/p80 (totals) + component means
    stats_df: pd.DataFrame    # statistic, rho, rho_weighted, p_permutation


def compute_pair_level(
    occ_df: pd.DataFrame,
    wages: pd.DataFrame,
    education: pd.Series,
    *,
    n_bins: int = 30,
    n_permutations: int = 1000,
    seed: int = 0,
    write_csv: bool = True,
) -> PairLevelResult:
    """Pair-level wage/education differences vs geometric distances.

    occ_df needs columns onet_code, xi, chi. Wages are merged on the 7-char
    SOC prefix of onet_code; education on onet_code.

    n_permutations controls the occupation-level permutation test (roughly
    5-10 minutes at 1000 draws). Set 0 to skip; p-values are then NaN.
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
    emp = m["TOT_EMP"].to_numpy(float)

    # Geometry-only decomposition: same specification as the Mincer
    # geometry regression (Eq. mincer-geometry)
    Xg = np.column_stack([np.ones(n), np.cos(xi), np.sin(xi), chi])
    beta_w = np.linalg.lstsq(Xg, lnw, rcond=None)[0]
    beta_E = np.linalg.lstsq(Xg, E_bar, rcond=None)[0]
    fit_w, fit_E = Xg @ beta_w, Xg @ beta_E
    res_w, res_E = lnw - fit_w, E_bar - fit_E
    m["lnw"], m["fit_lnw"], m["res_lnw"] = lnw, fit_w, res_w
    m["fit_E"], m["res_E"] = fit_E, res_E

    iu = np.triu_indices(n, k=1)

    def distances(xi_, chi_):
        dd = np.mod(xi_[iu[0]] - xi_[iu[1]], 2 * np.pi)
        dxi_ = np.minimum(dd, 2 * np.pi - dd)
        dchi_ = np.abs(chi_[iu[0]] - chi_[iu[1]])
        x_, y_ = np.cos(xi_) * chi_, np.sin(xi_) * chi_
        dEuc_ = np.hypot(x_[iu[0]] - x_[iu[1]], y_[iu[0]] - y_[iu[1]])
        return dxi_, dchi_, dEuc_

    dxi, dchi, dEuc = distances(xi, chi)

    pairs_df = pd.DataFrame({
        "i": iu[0], "j": iu[1],
        "dxi_deg": np.degrees(dxi),
        "dchi": dchi,
        "d_euclid": dEuc,
        "abs_dlnw": np.abs(lnw[iu[0]] - lnw[iu[1]]),
        "abs_dE": np.abs(E_bar[iu[0]] - E_bar[iu[1]]),
        "abs_dfit_lnw": np.abs(fit_w[iu[0]] - fit_w[iu[1]]),
        "abs_dres_lnw": np.abs(res_w[iu[0]] - res_w[iu[1]]),
        "abs_dfit_E": np.abs(fit_E[iu[0]] - fit_E[iu[1]]),
        "abs_dres_E": np.abs(res_E[iu[0]] - res_E[iu[1]]),
        "weight": np.sqrt(emp[iu[0]] * emp[iu[1]]),
    })

    # Binned curves with dispersion (totals) and component means
    edges = np.linspace(0.0, 180.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.clip(np.digitize(pairs_df["dxi_deg"], edges) - 1, 0, n_bins - 1)
    rows = []
    for k in range(n_bins):
        mask = bin_idx == k
        sub = pairs_df.loc[mask]
        rows.append({
            "bin_center_deg": float(centers[k]),
            "mean_abs_dlnw": float(sub["abs_dlnw"].mean()),
            "p20_abs_dlnw": float(sub["abs_dlnw"].quantile(0.20)),
            "p80_abs_dlnw": float(sub["abs_dlnw"].quantile(0.80)),
            "mean_abs_dfit_lnw": float(sub["abs_dfit_lnw"].mean()),
            "mean_abs_dres_lnw": float(sub["abs_dres_lnw"].mean()),
            "mean_abs_dE": float(sub["abs_dE"].mean()),
            "p20_abs_dE": float(sub["abs_dE"].quantile(0.20)),
            "p80_abs_dE": float(sub["abs_dE"].quantile(0.80)),
            "mean_abs_dfit_E": float(sub["abs_dfit_E"].mean()),
            "mean_abs_dres_E": float(sub["abs_dres_E"].mean()),
            "n_pairs": int(mask.sum()),
        })
    binned_df = pd.DataFrame(rows)

    # Observed statistics: outcome ranks are fixed under position permutation
    radw = rankdata(pairs_df["abs_dlnw"])
    radE = rankdata(pairs_df["abs_dE"])
    rdxi, rdchi, rdEuc = rankdata(dxi), rankdata(dchi), rankdata(dEuc)
    obs = _all_stats(rdxi, rdchi, rdEuc, radw, radE)

    wstats = {
        "d_w": _wpearson(rdEuc, radw, pairs_df["weight"]),
        "d_E": _wpearson(rdEuc, radE, pairs_df["weight"]),
        "dxi_w": _wpearson(rdxi, radw, pairs_df["weight"]),
        "dxi_E": _wpearson(rdxi, radE, pairs_df["weight"]),
        "dchi_w": _wpearson(rdchi, radw, pairs_df["weight"]),
        "dchi_E": _wpearson(rdchi, radE, pairs_df["weight"]),
    }

    # Occupation-level permutation test (hypothesis-testing statistics only)
    hits = {k: 0 for k in obs}
    if n_permutations > 0:
        rng = np.random.default_rng(seed)
        for _ in range(n_permutations):
            p = rng.permutation(n)
            dxi_p, dchi_p, dEuc_p = distances(xi[p], chi[p])
            perm = _all_stats(rankdata(dxi_p), rankdata(dchi_p),
                              rankdata(dEuc_p), radw, radE)
            for k in obs:
                if abs(perm[k]) >= abs(obs[k]):
                    hits[k] += 1
        p_perm = {k: (hits[k] + 1) / (n_permutations + 1) for k in obs}
    else:
        p_perm = {k: float("nan") for k in obs}

    stats_df = pd.DataFrame([
        {"statistic": k, "rho": obs[k],
         "rho_weighted": wstats.get(k, np.nan),
         "p_permutation": p_perm[k]}
        for k in obs
    ])

    # Decomposition diagnostics (descriptive, no permutation p)
    decomp = {
        "dxi_fit_w": _pearson(rdxi, rankdata(pairs_df["abs_dfit_lnw"])),
        "dxi_res_w": _pearson(rdxi, rankdata(pairs_df["abs_dres_lnw"])),
        "dxi_fit_E": _pearson(rdxi, rankdata(pairs_df["abs_dfit_E"])),
        "dxi_res_E": _pearson(rdxi, rankdata(pairs_df["abs_dres_E"])),
    }
    stats_df = pd.concat([stats_df, pd.DataFrame(
        [{"statistic": k, "rho": v, "rho_weighted": np.nan,
          "p_permutation": np.nan} for k, v in decomp.items()])],
        ignore_index=True)

    if write_csv:
        binned_df.to_csv(infra.RP.export_fp("pair_level_wage_edu_vs_dxi__binned.csv"),
                         index=False)
        stats_df.to_csv(infra.RP.export_fp("pair_level_geometry_wage_education.csv"),
                        index=False)

    log(f"[P3 pair-level] {n} occs, {len(pairs_df)} pairs | "
        f"rho(dxi,|dlnw|)={obs['dxi_w']:+.3f} (fit {decomp['dxi_fit_w']:+.3f}, "
        f"res {decomp['dxi_res_w']:+.3f}) | "
        f"rho(dxi,|dE|)={obs['dxi_E']:+.3f} (fit {decomp['dxi_fit_E']:+.3f}, "
        f"res {decomp['dxi_res_E']:+.3f}) | permutations={n_permutations}")

    return PairLevelResult(
        n_occ=n,
        n_pairs=len(pairs_df),
        occ_df=m,
        pairs_df=pairs_df,
        binned_df=binned_df,
        stats_df=stats_df,
    )


def plot_pair_level(
    result: PairLevelResult,
    *,
    figsize: tuple[float, float] = (11, 4.2),
    show: bool = True,
) -> dict:
    """Two-panel plot: wage and education differences vs angular distance.

    Each panel shows the 20th-80th percentile band of total differences,
    the binned mean of total differences, and the binned means of the
    geometry-predicted and residual components."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    b = result.binned_df
    s = result.stats_df.set_index("statistic")["rho"]

    panels = [
        ("mean_abs_dlnw", "p20_abs_dlnw", "p80_abs_dlnw",
         "mean_abs_dfit_lnw", "mean_abs_dres_lnw",
         s["dxi_w"], s["dxi_fit_w"], s["dxi_res_w"],
         r"|$\Delta$ ln w|", "(A) Wage differences"),
        ("mean_abs_dE", "p20_abs_dE", "p80_abs_dE",
         "mean_abs_dfit_E", "mean_abs_dres_E",
         s["dxi_E"], s["dxi_fit_E"], s["dxi_res_E"],
         r"|$\Delta\,\bar{E}$| (12-point scale)", "(B) Education differences"),
    ]
    for ax, (mc, lo, hi, fc, rc, rho_t, rho_f, rho_r, ylab, title) in zip(axes, panels):
        ax.fill_between(b["bin_center_deg"], b[lo], b[hi],
                        color="#3A6EA5", alpha=0.15, lw=0,
                        label="20th–80th pct (total)")
        ax.plot(b["bin_center_deg"], b[mc], "o-", lw=1.5, color="#3A6EA5",
                markersize=4, label=f"Total (ρ = {rho_t:+.2f})")
        ax.plot(b["bin_center_deg"], b[fc], "-", lw=1.8, color="#16324F",
                label=f"Geometry-predicted (ρ = {rho_f:+.2f})")
        ax.plot(b["bin_center_deg"], b[rc], "--", lw=1.5, color="0.45",
                label=f"Residual (ρ = {rho_r:+.2f})")
        ax.set_xlabel("Angular distance Δξ (deg)")
        ax.set_ylabel(ylab)
        ax.set_xlim(0, 180)
        ax.set_xticks([0, 45, 90, 135, 180])
        ax.grid(alpha=0.3)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc="upper left", frameon=False)

    fig.suptitle("Pair-level differences in wage and education vs angular distance",
                 fontsize=12, y=1.02)
    fig.tight_layout()

    out_pdf = infra.RP.figure_fp("pair_level_wage_edu_vs_dxi.pdf")
    out_png = infra.RP.figure_fp("pair_level_wage_edu_vs_dxi.png")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    log(f"Saved figure: {out_pdf.name}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {"pdf": out_pdf, "png": out_png}