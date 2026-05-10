"""
radial_specialization.py
------------------------
Two complementary analyses of the polar geometry:

Track 1 — Angular variance decomposition
    For each descriptor (skill / ability), partition its variance into
    a contribution from angular position (xi) and an additional
    contribution from radial position (chi) after xi-residualization.
    Quantifies the claim "xi carries most of the structure".

Track 2 — Radial intensification within direction
    Tests three operationalizations of "specialization within a direction":
      (i)   intensification — distance from the global mean profile (POSITIVE)
      (ii)  focusing on residuals — Gini / participation on |resid| (NULL)
      (iii) not tested here (was kernel-self-fit confounded)
    Statistics: global Spearman, partial Spearman controlling for xi,
    and per-sector consistency over 8 sectors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata

import infra
from infra import log

from overlays import OverlayResult


# ─────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────

VON_MISES_KAPPA = 16.0
N_SECTORS_DEFAULT = 8


def _circ_kernel_predict(theta_obs, y_obs, theta_query, kappa=VON_MISES_KAPPA):
    """Von Mises kernel regression on the circle."""
    theta_obs = np.mod(theta_obs, 2 * np.pi)
    theta_query = np.mod(theta_query, 2 * np.pi)
    d = theta_query[:, None] - theta_obs[None, :]
    K = np.exp(kappa * np.cos(d))
    num = np.nansum(K * y_obs[None, :], axis=1)
    den = np.nansum(K, axis=1)
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def _gini(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0 or np.any(x < 0) or np.sum(x) <= 0:
        return float("nan")
    n = x.size
    xs = np.sort(x)
    ranks = np.arange(1, n + 1, dtype=float)
    total = float(np.sum(xs))
    return float((2.0 * np.sum(ranks * xs)) / (n * total) - (n + 1.0) / n)


def _participation(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    s = float(np.sum(x))
    s2 = float(np.sum(x * x))
    if s <= 0 or s2 <= 0:
        return float("nan")
    return float((s * s) / (x.size * s2))


def _rank_residualize(y, x, kappa=VON_MISES_KAPPA):
    """Residualize y on circular x via von Mises kernel regression on ranks."""
    yr = rankdata(y)
    pred = _circ_kernel_predict(x, yr, x, kappa=kappa)
    return yr - pred


def _spearman_partial(y, chi, xi, kappa=VON_MISES_KAPPA):
    """Spearman partial correlation between y and chi, controlling for xi.

    Implementation: rank-residualize both y and chi against xi via von Mises
    kernel regression, then take Pearson correlation of the residuals.
    """
    m = np.isfinite(y) & np.isfinite(chi) & np.isfinite(xi)
    if m.sum() < 20:
        return float("nan"), int(m.sum())
    yr = _rank_residualize(y[m], xi[m], kappa=kappa)
    cr = _rank_residualize(chi[m], xi[m], kappa=kappa)
    a = yr - np.mean(yr)
    b = cr - np.mean(cr)
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denom <= 0:
        return float("nan"), int(m.sum())
    return float(np.sum(a * b) / denom), int(m.sum())


def _labelname(label: str) -> str:
    if label == "skill":
        return "Skills"
    if label == "ability":
        return "Abilities"
    return f"{label}s"


# ─────────────────────────────────────────────────────────────
# Track 1 — Angular variance decomposition
# ─────────────────────────────────────────────────────────────

@dataclass
class AngularVarianceResult:
    label: str
    df: pd.DataFrame   # one row per descriptor: R2_xi, R2_chi_of_total, R2_residual, n
    summary: dict      # aggregate statistics


def compute_angular_variance(
    overlay: OverlayResult,
    *,
    kappa: float = VON_MISES_KAPPA,
    write_csv: bool = True,
) -> AngularVarianceResult:
    """For each descriptor, decompose variance into xi, chi (after xi), and residual."""
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        have_lowess = True
    except Exception:
        have_lowess = False

    occ = overlay.occ
    names = overlay.chosen
    xi = np.mod(occ["xi"].to_numpy(float), 2 * np.pi)
    chi = occ["chi"].to_numpy(float)

    rows = []
    for n in names:
        v = occ[n].to_numpy(float)
        m = np.isfinite(v) & np.isfinite(xi) & np.isfinite(chi)
        if m.sum() < 20:
            continue
        v_, xi_, chi_ = v[m], xi[m], chi[m]
        ss_total = float(np.sum((v_ - np.mean(v_)) ** 2))
        if ss_total <= 0:
            continue

        # f(xi) via von Mises kernel regression
        v_xi_pred = _circ_kernel_predict(xi_, v_, xi_, kappa=kappa)
        ss_xi = float(np.sum((v_xi_pred - np.mean(v_)) ** 2))
        r2_xi = ss_xi / ss_total

        # g(chi) on residuals
        resid_xi = v_ - v_xi_pred
        if have_lowess and chi_.size >= 50:
            sm = lowess(resid_xi, chi_, frac=0.4, return_sorted=True, it=1)
            g_pred = np.interp(chi_, sm[:, 0], sm[:, 1])
        else:
            coeffs = np.polyfit(chi_, resid_xi, deg=3)
            g_pred = np.polyval(coeffs, chi_)
        ss_chi = float(np.sum((g_pred - np.mean(resid_xi)) ** 2))
        r2_chi_total = ss_chi / ss_total

        rows.append({
            "descriptor": n,
            "n": int(m.sum()),
            "R2_xi": r2_xi,
            "R2_chi_of_total": r2_chi_total,
            "R2_residual": max(0.0, 1.0 - r2_xi - r2_chi_total),
        })

    df = pd.DataFrame(rows).sort_values("R2_xi", ascending=False).reset_index(drop=True)

    summary = {
        "median_R2_xi": float(df["R2_xi"].median()),
        "median_R2_chi_of_total": float(df["R2_chi_of_total"].median()),
        "median_R2_residual": float(df["R2_residual"].median()),
        "frac_descriptors_R2_xi_above_0.3": float((df["R2_xi"] > 0.3).mean()),
        "frac_descriptors_R2_xi_above_0.5": float((df["R2_xi"] > 0.5).mean()),
        "n_descriptors": int(len(df)),
    }

    if write_csv:
        df.to_csv(infra.RP.export_fp(f"radial_spec__angular_variance__{overlay.label}s.csv"),
                  index=False)
        pd.DataFrame([summary]).to_csv(
            infra.RP.export_fp(f"radial_spec__angular_variance__{overlay.label}s__summary.csv"),
            index=False,
        )

    log(f"[Angular variance] {overlay.label}s: median R²(ξ)={summary['median_R2_xi']:.3f}, "
        f"R²(χ)={summary['median_R2_chi_of_total']:.3f}")

    return AngularVarianceResult(label=overlay.label, df=df, summary=summary)


def plot_angular_variance(
    skills_result: AngularVarianceResult,
    abilities_result: AngularVarianceResult,
    *,
    figsize: tuple[float, float] | None = None,
    show: bool = True,
) -> dict:
    """Stacked bar chart per descriptor: R²(ξ) + R²(χ) + R²(residual) = 1.

    Each bar is one descriptor, sorted by R²(ξ) descending. Three stacked
    segments show the variance contributions. Reads at a glance:
      - tall blue base => ξ dominates
      - thin orange band => χ adds little
      - grey top => unexplained variance
    """
    color_xi = "#3A6EA5"
    color_chi = "#e76f51"
    color_resid = "#bbbbbb"

    # Width scales with descriptor count so bars stay readable in both panels
    n_skills = len(skills_result.df)
    n_abil = len(abilities_result.df)
    width_per_bar = 0.18  # inches
    panel_widths = [max(4.0, width_per_bar * n_skills),
                    max(4.0, width_per_bar * n_abil)]
    if figsize is None:
        figsize = (sum(panel_widths) + 1.5, 6.5)

    fig, axes = plt.subplots(
        1, 2, figsize=figsize,
        gridspec_kw={"width_ratios": panel_widths},
    )

    for ax, res, name in zip(
        axes, [skills_result, abilities_result], ["Skills", "Abilities"]
    ):
        df = res.df  # already sorted by R²(ξ) descending
        labels = df["descriptor"].tolist()
        n = len(labels)
        x = np.arange(n)

        r2_xi = df["R2_xi"].to_numpy(float)
        r2_chi = df["R2_chi_of_total"].to_numpy(float)
        r2_res = df["R2_residual"].to_numpy(float)

        ax.bar(x, r2_xi, color=color_xi, label="R²(ξ)")
        ax.bar(x, r2_chi, bottom=r2_xi, color=color_chi, label="R²(χ of total)")
        ax.bar(x, r2_res, bottom=r2_xi + r2_chi, color=color_resid, label="R²(residual)")

        med_xi = res.summary["median_R2_xi"]
        med_chi = res.summary["median_R2_chi_of_total"]
        ax.axhline(med_xi, color=color_xi, ls="--", lw=1.0, alpha=0.8)
        ax.text(n - 0.5, med_xi + 0.01,
                f"median R²(ξ) = {med_xi:.2f}",
                ha="right", va="bottom", fontsize=8, color=color_xi)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Variance share")
        ax.set_title(
            f"{name} ({n} descriptors)\n"
            f"R²(ξ): median {med_xi:.2f}, > 0.3 in "
            f"{int(round(res.summary['frac_descriptors_R2_xi_above_0.3'] * n))}/{n}"
        )
        ax.set_xlim(-0.6, n - 0.4)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.suptitle("Angular variance decomposition per descriptor",
                 fontsize=13, y=1.00)
    fig.tight_layout()

    out_pdf = infra.RP.figure_fp("radial_spec__angular_variance.pdf")
    out_png = infra.RP.figure_fp("radial_spec__angular_variance.png")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    log(f"Saved figure: {out_pdf.name}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {"pdf": out_pdf, "png": out_png}


# ─────────────────────────────────────────────────────────────
# Track 2 — Radial intensification within direction
# ─────────────────────────────────────────────────────────────

INTENSIFICATION_METRIC = "norm_v_minus_global"
NULL_METRIC_GINI = "gini_resid_abs"
NULL_METRIC_PARTICIPATION = "participation_resid_abs"

METRICS_TRACK2 = [INTENSIFICATION_METRIC, NULL_METRIC_GINI, NULL_METRIC_PARTICIPATION]

METRIC_DISPLAY = {
    INTENSIFICATION_METRIC:   "Distance from global mean profile",
    NULL_METRIC_GINI:         "Gini on |residual|",
    NULL_METRIC_PARTICIPATION: "Participation on |residual|",
}


@dataclass
class RadialIntensificationResult:
    label: str
    metrics_df: pd.DataFrame    # per occupation: onet_code, xi, chi, three metrics
    statistics_df: pd.DataFrame # per metric: rho_global, p_global, rho_partial, p_partial, ...
    per_sector_df: pd.DataFrame # per metric x sector: rho, p, n, sector_center_deg


def _per_occupation_metrics(overlay: OverlayResult, kappa: float) -> pd.DataFrame:
    occ = overlay.occ
    names = overlay.chosen
    base = occ[["onet_code", "xi", "chi"]].copy()

    V = occ[names].to_numpy(float)
    xi = np.mod(occ["xi"].to_numpy(float), 2 * np.pi)
    n_occ = V.shape[0]

    # xi-expected descriptor profile per occupation (for residuals)
    V_xi = np.zeros_like(V)
    for j in range(V.shape[1]):
        col = V[:, j]
        m = np.isfinite(col)
        if m.sum() < 5:
            V_xi[:, j] = np.nan
            continue
        V_xi[:, j] = _circ_kernel_predict(xi[m], col[m], xi, kappa=kappa)

    # Global mean vector
    V_mean_global = np.nanmean(V, axis=0)

    out = {
        INTENSIFICATION_METRIC:    np.full(n_occ, np.nan),
        NULL_METRIC_GINI:          np.full(n_occ, np.nan),
        NULL_METRIC_PARTICIPATION: np.full(n_occ, np.nan),
    }

    for i in range(n_occ):
        v = V[i]
        v_xi = V_xi[i]
        m = np.isfinite(v) & np.isfinite(v_xi) & np.isfinite(V_mean_global)
        if m.sum() < 5:
            continue
        v_ = v[m]
        v_xi_ = v_xi[m]
        v_glob_ = V_mean_global[m]
        resid = v_ - v_xi_

        out[INTENSIFICATION_METRIC][i]    = float(np.linalg.norm(v_ - v_glob_))
        out[NULL_METRIC_GINI][i]          = _gini(np.abs(resid))
        out[NULL_METRIC_PARTICIPATION][i] = _participation(np.abs(resid))

    df = base.copy()
    for k, v in out.items():
        df[k] = v
    return df


def _spearman_with_p(y, x):
    m = np.isfinite(y) & np.isfinite(x)
    if m.sum() < 10:
        return float("nan"), float("nan"), int(m.sum())
    rho, p = spearmanr(y[m], x[m])
    return float(rho), float(p), int(m.sum())


def _build_circular_smoother(xi: np.ndarray, kappa: float = VON_MISES_KAPPA) -> np.ndarray:
    """Build the row-normalized von Mises smoother S so that residual(z) = z - S @ z.

    Depends only on xi, so it can be precomputed once and reused across many
    metrics or permutation rounds.
    """
    xi_m = np.mod(xi, 2 * np.pi)
    d = xi_m[:, None] - xi_m[None, :]
    K = np.exp(kappa * np.cos(d))
    row_sums = K.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    return K / row_sums


def _spearman_partial_with_p(y, chi, xi, n_perm: int = 0, kappa: float = VON_MISES_KAPPA,
                              seed: int = 42):
    """Partial Spearman + permutation p-value.

    Optimizations vs the naive implementation:
      - The von Mises smoother S depends only on xi, so it is built once
        per call and reused across all permutations.
      - y is fixed across permutations, so its rank-residual is computed once.
      - We permute the *ranks* of chi (Spearman is rank-based) and apply the
        precomputed smoother to all permutations in a single matrix product.

    Speedup is roughly 50-100x for n_perm=1000 vs the naive recomputation.
    """
    m = np.isfinite(y) & np.isfinite(chi) & np.isfinite(xi)
    n = int(m.sum())
    if n < 20:
        return float("nan"), float("nan"), n

    y_m = y[m]
    chi_m = chi[m]
    xi_m = np.mod(xi[m], 2 * np.pi)

    S = _build_circular_smoother(xi_m, kappa=kappa)

    # Observed partial rho
    yr = rankdata(y_m)
    cr = rankdata(chi_m)
    yr_resid = yr - S @ yr
    cr_resid = cr - S @ cr

    a = yr_resid - yr_resid.mean()
    b = cr_resid - cr_resid.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom <= 0:
        return float("nan"), float("nan"), n
    rho = float((a * b).sum() / denom)

    if n_perm <= 0:
        return rho, float("nan"), n

    # Vectorized permutations.
    rng = np.random.default_rng(seed)
    perm_ranks = np.empty((n_perm, n), dtype=float)
    for k in range(n_perm):
        perm_ranks[k] = rng.permutation(cr)

    smoothed = (S @ perm_ranks.T).T               # (n_perm, n)
    perm_resid = perm_ranks - smoothed
    perm_resid -= perm_resid.mean(axis=1, keepdims=True)

    num = perm_resid @ a                          # (n_perm,)
    norms = np.sqrt((perm_resid * perm_resid).sum(axis=1))
    denom_a = np.sqrt((a * a).sum())
    valid = (norms > 0) & np.isfinite(norms)
    null_rhos = np.where(valid, num / np.where(valid, norms * denom_a, 1.0), 0.0)

    p = float((np.abs(null_rhos) >= abs(rho)).mean())
    return rho, p, n


def compute_radial_intensification(
    overlay: OverlayResult,
    *,
    kappa: float = VON_MISES_KAPPA,
    n_sectors: int = N_SECTORS_DEFAULT,
    n_permutations: int = 1000,
    write_csv: bool = True,
) -> RadialIntensificationResult:
    """Compute three per-occupation metrics, plus global, partial, and per-sector statistics."""
    metrics_df = _per_occupation_metrics(overlay, kappa=kappa)

    chi = metrics_df["chi"].to_numpy(float)
    xi = metrics_df["xi"].to_numpy(float)

    # Global + partial Spearman per metric
    stat_rows = []
    for col in METRICS_TRACK2:
        y = metrics_df[col].to_numpy(float)
        rho_g, p_g, n_g = _spearman_with_p(y, chi)
        rho_p, p_p, n_p = _spearman_partial_with_p(
            y, chi, xi, n_perm=n_permutations, kappa=kappa,
        )
        stat_rows.append({
            "metric": col,
            "display_name": METRIC_DISPLAY[col],
            "rho_global": rho_g,
            "p_global": p_g,
            "n_global": n_g,
            "rho_partial_xi": rho_p,
            "p_partial_xi": p_p,
            "n_partial": n_p,
        })

    # Per-sector Spearman
    sector_width = 2 * np.pi / n_sectors
    sector_idx = (np.mod(xi, 2 * np.pi) // sector_width).astype(int)
    sector_idx = np.clip(sector_idx, 0, n_sectors - 1)

    per_sector_rows = []
    for col in METRICS_TRACK2:
        y = metrics_df[col].to_numpy(float)
        for s in range(n_sectors):
            mask = (sector_idx == s) & np.isfinite(y) & np.isfinite(chi)
            center_deg = float(np.degrees(sector_width * (s + 0.5)))
            if mask.sum() < 10:
                per_sector_rows.append({
                    "metric": col, "sector": s, "sector_center_deg": center_deg,
                    "rho": float("nan"), "p": float("nan"), "n": int(mask.sum()),
                })
                continue
            rho, p = spearmanr(y[mask], chi[mask])
            per_sector_rows.append({
                "metric": col, "sector": s, "sector_center_deg": center_deg,
                "rho": float(rho), "p": float(p), "n": int(mask.sum()),
            })

    statistics_df = pd.DataFrame(stat_rows)
    per_sector_df = pd.DataFrame(per_sector_rows)

    # Add per-sector summary to statistics
    for col in METRICS_TRACK2:
        sub = per_sector_df[per_sector_df["metric"] == col]
        rho_vals = sub["rho"].dropna().to_numpy(float)
        if rho_vals.size > 0:
            frac_pos = float((rho_vals > 0).mean())
            median_rho = float(np.median(rho_vals))
        else:
            frac_pos = float("nan")
            median_rho = float("nan")
        idx = statistics_df.index[statistics_df["metric"] == col]
        statistics_df.loc[idx, "frac_sectors_positive"] = frac_pos
        statistics_df.loc[idx, "median_sector_rho"] = median_rho
        statistics_df.loc[idx, "n_sectors"] = n_sectors

    if write_csv:
        metrics_df.to_csv(infra.RP.export_fp(f"radial_spec__intensification__metrics__{overlay.label}s.csv"),
                          index=False)
        statistics_df.to_csv(infra.RP.export_fp(f"radial_spec__intensification__statistics__{overlay.label}s.csv"),
                             index=False)
        per_sector_df.to_csv(infra.RP.export_fp(f"radial_spec__intensification__per_sector__{overlay.label}s.csv"),
                             index=False)

    log(f"[Radial intensification] {overlay.label}s: "
        f"{INTENSIFICATION_METRIC} partial ρ="
        f"{statistics_df.loc[statistics_df['metric']==INTENSIFICATION_METRIC, 'rho_partial_xi'].iloc[0]:+.3f}")

    return RadialIntensificationResult(
        label=overlay.label,
        metrics_df=metrics_df,
        statistics_df=statistics_df,
        per_sector_df=per_sector_df,
    )


def _lowess_or_polyfit(x: np.ndarray, y: np.ndarray, n_grid: int = 200):
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size < 4:
        return np.array([]), np.array([])
    xg = np.linspace(np.nanmin(x), np.nanmax(x), n_grid)
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        sm = lowess(y, x, frac=0.4, return_sorted=True, it=1)
        return xg, np.interp(xg, sm[:, 0], sm[:, 1])
    except Exception:
        coeffs = np.polyfit(x, y, deg=3)
        return xg, np.polyval(coeffs, xg)


def plot_radial_intensification(
    skills_result: RadialIntensificationResult,
    abilities_result: RadialIntensificationResult,
    *,
    show_null_metrics: bool = False,
    figsize: tuple[float, float] | None = None,
    point_alpha: float = 0.35,
    point_size: float = 12,
    show_sector_overlays: bool = True,
    show: bool = True,
) -> dict:
    """Paper figure for radial intensification.

    Default (``show_null_metrics=False``):
        1×2 layout — one panel per descriptor family, showing only the
        intensification metric (distance from global mean profile).
        This is the figure used in the paper.

    Optional (``show_null_metrics=True``):
        3×2 layout — adds two rows for the null metrics (Gini and
        participation ratio on residuals). Saved with a separate filename
        suffix and intended as supplementary material.
    """
    metrics_to_plot = METRICS_TRACK2 if show_null_metrics else [INTENSIFICATION_METRIC]
    n_rows = len(metrics_to_plot)

    if figsize is None:
        figsize = (11, 4.0) if n_rows == 1 else (13.5, 9.0)

    if n_rows == 1:
        fig, axes_arr = plt.subplots(1, 2, figsize=figsize, sharex=False)
        axes = np.array([[axes_arr[0], axes_arr[1]]])  # shape (1, 2) for unified indexing
    else:
        fig, axes = plt.subplots(n_rows, 2, figsize=figsize, sharex=False)

    cmap = plt.cm.twilight

    for col_idx, (res, col_title) in enumerate([
        (skills_result, "Skills"),
        (abilities_result, "Abilities"),
    ]):
        chi = res.metrics_df["chi"].to_numpy(float)
        xi = res.metrics_df["xi"].to_numpy(float)
        n_sectors = int(res.statistics_df["n_sectors"].iloc[0])
        sector_width = 2 * np.pi / n_sectors
        sector_idx = (np.mod(xi, 2 * np.pi) // sector_width).astype(int)
        sector_idx = np.clip(sector_idx, 0, n_sectors - 1)
        sector_colors = [cmap(s / n_sectors) for s in range(n_sectors)]

        for row_idx, metric in enumerate(metrics_to_plot):
            ax = axes[row_idx, col_idx]
            y = res.metrics_df[metric].to_numpy(float)

            ax.scatter(chi, y, s=point_size, alpha=point_alpha,
                       color="#888888", edgecolors="none")

            xg, yg = _lowess_or_polyfit(chi, y)
            if xg.size:
                ax.plot(xg, yg, "-", color="black", lw=2.2, label="global trend")

            if show_sector_overlays:
                for s in range(n_sectors):
                    mask = (sector_idx == s) & np.isfinite(y) & np.isfinite(chi)
                    if mask.sum() < 6:
                        continue
                    xs = chi[mask]; ys = y[mask]
                    order = np.argsort(xs)
                    xs, ys = xs[order], ys[order]
                    try:
                        coeffs = np.polyfit(xs, ys, deg=2)
                        xg_s = np.linspace(xs.min(), xs.max(), 50)
                        yg_s = np.polyval(coeffs, xg_s)
                        ax.plot(xg_s, yg_s, "-", color=sector_colors[s], lw=1.0, alpha=0.85)
                    except Exception:
                        pass

            stat_row = res.statistics_df[res.statistics_df["metric"] == metric].iloc[0]
            rho_p = stat_row["rho_partial_xi"]
            p_p = stat_row["p_partial_xi"]
            frac_pos = stat_row["frac_sectors_positive"]
            txt = (f"partial ρ(χ|ξ) = {rho_p:+.3f}\n"
                   f"p (perm) = {p_p:.3g}\n"
                   f"sectors > 0: {int(round(frac_pos * n_sectors))}/{n_sectors}")
            ax.text(0.02, 0.97, txt, transform=ax.transAxes,
                    fontsize=9, ha="left", va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              alpha=0.9, edgecolor="lightgray"))

            ax.set_ylabel(METRIC_DISPLAY[metric])
            ax.grid(alpha=0.3)
            if row_idx == 0:
                ax.set_title(col_title, fontsize=12)
            if row_idx == n_rows - 1:
                ax.set_xlabel("Radial coordinate χ")

    fig.suptitle("Radial intensification within angular direction",
                 fontsize=13, y=1.00)
    fig.tight_layout()

    suffix = "" if not show_null_metrics else "__with_null_metrics"
    out_pdf = infra.RP.figure_fp(f"radial_spec__intensification{suffix}.pdf")
    out_png = infra.RP.figure_fp(f"radial_spec__intensification{suffix}.png")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    log(f"Saved figure: {out_pdf.name}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {"pdf": out_pdf, "png": out_png}