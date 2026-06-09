"""
hc_analysis.py
--------------
Test 1 — Angular neighbourhood -> HC similarity.

    "If the angular coordinate captures domain identity in a meaningful sense,
     occupations in nearby directions should draw on similar human capital."

    For all pairs of occupations: cosine similarity over DEVIATION descriptor
    vectors (v - v_bar, where v_bar is the mean profile across occupations),
    plotted against circular angular distance.

Raw (uncentered) descriptor vectors are nonnegative and share a large common
profile, which compresses cosine similarity into roughly [0.6, 1.0]. Centering
on the global mean profile removes this common component: the similarity scale
spreads over [-1, 1], the Spearman correlation with angular distance
strengthens, and the binned mean follows a*cos(dxi) closely (the form a planar
signal-plus-noise model predicts). The raw-space Spearman is retained in the
result for reporting. The deviation v - v_bar is the same object whose
magnitude the radial intensification test uses.

Significance is assessed with an occupation-level permutation test: angular
positions are shuffled across occupations and the full pair-level statistic is
recomputed per draw. Pairs share occupations and are not independent, so the
asymptotic p-value from the pair-level Spearman would be anticonservative.

The previous Test 2 (global concentration vs chi) has been superseded by
the within-direction analyses in radial_specialization.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import infra
from infra import log

from overlays import OverlayResult


# ─────────────────────────────────────────────────────────────
# Helpers: descriptor matrix from an OverlayResult
# ─────────────────────────────────────────────────────────────

def _descriptor_matrix(overlay: OverlayResult):
    """Return:
        df_occ_clean: DataFrame with onet_code, xi, chi (for occupations with full data)
        V         : (n_occ, n_descriptors) matrix of raw descriptor values
        xi        : (n_occ,) angular coords in [0, 2*pi)
        chi       : (n_occ,) radial coords in [0, 1]
        names     : list of descriptor names (column order of V)
    """
    occ = overlay.occ
    names = overlay.chosen
    base_cols = ["onet_code", "xi", "chi"]
    sub = occ[base_cols + names].copy()

    mask_complete = sub[names].notna().all(axis=1) & sub["xi"].notna() & sub["chi"].notna()
    sub = sub.loc[mask_complete].reset_index(drop=True)

    V = sub[names].to_numpy(float)
    xi = np.mod(sub["xi"].to_numpy(float), 2 * np.pi)
    chi = sub["chi"].to_numpy(float)
    return sub[base_cols], V, xi, chi, names


def _circular_distance(theta1: np.ndarray, theta2: np.ndarray) -> np.ndarray:
    """Smallest absolute angular distance, in [0, pi]."""
    d = np.mod(theta1 - theta2, 2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


def _cosine_similarity_pairs(V: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity matrix for rows of V."""
    norms = np.linalg.norm(V, axis=1)
    norms = np.where(norms > 0, norms, 1.0)
    Vn = V / norms[:, None]
    return Vn @ Vn.T


def _permutation_p_xi(
    xi: np.ndarray,
    sims: np.ndarray,
    iu: tuple[np.ndarray, np.ndarray],
    observed_rho: float,
    n_permutations: int,
    seed: int = 0,
) -> float:
    """Occupation-level permutation test for the pair-level Spearman.

    Angular positions are shuffled across occupations; pairwise distances and
    the Spearman correlation are recomputed per draw. Two-sided p-value with
    the add-one correction (Phipson & Smyth).
    """
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_permutations):
        xp = rng.permutation(xi)
        dp = _circular_distance(xp[iu[0]], xp[iu[1]])
        r, _ = spearmanr(dp, sims)
        if abs(r) >= abs(observed_rho):
            hits += 1
    return (hits + 1) / (n_permutations + 1)


# ─────────────────────────────────────────────────────────────
# Test 1 — Angular neighbourhood vs HC similarity
# ─────────────────────────────────────────────────────────────

@dataclass
class AngularSimilarityResult:
    label: str
    n_occ: int
    n_pairs: int
    centered: bool
    binned_df: pd.DataFrame   # bin_center_deg, mean/median/p20/p80, n_pairs
    pairs_df: pd.DataFrame    # full pair-level data (dxi, sim, mean_chi)
    rho: float                # pair-level Spearman, similarity vs dxi
    p_value: float            # occupation-level permutation p (NaN if skipped)
    rho_raw: float            # same Spearman on uncentered vectors (footnote value)
    cos_fit_a: float          # binned mean ~ a*cos(dxi)
    cos_fit_r2: float
    rho_chibar_near: float    # Spearman(mean_chi, sim) for dxi < 30 deg
    rho_chibar_far: float     # Spearman(mean_chi, sim) for dxi > 150 deg


def compute_angular_similarity(
    overlay: OverlayResult,
    *,
    n_bins: int = 30,
    center: bool = True,
    n_permutations: int = 1000,
    write_csv: bool = True,
) -> AngularSimilarityResult:
    """Compute HC similarity vs circular angular distance.

    With center=True (default), similarity is the cosine between deviation
    vectors v - v_bar. The raw-vector Spearman is computed alongside and
    stored as rho_raw.

    n_permutations controls the occupation-level permutation test for the
    Spearman correlation (roughly 2-3 minutes per family at 1000 draws).
    Set n_permutations=0 to skip it during iteration; p_value is then NaN.
    """
    label = overlay.label
    _, V, xi, chi, _ = _descriptor_matrix(overlay)
    n = V.shape[0]
    if n < 3:
        raise RuntimeError(f"Too few occupations: {n}")

    iu = np.triu_indices(n, k=1)
    dxi = _circular_distance(xi[iu[0]], xi[iu[1]])

    sims_raw = _cosine_similarity_pairs(V)[iu]
    if center:
        sims = _cosine_similarity_pairs(V - V.mean(axis=0))[iu]
    else:
        sims = sims_raw

    edges = np.linspace(0.0, np.pi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.clip(np.digitize(dxi, edges) - 1, 0, n_bins - 1)

    rows = []
    for k in range(n_bins):
        mask = bin_idx == k
        rows.append({
            "bin_center_rad": float(centers[k]),
            "bin_center_deg": float(np.degrees(centers[k])),
            "mean_sim": float(np.mean(sims[mask])) if mask.any() else np.nan,
            "median_sim": float(np.median(sims[mask])) if mask.any() else np.nan,
            "p20_sim": float(np.quantile(sims[mask], 0.20)) if mask.any() else np.nan,
            "p80_sim": float(np.quantile(sims[mask], 0.80)) if mask.any() else np.nan,
            "n_pairs": int(mask.sum()),
        })
    binned_df = pd.DataFrame(rows)

    mean_chi = 0.5 * (chi[iu[0]] + chi[iu[1]])
    pairs_df = pd.DataFrame({
        "i": iu[0],
        "j": iu[1],
        "dxi_rad": dxi,
        "dxi_deg": np.degrees(dxi),
        "cos_similarity": sims,
        "cos_similarity_raw": sims_raw,
        "mean_chi": mean_chi,
    })

    # Headline statistics
    rho, _ = spearmanr(dxi, sims)
    rho_raw, _ = spearmanr(dxi, sims_raw)

    p_value = (
        _permutation_p_xi(xi, sims, iu, rho, n_permutations)
        if n_permutations > 0 else float("nan")
    )

    m = binned_df["mean_sim"].to_numpy()
    cosx = np.cos(binned_df["bin_center_rad"].to_numpy())
    a = float((m @ cosx) / (cosx @ cosx))
    r2 = float(1.0 - np.var(m - a * cosx) / np.var(m))

    deg = pairs_df["dxi_deg"]
    near = deg < 30.0
    far = deg > 150.0
    rho_near, _ = spearmanr(mean_chi[near], sims[near])
    rho_far, _ = spearmanr(mean_chi[far], sims[far])

    if write_csv:
        binned_df.to_csv(infra.RP.export_fp(f"hc__angular_similarity__{label}s__binned.csv"),
                         index=False)
        pairs_df.to_csv(infra.RP.export_fp(f"hc__angular_similarity__{label}s__pairs.csv"),
                        index=False)
        stats = pd.DataFrame([
            ("centered", center),
            ("n_occ", n),
            ("n_pairs", len(sims)),
            ("rho_spearman", rho),
            ("p_permutation", p_value),
            ("n_permutations", n_permutations),
            ("rho_spearman_raw", rho_raw),
            ("cos_fit_a", a),
            ("cos_fit_r2", r2),
            ("rho_chibar_near30", rho_near),
            ("rho_chibar_far150", rho_far),
            ("mean_sim_first_bin", binned_df["mean_sim"].iloc[0]),
            ("mean_sim_last_bin", binned_df["mean_sim"].iloc[-1]),
        ], columns=["stat", "value"])
        stats.to_csv(infra.RP.export_fp(f"hc__angular_similarity__{label}s__stats.csv"),
                     index=False)

    log(f"[Test 1] {label}s: {n} occs, {len(sims)} pairs, {n_bins} bins, "
        f"centered={center}, rho={rho:.3f} (raw {rho_raw:.3f}), "
        f"p_perm={p_value:.4g} ({n_permutations} draws), "
        f"cos-fit a={a:.3f} R2={r2:.3f}")

    return AngularSimilarityResult(
        label=label,
        n_occ=n,
        n_pairs=len(sims),
        centered=center,
        binned_df=binned_df,
        pairs_df=pairs_df,
        rho=float(rho),
        p_value=float(p_value),
        rho_raw=float(rho_raw),
        cos_fit_a=a,
        cos_fit_r2=r2,
        rho_chibar_near=float(rho_near),
        rho_chibar_far=float(rho_far),
    )


def plot_angular_similarity(
    skills_result: AngularSimilarityResult,
    abilities_result: AngularSimilarityResult,
    *,
    figsize: tuple[float, float] = (11, 4.2),
    show_cos_fit: bool = True,
    show: bool = True,
) -> dict:
    """Two-panel plot of HC similarity vs angular distance, Skills + Abilities.

    Shows the binned mean, a 20th-80th percentile band, and (optionally)
    the fitted a*cos(dxi) reference curve.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)

    centered = skills_result.centered
    ylabel = "Deviation-profile cosine similarity" if centered else "Cosine similarity"

    for ax, res, panel_title in zip(
        axes,
        [skills_result, abilities_result],
        [f"Skills ({skills_result.n_occ} occupations)",
         f"Abilities ({abilities_result.n_occ} occupations)"],
    ):
        x = res.binned_df["bin_center_deg"].to_numpy(float)
        y = res.binned_df["mean_sim"].to_numpy(float)

        ax.fill_between(x, res.binned_df["p20_sim"], res.binned_df["p80_sim"],
                        color="#3A6EA5", alpha=0.18, lw=0,
                        label="20th–80th percentile")
        if centered:
            ax.axhline(0.0, color="0.5", lw=0.8, ls="--")
        if show_cos_fit and centered:
            xx = np.linspace(0, 180, 200)
            ax.plot(xx, res.cos_fit_a * np.cos(np.radians(xx)), ls=":",
                    lw=1.2, color="0.25",
                    label=f"{res.cos_fit_a:.2f}·cos Δξ")
        ax.plot(x, y, "o-", lw=1.5, color="#3A6EA5", markersize=4, label="Mean")

        ax.set_xlabel("Angular distance Δξ (deg)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 180)
        ax.set_xticks([0, 45, 90, 135, 180])
        ax.grid(alpha=0.3)
        ax.set_title(f"{panel_title}\nSpearman ρ = {res.rho:.3f}", fontsize=10)
        ax.legend(fontsize=8, loc="upper right" if centered else "lower left",
                  frameon=False)

    suffix = " (deviations from the mean profile)" if centered else ""
    fig.suptitle(
        f"Capability similarity vs angular distance between occupations{suffix}",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()

    out_pdf = infra.RP.figure_fp("hc__angular_similarity.pdf")
    out_png = infra.RP.figure_fp("hc__angular_similarity.png")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    log(f"Saved figure: {out_pdf.name}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {"pdf": out_pdf, "png": out_png}