"""
hc_analysis.py
--------------
Test 1 — Angular neighbourhood -> HC similarity.

    "If the angular coordinate captures domain identity in a meaningful sense,
     occupations in nearby directions should draw on similar human capital."

    For all pairs of occupations: cosine similarity over the descriptor vector
    (skills or abilities, separately) plotted against circular angular distance.

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


# ─────────────────────────────────────────────────────────────
# Test 1 — Angular neighbourhood vs HC similarity
# ─────────────────────────────────────────────────────────────

@dataclass
class AngularSimilarityResult:
    label: str
    n_occ: int
    n_pairs: int
    binned_df: pd.DataFrame   # bin_center_deg, mean_sim, n_pairs
    pairs_df: pd.DataFrame    # full pair-level data (xi_i, xi_j, dxi, sim)


def compute_angular_similarity(
    overlay: OverlayResult,
    *,
    n_bins: int = 30,
    write_csv: bool = True,
) -> AngularSimilarityResult:
    """Compute mean HC similarity binned by circular angular distance."""
    label = overlay.label
    _, V, xi, _, _ = _descriptor_matrix(overlay)
    n = V.shape[0]
    if n < 3:
        raise RuntimeError(f"Too few occupations: {n}")

    S = _cosine_similarity_pairs(V)
    iu = np.triu_indices(n, k=1)
    sims = S[iu]
    dxi = _circular_distance(xi[iu[0]], xi[iu[1]])

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
            "n_pairs": int(mask.sum()),
        })
    binned_df = pd.DataFrame(rows)

    pairs_df = pd.DataFrame({
        "i": iu[0],
        "j": iu[1],
        "dxi_rad": dxi,
        "dxi_deg": np.degrees(dxi),
        "cos_similarity": sims,
    })

    if write_csv:
        binned_df.to_csv(infra.RP.export_fp(f"hc_test1__angular_similarity__{label}s__binned.csv"),
                         index=False)
        pairs_df.to_csv(infra.RP.export_fp(f"hc_test1__angular_similarity__{label}s__pairs.csv"),
                        index=False)

    log(f"[Test 1] {label}s: {n} occs, {len(sims)} pairs, {n_bins} bins")

    return AngularSimilarityResult(
        label=label,
        n_occ=n,
        n_pairs=len(sims),
        binned_df=binned_df,
        pairs_df=pairs_df,
    )


def plot_test1_angular_similarity(
    skills_result: AngularSimilarityResult,
    abilities_result: AngularSimilarityResult,
    *,
    figsize: tuple[float, float] = (11, 4.2),
    show: bool = True,
) -> dict:
    """Two-panel plot of mean HC similarity vs angular distance, Skills + Abilities."""
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)

    for ax, res, panel_title in zip(
        axes,
        [skills_result, abilities_result],
        [f"Skills ({skills_result.n_occ} occupations)",
         f"Abilities ({abilities_result.n_occ} occupations)"],
    ):
        x = res.binned_df["bin_center_deg"].to_numpy(float)
        y = res.binned_df["mean_sim"].to_numpy(float)

        rho, p = spearmanr(res.pairs_df["dxi_rad"], res.pairs_df["cos_similarity"])

        ax.plot(x, y, "o-", lw=1.5, color="#3A6EA5", markersize=4)
        ax.set_xlabel("Angular distance Δξ (deg)")
        ax.set_ylabel("Mean cosine similarity")
        ax.set_xlim(0, 180)
        ax.set_xticks([0, 45, 90, 135, 180])
        ax.grid(alpha=0.3)
        ax.set_title(f"{panel_title}\nSpearman ρ = {rho:.3f}", fontsize=10)

    fig.suptitle(
        "Test 1: HC similarity vs angular distance between occupations",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()

    out_pdf = infra.RP.figure_fp("hc_test1__angular_similarity.pdf")
    out_png = infra.RP.figure_fp("hc_test1__angular_similarity.png")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    log(f"Saved figure: {out_pdf.name}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {"pdf": out_pdf, "png": out_png}
