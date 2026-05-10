"""
profiles.py
-----------
Continuous angular profiles, profile clustering, and paper figures (Fig. 2 + Fig. 3).

The pipeline:

  1) build_continuous_profiles(overlay)
       von Mises kernel regression of descriptor values over xi.
       Returns ProfileSet with names, theta_grid, P (M x N matrix).

  2) cluster_profiles(profile_set, label)
       Cluster directional *trends* (cosine on circular first derivative).
       K is selected via subsampling-ARI stability with a fragmentation penalty.
       Returns ClusterResult with labels, bundles (median + 20/80 band),
       typicality, and a stable cluster ranking.

  3) plot_profiles_over_bands(profile_set, cluster_result, ...)
       Figure 2: individual profiles overlaid on cluster bands.

  4) plot_gradient_compass(skills_overlay, abilities_overlay, ...,
                           skills_clusters=None, abilities_clusters=None)
       Figure 3: directional maxima + R for both Skills and Abilities,
       cluster colors derived from each domain's clustering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

import infra
from infra import log
from plotutils import place_polar_labels_no_overlap_fast2

from overlays import OverlayResult


# ─────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────

VON_MISES_KAPPA = 16.0
N_THETA_PROFILE = 720
DEFAULT_USE_COL = "value_norm01_global"
DEFAULT_ZONE = "all"  # "all" | "inner" | "outer"

# Stability / clustering defaults
K_MIN = 2
K_MAX_DEFAULT = 20
MIN_CLUSTER_SIZE = 5
SUBSAMPLE_FRAC = 0.85
N_REPS = 40
STABILITY_SEED = 42

# Bundle band quantiles
BAND_QLO = 0.20
BAND_QHI = 0.80


# ─────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────

@dataclass
class ProfileSet:
    label: str                          # "skill" | "ability"
    use_col: str
    zone: str
    kappa: float
    theta_grid: np.ndarray              # (N,) radians in [0, 2*pi)
    names: list[str]                    # length M
    P: np.ndarray                       # (M, N) profile matrix


@dataclass
class ClusterResult:
    label: str
    K: int
    members_df: pd.DataFrame            # name, cluster, cluster_rank, typicality_cos
    bundle_df: pd.DataFrame             # cluster, cluster_rank, theta, deg, center, lo, hi
    cluster_meta: pd.DataFrame          # cluster, cluster_rank, n, center_median, band_*
    stability_df: pd.DataFrame
    typicality_df: pd.DataFrame
    cluster_order: list[int]            # cluster ids ordered by rank (1, 2, 3, ...)
    cluster_rank: dict[int, int]        # cluster id -> rank


# ─────────────────────────────────────────────────────────────
# 1) Build continuous angular profiles
# ─────────────────────────────────────────────────────────────

def _circ_kernel_regression(theta_samples: np.ndarray,
                            y: np.ndarray,
                            theta_grid: np.ndarray,
                            kappa: float) -> np.ndarray:
    """Von Mises kernel regression on the circle."""
    d = theta_grid[:, None] - theta_samples[None, :]
    K = np.exp(kappa * np.cos(d))
    num = np.nansum(K * y[None, :], axis=1)
    den = np.nansum(K, axis=1)
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def build_continuous_profiles(
    overlay: OverlayResult,
    *,
    use_col: str = DEFAULT_USE_COL,
    zone: str = DEFAULT_ZONE,
    kappa: float = VON_MISES_KAPPA,
    n_theta: int = N_THETA_PROFILE,
) -> ProfileSet:
    """Compute one continuous angular profile per descriptor by von Mises kernel regression."""
    label = overlay.label
    df = overlay.long_df

    needed = {"xi", "chi", label, use_col}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"long_df missing columns: {sorted(missing)}")

    # Zone filter on chi
    chi_med = float(np.nanmedian(df["chi"].to_numpy(float)))
    if zone == "outer":
        sub = df[df["chi"] >= chi_med].copy()
    elif zone == "inner":
        sub = df[df["chi"] < chi_med].copy()
    elif zone == "all":
        sub = df.copy()
    else:
        raise ValueError(f"Unknown zone: {zone!r}")

    names = sorted(sub[label].dropna().unique().tolist())
    theta_grid = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)

    rows = []
    for n in names:
        m = sub[sub[label] == n]
        if m.empty:
            continue
        y = m[use_col].to_numpy(float)
        t = np.mod(m["xi"].to_numpy(float), 2 * np.pi)
        prof = _circ_kernel_regression(t, y, theta_grid, kappa)
        rows.append((n, prof))

    # Sort by start value at theta=0 (descending) for stable display order
    rows.sort(key=lambda tup: float(tup[1][0]), reverse=True)
    final_names = [n for (n, _) in rows]
    P = np.vstack([prof for (_, prof) in rows]).astype(float)

    return ProfileSet(
        label=label,
        use_col=use_col,
        zone=zone,
        kappa=kappa,
        theta_grid=theta_grid,
        names=final_names,
        P=P,
    )


# ─────────────────────────────────────────────────────────────
# 2) Cluster profiles by directional trend
# ─────────────────────────────────────────────────────────────

def _zscore_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mu = np.nanmean(X, axis=1, keepdims=True)
    sd = np.nanstd(X, axis=1, keepdims=True)
    return (X - mu) / np.maximum(sd, eps)


def _circular_derivative(P: np.ndarray, n_theta: int, smooth_win: int = 9) -> np.ndarray:
    """Central-difference derivative on the circle, with optional moving-average smoothing."""
    dtheta = 2 * np.pi / n_theta
    D = (np.roll(P, -1, axis=1) - np.roll(P, +1, axis=1)) / (2.0 * dtheta)
    if smooth_win and smooth_win >= 3:
        k = smooth_win // 2
        D = np.nanmean(
            np.stack([np.roll(D, s, axis=1) for s in range(-k, k + 1)], axis=0),
            axis=0,
        )
    return D


def _cosine_sim_rows_to_vec(Xrows: np.ndarray, v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    num = Xrows @ v
    den = np.linalg.norm(Xrows, axis=1) * np.linalg.norm(v) + eps
    return num / den


def cluster_profiles(
    profile_set: ProfileSet,
    *,
    k_min: int = K_MIN,
    k_max: int = K_MAX_DEFAULT,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    subsample_frac: float = SUBSAMPLE_FRAC,
    n_reps: int = N_REPS,
    seed: int = STABILITY_SEED,
    band_qlo: float = BAND_QLO,
    band_qhi: float = BAND_QHI,
    write_csv: bool = True,
) -> ClusterResult:
    """Cluster directional trend (derivative of profile, cosine), select K via subsampling ARI."""
    label = profile_set.label
    P = profile_set.P
    M, N = P.shape
    if M < 2:
        raise RuntimeError(f"Need at least 2 profiles, got {M}")

    names = profile_set.names

    D = _circular_derivative(P, n_theta=N, smooth_win=9)
    X = np.nan_to_num(_zscore_rows(D), nan=0.0, posinf=0.0, neginf=0.0)

    # Feasible K range given min cluster size
    k_max_feas = min(k_max, M - 1, M // max(min_cluster_size, 1))
    if k_max_feas < k_min:
        # Relax min_cluster_size automatically
        min_cluster_size = max(2, min(M // k_min, min_cluster_size))
        k_max_feas = min(k_max, M - 1, M // min_cluster_size)
    if k_max_feas < k_min:
        raise RuntimeError(f"Too few profiles for stability selection: M={M}")

    rng = np.random.default_rng(seed)
    idx_all = np.arange(M)

    def _fit(indices: np.ndarray, k: int) -> np.ndarray:
        model = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
        return model.fit_predict(X[indices])

    stability_rows: list[tuple] = []
    best = {"k": None, "score": -np.inf, "ari_mean": None, "ari_std": None, "frag": None}

    for k in range(k_min, k_max_feas + 1):
        ref = _fit(idx_all, k)
        ref_counts = np.bincount(ref, minlength=k)
        if ref_counts.min() < min_cluster_size:
            continue

        aris: list[float] = []
        sub_n = int(np.floor(subsample_frac * M))
        for _ in range(n_reps):
            sub_idx = rng.choice(idx_all, size=sub_n, replace=False)
            if sub_idx.size < max(10, int(0.2 * M)):
                continue
            sub_lab = _fit(sub_idx, k)
            aris.append(adjusted_rand_score(ref[sub_idx], sub_lab))

        if len(aris) < max(10, n_reps // 3):
            continue

        ari_mean = float(np.mean(aris))
        ari_std = float(np.std(aris))
        frag = float(ref_counts.min() / max(min_cluster_size, 1))
        score = ari_mean - 0.05 * ari_std + 0.02 * np.log(frag)

        stability_rows.append((k, ari_mean, ari_std, frag, len(aris), score))
        if score > best["score"]:
            best.update({"k": k, "score": score, "ari_mean": ari_mean, "ari_std": ari_std, "frag": frag})

    if best["k"] is None:
        raise RuntimeError("No stable K found. Lower min_cluster_size or relax constraints.")

    K = int(best["k"])
    log(f"[{label}] selected K={K}  ARI_mean={best['ari_mean']:.3f}  std={best['ari_std']:.3f}  frag={best['frag']:.2f}")

    stability_df = pd.DataFrame(
        stability_rows,
        columns=["k", "ari_mean", "ari_std", "frag_min_over_req", "n_valid", "score"],
    ).sort_values("k").reset_index(drop=True)

    # Final fit on full data
    labels_arr = AgglomerativeClustering(
        n_clusters=K, metric="cosine", linkage="average"
    ).fit_predict(X)

    members_df = pd.DataFrame({"name": names, "cluster": labels_arr})

    # Bundles + meta
    bundle_parts: list[pd.DataFrame] = []
    cluster_rows: list[tuple] = []
    xdeg = np.degrees(profile_set.theta_grid)

    for c in range(K):
        idx = np.where(labels_arr == c)[0]
        if idx.size == 0:
            continue
        Pc = P[idx, :]
        center = np.nanmedian(Pc, axis=0)
        lo = np.nanquantile(Pc, band_qlo, axis=0)
        hi = np.nanquantile(Pc, band_qhi, axis=0)
        sort_key = float(np.nanmedian(center))
        band = hi - lo
        cluster_rows.append((
            c,
            sort_key,
            int(idx.size),
            float(np.nanmean(band)),
            float(np.nanquantile(band, 0.90)),
            float(np.nanmax(band)),
        ))
        bundle_parts.append(pd.DataFrame({
            "cluster": c,
            "theta": profile_set.theta_grid.astype(float),
            "deg": xdeg.astype(float),
            "center": center.astype(float),
            "lo": lo.astype(float),
            "hi": hi.astype(float),
        }))

    bundle_df = pd.concat(bundle_parts, ignore_index=True)

    cluster_order_full = sorted(cluster_rows, key=lambda t: t[1], reverse=True)
    cluster_rank = {c: i + 1 for i, (c, *_rest) in enumerate(cluster_order_full)}
    cluster_order = [c for (c, *_rest) in cluster_order_full]

    bundle_df["cluster_rank"] = bundle_df["cluster"].map(cluster_rank).astype(int)
    members_df["cluster_rank"] = members_df["cluster"].map(cluster_rank).astype(int)

    cluster_meta = pd.DataFrame(
        cluster_order_full,
        columns=["cluster", "center_median", "n", "band_mean", "band_p90", "band_max"],
    )
    cluster_meta["cluster_rank"] = cluster_meta["cluster"].map(cluster_rank).astype(int)
    cluster_meta = cluster_meta.sort_values("cluster_rank").reset_index(drop=True)

    # Typicality
    typ_rows: list[tuple] = []
    for c in cluster_order:
        idx = np.where(labels_arr == c)[0]
        if idx.size == 0:
            continue
        v = np.nan_to_num(np.nanmedian(X[idx, :], axis=0), nan=0.0)
        sims = _cosine_sim_rows_to_vec(X[idx, :], v)
        for (i_local, sim) in zip(idx, sims):
            typ_rows.append((names[i_local], c, cluster_rank[c], float(sim)))
    typicality_df = pd.DataFrame(
        typ_rows, columns=["name", "cluster", "cluster_rank", "typicality_cos"]
    )

    members_df = members_df.merge(
        typicality_df[["name", "typicality_cos"]], on="name", how="left"
    ).sort_values(["cluster_rank", "name"]).reset_index(drop=True)

    if write_csv:
        prefix = f"{label}s_profile_clusters__{profile_set.zone}__{profile_set.use_col}__K{K}"
        members_df.to_csv(infra.RP.export_fp(f"{prefix}__members.csv"), index=False)
        bundle_df.to_csv(infra.RP.export_fp(f"{prefix}__bundles.csv"), index=False)
        cluster_meta.to_csv(infra.RP.export_fp(f"{prefix}__meta.csv"), index=False)
        typicality_df.to_csv(infra.RP.export_fp(f"{prefix}__typicality.csv"), index=False)
        stability_df.to_csv(
            infra.RP.export_fp(
                f"{label}s_profile_clusters__{profile_set.zone}__{profile_set.use_col}__stability_ari.csv"
            ),
            index=False,
        )

    return ClusterResult(
        label=label,
        K=K,
        members_df=members_df,
        bundle_df=bundle_df,
        cluster_meta=cluster_meta,
        stability_df=stability_df,
        typicality_df=typicality_df,
        cluster_order=cluster_order,
        cluster_rank=cluster_rank,
    )


# ─────────────────────────────────────────────────────────────
# 3) Figure 2 — profiles over bands
# ─────────────────────────────────────────────────────────────

# Discrete band palette (blue + brown family). Used for both labels.
_BAND_PALETTE = [
    "#8B5A2B", "#3A6EA5", "#6B4F3A", "#4C566A",
    "#7C6F64", "#2F4F4F", "#5E4B3C", "#3E5870",
    "#7A5A3A", "#334B5E", "#6A5C50", "#3F4E5A",
]
_AX_GREY = "#9AA0A6"


def _band_prefix_for_label(label: str) -> str:
    if label == "skill":
        return "S"
    if label == "ability":
        return "A"
    return label[:1].upper()


def _labelname(label: str) -> str:
    if label == "skill":
        return "Skills"
    if label == "ability":
        return "Abilities"
    return f"{label}s"


def plot_profiles_over_bands(
    profile_set: ProfileSet,
    cluster_result: ClusterResult,
    *,
    figsize: tuple[float, float] = (12, 7),
    band_alpha: float = 0.25,
    band_line_alpha: float = 0.65,
    band_line_lw: float = 3.0,
    line_alpha: float = 0.45,
    line_lw: float = 0.9,
    annotate: bool = True,
    annotate_fontsize: int = 7,
    annotate_jitter_deg: float = 8.0,
    show: bool = True,
) -> dict:
    """Figure 2: individual profiles overlaid on cluster bands.

    Returns a dict with the saved file paths.
    """
    label = profile_set.label
    if label != cluster_result.label:
        raise ValueError(f"Label mismatch: profile_set={label!r}, cluster_result={cluster_result.label!r}")

    xdeg = np.degrees(profile_set.theta_grid)
    P = profile_set.P
    bundle_df = cluster_result.bundle_df
    cluster_order = cluster_result.cluster_order
    cluster_rank = cluster_result.cluster_rank

    band_color = {c: _BAND_PALETTE[i % len(_BAND_PALETTE)] for i, c in enumerate(cluster_order)}
    band_prefix = _band_prefix_for_label(label)
    labelname = _labelname(label)

    fig, ax = plt.subplots(figsize=figsize)

    # Bands
    for c in cluster_order:
        cr = int(cluster_rank[c])
        b = bundle_df[bundle_df["cluster"] == c].sort_values("deg")
        xx = b["deg"].to_numpy(float)
        center = b["center"].to_numpy(float)
        lo = b["lo"].to_numpy(float)
        hi = b["hi"].to_numpy(float)
        col = band_color[c]
        ax.fill_between(xx, lo, hi, alpha=band_alpha, linewidth=0, color=col)
        ax.plot(xx, center, lw=band_line_lw, alpha=band_line_alpha, color=col)
        # Label near 20 deg
        j = int(np.argmin(np.abs(xx - 20.0)))
        ax.text(xx[j] + 2.0, center[j], f"{band_prefix}{cr}",
                color=col, fontsize=24, va="center", ha="left", alpha=0.95)

    # Individual profiles
    lines = []
    for i in range(P.shape[0]):
        (ln,) = ax.plot(xdeg, P[i], lw=line_lw, alpha=line_alpha)
        lines.append(ln)

    # Numbering
    if annotate:
        rng = np.random.default_rng(42)
        for i, ln in enumerate(lines):
            y0 = float(P[i, 0])
            y_span = float(np.nanmax(P[i]) - np.nanmin(P[i]) + 1e-12)
            yj = float(rng.uniform(-1, 1)) * 0.01 * y_span
            col = ln.get_color()
            if i % 2 == 0:
                x0 = -float(rng.uniform(0.5, 1.5) * annotate_jitter_deg)
                ha = "right"
            else:
                x0 = 360.0 + float(rng.uniform(0.5, 1.5) * annotate_jitter_deg)
                ha = "left"
            ax.text(x0, y0 + yj, str(i + 1), ha=ha, va="center",
                    fontsize=annotate_fontsize, color=col)

    ax.set_title(
        f"O*NET {labelname} -- individual angular profiles over clustered bands",
        fontsize=13, color=_AX_GREY,
    )
    ax.set_xlabel("Direction (deg)", color=_AX_GREY)
    ax.set_ylabel(f"Normalized {label} value", color=_AX_GREY)
    ax.set_xticks(np.arange(0, 360, 45))
    ax.grid(alpha=0.22)
    for spine in ax.spines.values():
        spine.set_color(_AX_GREY)
    ax.tick_params(axis="both", colors=_AX_GREY)
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    out_base = f"{label}s_profiles_over_bands__{profile_set.zone}__{profile_set.use_col}"
    out_png = infra.RP.figure_fp(f"{out_base}.png")
    out_pdf = infra.RP.figure_fp(f"{out_base}.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    log(f"Saved figure: {out_pdf.name}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {"png": out_png, "pdf": out_pdf}


# ─────────────────────────────────────────────────────────────
# 4) Figure 3 — gradient compass
# ─────────────────────────────────────────────────────────────

# Cluster colors keyed by (label, rank). Pole 1 = teal family, Pole 2 = orange-red.
# Rank 1 -> Pole 1, rank 2 -> Pole 2, etc. With K=2 per domain we get the pairing
# S1/A1 (teal) and S2/A2 (orange-red).
_POLE_COLORS = {
    1: "#2a9d8f",  # Pole 1 (Social/Cognitive)
    2: "#e76f51",  # Pole 2 (Technical/Physical)
    3: "#9C6F44",
    4: "#5E4B3C",
}
_MARKERS = {"skill": "o", "ability": "^"}
_SECTOR_LABELS = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
_SECTOR_LABEL_R = 1.16   # radial position of N/E/S/W and corner labels


def _sector_label(theta_deg: float) -> str:
    d = float(theta_deg) % 360.0
    return _SECTOR_LABELS[int((d + 22.5) // 45) % 8]


def _build_compass_df(
    overlay: OverlayResult,
    cluster_result: ClusterResult | None,
) -> pd.DataFrame:
    """Combine overlay angles with cluster ranks per descriptor."""
    df = overlay.angles_df[[overlay.label, "theta_max_deg", "resultant_R"]].copy()
    df = df.rename(columns={overlay.label: "name"})
    df["theta_max_deg"] = pd.to_numeric(df["theta_max_deg"], errors="coerce")
    df["resultant_R"] = pd.to_numeric(df["resultant_R"], errors="coerce")
    df = df.dropna(subset=["theta_max_deg", "resultant_R"]).reset_index(drop=True)
    df["label"] = overlay.label
    df["type"] = "Skill" if overlay.label == "skill" else "Ability"

    if cluster_result is None:
        df["cluster_rank"] = np.nan
    else:
        rank_by_name = (
            cluster_result.members_df.set_index("name")["cluster_rank"].to_dict()
        )
        df["cluster_rank"] = df["name"].map(rank_by_name)

    return df


def plot_gradient_compass(
    skills_overlay: OverlayResult,
    abilities_overlay: OverlayResult,
    *,
    skills_clusters: ClusterResult | None = None,
    abilities_clusters: ClusterResult | None = None,
    figsize: tuple[float, float] = (14, 11),
    show: bool = True,
) -> dict:
    """Figure 3: directional maxima + R for Skills and Abilities, colored by cluster rank."""
    if skills_overlay.label != "skill" or abilities_overlay.label != "ability":
        raise ValueError("Expected skills_overlay.label == 'skill' and abilities_overlay.label == 'ability'.")

    df_s = _build_compass_df(skills_overlay, skills_clusters)
    df_a = _build_compass_df(abilities_overlay, abilities_clusters)
    df = pd.concat([df_s, df_a], ignore_index=True)

    # Map cluster_rank -> color (per descriptor)
    df["color"] = df["cluster_rank"].map(_POLE_COLORS).fillna("#888888")

    # Save membership csv (paper-relevant: pole assignment)
    mem_fp = infra.RP.export_fp("gradient_compass_clusters_membership.csv")
    cols = ["type", "label", "name", "theta_max_deg", "resultant_R", "cluster_rank"]
    df[cols].sort_values(["type", "cluster_rank", "resultant_R"],
                        ascending=[True, True, False]).to_csv(mem_fp, index=False)
    log(f"Saved compass membership: {mem_fp.name}")

    # Plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    ax.set_position([0.0, 0.04, 1.0, 0.96])

    # Sector labels (E, NE, N, ...) — placed at _SECTOR_LABEL_R below
    for ang in range(0, 360, 45):
        ax.text(np.deg2rad(ang), _SECTOR_LABEL_R, _sector_label(ang),
                ha="center", va="center", fontsize=11, color="gray", alpha=0.55)

    # Scatter points
    for label_typ in ("skill", "ability"):
        sub = df[df["label"] == label_typ]
        if sub.empty:
            continue
        is_skill = label_typ == "skill"
        marker = _MARKERS[label_typ]
        size = 26 if is_skill else 34
        edge = (0, 0, 0, 0.55) if is_skill else (0, 0, 0, 0.85)
        lw = 0.6 if is_skill else 0.9
        ax.scatter(
            np.deg2rad(sub["theta_max_deg"].to_numpy(float)),
            sub["resultant_R"].to_numpy(float),
            s=size, c=sub["color"].tolist(), marker=marker,
            alpha=0.90, linewidths=lw, edgecolors=edge, zorder=3,
        )

    # Labels
    place_polar_labels_no_overlap_fast2(
        ax,
        theta_list=np.deg2rad(df["theta_max_deg"].to_numpy(float)).tolist(),
        chi_list=df["resultant_R"].to_numpy(float).tolist(),
        label_list=df["name"].tolist(),
        base_offset=0.45, fontsize=9, min_sep_px=10, max_iter=180,
        step_r_px=7.0, step_t_px=4.0, init_jitter_px=8.0,
        outside_push_px=10.0, tangential_nudge_px=3.0,
        collision_jitter=0.35, safety_iters=18,
        leader_lw=0.7, leader_alpha=0.35, leader_curved=True,
        keep_inside_axes=False, force_outward=True, expand_rmax_if_needed=True,
        text_color=df["color"].tolist(), leader_color=df["color"].tolist(),
        text_alpha=0.95,
    )

    for t in ax.texts:
        t.set_clip_on(False)
    for ln in ax.lines:
        ln.set_clip_on(False)

    # Custom perimeter at r=1
    rmax = 1.70
    ax.set_ylim(0, rmax)
    ax.spines["polar"].set_visible(False)
    ax.patch.set_edgecolor("none")
    tt = np.linspace(0, 2 * np.pi, 720)
    ax.plot(tt, np.full_like(tt, 1.0),
            color="lightgray", alpha=0.95, linewidth=1.2, zorder=2)

    ax.set_yticks([])
    # We do NOT use ax.grid(True) for the radial spokes because matplotlib's
    # polar grid lines extend all the way to ymax (1.70), which makes them
    # shoot out beyond the unit circle. Instead, draw spokes manually only
    # out to r = 1.0.
    ax.grid(False)
    _SPOKE_ANGLES = np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315])
    for ang in _SPOKE_ANGLES:
        ax.plot([ang, ang], [0.0, 1.0],
                color="lightgray", alpha=0.55,
                linestyle="-", linewidth=0.8, zorder=1)
    ax.set_xticks(_SPOKE_ANGLES)
    # Hide auto-placed tick labels and tick lines. We place degree labels
    # manually so we can control their radius.
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0)
    # Shared radial layout for axis annotations:
    #   r = 1.00   unit circle (drawn above)
    #   r = 1.03   degree labels (0°, 45°, ...)
    #   r = 1.13   sector labels (E, NE, N, ...)
    _DEG_LABEL_R = 1.06
    for ang_deg in [0, 45, 90, 135, 180, 225, 270, 315]:
        ax.text(np.deg2rad(ang_deg), _DEG_LABEL_R, f"{ang_deg}°",
                ha="center", va="center",
                fontsize=10, color="gray", alpha=0.55)

    # Legend
    pole1 = _POLE_COLORS[1]
    pole2 = _POLE_COLORS[2]
    legend_elements = [
        Line2D([0], [0], marker="o", linestyle="none", color="none",
               markerfacecolor="gray", markeredgecolor=(0, 0, 0, 0.6),
               markeredgewidth=0.8, markersize=8, label="Skills (marker)"),
        Line2D([0], [0], marker="^", linestyle="none", color="none",
               markerfacecolor="gray", markeredgecolor=(0, 0, 0, 0.85),
               markeredgewidth=1.0, markersize=9, label="Abilities (marker)"),
        Line2D([0], [0], color=pole1, lw=5, label="Pole 1: Social/Cognitive"),
        Line2D([0], [0], color=pole2, lw=5, label="Pole 2: Technical/Physical"),
        Line2D([0], [0], marker="o", linestyle="none", color="none",
               markerfacecolor=pole1, markeredgecolor=(0, 0, 0, 0.6),
               markeredgewidth=0.8, markersize=8, label="S1 (Skills, Pole 1)"),
        Line2D([0], [0], marker="^", linestyle="none", color="none",
               markerfacecolor=pole1, markeredgecolor=(0, 0, 0, 0.85),
               markeredgewidth=1.0, markersize=9, label="A1 (Abilities, Pole 1)"),
        Line2D([0], [0], marker="o", linestyle="none", color="none",
               markerfacecolor=pole2, markeredgecolor=(0, 0, 0, 0.6),
               markeredgewidth=0.8, markersize=8, label="S2 (Skills, Pole 2)"),
        Line2D([0], [0], marker="^", linestyle="none", color="none",
               markerfacecolor=pole2, markeredgecolor=(0, 0, 0, 0.85),
               markeredgewidth=1.0, markersize=9, label="A2 (Abilities, Pole 2)"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper center", bbox_to_anchor=(0.5, 0.08),
        ncol=2, fontsize=10, frameon=True,
        title="Clusters and functional poles",
        borderaxespad=0.0, handletextpad=0.6, columnspacing=1.2,
    )

    out_pdf = infra.RP.figure_fp("master_gradient_compass_4clusters.pdf")
    out_png = infra.RP.figure_fp("master_gradient_compass_4clusters.png")
    plt.savefig(out_pdf, dpi=400, bbox_inches="tight")
    plt.savefig(out_png, dpi=400, bbox_inches="tight")
    log(f"Saved figure: {out_pdf.name}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {"pdf": out_pdf, "png": out_png, "membership": mem_fp}