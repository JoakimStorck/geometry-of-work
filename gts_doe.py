"""
gts_doe.py
----------
Design of Experiments for GTS technology scenarios.

Builds factorial or custom parameter grids, runs all combinations via
gts_cache (so results are cached and re-used), and returns tidy DataFrames
for analysis and plotting.

Typical usage
-------------
    import gts_doe as doe

    # Factorial grid over z and A
    grid = doe.factorial(
        xi_deg = 40.6,
        chi    = 0.35,
        R      = 0.60,
        gamma  = 0.3,
        z      = [0.15, 0.20, 0.30, 0.40, 0.50],
        A      = [0.5, 1.0, 2.0, 4.0],
    )

    results = doe.run(grid)          # runs via gts_cache
    df      = doe.to_dataframe(results)
    doe.heatmap(df, row='z', col='A', metric='op_share_total')
    doe.heatmap(df, row='z', col='A', metric='Delta_Gamma_D')
    doe.sector_heatmap(df, row='z', col='A', sector='W (manual)')
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import gts_cache

# ---------------------------------------------------------------------------
# Sector assignment (mirrors gts_analysis_cell logic)
# ---------------------------------------------------------------------------

def _sektor(p_xy):
    xi = math.degrees(math.atan2(p_xy[1], p_xy[0])) % 360
    if xi < 45 or xi >= 315: return 'E'
    elif xi < 135:            return 'N'
    elif xi < 225:            return 'W'
    else:                     return 'S'


# ---------------------------------------------------------------------------
# Grid builders
# ---------------------------------------------------------------------------

def factorial(xi_deg: float, chi: float, R: float, gamma: float,
              z: list, A: list,
              name_prefix: str = "DOE") -> list[dict]:
    """
    Full factorial grid over z and A with all other parameters fixed.

    Returns a list of scenario dicts compatible with gts.run_scenario.
    """
    scenarios = []
    for z_val, A_val in itertools.product(z, A):
        scenarios.append(dict(
            name   = f"{name_prefix}  z={z_val:.2f}  A={A_val:.1f}",
            xi_deg = xi_deg,
            chi    = chi,
            z      = z_val,
            A      = A_val,
            R      = R,
            gamma  = gamma,
            plot   = dict(show_allocation=True, show_tech_effect=True),
        ))
    return scenarios


def custom(base: dict, **sweeps) -> list[dict]:
    """
    Sweep one or more parameters over lists of values, holding the rest fixed.

    Example
    -------
        scenarios = doe.custom(
            base   = dict(xi_deg=40.6, chi=0.35, z=0.30, A=2.0, R=0.60, gamma=0.3),
            z      = [0.15, 0.30, 0.50],
            gamma  = [0.0, 0.3, 0.6],
        )
    """
    keys   = list(sweeps.keys())
    values = [sweeps[k] for k in keys]
    scenarios = []
    for combo in itertools.product(*values):
        cfg = dict(base)
        for k, v in zip(keys, combo):
            cfg[k] = v
        parts = "  ".join(f"{k}={v}" for k, v in zip(keys, combo))
        cfg['name'] = cfg.get('name', 'DOE') + f"  {parts}"
        cfg.setdefault('plot', dict(show_allocation=True, show_tech_effect=True))
        scenarios.append(cfg)
    return scenarios


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(scenarios: list[dict], run_fn=None) -> list[dict]:
    """
    Run all scenarios via gts_cache.

    Parameters
    ----------
    scenarios : list of scenario dicts
    run_fn    : callable, default gts.run_scenario (imported lazily)

    Returns
    -------
    list of (scenario, result) tuples
    """
    if run_fn is None:
        import gts_core as gts
        run_fn = gts.run_scenario

    pairs = []
    for cfg in scenarios:
        res = gts_cache.run(cfg, run_fn)
        pairs.append((cfg, res))
    return pairs


# ---------------------------------------------------------------------------
# Aggregate to DataFrame
# ---------------------------------------------------------------------------

def to_dataframe(pairs: list[tuple], include_sector: bool = True) -> pd.DataFrame:
    """
    Convert (scenario, result) pairs to a tidy DataFrame.

    Columns
    -------
    name, xi_deg, chi, z, A, R, gamma,
    op_share_total, op_share_total_g0,
    Delta_Gamma_D, gamma_eff, new_task_mass,
    <family>_dw, <family>_net  (per job family)
    [sector_<s>_mean_dw, sector_<s>_mean_net]  (if include_sector)
    """
    import gts_core as gts

    fam_names = gts.fam_names()
    fam_init  = gts.families_init()
    geo       = gts.geo_polar()

    rows = []
    for cfg, res in pairs:
        omega_map = {g: fam_init[g].get('omega', 1.0) for g in fam_names}

        row = dict(
            name          = cfg['name'],
            xi_deg        = cfg['xi_deg'],
            chi           = cfg['chi'],
            z             = cfg['z'],
            A             = cfg['A'],
            R             = cfg['R'],
            gamma         = cfg['gamma'],
            op_share_total     = sum(res['op_share'][g]    for g in fam_names),
            op_share_total_g0  = sum(res['op_share_g0'][g] for g in fam_names),
            Delta_Gamma_D      = res.get('Delta_Gamma_D', float('nan')),
            gamma_eff          = res.get('gamma_eff', 0.0),
            new_task_mass      = res.get('gamma_eff', 0.0) * res.get('Delta_Gamma_D', 0.0),
        )

        # Per-family wage changes (observed: omega * w)
        for g in fam_names:
            om   = omega_map[g]
            w0   = res['w_base'][g]  * om
            w1   = res['w_post'][g]  * om
            w_g0 = res['w_post_g0'][g] * om
            row[f'{g}__dw']  = 100 * (w1   - w0) / w0
            row[f'{g}__net'] = 100 * (w1   - w_g0) / w0   # reinstatement net
            row[f'{g}__dw_g0'] = 100 * (w_g0 - w0) / w0

        # Sector aggregates
        if include_sector:
            for sek in ('E', 'N', 'W', 'S'):
                members = [g for g in fam_names
                           if _sektor(geo[g]['p_xy']) == sek]
                if members:
                    row[f'sek_{sek}_mean_dw']  = np.mean([row[f'{g}__dw']  for g in members])
                    row[f'sek_{sek}_mean_net'] = np.mean([row[f'{g}__net'] for g in members])
                    row[f'sek_{sek}_mean_dw_g0'] = np.mean([row[f'{g}__dw_g0'] for g in members])

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def heatmap(df: pd.DataFrame, row: str, col: str, metric: str,
            title: str | None = None,
            fmt: str = '.2f',
            cmap: str = 'RdYlGn',
            center: float | None = 0.0,
            save_path=None, dpi: int = 130):
    """
    Pivot heatmap of a scalar metric over two DOE axes.

    Parameters
    ----------
    row, col : str   column names in df (e.g. 'z', 'A')
    metric   : str   column to display (e.g. 'op_share_total', 'Delta_Gamma_D',
                     'sek_W_mean_net')
    """
    pivot = df.pivot(index=row, columns=col, values=metric)
    pivot = pivot.sort_index(ascending=False)

    fig, ax = plt.subplots(figsize=(max(5, len(pivot.columns)*1.2),
                                    max(4, len(pivot)*1.0)), dpi=dpi)

    if center is not None:
        norm = mcolors.TwoSlopeNorm(vmin=pivot.values.min(),
                                    vcenter=center,
                                    vmax=pivot.values.max())
    else:
        norm = None

    im = ax.imshow(pivot.values, cmap=cmap, norm=norm, aspect='auto')

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels([f"{col}={v}" for v in pivot.columns], fontsize=9)
    ax.set_yticklabels([f"{row}={v}" for v in pivot.index],   fontsize=9)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, format(val, fmt),
                    ha='center', va='center', fontsize=9,
                    color='white' if abs(val) > 0.6 * abs(pivot.values).max() else 'black')

    ax.set_xlabel(col, fontsize=10)
    ax.set_ylabel(row, fontsize=10)
    _title = title or f"{metric}  ({row} × {col})"
    ax.set_title(_title, fontsize=11, fontweight='bold', pad=10)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=dpi)
        print(f"Saved: {save_path}")
    plt.show()


def sector_heatmap(df: pd.DataFrame, row: str, col: str,
                   metric: str = 'mean_net',
                   sectors: tuple = ('E', 'N', 'W', 'S'),
                   save_path=None, dpi: int = 130):
    """
    2×2 (or 1×4) grid of heatmaps — one per sector.

    metric: 'mean_net', 'mean_dw', or 'mean_dw_g0'
    """
    n = len(sectors)
    ncols = min(n, 2)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 5.5, nrows * 4.5), dpi=dpi)
    axes = np.array(axes).flatten()

    SEKTOR_LABEL = {'E': 'E (cognitive)', 'N': 'N (investigative)',
                    'W': 'W (manual)',    'S': 'S (service)'}
    SEKTOR_COLOR = {'E': 'RdYlBu', 'N': 'RdYlGn',
                    'W': 'RdYlGn', 'S': 'PuOr'}

    for ax, sek in zip(axes, sectors):
        col_name = f"sek_{sek}_{metric}"
        if col_name not in df.columns:
            ax.set_visible(False)
            continue

        pivot = df.pivot(index=row, columns=col, values=col_name)
        pivot = pivot.sort_index(ascending=False)

        vmax = max(abs(pivot.values.min()), abs(pivot.values.max()))
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        im = ax.imshow(pivot.values, cmap=SEKTOR_COLOR[sek],
                       norm=norm, aspect='auto')

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_xticklabels([f"{v}" for v in pivot.columns], fontsize=8)
        ax.set_yticklabels([f"{v}" for v in pivot.index],   fontsize=8)
        ax.set_xlabel(col, fontsize=9)
        ax.set_ylabel(row, fontsize=9)
        ax.set_title(SEKTOR_LABEL[sek], fontsize=10, fontweight='bold')

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                ax.text(j, i, f"{val:+.1f}%",
                        ha='center', va='center', fontsize=8.5,
                        color='white' if abs(val) > 0.6*vmax else 'black')

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[len(sectors):]:
        ax.set_visible(False)

    metric_label = {'mean_net': 'Netto reinstatement (%)',
                    'mean_dw':  'Löneförändring γ>0 (%)',
                    'mean_dw_g0': 'Löneförändring γ=0 (%)'}
    fig.suptitle(f"Sektorutfall: {metric_label.get(metric, metric)}\n"
                 f"per {row} × {col}",
                 fontsize=12, fontweight='bold', y=1.02)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=dpi)
        print(f"Saved: {save_path}")
    plt.show()


def lineplot(df: pd.DataFrame, x: str, hue: str, metric: str,
             title: str | None = None,
             save_path=None, dpi: int = 130):
    """
    Line plot of metric vs x, one line per hue value.
    Useful for single-axis sweeps.
    """
    fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)
    for val, grp in df.groupby(hue):
        grp = grp.sort_values(x)
        ax.plot(grp[x], grp[metric], marker='o', label=f"{hue}={val}")

    ax.axhline(0, color='black', lw=0.8)
    ax.set_xlabel(x, fontsize=10)
    ax.set_ylabel(metric, fontsize=10)
    ax.set_title(title or f"{metric} vs {x}", fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=dpi)
        print(f"Saved: {save_path}")
    plt.show()