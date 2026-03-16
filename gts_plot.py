"""
gts_plot.py
-----------
Plotting toolbox for the Geometric Task Space model.

All save paths are supplied by the caller — nothing is hardcoded here.
SHORT is defined once in this module; gts_core imports it from here if needed.

Functions
---------
plot_polar(res, label, fam_names, geo_polar, fam_color, ...)
    Unified polar task-space figure with two independent layers:

    show_allocation : bool  (default True)
        Colour every task point by its owning job family.
        Works for baseline (no technology) and all tech cases.

    show_tech_effect : bool  (default False)
        Overlay the technology field:
          - non-operated tasks → light grey background
          - operated tasks     → family colour, alpha ∝ φ_K intensity
          - star marker at p_K, dashed circle at z_K radius

    Combinations:
        Baseline            show_allocation=True,  show_tech_effect=False
        Tech (effect only)  show_allocation=False, show_tech_effect=True
        Tech (combined)     show_allocation=True,  show_tech_effect=True

plot_gradient_overlay(res, label, fam_names, geo_polar, fam_color, ...)
    Polar figure overlaying ||∇φ_K|| on the ownership map.
    Designed for task-creation scenarios (γ > 0): shows where new task
    density concentrates (warm colours = high gradient = boundary zone).

plot_wage_changes(results, labels, fam_names, ...)
    Grouped bar chart of wage changes across scenarios.

plot_scenario(res, save_path=None)
    Convenience wrapper: plots one result dict using its embedded plot config.
    Reads fam_names / geo_polar / fam_color from the active gts_core session.

plot_scenarios(results)
    Calls plot_scenario() for each result in a list.
"""

import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

# ── Short display names (sole definition — import from here if needed) ────────
SHORT = {
    'Architecture and Engineering':                   'Arch. & Eng.',
    'Arts, Design, Entertainment, Sports, and Media': 'Arts & Media',
    'Building and Grounds Cleaning and Maintenance':  'Building & Grounds',
    'Business and Financial Operations':              'Business & Finance',
    'Community and Social Service':                   'Community & Social',
    'Computer and Mathematical':                      'Computer & Math',
    'Construction and Extraction':                    'Construction',
    'Educational Instruction and Library':            'Education',
    'Farming, Fishing, and Forestry':                 'Farming',
    'Food Preparation and Serving Related':           'Food & Serving',
    'Healthcare Practitioners and Technical':         'Healthcare Pract.',
    'Healthcare Support':                             'Healthcare Support',
    'Installation, Maintenance, and Repair':          'Install. & Maint.',
    'Legal':                                          'Legal',
    'Life, Physical, and Social Science':             'Life & Science',
    'Management':                                     'Management',
    'Office and Administrative Support':              'Office & Admin',
    'Personal Care and Service':                      'Personal Care',
    'Production':                                     'Production',
    'Protective Service':                             'Protective Service',
    'Sales and Related':                              'Sales',
    'Transportation and Material Moving':             'Transportation',
}


# ==============================================================================
# Internal helpers
# ==============================================================================

def _polar_coords(r_xy):
    """Return (xi, chi) arrays from (N, 2) Cartesian task positions."""
    chi = np.sqrt(r_xy[:, 0]**2 + r_xy[:, 1]**2)
    xi  = np.arctan2(r_xy[:, 1], r_xy[:, 0]) % (2 * math.pi)
    return xi, chi


def _tech_circle(p_K, z_K, n=600):
    """Parametric circle in polar coords; preserves winding order."""
    th   = np.linspace(0, 2 * math.pi, n)
    rx   = p_K[0] + z_K * np.cos(th)
    ry   = p_K[1] + z_K * np.sin(th)
    rc   = np.sqrt(rx**2 + ry**2)
    xi_r = np.arctan2(ry, rx) % (2 * math.pi)
    return xi_r, rc


def _ax_style(ax):
    """Apply common polar-axis styling."""
    ax.set_rmax(0.9)
    ax.set_rticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ax.set_rlabel_position(12)
    ax.tick_params(axis='y', labelsize=7, labelcolor='gray')
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.set_thetagrids(range(0, 360, 45),
                      [f'{d}°' for d in range(0, 360, 45)], fontsize=8)
    ax.grid(color='gray', linewidth=0.4, alpha=0.4, ls='--')
    ax.set_facecolor('#f7f7f7')


def _save(fig, save_path, dpi):
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight', dpi=dpi)
        print(f"Saved: {save_path}")
    plt.show()
    plt.close(fig)


# ==============================================================================
# Main polar figure
# ==============================================================================

def _rho_colors(base_color, rho_vals, alpha_base=0.55, sat_boost=0.35):
    """
    Return per-point RGBA array with saturation and alpha boosted by task density.

    rho_vals = 1.0  → baseline appearance (same as gamma=0)
    rho_vals > 1.0  → higher saturation and alpha (extra task density)

    Parameters
    ----------
    base_color : matplotlib color
    rho_vals   : array, baseline = 1.0, higher = denser
    alpha_base : float  alpha at rho=1 (matches gamma=0 appearance)
    sat_boost  : float  maximum additional saturation fraction at rho peak
    """
    import matplotlib.colors as mc
    r, g, b, _ = mc.to_rgba(base_color)
    h, s_base, v = mc.rgb_to_hsv(np.array([[r, g, b]]))[0]

    # t in [0,1]: how much above baseline; rho=1 → t=0
    excess   = np.clip(rho_vals - 1.0, 0, None)
    exc_max  = float(excess.max()) if float(excess.max()) > 0 else 1.0
    t        = excess / exc_max          # 0 at baseline, 1 at peak

    sat  = np.clip(s_base + sat_boost * t, 0.0, 1.0)
    hsv  = np.stack([np.full_like(sat, h), sat, np.full_like(sat, v)], axis=1)
    rgb  = mc.hsv_to_rgb(hsv)
    alpha = np.clip(alpha_base + (1.0 - alpha_base) * t * 0.7, 0.15, 0.95)
    return np.column_stack([rgb, alpha])



def plot_polar(res, label, fam_names, geo_polar, fam_color,
               show_allocation=True,
               show_tech_effect=False,
               save_path=None,
               dpi=180):
    """
    Unified polar task-space figure.

    Parameters
    ----------
    res : dict
        Output of geo_baseline() or tech_shock().  Must contain:
        r_xy, owner_idx, phi_K, tech, op_share.
    label : str
        Figure title.
    fam_names : list[str]
    geo_polar : dict   {family: {'p_xy': array([x, y])}}
    fam_color : dict   {family: colour}
    show_allocation : bool
    show_tech_effect : bool
    save_path : path-like or None
    dpi : int
    """
    n     = len(fam_names)
    r_xy  = res['r_xy']
    owner = res['owner_idx']
    phi   = res.get('phi_K', np.zeros(len(owner)))
    tech  = res.get('tech', {'p_xy': np.array([0.0, 0.0]), 'z': 0.0, 'A': 0.0})
    rho   = res.get('rho_1', np.ones(len(owner)))   # task density [0,1]

    xi, chi = _polar_coords(r_xy)
    op_mask = owner >= n
    op_who  = owner - n

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'},
                           figsize=(8, 8), dpi=dpi)

    # ── Layer 1: allocation ───────────────────────────────────────────────────
    if show_allocation:
        uncovered = (owner == -1)
        ax.scatter(xi[uncovered], chi[uncovered],
                   s=0.8, color='#cccccc', alpha=0.35, zorder=1, linewidths=0)
        for i, g in enumerate(fam_names):
            mask = (owner == i)
            if mask.sum() > 0:
                rgba = _rho_colors(fam_color[g], rho[mask])
                ax.scatter(xi[mask], chi[mask], s=2.0,
                           c=rgba, zorder=2, linewidths=0)
        for i, g in enumerate(fam_names):
            mask = op_mask & (op_who == i)
            if mask.sum() > 0:
                rgba = _rho_colors(fam_color[g], rho[mask], sat_boost=0.5)
                ax.scatter(xi[mask], chi[mask], s=2.0,
                           c=rgba, zorder=3, linewidths=0)

    # ── Layer 2: tech effect ──────────────────────────────────────────────────
    if show_tech_effect:
        ax.scatter(xi[~op_mask], chi[~op_mask],
                   s=0.8, color='#cccccc', alpha=0.35,
                   zorder=4 if show_allocation else 1, linewidths=0)

        for i, g in enumerate(fam_names):
            mask = op_mask & (op_who == i)
            if mask.sum() > 5:
                rgba = _rho_colors(fam_color[g], rho[mask], sat_boost=0.25)
                ax.scatter(xi[mask], chi[mask], s=2.5,
                           c=rgba, zorder=5, linewidths=0)

        if tech['A'] > 0:
            p_K   = tech['p_xy']
            xi_K  = math.atan2(p_K[1], p_K[0]) % (2 * math.pi)
            chi_K = math.sqrt(p_K[0]**2 + p_K[1]**2)
            ax.scatter([xi_K], [chi_K],
                       s=260, marker='*', color='black', zorder=7)
            xi_r, rc = _tech_circle(p_K, tech['z'])
            ax.plot(xi_r, rc, color='black', lw=1.3, ls='--',
                    alpha=0.7, zorder=6)

    # ── Job-family centroids ──────────────────────────────────────────────────
    for g in fam_names:
        p_g   = geo_polar[g]['p_xy']
        xi_c  = math.atan2(p_g[1], p_g[0]) % (2 * math.pi)
        chi_c = math.sqrt(p_g[0]**2 + p_g[1]**2)
        ax.scatter([xi_c], [chi_c],
                   s=65, color=fam_color[g], marker='D',
                   zorder=8, edgecolors='white', linewidths=0.6)

    # ── Axes, title, annotation ───────────────────────────────────────────────
    _ax_style(ax)
    ax.set_title(label, fontsize=11, fontweight='bold', pad=18)

    if show_tech_effect:
        frac = sum(res.get('op_share', {}).get(g, 0) for g in fam_names)
        ax.text(0.5, -0.04, f'Tech-assisted task mass: {100*frac:.1f}%',
                transform=ax.transAxes, ha='center', fontsize=9)

    # ── Legend ────────────────────────────────────────────────────────────────
    handles = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=fam_color[g], markersize=7,
               label=SHORT.get(g, g))
        for g in fam_names
    ]
    if show_tech_effect and tech['A'] > 0:
        handles += [
            Line2D([0], [0], marker='*', color='black', markersize=10,
                   linestyle='none', label='Technology $p_K$'),
            Line2D([0], [0], color='black', lw=1.3, ls='--',
                   label='Tech. radius $z_K$'),
        ]
    fig.legend(handles=handles, loc='lower center',
               bbox_to_anchor=(0.5, -0.18), ncol=4, fontsize=7.5,
               framealpha=0.9,
               title='Job family  (◆ = centroid)', title_fontsize=8)

    plt.tight_layout()
    _save(fig, save_path, dpi)
    plt.close(fig)


# ==============================================================================
# Gradient-overlay figure  (task-creation boundary zone)
# ==============================================================================

def plot_gradient_overlay(res, label, fam_names, geo_polar, fam_color,
                          grad_threshold=0.05,
                          cmap_name='YlOrRd',
                          save_path=None,
                          dpi=200):
    """
    Polar task-space figure overlaying ||∇φ_K|| on the ownership map.

    Useful for task-creation scenarios (γ > 0): the gradient boundary zone
    (warm colours) marks where new task density concentrates, revealing
    whether that zone is controlled by labour or operated regimes.

    Parameters
    ----------
    res : dict
        Post-shock result from geo_eq_comparison().
        Must contain: r_xy, owner_idx, phi_K, tech.
    label : str
    fam_names : list[str]
    geo_polar : dict
    fam_color : dict
    grad_threshold : float
        Fraction of peak gradient below which points are not highlighted.
        Default 0.05 (show top 95 % of gradient range).
    cmap_name : str
        Matplotlib colormap for the gradient layer. Default 'YlOrRd'.
    save_path : path-like or None
    dpi : int
    """
    n     = len(fam_names)
    r_xy  = res['r_xy']
    owner = res['owner_idx']
    phi   = res.get('phi_K', np.zeros(len(owner)))
    tech  = res.get('tech', {'p_xy': np.array([0.0, 0.0]), 'z': 0.0, 'A': 0.0})
    rho   = res.get('rho_1', np.ones(len(owner)))   # task density [0,1]

    xi, chi = _polar_coords(r_xy)
    op_mask = owner >= n
    op_who  = owner - n

    # ── ||∇φ_K|| analytically from stored φ_K values ─────────────────────────
    p_K = tech['p_xy']
    z_K = tech['z']
    if z_K <= 0:
        raise ValueError(
            "plot_gradient_overlay requires an active technology (tech['z'] > 0).")
    dx       = r_xy[:, 0] - p_K[0]
    dy       = r_xy[:, 1] - p_K[1]
    d        = np.sqrt(dx**2 + dy**2)
    grad     = (d / z_K**2) * phi            # ||∇φ_K(r)|| = (1/z²)·d·φ_K
    grad_max = float(grad.max()) if grad.max() > 0 else 1.0
    grad_norm = grad / grad_max
    grad_mask = grad_norm > grad_threshold

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'},
                           figsize=(8, 8), dpi=dpi)

    # ── Layer 1: ownership background ────────────────────────────────────────
    uncovered = (owner == -1)
    ax.scatter(xi[uncovered], chi[uncovered],
               s=0.8, color='#cccccc', alpha=0.28, zorder=1, linewidths=0)
    for i, g in enumerate(fam_names):
        mask = (owner == i)
        if mask.sum() > 0:
            ax.scatter(xi[mask], chi[mask], s=1.5,
                       color=fam_color[g], alpha=0.30,
                       zorder=2, linewidths=0)
    for i, g in enumerate(fam_names):
        mask = op_mask & (op_who == i)
        if mask.sum() > 0:
            ax.scatter(xi[mask], chi[mask], s=1.5,
                       color=fam_color[g], alpha=0.58,
                       zorder=3, linewidths=0)

    # ── Layer 2: gradient heatmap ─────────────────────────────────────────────
    cmap_obj  = cm.get_cmap(cmap_name)
    rgba_vals = cmap_obj(grad_norm[grad_mask])
    rgba_vals[:, 3] = np.clip(0.12 + 0.38 * grad_norm[grad_mask], 0.0, 0.50)
    ax.scatter(xi[grad_mask], chi[grad_mask],
               s=3.0, c=rgba_vals, zorder=4, linewidths=0)

    # ── Layer 3: gradient-peak ring at ||r - p_K|| = z_K ─────────────────────
    if tech['A'] > 0:
        xi_ring, chi_ring = _tech_circle(p_K, z_K)
        ax.plot(xi_ring, chi_ring,
                color='#d62728', lw=1.6, ls='-', alpha=0.85, zorder=6,
                label=f'$\\|\\nabla\\phi_K\\|$ peak ring ($z_K={z_K:.2f}$)')
        xi_K  = math.atan2(p_K[1], p_K[0]) % (2 * math.pi)
        chi_K = math.sqrt(p_K[0]**2 + p_K[1]**2)
        ax.scatter([xi_K], [chi_K],
                   s=280, marker='*', color='black', zorder=8,
                   label='Technology centre $p_K$')

    # ── Layer 4: family centroids ─────────────────────────────────────────────
    for g in fam_names:
        p_g   = geo_polar[g]['p_xy']
        xi_c  = math.atan2(p_g[1], p_g[0]) % (2 * math.pi)
        chi_c = math.sqrt(p_g[0]**2 + p_g[1]**2)
        ax.scatter([xi_c], [chi_c],
                   s=65, color=fam_color[g], marker='D',
                   zorder=9, edgecolors='white', linewidths=0.6)

    # ── Axes and title ────────────────────────────────────────────────────────
    _ax_style(ax)
    ax.set_title(label, fontsize=11, fontweight='bold', pad=18)

    frac = sum(res.get('op_share', {}).get(g, 0) for g in fam_names)
    ax.text(0.5, -0.04,
            f'Operated task mass: {100*frac:.1f}%  |  '
            f'Gradient threshold: >{100*grad_threshold:.0f}% of peak',
            transform=ax.transAxes, ha='center', fontsize=8.5)

    # ── Colorbar ──────────────────────────────────────────────────────────────
    sm = cm.ScalarMappable(cmap=cmap_obj,
                           norm=mcolors.Normalize(vmin=0, vmax=grad_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal',
                        pad=0.08, fraction=0.03, aspect=35)
    cbar.set_label(r'$\|\nabla\phi_K(\mathbf{r})\|$  (normalised)',
                   fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5)

    # ── Legend ────────────────────────────────────────────────────────────────
    family_handles = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=fam_color[g], markersize=7,
               label=SHORT.get(g, g))
        for g in fam_names
    ]
    extra_handles = [
        Line2D([0], [0], marker='*', color='black', markersize=10,
               linestyle='none', label='Technology $p_K$'),
        Line2D([0], [0], color='#d62728', lw=1.6, ls='-',
               label=r'$\|\nabla\phi_K\|$ peak ring'),
    ]
    fig.legend(handles=family_handles + extra_handles,
               loc='lower center', bbox_to_anchor=(0.5, -0.22),
               ncol=4, fontsize=7.5, framealpha=0.9,
               title='Job family  (◆ = centroid)', title_fontsize=8)

    plt.tight_layout()
    _save(fig, save_path, dpi)
    plt.close(fig)


# ==============================================================================
# Wage-change bar chart
# ==============================================================================

def plot_wage_changes(results, labels, fam_names,
                      save_path=None, dpi=150, title=None):
    """
    Grouped bar chart of wage changes across scenarios.

    Parameters
    ----------
    results : list[dict]
        Each dict must contain w_post and w_base.
    labels : list[str]
        One label per result.
    fam_names : list[str]
    save_path : path-like or None
    dpi : int
    title : str or None
        Figure title. Defaults to standard formula title.
    """
    fam_sorted = sorted(fam_names, key=lambda g: results[0]['w_base'][g])
    x      = np.arange(len(fam_sorted))
    n_res  = len(results)
    colors = ['#2166ac', '#4dac26', '#d6604d', '#984ea3',
              '#ff7f00', '#a65628', '#f781bf', '#999999']
    width   = min(0.25, 0.8 / max(n_res, 1))
    offsets = np.linspace(-(n_res - 1) / 2, (n_res - 1) / 2, n_res) * width

    fig, ax = plt.subplots(figsize=(15, 6), dpi=dpi)
    for i, (res, lbl) in enumerate(zip(results, labels)):
        pcts = [100 * (res['w_post'][g] - res['w_base'][g]) / res['w_base'][g]
                for g in fam_sorted]
        ax.bar(x + offsets[i], pcts, width,
               label=lbl, color=colors[i % len(colors)],
               alpha=0.82, edgecolor='white', linewidth=0.4)

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT.get(g, g) for g in fam_sorted],
                       rotation=40, ha='right', fontsize=7.5)
    ax.set_ylabel('Wage change from geometric baseline (%)', fontsize=9)
    _title = title if title is not None else (
        'Wage changes — GTS scenarios\n'
        r'$\Delta w_g / w_g^{base} = (D_g^{post} / D_g^{base})^{1/\lambda} - 1$'
    )
    ax.set_title(_title, fontsize=10, fontweight='bold', pad=10)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(axis='y', linewidth=0.4, alpha=0.5)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    _save(fig, save_path, dpi)
    plt.show()


# ==============================================================================
# Convenience wrappers (use active gts_core session)
# ==============================================================================

def plot_scenario(res, save_path=None):
    """
    Plot one result dict using its embedded plot config.
    Reads fam_names / geo_polar / fam_color from the active gts_core session.

    Parameters
    ----------
    res : dict
        Result from gts_core.run_scenario() or gts_core.eq_comparison().
    save_path : path-like or None
        Overrides res['save_path'] if given.
    """
    import gts_core as gts
    path = save_path if save_path is not None else res.get('save_path')
    plot_polar(
        res, label=res['name'],
        fam_names=gts.fam_names(),
        geo_polar=gts.geo_polar(),
        fam_color=gts.fam_color(),
        save_path=path,
        **res.get('plot', {}),
    )


def plot_scenarios(results):
    """Call plot_scenario() for each result in a list."""
    for res in results:
        plot_scenario(res)