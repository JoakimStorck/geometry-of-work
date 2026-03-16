"""
gts_core.py
-----------
Geometric Task Space model — computation engine and session state.

This module merges the former ar_euclidean_absgolv.py (computation engine)
and ar_main.py (orchestration / lazy state) into a single clean module.

All file paths are supplied by the caller via init() — nothing is
hardcoded here.  Scenario definitions and save paths belong in the
notebook, not in this module.

Typical notebook usage
----------------------
    import gts_core as gts
    import gts_plot

    # 1. Initialise (once per session)
    gts.init(
        geo_path  = RP.cache / "job_family_centers_polar_scaled.csv",
        wage_path = RP.cache / "wage_per_job_family.csv",
        occ_path  = RP.cache / "occupation_embeddings_polar_scaled.csv",
        edu_path  = RP.cache / "rle_by_job_family.csv",
    )

    # 2. Define scenario and run
    scenario = dict(name="AI (γ=0)", xi_deg=40.6, chi=0.35,
                    z=0.30, A=2.0, R=0.60, gamma=0.0)
    b, p = gts.eq_comparison(scenario, theta=0.3)

    # 3. Plot
    gts_plot.plot_polar(p, label=scenario['name'],
                        fam_names=gts.fam_names(),
                        geo_polar=gts.geo_polar(),
                        fam_color=gts.fam_color(),
                        save_path=RP.exports / "fig_ai.png")
"""

import math
import copy
import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict

matplotlib.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm'})

# ── Module-level session state ─────────────────────────────────────────────────
_state = {}   # populated by init()

# ── Default solver parameters (can be overridden in init) ─────────────────────
_DEFAULTS = dict(lam=1.5, N=150_000, seed=1)

# ── Absolute productivity floor ────────────────────────────────────────────────
KAPPA_ABS = 0.10


# ==============================================================================
# Session management
# ==============================================================================

def init(geo_path, wage_path, occ_path, edu_path,
         lam=1.5, N=150_000, seed=1):
    """
    Load data and compute geometric baseline.  Must be called once before
    any run_scenario() / eq_comparison() call.

    Parameters
    ----------
    geo_path  : path-like   job_family_centers_polar_scaled.csv
    wage_path : path-like   wage_per_job_family.csv
    occ_path  : path-like   occupation_embeddings_polar_scaled.csv
    edu_path  : path-like   rle_by_job_family.csv
    lam       : float       CES elasticity (default 1.5)
    N         : int         MC sample size for baseline (default 150 000)
    seed      : int         RNG seed (default 1)
    """
    _state.clear()
    _state['cfg'] = dict(geo_path=str(geo_path), wage_path=str(wage_path),
                         occ_path=str(occ_path), edu_path=str(edu_path),
                         lam=lam, N=N, seed=seed)

    # Load families
    families = load_job_families(geo_path, wage_path, occ_path=occ_path)
    _state['families_init'] = families
    fams = list(families.keys())
    _state['fam_names'] = fams
    print(f"Loaded {len(fams)} job families.  κ_abs = {KAPPA_ABS}")

    # Attach education data
    import pandas as pd
    edu = pd.read_csv(edu_path).set_index('Job Family')
    edu_mean = float(edu['rle_mean_family'].mean())
    for g in fams:
        if g in edu.index:
            families[g]['edu'] = float(edu.loc[g, 'rle_mean_family'])
        else:
            families[g]['edu'] = edu_mean
            print(f"  Warning: no edu data for '{g}', using mean")

    # Calibrate ω_g to BLS wages (A_g = 1 throughout)
    families, base = calibrate_omega(
        families, lam=lam, N=N, seed=seed)
    _state['families_init'] = families
    base['tech']     = dict(p_xy=np.array([0.0, 0.0]), z=0.01, A=0.0, R=0.0)
    base['phi_K']    = np.zeros(len(base['r_xy']))
    base['op_share'] = {g: 0.0 for g in fams}
    _state['base'] = base

    # Plotting helpers
    n    = len(fams)
    cmap = matplotlib.colormaps.get_cmap('tab20').resampled(n)
    _state['fam_color'] = {g: cmap(i) for i, g in enumerate(fams)}
    _state['geo_polar'] = {g: {'p_xy': families[g]['p_xy']} for g in fams}

    # Print calibrated baseline wages (observed = ω_g · w_g)
    print("\n  Calibrated baseline wages:")
    print(f"  {'Job family':<45}  {'ω_g':>6}  {'ŵ':>6}  {'w_BLS':>6}  {'err%':>6}")
    for g in sorted(fams, key=lambda g: -base['families'][g]['w'] * families[g]['omega']):
        w_hat = base['families'][g]['w'] * families[g]['omega']
        bls   = families[g]['w_bls']
        omega = families[g]['omega']
        print(f"  {g:<45}  {omega:6.3f}  {w_hat:.3f}  {bls:.3f}  "
              f"{100*(w_hat/bls-1):+5.1f}%")


def reset():
    """Clear session state; next call to any accessor re-triggers init()."""
    _state.clear()
    print("Session state cleared.")


def _require_init():
    if not _state:
        raise RuntimeError(
            "gts_core not initialised — call gts.init(...) first.")


# ── Convenience accessors ──────────────────────────────────────────────────────

def fam_names():
    _require_init(); return _state['fam_names']

def families_init():
    _require_init(); return _state['families_init']

def fam_color():
    _require_init(); return _state['fam_color']

def geo_polar():
    _require_init(); return _state['geo_polar']

def base():
    _require_init(); return _state['base']


# ==============================================================================
# Scenario execution
# ==============================================================================

def pxy(xi_deg, chi):
    """Convert polar (xi°, chi) to Cartesian p_xy."""
    return np.array([chi * math.cos(math.radians(xi_deg)),
                     chi * math.sin(math.radians(xi_deg))])


def _tech_from_scenario(cfg):
    return dict(p_xy=pxy(cfg['xi_deg'], cfg['chi']),
                z=cfg['z'], A=cfg['A'], R=cfg['R'])


def run_scenario(cfg):
    """
    Run one scenario dict using the one-shot tech_shock approach.

    Parameters
    ----------
    cfg : dict with keys
        name, xi_deg, chi, z, A, R, gamma
        Optional: lam (overrides session lam)

    Returns
    -------
    Result dict with metadata (name, plot config) attached.
    """
    _require_init()
    _base = _state['base']
    _fam  = _state['families_init']
    lam   = cfg.get('lam', _state['cfg']['lam'])

    tech = _tech_from_scenario(cfg)

    if cfg['A'] == 0.0:
        res = dict(_base)
    else:
        res = tech_shock(_base, tech, _fam, lam=lam, gamma=cfg['gamma'])
        print_report(res, cfg['name'])

    res['name'] = cfg['name']
    res['plot'] = cfg.get('plot', dict(show_allocation=False,
                                       show_tech_effect=True))
    return res


def run_scenarios(scenario_list):
    """Run a list of scenario dicts; return list of result dicts."""
    results = []
    for cfg in scenario_list:
        print(f"\nRunning: {cfg['name']} ...")
        results.append(run_scenario(cfg))
    return results


def eq_comparison(scenario, theta=0.3, mu=0.3, N=None, N_inner=30_000):
    """
    Run baseline + post-shock geo_equilibrium_v2 and print comparison table.

    Parameters
    ----------
    scenario : dict with keys name, xi_deg, chi, z, A, R, gamma
    theta    : float   logit mobility elasticity (default 0.3)
    mu       : float   damping factor (default 0.3)
    N        : int     MC sample size (default: session N)
    N_inner  : int     inner-loop MC size (default 30 000)

    Returns
    -------
    (eq_base, eq_post)
    """
    _require_init()
    cfg   = _state['cfg']
    N     = N or cfg['N']
    lam   = scenario.get('lam', cfg['lam'])
    seed  = cfg['seed']

    tech  = _tech_from_scenario(scenario)
    gamma = scenario.get('gamma', 0.0)

    print(f"\n  === {scenario.get('name', 'Scenario')} ===")
    return geo_eq_comparison(
        _state['families_init'],
        tech=tech, gamma=gamma,
        theta=theta, mu=mu,
        N=N, N_inner=N_inner,
        seed=seed, lam=lam)


# ==============================================================================
# Geometry helpers
# ==============================================================================

def sample_disc(N, rng, scale=1.0):
    """Sample N points uniformly on disc of radius scale."""
    angle = rng.uniform(0.0, 2 * math.pi, N)
    r     = scale * np.sqrt(rng.uniform(0.0, 1.0, N))
    return np.column_stack([r * np.cos(angle), r * np.sin(angle)])


def dist(r_xy, p_xy):
    d = r_xy - np.asarray(p_xy, dtype=float).reshape(2,)
    return np.sqrt(np.sum(d * d, axis=-1))


def gauss(u):
    return np.exp(-0.5 * u * u)


# ==============================================================================
# Data loading
# ==============================================================================

def load_job_families(geo_path, wage_path, occ_path=None):
    """
    Load job families from CSV files.

    p_xy  = (chi·cos(ξ), chi·sin(ξ))  — Cartesian from polar (ξ, χ)
    z_g   = mean Euclidean distance to member occupations
    L_g   = BLS employment share
    w     = 1.0  (geometric baseline; BLS wages stored for reference)
    w_bls = BLS relative wage
    """
    geo = {}
    with open(geo_path) as f:
        for row in csv.DictReader(f):
            xi  = float(row['xi'])
            chi = float(row['chi'])
            geo[row['Job Family']] = dict(
                xi=xi, chi=chi,
                x=chi * math.cos(xi),
                y=chi * math.sin(xi))

    wage = {}
    with open(wage_path) as f:
        for row in csv.DictReader(f):
            wage[row['job_family']] = dict(
                w_bls=float(row['wage_rel_to_base']),
                emp=float(row['tot_emp']))

    z_eucl = {}
    if occ_path is not None:
        by_family = defaultdict(list)
        with open(occ_path) as f:
            for row in csv.DictReader(f):
                xi_o  = float(row['xi'])
                chi_o = float(row['chi'])
                by_family[row['Job Family']].append(
                    (chi_o * math.cos(xi_o), chi_o * math.sin(xi_o)))
        for name in geo:
            pts = np.array(by_family.get(name, []))
            if len(pts) >= 2:
                cx, cy  = geo[name]['x'], geo[name]['y']
                d2      = (pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2
                z_eucl[name] = float(np.sqrt(np.mean(d2)))   # RMS — consistent with R_o
            else:
                z_eucl[name] = 0.05

    total_emp = sum(v['emp'] for v in wage.values())
    families  = {}
    for name in geo:
        p_xy = np.array([geo[name]['x'], geo[name]['y']])
        families[name] = dict(
            p_xy  = p_xy,
            z     = z_eucl.get(name, 0.08),
            w     = 1.0,
            w_bls = wage[name]['w_bls'],
            emp   = wage[name]['emp'],
            L     = wage[name]['emp'] / total_emp,
            A     = 1.0,    # geometric productivity — fixed at 1, not calibrated
            omega = 1.0,    # wage premium — calibrated to BLS in init()
        )
    return families


# ==============================================================================
# Productivity and technology fields
# ==============================================================================

def labor_psi(r_xy, fam):
    u = dist(r_xy, fam['p_xy']) / max(fam['z'], 1e-12)
    return fam['A'] * gauss(u)


def tech_phi(r_xy, tech):
    if tech['A'] == 0:
        return np.zeros(len(r_xy))
    u = dist(r_xy, tech['p_xy']) / max(tech['z'], 1e-12)
    return tech['A'] * gauss(u)


def gradient_phi(r_xy, tech):
    if tech['A'] == 0:
        return np.zeros(len(r_xy))
    d   = dist(r_xy, tech['p_xy'])
    phi = tech_phi(r_xy, tech)
    return (1.0 / tech['z']**2) * d * phi


# ==============================================================================
# Task allocation
# ==============================================================================

def allocate(r_xy, families, tech, lam=1.5, gamma=0.0, tau=0.0, use_omega=True):
    """
    Cost-minimising task allocation.

      Pure regime g:    cost = (ω_g · w_g) / ψ_g(r)
      Operated g⊗K:    cost = (ω_g · w_g + R) / (ψ_g(r) · (1 + φ_K(r)))
      Eligible if ψ_g(r) ≥ KAPPA_ABS

    tau > 0: softmin (Boltzmann) assignment for convergence.
    tau = 0 (default): hard argmin — economically correct.
    gamma > 0: task-creation weighting via rho field.
    """
    fam_list = list(families.keys())
    psi_map  = {g: labor_psi(r_xy, families[g]) for g in fam_list}
    phi_K    = tech_phi(r_xy, tech)

    rho = np.ones(len(r_xy))
    if gamma > 0:
        S      = np.maximum.reduce([psi_map[g] for g in fam_list])
        grad_K = gradient_phi(r_xy, tech)
        rho    = 1.0 + gamma * S * grad_K

    INF  = 1e18
    n    = len(fam_list)
    costs = []

    # Pure-labour costs: (ω_g ·) w_g / ψ_g(r)
    # ω_g included only in tech_shock (use_omega=True); excluded from baseline
    for g in fam_list:
        omega = families[g].get('omega', 1.0) if use_omega else 1.0
        w_eff = omega * families[g]['w']
        elig  = psi_map[g] >= KAPPA_ABS
        c     = np.full(len(r_xy), INF)
        c[elig] = w_eff / np.maximum(psi_map[g][elig], 1e-300)
        costs.append(c)

    # Operated costs: (ω_g · w_g + R) / (ψ_g(r) · (1 + φ_K(r)))
    for g in fam_list:
        omega = families[g].get('omega', 1.0) if use_omega else 1.0
        w_eff = omega * families[g]['w']
        psi_e = psi_map[g] * (1.0 + phi_K)
        elig  = psi_map[g] >= KAPPA_ABS
        c     = np.full(len(r_xy), INF)
        c[elig] = (w_eff + tech['R']) / np.maximum(psi_e[elig], 1e-300)
        costs.append(c)

    C       = np.stack(costs, axis=0)       # (2n, N)
    covered = np.any(
        np.stack([psi_map[g] >= KAPPA_ABS for g in fam_list], axis=0), axis=0)

    rho_cov      = rho * covered
    rho_cov_mean = max(float(rho_cov.mean()), 1e-300)
    power        = lam - 1.0

    Gamma = {}; task_share = {}; pure_share = {}; op_share = {}

    if tau <= 0.0:
        # Hard argmin
        owner_idx = np.argmin(C, axis=0)
        owner_idx[~covered] = -1
        for i, g in enumerate(fam_list):
            mb = ((owner_idx == i) | (owner_idx == n + i)).astype(float)
            mp = (owner_idx == i).astype(float)
            mo = (owner_idx == n + i).astype(float)
            task_share[g] = float((mb * rho_cov).mean() / rho_cov_mean)
            pure_share[g] = float((mp * rho_cov).mean() / rho_cov_mean)
            op_share[g]   = float((mo * rho_cov).mean() / rho_cov_mean)
            Gamma[g]      = float(
                (mb * psi_map[g]**power * rho_cov).mean() / rho_cov_mean)
    else:
        # Softmin
        C_soft = np.where(C >= INF * 0.5, 1e6, C)
        C_min  = C_soft.min(axis=0, keepdims=True)
        logits = -(C_soft - C_min) / tau
        pi     = np.exp(logits)
        pi    /= pi.sum(axis=0, keepdims=True)
        for i, g in enumerate(fam_list):
            elig_i  = (psi_map[g] >= KAPPA_ABS).astype(float)
            pi[i]   *= elig_i
            pi[n+i] *= elig_i
        pi_sum = pi.sum(axis=0, keepdims=True)
        pi     = np.where(pi_sum > 0, pi / np.maximum(pi_sum, 1e-300), 0.0)

        owner_idx = np.argmin(C, axis=0)
        owner_idx[~covered] = -1
        for i, g in enumerate(fam_list):
            mb_soft = pi[i] + pi[n + i]
            mp_soft = pi[i]
            mo_soft = pi[n + i]
            task_share[g] = float((mb_soft * rho_cov).mean() / rho_cov_mean)
            pure_share[g] = float((mp_soft * rho_cov).mean() / rho_cov_mean)
            op_share[g]   = float((mo_soft * rho_cov).mean() / rho_cov_mean)
            Gamma[g]      = float(
                (mb_soft * psi_map[g]**power * rho_cov).mean() / rho_cov_mean)

    return Gamma, task_share, pure_share, op_share, owner_idx, phi_K


# ==============================================================================
# Convergence helpers
# ==============================================================================

def _recenter(log_v, weights):
    w_arr = np.asarray(weights, dtype=float)
    return log_v - float(np.dot(w_arr, log_v) / w_arr.sum())


def _compute_log_target(log_w, fam, fams, r_xy, tech_null, lam,
                        gamma_floor=0.10, tau=0.0, gamma=0.0):
    for i, g in enumerate(fams):
        fam[g]['w'] = math.exp(log_w[i])
    Gamma, D, *_ = allocate(r_xy, fam, tech_null, lam=lam, tau=tau, gamma=gamma)
    weights = np.array([fam[g]['L'] for g in fams], dtype=float)
    log_t   = np.array([
        (1.0 / lam) * math.log(
            max(Gamma[g], gamma_floor * fam[g]['L'], 1e-12) /
            max(fam[g]['L'], 1e-12))
        for g in fams
    ])
    log_t = _recenter(log_t, weights)
    return log_t, Gamma, D


def _anderson_safeguarded(x_hist, g_hist, m=6, ridge=1e-4):
    k = min(len(x_hist), m)
    if k <= 1:
        return g_hist[-1]
    R   = np.column_stack([g_hist[-k+i] - x_hist[-k+i] for i in range(k)])
    RtR = R.T @ R
    try:
        A   = np.vstack([RtR + ridge * np.eye(k), np.ones(k)])
        b   = np.append(np.zeros(k), 1.0)
        c, *_ = np.linalg.lstsq(A, b, rcond=None)
        c   = c[:k]
        if not np.all(np.isfinite(c)):
            return g_hist[-1]
        c = c / c.sum()
    except (np.linalg.LinAlgError, ZeroDivisionError):
        return g_hist[-1]
    return np.column_stack(g_hist[-k:]) @ c


def _picard_aa_phase(log_w0, fam, fams, r_xy, tech_null, lam,
                     gamma_floor, tau, alpha_picard, aa_start,
                     aa_window, aa_every, max_iter, eps, label="",
                     gamma=0.0):
    log_w      = log_w0.copy()
    x_hist     = []
    g_hist     = []
    aa_rejects = 0

    log_t, _, _ = _compute_log_target(
        log_w, fam, fams, r_xy, tech_null, lam, gamma_floor, tau, gamma=gamma)
    resid      = log_t - log_w
    resid_norm = float(np.max(np.abs(resid)))

    for it in range(max_iter):
        log_w_p  = log_w + alpha_picard * resid
        log_t_p, _, _ = _compute_log_target(
            log_w_p, fam, fams, r_xy, tech_null, lam, gamma_floor, tau, gamma=gamma)
        r_p_norm = float(np.max(np.abs(log_t_p - log_w_p)))

        x_hist.append(log_w.copy())
        g_hist.append(log_t.copy())

        used_aa = False
        if it >= aa_start and (it % aa_every == 0) and len(x_hist) >= 2:
            log_aa = _anderson_safeguarded(x_hist, g_hist, m=aa_window)
            log_t_aa, _, _ = _compute_log_target(
                log_aa, fam, fams, r_xy, tech_null, lam, gamma_floor, tau, gamma=gamma)
            r_aa_norm = float(np.max(np.abs(log_t_aa - log_aa)))
            if r_aa_norm < r_p_norm:
                log_w      = log_aa
                log_t      = log_t_aa
                resid      = log_t - log_w
                resid_norm = r_aa_norm
                used_aa    = True
            else:
                x_hist     = []
                g_hist     = []
                aa_rejects += 1

        if not used_aa:
            log_w      = log_w_p
            log_t      = log_t_p
            resid      = log_t - log_w
            resid_norm = r_p_norm

        if (it + 1) % 50 == 0:
            src = "AA" if used_aa else "Pi"
            print(f"    {label}iter {it+1:4d}  ||r||={resid_norm:.5f}  "
                  f"{src}  AA_rej={aa_rejects}")

        if resid_norm < eps:
            print(f"    {label}converged @ iter {it+1}  ||r||={resid_norm:.6f}")
            break

    return log_w, resid_norm, aa_rejects


# ==============================================================================
# Geometric baseline equilibrium
# ==============================================================================

def geo_baseline(families_init, N=150_000, seed=1, lam=1.5,
                 max_iter=500, eps=5e-4,
                 alpha_picard=0.02, aa_start=50, aa_window=6, aa_every=5,
                 gamma_floor=0.10,
                 tau_schedule=(0.10, 0.03, 0.003),
                 tau_eps_scale=5.0,
                 warm_start=False,
                 tech=None, gamma=0.0):
    """
    FOC wage equilibrium without technology.

    Convergence: softmin continuation (tau schedule) + safeguarded Anderson.
    Final evaluation always uses hard argmin (tau=0).
    """
    _tech_null = dict(p_xy=np.array([0.0, 0.0]), z=0.01, A=0.0, R=999.0)
    tech_use   = tech if tech is not None else _tech_null
    rng        = np.random.default_rng(seed)

    N_iter = min(N, 50_000)
    r_iter = sample_disc(N_iter, rng)
    r_full = sample_disc(N, rng) if N > N_iter else r_iter

    fam  = copy.deepcopy(families_init)
    fams = list(fam.keys())
    for g in fams:
        fam[g]['w']     = families_init[g].get('w', 1.0) if warm_start else 1.0
        fam[g]['A']     = families_init[g].get('A', 1.0)
        fam[g]['omega'] = families_init[g].get('omega', 1.0)

    log_w   = np.array([math.log(max(fam[g]['w'], 1e-12)) for g in fams])
    weights = np.array([fam[g]['L'] for g in fams])
    log_w  -= float(np.dot(weights, log_w) / weights.sum())

    for stage, tau in enumerate(tau_schedule):
        stage_eps = eps * tau_eps_scale if tau > 0.0 else eps
        label = f"τ={tau:.3f}  "
        print(f"  Stage {stage+1}/{len(tau_schedule)}: {label}")
        log_w, resid_norm, _ = _picard_aa_phase(
            log_w, fam, fams, r_iter, tech_use, lam,
            gamma_floor, tau, alpha_picard,
            aa_start, aa_window, aa_every,
            max_iter, stage_eps, label=label, gamma=gamma)
        if tau == 0.0 and resid_norm >= eps:
            print(f"  Warning: final phase did not reach eps={eps}  "
                  f"||r||={resid_norm:.5f}")

    for i, g in enumerate(fams):
        fam[g]['w'] = math.exp(log_w[i])

    Gamma, D, pure, op, owner, phi = allocate(
        r_full, fam, tech_use, lam=lam, gamma=gamma)

    return dict(families=fam, task_share=D, pure_share=pure, op_share=op,
                Gamma=Gamma, owner_idx=owner, r_xy=r_full, phi_K=phi,
                tech=tech_use)


# ==============================================================================
# Wage-premium calibration — match ω_g to BLS wages (Option B)
# ==============================================================================

def calibrate_omega(families_init, lam=1.5, N=50_000, seed=1,
                    tol=1e-3, max_outer=60, alpha=0.8,
                    geo_baseline_kwargs=None):
    """
    Calibrate per-family wage premium ω_g so that observed wages
    ω_g · w_g* reproduce BLS relative wages.

    ω_g enters the cost function as (ω_g · w_g) / ψ_g(r) — families that
    are expensive on the market lose task share to cheaper competitors.
    ω_g does NOT enter the FOC target, so the inner solver is unaffected.

    Algorithm (outer loop):
        1. Run geo_baseline with current ω_g
        2. Observed wage: ŵ_g = ω_g · w_g*
        3. Update: ω_g ← w_bls_g / w_g*   (direct reset, no step size needed)
        4. Repeat until max|ŵ_g - w_bls_g| / w_bls_g < tol

    The update rule sets ω_g so that IF w_g* were unchanged, ŵ_g = w_bls_g.
    Convergence occurs when w_g* is self-consistent with ω_g in the cost fn.

    Parameters
    ----------
    families_init       : dict from load_job_families
    lam                 : float   CES elasticity (default 1.5)
    N                   : int     MC sample size per inner solve (default 50k)
    seed                : int     RNG seed
    tol                 : float   convergence in max|ŵ/w_bls - 1| (default 1e-3)
    max_outer           : int     maximum outer iterations (default 30)
    alpha               : float   mixing weight for ω_g update (default 0.8)
                                  ω_new = (1-α)·ω_old + α·(w_bls/w*)
                                  lower α → more stable, slower convergence
    geo_baseline_kwargs : dict    extra kwargs forwarded to geo_baseline

    Returns
    -------
    (families, final_baseline)
    """
    gb_kw    = geo_baseline_kwargs or {}
    fam      = copy.deepcopy(families_init)
    fams     = list(fam.keys())

    for g in fams:
        fam[g]['A']     = 1.0
        fam[g]['omega'] = 1.0

    # Loose tolerance for inner solves
    inner_kw = dict(eps=5e-3, tau_schedule=(0.10, 0.03),
                    max_iter=300, **gb_kw)

    print("\nCalibrating ω_g  (A_g = 1, ω_g in cost, not in FOC)...")
    print(f"  {'Iter':>4}  {'max|err|':>9}  {'mean|err|':>9}")

    for outer in range(max_outer):
        warm   = outer > 0
        result = geo_baseline(fam, N=N, seed=seed, lam=lam,
                              warm_start=warm, **inner_kw)
        w_star = np.array([result['families'][g]['w'] for g in fams])
        omega  = np.array([fam[g]['omega'] for g in fams])
        w_bls  = np.array([fam[g]['w_bls'] for g in fams])
        w_obs  = omega * w_star

        rel_err  = np.abs(w_obs / w_bls - 1.0)
        max_err  = float(rel_err.max())
        mean_err = float(rel_err.mean())
        print(f"  {outer+1:>4}  {max_err:>9.5f}  {mean_err:>9.5f}")

        if max_err < tol:
            print(f"  Converged @ iter {outer+1}  max|err|={max_err:.6f}")
            break

        # Damped update: ω_new = (1-α)·ω_old + α·(w_bls/w*)
        omega_target = w_bls / np.maximum(w_star, 1e-12)
        for i, g in enumerate(fams):
            fam[g]['omega'] = (1.0 - alpha) * fam[g]['omega'] + alpha * float(omega_target[i])
            fam[g]['w']     = float(w_star[i])
    else:
        print(f"  Warning: did not converge in {max_outer} iters  "
              f"max|err|={max_err:.5f}")

    # Final full-precision solve
    print("\n  Final baseline solve with calibrated ω_g...")
    final = geo_baseline(fam, N=N, seed=seed, lam=lam,
                         warm_start=True, **gb_kw)
    for g in fams:
        fam[g]['w'] = final['families'][g]['w']

    print(f"\n  {'Job family':<45}  {'ω_g':>6}  {'ŵ':>6}  {'w_BLS':>6}  {'err%':>6}")
    for g in sorted(fams, key=lambda g: -fam[g]['w_bls']):
        w_hat = fam[g]['omega'] * fam[g]['w']
        print(f"  {g:<45}  {fam[g]['omega']:6.3f}  {w_hat:6.3f}"
              f"  {fam[g]['w_bls']:6.3f}  "
              f"{100*(w_hat/fam[g]['w_bls']-1):+5.1f}%")

    return fam, final


# ==============================================================================
# One-shot technology shock
# ==============================================================================

def tech_shock(baseline, tech, families_init, lam=1.5, gamma=0.0):
    """
    One-shot wage response to technology introduction.

      w_g_post = w_g_base · (D_g_post / D_g_base)^(1/λ)

    Employment-weighted mean is preserved by rescaling.

    Task creation (gamma > 0)
    -------------------------
    New task density is:
      rho_1(r) = rho_0(r) + (gamma * Delta_Gamma_D / G) * S(r) * ||grad phi_K(r)||

    where:
      Delta_Gamma_D  = operated task mass under pure automation (gamma=0)
                       i.e. the fraction of tasks where augmentation reduces
                       required labour input
      G              = integral of S(r)*||grad phi_K(r)|| over task space
                       (Monte Carlo normalisation constant)
      gamma          = reinstatement elasticity: fraction of displaced task
                       mass that returns as new tasks in the boundary zone

    gamma=0 is always run internally to obtain Delta_Gamma_D and the
    pure-automation outcome for reporting.
    """
    fam_list = list(baseline['families'].keys())
    D0       = baseline['task_share']
    w0       = {g: baseline['families'][g]['w'] for g in fam_list}
    r_xy     = baseline['r_xy']

    fam_post = copy.deepcopy(families_init)
    for g in fam_list:
        fam_post[g]['w']     = w0[g]
        fam_post[g]['A']     = 1.0
        fam_post[g]['omega'] = families_init[g].get('omega', 1.0)

    # ── Step 1: always run gamma=0 (pure automation) ─────────────────────────
    _, D1_g0, pure1_g0, op1_g0, owner1_g0, phi1 = allocate(
        r_xy, fam_post, tech, lam=lam, gamma=0.0)

    # Displacement = operated task mass under pure automation
    Delta_Gamma_D = sum(op1_g0[g] for g in fam_list)

    # ── Step 2: task creation if gamma > 0 ───────────────────────────────────
    if gamma > 1e-12 and Delta_Gamma_D > 1e-9:
        # Normalisation constant G = E[S * ||grad phi_K|| | covered]
        psi_map = {g: labor_psi(r_xy, fam_post[g]) for g in fam_list}
        covered = np.any(
            np.stack([psi_map[g] >= KAPPA_ABS for g in fam_list], axis=0), axis=0)
        S       = np.maximum.reduce([psi_map[g] for g in fam_list])
        grad_K  = gradient_phi(r_xy, tech)
        h       = S * grad_K * covered
        G       = float(h.mean()) / max(float(covered.mean()), 1e-9)

        gamma_eff = gamma * Delta_Gamma_D / max(G, 1e-9)

        _, D1, pure1, op1, owner1, phi1 = allocate(
            r_xy, fam_post, tech, lam=lam, gamma=gamma_eff)

        # rho_1 per sample point — raw, unnormalised
        # rho=1 is the uniform baseline; rho>1 indicates new task density
        rho_1 = 1.0 + gamma_eff * h
    else:
        D1, pure1, op1, owner1 = D1_g0, pure1_g0, op1_g0, owner1_g0
        gamma_eff = 0.0
        rho_1 = np.ones(len(r_xy))   # uniform — no density variation

    # ── Step 3: one-shot wage formula ─────────────────────────────────────────
    total_L = sum(families_init[g]['L'] for g in fam_list)
    mean_w0 = sum(w0[g] * families_init[g]['L'] / total_L for g in fam_list)

    def _floored(d, g):
        return max(d, families_init[g]['L'] / 2.0, 1e-9)

    w_post = {g: w0[g] * (_floored(D1[g], g) / _floored(D0[g], g))**(1/lam)
              for g in fam_list}
    mean1  = sum(w_post[g] * families_init[g]['L'] / total_L for g in fam_list)
    sc     = mean_w0 / max(mean1, 1e-12)
    w_post = {g: w_post[g] * sc for g in w_post}

    # gamma=0 wages (for reporting)
    w_post_g0 = {g: w0[g] * (_floored(D1_g0[g], g) / _floored(D0[g], g))**(1/lam)
                 for g in fam_list}
    mean1_g0  = sum(w_post_g0[g] * families_init[g]['L'] / total_L for g in fam_list)
    sc_g0     = mean_w0 / max(mean1_g0, 1e-12)
    w_post_g0 = {g: w_post_g0[g] * sc_g0 for g in w_post_g0}

    return dict(
        # Main result (with task creation if gamma > 0)
        w_post=w_post, w_base=w0, D_post=D1, D_base=D0,
        op_share=op1, pure_share=pure1, owner_idx=owner1,
        phi_K=phi1, r_xy=r_xy, tech=tech,
        gamma=gamma, gamma_eff=gamma_eff,
        rho_1=rho_1,
        # Pure-automation result (gamma=0) — always available
        w_post_g0=w_post_g0, D_post_g0=D1_g0,
        op_share_g0=op1_g0, owner_idx_g0=owner1_g0,
        Delta_Gamma_D=Delta_Gamma_D,
    )


# ==============================================================================
# Simultaneous (w_g, L_g) equilibrium — logit mobility
# ==============================================================================

def geo_equilibrium_v2(families_init, N=150_000, seed=1, lam=1.5,
                       theta=0.3, mu=0.3, max_outer=50, outer_eps=1e-3,
                       N_inner=30_000, verbose=True,
                       tech=None, gamma=0.0):
    """
    Simultaneous (w_g, L_g) equilibrium with logit labour mobility.

    Inner: FOC w_g = (Γ_g / L_g)^(1/λ) given current L_g.
    Outer: L_g ∝ w_g^θ · L_g^BLS  (logit anchored to BLS prior).

    theta=0 → L_g fixed at BLS shares (recovers geo_baseline).
    """
    fam_list  = list(families_init.keys())
    L_bls_arr = np.array([families_init[g]['L'] for g in fam_list])
    _label    = ("no tech" if tech is None or tech.get('A', 0) == 0
                 else f"p_K=({tech['p_xy'][0]:+.3f},{tech['p_xy'][1]:+.3f})")
    print(f"  geo_eq_v2  theta={theta}  mu={mu}  {_label}")

    fam    = copy.deepcopy(families_init)
    w_prev = None

    for outer_it in range(max_outer):
        use_N = N_inner if outer_it < max_outer - 1 else N
        inner = geo_baseline(fam, N=use_N, seed=seed, lam=lam,
                             warm_start=(outer_it > 0),
                             tech=tech, gamma=gamma)
        w     = {g: inner['families'][g]['w'] for g in fam_list}
        w_arr = np.array([w[g] for g in fam_list])

        log_unnorm  = (theta * np.log(np.maximum(w_arr, 1e-12))
                       + np.log(np.maximum(L_bls_arr, 1e-12)))
        log_unnorm -= log_unnorm.max()
        unnorm      = np.exp(log_unnorm)
        L_new_arr   = unnorm / unnorm.sum()

        L_arr     = np.array([fam[g]['L'] for g in fam_list])
        L_upd_arr = (1.0 - mu) * L_arr + mu * L_new_arr
        L_upd_arr = np.maximum(L_upd_arr, 1e-6)
        L_upd_arr /= L_upd_arr.sum()
        delta_L    = float(np.max(np.abs(L_upd_arr - L_arr)))

        if verbose:
            delta_w = float(np.max(np.abs(
                w_arr - (w_prev if w_prev is not None
                         else np.ones(len(fam_list))))))
            print(f"  Outer iter {outer_it+1:3d}  ΔL={delta_L:.5f}  "
                  f"max|Δw|={delta_w:.5f}")
        w_prev = w_arr.copy()

        for i, g in enumerate(fam_list):
            fam[g]['L'] = float(L_upd_arr[i])
            fam[g]['w'] = w[g]

        if delta_L < outer_eps:
            print(f"  Converged @ outer iter {outer_it+1}  ΔL={delta_L:.6f}")
            break

    final          = geo_baseline(fam, N=N, seed=seed, lam=lam,
                                  warm_start=True, tech=tech, gamma=gamma)
    final['w_eq']  = {g: final['families'][g]['w'] for g in fam_list}
    final['L_eq']  = {g: fam[g]['L'] for g in fam_list}
    final['L_bls'] = {g: families_init[g]['L'] for g in fam_list}
    return final


def geo_eq_comparison(families_init, tech, gamma=0.0,
                      theta=0.3, N=100_000, N_inner=30_000, mu=0.3,
                      seed=1, lam=1.5):
    """
    Run baseline and post-shock geo_equilibrium_v2; print comparison table.
    Returns (eq_base, eq_post).
    """
    print("\n  ── Baseline equilibrium ──────────────────────────────────")
    eq_base = geo_equilibrium_v2(
        families_init, N=N, N_inner=N_inner, seed=seed, lam=lam,
        theta=theta, mu=mu, tech=None, gamma=0.0)

    print("\n  ── Post-shock equilibrium ────────────────────────────────")
    eq_post = geo_equilibrium_v2(
        families_init, N=N, N_inner=N_inner, seed=seed, lam=lam,
        theta=theta, mu=mu, tech=tech, gamma=gamma)

    fam_list = list(families_init.keys())
    w_base   = eq_base['w_eq'];  w_post = eq_post['w_eq']
    L_base   = eq_base['L_eq'];  L_post = eq_post['L_eq']
    pairs    = sorted(fam_list, key=lambda g: -(w_post[g] / w_base[g] - 1))

    print(f"\n  {'Job family':<48} {'w_base':>6}  {'w_post':>6}  "
          f"{'Δw%':>7}  {'ΔL%':>7}")
    print(f"  {'-'*75}")
    for g in pairs:
        wb = w_base[g]; wp = w_post[g]
        print(f"  {g:<48} {wb:6.3f}  {wp:6.3f}  "
              f"{(wp/wb-1)*100:+6.1f}%  "
              f"{(L_post[g]/L_base[g]-1)*100:+6.1f}%")

    return eq_base, eq_post


# ==============================================================================
# Mobility matrix
# ==============================================================================

def mobility_matrix(families, sigma_c=None, beta=0.0):
    """
    Row-normalised mobility matrix M[g,h].

    Base weight:  M[g,h] = ψ_h(p_g)
    Optional friction penalties:
      sigma_c : distance switching cost  exp(-||p_g - p_h|| / sigma_c)
      beta    : education/entry barrier  exp(-β · e_h)
    """
    fam_list = list(families.keys())
    n = len(fam_list)
    M = np.zeros((n, n))
    for i, g in enumerate(fam_list):
        p_g = families[g]['p_xy']
        for j, h in enumerate(fam_list):
            d_gh = dist(np.array([p_g]), families[h]['p_xy'])[0]
            u    = d_gh / max(families[h]['z'], 1e-12)
            w    = families[h]['A'] * gauss(u)
            if sigma_c is not None:
                w *= math.exp(-d_gh / max(sigma_c, 1e-12))
            if beta != 0.0 and 'edu' in families[h]:
                w *= math.exp(-beta * families[h]['edu'])
            M[i, j] = w
        row_sum = M[i].sum()
        if row_sum > 1e-12:
            M[i] /= row_sum
    return M, fam_list


# ==============================================================================
# Reporting
# ==============================================================================

def print_report(result, case_name):
    """
    Print wage and task-share changes for a tech_shock result.

    When gamma > 0, shows three columns:
      ŵ_post(γ=0)  — pure automation outcome
      ŵ_post       — with task creation
      Δŵ%(γ)       — net reinstatement effect
    """
    fam_list  = list(result['w_post'].keys())
    tech      = result['tech']
    p_K       = tech['p_xy']
    gamma     = result['gamma']
    fams_init = _state.get('families_init', {})

    has_gamma = gamma > 1e-12 and 'w_post_g0' in result

    print(f"\n{'='*68}")
    print(f"  {case_name}")
    print(f"  p_K=({p_K[0]:+.4f},{p_K[1]:+.4f})  z_K={tech['z']:.3f}  "
          f"A_K={tech['A']:.2f}  R={tech['R']:.2f}  γ={gamma:.2f}")
    if has_gamma:
        eff = result.get('gamma_eff', 0.0)
        dGD = result.get('Delta_Gamma_D', 0.0)
        print(f"  ΔΓ^D={dGD:.4f}  γ_eff={eff:.4f}  "
              f"(new task mass = γ·ΔΓ^D = {gamma*dGD:.4f})")
    print(f"{'='*68}")

    if has_gamma:
        print(f"  {'Job family':<43}  {'ŵ_base':>7}  {'γ=0':>7}  "
              f"{'γ>0':>7}  {'Δγ=0%':>7}  {'Δγ%':>7}  {'op%':>5}")
    else:
        print(f"  {'Job family':<43}  {'ŵ_base':>7}  {'ŵ_post':>7}  "
              f"{'Δŵ%':>7}  {'ΔD':>6}  {'op%':>5}")

    changes = sorted(fam_list,
        key=lambda g: -(result['w_post'][g] - result['w_base'][g]) /
                       result['w_base'][g])

    for g in changes:
        omega = fams_init.get(g, {}).get('omega', 1.0)
        w0    = result['w_base'][g]  * omega
        w1    = result['w_post'][g]  * omega
        pct   = 100 * (w1 - w0) / w0
        ops   = 100 * result['op_share'][g]
        mark  = ' ◀' if abs(pct) > 1.0 else ''

        if has_gamma:
            w1g0  = result['w_post_g0'][g] * omega
            pctg0 = 100 * (w1g0 - w0) / w0
            print(f"  {g:<43}  {w0:.3f}   {w1g0:.3f}   {w1:.3f}  "
                  f"{pctg0:+6.1f}%  {pct:+6.1f}%  {ops:4.1f}%{mark}")
        else:
            dd   = 100 * (result['D_post'][g] - result['D_base'][g])
            print(f"  {g:<43}  {w0:.3f}   {w1:.3f}  {pct:+6.1f}%  "
                  f"{dd:+5.2f}  {ops:4.1f}%{mark}")

    total_op = sum(result['op_share'][g] for g in fam_list)
    print(f"\n  Tech-assisted task mass: {100*total_op:.1f}%")
    if has_gamma:
        total_op_g0 = sum(result['op_share_g0'][g] for g in fam_list)
        print(f"  Tech-assisted task mass (γ=0): {100*total_op_g0:.1f}%")