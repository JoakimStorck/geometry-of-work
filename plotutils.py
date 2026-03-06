import numpy as np
from matplotlib.patches import FancyArrowPatch


def place_polar_labels_no_overlap_fast2(
    ax,
    theta_list,
    chi_list,
    label_list,
    *,
    base_offset=0.30,
    fontsize=7,
    text_color=None,
    leader_color=None,
    text_alpha=0.95,
    min_sep_px=2,
    max_iter=60,
    step_r_px=3.0,
    step_t_px=2.0,
    leader_lw=0.6,
    leader_alpha=0.65,
    rlim_pad=0.03,
    keep_inside_axes=True,
    outside_push_px=6.0,
    tangential_nudge_px=2.0,
    edge_softzone=0.10,
    theta_bias_at_edge=2.0,
    return_spring=0.15,
    # randomness
    seed=None,
    init_jitter_px=2.0,
    collision_jitter=0.25,
    safety_iters=12,  # API compat
    # enforce "outside from origin" + optionally expand r-limits
    force_outward=True,
    outward_offset=None,
    expand_rmax_if_needed=True,
    clip_on=True,
    # curved leaderlines
    leader_curved=True,
    leader_curve_base=0.04,
    leader_curve_theta_scale=0.55,
    leader_curve_r_scale=0.25,
    leader_curve_max=0.22,
):
    rng = np.random.RandomState(seed) if seed is not None else None

    th = np.asarray(theta_list, dtype=float)
    rr = np.asarray(chi_list, dtype=float)
    labels = list(label_list)

    ok = np.isfinite(th) & np.isfinite(rr)
    th, rr = th[ok], rr[ok]
    keep_idx = np.flatnonzero(ok)
    labels = [labels[i] for i in keep_idx]
    n = len(labels)
    if n == 0:
        return [], []

    # -------------------------
    # Style normalization
    # -------------------------
    def _style_list(x, default):
        if x is None:
            return [default] * n
        if isinstance(x, (str, int, float)):
            return [x] * n
        if isinstance(x, (list, tuple, np.ndarray)):
            if len(x) != n:
                raise ValueError(f"Style list length mismatch: expected {n}, got {len(x)}")
            return list(x)
        return [x] * n

    text_color_l = _style_list(text_color, "black")
    leader_color_l = _style_list(leader_color, "gray")
    text_alpha_l = _style_list(text_alpha, 0.95)

    # -------------------------
    # Clusters by color
    # -------------------------
    cluster_key = []
    for i in range(n):
        c = text_color_l[i]
        if c is None:
            c = leader_color_l[i]
        if c is None:
            c = "__default__"
        cluster_key.append(str(c))

    uniq = {}
    cluster_id = np.empty(n, dtype=int)
    for i, k in enumerate(cluster_key):
        if k not in uniq:
            uniq[k] = len(uniq)
        cluster_id[i] = uniq[k]
    nC = len(uniq)

    # -------------------------
    # Geometry / limits
    # -------------------------
    out_off = float(base_offset if outward_offset is None else outward_offset)

    rmin, rmax = ax.get_ylim()
    if expand_rmax_if_needed and force_outward:
        need_rmax = float(np.nanmax(rr) + out_off + (float(rlim_pad) if rlim_pad is not None else 0.0))
        if need_rmax > rmax:
            ax.set_ylim(rmin, need_rmax)
            rmin, rmax = ax.get_ylim()

    rmin_eff = float(rmin)
    rmax_eff = float(rmax - float(rlim_pad)) if rlim_pad is not None else float(rmax)

    fig = ax.figure
    trans = ax.transData
    ax_bbox = ax.bbox

    canvas = fig.canvas
    renderer = canvas.get_renderer()
    if renderer is None:
        canvas.draw()
        renderer = canvas.get_renderer()

    # -------------------------
    # Screen-space local basis
    # -------------------------
    def _local_eps_r():
        return max(1e-4, 0.002 * (rmax_eff - rmin_eff + 1e-9))

    def _screen_basis(theta, r):
        eps_r = _local_eps_r()
        eps_t = 1e-3
        x0, y0 = trans.transform((float(theta), float(r)))
        xr, yr = trans.transform((float(theta), float(min(r + eps_r, rmax_eff))))
        xt, yt = trans.transform((float(theta + eps_t), float(r)))
        er0 = (xr - x0) / max(1e-9, eps_r)
        er1 = (yr - y0) / max(1e-9, eps_r)
        et0 = (xt - x0) / max(1e-9, eps_t)
        et1 = (yt - y0) / max(1e-9, eps_t)
        return (er0, er1), (et0, et1)

    def _local_scales(theta, r):
        (er0, er1), (et0, et1) = _screen_basis(theta, r)
        px_per_dr = float(np.hypot(er0, er1))
        px_per_dth = float(np.hypot(et0, et1))
        return max(1e-6, px_per_dr), max(1e-6, px_per_dth)

    # -------------------------
    # Floors / targets
    # -------------------------
    if force_outward:
        r_floor = np.clip(rr + out_off, rmin_eff, rmax_eff)
    else:
        r_floor = np.full_like(rr, rmin_eff, dtype=float)

    r_target = r_floor.copy()
    th_lab = th.copy()
    r_lab = r_target.copy()

    # -------------------------
    # Create texts
    # -------------------------
    texts = []
    for i in range(n):
        t = float(th_lab[i])
        r = float(r_lab[i])
        txt = ax.text(
            t,
            r,
            labels[i],
            fontsize=fontsize,
            ha=("left" if np.cos(t) >= 0 else "right"),
            va="center",
            color=text_color_l[i],
            alpha=float(text_alpha_l[i]),
            zorder=5,
            clip_on=bool(clip_on),
        )
        texts.append(txt)

    # Need one draw so text extents are valid
    canvas.draw()
    renderer = canvas.get_renderer()

    # -------------------------
    # Initial jitter
    # -------------------------
    if rng is not None and init_jitter_px and init_jitter_px > 0:
        px_per_dr0, px_per_dth0 = _local_scales(float(np.nanmean(th_lab)), float(np.nanmean(r_lab)))
        j_r = rng.uniform(-1.0, 1.0, size=n) * (0.4 * float(init_jitter_px)) / px_per_dr0
        j_t = rng.uniform(-1.0, 1.0, size=n) * (float(init_jitter_px)) / px_per_dth0
        r_lab = np.clip(r_lab + j_r, rmin_eff, rmax_eff)
        th_lab = th_lab + j_t
        if force_outward:
            r_lab = np.maximum(r_lab, r_floor)
        for i in range(n):
            t = float(th_lab[i])
            r = float(r_lab[i])
            texts[i].set_position((t, r))
            texts[i].set_ha("left" if np.cos(t) >= 0 else "right")

    # -------------------------
    # Bbox cache + mid-y cache (INKREMENTELL)
    # -------------------------
    def _bbox_one(i):
        return texts[i].get_window_extent(renderer=renderer).expanded(1.06, 1.10)

    bbs = [None] * n
    mid_y = np.zeros(n, dtype=float)

    def _update_one_bbox(i):
        bb = _bbox_one(i)
        bbs[i] = bb
        mid_y[i] = 0.5 * float(bb.y0 + bb.y1)
        return bb

    def _update_all_bboxes():
        for i in range(n):
            _update_one_bbox(i)

    def _order():
        return np.argsort(mid_y)

    # -------------------------
    # Overlap utilities
    # -------------------------
    def _overlap_amount(bb1, bb2, sep_px=0.0):
        sep = float(sep_px)
        x_ov = min(bb1.x1 + sep, bb2.x1) - max(bb1.x0 - sep, bb2.x0)
        y_ov = min(bb1.y1 + sep, bb2.y1) - max(bb1.y0 - sep, bb2.y0)
        return float(max(0.0, x_ov)), float(max(0.0, y_ov))

    def _overlap_any(bb1, bb2, sep_px=0.0):
        ovx, ovy = _overlap_amount(bb1, bb2, sep_px)
        return (ovx > 0.0) and (ovy > 0.0)

    def _overlap_score(bb1, bb2, sep_px=0.0, *, w_y=1.30, w_x=0.20, mode="lin"):
        ovx, ovy = _overlap_amount(bb1, bb2, sep_px)
        if mode == "area":
            return float(ovx * ovy)
        return float(w_y * ovy + w_x * ovx)

    def _need_px_from_overlap(bb1, bb2, sep_px, *, w_y=1.30, w_x=0.20, inflate=1.0):
        ovx, ovy = _overlap_amount(bb1, bb2, sep_px)
        need = float(sep_px + w_y * ovy + w_x * ovx)
        return float(inflate * need)

    # -------------------------
    # Wrap helpers
    # -------------------------
    def _wrap_pi(x):
        return (x + np.pi) % (2.0 * np.pi) - np.pi

    def _wrap_2pi(x):
        two_pi = 2.0 * np.pi
        return (x + two_pi) % two_pi

    # -------------------------
    # Keep-inside (inline 2x2 solve)
    # -------------------------
    def _push_inside(bb, theta, r, i, px_per_dr_ref=None, px_per_dth_ref=None):
        if not keep_inside_axes:
            return theta, r, False

        dx = 0.0
        dy = 0.0
        if bb.x0 < ax_bbox.x0:
            dx += (ax_bbox.x0 - bb.x0)
        if bb.x1 > ax_bbox.x1:
            dx -= (bb.x1 - ax_bbox.x1)
        if bb.y0 < ax_bbox.y0:
            dy += (ax_bbox.y0 - bb.y0)
        if bb.y1 > ax_bbox.y1:
            dy -= (bb.y1 - ax_bbox.y1)

        if dx == 0.0 and dy == 0.0:
            return theta, r, False

        (er0, er1), (et0, et1) = _screen_basis(float(theta), float(r))
        A00, A01 = float(er0), float(et0)
        A10, A11 = float(er1), float(et1)
        b0, b1 = float(dx), float(dy)

        det = A00 * A11 - A01 * A10
        if abs(det) < 1e-10:
            if px_per_dr_ref is None or px_per_dth_ref is None:
                px_per_dr, px_per_dth = _local_scales(theta, r)
            else:
                px_per_dr, px_per_dth = float(px_per_dr_ref), float(px_per_dth_ref)

            out = float(np.hypot(dx, dy))
            dr_in = min(float(outside_push_px), out) / max(1e-9, px_per_dr)
            r2 = max(rmin_eff, float(r) - dr_in)
            if force_outward:
                r2 = max(float(r_floor[i]), r2)

            sign = 1.0 if np.cos(theta) >= 0 else -1.0
            dth = sign * (float(tangential_nudge_px) / max(1e-9, px_per_dth))
            return float(theta + dth), float(min(r2, rmax_eff)), True

        inv_det = 1.0 / det
        dr_corr = (b0 * A11 - b1 * A01) * inv_det
        dth_corr = (-b0 * A10 + b1 * A00) * inv_det

        if px_per_dr_ref is None or px_per_dth_ref is None:
            px_per_dr_loc, px_per_dth_loc = _local_scales(theta, r)
        else:
            px_per_dr_loc, px_per_dth_loc = float(px_per_dr_ref), float(px_per_dth_ref)

        max_dr = float(max(1.0, outside_push_px) / max(1e-9, px_per_dr_loc))
        max_dth = float(max(1.0, tangential_nudge_px) / max(1e-9, px_per_dth_loc))

        dr_corr = float(np.clip(dr_corr, -2.0 * max_dr, 2.0 * max_dr))
        dth_corr = float(np.clip(dth_corr, -2.0 * max_dth, 2.0 * max_dth))

        r2 = float(np.clip(float(r) + dr_corr, rmin_eff, rmax_eff))
        if force_outward:
            r2 = max(float(r_floor[i]), r2)
        t2 = float(theta + dth_corr)
        return t2, r2, True

    # -------------------------
    # Band near periphery metrics
    # -------------------------
    bbs0 = [t.get_window_extent(renderer=renderer) for t in texts]
    w_px = np.array([bb.width for bb in bbs0], dtype=float)
    h_px = np.array([bb.height for bb in bbs0], dtype=float)

    r_push = rmin_eff + 0.82 * (rmax_eff - rmin_eff)   # mindre “ytterkrans”
    r_band = 0.55 * (rmax_eff - rmin_eff)              # mycket tjockare band    
        
    #r_push = float(rmin_eff + 0.90 * (rmax_eff - rmin_eff))
    #r_band = float(0.28 * (rmax_eff - rmin_eff))
    if n >= 70 or nC >= 4:
        r_band = float(0.34 * (rmax_eff - rmin_eff))

    r_low = float(np.clip(r_push - 0.5 * r_band, rmin_eff, rmax_eff))
    r_high = float(np.clip(r_push + 0.5 * r_band, rmin_eff, rmax_eff))

    px_per_dr_band, px_per_dth_band = _local_scales(float(np.nanmean(th)), float(r_push))
    band_px_avail = float((r_high - r_low) * px_per_dr_band)
    H_med = float(np.nanmedian(h_px) * 1.15)

    max_rows = max(1, int(band_px_avail // max(1.0, H_med)))
    max_rows = min(max_rows, 6)

    PAD_PX = 6.0
    GAP_RAD = np.deg2rad(2.5)

    # -------------------------
    # Required wedge per cluster
    # -------------------------
    req_wedge = np.zeros(nC, dtype=float)
    for c in range(nC):
        idx = np.flatnonzero(cluster_id == c)
        m = len(idx)
        if m == 0:
            req_wedge[c] = np.deg2rad(12.0)
            continue

        dth_need = (w_px[idx] + PAD_PX) / max(1e-9, px_per_dth_band)
        R = int(min(max_rows, max(1, int(np.ceil(m / 6.0)))))
        R = max(1, min(R, max_rows))

        rows = np.zeros(R, dtype=float)
        for x in np.sort(dth_need)[::-1]:
            j = int(np.argmin(rows))
            rows[j] += float(x)
        req_wedge[c] = float(rows.max())

    MIN_WEDGE = np.deg2rad(18.0)
    MAX_WEDGE = np.deg2rad(160.0)
    req_wedge = np.clip(req_wedge, MIN_WEDGE, MAX_WEDGE)

    two_pi = 2.0 * np.pi
    total_demand = float(req_wedge.sum() + nC * GAP_RAD)
    if total_demand > 0.98 * two_pi:
        scale = (0.98 * two_pi - nC * GAP_RAD) / max(1e-9, float(req_wedge.sum()))
        scale = max(0.25, min(1.0, scale))
        req_wedge = np.clip(req_wedge * scale, MIN_WEDGE * 0.75, MAX_WEDGE)

    # -------------------------
    # Centroids -> theta_target
    # -------------------------
    xy_anchor = np.asarray([trans.transform((float(th[i]), float(rr[i]))) for i in range(n)], dtype=float)
    g_cent = xy_anchor.mean(axis=0)

    c_cent = np.zeros((nC, 2), dtype=float)
    c_cnt = np.zeros(nC, dtype=float)
    for i in range(n):
        c = int(cluster_id[i])
        c_cent[c] += xy_anchor[i]
        c_cnt[c] += 1.0
    c_cnt = np.maximum(1.0, c_cnt)
    c_cent /= c_cnt[:, None]

    v = g_cent[None, :] - c_cent
    vnorm = np.sqrt((v * v).sum(axis=1)) + 1e-9
    vhat = v / vnorm[:, None]
    theta_target = np.arctan2(vhat[:, 1], vhat[:, 0])

    r_probe = float(np.nanmean(rr))
    eps = 1e-2
    for c in range(nC):
        vx, vy = float(v[c, 0]), float(v[c, 1])
        if (vx * vx + vy * vy) < 1e-12:
            continue
        t = float(theta_target[c])
        x0, y0 = trans.transform((t, r_probe))
        x1, y1 = trans.transform((t + eps, r_probe))
        dtx, dty = (x1 - x0), (y1 - y0)
        if (dtx * vx + dty * vy) < 0.0:
            theta_target[c] = t + np.pi
    theta_target = _wrap_pi(theta_target)

    tiny = vnorm < 1e-6
    if np.any(tiny):
        s_sum = np.bincount(cluster_id, weights=np.sin(th), minlength=nC)
        c_sum = np.bincount(cluster_id, weights=np.cos(th), minlength=nC)
        theta_mean_anchor = np.arctan2(s_sum, c_sum)
        theta_target[tiny] = theta_mean_anchor[tiny]

    # -------------------------
    # Wedge placement
    # -------------------------
    theta0 = _wrap_2pi(theta_target)
    orderC = np.argsort(theta0)
    wid = req_wedge[orderC].copy()

    total_wid = float(wid.sum())
    total_gap = float(nC * GAP_RAD)
    slack = two_pi - (total_wid + total_gap)
    gap_use = GAP_RAD
    slack = two_pi - (total_wid + nC * gap_use)
    if slack > 0:
        wid = wid + slack * (wid / max(1e-9, wid.sum()))

    cent_target_sorted = theta0[orderC].copy()
    left0 = float(cent_target_sorted[0] - 0.5 * wid[0])

    cent_sorted = np.zeros(nC, dtype=float)
    left = left0
    for k in range(nC):
        cent_sorted[k] = left + 0.5 * wid[k]
        left = left + wid[k] + gap_use
    cent_sorted = _wrap_2pi(cent_sorted)

    wedge_left = np.zeros(nC, dtype=float)
    wedge_right = np.zeros(nC, dtype=float)
    for k, c in enumerate(orderC):
        c_center = float(cent_sorted[k])
        w = float(wid[k])
        wedge_left[c] = c_center - 0.5 * w
        wedge_right[c] = c_center + 0.5 * w

    wedge_left = _wrap_2pi(wedge_left)
    wedge_right = _wrap_2pi(wedge_right)
    wedge_right = np.where(wedge_right < wedge_left, wedge_right + two_pi, wedge_right)

    GUARD_MIN = np.deg2rad(2.0)
    GUARD_MAX = np.deg2rad(6.0)
    GUARD = np.clip(0.02 * (wedge_right - wedge_left), np.deg2rad(0.6), np.deg2rad(2.0))

    wedge_left = wedge_left + GUARD
    wedge_right = wedge_right - GUARD
    wedge_width = np.maximum(1e-3, wedge_right - wedge_left)

    # -------------------------
    # Per-cluster radial band
    # -------------------------
    r_low_c = np.full(nC, r_low, dtype=float)
    r_high_c = np.full(nC, r_high, dtype=float)

    for c in range(nC):
        m = int(np.sum(cluster_id == c))
        if m <= 1:
            continue
        R = int(min(max_rows, max(1, int(np.ceil(m / 7.0)))))
        R = max(1, min(R, max_rows))
        need_px = float(R * H_med)
        crowd = need_px / max(1.0, band_px_avail)
        if crowd > 0.92:
            expand = min(2.4, max(1.0, crowd / 0.92))
            band = float((r_high - r_low) * expand)
            lo = float(np.clip(r_push - 0.5 * band, rmin_eff, rmax_eff))
            hi = float(np.clip(r_push + 0.5 * band, rmin_eff, rmax_eff))
            if hi - lo > 1e-6:
                r_low_c[c] = lo
                r_high_c[c] = hi

    def _expand_cluster_band(c, strength=1.0):
        clo = float(r_low_c[c])
        chi = float(r_high_c[c])
        span = chi - clo
        if span <= 1e-12:
            span = 0.05 * (rmax_eff - rmin_eff)

        add = (0.10 + 0.10 * float(strength)) * (rmax_eff - rmin_eff)
        add = min(add, 0.80 * span + 1e-9)

        new_lo = float(np.clip(clo - 0.5 * add, rmin_eff, rmax_eff))
        new_hi = float(np.clip(chi + 0.5 * add, rmin_eff, rmax_eff))

        if new_hi - new_lo > span + 1e-6:
            r_low_c[c] = new_lo
            r_high_c[c] = new_hi

    # -------------------------
    # Dynamics constants
    # -------------------------
    K_THETA_TO_TARGET = 0.10
    K_R_TO_TARGET = 0.12
    MAX_DTH = 0.08
    MAX_DR = 0.06

    px_per_dr0, px_per_dth0 = _local_scales(float(np.nanmean(th_lab)), float(np.nanmean(r_lab)))
    dr_step = float(step_r_px) / max(1e-9, px_per_dr0)
    dth_step = float(step_t_px) / max(1e-9, px_per_dth0)

    # -------------------------
    # Per-label targets
    # -------------------------
    u = np.zeros(n, dtype=float)
    for c in range(nC):
        idx = np.flatnonzero(cluster_id == c)
        m = len(idx)
        if m <= 1:
            u[idx] = 0.5
            continue
        d = (th[idx] - theta_target[c] + np.pi) % (2.0 * np.pi) - np.pi
        o = idx[np.argsort(d)]
        u[o] = np.linspace(0.0, 1.0, num=m)

    theta_lbl_target = wedge_left[cluster_id] + wedge_width[cluster_id] * u

    r_lbl_target = np.zeros(n, dtype=float)
    for c in range(nC):
        idx = np.flatnonzero(cluster_id == c)
        m = len(idx)
        if m == 0:
            continue

        R = int(min(max_rows, max(1, int(np.ceil(m / 7.0)))))
        R = max(1, min(R, max_rows))
        C = int(np.ceil(m / R))

        d = (th[idx] - theta_target[c] + np.pi) % (2.0 * np.pi) - np.pi
        o = idx[np.argsort(d)]

        lo = float(r_low_c[c])
        hi = float(r_high_c[c])

        for k, ii in enumerate(o):
            row = k % R
            col = k // R
            base = 0.5 if R == 1 else (row / (R - 1))
            jitter = 0.0 if C <= 1 else (col / (C - 1) - 0.5)
            micro = 0.35 / max(1.0, R)
            r_u = float(np.clip(base + micro * jitter, 0.0, 1.0))
            r_lbl_target[ii] = lo + (hi - lo) * r_u

    r_lbl_target = np.maximum(r_lbl_target, r_floor)

    # -------------------------
    # Helpers
    # -------------------------
    def _write_text(i):
        t = float(th_lab[i])
        r = float(r_lab[i])
        texts[i].set_position((t, r))
        texts[i].set_ha("left" if np.cos(t) >= 0 else "right")

    def _write_all_texts():
        for i in range(n):
            _write_text(i)

    sep = float(min_sep_px)

    def _find_overlap_neighbor(idx_sorted, pos):
        ii = int(idx_sorted[pos])
        bb_i = bbs[ii]
        y0 = bb_i.y0 - sep
        y1 = bb_i.y1 + sep

        j = pos - 1
        while j >= 0:
            jj = int(idx_sorted[j])
            bb_j = bbs[jj]
            if bb_j.y1 < y0:
                break
            if _overlap_any(bb_i, bb_j, sep):
                return jj
            j -= 1

        j = pos + 1
        while j < len(idx_sorted):
            jj = int(idx_sorted[j])
            bb_j = bbs[jj]
            if bb_j.y0 > y1:
                break
            if _overlap_any(bb_i, bb_j, sep):
                return jj
            j += 1
        return None

    def _any_overlap(idx_sorted):
        for pos in range(n):
            if _find_overlap_neighbor(idx_sorted, pos) is not None:
                return True
        return False

    # -------------------------
    # Initialize caches
    # -------------------------
    _write_all_texts()
    _update_all_bboxes()

    # -------------------------
    # Main iterations
    # -------------------------
    MAX_PASSES = 3
    MAX_FIX_PER_LABEL = 6

    for _it in range(int(max_iter)):
        dth = K_THETA_TO_TARGET * ((theta_lbl_target - th_lab + np.pi) % (2.0 * np.pi) - np.pi)
        dr = K_R_TO_TARGET * (r_lbl_target - r_lab)

        dth = np.clip(dth, -MAX_DTH, MAX_DTH)
        dr = np.clip(dr, -MAX_DR, MAX_DR)

        th_lab = th_lab + dth
        r_lab = r_lab + dr

        r_lab = np.clip(r_lab, rmin_eff, rmax_eff)
        if force_outward:
            r_lab = np.maximum(r_lab, r_floor)

        over = r_lab > r_target
        if np.any(over):
            r_lab2 = r_lab.copy()
            r_lab2[over] = r_lab[over] - return_spring * (r_lab[over] - r_target[over])
            if force_outward:
                r_lab2 = np.maximum(r_lab2, r_floor)
            r_lab = np.clip(r_lab2, rmin_eff, rmax_eff)

        # everyone moved -> full refresh once
        _write_all_texts()
        _update_all_bboxes()

        idx_sorted = _order()
        if not _any_overlap(idx_sorted):
            break

        overlap_hits = np.zeros(nC, dtype=int)

        for _pass in range(int(MAX_PASSES)):
            pass_moved = False
            idx_sorted = _order()

            for pos in range(n):
                ii = int(idx_sorted[pos])
                fixes = 0

                while fixes < int(MAX_FIX_PER_LABEL):
                    bb_i = bbs[ii]

                    t2, r2, did = _push_inside(bb_i, float(th_lab[ii]), float(r_lab[ii]), ii, px_per_dr_band, px_per_dth_band)
                    if did:
                        th_lab[ii], r_lab[ii] = float(t2), float(r2)
                        _write_text(ii)
                        _update_one_bbox(ii)
                        bb_i = bbs[ii]
                        pass_moved = True

                    jj = _find_overlap_neighbor(idx_sorted, pos)
                    if jj is None:
                        break

                    overlap_hits[int(cluster_id[ii])] += 1

                    need_px = _need_px_from_overlap(bb_i, bbs[jj], sep, w_y=1.15, w_x=0.35, inflate=1.0)
                    yi = 0.5 * float(bb_i.y0 + bb_i.y1)
                    yj = 0.5 * float(bbs[jj].y0 + bbs[jj].y1)
                    y_sign = -1.0 if yi < yj else 1.0

                    (er0, er1), (et0, et1) = _screen_basis(float(th_lab[ii]), float(r_lab[ii]))
                    A00, A01 = float(er0), float(et0)
                    A10, A11 = float(er1), float(et1)
                    det = A00 * A11 - A01 * A10

                    if abs(det) < 1e-10:
                        dr_need = float(y_sign) * float(need_px) / max(1e-6, px_per_dr_band)
                        dth_need = 0.0
                    else:
                        b1 = float(y_sign) * float(need_px)
                        inv_det = 1.0 / det
                        dr_need = (-b1 * A01) * inv_det
                        dth_need = ( b1 * A00) * inv_det

                    dr_need = float(np.clip(dr_need, -2.5 * dr_step, 2.5 * dr_step))
                    dth_need = float(np.clip(dth_need, -1.5 * dth_step, 1.5 * dth_step))

                    c = int(cluster_id[ii])
                    clo = float(r_low_c[c])
                    chi = float(r_high_c[c])

                    r_try = float(np.clip(float(r_lab[ii]) + dr_need, rmin_eff, rmax_eff))
                    if force_outward:
                        r_try = max(float(r_floor[ii]), r_try)

                    r_clamped = float(np.clip(r_try, max(rmin_eff, clo), min(rmax_eff, chi)))
                    hit_band = abs(r_clamped - r_try) > 1e-10
                    if hit_band:
                        _expand_cluster_band(c, strength=1.0)
                        clo = float(r_low_c[c])
                        chi = float(r_high_c[c])
                        r_clamped = float(np.clip(r_try, clo, chi))

                    th_lab[ii] = float(th_lab[ii]) + dth_need
                    r_lab[ii] = float(r_clamped)

                    if rng is not None and collision_jitter > 0 and rng.rand() < collision_jitter:
                        th_lab[ii] += (1.0 if np.cos(float(th_lab[ii])) >= 0 else -1.0) * (0.03 * dth_step)

                    _write_text(ii)
                    _update_one_bbox(ii)

                    pass_moved = True
                    fixes += 1

            if not pass_moved:
                break

        for c in range(nC):
            if overlap_hits[c] >= 6:
                _expand_cluster_band(c, strength=1.4)

        idx_sorted = _order()
        if not _any_overlap(idx_sorted):
            break

    # -------------------------
    # Strong cleanup (FULL QUALITY, but cached)
    # -------------------------
    def _any_overlaps_and_hits():
        idx_sorted = _order()
        hits = np.zeros(nC, dtype=int)
        any_ov = False
        for pos in range(n):
            ii = int(idx_sorted[pos])
            jj = _find_overlap_neighbor(idx_sorted, pos)
            if jj is None:
                continue
            any_ov = True
            hits[int(cluster_id[ii])] += 1
        return any_ov, hits

    any_ov, _hits = _any_overlaps_and_hits()
    if any_ov:
        MAX_CLEAN_ROUNDS = 4
        MAX_CLEAN_PASSES = 12
        MAX_FIX_PER_LABEL_C = 12
        MAX_PAIR_TRIES = 4
        EXPAND_ON_STUCK_TRY = 2

        def _apply_y_move(k, y_sign, need_px, *, dr_clip=6.0, dth_clip=3.5):
            t = float(th_lab[k])
            r = float(r_lab[k])

            (er0, er1), (et0, et1) = _screen_basis(t, r)
            A00, A01 = float(er0), float(et0)
            A10, A11 = float(er1), float(et1)
            det = A00 * A11 - A01 * A10

            if abs(det) < 1e-10:
                dr_need = float(y_sign) * float(need_px) / max(1e-6, px_per_dr_band)
                dth_need = 0.0
            else:
                b1 = float(y_sign) * float(need_px)
                inv_det = 1.0 / det
                dr_need = (-b1 * A01) * inv_det
                dth_need = ( b1 * A00) * inv_det

            dr_need = float(np.clip(dr_need, -dr_clip * dr_step, dr_clip * dr_step))
            dth_need = float(np.clip(dth_need, -dth_clip * dth_step, dth_clip * dth_step))

            c = int(cluster_id[k])
            clo = float(r_low_c[c])
            chi = float(r_high_c[c])

            r_try = float(np.clip(r + dr_need, rmin_eff, rmax_eff))
            if force_outward:
                r_try = max(float(r_floor[k]), r_try)

            r_clamped = float(np.clip(r_try, max(rmin_eff, clo), min(rmax_eff, chi)))
            hit_band = abs(r_clamped - r_try) > 1e-10

            th_lab[k] = float(t + dth_need)
            r_lab[k] = float(r_clamped)
            _write_text(k)
            _update_one_bbox(k)
            return bool(hit_band)

        def _apply_x_move(k, x_sign, need_px, *, dr_clip=6.0, dth_clip=3.5):
            """
            Move label k to achieve roughly delta screen ~= (x_sign*need_px, 0).
            Useful near N/S where y-push is mostly radial and doesn't separate in x.
            """
            t = float(th_lab[k])
            r = float(r_lab[k])

            (er0, er1), (et0, et1) = _screen_basis(t, r)
            A00, A01 = float(er0), float(et0)
            A10, A11 = float(er1), float(et1)
            det = A00 * A11 - A01 * A10

            if abs(det) < 1e-10:
                # fallback: pure tangential (theta) nudge using local scale
                _, px_per_dth_loc = _local_scales(t, r)
                dr_need = 0.0
                dth_need = float(x_sign) * float(need_px) / max(1e-6, px_per_dth_loc)
            else:
                # Solve A*[dr, dth] = [x_sign*need_px, 0]
                b0 = float(x_sign) * float(need_px)
                b1 = 0.0
                inv_det = 1.0 / det
                dr_need = ( b0 * A11 - b1 * A01) * inv_det
                dth_need = (-b0 * A10 + b1 * A00) * inv_det

            dr_need = float(np.clip(dr_need, -dr_clip * dr_step, dr_clip * dr_step))
            dth_need = float(np.clip(dth_need, -dth_clip * dth_step, dth_clip * dth_step))

            c = int(cluster_id[k])
            clo = float(r_low_c[c])
            chi = float(r_high_c[c])

            r_try = float(np.clip(r + dr_need, rmin_eff, rmax_eff))
            if force_outward:
                r_try = max(float(r_floor[k]), r_try)

            r_clamped = float(np.clip(r_try, max(rmin_eff, clo), min(rmax_eff, chi)))
            hit_band = abs(r_clamped - r_try) > 1e-10

            th_lab[k] = float(t + dth_need)
            r_lab[k] = float(r_clamped)
            _write_text(k)
            _update_one_bbox(k)
            return bool(hit_band)
            
        def _resolve_pair_best(ii, jj, need_px, y_sign_i, *, dr_clip=6.0, dth_clip=3.5):
            t_i0, r_i0 = float(th_lab[ii]), float(r_lab[ii])
            t_j0, r_j0 = float(th_lab[jj]), float(r_lab[jj])
            bb_i0, bb_j0 = bbs[ii], bbs[jj]
            s0 = _overlap_score(bb_i0, bb_j0, sep, w_y=1.30, w_x=0.20, mode="lin")
            if s0 <= 0.0:
                return False, False

            best_s = s0
            best_state = None

            def _rollback():
                th_lab[ii], r_lab[ii] = t_i0, r_i0
                th_lab[jj], r_lab[jj] = t_j0, r_j0
                _write_text(ii)
                _write_text(jj)
                bbs[ii], bbs[jj] = bb_i0, bb_j0
                mid_y[ii] = 0.5 * float(bb_i0.y0 + bb_i0.y1)
                mid_y[jj] = 0.5 * float(bb_j0.y0 + bb_j0.y1)

            def _commit_if_better():
                nonlocal best_s, best_state
                s = _overlap_score(bbs[ii], bbs[jj], sep, w_y=1.30, w_x=0.20, mode="lin")
                if s < best_s:
                    best_s = s
                    best_state = (float(th_lab[ii]), float(r_lab[ii]), float(th_lab[jj]), float(r_lab[jj]))

            hit_i = _apply_y_move(ii, float(y_sign_i), need_px, dr_clip=dr_clip, dth_clip=dth_clip)
            _commit_if_better()
            _rollback()

            hit_j = _apply_y_move(jj, -float(y_sign_i), need_px, dr_clip=dr_clip, dth_clip=dth_clip)
            _commit_if_better()
            _rollback()

            hit_i2 = _apply_y_move(ii, float(y_sign_i), 0.55 * need_px, dr_clip=dr_clip, dth_clip=dth_clip)
            hit_j2 = _apply_y_move(jj, -float(y_sign_i), 0.55 * need_px, dr_clip=dr_clip, dth_clip=dth_clip)
            _commit_if_better()
            _rollback()

            hit_i3 = _apply_y_move(ii, float(y_sign_i), 0.55 * need_px, dr_clip=dr_clip, dth_clip=dth_clip)
            hit_j3 = _apply_y_move(jj, -float(y_sign_i), 0.55 * need_px, dr_clip=dr_clip, dth_clip=dth_clip)
            jitter = 0.35 * float(dth_step)
            if rng is not None:
                jitter *= (1.0 if rng.rand() < 0.5 else -1.0)
            else:
                jitter *= (1.0 if np.cos(t_i0) >= 0 else -1.0)
            th_lab[ii] = float(th_lab[ii]) + float(jitter)
            _write_text(ii)
            _update_one_bbox(ii)
            _commit_if_better()
            _rollback()

            if best_state is None:
                return True, (hit_i or hit_j or hit_i2 or hit_j2 or hit_i3 or hit_j3)

            th_lab[ii], r_lab[ii], th_lab[jj], r_lab[jj] = best_state
            _write_text(ii)
            _write_text(jj)
            _update_one_bbox(ii)
            _update_one_bbox(jj)

            hit_any_band = False
            c_i = int(cluster_id[ii])
            c_j = int(cluster_id[jj])
            eps_r = 1e-6
            if abs(float(r_lab[ii]) - float(r_low_c[c_i])) < eps_r or abs(float(r_lab[ii]) - float(r_high_c[c_i])) < eps_r:
                hit_any_band = True
            if abs(float(r_lab[jj]) - float(r_low_c[c_j])) < eps_r or abs(float(r_lab[jj]) - float(r_high_c[c_j])) < eps_r:
                hit_any_band = True

            return True, hit_any_band

        for _round in range(int(MAX_CLEAN_ROUNDS)):
            for _pass in range(int(MAX_CLEAN_PASSES)):
                idx_sorted = _order()
                any_fix = False

                for pos in range(n):
                    ii = int(idx_sorted[pos])
                    fixes = 0

                    while fixes < int(MAX_FIX_PER_LABEL_C):
                        bb_i = bbs[ii]

                        t2, r2, did = _push_inside(bb_i, float(th_lab[ii]), float(r_lab[ii]), ii, px_per_dr_band, px_per_dth_band)
                        if did:
                            th_lab[ii], r_lab[ii] = float(t2), float(r2)
                            _write_text(ii)
                            _update_one_bbox(ii)
                            any_fix = True

                        jj = _find_overlap_neighbor(idx_sorted, pos)
                        if jj is None:
                            break

                        need_px = _need_px_from_overlap(bbs[ii], bbs[jj], sep, w_y=1.30, w_x=0.20, inflate=1.0)
                        yi = 0.5 * float(bbs[ii].y0 + bbs[ii].y1)
                        yj = 0.5 * float(bbs[jj].y0 + bbs[jj].y1)
                        y_sign_i = -1.0 if yi < yj else 1.0

                        stuck_hits = 0
                        hit_band_any = False

                        for _try in range(int(MAX_PAIR_TRIES)):
                            did_something, hit_band = _resolve_pair_best(ii, jj, need_px, y_sign_i)
                            if not did_something:
                                break

                            hit_band_any = hit_band_any or hit_band
                            any_fix = True

                            if not _overlap_any(bbs[ii], bbs[jj], sep):
                                break

                            stuck_hits += 1
                            if stuck_hits >= int(EXPAND_ON_STUCK_TRY):
                                _expand_cluster_band(int(cluster_id[ii]), strength=1.15)
                                _expand_cluster_band(int(cluster_id[jj]), strength=1.15)

                        fixes += 1
                        if hit_band_any and _overlap_any(bbs[ii], bbs[jj], sep):
                            break

                if not any_fix:
                    break

            any_ov2, hits = _any_overlaps_and_hits()
            if not any_ov2:
                break

            worst = int(np.argmax(hits))
            if hits[worst] > 0:
                _expand_cluster_band(worst, strength=1.6)
                for c in range(nC):
                    if c != worst and hits[c] > 0:
                        _expand_cluster_band(c, strength=0.8)

    # -------------------------
    # TOP-RESCUE micro-clean (fixar kvarvarande kollisioner nära N/90°)
    # -------------------------
    def _wrap_pi(x):
        return (x + np.pi) % (2.0 * np.pi) - np.pi

    def _pair_is_near_top(ii, jj, top_center=np.pi/2, top_window=np.deg2rad(25.0)):
        ti = float(th_lab[ii])
        tj = float(th_lab[jj])
        return (abs(_wrap_pi(ti - top_center)) < top_window) and (abs(_wrap_pi(tj - top_center)) < top_window)

    def _bb_mid_x(bb):
        return 0.5 * float(bb.x0 + bb.x1)

    def _find_worst_pair():
        idx_sorted = _order()
        worst = (None, None, 0.0, 0.0)
        for pos in range(n):
            ii = int(idx_sorted[pos])
            jj = _find_overlap_neighbor(idx_sorted, pos)
            if jj is None:
                continue
            s = _overlap_score(bbs[ii], bbs[jj], sep, w_y=1.30, w_x=0.20, mode="lin")
            if s > worst[2]:
                need_px = _need_px_from_overlap(bbs[ii], bbs[jj], sep, w_y=1.45, w_x=0.30, inflate=1.10)
                worst = (ii, jj, s, need_px)
        return worst[0], worst[1], worst[3]

    # Kör bara om något faktiskt överlappar nu
    any_ov3, _ = _any_overlaps_and_hits()
    if any_ov3:
        for _k in range(40):
            ii, jj, need_px = _find_worst_pair()
            if ii is None:
                break

            # Uppdatera bboxar (de är cache:ade, men säkra på att de är färska)
            # (Vanligtvis redan färska efter senaste åtgärd)
            bb_i = bbs[ii]
            bb_j = bbs[jj]
            if not _overlap_any(bb_i, bb_j, sep):
                continue

            if _pair_is_near_top(ii, jj):
                # separera i screen-x istället för screen-y
                xi = _bb_mid_x(bb_i)
                xj = _bb_mid_x(bb_j)
                x_sign_i = -1.0 if xi < xj else 1.0

                # prova split först (minimerar drift), annars full
                hit1 = _apply_x_move(ii, x_sign_i, 0.60 * need_px, dr_clip=8.0, dth_clip=6.0)
                hit2 = _apply_x_move(jj, -x_sign_i, 0.60 * need_px, dr_clip=8.0, dth_clip=6.0)

                if _overlap_any(bbs[ii], bbs[jj], sep):
                    _apply_x_move(ii, x_sign_i, 1.05 * need_px, dr_clip=10.0, dth_clip=7.5)

                if _overlap_any(bbs[ii], bbs[jj], sep):
                    # om pinning i bandet: öppna bandet lite
                    _expand_cluster_band(int(cluster_id[ii]), strength=1.25)
                    _expand_cluster_band(int(cluster_id[jj]), strength=1.25)
            else:
                # fallback: vanlig y-separation (som innan)
                yi = 0.5 * float(bb_i.y0 + bb_i.y1)
                yj = 0.5 * float(bb_j.y0 + bb_j.y1)
                y_sign_i = -1.0 if yi < yj else 1.0
                _resolve_pair_best(ii, jj, need_px, y_sign_i, dr_clip=9.0, dth_clip=5.0)

            # stoppa om vi blev rena
            if not _any_overlaps_and_hits()[0]:
                break
                
    # -------------------------
    # Final clamp
    # -------------------------
    for i in range(n):
        r_lab[i] = float(np.clip(r_lab[i], rmin_eff, rmax_eff))
        if force_outward:
            r_lab[i] = max(float(r_floor[i]), float(r_lab[i]))
        _write_text(i)
        _update_one_bbox(i)

    # -------------------------
    # Leader curvature
    # -------------------------
    def _wrap_dtheta(dth):
        return (dth + np.pi) % (2.0 * np.pi) - np.pi

    cluster_sign = np.array([1.0 if (c % 2 == 0) else -1.0 for c in range(nC)], dtype=float)
    cluster_mag = np.ones(nC, dtype=float)
    if nC > 1:
        sizes = np.bincount(cluster_id, minlength=nC).astype(float)
        sizes = np.maximum(1.0, sizes)
        cluster_mag = np.clip((np.median(sizes) / sizes) ** 0.25, 0.85, 1.25)

    def _leader_rad(t0, r0, t1, r1, c_id):
        dth = _wrap_dtheta(float(t1 - t0))
        dr = float(r1 - r0)

        th_term = min(1.0, abs(dth) / (np.deg2rad(30.0)))
        r_term = min(1.0, max(0.0, dr) / max(1e-9, 2.0 * out_off))

        mag = leader_curve_base + leader_curve_theta_scale * th_term + leader_curve_r_scale * r_term
        mag = float(min(leader_curve_max, max(0.0, mag)))

        if abs(dth) > 1e-6:
            sgn = 1.0 if dth > 0 else -1.0
        else:
            sgn = 1.0 if np.cos(t0) >= 0 else -1.0

        sgn *= float(cluster_sign[int(c_id)])
        mag *= float(cluster_mag[int(c_id)])
        return sgn * mag

    leader_lines = []
    for i, (t0, r0) in enumerate(zip(th, rr)):
        t1, r1 = float(th_lab[i]), float(r_lab[i])
        col = leader_color_l[i]
        if leader_curved:
            rad = _leader_rad(float(t0), float(r0), t1, r1, int(cluster_id[i]))
            patch = FancyArrowPatch(
                posA=(float(t0), float(r0)),
                posB=(t1, r1),
                arrowstyle="-",
                connectionstyle=f"arc3,rad={rad}",
                lw=float(leader_lw),
                alpha=float(leader_alpha),
                color=col,
                zorder=4,
                transform=ax.transData,
            )
            patch.set_clip_on(bool(clip_on))
            ax.add_patch(patch)
            leader_lines.append(patch)
        else:
            ln, = ax.plot([float(t0), t1], [float(r0), r1], color=col, lw=leader_lw, alpha=leader_alpha, zorder=4)
            ln.set_clip_on(bool(clip_on))
            leader_lines.append(ln)

    return texts, leader_lines