from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.colors as mcolors

from .db import OnetDB
from .elements import pick_scale, pivot_elements

RIASEC_ORDER = ["Realistic","Investigative","Artistic","Social","Enterprising","Conventional"]
AX_DEG = np.array([0, 60, 120, 180, 240, 300], dtype=float)
AX_RAD = np.deg2rad(AX_DEG)

def from_interests(
    db: OnetDB,
    *,
    scale_preference: Sequence[str] = ("OI","IM"),
    rho_percentiles: Tuple[float,float] = (5,95),
    hsv_value: float = 0.95,
    fillna: str = "col_mean",
) -> pd.DataFrame:
    """Derive RIASEC resultant (theta,rho) and colors from O*NET Interests."""
    raw = db.read("Interests")
    scale = pick_scale(raw, scale_preference)

    wide = pivot_elements(raw, scale_id=scale)
    # ensure required columns exist
    miss = [c for c in RIASEC_ORDER if c not in wide.columns]
    if miss:
        raise ValueError(f"Interests pivot missing RIASEC columns: {miss}")

    V = wide[RIASEC_ORDER].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if fillna == "col_mean":
        col_means = np.nanmean(V, axis=0)
        V = np.where(np.isnan(V), col_means, V)

    vx = (V * np.cos(AX_RAD)).sum(axis=1)
    vy = (V * np.sin(AX_RAD)).sum(axis=1)
    theta = (np.arctan2(vy, vx) + 2*np.pi) % (2*np.pi)
    rho   = np.hypot(vx, vy)

    finite = np.isfinite(rho)
    p_lo, p_hi = np.percentile(rho[finite], list(rho_percentiles)) if finite.any() else (0.0, 1.0)
    den = max(float(p_hi - p_lo), 1e-12)
    rho_scaled = np.clip((rho - p_lo) / den, 0, 1)

    dom_dim = np.array(RIASEC_ORDER, dtype=object)[np.argmax(V, axis=1)]

    d = (theta[:, None] - AX_RAD[None, :] + np.pi) % (2*np.pi) - np.pi
    idx = np.argmin(np.abs(d), axis=1)
    profile = np.array(RIASEC_ORDER, dtype=object)[idx]
    profile_center_deg = AX_DEG[idx]

    h = theta / (2*np.pi)
    s = rho_scaled
    v = np.full_like(h, float(hsv_value), dtype=float)
    rgb = mcolors.hsv_to_rgb(np.stack([h, s, v], axis=1))

    out = pd.DataFrame({
        "onet_code": wide["onet_code"].astype("string").str.strip(),
        "scale": scale,
        "onet_theta": theta,
        "onet_rho": rho,
        "onet_rho_scaled": rho_scaled,
        "RIASEC_dom": dom_dim,
        "riasec_profile": profile,
        "riasec_profile_center_deg": profile_center_deg,
        "hex_r": rgb[:, 0],
        "hex_g": rgb[:, 1],
        "hex_b": rgb[:, 2],
    })

    rgb255 = np.clip((out[["hex_r","hex_g","hex_b"]].to_numpy() * 255).round().astype(int), 0, 255)
    out["rgb_hex"] = [f"#{r:02X}{g:02X}{b:02X}" for r,g,b in rgb255]
    return out
