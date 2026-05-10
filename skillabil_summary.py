"""
skillabil_summary.py
--------------------
Build the Skills/Abilities summary table that combines headline results from
three analyses (Test 1, angular variance, radial intensification).

Produces three outputs:
  - a pandas DataFrame (return value)
  - a CSV file:   skillabil_summary.csv
  - a LaTeX file: skillabil_summary.tex   (tabularray syntax)

The function expects the result objects from prior analyses; it does not
recompute anything.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import infra
from infra import log

from hc_analysis import AngularSimilarityResult
from radial_specialization import (
    AngularVarianceResult,
    RadialIntensificationResult,
    INTENSIFICATION_METRIC,
)


# ─────────────────────────────────────────────────────────────
# Internal row representation
#   `name_latex` is the raw LaTeX label (no Unicode);
#   `name_unicode` is the same with Unicode for pretty-printing.
# ─────────────────────────────────────────────────────────────

@dataclass
class _Row:
    name_unicode: str
    name_latex: str
    skills_plain: str       # for CSV + pretty-print
    skills_latex: str       # for LaTeX
    abilities_plain: str
    abilities_latex: str
    is_section: bool = False


# ─────────────────────────────────────────────────────────────
# Pull headline numbers
# ─────────────────────────────────────────────────────────────

def _similarity_spearman(test1: AngularSimilarityResult) -> tuple[float, float]:
    from scipy.stats import spearmanr
    rho, p = spearmanr(test1.pairs_df["dxi_rad"], test1.pairs_df["cos_similarity"])
    return float(rho), float(p)


def _intensification_partial(rad: RadialIntensificationResult) -> tuple[float, float]:
    row = rad.statistics_df[
        rad.statistics_df["metric"] == INTENSIFICATION_METRIC
    ].iloc[0]
    return float(row["rho_partial_xi"]), float(row["p_partial_xi"])


def _stars_plain(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _stars_latex(p: float) -> str:
    s = _stars_plain(p)
    return f"^{{{s}}}" if s else ""


def _val_with_stars_latex(value: float, p: float, *, fmt: str = "+.3f") -> str:
    """Format a signed value with significance stars, wrapped in math mode."""
    stars = _stars_latex(p)
    return f"${format(value, fmt)}{stars}$"


# ─────────────────────────────────────────────────────────────
# Build the table
# ─────────────────────────────────────────────────────────────

def build_summary_table(
    *,
    test1_skills: AngularSimilarityResult,
    test1_abilities: AngularSimilarityResult,
    ang_skills: AngularVarianceResult,
    ang_abilities: AngularVarianceResult,
    rad_skills: RadialIntensificationResult,
    rad_abilities: RadialIntensificationResult,
    write_csv: bool = True,
    write_latex: bool = True,
) -> pd.DataFrame:
    """Assemble the Skills/Abilities summary table and write CSV + LaTeX files.

    Returns a flat pandas DataFrame (Statistic, Skills, Abilities).
    Section header rows have empty Skills and Abilities cells.
    """

    # Pull all numbers
    s1_rho, s1_p = _similarity_spearman(test1_skills)
    a1_rho, a1_p = _similarity_spearman(test1_abilities)

    s_med_xi = ang_skills.summary["median_R2_xi"]
    a_med_xi = ang_abilities.summary["median_R2_xi"]
    s_med_chi = ang_skills.summary["median_R2_chi_of_total"]
    a_med_chi = ang_abilities.summary["median_R2_chi_of_total"]

    s_rho, s_p = _intensification_partial(rad_skills)
    a_rho, a_p = _intensification_partial(rad_abilities)

    rows: list[_Row] = [
        _Row(
            name_unicode="Test 1 — angular variation",
            name_latex=r"Test 1 --- angular variation",
            skills_plain="", skills_latex="",
            abilities_plain="", abilities_latex="",
            is_section=True,
        ),
        _Row(
            name_unicode="Spearman ρ (HC similarity vs Δξ)",
            name_latex=r"Spearman $\rho$ (HC similarity vs $\Delta\xi$)",
            skills_plain=f"{s1_rho:+.3f}{_stars_plain(s1_p)}",
            skills_latex=_val_with_stars_latex(s1_rho, s1_p),
            abilities_plain=f"{a1_rho:+.3f}{_stars_plain(a1_p)}",
            abilities_latex=_val_with_stars_latex(a1_rho, a1_p),
        ),
        _Row(
            name_unicode="Angular variance decomposition",
            name_latex=r"Angular variance decomposition",
            skills_plain="", skills_latex="",
            abilities_plain="", abilities_latex="",
            is_section=True,
        ),
        _Row(
            name_unicode="Median R²(ξ)",
            name_latex=r"Median $R^{2}(\xi)$",
            skills_plain=f"{s_med_xi:.2f}",
            skills_latex=f"{s_med_xi:.2f}",
            abilities_plain=f"{a_med_xi:.2f}",
            abilities_latex=f"{a_med_xi:.2f}",
        ),
        _Row(
            name_unicode="Median R²(χ)",
            name_latex=r"Median $R^{2}(\chi)$",
            skills_plain=f"{s_med_chi:.3f}",
            skills_latex=f"{s_med_chi:.3f}",
            abilities_plain=f"{a_med_chi:.3f}",
            abilities_latex=f"{a_med_chi:.3f}",
        ),
        _Row(
            name_unicode="Test 2 — radial intensification",
            name_latex=r"Test 2 --- radial intensification",
            skills_plain="", skills_latex="",
            abilities_plain="", abilities_latex="",
            is_section=True,
        ),
        _Row(
            name_unicode="Partial Spearman ρ(χ|ξ)",
            name_latex=r"Partial Spearman $\rho(\chi\,\vert\,\xi)$",
            skills_plain=f"{s_rho:+.3f}{_stars_plain(s_p)}",
            skills_latex=_val_with_stars_latex(s_rho, s_p),
            abilities_plain=f"{a_rho:+.3f}{_stars_plain(a_p)}",
            abilities_latex=_val_with_stars_latex(a_rho, a_p),
        ),
    ]

    # CSV / DataFrame uses the unicode + plain forms
    df = pd.DataFrame(
        [(r.name_unicode, r.skills_plain, r.abilities_plain) for r in rows],
        columns=["Statistic", "Skills", "Abilities"],
    )

    if write_csv:
        out_csv = infra.RP.export_fp("skillabil_summary.csv")
        df.to_csv(out_csv, index=False)
        log(f"Saved CSV:   {out_csv.name}")

    if write_latex:
        out_tex = infra.RP.export_fp("skillabil_summary.tex")
        out_tex.write_text(_render_latex(rows), encoding="utf-8")
        log(f"Saved LaTeX: {out_tex.name}")

    _pretty_print(rows)
    return df


# ─────────────────────────────────────────────────────────────
# Pretty-print
# ─────────────────────────────────────────────────────────────

def _pretty_print(rows: list[_Row]) -> None:
    name_w = max(len(r.name_unicode) for r in rows) + 2
    s_w = max(len(r.skills_plain) for r in rows) + 2
    a_w = max(len(r.abilities_plain) for r in rows) + 2
    s_w = max(s_w, len("Skills") + 2)
    a_w = max(a_w, len("Abilities") + 2)

    sep = "─" * (name_w + s_w + a_w + 4)
    print()
    print(sep)
    print(f"{'Statistic':<{name_w}}{'Skills':>{s_w}}{'Abilities':>{a_w}}")
    print(sep)
    for r in rows:
        if r.is_section:
            print()
            print(r.name_unicode)
        else:
            print(f"  {r.name_unicode:<{name_w-2}}{r.skills_plain:>{s_w}}{r.abilities_plain:>{a_w}}")
    print(sep)
    print("Significance: *** p<0.001  ** p<0.01  * p<0.05 (permutation test)")
    print()


# ─────────────────────────────────────────────────────────────
# LaTeX rendering (tabularray)
# ─────────────────────────────────────────────────────────────

def _render_latex(rows: list[_Row]) -> str:
    body_lines = []
    for r in rows:
        if r.is_section:
            body_lines.append(rf"\textbf{{{r.name_latex}}} & & \\")
        else:
            body_lines.append(
                rf"\quad {r.name_latex} & {r.skills_latex} & {r.abilities_latex} \\"
            )

    body = "\n".join(body_lines)

    return (
        r"\begin{table}[!htbp]" + "\n"
        r"\centering" + "\n"
        r"\caption{Headline statistics for Skills and Abilities. "
        r"Test 1 reports pair-level Spearman correlation between cosine "
        r"similarity in descriptor space and circular angular distance "
        r"$\Delta\xi$ across all occupation pairs. "
        r"$R^{2}(\xi)$ and $R^{2}(\chi)$ are per-descriptor variance shares; "
        r"medians are taken across descriptors. "
        r"Partial Spearman $\rho(\chi\,\vert\,\xi)$ uses rank residualization "
        r"against $\xi$ via von Mises kernel regression. "
        r"Significance: "
        r"$^{***}\,p<0.001$ (Test 2 by 1000-permutation test).}" + "\n"
        r"\label{tab:skillabil-summary}" + "\n"
        r"\begin{tblr}{" + "\n"
        r"  colspec = {l c c}," + "\n"
        r"}" + "\n"
        r"\toprule" + "\n"
        r"\textbf{Statistic} & \textbf{Skills} & \textbf{Abilities} \\" + "\n"
        r"\midrule" + "\n"
        + body + "\n"
        r"\bottomrule" + "\n"
        r"\end{tblr}" + "\n"
        r"\end{table}" + "\n"
    )