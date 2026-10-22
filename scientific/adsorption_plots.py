"""
Publication-quality adsorption-site analysis plots.

Generates:
  * Energy histogram of all relaxed candidates
  * Energy landscape (rank vs E_ads, colored by site type)
  * Site-type distribution
  * Final global-minimum site in fractional cell projection
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pymatgen.core import Structure

if TYPE_CHECKING:
    from scientific.adsorption_workflow import AdsorptionSearchResult

_SITE_COLORS = {
    "mu3_O": "#d62728",
    "open_metal_site": "#ff7f0e",
    "open_metal_adjacent_O": "#e377c2",
    "linker_O": "#2ca02c",
    "terminal_O": "#1f77b4",
    "bridge_O": "#9467bd",
    "carboxylate_O": "#2ca02c",
    "pore_grid": "#8c564b",
    "voronoi_void": "#bcbd22",
    "electrostatic_minimum": "#17becf",
}


def _color_for(site_type: str) -> str:
    return _SITE_COLORS.get(site_type, "#7f7f7f")


def plot_adsorption_analysis(
    mof: str,
    bare: Structure,
    result: "AdsorptionSearchResult",
    output_dir: Path,
) -> list[Path]:
    """Write adsorption diagnostic figures; return paths created."""
    plots_dir = output_dir / "plots" / "adsorption"
    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    relaxed_ok = [s for s in result.all_relaxed if s.success and np.isfinite(s.adsorption_energy_ev)]
    if not relaxed_ok:
        return written

    energies = np.array([s.adsorption_energy_ev for s in relaxed_ok])
    types = [s.site_type for s in relaxed_ok]

    # ── 1. Histogram of candidate energies ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(energies, bins=min(40, max(10, len(energies) // 5)), color="#4c72b0", alpha=0.85, edgecolor="white")
    if result.best and np.isfinite(result.best.adsorption_energy_ev):
        ax.axvline(
            result.best.adsorption_energy_ev, color="#d62728", ls="--", lw=2,
            label=f"Global min ({result.best.adsorption_energy_ev:.3f} eV/K)",
        )
    ax.set_xlabel("Adsorption energy (eV/K)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"{mof} — Adsorption Energy Histogram", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, ls=":", alpha=0.4)
    fig.tight_layout()
    p1 = plots_dir / f"{mof}_adsorption_energy_histogram.png"
    fig.savefig(p1, dpi=200)
    plt.close(fig)
    written.append(p1)

    # ── 2. Energy landscape (sorted rank vs energy) ───────────────────────────
    order = np.argsort(energies)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, idx in enumerate(order):
        st = types[idx]
        ax.scatter(i + 1, energies[idx], c=_color_for(st), s=28, alpha=0.8, edgecolors="none")
    if result.local_minima:
        min_energies = {s.adsorption_energy_ev for s in result.local_minima}
        for i, idx in enumerate(order):
            if energies[idx] in min_energies:
                ax.scatter(i + 1, energies[idx], facecolors="none", edgecolors="black", s=80, lw=1.5)
    ax.set_xlabel("Candidate rank (by energy)", fontsize=11)
    ax.set_ylabel("Adsorption energy (eV/K)", fontsize=11)
    ax.set_title(f"{mof} — Adsorption Energy Landscape", fontsize=12, fontweight="bold")
    ax.grid(True, ls=":", alpha=0.4)
    # legend by site type
    seen: set[str] = set()
    for st in sorted(set(types)):
        if st not in seen:
            ax.scatter([], [], c=_color_for(st), label=st, s=40)
            seen.add(st)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    p2 = plots_dir / f"{mof}_adsorption_energy_landscape.png"
    fig.savefig(p2, dpi=200)
    plt.close(fig)
    written.append(p2)

    # ── 3. Site-type distribution ─────────────────────────────────────────────
    from collections import Counter
    counts = Counter(types)
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = list(counts.keys())
    vals = [counts[k] for k in labels]
    colors = [_color_for(k) for k in labels]
    ax.barh(labels, vals, color=colors, edgecolor="white")
    ax.set_xlabel("Relaxed candidates", fontsize=11)
    ax.set_title(f"{mof} — Adsorption Site-Type Distribution", fontsize=12, fontweight="bold")
    ax.grid(True, axis="x", ls=":", alpha=0.4)
    fig.tight_layout()
    p3 = plots_dir / f"{mof}_adsorption_site_distribution.png"
    fig.savefig(p3, dpi=200)
    plt.close(fig)
    written.append(p3)

    # ── 4. Fractional-cell projection of K positions ────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    planes = [("a", "b", 2), ("a", "c", 1), ("b", "c", 0)]
    titles = ["ab plane", "ac plane", "bc plane"]
    for ax, (pa, pb, drop), title in zip(axes, planes, titles):
        idx_map = {"a": 0, "b": 1, "c": 2}
        ia, ib = idx_map[pa], idx_map[pb]
        for s in relaxed_ok:
            f = s.k_frac_coords
            ax.scatter(f[ia], f[ib], c=_color_for(s.site_type), s=12, alpha=0.5)
        if result.best:
            f = result.best.k_frac_coords
            ax.scatter(
                f[ia], f[ib], c="none", edgecolors="black", s=200, lw=2.5,
                marker="*", label="Global minimum",
            )
        ax.set_xlabel(f"Fractional {pa}")
        ax.set_ylabel(f"Fractional {pb}")
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, ls=":", alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"{mof} — K Adsorption Site Positions (fractional cell)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    p4 = plots_dir / f"{mof}_adsorption_site_positions.png"
    fig.savefig(p4, dpi=200)
    plt.close(fig)
    written.append(p4)

    # ── 5. Local minima only (clustered landscape) ────────────────────────────
    if result.local_minima:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        mins = result.local_minima
        x = np.arange(1, len(mins) + 1)
        colors = [_color_for(s.site_type) for s in mins]
        ax.bar(x, [s.adsorption_energy_ev for s in mins], color=colors, edgecolor="white")
        ax.set_xlabel("Local minimum index (ranked)", fontsize=11)
        ax.set_ylabel("Adsorption energy (eV/K)", fontsize=11)
        ax.set_title(f"{mof} — Unique Local Minima ({len(mins)} clusters)", fontsize=12, fontweight="bold")
        ax.grid(True, axis="y", ls=":", alpha=0.4)
        fig.tight_layout()
        p5 = plots_dir / f"{mof}_adsorption_local_minima.png"
        fig.savefig(p5, dpi=200)
        plt.close(fig)
        written.append(p5)

    return written
