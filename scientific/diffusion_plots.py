"""
Publication-quality diffusion analysis plots.

Generates:
  * MSD vs time (lab + framework-relative, fit window shaded)
  * log(MSD) vs log(t) with anomalous exponent α
  * Block-averaged D convergence vs block size
  * NEB energy profile for selected hop (when available)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

from scientific.diffusion_analysis import (
    MIN_ANOMALOUS_EXPONENT,
    anomalous_diffusion_exponent,
    block_averaged_diffusion,
)


def plot_diffusion_analysis(
    mof: str,
    time_ps: np.ndarray,
    msd_lab: np.ndarray,
    msd_fw_rel: np.ndarray,
    fit: dict[str, Any],
    output_dir: Path,
    *,
    neb_energy_profile: list[float] | None = None,
) -> list[Path]:
    """Write diffusion diagnostic figures; return paths created."""
    plots_dir = output_dir / "plots" / "diffusion"
    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if len(time_ps) < 3 or len(msd_fw_rel) < 3:
        return written

    fit_start = int(fit.get("start_frame", -1))
    fit_end = int(fit.get("end_frame", -1))
    alpha = float(fit.get("anomalous_exponent", float("nan")))
    slope = float(fit.get("slope_a2_ps", float("nan")))
    intercept = float(fit.get("intercept_a2", float("nan")))
    d_reported = float(fit.get("diffusion_cm2_s", float("nan")))
    r2 = float(fit.get("fit_r2", float("nan")))

    # ── 1. MSD vs time ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if len(msd_lab) == len(time_ps):
        ax.plot(time_ps, msd_lab, color="#aec7e8", lw=1.5, alpha=0.8, label="MSD (lab frame)")
    ax.plot(time_ps, msd_fw_rel, color="#1f77b4", lw=2, label="MSD (framework-relative)")
    if (
        0 <= fit_start < len(time_ps)
        and 0 <= fit_end < len(time_ps)
        and np.isfinite(slope)
    ):
        t_fit = time_ps[fit_start : fit_end + 1]
        y_fit = slope * t_fit + intercept
        ax.plot(t_fit, y_fit, color="#d62728", ls="--", lw=2, label=f"Einstein fit (R²={r2:.3f})")
        ax.axvspan(t_fit[0], t_fit[-1], color="#ff7f0e", alpha=0.15, label="Fit window")
    ax.set_xlabel("Time (ps)", fontsize=11)
    ax.set_ylabel("MSD (Å²)", fontsize=11)
    ax.set_title(f"{mof} — K-ion MSD", fontsize=12, fontweight="bold")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    p1 = plots_dir / f"{mof}_msd_vs_time.png"
    fig.savefig(p1, dpi=200)
    plt.close(fig)
    written.append(p1)

    # ── 2. log-log MSD (anomalous exponent) ───────────────────────────────────
    if fit_start >= 0 and fit_end > fit_start:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        t_win = time_ps[fit_start : fit_end + 1]
        m_win = msd_fw_rel[fit_start : fit_end + 1]
        mask = (t_win > 0) & (m_win > 0)
        if mask.sum() >= 5:
            log_t = np.log10(t_win[mask])
            log_m = np.log10(m_win[mask])
            ax.scatter(log_t, log_m, s=12, color="#1f77b4", alpha=0.7, label="MSD data")
            if not np.isfinite(alpha):
                alpha = anomalous_diffusion_exponent(msd_fw_rel, time_ps, fit_start, fit_end + 1)
            if np.isfinite(alpha):
                lr = linregress(log_t, log_m)
                x_line = np.linspace(log_t.min(), log_t.max(), 50)
                y_line = lr.slope * x_line + lr.intercept
                ax.plot(x_line, y_line, color="#d62728", lw=2, ls="--", label=f"α = {alpha:.3f}")
                fickian = "Fickian" if alpha >= MIN_ANOMALOUS_EXPONENT else "subdiffusive"
                ax.text(
                    0.05, 0.95, f"α = {alpha:.3f} ({fickian})",
                    transform=ax.transAxes, fontsize=10, va="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )
        ax.set_xlabel("log₁₀(time / ps)", fontsize=11)
        ax.set_ylabel("log₁₀(MSD / Å²)", fontsize=11)
        ax.set_title(f"{mof} — Anomalous Diffusion Exponent", fontsize=12, fontweight="bold")
        ax.grid(True, ls=":", alpha=0.4)
        ax.legend(fontsize=9)
        fig.tight_layout()
        p2 = plots_dir / f"{mof}_loglog_msd.png"
        fig.savefig(p2, dpi=200)
        plt.close(fig)
        written.append(p2)

    # ── 3. Block-averaged D convergence ───────────────────────────────────────
    if fit_start >= 0 and fit_end > fit_start:
        t = time_ps[fit_start : fit_end + 1]
        m = msd_fw_rel[fit_start : fit_end + 1]
        n = len(t)
        block_sizes: list[int] = []
        d_vals: list[float] = []
        d_errs: list[float] = []
        b = 4
        while b <= n // 4:
            block_sizes.append(b)
            blk = block_averaged_diffusion(m, t, 0, n)
            d_vals.append(float(blk.get("diffusion_cm2_s", float("nan"))))
            d_errs.append(float(blk.get("diffusion_stderr_cm2_s", float("nan"))))
            b *= 2

        if block_sizes and any(np.isfinite(v) for v in d_vals):
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            valid = [i for i, v in enumerate(d_vals) if np.isfinite(v)]
            bs = [block_sizes[i] for i in valid]
            ds = [d_vals[i] for i in valid]
            es = [d_errs[i] if np.isfinite(d_errs[i]) else 0.0 for i in valid]
            ax.errorbar(bs, ds, yerr=es, fmt="o-", color="#2ca02c", capsize=3, lw=2)
            if np.isfinite(d_reported):
                ax.axhline(d_reported, color="#d62728", ls="--", lw=1.5, label=f"Reported D = {d_reported:.2e}")
            ax.set_xscale("log", base=2)
            ax.set_xlabel("Block size (frames)", fontsize=11)
            ax.set_ylabel("D (cm²/s)", fontsize=11)
            ax.set_title(f"{mof} — Block-Averaged D Convergence", fontsize=12, fontweight="bold")
            ax.grid(True, ls=":", alpha=0.4)
            ax.legend(fontsize=9)
            fig.tight_layout()
            p3 = plots_dir / f"{mof}_block_d_convergence.png"
            fig.savefig(p3, dpi=200)
            plt.close(fig)
            written.append(p3)

    # ── 4. NEB energy profile ─────────────────────────────────────────────────
    if neb_energy_profile and len(neb_energy_profile) >= 3:
        energies = np.array(neb_energy_profile, dtype=float)
        e_ref = float(min(energies[0], energies[-1]))
        rel_e = energies - e_ref
        images = np.arange(len(energies))
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(images, rel_e, "o-", color="#9467bd", lw=2, markersize=6)
        ax.axhline(0.0, color="gray", ls=":", lw=1)
        barrier = float(np.nanmax(rel_e))
        ax.set_xlabel("NEB image index", fontsize=11)
        ax.set_ylabel("Relative energy (eV)", fontsize=11)
        ax.set_title(
            f"{mof} — NEB Migration Barrier (Eₐ ≈ {barrier:.3f} eV)",
            fontsize=12, fontweight="bold",
        )
        ax.grid(True, ls=":", alpha=0.4)
        fig.tight_layout()
        p4 = plots_dir / f"{mof}_neb_energy_profile.png"
        fig.savefig(p4, dpi=200)
        plt.close(fig)
        written.append(p4)

    return written
