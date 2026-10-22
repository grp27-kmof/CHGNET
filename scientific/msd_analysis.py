"""
MSD and Einstein diffusion analysis with publication-grade statistics.

References:
  - Einstein (1905); Frenkel (1926) — MSD = 2dDt in d dimensions
  - Allen & Tildesley, *Computer Simulation of Liquids* — unwrapping, block averaging
  - Heermann et al. — equilibration via block error analysis

Design choices:
  * Sequential MIC unwrapping avoids underestimating MSD when ions hop cells.
  * Equilibration is detected via sliding-window energy/MSD stationarity, not a
    fixed 20% discard (which can be wrong for slow MOF equilibration).
  * D uncertainty propagated from linregress slope stderr: σ_D = σ_slope / (2d)
    then converted to cm²/s. 95% CI uses t-distribution with n-2 dof.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.stats import linregress

A2_PER_PS_TO_CM2_PER_S = 1e-4


def unwrap_positions_mic(
    positions: np.ndarray,
    cell: np.ndarray,
) -> np.ndarray:
    """
    Unwrap Cartesian positions using consecutive minimum-image displacements.

    Parameters
    ----------
    positions : (n_frames, n_atoms, 3)
    cell      : (3, 3) lattice matrix

    Returns
    -------
    unwrapped : (n_frames, n_atoms, 3) continuous trajectory
    """
    n_frames, n_atoms, _ = positions.shape
    if n_frames < 2:
        return positions.copy()

    inv_cell = np.linalg.inv(cell)
    unwrapped = np.zeros_like(positions, dtype=float)
    unwrapped[0] = positions[0].copy()

    for t in range(1, n_frames):
        delta_cart = positions[t] - positions[t - 1]
        delta_frac = delta_cart @ inv_cell
        delta_frac -= np.round(delta_frac)
        delta_mic = delta_frac @ cell
        unwrapped[t] = unwrapped[t - 1] + delta_mic

    return unwrapped


def detect_equilibration_index(
    series: np.ndarray,
    *,
    block_fraction: float = 0.10,
    min_start_fraction: float = 0.05,
    max_start_fraction: float = 0.50,
    stability_threshold: float = 0.02,
) -> int:
    """
    Detect equilibration onset via sliding-block relative drift.

    For each candidate start index, compare mean of first block vs last block
    in the remaining series.  The earliest index where relative drift falls
    below `stability_threshold` is returned.

    This is standard block-averaging equilibration analysis (Allen & Tildesley).

    Returns index in [0, n-1]; 0 if series too short or never stabilizes.
    """
    n = len(series)
    if n < 20:
        return max(0, int(n * min_start_fraction))

    finite = np.isfinite(series)
    if finite.sum() < 10:
        return 0

    s = series.copy()
    s[~finite] = np.nanmedian(s[finite])

    start_min = int(n * min_start_fraction)
    start_max = int(n * max_start_fraction)
    block = max(5, int(n * block_fraction))

    best_start = start_min
    best_drift = float("inf")

    for start in range(start_min, start_max + 1, max(1, block // 2)):
        tail = s[start:]
        if len(tail) < 2 * block:
            continue
        m1 = float(np.mean(tail[:block]))
        m2 = float(np.mean(tail[-block:]))
        denom = max(abs(m1), abs(m2), 1e-12)
        drift = abs(m2 - m1) / denom
        if drift < best_drift:
            best_drift = drift
            best_start = start
        if drift <= stability_threshold:
            return start

    return best_start


def einstein_fit_with_uncertainty(
    msd: np.ndarray,
    time_ps: np.ndarray,
    start: int,
    end: int,
    dimensions: int = 3,
    confidence: float = 0.95,
) -> dict[str, float]:
    """
    Linear Einstein fit MSD = slope * t + intercept with full statistics.

    Returns slope, intercept, R², slope stderr, t-stat, p-value, and
    diffusion coefficient with confidence interval.
    """
    out: dict[str, float] = {
        "slope_a2_ps": float("nan"),
        "intercept_a2": float("nan"),
        "fit_r2": float("nan"),
        "slope_stderr": float("nan"),
        "slope_tstat": float("nan"),
        "slope_pvalue": float("nan"),
        "diffusion_cm2_s": float("nan"),
        "diffusion_cm2_s_lo": float("nan"),
        "diffusion_cm2_s_hi": float("nan"),
        "diffusion_stderr_cm2_s": float("nan"),
        "fit_dof": float("nan"),
        "start_frame": float(start),
        "end_frame": float(end - 1),
        "num_frames": float(end - start),
    }

    if end - start < 3:
        return out

    t_win = time_ps[start:end]
    msd_win = msd[start:end]
    mask = np.isfinite(t_win) & np.isfinite(msd_win)
    if mask.sum() < 3:
        return out

    t_win = t_win[mask]
    msd_win = msd_win[mask]
    result = linregress(t_win, msd_win)
    slope = float(result.slope)
    intercept = float(result.intercept)
    r2 = float(result.rvalue ** 2)
    stderr = float(result.stderr) if result.stderr is not None else float("nan")
    tstat = float(abs(slope) / stderr) if np.isfinite(stderr) and stderr > 0 else float("nan")
    pvalue = float(result.pvalue) if result.pvalue is not None else float("nan")
    dof = len(t_win) - 2

    d_cm2_s, d_lo, d_hi, d_stderr = diffusion_from_slope_with_ci(
        slope, stderr, dimensions, dof, confidence
    )

    out.update(
        {
            "slope_a2_ps": slope,
            "intercept_a2": intercept,
            "fit_r2": r2,
            "slope_stderr": stderr,
            "slope_tstat": tstat,
            "slope_pvalue": pvalue,
            "diffusion_cm2_s": d_cm2_s,
            "diffusion_cm2_s_lo": d_lo,
            "diffusion_cm2_s_hi": d_hi,
            "diffusion_stderr_cm2_s": d_stderr,
            "fit_dof": float(dof),
        }
    )
    return out


def diffusion_from_slope_with_ci(
    slope_a2_ps: float,
    slope_stderr: float,
    dimensions: int = 3,
    dof: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float, float, float]:
    """
    Convert Einstein slope to D (cm²/s) and propagate uncertainty.

    D = slope / (2d)  in Å²/ps → multiply by 10⁻⁴ for cm²/s.

    Returns (D, D_lo, D_hi, D_stderr).
    """
    if not np.isfinite(slope_a2_ps) or slope_a2_ps <= 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    factor = A2_PER_PS_TO_CM2_PER_S / (2.0 * dimensions)
    d = slope_a2_ps * factor

    if not np.isfinite(slope_stderr) or dof < 1:
        return d, float("nan"), float("nan"), float("nan")

    d_stderr = slope_stderr * factor
    if slope_stderr == 0:
        return d, d, d, 0.0

    alpha = 1.0 - confidence
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, dof))
    margin = t_crit * d_stderr
    return d, d - margin, d + margin, d_stderr


def compute_msd_unwrapped(
    unwrapped_positions: np.ndarray,
    k_indices: list[int],
    framework_unwrapped: np.ndarray | None = None,
) -> np.ndarray:
    """
    MSD from unwrapped positions, optionally framework-relative.

    Parameters
    ----------
    unwrapped_positions : (n_frames, n_atoms, 3)
    k_indices : indices of K ions
    framework_unwrapped : optional (n_frames, n_fw, 3) for drift subtraction
    """
    n_frames = unwrapped_positions.shape[0]
    ref_k = unwrapped_positions[0, k_indices]
    msd = np.zeros(n_frames, dtype=float)

    if framework_unwrapped is not None and len(framework_unwrapped) > 0:
        ref_fw = framework_unwrapped[0]

    for t in range(n_frames):
        disp = unwrapped_positions[t, k_indices] - ref_k
        if framework_unwrapped is not None and len(framework_unwrapped) > 0:
            fw_disp = framework_unwrapped[t] - ref_fw
            fw_drift = np.mean(fw_disp, axis=0)
            disp = disp - fw_drift
        msd[t] = float(np.mean(np.sum(disp ** 2, axis=1)))
    return msd
