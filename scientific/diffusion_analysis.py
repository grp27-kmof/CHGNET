"""
Publication-grade diffusion analysis for MOF alkali-ion MD trajectories.

References:
  - Einstein (1905); Frenkel (1926) — MSD = 2dDt in d dimensions
  - Allen & Tildesley, *Computer Simulation of Liquids* — unwrapping, block averaging
  - Flyvbjerg & Petersen, J. Chem. Phys. 1989 — block averaging error estimates
  - Hummer et al. — anomalous subdiffusion in confined pores (MSD ~ t^α)
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.stats import linregress

from scientific.msd_analysis import (
    A2_PER_PS_TO_CM2_PER_S,
    detect_equilibration_index,
    diffusion_from_slope_with_ci,
    einstein_fit_with_uncertainty,
)

# Publication gates (Fickian / Einstein regime)
MIN_ANOMALOUS_EXPONENT = 0.85
MIN_FICKIAN_FIT_R2 = 0.92
MIN_MONOTONIC_MSD_FRACTION = 0.95  # allow 5% noise dips
POST_EQUIL_MIN_WINDOW_FRACTION = 0.30
DEFAULT_HOP_THRESHOLD_A = 2.5
MD_NEB_AGREEMENT_LOG10_MAX = 1.5


def msd_is_monotonic(msd: np.ndarray, tolerance: float = 0.01) -> bool:
    """True if MSD is non-decreasing within tolerance (Å²) between consecutive points."""
    if len(msd) < 3:
        return True
    diffs = np.diff(msd[1:])
    return bool(np.all(diffs >= -abs(tolerance)))


def monotonic_fraction(msd: np.ndarray) -> float:
    """Fraction of consecutive MSD increments that are non-negative."""
    if len(msd) < 2:
        return 1.0
    diffs = np.diff(msd)
    return float(np.mean(diffs >= -1e-6))


def anomalous_diffusion_exponent(
    msd: np.ndarray,
    time_ps: np.ndarray,
    start: int,
    end: int,
) -> float:
    """
    Fit log(MSD) = α log(t) + c in window [start, end).

    α ≈ 1 → Fickian diffusion; α < 1 → subdiffusive (cage rattling / confinement).
    """
    if end - start < 5:
        return float("nan")
    t = time_ps[start:end]
    m = msd[start:end]
    mask = (t > 0) & (m > 0) & np.isfinite(t) & np.isfinite(m)
    if mask.sum() < 5:
        return float("nan")
    log_t = np.log(t[mask])
    log_m = np.log(m[mask])
    result = linregress(log_t, log_m)
    return float(result.slope)


def detect_hop_events(
    unwrapped_k_positions: np.ndarray,
    threshold_a: float = DEFAULT_HOP_THRESHOLD_A,
) -> dict:
    """
    Detect long-range K hops by clustering instantaneous positions.

    Parameters
    ----------
    unwrapped_k_positions : (n_frames, n_k, 3) or (n_frames, 3) for single K

    Returns
    -------
    dict with hop_count, site_labels per frame, n_unique_sites
    """
    pos = np.asarray(unwrapped_k_positions, dtype=float)
    if pos.ndim == 2:
        pos = pos[:, np.newaxis, :]
    n_frames, n_k, _ = pos.shape
    if n_frames < 2 or n_k < 1:
        return {"hop_count": 0, "n_unique_sites": 1, "per_k_hop_counts": [], "site_ids": []}

    per_k_hops: list[int] = []
    all_site_ids: list[list[int]] = []

    for k in range(n_k):
        traj = pos[:, k, :]
        site_centroids: list[np.ndarray] = []
        site_ids: list[int] = []
        hops = 0
        current_site = 0

        for t in range(n_frames):
            p = traj[t]
            if not site_centroids:
                site_centroids.append(p.copy())
                site_ids.append(0)
                current_site = 0
                continue
            dists = [float(np.linalg.norm(p - c)) for c in site_centroids]
            nearest = int(np.argmin(dists))
            if dists[nearest] <= threshold_a:
                # Update running centroid (incremental mean)
                n_in = sum(1 for s in site_ids if s == nearest)
                site_centroids[nearest] = (
                    site_centroids[nearest] * n_in + p
                ) / (n_in + 1)
                if nearest != current_site:
                    hops += 1
                current_site = nearest
                site_ids.append(nearest)
            else:
                new_id = len(site_centroids)
                site_centroids.append(p.copy())
                if current_site != new_id:
                    hops += 1
                current_site = new_id
                site_ids.append(new_id)

        per_k_hops.append(hops)
        all_site_ids.append(site_ids)

    return {
        "hop_count": int(max(per_k_hops) if per_k_hops else 0),
        "per_k_hop_counts": per_k_hops,
        "n_unique_sites": int(max(len(set(s)) for s in all_site_ids) if all_site_ids else 1),
        "site_ids": all_site_ids,
    }


def _window_is_valid(
    msd: np.ndarray,
    time_ps: np.ndarray,
    start: int,
    end: int,
    min_r2: float = MIN_FICKIAN_FIT_R2,
    min_alpha: float = MIN_ANOMALOUS_EXPONENT,
) -> tuple[bool, dict[str, float]]:
    """Check monotonicity, R², and α for a candidate fit window."""
    msd_win = msd[start:end]
    if not msd_is_monotonic(msd_win):
        return False, {}
    if monotonic_fraction(msd_win) < MIN_MONOTONIC_MSD_FRACTION:
        return False, {}

    fit = einstein_fit_with_uncertainty(msd, time_ps, start, end, dimensions=3)
    r2 = float(fit.get("fit_r2", float("nan")))
    slope = float(fit.get("slope_a2_ps", float("nan")))
    if not np.isfinite(r2) or r2 < min_r2 or not np.isfinite(slope) or slope <= 0:
        return False, fit

    alpha = anomalous_diffusion_exponent(msd, time_ps, start, end)
    if np.isfinite(alpha) and alpha < min_alpha:
        return False, fit

    fit["anomalous_exponent"] = alpha
    return True, fit


def select_longest_diffusive_window(
    msd: np.ndarray,
    time_ps: np.ndarray,
    eq_frame: int = 0,
    *,
    min_r2: float = MIN_FICKIAN_FIT_R2,
    min_alpha: float = MIN_ANOMALOUS_EXPONENT,
    min_window_fraction: float = POST_EQUIL_MIN_WINDOW_FRACTION,
    stride: int = 5,
) -> dict[str, float]:
    """
    Select the longest time window satisfying Fickian gates (R², α, monotonic MSD).

    Prefers the longest statistically valid plateau over short high-R² segments.
    """
    out: dict[str, float] = {
        "diffusion_cm2_s": float("nan"),
        "fit_r2": float("nan"),
        "slope_a2_ps": float("nan"),
        "intercept_a2": float("nan"),
        "start_frame": -1.0,
        "end_frame": -1.0,
        "num_frames": 0.0,
        "slope_tstat": float("nan"),
        "slope_pvalue": float("nan"),
        "slope_stderr": float("nan"),
        "diffusion_stderr_cm2_s": float("nan"),
        "diffusion_cm2_s_lo": float("nan"),
        "diffusion_cm2_s_hi": float("nan"),
        "anomalous_exponent": float("nan"),
        "fit_window_ps": float("nan"),
        "equilibration_frame": float(eq_frame),
        "equilibration_time_ps": float(time_ps[eq_frame]) if len(time_ps) > eq_frame else float("nan"),
    }
    n = len(msd)
    if n < 10 or len(time_ps) != n:
        return out

    start_min = max(int(eq_frame), 0)
    post_eq = n - start_min
    min_window = max(10, int(post_eq * min_window_fraction))
    end_max = n

    best_fit: dict[str, float] | None = None
    best_length = -1

    # Search longest valid windows: fix start, extend end
    for start in range(start_min, end_max - min_window, stride):
        for end in range(start + min_window, end_max + 1, stride):
            length = end - start
            if length <= best_length:
                continue
            ok, fit = _window_is_valid(msd, time_ps, start, end, min_r2, min_alpha)
            if ok and length > best_length:
                best_length = length
                best_fit = fit
                best_fit["start_frame"] = float(start)
                best_fit["end_frame"] = float(end - 1)
                best_fit["num_frames"] = float(length)

    if best_fit is None:
        # Relaxed fallback: longest window with R² ≥ 0.90 and positive slope
        for start in range(start_min, end_max - min_window, stride):
            end = end_max
            length = end - start
            if length <= best_length:
                continue
            fit = einstein_fit_with_uncertainty(msd, time_ps, start, end, dimensions=3)
            r2 = float(fit.get("fit_r2", float("nan")))
            slope = float(fit.get("slope_a2_ps", float("nan")))
            if np.isfinite(r2) and r2 >= 0.90 and np.isfinite(slope) and slope > 0:
                if msd_is_monotonic(msd[start:end]):
                    best_length = length
                    best_fit = fit
                    best_fit["start_frame"] = float(start)
                    best_fit["end_frame"] = float(end - 1)
                    best_fit["num_frames"] = float(length)
                    best_fit["anomalous_exponent"] = anomalous_diffusion_exponent(
                        msd, time_ps, start, end
                    )

    if best_fit is not None:
        start = int(best_fit["start_frame"])
        end = int(best_fit["end_frame"]) + 1
        out.update(best_fit)
        if len(time_ps) > end - 1 and start < end:
            out["fit_window_ps"] = float(time_ps[end - 1] - time_ps[start])
        block = block_averaged_diffusion(msd, time_ps, start, end)
        if np.isfinite(block.get("diffusion_cm2_s", float("nan"))):
            out["diffusion_cm2_s"] = block["diffusion_cm2_s"]
            out["diffusion_stderr_cm2_s"] = block["diffusion_stderr_cm2_s"]
            out["diffusion_cm2_s_lo"] = block["diffusion_cm2_s_lo"]
            out["diffusion_cm2_s_hi"] = block["diffusion_cm2_s_hi"]
            out["block_stderr_cm2_s"] = block["diffusion_stderr_cm2_s"]
        elif not np.isfinite(out.get("diffusion_cm2_s", float("nan"))):
            slope = float(out.get("slope_a2_ps", float("nan")))
            if np.isfinite(slope) and slope > 0:
                out["diffusion_cm2_s"] = slope / 6.0 * A2_PER_PS_TO_CM2_PER_S

    return out


def block_averaged_diffusion(
    msd: np.ndarray,
    time_ps: np.ndarray,
    start: int,
    end: int,
    dimensions: int = 3,
) -> dict[str, float]:
    """
    Flyvbjerg block averaging on Einstein slopes from sub-blocks.

    Estimates D and standard error from convergence of block-averaged slopes.
    """
    out: dict[str, float] = {
        "diffusion_cm2_s": float("nan"),
        "diffusion_stderr_cm2_s": float("nan"),
        "diffusion_cm2_s_lo": float("nan"),
        "diffusion_cm2_s_hi": float("nan"),
        "n_blocks": 0.0,
    }
    if end - start < 10:
        return out

    t = time_ps[start:end]
    m = msd[start:end]
    n = len(t)
    block_sizes: list[int] = []
    d_blocks: list[float] = []
    b = 4
    while b <= n // 4:
        block_sizes.append(b)
        n_blk = n // b
        slopes = []
        for i in range(n_blk):
            s = i * b
            e = s + b
            if e - s < 3:
                continue
            fit = linregress(t[s:e], m[s:e])
            if fit.slope > 0:
                slopes.append(fit.slope / (2.0 * dimensions) * A2_PER_PS_TO_CM2_PER_S)
        if slopes:
            d_blocks.append(float(np.mean(slopes)))
        b *= 2

    if len(d_blocks) < 2:
        fit = einstein_fit_with_uncertainty(m, t, 0, n, dimensions=dimensions)
        out["diffusion_cm2_s"] = float(fit.get("diffusion_cm2_s", float("nan")))
        out["diffusion_stderr_cm2_s"] = float(fit.get("diffusion_stderr_cm2_s", float("nan")))
        out["diffusion_cm2_s_lo"] = float(fit.get("diffusion_cm2_s_lo", float("nan")))
        out["diffusion_cm2_s_hi"] = float(fit.get("diffusion_cm2_s_hi", float("nan")))
        out["n_blocks"] = 1.0
        return out

    # Flyvbjerg: variance of block means decreases; use block ensemble
    d_arr = np.array(d_blocks, dtype=float)
    d_mean = float(np.mean(d_arr))
    d_std = float(np.std(d_arr, ddof=1)) if len(d_arr) > 1 else float("nan")
    out["diffusion_cm2_s"] = d_mean
    out["diffusion_stderr_cm2_s"] = d_std
    out["n_blocks"] = float(len(d_blocks))
    if np.isfinite(d_std) and len(d_arr) > 1:
        dof = len(d_arr) - 1
        t_crit = float(stats.t.ppf(0.975, dof))
        margin = t_crit * d_std / np.sqrt(len(d_arr))
        out["diffusion_cm2_s_lo"] = d_mean - margin
        out["diffusion_cm2_s_hi"] = d_mean + margin
    return out


def md_neb_agreement_log10(d_md: float, d_neb: float) -> float:
    """Absolute log10 difference between MD and NEB+TST diffusion estimates."""
    if not np.isfinite(d_md) or not np.isfinite(d_neb) or d_md <= 0 or d_neb <= 0:
        return float("nan")
    return float(abs(np.log10(d_md) - np.log10(d_neb)))


def classify_diffusion_with_hops(
    base_reliability: str,
    *,
    hop_count: int,
    anomalous_exponent: float,
    md_neb_log10_agreement: float,
    outside_screening_range: bool = False,
) -> str:
    """Refine reliability using hop count, α, and MD-NEB agreement."""
    rel = base_reliability
    if np.isfinite(anomalous_exponent) and anomalous_exponent < MIN_ANOMALOUS_EXPONENT:
        return "Very Low"
    if hop_count == 0:
        rel = _downgrade_reliability(rel, steps=1)
    if np.isfinite(md_neb_log10_agreement) and md_neb_log10_agreement > MD_NEB_AGREEMENT_LOG10_MAX:
        rel = _downgrade_reliability(rel, steps=1)
    if outside_screening_range:
        rel = _downgrade_reliability(rel, steps=1)
    return rel


def _downgrade_reliability(rel: str, steps: int = 1) -> str:
    order = ["High", "Moderate", "Low", "Very Low", "Rejected"]
    try:
        idx = order.index(rel)
    except ValueError:
        idx = 3
    return order[min(idx + steps, len(order) - 1)]


def apply_best_estimate_policy(
    d_md: float,
    d_neb: float,
    *,
    hop_count: int,
    md_reliability: str,
    neb_reliability: str,
) -> tuple[float, str, str]:
    """
    Choose reported D and method.

    - ≥1 hop and valid MD → MD Einstein D
    - 0 hops (localized) → NEB+TST D if available, else MD with warning
    """
    md_ok = (
        np.isfinite(d_md) and d_md > 0
        and md_reliability not in ("Rejected", "Very Low")
    )
    neb_ok = np.isfinite(d_neb) and d_neb > 0

    if hop_count >= 1 and md_ok:
        return d_md, "md", md_reliability
    if hop_count == 0 and neb_ok:
        return d_neb, "neb_tst", neb_reliability
    if md_ok:
        return d_md, "md", md_reliability
    if neb_ok:
        return d_neb, "neb_tst", neb_reliability
    if np.isfinite(d_md) and d_md > 0:
        return d_md, "md", md_reliability
    if np.isfinite(d_neb) and d_neb > 0:
        return d_neb, "neb_tst", neb_reliability
    return float("nan"), "none", "Very Low"
