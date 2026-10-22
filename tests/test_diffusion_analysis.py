"""Tests for publication-grade diffusion analysis helpers."""

from __future__ import annotations

import numpy as np

from scientific.diffusion_analysis import (
    MIN_ANOMALOUS_EXPONENT,
    apply_best_estimate_policy,
    anomalous_diffusion_exponent,
    block_averaged_diffusion,
    detect_hop_events,
    md_neb_agreement_log10,
    select_longest_diffusive_window,
)
from scientific.msd_analysis import A2_PER_PS_TO_CM2_PER_S


def _linear_msd(time_ps: np.ndarray, d_cm2_s: float) -> np.ndarray:
    """Synthetic Fickian MSD: slope = 6D in Å²/ps."""
    slope = d_cm2_s / A2_PER_PS_TO_CM2_PER_S * 6.0
    return slope * time_ps + 0.05


def _subdiffusive_msd(time_ps: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    return 0.5 * (time_ps ** alpha) + 0.01


def test_anomalous_exponent_linear():
    t = np.linspace(0.1, 50.0, 500)
    d_true = 1e-6
    msd = _linear_msd(t, d_true)
    alpha = anomalous_diffusion_exponent(msd, t, 50, 450)
    assert np.isfinite(alpha)
    assert alpha > 0.9


def test_anomalous_exponent_subdiffusive():
    t = np.linspace(0.1, 50.0, 500)
    msd = _subdiffusive_msd(t, alpha=0.6)
    alpha = anomalous_diffusion_exponent(msd, t, 50, 450)
    assert np.isfinite(alpha)
    assert alpha < MIN_ANOMALOUS_EXPONENT


def test_block_averaged_diffusion_recovers_linear():
    t = np.linspace(0.1, 50.0, 500)
    d_true = 2e-7
    msd = _linear_msd(t, d_true)
    blk = block_averaged_diffusion(msd, t, 50, 450)
    d_fit = float(blk["diffusion_cm2_s"])
    assert np.isfinite(d_fit)
    assert abs(np.log10(d_fit) - np.log10(d_true)) < 0.5


def test_select_longest_window_prefers_plateau():
    t = np.linspace(0.1, 50.0, 500)
    d_true = 5e-7
    msd = _linear_msd(t, d_true)
    # Inject a short high-R² noisy segment at the start
    msd[100:130] += np.linspace(0, 2.0, 30)

    fit = select_longest_diffusive_window(msd, t, eq_frame=50)
    start = int(fit["start_frame"])
    end = int(fit["end_frame"])
    window_len = end - start + 1
    assert window_len >= 100
    assert float(fit.get("fit_r2", 0)) >= 0.90
    alpha = float(fit.get("anomalous_exponent", float("nan")))
    if np.isfinite(alpha):
        assert alpha >= MIN_ANOMALOUS_EXPONENT - 0.05


def test_detect_hop_events_zero_hops():
    n_frames = 200
    pos = np.zeros((n_frames, 1, 3), dtype=float)
    pos[:, 0, 0] = np.linspace(0, 0.5, n_frames)  # small cage rattling
    info = detect_hop_events(pos, threshold_a=2.5)
    assert info["hop_count"] == 0


def test_detect_hop_events_with_hop():
    n_frames = 100
    pos = np.zeros((n_frames, 1, 3), dtype=float)
    pos[:50, 0, 0] = 0.0
    pos[50:, 0, 0] = 5.0  # clear hop
    info = detect_hop_events(pos, threshold_a=2.5)
    assert info["hop_count"] >= 1


def test_apply_best_estimate_policy_zero_hops_uses_neb():
    d_md = 3.89e-5
    d_neb = 2.5e-7
    d_final, method, rel = apply_best_estimate_policy(
        d_md, d_neb, hop_count=0, md_reliability="Low", neb_reliability="Moderate"
    )
    assert method == "neb_tst"
    assert abs(d_final - d_neb) / d_neb < 1e-6


def test_apply_best_estimate_policy_with_hops_uses_md():
    d_md = 1e-6
    d_neb = 2e-7
    d_final, method, _ = apply_best_estimate_policy(
        d_md, d_neb, hop_count=2, md_reliability="Moderate", neb_reliability="Moderate"
    )
    assert method == "md"
    assert abs(d_final - d_md) / d_md < 1e-6


def test_md_neb_agreement_log10():
    agree = md_neb_agreement_log10(1e-5, 1e-7)
    assert np.isfinite(agree)
    assert abs(agree - 2.0) < 0.01
