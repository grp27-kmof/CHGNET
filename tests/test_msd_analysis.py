"""Unit tests for MSD unwrapping, equilibration detection, and Einstein D statistics."""

from __future__ import annotations

import numpy as np
import pytest

from scientific.msd_analysis import (
    detect_equilibration_index,
    diffusion_from_slope_with_ci,
    einstein_fit_with_uncertainty,
    unwrap_positions_mic,
)


def test_unwrap_linear_drift_recovers_true_displacement():
    """Unwrapped trajectory should recover linear motion across periodic boundaries."""
    cell = np.eye(3) * 10.0
    n_frames = 50
    drift_per_frame = np.array([0.15, 0.0, 0.0])
    positions = np.zeros((n_frames, 1, 3))
    for t in range(n_frames):
        positions[t, 0] = (t * drift_per_frame) % 10.0  # wraps in x

    unwrapped = unwrap_positions_mic(positions, cell)
    expected = np.stack([t * drift_per_frame for t in range(n_frames)])
    np.testing.assert_allclose(unwrapped[:, 0], expected, atol=1e-10)


def test_einstein_fit_recovers_diffusion_slope():
    """Synthetic MSD = 2*d*D*t should yield correct D within CI."""
    d_true_cm2_s = 1e-6
    d_ang2_ps = d_true_cm2_s / 1e-4
    slope_true = 6.0 * d_ang2_ps  # 3D Einstein
    time_ps = np.linspace(0.0, 100.0, 200)
    msd = slope_true * time_ps + 0.5
    fit = einstein_fit_with_uncertainty(msd, time_ps, 20, 200, dimensions=3)
    assert fit["fit_r2"] > 0.99
    assert fit["slope_a2_ps"] == pytest.approx(slope_true, rel=0.01)
    assert fit["diffusion_cm2_s"] == pytest.approx(d_true_cm2_s, rel=0.02)
    assert fit["diffusion_cm2_s_lo"] <= fit["diffusion_cm2_s"] <= fit["diffusion_cm2_s_hi"]


def test_detect_equilibration_finds_stationary_region():
    """Equilibration index should lie after transient drift."""
    n = 500
    series = np.concatenate([np.linspace(10, 20, 100), np.full(400, 20.0)])
    eq = detect_equilibration_index(series, min_start_fraction=0.05)
    assert eq >= 50
    assert eq <= 150


def test_diffusion_ci_width_scales_with_stderr():
    d, lo, hi, stderr = diffusion_from_slope_with_ci(
        slope_a2_ps=6.0,
        slope_stderr=0.5,
        dimensions=3,
        dof=50,
    )
    assert np.isfinite(d)
    assert hi - lo > 2 * stderr
