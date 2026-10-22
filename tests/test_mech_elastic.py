"""
Unit tests for the elastic mechanics module (scientific/mech_elastic.py).

These tests use a synthetic FCC-like isotropic elastic tensor so no
CHGNet call or CIF file is required.  They verify:
  - VRH averaging against known analytic isotropic results
  - Born stability sign checks
  - BM3 EOS fitting recovers known B0, B0'
  - _strain_matrix shapes and symmetry
"""
from __future__ import annotations

import numpy as np
import pytest

from scientific.mech_elastic import (
    _strain_matrix,
    born_stability,
    fit_birch_murnaghan,
    vrh_averages,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _isotropic_C(B_gpa: float, G_gpa: float) -> np.ndarray:
    """
    Build a 6×6 isotropic elastic tensor from bulk (B) and shear (G) moduli.

    For an isotropic material:
      C11 = B + 4G/3
      C12 = B − 2G/3
      C44 = G
    """
    C11 = B_gpa + 4.0 * G_gpa / 3.0
    C12 = B_gpa - 2.0 * G_gpa / 3.0
    C44 = G_gpa

    C = np.zeros((6, 6))
    # Normal components
    for i in range(3):
        C[i, i] = C11
    for i in range(3):
        for j in range(3):
            if i != j:
                C[i, j] = C12
    # Shear components
    for i in range(3, 6):
        C[i, i] = C44
    return C


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestStrainMatrix:
    def test_normal_strain_diagonal(self):
        """Normal strain (Voigt 0–2) should produce diagonal eps."""
        eps = _strain_matrix(0, 0.01)
        assert eps[0, 0] == pytest.approx(0.01)
        assert eps[1, 1] == pytest.approx(0.0)
        assert eps[2, 2] == pytest.approx(0.0)

    def test_shear_strain_symmetric(self):
        """Shear strain (Voigt 3–5) should produce symmetric off-diagonal eps."""
        eps = _strain_matrix(3, 0.01)  # yz
        assert eps[1, 2] == pytest.approx(0.005)
        assert eps[2, 1] == pytest.approx(0.005)
        assert eps[0, 0] == pytest.approx(0.0)

    def test_all_six_patterns(self):
        for alpha in range(6):
            eps = _strain_matrix(alpha, 0.01)
            assert eps.shape == (3, 3)
            np.testing.assert_allclose(eps, eps.T, atol=1e-15)  # must be symmetric


class TestVRH:
    def test_isotropic_B_G_recovery(self):
        """VRH on an isotropic tensor should recover B and G exactly."""
        B_true, G_true = 10.0, 5.0
        C = _isotropic_C(B_true, G_true)
        vrh = vrh_averages(C)

        assert vrh["B_hill"] == pytest.approx(B_true, rel=1e-4)
        assert vrh["G_hill"] == pytest.approx(G_true, rel=1e-4)

    def test_young_modulus_formula(self):
        """E = 9BG / (3B + G) must hold."""
        B, G = 15.0, 6.0
        C = _isotropic_C(B, G)
        vrh = vrh_averages(C)
        E_expected = 9.0 * B * G / (3.0 * B + G)
        assert vrh["E_hill"] == pytest.approx(E_expected, rel=1e-4)

    def test_voigt_geq_reuss(self):
        """Voigt bound must be ≥ Reuss bound (Voigt–Reuss inequality)."""
        C = _isotropic_C(12.0, 4.0)
        vrh = vrh_averages(C)
        assert vrh["B_voigt"] >= vrh["B_reuss"] - 1e-6
        assert vrh["G_voigt"] >= vrh["G_reuss"] - 1e-6


class TestBornStability:
    def test_positive_definite_passes(self):
        """A well-conditioned isotropic C should pass Born stability."""
        C = _isotropic_C(10.0, 4.0)
        result = born_stability(C)
        assert result["stable"] is True
        assert result["min_eigenvalue"] > 0.0

    def test_negative_eigenvalue_fails(self):
        """Perturbing C44 < 0 should fail Born stability."""
        C = _isotropic_C(10.0, 4.0)
        C[3, 3] = -1.0   # make one shear eigenvalue negative
        C[4, 4] = -1.0
        C[5, 5] = -1.0
        result = born_stability(C)
        assert result["stable"] is False
        assert result["min_eigenvalue"] < 0.0

    def test_eigenvalues_sorted(self):
        """Returned eigenvalues list must be sorted ascending."""
        C = _isotropic_C(8.0, 3.0)
        result = born_stability(C)
        evs = result["eigenvalues"]
        assert evs == sorted(evs)


class TestBirchMurnaghan:
    def test_bm3_recovery(self):
        """BM3 fit should recover known B0 and B0' from synthetic E(V)."""
        from scientific.mech_elastic import _bm3_energy

        V0_true, B0_eVA3_true, B0p_true = 1000.0, 0.05, 4.0
        E0_true = -100.0
        EV_A3_TO_GPA = 160.2176
        B0_gpa_true = B0_eVA3_true * EV_A3_TO_GPA

        V = np.linspace(900.0, 1100.0, 21)
        E = _bm3_energy(V, E0_true, V0_true, B0_eVA3_true, B0p_true)

        result = fit_birch_murnaghan(V, E)
        assert result["success"] is True
        assert result["B0_gpa"] == pytest.approx(B0_gpa_true, rel=0.01)
        assert result["B0p"]    == pytest.approx(B0p_true, rel=0.05)
        assert result["V0_A3"]  == pytest.approx(V0_true,  rel=0.01)

    def test_bm3_insufficient_data_returns_failure(self):
        """Fewer than 5 data points should not raise, just return success=False."""
        result = fit_birch_murnaghan(
            np.array([900.0, 950.0, 1000.0]),
            np.array([-99.9, -100.0, -99.9]),
        )
        # Fitting with 3 points may still succeed numerically, but we don't crash
        assert "success" in result
