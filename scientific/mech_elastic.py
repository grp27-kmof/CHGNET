"""
mech_elastic.py — Physics-based elastic moduli and Born stability for periodic crystals.

All calculations use finite-difference stress responses to applied strains.
CHGNet (or any ASE calculator) provides stresses at each strained configuration.

Physical workflow
-----------------
1.  Apply 6 independent Voigt strain patterns ±δ to the optimized cell.
2.  Compute the stress tensor σ at each strained configuration.
3.  Fit the linear response:  C_ij = ∂σ_i / ∂ε_j  (central differences).
4.  Derive bulk (B), shear (G), Young's (E) moduli via Voigt–Reuss–Hill averages.
5.  Check Born stability: all eigenvalues of C_ij > 0.
6.  Optionally scan hydrostatic pressure to find where Born stability first fails
    (critical pressure P_c).

References
----------
  Le Page & Saxe, Phys. Rev. B 65, 104104 (2002) — strain protocol.
  Mouhat & Coudert, Phys. Rev. B 90, 224104 (2014) — Born criteria review.
  Birch–Murnaghan EOS: Murnaghan, Proc. Natl. Acad. Sci. 30 (1944).
"""
from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import curve_fit

if TYPE_CHECKING:
    from ase import Atoms

logger = logging.getLogger(__name__)

# ── Strain amplitude ──────────────────────────────────────────────────────────
# 1 % strain (0.01) balances linearity of response with numerical noise.
# Values above 2 % enter the nonlinear regime for stiff MOFs.
DEFAULT_DELTA = 0.01

# ── Voigt notation index map ──────────────────────────────────────────────────
# Voigt: 0=xx, 1=yy, 2=zz, 3=yz, 4=xz, 5=xy
_VOIGT_IDX = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


# ─────────────────────────────────────────────────────────────────────────────
# Strain helpers
# ─────────────────────────────────────────────────────────────────────────────

def _strain_matrix(voigt_index: int, delta: float) -> np.ndarray:
    """3×3 strain tensor for a single Voigt component."""
    eps = np.zeros((3, 3))
    i, j = _VOIGT_IDX[voigt_index]
    if i == j:
        eps[i, i] = delta
    else:
        # Engineering shear strain: ε_ij = δ/2 (symmetric)
        eps[i, j] = delta / 2.0
        eps[j, i] = delta / 2.0
    return eps


def _deform_cell(atoms: "Atoms", eps: np.ndarray) -> "Atoms":
    """Return a new Atoms with cell and positions deformed by strain tensor eps."""
    import copy
    deformed = copy.deepcopy(atoms)
    F = np.eye(3) + eps                        # deformation gradient
    new_cell = atoms.get_cell().array @ F.T
    scaled = atoms.get_scaled_positions()
    deformed.set_cell(new_cell, scale_atoms=False)
    new_pos = scaled @ new_cell
    deformed.set_positions(new_pos)
    return deformed


def _stress_voigt(atoms: "Atoms") -> np.ndarray:
    """
    Return the stress in Voigt notation (GPa), sign-corrected.

    ASE convention: stress = [σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy] in eV/Å³.
    We convert to GPa:  1 eV/Å³ = 160.2176 GPa.
    """
    EV_A3_TO_GPA = 160.2176
    raw = atoms.get_stress()           # eV/Å³, Voigt
    return raw * EV_A3_TO_GPA


# ─────────────────────────────────────────────────────────────────────────────
# Elastic tensor
# ─────────────────────────────────────────────────────────────────────────────

def compute_elastic_tensor(
    atoms: "Atoms",
    delta: float = DEFAULT_DELTA,
    *,
    verbose: bool = True,
) -> np.ndarray:
    """
    Compute the 6×6 elastic stiffness tensor C_ij (GPa) by finite differences.

    For each of the 6 Voigt strain components α, apply +δ and −δ and evaluate
    the stress tensor σ.  Central differences give:

        C_iα = (σ_i(+δ) − σ_i(−δ)) / (2δ)

    The matrix is symmetrized to remove numerical noise.

    Parameters
    ----------
    atoms   : ASE Atoms with a calculator attached.
    delta   : Strain amplitude (dimensionless). Default 0.01.
    verbose : Log progress.

    Returns
    -------
    C : (6, 6) array, GPa.
    """
    C = np.zeros((6, 6))

    for alpha in range(6):
        eps_plus  = _strain_matrix(alpha, +delta)
        eps_minus = _strain_matrix(alpha, -delta)

        at_plus  = _deform_cell(atoms, eps_plus)
        at_minus = _deform_cell(atoms, eps_minus)

        s_plus  = _stress_voigt(at_plus)
        s_minus = _stress_voigt(at_minus)

        C[:, alpha] = (s_plus - s_minus) / (2.0 * delta)

        if verbose:
            logger.debug("Elastic column %d/%d done.", alpha + 1, 6)

    # Symmetrize
    C = 0.5 * (C + C.T)
    return C


# ─────────────────────────────────────────────────────────────────────────────
# Voigt–Reuss–Hill averages
# ─────────────────────────────────────────────────────────────────────────────

def vrh_averages(C: np.ndarray) -> dict[str, float]:
    """
    Voigt–Reuss–Hill bulk modulus B, shear modulus G, Young's modulus E,
    and Poisson's ratio ν from a 6×6 Voigt stiffness matrix C (GPa).

    Voigt bound (upper): B_V, G_V from C_ij components.
    Reuss bound (lower): B_R, G_R from compliance S = C^{-1}.
    Hill average: B = (B_V + B_R)/2, G = (G_V + G_R)/2.
    """
    # --- Voigt ---
    B_V = (C[0, 0] + C[1, 1] + C[2, 2] + 2*(C[0, 1] + C[0, 2] + C[1, 2])) / 9.0
    G_V = (C[0, 0] + C[1, 1] + C[2, 2]
           - C[0, 1] - C[0, 2] - C[1, 2]
           + 3*(C[3, 3] + C[4, 4] + C[5, 5])) / 15.0

    # --- Reuss ---
    try:
        S = np.linalg.inv(C)
        B_R = 1.0 / (S[0, 0] + S[1, 1] + S[2, 2]
                     + 2*(S[0, 1] + S[0, 2] + S[1, 2]))
        G_R = 15.0 / (4*(S[0, 0] + S[1, 1] + S[2, 2])
                      - 4*(S[0, 1] + S[0, 2] + S[1, 2])
                      + 3*(S[3, 3] + S[4, 4] + S[5, 5]))
    except np.linalg.LinAlgError:
        B_R, G_R = float("nan"), float("nan")

    # --- Hill ---
    B = 0.5 * (B_V + B_R)
    G = 0.5 * (G_V + G_R)
    E = 9.0 * B * G / (3.0 * B + G) if (3.0 * B + G) > 0 else float("nan")
    nu = (3.0 * B - 2.0 * G) / (2.0 * (3.0 * B + G)) if (3.0 * B + G) > 0 else float("nan")

    return {
        "B_voigt": float(B_V), "G_voigt": float(G_V),
        "B_reuss": float(B_R), "G_reuss": float(G_R),
        "B_hill":  float(B),   "G_hill":  float(G),
        "E_hill":  float(E),   "nu_hill": float(nu),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Born stability criteria
# ─────────────────────────────────────────────────────────────────────────────

def born_stability(C: np.ndarray) -> dict:
    """
    Check Born mechanical stability: all eigenvalues of C_ij must be positive.

    This is the necessary and sufficient condition for mechanical stability of a
    crystal in the harmonic approximation (Mouhat & Coudert, PRB 2014).

    Returns
    -------
    dict with keys:
      'stable'     : bool
      'eigenvalues': list[float]   (sorted ascending)
      'min_eigenvalue': float
      'reason'     : str description
    """
    eigvals = np.linalg.eigvalsh(C)
    min_ev = float(np.min(eigvals))
    stable = bool(min_ev > 0)

    reason = "All eigenvalues positive → mechanically stable" if stable else (
        f"Minimum eigenvalue = {min_ev:.4f} GPa < 0 → mechanically unstable"
    )
    return {
        "stable": stable,
        "eigenvalues": sorted(float(v) for v in eigvals),
        "min_eigenvalue": min_ev,
        "reason": reason,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Birch–Murnaghan EOS
# ─────────────────────────────────────────────────────────────────────────────

def _bm3_energy(V: np.ndarray, E0: float, V0: float, B0: float, B0p: float) -> np.ndarray:
    """Third-order Birch–Murnaghan energy as function of volume."""
    eta = (V0 / V) ** (2.0 / 3.0)
    return (E0
            + 9.0 * V0 * B0 / 16.0
            * (((eta - 1.0)**3) * B0p
               + (eta - 1.0)**2 * (6.0 - 4.0 * eta)))


def _bm3_pressure(V: np.ndarray, V0: float, B0: float, B0p: float) -> np.ndarray:
    """Third-order Birch–Murnaghan pressure P(V)."""
    f = 0.5 * ((V0 / V) ** (2.0 / 3.0) - 1.0)
    return (3.0 * B0 * f * (1.0 + f) ** (5.0 / 2.0)
            * (1.0 + 1.5 * (B0p - 4.0) * f))


def fit_birch_murnaghan(
    volumes: np.ndarray,
    energies: np.ndarray,
) -> dict[str, float]:
    """
    Fit the 3rd-order Birch–Murnaghan EOS to E(V) data.

    Returns
    -------
    dict with E0, V0, B0 (GPa), B0p (dimensionless).
    """
    EV_A3_TO_GPA = 160.2176

    V = np.asarray(volumes, dtype=float)
    E = np.asarray(energies, dtype=float)
    i_min = int(np.argmin(E))

    try:
        p0 = [E[i_min], V[i_min], 10.0, 4.0]
        popt, _ = curve_fit(_bm3_energy, V, E, p0=p0, maxfev=5000)
        E0, V0, B0_eVA3, B0p = popt
        B0_gpa = float(B0_eVA3 * EV_A3_TO_GPA)
        return {
            "E0_eV": float(E0),
            "V0_A3": float(V0),
            "B0_gpa": B0_gpa,
            "B0p": float(B0p),
            "success": True,
        }
    except Exception as exc:
        logger.warning("BM3 fit failed: %s", exc)
        return {"success": False, "error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# Critical pressure from Born stability scan
# ─────────────────────────────────────────────────────────────────────────────

def find_critical_pressure(
    atoms_ref: "Atoms",
    pressures_gpa: np.ndarray | None = None,
    delta: float = DEFAULT_DELTA,
) -> dict:
    """
    Scan a series of isotropic (hydrostatic) compressions and find the pressure
    at which Born stability first fails.

    At each pressure P, the unit cell is hydrostatically compressed by a
    scale factor α = (V(P)/V0)^(1/3), where V(P) is obtained by inverting the
    Birch–Murnaghan EOS using the zero-pressure bulk modulus.  The elastic
    tensor is then recomputed at the compressed geometry and its minimum
    eigenvalue is checked.

    Parameters
    ----------
    atoms_ref      : Optimized ASE Atoms with calculator attached.
    pressures_gpa  : 1D array of trial pressures (GPa). If None, uses a
                     logarithmic grid from 0.01 to 5 GPa.
    delta          : Strain amplitude for elastic tensor calculation.

    Returns
    -------
    dict with:
      'critical_pressure_gpa' : float or None
      'pressure_scan'         : list of (P_gpa, min_eigenvalue) tuples
      'note'                  : str
    """
    import copy

    if pressures_gpa is None:
        pressures_gpa = np.concatenate([
            np.linspace(0.0, 0.5, 11),
            np.linspace(0.6, 2.0, 15),
        ])

    # Zero-pressure reference
    C0 = compute_elastic_tensor(atoms_ref, delta, verbose=False)
    vrh0 = vrh_averages(C0)
    B0 = vrh0["B_hill"]   # GPa
    B0p = 4.0              # typical Murnaghan exponent

    V0 = float(atoms_ref.get_volume())
    scan: list[tuple[float, float]] = []
    critical_P: float | None = None

    for P in pressures_gpa:
        # Volume at pressure P from inverted 3rd-order BM EOS (approximate)
        # V(P) ≈ V0 * (1 + B0p * P / B0)^(-1/B0p)  [Murnaghan approximation]
        if P <= 0.0:
            V_P = V0
        else:
            if B0 <= 0:
                break
            V_P = V0 * (1.0 + B0p * P / B0) ** (-1.0 / B0p)

        alpha = (V_P / V0) ** (1.0 / 3.0)

        compressed = copy.deepcopy(atoms_ref)
        new_cell = atoms_ref.get_cell().array * alpha
        scaled = atoms_ref.get_scaled_positions()
        compressed.set_cell(new_cell, scale_atoms=False)
        compressed.set_positions(scaled @ new_cell)

        try:
            C_P = compute_elastic_tensor(compressed, delta, verbose=False)
            bs = born_stability(C_P)
            min_ev = bs["min_eigenvalue"]
        except Exception as exc:
            logger.warning("Elastic tensor failed at P=%.3f GPa: %s", P, exc)
            min_ev = float("nan")

        scan.append((float(P), float(min_ev)))
        logger.debug("P = %.3f GPa  →  min_eigenvalue = %.4f GPa", P, min_ev)

        if np.isfinite(min_ev) and min_ev <= 0.0 and critical_P is None:
            critical_P = float(P)
            break   # Found; no need to scan further

    note = (
        f"Born stability first violated at P ≈ {critical_P:.4f} GPa"
        if critical_P is not None
        else (
            "Born stability maintained across entire scan range "
            f"(max tested: {float(pressures_gpa[-1]):.2f} GPa). "
            "True P_c may exceed the scanned range."
        )
    )
    return {
        "critical_pressure_gpa": critical_P,
        "pressure_scan": scan,
        "note": note,
    }
