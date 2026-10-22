#!/usr/bin/env python3
"""
analyse.py — Independent Post-Processing for MOF Screening (Step 2 of 2).

Reads MD trajectory files produced by run_md.py and computes three
screening parameters directly from raw simulation data:

    1. Adsorption Energy       (eV/K)
    2. MSD                     (Å², framework-relative, production-averaged)
    3. Mechanical Stability    (MSI 0–1, mech_stabb)

Results are written to outputs/results.csv and printed to the terminal.

Usage:
    python analyse.py            # Analyse all MOFs in outputs/structures/
    python analyse.py --device cpu

Run AFTER run_md.py has completed.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — no GUI window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.io.trajectory import Trajectory
from pymatgen.core import Structure
from tqdm import tqdm

# ── Monkey-patch ase.geometry.find_mic to accept raw numpy arrays ─────────────
import ase.geometry
from ase.cell import Cell
from ase.constraints import FixAtoms
from ase.optimize import FIRE

_orig_find_mic = ase.geometry.find_mic


def _patched_find_mic(vectors, cell, pbc=True):
    if not isinstance(cell, Cell):
        cell = Cell(cell)
    return _orig_find_mic(vectors, cell, pbc)


ase.geometry.find_mic = _patched_find_mic

# ── Ensure run_pipeline is importable ─────────────────────────────────────────
sys.path.append(str(Path(__file__).parent.resolve()))
import run_pipeline

# Force CHGNet single-point evaluations to CUDA
# (Ensure GPU is not heavily used by another process)
run_pipeline.USE_DEVICE = "cuda"

# ── Vectorised trajectory helpers (100x–800x faster than the loop versions) ──

def _fast_compute_k_msd(trajectory: Trajectory, k_indices: list[int]) -> np.ndarray:
    """Vectorised MSD(t) = <|r(t)−r(0)|²> for K ions with PBC MIC."""
    n_frames = len(trajectory)
    msd = np.zeros(n_frames, dtype=float)
    ref_pos = trajectory[0].get_positions()
    ref_k   = ref_pos[k_indices]
    for t in range(n_frames):
        pos     = trajectory[t].get_positions()[k_indices]
        cell    = Cell(trajectory[t].get_cell())
        raw     = pos - ref_k                              # (N_k, 3)
        mic, _  = ase.geometry.find_mic(raw, cell)
        msd[t]  = float(np.mean(np.sum(mic * mic, axis=1)))
    return msd


def _fast_compute_k_msd_framework_relative(
    trajectory: Trajectory,
    k_indices: list[int],
    framework_idx: list[int],
) -> np.ndarray:
    """Framework-relative MSD(t) for K ions with PBC MIC."""
    n_frames = len(trajectory)
    msd = np.zeros(n_frames, dtype=float)
    if not k_indices:
        return msd

    ref_pos = trajectory[0].get_positions()
    ref_k   = ref_pos[k_indices]
    ref_fw  = ref_pos[framework_idx] if framework_idx else np.empty((0, 3))

    for t in range(n_frames):
        pos  = trajectory[t].get_positions()
        cell = Cell(trajectory[t].get_cell())
        if len(framework_idx):
            fw_mic, _ = ase.geometry.find_mic(pos[framework_idx] - ref_fw, cell)
            fw_drift  = np.mean(fw_mic, axis=0)
        else:
            fw_drift = np.zeros(3, dtype=float)

        raw    = pos[k_indices] - ref_k
        mic, _ = ase.geometry.find_mic(raw, cell)
        rel    = mic - fw_drift
        msd[t] = float(np.mean(np.sum(rel * rel, axis=1)))
    return msd


def _fast_framework_displacements(
    ref_positions: np.ndarray,
    positions: np.ndarray,
    framework_idx: list[int],
    cell: np.ndarray,
) -> tuple[float, float]:
    if not framework_idx:
        return float("nan"), float("nan")
    cell_obj = Cell(cell) if not isinstance(cell, Cell) else cell
    raw  = positions[framework_idx] - ref_positions[framework_idx]  # (N, 3)
    mic, _ = ase.geometry.find_mic(raw, cell_obj)
    disps  = np.linalg.norm(mic, axis=1)
    return float(np.sqrt(np.mean(disps ** 2))), float(np.max(disps))


def _fast_bond_preservation(
    positions: np.ndarray,
    bond_pairs: list[tuple[int, int]],
    initial_distances: dict[tuple[int, int], float],
    cell: np.ndarray,
) -> float:
    if not bond_pairs:
        return float("nan")
    cell_obj = Cell(cell) if not isinstance(cell, Cell) else cell
    idx_i = [p[0] for p in bond_pairs]
    idx_j = [p[1] for p in bond_pairs]
    raw   = positions[idx_j] - positions[idx_i]           # (M, 3)
    mic, _ = ase.geometry.find_mic(raw, cell_obj)
    dists  = np.linalg.norm(mic, axis=1)
    d0     = np.array([initial_distances[p] for p in bond_pairs])
    return float(np.sum(dists <= d0 * 1.30) / len(bond_pairs))


# Patch run_pipeline to use the fast versions
run_pipeline.compute_k_msd                      = _fast_compute_k_msd
run_pipeline.compute_k_msd_framework_relative   = _fast_compute_k_msd_framework_relative
run_pipeline.framework_displacements            = _fast_framework_displacements
run_pipeline.bond_preservation_fraction         = _fast_bond_preservation

warnings.filterwarnings("ignore", module="pymatgen")
warnings.filterwarnings("ignore", module="ase")

# =============================================================================
# PATHS
# =============================================================================

OUTPUT_DIR     = Path("outputs")
STRUCTURES_DIR = OUTPUT_DIR / "structures"
RESULTS_CSV    = OUTPUT_DIR / "results.csv"
BEST_CIF_CSV   = OUTPUT_DIR / "best_cif.csv"
BEST_CIFS_DIR  = OUTPUT_DIR / "best_cifs"

# =============================================================================
# MSD — framework-relative production average (Allen & Tildesley)
# =============================================================================

from scientific.msd_analysis import detect_equilibration_index


def compute_production_msd_a2(
    msd_curve: np.ndarray,
    time_ps: np.ndarray,
    equilibration_series: np.ndarray | None = None,
) -> tuple[float, int]:
    """
    Time-averaged MSD over the equilibrated production window.

    Uses sliding-block equilibration detection on potential energy when
    available, otherwise on the MSD curve itself.
    """
    if len(msd_curve) == 0:
        return float("nan"), 0

    if equilibration_series is not None and len(equilibration_series) == len(msd_curve):
        eq_idx = detect_equilibration_index(equilibration_series)
    else:
        eq_idx = detect_equilibration_index(msd_curve)

    prod_msd = msd_curve[eq_idx:]
    if len(prod_msd) == 0:
        return float(msd_curve[-1]), eq_idx

    if len(time_ps) == len(msd_curve) and eq_idx < len(time_ps) - 1:
        t_prod = time_ps[eq_idx:]
        dt = np.diff(t_prod, prepend=t_prod[0])
        weights = np.maximum(dt, 0.0)
        if float(weights.sum()) > 0:
            return float(np.average(prod_msd, weights=weights)), eq_idx

    return float(np.mean(prod_msd)), eq_idx


# =============================================================================
# RESULT DATACLASS
# =============================================================================

@dataclass
class MOFScreeningResult:
    """All five screening parameters for one MOF candidate."""

    mof_name: str

    # ── Parameter 1: Adsorption Energy ────────────────────────────────────────
    adsorption_energy_ev: float = float("nan")
    e_ads_1k_ev: float = float("nan")
    e_ads_avg_loaded_ev_per_k: float = float("nan")
    n_k: int = 0
    min_k_k_dist_a: float = float("nan")
    adsorption_label: str = "Unknown"          # e.g. "Moderate", "Strong"
    adsorption_site_type: str = ""
    adsorption_reference: str = "bulk"
    adsorption_k_o_distance_a: float = float("nan")

    # ── Parameter 2: MSD ──────────────────────────────────────────────────────
    msd_a2: float = float("nan")               # final MSD value in Å²

    # ── Parameter 3: Diffusion Coefficient ───────────────────────────────────
    diffusion_coefficient_cm2_s: float = float("nan")
    diffusion_slope_a2_ps: float = float("nan")
    diffusion_fit_r2: float = float("nan")
    diffusion_fit_start: int = -1
    diffusion_fit_end: int = -1
    diffusion_stderr_cm2_s: float = float("nan")
    diffusion_ci_lo_cm2_s: float = float("nan")
    diffusion_ci_hi_cm2_s: float = float("nan")
    slope_stderr_a2_ps: float = float("nan")
    slope_pvalue: float = float("nan")
    equilibration_time_ps: float = float("nan")
    diffusion_reliability: str = "Very Low"   # High / Moderate / Low / Very Low / Rejected
    diffusion_method: str = "none"            # md / neb_tst / none
    diffusion_warning: str = ""
    migration_barrier_eV: float = float("nan")
    hop_distance_A: float = float("nan")
    selected_hop_index: int = -1
    number_of_candidate_hops: int = 0
    rejected_hop_count: int = 0
    barrier_min_eV: float = float("nan")
    barrier_median_eV: float = float("nan")
    barrier_max_eV: float = float("nan")
    diffusion_anomalous_exponent: float = float("nan")
    diffusion_hop_count: int = -1
    diffusion_md_cm2_s: float = float("nan")
    diffusion_ne_tst_cm2_s: float = float("nan")
    diffusion_md_ne_agreement_log10: float = float("nan")

    # ── Parameter 4: Mechanical Stability ────────────────────────────────────
    mechanical_stability: str = "Collapsed"    # Stable / Marginal / Collapsed
    mechanical_stability_index: float = float("nan")  # MSI 0–1
    framework_rmsd: float = float("nan")       # Å
    maximum_displacement: float = float("nan") # Å
    bond_preservation: float = float("nan")    # fraction 0–1
    energy_fluctuation: float = float("nan")
    volume_fluctuation_frac: float = float("nan")
    coordination_preservation: float = float("nan")
    initial_max_force: float = float("nan")
    temperature_mean: float = float("nan")
    temperature_std: float = float("nan")
    energy_drift_ev_per_atom_ps: float = float("nan")
    n_metal_bonds_initial: int = 0
    n_metal_bonds_broken: int = 0
    n_metal_bonds_new: int = 0
    coordination_start: dict = field(default_factory=dict)
    coordination_end: dict = field(default_factory=dict)
    metal_coordination: str = ""
    insertion_n_max_stable: int = -1
    insertion_capacity_mAh_g: float = float("nan")
    insertion_average_voltage_V: float = float("nan")
    insertion_max_volume_expansion_percent: float = float("nan")

    # ── Parameter 5: Volume Change ────────────────────────────────────────────
    volume_bare_A3: float = float("nan")
    volume_k_A3: float = float("nan")
    volume_change_pct: float = float("nan")    # 100 * (V_k - V_bare) / V_bare

    # ── Metadata ──────────────────────────────────────────────────────────────
    simulation_time_ps: float = float("nan")
    stage_used: str = "none"                   # "stage1" or "stage2"
    confidence_score: float = float("nan")
    overall_score: float = float("nan")
    recommendation: str = "Not Recommended"
    reason: str = "Analysis did not complete."
    flags: List[str] = field(default_factory=list)


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("chgnett_analysis")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(output_dir / "analysis.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# =============================================================================
# HELPER UTILITIES
# =============================================================================

def get_time_axis_ps(log_path: Optional[Path], n_frames: int) -> np.ndarray:
    """Build a time axis in ps, parsing the MD log if available."""
    if log_path and log_path.is_file():
        try:
            times: list[float] = []
            with open(log_path, encoding="utf-8") as f:
                header = f.readline()
                if "Time[ps]" in header:
                    for line in f:
                        parts = line.split()
                        if parts:
                            times.append(float(parts[0]))
            if len(times) == n_frames:
                return np.array(times, dtype=float)
            if len(times) > 1:
                dt = times[1] - times[0]
                return np.arange(n_frames) * dt
        except Exception:
            pass

    # Fallback: reconstruct from pipeline configuration
    timestep_fs  = getattr(run_pipeline, "TIME_STEP_FS", 1.0)
    log_interval = getattr(run_pipeline, "MD_LOGINTERVAL", 1)
    dt_ps = (timestep_fs * log_interval) / 1000.0
    return np.arange(n_frames) * dt_ps


def parse_md_log_correct(
    log_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse the CHGNet MD log file with the correct column order.

    Actual log format (confirmed from file header):
        Time[ps]  Etot[eV]  Epot[eV]  Ekin[eV]  T[K]

    Returns (temperature_K, potential_energy_eV, total_energy_eV).
    Note: run_pipeline.parse_md_log reads the WRONG columns (written for a
    different column order), so we always use this function instead.
    """
    temps:    list[float] = []
    pot_evs:  list[float] = []
    tot_evs:  list[float] = []

    try:
        with open(log_path, encoding="utf-8") as f:
            header = f.readline()  # consume header line
            # Detect column order from header
            cols = header.split()
            # Expected: Time[ps] Etot[eV] Epot[eV] Ekin[eV] T[K]
            # Find indices dynamically so we are robust to any ordering
            try:
                t_col    = next(i for i, c in enumerate(cols) if "T[K]" in c or c == "T")
                epot_col = next(i for i, c in enumerate(cols) if "Epot" in c)
                etot_col = next(i for i, c in enumerate(cols) if "Etot" in c)
            except StopIteration:
                # Fallback to known position if header labels differ
                t_col, epot_col, etot_col = 4, 2, 1

            for line in f:
                parts = line.split()
                if len(parts) <= max(t_col, epot_col, etot_col):
                    continue
                try:
                    temps.append(float(parts[t_col]))
                    pot_evs.append(float(parts[epot_col]))
                    tot_evs.append(float(parts[etot_col]))
                except ValueError:
                    continue
    except Exception:
        pass

    return (
        np.asarray(temps,   dtype=float),
        np.asarray(pot_evs, dtype=float),
        np.asarray(tot_evs, dtype=float),
    )


def load_isolated_k_energy(
    mof_dir: Path, outputs_dir: Path, logger: logging.Logger
) -> float:
    """Load cached isolated-K energy or recompute via CHGNet."""
    for path in [
        mof_dir / "isolated_K_energy.json",
        outputs_dir / "isolated_K_energy.json",
        Path("isolated_K_energy.json"),
    ]:
        if path.is_file():
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                energy = float(data["energy_eV"])
                logger.info("Isolated K energy from %s: %.6f eV", path.name, energy)
                return energy
            except (json.JSONDecodeError, KeyError, ValueError) as err:
                logger.warning("Could not read %s: %s", path, err)

    logger.info("Isolated K energy not cached — computing via CHGNet ...")
    return run_pipeline.get_isolated_k_energy()


def select_best_trajectory(
    mof_dir: Path, logger: logging.Logger
) -> tuple[Optional[Path], Optional[Path], str]:
    """
    Choose Stage 2 trajectory if it exists and has ≥1000 frames,
    otherwise fall back to Stage 1.

    Returns (traj_path, log_path, stage_label).
    """
    for stage in ("stage2", "stage1"):
        traj = mof_dir / f"{stage}_md.traj"
        log  = mof_dir / f"{stage}_md.log"
        if traj.is_file():
            try:
                with Trajectory(str(traj), mode="r") as t:
                    n = len(t)
                if stage == "stage2" and n < 1000:
                    logger.info(
                        "  Stage 2 trajectory incomplete (%d frames) — trying Stage 1", n
                    )
                    continue
                logger.info("  Using %s trajectory (%d frames)", stage, n)
                return (
                    traj,
                    log if log.is_file() else None,
                    stage,
                )
            except Exception as exc:
                logger.warning("  Cannot read %s: %s", traj.name, exc)

    return None, None, "none"


def _structure_matches_atoms(structure: Structure, atoms) -> bool:
    return len(structure) == len(atoms) and [
        site.specie.symbol for site in structure
    ] == atoms.get_chemical_symbols()


def _load_frame0_structure(traj_path: Path) -> tuple[object, Structure]:
    from pymatgen.io.ase import AseAtomsAdaptor

    with Trajectory(str(traj_path), mode="r") as trajectory:
        frame = trajectory[0]
    return frame, AseAtomsAdaptor.get_structure(frame)


# =============================================================================
# PARAMETER 3: DIFFUSION COEFFICIENT — FALLBACK STRATEGY
#
#   Tier 1 (MD)      : Linear MSD fit (Einstein) — Stage 2 production MD only (≥10 ps).
#   Tier 2 (NEB+TST) : CHGNet NEB-lite barrier + Arrhenius/TST (Stage 1 fallback).
#   Tier 3 (BVSE)    : Bond-valence site energy landscape → barrier → Arrhenius.
#   Tier 4 (floor)   : Screening minimum D when all tiers fail (effectively immobile).
# =============================================================================

# ── Physical constants ────────────────────────────────────────────────────────
_KB_EV   = 8.617333e-5   # Boltzmann constant  (eV/K)
_H_EV_S  = 4.135668e-15  # Planck constant     (eV·s)
_T_K     = 300.0         # Temperature for TST/Arrhenius estimates

# Attempt-frequency prefactor used when TST hopping rate is converted to D
# via  D ≈ Γ · l² / 6  (l = hop distance in Å → convert to cm)
_NU0_HZ  = 1e12          # 1 THz, typical phonon attempt frequency

# NEB-lite: number of images interpolated between two K-ion sites
_NEB_N_IMAGES = 18
_NEB_RELAX_STEPS = 30
_NEB_RELAX_FMAX = 0.25
_NEB_MIN_K_FW_DIST_A = 1.5
_NEB_MIN_VALID_IMAGE_FRACTION = 0.70
_NEB_MAX_HOPS_TOTAL = 12
_NEB_MAX_HOPS_PER_K = 4
_NEB_MIN_HOP_A = 1.0
_NEB_MAX_HOP_A = 6.0
_NEB_SELECT_MIN_HOP_A = 1.0
_NEB_SELECT_MAX_HOP_A = 5.0
_NEB_MIN_BARRIER_EV = 0.10
_NEB_MAX_BARRIER_EV = 2.0
_NEB_SANITY_BARRIER_MAX_EV = 1.5
_NEB_LOW_BARRIER_TOLERANCE_EV = 0.02
_NEB_OVERLAP_MIN_FW_A = 1.0
_NEB_FAST_DIFFUSION_CM2_S = 1e-7
_NEB_PROFILE_SMOOTHNESS_MAX = 1.5

# BVSE: grid resolution in Å for the vacancy scan
_BVSE_GRID_SPACING_A = 0.3

# ── Tier 1: MD linear MSD fitting (delegates to run_pipeline.evaluate_md_diffusion) ──

# ── Tier 2: NEB-lite + TST ────────────────────────────────────────────────────

def _tst_diffusion_cm2_s(barrier_ev: float, hop_len_a: float) -> float:
    l_cm = hop_len_a * 1e-8
    gamma = _NU0_HZ * np.exp(-barrier_ev / (_KB_EV * _T_K))
    return float(gamma * (l_cm ** 2) / 6.0)


def _tier1_localized_rejection(rejection_reason: str, msd_lab_final: float) -> bool:
    reason = rejection_reason.lower()
    if "localized" in reason or "msd–d inconsistency" in reason or "msd-d inconsistency" in reason:
        return True
    return np.isfinite(msd_lab_final) and msd_lab_final < run_pipeline.CANDIDATE_MSD_MIN_A2


def _neb_screening_range_warning(d_cm2_s: float) -> str:
    if not np.isfinite(d_cm2_s):
        return ""
    lo = run_pipeline.CANDIDATE_DIFFUSION_MIN_CM2_S
    hi = run_pipeline.CANDIDATE_DIFFUSION_MAX_CM2_S
    if d_cm2_s < lo or d_cm2_s > hi:
        return "Outside expected screening range"
    return ""


def _neb_physical_sanity(
    barrier_ev: float,
    hop_len_a: float,
    d_cm2_s: float,
) -> tuple[str, str]:
    """Return (reliability_cap, warning) for NEB/TST estimates."""
    warning_parts: list[str] = []
    reliability = ""

    if np.isfinite(hop_len_a) and (
        hop_len_a < _NEB_SELECT_MIN_HOP_A or hop_len_a > _NEB_SELECT_MAX_HOP_A
    ):
        warning_parts.append(
            f"hop distance {hop_len_a:.2f} Å outside preferred {_NEB_SELECT_MIN_HOP_A:.1f}–"
            f"{_NEB_SELECT_MAX_HOP_A:.1f} Å"
        )
        reliability = "Low"
    if np.isfinite(barrier_ev) and (
        barrier_ev < _NEB_MIN_BARRIER_EV or barrier_ev > _NEB_SANITY_BARRIER_MAX_EV
    ):
        warning_parts.append(
            f"barrier {barrier_ev:.3f} eV outside preferred "
            f"{_NEB_MIN_BARRIER_EV:.2f}–{_NEB_SANITY_BARRIER_MAX_EV:.2f} eV"
        )
        reliability = "Low"
    range_warn = _neb_screening_range_warning(d_cm2_s)
    if range_warn:
        warning_parts.append(range_warn)
        reliability = "Low"
    return reliability, "; ".join(warning_parts)


def _low_barrier_corroborated(hops: list[dict]) -> set[int]:
    """Hop IDs with barrier < 0.10 eV that lack independent corroboration."""
    low = [h for h in hops if h["barrier_ev"] < _NEB_MIN_BARRIER_EV]
    if not low:
        return set()
    uncorroborated: set[int] = set()
    for hop in low:
        matches = [
            other
            for other in hops
            if other["hop_id"] != hop["hop_id"]
            and abs(other["barrier_ev"] - hop["barrier_ev"]) <= _NEB_LOW_BARRIER_TOLERANCE_EV
            and (
                other["moving_k_index"] != hop["moving_k_index"]
                or float(np.linalg.norm(other["c_end"] - hop["c_end"])) > 0.8
            )
        ]
        if not matches:
            uncorroborated.add(hop["hop_id"])
    return uncorroborated


def _hop_selection_score(hop: dict) -> tuple:
    rel_rank = {"High": 4, "Moderate": 3, "Low": 2, "Very Low": 1}.get(
        hop["diffusion_reliability"], 0
    )
    ideal_hop = 3.0
    hop_penalty = abs(hop["hop_length_a"] - ideal_hop)
    return (-rel_rank, hop["smoothness"], hop_penalty, -hop["converged_frac"])

def _min_k_framework_distance(
    cart_k: np.ndarray,
    fw_cart: np.ndarray,
    lattice: np.ndarray,
) -> float:
    if len(fw_cart) == 0:
        return float("inf")
    raw = cart_k[None, :] - fw_cart
    frac = raw @ np.linalg.inv(lattice)
    frac -= np.round(frac)
    mic = frac @ lattice
    return float(np.min(np.linalg.norm(mic, axis=1)))


def _candidate_hops_for_k(
    lattice: np.ndarray,
    cart_k: np.ndarray,
    fw_cart: np.ndarray,
    other_k_cart: np.ndarray,
) -> list[np.ndarray]:
    """Generate nearby low-density candidate endpoints for one K ion."""
    a_vec, b_vec, c_vec = lattice
    dirs = [
        a_vec, -a_vec, b_vec, -b_vec, c_vec, -c_vec,
        a_vec + b_vec, a_vec - b_vec, -a_vec + b_vec, -(a_vec + b_vec),
        a_vec + c_vec, a_vec - c_vec, b_vec + c_vec, b_vec - c_vec,
    ]
    dirs = [d / max(np.linalg.norm(d), 1e-8) for d in dirs]
    radii = (2.5, 3.2, 4.0)
    candidates: list[tuple[float, np.ndarray]] = []

    for r in radii:
        for u in dirs:
            end = cart_k + r * u
            hop = float(np.linalg.norm(end - cart_k))
            if hop < _NEB_MIN_HOP_A or hop > _NEB_MAX_HOP_A:
                continue
            min_fw = _min_k_framework_distance(end, fw_cart, lattice)
            if min_fw < _NEB_MIN_K_FW_DIST_A:
                continue
            if len(other_k_cart):
                kk = np.linalg.norm(other_k_cart - end[None, :], axis=1)
                if float(np.min(kk)) < 1.8:
                    continue
            # Lower framework crowding score is better for a vacancy-like endpoint.
            score = 1.0 / max(min_fw, 1e-6)
            candidates.append((score, end))

    if not candidates:
        return []
    candidates.sort(key=lambda x: x[0])
    uniq: list[np.ndarray] = []
    for _, end in candidates:
        if all(np.linalg.norm(end - u) > 0.8 for u in uniq):
            uniq.append(end)
        if len(uniq) >= _NEB_MAX_HOPS_PER_K:
            break
    return uniq


def _relax_image_energy(
    base_struct: "Structure",
    moving_k_idx: int,
    cart_target: np.ndarray,
    k_indices: list[int],
    calc,
    lattice: np.ndarray,
) -> tuple[float, bool, float]:
    """Relax only inserted K ions and return (energy, converged, min_k_fw_dist)."""
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.core import Species

    struct_img = base_struct.copy()
    frac_target = np.linalg.solve(lattice.T, cart_target)
    struct_img.replace(
        moving_k_idx,
        species=Species("K"),
        coords=frac_target,
        coords_are_cartesian=False,
    )
    atoms = AseAtomsAdaptor.get_atoms(struct_img)
    fixed_mask = np.array([i not in k_indices for i in range(len(atoms))], dtype=bool)
    atoms.set_constraint(FixAtoms(mask=fixed_mask))
    atoms.calc = calc

    converged = False
    try:
        opt = FIRE(atoms, logfile=None)
        converged = bool(opt.run(fmax=_NEB_RELAX_FMAX, steps=_NEB_RELAX_STEPS))
    except Exception:
        converged = False

    try:
        energy = float(atoms.get_potential_energy())
    except Exception:
        return float("nan"), converged, float("nan")

    pos = atoms.get_positions()
    fw_idx = [i for i in range(len(atoms)) if i not in k_indices]
    min_fw = float("inf")
    if fw_idx:
        min_fw = _min_k_framework_distance(pos[moving_k_idx], pos[fw_idx], atoms.get_cell().array)
    return energy, converged, min_fw


def _path_reliability(
    barrier_ev: float,
    hop_len_a: float,
    n_valid: int,
    n_total: int,
    converged_frac: float,
    smoothness: float,
    min_fw_dist: float,
) -> str:
    score = 0
    if n_valid >= int(0.9 * n_total):
        score += 2
    elif n_valid >= int(0.8 * n_total):
        score += 1
    if converged_frac >= 0.85:
        score += 2
    elif converged_frac >= 0.65:
        score += 1
    if smoothness <= 0.15:
        score += 2
    elif smoothness <= 0.30:
        score += 1
    if 2.0 <= hop_len_a <= 5.0:
        score += 1
    if min_fw_dist >= 1.8:
        score += 1
    if 0.15 <= barrier_ev <= 0.8:
        score += 2
    elif barrier_ev <= 1.2:
        score += 1

    if score >= 8:
        return "High"
    if score >= 6:
        return "Moderate"
    if score >= 4:
        return "Low"
    return "Very Low"


def _neb_tst_diffusion(
    k_opt: "Structure",
    k_indices: list,
    logger: logging.Logger,
) -> dict:
    """
    Fast NEB-lite with constrained image relaxations (K-only) for screening.

    Physically correct barrier selection:
      Among hops that survive quality filters (smooth profile, converged
      images, no framework overlap), choose the LOWEST reliable migration
      barrier. The rate-limiting elementary hop that dominates Arrhenius
      kinetics at finite T is the lowest valid E_a on the connected network
      (transition-state theory), not the median of sampled hops.

    Median selection was previously used as a conservative screen but is
    physically incorrect for reporting activation energy / D.
    """
    try:
        from pymatgen.io.ase import AseAtomsAdaptor
        calc = run_pipeline.get_calculator()
    except Exception as exc:
        logger.warning("  [NEB+TST] CHGNet calculator unavailable: %s", exc)
        return {}

    if len(k_indices) < 1:
        logger.warning("  [NEB+TST] No K indices supplied — skipping")
        return {}

    try:
        lattice = k_opt.lattice.matrix
        atoms_ref = AseAtomsAdaptor.get_atoms(k_opt)
        pos_ref = atoms_ref.get_positions()
        fw_idx = [i for i in range(len(atoms_ref)) if i not in set(k_indices)]
        fw_cart = pos_ref[fw_idx] if fw_idx else np.empty((0, 3))

        hop_tasks: list[tuple[int, np.ndarray, np.ndarray]] = []
        for moving_idx in k_indices:
            start = pos_ref[moving_idx]
            other_idx = [i for i in k_indices if i != moving_idx]
            other_k = pos_ref[other_idx] if other_idx else np.empty((0, 3))
            for end in _candidate_hops_for_k(lattice, start, fw_cart, other_k):
                hop_tasks.append((moving_idx, start, end))
        hop_tasks = hop_tasks[:_NEB_MAX_HOPS_TOTAL]
        n_candidates = len(hop_tasks)

        logger.info("  [NEB+TST] Generated %d candidate hop(s)", n_candidates)
        if not hop_tasks:
            logger.warning("  [NEB+TST] No physically valid candidate hops found")
            return {"number_of_candidate_hops": 0, "rejected_hops": []}

        rejected_hops: list[dict] = []
        valid_hops: list[dict] = []

        for hop_id, (moving_idx, c_start, c_end) in enumerate(hop_tasks, start=1):
            hop_len_a = float(np.linalg.norm(c_end - c_start))
            min_fw_start = _min_k_framework_distance(c_start, fw_cart, lattice)
            min_fw_end = _min_k_framework_distance(c_end, fw_cart, lattice)
            min_fw_hop = min(min_fw_start, min_fw_end)

            if hop_len_a < _NEB_SELECT_MIN_HOP_A or hop_len_a > _NEB_SELECT_MAX_HOP_A:
                reason = (
                    f"hop distance {hop_len_a:.2f} Å outside "
                    f"{_NEB_SELECT_MIN_HOP_A:.1f}–{_NEB_SELECT_MAX_HOP_A:.1f} Å"
                )
                rejected_hops.append({"hop_id": hop_id, "reason": reason})
                logger.info("  [NEB-lite] Hop %d rejected: %s", hop_id, reason)
                continue
            if min_fw_hop < _NEB_MIN_K_FW_DIST_A:
                reason = (
                    f"unrealistic geometry (min K-framework {min_fw_hop:.2f} Å "
                    f"< {_NEB_MIN_K_FW_DIST_A:.2f} Å)"
                )
                rejected_hops.append({"hop_id": hop_id, "reason": reason})
                logger.info("  [NEB-lite] Hop %d rejected: %s", hop_id, reason)
                continue

            images_cart = np.linspace(c_start, c_end, _NEB_N_IMAGES + 2)
            energies = np.full(len(images_cart), np.nan, dtype=float)
            converged = np.zeros(len(images_cart), dtype=bool)
            min_fw_profile = np.full(len(images_cart), np.nan, dtype=float)

            for ii, img_cart in enumerate(images_cart):
                e, conv, min_fw = _relax_image_energy(
                    k_opt, moving_idx, img_cart, k_indices, calc, lattice
                )
                energies[ii] = e
                converged[ii] = conv
                min_fw_profile[ii] = min_fw
                if np.isfinite(min_fw) and min_fw < _NEB_OVERLAP_MIN_FW_A:
                    energies[ii] = float("nan")

            valid = np.isfinite(energies)
            n_valid = int(np.sum(valid))
            n_total = len(energies)
            n_conv = int(np.sum(converged & valid))
            converged_frac = float(n_conv / max(n_valid, 1))

            if n_valid < int(np.ceil(_NEB_MIN_VALID_IMAGE_FRACTION * n_total)):
                reason = f"poor image convergence ({n_valid}/{n_total} valid images)"
                rejected_hops.append({"hop_id": hop_id, "reason": reason})
                logger.info("  [NEB-lite] Hop %d rejected: %s", hop_id, reason)
                continue
            if not (valid[0] and valid[-1]):
                reason = "invalid endpoints after relaxation"
                rejected_hops.append({"hop_id": hop_id, "reason": reason})
                logger.info("  [NEB-lite] Hop %d rejected: %s", hop_id, reason)
                continue
            if float(np.nanmin(min_fw_profile)) < _NEB_OVERLAP_MIN_FW_A:
                reason = (
                    f"atomic overlap along path (min K-framework "
                    f"{float(np.nanmin(min_fw_profile)):.2f} Å)"
                )
                rejected_hops.append({"hop_id": hop_id, "reason": reason})
                logger.info("  [NEB-lite] Hop %d rejected: %s", hop_id, reason)
                continue

            valid_e = energies[valid]
            e_ref = float(min(energies[0], energies[-1]))
            e_peak = float(np.max(valid_e))
            barrier = max(0.0, e_peak - e_ref)

            diffs = np.abs(np.diff(valid_e))
            smoothness = (
                float(np.std(diffs) / max(np.mean(diffs), 1e-6)) if len(diffs) > 1 else 0.0
            )
            if smoothness > _NEB_PROFILE_SMOOTHNESS_MAX:
                reason = f"discontinuous energy profile (smoothness={smoothness:.2f})"
                rejected_hops.append({"hop_id": hop_id, "reason": reason})
                logger.info("  [NEB-lite] Hop %d rejected: %s", hop_id, reason)
                continue
            if barrier > _NEB_MAX_BARRIER_EV:
                reason = f"barrier too high ({barrier:.3f} eV > {_NEB_MAX_BARRIER_EV:.2f} eV)"
                rejected_hops.append({"hop_id": hop_id, "reason": reason})
                logger.info("  [NEB-lite] Hop %d rejected: %s", hop_id, reason)
                continue

            rel = _path_reliability(
                barrier_ev=barrier,
                hop_len_a=hop_len_a,
                n_valid=n_valid,
                n_total=n_total,
                converged_frac=converged_frac,
                smoothness=smoothness,
                min_fw_dist=float(np.nanmin(min_fw_profile)),
            )
            hop_rec = {
                "hop_id": hop_id,
                "moving_k_index": int(moving_idx),
                "barrier_ev": barrier,
                "hop_length_a": hop_len_a,
                "diffusion_reliability": rel,
                "smoothness": smoothness,
                "converged_frac": converged_frac,
                "c_end": c_end.copy(),
                "neb_energies_ev": energies.tolist(),
            }
            valid_hops.append(hop_rec)
            logger.info(
                "  [NEB-lite] Hop %d | K index=%d | hop=%.2f A | min K-fw=%.2f A | "
                "images relaxed=%d/%d | barrier=%.3f eV | reliability=%s",
                hop_id,
                moving_idx,
                hop_len_a,
                float(np.nanmin(min_fw_profile)),
                n_conv,
                n_total,
                barrier,
                rel,
            )

        if not valid_hops:
            logger.warning(
                "  [NEB+TST] No valid relaxed hops survived quality checks (%d rejected)",
                len(rejected_hops),
            )
            return {
                "number_of_candidate_hops": n_candidates,
                "rejected_hops": rejected_hops,
            }

        uncorroborated_low = _low_barrier_corroborated(valid_hops)
        if uncorroborated_low:
            kept: list[dict] = []
            for hop in valid_hops:
                if hop["hop_id"] in uncorroborated_low:
                    reason = (
                        f"barrier {hop['barrier_ev']:.3f} eV < {_NEB_MIN_BARRIER_EV:.2f} eV "
                        "without independent corroboration"
                    )
                    rejected_hops.append({"hop_id": hop["hop_id"], "reason": reason})
                    logger.info("  [NEB-lite] Hop %d rejected: %s", hop["hop_id"], reason)
                else:
                    kept.append(hop)
            valid_hops = kept

        if not valid_hops:
            logger.warning("  [NEB+TST] All hops rejected after barrier corroboration checks")
            return {
                "number_of_candidate_hops": n_candidates,
                "rejected_hops": rejected_hops,
            }

        barriers = [h["barrier_ev"] for h in valid_hops]
        barrier_stats = {
            "min": float(np.min(barriers)),
            "median": float(np.median(barriers)),
            "max": float(np.max(barriers)),
            "n_valid": len(valid_hops),
        }
        logger.info(
            "  [NEB+TST] Barrier statistics: min=%.3f, median=%.3f, max=%.3f eV (%d valid hops)",
            barrier_stats["min"],
            barrier_stats["median"],
            barrier_stats["max"],
            barrier_stats["n_valid"],
        )

        # Select the lowest reliable barrier among quality-filtered hops.
        # Tie-break with path reliability / smoothness via _hop_selection_score.
        def _lowest_reliable_key(h: dict) -> tuple:
            return (
                float(h["barrier_ev"]),
                _hop_selection_score(h),
            )

        selected = sorted(valid_hops, key=_lowest_reliable_key)[0]
        use_barrier = float(selected["barrier_ev"])
        l_a = float(selected["hop_length_a"])
        d_cm2_s = _tst_diffusion_cm2_s(use_barrier, l_a)

        rel_cap, sanity_warn = _neb_physical_sanity(use_barrier, l_a, d_cm2_s)
        reliability = selected["diffusion_reliability"]
        if rel_cap == "Low":
            reliability = "Low"

        warnings: list[str] = []
        if sanity_warn:
            warnings.append(sanity_warn)
        if len(valid_hops) == 1:
            warnings.append("single valid hop; verify with longer MD or DFT-NEB")
        if d_cm2_s > _NEB_FAST_DIFFUSION_CM2_S:
            warnings.append("Potentially non-physical; verify with full DFT-NEB or longer MD.")

        logger.info(
            "  [NEB+TST] Selected hop %d (K index=%d): LOWEST reliable barrier=%.3f eV "
            "(min=%.3f, median=%.3f eV), hop=%.2f A, D=%.3e cm²/s, reliability=%s",
            selected["hop_id"],
            selected["moving_k_index"],
            use_barrier,
            barrier_stats["min"],
            barrier_stats["median"],
            l_a,
            d_cm2_s,
            reliability,
        )

        return {
            "diffusion_cm2_s": d_cm2_s,
            "barrier_ev": use_barrier,
            "hop_length_a": l_a,
            "diffusion_reliability": reliability,
            "migration_barrier_eV": use_barrier,
            "hop_distance_A": l_a,
            "selected_hop_index": int(selected["hop_id"]),
            "number_of_candidate_hops": n_candidates,
            "rejected_hops": rejected_hops,
            "barrier_statistics": barrier_stats,
            "diffusion_warning": "; ".join(warnings),
            "moving_k_index": int(selected["moving_k_index"]),
            "activation_energy_eV": use_barrier,
            "selected_pathway": (
                f"hop_{selected['hop_id']}_K{selected['moving_k_index']}"
            ),
            "neb_energy_profile_ev": list(selected.get("neb_energies_ev", [])),
            "valid_hops": valid_hops,
        }
    except Exception as exc:
        logger.warning("  [NEB+TST] Calculation failed: %s", exc)
        return {}


# ── Tier 3: BVSE ─────────────────────────────────────────────────────────────

def _bvse_diffusion(
    k_opt: "Structure",
    logger: logging.Logger,
) -> dict:
    """
    Estimate D from a Bond-Valence Site Energy (BVSE) landscape scan.

    Method
    ------
    1. Remove all K ions from the structure to get the host framework.
    2. Scan a Cartesian grid at _BVSE_GRID_SPACING_A resolution.
    3. At each grid point insert a test K and evaluate the simplified BVSE
       energy:  E_BVSE(r) = Σ_{j ∈ framework} [exp((R0_j - d_j)/b) - (R0_j/d_j)]^2
       where R0_j are tabulated bond-valence radii and b=0.37 Å.
    4. E_b = max(E_BVSE along minimum-energy path estimate) − min(E_BVSE).
       Here we approximate the path as the difference between the 10th and
       90th percentile of the lowest-energy grid band.
    5. D ≈ ν₀ · exp(−E_b / k_B T) · l² / 6,  l from grid spacing.

    Returns dict with 'diffusion_cm2_s', 'barrier_ev', 'diffusion_reliability',
    or empty dict on failure.
    """
    # Bond-valence parameters R0 (Å) for K–X pairs (from Brown 2002 tables)
    # Key: anion symbol, Value: R0 for K–X bond
    BV_R0: dict[str, float] = {
        "O":  2.132, "F":  2.053, "Cl": 2.593, "Br": 2.732,
        "I":  2.972, "N":  2.284, "S":  2.551, "Se": 2.681,
        "P":  2.342, "As": 2.450, "C":  2.100,
    }
    BV_B = 0.37   # universal softness parameter (Å)

    try:
        framework = k_opt.copy()
        framework.remove_species(["K"])
        lattice = framework.lattice.matrix

        # Extract framework atom positions and R0 values
        fw_symbols = [site.specie.symbol for site in framework]
        fw_cart    = np.array([site.coords for site in framework])
        fw_r0      = np.array([BV_R0.get(s, 2.0) for s in fw_symbols])

        if len(fw_cart) == 0:
            logger.warning("  [BVSE] Empty framework after K removal")
            return {}

        # Build grid in fractional coordinates
        abc     = framework.lattice.abc
        n_pts   = [max(2, int(np.ceil(a / _BVSE_GRID_SPACING_A))) for a in abc]
        fa      = np.linspace(0.0, 1.0, n_pts[0], endpoint=False)
        fb      = np.linspace(0.0, 1.0, n_pts[1], endpoint=False)
        fc      = np.linspace(0.0, 1.0, n_pts[2], endpoint=False)

        # Limit grid to a manageable size (≤ 20^3 = 8000 points)
        max_pts = 8000
        total   = n_pts[0] * n_pts[1] * n_pts[2]
        if total > max_pts:
            factor  = (max_pts / total) ** (1 / 3)
            n_pts   = [max(2, int(n * factor)) for n in n_pts]
            fa      = np.linspace(0.0, 1.0, n_pts[0], endpoint=False)
            fb      = np.linspace(0.0, 1.0, n_pts[1], endpoint=False)
            fc      = np.linspace(0.0, 1.0, n_pts[2], endpoint=False)

        fa3, fb3, fc3 = np.meshgrid(fa, fb, fc, indexing="ij")
        frac_grid     = np.stack([fa3.ravel(), fb3.ravel(), fc3.ravel()], axis=1)  # (N, 3)
        cart_grid     = frac_grid @ lattice                                          # (N, 3)

        # Vectorised BVSE: compute pairwise distances with PBC via minimum image
        n_grid     = len(cart_grid)
        bvse_vals  = np.zeros(n_grid, dtype=float)
        cutoff_a   = 6.0   # ignore framework atoms beyond this distance

        for gi in range(n_grid):
            r_vec    = cart_grid[gi] - fw_cart           # (M, 3)
            # Minimum image convention
            frac_dv  = r_vec @ np.linalg.inv(lattice)
            frac_dv -= np.round(frac_dv)
            r_mic    = frac_dv @ lattice
            dists    = np.linalg.norm(r_mic, axis=1)   # (M,)
            mask     = (dists > 0.5) & (dists < cutoff_a)
            if not np.any(mask):
                bvse_vals[gi] = 0.0
                continue
            d_m  = dists[mask]
            r0_m = fw_r0[mask]
            bv   = np.exp((r0_m - d_m) / BV_B) - (r0_m / d_m)
            bvse_vals[gi] = float(np.sum(bv ** 2))

        e_min  = float(np.min(bvse_vals))
        e_low  = float(np.percentile(bvse_vals, 10))  # accessible low-E band
        e_high = float(np.percentile(bvse_vals, 30))  # saddle-point estimate
        e_b    = max(0.0, e_high - e_low)

        # Crude scaling: BVSE energies are dimensionless sums-of-squares;
        # empirically, 1 BVSE unit ≈ 0.05 eV for K in oxide frameworks.
        # This is a rough heuristic — treat the result accordingly.
        BVSE_TO_EV = 0.05
        e_b_ev = e_b * BVSE_TO_EV
        # Collect low-energy sites as NEB pathway guidance only.
        order = np.argsort(bvse_vals)
        low_fracs = [frac_grid[i].tolist() for i in order[:12]]
        suggested_hop = float(_BVSE_GRID_SPACING_A * 3.0)

        d_cm2_s = _tst_diffusion_cm2_s(e_b_ev, suggested_hop)
        logger.info(
            "  [BVSE] E_b(raw)=%.3f (~%.3f eV), hop~%.2f Å, D~%.3e cm²/s "
            "(%d low-E sites; used only when MD/NEB fail)",
            e_b,
            e_b_ev,
            suggested_hop,
            d_cm2_s,
            len(low_fracs),
        )
        return {
            "diffusion_cm2_s": d_cm2_s,
            "migration_barrier_eV": e_b_ev,
            "hop_distance_A": suggested_hop,
            "barrier_ev_estimate": e_b_ev,
            "suggested_hop_a": suggested_hop,
            "low_energy_fracs": low_fracs,
            "diffusion_reliability": "Very Low",
        }

    except Exception as exc:
        logger.warning("  [BVSE] Calculation failed: %s", exc)
        return {}


# ── Master orchestrator: MD → NEB+TST → BVSE → failure ───────────────────────

from scientific.diffusion_analysis import (
    apply_best_estimate_policy,
    classify_diffusion_with_hops,
    detect_hop_events,
    md_neb_agreement_log10,
)
from scientific.diffusion_plots import plot_diffusion_analysis
from scientific.msd_analysis import unwrap_positions_mic

def _screening_range_note(d_cm2_s: float) -> str:
    if run_pipeline.diffusion_outside_screening_range(d_cm2_s):
        return (
            f" outside screening range "
            f"({run_pipeline.CANDIDATE_DIFFUSION_MIN_CM2_S:.0e}–"
            f"{run_pipeline.CANDIDATE_DIFFUSION_MAX_CM2_S:.0e} cm²/s)"
        )
    return ""


def _apply_neb_cross_validation(
    neb_result: dict,
    *,
    tier1_rejection: str,
    msd_lab_final: float,
    mechanical_stability: str,
    barrier_stats: dict,
) -> dict:
    """Downgrade NEB reliability when inconsistent with MD observations."""
    out = dict(neb_result)
    warnings = [w for w in str(out.get("diffusion_warning", "")).split("; ") if w]
    reliability = str(out.get("diffusion_reliability", "Very Low"))
    d_val = float(out.get("diffusion_cm2_s", float("nan")))

    localized = _tier1_localized_rejection(tier1_rejection, msd_lab_final)
    if localized and np.isfinite(d_val) and d_val > _NEB_FAST_DIFFUSION_CM2_S:
        reliability = "Low"
        warnings.append("NEB prediction inconsistent with MD trajectory.")

    if mechanical_stability == "Collapsed" and np.isfinite(d_val) and d_val > _NEB_FAST_DIFFUSION_CM2_S:
        reliability = "Low"
        warnings.append("Potentially non-physical; verify with full DFT-NEB or longer MD.")

    n_valid = int(barrier_stats.get("n_valid", 0)) if barrier_stats else 0
    if n_valid == 1 and np.isfinite(d_val) and d_val > _NEB_FAST_DIFFUSION_CM2_S:
        reliability = "Low"
        if "isolated hop" not in " ".join(warnings).lower():
            warnings.append("migration barrier is an isolated outlier")

    out["diffusion_reliability"] = reliability
    out["diffusion_warning"] = "; ".join(dict.fromkeys(warnings))
    return out


def compute_diffusion_coefficient(
    msd_fw_rel: np.ndarray,
    time_ps: np.ndarray,
    k_opt: Optional["Structure"],
    k_indices: list,
    logger: logging.Logger,
    *,
    stage_label: str,
    mechanical_stability: str,
    md_converged: bool,
    flags: list[str],
    msd_lab_final: float,
    simulation_time_ps: float,
    equilibration_series: np.ndarray | None = None,
    hop_count: int = -1,
    diagnostics_path: Path | None = None,
) -> dict:
    """
    Compute diffusion coefficient with MD + always-on NEB cross-check.

    Tier 1 — MD Einstein fit (framework-relative, longest valid window).
    Tier 2 — NEB+TST (always run when K structure available).
    Tier 3 — BVSE Arrhenius fallback when MD and NEB fail.
    Tier 4 — Screening minimum D when all methods fail.

    Best-estimate policy:
      ≥1 hop + valid MD → report MD D
      0 hops (localized K) → report NEB+TST D, keep MD in supplementary fields
    """
    base: dict = {
        "diffusion_cm2_s":       float("nan"),
        "diffusion_reliability": "Very Low",
        "diffusion_method":      "none",
        "fit_r2":                float("nan"),
        "slope_a2_ps":           float("nan"),
        "intercept_a2":          float("nan"),
        "start_frame":           -1.0,
        "end_frame":             -1.0,
        "barrier_ev":            float("nan"),
        "diffusion_note":        "No reliable diffusion pathway found.",
        "diffusion_warning":     "",
        "migration_barrier_eV":  float("nan"),
        "hop_distance_A":        float("nan"),
        "selected_hop_index":    -1,
        "number_of_candidate_hops": 0,
        "rejected_hops":         [],
        "barrier_statistics":    {},
        "anomalous_exponent":    float("nan"),
        "fit_window_ps":         float("nan"),
        "hop_count":             float(hop_count) if hop_count >= 0 else float("nan"),
        "diffusion_md_cm2_s":    float("nan"),
        "diffusion_ne_tst_cm2_s": float("nan"),
        "diffusion_md_ne_agreement_log10": float("nan"),
        "neb_energy_profile_ev": [],
    }

    diagnostics: dict = {
        "stage_label": stage_label,
        "simulation_time_ps": simulation_time_ps,
        "hop_count": hop_count,
        "md": {},
        "neb": {},
        "policy": {},
        "warnings": [],
    }

    # ── Tier 1: MD (Stage 2 production only) ─────────────────────────────────
    tier1_allowed = run_pipeline.md_diffusion_tier1_allowed(stage_label, simulation_time_ps)
    rejection = ""
    md_fit: dict = {}
    d_md = float("nan")
    r2_md = float("nan")
    md_reliability = "Rejected"

    if tier1_allowed:
        logger.info("  [Diffusion] Tier 1: MD linear MSD fit (framework-relative) ...")
        md_fit = run_pipeline.evaluate_md_diffusion(
            msd_fw_rel,
            time_ps,
            mechanical_stability=mechanical_stability,
            md_converged=md_converged,
            flags=flags,
            msd_lab_final=msd_lab_final,
            simulation_time_ps=simulation_time_ps,
            equilibration_series=equilibration_series,
            hop_count=hop_count,
            logger=logger,
        )
        d_md = float(md_fit.get("diffusion_cm2_s", float("nan")))
        r2_md = float(md_fit.get("fit_r2", float("nan")))
        rejection = str(md_fit.get("rejection_reason", ""))
        md_reliability = str(md_fit.get("diffusion_reliability", "Rejected"))
        base["anomalous_exponent"] = float(md_fit.get("anomalous_exponent", float("nan")))
        base["fit_window_ps"] = float(md_fit.get("fit_window_ps", float("nan")))
        for key in (
            "fit_r2", "slope_a2_ps", "intercept_a2", "start_frame", "end_frame",
            "slope_stderr", "slope_pvalue", "diffusion_stderr_cm2_s",
            "diffusion_cm2_s_lo", "diffusion_cm2_s_hi", "equilibration_time_ps",
        ):
            if key in md_fit:
                base[key] = md_fit[key]
        diagnostics["md"] = {
            k: md_fit.get(k) for k in (
                "diffusion_cm2_s", "fit_r2", "anomalous_exponent", "fit_window_ps",
                "start_frame", "end_frame", "num_frames", "diffusion_reliability",
                "rejection_reason", "diffusion_warning", "hop_count",
            )
        }
    elif stage_label == "stage1":
        logger.info(
            "  [Diffusion] Stage 1 fast MD: Tier 1 skipped — using NEB+TST / BVSE."
        )
        rejection = "stage1 screening gate"
    else:
        logger.info(
            "  [Diffusion] Tier 1 skipped — trajectory %.1f ps < %.1f ps required.",
            simulation_time_ps,
            run_pipeline.DIFFUSION_SUFFICIENT_SIM_TIME_PS,
        )
        rejection = "insufficient production MD time"

    md_ok = (
        tier1_allowed
        and np.isfinite(d_md)
        and d_md > 0
        and md_reliability != "Rejected"
    )
    if md_ok:
        md_final = run_pipeline.finalize_reported_diffusion(md_fit, method="md")
        if md_final is not None:
            d_md = float(md_final["diffusion_cm2_s"])
            md_reliability = str(md_final["diffusion_reliability"])
        logger.info(
            "  [Diffusion] Tier 1 (MD): D=%.3e cm²/s, R²=%.3f, α=%.3f, hops=%s, reliability=%s",
            d_md,
            r2_md,
            base["anomalous_exponent"],
            hop_count if hop_count >= 0 else "?",
            md_reliability,
        )
    else:
        logger.info(
            "  [Diffusion] Tier 1 (MD) not accepted: D=%s, R²=%s%s",
            f"{d_md:.3e}" if np.isfinite(d_md) else "NaN",
            f"{r2_md:.3f}" if np.isfinite(r2_md) else "NaN",
            f" ({rejection})" if rejection else "",
        )

    base["diffusion_md_cm2_s"] = (
        d_md if np.isfinite(d_md) and d_md > 0
        else float(md_fit.get("diffusion_md_fit_cm2_s", float("nan")))
    )

    # ── Tier 2: NEB + TST (always when K structure available) ─────────────────
    neb_res: dict = {}
    d_neb = float("nan")
    neb_reliability = "Very Low"

    if k_opt is not None and k_indices:
        logger.info("  [Diffusion] Tier 2: NEB+TST cross-check (always run) ...")
        neb_res = _neb_tst_diffusion(k_opt, k_indices, logger)
        if neb_res:
            neb_res = _apply_neb_cross_validation(
                neb_res,
                tier1_rejection=rejection,
                msd_lab_final=msd_lab_final,
                mechanical_stability=mechanical_stability,
                barrier_stats=neb_res.get("barrier_statistics", {}),
            )
        d_neb = float(neb_res.get("diffusion_cm2_s", float("nan"))) if neb_res else float("nan")
        neb_final = (
            run_pipeline.finalize_reported_diffusion(neb_res, method="neb_tst")
            if neb_res and np.isfinite(d_neb) and d_neb > 0
            else None
        )
        if neb_final is not None:
            d_neb = float(neb_final["diffusion_cm2_s"])
            neb_reliability = str(neb_res.get("diffusion_reliability", neb_final["diffusion_reliability"]))
            logger.info(
                "  [Diffusion] Tier 2 (NEB+TST): D=%.3e cm²/s, E_b=%.3f eV, reliability=%s",
                d_neb,
                neb_res.get("migration_barrier_eV", float("nan")),
                neb_reliability,
            )
        elif np.isfinite(d_neb) and d_neb > 0:
            logger.info("  [Diffusion] Tier 2 (NEB+TST) D failed physical sanity checks.")
        else:
            logger.info("  [Diffusion] Tier 2 (NEB+TST) did not produce a valid D.")
        diagnostics["neb"] = {
            k: neb_res.get(k) for k in (
                "diffusion_cm2_s", "migration_barrier_eV", "hop_distance_A",
                "selected_hop_index", "number_of_candidate_hops", "barrier_statistics",
                "diffusion_reliability", "diffusion_warning", "rejected_hops",
            )
        } if neb_res else {}
    else:
        logger.info("  [Diffusion] Tier 2 (NEB+TST) skipped (no K structure).")

    base["diffusion_ne_tst_cm2_s"] = d_neb if np.isfinite(d_neb) and d_neb > 0 else float("nan")
    agreement = md_neb_agreement_log10(d_md, d_neb)
    base["diffusion_md_ne_agreement_log10"] = agreement

    # ── Best-estimate policy ──────────────────────────────────────────────────
    effective_hop_count = hop_count if hop_count >= 0 else 0
    d_final, method, rel = apply_best_estimate_policy(
        d_md,
        d_neb,
        hop_count=effective_hop_count,
        md_reliability=md_reliability,
        neb_reliability=neb_reliability,
    )

    alpha = float(base.get("anomalous_exponent", float("nan")))
    outside = bool(np.isfinite(d_final) and run_pipeline.diffusion_outside_screening_range(d_final))
    rel = classify_diffusion_with_hops(
        rel,
        hop_count=effective_hop_count,
        anomalous_exponent=alpha,
        md_neb_log10_agreement=agreement,
        outside_screening_range=outside,
    )

    warnings: list[str] = []
    if str(md_fit.get("diffusion_warning", "")):
        warnings.append(str(md_fit["diffusion_warning"]))
    if neb_res and str(neb_res.get("diffusion_warning", "")):
        warnings.append(str(neb_res["diffusion_warning"]))
    if hop_count == 0 and method == "neb_tst":
        warnings.append(
            "no K hops in MD trajectory; reporting NEB+TST D (MD estimate in Diffusion_MD_cm2_s)"
        )
    if np.isfinite(agreement) and agreement > 1.5:
        warnings.append(f"MD-NEB disagreement: {agreement:.2f} log₁₀ decades")

    diagnostics["policy"] = {
        "reported_d_cm2_s": d_final,
        "method": method,
        "reliability": rel,
        "md_neb_agreement_log10": agreement,
        "hop_count": effective_hop_count,
    }

    # ── Populate result or fall through to BVSE guidance ─────────────────────
    if np.isfinite(d_final) and d_final > 0 and run_pipeline.diffusion_passes_physical_sanity(d_final):
        base["diffusion_cm2_s"] = d_final
        base["diffusion_method"] = method
        base["diffusion_reliability"] = rel
        base["diffusion_warning"] = "; ".join(dict.fromkeys(w for w in warnings if w))
        if neb_res:
            base["migration_barrier_eV"] = float(neb_res.get("migration_barrier_eV", float("nan")))
            base["hop_distance_A"] = float(neb_res.get("hop_distance_A", float("nan")))
            base["selected_hop_index"] = int(neb_res.get("selected_hop_index", -1))
            base["number_of_candidate_hops"] = int(neb_res.get("number_of_candidate_hops", 0))
            base["rejected_hops"] = list(neb_res.get("rejected_hops", []))
            base["barrier_statistics"] = dict(neb_res.get("barrier_statistics", {}))
            base["barrier_ev"] = float(neb_res.get("barrier_ev", float("nan")))
            base["neb_energy_profile_ev"] = list(neb_res.get("neb_energy_profile_ev", []))
        range_note = _screening_range_note(d_final)
        if method == "md":
            base["diffusion_note"] = (
                f"MD Einstein fit: D={d_final:.3e} cm²/s, R²={r2_md:.3f}, "
                f"hops={effective_hop_count} [{rel}]{range_note}"
            )
        else:
            base["diffusion_note"] = (
                f"NEB+TST estimate: D={d_final:.3e} cm²/s, "
                f"E_b={base['migration_barrier_eV']:.3f} eV "
                f"[{rel}]{range_note}"
            )
        logger.info(
            "  [Diffusion] Final D=%.3e cm²/s (method=%s, reliability=%s, agreement=%s)",
            d_final,
            method,
            rel,
            f"{agreement:.2f}" if np.isfinite(agreement) else "N/A",
        )
        if diagnostics_path is not None:
            diagnostics["warnings"] = warnings
            diagnostics["final"] = dict(base)
            try:
                diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
                with open(diagnostics_path, "w", encoding="utf-8") as f:
                    json.dump(diagnostics, f, indent=2, default=str)
            except OSError as exc:
                logger.warning("  [Diffusion] Could not write diagnostics: %s", exc)
        return base

    # ── Tier 3: BVSE Arrhenius fallback ───────────────────────────────────────
    d_bvse = float("nan")
    bvse_res: dict = {}
    if k_opt is not None:
        bvse_res = _bvse_diffusion(k_opt, logger)
        if bvse_res:
            base["bvse_guidance"] = bvse_res
            diagnostics["bvse"] = bvse_res
            d_bvse = float(bvse_res.get("diffusion_cm2_s", float("nan")))

    if np.isfinite(d_bvse) and d_bvse > 0 and run_pipeline.diffusion_passes_physical_sanity(d_bvse):
        base["diffusion_cm2_s"] = d_bvse
        base["diffusion_method"] = "bvse"
        base["diffusion_reliability"] = "Very Low"
        base["migration_barrier_eV"] = float(bvse_res.get("migration_barrier_eV", float("nan")))
        base["hop_distance_A"] = float(bvse_res.get("hop_distance_A", float("nan")))
        base["barrier_ev"] = float(bvse_res.get("barrier_ev_estimate", float("nan")))
        range_note = _screening_range_note(d_bvse)
        base["diffusion_note"] = (
            f"BVSE fallback: D={d_bvse:.3e} cm²/s, "
            f"E_b={base['migration_barrier_eV']:.3f} eV [Very Low]{range_note}"
        )
        base["diffusion_warning"] = "; ".join(dict.fromkeys(
            w for w in [
                *warnings,
                "MD and NEB failed; BVSE Arrhenius estimate used as fallback",
            ] if w
        ))
        logger.info(
            "  [Diffusion] Tier 3 (BVSE fallback): D=%.3e cm²/s, E_b=%.3f eV",
            d_bvse,
            base["migration_barrier_eV"],
        )
        if diagnostics_path is not None:
            diagnostics["warnings"] = warnings + ["BVSE fallback used"]
            diagnostics["final"] = dict(base)
            try:
                diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
                with open(diagnostics_path, "w", encoding="utf-8") as f:
                    json.dump(diagnostics, f, indent=2, default=str)
            except OSError as exc:
                logger.warning("  [Diffusion] Could not write diagnostics: %s", exc)
        return base

    # ── Tier 4: screening floor (always report a finite D) ────────────────────
    d_floor = float(run_pipeline.CANDIDATE_DIFFUSION_MIN_CM2_S)
    base["diffusion_cm2_s"] = d_floor
    base["diffusion_method"] = "floor"
    base["diffusion_reliability"] = "Very Low"
    base["diffusion_note"] = (
        f"Screening floor: D={d_floor:.3e} cm²/s (MD, NEB, and BVSE all failed)"
    )
    base["diffusion_warning"] = "; ".join(dict.fromkeys(
        w for w in [
            *warnings,
            "no reliable diffusion pathway; reporting screening minimum D",
        ] if w
    ))
    logger.warning(
        "  [Diffusion] All tiers failed; reporting screening floor D=%.3e cm²/s",
        d_floor,
    )
    if diagnostics_path is not None:
        diagnostics["warnings"] = warnings
        diagnostics["final"] = dict(base)
        try:
            diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(diagnostics_path, "w", encoding="utf-8") as f:
                json.dump(diagnostics, f, indent=2, default=str)
        except OSError as exc:
            logger.warning("  [Diffusion] Could not write diagnostics: %s", exc)
    return base


# =============================================================================
# PLOTTING
# =============================================================================

def generate_plots(
    mof_name: str,
    stage_label: str,
    time_ps: np.ndarray,
    msd: np.ndarray,
    temp_vals: np.ndarray,
    energy_ev: np.ndarray,
    fit_start: int,
    fit_end: int,
    fit_slope: float,
    fit_intercept: float,
    output_dir: Path,
) -> None:
    """Save MSD, energy, and temperature diagnostic plots."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ── MSD vs Time ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(time_ps, msd, color="#1f77b4", lw=2, label="MSD (K, framework-relative)")
    if (
        0 <= fit_start < len(time_ps)
        and 0 <= fit_end < len(time_ps)
        and np.isfinite(fit_slope)
    ):
        t_fit = time_ps[fit_start : fit_end + 1]
        y_fit = fit_slope * t_fit + fit_intercept
        ax.plot(t_fit, y_fit, color="#d62728", ls="--", lw=2, label="Linear fit (6Dt)")
        ax.axvspan(t_fit[0], t_fit[-1], color="#ff7f0e", alpha=0.15, label="Fitted region")
    ax.set_xlabel("Time (ps)", fontsize=10)
    ax.set_ylabel("MSD (Å²)", fontsize=10)
    ax.set_title(f"{mof_name} ({stage_label}) — K-ion MSD (framework-relative)", fontsize=11, fontweight="bold")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="upper left", frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / f"{mof_name}_msd_vs_time.png", dpi=150)
    plt.close(fig)

    # ── Potential Energy vs Time ──────────────────────────────────────────────
    if len(energy_ev) > 0:
        t_ax = np.linspace(0.0, time_ps[-1], len(energy_ev))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(t_ax, energy_ev, color="#2ca02c", lw=1.5)
        ax.set_xlabel("Time (ps)", fontsize=10)
        ax.set_ylabel("Potential energy (eV)", fontsize=10)
        ax.set_title(f"{mof_name} — Energy vs Time", fontsize=11, fontweight="bold")
        ax.grid(True, ls=":", alpha=0.5)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{mof_name}_energy_vs_time.png", dpi=150)
        plt.close(fig)

    # ── Temperature vs Time ───────────────────────────────────────────────────
    if len(temp_vals) > 0:
        t_ax = np.linspace(0.0, time_ps[-1], len(temp_vals))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(t_ax, temp_vals, color="#9467bd", lw=1.5)
        ax.axhline(y=300.0, color="#d62728", ls=":", alpha=0.8, label="Target 300 K")
        ax.set_xlabel("Time (ps)", fontsize=10)
        ax.set_ylabel("Temperature (K)", fontsize=10)
        ax.set_title(f"{mof_name} — Temperature vs Time", fontsize=11, fontweight="bold")
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{mof_name}_temperature_vs_time.png", dpi=150)
        plt.close(fig)


# =============================================================================
# CORE: ANALYSE ONE MOF
# =============================================================================

def analyse_mof(
    mof_dir: Path,
    isolated_k_energy: float,
    logger: logging.Logger,
) -> Optional[MOFScreeningResult]:
    """
    Compute all 5 screening parameters for a single MOF from raw output files.
    Returns None if the required structural files are missing.
    """
    mof_name = mof_dir.name
    logger.info("-- Analysing: %s", mof_name)

    result = MOFScreeningResult(mof_name=mof_name)

    # ── Load optimised structures (required) ──────────────────────────────────
    relax_summary_path = mof_dir / "relax_summary.json"
    if relax_summary_path.is_file():
        try:
            with open(relax_summary_path, encoding="utf-8") as f:
                relax_summary = json.load(f)
            relax_status = relax_summary.get("status")
            if relax_status != "relaxed":
                logger.warning(
                    "  Skipping %s: bare relaxation status=%s%s",
                    mof_name,
                    relax_status,
                    f" ({relax_summary.get('error')})" if relax_summary.get("error") else "",
                )
                return None
        except (OSError, json.JSONDecodeError) as err:
            logger.warning("  Invalid relaxation summary for %s — skipping: %s", mof_name, err)
            return None

    bare_path = mof_dir / "bare_relaxed.cif"
    if not bare_path.is_file():
        bare_path = mof_dir / "bare_optimized.cif"
    k_path    = mof_dir / "k_on_bare_optimized.cif"
    if not k_path.is_file():
        k_path = mof_dir / "k_inserted_optimized.cif"

    if not bare_path.is_file():
        logger.warning("  bare_optimized.cif not found — skipping %s", mof_name)
        return None
    if not k_path.is_file():
        logger.warning("  K-inserted optimised CIF not found — skipping %s", mof_name)
        return None

    try:
        bare_opt = Structure.from_file(str(bare_path))
        k_opt    = Structure.from_file(str(k_path))
    except Exception as err:
        logger.error("  Failed to load CIF files for %s: %s", mof_name, err)
        return None

    # =========================================================================
    # Volume change (internal metadata for structure validation only)
    # =========================================================================
    try:
        result.volume_bare_A3, result.volume_k_A3, result.volume_change_pct = (
            run_pipeline.compute_volume_change_pct(bare_opt, k_opt)
        )
    except Exception as err:
        logger.error("  Volume change check failed: %s", err)

    # =========================================================================
    # PARAMETER 1: ADSORPTION ENERGY
    # Prefer site-resolved ranking from run_md / compute_adsorption_energy.
    # Fall back to fixed-framework protocol if ranking is absent.
    # =========================================================================
    n_k = (
        run_pipeline.count_element(k_opt, "K")
        - run_pipeline.count_element(bare_opt, "K")
    )
    if n_k <= 0:
        n_k = run_pipeline.count_element(k_opt, "K")

    site_meta = run_pipeline.load_adsorption_site_metadata(mof_name)
    if site_meta.get("best_adsorption_energy_ev") is not None:
        result.e_ads_1k_ev = float(site_meta["best_adsorption_energy_ev"])
        result.adsorption_energy_ev = result.e_ads_1k_ev
        result.adsorption_site_type = str(site_meta.get("best_site_type", ""))
        result.adsorption_reference = str(site_meta.get("reference", run_pipeline.ADSORPTION_REFERENCE))
        result.adsorption_k_o_distance_a = float(site_meta.get("best_k_o_distance_a", float("nan")))
        result.adsorption_label = run_pipeline.classify_adsorption(result.e_ads_1k_ev)
        logger.info(
            "  [1] Adsorption energy : %.4f eV/K  (%s, site=%s, ref=%s, N_K=%d)",
            result.adsorption_energy_ev,
            result.adsorption_label,
            result.adsorption_site_type or "unknown",
            result.adsorption_reference,
            n_k,
        )
    elif n_k > 0:
        try:
            k_indices_ads = run_pipeline.get_inserted_k_indices(bare_opt, k_opt)
            if not k_indices_ads:
                k_indices_ads = run_pipeline.get_atom_indices(k_opt, "K")
            result.e_ads_1k_ev = run_pipeline.compute_adsorption_energy(
                bare_opt,
                k_opt,
                k_indices_ads,
                n_k,
                mof_name,
                reference=run_pipeline.ADSORPTION_REFERENCE,
                search_sites=(n_k == 1),
            )
            site_meta = run_pipeline.load_adsorption_site_metadata(mof_name)
            result.adsorption_site_type = str(site_meta.get("best_site_type", ""))
            result.adsorption_reference = str(
                site_meta.get("reference", run_pipeline.ADSORPTION_REFERENCE)
            )
            result.adsorption_k_o_distance_a = float(
                site_meta.get("best_k_o_distance_a", float("nan"))
            )
            result.adsorption_energy_ev = result.e_ads_1k_ev
            result.adsorption_label = run_pipeline.classify_adsorption(result.e_ads_1k_ev)
            logger.info(
                "  [1] Adsorption energy : %.4f eV/K  (%s, site=%s, ref=%s, N_K=%d)",
                result.adsorption_energy_ev,
                result.adsorption_label,
                result.adsorption_site_type or "provided",
                result.adsorption_reference,
                n_k,
            )
        except Exception as err:
            logger.error("  [1] Adsorption energy failed: %s", err)
    else:
        logger.warning("  [1] No inserted K atoms detected — adsorption energy = NaN")

    # ── Select best trajectory (Stage 2 preferred) ───────────────────────────
    traj_path, log_path, stage_label = select_best_trajectory(mof_dir, logger)
    result.stage_used = stage_label

    if traj_path is None:
        logger.warning("  No MD trajectory found for %s — skipping MD-based parameters", mof_name)
        result.reason = "No MD trajectory file found."
        result.flags.append("no_trajectory")
        return result

    # Compute time axis
    try:
        with Trajectory(str(traj_path), mode="r") as t:
            n_frames = len(t)
    except Exception as err:
        logger.error("  Cannot read trajectory for %s: %s", mof_name, err)
        return result

    try:
        frame0, trajectory_structure = _load_frame0_structure(traj_path)
        n_k = run_pipeline.count_element(trajectory_structure, "K")
        result.n_k = n_k
        k_indices = run_pipeline.get_atom_indices(trajectory_structure, "K")
        framework_idx = run_pipeline.get_framework_indices(trajectory_structure)
        symbols = [site.specie.symbol for site in trajectory_structure]
        if not _structure_matches_atoms(k_opt, frame0):
            logger.warning(
                "Optimized CIF species/atom count does not match trajectory frame 0; "
                "using frame 0 for indexing and its stored energy."
            )
        try:
            frame0_energy = float(frame0.get_potential_energy())
        except Exception:
            frame0_energy = float("nan")
        bare_energy = float(relax_summary.get("energy_final_eV", float("nan")))
        e_k_bulk = run_pipeline.get_bulk_k_energy()
        if np.isfinite(frame0_energy) and np.isfinite(bare_energy) and n_k:
            result.e_ads_avg_loaded_ev_per_k = (
                frame0_energy - bare_energy - n_k * e_k_bulk
            ) / n_k
            result.adsorption_energy_ev = result.e_ads_avg_loaded_ev_per_k
        min_kk = run_pipeline.min_k_k_distance(trajectory_structure, k_indices)
        result.min_k_k_dist_a = min_kk
        if np.isfinite(min_kk) and min_kk < run_pipeline.MIN_KK:
            result.flags.append("k_too_close")
            logger.warning("K–K distance %.3f Å is below %.3f Å", min_kk, run_pipeline.MIN_KK)
        result.volume_k_A3 = float(trajectory_structure.volume)
        result.n_k = n_k
    except Exception as err:
        logger.error("  Failed to establish trajectory frame-0 reference: %s", err)
        return result

    time_axis = get_time_axis_ps(log_path, n_frames)
    result.simulation_time_ps = float(time_axis[-1]) if n_frames > 0 else 0.0

    # ── Prepare structural indices ────────────────────────────────────────────
    # All index-based analysis uses frame 0 of the selected trajectory.

    # =========================================================================
    # PARAMETERS 2–3: MSD + MECHANICAL STABILITY
    # MSD: sequentially unwrapped, framework-relative, production-averaged.
    # =========================================================================
    md_result = None
    msd_fw_curve = np.array([], dtype=float)
    eq_idx = 0
    try:
        pot_vals_eq: np.ndarray | None = None
        if log_path and log_path.is_file():
            _, pot_vals_eq, _ = run_pipeline.parse_md_log(log_path)

        md_result = run_pipeline.analyze_md_trajectory(
            mof=mof_name,
            traj_path=traj_path,
            log_path=log_path,
            k_indices=k_indices,
            framework_idx=framework_idx,
            reference_structure=trajectory_structure,
            symbols=symbols,
            time_ps=result.simulation_time_ps,
            stage_label=stage_label,
            skip_diffusion=True,
        )

        if md_result.success:
            result.mechanical_stability       = md_result.mechanical_stability
            result.mechanical_stability_index = md_result.mechanical_stability_index
            result.framework_rmsd             = md_result.framework_rmsd
            result.maximum_displacement       = md_result.max_displacement
            result.bond_preservation          = md_result.bond_preservation
            result.energy_fluctuation         = md_result.energy_fluctuation
            result.volume_fluctuation_frac    = getattr(
                md_result, "volume_fluctuation_frac", float("nan")
            )
            result.coordination_preservation = getattr(
                md_result, "coordination_preservation", float("nan")
            )
            result.initial_max_force = getattr(md_result, "initial_max_force", float("nan"))
            result.temperature_mean = getattr(md_result, "temperature_mean", float("nan"))
            result.temperature_std = getattr(md_result, "temperature_std", float("nan"))
            result.energy_drift_ev_per_atom_ps = getattr(
                md_result, "energy_drift_ev_per_atom_ps", float("nan")
            )
            result.n_metal_bonds_initial = getattr(md_result, "n_metal_bonds_initial", 0)
            result.n_metal_bonds_broken = getattr(md_result, "n_metal_bonds_broken", 0)
            result.n_metal_bonds_new = getattr(md_result, "n_metal_bonds_new", 0)
            result.coordination_start = getattr(md_result, "coordination_start", {})
            result.coordination_end = getattr(md_result, "coordination_end", {})
            coordination_parts = []
            for index in sorted(
                set(result.coordination_start) | set(result.coordination_end),
                key=lambda value: int(value),
            ):
                atom_index = int(index)
                element = symbols[atom_index] if atom_index < len(symbols) else "M"
                coordination_parts.append(
                    f"{element}{index}:"
                    f"{result.coordination_start.get(index, 0)}->"
                    f"{result.coordination_end.get(index, 0)}"
                )
            result.metal_coordination = ";".join(coordination_parts)
            logger.info(
                "  [3] Mech. stability   : %s  (mech_stabb=%.4f, RMSD=%.3f Å, coord=%.2f)",
                result.mechanical_stability,
                result.mechanical_stability_index,
                result.framework_rmsd,
                result.coordination_preservation
                if np.isfinite(result.coordination_preservation)
                else float("nan"),
            )
        else:
            logger.warning("  [3] MD analysis failed for %s: %s", mof_name, md_result.error)
            result.flags.append("md_analysis_failed")

        with Trajectory(str(traj_path)) as traj:
            msd_fw_curve = run_pipeline.compute_k_msd_unwrapped_trajectory(
                traj, k_indices, framework_idx
            )

        min_len = min(len(time_axis), len(msd_fw_curve))
        time_fit = time_axis[:min_len]
        msd_fw_fit = msd_fw_curve[:min_len]
        eq_series = None
        if pot_vals_eq is not None and len(pot_vals_eq) >= min_len:
            eq_series = pot_vals_eq[:min_len]

        result.msd_a2, eq_idx = compute_production_msd_a2(
            msd_fw_fit, time_fit, eq_series
        )
        result.equilibration_time_ps = (
            float(time_fit[eq_idx]) if eq_idx < len(time_fit) else float("nan")
        )
        logger.info(
            "  [2] MSD (framework-relative, production avg) : %.4f Å²  "
            "(equilibrated from %.1f ps)",
            result.msd_a2,
            result.equilibration_time_ps if np.isfinite(result.equilibration_time_ps) else 0.0,
        )

        md_ok = bool(md_result and md_result.success and md_result.md_converged)
        mof_res = run_pipeline.MOFResult(
            mof=mof_name,
            adsorption_energy_ev=result.adsorption_energy_ev,
            adsorption_binding_class=result.adsorption_label,
            adsorption_site_type=result.adsorption_site_type,
            adsorption_reference=result.adsorption_reference,
            adsorption_k_o_distance_a=result.adsorption_k_o_distance_a,
            mechanical_stability=result.mechanical_stability if (md_result and md_result.success) else "Unknown",
            mechanical_stability_index=result.mechanical_stability_index,
            framework_rmsd=result.framework_rmsd,
            max_displacement=result.maximum_displacement,
            energy_fluctuation=result.energy_fluctuation,
            volume_fluctuation_frac=result.volume_fluctuation_frac,
            msd_a2=result.msd_a2,
            volume_bare_A3=result.volume_bare_A3,
            volume_k_A3=result.volume_k_A3,
            volume_change_pct=result.volume_change_pct,
            md_converged=md_ok,
            simulation_time_ps=result.simulation_time_ps,
            status="Complete",
            stage=2 if stage_label == "stage2" else 1,
        )

        result.flags = list(dict.fromkeys(
            result.flags
            + list(getattr(md_result, "flagged", []))
            + run_pipeline.validate_result(mof_res)
        ))
        mof_res.flags = result.flags
        result.confidence_score = run_pipeline.confidence_score(
            mof_res, opt_ok=True, md_ok=md_ok
        )
        mof_res.confidence_score = result.confidence_score
        result.overall_score = run_pipeline.compute_overall_score(mof_res)
        mof_res.overall_score = result.overall_score
        mof_res.screening_score = result.overall_score
        result.recommendation = "Not scored (scoring pending)"
        result.reason = "Recommendation scoring is pending."

        insertion_summary_path = mof_dir / "insertion" / "insertion_summary.json"
        if insertion_summary_path.is_file():
            try:
                with open(insertion_summary_path, encoding="utf-8") as f:
                    insertion = json.load(f)
                result.insertion_n_max_stable = int(insertion.get("n_max_stable", -1))
                result.insertion_capacity_mAh_g = float(
                    insertion.get("capacity_at_n_max_stable_mAh_g", float("nan"))
                )
                result.insertion_average_voltage_V = float(
                    insertion.get("average_voltage_V", float("nan"))
                )
                result.insertion_max_volume_expansion_percent = float(
                    insertion.get("max_volume_expansion_percent", float("nan"))
                )
            except (OSError, ValueError, json.JSONDecodeError) as err:
                logger.warning("Could not read insertion summary for %s: %s", mof_name, err)

    except Exception as err:
        logger.error("  [2/3] MSD / mechanical stability failed for %s: %s", mof_name, err)

    # ── Diagnostic plots ──────────────────────────────────────────────────────
    try:
        temp_vals = np.array([], dtype=float)
        pot_vals  = np.array([], dtype=float)
        if log_path and log_path.is_file():
            temp_vals, pot_vals, _ = parse_md_log_correct(log_path)

        if len(msd_fw_curve) == 0:
            with Trajectory(str(traj_path)) as traj:
                msd_fw_curve = run_pipeline.compute_k_msd_framework_relative(
                    traj, k_indices, framework_idx
                )

        min_len_plot = min(len(time_axis), len(msd_fw_curve))
        time_axis_plot = time_axis[:min_len_plot]
        msd_for_plot = msd_fw_curve[:min_len_plot]

        generate_plots(
            mof_name=mof_name,
            stage_label=stage_label,
            time_ps=time_axis_plot,
            msd=msd_for_plot,
            temp_vals=temp_vals,
            energy_ev=pot_vals,
            fit_start=eq_idx,
            fit_end=len(msd_for_plot) - 1,
            fit_slope=float("nan"),
            fit_intercept=float("nan"),
            output_dir=OUTPUT_DIR,
        )
    except Exception as plot_err:
        logger.warning("  Plot generation failed for %s: %s", mof_name, plot_err)

    return result


# =============================================================================
# TERMINAL DISPLAY
# =============================================================================

_SEP = "=" * 62

def print_mof_result(result: MOFScreeningResult) -> None:
    """Print a formatted 3-parameter summary for one MOF."""
    ads_str  = (
        f"{result.adsorption_energy_ev:+.4f} eV/K  [{result.adsorption_label}]"
        if np.isfinite(result.adsorption_energy_ev)
        else "NaN"
    )
    msd_str  = (
        f"{result.msd_a2:.4f} Å²"
        if np.isfinite(result.msd_a2)
        else "NaN"
    )
    msi_str  = (
        f"{result.mechanical_stability}  (mech_stabb={result.mechanical_stability_index:.4f})"
        if np.isfinite(result.mechanical_stability_index)
        else "Unknown"
    )

    print(f"\n{_SEP}")
    print(f"  MOF: {result.mof_name}   |   Stage: {result.stage_used}")
    print(_SEP)
    site_str = (
        f"  site={result.adsorption_site_type}, ref={result.adsorption_reference}"
        if result.adsorption_site_type or result.adsorption_reference
        else ""
    )
    print(f"  1. Adsorption Energy     :  {ads_str}{site_str}")
    if np.isfinite(result.e_ads_1k_ev):
        print(f"     |-- E_ads_1K          :  {result.e_ads_1k_ev:+.4f} eV")
    if np.isfinite(result.e_ads_avg_loaded_ev_per_k):
        print(
            f"     `-- E_ads_avg_loaded :  {result.e_ads_avg_loaded_ev_per_k:+.4f} eV/K"
            f"  (N_K={result.n_k})"
        )
    print(f"  2. MSD (K-ion)           :  {msd_str}")
    print(f"  3. Mechanical Stability  :  {msi_str}")
    print(f"     |-- Framework RMSD     :  {result.framework_rmsd:.4f} A"
          if np.isfinite(result.framework_rmsd) else "     |-- Framework RMSD     :  N/A")
    print(f"     |-- Max displacement   :  {result.maximum_displacement:.4f} A"
          if np.isfinite(result.maximum_displacement) else "     |-- Max displacement   :  N/A")
    print(f"     `-- Bond preservation  :  {result.bond_preservation*100:.1f}%"
          if np.isfinite(result.bond_preservation) else "     `-- Bond preservation  :  N/A")
    if np.isfinite(result.initial_max_force):
        print(f"     |-- Initial max force :  {result.initial_max_force:.3f} eV/A")
    if np.isfinite(result.temperature_mean):
        print(
            f"     |-- Temperature       :  {result.temperature_mean:.1f} +/- "
            f"{result.temperature_std:.1f} K"
        )
    print(
        f"     |-- Metal bonds       :  {result.n_metal_bonds_initial} initial, "
        f"{result.n_metal_bonds_broken} broken, {result.n_metal_bonds_new} new"
    )
    if result.insertion_n_max_stable >= 0:
        print(
            f"  Insertion (uncalibrated): n_max={result.insertion_n_max_stable}, "
            f"capacity={result.insertion_capacity_mAh_g:.2f} mAh/g, "
            f"avg V={result.insertion_average_voltage_V:.3f} V"
        )
    if np.isfinite(result.overall_score):
        print(f"  Overall Score            :  {result.overall_score:.1f}/100")
    print(f"  Recommendation           :  {result.recommendation}")
    print(f"  Reason                   :  {result.reason}")
    if result.flags:
        print(f"  Flags                    :  {', '.join(result.flags)}")
    print(_SEP)


def print_final_summary(results: List[MOFScreeningResult]) -> None:
    """Print an aggregate summary table after all MOFs are processed."""
    if not results:
        print("\nNo results to summarise.")
        return

    ads_vals  = [r.adsorption_energy_ev           for r in results if np.isfinite(r.adsorption_energy_ev)]
    msd_vals  = [r.msd_a2                          for r in results if np.isfinite(r.msd_a2)]
    msi_vals  = [r.mechanical_stability_index      for r in results if np.isfinite(r.mechanical_stability_index)]

    _good_labels = {
        run_pipeline.RECOMMENDATION_EXCELLENT,
        run_pipeline.RECOMMENDATION_VERY_GOOD,
        run_pipeline.RECOMMENDATION_GOOD,
    }

    good_candidates = [r for r in results if r.recommendation in _good_labels]
    n_good = len(good_candidates)
    n_val = sum(1 for r in results if r.recommendation == run_pipeline.RECOMMENDATION_VALIDATE)
    n_rej = sum(
        1
        for r in results
        if r.recommendation in {
            run_pipeline.RECOMMENDATION_NOT,
            run_pipeline.RECOMMENDATION_POOR,
        }
    )

    print(f"\n\n{'=' * 62}")
    print("  FINAL SCREENING SUMMARY".center(62))
    print(f"{'=' * 62}")
    print(f"  Total MOFs analysed      :  {len(results)}")
    print(f"  Good candidates          :  {n_good}")
    print(f"  Needs validation         :  {n_val}")
    print(f"  Not recommended          :  {n_rej}")

    # ── List every good candidate by name ──────────────────────────
    if good_candidates:
        # Sort best-first by confidence score (descending)
        good_sorted = sorted(
            good_candidates,
            key=lambda r: r.confidence_score if np.isfinite(r.confidence_score) else -1,
            reverse=True,
        )
        print(f"{'─' * 62}")
        print("  GOOD CANDIDATE CIF NAMES")
        print(f"{'─' * 62}")
        for rank, r in enumerate(good_sorted, start=1):
            score_str = (
                f"  (score={r.confidence_score:.1f})"
                if np.isfinite(r.confidence_score)
                else ""
            )
            rec_tag = r.recommendation if r.recommendation else ""
            print(f"  {rank:>2}. {r.mof_name}.cif"
                  f"  [{rec_tag}]{score_str}")
    print(f"{'─' * 62}")
    print("  PARAMETER AVERAGES")
    print(f"{'─' * 62}")
    print(f"  1. Adsorption Energy     :  "
          f"{np.mean(ads_vals):+.4f} eV/K" if ads_vals else "  1. Adsorption Energy     :  N/A")
    print(f"  2. MSD (K-ion)           :  "
          f"{np.mean(msd_vals):.4f} Å²" if msd_vals else "  2. MSD (K-ion)           :  N/A")
    print(f"  3. Mechanical Stability  :  "
          f"mech_stabb avg = {np.mean(msi_vals):.4f}" if msi_vals else "  3. Mechanical Stability  :  N/A")
    print(f"{'─' * 62}")

    # Best and worst by adsorption (closest to –1.5 eV/K)
    if results:
        by_conf = sorted(results, key=lambda r: r.confidence_score
                         if np.isfinite(r.confidence_score) else -1, reverse=True)
        print(f"  Highest confidence       :  {by_conf[0].mof_name}"
              f"  (score={by_conf[0].confidence_score:.1f})"
              if np.isfinite(by_conf[0].confidence_score) else "")
        print(f"  Lowest confidence        :  {by_conf[-1].mof_name}"
              if len(by_conf) > 1 else "")

    print(f"{'=' * 62}")
    print(f"  Results CSV  ->  {RESULTS_CSV}")
    print(f"  Plots        ->  {OUTPUT_DIR / 'plots'}")
    print(f"{'=' * 62}\n")


# =============================================================================
# CSV OUTPUT
# =============================================================================

CSV_COLUMNS = [
    "MOF",
    "adsorption_energy",
    "E_ads_1K_eV",
    "E_ads_avg_loaded_eV_per_K",
    "N_K",
    "initial_max_force_eV_A",
    "temperature_mean_K",
    "temperature_std_K",
    "energy_drift_eV_atom_ps",
    "n_metal_bonds_initial",
    "n_metal_bonds_broken",
    "n_metal_bonds_new",
    "flags",
    "framework_rmsd_A",
    "max_displacement_A",
    "bond_preservation",
    "min_K_K_dist_A",
    "metal_coordination_start_end",
    "recommendation",
    "insertion_n_max_stable",
    "insertion_capacity_mAh_g",
    "insertion_average_voltage_V_uncalibrated",
    "insertion_max_volume_expansion_percent",
    "msd",
    "mech_stabb",
]


def result_to_row(r: MOFScreeningResult) -> dict:
    def fmt(val):
        return "" if (isinstance(val, float) and not np.isfinite(val)) else val

    return {
        "MOF":               r.mof_name,
        "adsorption_energy": fmt(r.adsorption_energy_ev),
        "E_ads_1K_eV": fmt(r.e_ads_1k_ev),
        "E_ads_avg_loaded_eV_per_K": fmt(r.e_ads_avg_loaded_ev_per_k),
        "N_K": r.n_k,
        "initial_max_force_eV_A": fmt(r.initial_max_force),
        "temperature_mean_K": fmt(r.temperature_mean),
        "temperature_std_K": fmt(r.temperature_std),
        "energy_drift_eV_atom_ps": fmt(r.energy_drift_ev_per_atom_ps),
        "n_metal_bonds_initial": r.n_metal_bonds_initial,
        "n_metal_bonds_broken": r.n_metal_bonds_broken,
        "n_metal_bonds_new": r.n_metal_bonds_new,
        "flags": ";".join(r.flags),
        "framework_rmsd_A": fmt(r.framework_rmsd),
        "max_displacement_A": fmt(r.maximum_displacement),
        "bond_preservation": fmt(r.bond_preservation),
        "min_K_K_dist_A": fmt(r.min_k_k_dist_a),
        "metal_coordination_start_end": r.metal_coordination,
        "recommendation": r.recommendation,
        "insertion_n_max_stable": r.insertion_n_max_stable,
        "insertion_capacity_mAh_g": fmt(r.insertion_capacity_mAh_g),
        "insertion_average_voltage_V_uncalibrated": fmt(r.insertion_average_voltage_V),
        "insertion_max_volume_expansion_percent": fmt(r.insertion_max_volume_expansion_percent),
        "msd":               fmt(r.msd_a2),
        "mech_stabb":        fmt(r.mechanical_stability_index),
    }


def write_results_csv(results: List[MOFScreeningResult]) -> None:
    rows = [result_to_row(r) for r in results]
    df   = pd.DataFrame(rows, columns=CSV_COLUMNS)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_CSV, index=False)


def is_inbetween_thresholds(r: MOFScreeningResult) -> bool:
    if not (np.isfinite(r.adsorption_energy_ev) and run_pipeline.CANDIDATE_ADS_MIN_EV <= r.adsorption_energy_ev <= run_pipeline.CANDIDATE_ADS_MAX_EV):
        return False
    if not (np.isfinite(r.mechanical_stability_index) and r.mechanical_stability_index >= run_pipeline.CANDIDATE_MSI_MIN and r.mechanical_stability != "Collapsed"):
        return False
    if not (np.isfinite(r.msd_a2) and r.msd_a2 >= run_pipeline.CANDIDATE_MSD_MIN_A2):
        return False
    return True


def write_best_cif_csv(results: List[MOFScreeningResult], top_n: int | None = None) -> int:
    """
    Write best_cif.csv containing only results inbetween the thresholds, ranked by Overall Score.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BEST_CIFS_DIR.mkdir(parents=True, exist_ok=True)

    if not results:
        pd.DataFrame(columns=run_pipeline.BEST_CIF_COLUMNS).to_csv(BEST_CIF_CSV, index=False)
        return 0

    # Filter results that are inbetween the thresholds
    selected = [r for r in results if is_inbetween_thresholds(r)]

    selected = sorted(
        selected,
        key=lambda r: (
            r.overall_score if np.isfinite(r.overall_score) else float("-inf"),
            r.confidence_score if np.isfinite(r.confidence_score) else float("-inf"),
        ),
        reverse=True,
    )

    rows = []
    for r in selected:
        cif_name = f"{r.mof_name}.cif"
        src = STRUCTURES_DIR / r.mof_name / "k_on_bare_optimized.cif"
        if not src.is_file():
            src = STRUCTURES_DIR / r.mof_name / "k_inserted_optimized.cif"
        if src.is_file():
            shutil.copy2(src, BEST_CIFS_DIR / cif_name)
        rows.append({
            "best_cif": cif_name,
            "adsorption_energy": r.adsorption_energy_ev,
            "msd": r.msd_a2,
            "mech_stabb": r.mechanical_stability_index,
        })

    pd.DataFrame(rows, columns=run_pipeline.BEST_CIF_COLUMNS).to_csv(BEST_CIF_CSV, index=False)
    return len(rows)


# =============================================================================
# ENTRY POINT
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute 3 MOF screening parameters from MD trajectories."
    )
    parser.add_argument(
        "--device", type=str, default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device for CHGNet single-point calculations (default: auto-detect)"
    )
    parser.add_argument(
        "--structures-dir", type=Path, default=STRUCTURES_DIR,
        help=f"Directory containing per-MOF output folders (default: {STRUCTURES_DIR})"
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        choices=["isolated", "bulk"],
        help="K adsorption reference: isolated (default) or bulk BCC K",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help=f"Number of top MOFs to export to best_cif.csv (default: {run_pipeline.BEST_CIF_TOP_N})",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=None,
        help="Optional literature CSV for validation report (MAE/RMSE/R²)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Apply device before loading CHGNet
    run_pipeline.USE_DEVICE = args.device
    if args.reference:
        run_pipeline.ADSORPTION_REFERENCE = args.reference
    if args.top_n is not None:
        run_pipeline.BEST_CIF_TOP_N = max(1, int(args.top_n))

    logger = setup_logging(OUTPUT_DIR)
    run_pipeline._logger = logger  # Route run_pipeline utilities to analysis.log

    from scientific.manifest import write_run_manifest

    manifest_path = write_run_manifest(
        OUTPUT_DIR,
        run_pipeline,
        cli_args={
            "device": args.device,
            "reference": args.reference or run_pipeline.ADSORPTION_REFERENCE,
            "structures_dir": str(args.structures_dir),
            "benchmark": str(args.benchmark) if args.benchmark else None,
        },
    )
    logger.info("Run manifest written: %s", manifest_path)

    logger.info("=" * 62)
    logger.info("CHGNet MOF Screening — Independent Parameter Analysis")
    logger.info("=" * 62)
    logger.info("Structures dir : %s", args.structures_dir.resolve())
    logger.info("Device         : %s", args.device)
    logger.info("K reference    : %s", run_pipeline.ADSORPTION_REFERENCE)
    logger.info("Best-CIF top-N : %d", run_pipeline.BEST_CIF_TOP_N)
    logger.info("Parameters     : Adsorption Energy | MSD | Mechanical Stability (mech_stabb)")

    # ── 1. Discover MOF directories ───────────────────────────────────────────
    structures_dir = args.structures_dir
    if not structures_dir.is_dir():
        logger.error("Structures directory not found: %s", structures_dir.resolve())
        logger.error("Run  python run_md.py  first to generate trajectory files.")
        sys.exit(1)

    mof_dirs = sorted(
        d for d in structures_dir.iterdir()
        if d.is_dir()
        and d.name != "insertion"
        and not d.name.endswith("_backup")
        and not d.name.startswith("md_")
    )
    if not mof_dirs:
        logger.error("No MOF subdirectories found in %s", structures_dir.resolve())
        sys.exit(1)

    logger.info("Found %d MOF directories to process", len(mof_dirs))

    # ── 2. Load isolated K energy ─────────────────────────────────────────────
    isolated_k_energy = load_isolated_k_energy(mof_dirs[0], OUTPUT_DIR, logger)

    # ── 3. Initialise CSV with header ─────────────────────────────────────────
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV, "w", encoding="utf-8") as f:
        f.write(",".join(CSV_COLUMNS) + "\n")

    # ── 4. Process every MOF ──────────────────────────────────────────────────
    results: List[MOFScreeningResult] = []

    for mof_dir in tqdm(mof_dirs, desc="Analysing MOFs", unit="MOF"):
        try:
            res = analyse_mof(mof_dir, isolated_k_energy, logger)
        except Exception as exc:
            logger.error("Fatal error for %s: %s", mof_dir.name, exc)
            res = None

        if res is not None:
            results.append(res)
            print_mof_result(res)

            # Append row to CSV immediately (incremental — safe if interrupted)
            row = result_to_row(res)
            with open(RESULTS_CSV, "a", encoding="utf-8") as f:
                vals = [str(row.get(col, "")) for col in CSV_COLUMNS]
                f.write(",".join(vals) + "\n")

    # Ensure a clean final CSV snapshot (header + all rows), even if
    # incremental appends were interrupted or manually edited.
    write_results_csv(results)

    # ── 5. Final summary ──────────────────────────────────────────────────────
    print_final_summary(results)
    n_best = write_best_cif_csv(results, top_n=args.top_n)
    logger.info("Best-CIF CSV written to %s (%d entries)", BEST_CIF_CSV, n_best)

    if args.benchmark is not None:
        try:
            from validate_benchmark import run_validation

            logger.info("Running literature validation against %s", args.benchmark)
            run_validation(args.benchmark, RESULTS_CSV, OUTPUT_DIR, logger=logger)
        except Exception as exc:
            logger.error("Benchmark validation failed: %s", exc)

    logger.info("Analysis complete. Results written to %s", RESULTS_CSV)


if __name__ == "__main__":
    main()
