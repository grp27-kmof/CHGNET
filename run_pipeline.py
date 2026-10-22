#!/usr/bin/env python3
"""
CHGNet-Based Automated MOF Screening Pipeline for Potassium-Ion Battery Materials.

All simulations use the CHGNet universal machine-learning interatomic potential.
CHGNet provides fast approximate energies, forces, and MD trajectories relative to
DFT; absolute adsorption energies and diffusion coefficients should be validated
against higher-level methods for publication-quality predictions.

Scientific design notes (high-throughput screening methodology)
--------------------------------------------------------------
* Adsorption: automatic multi-site search (μ3-O, carboxylate O, bridge O,
  open-metal-adjacent, pore minima); K-only relaxation on fixed framework;
  globally lowest E_ads selected. Reference: isolated K (default) or bulk BCC K.
* Optimization: fmax = 0.01 eV/Å with retry / unconverged detection.
* MD: Stage-1 10 ps Berendsen equilibration; Stage-2 50 ps Nose-Hoover production.
* Diffusion: Einstein MSD with statistical linear-regime detection; NEB+TST uses
  the lowest reliable barrier (not median). BVSE Arrhenius estimate is used as
  a last-resort fallback when MD and NEB both fail.
* Ranking: continuous Overall Score (0–100) with soft preference windows;
  best_cif.csv always exports top-N by score.
"""

from __future__ import annotations

import json
import hashlib
import logging
import shutil
import sys
import time
import warnings
import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from ase import Atoms
from ase.constraints import FixAtoms
from ase.data import atomic_numbers, covalent_radii
from ase.geometry import find_mic
from ase.io.trajectory import Trajectory
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.stats import linregress
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from chgnet.model.dynamics import CHGNetCalculator, MolecularDynamics, StructOptimizer
from chgnet.model.model import CHGNet

from scientific.adsorption_sites import (
    AdsorptionCandidate,
    analyze_relaxed_site_geometry,
    discover_adsorption_candidates,
    min_framework_distance,
    k_coordination_number_at_frac,
    score_candidate_pre_relaxation,
)
from scientific.adsorption_plots import plot_adsorption_analysis
from scientific.adsorption_workflow import run_adsorption_site_search, save_adsorption_results
from scientific.msd_analysis import (
    compute_msd_unwrapped,
    detect_equilibration_index,
    einstein_fit_with_uncertainty,
    unwrap_positions_mic,
)

class CudaOOMError(Exception):
    """Raised when a CUDA Out of Memory error occurs and device fallback is disabled."""
    pass
from scientific.diffusion_analysis import (
    MIN_ANOMALOUS_EXPONENT,
    MIN_FICKIAN_FIT_R2,
    anomalous_diffusion_exponent,
    msd_is_monotonic,
    select_longest_diffusive_window,
)
from scientific.msi_analysis import coordination_preservation_fraction

warnings.filterwarnings("ignore", module="pymatgen")
warnings.filterwarnings("ignore", module="ase")

# =============================================================================
# CONFIGURATION — edit simulation parameters here only
# =============================================================================

TEMPERATURE_K = 300
TIME_STEP_FS = 1
# Stage 1: equilibration. 10 ps is a common MLIP/MOF NVT equilibration length
# that allows thermostat settling before production sampling (cf. typical
# AIMD/MLIP MOF protocols of 5–20 ps equilibration).
FAST_MD_TIME_PS = 10
# Stage 2: production. 50 ps improves Einstein-regime sampling for dilute K
# diffusion relative to 20 ps while remaining tractable for high-throughput
# CHGNet screening (literature MOF MLIP MD often uses 50–100+ ps production).
FINAL_MD_TIME_PS = 50
TOP_PERCENT = 20
# Geometry optimization force tolerance.
# 0.01 eV/Å is a standard DFT ionic-relaxation target (VASP/QE tutorials);
# CHGNet can reach this for most MOFs and yields more transferable adsorption
# energies than the previous 0.05 eV/Å screening default.
FORCE_TOL = 0.01
FORCE_TOL_LOOSE = 0.05  # fallback if tight convergence is not reached
MAX_OPT_STEPS = 3000
MAX_OPT_RETRIES = 2
N_PRESCREEN = 10
N_FULL = 3
ADSORPTION_FULL_FMAX = 0.05
ADSORPTION_FULL_MAX_STEPS = 1000
INSERTION_N_MAX = 12
INSERTION_N_PRESCREEN = 8
INSERTION_N_FULL = 2
INSERTION_MIN_KK = 3.8
INSERTION_V_THRESHOLD = 1.0
INSERTION_FMAX = 0.05
INSERTION_MAX_STEPS = 1500
BARE_RELAX_FMAX = 0.01
BARE_RELAX_MAX_STEPS = 2000
BARE_RELAX_CELL = True
BARE_RELAX_ACCEPTANCE_FMAX = 0.05
BARE_RELAX_MIN_DISTANCE = 0.8
ADSORBATE = "K"
RANDOM_SEED = 42

# Adsorption energy reference:
#   "isolated" — single K in a large vacuum box (default; matches gas-phase
#                adsorption conventions used in many MOF DFT studies)
#   "bulk"     — BCC metallic K (formation-energy style reference; useful when
#                comparing intercalation/insertion energetics to electrode metals)
ADSORPTION_REFERENCE = "bulk"  # "isolated" | "bulk"
BULK_K_BCC_A_A = 5.225  # experimental BCC K lattice constant (Å)

# When True, screening outputs and scoring use only adsorption, MSD, and mech_stabb.
THREE_PARAMETER_SCREENING = True

WEIGHTS = {
    # Multi-objective overall score for the three reported screening parameters.
    "adsorption": 0.40,
    "stability": 0.35,
    "msd": 0.20,
    "confidence": 0.05,
    # Legacy keys retained at zero weight for backward-compatible scoring calls.
    "diffusion": 0.0,
    "volume": 0.0,
}

# Paths
BARE_CIFS_DIR = Path("inputs/bare_cifs")
CIFS_DIR = Path("inputs/cifs")
OUTPUT_DIR = Path("outputs")
BEST_CIF_TOP_N = 10  # always export top-N by overall score (not hard-threshold only)

# Additional simulation settings (paths excluded from "simulation" scope)
RELAX_CELL = False
USE_DEVICE: str | None = "cuda"  # "cpu", "cuda", "mps", or None for auto
ISOLATED_K_BOX_SIZE_A = 20.0
ADSORPTION_CHECKPOINT_KEY = "adsorption_energy_ev_per_k_v4"
MIN_DIFFUSION_REPORT_R2 = 0.70
# Tier-1 Einstein acceptance: R² ≥ 0.90 is statistically meaningful for
# high-throughput screening. The previous 0.98 gate rejected many physically
# valid dilute-ion trajectories with natural MSD noise.
MD_DIFFUSION_ACCEPT_R2 = 0.90
MD_DIFFUSION_MIN_TIME_PS = 10.0
MIN_DIFFUSION_FIT_POINTS = 50
MD_DIFFUSION_MIN_FIT_FRAMES = MIN_DIFFUSION_FIT_POINTS
# MD stability gates — reject only genuinely unstable runs.
# Energy drift 0.05 eV/ps (~3 eV over 50 ps for ~100-atom cells) is a soft
# MLIP NVT tolerance; Nose-Hoover can show mild total-energy fluctuations
# without structural collapse. Force and temperature gates similarly avoid
# false rejection of thermally fluctuating but intact frameworks.
MAX_MD_ENERGY_DRIFT_EV_PS = 0.05
MAX_MD_FORCE_EV_A = 5.0
MAX_MD_TEMPERATURE_STD_K = 100.0
MD_THERMOSTAT_EQUIL = "Berendsen"          # Stage 1 equilibration
MD_THERMOSTAT_PRODUCTION = "Nose-Hoover"   # Stage 2 production (canonical kinetics)
# Legacy alias (equilibration default)
MD_THERMOSTAT = MD_THERMOSTAT_EQUIL
MD_LOGINTERVAL = 1

# Adsorption scoring: favor moderately negative binding (eV per K atom)
ADSORPTION_TARGET_EV_PER_K = -1.5
ADSORPTION_SIGMA_EV = 0.75
ADSORPTION_TARGET_MIN_EV_PER_K = -2.5
ADSORPTION_TARGET_MAX_EV_PER_K = -0.5
ADSORPTION_VERY_WEAK_EV_PER_K = -0.2
ADSORPTION_VERY_STRONG_EV_PER_K = -3.0

# Automatic adsorption-site search (publication-quality multi-start global optimization)
# Chemistry-based distances are computed inside scientific/adsorption_sites.py from covalent radii.
ADSORPTION_MAX_RAW_CANDIDATES = 800
ADSORPTION_MAX_UNIQUE_CANDIDATES = 400
ADSORPTION_PARALLEL_WORKERS = 1  # increase on CPU; keep 1 on GPU
# Legacy aliases (backward compatibility for external scripts)
ADSORPTION_K_O_DISTANCE_A = 2.70
ADSORPTION_METAL_O_CUTOFF_A = 2.60
ADSORPTION_MIN_K_FW_DIST_A = 1.80
ADSORPTION_PORE_GRID_SPACING_A = 1.0
ADSORPTION_MAX_CANDIDATES = ADSORPTION_MAX_UNIQUE_CANDIDATES

# Mechanical stability thresholds (Å)
STABILITY_RMSD_STABLE_A = 0.5
STABILITY_RMSD_MARGINAL_A = 1.5
STABILITY_MAX_DISP_STABLE_A = 1.0
STABILITY_MAX_DISP_MARGINAL_A = 3.0
BOND_PRESERVATION_STABLE = 0.90
BOND_PRESERVATION_MARGINAL = 0.70
BOND_TOLERANCE_FACTOR = 1.30

# MSD / diffusion analysis
MD_EQUILIBRATION_FRACTION = 0.20
MIN_DIFFUSION_FIT_R2 = 0.85
DIFFUSION_DIMENSIONS = 3
MIN_LINEAR_WINDOW_FRACTION = 0.20
LINEAR_WINDOW_STRIDE = 10
DIFFUSION_SUFFICIENT_SIM_TIME_PS = 20.0
DIFFUSION_STAGE1_MIN_TIME_PS = 10.0
DIFFUSION_MSD_D_CONSISTENCY_FACTOR = 5.0
DIFFUSION_FALLBACK_START_FRACTION = 0.40  # fit last 60% when no ideal linear region
# Statistical linear-regime detection: require slope significance (t-test)
# rather than a fixed ultra-high R² alone.
DIFFUSION_MIN_SLOPE_TSTAT = 2.0

# Soft preference windows for multi-objective scoring (not hard reject gates).
# Hard all-or-nothing thresholds are replaced by continuous scores so a MOF
# that barely misses one metric is not discarded if others are strong.
CANDIDATE_ADS_MIN_EV = -2.5
CANDIDATE_ADS_MAX_EV = -0.5
CANDIDATE_MSI_MIN = 0.5
CANDIDATE_DIFFUSION_MIN_CM2_S = 1e-12
CANDIDATE_DIFFUSION_MAX_CM2_S = 1e-8
CANDIDATE_MSD_MIN_A2 = 5.0
# Volume change ΔV/V (%) between bare and K-inserted optimized cells
CANDIDATE_VOLUME_CHANGE_MAX_PCT = 5.0
VOLUME_CHANGE_WARN_PCT = 2.0
# Soft volume-fluctuation scale for MSI (relative std of cell volume during MD).
# MOF frameworks with |σ_V / ⟨V⟩| ≲ 1–2% are typically intact under NVT.
MSI_MAX_VOLUME_FLUCT_FRAC = 0.02

# Absolute physical bounds for reporting any tier (cm²/s).
DIFFUSION_D_ABS_MIN_CM2_S = 1e-300
DIFFUSION_D_ABS_MAX_CM2_S = 1.0

# MSI (Mechanical Stability Index) configuration
# Weights emphasize bond preservation and RMSD because framework integrity
# in MOF MD literature is primarily judged by bond retention and atomic
# displacement from the initial topology (cf. MOF flexibility / collapse
# analyses). Energy and volume fluctuations are secondary NVT diagnostics.
# No single community-standard MSI weight set exists for CHGNet MOF screening;
# these weights are therefore justified by physical priority, not arbitrary
# equal weighting.
MSI_WEIGHTS = {
    "rmsd": 0.25,
    "bond": 0.25,
    "displacement": 0.15,
    "coordination": 0.15,  # K–O contact preservation (linker dissociation proxy)
    "energy": 0.10,
    "volume": 0.10,
}
MSI_STABLE_THRESHOLD = 0.70
MSI_MARGINAL_THRESHOLD = 0.45
MSI_MAX_ENERGY_FLUCT_EV_PER_ATOM = 0.05  # σ_E per atom (eV/atom)

# K insertion sanity checks (dilute screening / validation studies)
MIN_K_K_SPACING_A = 3.50
CANDIDATE_MAX_N_K = 4
MIN_KK = 3.8

# Confidence score: normalized components in [0,1], then weighted to 0–100.
# Each weight reflects how strongly that evidence supports trusting the result.
CONFIDENCE_OPT_WEIGHT = 0.15
CONFIDENCE_MD_WEIGHT = 0.20
CONFIDENCE_MSI_WEIGHT = 0.20
CONFIDENCE_DIFFUSION_WEIGHT = 0.20
CONFIDENCE_ADSORPTION_WEIGHT = 0.15
CONFIDENCE_TRAJECTORY_WEIGHT = 0.10

# Overall recommendation score bands (0–100 continuous ranking)
OVERALL_SCORE_EXCELLENT = 85.0
OVERALL_SCORE_VERY_GOOD = 70.0
OVERALL_SCORE_GOOD = 55.0
OVERALL_SCORE_VALIDATE = 40.0
OVERALL_SCORE_POOR = 25.0

# Covalent radii sum multiplier for initial bond graph (Å)
BOND_CUTOFF_SCALE = 1.20

# =============================================================================
# END CONFIGURATION
# =============================================================================

MOF_PARAM_COLUMNS = [
    "MOF",
    "adsorption_energy",
    "msd",
    "mech_stabb",
]

# Legacy alias used by resume logic
RESULTS_COLUMNS = MOF_PARAM_COLUMNS

BEST_CIF_COLUMNS = [
    "best_cif",
    "adsorption_energy",
    "msd",
    "mech_stabb",
]

RECOMMENDATION_EXCELLENT = "Excellent"
RECOMMENDATION_VERY_GOOD = "Very Good"
RECOMMENDATION_GOOD = "Good"
RECOMMENDATION_VALIDATE = "Needs Validation"
RECOMMENDATION_POOR = "Poor"
RECOMMENDATION_NOT = "Not Recommended"
# Backward-compatible aliases used by older callers
RECOMMENDATION_GOOD_CANDIDATE = RECOMMENDATION_GOOD
RECOMMENDATION_NEEDS_VALIDATION = RECOMMENDATION_VALIDATE

STABILITY_SCORES = {"Stable": 1.0, "Marginal": 0.5, "Collapsed": 0.0}

A2_PER_PS_TO_CM2_PER_S = 1e-4  # 1 Å²/ps = 10⁻⁴ cm²/s


def covalent_radius(symbol: str) -> float:
    """Covalent radius in Å from ASE data tables."""
    z = atomic_numbers.get(symbol, 0)
    if z <= 0:
        return 1.0
    return float(covalent_radii[z])


@dataclass
class OptimizationResult:
    structure: Structure | None = None
    energy: float = float("nan")
    n_steps: int = 0
    max_force: float = float("nan")
    success: bool = False
    converged: bool = False
    fmax_target: float = float("nan")
    error: str = ""
    max_force_K_only: float = float("nan")
    max_force_all: float = float("nan")


@dataclass
class BareRelaxationResult:
    structure: Structure | None = None
    status: str = "not_relaxed"
    final_fmax: float = float("nan")
    n_steps: int = 0
    used_fallback: bool = False
    hit_max_steps: bool = False
    min_dist_initial: float = float("nan")
    energy_initial: float = float("nan")
    energy_final: float = float("nan")
    energy_per_atom_final: float = float("nan")
    volume_initial: float = float("nan")
    volume_final: float = float("nan")
    volume_change_percent: float = float("nan")
    lattice_initial: dict[str, float] = field(default_factory=dict)
    lattice_final: dict[str, float] = field(default_factory=dict)
    k_cif_cell_mismatch: bool = False
    wall_time_s: float = 0.0
    error: str = ""

    @property
    def success(self) -> bool:
        return self.status == "relaxed"

    def to_summary(self, mof_name: str) -> dict[str, Any]:
        return {
            "mof_name": mof_name,
            "status": self.status,
            "final_fmax_eV_per_A": self.final_fmax,
            "n_steps": self.n_steps,
            "used_fallback": self.used_fallback,
            "hit_max_steps": self.hit_max_steps,
            "min_dist_initial_A": self.min_dist_initial,
            "energy_initial_eV": self.energy_initial,
            "energy_final_eV": self.energy_final,
            "energy_per_atom_final_eV": self.energy_per_atom_final,
            "volume_initial_A3": self.volume_initial,
            "volume_final_A3": self.volume_final,
            "volume_change_percent": self.volume_change_percent,
            "lattice_initial": self.lattice_initial,
            "lattice_final": self.lattice_final,
            "k_cif_cell_mismatch": self.k_cif_cell_mismatch,
            "wall_time_s": self.wall_time_s,
            "fmax_target_eV_per_A": BARE_RELAX_FMAX,
            "max_steps": BARE_RELAX_MAX_STEPS,
            "relax_cell": BARE_RELAX_CELL,
        }


@dataclass
class AdsorptionSiteResult:
    """One automatically discovered and scored K adsorption site."""
    site_type: str = "unknown"
    rank: int = -1
    cluster_id: int = -1
    is_global_minimum: bool = False
    candidate_id: int = -1
    adsorption_energy_ev: float = float("nan")
    k_frac_coords: list[float] = field(default_factory=list)
    k_o_distance_a: float = float("nan")
    k_metal_distance_a: float = float("nan")
    oxygen_index: int = -1
    coordination_number: int = -1
    nearest_atoms: list[dict[str, Any]] = field(default_factory=list)
    local_environment_descriptor: str = ""
    optimization_converged: bool = False
    max_force_ev_a: float = float("nan")
    confidence_score: float = float("nan")
    pre_score: float = float("nan")
    structure: Structure | None = None
    success: bool = False
    notes: str = ""


@dataclass
class MDAnalysisResult:
    mechanical_stability: str = "Collapsed"
    mechanical_stability_index: float = float("nan")
    framework_rmsd: float = float("nan")
    max_displacement: float = float("nan")
    energy_fluctuation: float = float("nan")
    volume_fluctuation_frac: float = float("nan")
    msd_a2: float = float("nan")
    diffusion_cm2_s: float = float("nan")
    diffusion_slope_a2_ps: float = float("nan")
    diffusion_fit_r2: float = float("nan")
    diffusion_fit_start: int = -1
    diffusion_fit_end: int = -1
    diffusion_fit_num_frames: int = 0
    diffusion_stderr_cm2_s: float = float("nan")
    diffusion_ci_lo_cm2_s: float = float("nan")
    diffusion_ci_hi_cm2_s: float = float("nan")
    slope_stderr_a2_ps: float = float("nan")
    slope_pvalue: float = float("nan")
    equilibration_frame: int = -1
    equilibration_time_ps: float = float("nan")
    coordination_preservation: float = float("nan")
    diffusion_reliability: str = "Very Low"
    diffusion_warning: str = ""
    bond_preservation: float = float("nan")
    temperature_std: float = float("nan")
    temperature_mean: float = float("nan")
    energy_drift_ev_ps: float = float("nan")
    energy_drift_ev_per_atom_ps: float = float("nan")
    initial_max_force: float = float("nan")
    max_force_md: float = float("nan")
    n_metal_bonds_initial: int = 0
    n_metal_bonds_broken: int = 0
    n_metal_bonds_new: int = 0
    coordination_start: dict[str, int] = field(default_factory=dict)
    coordination_end: dict[str, int] = field(default_factory=dict)
    metal_carbon_contact: bool = False
    md_converged: bool = False
    simulation_time_ps: float = float("nan")
    flagged: list[str] = field(default_factory=list)
    success: bool = False
    error: str = ""


import json

def load_literature_mech_stab() -> dict:
    path = Path("INPUTS/literature_mech_stab.json")
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

LIT_MECH_STAB = load_literature_mech_stab()

@dataclass
class MOFResult:
    mof: str
    adsorption_energy_ev: float = float("nan")
    adsorption_binding_class: str = ""
    adsorption_site_type: str = ""
    adsorption_reference: str = ADSORPTION_REFERENCE
    adsorption_k_o_distance_a: float = float("nan")
    adsorption_site_rank: int = -1
    mechanical_stability: str = ""
    mechanical_stability_index: float = float("nan")
    framework_rmsd: float = float("nan")
    max_displacement: float = float("nan")
    bond_preservation: float = float("nan")
    energy_fluctuation: float = float("nan")
    volume_fluctuation_frac: float = float("nan")
    msd_a2: float = float("nan")
    diffusion_cm2_s: float = float("nan")
    diffusion_fit_r2: float = float("nan")
    diffusion_slope_a2_ps: float = float("nan")
    diffusion_fit_start: int = -1
    diffusion_fit_end: int = -1
    diffusion_fit_num_frames: int = 0
    diffusion_stderr_cm2_s: float = float("nan")
    diffusion_ci_lo_cm2_s: float = float("nan")
    diffusion_ci_hi_cm2_s: float = float("nan")
    equilibration_time_ps: float = float("nan")
    coordination_preservation: float = float("nan")
    diffusion_simulation_duration_ps: float = float("nan")
    diffusion_reliability: str = "Very Low"
    diffusion_warning: str = ""
    migration_barrier_ev: float = float("nan")
    volume_bare_A3: float = float("nan")
    volume_k_A3: float = float("nan")
    volume_change_pct: float = float("nan")  # 100 * (V_k - V_bare) / V_bare
    optimization_energy: float = float("nan")
    optimization_converged: bool = False
    confidence_score: float = float("nan")
    overall_score: float = float("nan")
    uncertainty_flag: str = ""
    md_converged: bool = False
    simulation_time_ps: float = float("nan")
    status: str = "Pending"
    stage: int = 0
    screening_score: float = float("nan")
    flags: list[str] = field(default_factory=list)

    def to_results_row(self) -> dict[str, Any]:
        return {
            "MOF": self.mof,
            "adsorption_energy": self.adsorption_energy_ev,
            "msd": self.msd_a2,
            "mech_stabb": self.mechanical_stability_index,
        }


_chgnet_model: CHGNet | None = None
_relaxer: StructOptimizer | None = None
_calculator: CHGNetCalculator | None = None
_isolated_k_energy: float | None = None
_bulk_k_energy: float | None = None
_logger: logging.Logger | None = None
_active_device: str | None = None
_cpu_fallback_mofs: set[str] = set()


def setup_logging(output_dir: Path) -> logging.Logger:
    """Configure file and console logging."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("chgnett_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(output_dir / "pipeline.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def get_logger() -> logging.Logger:
    """Return pipeline logger; auto-initialize a basic logger if needed."""
    global _logger
    if _logger is None:
        _logger = logging.getLogger("chgnett")
        if not _logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            _logger.addHandler(handler)
            _logger.setLevel(logging.INFO)
    return _logger


def clear_cuda_memory(full: bool = True) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if full:
            torch.cuda.ipc_collect()


def _is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "CUDA out of memory" in str(exc)


def _resolve_initial_device() -> str | None:
    if USE_DEVICE in (None, "cuda"):
        if torch.cuda.is_available():
            return "cuda"
        return "cpu" if USE_DEVICE == "cuda" else None
    return USE_DEVICE


def get_active_device() -> str | None:
    global _active_device
    if _active_device is None:
        _active_device = _resolve_initial_device()
    return _active_device


def _reset_chgnet_globals() -> None:
    global _chgnet_model, _relaxer, _calculator
    _chgnet_model = None
    _relaxer = None
    _calculator = None


def switch_to_cpu(mof: str | None = None) -> None:
    global _active_device
    logger = get_logger()
    if mof:
        logger.warning("CUDA out of memory for %s. Falling back to CPU.", mof)
        _cpu_fallback_mofs.add(mof)
    clear_cuda_memory(full=True)
    _active_device = "cpu"
    _reset_chgnet_globals()


def run_with_device_fallback(
    func: Any,
    *args: Any,
    mof: str | None = None,
    on_fallback: Any | None = None,
    **kwargs: Any,
) -> Any:
    try:
        return func(*args, **kwargs)
    except Exception as exc:
        if _is_cuda_oom(exc) and get_active_device() == "cuda":
            raise CudaOOMError(f"CUDA OOM for {mof}: {str(exc)}") from exc
        raise


def log_device_for_mof(mof: str) -> None:
    logger = get_logger()
    if mof in _cpu_fallback_mofs or get_active_device() == "cpu":
        logger.info("%s | Device: CPU (CUDA OOM fallback)", mof)
    elif get_active_device() == "cuda":
        logger.info("%s | Device: CUDA", mof)
    else:
        logger.info("%s | Device: %s", mof, str(get_active_device()).upper())


def get_chgnet() -> CHGNet:
    global _chgnet_model
    if _chgnet_model is None:
        device = get_active_device()
        get_logger().info("Loading CHGNet model on %s", device if device is not None else "auto")
        _chgnet_model = CHGNet.load(verbose=False, use_device=device)
    return _chgnet_model


def get_relaxer() -> StructOptimizer:
    global _relaxer
    if _relaxer is None:
        device = get_active_device()
        _relaxer = StructOptimizer(
            model=get_chgnet(),
            use_device=device,
            on_isolated_atoms="warn",
        )
    return _relaxer


def get_calculator() -> CHGNetCalculator:
    global _calculator
    if _calculator is None:
        device = get_active_device()
        _calculator = CHGNetCalculator(
            model=get_chgnet(),
            use_device=device,
            on_isolated_atoms="ignore",
        )
    return _calculator


def mof_work_dir(mof: str) -> Path:
    safe = mof.replace("/", "_").replace("\\", "_")
    return OUTPUT_DIR / "structures" / safe


def _lattice_dict(structure: Structure) -> dict[str, float]:
    a, b, c = structure.lattice.abc
    alpha, beta, gamma = structure.lattice.angles
    return {
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
    }


def minimum_interatomic_distance(structure: Structure) -> float:
    """Return the shortest periodic interatomic distance in Å."""
    if len(structure) < 2:
        return float("inf")
    distances = np.asarray(structure.distance_matrix, dtype=float)
    distances = distances[np.triu_indices_from(distances, k=1)]
    finite = distances[np.isfinite(distances) & (distances > 1e-12)]
    return float(np.min(finite)) if finite.size else float("inf")


def lattice_cell_mismatch(
    relaxed: Structure,
    k_structure: Structure,
    length_tolerance: float = 0.02,
    volume_tolerance: float = 0.03,
) -> bool:
    """Return whether relaxed bare and K-loaded cells differ beyond tolerance."""
    relaxed_lengths = np.asarray(relaxed.lattice.abc, dtype=float)
    k_lengths = np.asarray(k_structure.lattice.abc, dtype=float)
    length_change = np.max(
        np.abs(relaxed_lengths - k_lengths) / np.maximum(np.abs(k_lengths), 1e-12)
    )
    volume_change = abs(relaxed.volume - k_structure.volume) / max(abs(k_structure.volume), 1e-12)
    return bool(length_change > length_tolerance or volume_change > volume_tolerance)


def classify_bare_relaxation(final_fmax: float) -> str:
    """Apply the hard post-relaxation force acceptance rule."""
    return (
        "relaxed"
        if np.isfinite(final_fmax) and final_fmax <= BARE_RELAX_ACCEPTANCE_FMAX
        else "not_relaxed"
    )


def _write_bare_relaxation_summary(mof: str, result: BareRelaxationResult) -> Path:
    path = mof_work_dir(mof) / "relax_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_summary(mof), f, indent=2)
    except Exception as exc:
        get_logger().error("Failed to save relaxation summary %s: %s", path, exc)
    return path


def relax_bare_structure(
    structure: Structure,
    mof: str,
    *,
    k_structure: Structure | None = None,
    resume: bool = True,
    fmax: float | None = None,
    max_steps: int | None = None,
    relax_cell: bool | None = None,
) -> BareRelaxationResult:
    """Strictly relax a bare framework with FIRE and optional cell relaxation."""
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE

    logger = get_logger()
    started = time.perf_counter()
    target_fmax = float(BARE_RELAX_FMAX if fmax is None else fmax)
    target_steps = int(BARE_RELAX_MAX_STEPS if max_steps is None else max_steps)
    cell_relax = bool(BARE_RELAX_CELL if relax_cell is None else relax_cell)
    work = mof_work_dir(mof)
    work.mkdir(parents=True, exist_ok=True)
    relaxed_path = work / "bare_relaxed.cif"
    summary_path = work / "relax_summary.json"

    if resume and relaxed_path.is_file() and summary_path.is_file():
        try:
            with open(summary_path, encoding="utf-8") as f:
                cached = json.load(f)
            same_settings = (
                cached.get("status") == "relaxed"
                and abs(float(cached.get("fmax_target_eV_per_A", -1.0)) - target_fmax) <= 1e-12
                and int(cached.get("max_steps", -1)) == target_steps
                and bool(cached.get("relax_cell")) == cell_relax
            )
            if same_settings:
                logger.info("Reusing fully relaxed structure: %s", relaxed_path.name)
                result = BareRelaxationResult(
                    structure=load_structure(relaxed_path),
                    status="relaxed",
                    final_fmax=float(cached["final_fmax_eV_per_A"]),
                    n_steps=int(cached["n_steps"]),
                    used_fallback=bool(cached["used_fallback"]),
                    hit_max_steps=bool(cached.get("hit_max_steps", False)),
                    min_dist_initial=float(cached["min_dist_initial_A"]),
                    energy_initial=float(cached["energy_initial_eV"]),
                    energy_final=float(cached["energy_final_eV"]),
                    energy_per_atom_final=float(cached["energy_per_atom_final_eV"]),
                    volume_initial=float(cached["volume_initial_A3"]),
                    volume_final=float(cached["volume_final_A3"]),
                    volume_change_percent=float(cached["volume_change_percent"]),
                    lattice_initial=dict(cached["lattice_initial"]),
                    lattice_final=dict(cached["lattice_final"]),
                    k_cif_cell_mismatch=bool(cached["k_cif_cell_mismatch"]),
                    wall_time_s=float(cached["wall_time_s"]),
                )
                if k_structure is not None:
                    result.k_cif_cell_mismatch = lattice_cell_mismatch(result.structure, k_structure)
                return result
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring invalid bare relaxation cache for %s: %s", mof, exc)

    result = BareRelaxationResult()
    result.min_dist_initial = minimum_interatomic_distance(structure)
    result.volume_initial = float(structure.volume)
    result.lattice_initial = _lattice_dict(structure)
    if result.min_dist_initial < BARE_RELAX_MIN_DISTANCE:
        result.status = "broken_geometry"
        result.error = f"minimum interatomic distance {result.min_dist_initial:.4f} Å < {BARE_RELAX_MIN_DISTANCE:.2f} Å"
        result.wall_time_s = time.perf_counter() - started
        logger.error("%s | broken_geometry: %s", mof, result.error)
        _write_bare_relaxation_summary(mof, result)
        return result

    try:
        atoms = AseAtomsAdaptor.get_atoms(structure)
        atoms.calc = get_calculator()
        result.energy_initial = float(atoms.get_potential_energy())
        final_atoms = atoms
        used_fallback = False
        hit_max_steps = False
        current_fmax = target_fmax
        n_steps = 0

        for attempt in range(MAX_OPT_RETRIES + 1):
            final_atoms = AseAtomsAdaptor.get_atoms(
                AseAtomsAdaptor.get_structure(final_atoms)
            )
            final_atoms.calc = get_calculator()
            target = FrechetCellFilter(final_atoms) if cell_relax else final_atoms
            optimizer = FIRE(target)
            run_with_device_fallback(
                lambda: optimizer.run(fmax=current_fmax, steps=target_steps),
                mof=mof,
            )
            attempt_steps = optimizer.get_number_of_steps()
            n_steps += attempt_steps
            hit_max_steps = hit_max_steps or attempt_steps >= target_steps
            final_atoms.calc = get_calculator()
            true_forces = np.asarray(
                run_with_device_fallback(final_atoms.get_forces, mof=mof)
            )
            true_fmax = float(np.max(np.linalg.norm(true_forces, axis=1)))
            if true_fmax <= target_fmax:
                break
            if attempt >= MAX_OPT_RETRIES:
                break
            used_fallback = True
            current_fmax = max(current_fmax, FORCE_TOL_LOOSE)
            logger.warning(
                "%s | bare relaxation retry %d: true fmax=%.4f eV/Å",
                mof,
                attempt + 1,
                true_fmax,
            )

        final_structure = AseAtomsAdaptor.get_structure(final_atoms)
        final_atoms.calc = get_calculator()
        result.final_fmax = float(
            np.max(np.linalg.norm(np.asarray(final_atoms.get_forces()), axis=1))
        )
        result.energy_final = float(final_atoms.get_potential_energy())
        result.energy_per_atom_final = result.energy_final / max(len(final_structure), 1)
        result.structure = final_structure
        result.n_steps = n_steps
        result.used_fallback = used_fallback
        result.hit_max_steps = hit_max_steps
        result.volume_final = float(final_structure.volume)
        result.volume_change_percent = (
            100.0 * (result.volume_final - result.volume_initial) / result.volume_initial
            if result.volume_initial
            else float("nan")
        )
        result.lattice_final = _lattice_dict(final_structure)
        result.status = classify_bare_relaxation(result.final_fmax)
        if k_structure is not None:
            result.k_cif_cell_mismatch = lattice_cell_mismatch(final_structure, k_structure)
            if result.k_cif_cell_mismatch:
                logger.warning(
                    "K-loaded CIF was built on unrelaxed cell; regenerate K positions on relaxed framework"
                )
        try:
            final_structure.to(fmt="cif", filename=str(relaxed_path))
        except Exception as exc:
            result.status = "io_error"
            result.error = f"Failed to save relaxed structure {relaxed_path}: {exc}"
            logger.error("%s", result.error)
        if result.status == "not_relaxed":
            logger.error(
                "%s | not_relaxed: true final fmax=%.4f eV/Å exceeds %.4f eV/Å",
                mof, result.final_fmax, BARE_RELAX_ACCEPTANCE_FMAX,
            )
    except (CudaOOMError, RuntimeError, ValueError, OSError) as exc:
        result.status = "not_relaxed"
        result.error = str(exc)
        logger.error("%s | bare relaxation failed: %s", mof, exc)

    result.wall_time_s = time.perf_counter() - started
    _write_bare_relaxation_summary(mof, result)
    return result


def checkpoint_path(mof: str) -> Path:
    return OUTPUT_DIR / "checkpoints" / f"{mof}.json"


def ps_to_steps(time_ps: float) -> int:
    return max(1, int(round(time_ps * 1000.0 / TIME_STEP_FS)))


def md_thermostat_for_stage(stage_label: str) -> str:
    """Berendsen for Stage-1 equilibration; Nose-Hoover for Stage-2 production."""
    if stage_label == "stage2":
        return MD_THERMOSTAT_PRODUCTION
    return MD_THERMOSTAT_EQUIL


def md_diffusion_tier1_allowed(stage_label: str, simulation_time_ps: float) -> bool:
    """
    Whether Tier-1 MD Einstein diffusion is appropriate.

    Stage-1 equilibration is for thermalization / structural screening only.
    Rare-event K hopping is not expected to enter a linear Einstein regime
    until Stage-2 production (≥ DIFFUSION_SUFFICIENT_SIM_TIME_PS).
    """
    if stage_label != "stage2":
        return False
    return (
        np.isfinite(simulation_time_ps)
        and simulation_time_ps >= DIFFUSION_SUFFICIENT_SIM_TIME_PS - 0.1
    )


def load_structure(path: Path) -> Structure:
    return Structure.from_file(str(path))


def structure_fingerprint(structure: Structure, source_path: Path | None = None) -> dict[str, Any]:
    """Fingerprint a structure for validating MD-related caches."""
    source_sha256 = None
    if source_path is not None and source_path.is_file():
        source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    return {
        "source_sha256": source_sha256,
        "n_atoms": len(structure),
        "species": [site.specie.symbol for site in structure],
        "n_k": count_element(structure, ADSORBATE),
    }


def fingerprint_matches(expected: dict[str, Any], actual: dict[str, Any]) -> tuple[bool, str]:
    for key in ("source_sha256", "n_atoms", "species", "n_k"):
        if expected.get(key) != actual.get(key):
            return False, f"{key} changed"
    return True, ""


def count_element(structure: Structure, symbol: str) -> int:
    return sum(1 for site in structure if site.specie.symbol == symbol)


def get_atom_indices(structure: Structure, symbol: str) -> list[int]:
    return [i for i, site in enumerate(structure) if site.specie.symbol == symbol]


def get_framework_indices(structure: Structure) -> list[int]:
    return [i for i, site in enumerate(structure) if site.specie.symbol != ADSORBATE]


def count_inserted_ions(bare: Structure, inserted: Structure) -> int:
    n_k = count_element(inserted, ADSORBATE) - count_element(bare, ADSORBATE)
    if n_k < 0:
        raise ValueError(
            f"Inserted structure has fewer {ADSORBATE} atoms than bare structure"
        )
    return n_k


def _periodic_distance(frac_a: np.ndarray, frac_b: np.ndarray, lattice_matrix: np.ndarray) -> float:
    delta = frac_a - frac_b
    delta -= np.round(delta)
    cart = np.dot(delta, lattice_matrix)
    return float(np.linalg.norm(cart))


def get_inserted_k_indices(bare: Structure, inserted: Structure) -> list[int]:
    """
    Identify newly inserted K atoms by matching any pre-existing K atoms
    between bare and inserted structures in fractional coordinates.
    """
    inserted_k_idx = [i for i, s in enumerate(inserted) if s.specie.symbol == ADSORBATE]
    bare_k_frac = [site.frac_coords for site in bare if site.specie.symbol == ADSORBATE]
    if not bare_k_frac:
        return inserted_k_idx

    lattice = inserted.lattice.matrix
    unmatched = set(inserted_k_idx)
    used_bare: set[int] = set()
    match_tol_a = 0.35
    for idx in inserted_k_idx:
        f_ins = inserted[idx].frac_coords
        best_j = None
        best_d = float("inf")
        for j, f_bare in enumerate(bare_k_frac):
            if j in used_bare:
                continue
            d = _periodic_distance(f_ins, np.asarray(f_bare), lattice)
            if d < best_d:
                best_d = d
                best_j = j
        if best_j is not None and best_d <= match_tol_a:
            used_bare.add(best_j)
            unmatched.discard(idx)
    return sorted(unmatched)


def classify_adsorption(ads_ev_per_k: float) -> str:
    """Classify binding strength from adsorption energy per K atom (eV/K)."""
    if not np.isfinite(ads_ev_per_k):
        return "Unknown"
    if ads_ev_per_k > ADSORPTION_VERY_WEAK_EV_PER_K:
        return "Very Weak"
    if ads_ev_per_k >= ADSORPTION_TARGET_MAX_EV_PER_K:
        return "Moderate"
    if ads_ev_per_k >= ADSORPTION_VERY_STRONG_EV_PER_K:
        return "Strong"
    return "Very Strong"


def compute_volume_change_pct(bare: Structure, k_inserted: Structure) -> tuple[float, float, float]:
    """
    Relative cell-volume change (%) upon K insertion.

    Returns (V_bare, V_k, ΔV/V_bare × 100).  With RELAX_CELL=False the
    optimized cells usually match the input CIFs, so this mainly reflects
    any pre-existing cell difference between the bare and K-inserted files.
    """
    v_bare = float(bare.volume)
    v_k = float(k_inserted.volume)
    if not (np.isfinite(v_bare) and np.isfinite(v_k)) or v_bare <= 0.0:
        return v_bare, v_k, float("nan")
    return v_bare, v_k, 100.0 * (v_k - v_bare) / v_bare


def uncertainty_from_confidence(confidence: float) -> str:
    if not np.isfinite(confidence):
        return "Very High"
    if confidence >= 90:
        return "Low"
    if confidence >= 75:
        return "Moderate"
    if confidence >= 50:
        return "High"
    return "Very High"


def confidence_score(result: MOFResult, opt_ok: bool, md_ok: bool) -> float:
    """
    Publication-style confidence in [0, 100] from normalized evidence terms.

    Each component is mapped to [0, 1] before weighting so no arbitrary
    absolute constants dominate. Trajectory quality uses production length
    relative to FINAL_MD_TIME_PS (longer sampling → higher confidence).
    """
    # Optimization convergence evidence
    opt_score = 1.0 if opt_ok or result.optimization_converged else 0.0

    # MD convergence evidence
    md_score = 1.0 if md_ok or result.md_converged else 0.0

    # MSI already on [0, 1]
    msi_score = float(
        np.clip(
            result.mechanical_stability_index
            if np.isfinite(result.mechanical_stability_index)
            else 0.0,
            0.0,
            1.0,
        )
    )

    # Diffusion reliability: prefer R² when MD-based; otherwise map label
    if THREE_PARAMETER_SCREENING:
        diff_score = 0.0
    elif np.isfinite(result.diffusion_fit_r2):
        diff_score = float(np.clip(result.diffusion_fit_r2, 0.0, 1.0))
    else:
        rel_map = {"High": 0.85, "Moderate": 0.65, "Low": 0.40, "Very Low": 0.20, "Rejected": 0.0}
        diff_score = rel_map.get(result.diffusion_reliability, 0.25)
        if not np.isfinite(result.diffusion_cm2_s):
            diff_score = 0.0

    # Adsorption reliability: finite energy near the chemically preferred window
    if not np.isfinite(result.adsorption_energy_ev):
        ads_score = 0.0
    else:
        ads = result.adsorption_energy_ev
        # Gaussian preference around ADSORPTION_TARGET_EV_PER_K, clipped to [0,1]
        ads_score = float(
            np.exp(-0.5 * ((ads - ADSORPTION_TARGET_EV_PER_K) / ADSORPTION_SIGMA_EV) ** 2)
        )
        if result.adsorption_site_type:
            ads_score = min(1.0, ads_score + 0.05)  # small bonus for site-resolved search
        if "adsorption_extreme_positive" in result.flags or "adsorption_extreme_negative" in result.flags:
            ads_score *= 0.25

    # Trajectory quality: fraction of intended production length achieved
    traj_score = 0.0
    if np.isfinite(result.simulation_time_ps) and FINAL_MD_TIME_PS > 0:
        traj_score = float(np.clip(result.simulation_time_ps / FINAL_MD_TIME_PS, 0.0, 1.0))

    # Soft penalty for many independent warning flags (normalized, not arbitrary)
    n_flags = len(result.flags)
    warning_factor = float(np.clip(1.0 - 0.05 * n_flags, 0.50, 1.0))

    score = (
        CONFIDENCE_OPT_WEIGHT * opt_score
        + CONFIDENCE_MD_WEIGHT * md_score
        + CONFIDENCE_MSI_WEIGHT * msi_score
        + CONFIDENCE_DIFFUSION_WEIGHT * diff_score
        + CONFIDENCE_ADSORPTION_WEIGHT * ads_score
        + CONFIDENCE_TRAJECTORY_WEIGHT * traj_score
    )
    return float(np.clip(100.0 * score * warning_factor, 0.0, 100.0))


def save_md_diagnostics(
    mof: str,
    stage_label: str,
    time_ps: np.ndarray,
    msd: np.ndarray,
    temperature_k: np.ndarray,
    energy_ev: np.ndarray,
    fit_start: int,
    fit_end: int,
    fit_slope_a2_ps: float,
    fit_intercept_a2: float,
) -> None:
    plots_dir = OUTPUT_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    base = plots_dir / f"{mof}_{stage_label}"

    plt.figure(figsize=(6, 4))
    plt.plot(time_ps, msd, lw=1.5, label="MSD")
    if 0 <= fit_start < len(time_ps) and 0 <= fit_end < len(time_ps) and np.isfinite(fit_slope_a2_ps):
        t_fit = time_ps[fit_start : fit_end + 1]
        y_fit = fit_slope_a2_ps * t_fit + fit_intercept_a2
        plt.plot(t_fit, y_fit, "r--", lw=2.0, label="Linear fit")
        plt.axvspan(t_fit[0], t_fit[-1], color="orange", alpha=0.15, label="Fitted region")
    plt.xlabel("Time (ps)")
    plt.ylabel("MSD (Å²)")
    plt.title(f"{mof} ({stage_label}) - K-ion MSD")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(str(base) + "_msd_vs_time.png", dpi=150)
    plt.close()

    if len(energy_ev) == len(time_ps):
        plt.figure(figsize=(6, 4))
        plt.plot(time_ps, energy_ev, lw=1.5)
        plt.xlabel("Time (ps)")
        plt.ylabel("Potential Energy (eV)")
        plt.title(f"{mof} - Energy vs Time")
        plt.tight_layout()
        plt.savefig(str(base) + "_energy_vs_time.png", dpi=150)
        plt.close()

    if len(temperature_k) == len(time_ps):
        plt.figure(figsize=(6, 4))
        plt.plot(time_ps, temperature_k, lw=1.5)
        plt.xlabel("Time (ps)")
        plt.ylabel("Temperature (K)")
        plt.title(f"{mof} - Temperature vs Time")
        plt.tight_layout()
        plt.savefig(str(base) + "_temperature_vs_time.png", dpi=150)
        plt.close()


def pair_mof_files() -> list[tuple[str, Path, Path]]:
    """Pair bare and K-inserted CIFs by identical filename."""
    bare_dir = BARE_CIFS_DIR
    k_dir = CIFS_DIR
    pairs: list[tuple[str, Path, Path]] = []
    logger = get_logger()

    if not bare_dir.is_dir():
        logger.error("Bare CIF directory not found: %s", bare_dir)
        return pairs
    if not k_dir.is_dir():
        logger.error("K-inserted CIF directory not found: %s", k_dir)
        return pairs

    bare_files = {p.name: p for p in bare_dir.glob("*.cif")}
    k_files = {p.name: p for p in k_dir.glob("*.cif")}
    all_names = sorted(set(bare_files) | set(k_files))

    for name in all_names:
        bare_path = bare_files.get(name)
        k_path = k_files.get(name)
        mof_name = Path(name).stem
        if bare_path is None:
            logger.warning("Missing bare CIF for %s — skipping", name)
            continue
        if k_path is None:
            logger.warning("Missing K-inserted CIF for %s — skipping", name)
            continue
        pairs.append((mof_name, bare_path, k_path))

    logger.info("Paired %d MOF structures for screening", len(pairs))
    return pairs


def load_checkpoint(mof: str) -> dict[str, Any]:
    path = checkpoint_path(mof)
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_checkpoint(mof: str, data: dict[str, Any]) -> None:
    path = checkpoint_path(mof)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_isolated_k_energy() -> float:
    """Compute CHGNet energy of an isolated K atom in a large periodic box."""
    global _isolated_k_energy
    cache = OUTPUT_DIR / "isolated_K_energy.json"
    if _isolated_k_energy is not None:
        return _isolated_k_energy
    if cache.is_file():
        with open(cache, encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("box_size_A") == ISOLATED_K_BOX_SIZE_A:
            _isolated_k_energy = float(cached["energy_eV"])
            get_logger().info("Loaded cached isolated K energy: %.4f eV", _isolated_k_energy)
            return _isolated_k_energy
        get_logger().info(
            "Isolated K cache invalidated (box size %.1f -> %.1f Å)",
            cached.get("box_size_A", 0.0),
            ISOLATED_K_BOX_SIZE_A,
        )

    logger = get_logger()
    logger.info("Computing isolated %s atom energy with CHGNet", ADSORBATE)
    half = ISOLATED_K_BOX_SIZE_A / 2.0
    atoms = Atoms(
        ADSORBATE,
        positions=[[half, half, half]],
        cell=[[ISOLATED_K_BOX_SIZE_A, 0, 0], [0, ISOLATED_K_BOX_SIZE_A, 0], [0, 0, ISOLATED_K_BOX_SIZE_A]],
        pbc=True,
    )
    def _compute() -> float:
        atoms.calc = get_calculator()
        return float(atoms.get_potential_energy())

    energy = run_with_device_fallback(_compute, mof="isolated_K")
    _isolated_k_energy = energy
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w", encoding="utf-8") as f:
        json.dump({"energy_eV": energy, "box_size_A": ISOLATED_K_BOX_SIZE_A}, f, indent=2)
    logger.info("Isolated %s energy: %.4f eV (cached)", ADSORBATE, energy)
    return energy


def get_bulk_k_energy() -> float:
    """
    CHGNet energy per atom of BCC potassium (2-atom conventional cell).

    Bulk metal references are standard in intercalation/formation-energy
    thermochemistry (cf. Materials Project / electrode voltage conventions).
    Optional via ADSORPTION_REFERENCE='bulk'; default remains isolated K.
    """
    global _bulk_k_energy
    cache = OUTPUT_DIR / "bulk_K_energy.json"
    if _bulk_k_energy is not None:
        return _bulk_k_energy
    if cache.is_file():
        with open(cache, encoding="utf-8") as f:
            cached = json.load(f)
        if abs(float(cached.get("a_A", 0.0)) - BULK_K_BCC_A_A) < 1e-6:
            _bulk_k_energy = float(cached["energy_eV_per_atom"])
            get_logger().info("Loaded cached bulk BCC K energy: %.4f eV/atom", _bulk_k_energy)
            return _bulk_k_energy

    logger = get_logger()
    logger.info("Computing bulk BCC %s energy with CHGNet (a=%.3f Å)", ADSORBATE, BULK_K_BCC_A_A)
    a = BULK_K_BCC_A_A
    # Conventional BCC cell: atoms at (0,0,0) and (0.5,0.5,0.5)
    atoms = Atoms(
        f"{ADSORBATE}2",
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        cell=[[a, 0, 0], [0, a, 0], [0, 0, a]],
        pbc=True,
    )

    def _compute() -> float:
        atoms.calc = get_calculator()
        return float(atoms.get_potential_energy()) / 2.0

    energy = run_with_device_fallback(_compute, mof="bulk_K")
    _bulk_k_energy = energy
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w", encoding="utf-8") as f:
        json.dump({"energy_eV_per_atom": energy, "a_A": a, "structure": "bcc"}, f, indent=2)
    logger.info("Bulk BCC %s energy: %.4f eV/atom (cached)", ADSORBATE, energy)
    return energy


def get_k_reference_energy(reference: str | None = None) -> tuple[float, str]:
    """
    Return (E_K_ref, reference_label) for adsorption energy formulas.

    reference:
      'isolated' (default) — gas-phase-like isolated atom
      'bulk'               — BCC metallic K per atom
    """
    ref = (reference or ADSORPTION_REFERENCE or "isolated").strip().lower()
    if ref in {"bulk", "bcc", "metal"}:
        return get_bulk_k_energy(), "bulk"
    return get_isolated_k_energy(), "isolated"


def optimize_structure(
    structure: Structure,
    label: str,
    mof: str,
    fmax: float | None = None,
    allow_retry: bool = True,
    resume: bool = True,
    input_fingerprint: dict[str, Any] | None = None,
) -> OptimizationResult:
    """
    Relax a structure with CHGNet StructOptimizer.

    Targets FORCE_TOL (0.01 eV/Å) by default. If the structure does not
    converge, optionally retries with FORCE_TOL_LOOSE and flags unconverged
    geometries rather than silently accepting them.
    """
    logger = get_logger()
    work = mof_work_dir(mof)
    work.mkdir(parents=True, exist_ok=True)
    opt_path = work / f"{label}_optimized.cif"
    meta_path = work / f"{label}_opt_meta.json"
    target_fmax = float(FORCE_TOL if fmax is None else fmax)

    if resume and opt_path.is_file() and meta_path.is_file():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("success"):
            valid_cache = True
            if input_fingerprint is not None:
                cached_fp = meta.get("input_fingerprint")
                if cached_fp is None:
                    logger.info("legacy cache without fingerprint — recomputing")
                    valid_cache = False
                else:
                    valid_cache, reason = fingerprint_matches(input_fingerprint, cached_fp)
                    if not valid_cache:
                        logger.info("cache invalidated: %s", reason)
            if valid_cache:
                cached_fmax = float(meta.get("fmax_target", FORCE_TOL_LOOSE))
                if cached_fmax <= target_fmax + 1e-9 or meta.get("converged", False):
                    logger.info("Reusing optimized structure: %s", opt_path.name)
                    return OptimizationResult(
                        structure=load_structure(opt_path),
                        energy=meta["energy"],
                        n_steps=meta["n_steps"],
                        max_force=meta["max_force"],
                        success=True,
                        converged=bool(meta.get("converged", meta["max_force"] <= target_fmax)),
                        fmax_target=cached_fmax,
                    )

    result = OptimizationResult(fmax_target=target_fmax)
    try:
        current_fmax = target_fmax
        final_struct: Structure | None = None
        energy = float("nan")
        n_steps = 0
        max_force = float("nan")
        converged = False

        for attempt in range(MAX_OPT_RETRIES + 1):
            logger.info(
                "Optimizing %s (%s) attempt %d/%d with fmax=%.3f eV/Å",
                mof,
                label,
                attempt + 1,
                MAX_OPT_RETRIES + 1,
                current_fmax,
            )
            start_struct = structure if final_struct is None else final_struct
            relax_output = run_with_device_fallback(
                lambda s=start_struct, fm=current_fmax: get_relaxer().relax(
                    s,
                    fmax=fm,
                    steps=MAX_OPT_STEPS,
                    relax_cell=RELAX_CELL,
                    verbose=False,
                ),
                mof=mof,
            )
            traj = relax_output["trajectory"]
            final_struct = relax_output["final_structure"]
            forces = np.asarray(traj.forces[-1])
            max_force = float(np.max(np.linalg.norm(forces, axis=1)))
            energy = float(traj.energies[-1])
            n_steps += len(traj.energies) - 1
            converged = max_force <= target_fmax + 1e-6
            if converged:
                break
            if not allow_retry or attempt >= MAX_OPT_RETRIES:
                break
            # Physical rationale: if tight fmax fails, continue from the best
            # geometry with a looser tolerance so screening still obtains a
            # usable structure, while recording that it is not fully converged.
            current_fmax = max(current_fmax, FORCE_TOL_LOOSE)
            logger.warning(
                "Optimization for %s (%s) not converged (fmax=%.4f > %.4f); retrying",
                mof,
                label,
                max_force,
                target_fmax,
            )

        if final_struct is None:
            raise RuntimeError("Optimizer returned no final structure")

        final_struct.to(fmt="cif", filename=str(opt_path))
        meta = {
            "energy": energy,
            "n_steps": n_steps,
            "max_force": max_force,
            "success": True,
            "converged": converged,
            "fmax_target": target_fmax,
            "fmax_used": current_fmax,
            "input_fingerprint": input_fingerprint,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        result = OptimizationResult(
            structure=final_struct,
            energy=energy,
            n_steps=n_steps,
            max_force=max_force,
            success=True,
            converged=converged,
            fmax_target=target_fmax,
        )
        if converged:
            logger.info(
                "Optimization complete (%s): E=%.4f eV, steps=%d, fmax=%.4f eV/Å (converged)",
                label,
                energy,
                n_steps,
                max_force,
            )
        else:
            logger.warning(
                "Optimization UNCONVERGED (%s): E=%.4f eV, steps=%d, fmax=%.4f eV/Å "
                "(target %.4f) — structure saved but flagged",
                label,
                energy,
                n_steps,
                max_force,
                target_fmax,
            )
    except Exception as exc:
        if isinstance(exc, CudaOOMError):
            raise
        result.error = str(exc)
        logger.error("Optimization failed for %s (%s): %s", mof, label, exc)
    return result


def build_k_on_bare_framework(
    bare_opt: Structure,
    k_struct: Structure,
    k_indices: list[int],
) -> tuple[Structure, list[int]]:
    """Place inserted K ions on the relaxed bare MOF framework (fixed lattice)."""
    if not k_indices:
        raise ValueError(f"No inserted {ADSORBATE} indices supplied")
    bare_lengths = np.array(bare_opt.lattice.abc)
    k_lengths = np.array(k_struct.lattice.abc)
    if np.any(np.abs(bare_lengths - k_lengths) / np.maximum(k_lengths, 1e-6) > 0.02):
        get_logger().warning(
            "Bare/K lattice mismatch >2%% — using bare lattice for K placement"
        )
    merged = bare_opt.copy()
    new_k_indices: list[int] = []
    for idx in k_indices:
        k_site = k_struct[idx]
        merged.append(k_site.specie, k_site.frac_coords, coords_are_cartesian=False)
        new_k_indices.append(len(merged) - 1)
    return merged, new_k_indices


# Legacy path: despite its name, this relaxes K plus the framework atoms.
def relax_k_ions_only(
    structure: Structure,
    k_indices: list[int],
    mof: str,
    label: str = "k_on_bare",
    reuse_cache: bool = True,
    write_output_files: bool = True,
) -> OptimizationResult:
    """Relax inserted K ions and MOF framework."""
    logger = get_logger()
    work = mof_work_dir(mof)
    work.mkdir(parents=True, exist_ok=True)
    opt_path = work / f"{label}_optimized.cif"
    meta_path = work / f"{label}_opt_meta.json"

    if reuse_cache and opt_path.is_file() and meta_path.is_file():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("success"):
            logger.info("Reusing fully relaxed structure: %s", opt_path.name)
            return OptimizationResult(
                structure=load_structure(opt_path),
                energy=meta["energy"],
                n_steps=meta["n_steps"],
                max_force=meta["max_force"],
                success=True,
                converged=bool(meta.get("converged", True)),
                fmax_target=float(meta.get("fmax_target", FORCE_TOL)),
            )

    result = OptimizationResult(fmax_target=FORCE_TOL)
    try:
        logger.info("Relaxing inserted %s ions and bare framework for %s", ADSORBATE, mof)
        atoms = AseAtomsAdaptor.get_atoms(structure)
        atoms.calc = get_calculator()
        optimizer = FIRE(atoms)

        def _rebind_cpu_calculator() -> None:
            atoms.calc = get_calculator()

        run_with_device_fallback(
            lambda: optimizer.run(fmax=FORCE_TOL, steps=MAX_OPT_STEPS),
            mof=mof,
            on_fallback=_rebind_cpu_calculator,
        )
        forces = np.asarray(
            run_with_device_fallback(
                atoms.get_forces,
                mof=mof,
                on_fallback=_rebind_cpu_calculator,
            )
        )
        k_forces = forces[k_indices] if k_indices else forces
        max_force = float(np.max(np.linalg.norm(k_forces, axis=1))) if len(k_forces) else float("nan")
        energy = run_with_device_fallback(
            lambda: float(atoms.get_potential_energy()),
            mof=mof,
            on_fallback=_rebind_cpu_calculator,
        )
        n_steps = optimizer.get_number_of_steps()
        final_struct = AseAtomsAdaptor.get_structure(atoms)
        converged = bool(np.isfinite(max_force) and max_force <= FORCE_TOL + 1e-6)
        
        if write_output_files:
            final_struct.to(fmt="cif", filename=str(opt_path))
            meta = {
                "energy": energy,
                "n_steps": n_steps,
                "max_force": max_force,
                "success": True,
                "converged": converged,
                "fmax_target": FORCE_TOL,
                "method": "fixed_framework_k_relaxation",
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        result = OptimizationResult(
            structure=final_struct,
            energy=energy,
            n_steps=n_steps,
            max_force=max_force,
            success=True,
            converged=converged,
            fmax_target=FORCE_TOL,
        )
        logger.info(
            "K-only relaxation complete: E=%.4f eV, steps=%d, fmax(K)=%.4f eV/Å%s",
            energy,
            n_steps,
            max_force,
            "" if converged else " (UNCONVERGED)",
        )
    except Exception as exc:
        if isinstance(exc, CudaOOMError):
            raise
        result.error = str(exc)
        logger.error("K-only relaxation failed for %s: %s", mof, exc)
    return result


def relax_k_only_fixed_framework(
    structure: Structure,
    k_indices: list[int],
    mof: str,
    *,
    fmax: float = 0.05,
    max_steps: int = 1000,
) -> OptimizationResult:
    """Relax only K atoms while fixing all framework atoms and the cell."""
    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.calc = get_calculator()
    atoms.set_constraint(FixAtoms(indices=get_framework_indices(structure)))
    logger = get_logger()
    logger.info(
        "K-only prescreen relaxation for %s: constrained atoms=%d",
        mof,
        len(atoms.constraints[0].index) if atoms.constraints else 0,
    )
    optimizer = FIRE(atoms)
    run_with_device_fallback(
        lambda: optimizer.run(fmax=fmax, steps=max_steps),
        mof=mof,
    )
    atoms.calc = get_calculator()
    forces = _unconstrained_forces(atoms, mof)
    k_force = float(np.max(np.linalg.norm(forces[k_indices], axis=1))) if k_indices else float("nan")
    all_force = float(np.max(np.linalg.norm(forces, axis=1))) if len(forces) else float("nan")
    final = AseAtomsAdaptor.get_structure(atoms)
    return OptimizationResult(
        structure=final,
        energy=float(run_with_device_fallback(atoms.get_potential_energy, mof=mof)),
        n_steps=optimizer.get_number_of_steps(),
        max_force=k_force,
        max_force_K_only=k_force,
        max_force_all=all_force,
        success=True,
        converged=bool(np.isfinite(all_force) and all_force <= fmax),
        fmax_target=fmax,
    )


def relax_k_full_fixed_cell(
    structure: Structure,
    mof: str,
    *,
    fmax: float = ADSORPTION_FULL_FMAX,
    max_steps: int = ADSORPTION_FULL_MAX_STEPS,
) -> OptimizationResult:
    """Relax K and framework atoms while keeping the relaxed bare cell fixed."""
    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.set_constraint([])
    atoms.calc = get_calculator()
    logger = get_logger()
    logger.info("Full adsorption relaxation for %s: constrained atoms=0", mof)
    optimizer = FIRE(atoms)
    run_with_device_fallback(
        lambda: optimizer.run(fmax=fmax, steps=max_steps),
        mof=mof,
    )
    atoms.calc = get_calculator()
    forces = _unconstrained_forces(atoms, mof)
    all_force = float(np.max(np.linalg.norm(forces, axis=1))) if len(forces) else float("nan")
    k_indices = [i for i, symbol in enumerate(atoms.get_chemical_symbols()) if symbol == ADSORBATE]
    k_force = float(np.max(np.linalg.norm(forces[k_indices], axis=1))) if k_indices else float("nan")
    final = AseAtomsAdaptor.get_structure(atoms)
    return OptimizationResult(
        structure=final,
        energy=float(run_with_device_fallback(atoms.get_potential_energy, mof=mof)),
        n_steps=optimizer.get_number_of_steps(),
        max_force=all_force,
        max_force_K_only=k_force,
        max_force_all=all_force,
        success=True,
        converged=bool(np.isfinite(all_force) and all_force <= fmax),
        fmax_target=fmax,
    )


def _unconstrained_forces(atoms: Atoms, mof: str) -> np.ndarray:
    """Measure physical forces without applying any ASE constraints."""
    unconstrained = atoms.copy()
    unconstrained.set_constraint([])
    unconstrained.calc = atoms.calc
    return np.asarray(
        run_with_device_fallback(
            lambda: unconstrained.get_forces(apply_constraint=False),
            mof=mof,
        )
    )


def relax_k_full_relaxed_cell(
    structure: Structure,
    mof: str,
    *,
    fmax: float = INSERTION_FMAX,
    max_steps: int = INSERTION_MAX_STEPS,
) -> OptimizationResult:
    """Relax all atoms and the cell for sequential K insertion."""
    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.set_constraint([])
    atoms.calc = get_calculator()
    optimizer = FIRE(FrechetCellFilter(atoms))
    run_with_device_fallback(
        lambda: optimizer.run(fmax=fmax, steps=max_steps),
        mof=mof,
    )
    atoms.calc = get_calculator()
    forces = _unconstrained_forces(atoms, mof)
    all_force = float(np.max(np.linalg.norm(forces, axis=1))) if len(forces) else float("nan")
    k_indices = [i for i, symbol in enumerate(atoms.get_chemical_symbols()) if symbol == ADSORBATE]
    k_force = float(np.max(np.linalg.norm(forces[k_indices], axis=1))) if k_indices else float("nan")
    final = AseAtomsAdaptor.get_structure(atoms)
    return OptimizationResult(
        structure=final,
        energy=float(run_with_device_fallback(atoms.get_potential_energy, mof=mof)),
        n_steps=optimizer.get_number_of_steps(),
        max_force=all_force,
        max_force_K_only=k_force,
        max_force_all=all_force,
        success=True,
        converged=bool(np.isfinite(all_force) and all_force <= fmax),
        fmax_target=fmax,
    )


def _minimum_k_k_distance(structure: Structure) -> float:
    k_indices = get_atom_indices(structure, ADSORBATE)
    if len(k_indices) < 2:
        return float("inf")
    return min(
        _periodic_distance(
            structure[i].frac_coords,
            structure[j].frac_coords,
            structure.lattice.matrix,
        )
        for pos, i in enumerate(k_indices)
        for j in k_indices[pos + 1:]
    )


def _framework_site_map(
    bare: Structure,
    structure: Structure,
    *,
    logger: logging.Logger | None = None,
) -> dict[int, int]:
    """Map bare framework sites to structure sites by species and PBC distance."""
    mapping: dict[int, int] = {}
    for symbol in sorted({site.specie.symbol for site in bare if site.specie.symbol != ADSORBATE}):
        bare_indices = [i for i, site in enumerate(bare) if site.specie.symbol == symbol]
        structure_indices = [i for i, site in enumerate(structure) if site.specie.symbol == symbol]
        if len(bare_indices) != len(structure_indices):
            raise ValueError(
                f"framework species count mismatch for {symbol}: "
                f"{len(bare_indices)} vs {len(structure_indices)}"
            )
        cost = np.array([
            [
                _periodic_distance(
                    bare[bare_index].frac_coords,
                    structure[structure_index].frac_coords,
                    bare.lattice.matrix,
                )
                for structure_index in structure_indices
            ]
            for bare_index in bare_indices
        ])
        rows, columns = linear_sum_assignment(cost)
        for row, column in zip(rows, columns):
            distance = float(cost[row, column])
            if distance > 1.0 and logger is not None:
                logger.warning(
                    "Framework mapping %s[%d] -> structure[%d] is %.3f Å",
                    symbol,
                    bare_indices[row],
                    structure_indices[column],
                    distance,
                )
            mapping[bare_indices[row]] = structure_indices[column]
    return mapping


def _insertion_framework_rmsd(
    bare: Structure,
    structure: Structure,
    mapping: dict[int, int] | None = None,
) -> float:
    mapping = mapping or _framework_site_map(bare, structure, logger=get_logger())
    framework_indices = sorted(mapping)
    bare_frac = np.asarray([bare[i].frac_coords for i in framework_indices])
    current_frac = np.asarray([structure[mapping[i]].frac_coords for i in framework_indices])
    delta = current_frac - bare_frac
    delta -= np.floor(delta + 0.5)
    displacement = delta @ bare.lattice.matrix
    return float(np.sqrt(np.mean(np.sum(displacement * displacement, axis=1))))


def _insertion_bond_preservation(
    bare: Structure,
    structure: Structure,
    mapping: dict[int, int] | None = None,
) -> float:
    mapping = mapping or _framework_site_map(bare, structure, logger=get_logger())
    framework_indices = list(range(len(bare)))
    symbols = [bare[i].specie.symbol for i in framework_indices]
    pairs = build_bond_pairs(bare.cart_coords, symbols, framework_indices)
    initial: dict[tuple[int, int], float] = {}
    for i, j in pairs:
        mic, _ = find_mic(bare.cart_coords[j] - bare.cart_coords[i], bare.lattice.matrix)
        initial[(i, j)] = float(np.linalg.norm(mic))
    mapped_pairs = [(mapping[i], mapping[j]) for i, j in pairs]
    mapped_initial = {
        (mapping[i], mapping[j]): distance
        for (i, j), distance in initial.items()
    }
    return bond_preservation_fraction(
        structure.cart_coords,
        mapped_pairs,
        mapped_initial,
        structure.lattice.matrix,
    )


def _insertion_candidate_metrics(
    bare: Structure,
    structure: Structure,
    energy: float,
    previous_energy: float,
    bulk_k_energy: float,
    bare_energy: float,
    *,
    final_fmax: float,
    min_kk: float,
) -> dict[str, Any]:
    mapping = _framework_site_map(bare, structure, logger=get_logger())
    n_k = count_element(structure, ADSORBATE)
    energy = float(energy)
    step_voltage = -(energy - previous_energy - bulk_k_energy)
    return {
        "n_K": n_k,
        "energy_eV": energy,
        "volume_A3": float(structure.volume),
        "volume_change_vs_bare_percent": 100.0 * (structure.volume - bare.volume) / bare.volume,
        "formation_energy_eV": energy - bare_energy - n_k * bulk_k_energy,
        "step_voltage_V": step_voltage,
        "min_K_K_dist_A": _minimum_k_k_distance(structure),
        "min_K_framework_dist_A": min(
            min_framework_distance(structure, structure[i].frac_coords)
            for i in get_atom_indices(structure, ADSORBATE)
        ),
        "framework_rmsd_vs_bare_A": _insertion_framework_rmsd(bare, structure, mapping),
        "bond_preservation": _insertion_bond_preservation(bare, structure, mapping),
        "final_fmax": final_fmax,
    }


def convex_hull_vertices(points: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Return lower convex-hull vertices sorted by loading."""
    hull: list[tuple[int, float]] = []
    for point in sorted(points):
        while len(hull) >= 2:
            x1, y1 = hull[-2]
            x2, y2 = hull[-1]
            x3, y3 = point
            if (y2 - y1) * (x3 - x2) >= (y3 - y2) * (x2 - x1):
                hull.pop()
            else:
                break
        hull.append(point)
    return hull


def insertion_capacity_mAh_g(n_k: int, cell_mass_g_mol: float) -> float:
    return float(26801.0 * n_k / cell_mass_g_mol)


def _insertion_rejection_reasons(
    metrics: dict[str, Any],
    *,
    min_kk: float,
) -> list[str]:
    reasons: list[str] = []
    if not metrics["converged"]:
        reasons.append("not_converged")
    if metrics["min_K_K_dist_A"] < min_kk:
        reasons.append("k_clustering")
    if metrics["bond_preservation"] < 0.90 or metrics["framework_rmsd_vs_bare_A"] > 1.0:
        reasons.append("framework_damage")
    if metrics["step_voltage_V"] < 0:
        reasons.append("negative_step_voltage")
    return reasons


def run_sequential_k_insertion(
    bare: Structure,
    mof: str,
    e_bare: float,
    e_k_bulk: float,
    *,
    n_max: int = INSERTION_N_MAX,
    n_prescreen: int = INSERTION_N_PRESCREEN,
    n_full: int = INSERTION_N_FULL,
    min_kk: float = INSERTION_MIN_KK,
    v_threshold: float = INSERTION_V_THRESHOLD,
    fmax: float = INSERTION_FMAX,
    max_steps: int = INSERTION_MAX_STEPS,
    resume: bool = True,
) -> dict[str, Any]:
    """Sequentially insert and relax K, stopping at the first unstable loading."""
    started = time.perf_counter()
    work = mof_work_dir(mof) / "insertion"
    work.mkdir(parents=True, exist_ok=True)
    settings = {
        "n_max": n_max, "n_prescreen": n_prescreen, "n_full": n_full,
        "min_kk": min_kk, "v_threshold": v_threshold,
        "fmax": fmax, "max_steps": max_steps,
    }
    steps_path = work / "insertion_steps.csv"
    rows: list[dict[str, Any]] = []
    current = bare.copy()
    current_energy = float(e_bare)
    start_n = 0
    if resume and steps_path.is_file():
        try:
            cached = pd.read_csv(steps_path)
            cached_settings = json.loads(
                (work / "insertion_summary.json").read_text(encoding="utf-8")
            ).get("settings")
            if cached_settings == settings:
                for _, row in cached.sort_values("n_K").iterrows():
                    n = int(row["n_K"])
                    path = work / f"k{n}_relaxed.cif"
                    if path.is_file():
                        rows.append(row.to_dict())
                        current = load_structure(path)
                        current_energy = float(row["energy_eV"])
                        start_n = n
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            get_logger().warning("%s | ignoring insertion cache: %s", mof, exc)
    if not rows:
        current.to(fmt="cif", filename=str(work / "k0_relaxed.cif"))
        rows.append({
            "n_K": 0, "energy_eV": e_bare, "volume_A3": bare.volume,
            "volume_change_vs_bare_percent": 0.0, "formation_energy_eV": 0.0,
            "step_voltage_V": float("nan"), "min_K_K_dist_A": float("inf"),
            "min_K_framework_dist_A": float("inf"), "framework_rmsd_vs_bare_A": 0.0,
            "bond_preservation": 1.0, "final_fmax": 0.0, "converged": True,
            "wall_time_s": 0.0,
        })
    stop_reason = "n_max" if start_n >= n_max else ""
    while start_n < n_max and not stop_reason:
        n = start_n + 1
        candidates: list[tuple[AdsorptionCandidate, OptimizationResult]] = []
        if n == 1:
            seed = load_structure(mof_work_dir(mof) / "best_k1.cif")
            full = relax_k_full_relaxed_cell(seed, mof, fmax=fmax, max_steps=max_steps)
            candidates = [(
                AdsorptionCandidate(
                    candidate_id=0,
                    site_type="best_k1",
                    k_frac=list(seed[-1].frac_coords),
                    anchor_symbol="K",
                ),
                full,
            )]
        else:
            discovered = discover_adsorption_candidates(
                current, max_raw_candidates=ADSORPTION_MAX_RAW_CANDIDATES,
                max_unique_candidates=ADSORPTION_MAX_UNIQUE_CANDIDATES,
            )
            existing_k = get_atom_indices(current, ADSORBATE)
            discovered = [
                c for c in discovered
                if min(
                    _periodic_distance(c.k_frac, current[i].frac_coords, current.lattice.matrix)
                    for i in existing_k
                ) >= min_kk
            ]
            for candidate in discovered:
                candidate.pre_score, candidate.pre_score_components = score_candidate_pre_relaxation(current, candidate)
            selected = _select_diverse_adsorption_candidates(discovered, n_prescreen)
            prescreen_results: list[tuple[AdsorptionCandidate, OptimizationResult]] = []
            for candidate in selected:
                trial = current.copy()
                trial.append(ADSORBATE, candidate.k_frac, coords_are_cartesian=False)
                result = relax_k_only_fixed_framework(
                    trial, get_atom_indices(trial, ADSORBATE), mof,
                    fmax=fmax, max_steps=max_steps,
                )
                if result.structure is not None:
                    prescreen_results.append((candidate, result))
            for candidate, result in sorted(
                prescreen_results, key=lambda item: item[1].energy
            )[:n_full]:
                full = relax_k_full_relaxed_cell(
                    result.structure, mof, fmax=fmax, max_steps=max_steps
                )
                if full.structure is not None:
                    candidates.append((candidate, full))
        if not candidates:
            stop_reason = "no_site"
            break
        evaluated: list[tuple[dict[str, Any], Structure, OptimizationResult]] = []
        for candidate, result in candidates:
            if result.structure is None:
                continue
            metrics = _insertion_candidate_metrics(
                bare, result.structure, result.energy, current_energy, e_k_bulk,
                e_bare,
                final_fmax=result.max_force_all, min_kk=min_kk,
            )
            metrics["converged"] = bool(result.converged)
            metrics["wall_time_s"] = time.perf_counter() - started
            rejection = _insertion_rejection_reasons(metrics, min_kk=min_kk)
            metrics["_rejection"] = ",".join(dict.fromkeys(rejection))
            get_logger().info(
                "%s | insertion n=%d candidate=%s: FIRE steps=%d, "
                "framework RMSD=%.6f Å, bond preservation=%.6f, "
                "K-K=%.6f Å, step voltage=%.6f V, final fmax=%.6f eV/Å, "
                "rejection=%s",
                mof,
                n,
                candidate.site_type,
                result.n_steps,
                metrics["framework_rmsd_vs_bare_A"],
                metrics["bond_preservation"],
                metrics["min_K_K_dist_A"],
                metrics["step_voltage_V"],
                metrics["final_fmax"],
                metrics["_rejection"] or "none",
            )
            evaluated.append((metrics, result.structure, result))
        evaluated.sort(key=lambda item: item[0]["energy_eV"])
        accepted = next((item for item in evaluated if not item[0]["_rejection"]), None)
        if accepted is None:
            stop_reason = evaluated[0][0]["_rejection"].split(",")[0] or "not_converged"
            for metrics, _, _ in evaluated:
                get_logger().warning("%s | rejected K loading %d: %s", mof, n, metrics["_rejection"])
            break
        metrics, current, result = accepted
        metrics.pop("_rejection", None)
        current_energy = result.energy
        current.to(fmt="cif", filename=str(work / f"k{n}_relaxed.cif"))
        rows.append(metrics)
        start_n = n
        pd.DataFrame(rows).to_csv(steps_path, index=False)
    if not stop_reason:
        stop_reason = "n_max"
    stable_n = max(int(row["n_K"]) for row in rows)
    hull = convex_hull_vertices([(int(row["n_K"]), float(row["formation_energy_eV"])) for row in rows])
    plateau = [
        -(hull[i + 1][1] - hull[i][1]) / (hull[i + 1][0] - hull[i][0])
        for i in range(len(hull) - 1)
    ]
    mass = float(bare.composition.weight)
    capacities = [insertion_capacity_mAh_g(int(row["n_K"]), mass) for row in rows]
    below = 0.0
    for i, voltage in enumerate(plateau):
        if voltage < v_threshold:
            below += insertion_capacity_mAh_g(hull[i + 1][0] - hull[i][0], mass)
    summary = {
        "mof_name": mof, "n_max_stable": stable_n, "stop_reason": stop_reason,
        "hull_points": hull, "plateau_voltages_V": plateau,
        "average_voltage_V": float(np.average(plateau, weights=np.diff([p[0] for p in hull]))) if plateau else float("nan"),
        "capacity_at_n_max_stable_mAh_g": insertion_capacity_mAh_g(stable_n, mass),
        "capacity_below_threshold_mAh_g": below,
        "v_threshold_V": v_threshold, "max_volume_expansion_percent": max(float(r["volume_change_vs_bare_percent"]) for r in rows),
        "M_cell_g_mol": mass, "voltages_calibrated": False, "settings": settings,
        "total_wall_time_s": time.perf_counter() - started,
    }
    with open(work / "insertion_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=True)
    pd.DataFrame(rows).to_csv(steps_path, index=False)
    fig, ax = plt.subplots()
    ax.plot(
        capacities,
        [float(r["step_voltage_V"]) if np.isfinite(r["step_voltage_V"]) else np.nan for r in rows],
        "o-",
        label="step voltage",
    )
    hull_capacities = [insertion_capacity_mAh_g(point[0], mass) for point in hull]
    if plateau:
        ax.step(
            hull_capacities,
            [plateau[0], *plateau],
            where="post",
            label="hull voltage",
        )
    ax.set(xlabel="Capacity (mAh/g)", ylabel="Step voltage (V)", title=f"{mof} (uncalibrated)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(work / "voltage_profile.png", dpi=150)
    plt.close(fig)
    return summary


def _normalize_vec(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return v / n


def _min_framework_distance(structure: Structure, frac: np.ndarray) -> float:
    """Minimum MIC distance from a fractional coordinate to any framework atom."""
    lattice = structure.lattice.matrix
    min_d = float("inf")
    for site in structure:
        if site.specie.symbol == ADSORBATE:
            continue
        d = _periodic_distance(np.asarray(frac, dtype=float), np.asarray(site.frac_coords, dtype=float), lattice)
        min_d = min(min_d, d)
    return float(min_d)


def discover_adsorption_sites(
    bare: Structure,
    k_o_distance: float = ADSORPTION_K_O_DISTANCE_A,
    metal_o_cutoff: float = ADSORPTION_METAL_O_CUTOFF_A,
    max_candidates: int = ADSORPTION_MAX_CANDIDATES,
) -> list[dict[str, Any]]:
    """
    Discover symmetry-unique adsorption candidates (dict API for legacy callers).

    Delegates to scientific.adsorption_sites.discover_adsorption_candidates.
    Distances/cutoffs are chemistry-derived inside the scientific module; arguments
    are retained for backward compatibility only.
    """
    logger = get_logger()
    _ = k_o_distance, metal_o_cutoff  # legacy signature; not used
    candidates = discover_adsorption_candidates(
        bare,
        max_raw_candidates=ADSORPTION_MAX_RAW_CANDIDATES,
        max_unique_candidates=min(max_candidates, ADSORPTION_MAX_UNIQUE_CANDIDATES),
    )
    out = [
        {
            "site_type": c.site_type,
            "oxygen_index": c.anchor_index if c.anchor_symbol == "O" else -1,
            "anchor_index": c.anchor_index,
            "k_frac": c.k_frac,
            "k_o_distance_a": c.k_o_distance_a,
            "pre_score": c.pre_score,
            "candidate_id": c.candidate_id,
        }
        for c in candidates
    ]
    logger.info(
        "Discovered %d adsorption-site candidates (%s)",
        len(out),
        ", ".join(sorted({c["site_type"] for c in out})) if out else "none",
    )
    return out


def _relaxed_site_to_result(site: Any) -> AdsorptionSiteResult:
    """Map scientific.RelaxedAdsorptionSite → pipeline AdsorptionSiteResult."""
    return AdsorptionSiteResult(
        site_type=site.site_type,
        rank=site.rank,
        cluster_id=site.cluster_id,
        is_global_minimum=site.is_global_minimum,
        candidate_id=site.candidate_id,
        adsorption_energy_ev=site.adsorption_energy_ev,
        k_frac_coords=site.k_frac_coords,
        k_o_distance_a=site.k_o_distance_a,
        k_metal_distance_a=site.k_metal_distance_a,
        oxygen_index=site.oxygen_index,
        coordination_number=site.coordination_number,
        nearest_atoms=site.nearest_atoms,
        local_environment_descriptor=site.local_environment_descriptor,
        optimization_converged=site.optimization_converged,
        max_force_ev_a=site.max_force_ev_a,
        confidence_score=site.confidence_score,
        pre_score=site.pre_score,
        structure=site.structure,
        success=site.success,
        notes=site.notes,
    )


def score_adsorption_sites(
    bare_opt: Structure,
    mof: str,
    reference: str | None = None,
    candidates: list[dict[str, Any]] | None = None,
) -> list[AdsorptionSiteResult]:
    """
    Relax K-only on each candidate site (framework fixed) and rank by E_ads.

    Uses multi-start global optimization workflow:
      discover → pre-score → relax all → cluster duplicates → rank minima.

    The globally lowest (most negative) adsorption energy is the physically
    preferred site at 0 K under the chosen reference — standard DFT practice.
    """
    search, ranked = _execute_adsorption_search(bare_opt, mof, reference, candidates)
    _ = search  # callers may use save_adsorption_results via compute_adsorption_energy
    failed = [
        _relaxed_site_to_result(s)
        for s in search.all_relaxed
        if not s.success
    ]
    return ranked + failed


def _execute_adsorption_search(
    bare_opt: Structure,
    mof: str,
    reference: str | None = None,
    candidates: list[dict[str, Any]] | None = None,
) -> tuple[Any, list[AdsorptionSiteResult]]:
    """Run full adsorption search; return (AdsorptionSearchResult, ranked minima)."""
    logger = get_logger()
    e_ref, ref_label = get_k_reference_energy(reference)

    candidates_override: list[AdsorptionCandidate] | None = None
    if candidates is not None:
        candidates_override = [
            AdsorptionCandidate(
                candidate_id=int(c.get("candidate_id", i + 1)),
                site_type=str(c.get("site_type", "unknown")),
                k_frac=list(c.get("k_frac", [])),
                anchor_index=int(c.get("oxygen_index", c.get("anchor_index", -1))),
                anchor_symbol="O" if c.get("oxygen_index", c.get("anchor_index", -1)) >= 0 else "",
                k_o_distance_a=float(c.get("k_o_distance_a", float("nan"))),
                pre_score=float(c.get("pre_score", float("nan"))),
            )
            for i, c in enumerate(candidates)
        ]

    def _relax(structure: Structure, k_indices: list[int], label: str) -> OptimizationResult:
        return relax_k_ions_only(structure, k_indices, mof=mof, label=label, reuse_cache=True, write_output_files=False)

    def _energy(structure: Structure, mof_name: str) -> float:
        return structure_energy(structure, mof=mof_name)

    workers = ADSORPTION_PARALLEL_WORKERS
    if get_active_device() == "cuda":
        workers = 1

    search = run_adsorption_site_search(
        bare_opt,
        mof,
        e_ref,
        ref_label,
        _relax,
        _energy,
        adsorbate=ADSORBATE,
        max_raw_candidates=ADSORPTION_MAX_RAW_CANDIDATES,
        max_unique_candidates=ADSORPTION_MAX_UNIQUE_CANDIDATES,
        parallel_workers=workers,
        candidates_override=candidates_override,
        logger=logger,
    )
    ranked = [_relaxed_site_to_result(s) for s in search.local_minima]
    return search, ranked


def _select_diverse_adsorption_candidates(
    candidates: list[AdsorptionCandidate],
    n_prescreen: int,
) -> list[AdsorptionCandidate]:
    def family(site_type: str) -> str:
        if "metal_adjacent" in site_type:
            return "metal-adjacent oxygen"
        if "open_metal" in site_type:
            return "open metal"
        if "voronoi" in site_type:
            return "Voronoi void"
        if "electrostatic" in site_type:
            return "electrostatic minimum"
        if "pore_grid" in site_type:
            return "pore grid"
        return "oxygen"

    ranked = sorted(candidates, key=lambda c: c.pre_score, reverse=True)
    selected: list[AdsorptionCandidate] = []
    seen_types: set[str] = set()
    for candidate in ranked:
        group = family(candidate.site_type)
        if group not in seen_types and len(selected) < n_prescreen + 6:
            selected.append(candidate)
            seen_types.add(group)
    for candidate in ranked:
        if candidate not in selected:
            selected.append(candidate)
        if len(selected) >= n_prescreen + 6:
            break
    return selected


def _adsorption_validity(
    bare: Structure,
    complex_structure: Structure,
    k_index: int,
    final_fmax: float,
) -> tuple[list[str], float, float, float]:
    framework_idx = get_framework_indices(bare)
    rmsd, _ = framework_displacements(
        np.asarray(bare.cart_coords),
        np.asarray(complex_structure.cart_coords),
        framework_idx,
        bare.lattice.matrix,
    )
    symbols = [site.specie.symbol for site in bare]
    bond_pairs = build_bond_pairs(bare.cart_coords, symbols, framework_idx)
    initial_distances: dict[tuple[int, int], float] = {}
    for i, j in bond_pairs:
        mic, _ = find_mic(bare.cart_coords[j] - bare.cart_coords[i], bare.lattice.matrix)
        initial_distances[(i, j)] = float(np.linalg.norm(mic))
    bond_fraction = bond_preservation_fraction(
        np.asarray(complex_structure.cart_coords),
        bond_pairs,
        initial_distances,
        bare.lattice.matrix,
    )
    min_k_fw = min_framework_distance(
        complex_structure, complex_structure[k_index].frac_coords
    )
    flags: list[str] = []
    if not np.isfinite(final_fmax) or final_fmax > ADSORPTION_FULL_FMAX:
        flags.append("final_fmax_above_0.05_eV_per_A")
    if not np.isfinite(min_k_fw) or min_k_fw < 2.3:
        flags.append("min_K_framework_distance_below_2.3_A")
    if not np.isfinite(rmsd) or rmsd > 0.5:
        flags.append("framework_rmsd_above_0.5_A")
    if np.isfinite(rmsd) and rmsd < 1e-6:
        get_logger().warning("framework did not move in full relaxation")
        flags.append("framework_frozen")
    if not np.isfinite(bond_fraction) or bond_fraction < 0.95:
        flags.append("bond_preservation_below_95_percent")
    return flags, min_k_fw, rmsd, bond_fraction


def adsorption_energy_and_voltage(
    complex_energy: float,
    bare_energy: float,
    k_reference_energy: float,
) -> tuple[float, float]:
    """Return single-K adsorption energy and the corresponding voltage sign."""
    e_ads = float(complex_energy - bare_energy - k_reference_energy)
    return e_ads, -e_ads


def run_single_k_adsorption(
    bare: Structure,
    mof: str,
    e_bare: float,
    *,
    e_k_ref: float | None = None,
    n_prescreen: int = N_PRESCREEN,
    n_full: int = N_FULL,
    fmax: float = ADSORPTION_FULL_FMAX,
    max_steps: int = ADSORPTION_FULL_MAX_STEPS,
    resume: bool = True,
) -> dict[str, Any]:
    """Run automatic single-K adsorption on a strictly relaxed bare framework."""
    started = time.perf_counter()
    work = mof_work_dir(mof)
    work.mkdir(parents=True, exist_ok=True)
    best_path = work / "best_k1.cif"
    summary_path = work / "adsorption_k1_summary.json"
    settings = {
        "n_prescreen": n_prescreen,
        "n_full": n_full,
        "fmax": fmax,
        "max_steps": max_steps,
        "reference": "bulk",
    }
    if resume and best_path.is_file() and summary_path.is_file():
        with open(summary_path, encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("settings") == settings:
            return cached

    e_ref = float(get_bulk_k_energy() if e_k_ref is None else e_k_ref)
    candidates = discover_adsorption_candidates(
        bare,
        max_raw_candidates=ADSORPTION_MAX_RAW_CANDIDATES,
        max_unique_candidates=ADSORPTION_MAX_UNIQUE_CANDIDATES,
    )
    for candidate in candidates:
        candidate.pre_score, candidate.pre_score_components = score_candidate_pre_relaxation(
            bare, candidate
        )
    selected = _select_diverse_adsorption_candidates(candidates, n_prescreen)
    prescreened: list[tuple[AdsorptionCandidate, OptimizationResult]] = []
    for candidate in selected:
        trial = bare.copy()
        trial.append(ADSORBATE, candidate.k_frac, coords_are_cartesian=False)
        trial_result = relax_k_only_fixed_framework(
            trial, [len(trial) - 1], mof, fmax=fmax, max_steps=max_steps
        )
        if trial_result.success and trial_result.structure is not None:
            prescreened.append((candidate, trial_result))
    prescreened.sort(key=lambda item: item[1].energy)
    full_results: list[tuple[AdsorptionCandidate, OptimizationResult]] = []
    for candidate, prescreen_result in prescreened[:n_full]:
        full_result = relax_k_full_fixed_cell(
            prescreen_result.structure, mof, fmax=fmax, max_steps=max_steps
        )
        if full_result.success and full_result.structure is not None:
            full_results.append((candidate, full_result))

    if not full_results:
        summary: dict[str, Any] = {
            "mof_name": mof, "status": "flagged", "flags": ["no_successful_full_relaxation"],
            "E_ads_eV": float("nan"), "voltage_1K_V": float("nan"),
            "E_complex_eV": float("nan"), "E_bare_eV": e_bare,
            "E_K_ref_eV_per_atom": e_ref, "best_site_type": "",
            "K_O_min_dist_A": float("nan"), "K_coordination": -1,
            "min_K_framework_dist_A": float("nan"), "framework_rmsd_A": float("nan"),
            "bond_preservation": float("nan"), "final_fmax_eV_per_A": float("nan"),
            "max_force_K_only": float("nan"), "n_candidates_found": len(candidates),
            "n_prescreened": len(selected), "n_full_relaxed": 0,
            "all_site_energies": [], "wall_time_s": time.perf_counter() - started,
            "settings": settings,
        }
    else:
        candidate, best = min(full_results, key=lambda item: item[1].energy)
        final = best.structure
        k_idx = len(final) - 1
        geom = analyze_relaxed_site_geometry(final, k_idx, candidate.site_type)
        flags, min_k_fw, rmsd, bond_fraction = _adsorption_validity(
            bare, final, k_idx, best.max_force_all
        )
        e_ads, voltage = adsorption_energy_and_voltage(best.energy, e_bare, e_ref)
        final.to(fmt="cif", filename=str(best_path))
        summary = {
            "mof_name": mof, "status": "ok" if not flags else "flagged", "flags": flags,
            "E_ads_eV": e_ads, "voltage_1K_V": voltage,
            "E_complex_eV": best.energy, "E_bare_eV": e_bare,
            "E_K_ref_eV_per_atom": e_ref, "best_site_type": candidate.site_type,
            "K_O_min_dist_A": geom["k_o_distance_a"],
            "K_coordination": geom["coordination_number"],
            "min_K_framework_dist_A": min_k_fw, "framework_rmsd_A": rmsd,
            "bond_preservation": bond_fraction,
            "final_fmax_eV_per_A": best.max_force_all,
            "max_force_K_only": best.max_force_K_only,
            "n_candidates_found": len(candidates), "n_prescreened": len(selected),
            "n_full_relaxed": len(full_results),
            "all_site_energies": [r.energy - e_bare - e_ref for _, r in full_results],
            "wall_time_s": time.perf_counter() - started, "settings": settings,
        }
        ranking = {
            "algorithm_version": "adsorption_k1_change_b",
            "reference": "bulk", "best_site_type": candidate.site_type,
            "best_adsorption_energy_ev": e_ads,
            "n_candidates_discovered": len(candidates),
            "n_prescreened": len(selected), "n_full_relaxed": len(full_results),
            "sites": [
                {
                    "candidate_id": c.candidate_id, "site_type": c.site_type,
                    "pre_score": c.pre_score,
                    "adsorption_energy_ev": r.energy - e_bare - e_ref,
                    "final_fmax_eV_per_A": r.max_force_all,
                    "max_force_K_only": r.max_force_K_only,
                }
                for c, r in full_results
            ],
        }
        with open(work / "adsorption_site_ranking.json", "w", encoding="utf-8") as f:
            json.dump(ranking, f, indent=2)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def compute_adsorption_energy(
    bare_opt: Structure,
    k_struct: Structure,
    k_indices: list[int],
    n_k: int,
    mof: str,
    reference: str | None = None,
    search_sites: bool = True,
) -> float:
    """
    Adsorption energy per K (eV/K) with selectable reference.

        E_ads/K = [E(K@bare*) - E(bare*) - N_K * E_ref(K)] / N_K

    When search_sites=True and N_K==1, runs publication-quality multi-start
    site search (hundreds of candidates, full relaxation, clustered minima).

    For multi-K structures, relaxes provided K positions on fixed framework.
    """
    if n_k <= 0:
        return float("nan")
    logger = get_logger()

    if search_sites and n_k == 1:
        search, ranked = _execute_adsorption_search(bare_opt, mof, reference)
        best = next((r for r in ranked if r.success and r.rank == 1), None)
        if best is not None and best.structure is not None:
            work = mof_work_dir(mof)
            work.mkdir(parents=True, exist_ok=True)
            save_adsorption_results(search, work, mof)
            try:
                plot_paths = plot_adsorption_analysis(mof, bare_opt, search, OUTPUT_DIR)
                for pp in plot_paths:
                    logger.info("Adsorption plot: %s", pp)
            except Exception as plot_exc:
                logger.warning("Adsorption plot generation failed for %s: %s", mof, plot_exc)

            _, ref_label = get_k_reference_energy(reference)
            logger.info(
                "Best adsorption site for %s: %s  E_ads=%.4f eV/K  "
                "(confidence=%.1f, ref=%s, %d local minima)",
                mof,
                best.site_type,
                best.adsorption_energy_ev,
                best.confidence_score if np.isfinite(best.confidence_score) else float("nan"),
                ref_label,
                search.n_local_minima,
            )
            return float(best.adsorption_energy_ev)

    e_ref, _ = get_k_reference_energy(reference)
    merged, k_only_indices = build_k_on_bare_framework(bare_opt, k_struct, k_indices)
    k_relaxed = relax_k_ions_only(merged, k_only_indices, mof)
    if not k_relaxed.success or k_relaxed.structure is None:
        return float("nan")
    e_bare = structure_energy(bare_opt, mof=mof)
    e_k_system = structure_energy(k_relaxed.structure, mof=mof)
    return (e_k_system - e_bare - n_k * e_ref) / n_k


def load_adsorption_site_metadata(mof: str) -> dict[str, Any]:
    """Load adsorption_site_ranking.json if present."""
    path = mof_work_dir(mof) / "adsorption_site_ranking.json"
    if not path.is_file():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def framework_reference_from_k_structure(k_structure: Structure) -> Structure:
    """Bare-framework reference using the relaxed geometry from the K-inserted MOF."""
    ref = k_structure.copy()
    ref.remove_species([ADSORBATE])
    return ref


def effective_inserted_k_count(bare: Structure, inserted: Structure) -> int:
    """Count inserted K ions using coordinate matching when available."""
    matched = len(get_inserted_k_indices(bare, inserted))
    if matched > 0:
        return matched
    return count_inserted_ions(bare, inserted)


def min_k_k_distance(structure: Structure, k_indices: list[int]) -> float:
    """Minimum periodic MIC distance (Å) between any pair of inserted K ions."""
    if len(k_indices) < 2:
        return float("inf")
    lattice = structure.lattice.matrix
    min_dist = float("inf")
    for i, idx_i in enumerate(k_indices):
        fi = np.asarray(structure[idx_i].frac_coords, dtype=float)
        for idx_j in k_indices[i + 1 :]:
            fj = np.asarray(structure[idx_j].frac_coords, dtype=float)
            min_dist = min(min_dist, _periodic_distance(fi, fj, lattice))
    return float(min_dist)


def validate_k_insertion(bare: Structure, inserted: Structure) -> list[str]:
    """
    Flag overpacked or overcrowded K insertion in input structures.

    Screening assumes dilute insertion (few K per cell) with ≥3.50 Å K–K spacing.
    """
    flags: list[str] = []
    n_k = effective_inserted_k_count(bare, inserted)
    if n_k <= 0:
        return flags
    k_indices = get_inserted_k_indices(bare, inserted)
    if len(k_indices) >= 2:
        min_dist = min_k_k_distance(inserted, k_indices)
        if min_dist < MIN_KK:
            flags.append("k_too_close")
    return flags


def structure_energy(structure: Structure, mof: str = "unknown") -> float:
    """Single-point CHGNet energy for a pymatgen Structure."""
    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.calc = get_calculator()

    def _rebind_cpu_calculator() -> None:
        atoms.calc = get_calculator()

    return run_with_device_fallback(
        lambda: float(atoms.get_potential_energy()),
        mof=mof,
        on_fallback=_rebind_cpu_calculator,
    )


def trajectory_frame_count(traj_path: Path) -> int | None:
    """Return frame count from a trajectory file, closing handles immediately."""
    try:
        with Trajectory(str(traj_path), mode="r") as traj:
            return len(traj)
    except Exception:
        return None


def remove_md_files_with_retry(
    traj_path: Path,
    log_path: Path,
    attempts: int = 6,
) -> bool:
    """Remove MD artifacts; retry to tolerate transient Windows file locks."""
    for attempt in range(attempts):
        try:
            if traj_path.is_file():
                traj_path.unlink()
            if log_path.is_file():
                log_path.unlink()
            return True
        except OSError:
            if attempt == attempts - 1:
                return False
            time.sleep(0.5 * (attempt + 1))
    return False


def _load_last_frame_state(
    traj_path: Path,
    logger: logging.Logger,
) -> tuple["np.ndarray | None", "np.ndarray | None"]:
    """
    Read positions and momenta from the last frame of an ASE trajectory.

    Returns (positions_cartesian, momenta) as numpy arrays, or (None, None)
    on any failure.  Both arrays have shape (N_atoms, 3).
    """
    try:
        with Trajectory(str(traj_path), mode="r") as traj:
            if len(traj) == 0:
                return None, None
            last = traj[-1]
            positions = last.get_positions().copy()
            try:
                momenta = last.get_momenta().copy()
            except Exception:
                momenta = None
        logger.info(
            "  Continuing from last frame of %s (velocities %s)",
            traj_path.name,
            "loaded" if momenta is not None else "unavailable — will re-thermalise",
        )
        return positions, momenta
    except Exception as exc:
        logger.warning("  Could not read last frame from %s: %s", traj_path.name, exc)
        return None, None


def run_md(
    structure: Structure,
    mof: str,
    stage_label: str,
    time_ps: float,
    init_traj_path: "Path | None" = None,
    resume: bool = True,
    input_fingerprint: dict[str, Any] | None = None,
) -> tuple["Path | None", "Path | None"]:
    """
    Run CHGNet NVT MD and return trajectory and logfile paths.

    Parameters
    ----------
    structure      : Starting pymatgen Structure (used for topology / cell).
    mof            : MOF identifier string.
    stage_label    : 'stage1', 'stage2', etc.
    time_ps        : Total simulation time in picoseconds.
    init_traj_path : Optional path to a prior .traj file whose last frame
                     supplies atomic positions and momenta.  When supplied,
                     Stage 2 continues seamlessly from Stage 1 instead of
                     re-thermalising from the static CIF geometry.
    """
    logger = get_logger()
    work = mof_work_dir(mof)
    work.mkdir(parents=True, exist_ok=True)
    traj_path = work / f"{stage_label}_md.traj"
    log_path = work / f"{stage_label}_md.log"
    steps = ps_to_steps(time_ps)

    fingerprint_path = work / f"{stage_label}_md_fingerprint.json"
    if resume and traj_path.is_file() and log_path.is_file():
        n_frames = trajectory_frame_count(traj_path)
        cached_fp = None
        if fingerprint_path.is_file():
            with open(fingerprint_path, encoding="utf-8") as f:
                cached_fp = json.load(f)
        if cached_fp is None:
            logger.info("legacy cache without fingerprint — recomputing")
        matches = cached_fp is not None and (
            input_fingerprint is None or cached_fp == input_fingerprint
        )
        if n_frames is not None and n_frames >= steps and matches:
            logger.info("Reusing MD trajectory: %s (%d frames)", traj_path.name, n_frames)
            return traj_path, log_path
        if cached_fp is not None and not matches:
            logger.info("cache invalidated: MD input fingerprint changed")
        if n_frames is not None and n_frames > 0:
            logger.warning(
                "Incomplete %s trajectory (%d / %d frames) — removing before rerun",
                traj_path.name,
                n_frames,
                steps,
            )
        else:
            logger.warning("Unreadable trajectory %s — removing before rerun", traj_path.name)
        if not remove_md_files_with_retry(traj_path, log_path):
            stamp = int(time.time())
            traj_path = work / f"{stage_label}_md_{stamp}.traj"
            log_path = work / f"{stage_label}_md_{stamp}.log"
            fingerprint_path = work / f"{stage_label}_md_{stamp}_fingerprint.json"
            logger.warning(
                "Existing %s MD files are locked; writing to %s instead",
                stage_label,
                traj_path.name,
            )

    for attempt in range(3):
        try:
            logger.info(
                "Running %s MD for %s (%.2f ps, %d steps)",
                stage_label,
                mof,
                time_ps,
                steps,
            )
            np.random.seed(RANDOM_SEED)

            # ── Build ASE Atoms from the pymatgen Structure ───────────────────
            from pymatgen.io.ase import AseAtomsAdaptor
            atoms = AseAtomsAdaptor.get_atoms(structure)

            # ── Seed positions/velocities from a prior trajectory ─────────────
            starting_temp = TEMPERATURE_K   # default: thermalise fresh
            if init_traj_path is not None and init_traj_path.is_file():
                prior_pos, prior_mom = _load_last_frame_state(init_traj_path, logger)
                if prior_pos is not None and prior_pos.shape == atoms.get_positions().shape:
                    atoms.set_positions(prior_pos)
                    if prior_mom is not None and prior_mom.shape == atoms.get_positions().shape:
                        atoms.set_momenta(prior_mom)
                        starting_temp = None  # skip thermalisation — velocities already set
                        logger.info(
                            "  Stage 2 will continue from Stage 1 final state "
                            "(positions + momenta transferred)."
                        )
                    else:
                        logger.info(
                            "  Stage 2: positions from Stage 1, but momenta unavailable — "
                            "will re-thermalise at %d K.", TEMPERATURE_K
                        )
                else:
                    logger.warning(
                        "  init_traj_path frame shape mismatch — starting Stage 2 fresh."
                    )

            def _run_md_once() -> None:
                thermostat = md_thermostat_for_stage(stage_label)
                md = MolecularDynamics(
                    atoms=atoms,
                    model=get_chgnet(),
                    ensemble="nvt",
                    thermostat=thermostat,
                    temperature=TEMPERATURE_K,
                    starting_temperature=starting_temp,
                    timestep=TIME_STEP_FS,
                    trajectory=str(traj_path),
                    logfile=str(log_path),
                    loginterval=MD_LOGINTERVAL,
                    append_trajectory=False,
                    use_device=get_active_device(),
                    on_isolated_atoms="warn",
                )
                md.run(steps)

            logger.info("  MD thermostat: %s", md_thermostat_for_stage(stage_label))
            run_with_device_fallback(_run_md_once, mof=mof)
            logger.info("MD complete: %s", traj_path.name)
            if input_fingerprint is not None:
                with open(fingerprint_path, "w", encoding="utf-8") as f:
                    json.dump(input_fingerprint, f, indent=2)
            return traj_path, log_path
        except OSError as exc:
            if getattr(exc, "winerror", None) == 32 or "being used by another process" in str(exc):
                if attempt < 2:
                    logger.warning(
                        "MD file lock on %s (attempt %d/3) — retrying",
                        traj_path.name,
                        attempt + 1,
                    )
                    if not remove_md_files_with_retry(traj_path, log_path):
                        stamp = int(time.time())
                        traj_path = work / f"{stage_label}_md_{stamp}.traj"
                        log_path = work / f"{stage_label}_md_{stamp}.log"
                    time.sleep(1.0 * (attempt + 1))
                    continue
            logger.error("MD failed for %s (%s): %s", mof, stage_label, exc)
            return None, None
        except Exception as exc:
            if isinstance(exc, CudaOOMError):
                raise
            logger.error("MD failed for %s (%s): %s", mof, stage_label, exc)
            return None, None

    logger.error("MD failed for %s (%s): file remained locked after retries", mof, stage_label)
    return None, None


def build_bond_pairs(positions: np.ndarray, symbols: list[str], indices: list[int]) -> list[tuple[int, int]]:
    """Identify framework bonds from initial covalent radii cutoffs."""
    pairs: list[tuple[int, int]] = []
    idx_array = np.array(indices)
    sub_pos = positions[idx_array]
    n = len(indices)
    for i in range(n):
        ri = covalent_radius(symbols[indices[i]])
        for j in range(i + 1, n):
            rj = covalent_radius(symbols[indices[j]])
            cutoff = (ri + rj) * BOND_CUTOFF_SCALE
            dist = np.linalg.norm(sub_pos[i] - sub_pos[j])
            if dist <= cutoff:
                pairs.append((indices[i], indices[j]))
    return pairs


def build_bond_pairs_crystalnn(structure: Structure, framework_indices: list[int]) -> list[tuple[int, int]]:
    """Chemically informed bond graph using CrystalNN; may fail on some MOFs."""
    nn = CrystalNN(distance_cutoffs=None, x_diff_weight=0.0, porous_adjustment=True)
    allowed = set(framework_indices)
    pairs: set[tuple[int, int]] = set()
    for i in framework_indices:
        try:
            for info in nn.get_nn_info(structure, i):
                j = int(info["site_index"])
                if j in allowed and i != j:
                    pairs.add((min(i, j), max(i, j)))
        except Exception:
            continue
    return sorted(pairs)


def bond_preservation_fraction(
    positions: np.ndarray,
    bond_pairs: list[tuple[int, int]],
    initial_distances: dict[tuple[int, int], float],
    cell: np.ndarray,
) -> float:
    if not bond_pairs:
        return float("nan")
    preserved = 0
    for i, j in bond_pairs:
        d0 = initial_distances[(i, j)]
        raw = positions[j] - positions[i]
        mic, _ = find_mic(raw, cell)
        dist = float(np.linalg.norm(mic))
        if dist <= d0 * BOND_TOLERANCE_FACTOR:
            preserved += 1
    return preserved / len(bond_pairs)


def _kabsch_align(
    ref_positions: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    """Rigid-body align positions to ref_positions (Kabsch algorithm)."""
    ref_c = ref_positions - ref_positions.mean(axis=0)
    pos_c = positions - positions.mean(axis=0)
    h_mat = pos_c.T @ ref_c
    u_mat, _, vt_mat = np.linalg.svd(h_mat)
    rot = vt_mat.T @ u_mat.T
    if np.linalg.det(rot) < 0:
        vt_mat[-1, :] *= -1
        rot = vt_mat.T @ u_mat.T
    aligned = (rot @ pos_c.T).T + ref_positions.mean(axis=0)
    return aligned


def framework_displacements(
    ref_positions: np.ndarray,
    positions: np.ndarray,
    framework_idx: list[int],
    cell: np.ndarray,
    *,
    use_kabsch: bool = True,
) -> tuple[float, float]:
    """Return (RMSD, max displacement) for framework atoms with MIC.

    When use_kabsch=True (default), framework atoms are rigid-body aligned
    before displacement measurement — standard practice in MOF flexibility
  analyses to remove global translation/rotation artifacts.
    """
    if not framework_idx:
        return float("nan"), float("nan")
    ref_fw = ref_positions[framework_idx]
    pos_fw = positions[framework_idx]
    if use_kabsch and len(framework_idx) >= 3:
        pos_fw = _kabsch_align(ref_fw, pos_fw)
    disp_sq: list[float] = []
    max_disp = 0.0
    for i, idx in enumerate(framework_idx):
        raw = pos_fw[i] - ref_fw[i]
        mic, _ = find_mic(raw, cell)
        d = float(np.linalg.norm(mic))
        disp_sq.append(d * d)
        max_disp = max(max_disp, d)
    return float(np.sqrt(np.mean(disp_sq))), max_disp


def parse_md_log(log_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse temperature, potential energy, and total energy from ASE/CHGNet MD logfile.

    Actual CHGNet MolecularDynamics log column order:
        Time[ps]  Etot[eV]  Epot[eV]  Ekin[eV]  T[K]
    """
    temps: list[float] = []
    energies: list[float] = []
    totals: list[float] = []
    with open(log_path, encoding="utf-8") as f:
        header = f.readline()
        header_l = header.lower()
        # Detect column layout from header when present
        use_chgnet_order = ("time[ps]" in header_l) or ("etot" in header_l and "epot" in header_l)
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                if use_chgnet_order:
                    # Time, Etot, Epot, Ekin, T
                    totals.append(float(parts[1]))
                    energies.append(float(parts[2]))
                    temps.append(float(parts[4]))
                else:
                    # Legacy fallback
                    temps.append(float(parts[1]))
                    energies.append(float(parts[2]))
                    totals.append(float(parts[4]))
            except ValueError:
                continue
    return (
        np.asarray(temps, dtype=float),
        np.asarray(energies, dtype=float),
        np.asarray(totals, dtype=float),
    )


def classify_diffusion_reliability(
    r2: float,
    sim_time_ps: float,
    *,
    hop_count: int = -1,
    anomalous_exponent: float = float("nan"),
    md_neb_log10_agreement: float = float("nan"),
) -> str:
    if np.isfinite(anomalous_exponent) and anomalous_exponent < MIN_ANOMALOUS_EXPONENT:
        return "Very Low"
    if sim_time_ps < DIFFUSION_SUFFICIENT_SIM_TIME_PS:
        base = "Very Low"
    elif not np.isfinite(r2):
        base = "Very Low"
    elif r2 >= 0.95:
        base = "High"
    elif r2 >= 0.85:
        base = "Moderate"
    elif r2 >= 0.70:
        base = "Low"
    else:
        base = "Very Low"

    if hop_count == 0 and base in {"High", "Moderate"}:
        base = "Low"
    elif hop_count == 0 and base == "Low":
        base = "Very Low"

    if np.isfinite(md_neb_log10_agreement) and md_neb_log10_agreement > 1.5:
        order = ["High", "Moderate", "Low", "Very Low"]
        try:
            idx = order.index(base)
            base = order[min(idx + 1, len(order) - 1)]
        except ValueError:
            base = "Low"
    return base


def diffusion_outside_screening_range(d_cm2_s: float) -> bool:
    """Whether D lies outside the candidate battery screening window."""
    if not np.isfinite(d_cm2_s):
        return False
    return (
        d_cm2_s < CANDIDATE_DIFFUSION_MIN_CM2_S
        or d_cm2_s > CANDIDATE_DIFFUSION_MAX_CM2_S
    )


def diffusion_passes_physical_sanity(d_cm2_s: float) -> bool:
    """
    Finite, positive D within numerically stable bounds.
    Rejects NaN, zero/negative, and overflow/underflow artifacts.
    """
    if not np.isfinite(d_cm2_s) or d_cm2_s <= 0:
        return False
    if d_cm2_s < DIFFUSION_D_ABS_MIN_CM2_S:
        return False
    if d_cm2_s > DIFFUSION_D_ABS_MAX_CM2_S:
        return False
    return True


def apply_screening_range_reliability_cap(
    d_cm2_s: float,
    base_reliability: str,
) -> str:
    """
    Flag D outside the candidate screening window as Low confidence.
    Values inside the window keep their tier-assigned reliability.
    """
    if not diffusion_outside_screening_range(d_cm2_s):
        return base_reliability
    return "Low"


def finalize_reported_diffusion(
    tier_result: dict,
    *,
    method: str,
) -> dict | None:
    """
    Validate a tier result for CSV reporting.
    Returns None only when D fails physical sanity checks.
    """
    d = float(tier_result.get("diffusion_cm2_s", float("nan")))
    if not diffusion_passes_physical_sanity(d):
        return None
    out = dict(tier_result)
    out["diffusion_cm2_s"] = d
    out["diffusion_method"] = method
    base_rel = str(tier_result.get("diffusion_reliability", "Very Low"))
    out["diffusion_reliability"] = apply_screening_range_reliability_cap(d, base_rel)
    return out


def compute_k_msd_unwrapped_trajectory(
    trajectory: Trajectory,
    k_indices: list[int],
    framework_idx: list[int] | None = None,
) -> np.ndarray:
    """
    MSD from sequentially unwrapped K trajectories (publication standard).

    Framework-relative MSD subtracts mean framework drift computed from
    unwrapped framework atom positions.
    """
    n_frames = len(trajectory)
    if n_frames < 1 or not k_indices:
        return np.zeros(n_frames, dtype=float)

    all_pos = np.stack([traj.get_positions() for traj in trajectory], axis=0)
    cell0 = trajectory[0].get_cell().array

    k_unwrapped = unwrap_positions_mic(all_pos[:, k_indices, :], cell0)

    fw_unwrapped = None
    if framework_idx:
        fw_unwrapped = unwrap_positions_mic(all_pos[:, framework_idx, :], cell0)

    return compute_msd_unwrapped(k_unwrapped, k_indices=list(range(len(k_indices))), framework_unwrapped=fw_unwrapped)


def compute_k_msd(trajectory: Trajectory, k_indices: list[int]) -> np.ndarray:
    """MSD(t) = <|r(t)-r(0)|²> for K ions with PBC MIC."""
    n_frames = len(trajectory)
    msd = np.zeros(n_frames, dtype=float)
    ref_atoms = trajectory[0]
    ref_pos = ref_atoms.get_positions()
    ref_k = ref_pos[k_indices]

    for t in range(n_frames):
        pos = trajectory[t].get_positions()[k_indices]
        cell = trajectory[t].get_cell().array
        sq: list[float] = []
        for i in range(len(k_indices)):
            raw = pos[i] - ref_k[i]
            mic, _ = find_mic(raw, cell)
            sq.append(float(np.dot(mic, mic)))
        msd[t] = float(np.mean(sq))
    return msd


def compute_k_msd_framework_relative(
    trajectory: Trajectory,
    k_indices: list[int],
    framework_idx: list[int],
) -> np.ndarray:
    """MSD(t) for K ions relative to mean framework drift (MIC-corrected)."""
    n_frames = len(trajectory)
    msd = np.zeros(n_frames, dtype=float)
    if not k_indices:
        return msd

    ref_atoms = trajectory[0]
    ref_pos = ref_atoms.get_positions()
    ref_k = ref_pos[k_indices]
    ref_fw = ref_pos[framework_idx] if framework_idx else np.empty((0, 3))

    for t in range(n_frames):
        atoms = trajectory[t]
        pos = atoms.get_positions()
        cell = atoms.get_cell().array
        if len(framework_idx):
            fw_mic, _ = find_mic(pos[framework_idx] - ref_fw, cell)
            fw_drift = np.mean(fw_mic, axis=0)
        else:
            fw_drift = np.zeros(3, dtype=float)

        sq: list[float] = []
        for i in range(len(k_indices)):
            raw = pos[k_indices[i]] - ref_k[i]
            mic, _ = find_mic(raw, cell)
            rel = mic - fw_drift
            sq.append(float(np.dot(rel, rel)))
        msd[t] = float(np.mean(sq))
    return msd


def _diffusion_cm2_s_from_slope(slope_a2_ps: float) -> float:
    if not np.isfinite(slope_a2_ps) or slope_a2_ps <= 0:
        return float("nan")
    d_ang2_per_ps = slope_a2_ps / (2.0 * DIFFUSION_DIMENSIONS)
    return float(d_ang2_per_ps * A2_PER_PS_TO_CM2_PER_S)


def _fit_msd_window(
    msd: np.ndarray,
    time_ps: np.ndarray,
    start: int,
    end: int,
) -> dict[str, float] | None:
    """
    Linear regression of MSD vs time with statistical diagnostics and D CI.

    Returns slope t-statistic so linear-regime selection can prefer windows
    with a statistically significant positive slope (not R² alone).
    """
    if end - start < 2:
        return None
    fit = einstein_fit_with_uncertainty(
        msd, time_ps, start, end, dimensions=DIFFUSION_DIMENSIONS
    )
    if not np.isfinite(fit.get("slope_a2_ps", float("nan"))):
        return None
    return {
        "fit_r2": float(fit["fit_r2"]),
        "slope_a2_ps": float(fit["slope_a2_ps"]),
        "intercept_a2": float(fit["intercept_a2"]),
        "start_frame": float(fit["start_frame"]),
        "end_frame": float(fit["end_frame"]),
        "num_frames": float(fit["num_frames"]),
        "slope_tstat": float(fit["slope_tstat"]),
        "slope_pvalue": float(fit["slope_pvalue"]),
        "slope_stderr": float(fit["slope_stderr"]),
        "diffusion_cm2_s": float(fit["diffusion_cm2_s"]),
        "diffusion_cm2_s_lo": float(fit["diffusion_cm2_s_lo"]),
        "diffusion_cm2_s_hi": float(fit["diffusion_cm2_s_hi"]),
        "diffusion_stderr_cm2_s": float(fit["diffusion_stderr_cm2_s"]),
    }


def detect_linear_diffusive_regime(
    msd: np.ndarray,
    time_ps: np.ndarray,
    *,
    equilibration_series: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Fit Einstein relation in the linear regime using the longest valid diffusive window.

    Window selection prefers the longest plateau with R² ≥ 0.92, α ≥ 0.85,
    and monotonic MSD (publication standard for MOF ionic diffusion).
    """
    out = {
        "diffusion_cm2_s": float("nan"),
        "final_msd_a2": float(msd[-1]) if len(msd) else float("nan"),
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
        "equilibration_frame": -1.0,
        "equilibration_time_ps": float("nan"),
        "anomalous_exponent": float("nan"),
        "fit_window_ps": float("nan"),
    }
    n = len(msd)
    if n < 2:
        return out

    msd_deriv = np.gradient(msd, time_ps) if len(time_ps) == n else np.gradient(msd)
    eq_msd = detect_equilibration_index(msd_deriv)
    eq_energy = 0
    if equilibration_series is not None and len(equilibration_series) == n:
        eq_energy = detect_equilibration_index(equilibration_series)
    eq_frame = max(eq_msd, eq_energy)

    fit = select_longest_diffusive_window(
        msd,
        time_ps,
        eq_frame,
        min_r2=MIN_FICKIAN_FIT_R2,
        min_alpha=MIN_ANOMALOUS_EXPONENT,
        min_window_fraction=0.30,
        stride=max(1, LINEAR_WINDOW_STRIDE),
    )
    out.update(fit)
    out["final_msd_a2"] = float(msd[-1]) if len(msd) else float("nan")
    return out


def md_simulation_time_sufficient(
    simulation_time_ps: float,
    fit_r2: float,
    mechanical_stability: str,
) -> bool:
    """Whether trajectory length supports reporting an MD diffusion coefficient."""
    if not np.isfinite(simulation_time_ps):
        return False
    if simulation_time_ps >= DIFFUSION_SUFFICIENT_SIM_TIME_PS:
        return True
    if simulation_time_ps >= DIFFUSION_STAGE1_MIN_TIME_PS:
        return (
            np.isfinite(fit_r2)
            and fit_r2 >= MIN_DIFFUSION_FIT_R2
            and mechanical_stability in ("Stable", "Marginal")
        )
    return False


def msd_d_consistency_ok(
    d_fit: float,
    msd_lab_final: float,
    simulation_time_ps: float,
) -> bool:
    """Reject MD D when linear fit greatly exceeds endpoint MSD (localized K)."""
    if not np.isfinite(d_fit) or not np.isfinite(msd_lab_final) or simulation_time_ps <= 0:
        return True
    t_s = simulation_time_ps * 1e-12
    d_end = msd_lab_final / (6.0 * t_s * 1e16)
    if not np.isfinite(d_end) or d_end <= 0:
        return msd_lab_final >= CANDIDATE_MSD_MIN_A2
    if (
        d_fit > DIFFUSION_MSD_D_CONSISTENCY_FACTOR * d_end
        and msd_lab_final < CANDIDATE_MSD_MIN_A2
    ):
        return False
    return True


def evaluate_md_diffusion(
    msd_fw_rel: np.ndarray,
    time_ps: np.ndarray,
    *,
    mechanical_stability: str,
    md_converged: bool,
    flags: list[str],
    msd_lab_final: float,
    simulation_time_ps: float,
    equilibration_series: np.ndarray | None = None,
    hop_count: int = -1,
    logger: logging.Logger | None = None,
) -> dict[str, float | str]:
    """
    Fit framework-relative MSD and return D when Tier-1 statistical gates pass.

    Uses longest-valid-window Einstein fitting with α exponent and hop-count
    validation for publication-quality diffusion reporting.
    """
    fit = detect_linear_diffusive_regime(
        msd_fw_rel, time_ps, equilibration_series=equilibration_series
    )
    d_fit = float(fit.get("diffusion_cm2_s", float("nan")))
    r2 = float(fit.get("fit_r2", float("nan")))
    tstat = float(fit.get("slope_tstat", float("nan")))
    fit_frames = int(fit.get("num_frames", 0.0))
    alpha = float(fit.get("anomalous_exponent", float("nan")))
    fit_start = int(fit.get("start_frame", -1))
    fit_end = int(fit.get("end_frame", -1))

    out: dict[str, float | str] = {
        "diffusion_cm2_s": float("nan"),
        "diffusion_method": "md",
        "fit_r2": r2,
        "slope_a2_ps": float(fit.get("slope_a2_ps", float("nan"))),
        "intercept_a2": float(fit.get("intercept_a2", float("nan"))),
        "start_frame": float(fit.get("start_frame", -1.0)),
        "end_frame": float(fit.get("end_frame", -1.0)),
        "num_frames": float(fit_frames),
        "slope_tstat": tstat,
        "slope_pvalue": float(fit.get("slope_pvalue", float("nan"))),
        "slope_stderr": float(fit.get("slope_stderr", float("nan"))),
        "diffusion_stderr_cm2_s": float(fit.get("diffusion_stderr_cm2_s", float("nan"))),
        "diffusion_cm2_s_lo": float(fit.get("diffusion_cm2_s_lo", float("nan"))),
        "diffusion_cm2_s_hi": float(fit.get("diffusion_cm2_s_hi", float("nan"))),
        "equilibration_frame": float(fit.get("equilibration_frame", -1.0)),
        "equilibration_time_ps": float(fit.get("equilibration_time_ps", float("nan"))),
        "anomalous_exponent": alpha,
        "fit_window_ps": float(fit.get("fit_window_ps", float("nan"))),
        "hop_count": float(hop_count) if hop_count >= 0 else float("nan"),
        "diffusion_md_fit_cm2_s": d_fit if np.isfinite(d_fit) and d_fit > 0 else float("nan"),
        "diffusion_reliability": "Rejected",
        "rejection_reason": "",
        "diffusion_warning": "",
    }

    rejection_reasons: list[str] = []
    soft_warnings: list[str] = []

    if mechanical_stability == "Collapsed":
        rejection_reasons.append("collapsed framework")
    if "high_energy_drift" in flags and not md_converged:
        rejection_reasons.append("excessive energy drift with failed MD convergence")
    elif "high_energy_drift" in flags:
        soft_warnings.append("elevated energy drift")

    if not md_converged:
        soft_warnings.append("MD convergence flags present")

    if fit_start >= 0 and fit_end > fit_start:
        win = msd_fw_rel[fit_start : fit_end + 1]
        if not msd_is_monotonic(win):
            rejection_reasons.append("non-monotonic MSD in selected fit window")
    elif "msd_non_monotonic" in flags:
        soft_warnings.append("non-monotonic MSD segments")

    if not np.isfinite(r2) or r2 < MD_DIFFUSION_ACCEPT_R2:
        rejection_reasons.append(
            f"weak linear regime (R²={r2:.3f}; need R²≥{MD_DIFFUSION_ACCEPT_R2:.2f})"
        )

    if np.isfinite(alpha) and alpha < MIN_ANOMALOUS_EXPONENT:
        rejection_reasons.append(
            f"subdiffusive regime (α={alpha:.3f}; need α≥{MIN_ANOMALOUS_EXPONENT:.2f})"
        )

    if not np.isfinite(simulation_time_ps) or simulation_time_ps < MD_DIFFUSION_MIN_TIME_PS:
        sim_txt = f"{simulation_time_ps:.1f}" if np.isfinite(simulation_time_ps) else "NaN"
        rejection_reasons.append(
            f"short trajectory ({sim_txt} ps < {MD_DIFFUSION_MIN_TIME_PS:.0f} ps)"
        )

    post_eq_frames = max(1, len(msd_fw_rel) - int(fit.get("equilibration_frame", 0)))
    min_fit_frames = max(
        MD_DIFFUSION_MIN_FIT_FRAMES,
        int(post_eq_frames * 0.30),
    )
    if fit_frames < min_fit_frames:
        rejection_reasons.append(
            f"insufficient diffusive frames ({fit_frames} < {min_fit_frames})"
        )

    if hop_count == 0:
        soft_warnings.append("no K hops detected in trajectory (cage rattling / localized ion)")

    if np.isfinite(d_fit) and not msd_d_consistency_ok(d_fit, msd_lab_final, simulation_time_ps):
        rejection_reasons.append("MSD–D inconsistency (localized motion)")

    log = logger
    if log is None:
        try:
            log = get_logger()
        except RuntimeError:
            log = None

    if rejection_reasons:
        out["rejection_reason"] = "; ".join(rejection_reasons)
        if log is not None:
            log.info("  [Diffusion] Tier 1 (MD) REJECTED: %s", out["rejection_reason"])
        return out

    if np.isfinite(d_fit) and d_fit > 0:
        out["diffusion_cm2_s"] = d_fit
        rel = classify_diffusion_reliability(
            r2,
            simulation_time_ps,
            hop_count=hop_count,
            anomalous_exponent=alpha,
        )
        if soft_warnings:
            if rel == "High":
                rel = "Moderate"
            elif rel == "Moderate":
                rel = "Low"
            out["diffusion_warning"] = "; ".join(soft_warnings)
        out["diffusion_reliability"] = rel
        if log is not None:
            log.info(
                "  [Diffusion] Tier 1 (MD) ACCEPTED: D=%.3e cm²/s, R²=%.3f, "
                "α=%.3f, hops=%s, frames=%d, window=%.1f ps, reliability=%s",
                d_fit,
                r2,
                alpha if np.isfinite(alpha) else float("nan"),
                hop_count if hop_count >= 0 else "?",
                fit_frames,
                float(fit.get("fit_window_ps", float("nan"))),
                out["diffusion_reliability"],
            )
    return out


def apply_diffusion_reporting_gate(result: MDAnalysisResult, time_ps: float) -> None:
    """Suppress diffusion values that are not physically reliable for screening."""
    if not np.isfinite(result.diffusion_cm2_s):
        return
    if np.isfinite(result.diffusion_fit_r2) and result.diffusion_fit_r2 < MIN_DIFFUSION_REPORT_R2:
        if "low_r2" not in result.flagged:
            result.flagged.append("low_r2")
    if np.isfinite(time_ps) and not md_simulation_time_sufficient(
        time_ps,
        result.diffusion_fit_r2,
        result.mechanical_stability,
    ):
        if "short_trajectory" not in result.flagged:
            result.flagged.append("short_trajectory")
    if not diffusion_reliability_criteria_met(
        mechanical_stability=result.mechanical_stability,
        md_converged=result.md_converged,
        flags=result.flagged,
        diffusion_slope_a2_ps=result.diffusion_slope_a2_ps,
        diffusion_fit_r2=result.diffusion_fit_r2,
        simulation_time_ps=time_ps,
    ):
        if "diffusion_not_reliable" not in result.flagged:
            result.flagged.append("diffusion_not_reliable")
        result.diffusion_cm2_s = float("nan")


def diffusion_reliability_criteria_met(
    *,
    mechanical_stability: str,
    md_converged: bool,
    flags: list[str],
    diffusion_slope_a2_ps: float,
    diffusion_fit_r2: float,
    simulation_time_ps: float,
) -> bool:
    """Whether MD and MSD fitting quality support a reportable diffusion coefficient."""
    return (
        mechanical_stability != "Collapsed"
        and md_converged
        and "msd_non_monotonic" not in flags
        and np.isfinite(diffusion_slope_a2_ps)
        and diffusion_slope_a2_ps > 0
        and np.isfinite(diffusion_fit_r2)
        and diffusion_fit_r2 >= MIN_DIFFUSION_REPORT_R2
        and md_simulation_time_sufficient(
            simulation_time_ps, diffusion_fit_r2, mechanical_stability
        )
    )


def diffusion_is_reliable(result: MOFResult) -> bool:
    return diffusion_reliability_criteria_met(
        mechanical_stability=result.mechanical_stability,
        md_converged=result.md_converged,
        flags=result.flags,
        diffusion_slope_a2_ps=result.diffusion_slope_a2_ps,
        diffusion_fit_r2=result.diffusion_fit_r2,
        simulation_time_ps=result.simulation_time_ps,
    ) and np.isfinite(result.diffusion_cm2_s)


def screening_threshold_status(result: MOFResult) -> dict[str, bool]:
    status = {
        "adsorption": (
            np.isfinite(result.adsorption_energy_ev)
            and CANDIDATE_ADS_MIN_EV <= result.adsorption_energy_ev <= CANDIDATE_ADS_MAX_EV
        ),
        "stability": (
            np.isfinite(result.mechanical_stability_index)
            and result.mechanical_stability_index >= CANDIDATE_MSI_MIN
            and result.mechanical_stability != "Collapsed"
        ),
        "msd": np.isfinite(result.msd_a2) and result.msd_a2 >= CANDIDATE_MSD_MIN_A2,
    }
    if not THREE_PARAMETER_SCREENING:
        status["diffusion"] = (
            np.isfinite(result.diffusion_cm2_s)
            and CANDIDATE_DIFFUSION_MIN_CM2_S <= result.diffusion_cm2_s <= CANDIDATE_DIFFUSION_MAX_CM2_S
        )
        status["volume"] = (
            np.isfinite(result.volume_change_pct)
            and abs(result.volume_change_pct) <= CANDIDATE_VOLUME_CHANGE_MAX_PCT
        )
    return status


def is_inbetween_thresholds(result: MOFResult) -> bool:
    return all(screening_threshold_status(result).values())


def explain_diffusion_unreliability(result: MOFResult) -> str:
    parts: list[str] = []
    if result.mechanical_stability == "Collapsed":
        parts.append("collapsed framework")
    if "high_energy_drift" in result.flags:
        parts.append("high MD energy drift")
    if "high_md_force" in result.flags:
        parts.append("high MD forces")
    if "high_temperature_std" in result.flags:
        parts.append("unstable temperature")
    if not result.md_converged and "MD not converged" not in parts:
        parts.append("MD not converged")
    if result.simulation_time_ps < DIFFUSION_SUFFICIENT_SIM_TIME_PS:
        parts.append(f"short trajectory ({result.simulation_time_ps:.1f} ps)")
    if np.isfinite(result.diffusion_fit_r2) and result.diffusion_fit_r2 < MIN_DIFFUSION_REPORT_R2:
        parts.append(f"low MSD fit R² ({result.diffusion_fit_r2:.2f})")
    elif not np.isfinite(result.diffusion_fit_r2):
        parts.append("no reliable MSD linear fit")
    if "msd_non_monotonic" in result.flags:
        parts.append("non-monotonic MSD")
    if not parts:
        parts.append("diffusion suppressed by reliability checks")
    return "; ".join(parts)


def recommend_mof(result: MOFResult) -> tuple[str, str]:
    """
    Multi-objective recommendation from continuous Overall Score (0–100).

    Soft preference windows still inform the score, but a MOF is no longer
    hard-rejected for barely missing one threshold. Collapse / overpacking
    remain hard disqualifiers because they invalidate the physical model.
    """
    if result.status in ("Failed", "Pending"):
        return RECOMMENDATION_NOT, "Simulation or analysis did not complete successfully."

    # Hard scientific disqualifiers
    hard_fail: list[str] = []
    if result.mechanical_stability == "Collapsed":
        hard_fail.append("framework collapsed during MD")
    if "k_too_close" in result.flags:
        hard_fail.append(f"K–K spacing below {MIN_KK:.2f} Å")
    if hard_fail:
        return RECOMMENDATION_NOT, "; ".join(hard_fail).capitalize() + "."

    score = float(result.overall_score) if np.isfinite(result.overall_score) else float("nan")
    if not np.isfinite(score):
        score = float(compute_overall_score(result))
        result.overall_score = score
        result.screening_score = score

    # Build explanatory notes from soft windows (not hard rejects)
    notes: list[str] = []
    checks = screening_threshold_status(result)
    if not checks["adsorption"] and np.isfinite(result.adsorption_energy_ev):
        notes.append(f"adsorption {result.adsorption_energy_ev:.2f} eV/K outside preferred window")
    if not checks["stability"] and np.isfinite(result.mechanical_stability_index):
        notes.append(f"MSI {result.mechanical_stability_index:.2f} below preferred {CANDIDATE_MSI_MIN:.1f}")
    if not checks["msd"] and np.isfinite(result.msd_a2):
        notes.append(f"MSD {result.msd_a2:.2f} Å² below preferred {CANDIDATE_MSD_MIN_A2:.0f} Å²")
    if not THREE_PARAMETER_SCREENING:
        if not checks.get("diffusion", True) and np.isfinite(result.diffusion_cm2_s):
            notes.append(f"diffusion {result.diffusion_cm2_s:.2e} cm²/s outside preferred window")
        if not checks.get("volume", True) and np.isfinite(result.volume_change_pct):
            notes.append(f"|ΔV| {abs(result.volume_change_pct):.2f}% above preferred {CANDIDATE_VOLUME_CHANGE_MAX_PCT:.1f}%")
        if not np.isfinite(result.diffusion_cm2_s):
            notes.append("diffusion coefficient unavailable")

    if score >= OVERALL_SCORE_EXCELLENT:
        rec = RECOMMENDATION_EXCELLENT
    elif score >= OVERALL_SCORE_VERY_GOOD:
        rec = RECOMMENDATION_VERY_GOOD
    elif score >= OVERALL_SCORE_GOOD:
        rec = RECOMMENDATION_GOOD
    elif score >= OVERALL_SCORE_VALIDATE:
        rec = RECOMMENDATION_VALIDATE
    elif score >= OVERALL_SCORE_POOR:
        rec = RECOMMENDATION_POOR
    else:
        rec = RECOMMENDATION_NOT

    reason = f"Overall score {score:.1f}/100"
    if notes:
        reason += "; " + "; ".join(notes)
    reason += "."
    return rec, reason


def _normalize_adsorption_score(ads_ev: float) -> float:
    """Gaussian preference around target adsorption energy → [0,1]."""
    if not np.isfinite(ads_ev):
        return 0.0
    return float(np.exp(-0.5 * ((ads_ev - ADSORPTION_TARGET_EV_PER_K) / ADSORPTION_SIGMA_EV) ** 2))


def _normalize_diffusion_score(d_cm2_s: float) -> float:
    """
    Map D onto [0,1] with peak inside the battery-relevant window.

    Uses log10(D) so decades of transport are treated evenly (standard for
    ionic diffusivity spanning 10⁻¹⁴–10⁻⁶ cm²/s).
    """
    if not np.isfinite(d_cm2_s) or d_cm2_s <= 0:
        return 0.0
    log_d = np.log10(d_cm2_s)
    log_lo = np.log10(CANDIDATE_DIFFUSION_MIN_CM2_S)
    log_hi = np.log10(CANDIDATE_DIFFUSION_MAX_CM2_S)
    log_mid = 0.5 * (log_lo + log_hi)
    # Triangular / tent score peaked at window center, soft outside
    half = max(log_hi - log_lo, 1e-6)
    score = 1.0 - abs(log_d - log_mid) / half
    return float(np.clip(score, 0.0, 1.0))


def _normalize_msd_score(msd_a2: float) -> float:
    if not np.isfinite(msd_a2) or msd_a2 < 0:
        return 0.0
    # Saturating score: MSD ≥ preferred minimum → 1
    return float(np.clip(msd_a2 / max(CANDIDATE_MSD_MIN_A2, 1e-6), 0.0, 1.0))


def _normalize_volume_score(dV_pct: float) -> float:
    if not np.isfinite(dV_pct):
        return 0.5
    # Prefer small |ΔV|; zero change → 1, at max gate → 0
    return float(np.clip(1.0 - abs(dV_pct) / max(CANDIDATE_VOLUME_CHANGE_MAX_PCT, 1e-6), 0.0, 1.0))


def compute_overall_score(result: MOFResult) -> float:
    """
    Weighted multi-objective overall score in [0, 100].

    All inputs are normalized to [0,1] before weighting (WEIGHTS). Missing
    metrics contribute 0 rather than aborting the ranking, so partial results
    remain comparable across the screening set.
    """
    comps: dict[str, float] = {
        "adsorption": _normalize_adsorption_score(result.adsorption_energy_ev),
        "stability": float(
            np.clip(
                result.mechanical_stability_index
                if np.isfinite(result.mechanical_stability_index)
                else 0.0,
                0.0,
                1.0,
            )
        ),
        "msd": _normalize_msd_score(result.msd_a2),
        "confidence": float(
            np.clip(
                (result.confidence_score / 100.0)
                if np.isfinite(result.confidence_score)
                else 0.0,
                0.0,
                1.0,
            )
        ),
    }
    if not THREE_PARAMETER_SCREENING:
        comps["diffusion"] = _normalize_diffusion_score(result.diffusion_cm2_s)
        comps["volume"] = _normalize_volume_score(result.volume_change_pct)
    total_w = sum(WEIGHTS.get(k, 0.0) for k in comps)
    if total_w <= 0:
        return 0.0
    score = sum(WEIGHTS.get(k, 0.0) * comps[k] for k in comps) / total_w
    return float(np.clip(100.0 * score, 0.0, 100.0))


def classify_candidate_assessment(result: MOFResult) -> str:
    """Legacy Good/Bad label derived from Recommendation."""
    recommendation, _ = recommend_mof(result)
    return "Good" if recommendation in {
        RECOMMENDATION_EXCELLENT,
        RECOMMENDATION_VERY_GOOD,
        RECOMMENDATION_GOOD,
    } else "Bad"


def mechanical_stability_index(
    framework_rmsd: float,
    max_disp: float,
    bond_frac: float,
    energy_fluct: float,
    volume_fluct_frac: float = float("nan"),
    coordination_frac: float = float("nan"),
) -> float:
    """
    Mechanical Stability Index in [0, 1].

    Components (see MSI_WEIGHTS for justification):
      RMSD, max displacement, bond preservation, energy fluctuation,
      and relative volume fluctuation during MD.
      
    Weights are dynamically renormalized over the available (finite) metrics.
    """
    w = MSI_WEIGHTS
    numerator = 0.0
    total_w = 0.0

    if np.isfinite(framework_rmsd):
        rmsd_score = float(np.clip(1.0 - (framework_rmsd / STABILITY_RMSD_MARGINAL_A), 0.0, 1.0))
        numerator += w.get("rmsd", 0.0) * rmsd_score
        total_w += w.get("rmsd", 0.0)

    if np.isfinite(max_disp):
        disp_score = float(np.clip(1.0 - (max_disp / STABILITY_MAX_DISP_MARGINAL_A), 0.0, 1.0))
        numerator += w.get("displacement", 0.0) * disp_score
        total_w += w.get("displacement", 0.0)

    if np.isfinite(bond_frac):
        bond_score = float(np.clip(bond_frac, 0.0, 1.0))
        numerator += w.get("bond", 0.0) * bond_score
        total_w += w.get("bond", 0.0)

    if np.isfinite(energy_fluct):
        energy_score = float(np.clip(1.0 - (energy_fluct / MSI_MAX_ENERGY_FLUCT_EV_PER_ATOM), 0.0, 1.0))
        numerator += w.get("energy", 0.0) * energy_score
        total_w += w.get("energy", 0.0)

    if np.isfinite(volume_fluct_frac):
        volume_score = float(np.clip(1.0 - (volume_fluct_frac / MSI_MAX_VOLUME_FLUCT_FRAC), 0.0, 1.0))
        numerator += w.get("volume", 0.0) * volume_score
        total_w += w.get("volume", 0.0)

    if np.isfinite(coordination_frac):
        coord_score = float(np.clip(coordination_frac, 0.0, 1.0))
        numerator += w.get("coordination", 0.0) * coord_score
        total_w += w.get("coordination", 0.0)

    if total_w <= 0:
        return 0.0
    return float(numerator / total_w)


def classify_mechanical_stability(msi: float) -> str:
    if not np.isfinite(msi):
        return "Collapsed"
    if msi >= MSI_STABLE_THRESHOLD:
        return "Stable"
    if msi >= MSI_MARGINAL_THRESHOLD:
        return "Marginal"
    return "Collapsed"


def analyze_md_trajectory(
    mof: str,
    traj_path: Path,
    log_path: Path | None,
    k_indices: list[int],
    framework_idx: list[int],
    reference_structure: Structure,
    symbols: list[str],
    time_ps: float,
    stage_label: str,
    skip_diffusion: bool = False,
) -> MDAnalysisResult:
    """Extract mechanical stability, MSD, and diffusion from an MD trajectory."""
    logger = get_logger()
    result = MDAnalysisResult(simulation_time_ps=time_ps)
    try:
        traj = Trajectory(str(traj_path))
        n_frames = len(traj)
        if n_frames < 2:
            result.error = "Trajectory too short"
            return result

        ref = traj[0]
        ref_pos = ref.get_positions()
        cell = ref.get_cell().array
        bond_pairs: list[tuple[int, int]] = []
        try:
            bond_pairs = build_bond_pairs_crystalnn(reference_structure, framework_idx)
            if not bond_pairs:
                raise ValueError("CrystalNN returned no framework bonds")
            logger.info("Bonding model for %s uses CrystalNN (%d bonds)", mof, len(bond_pairs))
        except Exception as exc:
            bond_pairs = build_bond_pairs(ref_pos, symbols, framework_idx)
            logger.warning(
                "CrystalNN bonding failed for %s (%s); fallback to covalent cutoff (%d bonds)",
                mof,
                exc,
                len(bond_pairs),
            )
        initial_dist: dict[tuple[int, int], float] = {}
        for i, j in bond_pairs:
            raw = ref_pos[j] - ref_pos[i]
            mic, _ = find_mic(raw, cell)
            initial_dist[(i, j)] = float(np.linalg.norm(mic))

        rmsd_vals: list[float] = []
        max_disp_vals: list[float] = []
        bond_fracs: list[float] = []

        all_pos = np.stack([atoms.get_positions() for atoms in traj], axis=0)
        unwrapped_positions = unwrap_positions_mic(all_pos, cell)

        all_cells = np.stack([atoms.get_cell().array for atoms in traj], axis=0)

        symbols_frame0 = ref.get_chemical_symbols()
        metal_indices = [
            i for i, symbol in enumerate(symbols_frame0)
            if symbol not in {ADSORBATE, "H", "C", "N", "O", "F", "S", "P"}
        ]
        ligand_indices = [
            i for i, symbol in enumerate(symbols_frame0)
            if symbol in {"O", "N"} and symbol != ADSORBATE
        ]
        half_start = max(n_frames // 2, 0)
        metal_bonds: set[tuple[int, int]] = set()
        for metal in metal_indices:
            for ligand in ligand_indices:
                d = _periodic_distance(
                    ref.get_scaled_positions()[metal],
                    ref.get_scaled_positions()[ligand],
                    cell,
                )
                if d <= 2.4:
                    metal_bonds.add((metal, ligand))
        result.n_metal_bonds_initial = len(metal_bonds)
        result.coordination_start = {
            str(metal): sum(1 for pair in metal_bonds if pair[0] == metal)
            for metal in metal_indices
        }
        broken = 0
        persistent_new = 0
        for metal in metal_indices:
            for ligand in ligand_indices:
                distances = []
                for frame_idx in range(half_start, n_frames):
                    delta = all_pos[frame_idx, ligand] - all_pos[frame_idx, metal]
                    mic, _ = find_mic(delta, all_cells[frame_idx])
                    distances.append(float(np.linalg.norm(mic)))
                if not distances:
                    continue
                fraction_long = sum(d > 2.6 for d in distances) / len(distances)
                pair = (metal, ligand)
                if pair in metal_bonds and fraction_long > 0.5:
                    broken += 1
                if pair not in metal_bonds and fraction_long <= 0.5 and sum(
                    d <= 2.4 for d in distances
                ) / len(distances) > 0.5:
                    persistent_new += 1
        result.n_metal_bonds_broken = broken
        result.n_metal_bonds_new = persistent_new
        result.coordination_end = {
            str(metal): sum(
                1
                for ligand in ligand_indices
                if np.linalg.norm(
                    find_mic(
                        all_pos[-1, ligand] - all_pos[-1, metal],
                        all_cells[-1],
                    )[0]
                ) <= 2.4
            )
            for metal in metal_indices
        }
        result.metal_carbon_contact = any(
            sum(
                np.linalg.norm(
                    find_mic(
                        all_pos[frame_idx, carbon] - all_pos[frame_idx, metal],
                        all_cells[frame_idx],
                    )[0]
                ) < 2.4
                for frame_idx in range(half_start, n_frames)
            ) > (n_frames - half_start) * 0.5
            for metal in metal_indices
            for carbon in [i for i, symbol in enumerate(symbols_frame0) if symbol == "C"]
        )

        for idx_frame in range(n_frames):
            pos_wrapped = all_pos[idx_frame]
            pos_unwrapped = unwrapped_positions[idx_frame]
            cell_t = all_cells[idx_frame]
            framework_displacements_cart = (
                pos_unwrapped[framework_idx] - ref_pos[framework_idx]
            )
            framework_displacements_cart -= np.mean(
                framework_displacements_cart, axis=0, keepdims=True
            )
            rmsd = float(np.sqrt(np.mean(np.sum(framework_displacements_cart**2, axis=1))))
            max_d = float(np.max(np.linalg.norm(framework_displacements_cart, axis=1)))
            rmsd_vals.append(rmsd)
            max_disp_vals.append(max_d)
            bond_fracs.append(
                bond_preservation_fraction(pos_wrapped, bond_pairs, initial_dist, cell_t)
            )

        result.framework_rmsd = float(np.mean(rmsd_vals))
        result.max_displacement = float(np.max(max_disp_vals))
        result.bond_preservation = float(np.mean(bond_fracs))

        try:
            volumes = np.abs(np.linalg.det(all_cells))
        except Exception:
            volumes = np.array([])
        if len(volumes) > 1:
            v_mean = float(np.mean(volumes))
            result.volume_fluctuation_frac = (
                float(np.std(volumes) / v_mean) if v_mean > 0 else float("nan")
            )

        temp_vals = np.array([], dtype=float)
        pot_vals = np.array([], dtype=float)
        total_vals = np.array([], dtype=float)
        if log_path and log_path.is_file():
            temp_vals, pot_vals, total_vals = parse_md_log(log_path)
            if len(pot_vals) > 1:
                prod = pot_vals[int(len(pot_vals) * MD_EQUILIBRATION_FRACTION) :]
                prod = prod[np.isfinite(prod)]
                n_atoms = max(len(ref), 1)
                if len(prod) > 1:
                    energy_per_atom = prod / n_atoms
                    result.energy_fluctuation = float(np.std(energy_per_atom))
                t_axis = np.linspace(0.0, time_ps, len(total_vals))
                finite = np.isfinite(total_vals)
                if finite.sum() > 2 and len(total_vals) == len(t_axis):
                    slope, _, _, _, _ = linregress(t_axis[finite], total_vals[finite])
                    result.energy_drift_ev_ps = float(slope)
                    result.energy_drift_ev_per_atom_ps = float(slope / max(len(ref), 1))
            if len(temp_vals) > 1:
                tv = temp_vals[np.isfinite(temp_vals)]
                if len(tv) > 1:
                    result.temperature_mean = float(np.mean(tv))
                    result.temperature_std = float(np.std(tv))

        max_force_md = 0.0
        force_frames = 0
        for atoms in traj:
            try:
                forces = atoms.get_forces()
            except Exception:
                continue
            force_mag = np.linalg.norm(np.asarray(forces), axis=1)
            if len(force_mag):
                max_force_md = max(max_force_md, float(np.max(force_mag)))
                force_frames += 1
        if force_frames > 0:
            result.max_force_md = max_force_md
        else:
            result.max_force_md = float("nan")
        try:
            initial_forces = np.asarray(traj[0].get_forces(apply_constraint=False))
            result.initial_max_force = float(
                np.max(np.linalg.norm(initial_forces, axis=1))
            )
        except Exception:
            result.initial_max_force = float("nan")

        traj_positions = [all_pos[i] for i in range(n_frames)]
        traj_cells = [all_cells[i] for i in range(n_frames)]
        result.coordination_preservation = coordination_preservation_fraction(
            reference_structure, traj_positions, k_indices, cells=traj_cells
        )

        result.mechanical_stability_index = mechanical_stability_index(
            result.framework_rmsd,
            result.max_displacement,
            result.bond_preservation,
            result.energy_fluctuation,
            result.volume_fluctuation_frac,
            result.coordination_preservation,
        )
        result.mechanical_stability = classify_mechanical_stability(result.mechanical_stability_index)
        if result.mechanical_stability == "Collapsed":
            logger.warning(
                "Mechanical Stability marked as Collapsed for %s because MSI %.3f is below %.3f "
                "(RMSD %.3f Å, bond preservation %.1f%%, max displacement %.3f Å).",
                mof,
                result.mechanical_stability_index,
                MSI_MARGINAL_THRESHOLD,
                result.framework_rmsd,
                100.0 * result.bond_preservation if np.isfinite(result.bond_preservation) else float("nan"),
                result.max_displacement,
            )

        result.md_converged = True
        if (
            np.isfinite(result.temperature_mean)
            and abs(result.temperature_mean - TEMPERATURE_K) > 30.0
        ) or (
            np.isfinite(result.temperature_std) and result.temperature_std > 50.0
        ):
            result.md_converged = False
            result.flagged.append("md_unstable_temperature")
            logger.warning("MD warning for %s: mean/std temperature = %.2f/%.2f K", mof, result.temperature_mean, result.temperature_std)
        if np.isfinite(result.energy_drift_ev_per_atom_ps) and abs(result.energy_drift_ev_per_atom_ps) > MAX_MD_ENERGY_DRIFT_EV_PS:
            result.md_converged = False
            result.flagged.append("md_energy_drift")
            logger.warning("MD warning for %s: high energy drift = %.4f eV/atom/ps", mof, result.energy_drift_ev_per_atom_ps)
        if np.isfinite(result.initial_max_force) and result.initial_max_force > 0.1:
            result.md_converged = False
            result.flagged.append("md_bad_start")
            logger.warning("MD warning for %s: initial maximum force = %.3f eV/Å", mof, result.initial_max_force)
        if result.n_metal_bonds_broken or result.n_metal_bonds_new:
            result.flagged.append("node_rearrangement")
        if result.metal_carbon_contact:
            result.flagged.append("metal_carbon_contact")

        msd_lab = compute_k_msd_unwrapped_trajectory(traj, k_indices)
        time_axis = np.linspace(0.0, time_ps, n_frames)
        result.msd_a2 = float(msd_lab[-1]) if len(msd_lab) else float("nan")
        if len(msd_lab) > 2 and np.any(np.diff(msd_lab[1:]) < -0.01):
            result.flagged.append("msd_non_monotonic")

        fit: dict[str, float | str] = {"intercept_a2": float("nan")}
        msd_for_plot = msd_lab
        if not skip_diffusion:
            msd_fw = compute_k_msd_unwrapped_trajectory(traj, k_indices, framework_idx)
            msd_for_plot = msd_fw
            if md_diffusion_tier1_allowed(stage_label, time_ps):
                eq_series = pot_vals if len(pot_vals) == n_frames else None
                md_eval = evaluate_md_diffusion(
                    msd_fw,
                    time_axis,
                    mechanical_stability=result.mechanical_stability,
                    md_converged=result.md_converged,
                    flags=result.flagged,
                    msd_lab_final=result.msd_a2,
                    simulation_time_ps=time_ps,
                    equilibration_series=eq_series,
                    logger=logger,
                )
                fit = md_eval
                result.diffusion_cm2_s = float(md_eval.get("diffusion_cm2_s", float("nan")))
                result.diffusion_slope_a2_ps = float(md_eval.get("slope_a2_ps", float("nan")))
                result.diffusion_fit_r2 = float(md_eval.get("fit_r2", float("nan")))
                result.diffusion_fit_start = int(md_eval.get("start_frame", -1))
                result.diffusion_fit_end = int(md_eval.get("end_frame", -1))
                result.diffusion_stderr_cm2_s = float(
                    md_eval.get("diffusion_stderr_cm2_s", float("nan"))
                )
                result.diffusion_ci_lo_cm2_s = float(
                    md_eval.get("diffusion_cm2_s_lo", float("nan"))
                )
                result.diffusion_ci_hi_cm2_s = float(
                    md_eval.get("diffusion_cm2_s_hi", float("nan"))
                )
                result.slope_stderr_a2_ps = float(md_eval.get("slope_stderr", float("nan")))
                result.slope_pvalue = float(md_eval.get("slope_pvalue", float("nan")))
                result.equilibration_frame = int(md_eval.get("equilibration_frame", -1))
                result.equilibration_time_ps = float(
                    md_eval.get("equilibration_time_ps", float("nan"))
                )
                result.diffusion_reliability = str(md_eval.get("diffusion_reliability", "Very Low"))
                rejection = str(md_eval.get("rejection_reason", ""))
                if rejection:
                    result.diffusion_warning = f"MD diffusion rejected: {rejection}."
                    logger.info("MD diffusion rejected for %s: %s", mof, rejection)
                elif result.diffusion_reliability in {"Low", "Very Low"}:
                    result.diffusion_warning = (
                        f"Diffusion coefficient computed from a {time_ps:.1f} ps trajectory. "
                        f"Linear fit R² = {result.diffusion_fit_r2:.2f}. "
                        "Value reported for comparison but should be interpreted with caution."
                    )
                    logger.warning("%s [%s]", result.diffusion_warning, mof)
                    result.flagged.append("diffusion_not_reliable")
                elif result.diffusion_reliability == "Moderate":
                    result.diffusion_warning = (
                        f"Diffusion fit quality is moderate (R² = {result.diffusion_fit_r2:.2f}, "
                        f"{time_ps:.1f} ps trajectory)."
                    )
                if not np.isfinite(result.diffusion_cm2_s):
                    if not result.diffusion_warning:
                        result.diffusion_warning = (
                            "Diffusion coefficient unavailable because MD reliability gates were not met."
                        )
                    logger.warning("%s [%s]", result.diffusion_warning, mof)

                apply_diffusion_reporting_gate(result, time_ps)
                result.diffusion_fit_num_frames = max(
                    0, result.diffusion_fit_end - result.diffusion_fit_start + 1
                )
            else:
                if stage_label == "stage1":
                    logger.info(
                        "Stage 1 fast MD for %s: Tier-1 MD diffusion skipped — "
                        "rare-event K hopping not sampled in %.1f ps.",
                        mof,
                        time_ps,
                    )
                else:
                    logger.info(
                        "Tier-1 MD diffusion skipped for %s — trajectory %.1f ps < %.1f ps minimum.",
                        mof,
                        time_ps,
                        DIFFUSION_SUFFICIENT_SIM_TIME_PS,
                    )
                result.flagged.append("diffusion_not_reliable")
                if not result.diffusion_warning:
                    result.diffusion_warning = (
                        "MD Tier-1 diffusion skipped; use NEB+TST/BVSE or Stage-2 production MD."
                    )

        result.success = True
        if skip_diffusion:
            logger.info(
                "MD analysis: stability=%s (MSI=%.3f), RMSD=%.3f A, MSD(K)=%.3f A^2",
                result.mechanical_stability,
                result.mechanical_stability_index,
                result.framework_rmsd,
                result.msd_a2,
            )
        else:
            logger.info(
                "MD analysis: stability=%s (MSI=%.3f), RMSD=%.3f A, MSD(K)=%.3f A^2, D=%.3e cm^2/s",
                result.mechanical_stability,
                result.mechanical_stability_index,
                result.framework_rmsd,
                result.msd_a2,
                result.diffusion_cm2_s if np.isfinite(result.diffusion_cm2_s) else float("nan"),
            )
        if not skip_diffusion and np.isfinite(result.diffusion_fit_r2):
            logger.info(
                "Diffusion linear fit for %s: slope=%.4f Å²/ps, R²=%.3f, frames=%d-%d",
                mof,
                result.diffusion_slope_a2_ps,
                result.diffusion_fit_r2,
                result.diffusion_fit_start,
                result.diffusion_fit_end,
            )
        try:
            save_md_diagnostics(
                mof=mof,
                stage_label=stage_label,
                time_ps=time_axis,
                msd=msd_for_plot,
                temperature_k=temp_vals,
                energy_ev=pot_vals,
                fit_start=result.diffusion_fit_start,
                fit_end=result.diffusion_fit_end,
                fit_slope_a2_ps=result.diffusion_slope_a2_ps,
                fit_intercept_a2=float(fit.get("intercept_a2", float("nan"))),
            )
        except Exception as exc:
            logger.warning("Plot generation failed for %s: %s", mof, exc)
    except Exception as exc:
        result.error = str(exc)
        logger.error("MD analysis failed: %s", exc)
    return result


def validate_result(result: MOFResult) -> list[str]:
    """Flag suspicious outputs without altering computed values."""
    flags: list[str] = []
    if np.isfinite(result.adsorption_energy_ev) and result.adsorption_energy_ev > 0:
        flags.append("positive_adsorption_energy")
    if np.isfinite(result.adsorption_energy_ev):
        if result.adsorption_energy_ev > 0.5:
            flags.append("adsorption_extreme_positive")
        if result.adsorption_energy_ev < -5.0:
            flags.append("adsorption_extreme_negative")
    if np.isfinite(result.diffusion_cm2_s) and result.diffusion_cm2_s < 0:
        flags.append("negative_diffusion")
    if result.mechanical_stability == "Stable" and np.isfinite(result.mechanical_stability_index):
        if result.mechanical_stability_index < MSI_STABLE_THRESHOLD:
            flags.append("stability_msi_mismatch")
    if result.mechanical_stability == "Collapsed" and np.isfinite(result.mechanical_stability_index):
        if result.mechanical_stability_index >= MSI_MARGINAL_THRESHOLD:
            flags.append("stability_msi_mismatch")
    if not np.isfinite(result.msd_a2):
        flags.append("missing_msd")
    if not THREE_PARAMETER_SCREENING:
        if not np.isfinite(result.diffusion_cm2_s):
            flags.append("missing_diffusion")
        else:
            if not diffusion_is_reliable(result):
                flags.append("diffusion_not_reliable")
            if np.isfinite(result.diffusion_fit_r2) and result.diffusion_fit_r2 < MIN_DIFFUSION_REPORT_R2:
                flags.append("low_r2")
            sim_time = (
                result.diffusion_simulation_duration_ps
                if np.isfinite(result.diffusion_simulation_duration_ps)
                else result.simulation_time_ps
            )
            if np.isfinite(sim_time) and sim_time < DIFFUSION_SUFFICIENT_SIM_TIME_PS:
                flags.append("short_trajectory")
        if not np.isfinite(result.diffusion_fit_r2):
            flags.append("missing_diffusion_fit_r2")
    if not np.isfinite(result.adsorption_energy_ev):
        flags.append("missing_adsorption")
    if not np.isfinite(result.volume_change_pct):
        flags.append("missing_volume_change")
    elif abs(result.volume_change_pct) > CANDIDATE_VOLUME_CHANGE_MAX_PCT:
        flags.append("large_volume_change")
    elif abs(result.volume_change_pct) > VOLUME_CHANGE_WARN_PCT:
        flags.append("moderate_volume_change")
    return flags


def log_flag_reasons(mof: str, flags: list[str], logger: logging.Logger) -> None:
    explanations = {
        "positive_adsorption_energy": "adsorption energy is positive (weak insertion driving force).",
        "adsorption_extreme_positive": "adsorption energy is extremely positive and suspicious.",
        "adsorption_extreme_negative": "adsorption energy is extremely negative and suspicious.",
        "negative_diffusion": "diffusion coefficient is negative (non-physical).",
        "stability_msi_mismatch": "stability label and MSI threshold are inconsistent.",
        "missing_msd": "MSD could not be computed for inserted K trajectory.",
        "missing_diffusion": "diffusion coefficient unavailable due to non-linear MSD regime.",
        "missing_diffusion_fit_r2": "diffusion linear fit R² unavailable.",
        "missing_adsorption": "adsorption energy not computed.",
        "md_bad_start": "initial-frame maximum force exceeds 0.1 eV/Å.",
        "md_unstable_temperature": "production temperature mean or variance is outside tolerance.",
        "md_energy_drift": "per-atom production energy drift exceeds tolerance.",
        "node_rearrangement": "persistent metal-node bond rearrangement detected.",
        "metal_carbon_contact": "persistent metal–carbon contact detected.",
        "diffusion_not_reliable": "MSD linear fit did not satisfy reliability threshold.",
        "msd_non_monotonic": "MSD contains pronounced non-monotonic behavior.",
        "low_r2": "diffusion coefficient fit R² is below the reporting threshold.",
        "short_trajectory": "MD simulation trajectory is shorter than the preferred duration.",
        "missing_volume_change": "cell volume change could not be computed.",
        "moderate_volume_change": "cell volume change is moderately large upon K insertion.",
        "large_volume_change": "cell volume change exceeds the screening threshold upon K insertion.",
        "k_too_close": "inserted K ions are closer than the minimum K–K spacing.",
    }
    for flag in flags:
        why = explanations.get(flag, "flagged by internal validation checks.")
        logger.warning("Validation flag for %s: %s -> %s", mof, flag, why)


def normalize_series(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    valid = values.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=values.index)
    vmin, vmax = valid.min(), valid.max()
    if np.isclose(vmin, vmax):
        norm = pd.Series(0.5, index=values.index)
    else:
        norm = (values - vmin) / (vmax - vmin)
    if not higher_is_better:
        norm = 1.0 - norm
    return norm.clip(0.0, 1.0)


def adsorption_score(values: pd.Series, n_k_map: dict[str, int]) -> pd.Series:
    """Reward target adsorption window and penalize weak/over-strong binding (eV/K)."""
    scores = []
    for mof, per_k in values.items():
        if not np.isfinite(per_k):
            scores.append(np.nan)
            continue
        if ADSORPTION_TARGET_MIN_EV_PER_K <= per_k <= ADSORPTION_TARGET_MAX_EV_PER_K:
            center = 0.5 * (ADSORPTION_TARGET_MIN_EV_PER_K + ADSORPTION_TARGET_MAX_EV_PER_K)
            z = (per_k - center) / ADSORPTION_SIGMA_EV
            score = float(np.exp(-0.5 * z * z))
        elif per_k > ADSORPTION_TARGET_MAX_EV_PER_K:
            delta = per_k - ADSORPTION_TARGET_MAX_EV_PER_K
            score = float(np.exp(-(delta / max(ADSORPTION_SIGMA_EV, 1e-6))))
        else:
            delta = ADSORPTION_TARGET_MIN_EV_PER_K - per_k
            score = float(np.exp(-(delta / max(ADSORPTION_SIGMA_EV, 1e-6))))
        scores.append(float(np.clip(score, 0.0, 1.0)))
    return pd.Series(scores, index=values.index)


def compute_screening_scores(
    results: list[MOFResult],
    n_k_map: dict[str, int],
) -> pd.DataFrame:
    """
    Rank MOFs by multi-objective Overall Score.

    Uses per-MOF compute_overall_score (normalized metrics × WEIGHTS) so
    ranking is consistent with recommend_mof / best_cif export.
    """
    rows = []
    for r in results:
        if not np.isfinite(r.overall_score):
            r.overall_score = compute_overall_score(r)
        r.screening_score = r.overall_score
        rows.append(
            {
                "MOF": r.mof,
                "Adsorption_Energy_eV": r.adsorption_energy_ev,
                "Mechanical_Stability": (
                    r.mechanical_stability
                    or classify_mechanical_stability(r.mechanical_stability_index)
                ),
                "Mechanical_Stability_Value": r.mechanical_stability_index,
                "MSD_A2": r.msd_a2,
                "Diffusion_Coefficient_cm2_s": r.diffusion_cm2_s,
                "Volume_Change_pct": r.volume_change_pct,
                "Confidence_Score": r.confidence_score,
                "Uncertainty_Flag": r.uncertainty_flag,
                "Overall_Score": r.overall_score,
                "N_K": n_k_map.get(r.mof, 0),
            }
        )
    df = pd.DataFrame(rows).set_index("MOF")
    df["Screening_Score"] = df["Overall_Score"]
    return df


def recommendation_from_score(score: float, stability: str) -> tuple[str, str]:
    """Map continuous score to recommendation labels (legacy helper)."""
    if stability == "Collapsed":
        return RECOMMENDATION_NOT, "Framework collapsed during MD; unsuitable despite other metrics."
    if not np.isfinite(score):
        return RECOMMENDATION_POOR, "Insufficient data for reliable screening."
    dummy = MOFResult(mof="tmp", overall_score=score, mechanical_stability=stability, status="Complete")
    return recommend_mof(dummy)


def print_mof_progress(
    index: int,
    total: int,
    result: MOFResult,
) -> None:
    print("\n" + "=" * 36)
    print(f"Processing {index} / {total}")
    print(f"\nMOF : {result.mof}")
    print(f"\nOptimization : {'Complete' if np.isfinite(result.optimization_energy) else 'Failed'}")
    if np.isfinite(result.adsorption_energy_ev):
        site = f" [{result.adsorption_site_type}]" if result.adsorption_site_type else ""
        print(f"\nAdsorption : {result.adsorption_energy_ev:.2f} eV/K{site}")
    print(f"\nFast MD : {'Complete' if result.stage >= 1 else 'Pending'}")
    if result.mechanical_stability:
        print(f"\nMechanical Stability : {result.mechanical_stability}")
    if np.isfinite(result.mechanical_stability_index):
        print(f"\nMechanical Stability Index : {result.mechanical_stability_index:.3f}")
    if np.isfinite(result.msd_a2):
        print(f"\nMSD(K) : {result.msd_a2:.2f} Å²")
    if np.isfinite(result.diffusion_cm2_s):
        print(f"\nDiffusion : {result.diffusion_cm2_s:.2e} cm²/s")
        print(f"\nDiffusion Reliability : {result.diffusion_reliability}")
    if np.isfinite(result.volume_change_pct):
        print(f"\nVolume Change : {result.volume_change_pct:+.2f} %")
    if np.isfinite(result.overall_score):
        print(f"\nOverall Score : {result.overall_score:.1f}/100")
    elif np.isfinite(result.screening_score):
        print(f"\nCurrent Screening Score : {result.screening_score:.1f}")
    if np.isfinite(result.confidence_score):
        print(f"\nConfidence Score : {result.confidence_score:.1f} ({result.uncertainty_flag})")
    print("\n" + "=" * 36)


def write_results_csv(results: list[MOFResult]) -> None:
    df = pd.DataFrame([r.to_results_row() for r in results], columns=MOF_PARAM_COLUMNS)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "results.csv", index=False)


def passing_results(results: list[MOFResult]) -> list[MOFResult]:
    """MOFs with Good / Very Good / Excellent recommendation."""
    good_labels = {RECOMMENDATION_EXCELLENT, RECOMMENDATION_VERY_GOOD, RECOMMENDATION_GOOD}
    return [r for r in results if recommend_mof(r)[0] in good_labels]


def write_best_cif_csv(
    results: list[MOFResult],
    n_k_map: dict[str, int],
    top_n: int | None = None,
) -> int:
    """
    Write best_cif.csv containing only results inbetween the thresholds, ranked by Overall Score.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cif_dir = OUTPUT_DIR / "best_cifs"
    cif_dir.mkdir(parents=True, exist_ok=True)

    if not results:
        pd.DataFrame(columns=BEST_CIF_COLUMNS).to_csv(OUTPUT_DIR / "best_cif.csv", index=False)
        return 0

    # Ensure scores are populated
    for r in results:
        if not np.isfinite(r.confidence_score):
            r.confidence_score = confidence_score(r, opt_ok=True, md_ok=r.md_converged)
        if not np.isfinite(r.overall_score):
            r.overall_score = compute_overall_score(r)
        r.screening_score = r.overall_score

    # Filter results that are inbetween the thresholds
    selected = [r for r in results if is_inbetween_thresholds(r)]

    # Rank selected results by overall score
    selected = sorted(
        selected,
        key=lambda r: (
            r.overall_score if np.isfinite(r.overall_score) else float("-inf"),
            r.confidence_score if np.isfinite(r.confidence_score) else float("-inf"),
        ),
        reverse=True,
    )

    rows: list[dict[str, Any]] = []
    for result in selected:
        cif_name = f"{result.mof}.cif"
        work = mof_work_dir(result.mof)
        # Prefer globally best adsorption geometry when available
        k_src = work / "k_on_bare_optimized.cif"
        if not k_src.is_file():
            k_src = work / "k_inserted_optimized.cif"
        if k_src.is_file():
            shutil.copy2(k_src, cif_dir / cif_name)
        params = result.to_results_row()
        rows.append(
            {
                "best_cif": cif_name,
                "adsorption_energy": params["adsorption_energy"],
                "msd": params["msd"],
                "mech_stabb": params["mech_stabb"],
            }
        )

    pd.DataFrame(rows, columns=BEST_CIF_COLUMNS).to_csv(OUTPUT_DIR / "best_cif.csv", index=False)
    get_logger().info(
        "Wrote %d threshold-passing MOFs to %s/best_cif.csv",
        len(rows),
        OUTPUT_DIR,
    )
    return len(rows)


def select_best_result(
    results: list[MOFResult],
    n_k_map: dict[str, int],
) -> MOFResult | None:
    if not results:
        return None
    for r in results:
        if not np.isfinite(r.overall_score):
            r.overall_score = compute_overall_score(r)
    return max(
        results,
        key=lambda r: (
            r.overall_score if np.isfinite(r.overall_score) else float("-inf"),
            r.confidence_score if np.isfinite(r.confidence_score) else float("-inf"),
        ),
    )


def export_best_mof_cifs(best: MOFResult) -> None:
    """Copy optimized bare and K-inserted CIFs for the best MOF into outputs/."""
    work = mof_work_dir(best.mof)
    bare_src = work / "bare_optimized.cif"
    k_src = work / "k_inserted_optimized.cif"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if bare_src.is_file():
        shutil.copy2(bare_src, OUTPUT_DIR / "best_mof_bare.cif")
    if k_src.is_file():
        shutil.copy2(k_src, OUTPUT_DIR / "best_mof_k_inserted.cif")


def write_best_mof_outputs(results: list[MOFResult], n_k_map: dict[str, int]) -> MOFResult | None:
    n_passed = write_best_cif_csv(results, n_k_map)
    best = select_best_result(results, n_k_map)
    if best is None:
        return None
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_df = pd.DataFrame([best.to_results_row()], columns=MOF_PARAM_COLUMNS)
    best_df.to_csv(OUTPUT_DIR / "best_mof.csv", index=False)
    export_best_mof_cifs(best)
    get_logger().info(
        "Threshold-passing MOFs: %d written to %s/best_cif.csv",
        n_passed,
        OUTPUT_DIR,
    )
    return best


def write_ranked_csv(
    results: list[MOFResult],
    n_k_map: dict[str, int],
) -> pd.DataFrame:
    """Internal ranking table (not written to disk)."""
    return compute_screening_scores(results, n_k_map).reset_index()


def process_mof_stage2(
    mof: str,
    bare_path: Path,
    k_path: Path,
    md_time_ps: float,
    n_k: int,
    prior: MOFResult,
) -> MOFResult:
    """Stage 2: long MD only; retain Stage 1 adsorption and optimization energies."""
    logger = get_logger()
    t0 = time.perf_counter()
    result = MOFResult(
        mof=mof,
        adsorption_energy_ev=prior.adsorption_energy_ev,
        adsorption_binding_class=prior.adsorption_binding_class,
        adsorption_site_type=getattr(prior, "adsorption_site_type", ""),
        adsorption_reference=getattr(prior, "adsorption_reference", ADSORPTION_REFERENCE),
        adsorption_k_o_distance_a=getattr(prior, "adsorption_k_o_distance_a", float("nan")),
        adsorption_site_rank=getattr(prior, "adsorption_site_rank", -1),
        optimization_energy=prior.optimization_energy,
        optimization_converged=getattr(prior, "optimization_converged", False),
        volume_bare_A3=prior.volume_bare_A3,
        volume_k_A3=prior.volume_k_A3,
        volume_change_pct=prior.volume_change_pct,
    )
    if not result.adsorption_binding_class:
        result.adsorption_binding_class = classify_adsorption(result.adsorption_energy_ev)
    try:
        log_device_for_mof(mof)
        bare_struct = load_structure(bare_path)
        k_struct = load_structure(k_path)
        k_indices = get_inserted_k_indices(bare_struct, k_struct)
        framework_idx = get_framework_indices(k_struct)
        symbols = [site.specie.symbol for site in k_struct]
        result.flags.extend(validate_k_insertion(bare_struct, k_struct))

        k_opt = optimize_structure(k_struct, "k_inserted", mof)
        if not k_opt.success:
            result.status = "Failed"
            result.flags.append("optimization_failed")
            return result

        # Prefer optimized cells; fall back to Stage-1 values carried in prior
        work = mof_work_dir(mof)
        bare_opt_path = work / "bare_optimized.cif"
        if bare_opt_path.is_file():
            try:
                bare_opt_struct = load_structure(bare_opt_path)
                result.volume_bare_A3, result.volume_k_A3, result.volume_change_pct = (
                    compute_volume_change_pct(bare_opt_struct, k_opt.structure)
                )
            except Exception as vol_exc:
                logger.warning("Volume change recomputation failed for %s: %s", mof, vol_exc)

        traj_path, log_path = run_md(k_opt.structure, mof, "stage2", md_time_ps)
        if traj_path is None:
            result.status = "Failed"
            result.flags.append("md_failed")
            return result

        md_result = analyze_md_trajectory(
            mof=mof,
            traj_path=traj_path,
            log_path=log_path,
            k_indices=k_indices,
            framework_idx=framework_idx,
            reference_structure=k_opt.structure,
            symbols=symbols,
            time_ps=md_time_ps,
            stage_label="stage2",
        )
        if not md_result.success:
            result.status = "Failed"
            result.flags.append("md_analysis_failed")
            return result

        result.mechanical_stability = md_result.mechanical_stability
        result.mechanical_stability_index = md_result.mechanical_stability_index
        result.framework_rmsd = md_result.framework_rmsd
        result.max_displacement = md_result.max_displacement
        result.bond_preservation = md_result.bond_preservation
        result.energy_fluctuation = md_result.energy_fluctuation
        result.volume_fluctuation_frac = getattr(md_result, "volume_fluctuation_frac", float("nan"))
        result.coordination_preservation = getattr(md_result, "coordination_preservation", float("nan"))
        result.msd_a2 = md_result.msd_a2
        result.diffusion_cm2_s = md_result.diffusion_cm2_s
        result.diffusion_fit_r2 = md_result.diffusion_fit_r2
        result.diffusion_slope_a2_ps = md_result.diffusion_slope_a2_ps
        result.diffusion_fit_start = md_result.diffusion_fit_start
        result.diffusion_fit_end = md_result.diffusion_fit_end
        result.diffusion_fit_num_frames = md_result.diffusion_fit_num_frames
        result.diffusion_simulation_duration_ps = md_result.simulation_time_ps
        result.diffusion_reliability = md_result.diffusion_reliability
        result.diffusion_warning = md_result.diffusion_warning
        result.md_converged = md_result.md_converged
        result.simulation_time_ps = md_result.simulation_time_ps
        result.flags.extend(validate_result(result))
        result.confidence_score = confidence_score(result, opt_ok=True, md_ok=result.md_converged)
        result.overall_score = compute_overall_score(result)
        result.screening_score = result.overall_score
        result.uncertainty_flag = uncertainty_from_confidence(result.confidence_score)
        if result.flags:
            log_flag_reasons(mof, result.flags, logger)
        result.status = "Flagged" if result.flags else "Complete"
        result.stage = 2

        ckpt = load_checkpoint(mof)
        ckpt.update(
            {
                "stage": 2,
                "simulation_time_ps": result.simulation_time_ps,
                "confidence_score": result.confidence_score,
                "overall_score": result.overall_score,
            }
        )
        save_checkpoint(mof, ckpt)

    except Exception as exc:
        logger.error("Stage 2 failed for %s: %s", mof, exc)
        result.status = "Failed"
        result.flags.append(str(exc))

    elapsed = time.perf_counter() - t0
    logger.info("MOF %s (stage2) finished in %.1f s — status=%s", mof, elapsed, result.status)
    return result


def process_mof_stage(
    mof: str,
    bare_path: Path,
    k_path: Path,
    md_time_ps: float,
    n_k: int,
) -> MOFResult:
    """Run optimization, adsorption, and MD for one MOF (Stage 1)."""
    logger = get_logger()
    t0 = time.perf_counter()
    result = MOFResult(mof=mof)
    ckpt = load_checkpoint(mof)

    try:
        log_device_for_mof(mof)
        bare_struct = load_structure(bare_path)
        k_struct = load_structure(k_path)
        k_indices = get_inserted_k_indices(bare_struct, k_struct)
        framework_idx = get_framework_indices(k_struct)
        symbols = [site.specie.symbol for site in k_struct]

        n_k_effective = effective_inserted_k_count(bare_struct, k_struct)
        if n_k_effective <= 0:
            raise ValueError(f"No inserted {ADSORBATE} atoms detected for {mof}")
        if len(k_indices) != n_k_effective:
            logger.warning(
                "Inserted K matching mismatch for %s: counted=%d, matched=%d; using matched count.",
                mof,
                n_k,
                len(k_indices),
            )
            n_k_effective = len(k_indices) if k_indices else n_k_effective

        result.flags.extend(validate_k_insertion(bare_struct, k_struct))
        if any(f in result.flags for f in ("k_overpacked", "k_ions_too_close")):
            log_flag_reasons(mof, [f for f in result.flags if f in ("k_overpacked", "k_ions_too_close")], logger)

        bare_relax = relax_bare_structure(
            bare_struct,
            mof,
            k_structure=k_struct,
            resume=True,
        )
        if not bare_relax.success:
            result.status = bare_relax.status
            result.flags.append(f"bare_relaxation_{bare_relax.status}")
            logger.error(
                "Skipping %s after bare relaxation: status=%s reason=%s",
                mof,
                bare_relax.status,
                bare_relax.error or "final force acceptance failed",
            )
            return result
        k_opt = optimize_structure(k_struct, "k_inserted", mof)
        if not k_opt.success:
            result.status = "Failed"
            result.flags.append("optimization_failed")
            return result

        result.optimization_energy = k_opt.energy

        if ckpt.get(ADSORPTION_CHECKPOINT_KEY) is not None:
            result.adsorption_energy_ev = ckpt[ADSORPTION_CHECKPOINT_KEY]
            result.adsorption_site_type = str(ckpt.get("adsorption_site_type", ""))
            result.adsorption_reference = str(ckpt.get("adsorption_reference", ADSORPTION_REFERENCE))
            result.adsorption_k_o_distance_a = float(
                ckpt.get("adsorption_k_o_distance_a", float("nan"))
            )
            result.adsorption_site_rank = int(ckpt.get("adsorption_site_rank", 1))
        else:
            result.adsorption_energy_ev = compute_adsorption_energy(
                bare_relax.structure,
                k_struct,
                k_indices,
                n_k_effective,
                mof,
                reference=ADSORPTION_REFERENCE,
                search_sites=True,
            )
            meta = load_adsorption_site_metadata(mof)
            result.adsorption_site_type = str(meta.get("best_site_type", ""))
            result.adsorption_reference = str(meta.get("reference", ADSORPTION_REFERENCE))
            result.adsorption_k_o_distance_a = float(
                meta.get("best_k_o_distance_a", float("nan"))
            )
            result.adsorption_site_rank = 1
            ckpt[ADSORPTION_CHECKPOINT_KEY] = result.adsorption_energy_ev
            ckpt["adsorption_site_type"] = result.adsorption_site_type
            ckpt["adsorption_reference"] = result.adsorption_reference
            ckpt["adsorption_k_o_distance_a"] = result.adsorption_k_o_distance_a
            ckpt["adsorption_site_rank"] = result.adsorption_site_rank
            ckpt.pop("adsorption_energy_ev_per_k", None)
            ckpt.pop("adsorption_energy_ev_per_k_v2", None)
            save_checkpoint(mof, ckpt)
        result.adsorption_binding_class = classify_adsorption(result.adsorption_energy_ev)
        result.optimization_converged = bool(
            bare_relax.success and getattr(k_opt, "converged", True)
        )
        if not result.optimization_converged:
            result.flags.append("optimization_unconverged")
        logger.info(
            "Adsorption for %s: %.3f eV/K (%s site, ref=%s, %d %s ions)",
            mof,
            result.adsorption_energy_ev,
            result.adsorption_site_type or "provided",
            result.adsorption_reference or ADSORPTION_REFERENCE,
            n_k_effective,
            ADSORBATE,
        )
        result.volume_bare_A3, result.volume_k_A3, result.volume_change_pct = (
            compute_volume_change_pct(bare_relax.structure, k_opt.structure)
        )
        logger.info(
            "Volume change for %s: %+.2f %% (%.1f → %.1f Å³)",
            mof,
            result.volume_change_pct,
            result.volume_bare_A3,
            result.volume_k_A3,
        )
        if result.adsorption_binding_class == "Very Weak":
            logger.warning(
                "Adsorption for %s is Very Weak (%.3f eV/K).",
                mof,
                result.adsorption_energy_ev,
            )
        if result.adsorption_binding_class == "Very Strong":
            logger.warning(
                "Adsorption for %s is Very Strong (%.3f eV/K).",
                mof,
                result.adsorption_energy_ev,
            )

        traj_path, log_path = run_md(k_opt.structure, mof, "stage1", md_time_ps)
        if traj_path is None:
            result.status = "Failed"
            result.flags.append("md_failed")
            return result

        md_result = analyze_md_trajectory(
            mof=mof,
            traj_path=traj_path,
            log_path=log_path,
            k_indices=k_indices,
            framework_idx=framework_idx,
            reference_structure=k_opt.structure,
            symbols=symbols,
            time_ps=md_time_ps,
            stage_label="stage1",
        )
        if not md_result.success:
            result.status = "Failed"
            result.flags.append("md_analysis_failed")
            return result

        result.mechanical_stability = md_result.mechanical_stability
        result.mechanical_stability_index = md_result.mechanical_stability_index
        result.framework_rmsd = md_result.framework_rmsd
        result.max_displacement = md_result.max_displacement
        result.energy_fluctuation = md_result.energy_fluctuation
        result.volume_fluctuation_frac = getattr(md_result, "volume_fluctuation_frac", float("nan"))
        result.msd_a2 = md_result.msd_a2
        result.diffusion_cm2_s = md_result.diffusion_cm2_s
        result.diffusion_fit_r2 = md_result.diffusion_fit_r2
        result.diffusion_slope_a2_ps = md_result.diffusion_slope_a2_ps
        result.diffusion_fit_start = md_result.diffusion_fit_start
        result.diffusion_fit_end = md_result.diffusion_fit_end
        result.diffusion_fit_num_frames = md_result.diffusion_fit_num_frames
        result.diffusion_simulation_duration_ps = md_result.simulation_time_ps
        result.diffusion_reliability = md_result.diffusion_reliability
        result.diffusion_warning = md_result.diffusion_warning
        result.md_converged = md_result.md_converged
        result.simulation_time_ps = md_result.simulation_time_ps
        result.flags.extend(md_result.flagged)
        result.flags.extend(validate_result(result))
        result.confidence_score = confidence_score(
            result, opt_ok=result.optimization_converged, md_ok=result.md_converged
        )
        result.overall_score = compute_overall_score(result)
        result.screening_score = result.overall_score
        result.uncertainty_flag = uncertainty_from_confidence(result.confidence_score)
        if result.flags:
            log_flag_reasons(mof, result.flags, logger)
        result.status = "Flagged" if result.flags else "Complete"
        result.stage = 1

        ckpt.update(
            {
                "stage": result.stage,
                "simulation_time_ps": result.simulation_time_ps,
                "mechanical_stability": result.mechanical_stability,
                "mechanical_stability_index": result.mechanical_stability_index,
                "confidence_score": result.confidence_score,
                "overall_score": result.overall_score,
                "adsorption_site_type": result.adsorption_site_type,
                "adsorption_reference": result.adsorption_reference,
            }
        )
        save_checkpoint(mof, ckpt)

    except Exception as exc:
        logger.error("Processing failed for %s: %s", mof, exc)
        result.status = "Failed"
        result.flags.append(str(exc))

    elapsed = time.perf_counter() - t0
    logger.info("MOF %s (stage1) finished in %.1f s — status=%s", mof, elapsed, result.status)
    return result


def refresh_adsorption_energy_if_needed(
    mof: str,
    bare_path: Path,
    k_path: Path,
    result: MOFResult,
) -> MOFResult:
    """Recompute adsorption when legacy checkpoints are encountered."""
    ckpt = load_checkpoint(mof)
    if ckpt.get(ADSORPTION_CHECKPOINT_KEY) is not None:
        result.adsorption_energy_ev = ckpt[ADSORPTION_CHECKPOINT_KEY]
        result.adsorption_binding_class = classify_adsorption(result.adsorption_energy_ev)
        return result

    logger = get_logger()
    bare_struct = load_structure(bare_path)
    k_struct = load_structure(k_path)
    k_indices = get_inserted_k_indices(bare_struct, k_struct)
    n_k_effective = len(k_indices) if k_indices else effective_inserted_k_count(bare_struct, k_struct)
    bare_relax = relax_bare_structure(
        bare_struct,
        mof,
        k_structure=k_struct,
        resume=True,
    )
    if not bare_relax.success or n_k_effective <= 0 or not k_indices:
        return result

    result.adsorption_energy_ev = compute_adsorption_energy(
        bare_relax.structure,
        k_struct,
        k_indices,
        n_k_effective,
        mof,
    )
    result.adsorption_binding_class = classify_adsorption(result.adsorption_energy_ev)
    ckpt[ADSORPTION_CHECKPOINT_KEY] = result.adsorption_energy_ev
    ckpt.pop("adsorption_energy_ev_per_k", None)
    save_checkpoint(mof, ckpt)
    logger.info(
        "Refreshed adsorption for %s: %.3f eV/K (%d %s ions)",
        mof,
        result.adsorption_energy_ev,
        n_k_effective,
        ADSORBATE,
    )
    return result


def load_existing_results() -> dict[str, MOFResult]:
    """Resume partial results from results.csv if present."""
    path = OUTPUT_DIR / "results.csv"
    existing: dict[str, MOFResult] = {}
    if not path.is_file():
        return existing
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        msi = row.get(
            "Mechanical_Stability_Value",
            row.get("Mechanical_Stability_Index", np.nan),
        )
        mech_stability_text = str(row.get("Mechanical_Stability", "")).strip()
        if not mech_stability_text or mech_stability_text.lower() == "nan":
            mech_stability_text = classify_mechanical_stability(msi)
        elif mech_stability_text.lower() == "collapse":
            mech_stability_text = "Collapsed"

        status_raw = str(
            row.get("Candidate_Assessment", row.get("Status", ""))
        ).strip()
        has_md_metrics = np.isfinite(row.get("MSD_A2", np.nan))
        if status_raw.lower() in {"pass", "good", "bad"}:
            normalized_status = "Complete" if has_md_metrics else "Failed"
        elif status_raw.lower() == "fail":
            normalized_status = "Failed"
        elif status_raw:
            normalized_status = status_raw
        else:
            normalized_status = "Complete" if has_md_metrics else "Pending"

        r = MOFResult(
            mof=str(row["MOF"]),
            adsorption_energy_ev=row.get("adso_ener", row.get("Adsorption_Energy_eV", np.nan)),
            adsorption_binding_class=str(row.get("Adsorption_Binding_Class", "")),
            mechanical_stability=mech_stability_text,
            mechanical_stability_index=msi,
            framework_rmsd=row.get("Framework_RMSD", np.nan),
            max_displacement=row.get("Maximum_Displacement", np.nan),
            energy_fluctuation=row.get("Energy_Fluctuation", np.nan),
            msd_a2=row.get("msd", row.get("MSD_A2", np.nan)),
            diffusion_cm2_s=row.get("diff_coeff", row.get("Diffusion_Coefficient_cm2_s", np.nan)),
            diffusion_fit_r2=row.get("Diffusion_Fit_R2", np.nan),
            diffusion_slope_a2_ps=row.get("Diffusion_Slope_A2_ps", np.nan),
            diffusion_fit_start=int(row.get("Diffusion_Fit_Start_Frame", -1)),
            diffusion_fit_end=int(row.get("Diffusion_Fit_End_Frame", -1)),
            diffusion_fit_num_frames=int(row.get("Diffusion_Fit_Num_Frames", 0)),
            diffusion_simulation_duration_ps=row.get("Diffusion_Simulation_Duration_ps", np.nan),
            diffusion_reliability=str(row.get("Diffusion_Reliability", "Very Low")),
            diffusion_warning=str(row.get("Diffusion_Warning", "")),
            volume_bare_A3=row.get("Volume_Bare_A3", np.nan),
            volume_k_A3=row.get("Volume_K_A3", np.nan),
            volume_change_pct=row.get("Volume_Change_pct", np.nan),
            optimization_energy=row.get("Optimization_Energy", np.nan),
            confidence_score=row.get("Confidence_Score", np.nan),
            uncertainty_flag=str(row.get("Uncertainty_Flag", "")),
            simulation_time_ps=row.get("Simulation_Time_ps", np.nan),
            status=normalized_status,
        )
        r.stage = 2 if r.simulation_time_ps >= FINAL_MD_TIME_PS - 0.1 else 1
        existing[r.mof] = r
    return existing


def main() -> None:
    global _logger
    np.random.seed(RANDOM_SEED)
    t_start = time.perf_counter()
    _logger = setup_logging(OUTPUT_DIR)
    logger = get_logger()

    logger.info("=" * 60)
    logger.info("CHGNet MOF K-ion Battery Screening Pipeline")
    logger.info("=" * 60)
    logger.info(
        "Stage 1: %.1f ps MD | Stage 2: %.1f ps MD | Top %.0f%% advance",
        FAST_MD_TIME_PS,
        FINAL_MD_TIME_PS,
        TOP_PERCENT,
    )
    logger.info(
        "CHGNet limitation: MLIP energies/MD are approximate vs DFT; "
        "short NVT trajectories may not fully sample slow K diffusion."
    )

    pairs = pair_mof_files()
    if not pairs:
        logger.error("No valid MOF pairs found — exiting")
        sys.exit(1)

    n_k_map: dict[str, int] = {}
    for mof, bare_p, k_p in pairs:
        try:
            n_k_map[mof] = count_inserted_ions(load_structure(bare_p), load_structure(k_p))
        except Exception as exc:
            logger.warning("Cannot count %s ions for %s: %s", ADSORBATE, mof, exc)
            n_k_map[mof] = 0

    get_isolated_k_energy()
    existing = load_existing_results()
    stage1_results: list[MOFResult] = []
    total = len(pairs)

    for idx, (mof, bare_p, k_p) in enumerate(
        tqdm(pairs, desc="Stage 1 fast screening", unit="MOF"), start=1
    ):
        if mof in existing and existing[mof].stage >= 1 and existing[mof].status != "Failed":
            result = refresh_adsorption_energy_if_needed(mof, bare_p, k_p, existing[mof])
            logger.info("Skipping Stage 1 for %s (already complete)", mof)
        else:
            result = process_mof_stage(
                mof, bare_p, k_p, FAST_MD_TIME_PS, n_k_map[mof]
            )
        scored_preview = compute_screening_scores([result], n_k_map)
        result.screening_score = float(scored_preview["Screening_Score"].iloc[0])
        print_mof_progress(idx, total, result)
        stage1_results.append(result)
        write_results_csv(stage1_results)

    valid_stage1 = [r for r in stage1_results if r.status in ("Complete", "Flagged", "Stage1_Only")]
    if not valid_stage1:
        logger.error("No MOFs completed Stage 1 — cannot rank")
        write_best_mof_outputs(stage1_results, n_k_map)
        sys.exit(1)

    scored_df = compute_screening_scores(valid_stage1, n_k_map)
    n_select = max(1, int(np.ceil(len(valid_stage1) * TOP_PERCENT / 100.0)))
    selected = (
        scored_df.sort_values("Screening_Score", ascending=False)
        .head(n_select)
        .index.tolist()
    )
    logger.info("Stage 2: %d / %d MOFs selected (top %.0f%%)", len(selected), len(valid_stage1), TOP_PERCENT)

    result_map = {r.mof: r for r in stage1_results}
    pair_map = {mof: (bare, k) for mof, bare, k in pairs}

    for mof in tqdm(selected, desc="Stage 2 accurate screening", unit="MOF"):
        if mof in existing and existing[mof].stage >= 2:
            result_map[mof] = existing[mof]
            logger.info("Skipping Stage 2 for %s (already complete)", mof)
            continue
        bare_p, k_p = pair_map[mof]
        prior = result_map[mof]
        stage2 = process_mof_stage2(mof, bare_p, k_p, FINAL_MD_TIME_PS, n_k_map[mof], prior)
        if stage2.status != "Failed":
            result_map[mof] = stage2
        else:
            prev = result_map[mof]
            prev.status = "Stage1_Only"
            result_map[mof] = prev

    final_results = [result_map[mof] for mof, _, _ in pairs if mof in result_map]
    write_results_csv(final_results)
    best = write_best_mof_outputs(final_results, n_k_map)

    elapsed = time.perf_counter() - t_start
    logger.info("Pipeline complete in %.1f s", elapsed)
    logger.info("All MOFs: %s/results.csv", OUTPUT_DIR)
    logger.info("Passed thresholds: %s/best_cif.csv", OUTPUT_DIR)
    logger.info("Top MOF: %s/best_mof.csv", OUTPUT_DIR)
    logger.info("Passed CIFs: %s/best_cifs/", OUTPUT_DIR)
    if best is not None:
        logger.info("Top candidate: %s", best.mof)


if __name__ == "__main__":
    main()
