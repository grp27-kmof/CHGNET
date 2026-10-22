"""
Reproducibility manifest for publication-quality CHGNet MOF screening.

Writes `outputs/run_manifest.json` capturing software versions, random seeds,
simulation parameters, and git commit hash when available.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _package_version(name: str) -> str:
    try:
        import importlib.metadata as im

        return im.version(name)
    except Exception:
        return "unknown"


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parent.parent,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def collect_manifest(
    config_module: Any,
    cli_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Gather reproducibility metadata from a config module (e.g. run_pipeline)."""
    torch_version = "not_installed"
    cuda_available = False
    try:
        import torch

        torch_version = torch.__version__
        cuda_available = bool(torch.cuda.is_available())
    except ImportError:
        pass

    chgnet_version = _package_version("chgnet")

    manifest: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python": sys.version,
            "processor": platform.processor(),
        },
        "software": {
            "chgnet": chgnet_version,
            "torch": torch_version,
            "cuda_available": cuda_available,
            "numpy": np.__version__,
            "ase": _package_version("ase"),
            "pymatgen": _package_version("pymatgen"),
            "scipy": _package_version("scipy"),
            "pandas": _package_version("pandas"),
        },
        "git_commit": _git_commit(),
        "random_seed": getattr(config_module, "RANDOM_SEED", None),
        "md": {
            "temperature_K": getattr(config_module, "TEMPERATURE_K", None),
            "timestep_fs": getattr(config_module, "TIME_STEP_FS", None),
            "stage1_ps": getattr(config_module, "FAST_MD_TIME_PS", None),
            "stage2_ps": getattr(config_module, "FINAL_MD_TIME_PS", None),
            "thermostat_equil": getattr(config_module, "MD_THERMOSTAT_EQUIL", None),
            "thermostat_production": getattr(config_module, "MD_THERMOSTAT_PRODUCTION", None),
            "equilibration_fraction": getattr(config_module, "MD_EQUILIBRATION_FRACTION", None),
        },
        "optimization": {
            "force_tol_ev_a": getattr(config_module, "FORCE_TOL", None),
            "force_tol_loose_ev_a": getattr(config_module, "FORCE_TOL_LOOSE", None),
            "max_steps": getattr(config_module, "MAX_OPT_STEPS", None),
            "relax_cell": getattr(config_module, "RELAX_CELL", None),
        },
        "adsorption": {
            "algorithm_version": "adsorption_v4",
            "reference": getattr(config_module, "ADSORPTION_REFERENCE", None),
            "max_raw_candidates": getattr(config_module, "ADSORPTION_MAX_RAW_CANDIDATES", None),
            "max_unique_candidates": getattr(config_module, "ADSORPTION_MAX_UNIQUE_CANDIDATES", None),
            "parallel_workers": getattr(config_module, "ADSORPTION_PARALLEL_WORKERS", None),
        },
        "diffusion": {
            "accept_r2": getattr(config_module, "MD_DIFFUSION_ACCEPT_R2", None),
            "min_fit_points": getattr(config_module, "MIN_DIFFUSION_FIT_POINTS", None),
            "dimensions": getattr(config_module, "DIFFUSION_DIMENSIONS", None),
            "min_slope_tstat": getattr(config_module, "DIFFUSION_MIN_SLOPE_TSTAT", None),
        },
        "scoring_weights": getattr(config_module, "WEIGHTS", None),
        "msi_weights": getattr(config_module, "MSI_WEIGHTS", None),
        "cli_args": cli_args or {},
    }
    return manifest


def write_run_manifest(
    output_dir: Path,
    config_module: Any,
    cli_args: dict[str, Any] | None = None,
) -> Path:
    """Write run_manifest.json to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = collect_manifest(config_module, cli_args)
    path = output_dir / "run_manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    return path
