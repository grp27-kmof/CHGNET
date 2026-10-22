"""
Publication-quality scientific analysis utilities for CHGNETT.

Modules:
  msd_analysis          — trajectory unwrapping, equilibration detection, Einstein D + CI
  adsorption_sites      — chemistry-based site enumeration, clustering, confidence
  adsorption_workflow   — multi-start global adsorption search orchestration
  adsorption_plots      — publication adsorption energy landscape figures
  manifest              — reproducibility metadata capture
  msi_analysis          — coordination preservation metrics
"""

from scientific.msd_analysis import (
    detect_equilibration_index,
    diffusion_from_slope_with_ci,
    einstein_fit_with_uncertainty,
    unwrap_positions_mic,
)
from scientific.manifest import write_run_manifest

__all__ = [
    "detect_equilibration_index",
    "diffusion_from_slope_with_ci",
    "einstein_fit_with_uncertainty",
    "unwrap_positions_mic",
    "write_run_manifest",
]
