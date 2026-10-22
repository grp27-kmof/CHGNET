"""
Orchestration for publication-quality adsorption-site search.

Workflow (standard multi-start global optimization):
  1. Discover hundreds of symmetry-unique candidate placements
  2. Pre-score by local environment (no MLIP)
  3. Relax K on fixed framework for EVERY unique candidate (CHGNet)
  4. Cluster relaxed structures → unique local minima
  5. Rank by adsorption energy; global minimum = best site
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from pymatgen.core import Structure

from scientific.adsorption_sites import (
    AdsorptionCandidate,
    RelaxedAdsorptionSite,
    analyze_relaxed_site_geometry,
    cluster_relaxed_minima,
    compute_adsorption_confidence,
    discover_adsorption_candidates,
)

ALGORITHM_VERSION = "adsorption_v4"

RelaxFn = Callable[[Structure, list[int], str], Any]  # -> OptimizationResult-like
EnergyFn = Callable[[Structure, str], float]


@dataclass
class AdsorptionSearchResult:
    """Full adsorption search output."""
    best: RelaxedAdsorptionSite | None
    all_relaxed: list[RelaxedAdsorptionSite]
    local_minima: list[RelaxedAdsorptionSite]
    candidates: list[AdsorptionCandidate]
    reference: str
    e_bare: float
    e_ref: float
    n_discovered: int = 0
    n_unique: int = 0
    n_relaxed: int = 0
    n_local_minima: int = 0

    def to_ranking_dict(self) -> dict[str, Any]:
        best = self.best
        return {
            "algorithm_version": ALGORITHM_VERSION,
            "reference": self.reference,
            "best_site_type": best.site_type if best else "",
            "best_adsorption_energy_ev": best.adsorption_energy_ev if best else float("nan"),
            "best_k_o_distance_a": best.k_o_distance_a if best else float("nan"),
            "best_k_frac_coords": best.k_frac_coords if best else [],
            "best_confidence_score": best.confidence_score if best else float("nan"),
            "best_local_environment_descriptor": (
                best.local_environment_descriptor if best else ""
            ),
            "n_candidates_discovered": self.n_discovered,
            "n_candidates_unique": self.n_unique,
            "n_relaxed": self.n_relaxed,
            "n_local_minima": self.n_local_minima,
            "sites": [s.to_dict() for s in self.all_relaxed if s.success],
            "local_minima": [s.to_dict() for s in self.local_minima],
            "candidates_summary": [c.to_dict() for c in self.candidates],
        }


def _relax_one_candidate(
    bare_opt: Structure,
    cand: AdsorptionCandidate,
    mof: str,
    e_bare: float,
    e_ref: float,
    ref_label: str,
    relax_fn: RelaxFn,
    energy_fn: EnergyFn,
    adsorbate: str,
) -> RelaxedAdsorptionSite:
    label = f"ads_site_{cand.candidate_id:04d}_{cand.site_type}"
    site = RelaxedAdsorptionSite(
        candidate_id=cand.candidate_id,
        site_type=cand.site_type,
        pre_score=cand.pre_score,
        oxygen_index=cand.anchor_index if cand.anchor_symbol == "O" else -1,
    )
    try:
        trial = bare_opt.copy()
        trial.append(adsorbate, cand.k_frac, coords_are_cartesian=False)
        k_idx = [len(trial) - 1]
        relaxed = relax_fn(trial, k_idx, label)
        if not getattr(relaxed, "success", False) or relaxed.structure is None:
            site.notes = getattr(relaxed, "error", "") or "relaxation failed"
            return site

        e_sys = energy_fn(relaxed.structure, mof)
        e_ads = e_sys - e_bare - e_ref
        geom = analyze_relaxed_site_geometry(relaxed.structure, k_idx[0], cand.site_type)

        site.adsorption_energy_ev = float(e_ads)
        site.k_frac_coords = geom["k_frac_coords"]
        site.k_o_distance_a = geom["k_o_distance_a"]
        site.k_metal_distance_a = geom["k_metal_distance_a"]
        site.oxygen_index = geom["oxygen_index"]
        site.coordination_number = geom["coordination_number"]
        site.nearest_atoms = geom["nearest_atoms"]
        site.local_environment_descriptor = geom["local_environment_descriptor"]
        site.optimization_converged = bool(getattr(relaxed, "converged", False))
        site.max_force_ev_a = float(getattr(relaxed, "max_force", float("nan")))
        site.structure = relaxed.structure
        site.success = True
        site.notes = f"reference={ref_label}"
    except Exception as exc:
        site.notes = str(exc)
    return site


def run_adsorption_site_search(
    bare_opt: Structure,
    mof: str,
    e_ref: float,
    ref_label: str,
    relax_fn: RelaxFn,
    energy_fn: EnergyFn,
    *,
    adsorbate: str = "K",
    max_raw_candidates: int = 800,
    max_unique_candidates: int = 400,
    parallel_workers: int = 1,
    candidates_override: list[AdsorptionCandidate] | None = None,
    logger: logging.Logger | None = None,
) -> AdsorptionSearchResult:
    """
    Full adsorption-site search for single-K MOFs.

    Parameters
    ----------
    relax_fn : callable(structure, k_indices, label) -> OptimizationResult
    energy_fn : callable(structure, mof) -> float
    parallel_workers : int
        Thread pool size for candidate relaxation. Use 1 on GPU (default).
        >1 can help on CPU when CHGNet releases the GIL during inference.
    """
    log = logger or logging.getLogger("adsorption_workflow")
    e_bare = energy_fn(bare_opt, mof)

    if candidates_override is not None:
        candidates = candidates_override
    else:
        candidates = discover_adsorption_candidates(
            bare_opt,
            max_raw_candidates=max_raw_candidates,
            max_unique_candidates=max_unique_candidates,
        )
    n_discovered = len(candidates)
    log.info(
        "Adsorption search %s: %d symmetry-unique candidates to relax "
        "(pre-scored, chemistry-based enumeration)",
        mof,
        n_discovered,
    )

    all_relaxed: list[RelaxedAdsorptionSite] = []

    def _run_one(cand: AdsorptionCandidate) -> RelaxedAdsorptionSite:
        return _relax_one_candidate(
            bare_opt, cand, mof, e_bare, e_ref, ref_label,
            relax_fn, energy_fn, adsorbate,
        )

    if parallel_workers > 1:
        with ThreadPoolExecutor(max_workers=parallel_workers) as pool:
            futures = {pool.submit(_run_one, c): c for c in candidates}
            for i, fut in enumerate(as_completed(futures), start=1):
                cand = futures[fut]
                site = fut.result()
                all_relaxed.append(site)
                if site.success:
                    log.info(
                        "  [%d/%d] %s (id=%d): E_ads=%.4f eV/K  K–O=%.2f Å  conv=%s",
                        i, n_discovered, cand.site_type, cand.candidate_id,
                        site.adsorption_energy_ev, site.k_o_distance_a,
                        site.optimization_converged,
                    )
    else:
        for i, cand in enumerate(candidates, start=1):
            site = _run_one(cand)
            all_relaxed.append(site)
            if site.success:
                log.info(
                    "  [%d/%d] %s (id=%d): E_ads=%.4f eV/K  K–O=%.2f Å  conv=%s",
                    i, n_discovered, cand.site_type, cand.candidate_id,
                    site.adsorption_energy_ev, site.k_o_distance_a,
                    site.optimization_converged,
                )

    local_minima = cluster_relaxed_minima(all_relaxed)
    energies = [s.adsorption_energy_ev for s in local_minima]
    for s in local_minima:
        s.confidence_score = compute_adsorption_confidence(s, energies)

    # Also assign confidence to all relaxed (for histogram)
    all_energies = [s.adsorption_energy_ev for s in all_relaxed if s.success]
    for s in all_relaxed:
        if s.success and not np.isfinite(s.confidence_score):
            s.confidence_score = compute_adsorption_confidence(s, all_energies)

    best = local_minima[0] if local_minima else None
    if best:
        log.info(
            "Global minimum for %s: %s  E_ads=%.4f eV/K  "
            "(cluster %d, %d unique minima from %d relaxations)",
            mof, best.site_type, best.adsorption_energy_ev,
            best.cluster_id, len(local_minima), len(all_relaxed),
        )

    return AdsorptionSearchResult(
        best=best,
        all_relaxed=all_relaxed,
        local_minima=local_minima,
        candidates=candidates,
        reference=ref_label,
        e_bare=e_bare,
        e_ref=e_ref,
        n_discovered=n_discovered,
        n_unique=n_discovered,
        n_relaxed=sum(1 for s in all_relaxed if s.success),
        n_local_minima=len(local_minima),
    )


def save_adsorption_results(
    result: AdsorptionSearchResult,
    work_dir: Path,
    mof: str,
) -> Path:
    """Write adsorption_site_ranking.json and best-structure CIF."""
    work_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = work_dir / "adsorption_site_ranking.json"
    with open(ranking_path, "w", encoding="utf-8") as f:
        json.dump(result.to_ranking_dict(), f, indent=2)

    if result.best and result.best.structure is not None:
        result.best.structure.to(
            fmt="cif",
            filename=str(work_dir / "k_on_bare_optimized.cif"),
        )
    return ranking_path
