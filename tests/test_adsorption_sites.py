"""Tests for publication-quality adsorption site enumeration."""

from __future__ import annotations

import numpy as np
from pymatgen.core import Lattice, Structure

from scientific.adsorption_sites import (
    AdsorptionCandidate,
    analyze_relaxed_site_geometry,
    bond_cutoff,
    cluster_relaxed_minima,
    deduplicate_site_dicts,
    deduplicate_sites_by_symmetry,
    discover_adsorption_candidates,
    k_oxygen_contact_distance,
    nearest_site_by_symbol,
    RelaxedAdsorptionSite,
    score_candidate_pre_relaxation,
)


def test_deduplicate_removes_near_duplicate_fractional_sites():
    lattice = Lattice.cubic(20.0)
    bare = Structure(lattice, ["C"], [[0.0, 0.0, 0.0]])
    candidates = [
        {"site_type": "pore", "k_frac": [0.5, 0.5, 0.5], "oxygen_index": -1},
        {"site_type": "pore", "k_frac": [0.501, 0.5, 0.5], "oxygen_index": -1},
        {"site_type": "pore", "k_frac": [0.1, 0.1, 0.1], "oxygen_index": -1},
    ]
    unique = deduplicate_site_dicts(bare, candidates, tolerance=0.75)
    assert len(unique) == 2


def test_chemistry_based_k_o_distance_is_physical():
    d = k_oxygen_contact_distance()
    assert 2.4 < d < 3.2


def test_bond_cutoff_from_covalent_radii():
    d = bond_cutoff("K", "O")
    assert 2.4 < d < 3.5


def test_discover_candidates_on_simple_mof_like_cell():
    """UiO-66-free: Zn4O cluster proxy in a large cell."""
    lattice = Lattice.cubic(26.0)
    species = ["Zn", "Zn", "Zn", "Zn", "O", "O", "O", "O", "O", "O"]
    coords = [
        [0.10, 0.10, 0.10],
        [0.90, 0.10, 0.10],
        [0.10, 0.90, 0.10],
        [0.10, 0.10, 0.90],
        [0.05, 0.05, 0.05],
        [0.15, 0.05, 0.05],
        [0.05, 0.15, 0.05],
        [0.05, 0.05, 0.15],
        [0.50, 0.50, 0.50],
        [0.55, 0.50, 0.50],
    ]
    bare = Structure(lattice, species, coords)
    cands = discover_adsorption_candidates(bare, max_raw_candidates=200, max_unique_candidates=100)
    assert len(cands) >= 5
    types = {c.site_type for c in cands}
    assert types  # non-empty
    assert all(np.isfinite(c.pre_score) for c in cands)


def test_pre_relaxation_scoring_returns_components():
    lattice = Lattice.cubic(20.0)
    bare = Structure(lattice, ["O"], [[0.5, 0.5, 0.5]])
    cand = AdsorptionCandidate(
        candidate_id=1,
        site_type="mu3_O",
        k_frac=[0.52, 0.5, 0.5],
        anchor_index=0,
        anchor_symbol="O",
    )
    score, comps = score_candidate_pre_relaxation(bare, cand)
    assert 0.0 <= score <= 1.0
    assert "electrostatic" in comps
    assert "steric" in comps


def test_cluster_relaxed_minima_merges_duplicates():
    s1 = RelaxedAdsorptionSite(
        candidate_id=1, site_type="mu3_O", success=True,
        adsorption_energy_ev=-2.0, k_frac_coords=[0.5, 0.5, 0.5],
    )
    s2 = RelaxedAdsorptionSite(
        candidate_id=2, site_type="mu3_O", success=True,
        adsorption_energy_ev=-2.1, k_frac_coords=[0.5001, 0.5, 0.5],
    )
    s3 = RelaxedAdsorptionSite(
        candidate_id=3, site_type="pore_grid", success=True,
        adsorption_energy_ev=-1.0, k_frac_coords=[0.1, 0.1, 0.1],
    )
    lattice = Lattice.cubic(20.0)
    s1.structure = Structure(lattice, ["K"], [[0.5, 0.5, 0.5]])
    s2.structure = s1.structure
    s3.structure = Structure(lattice, ["K"], [[0.1, 0.1, 0.1]])

    minima = cluster_relaxed_minima([s1, s2, s3], cluster_distance_a=0.5)
    assert len(minima) == 2
    assert minima[0].adsorption_energy_ev == -2.1
    assert minima[0].is_global_minimum


def test_symmetry_dedup_on_candidate_objects():
    lattice = Lattice.cubic(20.0)
    bare = Structure(lattice, ["C"], [[0.0, 0.0, 0.0]])
    cands = [
        AdsorptionCandidate(1, "pore_grid", [0.5, 0.5, 0.5], pre_score=0.9),
        AdsorptionCandidate(2, "pore_grid", [0.501, 0.5, 0.5], pre_score=0.8),
        AdsorptionCandidate(3, "pore_grid", [0.1, 0.1, 0.1], pre_score=0.7),
    ]
    unique = deduplicate_sites_by_symmetry(bare, cands, tolerance_a=0.75)
    assert len(unique) == 2


def test_nearest_oxygen_always_found_after_relaxation():
    """K–O distance must be finite when O exists in the framework."""
    lattice = Lattice.cubic(12.0)
    # Zr node proxy: O at origin, K placed ~2.8 Å along z
    bare = Structure(lattice, ["O", "Zr"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    k_frac = np.array([0.0, 0.0, 2.8 / 12.0])
    relaxed = bare.copy()
    relaxed.append("K", k_frac, coords_are_cartesian=False)
    k_idx = len(relaxed) - 1

    d_o, o_idx = nearest_site_by_symbol(relaxed, k_idx, "O")
    assert np.isfinite(d_o)
    assert o_idx == 0
    assert 2.5 <= d_o <= 3.2

    geom = analyze_relaxed_site_geometry(relaxed, k_idx, "mu3_O")
    assert np.isfinite(geom["k_o_distance_a"])
    assert geom["oxygen_index"] == 0
    assert geom["coordination_number"] >= 1
    assert any(n["symbol"] == "O" for n in geom["nearest_atoms"])
