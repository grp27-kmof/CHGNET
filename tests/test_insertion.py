"""Mock-only tests for Change C insertion calculations and stop rules."""

from __future__ import annotations

import json

import numpy as np

import run_pipeline


def test_step_voltage_and_formation_energy():
    previous = -10.0
    current = -11.2
    bulk = -1.0
    assert np.isclose(-(current - previous - bulk), 0.2)
    assert np.isclose(current - (-9.0) - 2 * bulk, -0.2)


def test_convex_hull_selection():
    assert run_pipeline.convex_hull_vertices(
        [(0, 0.0), (1, -1.0), (2, 0.0), (3, -0.5)]
    ) == [(0, 0.0), (1, -1.0), (3, -0.5)]


def test_capacity_formula():
    assert run_pipeline.insertion_capacity_mAh_g(2, 100.0) == 536.02


def test_framework_mapping_handles_k_in_middle_of_species_order():
    from pymatgen.core import Lattice, Structure

    lattice = Lattice.cubic(10)
    bare = Structure(lattice, ["C", "O", "N"], [[0.1, 0.1, 0.1], [0.12, 0.12, 0.12], [0.14, 0.14, 0.14]])
    with_k = Structure(
        lattice,
        ["C", "K", "O", "N"],
        [[0.1, 0.1, 0.1], [0.5, 0.5, 0.5], [0.12, 0.12, 0.12], [0.14, 0.14, 0.14]],
    )
    mapping = run_pipeline._framework_site_map(bare, with_k)
    assert mapping == {0: 0, 1: 2, 2: 3}
    assert run_pipeline._insertion_framework_rmsd(bare, with_k, mapping) < 1e-12
    assert run_pipeline._insertion_bond_preservation(bare, with_k, mapping) == 1.0


def test_each_insertion_stop_condition():
    base = {
        "converged": True,
        "min_K_K_dist_A": 4.0,
        "bond_preservation": 1.0,
        "framework_rmsd_vs_bare_A": 0.1,
        "step_voltage_V": 1.0,
    }
    cases = [
        ({"converged": False}, "not_converged"),
        ({"min_K_K_dist_A": 3.0}, "k_clustering"),
        ({"bond_preservation": 0.8}, "framework_damage"),
        ({"framework_rmsd_vs_bare_A": 1.1}, "framework_damage"),
        ({"step_voltage_V": -0.1}, "negative_step_voltage"),
    ]
    for overrides, expected in cases:
        metrics = {**base, **overrides}
        assert expected in run_pipeline._insertion_rejection_reasons(metrics, min_kk=3.8)


def test_resume_reuses_cached_steps(tmp_path, monkeypatch):
    from pymatgen.core import Lattice, Structure
    bare = Structure(Lattice.cubic(10), ["C"], [[0, 0, 0]])
    mof = "resume_mof"
    work = tmp_path / "structures" / mof / "insertion"
    work.mkdir(parents=True)
    monkeypatch.setattr(run_pipeline, "OUTPUT_DIR", tmp_path)
    (work / "k0_relaxed.cif").write_text(bare.to(fmt="cif"), encoding="utf-8")
    rows = [{
        "n_K": 0, "energy_eV": -1.0, "volume_A3": 1000.0,
        "volume_change_vs_bare_percent": 0.0, "formation_energy_eV": 0.0,
        "step_voltage_V": np.nan, "min_K_K_dist_A": np.inf,
        "min_K_framework_dist_A": np.inf, "framework_rmsd_vs_bare_A": 0.0,
        "bond_preservation": 1.0, "final_fmax": 0.0, "converged": True,
        "wall_time_s": 0.0,
    }]
    import pandas as pd
    settings = {
        "n_max": 0, "n_prescreen": 8, "n_full": 2, "min_kk": 3.8,
        "v_threshold": 1.0, "fmax": 0.05, "max_steps": 1500,
    }
    pd.DataFrame(rows).to_csv(work / "insertion_steps.csv", index=False)
    (work / "insertion_summary.json").write_text(
        json.dumps({"settings": settings}), encoding="utf-8"
    )
    result = run_pipeline.run_sequential_k_insertion(
        bare, mof, -1.0, -1.0, n_max=0, resume=True
    )
    assert result["n_max_stable"] == 0
