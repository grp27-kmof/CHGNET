"""Mock-only tests for Change B single-K adsorption bookkeeping."""

from __future__ import annotations

import json
import logging

import numpy as np
from pymatgen.core import Lattice, Structure

import run_pipeline


def _bare() -> Structure:
    return Structure(Lattice.cubic(10), ["C", "O"], [[0, 0, 0], [0.3, 0, 0]])


def test_adsorption_energy_and_voltage_sign():
    e_ads, voltage = run_pipeline.adsorption_energy_and_voltage(-10.0, -8.0, -1.5)
    assert e_ads == -0.5
    assert voltage == 0.5


def test_validity_flags_are_independent(monkeypatch):
    bare = _bare()
    complex_structure = bare.copy()
    complex_structure.append("K", [0.5, 0.5, 0.5])
    monkeypatch.setattr(run_pipeline, "framework_displacements", lambda *args, **kwargs: (0.6, 0.6))
    monkeypatch.setattr(run_pipeline, "bond_preservation_fraction", lambda *args, **kwargs: 0.90)
    monkeypatch.setattr(run_pipeline, "min_framework_distance", lambda *args, **kwargs: 2.0)
    flags, *_ = run_pipeline._adsorption_validity(bare, complex_structure, 2, 0.06)
    assert {
        "final_fmax_above_0.05_eV_per_A",
        "min_K_framework_distance_below_2.3_A",
        "framework_rmsd_above_0.5_A",
        "bond_preservation_below_95_percent",
    } <= set(flags)


def test_non_relaxed_mof_is_skipped(tmp_path, monkeypatch):
    import run_md

    monkeypatch.setattr(run_pipeline, "OUTPUT_DIR", tmp_path)
    bare_dir = tmp_path / "bare"
    bare_dir.mkdir()
    _bare().to(fmt="cif", filename=str(bare_dir / "MOF.cif"))
    work = tmp_path / "structures" / "MOF"
    work.mkdir(parents=True)
    with open(work / "relax_summary.json", "w", encoding="utf-8") as f:
        json.dump({"status": "not_relaxed"}, f)
    logger = logging.getLogger("test_adsorb_only")
    run_md.run_adsorb_only(bare_dir, True, logger, 10, 3, 0.05, 1000)
    assert not (work / "adsorption_k1_summary.json").exists()


def test_required_summary_keys_with_mocked_relaxation(tmp_path, monkeypatch):
    bare = _bare()
    monkeypatch.setattr(run_pipeline, "OUTPUT_DIR", tmp_path)
    candidate = run_pipeline.AdsorptionCandidate(
        candidate_id=1, site_type="pore_grid", k_frac=[0.5, 0.5, 0.5], pre_score=1.0
    )
    monkeypatch.setattr(run_pipeline, "discover_adsorption_candidates", lambda *a, **k: [candidate])
    monkeypatch.setattr(
        run_pipeline,
        "score_candidate_pre_relaxation",
        lambda *a, **k: (1.0, {}),
    )
    monkeypatch.setattr(run_pipeline, "get_bulk_k_energy", lambda: -1.5)
    monkeypatch.setattr(
        run_pipeline,
        "analyze_relaxed_site_geometry",
        lambda *a, **k: {"k_o_distance_a": 2.7, "coordination_number": 4},
    )
    monkeypatch.setattr(
        run_pipeline,
        "_adsorption_validity",
        lambda *a, **k: ([], 2.5, 0.1, 1.0),
    )

    def fake_relax(structure, *args, **kwargs):
        return run_pipeline.OptimizationResult(
            structure=structure,
            energy=-10.0,
            success=True,
            converged=True,
            max_force=0.02,
            max_force_all=0.02,
            max_force_K_only=0.01,
        )

    monkeypatch.setattr(run_pipeline, "relax_k_only_fixed_framework", fake_relax)
    monkeypatch.setattr(run_pipeline, "relax_k_full_fixed_cell", fake_relax)
    summary = run_pipeline.run_single_k_adsorption(bare, "MOF", -8.0, resume=False)
    required = {
        "mof_name", "status", "flags", "E_ads_eV", "voltage_1K_V",
        "E_complex_eV", "E_bare_eV", "E_K_ref_eV_per_atom", "best_site_type",
        "K_O_min_dist_A", "K_coordination", "min_K_framework_dist_A",
        "framework_rmsd_A", "bond_preservation", "final_fmax_eV_per_A",
        "n_candidates_found", "n_prescreened", "n_full_relaxed",
        "all_site_energies", "wall_time_s",
    }
    assert required <= summary.keys()
    assert summary["E_ads_eV"] == -0.5
    assert summary["voltage_1K_V"] == 0.5


def test_unconstrained_force_measurement_clears_constraints_and_disables_application():
    seen = {}

    class FakeAtoms:
        def __init__(self):
            self.constraints = ["frozen"]
            self.calc = object()
            self.apply_constraint = None

        def copy(self):
            copied = FakeAtoms()
            copied.calc = self.calc
            return copied

        def set_constraint(self, constraints):
            self.constraints = constraints

        def get_forces(self, *, apply_constraint=True):
            self.apply_constraint = apply_constraint
            seen["apply_constraint"] = apply_constraint
            return np.array([[1.0, 0.0, 0.0], [0.25, 0.0, 0.0]])

    atoms = FakeAtoms()
    forces = run_pipeline._unconstrained_forces(atoms, "MOF")
    assert atoms.constraints == ["frozen"]
    assert seen["apply_constraint"] is False
    assert forces[:, 0].tolist() == [1.0, 0.25]


def test_full_relaxation_starts_without_constraints(monkeypatch):
    bare = _bare()
    bare.append("K", [0.5, 0.5, 0.5])
    seen = {}

    class FakeAtoms:
        def __init__(self):
            self.constraints = ["prescreen constraint"]
            self.calc = None

        def set_constraint(self, constraints):
            self.constraints = constraints
            seen["constraints"] = list(constraints)

        def copy(self):
            copied = FakeAtoms()
            copied.constraints = list(self.constraints)
            copied.calc = self.calc
            return copied

        def get_forces(self, *, apply_constraint=True):
            seen["apply_constraint"] = apply_constraint
            return np.zeros((3, 3))

        def get_potential_energy(self):
            return -1.0

        def get_chemical_symbols(self):
            return ["C", "O", "K"]

        def __len__(self):
            return 3

    class FakeOptimizer:
        def __init__(self, atoms):
            seen["optimizer_constraints"] = list(atoms.constraints)

        def run(self, **kwargs):
            return True

        def get_number_of_steps(self):
            return 1

    monkeypatch.setattr(run_pipeline.AseAtomsAdaptor, "get_atoms", lambda structure: FakeAtoms())
    monkeypatch.setattr(run_pipeline.AseAtomsAdaptor, "get_structure", lambda atoms: bare)
    monkeypatch.setattr(run_pipeline, "get_calculator", lambda: object())
    monkeypatch.setattr(run_pipeline, "FIRE", FakeOptimizer)
    monkeypatch.setattr(run_pipeline, "run_with_device_fallback", lambda fn, **kwargs: fn())

    run_pipeline.relax_k_full_fixed_cell(bare, "MOF")
    assert seen["constraints"] == []
    assert seen["optimizer_constraints"] == []
    assert seen["apply_constraint"] is False


def test_framework_frozen_flag_and_warning(caplog):
    bare = _bare()
    complex_structure = bare.copy()
    complex_structure.append("K", [0.5, 0.5, 0.5])
    with caplog.at_level("WARNING"):
        flags, _, rmsd, _ = run_pipeline._adsorption_validity(
            bare, complex_structure, 2, 0.01
        )
    assert rmsd < 1e-6
    assert "framework_frozen" in flags
    assert "framework did not move in full relaxation" in caplog.text
