"""Tests for strict bare-framework relaxation bookkeeping without CHGNet runs."""

from __future__ import annotations

import json

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from pymatgen.core import Lattice, Structure

import run_pipeline


def _structure(a: float = 10.0) -> Structure:
    return Structure(
        Lattice.cubic(a),
        ["C", "O"],
        [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]],
    )


def test_hard_force_acceptance_boundary():
    assert run_pipeline.classify_bare_relaxation(0.04) == "relaxed"
    assert run_pipeline.classify_bare_relaxation(0.06) == "not_relaxed"


def test_fallback_does_not_override_hard_force_acceptance():
    result = run_pipeline.BareRelaxationResult(
        status=run_pipeline.classify_bare_relaxation(0.06),
        final_fmax=0.06,
        used_fallback=True,
    )
    assert result.status == "not_relaxed"
    assert result.used_fallback is True


def test_minimum_distance_detects_broken_geometry():
    structure = Structure(
        Lattice.cubic(10.0),
        ["C", "O"],
        [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]],
    )
    np.testing.assert_allclose(
        run_pipeline.minimum_interatomic_distance(structure),
        0.5,
    )


def test_lattice_mismatch_thresholds():
    relaxed = _structure(10.5)
    k_structure = _structure(10.0)
    assert run_pipeline.lattice_cell_mismatch(relaxed, k_structure) is True

    relaxed = Structure(
        Lattice.from_parameters(10.1, 10.0, 10.0, 90.0, 90.0, 90.0),
        ["C", "O"],
        [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]],
    )
    assert run_pipeline.lattice_cell_mismatch(relaxed, k_structure) is False


def test_relax_summary_contains_required_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(run_pipeline, "OUTPUT_DIR", tmp_path)
    result = run_pipeline.BareRelaxationResult(
        status="relaxed",
        final_fmax=0.04,
        n_steps=12,
        min_dist_initial=1.2,
        energy_initial=-2.0,
        energy_final=-2.5,
        energy_per_atom_final=-1.25,
        volume_initial=1000.0,
        volume_final=1010.0,
        volume_change_percent=1.0,
        lattice_initial={"a": 10.0, "b": 10.0, "c": 10.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
        lattice_final={"a": 10.03, "b": 10.03, "c": 10.03, "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
    )
    path = run_pipeline._write_bare_relaxation_summary("MOF", result)
    with open(path, encoding="utf-8") as f:
        summary = json.load(f)

    required = {
        "mof_name",
        "status",
        "final_fmax_eV_per_A",
        "n_steps",
        "used_fallback",
        "hit_max_steps",
        "min_dist_initial_A",
        "energy_initial_eV",
        "energy_final_eV",
        "energy_per_atom_final_eV",
        "volume_initial_A3",
        "volume_final_A3",
        "volume_change_percent",
        "lattice_initial",
        "lattice_final",
        "k_cif_cell_mismatch",
        "wall_time_s",
    }
    assert required <= summary.keys()


def test_max_steps_with_accepted_force_is_relaxed_and_creates_output_dir(
    tmp_path, monkeypatch
):
    class ConstantForceCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results["energy"] = -1.0
            self.results["forces"] = np.tile([0.02, 0.0, 0.0], (len(atoms), 1))

    class MaxStepOptimizer:
        def __init__(self, target):
            self.steps = 0

        def run(self, fmax, steps):
            self.steps = steps

        def get_number_of_steps(self):
            return self.steps

    monkeypatch.setattr(run_pipeline, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(run_pipeline, "get_calculator", ConstantForceCalculator)
    monkeypatch.setattr(run_pipeline, "FIRE", MaxStepOptimizer, raising=False)

    # The relaxation function imports FIRE locally, so patch the ASE optimizer.
    import ase.optimize

    monkeypatch.setattr(ase.optimize, "FIRE", MaxStepOptimizer)
    result = run_pipeline.relax_bare_structure(
        _structure(),
        "max-step-mof",
        resume=False,
        max_steps=3,
        relax_cell=False,
    )

    output_dir = tmp_path / "structures" / "max-step-mof"
    assert output_dir.is_dir()
    assert result.status == "relaxed"
    assert result.hit_max_steps is True
    assert (output_dir / "bare_relaxed.cif").is_file()
    with open(output_dir / "relax_summary.json", encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["hit_max_steps"] is True
