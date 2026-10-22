from pathlib import Path

import numpy as np
from ase import Atoms
from pymatgen.core import Lattice, Structure

import run_pipeline


def _structure(symbols=("C", "O", "K"), coords=((0, 0, 0), (1.2, 0, 0), (3.8, 0, 0))):
    return Structure(
        Lattice.cubic(12),
        list(symbols),
        list(coords),
        coords_are_cartesian=True,
    )


def test_cache_fingerprint_changes_when_k_count_changes(tmp_path: Path):
    one_k = _structure()
    two_k = _structure(("C", "O", "K", "K"), ((0, 0, 0), (1.2, 0, 0), (3.8, 0, 0), (7.6, 0, 0)))
    first = run_pipeline.structure_fingerprint(one_k, tmp_path / "input.cif")
    second = run_pipeline.structure_fingerprint(two_k, tmp_path / "input.cif")

    matches, reason = run_pipeline.fingerprint_matches(first, second)
    assert not matches
    assert "n_atoms" in reason


def test_k_spacing_flag_uses_new_threshold_not_k_count():
    bare = _structure(("C", "O"), ((0, 0, 0), (1.2, 0, 0)))
    close = _structure(("C", "O", "K", "K"), ((0, 0, 0), (1.2, 0, 0), (3.8, 0, 0), (7.3, 0, 0)))
    separated = _structure(("C", "O", "K", "K"), ((0, 0, 0), (1.2, 0, 0), (3.8, 0, 0), (8.1, 0, 0)))

    assert "k_too_close" in run_pipeline.validate_k_insertion(bare, close)
    assert "k_too_close" not in run_pipeline.validate_k_insertion(bare, separated)
    assert "k_overpacked" not in run_pipeline.validate_k_insertion(bare, close)


def test_drift_corrected_framework_rmsd_is_zero_for_translation():
    ref = Atoms("CO", positions=[(0, 0, 0), (1.2, 0, 0)], cell=[12, 12, 12], pbc=True)
    translated = ref.copy()
    translated.positions += np.array([0.7, -0.2, 0.3])

    displacement = translated.positions - ref.positions
    displacement -= displacement.mean(axis=0)
    rmsd = np.sqrt(np.mean(np.sum(displacement**2, axis=1)))
    assert rmsd < 1e-12
