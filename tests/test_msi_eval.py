import numpy as np
import pytest
from pymatgen.core import Structure, Lattice
from scientific.msi_analysis import coordination_preservation_fraction
from run_pipeline import mechanical_stability_index

def test_coordination_preservation_fraction_pbc():
    # Setup a cubic structure: a = 10.0
    # Oxygen at (0, 0, 0), Potassium at fractional (0.9, 0.0, 0.0) -> Cartesian (9.0, 0.0, 0.0)
    # PBC distance should be 1.0, which is within the K-O covalent cutoff
    lattice = Lattice.cubic(10.0)
    species = ["O", "K"]
    coords = [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]]
    structure = Structure(lattice, species, coords)

    k_indices = [1]
    
    # Trajectory positions:
    # t=0: same as reference
    # t=1: K moves to fractional (0.91, 0.0, 0.0) -> Cartesian (9.1, 0.0, 0.0) (dist = 0.9 under PBC)
    # t=2: K moves to fractional (0.80, 0.0, 0.0) -> Cartesian (8.0, 0.0, 0.0) (dist = 2.0 under PBC)
    
    pos_t0 = np.array([[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
    pos_t1_preserved = np.array([[0.0, 0.0, 0.0], [9.1, 0.0, 0.0]]) # dist = 0.9 <= 1.0 * 1.3
    pos_t2_broken = np.array([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]]) # dist = 2.0 > 1.0 * 1.3
    
    traj_positions = [pos_t0, pos_t1_preserved, pos_t2_broken]
    cells = [lattice.matrix] * 3

    # Run with PBC cells
    frac = coordination_preservation_fraction(
        ref_structure=structure,
        traj_positions=traj_positions,
        k_indices=k_indices,
        bond_cutoff_scale=1.3,
        cells=cells
    )
    
    # Expected preservation at t=1: 1.0 (preserved), at t=2: 0.0 (broken)
    # Mean preservation = (1.0 + 0.0) / 2 = 0.5
    assert np.isclose(frac, 0.5)


def test_mechanical_stability_index_renormalization():
    # If all metrics are finite and perfect
    msi_perfect = mechanical_stability_index(
        framework_rmsd=0.0,
        max_disp=0.0,
        bond_frac=1.0,
        energy_fluct=0.0,
        volume_fluct_frac=0.0,
        coordination_frac=1.0
    )
    assert np.isclose(msi_perfect, 1.0)

    # If coordination is nan, weights of other components (0.25, 0.25, 0.15, 0.10, 0.10) sum to 0.85.
    # If others are perfect, MSI should still be 1.0 (renormalized) rather than dragged down by coordination=0.5.
    msi_partial = mechanical_stability_index(
        framework_rmsd=0.0,
        max_disp=0.0,
        bond_frac=1.0,
        energy_fluct=0.0,
        volume_fluct_frac=0.0,
        coordination_frac=float("nan")
    )
    assert np.isclose(msi_partial, 1.0)
    
    # Check if a non-perfect metric scales correctly when coordination is nan
    # E.g. bond_frac = 0.8. Since bond weight is 0.25, with other perfect:
    # perfect: rmsd (0.25*1.0), disp (0.15*1.0), energy (0.10*1.0), volume (0.10*1.0)
    # bond: 0.25*0.8 = 0.20
    # Sum of weighted score = 0.25 + 0.15 + 0.10 + 0.10 + 0.20 = 0.80.
    # Renormalized weight sum = 0.85
    # Expected MSI = 0.80 / 0.85 = 0.941176
    msi_partial_imperfect = mechanical_stability_index(
        framework_rmsd=0.0,
        max_disp=0.0,
        bond_frac=0.8,
        energy_fluct=0.0,
        volume_fluct_frac=0.0,
        coordination_frac=float("nan")
    )
    assert np.isclose(msi_partial_imperfect, 0.80 / 0.85)
