"""
Mechanical stability descriptors beyond composite MSI.

Adds coordination-number preservation for K ions and framework metals,
which correlates with linker dissociation in MOF MD literature.
"""

from __future__ import annotations

import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN


def k_coordination_numbers(
    structure: Structure,
    k_indices: list[int],
) -> dict[int, int]:
    """Return coordination number of each K site via CrystalNN."""
    nn = CrystalNN(distance_cutoffs=None, x_diff_weight=0.0, porous_adjustment=True)
    out: dict[int, int] = {}
    for idx in k_indices:
        try:
            info = nn.get_nn_info(structure, idx)
            out[idx] = len(info)
        except Exception:
            out[idx] = -1
    return out


def coordination_preservation_fraction(
    ref_structure: Structure,
    traj_positions: list[np.ndarray],
    k_indices: list[int],
    bond_cutoff_scale: float = 1.3,
    cells: list[np.ndarray] | None = None,
) -> float:
    """
    Fraction of K–O contacts (within covalent cutoff) preserved vs t=0.

    PBC-aware distance calculations are used for correct coordination tracking.
    """
    if not k_indices or not traj_positions:
        return float("nan")

    from ase.data import covalent_radii, atomic_numbers

    ref_pos = traj_positions[0]
    
    if cells is None:
        ref_cell = ref_structure.lattice.matrix
        cells = [ref_cell] * len(traj_positions)
    else:
        if len(cells) < len(traj_positions):
            cells = list(cells) + [cells[-1]] * (len(traj_positions) - len(cells))
        elif len(cells) > len(traj_positions):
            cells = cells[:len(traj_positions)]
        ref_cell = cells[0]

    o_indices = [i for i, s in enumerate(ref_structure) if s.specie.symbol == "O"]
    if not o_indices:
        return float("nan")

    inv_ref_cell = np.linalg.inv(ref_cell)

    # Initial K–O pairs within cutoff under PBC
    pairs: list[tuple[int, int, float]] = []
    for k_idx in k_indices:
        rk = covalent_radii[atomic_numbers["K"]]
        for o_idx in o_indices:
            ro = covalent_radii[atomic_numbers["O"]]
            cutoff = (rk + ro) * bond_cutoff_scale
            
            # PBC-aware distance calculation
            diff_cart = ref_pos[o_idx] - ref_pos[k_idx]
            diff_frac = diff_cart @ inv_ref_cell
            diff_frac -= np.round(diff_frac)
            d0 = float(np.linalg.norm(diff_frac @ ref_cell))
            
            if d0 <= cutoff:
                pairs.append((k_idx, o_idx, d0))

    if not pairs:
        return float("nan")

    # Invert all cells beforehand for efficiency
    inv_cells = [np.linalg.inv(cell) for cell in cells]

    preserved_counts: list[float] = []
    for t, pos in enumerate(traj_positions[1:], start=1):
        cell_t = cells[t]
        inv_cell_t = inv_cells[t]
        preserved = 0
        for k_idx, o_idx, d0 in pairs:
            # PBC-aware distance calculation
            diff_cart = pos[o_idx] - pos[k_idx]
            diff_frac = diff_cart @ inv_cell_t
            diff_frac -= np.round(diff_frac)
            d = float(np.linalg.norm(diff_frac @ cell_t))
            
            if d <= d0 * bond_cutoff_scale:
                preserved += 1
        preserved_counts.append(preserved / len(pairs))

    if not preserved_counts:
        return 1.0
    return float(np.mean(preserved_counts))
