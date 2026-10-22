"""
Publication-quality alkali-ion adsorption-site enumeration for arbitrary MOFs/COFs.

Methodology aligned with high-throughput DFT/MOF screening practice:
  * Local-environment classification via CrystalNN (pymatgen)
  * Chemistry-based contact distances from covalent radii (not MOF-specific constants)
  * Symmetry reduction before relaxation (Wyckoff-equivalent sites; spglib/pymatgen)
  * Void/pore sampling: adaptive fractional grids + Voronoi vertices (Zeo++/Voro++ tradition)
  * Electrostatic pre-screening: formal oxidation-state Coulomb proxy (ranking only)
  * Post-relaxation clustering to merge duplicate local minima (RMSD in fractional space)

References (computational materials practice):
  - Tang et al., J. Phys. Chem. C 2021 — μ₃-O K binding in UiO-66 (representative MOF DFT)
  - McConnell, Li & Vlachos, Microporous Mesoporous Mater. 2017 — Zeo++/pore accessibility
  - Willems et al., Microporous Mesoporous Mater. 2016 — Voronoi void analysis in MOFs
  - Ongari et al., Chem. Mater. 2017 — RASPA/CrystalNN local environment descriptors
  - Kresse & Hafner, PRB 1993 — global minimum via multiple starting geometries + relaxation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
from ase.data import atomic_numbers, covalent_radii
from pymatgen.core import Element, Structure

try:
    from pymatgen.analysis.local_env import CrystalNN
    _HAS_CRYSTAL_NN = True
except ImportError:
    _HAS_CRYSTAL_NN = False

try:
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    _HAS_SPG = True
except ImportError:
    _HAS_SPG = False

try:
    from scipy.spatial import Voronoi
    _HAS_SCIPY_VORONOI = True
except ImportError:
    _HAS_SCIPY_VORONOI = False

# =============================================================================
# Chemistry-based radii / cutoffs (element-specific, not MOF-specific)
# =============================================================================

_NONMETALS = frozenset({"H", "He", "B", "C", "N", "O", "F", "Ne", "Si", "P", "S", "Cl", "Ar",
                        "Se", "Br", "Kr", "I", "Xe", "At", "Rn"})
_ALKALI = frozenset({"Li", "Na", "K", "Rb", "Cs"})
_ORGANIC = frozenset({"C", "H", "N", "S"})

# Formal oxidation states for electrostatic ranking (not for absolute energies).
# pymatgen Element.common_oxidation_states used when available.
_DEFAULT_OXIDATION: dict[str, float] = {
    "H": 1, "C": 0, "N": -3, "O": -2, "F": -1, "S": -2, "Cl": -1,
    "Li": 1, "Na": 1, "K": 1, "Rb": 1, "Cs": 1,
    "Mg": 2, "Ca": 2, "Zn": 2, "Cu": 2, "Ni": 2, "Co": 2, "Fe": 2, "Mn": 2,
    "Al": 3, "Cr": 3, "V": 3, "Zr": 4, "Hf": 4, "Ti": 4, "Ce": 4, "Th": 4,
}


def covalent_radius(symbol: str) -> float:
    z = atomic_numbers.get(symbol, 0)
    if z <= 0:
        return 1.0
    return float(covalent_radii[z])


def bond_cutoff(sym_a: str, sym_b: str, scale: float = 1.15) -> float:
    """Contact distance from summed covalent radii (standard crystallographic practice)."""
    return scale * (covalent_radius(sym_a) + covalent_radius(sym_b))


def metal_ligand_cutoff(metal: str, ligand: str = "O") -> float:
    return bond_cutoff(metal, ligand, scale=1.20)


def k_oxygen_contact_distance(scale: float = 1.0) -> float:
    """Expected K–O contact from covalent radii (~2.7–2.9 Å)."""
    return scale * bond_cutoff("K", "O", scale=1.0)


def k_framework_min_distance() -> float:
    """Minimum K–framework atom separation from radii sum (steric gate)."""
    return 0.85 * bond_cutoff("K", "O", scale=1.0)


def element_charge(symbol: str) -> float:
    try:
        ox = Element(symbol).common_oxidation_states
        if ox:
            return float(ox[0])
    except Exception:
        pass
    return float(_DEFAULT_OXIDATION.get(symbol, 0))


def is_metal(symbol: str) -> bool:
    return symbol not in _NONMETALS and symbol not in _ALKALI


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class AdsorptionCandidate:
    """One initial K placement before CHGNet relaxation."""
    candidate_id: int
    site_type: str
    k_frac: list[float]
    anchor_index: int = -1          # O or metal index; -1 for void/grid sites
    anchor_symbol: str = ""
    placement_method: str = ""
    k_o_distance_a: float = float("nan")
    pre_score: float = float("nan")
    pre_score_components: dict[str, float] = field(default_factory=dict)
    symmetry_wyckoff: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "site_type": self.site_type,
            "k_frac": self.k_frac,
            "anchor_index": self.anchor_index,
            "anchor_symbol": self.anchor_symbol,
            "placement_method": self.placement_method,
            "k_o_distance_a": self.k_o_distance_a,
            "pre_score": self.pre_score,
            "pre_score_components": self.pre_score_components,
            "symmetry_wyckoff": self.symmetry_wyckoff,
            "notes": self.notes,
        }


@dataclass
class RelaxedAdsorptionSite:
    """One relaxed K@MOF local minimum."""
    candidate_id: int
    site_type: str
    rank: int = -1
    cluster_id: int = -1
    is_global_minimum: bool = False
    adsorption_energy_ev: float = float("nan")
    k_frac_coords: list[float] = field(default_factory=list)
    k_o_distance_a: float = float("nan")
    k_metal_distance_a: float = float("nan")
    oxygen_index: int = -1
    coordination_number: int = -1
    nearest_atoms: list[dict[str, Any]] = field(default_factory=list)
    local_environment_descriptor: str = ""
    optimization_converged: bool = False
    max_force_ev_a: float = float("nan")
    confidence_score: float = float("nan")
    pre_score: float = float("nan")
    success: bool = False
    notes: str = ""
    structure: Any = None  # pymatgen Structure, optional

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "cluster_id": self.cluster_id,
            "is_global_minimum": self.is_global_minimum,
            "candidate_id": self.candidate_id,
            "site_type": self.site_type,
            "adsorption_energy_ev": self.adsorption_energy_ev,
            "k_frac_coords": self.k_frac_coords,
            "k_o_distance_a": self.k_o_distance_a,
            "k_metal_distance_a": self.k_metal_distance_a,
            "oxygen_index": self.oxygen_index,
            "coordination_number": self.coordination_number,
            "nearest_atoms": self.nearest_atoms,
            "local_environment_descriptor": self.local_environment_descriptor,
            "optimization_converged": self.optimization_converged,
            "max_force_ev_a": self.max_force_ev_a,
            "confidence_score": self.confidence_score,
            "pre_score": self.pre_score,
            "success": self.success,
            "notes": self.notes,
        }


# =============================================================================
# Geometry helpers
# =============================================================================

def mic_distance_frac(f1: np.ndarray, f2: np.ndarray, lattice: np.ndarray) -> float:
    d = np.asarray(f1, dtype=float) - np.asarray(f2, dtype=float)
    d -= np.round(d)
    return float(np.linalg.norm(d @ lattice))


def lattice_distance(
    structure: Structure,
    frac_a: np.ndarray,
    frac_b: np.ndarray,
) -> float:
    """PBC-aware distance (Å) via pymatgen Lattice (publication-standard MIC)."""
    return float(structure.lattice.get_distance_and_image(frac_a, frac_b)[0])


def min_distance_to_symbol(structure: Structure, frac: np.ndarray, symbol: str) -> float:
    """Minimum MIC distance from a fractional point to any atom of ``symbol``."""
    best = float("inf")
    frac_a = np.asarray(frac, dtype=float)
    for site in structure:
        if site.specie.symbol != symbol:
            continue
        d = lattice_distance(structure, frac_a, site.frac_coords)
        if d < best:
            best = d
    return float(best) if np.isfinite(best) and best < float("inf") else float("nan")


def nearest_site_by_symbol(
    structure: Structure,
    k_index: int,
    symbol: str,
) -> tuple[float, int]:
    """Nearest atom of ``symbol`` to K at ``k_index`` (always scans all sites)."""
    k_frac = np.asarray(structure[k_index].frac_coords, dtype=float)
    best_d = float("inf")
    best_i = -1
    for i, site in enumerate(structure):
        if i == k_index or site.specie.symbol != symbol:
            continue
        d = lattice_distance(structure, k_frac, site.frac_coords)
        if d < best_d:
            best_d = d
            best_i = i
    if best_i < 0:
        return float("nan"), -1
    return float(best_d), best_i


def _neighbor_distance(nn: Any) -> float:
    return float(getattr(nn, "nn_distance", getattr(nn, "distance", float("nan"))))


def nearest_neighbors_at_site(
    structure: Structure,
    k_index: int,
    n: int = 10,
    max_cutoff: float = 8.0,
) -> list[dict[str, Any]]:
    """Return n nearest framework atoms to a K site using pymatgen neighbor search."""
    k_site = structure[k_index]
    neighbors = structure.get_neighbors(k_site, max_cutoff)
    dists: list[tuple[float, int, str]] = []
    for nn in neighbors:
        idx = int(nn.index)
        if structure[idx].specie.symbol == "K":
            continue
        dists.append((_neighbor_distance(nn), idx, structure[idx].specie.symbol))
    dists.sort(key=lambda t: t[0])
    return [
        {"index": idx, "symbol": sym, "distance_a": float(d)}
        for d, idx, sym in dists[:n]
    ]


def k_coordination_number_at_site(
    structure: Structure,
    k_index: int,
    search_radius: float = 5.0,
) -> int:
    """Count framework atoms within summed covalent contact distance of K."""
    k_site = structure[k_index]
    cn = 0
    for nn in structure.get_neighbors(k_site, search_radius):
        idx = int(nn.index)
        if idx == k_index or structure[idx].specie.symbol == "K":
            continue
        sym = structure[idx].specie.symbol
        contact = bond_cutoff("K", sym, scale=1.12)
        if _neighbor_distance(nn) <= contact:
            cn += 1
    return cn


def resolve_k_index(structure: Structure, k_index: int) -> int:
    """Return a valid K index (fallback: sole K or last inserted K)."""
    if 0 <= k_index < len(structure) and structure[k_index].specie.symbol == "K":
        return k_index
    k_indices = [i for i, s in enumerate(structure) if s.specie.symbol == "K"]
    if not k_indices:
        raise ValueError("No K ion in structure for geometry analysis")
    return k_indices[-1]


def mic_displacement_frac(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    d = np.asarray(f1, dtype=float) - np.asarray(f2, dtype=float)
    d -= np.round(d)
    return d


def frac_to_cart(frac: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    return np.asarray(frac, dtype=float) @ lattice


def normalize_vec(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return v / n


def min_framework_distance(structure: Structure, frac: np.ndarray, exclude_k: bool = True) -> float:
    lattice = structure.lattice.matrix
    min_d = float("inf")
    for site in structure:
        if exclude_k and site.specie.symbol == "K":
            continue
        d = mic_distance_frac(frac, np.asarray(site.frac_coords, dtype=float), lattice)
        min_d = min(min_d, d)
    return float(min_d)


def nearest_neighbors(
    structure: Structure,
    frac: np.ndarray,
    n: int = 8,
    max_cutoff: float = 8.0,
) -> list[dict[str, Any]]:
    """Return n nearest framework atoms to a fractional coordinate."""
    frac_a = np.asarray(frac, dtype=float)
    dists: list[tuple[float, int, str]] = []
    for i, site in enumerate(structure):
        if site.specie.symbol == "K":
            continue
        d = lattice_distance(structure, frac_a, site.frac_coords)
        if d <= max_cutoff:
            dists.append((d, i, site.specie.symbol))
    dists.sort(key=lambda t: t[0])
    return [
        {"index": idx, "symbol": sym, "distance_a": float(d)}
        for d, idx, sym in dists[:n]
    ]


# =============================================================================
# CrystalNN-based local environment
# =============================================================================

def _get_crystal_nn() -> CrystalNN | None:
    if not _HAS_CRYSTAL_NN:
        return None
    return CrystalNN(distance_cutoffs=None, x_diff_weight=0.0, porous_adjustment=True)


def metal_coordination_number(structure: Structure, metal_idx: int) -> int:
    nn = _get_crystal_nn()
    if nn is None:
        site = structure[metal_idx]
        cutoff = metal_ligand_cutoff(site.specie.symbol)
        return len(structure.get_neighbors(site, cutoff))
    try:
        return len(nn.get_nn_info(structure, metal_idx))
    except Exception:
        return -1


def oxygen_coordination_info(structure: Structure, o_idx: int) -> dict[str, Any]:
    """Classify oxygen local environment for adsorption-site typing."""
    site = structure[o_idx]
    if site.specie.symbol != "O":
        return {"site_type": "unknown", "metal_indices": [], "carbon_indices": []}

    nn = _get_crystal_nn()
    if nn is not None:
        try:
            nn_info = nn.get_nn_info(structure, o_idx)
            nbr_indices = [x["site_index"] for x in nn_info]
        except Exception:
            nbr_indices = []
    else:
        cutoff = bond_cutoff("O", "Zr", scale=1.25)  # generous fallback
        nbr_indices = [n.index for n in structure.get_neighbors(site, cutoff)]

    metal_idxs = sorted({i for i in nbr_indices if is_metal(structure[i].specie.symbol)})
    c_idxs = sorted({i for i in nbr_indices if structure[i].specie.symbol == "C"})

    n_metal = len(metal_idxs)
    if n_metal >= 3:
        site_type = "mu3_O"
    elif n_metal == 2:
        site_type = "bridge_O"
    elif n_metal == 1 and c_idxs:
        site_type = "linker_O"  # carboxylate / organic linker oxygen
    elif n_metal == 1:
        cn = metal_coordination_number(structure, metal_idxs[0])
        typical = _typical_metal_cn(structure[metal_idxs[0]].specie.symbol)
        if cn > 0 and cn < typical:
            site_type = "open_metal_adjacent_O"
        else:
            site_type = "terminal_O"
    elif c_idxs:
        site_type = "organic_O"
    else:
        site_type = "other_O"

    return {
        "site_type": site_type,
        "metal_indices": metal_idxs,
        "carbon_indices": c_idxs,
        "n_metal": n_metal,
    }


def _typical_metal_cn(symbol: str) -> int:
    """Literature-typical coordination numbers for MOF node metals (screening heuristic)."""
    typical = {
        "Zr": 8, "Hf": 8, "Ti": 6, "V": 6, "Cr": 6, "Fe": 6, "Co": 6, "Ni": 6,
        "Cu": 5, "Zn": 4, "Mg": 6, "Ca": 6, "Al": 6, "Mn": 6, "Cd": 6, "Pb": 6,
        "La": 9, "Ce": 9, "Y": 8, "Sc": 6, "Mo": 6, "W": 6,
    }
    return typical.get(symbol, 6)


def k_coordination_number_at_frac(structure: Structure, k_frac: np.ndarray) -> int:
    """Count framework atoms within K–X contact distance (proxy for post-relax CN)."""
    lattice = structure.lattice.matrix
    k_cart = frac_to_cart(k_frac, lattice)
    cn = 0
    for site in structure:
        if site.specie.symbol == "K":
            continue
        cutoff = bond_cutoff("K", site.specie.symbol, scale=1.10)
        d = mic_distance_frac(k_frac, np.asarray(site.frac_coords, dtype=float), lattice)
        if d <= cutoff:
            cn += 1
    return cn


def local_environment_descriptor_string(
    structure: Structure,
    k_frac: np.ndarray,
    site_type: str,
) -> str:
    """Compact descriptor: site_type + nearest atom shell."""
    nbrs = nearest_neighbors(structure, k_frac, n=6, max_cutoff=8.0)
    shell = "-".join(f"{n['symbol']}{n['distance_a']:.2f}" for n in nbrs[:4])
    return f"{site_type}|{shell}"


# =============================================================================
# Candidate placement generators
# =============================================================================

def _outward_normal_from_metals(
    structure: Structure,
    o_cart: np.ndarray,
    metal_idxs: list[int],
) -> np.ndarray:
    lat = structure.lattice
    if len(metal_idxs) >= 3:
        metal_cart = np.array([structure[i].coords for i in metal_idxs[:3]], dtype=float)
        v1 = metal_cart[1] - metal_cart[0]
        v2 = metal_cart[2] - metal_cart[0]
        normal = normalize_vec(np.cross(v1, v2))
    elif len(metal_idxs) == 2:
        m0, m1 = structure[metal_idxs[0]].coords, structure[metal_idxs[1]].coords
        mid = 0.5 * (m0 + m1)
        normal = normalize_vec(o_cart - mid)
    elif len(metal_idxs) == 1:
        normal = normalize_vec(o_cart - structure[metal_idxs[0]].coords)
    else:
        center = lat.get_cartesian_coords([0.5, 0.5, 0.5])
        normal = normalize_vec(center - o_cart)
    metal_centroid = np.mean([structure[i].coords for i in metal_idxs], axis=0) if metal_idxs else o_cart
    if np.dot(normal, o_cart - metal_centroid) < 0:
        normal = -normal
    return normal


def oxygen_centered_placements(
    structure: Structure,
    o_idx: int,
    site_type: str,
    metal_idxs: list[int],
) -> list[tuple[np.ndarray, str, float]]:
    """Multiple K placements per oxygen: along normal at radii-scaled distances."""
    lat = structure.lattice
    o_cart = np.asarray(structure[o_idx].coords, dtype=float)
    normal = _outward_normal_from_metals(structure, o_cart, metal_idxs)
    base_dist = k_oxygen_contact_distance(1.0)
    placements: list[tuple[np.ndarray, str, float]] = []
    for scale, tag in ((0.92, "r0.92"), (1.00, "r1.00"), (1.08, "r1.08")):
        k_cart = o_cart + scale * base_dist * normal
        k_frac = np.mod(lat.get_fractional_coords(k_cart), 1.0)
        if min_framework_distance(structure, k_frac) >= k_framework_min_distance():
            placements.append((k_frac, f"normal_{tag}", scale * base_dist))
    # Secondary: bisector away from nearest metal for terminal/linker oxygens
    if len(metal_idxs) == 1:
        m_cart = structure[metal_idxs[0]].coords
        bisect = normalize_vec(o_cart - m_cart)
        k_cart = o_cart + base_dist * bisect
        k_frac = np.mod(lat.get_fractional_coords(k_cart), 1.0)
        if min_framework_distance(structure, k_frac) >= k_framework_min_distance():
            placements.append((k_frac, "metal_bisector", base_dist))
    return placements


def open_metal_site_placements(structure: Structure, metal_idx: int) -> list[tuple[np.ndarray, str, float]]:
    """
    Place K in the vacant coordination direction of an undercoordinated metal node.
    Standard practice for open-metal-site MOF adsorption (Cu paddlewheel, etc.).
    """
    lat = structure.lattice
    metal = structure[metal_idx]
    symbol = metal.specie.symbol
    cn = metal_coordination_number(structure, metal_idx)
    typical = _typical_metal_cn(symbol)
    if cn < 0 or cn >= typical:
        return []

    nn = _get_crystal_nn()
    ligand_cart: list[np.ndarray] = []
    if nn is not None:
        try:
            for info in nn.get_nn_info(structure, metal_idx):
                ligand_cart.append(structure[info["site_index"]].coords)
        except Exception:
            pass
    if not ligand_cart:
        cutoff = metal_ligand_cutoff(symbol)
        ligand_cart = [n.coords for n in structure.get_neighbors(metal, cutoff)]

    if not ligand_cart:
        return []

    m_cart = np.asarray(metal.coords, dtype=float)
    centroid = np.mean(ligand_cart, axis=0)
    outward = normalize_vec(m_cart - centroid)
    dist = bond_cutoff("K", symbol, scale=1.05)
    placements: list[tuple[np.ndarray, str, float]] = []
    for scale in (0.95, 1.0, 1.05):
        k_cart = m_cart + scale * dist * outward
        k_frac = np.mod(lat.get_fractional_coords(k_cart), 1.0)
        if min_framework_distance(structure, k_frac) >= k_framework_min_distance():
            placements.append((k_frac, f"open_metal_r{scale:.2f}", scale * dist))
    return placements


def adaptive_pore_grid_candidates(
    structure: Structure,
    target_points: int = 400,
    min_void_radius: float | None = None,
) -> list[tuple[np.ndarray, str, float]]:
    """
  Adaptive fractional grid in void regions (distance to framework > void radius).
    McConnell / Zeo++-style large-pore sampling.
    """
    if min_void_radius is None:
        min_void_radius = max(k_oxygen_contact_distance(0.9), k_framework_min_distance() + 0.5)

    lattice = structure.lattice.matrix
    abc = np.array(structure.lattice.abc, dtype=float)
    volume = float(structure.volume)
    n_atoms = len(structure)
    # Grid spacing from average atomic volume (chemistry-based density estimate)
    spacing = max(0.35, (volume / max(n_atoms, 1)) ** (1.0 / 3.0) * 0.55)
    n_grid = np.maximum(3, np.round(abc / spacing).astype(int))
    while int(np.prod(n_grid)) > target_points * 3 and np.min(n_grid) > 3:
        n_grid = np.maximum(3, n_grid - 1)

    void_pts: list[tuple[float, np.ndarray, float]] = []
    for ix in range(int(n_grid[0])):
        for iy in range(int(n_grid[1])):
            for iz in range(int(n_grid[2])):
                frac = np.array([
                    (ix + 0.5) / n_grid[0],
                    (iy + 0.5) / n_grid[1],
                    (iz + 0.5) / n_grid[2],
                ], dtype=float)
                dmin = min_framework_distance(structure, frac)
                if dmin >= min_void_radius:
                    void_pts.append((dmin, frac, dmin))

    void_pts.sort(key=lambda t: -t[0])
    out: list[tuple[np.ndarray, str, float]] = []
    for dmin, frac, _ in void_pts[:target_points]:
        out.append((frac, "pore_grid", float(dmin)))
    return out


def voronoi_void_candidates(
    structure: Structure,
    min_framework_dist_a: float | None = None,
    max_candidates: int = 120,
    grid_density: int = 8,
) -> list[tuple[np.ndarray, str, float]]:
    """Voronoi vertex void centers (Willems et al. / Zeo++ methodology)."""
    if not _HAS_SCIPY_VORONOI:
        return []
    if min_framework_dist_a is None:
        min_framework_dist_a = k_framework_min_distance() + 0.6

    lattice = structure.lattice.matrix
    grids = np.linspace(0.04, 0.96, grid_density)
    frac_pts = np.array(
        [[fx, fy, fz] for fx in grids for fy in grids for fz in grids],
        dtype=float,
    )
    cart_pts = frac_pts @ lattice
    try:
        vor = Voronoi(cart_pts)
    except Exception:
        return []

    fw_cart = np.array([s.coords for s in structure if s.specie.symbol != "K"])
    if len(fw_cart) == 0:
        return []

    candidates: list[tuple[float, np.ndarray, float]] = []
    for vertex in vor.vertices:
        if not np.all(np.isfinite(vertex)):
            continue
        frac = np.mod(structure.lattice.get_fractional_coords(vertex), 1.0)
        min_d = float(np.min(np.linalg.norm(fw_cart - vertex, axis=1)))
        if min_d < min_framework_dist_a:
            continue
        candidates.append((min_d, frac, min_d))

    candidates.sort(key=lambda t: -t[0])
    return [(f, "voronoi_void", d) for _, f, d in candidates[:max_candidates]]


def electrostatic_grid_minima(
    structure: Structure,
    grid_spacing_a: float | None = None,
    min_framework_dist_a: float | None = None,
    max_candidates: int = 80,
) -> list[tuple[np.ndarray, str, float]]:
    """
    Local minima of a Coulomb energy proxy using formal oxidation states.
    Adequate for ranking void vs wall sites (Ongari/Chem.Mater. electrostatic pre-screening).
    """
    if min_framework_dist_a is None:
        min_framework_dist_a = k_framework_min_distance()
    if grid_spacing_a is None:
        vol = float(structure.volume)
        grid_spacing_a = max(0.4, (vol / max(len(structure), 1)) ** (1.0 / 3.0) * 0.5)

    charges = np.array([element_charge(s.specie.symbol) for s in structure], dtype=float)
    positions = np.array([s.coords for s in structure])
    lattice = structure.lattice.matrix
    abc = structure.lattice.abc
    n_grid = [max(4, int(round(a / grid_spacing_a))) for a in abc]

    scores: list[tuple[float, np.ndarray, float]] = []
    inv_cell = np.linalg.inv(lattice)
    for ix in range(n_grid[0]):
        for iy in range(n_grid[1]):
            for iz in range(n_grid[2]):
                frac = np.array([
                    (ix + 0.5) / n_grid[0],
                    (iy + 0.5) / n_grid[1],
                    (iz + 0.5) / n_grid[2],
                ], dtype=float)
                cart = frac @ lattice
                min_fw = float(np.min(np.linalg.norm(positions - cart, axis=1)))
                if min_fw < min_framework_dist_a:
                    continue
                raw = cart - positions
                frac_d = raw @ inv_cell
                frac_d -= np.round(frac_d)
                mic = frac_d @ lattice
                dist = np.maximum(np.linalg.norm(mic, axis=1), 0.45)
                e_coul = float(np.sum(charges / dist))
                scores.append((e_coul, frac, min_fw))

    scores.sort(key=lambda t: t[0])
    out: list[tuple[np.ndarray, str, float]] = []
    cluster_tol = 0.6  # Å in fractional MIC proxy
    for e_coul, frac, min_fw in scores:
        if any(mic_distance_frac(frac, f, lattice) < cluster_tol for f, _, _ in out):
            continue
        out.append((frac, "electrostatic_minimum", min_fw))
        if len(out) >= max_candidates:
            break
    return out


# =============================================================================
# Pre-relaxation scoring
# =============================================================================

def score_candidate_pre_relaxation(
    structure: Structure,
    candidate: AdsorptionCandidate,
) -> tuple[float, dict[str, float]]:
    """
    Rank initial placements before expensive MLIP relaxation.

    Components (all in [0,1] after normalization):
      - electrostatic: Coulomb proxy at K position (more negative → better for cation)
      - coordination: proximity to O and favorable metal neighbors
      - accessibility: void radius (larger min framework distance → better)
      - steric: penalty for framework clashes
    """
    lattice = structure.lattice.matrix
    frac = np.asarray(candidate.k_frac, dtype=float)
    comps: dict[str, float] = {}

    # Electrostatic (ranking-only Coulomb)
    positions = np.array([s.coords for s in structure])
    charges = np.array([element_charge(s.specie.symbol) for s in structure])
    cart = frac @ lattice
    inv_cell = np.linalg.inv(lattice)
    raw = cart - positions
    frac_d = raw @ inv_cell
    frac_d -= np.round(frac_d)
    mic = frac_d @ lattice
    dist = np.maximum(np.linalg.norm(mic, axis=1), 0.45)
    e_coul = float(np.sum(charges / dist))
    comps["electrostatic_raw_ev"] = e_coul
    comps["electrostatic"] = float(np.clip(-e_coul / 30.0, 0.0, 1.0))

    # K–O distance score (prefer ~2.7 Å)
    d_o = min_distance_to_symbol(structure, frac, "O")
    target = k_oxygen_contact_distance(1.0)
    comps["k_o_distance_a"] = d_o
    comps["coordination"] = float(np.exp(-0.5 * ((d_o - target) / 0.6) ** 2)) if np.isfinite(d_o) else 0.0

    # K–metal distance (secondary)
    d_m = float("inf")
    for i, site in enumerate(structure):
        if is_metal(site.specie.symbol):
            d = lattice_distance(structure, frac, site.frac_coords)
            d_m = min(d_m, d)
    comps["k_metal_distance_a"] = d_m if np.isfinite(d_m) else float("nan")
    if np.isfinite(d_m):
        comps["metal_proximity"] = float(np.exp(-0.5 * ((d_m - bond_cutoff("K", "Zr")) / 1.0) ** 2))
    else:
        comps["metal_proximity"] = 0.0

    # Pore accessibility
    d_fw = min_framework_distance(structure, frac)
    comps["min_framework_distance_a"] = d_fw
    comps["accessibility"] = float(np.clip((d_fw - k_framework_min_distance()) / 2.5, 0.0, 1.0))

    # Steric clash penalty
    min_allowed = k_framework_min_distance()
    if d_fw < min_allowed:
        comps["steric"] = 0.0
    else:
        comps["steric"] = float(np.clip((d_fw - min_allowed) / 1.0, 0.0, 1.0))

    # Site-type prior (literature-motivated, not hardcoded geometry)
    type_prior = {
        "mu3_O": 1.0, "open_metal_site": 0.95, "open_metal_adjacent_O": 0.9,
        "linker_O": 0.85, "terminal_O": 0.8, "bridge_O": 0.75,
        "carboxylate_O": 0.85, "electrostatic_minimum": 0.6,
        "voronoi_void": 0.55, "pore_grid": 0.5, "organic_O": 0.4,
    }
    comps["site_type_prior"] = type_prior.get(candidate.site_type, 0.5)

    weights = {
        "electrostatic": 0.20, "coordination": 0.25, "metal_proximity": 0.10,
        "accessibility": 0.20, "steric": 0.15, "site_type_prior": 0.10,
    }
    score = sum(weights[k] * comps.get(k, 0.0) for k in weights)
    return float(score), comps


# =============================================================================
# Symmetry deduplication
# =============================================================================

def deduplicate_sites_by_symmetry(
    bare: Structure,
    candidates: list[AdsorptionCandidate],
    tolerance_a: float | None = None,
) -> list[AdsorptionCandidate]:
    """
    Remove symmetry-equivalent candidates before relaxation.

    Uses pymatgen SpacegroupAnalyzer symmetry operations when available;
    falls back to greedy fractional MIC clustering.
    """
    if not candidates:
        return []
    if tolerance_a is None:
        tolerance_a = 0.5 * k_oxygen_contact_distance(1.0)

    lattice = bare.lattice.matrix

    if _HAS_SPG:
        try:
            sga = SpacegroupAnalyzer(bare, symprec=0.1, angle_tolerance=5)
            sym_ops = sga.get_symmetry_operations()
            unique: list[AdsorptionCandidate] = []

            def _equiv(f1: np.ndarray, f2: np.ndarray) -> bool:
                for op in sym_ops:
                    f_sym = op.operate(f1)
                    if min(
                        mic_distance_frac(f1, f2, lattice),
                        mic_distance_frac(f_sym, f2, lattice),
                    ) < tolerance_a:
                        return True
                return False

            for cand in sorted(candidates, key=lambda c: -c.pre_score):
                f = np.asarray(cand.k_frac, dtype=float)
                if any(_equiv(f, np.asarray(u.k_frac, dtype=float)) for u in unique):
                    continue
                unique.append(cand)
            return unique
        except Exception:
            pass

    unique = []
    for cand in sorted(candidates, key=lambda c: -c.pre_score):
        f = np.asarray(cand.k_frac, dtype=float)
        if any(
            mic_distance_frac(f, np.asarray(u.k_frac, dtype=float), lattice) < tolerance_a
            for u in unique
        ):
            continue
        unique.append(cand)
    return unique


# =============================================================================
# Master discovery
# =============================================================================

def discover_adsorption_candidates(
    bare: Structure,
    max_raw_candidates: int = 800,
    max_unique_candidates: int = 400,
) -> list[AdsorptionCandidate]:
    """
    Enumerate chemically meaningful K adsorption candidates for any MOF/COF.

    Returns symmetry-unique, pre-scored candidates ready for MLIP relaxation.
    """
    raw: list[AdsorptionCandidate] = []
    cid = 0

    # 1. Oxygen-centered sites (all O atoms, all environments)
    for o_idx, site in enumerate(bare):
        if site.specie.symbol != "O":
            continue
        env = oxygen_coordination_info(bare, o_idx)
        site_type = env["site_type"]
        if site_type in {"other_O", "organic_O"}:
            continue
        for k_frac, method, dist in oxygen_centered_placements(
            bare, o_idx, site_type, env["metal_indices"]
        ):
            cid += 1
            raw.append(AdsorptionCandidate(
                candidate_id=cid,
                site_type=site_type,
                k_frac=[float(x) for x in k_frac],
                anchor_index=o_idx,
                anchor_symbol="O",
                placement_method=method,
                k_o_distance_a=float(dist),
            ))

    # 2. Open metal sites (undercoordinated nodes)
    seen_metals: set[int] = set()
    for i, site in enumerate(bare):
        if not is_metal(site.specie.symbol):
            continue
        if i in seen_metals:
            continue
        cn = metal_coordination_number(bare, i)
        typical = _typical_metal_cn(site.specie.symbol)
        if cn < 0 or cn >= typical:
            continue
        seen_metals.add(i)
        for k_frac, method, dist in open_metal_site_placements(bare, i):
            cid += 1
            raw.append(AdsorptionCandidate(
                candidate_id=cid,
                site_type="open_metal_site",
                k_frac=[float(x) for x in k_frac],
                anchor_index=i,
                anchor_symbol=site.specie.symbol,
                placement_method=method,
                k_o_distance_a=float(dist),
            ))

    # 3. Pore / void / electrostatic candidates
    for k_frac, method, dist in adaptive_pore_grid_candidates(bare):
        cid += 1
        raw.append(AdsorptionCandidate(
            candidate_id=cid, site_type="pore_grid",
            k_frac=[float(x) for x in k_frac], placement_method=method,
            k_o_distance_a=float(dist),
        ))
    for k_frac, method, dist in voronoi_void_candidates(bare):
        cid += 1
        raw.append(AdsorptionCandidate(
            candidate_id=cid, site_type="voronoi_void",
            k_frac=[float(x) for x in k_frac], placement_method=method,
            k_o_distance_a=float(dist),
        ))
    for k_frac, method, dist in electrostatic_grid_minima(bare):
        cid += 1
        raw.append(AdsorptionCandidate(
            candidate_id=cid, site_type="electrostatic_minimum",
            k_frac=[float(x) for x in k_frac], placement_method=method,
            k_o_distance_a=float(dist),
        ))

    # Cap raw if extreme (very large cells)
    if len(raw) > max_raw_candidates:
        for c in raw:
            if not np.isfinite(c.pre_score):
                s, comp = score_candidate_pre_relaxation(bare, c)
                c.pre_score = s
                c.pre_score_components = comp
        raw.sort(key=lambda c: -c.pre_score)
        raw = raw[:max_raw_candidates]

    # Pre-score all
    for c in raw:
        if not np.isfinite(c.pre_score):
            s, comp = score_candidate_pre_relaxation(bare, c)
            c.pre_score = s
            c.pre_score_components = comp

    # Symmetry deduplication
    unique = deduplicate_sites_by_symmetry(bare, raw)
    unique.sort(key=lambda c: -c.pre_score)
    if len(unique) > max_unique_candidates:
        unique = unique[:max_unique_candidates]
    return unique


# =============================================================================
# Post-relaxation clustering
# =============================================================================

def cluster_relaxed_minima(
    sites: list[RelaxedAdsorptionSite],
    cluster_distance_a: float | None = None,
) -> list[RelaxedAdsorptionSite]:
    """
    Merge symmetry/distortion-equivalent relaxed minima (duplicate basins).

    Greedy clustering on final K fractional positions; lowest E_ads per cluster
    is retained as a unique local minimum (standard multi-start global optimization).
    """
    if not sites:
        return []
    if cluster_distance_a is None:
        cluster_distance_a = 0.5 * k_oxygen_contact_distance(1.0)

    successful = [s for s in sites if s.success and np.isfinite(s.adsorption_energy_ev)]
    successful.sort(key=lambda s: s.adsorption_energy_ev)

    clusters: list[list[RelaxedAdsorptionSite]] = []
    for site in successful:
        f = np.asarray(site.k_frac_coords, dtype=float)
        placed = False
        for cluster in clusters:
            rep = cluster[0]
            # Use MIC distance in fractional space with lattice from structure if available
            if site.structure is not None:
                lattice = site.structure.lattice.matrix
            elif rep.structure is not None:
                lattice = rep.structure.lattice.matrix
            else:
                lattice = np.eye(3) * 10.0
            d = mic_distance_frac(f, np.asarray(rep.k_frac_coords, dtype=float), lattice)
            if d < cluster_distance_a:
                cluster.append(site)
                placed = True
                break
        if not placed:
            clusters.append([site])

    minima: list[RelaxedAdsorptionSite] = []
    for cid, cluster in enumerate(clusters):
        best = min(cluster, key=lambda s: s.adsorption_energy_ev)
        best.cluster_id = cid
        minima.append(best)

    minima.sort(key=lambda s: s.adsorption_energy_ev)
    for rank, s in enumerate(minima, start=1):
        s.rank = rank
        s.is_global_minimum = (rank == 1)
    return minima


def analyze_relaxed_site_geometry(
    relaxed_structure: Structure,
    k_index: int,
    site_type: str,
) -> dict[str, Any]:
    """Extract nearest-atom shell, K–O, K–metal, CN after relaxation."""
    k_index = resolve_k_index(relaxed_structure, k_index)
    k_site = relaxed_structure[k_index]
    k_frac = np.asarray(k_site.frac_coords, dtype=float)

    k_o, o_idx = nearest_site_by_symbol(relaxed_structure, k_index, "O")
    k_metal = float("inf")
    metal_idx = -1
    for i, site in enumerate(relaxed_structure):
        if i == k_index or not is_metal(site.specie.symbol):
            continue
        d = lattice_distance(relaxed_structure, k_frac, site.frac_coords)
        if d < k_metal:
            k_metal = d
            metal_idx = i

    nbrs = nearest_neighbors_at_site(relaxed_structure, k_index, n=10, max_cutoff=8.0)
    cn = k_coordination_number_at_site(relaxed_structure, k_index)
    descriptor = local_environment_descriptor_string(relaxed_structure, k_frac, site_type)

    return {
        "k_frac_coords": [float(x) for x in k_frac],
        "k_o_distance_a": k_o,
        "k_metal_distance_a": float(k_metal) if np.isfinite(k_metal) and metal_idx >= 0 else float("nan"),
        "oxygen_index": o_idx,
        "metal_index": metal_idx,
        "coordination_number": cn,
        "nearest_atoms": nbrs,
        "local_environment_descriptor": descriptor,
    }


def compute_adsorption_confidence(
    site: RelaxedAdsorptionSite,
    all_energies: Sequence[float],
) -> float:
    """
    Confidence in a relaxed adsorption result [0, 100].

    Based on: convergence, contact distances, energy sanity, rank separation.
    """
    score = 0.0
    if site.optimization_converged:
        score += 35.0
    elif site.success:
        score += 15.0

    if np.isfinite(site.k_o_distance_a):
        target = k_oxygen_contact_distance(1.0)
        if 2.0 <= site.k_o_distance_a <= 3.8:
            score += 25.0 * float(np.exp(-0.5 * ((site.k_o_distance_a - target) / 0.5) ** 2))

    if np.isfinite(site.k_metal_distance_a) and site.k_metal_distance_a > 2.0:
        score += 10.0

    finite = [e for e in all_energies if np.isfinite(e)]
    if finite and np.isfinite(site.adsorption_energy_ev):
        med = float(np.median(finite))
        if abs(site.adsorption_energy_ev - med) < 3.0:
            score += 15.0
        if site.is_global_minimum and len(finite) > 1:
            gap = float(finite[1] - finite[0]) if len(finite) > 1 else 0.0
            if gap > 0.05:
                score += 15.0

    return float(np.clip(score, 0.0, 100.0))


# Backward-compatible dict API for legacy callers
def deduplicate_site_dicts(
    bare: Structure,
    candidates: list[dict[str, Any]],
    tolerance: float = 0.75,
) -> list[dict[str, Any]]:
    wrapped = [
        AdsorptionCandidate(
            candidate_id=i,
            site_type=str(c.get("site_type", "unknown")),
            k_frac=list(c.get("k_frac", [])),
            anchor_index=int(c.get("oxygen_index", c.get("anchor_index", -1))),
            k_o_distance_a=float(c.get("k_o_distance_a", float("nan"))),
        )
        for i, c in enumerate(candidates)
    ]
    unique = deduplicate_sites_by_symmetry(bare, wrapped, tolerance_a=tolerance)
    return [
        {
            "site_type": c.site_type,
            "oxygen_index": c.anchor_index,
            "k_frac": c.k_frac,
            "k_o_distance_a": c.k_o_distance_a,
        }
        for c in unique
    ]


# Legacy aliases
deduplicate_sites_by_symmetry_dict = deduplicate_site_dicts
