#!/usr/bin/env python3
"""
Generate K-inserted UiO-66 candidate CIFs near mu3-O sites (K2-like),
then score all candidates using existing CHGNet optimization/adsorption routines.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pymatgen.core import Structure

sys.path.append(str(Path(__file__).parent.resolve()))
import run_pipeline


@dataclass
class CandidateResult:
    site_number: int
    oxygen_index: int
    oxygen_frac: np.ndarray
    k_frac_initial: np.ndarray
    k_frac_optimized: np.ndarray
    k_o_distance_a: float
    adsorption_energy_ev: float
    candidate_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Insert one K near each mu3-O site using an outward normal from the Zr3 plane, "
            "then optimize/score and select best adsorption site."
        )
    )
    parser.add_argument(
        "--bare-cif",
        type=Path,
        default=Path("INPUTS/bare_cifs/QMOF-Relaxed/QMOF-relaxed/UiO-66.cif"),
        help="Path to bare UiO-66 CIF file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("INPUTS/cifs/q-mof-K+ions/hmof-K+ions"),
        help="Directory to write all candidate CIF files.",
    )
    parser.add_argument(
        "--k-o-distance",
        type=float,
        default=2.78,
        help="Initial K-O distance in Angstrom along outward normal.",
    )
    parser.add_argument(
        "--zr-o-cutoff",
        type=float,
        default=2.5,
        help="Cutoff (Angstrom) to detect mu3-O coordination to Zr atoms.",
    )
    return parser.parse_args()


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        raise ValueError("Encountered near-zero vector during normal construction.")
    return v / n


def _pbc_displacement(frac_from: np.ndarray, frac_to: np.ndarray) -> np.ndarray:
    d = frac_to - frac_from
    d -= np.round(d)
    return d


def find_mu3_oxygen_sites(structure: Structure, zr_o_cutoff: float) -> list[tuple[int, list[int]]]:
    """
    Return list of (oxygen_index, [zr_index0, zr_index1, zr_index2]) for mu3-O sites.
    """
    mu3_sites: list[tuple[int, list[int]]] = []
    for o_idx, site in enumerate(structure):
        if site.specie.symbol != "O":
            continue
        nbrs = structure.get_neighbors(site, zr_o_cutoff)
        zr_indices = sorted({n.index for n in nbrs if n.specie.symbol == "Zr"})
        if len(zr_indices) == 3:
            mu3_sites.append((o_idx, zr_indices))
    return mu3_sites


def place_k_outward_normal(
    structure: Structure,
    oxygen_index: int,
    zr_indices: list[int],
    k_o_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Place K at oxygen + distance * outward_normal.
    Outward is defined as pointing away from Zr3 centroid.
    Returns (k_frac, o_frac).
    """
    lattice = structure.lattice
    o_frac = np.array(structure[oxygen_index].frac_coords, dtype=float)

    zr_frac = [np.array(structure[i].frac_coords, dtype=float) for i in zr_indices]
    zr_cart = []
    for f in zr_frac:
        d_frac = _pbc_displacement(o_frac, f)
        zr_cart.append(lattice.get_cartesian_coords(o_frac + d_frac))
    zr_cart = [np.array(v, dtype=float) for v in zr_cart]

    z1, z2, z3 = zr_cart
    v1 = z2 - z1
    v2 = z3 - z1
    normal = _normalize(np.cross(v1, v2))

    o_cart = np.array(structure[oxygen_index].coords, dtype=float)
    zr_centroid = (z1 + z2 + z3) / 3.0
    outward_hint = _normalize(o_cart - zr_centroid)
    if float(np.dot(normal, outward_hint)) < 0.0:
        normal = -normal

    k_cart = o_cart + k_o_distance * normal
    k_frac = np.array(lattice.get_fractional_coords(k_cart), dtype=float) % 1.0
    return k_frac, o_frac


def write_candidate_cifs(
    bare: Structure,
    stem: str,
    output_dir: Path,
    mu3_sites: list[tuple[int, list[int]]],
    k_o_distance: float,
) -> list[tuple[int, int, np.ndarray, np.ndarray, Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[tuple[int, int, np.ndarray, np.ndarray, Path]] = []
    for i, (o_idx, zr_idx) in enumerate(mu3_sites, start=1):
        k_frac, o_frac = place_k_outward_normal(
            structure=bare,
            oxygen_index=o_idx,
            zr_indices=zr_idx,
            k_o_distance=k_o_distance,
        )
        candidate = bare.copy()
        candidate.append("K", k_frac, validate_proximity=False)
        out_path = output_dir / f"{stem}_Ksite{i}.cif"
        candidate.to(fmt="cif", filename=str(out_path))
        records.append((i, o_idx, o_frac, k_frac, out_path))
    return records


def score_candidates(
    bare_original: Structure,
    candidate_records: list[tuple[int, int, np.ndarray, np.ndarray, Path]],
    stem: str,
) -> list[CandidateResult]:
    logger = run_pipeline.setup_logging(run_pipeline.OUTPUT_DIR)
    run_pipeline._logger = logger

    bare_opt = run_pipeline.optimize_structure(
        bare_original,
        "bare",
        f"{stem}_site_scan",
    )
    if not bare_opt.success or bare_opt.structure is None:
        raise RuntimeError(f"Bare optimization failed: {bare_opt.error}")
    bare_opt_struct = bare_opt.structure

    results: list[CandidateResult] = []
    for site_num, o_idx, o_frac, k_frac0, path in candidate_records:
        candidate_struct = run_pipeline.load_structure(path)
        mof_tag = f"{stem}_Ksite{site_num}"
        k_opt = run_pipeline.optimize_structure(candidate_struct, "k_inserted", mof_tag)
        if not k_opt.success or k_opt.structure is None:
            raise RuntimeError(f"Candidate optimization failed for site {site_num}: {k_opt.error}")

        k_indices = run_pipeline.get_inserted_k_indices(bare_original, k_opt.structure)
        n_k = run_pipeline.effective_inserted_k_count(bare_original, k_opt.structure)
        ads_ev = run_pipeline.compute_adsorption_energy(
            bare_opt_struct,
            k_opt.structure,
            k_indices,
            n_k,
            mof_tag,
        )
        if not np.isfinite(ads_ev):
            raise RuntimeError(f"Adsorption energy failed (NaN) for site {site_num}")

        k_idx = k_indices[0]
        k_frac_opt = np.array(k_opt.structure[k_idx].frac_coords, dtype=float)
        k_o_distance = float(
            k_opt.structure.lattice.get_distance_and_image(
                k_frac_opt,
                np.array(o_frac, dtype=float),
            )[0]
        )

        results.append(
            CandidateResult(
                site_number=site_num,
                oxygen_index=o_idx,
                oxygen_frac=np.array(o_frac, dtype=float),
                k_frac_initial=np.array(k_frac0, dtype=float),
                k_frac_optimized=k_frac_opt,
                k_o_distance_a=k_o_distance,
                adsorption_energy_ev=float(ads_ev),
                candidate_path=path,
            )
        )
    return results


def main() -> None:
    args = parse_args()
    bare_path: Path = args.bare_cif
    output_dir: Path = args.output_dir
    stem = bare_path.stem

    if not bare_path.is_file():
        raise FileNotFoundError(f"Bare CIF not found: {bare_path}")

    bare_struct = Structure.from_file(str(bare_path))
    bare_struct.remove_species(["K"])

    mu3_sites = find_mu3_oxygen_sites(bare_struct, args.zr_o_cutoff)
    if not mu3_sites:
        raise RuntimeError("No mu3-O sites found (O with exactly three Zr neighbors).")

    records = write_candidate_cifs(
        bare=bare_struct,
        stem=stem,
        output_dir=output_dir,
        mu3_sites=mu3_sites,
        k_o_distance=args.k_o_distance,
    )
    print(f"Generated {len(records)} candidate CIF(s) in: {output_dir.resolve()}")

    scored = score_candidates(
        bare_original=bare_struct,
        candidate_records=records,
        stem=stem,
    )
    best = min(scored, key=lambda r: r.adsorption_energy_ev)

    best_src = run_pipeline.mof_work_dir(f"{stem}_Ksite{best.site_number}") / "k_inserted_optimized.cif"
    best_out = output_dir / f"{stem}_bestK.cif"
    best_struct = (
        run_pipeline.load_structure(best_src)
        if best_src.is_file()
        else run_pipeline.load_structure(best.candidate_path)
    )
    best_struct.to(fmt="cif", filename=str(best_out))

    k_flags = run_pipeline.validate_k_insertion(bare_struct, best_struct)
    if k_flags:
        print(f"Warning: best K structure failed insertion validation: {', '.join(k_flags)}")

    print("\nBest candidate (lowest adsorption energy):")
    print(f"Site number      : {best.site_number}")
    print(f"Adsorption energy: {best.adsorption_energy_ev:.4f} eV/K")
    print(
        "K coordinates    : "
        f"({best.k_frac_optimized[0]:.6f}, {best.k_frac_optimized[1]:.6f}, {best.k_frac_optimized[2]:.6f}) frac"
    )
    print(f"K-O distance     : {best.k_o_distance_a:.4f} A")
    print(f"Saved best CIF   : {best_out}")


if __name__ == "__main__":
    main()
