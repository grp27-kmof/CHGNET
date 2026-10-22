#!/usr/bin/env python3
"""
mech_stability_cif.py — Compute the mechanical stability of a MOF from a CIF file.

Usage
--
    python mech_stability_cif.py  path/to/structure.cif  [--device cpu|cuda]
                                  [--delta 0.01]
                                  [--pressure-max 3.0]
                                  [--pressure-steps 30]
                                  [--no-pressure-scan]
                                  [--output report.txt]

What is computed
----------------
1.  **Geometry optimization** — CHGNet relaxes the CIF structure to a local
    energy minimum (forces < 0.01 eV/Å, stress < 0.01 GPa).
2.  **Elastic stiffness tensor C_ij** — finite-difference stress response to
    six independent Voigt strain patterns ±δ.
3.  **Voigt–Reuss–Hill elastic moduli** — bulk modulus B, shear modulus G,
    Young's modulus E, Poisson's ratio ν derived from C_ij.
4.  **Born mechanical stability** — positive definiteness of C_ij (all
    eigenvalues > 0 is required for mechanical stability).
5.  **Birch–Murnaghan EOS** — E(V) curve from hydrostatic compression,
    giving an independent estimate of B and B' (pressure derivative).
6.  **Critical pressure P_c** — pressure at which Born stability first fails
    under simulated hydrostatic compression.  This is the primary output.

All calculations use the CHGNet machine-learning interatomic potential, which
provides DFT-quality energies, forces, and stresses for transition-metal MOFs.

References
----------
  Le Page & Saxe, PRB 65, 104104 (2002) — strain protocol.
  Mouhat & Coudert, PRB 90, 224104 (2014) — Born stability.
  Birch, Phys. Rev. 71, 809 (1947); Murnaghan, PNAS 30, 244 (1944).
"""
from __future__ import annotations

import argparse
import logging
import sys
import textwrap
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("mech_stability")
EV_A3_TO_GPA = 160.21766208


# ─────────────────────────────────────────────────────────────────────────────
# Load CHGNet
# ─────────────────────────────────────────────────────────────────────────────

def _load_chgnet(device: str):
    from chgnet.model.model import CHGNet
    from chgnet.model.dynamics import CHGNetCalculator
    logger.info("Loading CHGNet v0.3.0 on device '%s' …", device)
    model = CHGNet.load()
    calc  = CHGNetCalculator(model=model, use_device=device)
    logger.info("CHGNet ready.")
    return model, calc


# ─────────────────────────────────────────────────────────────────────────────
# CIF → ASE Atoms
# ─────────────────────────────────────────────────────────────────────────────

def _cif_to_atoms(cif_path: Path):
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    logger.info("Reading CIF: %s", cif_path)
    structure = Structure.from_file(str(cif_path))
    atoms = AseAtomsAdaptor.get_atoms(structure)
    logger.info(
        "  Formula: %s   |   Natoms: %d   |   V0 = %.2f Å³",
        structure.formula, len(atoms), atoms.get_volume(),
    )
    return atoms, structure.formula


# ─────────────────────────────────────────────────────────────────────────────
# Geometry optimization
# ─────────────────────────────────────────────────────────────────────────────

def _attach_calculator_and_measure(atoms, model, device: str, fmax: float):
    from chgnet.model.dynamics import CHGNetCalculator

    final_atoms = atoms.copy()
    final_atoms.set_constraint([])
    final_atoms.calc = CHGNetCalculator(model=model, use_device=device)
    forces = np.asarray(final_atoms.get_forces(apply_constraint=False))
    final_fmax = float(np.max(np.linalg.norm(forces, axis=1))) if len(forces) else float("nan")
    energy = float(final_atoms.get_potential_energy())
    converged = bool(np.isfinite(final_fmax) and final_fmax <= fmax)
    return final_atoms, energy, final_fmax, converged


def _optimize(
    atoms,
    model,
    device: str,
    fmax: float = 0.01,
    relax_steps: int = 500,
):
    from chgnet.model.dynamics import StructOptimizer
    logger.info("Relaxing structure (fmax = %.3f eV/Å) …", fmax)
    relax = StructOptimizer(model=model, use_device=device)
    result = relax.relax(atoms, fmax=fmax, steps=relax_steps, relax_cell=True)
    if "final_structure" in result:
        from pymatgen.io.ase import AseAtomsAdaptor
        final_atoms = AseAtomsAdaptor.get_atoms(result["final_structure"])
    elif "final_atoms" in result:
        final_atoms = result["final_atoms"]
    else:
        raise KeyError("StructOptimizer result contains neither 'final_structure' nor 'final_atoms'")

    final_atoms, energy, fmax_reached, converged = _attach_calculator_and_measure(
        final_atoms, model, device, fmax
    )
    logger.info("  Optimization done.  E = %.6f eV   fmax = %.4f eV/Å", energy, fmax_reached)
    if not converged:
        logger.warning("  Optimization may not be fully converged (fmax = %.4f)", fmax_reached)
    return final_atoms, converged, fmax_reached


def _residual_stress(atoms) -> tuple[np.ndarray, float]:
    stress_gpa = np.asarray(atoms.get_stress(voigt=True), dtype=float) * EV_A3_TO_GPA
    max_abs = float(np.max(np.abs(stress_gpa)))
    logger.info(
        "Residual stress (GPa): %s; max |stress| = %.6f GPa",
        np.array2string(stress_gpa, precision=6),
        max_abs,
    )
    if max_abs > 0.1:
        logger.warning("Residual stress exceeds 0.1 GPa; elastic constants may be less accurate.")
    return stress_gpa, max_abs


# ─────────────────────────────────────────────────────────────────────────────
# BM3 EOS scan
# ─────────────────────────────────────────────────────────────────────────────

def _eos_scan(atoms_ref, model, device: str, n_points: int = 11):
    """
    E(V) scan: compress/expand cell isotropically over ±6 % volume range,
    evaluate energy, fit 3rd-order Birch–Murnaghan EOS.
    """
    import copy
    from chgnet.model.dynamics import CHGNetCalculator
    from scientific.mech_elastic import fit_birch_murnaghan

    calc = CHGNetCalculator(model=model, use_device=device)
    V0 = atoms_ref.get_volume()
    scale_factors = np.linspace(0.94, 1.06, n_points) ** (1.0 / 3.0)   # ±6 % V

    volumes, energies = [], []
    logger.info("EOS scan: %d points over ±6%% volume range …", n_points)
    for s in scale_factors:
        at = copy.deepcopy(atoms_ref)
        new_cell = atoms_ref.get_cell().array * s
        at.set_cell(new_cell, scale_atoms=True)
        at.calc = calc
        try:
            E = float(at.get_potential_energy())
            volumes.append(at.get_volume())
            energies.append(E)
        except Exception as exc:
            logger.warning("  EOS point skipped: %s", exc)

    if len(volumes) < 5:
        logger.warning("  Too few EOS points for reliable BM3 fit.")
        return {"success": False}

    result = fit_birch_murnaghan(np.array(volumes), np.array(energies))
    if result["success"]:
        logger.info(
            "  BM3 fit:  B0 = %.2f GPa   B0' = %.3f   V0 = %.2f Å³",
            result["B0_gpa"], result["B0p"], result["V0_A3"],
        )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis function
# ─────────────────────────────────────────────────────────────────────────────

def analyze_mof_cif(
    cif_path: str | Path,
    device: str = "cpu",
    delta: float = 0.01,
    pressure_max_gpa: float = 3.0,
    pressure_steps: int = 30,
    run_pressure_scan: bool = True,
    skip_relax: bool = False,
    relax_steps: int = 500,
) -> dict:
    """
    Full mechanical stability analysis for a MOF CIF file.

    Returns a results dict containing all computed quantities and a formatted
    report string under the key ``'report'``.
    """
    from scientific.mech_elastic import (
        compute_elastic_tensor,
        vrh_averages,
        born_stability,
        find_critical_pressure,
    )

    cif_path = Path(cif_path)
    if not cif_path.exists():
        raise FileNotFoundError(f"CIF file not found: {cif_path}")

    mof_name = cif_path.stem
    results: dict = {"mof_name": mof_name, "cif_path": str(cif_path)}

    # ── 1. Load CHGNet ────────────────────────────────────────────────────────
    model, calc = _load_chgnet(device)

    # ── 2. Read CIF ───────────────────────────────────────────────────────────
    atoms_raw, formula = _cif_to_atoms(cif_path)
    results["formula"] = formula

    # ── 3. Optimize ───────────────────────────────────────────────────────────
    if skip_relax:
        atoms_opt, energy, final_fmax, converged = _attach_calculator_and_measure(
            atoms_raw, model, device, 0.01
        )
        logger.info("Skipping relaxation; measured final fmax = %.4f eV/Å", final_fmax)
    else:
        atoms_opt, converged, final_fmax = _optimize(
            atoms_raw, model, device, relax_steps=relax_steps
        )
    results["optimization_converged"] = converged
    results["converged"] = converged
    results["final_fmax"] = final_fmax
    results["volume_A3"] = float(atoms_opt.get_volume())
    results["energy_eV"] = float(atoms_opt.get_potential_energy())

    residual_stress, max_residual_stress = _residual_stress(atoms_opt)
    results["residual_stress_GPa"] = residual_stress.tolist()
    results["residual_stress_max_abs_GPa"] = max_residual_stress

    # ── 4. Elastic tensor ─────────────────────────────────────────────────────
    logger.info("Computing elastic stiffness tensor C_ij (δ = %.3f) …", delta)
    C = compute_elastic_tensor(atoms_opt, delta, verbose=True)
    results["elastic_tensor_gpa"] = C.tolist()
    logger.info("  C_ij computed successfully.")

    # ── 5. Moduli ─────────────────────────────────────────────────────────────
    vrh = vrh_averages(C)
    results.update(vrh)
    logger.info(
        "  VRH moduli:  B = %.2f GPa   G = %.2f GPa   E = %.2f GPa   ν = %.3f",
        vrh["B_hill"], vrh["G_hill"], vrh["E_hill"], vrh["nu_hill"],
    )

    # ── 6. Born stability ─────────────────────────────────────────────────────
    bs = born_stability(C)
    results["born_stable"]          = bs["stable"]
    results["born_eigenvalues"]     = bs["eigenvalues"]
    results["born_min_eigenvalue"]  = bs["min_eigenvalue"]
    results["born_reason"]          = bs["reason"]
    logger.info("  Born stability: %s  (min λ = %.4f GPa)", "PASS" if bs["stable"] else "FAIL", bs["min_eigenvalue"])

    # ── 7. BM3 EOS ───────────────────────────────────────────────────────────
    eos = _eos_scan(atoms_opt, model, device)
    results["eos"] = eos
    results["birch_murnaghan"] = {
        key: eos[key]
        for key in ("V0_A3", "B0_gpa", "B0p")
        if key in eos
    }

    # ── 8. Critical pressure ──────────────────────────────────────────────────
    critical_pressure_gpa: float | None = None
    pressure_scan_note = "Not run."

    if not bs["stable"]:
        # Already unstable at P=0; critical pressure is undefined
        critical_pressure_gpa = 0.0
        pressure_scan_note = "Structure is mechanically unstable at zero pressure (Born criteria already violated). P_c = 0."
        logger.warning("  Structure is unstable at P=0. Skipping pressure scan.")

    elif run_pressure_scan:
        logger.info("Pressure scan: 0 → %.2f GPa in %d steps …", pressure_max_gpa, pressure_steps)
        pressures = np.concatenate([
            np.linspace(0.0, min(pressure_max_gpa, 0.5), max(pressure_steps // 3, 6)),
            np.linspace(0.5, pressure_max_gpa, max(pressure_steps * 2 // 3, 15)),
        ])
        pressures = np.unique(pressures)

        pc_result = find_critical_pressure(atoms_opt, pressures_gpa=pressures, delta=delta)
        critical_pressure_gpa = pc_result["critical_pressure_gpa"]
        pressure_scan_note    = pc_result["note"]
        results["pressure_scan"] = pc_result["pressure_scan"]

        if critical_pressure_gpa is not None:
            logger.info("  → Critical pressure: %.4f GPa", critical_pressure_gpa)
        else:
            logger.info("  → Born stability maintained across full scan range.")

    results["critical_pressure_gpa"] = critical_pressure_gpa
    results["pressure_scan_note"]    = pressure_scan_note

    # ── 9. Method and confidence ──────────────────────────────────────────────
    method_parts = [
        "CHGNet v0.3.0 MLIP (DFT-quality)",
        f"Geometry optimized to fmax ≤ {0.01} eV/Å",
        f"Elastic tensor by finite-difference stress (δ = {delta:.3f})",
        "Voigt–Reuss–Hill moduli averaging",
        "Born stability (positive definiteness of C_ij)",
    ]
    if run_pressure_scan and bs["stable"]:
        method_parts.append(
            "Critical pressure from Born-stability scan under hydrostatic compression "
            f"(Murnaghan volume compression up to {pressure_max_gpa:.1f} GPa)"
        )
    if eos.get("success"):
        method_parts.append(
            f"3rd-order Birch–Murnaghan EOS (B0 = {eos['B0_gpa']:.2f} GPa)"
        )

    results["method"] = "; ".join(method_parts)

    # Confidence assessment
    notes: list[str] = []
    if not converged:
        notes.append("geometry optimization not fully converged")
    if not bs["stable"]:
        notes.append("Born criteria violated at zero pressure — P_c is not a reliable upper bound")
    if critical_pressure_gpa is None:
        notes.append(
            f"P_c exceeds scanned range ({pressure_max_gpa:.1f} GPa); increase --pressure-max"
        )
    if eos.get("success") and abs(eos["B0_gpa"] - vrh["B_hill"]) > 5.0:
        notes.append(
            f"BM3 B0 ({eos['B0_gpa']:.2f} GPa) and VRH B ({vrh['B_hill']:.2f} GPa) "
            "differ by > 5 GPa — check strain amplitude"
        )

    if not notes:
        confidence = "High — all calculations converged, Born criteria met, P_c located within scan."
    elif len(notes) == 1:
        confidence = f"Moderate — one caveat: {notes[0]}."
    else:
        confidence = "Low — multiple caveats: " + "; ".join(notes) + "."

    results["confidence"] = confidence

    # ── 10. Stability label ───────────────────────────────────────────────────
    if critical_pressure_gpa is None:
        stability_label = f"Mechanically robust (P_c > {pressure_max_gpa:.1f} GPa)"
    elif critical_pressure_gpa == 0.0:
        stability_label = "Mechanically unstable at ambient pressure"
    elif critical_pressure_gpa >= 1.0:
        stability_label = f"Mechanically robust (P_c = {critical_pressure_gpa:.3f} GPa)"
    elif critical_pressure_gpa >= 0.1:
        stability_label = f"Moderately stable (P_c = {critical_pressure_gpa:.3f} GPa)"
    else:
        stability_label = f"Mechanically fragile (P_c = {critical_pressure_gpa:.3f} GPa)"

    results["stability_label"] = stability_label

    json_results = {
        "converged": converged,
        "final_fmax": final_fmax,
        "residual_stress_GPa": residual_stress.tolist(),
        "residual_stress_max_abs_GPa": max_residual_stress,
        "elastic_tensor_GPa": C.tolist(),
        "eigenvalues": bs["eigenvalues"],
        "born_stable": bs["stable"],
        "B_voigt": vrh["B_voigt"],
        "B_reuss": vrh["B_reuss"],
        "B_hill": vrh["B_hill"],
        "G_voigt": vrh["G_voigt"],
        "G_reuss": vrh["G_reuss"],
        "G_hill": vrh["G_hill"],
        "youngs_modulus": vrh["E_hill"],
        "poisson_ratio": vrh["nu_hill"],
        "Birch–Murnaghan": results["birch_murnaghan"],
        "V0_A3": eos.get("V0_A3"),
        "B0_GPa": eos.get("B0_gpa"),
        "B0_prime": eos.get("B0p"),
    }
    json_path = cif_path.with_name(f"{cif_path.stem}_mech.json")
    import json
    json_path.write_text(json.dumps(json_results, indent=2, allow_nan=True), encoding="utf-8")
    logger.info("JSON results written to %s", json_path)

    # ── 11. Format report ─────────────────────────────────────────────────────
    def _fmt_gpa(v):
        return f"{v:.4f}" if (v is not None and np.isfinite(v)) else "Cannot be determined"

    pc_str = (
        f"{critical_pressure_gpa:.4f} GPa" if (
            critical_pressure_gpa is not None and critical_pressure_gpa > 0
        )
        else ("0 GPa (unstable at ambient)" if critical_pressure_gpa == 0.0 else
              f"Could not be determined within scan range (> {pressure_max_gpa:.1f} GPa)")
    )

    report = textwrap.dedent(f"""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║         MOF Mechanical Stability Report — CHGNet MLIP               ║
    ╚══════════════════════════════════════════════════════════════════════╝

    MOF Name            : {mof_name}
    Formula             : {formula}
    CIF path            : {cif_path}

    ── Elastic Moduli (Voigt–Reuss–Hill) ───────────────────────────────────
    Bulk Modulus     B  : {_fmt_gpa(vrh["B_hill"])} GPa
    Shear Modulus    G  : {_fmt_gpa(vrh["G_hill"])} GPa
    Young's Modulus  E  : {_fmt_gpa(vrh["E_hill"])} GPa
    Poisson's Ratio  ν  : {vrh["nu_hill"]:.4f}

    Residual stress (GPa): {", ".join(f"{v:.4f}" for v in residual_stress)}
    Max |residual stress|: {max_residual_stress:.4f} GPa
    Final fmax           : {final_fmax:.6f} eV/Å

    BM3 EOS (independent check)
      B0 (EOS)          : {_fmt_gpa(eos.get("B0_gpa", float("nan"))) if eos.get("success") else "EOS fit failed"} GPa
      B0' (EOS)         : {f"{eos['B0p']:.3f}" if eos.get("success") else "—"}

    ── Born Stability ───────────────────────────────────────────────────────
    Born Stability      : {"✓ PASS" if bs["stable"] else "✗ FAIL"}
    Min eigenvalue λ₁   : {bs["min_eigenvalue"]:.4f} GPa
    All eigenvalues (GPa): {", ".join(f"{v:.3f}" for v in bs["eigenvalues"])}
    Reason              : {bs["reason"]}

    ── Critical Pressure ────────────────────────────────────────────────────
    Critical Pressure   : {pc_str}
    ({pressure_scan_note})

    ── Assessment ───────────────────────────────────────────────────────────
    Mechanical Stability: {stability_label}

    Method Used         :
      {chr(10).join("  • " + p for p in method_parts)}

    Confidence          : {confidence}

    ── Literature Benchmarks ────────────────────────────────────────────────
    UiO-66(Zr)  P_c ≈ 1.83–1.84 GPa   (diamond-anvil cell, Moggach 2009)
    MOF-5(Zn)   P_c ≈ 0.189 GPa       (DAC + PXRD, Chapman 2009)
    MIL-47(V)   P_c ≈ 0.123 GPa LP→CP (DAC, Yot 2012)
    MIL-53(Al)  P_c ≈ 0.030 GPa LP→CP (DAC, Serre 2002)

    NOTE: CHGNet pressures are derived from mechanical stability criteria
    under simulated hydrostatic compression, not from DAC experiments.
    Values are expected to be qualitatively consistent with experiment but
    may differ quantitatively due to MLIP accuracy and temperature effects.
    ════════════════════════════════════════════════════════════════════════
    """).strip()

    results["report"] = report
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("cif", help="Path to the MOF CIF file.")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"],
                   help="Compute device for CHGNet (default: cpu).")
    p.add_argument("--delta", type=float, default=0.01,
                   help="Strain amplitude for elastic tensor (default: 0.01).")
    p.add_argument("--pressure-max", type=float, default=3.0,
                   help="Maximum pressure for Born-stability scan, GPa (default: 3.0).")
    p.add_argument("--pressure-steps", type=int, default=30,
                   help="Number of pressure grid points (default: 30).")
    p.add_argument("--no-pressure-scan", action="store_true",
                   help="Skip the pressure scan (only compute elastic moduli).")
    p.add_argument("--skip-relax", action="store_true",
                   help="Skip geometry relaxation; input CIF is already relaxed.")
    p.add_argument("--relax-steps", type=int, default=500,
                   help="Maximum geometry-relaxation steps (default: 500).")
    p.add_argument("--output", default=None,
                   help="Save report to this text file.")
    return p


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    results = analyze_mof_cif(
        cif_path=args.cif,
        device=args.device,
        delta=args.delta,
        pressure_max_gpa=args.pressure_max,
        pressure_steps=args.pressure_steps,
        run_pressure_scan=not args.no_pressure_scan,
        skip_relax=args.skip_relax,
        relax_steps=args.relax_steps,
    )

    print("\n" + results["report"])
    print(
        "\nFinal summary: Born stable %s; smallest eigenvalue %.6f GPa; "
        "B_hill %.6f GPa; G_hill %.6f GPa"
        % (
            "yes" if results["born_stable"] else "no",
            results["born_min_eigenvalue"],
            results["B_hill"],
            results["G_hill"],
        )
    )

    if args.output:
        out = Path(args.output)
        out.write_text(results["report"], encoding="utf-8")
        logger.info("Report written to %s", out)

    return results


if __name__ == "__main__":
    main()
