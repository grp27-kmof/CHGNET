#!/usr/bin/env python3
"""
run_md.py — CHGNet MD Simulation Runner (Step 1 of 2).

Runs geometry optimization and NVT Molecular Dynamics for all MOF CIF pairs.
Saves .traj and .log trajectory files to outputs/structures/<MOF>/.

Run this FIRST, then run analyse.py to extract the 5 screening parameters.

Usage:
    python run_md.py                            # Use default CIF directories
    python run_md.py --bare inputs/bare_cifs_100 --k inputs/cifs_100
    python run_md.py --stage1-only              # Skip Stage 2 (long MD)
    python run_md.py --no-resume                # Ignore existing trajectories

Directory conventions:
    --bare  (bare_cifs)  : MOF CIFs WITHOUT K atoms  — the bare framework
    --k     (cifs)       : MOF CIFs WITH K atoms inserted — K@MOF structures
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from profiler import StageProfiler

profiler = StageProfiler()

# Ensure run_pipeline is importable from the same directory
sys.path.append(str(Path(__file__).parent.resolve()))
import run_pipeline

# =============================================================================
# CONFIGURATION — override via command-line flags or edit here
# =============================================================================

# bare_cifs  →  MOF CIFs WITHOUT K  (bare framework structures)
# cifs       →  MOF CIFs WITH K inserted  (K@MOF structures)
# Must match the directory names used by run_pipeline.py (case-insensitive on Windows)
DEFAULT_BARE_CIFS_DIR = Path("inputs/bare_cifs")   # bare MOF, no K atoms
DEFAULT_CIFS_DIR      = Path("inputs/cifs")         # K-inserted MOF structures
OUTPUT_DIR            = Path("outputs")

# =============================================================================
# END CONFIGURATION
# =============================================================================


def setup_logging(output_dir: Path) -> logging.Logger:
    """Configure file + console logging for run_md.py."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("run_md")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(output_dir / "run_md.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger



def _log_oom_retry(mof: str) -> None:
    """Append MOF name to retry_gpu.txt (deduplicates; safe to call multiple times)."""
    retry_file = OUTPUT_DIR / "retry_gpu.txt"
    existing: set[str] = set()
    if retry_file.is_file():
        with open(retry_file, "r", encoding="utf-8") as f:
            existing = {line.strip() for line in f if line.strip()}
    if mof not in existing:
        with open(retry_file, "a", encoding="utf-8") as f:
            f.write(f"{mof}\n")


def pair_cif_files(
    bare_dir: Path, k_dir: Path, logger: logging.Logger
) -> list[tuple[str, Path, Path]]:
    """Pair bare and K-inserted CIF files by identical filename."""
    if not bare_dir.is_dir():
        logger.error("Bare CIF directory not found: %s", bare_dir.resolve())
        return []
    if not k_dir.is_dir():
        logger.error("K-inserted CIF directory not found: %s", k_dir.resolve())
        return []

    bare_files = {p.name: p for p in bare_dir.rglob("*.cif")}
    k_files    = {p.name: p for p in k_dir.rglob("*.cif")}
    # Skip insert.py candidate files (UiO-66_Ksite1.cif, …) — not paired MOFs
    k_files = {
        name: path
        for name, path in k_files.items()
        if "_Ksite" not in name and "_bestK" not in name
    }
    all_names  = sorted(set(bare_files) | set(k_files))

    pairs: list[tuple[str, Path, Path]] = []
    for name in all_names:
        bp = bare_files.get(name)
        kp = k_files.get(name)
        if bp is None:
            logger.warning("Missing bare CIF for %s — skipping", name)
            continue
        if kp is None:
            logger.warning("Missing K-inserted CIF for %s — skipping", name)
            continue
        pairs.append((Path(name).stem, bp, kp))

    logger.info("Discovered %d MOF CIF pairs for simulation", len(pairs))
    return pairs


def trajectory_is_complete(traj_path: Path, required_frames: int) -> bool:
    """Return True if the trajectory exists and has at least required_frames frames."""
    if not traj_path.is_file():
        return False
    n = run_pipeline.trajectory_frame_count(traj_path)
    return n is not None and n >= required_frames


def run_stage1_md(
    mof: str,
    bare_path: Path,
    k_path: Path,
    resume: bool,
    logger: logging.Logger,
) -> dict:
    """
    Run geometry optimization + Stage 1 (fast) MD for one MOF.

    Returns a status dict with keys:
        success (bool), adsorption_energy_ev (float), stage (int),
        screening_score (float), status (str)
    """
    t0 = time.perf_counter()
    work_dir = run_pipeline.mof_work_dir(mof)
    work_dir.mkdir(parents=True, exist_ok=True)

    stage1_steps = run_pipeline.ps_to_steps(run_pipeline.FAST_MD_TIME_PS)
    traj_path    = work_dir / "stage1_md.traj"
    log_path     = work_dir / "stage1_md.log"

    try:
        k_struct_for_cache = run_pipeline.load_structure(k_path)
        input_fingerprint = run_pipeline.structure_fingerprint(
            k_struct_for_cache, k_path
        )
    except Exception as exc:
        logger.error("  Could not fingerprint K-loaded input for %s: %s", mof, exc)
        return {"success": False, "status": "Failed", "stage": 0}

    # ── Resume check ──────────────────────────────────────────────────────────
    if resume and trajectory_is_complete(traj_path, stage1_steps):
        ckpt = run_pipeline.load_checkpoint(mof)
        cached_fp = ckpt.get("input_fingerprint")
        fp_path = work_dir / "stage1_md_fingerprint.json"
        valid = fp_path.is_file() and cached_fp is not None
        if not valid:
            logger.info("legacy cache without fingerprint — recomputing")
        else:
            with open(fp_path, encoding="utf-8") as f:
                valid = json.load(f) == input_fingerprint
            valid = valid and cached_fp == input_fingerprint
            if not valid:
                logger.info("cache invalidated: Stage 1 input fingerprint changed")
        if valid:
            logger.info("  [SKIP] Stage 1 MD already complete for %s (%d frames)", mof, stage1_steps)
            return {
                "success": True,
                "adsorption_energy_ev": ckpt.get(run_pipeline.ADSORPTION_CHECKPOINT_KEY, float("nan")),
                "stage": ckpt.get("stage", 1),
                "screening_score": ckpt.get("screening_score", float("nan")),
                "status": "Complete",
                "skipped": True,
            }

    logger.info("─" * 50)
    logger.info("MOF: %s  |  Stage 1 MD  (%.1f ps)", mof, run_pipeline.FAST_MD_TIME_PS)

    try:
        bare_struct = run_pipeline.load_structure(bare_path)
        k_struct    = run_pipeline.load_structure(k_path)
        k_indices   = run_pipeline.get_inserted_k_indices(bare_struct, k_struct)

        n_k = run_pipeline.effective_inserted_k_count(bare_struct, k_struct)
        if n_k <= 0:
            logger.error("  No inserted K atoms detected for %s — skipping", mof)
            return {"success": False, "status": "Failed", "stage": 0}

        k_flags = run_pipeline.validate_k_insertion(bare_struct, k_struct)
        if k_flags:
            logger.warning(
                "  K insertion validation for %s: %s (N_K=%d)",
                mof,
                ", ".join(k_flags),
                n_k,
            )

        # ── Strict bare-framework relaxation ─────────────────────────────────
        logger.info("  Relaxing bare framework ...")
        bare_relax = run_pipeline.relax_bare_structure(
            bare_struct,
            mof,
            k_structure=k_struct,
            resume=resume,
        )
        if not bare_relax.success:
            logger.error(
                "  Skipping %s after bare relaxation: status=%s reason=%s",
                mof,
                bare_relax.status,
                bare_relax.error or "final force acceptance failed",
            )
            return {
                "success": False,
                "status": bare_relax.status,
                "stage": 0,
                "reason": bare_relax.error,
            }

        logger.info("  Optimizing K-inserted structure ...")
        input_fingerprint = run_pipeline.structure_fingerprint(k_struct, k_path)
        k_opt = run_pipeline.optimize_structure(
            k_struct,
            "k_inserted",
            mof,
            resume=resume,
            input_fingerprint=input_fingerprint,
        )
        if not k_opt.success:
            logger.error("  K-inserted optimization failed for %s", mof)
            return {"success": False, "status": "Failed", "stage": 0}

        # ── Adsorption energy (checkpoint-aware) ─────────────────────────────
        ckpt = run_pipeline.load_checkpoint(mof)
        cached_fingerprint = ckpt.get("input_fingerprint")
        if not resume:
            ckpt = {}
        elif cached_fingerprint is None:
            logger.info("legacy cache without fingerprint — recomputing")
            ckpt = {}
        else:
            matches, reason = run_pipeline.fingerprint_matches(
                input_fingerprint, cached_fingerprint
            )
            if not matches:
                logger.info("cache invalidated: %s", reason)
                ckpt = {}
        if ckpt.get(run_pipeline.ADSORPTION_CHECKPOINT_KEY) is not None:
            ads_ev = float(ckpt[run_pipeline.ADSORPTION_CHECKPOINT_KEY])
            logger.info("  Adsorption energy (cached): %.4f eV/K", ads_ev)
        else:
            logger.info(
                "  Computing adsorption energy (%d K atoms, site search, ref=%s) ...",
                n_k,
                run_pipeline.ADSORPTION_REFERENCE,
            )
            ads_ev = run_pipeline.compute_adsorption_energy(
                bare_relax.structure,
                k_struct,
                k_indices,
                n_k,
                mof,
                reference=run_pipeline.ADSORPTION_REFERENCE,
                search_sites=True,
            )
            meta = run_pipeline.load_adsorption_site_metadata(mof)
            ckpt[run_pipeline.ADSORPTION_CHECKPOINT_KEY] = ads_ev
            ckpt["adsorption_site_type"] = meta.get("best_site_type", "")
            ckpt["adsorption_reference"] = meta.get(
                "reference", run_pipeline.ADSORPTION_REFERENCE
            )
            ckpt["adsorption_k_o_distance_a"] = meta.get("best_k_o_distance_a", float("nan"))
            ckpt["adsorption_site_rank"] = 1
            ckpt.pop("adsorption_energy_ev_per_k_v2", None)
            logger.info(
                "  Adsorption energy: %.4f eV/K  (site=%s, ref=%s)",
                ads_ev,
                meta.get("best_site_type", "provided"),
                meta.get("reference", run_pipeline.ADSORPTION_REFERENCE),
            )
            if not getattr(k_opt, "converged", True):
                logger.warning(
                    "  Optimization not fully converged for %s "
                    "(k fmax=%.4f; target=%.3f eV/Å)",
                    mof,
                    getattr(k_opt, "max_force", float("nan")),
                    run_pipeline.FORCE_TOL,
                )

        # ── Stage 1 MD ────────────────────────────────────────────────────────
        logger.info("  Running Stage 1 MD (%.1f ps) ...", run_pipeline.FAST_MD_TIME_PS)
        traj_p, log_p = run_pipeline.run_md(
            k_opt.structure, mof, "stage1", run_pipeline.FAST_MD_TIME_PS
            , resume=resume, input_fingerprint=input_fingerprint
        )
        if traj_p is None:
            logger.error("  Stage 1 MD failed for %s", mof)
            return {"success": False, "status": "Failed", "stage": 0}

        elapsed = time.perf_counter() - t0
        logger.info("  ✓ Stage 1 complete for %s in %.1f s", mof, elapsed)

        # Quick screening score for Stage 2 selection
        dummy_result = run_pipeline.MOFResult(
            mof=mof,
            adsorption_energy_ev=ads_ev,
            adsorption_binding_class=run_pipeline.classify_adsorption(ads_ev),
            status="Complete",
            stage=1,
        )
        scored = run_pipeline.compute_screening_scores([dummy_result], {mof: n_k})
        screening_score = float(scored["Screening_Score"].iloc[0]) if not scored.empty else float("nan")

        ckpt.update({"stage": 1, "screening_score": screening_score})
        ckpt["input_fingerprint"] = input_fingerprint
        run_pipeline.save_checkpoint(mof, ckpt)

        return {
            "success": True,
            "adsorption_energy_ev": ads_ev,
            "stage": 1,
            "screening_score": screening_score,
            "status": "Complete",
            "skipped": False,
        }

    except run_pipeline.CudaOOMError as exc:
        logger.error("  Stage 1 failed due to CUDA OOM for %s: %s", mof, exc)
        _log_oom_retry(mof)
        ckpt = run_pipeline.load_checkpoint(mof)
        ckpt.update({"stage": 0, "status": "GPU_OOM"})
        run_pipeline.save_checkpoint(mof, ckpt)
        run_pipeline.clear_cuda_memory(full=True)
        return {"success": False, "status": "GPU_OOM", "stage": 0, "error": str(exc)}
    except Exception as exc:
        logger.error("  Stage 1 failed for %s: %s", mof, exc)
        return {"success": False, "status": "Failed", "stage": 0, "error": str(exc)}


def run_stage2_md(
    mof: str,
    resume: bool,
    logger: logging.Logger,
) -> dict:
    """
    Run Stage 2 (long) MD for one MOF, continuing from Stage 1's last frame.

    Requires that Stage 1 optimization outputs already exist in outputs/structures/<MOF>/.
    Stage 2 inherits Stage 1's atomic positions AND momenta (velocities) so the
    trajectory is a seamless continuation rather than a fresh thermalisation.

    Returns a status dict with keys: success (bool), stage (int), status (str)
    """
    t0 = time.perf_counter()
    work_dir = run_pipeline.mof_work_dir(mof)

    stage2_steps = run_pipeline.ps_to_steps(run_pipeline.FINAL_MD_TIME_PS)
    traj_path    = work_dir / "stage2_md.traj"

    # ── Resume check ──────────────────────────────────────────────────────────
    if resume and trajectory_is_complete(traj_path, stage2_steps):
        logger.info("  [SKIP] Stage 2 MD already complete for %s (%d frames)", mof, stage2_steps)
        return {"success": True, "stage": 2, "status": "Complete", "skipped": True}

    logger.info("-" * 50)
    logger.info("MOF: %s  |  Stage 2 MD  (%.1f ps)", mof, run_pipeline.FINAL_MD_TIME_PS)

    # ── Load the K-inserted optimized structure produced in Stage 1 ───────────
    # (This provides cell, species, and topology.  Actual positions/momenta
    #  will be overridden by the Stage 1 trajectory if available.)
    k_opt_path = work_dir / "k_inserted_optimized.cif"
    if not k_opt_path.is_file():
        logger.error(
            "  k_inserted_optimized.cif not found for %s — run Stage 1 first", mof
        )
        return {"success": False, "stage": 1, "status": "Failed"}

    # ── Locate Stage 1 trajectory for warm-start ──────────────────────────────
    stage1_traj = work_dir / "stage1_md.traj" if resume else None
    if stage1_traj.is_file():
        n_s1 = run_pipeline.trajectory_frame_count(stage1_traj)
        if n_s1 and n_s1 > 0:
            logger.info(
                "  Stage 2 will continue from Stage 1 last frame "
                "(%d frames in %s).", n_s1, stage1_traj.name
            )
        else:
            logger.warning(
                "  Stage 1 trajectory exists but is empty — Stage 2 will "
                "thermalise fresh from the optimized CIF."
            )
            stage1_traj = None
    else:
        logger.warning(
            "  Stage 1 trajectory not found — Stage 2 will thermalise "
            "fresh from the optimized CIF."
        )
        stage1_traj = None

    try:
        k_opt_struct = run_pipeline.load_structure(k_opt_path)
        input_fingerprint = run_pipeline.structure_fingerprint(k_opt_struct, k_opt_path)

        logger.info("  Running Stage 2 MD (%.1f ps) ...", run_pipeline.FINAL_MD_TIME_PS)
        traj_p, log_p = run_pipeline.run_md(
            k_opt_struct, mof, "stage2", run_pipeline.FINAL_MD_TIME_PS,
            init_traj_path=stage1_traj,   # ← warm-start from Stage 1
            resume=resume, input_fingerprint=input_fingerprint,
        )
        if traj_p is None:
            logger.error("  Stage 2 MD failed for %s", mof)
            return {"success": False, "stage": 1, "status": "Failed"}

        elapsed = time.perf_counter() - t0
        logger.info("  ✓ Stage 2 complete for %s in %.1f s", mof, elapsed)

        ckpt = run_pipeline.load_checkpoint(mof)
        ckpt["stage"] = 2
        run_pipeline.save_checkpoint(mof, ckpt)

        return {"success": True, "stage": 2, "status": "Complete", "skipped": False}

    except run_pipeline.CudaOOMError as exc:
        logger.error("  Stage 2 failed due to CUDA OOM for %s: %s", mof, exc)
        _log_oom_retry(mof)
        ckpt = run_pipeline.load_checkpoint(mof)
        ckpt.update({"stage": 1, "status": "GPU_OOM"})
        run_pipeline.save_checkpoint(mof, ckpt)
        run_pipeline.clear_cuda_memory(full=True)
        return {"success": False, "stage": 1, "status": "GPU_OOM", "error": str(exc)}
    except Exception as exc:
        logger.error("  Stage 2 failed for %s: %s", mof, exc)
        return {"success": False, "stage": 1, "status": "Failed", "error": str(exc)}


def save_md_progress(
    progress: dict[str, dict], output_dir: Path
) -> None:
    """Persist simulation progress to a JSON file for inspection / resume."""
    path = output_dir / "md_progress.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, default=str)


def print_run_summary(
    stage1_results: dict[str, dict],
    stage2_results: dict[str, dict],
    logger: logging.Logger,
) -> None:
    """Print a concise summary after all simulations."""
    n_s1_ok  = sum(1 for v in stage1_results.values() if v.get("success"))
    n_s1_bad = len(stage1_results) - n_s1_ok
    n_s2_ok  = sum(1 for v in stage2_results.values() if v.get("success"))
    n_s2_bad = len(stage2_results) - n_s2_ok

    logger.info("=" * 60)
    logger.info("RUN_MD SIMULATION SUMMARY")
    logger.info("=" * 60)
    logger.info("Stage 1 (%.1f ps MD) : %d succeeded, %d failed",
                run_pipeline.FAST_MD_TIME_PS, n_s1_ok, n_s1_bad)
    logger.info("Stage 2 (%.1f ps MD) : %d succeeded, %d failed",
                run_pipeline.FINAL_MD_TIME_PS, n_s2_ok, n_s2_bad)
    logger.info("Trajectory files saved to : %s/structures/", run_pipeline.OUTPUT_DIR)
    logger.info("Next step: run  python analyse.py  to compute screening parameters.")
    logger.info("=" * 60)


def run_relax_only(
    bare_dir: Path,
    k_dir: Path,
    resume: bool,
    logger: logging.Logger,
) -> None:
    """Relax every bare CIF and write one combined relaxation summary CSV."""
    if not bare_dir.is_dir():
        logger.error("Bare CIF directory not found: %s", bare_dir.resolve())
        raise SystemExit(1)
    bare_files = sorted(bare_dir.rglob("*.cif"))
    k_files = {path.name: path for path in k_dir.rglob("*.cif")} if k_dir.is_dir() else {}
    summaries: list[dict] = []
    for bare_path in tqdm(bare_files, desc="Bare relaxation", unit="MOF"):
        mof = bare_path.stem
        try:
            bare = run_pipeline.load_structure(bare_path)
            k = run_pipeline.load_structure(k_files[bare_path.name]) if bare_path.name in k_files else None
            result = run_pipeline.relax_bare_structure(
                bare,
                mof,
                k_structure=k,
                resume=resume,
            )
            summaries.append(result.to_summary(mof))
        except (OSError, ValueError, RuntimeError) as exc:
            logger.error("%s | relaxation setup failed: %s", mof, exc)
            summaries.append(
                {
                    "mof_name": mof,
                    "status": "not_relaxed",
                    "error": str(exc),
                }
            )

    summary_path = OUTPUT_DIR / "relax_summary.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    logger.info("Combined relaxation summary written: %s", summary_path)


def run_adsorb_only(
    bare_dir: Path,
    resume: bool,
    logger: logging.Logger,
    n_prescreen: int,
    n_full: int,
    fmax: float,
    max_steps: int,
) -> None:
    """Run Change B single-K adsorption using only accepted relaxed bare CIFs."""
    if not bare_dir.is_dir():
        logger.error("Bare CIF directory not found: %s", bare_dir.resolve())
        raise SystemExit(1)
    rows: list[dict] = []
    for bare_path in tqdm(sorted(bare_dir.rglob("*.cif")), desc="Single-K adsorption", unit="MOF"):
        mof = bare_path.stem
        work = run_pipeline.mof_work_dir(mof)
        summary_path = work / "relax_summary.json"
        if not summary_path.is_file():
            logger.warning("%s | skipping adsorption: relax_summary.json not found", mof)
            continue
        try:
            with open(summary_path, encoding="utf-8") as f:
                relax_summary = json.load(f)
            if relax_summary.get("status") != "relaxed":
                logger.warning(
                    "%s | skipping adsorption: bare relaxation status=%s",
                    mof,
                    relax_summary.get("status"),
                )
                continue
            bare = run_pipeline.load_structure(work / "bare_relaxed.cif")
            result = run_pipeline.run_single_k_adsorption(
                bare,
                mof,
                float(relax_summary["energy_final_eV"]),
                e_k_ref=run_pipeline.get_bulk_k_energy(),
                n_prescreen=n_prescreen,
                n_full=n_full,
                fmax=fmax,
                max_steps=max_steps,
                resume=resume,
            )
            rows.append(result)
        except (OSError, KeyError, ValueError, RuntimeError) as exc:
            logger.error("%s | adsorption failed: %s", mof, exc)
    summary_path = run_pipeline.OUTPUT_DIR / "adsorption_k1_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    logger.info("Combined adsorption summary written: %s", summary_path)


def run_insert_only(
    bare_dir: Path,
    resume: bool,
    logger: logging.Logger,
    n_max: int,
    n_prescreen: int,
    n_full: int,
    min_kk: float,
    v_threshold: float,
    fmax: float,
    max_steps: int,
    capacity_csv: Path | None,
) -> None:
    """Run Change C sequential K insertion for accepted Change B MOFs."""
    capacity_limits: dict[str, int] = {}
    if capacity_csv is not None:
        try:
            capacity_frame = pd.read_csv(capacity_csv)
            name_col = next(
                (column for column in capacity_frame.columns if column.lower() in {"mof", "mof_name", "name", "filename"}),
                capacity_frame.columns[0],
            )
            limit_col = next(
                (column for column in capacity_frame.columns if "max" in column.lower() and "k" in column.lower()),
                None,
            )
            if limit_col is None:
                raise ValueError("capacity CSV must contain a Max_K_Ions_n column")
            for _, row in capacity_frame.iterrows():
                stem = Path(str(row[name_col])).stem
                capacity_limits[stem] = int(row[limit_col])
        except (OSError, ValueError, IndexError) as exc:
            logger.error("Could not read capacity CSV %s: %s", capacity_csv, exc)
            raise SystemExit(1)

    rows: list[dict] = []
    for bare_path in tqdm(sorted(bare_dir.rglob("*.cif")), desc="Sequential K insertion", unit="MOF"):
        mof = bare_path.stem
        work = run_pipeline.mof_work_dir(mof)
        try:
            with open(work / "relax_summary.json", encoding="utf-8") as f:
                relax_summary = json.load(f)
            with open(work / "adsorption_k1_summary.json", encoding="utf-8") as f:
                adsorption_summary = json.load(f)
        except (OSError, json.JSONDecodeError):
            logger.warning("%s | skipping insertion: required Change A/B summary missing", mof)
            continue
        if relax_summary.get("status") != "relaxed":
            logger.warning("%s | skipping insertion: bare relaxation status=%s", mof, relax_summary.get("status"))
            continue
        if adsorption_summary.get("status") != "ok":
            logger.warning("%s | skipping insertion: adsorption status=%s", mof, adsorption_summary.get("status"))
            continue
        limit = capacity_limits.get(mof, n_max)
        if mof not in capacity_limits and capacity_csv is not None:
            logger.warning("%s | not found in capacity CSV; using N_MAX=%d", mof, n_max)
        try:
            result = run_pipeline.run_sequential_k_insertion(
                run_pipeline.load_structure(work / "bare_relaxed.cif"),
                mof,
                float(relax_summary["energy_final_eV"]),
                run_pipeline.get_bulk_k_energy(),
                n_max=min(n_max, limit),
                n_prescreen=n_prescreen,
                n_full=n_full,
                min_kk=min_kk,
                v_threshold=v_threshold,
                fmax=fmax,
                max_steps=max_steps,
                resume=resume,
            )
            rows.append(result)
        except (OSError, KeyError, ValueError, RuntimeError) as exc:
            logger.error("%s | insertion failed: %s", mof, exc)
    pd.DataFrame(rows).to_csv(run_pipeline.OUTPUT_DIR / "insertion_summary.csv", index=False)
    logger.info("Combined insertion summary written: %s", run_pipeline.OUTPUT_DIR / "insertion_summary.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CHGNet MD simulation runner — Step 1 of the MOF screening pipeline."
    )
    parser.add_argument(
        "--bare", type=Path, default=DEFAULT_BARE_CIFS_DIR,
        help=(
            f"Directory with bare MOF CIFs (NO K atoms). "
            f"Default: {DEFAULT_BARE_CIFS_DIR}"
        )
    )
    parser.add_argument(
        "--k", type=Path, default=DEFAULT_CIFS_DIR,
        help=(
            f"Directory with K-inserted MOF CIFs (WITH K atoms). "
            f"Default: {DEFAULT_CIFS_DIR}"
        )
    )
    parser.add_argument(
        "--stage1-only", action="store_true",
        help="Run Stage 1 (fast) MD only; skip Stage 2 (long) MD entirely."
    )
    parser.add_argument(
        "--relax-only", action="store_true",
        help="Run only strict bare-framework relaxation; skip K optimization and MD."
    )
    parser.add_argument(
        "--adsorb-only", action="store_true",
        help="Run Change B single-K adsorption on accepted bare_relaxed.cif files only."
    )
    parser.add_argument(
        "--insert-only", action="store_true",
        help="Run Change C sequential K insertion on accepted Change A/B MOFs only."
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Ignore existing trajectory files and re-run all simulations."
    )
    parser.add_argument(
        "--top-percent", type=float, default=run_pipeline.TOP_PERCENT,
        help=f"Top %% of Stage-1 MOFs to advance to Stage 2 (default: {run_pipeline.TOP_PERCENT})"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device for CHGNet (default: auto-detect)"
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        choices=["isolated", "bulk"],
        help=(
            "K adsorption energy reference: 'isolated' (default) or 'bulk' BCC K. "
            "Does not change the rest of the workflow."
        ),
    )
    parser.add_argument(
        "--retry", type=Path, default=None,
        help="Path to a text file (e.g. retry_gpu.txt) containing MOFs to retry."
    )
    parser.add_argument(
        "--relax-fmax", type=float, default=run_pipeline.BARE_RELAX_FMAX,
        help=f"Bare relaxation FIRE target fmax in eV/Å (default: {run_pipeline.BARE_RELAX_FMAX})",
    )
    parser.add_argument(
        "--relax-max-steps", type=int, default=run_pipeline.BARE_RELAX_MAX_STEPS,
        help=f"Maximum FIRE steps for bare relaxation (default: {run_pipeline.BARE_RELAX_MAX_STEPS})",
    )
    parser.add_argument(
        "--relax-cell", action=argparse.BooleanOptionalAction,
        default=run_pipeline.BARE_RELAX_CELL,
        help="Relax the bare framework cell (default: enabled).",
    )
    parser.add_argument("--n-prescreen", type=int, default=run_pipeline.N_PRESCREEN)
    parser.add_argument("--n-full", type=int, default=run_pipeline.N_FULL)
    parser.add_argument("--adsorption-fmax", type=float, default=run_pipeline.ADSORPTION_FULL_FMAX)
    parser.add_argument("--adsorption-max-steps", type=int, default=run_pipeline.ADSORPTION_FULL_MAX_STEPS)
    parser.add_argument("--n-max", type=int, default=run_pipeline.INSERTION_N_MAX)
    parser.add_argument("--insert-n-prescreen", type=int, default=run_pipeline.INSERTION_N_PRESCREEN)
    parser.add_argument("--insert-n-full", type=int, default=run_pipeline.INSERTION_N_FULL)
    parser.add_argument("--min-kk", type=float, default=run_pipeline.INSERTION_MIN_KK)
    parser.add_argument("--v-threshold", type=float, default=run_pipeline.INSERTION_V_THRESHOLD)
    parser.add_argument("--insertion-fmax", type=float, default=run_pipeline.INSERTION_FMAX)
    parser.add_argument("--insertion-max-steps", type=int, default=run_pipeline.INSERTION_MAX_STEPS)
    parser.add_argument("--capacity-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Apply device setting before any CHGNet model is loaded
    if args.device:
        run_pipeline.USE_DEVICE = args.device
    if args.reference:
        run_pipeline.ADSORPTION_REFERENCE = args.reference
    run_pipeline.BARE_RELAX_FMAX = args.relax_fmax
    run_pipeline.BARE_RELAX_MAX_STEPS = args.relax_max_steps
    run_pipeline.BARE_RELAX_CELL = args.relax_cell

    resume = not args.no_resume
    top_percent = args.top_percent

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(OUTPUT_DIR)

    # Patch run_pipeline's internal logger so its utilities log to run_md.log
    run_pipeline._logger = logger

    from scientific.manifest import write_run_manifest

    manifest_path = write_run_manifest(
        OUTPUT_DIR,
        run_pipeline,
        cli_args={
            "bare": str(args.bare),
            "k": str(args.k),
            "device": args.device,
            "reference": args.reference or run_pipeline.ADSORPTION_REFERENCE,
            "no_resume": args.no_resume,
            "stage1_only": getattr(args, "stage1_only", False),
        },
    )
    logger.info("Run manifest written: %s", manifest_path)

    logger.info("=" * 60)
    logger.info("CHGNet MOF Screening — MD Simulation Runner")
    logger.info("=" * 60)
    logger.info("Bare CIFs (no K)     : %s", args.bare.resolve())
    logger.info("K-inserted CIFs      : %s", args.k.resolve())
    logger.info("Stage 1 MD   : %.1f ps", run_pipeline.FAST_MD_TIME_PS)
    logger.info("Stage 2 MD   : %.1f ps", run_pipeline.FINAL_MD_TIME_PS)
    logger.info("Top %%        : %.0f%%", top_percent)
    logger.info("Resume       : %s", resume)
    logger.info("Device       : %s", args.device or "auto")
    logger.info("K reference  : %s", run_pipeline.ADSORPTION_REFERENCE)
    logger.info("Opt fmax     : %.3f eV/Å", run_pipeline.FORCE_TOL)
    logger.info(
        "Bare relax   : fmax=%.3f eV/Å, steps=%d, relax_cell=%s",
        run_pipeline.BARE_RELAX_FMAX,
        run_pipeline.BARE_RELAX_MAX_STEPS,
        run_pipeline.BARE_RELAX_CELL,
    )

    if args.relax_only:
        run_relax_only(args.bare, args.k, resume=resume, logger=logger)
        return
    if args.adsorb_only:
        run_adsorb_only(
            args.bare,
            resume=resume,
            logger=logger,
            n_prescreen=args.n_prescreen,
            n_full=args.n_full,
            fmax=args.adsorption_fmax,
            max_steps=args.adsorption_max_steps,
        )
        return
    if args.insert_only:
        run_insert_only(
            args.bare,
            resume=resume,
            logger=logger,
            n_max=args.n_max,
            n_prescreen=args.insert_n_prescreen,
            n_full=args.insert_n_full,
            min_kk=args.min_kk,
            v_threshold=args.v_threshold,
            fmax=args.insertion_fmax,
            max_steps=args.insertion_max_steps,
            capacity_csv=args.capacity_csv,
        )
        return

    # ── 1. Discover CIF pairs ─────────────────────────────────────────────────
    pairs = pair_cif_files(args.bare, args.k, logger)
    if not pairs:
        logger.error("No valid CIF pairs found — exiting.")
        sys.exit(1)
        
    if args.retry and args.retry.is_file():
        logger.info("Reading retry list from %s", args.retry)
        with open(args.retry, "r", encoding="utf-8") as f:
            retry_mofs = {line.strip() for line in f if line.strip()}
        pairs = [p for p in pairs if p[0] in retry_mofs]
        logger.info("Filtered to %d MOFs for retry.", len(pairs))
        if not pairs:
            logger.error("No matching MOFs found for retry — exiting.")
            sys.exit(0)

    # ── 2. Compute / cache isolated K energy ──────────────────────────────────
    logger.info("Computing isolated K atom energy (needed for adsorption) ...")
    iso_k = run_pipeline.get_isolated_k_energy()
    logger.info("Isolated K energy: %.4f eV", iso_k)

    # ── 3. Stage 1: optimization + fast MD for all MOFs ──────────────────────
    logger.info("\n%s", "=" * 60)
    logger.info("STAGE 1 — Optimization + Fast MD (%.1f ps)", run_pipeline.FAST_MD_TIME_PS)
    logger.info("=" * 60)

    stage1_results: dict[str, dict] = {}
    progress: dict[str, dict] = {}

    for mof, bare_p, k_p in tqdm(pairs, desc="Stage 1", unit="MOF"):
        profiler.begin(f"Stage 1 - {mof}")
        res = run_stage1_md(mof, bare_p, k_p, resume=resume, logger=logger)
        run_pipeline.clear_cuda_memory(full=False)
        stage1_results[mof] = res
        progress[mof] = {"stage1": res}
        save_md_progress(progress, OUTPUT_DIR)
        profiler.end()

        # Terminal per-MOF status line
        ads_str = (
            f"{res['adsorption_energy_ev']:.4f} eV/K"
            if np.isfinite(res.get("adsorption_energy_ev", float("nan")))
            else "N/A"
        )
        status_icon = "✓" if res.get("success") else "✗"
        skip_tag = " [skipped — already done]" if res.get("skipped") else ""
        print(f"  {status_icon} {mof:40s} | Adsorption: {ads_str}{skip_tag}")

    # ── 4. Select top MOFs for Stage 2 ───────────────────────────────────────
    if args.stage1_only:
        logger.info("\nStage 2 skipped (--stage1-only flag set).")
        print_run_summary(stage1_results, {}, logger)
        return

    successful_s1 = {
        mof: res for mof, res in stage1_results.items()
        if res.get("success")
    }
    if not successful_s1:
        logger.error("No MOFs completed Stage 1 — cannot proceed to Stage 2.")
        print_run_summary(stage1_results, {}, logger)
        sys.exit(1)

    # Rank by screening score (or all if ≤1 MOF)
    ranked = sorted(
        successful_s1.items(),
        key=lambda kv: kv[1].get("screening_score", float("-inf")),
        reverse=True,
    )
    n_select = max(1, int(np.ceil(len(ranked) * top_percent / 100.0)))
    selected_mofs = [mof for mof, _ in ranked[:n_select]]

    logger.info("\n%s", "=" * 60)
    logger.info(
        "STAGE 2 — Long MD (%.1f ps) for top %d / %d MOFs (%.0f%%)",
        run_pipeline.FINAL_MD_TIME_PS,
        n_select, len(ranked), top_percent,
    )
    logger.info("Selected: %s", ", ".join(selected_mofs))
    logger.info("=" * 60)

    # ── 5. Stage 2: long MD for selected MOFs ────────────────────────────────
    stage2_results: dict[str, dict] = {}

    for mof in tqdm(selected_mofs, desc="Stage 2", unit="MOF"):
        profiler.begin(f"Stage 2 - {mof}")
        res = run_stage2_md(mof, resume=resume, logger=logger)
        run_pipeline.clear_cuda_memory(full=False)
        stage2_results[mof] = res
        progress[mof]["stage2"] = res
        save_md_progress(progress, OUTPUT_DIR)
        profiler.end()

        status_icon = "✓" if res.get("success") else "✗"
        skip_tag = " [skipped — already done]" if res.get("skipped") else ""
        print(f"  {status_icon} {mof:40s} | Stage 2 MD{skip_tag}")

    # ── 6. Summary ────────────────────────────────────────────────────────────
    print_run_summary(stage1_results, stage2_results, logger)


if __name__ == "__main__":
    main()
