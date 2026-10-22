# MOF Screening Pipeline — Workflow, Settings, & Design

> Complete walkthrough of the **implemented CHGNet MOF battery screening workflow**
> running in the root workspace. Two main user-facing scripts communicating through 
> a shared core library ([run_pipeline.py](file:///c:/Users/eshan/OneDrive/Desktop/CHGNETT/run_pipeline.py)) and files on disk.

---

# PART 1 — Pipeline Overview

---

## What It Is

The MOF screening pipeline evaluates whether a metal-organic framework (MOF) is a viable **potassium-ion battery anode** using CHGNet (a pretrained universal graph neural network potential).

The workflow calculates **four core screening parameters** for each MOF:
1. **Adsorption Energy** (`adso_ener`): Thermodynamic binding energy (eV per K atom).
2. **Diffusion Coefficient** (`diff_coeff`): Kinetic transport rate of K ions (cm²/s).
3. **Mean Squared Displacement** (`msd`): Average displacement of K ions during MD (Å²).
4. **Mechanical Stability** (`mech_stab`): Framework structural integrity index (MSI, 0–1).

The current workflow is split into **two major steps**:
1. **Simulation Runner** (`run_md.py`): Performs structural optimizations, calculates adsorption sites, and runs molecular dynamics (equilibration + production).
2. **Post-Processing Analyzer** (`analyse.py`): Performs trajectory analysis, calculates MSD and diffusivity via multiple tiers (MD Einstein, NEB-lite TST, BVSE), computes mechanical stability, and writes final results.

They share a common core engine module, `run_pipeline.py`, which contains configuration constants, PyTorch/CHGNet models, and shared computational utilities.

```
                  [ cifs/ + bare_cifs/ ]
                            │
                            ▼
                        run_md.py
                     (Simulation)
                            │
                            ▼  [ outputs/structures/ & outputs/checkpoints/ ]
                        analyse.py
                    (Post-Processing)
                            │
                            ▼
              [ outputs/results.csv + best_cif.csv ]
```

---

## Run Workflow

| Step | Script | Input Dependencies | Produces |
|------|--------|--------------------|----------|
| 1 | `run_md.py` | `inputs/bare_cifs/`, `inputs/cifs/` | Optimized CIFs, MD trajectories, checkpoints JSON |
| 2 | `analyse.py` | Step 1 outputs | `results.csv`, `best_cif.csv`, `best_mof.csv`, `best_cifs/`, plots |

### Stage 1 & Stage 2 Simulation Strategy
To conserve computational resources, `run_md.py` operates in two stages:
- **Stage 1 (Fast Screening)**: Runs geometry optimization of bare and K-inserted frameworks, evaluates adsorption energy via global site search, and runs a short **10 ps equilibration MD** (Berendsen thermostat) for *all* candidate MOF pairs.
- **Stage 2 (Accurate Production)**: Evaluates a fast screening score for all completed MOFs and ranks them. The **top 20%** (`TOP_PERCENT` setting) are selected to continue with a long **50 ps Nose-Hoover NVT MD** production run (inheriting Stage 1 coordinates and velocities).

---

## Output Directory Tree

All outputs land under `outputs/` in the working directory:

```
outputs/
├── run_md.log                     ← Log file for run_md.py
├── analysis.log                   ← Log file for analyse.py
├── run_manifest.json              ← Settings / parameters manifest for audit
├── md_progress.json               ← Progress tracker for MD simulations
├── isolated_K_energy.json         ← Cached isolated K reference energy (or bulk BCC K)
├── results.csv                    ← Main database containing: MOF, adso_ener, diff_coeff, msd, mech_stab
├── best_cif.csv                   ← Filtered database containing: best_cif, adso_ener, diff_coeff, msd, mech_stab
├── best_mof.csv                   ← Details of the single top-ranked MOF candidate
├── best_cifs/                     ← Exported optimized K-inserted CIFs of best candidates
├── plots/                         ← Directory containing diffusion MSD and profile plots
├── checkpoints/
│   └── MOF_name.json              ← Checkpoint cache for resume/caching logic
└── structures/
    └── MOF_name/                  ← Trajectory & CIF outputs per MOF
        ├── stage1_md.traj         ← NVT MD trajectory (10 ps Berendsen)
        ├── stage1_md.log          ← MD thermodynamic log
        ├── stage2_md.traj         ← Continuation NVT MD trajectory (50 ps Nose-Hoover)
        ├── stage2_md.log          ← Continuation MD thermodynamic log
        ├── bare_optimized.cif     ← Optimized bare framework structure
        ├── k_inserted_optimized.cif ← Optimized K-inserted structure
        └── k_on_bare_optimized.cif  ← Globally best adsorption site structure
```

---

# PART 2 — Phase 0: Reference Energy (`run_md.py`)

---

Before any MOF calculations are performed, the pipeline establishes a reference energy for potassium. 

**Why?** Absolute interatomic potential energies are model-dependent. Adsorption energies are differences, meaning the reference state of the inserted K ion must be computed self-consistently with the exact same model and device.

### Reference Modes
Two reference modes are supported via the `--reference` CLI option:
1. **Isolated (`"isolated"`)** [Default]: Computes the energy of a single K atom in a large vacuum box ($20.0 \times 20.0 \times 20.0$ Å³). This is standard for gas-phase-like adsorption conventions.
2. **Bulk (`"bulk"`)**: Relax BCC metallic potassium (lattice constant $a = 5.225$ Å, 2 atoms per cell). Useful for electrode intercalation energetics where insertion is compared to bulk metal anodes.

Once computed, the reference energy is cached in `outputs/isolated_K_energy.json` (or `outputs/bulk_K_energy.json`).

---

# PART 3 — Phase 1: Simulations (`run_md.py`)

---

## Input Requirements
- `inputs/bare_cifs/`: Bare MOF CIF files (without K).
- `inputs/cifs/`: Paired K-inserted MOF CIF files. 
- Filenames in both folders must match exactly to pair successfully.

---

## Geometry Optimization & Adsorption
For each pair:
1. **Bare Optimization**: Bare framework is relaxed with a force tolerance `FORCE_TOL` ($0.01$ eV/Å) and its energy cached.
2. **Adsorption Site Search**: A global chemical site search discovers candidate sites (carboxylate oxygen, bridge sites, open metals, pore minima). K-only relaxation on the fixed framework determines the lowest-energy adsorption site.
3. **K-inserted Optimization**: The K-inserted framework is optimized.
4. **Adsorption Energy**: Calculated as:
   $$\text{adso\_ener} = \frac{E(\text{MOF} + n_{\text{K}}\text{K}) - E(\text{bare}) - n_{\text{K}} \times E(\text{ref})}{n_{\text{K}}}$$

---

## Molecular Dynamics Ensembles
- **Stage 1 (Equilibration)**: runs for 10 ps NVT at 300 K using a **Berendsen** thermostat. A larger timestep of 1 fs is utilized to guarantee stability while allowing initial thermalisation.
- **Stage 2 (Production)**: runs for 50 ps NVT Nose-Hoover to simulate physically accurate canonical ensemble kinetics, continuing directly from the final coordinates/velocities of Stage 1.

### Thermal Runaway Protection
If the instantaneous framework temperature spikes above a threshold (e.g. $T > \text{Target} + \text{Margin}$), the MD simulation is aborted. The partial trajectory is written to disk for debugging, and a validation flag is set.

---

# PART 4 — Phase 2: Post-Processing & Analysis (`analyse.py`)

---

## Four Parameters Calculated
`analyse.py` reads raw trajectories and outputs:
1. **Adsorption Energy** (`adso_ener`): Re-extracted from checkpoints or recalculated.
2. **Mean Squared Displacement** (`msd`): Evaluated over periodic-boundary unwrapped K trajectories.
3. **Diffusion Coefficient** (`diff_coeff`):
   - **Tier 1 (MD Einstein)**: Extracted via a linear fit on the unwrapped MSD trajectory (skipping 20% equilibration, fitting to the longest Fickian window). Requires $R^2 \ge 0.90$ and t-statistic $\ge 2.0$ for statistical validation.
   - **Tier 2 (NEB+TST)**: Runs K-only NEB-lite image relaxation on candidate hops to compute migration barriers ($E_a$) and transition-state theory Arrhenius diffusivity. Used to corroborate and cross-validate MD results.
   - **Tier 3 (BVSE Fallback)**: Computes a Bond-Valence Site Energy landscape scan to estimate diffusion if MD and NEB both fail due to localized ions or short simulations.
4. **Mechanical Stability** (`mech_stab`): A composite structural integrity index (weights: framework RMSD 25%, bond preservation 25%, atomic displacement 15%, coordination preservation 15%, energy fluctuation 10%, volume fluctuation 10%).

---

## Continuous Overall Score (0–100)
Candidates are ranked using a continuous weighted score combining the normalized parameters:
- **Diffusion** (`diff_coeff`): 30%
- **Adsorption** (`adso_ener`): 25% (rewards ideal window $[-2.5, -0.5]$ eV)
- **Mechanical Stability** (`mech_stab`): 20%
- **MSD** (`msd`): 10%
- **Volume Expansion** (internal metric): 10%
- **Confidence**: 5%

Recommendation categories range from **Excellent** (score $\ge 85$) to **Not Recommended** (for collapsed frameworks).

---

# PART 5 — Configuration & Settings

Tunable parameters are stored in the configuration block of `run_pipeline.py` and shared:

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `TEMPERATURE_K` | `300` | NVT MD target temperature (K) |
| `TIME_STEP_FS` | `1` | MD timestep (fs) |
| `FAST_MD_TIME_PS` | `10` | Stage 1 equilibration MD duration (ps) |
| `FINAL_MD_TIME_PS` | `50` | Stage 2 production MD duration (ps) |
| `FORCE_TOL` | `0.01` | Core geometry relaxation force threshold (eV/Å) |
| `FORCE_TOL_LOOSE` | `0.05` | Fallback geometry relaxation threshold (eV/Å) |
| `TOP_PERCENT` | `20` | Percentage of Stage 1 MOFs selected for Stage 2 |
| `ADSORPTION_REFERENCE`| `"isolated"` | K ion reference: `"isolated"` or `"bulk"` |
| `STABILITY_RMSD_STABLE_A`| `0.5` | Threshold for framework RMSD mechanical stability (Å) |
| `BOND_PRESERVATION_STABLE`| `0.90` | Fraction of framework bonds preserved in MD |
| `MIN_DIFFUSION_FIT_R2`| `0.85` | Minimum $R^2$ fit threshold for MD diffusion |

---

# PART 6 — Validation Flags

The pipeline sets validation flags to alert users of potential issues:

| Flag | Meaning | Code Response |
|------|---------|---------------|
| `k_overpacked` | Too many K ions inserted | Score penalized |
| `k_ions_too_close` | K–K distance $< 3.5$ Å | Score penalized |
| `optimization_failed` | Relaxation failed to converge | Marked as Failed |
| `md_failed` | Trajectory crashed or thermal runaway | Marked as Failed |
| `md_not_converged` | Large energy drift / temperature std | Flagged, confidence reduced |
| `diffusion_not_reliable`| MSD did not satisfy Fickian criteria | Fallback to NEB/BVSE |
| `large_volume_change` | Expansion exceeds threshold ($> 5.0\%$) | Flagged, score penalized |
| `msd_non_monotonic` | MSD shows localized/rattling behavior | Flagged, confidence reduced |
