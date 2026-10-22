# MOF Screening Pipeline — Supporting Engine Architecture

> Technical documentation of the core background utilities, graph construction, caching, checkpointing, and GPU VRAM management powering the MOF screening pipeline and the calculations of its four key parameters.

---

# 1. CHGNet Loading & Device Selection

Calculations in the pipeline (relaxation, molecular dynamics, NEB-lite) are computed via PyTorch using the **CHGNet interatomic potential**. 

## Model Initialization
- **Model weights**: Default is version `0.3.0` trained on Figshare MPtrj (Materials Project Trajectory) DFT calculations.
- **Class instantiation**: PyTorch loading is coordinated through `get_chgnet()` in [run_pipeline.py](file:///c:/Users/eshan/OneDrive/Desktop/CHGNETT/run_pipeline.py).
- **GPU memory mode**: Uses `torch.set_float32_matmul_precision("medium")` to speed up matrix operations on NVIDIA tensor cores.

## Multi-Device Selection Logic
The pipeline handles GPU device allocation dynamically:
1. Explicit CLI argument (`--device cpu` / `--device cuda`).
2. Checks if Apple Metal Performance Shaders (MPS) are available (`mps`).
3. Checks if NVIDIA CUDA is available (`cuda`).
4. Fallback to CPU (`cpu`).

To prevent VRAM conflicts, `analyse.py` monkey-patches the device configuration to run single-point evaluations on the **CPU** when the GPU is heavily loaded by running simulations.

---

# 2. Crystal Graph Converter

Before forward-passing structures into the GNN, pymatgen structures are converted into dual-graphs:
- **Atom Graph**: Captures neighbor connections within a cutoff of $6.0$ Å.
- **Bond Graph**: Captures three-body bond-angle correlations within a cutoff of $3.0$ Å.

The pipeline utilizes the **Cython** implementation (`cygraph.pyx`) when compiled, yielding a $10\times$ speedup over the pure-python fallback. In addition, the graph converter is set to ignore isolated atom errors during grid searches to prevent crashes.

---

# 3. JSON Checkpointing & Caching

The pipeline is designed to be fully idempotent and resume-safe. Progress is cached using three mechanisms on disk:

1. **Reference Energy Cache**:
   - `outputs/isolated_K_energy.json` (or `bulk_K_energy.json`) caches the relaxed reference energy of potassium (used to calculate `adso_ener`).
2. **Workflow Checkpoints**:
   - `outputs/checkpoints/<MOF>.json` stores complete state records.
   - Example Schema:
     ```json
     {
       "stage": 2,
       "simulation_time_ps": 50.0,
       "adsorption_energy_ev_per_k_v4": -1.234,
       "adsorption_site_type": "carboxylate_O",
       "mechanical_stability": "Stable",
       "mechanical_stability_index": 0.812,
       "overall_score": 76.5
     }
     ```
3. **Trajectory skip guards**:
   - When resuming, `run_md.py` counts the frames in the `.traj` file. If the file is incomplete (e.g., due to an interrupted run or crash), it deletes it and restarts the simulation.

---

# 4. Trajectory Unwrapping Engine (for `msd` & `diff_coeff`)

Calculating `msd` and `diff_coeff` requires unwrapped atomic coordinates to avoid spurious periodic-crossing jumps. Fractional coordinates are unwrapped inside `scientific/msd_analysis.py` using:

$$\Delta \mathbf{s}_f(t) = \mathbf{s}_f(t) - \mathbf{s}_f(t-1)$$
$$\mathbf{s}_{f,\text{unwrap}}(t) = \mathbf{s}_{f,\text{unwrap}}(t-1) + \left[ \Delta \mathbf{s}_f(t) - \text{round}\left(\Delta \mathbf{s}_f(t)\right) \right]$$

Unwrapped coordinates are converted to Cartesian space. The unwrapping process is fully vectorized in NumPy, speeding up post-processing by $100\times$ to $800\times$ compared to standard looping.

---

# 5. NEB-lite Engine (for `diff_coeff`)

Rather than invoking a heavy external NEB script, `analyse.py` implements an inline **NEB-lite** solver for transition state theory (TST) diffusion validation.

1. **Hop Discovery**: Candidate pathways are identified by matching stable sites within $3.5$ to $8.0$ Å.
2. **Image Generation**: Discovers paths and linear-interpolates 5 intermediate images.
3. **Image Relaxation**: For each image, host framework atoms are locked, and only the migrating K atom coordinates are relaxed under CHGNet forces.
4. **Path Validation**: Profiles are validated by checking energy smoothness, endpoint alignment, and ensuring no atom-framework overlaps occur ($d_{\text{min}} < 1.0$ Å).

---

# 6. BVSE Grid Scanner (for `diff_coeff` fallback)

When MD and NEB both fail (e.g. for slow-diffusing frameworks), Tier 3 diffusion estimates are computed using a simplified Bond-Valence Site Energy (BVSE) scan:
- **Grid Scan**: Scans the empty framework unit cell on a 3D Cartesian grid (spacing $0.3$ Å).
- **Simplified Valence**: Evaluates interaction energy sum-of-squares at each point:
  $$E_{\text{BVSE}}(\mathbf{r}) = \sum_j \left[ \exp\left(\frac{R_0 - d_j}{b}\right) - \frac{R_0}{d_j} \right]^2$$
- **Energy Conversion**: Scales the dimensionless sum-of-squares to physical energy ($1 \text{ BVSE unit} \approx 0.05 \text{ eV}$).
- **Barrier Extraction**: Barrier is estimated from the energy difference between the 30th and 10th percentiles of grid values.

---

# 7. Mechanical Stability Index (MSI) Evaluator (for `mech_stab`)

The **Mechanical Stability Index (MSI)** is a composite scalar in $[0, 1]$ that quantifies framework structural integrity across the MD trajectory. It is computed from six physical sub-scores, each normalized to $[0, 1]$, and combined by a dynamically renormalized weighted average.

## Sub-scores & Formulas

| # | Indicator | Formula | Weight |
|---|-----------|---------|--------|
| 1 | **RMSD** | $\text{score} = \max\!\left(0,\; 1 - \dfrac{\overline{\text{RMSD}}}{\text{RMSD}_\text{marginal}}\right)$, $\;\text{RMSD}_\text{marginal} = 1.5\,\text{Å}$ | 0.25 |
| 2 | **Bond Preservation** | $\text{score} = \overline{f_\text{bond}}$, mean fraction of covalent bonds within $1.3\times d_0$ across all frames | 0.25 |
| 3 | **Atomic Displacement** | $\text{score} = \max\!\left(0,\; 1 - \dfrac{\max(\delta_i)}{d_\text{marginal}}\right)$, $\;d_\text{marginal} = 3.0\,\text{Å}$ | 0.15 |
| 4 | **Coordination Preservation** | $\text{score} = \overline{f_\text{coord}}$, mean fraction of reference K–O contacts retained per frame (PBC-aware) | 0.15 |
| 5 | **Energy Fluctuation** | $\text{score} = \max\!\left(0,\; 1 - \dfrac{\sigma_E / \text{atom}}{\sigma_\text{max}}\right)$, $\;\sigma_\text{max} = 0.05\,\text{eV/atom}$ | 0.10 |
| 6 | **Volume Fluctuation** | $\text{score} = \max\!\left(0,\; 1 - \dfrac{\sigma_V / \langle V \rangle}{f_\text{max}}\right)$, $\;f_\text{max} = 0.02$ | 0.10 |

## Dynamic Weight Renormalization

A key design requirement is that **missing metrics must not penalise the MSI**. For example, when energy/volume data is unavailable (e.g. NVT with no log file) the old approach injected a hardcoded neutral score of `0.5`, silently biasing the result downward whenever the true sub-score would be close to 1.

The implementation now uses **strict renormalization**: only the weights of metrics that are **finite and available** are summed to form the denominator:

$$\text{MSI} = \frac{\sum_{i \in \mathcal{A}} w_i \cdot s_i}{\sum_{i \in \mathcal{A}} w_i}$$

where $\mathcal{A}$ is the set of available (finite) indicators. This guarantees:
- A perfectly stable structure scores **1.0** regardless of which subset of metrics is available.
- A completely unavailable metric contributes **zero** to both numerator and denominator.

## PBC-Aware Coordinate Handling

Two PBC corrections were added to eliminate systematic biases:

### 1. Kabsch Alignment on Unwrapped Positions
Before computing per-atom RMSD and max-displacement, the pipeline now **unwraps the trajectory** using sequential minimum-image displacements:

$$\Delta \mathbf{r}_\text{MIC}(t) = \mathbf{r}(t) - \mathbf{r}(t{-}1) - \mathbf{L}\,\text{round}\!\left(\frac{\mathbf{r}(t) - \mathbf{r}(t{-}1)}{\mathbf{L}}\right)$$

Kabsch rigid-body alignment is applied to unwrapped framework positions, eliminating ghost displacements that arise when atoms cross cell boundaries between frames.

### 2. PBC-Aware K–O Distance for Coordination Preservation
K–O contact distances are now computed using the minimum image convention:

$$d_\text{MIC}(\mathbf{r}_{KO}) = \left|\mathbf{L}\left(\text{frac}(\mathbf{r}_{KO} \cdot \mathbf{L}^{-1}) - \text{round}(\text{frac}(\mathbf{r}_{KO} \cdot \mathbf{L}^{-1}))\right)\right|$$

This correctly identifies contacts near periodic boundaries that would otherwise appear as broken bonds when measured as naive Cartesian distances. Cell matrices are extracted per-frame to support NPT-like volume drift.

## Classification Thresholds

| Label | MSI Range |
|-------|-----------|
| **Stable** | $\geq 0.70$ |
| **Marginal** | $0.45$ – $0.70$ |
| **Collapsed** | $< 0.45$ |

---

# 8. VRAM Management & CPU Fallback

To prevent PyTorch from raising Out-Of-Memory (OOM) errors on consumer-grade GPUs, the pipeline incorporates active memory garbage collection:
1. **ASE Calculator Detachment**: ASE `Atoms` objects retain references to their calculators, creating memory-leak loops. The pipeline breaks these cycles after evaluations by setting:
   ```python
   atoms.calc = None
   ```
2. **Torch Cache Flushing**: Triggers `torch.cuda.empty_cache()` and `torch.cuda.ipc_collect()` after each MOF evaluation.
3. **CPU Fallback Retries**: If a CUDA OOM error occurs during optimization, the GNN weights are moved to system memory (`device = "cpu"`), the calculation is retried on the CPU, and the model is returned to CUDA for subsequent calculations.
