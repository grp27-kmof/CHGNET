# MOF Screening Pipeline — Calculations & Physics

> Deep dive into the physical formulas, algorithms, and scoring metrics implemented in the CHGNet MOF screening workflow for the four calculated parameters.

---

## The Four Core Parameters

A viable metal-organic framework (MOF) anode for potassium-ion batteries must satisfy multiple physical requirements. The pipeline calculates **four core parameters**:
1. **Adsorption Energy** (`adso_ener`): Thermodynamic binding energy of potassium insertion (eV per K atom).
2. **Diffusion Coefficient** (`diff_coeff`): Kinetic transport rate of K ions through framework pores (cm²/s).
3. **Mean Squared Displacement** (`msd`): Average displacement of K ions over time (Å²).
4. **Mechanical Stability** (`mech_stab`): Framework structural integrity index (MSI, 0–1).

---

# 1. Adsorption Energy (`adso_ener`)

## Physical Definition & Formula
Adsorption energy measures the thermodynamic driving force for inserting potassium into the host framework. The average binding energy per K atom (`adso_ener`) is computed as:

$$\text{adso\_ener} = \frac{E(\text{MOF} + n_{\text{K}}\text{K}) - E(\text{bare MOF}) - n_{\text{K}} \times E(\text{ref})}{n_{\text{K}}}$$

Where:
- $E(\text{MOF} + n_{\text{K}}\text{K})$ is the relaxed energy of the K-inserted framework.
- $E(\text{bare MOF})$ is the relaxed energy of the bare host framework.
- $E(\text{ref})$ is the potassium reference energy, resolved under the selected reference mode:
  - **Isolated (`"isolated"`)** [Default]: Energy of a single K atom in a large vacuum box ($20.0 \times 20.0 \times 20.0$ Å³).
  - **Bulk (`"bulk"`)**: Energy per K atom in metallic BCC potassium.
- $n_{\text{K}}$ is the number of inserted K atoms.

## Algorithm: Adsorption Site Search
To find the global energy minimum:
1. Candidate sites are identified by scanning a grid inside the framework void channels, and detecting coordination shells (metals, carboxylate/bridge oxygens) using `CrystalNN`.
2. A fast, K-only geometry relaxation is performed at each candidate site with framework atoms fixed, saving CPU time.
3. The configuration with the lowest energy is selected for full relaxation to yield the globally optimized $E(\text{MOF} + n_{\text{K}}\text{K})$ structure.

## Scoring
The adsorption energy is scored using a Gaussian function centered around the target energy:
- **Target Optimum**: $-1.5$ eV/K
- **Ideal Range**: $[-2.5, -0.5]$ eV/K
- **Gaussian Standard Deviation ($\sigma$)**: $0.75$ eV

---

# 2. Diffusion Coefficient (`diff_coeff`) — Multi-Tier Engine

Ion transport is evaluated via a multi-tier approach to ensure reliable predictions across different simulation timescales.

```
                    ┌────────────────────────┐
                    │  Trajectories & CIFs   │
                    └───────────┬────────────┘
                                │
                                ▼
         [ Tier 1 ] ───► Einstein MSD Fit (MD)  ───► (Passed R² & t-stat?)
                                │                                │ Yes
                                │ No                             ▼
                                │                     Report Tier 1 D
                                ▼
         [ Tier 2 ] ───► NEB-lite + TST Barrier ───► (Converged & valid?)
                                │                                │ Yes
                                │ No                             ▼
                                ▼                     Report Tier 2 D
         [ Tier 3 ] ───► BVSE Landscape Fallback ───► Report Tier 3 D
```

---

## Tier 1: Einstein MD (Unwrapped MSD)
For liquid-like or fast-solid diffusion, the diffusion coefficient is extracted from the NVT MD trajectory.

### PBC Coordinate Unwrapping
Because trajectories in periodic boundary conditions (PBC) have coordinate jumps when atoms cross the unit cell borders, raw coordinate differences cannot be used. Fractional displacements are unwrapped step-by-step using minimum-image logic:

$$\Delta \mathbf{s}_f(t) = \mathbf{s}_f(t) - \mathbf{s}_f(t-1)$$

$$\Delta \mathbf{s}_{f,\text{unwrap}}(t) = \Delta \mathbf{s}_f(t) - \text{round}\left(\Delta \mathbf{s}_f(t)\right)$$

Unwrapped Cartesian positions are then reconstructed.

### Mean Squared Displacement (MSD) & Einstein Fit
The MSD for the $n_{\text{K}}$ potassium ions is computed over frames:

$$\text{MSD}(t) = \frac{1}{n_{\text{K}}} \sum_{i=1}^{n_{\text{K}}} \langle |\mathbf{r}_i(t) - \mathbf{r}_i(0)|^2 \rangle$$

The 3D isotropic Einstein diffusion coefficient is related to the slope of MSD vs. time:

$$\text{MSD}(t) = 6Dt \implies D = \frac{1}{6} \times \text{slope}(\text{MSD vs. } t)$$

- **Fit Window**: The first 20% of the trajectory is discarded (equilibration). The fit is optimized over the longest monotonic region.
- **Statistical Gating**: A fit is accepted only if $R^2 \ge 0.90$ and the slope t-statistic $t_{\text{stat}} \ge 2.0$.

---

## Tier 2: NEB-lite + Transition State Theory (NEB+TST)
When MD runs are too short to sample diffusion, or K ions are trapped, Tier 2 is triggered.

### Hop Path Search & NEB-lite
1. Pairwise hops between stable K positions are identified within a distance cutoff ($3.5$ to $8.0$ Å).
2. For each hop, intermediate images (default 5) are generated along the path.
3. A constrained **NEB-lite** relaxation is performed: intermediate images are relaxed with the mobile K atom free to move, but the host framework atoms fixed in place.
4. The migration barrier $E_a$ is the peak energy along the relaxed path:

$$E_a = \max(E_{\text{images}}) - \min(E_{\text{endpoints}})$$

### Arrhenius Diffusivity
The diffusivity is calculated using the transition state theory expression:

$$D_{\text{Arrhenius}} = \frac{a^2}{2d} \nu^* \exp\left(-\frac{E_a}{k_B T}\right)$$

Where:
- $a$ is the hop distance (Å).
- $d$ is the dimensionality of diffusion ($d = 3$ for isotropic).
- $\nu^*$ is the attempt frequency, estimated at 300 K as:
  
$$\nu^* = \frac{k_B T}{h} \approx 6.25 \times 10^{12} \text{ Hz}$$

- **Rate-Limiting Barrier Selection**: The rate-limiting elementary hop that dominates macroscopic network diffusion is the **lowest reliable migration barrier** on the connected network (not the median or maximum).

---

## Tier 3: BVSE Landscape Fallback
If MD and NEB both fail (e.g. no valid hops converged), the pipeline evaluates a Bond-Valence Site Energy (BVSE) grid scan as a last resort fallback.

### Formulation
The valence energy at grid position $\mathbf{r}$ (spacing $0.3$ Å) is computed by summing interactions with host atoms:

$$E_{\text{BVSE}}(\mathbf{r}) = \sum_{j \in \text{fw}} \left[ \exp\left(\frac{R_0 - d_j}{b}\right) - \frac{R_0}{d_j} \right]^2$$

Where $R_0$ is the bond valence parameter for $\text{K}^+ - \text{anion}$, $d_j$ is the distance to framework atom $j$, and $b = 0.37$ Å.
The dimensionless sum-of-squares is converted to physical energy scaling via:

$$1 \text{ BVSE unit} \approx 0.05 \text{ eV}$$

The migration barrier is estimated from the energy difference between the accessible channels (estimated using the 30th and 10th percentiles of the grid energy distribution).

---

# 3. Mean Squared Displacement (`msd`)

Mean Squared Displacement measures the average spatial displacement of potassium ions over the molecular dynamics simulation time:

$$\text{msd}(t) = \frac{1}{n_{\text{K}}} \sum_{i=1}^{n_{\text{K}}} \langle |\mathbf{r}_i(t) - \mathbf{r}_i(0)|^2 \rangle$$

The reported `msd` value represents the final, maximum MSD achieved by K ions at the end of the trajectory.
A larger MSD indicates a greater exploration of the framework void space, showing that K ions are mobile rather than localized or rattling in a single potential well.

---

# 4. Mechanical Stability (`mech_stab`)

Mechanical stability during MD is evaluated using a continuous index (MSI, 0 to 1) built from six normalized parameters:

$$\text{mech\_stab} = 0.25 S_{\text{RMSD}} + 0.25 S_{\text{bond}} + 0.15 S_{\text{disp}} + 0.15 S_{\text{coord}} + 0.10 S_{\text{energy}} + 0.10 S_{\text{volume}}$$

Where:
- $S_{\text{RMSD}}$: Score based on time-averaged framework root-mean-square displacement (threshold: stable $< 0.5$ Å).
- $S_{\text{bond}}$: Score based on the fraction of framework bonds preserved throughout the trajectory (threshold: stable $> 90\%$).
- $S_{\text{disp}}$: Score based on the maximum atomic displacement of any single framework atom (threshold: stable $< 1.0$ Å).
- $S_{\text{coord}}$: Score representing $\text{K}-\text{O}$ coordination shell preservation.
- $S_{\text{energy}}$: Score based on energy fluctuations per atom ($\sigma_E$ threshold: stable $< 0.05$ eV/atom).
- $S_{\text{volume}}$: Score based on volume fluctuations during NVT.

### Classification
- **Stable** ($\text{mech\_stab} \ge 0.70$): Framework remains fully intact.
- **Marginal** ($0.45 \le \text{mech\_stab} < 0.70$): Minor local distortions or bond breaking.
- **Collapsed** ($\text{mech\_stab} < 0.45$): Severe structural collapse or dissociation.
