# Research Paper References — CHGNet MOF Screening Pipeline

> Peer-reviewed citations justifying the physics, formulas, and methods implemented in the 2-script screening workflow (`run_md.py` → `analyse.py`) for the four calculated parameters (`adso_ener`, `diff_coeff`, `msd`, `mech_stab`).

---

## Core Citations Required for Publications

If you publish results generated using this pipeline, please cite the following papers:

1. **CHGNet Model**: Deng et al. (2023) — Universal machine learning interatomic potential.
2. **ASE Framework**: Larsen et al. (2017) — Trajectory I/O, relaxation, and MD thermostats.
3. **Property-Specific Citations**: Please cite the corresponding methodological papers listed in the tables below for the specific screening parameters you report.

---

# 1. CORE ENGINE & CALCULATORS

| Step / Component | Purpose | Citations | DOI / Link |
|------------------|---------|-----------|------------|
| Core Potential | Graph Neural Network interatomic potential predicting energy, forces, stress, and magnetic moments | Deng, B., et al. *Nature Machine Intelligence* (2023) | [10.1038/s42256-023-00716-3](https://doi.org/10.1038/s42256-023-00716-3) |
| MPtrj Dataset | Training set for the default 0.3.0 model weights | Figshare dataset (2023) | [10.6084/m9.figshare.23713842](https://doi.org/10.6084/m9.figshare.23713842) |
| r2SCAN Weights | Optional upgrade weights for transition-metal-heavy MOF nodes | Huang, X., et al. *npj Computational Materials* (2025) | [10.1038/s41524-025-01796-y](https://doi.org/10.1038/s41524-025-01796-y) |
| ASE Framework | Atomic simulation environment wrapping calculations and trajectories | Larsen, A. H., et al. *J. Phys.: Condens. Matter* (2017) | [10.1088/1361-648X/aa680e](https://doi.org/10.1088/1361-648X/aa680e) |

---

# 2. SIMULATIONS & THERMODYNAMICS (`run_md.py` / `adso_ener`)

| Parameter / Method | Justification | Citations | DOI / Link |
|------------------|---------------|-----------|------------|
| Adsorption Energy (`adso_ener`) | Standard reference-subtraction formula: $\Delta E = E_{\text{ins}} - E_{\text{bare}} - nE_{\text{ref}}$ | Yildirim, H., et al. *ACS Applied Materials & Interfaces* (2015) | [10.1021/acsami.5b04586](https://doi.org/10.1021/acsami.5b04586) |
| K-Metal Reference | Self-consistent references computed at the same level of theory as the compounds | Wang, L., et al. *Physical Review B* (2006) | [10.1103/PhysRevB.73.195107](https://doi.org/10.1103/PhysRevB.73.195107) |
| K-Ion Batteries | Scientific motivation for K-ion chemistry and anode targets | Tian, Y., et al. *Chemical Reviews* (2021) | [10.1021/acs.chemrev.0c00767](https://doi.org/10.1021/acs.chemrev.0c00767) |
| MOF Anodes | Rationale for Metal-Organic Frameworks as intercalation/insertion electrode candidates | Liang, Z., et al. *Advanced Materials* (2018) | [10.1002/adma.201702891](https://doi.org/10.1002/adma.201702891) |
| Berendsen Thermostat | NVT equilibration ensemble utilized in Stage 1 MD simulations | Berendsen, H. J. C., et al. *Journal of Chemical Physics* (1884) | [10.1063/1.448118](https://doi.org/10.1063/1.448118) |
| Nose-Hoover Thermostat | Production canonical MD ensemble (canonical kinetics) utilized in Stage 2 MD simulations | Nosé, S. *Journal of Chemical Physics* (1984) | [10.1063/1.447334](https://doi.org/10.1063/1.447334) |

---

# 3. ANALYSIS & TRANSPORT (`analyse.py` / `diff_coeff`, `msd`, `mech_stab`)

| Parameter / Method | Justification | Citations | DOI / Link |
|------------------|---------------|-----------|------------|
| Einstein MSD (`msd` & `diff_coeff`) | Diffusivity extracted from mean squared displacement under Fickian regimes | He, X., et al. *Energy & Environmental Science* (2018) | [10.1039/C8EE01848E](https://doi.org/10.1039/C8EE01848E) |
| PBC Unwrapping (`msd`) | Coordinate unwrapping via minimum-image convention to prevent coordinate jumps | Plimpton, S. J. *Journal of Computational Physics* (1995) | [10.1006/jcph.1995.1039](https://doi.org/10.1006/jcph.1995.1039) |
| Climbing-Image NEB (`diff_coeff` Tier 2) | Finding exact transition states and activation barriers along K hops | Henkelman, G., et al. *Journal of Chemical Physics* (2000) | [10.1063/1.1329672](https://doi.org/10.1063/1.1329672) |
| TST Diffusivity (`diff_coeff` Tier 2) | Arrhenius diffusivity calculated from activation barriers using Vineyard frequency factors | Vineyard, G. H. *Journal of Physics and Chemistry of Solids* (1957) | [10.1016/0022-3697(57)90059-8](https://doi.org/10.1016/0022-3697(57)90059-8) |
| BVSE Fallback (`diff_coeff` Tier 3) | Grid-based Bond-Valence Site Energy landscape calculation for fallback diffusivity | Adams, S., et al. *Journal of Materials Chemistry* (2011) | [10.1039/C1JM10839K](https://doi.org/10.1039/C1JM10839K) |
| Framework Integrity (`mech_stab`) | Open-framework electrode materials accommodate ions with minimal mechanical strain | Fang, Y., et al. *Advanced Science* (2017) | [10.1002/advs.201600392](https://doi.org/10.1002/advs.201600392) |
| Mechanical Strain (`mech_stab`) | Pulverization, volume changes, and mechanical degradation during cycling | Zhao, K., et al. *Journal of Applied Physics* (2011) | [10.1063/1.3492617](https://doi.org/10.1063/1.3492617) |
| HT MOF Screening | Design methodology for computational screening of MOF databases | Chung, Y. G., et al. *Chemistry of Materials* (2014) | [10.1021/cm502594j](https://doi.org/10.1021/cm502594j) |
| Materials Design | First-principles battery materials screening and multi-objective ranking | Ceder, G. *MRS Bulletin* (2010) | [10.1557/mrs2010.681](https://doi.org/10.1557/mrs2010.681) |

---

## BibTeX Entries

```bibtex
@article{deng_2023_chgnet,
  title   = {CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling},
  author  = {Deng, Bowen and Zhong, Peichen and Jun, KyuJung and Riebesell, Janosh and Han, Kevin and Bartel, Christopher J. and Ceder, Gerbrand},
  journal = {Nature Machine Intelligence},
  volume  = {5},
  pages   = {1031--1041},
  year    = {2023},
  doi     = {10.1038/s42256-023-00716-3}
}

@article{adams_2011_bvse,
  title   = {High power sodium ion batteries — Bond valence site energy pathways for mobile ions},
  author  = {Adams, Stefan and Prasada Rao, Hari},
  journal = {Journal of Materials Chemistry},
  volume  = {21},
  pages   = {9804--9811},
  year    = {2011},
  doi     = {10.1039/C1JM10839K}
}
```
