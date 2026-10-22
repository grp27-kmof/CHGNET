# CHGNet Screening Pipeline for MOF Anodes in Potassium-Ion Batteries

This repository screens metal–organic frameworks (MOFs) as anode materials for potassium-ion batteries (KIBs) using the [CHGNet](https://github.com/CederGroupHub/chgnet) machine-learning potential.

It takes MOF CIF files, experimental or generated, and estimates:

- how strongly a K atom binds to the framework
- how many K atoms the framework can store, giving the capacity and voltage profile
- whether the framework stays mechanically and thermally stable when loaded with K

It is part of the final-year project **"Computational design and generation of high-performance MOF anodes for potassium-ion batteries"** (Group 27).

---

## Workflow

```
CIF ──► 1. Relaxation ──► 2. Single-K adsorption ──► 3. Stepwise K insertion
                                                         │
          6. Analysis ◄── 5. MD at 300 K ◄── 4. Mechanical stability
```

| Step | What it does | Key output |
|------|--------------|------------|
| 1. Relaxation | Relaxes atoms and cell (FIRE + FrechetCellFilter). Accepted only if f_max ≤ 0.05 eV/Å | Relaxed bare MOF |
| 2. Single-K adsorption | Finds the best K site: quick prescreen with the framework fixed, then a full unconstrained relaxation | E_ads (eV) |
| 3. Stepwise insertion | Adds K one at a time and stops when insertion is no longer favourable | Max K loading, capacity (mAh/g), voltage profile, volume change |
| 4. Mechanical stability | Elastic tensor from ±1 % strain, Born stability test, Voigt–Reuss–Hill moduli | Bulk and shear modulus, stable / unstable |
| 5. Molecular dynamics | 300 K, 10 ps, Berendsen thermostat on the K-loaded MOF | RMSD, metal–node bond integrity, K MSD |
| 6. Analysis | Collects all results into summary tables | CSV summary |

**Formulas used**

- Step voltage: `V_n = −[E(n) − E(n−1) − E_K(bulk)]`
- Theoretical capacity: `C = 26801 · n / M` (mAh/g), where *n* is the number of K atoms and *M* the molar mass of the MOF in g/mol
- A metal–node bond counts as broken if it stays longer than 2.6 Å for more than 50 % of the second half of the MD run

---

## Installation

```bash
git clone https://github.com/grp27-kmof/CHGNET.git
cd CHGNET

conda create -n chgnet python=3.10 -y
conda activate chgnet
pip install chgnet pymatgen ase numpy pandas matplotlib pytest
```

A CUDA GPU is recommended for large batches; CPU and Apple Silicon also work.

---

## Usage

Put the input CIF files in one folder, then run from the repository root.

**Full pipeline (relaxation → adsorption → insertion)**
```bash
python run_pipeline.py
```

**Run only one stage**
```bash
python run_pipeline.py --relax-only
python run_pipeline.py --adsorb-only
python run_pipeline.py --insert-only --n-max 20
```

| Flag | Meaning |
|------|---------|
| `--relax-only` / `--adsorb-only` / `--insert-only` | Run a single stage |
| `--n-max N` | Maximum number of K atoms to insert |
| `--no-resume` | Ignore cached results and recompute everything |

**Mechanical stability**
```bash
python mech_stability_cif.py              # relax, then compute the elastic tensor
python mech_stability_cif.py --skip-relax # use already-relaxed structures
```

**Molecular dynamics**
```bash
python run_md.py
```

**Analysis**
```bash
python analyse.py
```

Run `python <script>.py --help` to see all options and the input and output paths.

---

## Tests

```bash
pytest tests/
```

The tests cover relaxation, single-K adsorption, stepwise insertion and the analysis fixes.

---

## Notes and limitations

- **CHGNet overbinds K.** Absolute voltages and adsorption energies are too strong (for example, about −2.7 eV from CHGNet versus about −1.8 eV from DFT for the same MOF). Use them to **compare** frameworks, not as final values.
- Results are cached and checked by a fingerprint of the structure and settings. Use `--no-resume` after changing the input structures or parameters.
- Final candidates should be confirmed with DFT (CP2K) and MACE diffusion-barrier calculations, which are in separate parts of the project.

---

## Project context

| Phase | Description |
|-------|-------------|
| Phase 1 | Screening of known MOFs: ALIGNN formation energy → theoretical capacity → **CHGNet (this repo)** → MACE diffusion → DFT |
| Phase 2 | Generation of new MOFs with deep generative models, followed by the same screening |

So far, one generated MOF (`fsc+N564+N85+E0+E163`, Zn-based) has passed every stage: 267.5 mAh/g, about 1 % volume change, a 0.149 eV diffusion barrier, and Born-stable and MD-stable when fully loaded with K.

---

## Acknowledgements

- CHGNet: Deng *et al.*, *Nature Machine Intelligence* 5, 1031 (2023)
- pymatgen and ASE for structure handling and simulations