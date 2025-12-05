# Robust Non-Singular Bouncing Cosmology (Version 3)

[![Python 3.8.2+](https://img.shields.io/badge/python-3.8.2+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2511.18522-b31b1b.svg)](https://arxiv.org/abs/2511.18522)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17684433.svg)](https://doi.org/10.5281/zenodo.17684433)

> ### License Note
>
> * **Code:** All Python source files (`*.py`) in this repository are licensed under the **MIT License**.
> * **Manuscript:** The article text (`main.tex`, `main.pdf`) and figures (`/figures/*.pdf`) are licensed under **[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)**.

## ✅ Key Advancements Over v2.

**Complete implementation and validation** of the sigmoid-regularized hyperbolic field space bouncing cosmology. **100% success** bounce+inflation with **Planck consistency**.

This version adds comprehensive validation of critical theoretical aspects:

| Feature | v2 Status | v3 Status | Impact |
|---------|-----------|-----------|---------|
| **BKL compatibility** | Not addressed | ✅ **Validated**: shear suppressed by 10⁻²¹ | Ensures isotropic evolution |
| **NEC preservation** | Asserted | ✅ **Verified**: ρ+p ≥ 0 throughout | No exotic matter needed |
| **Ghost-free** | Implied | ✅ **Proven**: positive-definite metric | Guarantees stability |
| **Flatness solution** | Not calculated | ✅ **Quantified**: only 3.5 e-folds needed | Solves post-bounce flatness problem |
| **α-independence** | Not tested | ✅ **Demonstrated**: n_s, r constant for α∈[0.1,10] | Universal predictions |
| **Trans-Planckian safety** | Not checked | ✅ **Confirmed**: λ_min > Planck length | Perturbations remain classical |

## 🚀 Quick Start

```bash
# Clone the repository (if not already)
git clone https://github.com/OkMathOrg/bouncing-cosmology.git
cd bouncing-cosmology/v3

# Check Python version
python --version  # Should be 3.8.2+

# Verify installations
python -c "import numpy, scipy, matplotlib; print('All packages installed')"

# Run comprehensive validation (generates all results)
python run_all_tests.py

# Generate publication-quality figures (PDF)
python generate_figures.py

# Run individual components
python bounce.py              # Background evolution and core validation
python perturbations.py       # Perturbation analysis through bounce

# Compile paper (requires LaTeX - TeX Live 2023+ or MiKTeX)
pdflatex main.tex
bibtex main     # if you add citations
pdflatex main.tex  # run twice for references
```

## 🔬 Model Overview

**Non-singular bounce in closed universe with sigmoid-regularized hyperbolic geometry**

### Theoretical Foundation
- **Field space metric**: g_χχ(φ) = 1 / (1 + exp(-2αφ/M_Pl))
- **Derivation**: Unique minimal-complexity solution satisfying:
  1. **Exponential suppression** during contraction: g → 0 as φ → -∞
  2. **Canonical normalization** during inflation: g → 1 as φ → +∞
  3. **Positive-definiteness** (ghost-free): g > 0 ∀φ
- **Potential**: Starobinsky-type V(φ) = V₀(1 - exp(-βφ/M_Pl))², β = √(2/3)

### Key Physics
- **Bounce mechanism**: Spatial curvature (k=+1) halts contraction → expansion
- **NEC preservation**: ρ + p = φ̇² + g_χχχ̇² ≥ 0 always
- **Sub-Planckian**: a_min ≈ 1.73×10⁵ M_Pl⁻¹, ρ_max ≈ 10⁻¹⁰ M_Pl⁴

## 🎯 Key Results (All Validated)

### 1. Robust Bounce & Inflation
- ✅ **Non-singular bounce**: Finite a_min, smooth H: - → +
- ✅ **60+ e-folds inflation**: N_post > 60 e-folds
- ✅ **100% success rate**: Across 16 orders of magnitude in initial χ̇
- ✅ **Basin improvement**: ∼10²¹× wider than exponential metric (v1)

### 2. Theoretical Consistency
- ✅ **NEC/WEC preserved**: No exotic matter needed
- ✅ **BKL-compatible**: Anisotropic shear suppressed by ∼10⁻²¹
- ✅ **Flatness solved**: Only ∼3.5 e-folds needed for |Ω_k| < 0.001
- ✅ **Trans-Planckian safe**: Minimum physical wavelength > Planck length

### 3. Observational Predictions (N=60)
- ✅ **Spectral index**: n_s = 0.967 (within 0.5σ of Planck 2018)
- ✅ **Tensor-to-scalar**: r = 0.0033 (testable by CMB-S4/LiteBIRD)
- ✅ **Amplitude**: A_s = 2.1×10⁻⁹ (matches Planck)
- ✅ **Universal**: Predictions independent of α (0.1 ≤ α ≤ 10)

## 📁 Files

| File | Purpose |
|------|---------|
| `bounce.py` | **Main simulation** (v5 - Final Comprehensive Edition)<br>Background evolution, validation suite, analytical calculations |
| `perturbations.py` | **Perturbation analysis**<br>Regularity through bounce, curvature conservation, Trans-Planckian check |
| `generate_figures.py` | **Publication figures**<br>Generates all 9 PDF figures used in paper |
| `run_all_tests.py` | **Complete validation**<br>8-category test suite, saves results to `output/` |
| `main.tex` | **LaTeX source** for arXiv submission |
| `main.pdf` | **Compiled paper** (20 pages, complete with all figures) |

## ▶️ How to Run

### 1. Basic Validation

```bash
python bounce.py
```

This runs the core background evolution and displays key results.

### 2. Complete Validation Suite

```bash
python run_all_tests.py
```

Runs all 8 test categories:

1. Bounce test - Finite a_min, smooth H=0 crossing
2. Friedmann constraint - Max error < 10⁻¹²
3. Energy conditions - NEC & WEC preserved
4. Flatness calculation - N_actual > N_required (60 > 3.5)
5. BKL compatibility - Shear/curvature ratio ∼ 10⁻²¹
6. α-independence - n_s, r constant across α ∈ [0.1, 10]
7. Perturbation regularity - Equations finite, ℛ conserved
8. Observational consistency - Planck 2018: n_s=0.967, r<0.036

Outputs: `output/final_results.json`, `output/results_summary.txt`

### 3. Generate Publication Figures

```bash
python generate_figures.py
```

Generates all 9 PDF figures used in the paper.

### 4. Perturbation Analysis

```bash
python perturbations.py
```

Analyzes perturbation behavior through bounce.

### 5. Verify Results

```bash
# Check that all tests passed
cat output/results_summary.txt

# Verify key results
python -c "import json; data=json.load(open('output/final_results.json')); print(f'n_s: {data[\"n_s\"]:.4f}, r: {data[\"r\"]:.4f}, A_s: {data[\"A_s\"]:.4e}')"

```

## 📐 Generated Figures

| Figure | Description | File |
|--------|-------------|------|
| **Fig 1** | Field space geometry & regularization | [field_space_geometry.pdf](figures/field_space_geometry.pdf) |
| **Fig 2** | Potential & inflationary dynamics | [potential_and_dynamics.pdf](figures/potential_and_dynamics.pdf) |
| **Fig 3** | Full background evolution | [background_evolution.pdf](figures/background_evolution.pdf) |
| **Fig 4** | Detailed bounce region | [bounce_zoom.pdf](figures/bounce_zoom.pdf) |
| **Fig 5** | Basin of attraction analysis | [basin_of_attraction.pdf](figures/basin_of_attraction.pdf) |
| **Fig 6** | Perturbation analysis | [perturbation_analysis.pdf](figures/perturbation_analysis.pdf) |
| **Fig 7** | α-independence of predictions | [alpha_independence.pdf](figures/alpha_independence.pdf) |
| **Fig 8** | Flatness evolution after bounce | [flatness_evolution.pdf](figures/flatness_evolution.pdf) |
| **Fig 9** | BKL compatibility analysis | [bkl_analysis.pdf](figures/bkl_analysis.pdf) |

## 👍 New in Version 3 (vs v2)

### Theoretical Extensions
- **BKL compatibility proof**: Shear suppressed by 10⁻²¹, curvature dominates
- **Flatness solution**: Quantitative calculation of Ω_k(N) after bounce
- **α-independence proof**: Observable predictions independent of regularization parameter
- **Enhanced derivation**: Sigmoid as unique minimal-complexity solution

### Numerical Additions
- **Complete validation suite**: 8 test categories, automated reporting
- **Bounce analysis**: Detailed energy components through bounce
- **Perturbation checks**: Trans-Planckian safety, equation regularity
- **Professional figures**: 9 publication-quality PDFs

### Code Improvements
- **Robust numerics**: Safe sigmoid computation for all φ
- **Modular structure**: Separate modules for background/perturbations/figures
- **Full reproducibility**: One-command regeneration of all results
- **Comprehensive docs**: Detailed docstrings and comments

## 📚 Citation

If you use this code or results, please cite:

```bibtex
@article{Kravchenko2025v3,
  title = {Robust Non-Singular Bouncing Cosmology from Regularized Hyperbolic Field Space},
  author = {Kravchenko, Oleksandr},
  journal = {arXiv preprint arXiv:2511.18522v3},
  year = {2025},
  eprint = {2511.18522v3},
  archivePrefix = {arXiv},
  primaryClass = {gr-qc}
}
```

**BibTeX for specific versions:**
- v3 (this version): arXiv:2511.18522v3
- v2 (sigmoid metric): arXiv:2511.18522v2
- v1 (initial exponential): arXiv:2511.18522v1

## ⁉️ Issue Reporting

Found a bug or have questions?
1. Check [GitHub Issues](https://github.com/OkMathOrg/bouncing-cosmology/issues)
2. Email: cosmology@okmath.org
3. **Note**: v1 and v2 are archived; v3 is actively maintained

## 🔗 Links

- **Full repository**: https://github.com/OkMathOrg/bouncing-cosmology
- **arXiv v3**: https://arxiv.org/abs/2511.18522v3
- **arXiv v2**: https://arxiv.org/abs/2511.18522v2
- **arXiv v1**: https://arxiv.org/abs/2511.18522v1

## 🤝 Contributing

We welcome issues and discussions! Please use GitHub issues for:
- Bug reports
- Theoretical questions  
- Suggestions for extensions
- Numerical validation on different systems

## 📧 Contact

- Email: cosmology@okmath.org
- Website: https://okmath.org
- GitHub: [@OkMathOrg/bouncing-cosmology](https://github.com/OkMathOrg/bouncing-cosmology)

© 2025 Oleksandr Kravchenko, [OkMath Research Initiative](https://okmath.org)