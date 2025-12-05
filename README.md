# Robust Non-Singular Bouncing Cosmology

[![Python 3.8.2+](https://img.shields.io/badge/python-3.8.2+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2511.18522-b31b1b.svg)](https://arxiv.org/abs/2511.18522)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17684433.svg)](https://doi.org/10.5281/zenodo.17684433)

> ### ⚖️ License Note
>
> * **Code:** All Python source files (`*.py`) in this repository are licensed under the **MIT License**.
> * **Manuscript:** The article text (`main.tex`, `main.pdf`) and figures (`/figures/*.pdf`) are licensed under **[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)**.

**Complete numerical framework** for a ghost-free, NEC-preserving bouncing cosmology in a **closed universe** ($k=+1$) that replaces the Big Bang singularity with a geometric bounce driven by regularized hyperbolic field space. Features three progressively refined implementations with improved stability and predictive power.

## 🎯 Key Breakthroughs

- **✅ Solves initial singularity** without exotic matter or modified gravity
- **✅ 100% success rate** across 16 orders of magnitude in initial conditions  
- **✅ NEC satisfied** throughout evolution - no energy condition violations
- **✅ Perturbation-safe** - Mukhanov-Sasaki equations remain regular through bounce
- **✅ Planck-compatible** - $n_s \approx 0.967$, $r \approx 0.003$, $A_s \approx 2.1\times 10^{-9}$
- **✅ Universal predictions** - observables independent of regularization parameter α
- **✅ Solves flatness problem** - only ~3.5 e-folds needed post-bounce
- **✅ BKL-compatible** - suppresses anisotropic shear, maintains isotropy

## 📁 Repository Structure

### 🔬 [Version 3: Comprehensive Final Model (RECOMMENDED)](./v3/)
**"Robust Non-Singular Bouncing Cosmology from Regularized Hyperbolic Field Space - Final Comprehensive Edition"**

* **Status**: Current, most robust version (arXiv-submitted)
* **Key Innovations**: 
  - Complete flatness solution calculation
  - BKL compatibility verification  
  - α-parameter independence proof
  - Enhanced perturbation analysis
* **Full Validation**: 10+ comprehensive tests including flatness, BKL, shear suppression
* **Requirements**: Python 3.8+, NumPy, SciPy, Matplotlib
* **Quick Start**: `cd v3 && python run_all_tests.py`

### 🚀 [Version 2: Robust Sigmoid Model](./v2/)
**"Robust Non-Singular Bouncing Cosmology from Regularized Hyperbolic Field Space"**

* **Status**: Previous, arXiv-submitted version
* **Key Innovation**: Sigmoid metric $g_{\chi\chi} = (1+e^{-2\alpha\phi})^{-1}$ derived from first principles
* **Breakthrough Results**:
  - **Basin improvement**: $10^{21}\times$ wider than exponential metric
  - **Full perturbation validation** through bounce
  - **Trans-Planckian safety** - fluctuations remain classical
  - **Publication-ready figures** and complete LaTeX paper
* **Requirements**: Python 3.8+, NumPy, SciPy, Matplotlib
* **Files**: `bounce.py`, `perturbations.py`, `generate_figures.py`, `main.tex`, `main.pdf`

### 📜 [Version 1: Exponential Model (LEGACY)](./v1/)
**"Non-Singular Bouncing Cosmology from Hyperbolic Field Space Geometry"**

* **Status**: Archived proof-of-concept
* **Limitations**: Narrow basin of attraction, perturbative unitarity issues at $\phi \to +\infty$
* **Educational Value**: Demonstrates bounce mechanism with pure exponential metric $e^{2\alpha\phi}$

## 🚀 Quick Start (Version 3)

```bash
cd v3
python run_all_tests.py             # Complete validation & figures
# OR
python bounce.py                    # Run core simulation only
python generate_figures.py          # Generate figures only
pdflatex main.tex                   # Compile paper
```

## 📊 Validation Suite

The code automatically runs 12+ validation checks:
- ✅ Background evolution through bounce
- ✅ Friedmann constraint (error < $10^{-6}$)
- ✅ Energy conditions (NEC, WEC)
- ✅ Basin of attraction (16 orders of magnitude)
- ✅ Perturbation robustness (20% noise tests)
- ✅ Observable predictions vs Planck data
- ✅ Trans-Planckian safety
- ✅ Mukhanov-Sasaki regularity
- ✅ Flatness solution (Ω_k < 0.001)
- ✅ BKL compatibility (shear suppression)
- ✅ α-parameter independence
- ✅ Single-field dominance

## 📚 Citation

Please cite our work:

```bibtex
@article{Kravchenko2025,
    title = {Robust Non-Singular Bouncing Cosmology from Regularized Hyperbolic Field Space},
    author = {Kravchenko, Oleksandr},
    year = {2025},
    eprint = {2511.18522},
    archivePrefix = {arXiv},
    primaryClass = {gr-qc},
    url = {https://arxiv.org/abs/2511.18522}
}
```

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

---

*This work provides the first complete, robust, and observationally viable alternative to singular Big Bang cosmology using standard general relativity.*
