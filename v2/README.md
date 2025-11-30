# Robust Non-Singular Bouncing Cosmology (Version 2)

[![Python 3.8.2+](https://img.shields.io/badge/python-3.8.2+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2511.18522v2-b31b1b.svg)](https://arxiv.org/abs/2511.18522v2)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17684433.svg)](https://doi.org/10.5281/zenodo.17684433)

**Validated implementation** of the sigmoid-regularized hyperbolic field space bouncing cosmology.

## ✅ Validation Results (Python 3.8.2)

- **12/12 tests passed** - Full validation suite
- **398 e-folds** of inflation post-bounce  
- **Trans-Planckian safe** - min wavelength = 3.47×10⁶ M_Pl⁻¹
- **Observables**: nₛ = 0.9667, r = 0.0033, Aₛ = 2.03×10⁻⁹
- **100% success rate** in basin tests (16 orders of magnitude)

## 🚀 Quick Start

```bash
# Install dependencies
pip install numpy scipy matplotlib

# Run comprehensive validation
python bounce.py

# Generate publication figures
python generate_figures.py

# Compile paper (requires LaTeX)
pdflatex main.tex
```

## 📋 Requirements

- **Python 3.8.2+** (tested on 3.8.2 and 3.13.9)
- **NumPy** (≥1.19.0)
- **SciPy** (≥1.5.0) 
- **Matplotlib** (≥3.3.0)

## 📁 Files

- `bounce.py` - Main simulation and validation
- `perturbations.py` - Perturbation analysis  
- `generate_figures.py` - Publication-quality figures
- `/figures` - Figures
- `main.tex`, `main.pdf` - Complete paper

> Compile paper (requires LaTeX - MiKTeX or TeX Live)

## 🔬 Testing

The validation suite checks:
- Background evolution through bounce
- Friedmann constraint (error < 10⁻⁶)
- Energy conditions (NEC, WEC)
- Basin of attraction (16 orders of magnitude)
- Perturbation robustness
- Observable predictions vs Planck
- Trans-Planckian safety
- Mukhanov-Sasaki regularity

## 📚 Citation

Please cite our work:

```bibtex
@article{Kravchenko2025,
    title = {Robust Non-Singular Bouncing Cosmology from Regularized Hyperbolic Field Space},
    author = {Kravchenko, Oleksandr},
    year = {2025},
    eprint = {2511.18522v2},
    archivePrefix = {arXiv},
    primaryClass = {gr-qc},
    url = {https://arxiv.org/abs/2511.18522v2}
}
```

## 🤝 Contributing

We welcome issues and discussions! Please use GitHub issues for:
- Bug reports
- Theoretical questions  
- Suggestions for extensions
- Numerical validation on different systems

## 📧 Contact

**Oleksandr Kravchenko**  
- Email: cosmology@okmath.org
- Website: https://okmath.org
- GitHub: [@OkMathOrg/bouncing-cosmology](https://github.com/OkMathOrg/bouncing-cosmology)

---

*Part of the [bouncing-cosmology](https://github.com/OkMathOrg/bouncing-cosmology) repository*