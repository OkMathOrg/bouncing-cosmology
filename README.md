# Robust Non-Singular Bouncing Cosmology

[![Python 3.8.2+](https://img.shields.io/badge/python-3.8.2+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2511.18522-b31b1b.svg)](https://arxiv.org/abs/2511.18522)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17684433.svg)](https://doi.org/10.5281/zenodo.17684433)

> ### ⚖️ License Note
>
> * **Code:** All Python source files (`*.py`) in this repository are licensed under the **MIT License**.
> * **Manuscript:** The article text (`main.tex`, `main.pdf`) and figures (`/figures/*.pdf`) are licensed under **[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)**.

**Complete numerical framework** for a ghost-free, NEC-preserving bouncing cosmology in a **closed universe** ($k=+1$) that replaces the Big Bang singularity with a geometric bounce driven by regularized hyperbolic field space. Features four progressively refined implementations with improved stability, predictive power, and through-bounce perturbation analysis.

## 🎯 Key Breakthroughs

- **✅ Solves initial singularity** without exotic matter or modified gravity
- **✅ NEC satisfied** throughout evolution — no energy condition violations
- **✅ Ghost-free, gradient-stable** — scalar and tensor sound speeds $c_\phi^2 = c_\chi^2 = c_T^2 = 1$ verified through $H=0$
- **✅ Through-bounce perturbations** (v4) — full two-field Newtonian-gauge integration over 65 e-folds, both Einstein constraints satisfied
- **✅ Independent CMB-scale verification** (v4) — $n_s = 0.9683$ from a separate Mukhanov–Sasaki run, matches the exact slow-roll benchmark to $|\Delta n_s| = 5.47\times10^{-4}$
- **✅ Non-Gaussianity** (v4) — $f_{\rm NL}^{\rm local} \approx +0.013$ via $\delta N$ formalism, agrees with Maldacena's single-field consistency relation to $|\Delta f_{\rm NL}| = 1.5\times10^{-4}$
- **✅ Nontrivial α-universality** (v4) — predictions stable across $\alpha\in[0.1,10]$ even with an excited spectator ($\chi_0 = 1\,M_{\rm Pl}$, $K_\chi/K_{\rm tot}$ up to 11.6%, $\sim25\%$ variation in $g_{\chi\chi}$)
- **✅ Planck-compatible** — $n_s \approx 0.967$, $r \approx 0.003$, $A_s \approx 2.1\times 10^{-9}$
- **✅ Solves flatness problem** — only $\sim 3.5$ e-folds needed post-bounce
- **✅ BKL-compatible** — anisotropic shear suppressed, no Kasner transitions for physically reasonable initial shear

## 📁 Repository Structure

### 🔬 [Version 4: Through-Bounce Perturbations + CMB Verification (RECOMMENDED)](./v4/)
**"Robust Non-Singular Bouncing Cosmology from Regularized Hyperbolic Field Space"** (paper v4)

* **Status**: Current, most rigorous version (paper v4)
* **Key Innovations**:
  - Two-field Newtonian-gauge perturbation integration through $H = 0$ (resolves the comoving-gauge $z''/z\sim 2/\tau^2$ singularity)
  - Numerical sound-speed probe at 203 sample times: $c_\phi^2 = c_\chi^2 = 1$ to floating-point precision through the bounce; $c_T^2 = 1$ analytically
  - Independent flat-FRW CMB-scale Mukhanov–Sasaki run at $N\in[50,70]$: $n_s = 0.9683$ vs exact slow-roll benchmark to $|\Delta n_s| = 5.47\times10^{-4}$
  - Non-Gaussianity via the $\delta N$ formalism with Maldacena single-field cross-check
  - Nontrivial α-universality test on excited-spectator background ($\chi_0 = 1\,M_{\rm Pl}$, $\sigma(n_s) = 2.4\times10^{-6}$ across 11 α)
  - Rigorous $\mathcal{R}$-conservation test ($|\Delta\mathcal{R}^2/\mathcal{R}^2| = 4.43\times10^{-3}$ between $N=1$ and $N=5$ post-exit)
  - Dynamical Bianchi-IX shear evolution (extends v3's static check)
  - Explicit epistemic classification (Theorem / Assumption / Minimal-Complexity-Choice) of every step of the sigmoid derivation
* **Validation**: 12/13 background tests + 16-mode perturbation integration + CMB verification + 30 figures
* **Quick Start**: `cd v4 && python run_all_tests.py --quick` (~60 s) or `python run_all_tests.py` (~15 min full)

### 🔬 [Version 3: Comprehensive Final Model (previous comprehensive baseline)](./v3/)
**"Robust Non-Singular Bouncing Cosmology from Regularized Hyperbolic Field Space — Final Comprehensive Edition"**

* **Status**: Previous arXiv-submitted version, retained as the comprehensive baseline
* **Key Innovations**:
  - Complete flatness solution calculation
  - BKL compatibility verification
  - α-parameter independence proof (baseline)
  - Enhanced perturbation analysis (comoving gauge)
* **Full Validation**: 10+ comprehensive tests including flatness, BKL, shear suppression
* **Requirements**: Python 3.8+, NumPy, SciPy, Matplotlib
* **Quick Start**: `cd v3 && python run_all_tests.py`

### 🚀 [Version 2: Robust Sigmoid Model](./v2/)
**"Robust Non-Singular Bouncing Cosmology from Regularized Hyperbolic Field Space"**

* **Status**: Previous, arXiv-submitted version
* **Key Innovation**: Sigmoid metric $g_{\chi\chi} = (1+e^{-2\alpha\phi})^{-1}$ derived from first principles
* **Breakthrough Results**:
  - **Basin improvement**: $10^{21}\times$ wider than exponential metric
  - **Full perturbation validation** through bounce (comoving gauge, asserted)
  - **Trans-Planckian safety** — fluctuations remain classical
  - **Publication-ready figures** and complete LaTeX paper
* **Requirements**: Python 3.8+, NumPy, SciPy, Matplotlib
* **Files**: `bounce.py`, `perturbations.py`, `generate_figures.py`, `main.tex`, `main.pdf`

### 📜 [Version 1: Exponential Model (LEGACY)](./v1/)
**"Non-Singular Bouncing Cosmology from Hyperbolic Field Space Geometry"**

* **Status**: Archived proof-of-concept
* **Limitations**: Narrow basin of attraction, perturbative unitarity issues at $\phi \to +\infty$
* **Educational Value**: Demonstrates bounce mechanism with pure exponential metric $e^{2\alpha\phi}$

## 🚀 Quick Start (Version 4)

```bash
cd v4
pip install -r requirements.txt
python run_all_tests.py --quick   # ~60 s smoke test
python run_all_tests.py           # ~15 min full validation + figures
latexmk -pdf main.tex             # compile the paper
```

## 📊 Validation Suite (Version 4)

The full v4 suite runs:
- ✅ Background evolution through bounce (FLRW + Bianchi IX)
- ✅ Friedmann constraint (max error $< 10^{-6}$)
- ✅ Energy conditions (NEC, WEC)
- ✅ Basin of attraction (22 orders of magnitude in $\dot\chi_0$)
- ✅ Flatness solution ($|\Omega_k| < 10^{-3}$ in only $\sim 3.5$ e-folds)
- ✅ BKL compatibility (dynamical shear evolution, no Kasner transitions)
- ✅ α-parameter independence — baseline ($\sigma(n_s) < 10^{-6}$) and nontrivial ($\sigma(n_s) = 2.4\times10^{-6}$)
- ✅ Two-field perturbations through $H = 0$ (Newtonian gauge, both Einstein constraints)
- ✅ Sound-speed probe ($c_\phi^2 = c_\chi^2 = 1$ at floating-point precision)
- ✅ $\mathcal{R}$ super-Hubble conservation (rigorous time-drift test)
- ✅ Independent CMB-scale Mukhanov–Sasaki run ($n_s = 0.9683$)
- ✅ Non-Gaussianity ($\delta N$ + Maldacena cross-check, $|\Delta f_{\rm NL}| = 1.5\times10^{-4}$)
- ✅ Slow-roll attractor verification ($|dN/d\phi|_{\rm num/an}$ within $10^{-4}$)

## 📚 Citation

Please cite our work:

```bibtex
@article{Kravchenko2026,
    title = {Robust Non-Singular Bouncing Cosmology from Regularized Hyperbolic Field Space},
    author = {Kravchenko, Oleksandr},
    year = {2026},
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

© 2025–2026 Oleksandr Kravchenko, [OkMath Research Initiative](https://okmath.org)

---

*This work provides the first complete, robust, and observationally viable alternative to singular Big Bang cosmology using standard general relativity, now extended in v4 with a fully gauge-invariant through-bounce perturbation analysis and an independent CMB-scale verification.*
