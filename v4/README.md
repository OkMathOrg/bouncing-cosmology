# Robust Non-Singular Bouncing Cosmology (Version 4)

[![Python 3.8.2+](https://img.shields.io/badge/python-3.8.2+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2511.18522-b31b1b.svg)](https://arxiv.org/abs/2511.18522)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17684433.svg)](https://doi.org/10.5281/zenodo.17684433)

> ### License Note
>
> * **Code:** All Python source files (`*.py`) in this repository are licensed under the **MIT License**.
> * **Manuscript:** The article text (`main.tex`, `main.pdf`) and figures (`/figures/*.pdf`) are licensed under **[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)**.

## ✅ Key Advancements Over v3

**Through-bounce perturbation analysis, independent CMB-scale verification, and non-Gaussianity** added on top of the v3 baseline. The bounce mechanism, BKL stability, NEC preservation, and Planck-compatible observables of v3 all carry over; v4 strengthens the perturbative and observational story.

| Feature | v3 Status | v4 Status | Impact |
|---------|-----------|-----------|---------|
| **Two-field perturbations through H=0** | Comoving-gauge regularity asserted | ✅ **Newtonian-gauge integration** of the full $(\delta\phi,\delta\chi,\Phi)$ system over 65 e-folds, both Einstein constraints verified *a posteriori* | Resolves the $z''/z\sim 2/\tau^2$ comoving-gauge singularity at the bounce |
| **Sound speeds at H=0** | Not measured | ✅ **Numerical probe** of the coded $k^2/a^2$ coefficient at 203 sample times (including $H{=}0$): $c_\phi^2=c_\chi^2=1$ to floating-point precision; $c_T^2=1$ analytically | Strict hyperbolicity through the bounce; no ghost / no gradient instability |
| **CMB-scale verification** | Slow-roll formula only | ✅ **Independent flat-FRW Mukhanov–Sasaki** integration at $N\in[50,70]$ via $u=a\,\delta\phi$, reproducing $n_s=0.9683$ to $\|\Delta n_s\|=5.47\times10^{-4}$ vs the exact slow-roll benchmark | Genuine numerical recovery of CMB observables, not just analytical extrapolation |
| **Non-Gaussianity** | Not computed | ✅ **$f_{\rm NL}^{\rm local}=+0.013$** via $\delta N$ formalism, agreeing with Maldacena's single-field consistency $\tfrac{5}{12}(1-n_s)$ to $\|\Delta f_{\rm NL}\|=1.54\times10^{-4}$ | Bounce-phase contribution at CMB scales suppressed to $10^{-2232}$ via $(k_H/k_{\rm CMB})^2$ matching |
| **Slow-roll attractor verification** | Asserted | ✅ **Numerically verified**: $\|dN/d\phi\|_{\rm num} / \|dN/d\phi\|_{\rm an}$ deviates from unity by median $1.4\times10^{-4}$ in the post-bounce $\phi$-window | Justifies use of $\delta N$ formalism on the simulated trajectory |
| **α-universality** | Verified on baseline ($\chi=\dot\chi=0$) | ✅ **Nontrivial test added**: spectator-displacement scan with $\chi_0=1\,M_{\rm Pl}$ ($K_\chi/K_{\rm tot}$ up to $11.6\%$, $g_{\chi\chi}(\phi_{\rm CMB})$ varies $\sim25\%$); universality survives with $\sigma(n_s)=2.4\times10^{-6}$ across 11 values of $\alpha$ | Universality is not a kinematic tautology of the trivial trajectory |
| **R conservation** | Implicit (constancy after exit) | ✅ **Rigorous time-drift test**: $\|\Delta\mathcal{R}^2/\mathcal{R}^2\|_{N=1\to5} = 4.43\times10^{-3}$ for $k=0.5\,k_H$ | Sub-percent super-Hubble conservation across explicit e-fold markers |
| **Bianchi-IX shear evolution** | Static comparison only | ✅ **Dynamical**: shear $\Sigma^2/a^6$ tracked through contraction across 31 orders of magnitude in initial $\Sigma_0^2$, no Kasner transitions for $\Sigma^2 \lesssim a_{\rm min}^4$ | Wall-free truncation explicitly self-consistent |
| **Theoretical classification** | Implicit | ✅ **Explicit Theorem / Assumption / Minimal-Complexity-Choice tagging** for each step of the sigmoid derivation (Table 1 in paper) | Disentangles physical input from mathematical consequence |

## 🚀 Quick Start

```bash
# From the repo root
cd v4

# Check Python version
python --version  # Should be 3.8.2+

# Install dependencies
pip install -r requirements.txt

# Quick smoke test (~60-70 s; reduced mode count, skips alpha-scan, BKL, full convergence and figure generation)
python run_all_tests.py --quick

# Full validation suite (~14-16 min; 13 background tests, 16-mode perturbation
# integration, CMB-scale verification, regenerates results_macros.tex and 30 figures)
python run_all_tests.py

# Compile the paper (requires TeX Live 2023+ or MiKTeX)
latexmk -pdf main.tex
```

## 🔬 Model Overview

**Non-singular bounce in a closed universe with sigmoid-regularized hyperbolic geometry.**

### Theoretical Foundation
- **Field space metric**: $g_{\chi\chi}(\phi) = (1 + e^{-2\alpha\phi/M_{\rm Pl}})^{-1}$
- **Derivation**: unique minimal-complexity solution satisfying three physical boundary conditions
  1. **Exponential suppression** during contraction: $g \to 0$ as $\phi \to -\infty$
  2. **Canonical normalization** during inflation: $g \to 1$ as $\phi \to +\infty$
  3. **Positive-definiteness** (ghost-free): $g(\phi) > 0$ for all $\phi$
- **Potential**: Starobinsky-type $V(\phi) = V_0(1 - e^{-\beta\phi/M_{\rm Pl}})^2$, $\beta = \sqrt{2/3}$, $V_0 = 10^{-10}\,M_{\rm Pl}^4$.

### Key Physics
- **Bounce mechanism**: spatial curvature ($k=+1$) halts contraction → expansion at $a_{\rm min} \approx 1.73\times10^5\,M_{\rm Pl}^{-1}$
- **NEC preservation**: $\rho + p = \dot\phi^2 + g_{\chi\chi}\dot\chi^2 \geq 0$ identically
- **Sub-Planckian densities**: $\rho_{\rm bounce} \sim V_0 \sim 10^{-10}\,M_{\rm Pl}^4$
- **Through-bounce perturbations**: integrated in Newtonian gauge where the comoving-gauge singularity at $H=0$ is absent

## 🎯 Key Results (All Numerically Validated)

### 1. Through-bounce perturbation regularity
- ✅ **Newtonian-gauge integration**: 16 modes spanning $k=0.3$–$200\,k_H$ over 65 post-bounce e-folds, no $1/H$ singularities
- ✅ **Both Einstein constraints**: momentum constraint to $<1.2\%$ median, Hamiltonian constraint to $\sim10^{-4}\%$ for $k\leq5\,k_H$
- ✅ **Resolution convergence**: $\Delta P_\mathcal{R}/P_\mathcal{R} < 10^{-6}$ on tightening tolerances by two orders
- ✅ **Window sensitivity**: $P_\mathcal{R}$ stable to $<0.03\%$ across extraction windows

### 2. Hyperbolicity at H=0
- ✅ **Scalar sound speeds**: $c_\phi^2,\,c_\chi^2 = 1$ to $\|c^2-1\|\leq 8\times10^{-16}$ (floating-point roundoff) at 203 sample times including points adjacent to $H=0$
- ✅ **Tensor sound speed**: $c_T^2 = 1$ inherited analytically from the Einstein–Hilbert action
- ✅ **No ghost / gradient instability** at the bounce
- ✅ **Curvature conservation**: $\|\Delta\mathcal{R}^2/\mathcal{R}^2\| = 4.43\times10^{-3}$ between $N=1$ and $N=5$ post-exit

### 3. CMB-scale observables (independent verification)
- ✅ **Spectral index**: $n_s = 0.9683$ from independent flat-FRW Mukhanov–Sasaki run at $N\in[50,70]$
- ✅ **Exact-fit benchmark**: $n_s^{\rm exact\,fit} = 0.9678$, $\|\Delta n_s\| = 5.47\times10^{-4}$ (integrator noise floor)
- ✅ **Tensor-to-scalar**: $r = 0.0033$ at $N=60$ (testable by LiteBIRD / CMB-S4 / PICO)
- ✅ **Amplitude calibration**: $A_s = 2.23\times10^{-9}$ at $V_0 = 10^{-10}\,M_{\rm Pl}^4$ (within $\sim6\%$ of the Planck central value; exact match at $V_0 = 0.94\times10^{-10}$)

### 4. Non-Gaussianity (CMB scales)
- ✅ **$\delta N$ formalism**: $f_{\rm NL}^{\rm local}(\phi_{\rm CMB}) = +0.0133$
- ✅ **Maldacena cross-check**: $\tfrac{5}{12}(1-n_s^{\rm exact}) = +0.0134$, $\|\Delta f_{\rm NL}\| = 1.54\times10^{-4}$
- ✅ **Bounce-phase dynamical contribution at CMB**: $\sim 10^{-2232}$, suppressed by Deruelle–Mukhanov $(k_H/k_{\rm CMB})^2$ matching
- ✅ **Slow-roll attractor verified**: $\|dN/d\phi\|_{\rm num/an}$ within $1.4\times10^{-4}$ of unity in post-bounce $\phi\in[9.84,9.96]\,M_{\rm Pl}$

### 5. α-universality (two distinct backgrounds)
- ✅ **Baseline scan** ($\chi=\dot\chi=0$, 11 values of $\alpha\in[0.1,10]$): $\sigma(n_s)<10^{-6}$, kinematic decoupling exact
- ✅ **Nontrivial scan** (excited spectator $\chi_0=1\,M_{\rm Pl}$, $\dot\chi_0=0$): $\sigma(n_s)=2.4\times10^{-6}$, peak-to-peak spread $8.25\times10^{-6}$, with $K_\chi/K_{\rm tot}$ up to $11.6\%$ at $\phi_{\rm CMB}$ and $\sim25\%$ variation in $g_{\chi\chi}(\phi_{\rm CMB})$
- ✅ Universality is **not** a tautology of the trivial trajectory

### 6. Theoretical robustness (carries over from v3)
- ✅ **NEC and WEC** preserved through the bounce
- ✅ **BKL-stable**: $\Sigma^2/a^4 < 1$ for all physically reasonable initial shear ($\Sigma_0^2 \lesssim 10^{18} \ll a_{\rm min}^4 \sim 10^{20}$)
- ✅ **Flatness solved**: $\sim 3.5$ post-bounce e-folds suffice for $\|\Omega_k\|<10^{-3}$
- ✅ **Robust bounce**: 22 orders of magnitude in $\dot\chi_0$ tested ($86\%$ overall success rate; $100\%$ in the small-$\dot\chi_0$ regime)

## 📁 Files

| File | Purpose |
|------|---------|
| `bounce.py` | **Background simulation** — FLRW + Bianchi-IX evolution, 13-test validation suite, baseline + nontrivial $\alpha$-scans, $\delta N$ non-Gaussianity |
| `perturbations.py` | **Two-field perturbations** — Newtonian-gauge integration through the bounce, sound-speed probe, R-conservation test, CMB-scale Mukhanov–Sasaki verification |
| `generate_figures.py` | **Publication figures** — 30 PDF panels grouped into 9 figures used in the paper |
| `run_all_tests.py` | **Master runner** — orchestrates background + perturbations + CMB verification + macro generation + figures |
| `main.tex` | **LaTeX source** (38 pages) for the paper |
| `main.pdf` | **Compiled paper** including all 9 figures |
| `results_macros.tex` | **Auto-generated** numerical macros consumed by `main.tex` (regenerated by full `run_all_tests.py`) |
| `requirements.txt` | Python dependencies (NumPy ≥ 1.20, SciPy ≥ 1.7, Matplotlib ≥ 3.4) |

## ▶️ How to Run

### 1. Quick smoke test

```bash
python run_all_tests.py --quick
```

~60–70 seconds. Runs a reduced suite (3 perturbation modes, no BKL, no convergence, no $\alpha$-scan, no figure generation). Useful for checking that the codebase still imports and integrates cleanly. **Note**: the `n_s` value reported in `--quick` is from a reduced 3-mode fit and is not physically meaningful; use the full suite for production numbers.

### 2. Full validation suite

```bash
python run_all_tests.py
```

~14–16 minutes. Runs:

1. **Bounce + Friedmann + NEC/WEC** sanity (background)
2. **Flatness** quantitative ($\Omega_k$ vs $N$)
3. **Baseline $\alpha$-scan** (11 values, $\chi=\dot\chi=0$)
4. **Nontrivial $\alpha$-scan** (11 values, $\chi_0=1\,M_{\rm Pl}$)
5. **BKL / Bianchi-IX** dynamical (31 orders of magnitude in initial $\Sigma^2$)
6. **Observables** ($n_s$, $r$, $A_s$ at $N=60$)
7. **Isocurvature mass-matrix** analysis at the bounce
8. **Non-Gaussianity** ($\delta N$ + Maldacena cross-check + slow-roll attractor)
9. **Two-field perturbations** through the bounce (16 modes, both vacuum modes, momentum + Hamiltonian constraints, sound-speed probe, R-conservation)
10. **CMB-scale verification** (independent flat-FRW Mukhanov–Sasaki, 11 modes at $N\in[50,70]$)
11. **Macro generation** (`results_macros.tex`)
12. **Figure regeneration** (30 PDF panels in `figures/`)

### 3. Compile the paper

```bash
latexmk -pdf main.tex
```

`main.tex` reads `results_macros.tex` for numerical values; the existing fallback `\providecommand` block in `main.tex` ensures the paper compiles even without a fresh full run.

## 📐 Generated Figures (30 PDF panels)

Grouped into 9 logical figures in the paper:

| Figure | Panels | Purpose |
|--------|--------|---------|
| Field-space geometry | `fsg_a_metric`, `fsg_b_curvature`, `fsg_c_decoupling` | Sigmoid metric, curvature transition, decoupling factor |
| Potential and dynamics | `pd_a_potential`, `pd_b_epsilon`, `pd_c_efolds`, `pd_d_ns_r` | Starobinsky potential, slow-roll, $N(\phi)$, $n_s$–$r$ plane |
| Background evolution | `bg_a_scale_factor`, `bg_b_hubble`, `bg_c_inflaton`, `bg_d_metric` | Full background $a(t)$, $H(t)$, $\phi(t)$, $g_{\chi\chi}(t)$ |
| Bounce zoom | `bz_a_scale_factor`, `bz_b_hubble`, `bz_c_energies` | Detail near $a_{\rm min}$, $H=0$ crossing, energy components |
| Basin of attraction | `ba_a_success_map`, `ba_b_efolds` | Success vs $\dot\chi_0$, post-bounce e-folds |
| Flatness | `fl_a_dilution`, `fl_b_compare` | $\|\Omega_k\|(N)$, required vs achieved e-folds |
| BKL analysis | `bkl_a_shear_vs_curv`, `bkl_b_kasner_ratio`, `bkl_c_at_bounce` | Shear vs curvature, Kasner ratio, bounce-point $R_K$ vs $\Sigma_0^2$ |
| Two-field perturbations | `pert_a_amplitudes`, `pert_b_curvature`, `pert_c_constraint`, `pert_d_power_spectrum`, `pert_e_residuals`, `pert_f_isocurvature` | Through-bounce field perturbations, $\mathcal{R}_k$, momentum constraint, $P_\mathcal{R}(k)$ with bounce bump, fit residuals, isocurvature transfer |
| α-independence | `ai_a_ns`, `ai_b_r`, `ai_c_g_saturation` | $n_s(\alpha)$, $r(\alpha)$, $g_{\chi\chi}(\phi_{\rm CMB})$ saturation |

## 👍 New in Version 4 (vs v3)

### Theoretical extensions
- **Explicit epistemic classification**: each step of the sigmoid derivation tagged Theorem / Assumption / Minimal-Complexity-Choice (Table 1)
- **Bianchi-IX dynamical analysis** beyond v3's static comparison — full shear evolution through contraction, Kasner-ratio scan
- **Hyperbolicity proof through $H=0$**: the scalar quadratic action has identical kinetic and gradient $G_{IJ}$ contraction, so $c_s^2 = 1$ identically; numerically verified

### Numerical additions
- **Two-field Newtonian-gauge solver** through the bounce (the v3 perturbation analysis was comoving-gauge and did not pass through $H=0$)
- **Independent CMB-scale Mukhanov–Sasaki verification** in flat FRW, decoupled from the bounce-region run
- **Sound-speed probe**: numerically extracts the coded $k^2/a^2$ coefficient from `_rhs` (not from analytical assumption)
- **Rigorous $\mathcal{R}$-conservation test** between explicit e-fold markers (genuine time drift, not std-around-mean)
- **$\delta N$ non-Gaussianity** with attractor verification and Maldacena cross-check
- **Nontrivial $\alpha$-universality test** (excited spectator) on top of the baseline scan

### Code improvements
- **Unified macro pipeline**: `run_all_tests.py` writes `results_macros.tex` with current numerics; the LaTeX source has matching `\providecommand` fallbacks so the paper builds without a fresh test run
- **Display-equation typesetting**: long inline expressions (e.g. exact-fit benchmark) moved to displayed equations to eliminate overfull-hbox warnings
- **Modular figures**: 30 separate PDF panels (instead of compound figures) for journal flexibility

## 📚 Citation

If you use this code or results, please cite:

```bibtex
@article{Kravchenko2025v4,
  title  = {Robust Non-Singular Bouncing Cosmology from Regularized Hyperbolic Field Space},
  author = {Kravchenko, Oleksandr},
  journal = {arXiv preprint arXiv:2511.18522},
  year   = {2026},
  eprint = {2511.18522},
  archivePrefix = {arXiv},
  primaryClass = {gr-qc}
}
```

**BibTeX for specific versions:**
- v4 (this version): arXiv:2511.18522 (latest)
- v3 (comprehensive baseline): arXiv:2511.18522v3
- v2 (sigmoid metric): arXiv:2511.18522v2
- v1 (initial exponential): arXiv:2511.18522v1

## ⁉️ Issue Reporting

Found a bug or have questions?
1. Check [GitHub Issues](https://github.com/OkMathOrg/bouncing-cosmology/issues)
2. Email: cosmology@okmath.org
3. **Note**: v1, v2, v3 are archived; v4 is actively maintained

## 🔗 Links

- **Repository**: https://github.com/OkMathOrg/bouncing-cosmology
- **arXiv (latest)**: https://arxiv.org/abs/2511.18522
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

© 2026 Oleksandr Kravchenko, [OkMath Research Initiative](https://okmath.org)
