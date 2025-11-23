# Non-Singular Bouncing Cosmology from Hyperbolic Field Space Geometry

**A ghost-free, NEC-satisfying cosmological model that replaces the Big Bang singularity with a geometric bounce.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17684434.svg)](https://doi.org/10.5281/zenodo.17684434)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

This repository contains the code and LaTeX source for the paper:

**"Non-Singular Bouncing Cosmology from Hyperbolic Field Space Geometry"**  
by Oleksandr Kravchenko

## Abstract

We investigate a two-field cosmological model in a closed ($k=+1$) universe where the field space is endowed with a hyperbolic geometry. We demonstrate that the curvature of the field space introduces a kinetic coupling that exponentially suppresses the scalar field kinetic energy, allowing the spatial curvature to dominate and trigger a non-singular bounce. Crucially, the model satisfies the Null Energy Condition (NEC) throughout, with the bounce driven entirely by the positive spatial curvature—not by exotic physics.

## Key Results

- **Ghost-free**: The model has positive-definite kinetic matrix for all finite field values
- **NEC-satisfying**: $\rho + p \geq 0$ throughout the evolution
- **Non-singular**: Scale factor remains positive: $a(t) \geq a_{\min} > 0$
- **Predictions**: $r \approx 0.003-0.005$, $n_s \approx 0.96-0.97$, $f_{\rm NL}^{\rm local} \sim 1$

## The Mechanism

The key equations in a closed universe ($k = +1$):

**Friedmann constraint:**
$$H^2 = \frac{\rho}{3M_{\rm Pl}^2} - \frac{1}{a^2}$$

**Acceleration equation:**
$$\dot{H} = -\frac{\rho + p}{2M_{\rm Pl}^2} + \frac{1}{a^2}$$

The $+1/a^2$ term from spatial curvature enables $\dot{H} > 0$ even when the NEC is satisfied!

## Files

- `main.tex` - LaTeX source of the paper
- `Bouncing_Cosmology_by_Oleksandr_Kravchenko_OkMathOrg.pdf` - Compiled paper with embedded figures
- `numerical_solution.py` - Python code for numerical solutions
- `verify_christoffel.py` - Symbolic verification of field-space geometry
- `bounce_closed.pdf` - Figure 1: Bounce solution
- `kinetic_suppression.pdf` - Figure 2: Kinetic suppression mechanism

## Requirements

```
numpy
scipy
matplotlib
sympy (for verification script)
```

## Running the Code

```bash
python numerical_solution.py
```

This will generate all figures and print a summary of the bounce solution.

## Citation

If you use this code or find this work useful, please cite:

```bibtex
@article{Kravchenko2025bounce,
  author = {Kravchenko, Oleksandr},
  title = {Non-Singular Bouncing Cosmology from Hyperbolic Field Space Geometry},
  year = {2025},
  eprint = {arXiv:XXXX.XXXXX},
  archivePrefix = {arXiv},
  primaryClass = {gr-qc}
}
```

## Contact

For questions or comments, please open an issue or contact the author.
