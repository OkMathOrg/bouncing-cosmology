#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding='utf-8')
"""
Generate single-panel figures for v4 paper.

Each panel in the paper corresponds to ONE stand-alone figure environment
with its own caption, following the visual style of physics journals
(PRD / JCAP / PRL).  Figures are stacked vertically in the paper, one per
panel, so each panel is large, clean, and carries its own caption.

Output PDFs are named <group>_<letter>_<short>.pdf (e.g. fsg_a_metric.pdf),
so that the correspondence with the LaTeX subfigure references is obvious.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

# =============================================================================
# Typography
# =============================================================================
#
# Figures are embedded at ~0.78*textwidth (centred) in the paper, with
# textwidth = 6.5 in.  The matplotlib figure size below (6.0 x 4.0 in) is
# therefore shown at roughly its native scale, so fonts set here render in
# the PDF at essentially the same point size.
PAPER_TEXTWIDTH_IN = 6.5
SINGLE_W = 6.0       # one-panel figure width in inches
SINGLE_H = 3.9       # default one-panel figure height (3:2-ish aspect)

plt.rcParams.update({
    'font.family':        'serif',
    'mathtext.fontset':   'cm',
    'font.size':          12,
    'axes.labelsize':     13,
    'axes.titlesize':     13,
    'xtick.labelsize':    11,
    'ytick.labelsize':    11,
    'legend.fontsize':    11,
    'figure.titlesize':   13,
    'lines.linewidth':    2.0,
    'lines.markersize':   6,
    'axes.linewidth':     0.9,
    'grid.linewidth':     0.5,
    'grid.alpha':         0.35,
    'figure.dpi':         200,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.10,
    'pdf.fonttype':       42,
})

# --- Professional palette (tab10 subset) -----------------------------------
C_BLUE   = '#1f77b4'
C_ORANGE = '#ff7f0e'
C_GREEN  = '#2ca02c'
C_RED    = '#d62728'
C_PURPLE = '#9467bd'
C_BROWN  = '#8c564b'
C_PINK   = '#e377c2'
C_GRAY   = '#7f7f7f'
C_OLIVE  = '#bcbd22'
C_CYAN   = '#17becf'


# =============================================================================
# Helpers — every panel routes through these so the visual style is uniform
# =============================================================================
def _new(w=SINGLE_W, h=SINGLE_H):
    """Return a one-panel fig+ax with clean journal styling applied."""
    fig, ax = plt.subplots(figsize=(w, h))
    _style(ax)
    return fig, ax


def _style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)


def _legend(ax, **kw):
    defaults = dict(frameon=False, loc='best')
    defaults.update(kw)
    ax.legend(**defaults)


def _save(fig, fname):
    path = os.path.join(FIGDIR, fname)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {fname}")


# =============================================================================
# Field-space geometry  (three panels)
# =============================================================================
def fig_field_space_geometry():
    from bounce import g_chichi, dg_chichi_dphi, alpha, M_Pl

    phi = np.linspace(-10, 10, 500)
    g_sig = g_chichi(phi)
    g_exp = np.exp(2 * alpha * phi / M_Pl)

    # (a) metric
    fig, ax = _new()
    ax.plot(phi, g_sig,                  color=C_BLUE, lw=2.4,
            label=r'Sigmoid $g^S_{\chi\chi}(\phi)$')
    ax.plot(phi, np.minimum(g_exp, 5),    color=C_RED,  lw=1.8, ls='--',
            label=r'Exponential $e^{2\alpha\phi/M_{\rm Pl}}$ (unbounded)')
    ax.axhline(1, color=C_GRAY, ls=':', alpha=0.7,
               label=r'saturation $g=1$')
    ax.set_xlabel(r'$\phi / M_{\rm Pl}$')
    ax.set_ylabel(r'$g_{\chi\chi}(\phi)$')
    ax.set_ylim(-0.05, 2.0)
    _legend(ax)
    _save(fig, 'fsg_a_metric.pdf')

    # (b) curvature
    fig, ax = _new()
    K = -0.5 * alpha ** 2 * (1 - g_sig) ** 2
    ax.plot(phi, K, color=C_BLUE, lw=2.4)
    ax.axhline(-alpha ** 2 / 2, color=C_GRAY, ls=':', alpha=0.8,
               label=r'hyperbolic asymptote $K=-\alpha^2/2$')
    ax.axhline(0,                 color=C_GRAY, ls='-', alpha=0.4,
               label=r'flat asymptote $K=0$')
    ax.set_xlabel(r'$\phi / M_{\rm Pl}$')
    ax.set_ylabel(r'Field-space curvature $K(\phi)$')
    _legend(ax)
    _save(fig, 'fsg_b_curvature.pdf')

    # (c) Christoffel / decoupling
    fig, ax = _new()
    Gamma = alpha / M_Pl * (1 - g_sig)
    ax.plot(phi, Gamma,     color=C_BLUE,   lw=2.4,
            label=r'$\Gamma^\chi_{\phi\chi}(\phi)$')
    ax.plot(phi, 1 - g_sig, color=C_ORANGE, lw=2.0, ls='--',
            label=r'$1-g_{\chi\chi}(\phi)$ (decoupling factor)')
    ax.set_xlabel(r'$\phi / M_{\rm Pl}$')
    ax.set_ylabel('coupling strength')
    _legend(ax)
    _save(fig, 'fsg_c_decoupling.pdf')


# =============================================================================
# Potential and inflationary dynamics  (four panels)
# =============================================================================
def fig_potential_and_dynamics():
    from bounce import (V, slow_roll_epsilon, N_to_end_analytical,
                        compute_observables_analytical, V0)
    obs = compute_observables_analytical(N=60)
    ns_anal, r_anal = obs['n_s'], obs['r']

    phi = np.linspace(-2, 8, 600)
    V_vals = np.array([V(p) for p in phi]) / V0

    # (a) potential
    fig, ax = _new()
    ax.plot(phi, V_vals, color=C_BLUE, lw=2.4)
    ax.set_xlabel(r'$\phi / M_{\rm Pl}$')
    ax.set_ylabel(r'$V(\phi) / V_0$')
    _save(fig, 'pd_a_potential.pdf')

    # (b) epsilon
    fig, ax = _new()
    eps = slow_roll_epsilon(phi)
    eps = np.clip(eps, 1e-10, 10)
    ax.semilogy(phi, eps, color=C_BLUE, lw=2.4)
    ax.axhline(1, color=C_RED, ls='--', lw=1.8, alpha=0.8,
               label=r'$\epsilon = 1$ (end of inflation)')
    ax.set_xlabel(r'$\phi / M_{\rm Pl}$')
    ax.set_ylabel(r'$\epsilon_V(\phi)$')
    ax.set_ylim(1e-4, 10)
    _legend(ax)
    _save(fig, 'pd_b_epsilon.pdf')

    # (c) N(phi)
    fig, ax = _new()
    phi_inf = np.linspace(1, 7, 300)
    N_vals = [N_to_end_analytical(p) for p in phi_inf]
    ax.plot(phi_inf, N_vals, color=C_BLUE, lw=2.4)
    ax.axhline(60, color=C_RED, ls='--', lw=1.8, alpha=0.8,
               label=r'$N_{\rm CMB} = 60$')
    ax.set_xlabel(r'$\phi / M_{\rm Pl}$')
    ax.set_ylabel(r'$N(\phi)$  (e-folds to end)')
    _legend(ax)
    _save(fig, 'pd_c_efolds.pdf')

    # (d) n_s-r plane
    fig, ax = _new()
    N_range = np.arange(40, 80)
    n_s_curve = [1 - 2/n for n in N_range]
    r_curve   = [12/n**2 for n in N_range]
    ax.plot(n_s_curve, r_curve, color=C_BLUE, lw=2.4,
            label=r'Starobinsky, $N\in[40,80]$')
    ax.axvspan(0.9649 - 0.0042, 0.9649 + 0.0042,
               alpha=0.15, color=C_RED, label=r'Planck 2018 $1\sigma$')
    ax.plot(ns_anal, r_anal, 'o', color=C_RED, ms=9,
            markeredgewidth=1.5, markerfacecolor='white',
            label='This work ($N=60$)')
    ax.set_xlabel(r'$n_s$')
    ax.set_ylabel(r'$r$')
    _legend(ax, loc='best')
    _save(fig, 'pd_d_ns_r.pdf')


# =============================================================================
# Background evolution through the bounce  (four panels)
# =============================================================================
def fig_background_evolution(bg):
    from bounce import omega

    t = bg['t']
    a = bg['a']
    H = bg['H']
    phi = bg['phi']
    g = bg['g']
    t_norm = t * omega

    # (a) scale factor
    fig, ax = _new()
    ax.semilogy(t_norm, a, color=C_BLUE, lw=2.0)
    ax.set_xlabel(r'$t \cdot \omega$')
    ax.set_ylabel(r'scale factor $a$')
    _save(fig, 'bg_a_scale_factor.pdf')

    # (b) Hubble
    fig, ax = _new()
    ax.plot(t_norm, H/omega, color=C_BLUE, lw=2.0)
    ax.axhline(0, color=C_GRAY, ls='-', alpha=0.4)
    ax.set_xlabel(r'$t \cdot \omega$')
    ax.set_ylabel(r'$H/\omega$')
    _save(fig, 'bg_b_hubble.pdf')

    # (c) inflaton
    fig, ax = _new()
    ax.plot(t_norm, phi, color=C_BLUE, lw=2.0)
    ax.set_xlabel(r'$t \cdot \omega$')
    ax.set_ylabel(r'$\phi / M_{\rm Pl}$')
    _save(fig, 'bg_c_inflaton.pdf')

    # (d) g_chichi saturation: plot |1 - g| on log scale so the
    # post-bounce convergence to canonical kinetic terms is actually
    # visible.  Plotting g directly forces a matplotlib offset
    # ("1e-9 + 9.9999999e-1") that obscures the saturation behavior.
    one_minus_g = np.maximum(np.abs(1.0 - g), 1e-18)
    fig, ax = _new()
    ax.semilogy(t_norm, one_minus_g, color=C_BLUE, lw=2.0)
    ax.set_xlabel(r'$t \cdot \omega$')
    ax.set_ylabel(r'$|1 - g_{\chi\chi}(\phi(t))|$')
    _save(fig, 'bg_d_metric.pdf')


# =============================================================================
# Bounce detail (three panels)
# =============================================================================
def fig_bounce_zoom(bg):
    from bounce import V, V0, omega

    i_b = bg['i_bounce']
    a_min = bg['a_min']
    t_bounce = bg['t'][i_b]
    # Wide slice (energy panel), then a tight slice for the (a)/(b)
    # bounce-region zooms.  Without the tight slice the parabolic
    # minimum of a(t) is invisible, because post-bounce inflation
    # raises a by ~10^8 within the wide window.
    width = max(500, len(bg['t']) // 20)
    sl = slice(max(0, i_b - width), min(len(bg['t']), i_b + width))

    t_wide = (bg['t'][sl] - t_bounce) * omega
    a_wide = bg['a'][sl]
    H_wide = bg['H'][sl] / omega
    phi_s = bg['phi'][sl]
    phidot_s = bg['phi_dot'][sl]
    chidot_s = bg['chi_dot'][sl]
    g_s = bg['g'][sl]
    V_s = np.array([V(p) for p in phi_s])
    K_phi = 0.5 * phidot_s ** 2
    K_chi = 0.5 * g_s * chidot_s ** 2

    # Tight zoom for panels (a) and (b): restrict to |t - t_bounce| <= 3/omega.
    # This is ~3 inflationary Hubble times and is just wide enough to show
    # the parabolic minimum of a(t) and the H = 0 crossing without being
    # swamped by the rapid post-bounce inflationary growth.
    tight_mask = np.abs(t_wide) <= 3.0
    t_tight = t_wide[tight_mask]
    a_tight = a_wide[tight_mask]
    H_tight = H_wide[tight_mask]

    # (a) scale factor near bounce — normalize by a_min so the parabolic
    # minimum is visible on a linear scale.
    fig, ax = _new()
    ax.plot(t_tight, a_tight / a_min, color=C_BLUE, lw=2.2)
    ax.axvline(0, color=C_GRAY, ls=':', alpha=0.6,
               label=r'bounce ($H=0$)')
    ax.set_xlabel(r'$(t - t_{\rm bounce})\cdot \omega$')
    ax.set_ylabel(r'$a / a_{\rm min}$')
    _legend(ax, loc='upper center')
    _save(fig, 'bz_a_scale_factor.pdf')

    # (b) Hubble through zero
    fig, ax = _new()
    ax.plot(t_tight, H_tight, color=C_BLUE, lw=2.2)
    ax.axhline(0, color=C_GRAY, ls='-', alpha=0.5)
    ax.axvline(0, color=C_GRAY, ls=':', alpha=0.6)
    ax.set_xlabel(r'$(t - t_{\rm bounce})\cdot \omega$')
    ax.set_ylabel(r'$H/\omega$')
    _save(fig, 'bz_b_hubble.pdf')

    # (c) energy components — keep the wider window so post-bounce decay
    # of the kinetic terms is visible, but use the same t - t_bounce
    # centering as panels (a)/(b).
    fig, ax = _new()
    ax.semilogy(t_wide, np.maximum(V_s, 1e-30), color=C_BLUE,  lw=2.2,
                label=r'potential $V$')
    ax.semilogy(t_wide, np.maximum(K_phi, 1e-30), color=C_RED, lw=2.0, ls='--',
                label=r'kinetic $\frac{1}{2}\dot\phi^2$')
    ax.semilogy(t_wide, np.maximum(K_chi, 1e-30), color=C_GREEN, lw=2.0, ls=':',
                label=r'kinetic $\frac{1}{2} g_{\chi\chi}\dot\chi^2$')
    ax.axvline(0, color=C_GRAY, ls=':', alpha=0.6)
    ax.set_xlabel(r'$(t - t_{\rm bounce})\cdot \omega$')
    ax.set_ylabel(r'energy density (in $M_{\rm Pl}^4$)')
    _legend(ax, loc='upper center', bbox_to_anchor=(0.5, -0.18),
            ncol=3, fontsize=10)
    _save(fig, 'bz_c_energies.pdf')


# =============================================================================
# Flatness (two panels)
# =============================================================================
def fig_flatness_evolution(bg):
    from bounce import V0

    a_b = bg['a_min']
    H_inf = np.sqrt(V0 / 3)

    N = np.linspace(0, 70, 300)
    Omega_k = 1.0 / (a_b ** 2 * np.exp(2 * N) * H_inf ** 2)

    # (a) curvature dilution
    fig, ax = _new()
    ax.semilogy(N, Omega_k, color=C_BLUE, lw=2.4)
    ax.axhline(0.001, color=C_RED, ls='--', lw=1.8,
               label=r'target $|\Omega_k| = 10^{-3}$')
    ax.axvline(3.5,   color=C_GRAY, ls=':', lw=1.5, alpha=0.7,
               label=r'$N = 3.5$ reaches target')
    ax.set_xlabel(r'post-bounce e-folds $N$')
    ax.set_ylabel(r'$|\Omega_k|(N)$')
    _legend(ax)
    _save(fig, 'fl_a_dilution.pdf')

    # (b) required vs achieved
    fig, ax = _new()
    labels = ['Required\n(target $|\\Omega_k|<10^{-3}$)',
              f'Achieved (simulation)']
    values = [3.5, float(bg['N_total'])]
    colors = [C_ORANGE, C_GREEN]
    bars = ax.bar(labels, values, color=colors, width=0.55, edgecolor='k',
                  linewidth=0.8)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width()/2, v*1.02, f'{v:.1f}',
                ha='center', va='bottom', fontsize=11)
    ax.set_ylabel('post-bounce e-folds')
    ax.set_ylim(0, max(values) * 1.18)
    _save(fig, 'fl_b_compare.pdf')


# =============================================================================
# Basin of attraction (two panels)
# =============================================================================
def fig_basin_of_attraction():
    from bounce import test_basin_robust

    rate, results = test_basin_robust(n_samples=21)

    logs    = np.array([r['log_chi_dot'] for r in results])
    success = np.array([r['success']     for r in results])
    N_post  = np.array([r.get('N_post', 0) for r in results])

    colors_bar = [C_GREEN if s else C_RED for s in success]

    # (a) success rate
    fig, ax = _new()
    ax.bar(logs, np.ones_like(logs), color=colors_bar, width=0.9,
           edgecolor='black', linewidth=0.5)
    ax.axvline(-4, color=C_GRAY, ls=':', lw=1.5,
               label=r'sigmoid suppression active: $\dot\chi_0 > 10^{-4}\sqrt{V_0}$')
    ax.set_xlabel(r'$\log_{10}(\dot\chi_0 / \sqrt{V_0})$')
    ax.set_yticks([])
    ax.set_ylim(0, 1.15)
    ax.set_title(f'overall success rate: {rate:.0f}% ({int(success.sum())}/{len(results)})',
                 pad=8)
    _legend(ax, loc='upper right')
    _save(fig, 'ba_a_success_map.pdf')

    # (b) N_post
    fig, ax = _new()
    ax.bar(logs, N_post, color=colors_bar, width=0.9,
           edgecolor='black', linewidth=0.5)
    ax.axhline(60, color=C_RED, ls='--', lw=1.8,
               label=r'$N_{\rm CMB} = 60$')
    ax.axvline(-4, color=C_GRAY, ls=':', lw=1.5, alpha=0.7)
    ax.set_xlabel(r'$\log_{10}(\dot\chi_0 / \sqrt{V_0})$')
    ax.set_ylabel('post-bounce e-folds $N$')
    _legend(ax, loc='upper right')
    _save(fig, 'ba_b_efolds.pdf')


# =============================================================================
# BKL / Bianchi IX (three panels)
# =============================================================================
def fig_bkl_analysis():
    from bounce import run_bianchi_ix_simulation, a_min_expected

    sigma_sq_values = [1e-5, 1e0, 1e5, 1e10, 1e15]
    runs = []
    for sig in sigma_sq_values:
        try:
            bg = run_bianchi_ix_simulation(sigma_sq_init=sig, n_points=50000)
            if bg is None:
                continue
            runs.append((sig, bg))
        except Exception:
            continue

    palette = [C_BLUE, C_ORANGE, C_GREEN, C_PURPLE, C_BROWN, C_PINK, C_OLIVE]

    # (a) shear vs curvature
    fig, ax = _new()
    for (sig, bg), col in zip(runs, palette):
        i_b = bg['i_bounce']
        a_arr = bg['a'][:i_b + 100] / a_min_expected
        ax.semilogy(a_arr, bg['shear_term'][:i_b + 100],
                    color=col, lw=1.6,
                    label=rf'$\Sigma_0^2 = 10^{{{int(np.log10(sig))}}}$')
    a_ref = np.logspace(np.log10(0.95 * a_min_expected),
                        np.log10(2.0 * a_min_expected), 100)
    ax.semilogy(a_ref / a_min_expected, 1.0 / a_ref ** 2,
                color='k', ls='--', lw=2.0, label=r'$1/a^2$ (curvature)')
    ax.set_xlabel(r'$a / a_{\rm min}$')
    ax.set_ylabel(r'energy density terms')
    _legend(ax, loc='upper center', bbox_to_anchor=(0.5, -0.18),
            ncol=3, fontsize=10)
    _save(fig, 'bkl_a_shear_vs_curv.pdf')

    # (b) Kasner ratio
    fig, ax = _new()
    for (sig, bg), col in zip(runs, palette):
        i_b = bg['i_bounce']
        a_arr = bg['a'][:i_b + 100] / a_min_expected
        ax.semilogy(a_arr, bg['kasner_ratio'][:i_b + 100],
                    color=col, lw=1.6,
                    label=rf'$\Sigma_0^2 = 10^{{{int(np.log10(sig))}}}$')
    ax.axhline(1, color=C_RED, ls='--', lw=2.0,
               label=r'Kasner threshold $R_K = 1$')
    ax.set_xlabel(r'$a / a_{\rm min}$')
    ax.set_ylabel(r'$R_K = \Sigma^2 / a^4$')
    _legend(ax, loc='upper center', bbox_to_anchor=(0.5, -0.18),
            ncol=3, fontsize=10)
    _save(fig, 'bkl_b_kasner_ratio.pdf')

    # (c) R_K at bounce vs initial Σ²
    fig, ax = _new()
    if runs:
        sigma_vals = np.array([s for s, _ in runs])
        ratios = np.array([bg['kasner_ratio'][bg['i_bounce']] for _, bg in runs])
        ax.loglog(sigma_vals, ratios, 'o-', color=C_BLUE, lw=2.2, ms=8,
                  markerfacecolor='white', markeredgewidth=1.8)
        ax.axhline(1, color=C_RED, ls='--', lw=2.0,
                   label=r'Kasner threshold $R_K = 1$')
    ax.set_xlabel(r'initial shear amplitude $\Sigma_0^2$')
    ax.set_ylabel(r'$R_K$ at the bounce point')
    _legend(ax)
    _save(fig, 'bkl_c_at_bounce.pdf')


# =============================================================================
# Alpha independence (three panels)
# =============================================================================
def fig_alpha_independence():
    from bounce import test_alpha_independence, compute_observables_analytical

    results = test_alpha_independence()
    alphas = sorted([a for a in results if results[a].get('success')])
    n_s_vals = np.array([results[a]['n_s_numerical'] for a in alphas])
    r_vals   = np.array([results[a]['r_numerical']   for a in alphas])
    g_vals   = np.array([results[a]['g_CMB']         for a in alphas])

    obs = compute_observables_analytical(N=60)

    # (a) n_s
    fig, ax = _new()
    ax.semilogx(alphas, n_s_vals, 'o-', color=C_BLUE, lw=2.0, ms=7,
                markerfacecolor='white', markeredgewidth=1.6,
                label=r'numerical $n_s^{\rm kin}$')
    ax.axhline(obs['n_s'], color=C_GRAY, ls='--', lw=1.8,
               label=rf"slow-roll $n_s = 1 - 2/60 = {obs['n_s']:.4f}$")
    ax.axhspan(0.9649 - 0.0042, 0.9649 + 0.0042, alpha=0.15, color=C_RED,
               label=r'Planck 2018 $1\sigma$')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'spectral index $n_s$')
    ax.set_ylim(0.955, 0.975)
    _legend(ax, loc='lower right', fontsize=10)
    _save(fig, 'ai_a_ns.pdf')

    # (b) r
    fig, ax = _new()
    ax.semilogx(alphas, r_vals, 'o-', color=C_BLUE, lw=2.0, ms=7,
                markerfacecolor='white', markeredgewidth=1.6,
                label=r'numerical $r$')
    ax.axhline(obs['r'], color=C_GRAY, ls='--', lw=1.8,
               label=rf"slow-roll $r = 12/60^2 = {obs['r']:.4f}$")
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'tensor-to-scalar ratio $r$')
    _legend(ax, loc='best', fontsize=10)
    _save(fig, 'ai_b_r.pdf')

    # (c) g(phi_CMB) saturation
    fig, ax = _new()
    one_minus_g = np.clip(1.0 - g_vals, 1e-16, None)
    ax.loglog(alphas, one_minus_g, 'o-', color=C_BLUE, lw=2.0, ms=7,
              markerfacecolor='white', markeredgewidth=1.6)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$1 - g_{\chi\chi}(\phi_{\rm CMB})$')
    _save(fig, 'ai_c_g_saturation.pdf')


# =============================================================================
# Perturbation analysis (six panels — one per physics observable)
# =============================================================================
def fig_perturbation_analysis(pert_results):
    ng = pert_results.get('ng_solver') or pert_results.get('ms_solver')
    mode_results = pert_results.get('mode_results', {})

    if not mode_results or ng is None:
        print("  (skipping perturbation figures: no mode_results)")
        return

    k_sorted = sorted(mode_results.keys())
    k_example = k_sorted[len(k_sorted) // 2]
    mode = mode_results[k_example]
    t_arr = mode['t']
    t_b = ng.bg['t'][ng.bg['i_bounce']]
    t_plot = t_arr - t_b

    # (a) delta phi and Phi through bounce
    fig, ax = _new()
    if 'dphi_sq' in mode:
        ax.semilogy(t_plot, np.maximum(mode['dphi_sq'], 1e-30),
                    color=C_BLUE, lw=1.8,
                    label=r'$|\delta\phi_k|^2$')
        ax.semilogy(t_plot, np.maximum(mode['Phi_sq'], 1e-30),
                    color=C_RED,  lw=1.8, ls='--',
                    label=r'$|\Phi_k|^2$')
    ax.axvline(0, color=C_GRAY, ls=':', alpha=0.8, label='bounce ($H=0$)')
    ax.set_xlabel(r'$t - t_{\rm bounce}$')
    ax.set_ylabel(r'amplitude$^2$')
    _legend(ax, loc='best')
    _save(fig, 'pert_a_amplitudes.pdf')

    # (b) R through bounce
    fig, ax = _new()
    R_sq = mode.get('R_sq')
    if R_sq is not None:
        usable = R_sq > 1e-30
        if np.any(usable):
            ax.semilogy(t_arr[usable] - t_b, R_sq[usable],
                        color=C_BLUE, lw=1.8)
    ax.axvline(0, color=C_GRAY, ls=':', alpha=0.8, label='bounce ($H=0$)')
    ax.set_xlabel(r'$t - t_{\rm bounce}$')
    ax.set_ylabel(r'$|\mathcal{R}_k|^2$')
    _legend(ax, loc='best')
    _save(fig, 'pert_b_curvature.pdf')

    # (c) momentum constraint median error
    fig, ax = _new()
    errs = np.array([mode_results[k].get('constraint_rel_err', 0)
                     for k in k_sorted])
    ax.semilogy(k_sorted, np.maximum(errs, 1e-15),
                'o-', color=C_BLUE, lw=1.6, ms=7,
                markerfacecolor='white', markeredgewidth=1.4)
    ax.axhline(0.01, color=C_RED, ls='--', lw=1.8,
               label=r'1% threshold')
    ax.set_xscale('log')
    ax.set_xlabel(r'$k$')
    ax.set_ylabel(r'MC relative error (median)')
    _legend(ax, loc='best')
    _save(fig, 'pert_c_constraint.pdf')

    # (d) power spectrum with fits
    fig, ax = _new()
    k_arr = pert_results.get('k_arr')
    P_arr = pert_results.get('P_arr')
    n_s_num = pert_results.get('n_s_numerical')
    if k_arr is not None and P_arr is not None:
        valid = (P_arr > 0) & np.isfinite(P_arr)
        if np.sum(valid) > 1:
            ax.loglog(k_arr[valid], P_arr[valid], 'o-', color=C_BLUE,
                      lw=1.6, ms=7, markerfacecolor='white',
                      markeredgewidth=1.4,
                      label='numerical modes')
            from bounce import compute_observables_analytical
            ns_anal = compute_observables_analytical(N=60)['n_s']
            k0 = k_arr[valid][len(k_arr[valid]) // 2]
            P0 = P_arr[valid][len(P_arr[valid]) // 2]
            kk = np.logspace(np.log10(k_arr[valid].min()),
                             np.log10(k_arr[valid].max()), 80)
            ax.loglog(kk, P0 * (kk / k0) ** (ns_anal - 1),
                      color=C_GRAY, ls='--', lw=1.6,
                      label=rf'slow-roll $n_s={ns_anal:.4f}$')
            if n_s_num is not None:
                ax.loglog(kk, P0 * (kk / k0) ** (n_s_num - 1),
                          color=C_RED, ls=':', lw=1.9,
                          label=rf'numerical slope $n_s^{{\rm fit}}={n_s_num:.3f}$')
    ax.set_xlabel(r'$k$')
    ax.set_ylabel(r'$P_{\mathcal{R}}(k)$')
    _legend(ax, loc='best')
    _save(fig, 'pert_d_power_spectrum.pdf')

    # (e) fit residuals
    fig, ax = _new()
    res  = pert_results.get('fit_residuals')
    lk   = pert_results.get('fit_log_k')
    if res is not None and lk is not None:
        ax.plot(np.exp(lk), res, 'o-', color=C_BLUE, lw=1.6, ms=7,
                markerfacecolor='white', markeredgewidth=1.4)
        ax.axhline( 0.1, color=C_RED, ls='--', lw=1.5, alpha=0.7,
                    label=r'$\pm 0.1$')
        ax.axhline(-0.1, color=C_RED, ls='--', lw=1.5, alpha=0.7)
        ax.axhline( 0.0, color=C_GRAY, ls='-', lw=1.0, alpha=0.5)
        ax.set_xscale('log')
    ax.set_xlabel(r'$k$')
    ax.set_ylabel(r'$\ln P - \mathrm{fit}$')
    _legend(ax, loc='best')
    _save(fig, 'pert_e_residuals.pdf')

    # (f) isocurvature transfer
    fig, ax = _new()
    T_RS = pert_results.get('T_RS_arr')
    if T_RS is not None and k_arr is not None:
        valid = np.isfinite(T_RS) & (T_RS >= 0)
        if np.any(valid):
            ax.semilogy(k_arr[valid], np.maximum(T_RS[valid], 1e-20),
                        'o-', color=C_BLUE, lw=1.6, ms=7,
                        markerfacecolor='white', markeredgewidth=1.4)
    ax.axhline(0.01, color=C_RED, ls='--', lw=1.8,
               label=r'1% threshold')
    ax.axhline(1e-4, color=C_GREEN, ls=':', lw=1.6,
               label=r'paper bound $T_{RS} < 10^{-4}$')
    ax.set_xscale('log')
    ax.set_xlabel(r'$k$')
    ax.set_ylabel(r'$T_{RS} = P_{SS} / P_{\mathcal{R}}$')
    _legend(ax, loc='best')
    _save(fig, 'pert_f_isocurvature.pdf')


# =============================================================================
# Orchestration
# =============================================================================
def generate_all_figures(bg=None, pert_results=None):
    print("\n  Generating figures...")
    fig_field_space_geometry()
    fig_potential_and_dynamics()
    if bg is not None:
        fig_background_evolution(bg)
        fig_flatness_evolution(bg)
        fig_bounce_zoom(bg)
    fig_basin_of_attraction()
    fig_bkl_analysis()
    fig_alpha_independence()
    if pert_results is not None:
        fig_perturbation_analysis(pert_results)
    print("  All figures generated.")


if __name__ == "__main__":
    print("Generating v4 figures...")
    from bounce import run_simulation_robust
    bg = run_simulation_robust(phi0=10.0, n_points=100000)
    pert_results = None
    if bg is not None:
        from perturbations import validate_perturbations_full
        pert_results = validate_perturbations_full(bg)
    generate_all_figures(bg, pert_results)
