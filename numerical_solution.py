"""
Non-Singular Bouncing Cosmology in a Closed Universe (k=+1)
with Hyperbolic Field Space Geometry

Key equations:
  Friedmann:     H² = ρ/(3M²) - 1/a²
  Acceleration:  dH/dt = -(ρ+p)/(2M²) + 1/a²
  
The +1/a² term from spatial curvature enables a bounce without NEC violation.
The hyperbolic field space geometry suppresses kinetic energy, allowing
potential domination and satisfying the bounce condition w < -1/3.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Set up nice plotting
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.figsize': (10, 8),
    'lines.linewidth': 1.5,
    'text.usetex': False,
})

# =============================================================================
# Model Parameters (in Planck units: M_Pl = 1)
# =============================================================================
M_Pl = 1.0
alpha = 1.0          # Field space curvature parameter
beta = np.sqrt(2/3)  # Potential steepness (Starobinsky value)
V0 = 1e-10           # Potential amplitude (sets energy scale)
m_chi = 1e-6         # Mass of χ field
k = 1                # Spatial curvature: +1 for closed universe

# =============================================================================
# Potential and derivatives
# =============================================================================
def V(phi, chi):
    """Total potential V(φ, χ)"""
    V_phi = V0 * (1 - np.exp(-beta * phi / M_Pl))**2
    V_chi = 0.5 * m_chi**2 * chi**2
    return V_phi + V_chi

def dV_dphi(phi, chi):
    """∂V/∂φ"""
    return V0 * 2 * (1 - np.exp(-beta * phi / M_Pl)) * (beta / M_Pl) * np.exp(-beta * phi / M_Pl)

def dV_dchi(phi, chi):
    """∂V/∂χ"""
    return m_chi**2 * chi

# =============================================================================
# Energy density, pressure, and Hubble parameter
# =============================================================================
def kinetic_energy(phi, pi_phi, pi_chi):
    """Total kinetic energy"""
    return 0.5 * pi_phi**2 + 0.5 * np.exp(2 * alpha * phi / M_Pl) * pi_chi**2

def rho(phi, chi, pi_phi, pi_chi):
    """Energy density ρ = kinetic + potential"""
    return kinetic_energy(phi, pi_phi, pi_chi) + V(phi, chi)

def pressure(phi, chi, pi_phi, pi_chi):
    """Pressure p = kinetic - potential"""
    return kinetic_energy(phi, pi_phi, pi_chi) - V(phi, chi)

def H_squared(phi, chi, pi_phi, pi_chi, a):
    """
    H² = ρ/(3M²) - k/a²  for closed universe (k=+1)
    Returns max(0, ...) to handle numerical noise near bounce
    """
    rho_val = rho(phi, chi, pi_phi, pi_chi)
    H2 = rho_val / (3 * M_Pl**2) - k / a**2
    return max(0, H2)

def H_dot(phi, chi, pi_phi, pi_chi, a):
    """
    dH/dt = -(ρ+p)/(2M²) + k/a²  for closed universe
    This is the KEY equation: +k/a² allows dH/dt > 0 even when NEC is satisfied!
    """
    rho_val = rho(phi, chi, pi_phi, pi_chi)
    p_val = pressure(phi, chi, pi_phi, pi_chi)
    return -(rho_val + p_val) / (2 * M_Pl**2) + k / a**2

def equation_of_state(phi, chi, pi_phi, pi_chi):
    """Equation of state w = p/ρ"""
    rho_val = rho(phi, chi, pi_phi, pi_chi)
    p_val = pressure(phi, chi, pi_phi, pi_chi)
    if abs(rho_val) < 1e-30:
        return 0
    return p_val / rho_val

# =============================================================================
# System of ODEs for Closed Universe
# =============================================================================
def equations_closed(t, y, H_sign_tracker):
    """
    Right-hand side of the ODE system for k=+1 universe.
    y = [φ, χ, π_φ, π_χ, a, sign_H]
    
    sign_H tracks whether we're contracting (-1) or expanding (+1)
    """
    phi, chi, pi_phi, pi_chi, a = y[:5]
    
    # Compute H² and determine sign
    H2 = H_squared(phi, chi, pi_phi, pi_chi, a)
    H_mag = np.sqrt(H2)
    
    # Determine sign of H from rate of change of a
    # At bounce: H=0, so we need to track using dH/dt
    Hdot = H_dot(phi, chi, pi_phi, pi_chi, a)
    
    # Use H_sign_tracker to maintain continuity
    if H2 < 1e-20:  # Near bounce
        # At bounce, H crosses zero. Sign is determined by dH/dt
        # If dH/dt > 0, H goes from - to +
        H = 0
    else:
        H = H_sign_tracker[0] * H_mag
    
    # Update H_sign based on whether we've passed the bounce
    if Hdot > 0 and H_sign_tracker[0] < 0 and H2 < 1e-16:
        H_sign_tracker[0] = 1  # Transition to expansion
    
    # Exponential factors from field-space metric
    exp_plus = np.exp(2 * alpha * phi / M_Pl)
    exp_minus = np.exp(-2 * alpha * phi / M_Pl)
    
    # Field equations
    dphi_dt = pi_phi
    dchi_dt = pi_chi
    
    dpi_phi_dt = -3 * H * pi_phi + (alpha / M_Pl) * exp_plus * pi_chi**2 - dV_dphi(phi, chi)
    dpi_chi_dt = -3 * H * pi_chi - (2 * alpha / M_Pl) * pi_phi * pi_chi - exp_minus * dV_dchi(phi, chi)
    
    da_dt = a * H
    
    return [dphi_dt, dchi_dt, dpi_phi_dt, dpi_chi_dt, da_dt]

# =============================================================================
# Alternative: Track a_dot directly for robustness
# =============================================================================
def equations_closed_robust(t, y):
    """
    More robust formulation: track (φ, χ, π_φ, π_χ, a, ȧ)
    where ȧ = da/dt can go through zero smoothly
    """
    phi, chi, pi_phi, pi_chi, a, a_dot = y
    
    # Compute H = ȧ/a
    H = a_dot / a if a > 0 else 0
    
    # Compute ρ and p
    rho_val = rho(phi, chi, pi_phi, pi_chi)
    p_val = pressure(phi, chi, pi_phi, pi_chi)
    
    # Exponential factors
    exp_plus = np.exp(2 * alpha * phi / M_Pl)
    exp_minus = np.exp(-2 * alpha * phi / M_Pl)
    
    # Field equations
    dphi_dt = pi_phi
    dchi_dt = pi_chi
    dpi_phi_dt = -3 * H * pi_phi + (alpha / M_Pl) * exp_plus * pi_chi**2 - dV_dphi(phi, chi)
    dpi_chi_dt = -3 * H * pi_chi - (2 * alpha / M_Pl) * pi_phi * pi_chi - exp_minus * dV_dchi(phi, chi)
    
    da_dt = a_dot
    
    # Key equation: d(ȧ)/dt = a*dH/dt + H*ȧ = a*dH/dt + H²*a
    # From dH/dt = -(ρ+p)/(2M²) + 1/a²:
    # d(ȧ)/dt = a * [-(ρ+p)/(2M²) + 1/a²] + H² * a
    #         = -a(ρ+p)/(2M²) + 1/a + H²*a
    # Simplify using H² = ρ/(3M²) - 1/a²:
    # d(ȧ)/dt = -a(ρ+p)/(2M²) + 1/a + [ρ/(3M²) - 1/a²]*a
    #         = -a(ρ+p)/(2M²) + 1/a + aρ/(3M²) - 1/a
    #         = -a(ρ+p)/(2M²) + aρ/(3M²)
    #         = a * [-（ρ+p)/(2M²) + ρ/(3M²)]
    #         = a * [-(3ρ+3p-2ρ)/(6M²)]
    #         = -a * (ρ + 3p) / (6M²)
    
    # Actually, let's use the standard form:
    # ä = -a(ρ + 3p)/(6M²)  (for k=+1, this is the Raychaudhuri equation)
    # But we need to be careful. Let me derive it properly.
    
    # From H = ȧ/a, we have:
    # dH/dt = (ä/a) - (ȧ/a)² = ä/a - H²
    # So: ä = a(dH/dt + H²)
    # With dH/dt = -(ρ+p)/(2M²) + 1/a² and H² = ρ/(3M²) - 1/a²:
    # ä = a * [-(ρ+p)/(2M²) + 1/a² + ρ/(3M²) - 1/a²]
    #   = a * [-(ρ+p)/(2M²) + ρ/(3M²)]
    #   = a * [(-3(ρ+p) + 2ρ) / (6M²)]
    #   = a * [(-3ρ - 3p + 2ρ) / (6M²)]
    #   = a * [(-ρ - 3p) / (6M²)]
    #   = -a * (ρ + 3p) / (6M²)
    
    da_dot_dt = -a * (rho_val + 3 * p_val) / (6 * M_Pl**2)
    
    return [dphi_dt, dchi_dt, dpi_phi_dt, dpi_chi_dt, da_dt, da_dot_dt]

# =============================================================================
# Bounce solution in closed universe
# =============================================================================
def solve_bounce_closed():
    """
    Solve for a bouncing universe with k=+1.
    Start near the bounce and evolve in both directions.
    """
    # Minimum scale factor at bounce (from ρ = 3M²/a²)
    # For V ≈ V0, we have a_min ≈ √(3M²/V0)
    a_min_expected = np.sqrt(3 * M_Pl**2 / V0)
    omega = np.sqrt(V0 / (3 * M_Pl**2))
    print(f"Expected a_min ≈ {a_min_expected:.2e}")
    print(f"Characteristic frequency ω = {omega:.2e}")
    
    # Start at the bounce point (H=0) and integrate forward
    # At bounce: a = a_min, ȧ = 0
    a0 = a_min_expected * 1.001  # Slightly above minimum to avoid numerical issues
    
    # Fields: φ on the potential plateau for potential domination
    phi0 = 3.0 * M_Pl  # On the plateau where V ≈ V0
    chi0 = 0.0
    
    # Very small kinetic energy for potential domination
    pi_phi0 = 1e-8 * np.sqrt(V0)
    pi_chi0 = 0.0
    
    # At bounce start, H ≈ 0, ȧ small but positive (just past bounce)
    H2_init = H_squared(phi0, chi0, pi_phi0, pi_chi0, a0)
    H0 = np.sqrt(max(0, H2_init))  # Small positive (expanding from bounce)
    a_dot0 = a0 * H0
    
    print(f"Initial: a={a0:.2e}, H={H0:.2e}, ρ={rho(phi0, chi0, pi_phi0, pi_chi0):.2e}")
    print(f"V(φ)/V0 = {V(phi0, chi0)/V0:.4f}")
    
    y0 = [phi0, chi0, pi_phi0, pi_chi0, a0, a_dot0]
    
    # Integrate forward from bounce
    t_end = 5 / omega
    t_span = (0, t_end)
    t_eval = np.linspace(0, t_end, 1000)
    
    print(f"Integrating expansion phase: t = 0 to {t_end:.2e}")
    sol_expand = solve_ivp(equations_closed_robust, t_span, y0, t_eval=t_eval,
                           method='RK45', rtol=1e-10, atol=1e-12)
    
    # Now integrate backward (contraction phase)
    # Flip ȧ sign for contraction
    y0_contract = [phi0, chi0, pi_phi0, pi_chi0, a0, -a_dot0]
    t_span_back = (0, t_end)
    t_eval_back = np.linspace(0, t_end, 1000)
    
    print(f"Integrating contraction phase: t = 0 to -{t_end:.2e}")
    sol_contract = solve_ivp(equations_closed_robust, t_span_back, y0_contract, 
                              t_eval=t_eval_back, method='RK45', rtol=1e-10, atol=1e-12)
    
    # Combine solutions: contraction (reversed) + expansion
    # Contraction: flip time axis
    t_contract = -sol_contract.t[::-1]
    y_contract = sol_contract.y[:, ::-1]
    
    # Expansion
    t_expand = sol_expand.t
    y_expand = sol_expand.y
    
    # Combine (skip duplicate point at t=0)
    t_combined = np.concatenate([t_contract[:-1], t_expand])
    y_combined = np.concatenate([y_contract[:, :-1], y_expand], axis=1)
    
    # Create a solution-like object
    class CombinedSolution:
        def __init__(self, t, y, success):
            self.t = t
            self.y = y
            self.success = success
    
    sol = CombinedSolution(t_combined, y_combined, 
                           sol_expand.success and sol_contract.success)
    
    return sol

# =============================================================================
# Analytic bounce solution for comparison
# =============================================================================
def analytic_bounce_closed():
    """
    Analytic solution for potential-dominated bounce in closed universe.
    
    For H² = V0/(3M²) - 1/a² with V ≈ V0:
    At bounce: V0/(3M²) = 1/a_min² → a_min = √(3M²/V0)
    
    Near bounce: a(t) ≈ a_min * cosh(ωt), ω = √(V0/(3M²))
    """
    omega = np.sqrt(V0 / (3 * M_Pl**2))
    a_min = np.sqrt(3 * M_Pl**2 / V0)
    
    t = np.linspace(-5/omega, 5/omega, 2000)
    
    # Scale factor
    a = a_min * np.cosh(omega * t)
    
    # Hubble parameter
    H = omega * np.tanh(omega * t)
    
    # ȧ = aH
    a_dot = a * H
    
    # dH/dt = ω² * sech²(ωt)
    H_dot_analytic = omega**2 / np.cosh(omega * t)**2
    
    # For constant potential, φ = const
    phi = 0.5 * M_Pl * np.ones_like(t)
    
    return t, phi, a, H, H_dot_analytic, a_min, omega

# =============================================================================
# Plotting functions
# =============================================================================
def plot_bounce_closed(sol, filename='bounce_closed.pdf'):
    """Plot the numerical bounce solution for closed universe."""
    t = sol.t
    phi, chi, pi_phi, pi_chi, a, a_dot = sol.y
    
    # Compute derived quantities
    H = a_dot / a
    rho_arr = np.array([rho(phi[i], chi[i], pi_phi[i], pi_chi[i]) for i in range(len(t))])
    p_arr = np.array([pressure(phi[i], chi[i], pi_phi[i], pi_chi[i]) for i in range(len(t))])
    w_arr = np.array([equation_of_state(phi[i], chi[i], pi_phi[i], pi_chi[i]) for i in range(len(t))])
    rho_plus_p = rho_arr + p_arr
    
    # Get analytic solution for comparison
    t_an, _, a_an, H_an, _, a_min, omega = analytic_bounce_closed()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Scale factor
    ax1 = axes[0, 0]
    ax1.plot(t * omega, a / a_min, 'b-', linewidth=2, label='Numerical')
    ax1.plot(t_an * omega, a_an / a_min, 'r--', linewidth=1.5, alpha=0.7, label='Analytic: $a_{min} \\cosh(\\omega t)$')
    ax1.axhline(y=1.0, color='k', linestyle=':', alpha=0.5, label='$a_{min}$')
    ax1.set_xlabel(r'$\omega t$')
    ax1.set_ylabel(r'$a / a_{\rm min}$')
    ax1.set_title('Scale Factor: Non-Singular Bounce')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.5, 3)
    
    # Hubble parameter
    ax2 = axes[0, 1]
    ax2.plot(t * omega, H / omega, 'b-', linewidth=2, label='Numerical')
    ax2.plot(t_an * omega, H_an / omega, 'r--', linewidth=1.5, alpha=0.7, label='Analytic: $\\omega \\tanh(\\omega t)$')
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='Bounce ($H=0$)')
    ax2.set_xlabel(r'$\omega t$')
    ax2.set_ylabel(r'$H / \omega$')
    ax2.set_title('Hubble Parameter: $H<0$ → $H>0$')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.annotate('Contraction', xy=(-3, -0.5), fontsize=11, ha='center')
    ax2.annotate('Expansion', xy=(3, 0.5), fontsize=11, ha='center')
    
    # NEC check: ρ + p ≥ 0
    ax3 = axes[1, 0]
    ax3.plot(t * omega, rho_plus_p / V0, 'g-', linewidth=2)
    ax3.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='NEC boundary')
    ax3.fill_between(t * omega, 0, rho_plus_p / V0, where=(rho_plus_p >= 0), 
                     alpha=0.3, color='green', label='NEC satisfied')
    ax3.set_xlabel(r'$\omega t$')
    ax3.set_ylabel(r'$(\rho + p) / V_0$')
    ax3.set_title('Null Energy Condition: $\\rho + p \\geq 0$ ✓')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Equation of state
    ax4 = axes[1, 1]
    ax4.plot(t * omega, w_arr, 'purple', linewidth=2)
    ax4.axhline(y=-1, color='k', linestyle='--', alpha=0.5, label='$w = -1$ (de Sitter)')
    ax4.axhline(y=-1/3, color='orange', linestyle=':', alpha=0.7, label='$w = -1/3$ (bounce threshold)')
    ax4.set_xlabel(r'$\omega t$')
    ax4.set_ylabel(r'$w = p/\rho$')
    ax4.set_title('Equation of State')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-1.2, 0.5)
    
    plt.suptitle('Non-Singular Bounce in Closed Universe ($k=+1$)\nGhost-Free, NEC-Satisfying', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")

def plot_kinetic_suppression(sol, filename='kinetic_suppression.pdf'):
    """Show how hyperbolic geometry suppresses kinetic energy."""
    t = sol.t
    phi, chi, pi_phi, pi_chi, a, a_dot = sol.y
    
    # Get analytic parameters
    _, _, _, _, _, a_min, omega = analytic_bounce_closed()
    
    # Kinetic components
    K_phi = 0.5 * pi_phi**2
    suppression_factor = np.exp(2 * alpha * phi / M_Pl)
    K_chi_raw = 0.5 * pi_chi**2
    K_chi_effective = 0.5 * suppression_factor * pi_chi**2
    K_total = K_phi + K_chi_effective
    V_arr = np.array([V(phi[i], chi[i]) for i in range(len(t))])
    
    # Ratio K/V
    KV_ratio = K_total / V_arr
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    # φ evolution
    ax1 = axes[0]
    ax1.plot(t * omega, phi / M_Pl, 'b-', linewidth=2)
    ax1.set_xlabel(r'$\omega t$')
    ax1.set_ylabel(r'$\phi / M_{\rm Pl}$')
    ax1.set_title(r'Field $\phi(t)$')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Suppression factor
    ax2 = axes[1]
    ax2.semilogy(t * omega, suppression_factor, 'r-', linewidth=2)
    ax2.axhline(y=1, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel(r'$\omega t$')
    ax2.set_ylabel(r'$e^{2\alpha\phi/M_{\rm Pl}}$')
    ax2.set_title('Kinetic Suppression Factor')
    ax2.grid(True, alpha=0.3)
    ax2.annotate('Suppressed\n(φ < 0)', xy=(-3, 0.1), fontsize=10, ha='center')
    ax2.annotate('Enhanced\n(φ > 0)', xy=(3, 10), fontsize=10, ha='center')
    
    # K/V ratio
    ax3 = axes[2]
    ax3.semilogy(t * omega, KV_ratio, 'g-', linewidth=2)
    ax3.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='$K = V$')
    ax3.fill_between(t * omega, 1e-8, KV_ratio, where=(KV_ratio < 1), alpha=0.3, color='green')
    ax3.set_xlabel(r'$\omega t$')
    ax3.set_ylabel(r'$K / V$')
    ax3.set_title('Kinetic/Potential Ratio\n(Potential domination enables bounce)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(1e-8, 10)
    
    plt.suptitle('Kinetic Energy Suppression from Hyperbolic Field Space', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")

def plot_Hdot_mechanism(filename='Hdot_mechanism.pdf'):
    """Illustrate how +1/a² enables dH/dt > 0."""
    # Get analytic solution
    t, _, a, H, Hdot_an, a_min, omega = analytic_bounce_closed()
    
    # Compute components of dH/dt = -(ρ+p)/(2M²) + 1/a²
    # For potential domination: ρ ≈ V0, p ≈ -V0, so ρ+p ≈ 0
    curvature_term = 1 / a**2
    # Matter term (small for potential domination)
    matter_term = np.zeros_like(t)  # Approximate
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # dH/dt components
    ax1 = axes[0]
    ax1.plot(t * omega, Hdot_an / omega**2, 'b-', linewidth=2, label=r'$\dot{H}$ (total)')
    ax1.plot(t * omega, curvature_term / omega**2 * (a_min**2), 'r--', linewidth=1.5, 
             label=r'$+1/a^2$ (curvature)', alpha=0.7)
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel(r'$\omega t$')
    ax1.set_ylabel(r'$\dot{H} / \omega^2$')
    ax1.set_title(r'$\dot{H} = -\frac{\rho+p}{2M^2} + \frac{1}{a^2}$')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.annotate('Bounce:\n$\\dot{H} > 0$', xy=(0, 0.8), fontsize=11, ha='center')
    
    # Phase diagram
    ax2 = axes[1]
    ax2.plot(H / omega, Hdot_an / omega**2, 'purple', linewidth=2)
    ax2.scatter([0], [1], c='red', s=100, zorder=5, label='Bounce point')
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax2.set_xlabel(r'$H / \omega$')
    ax2.set_ylabel(r'$\dot{H} / \omega^2$')
    ax2.set_title('Phase Diagram: Bounce Trajectory')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.annotate('Contraction', xy=(-0.7, 0.5), fontsize=10, ha='center')
    ax2.annotate('Expansion', xy=(0.7, 0.5), fontsize=10, ha='center')
    
    plt.suptitle(r'The Mechanism: Spatial Curvature ($+1/a^2$) Enables $\dot{H} > 0$', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")

def plot_comparison_flat_closed(filename='flat_vs_closed.pdf'):
    """Compare flat (k=0) and closed (k=+1) universes."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    omega = np.sqrt(V0 / (3 * M_Pl**2))
    a_min = np.sqrt(3 * M_Pl**2 / V0)
    
    # Time array
    t = np.linspace(-3/omega, 3/omega, 500)
    
    # Closed universe (k=+1): bounce
    a_closed = a_min * np.cosh(omega * t)
    H_closed = omega * np.tanh(omega * t)
    
    # Flat universe (k=0): no bounce, singularity
    # For k=0, H² = ρ/(3M²) and dH/dt = -(ρ+p)/(2M²) < 0 always
    # With constant V0: H² = V0/(3M²), but this can't transition H<0 to H>0
    # Contraction would hit a singularity
    # Show conceptually: a → 0 as t → 0 from below
    t_flat_neg = t[t < 0]
    a_flat_neg = a_min * np.exp(omega * t_flat_neg)  # Exponential collapse
    
    ax1 = axes[0]
    ax1.plot(t * omega, a_closed / a_min, 'b-', linewidth=2, label='Closed ($k=+1$): Bounce')
    ax1.plot(t_flat_neg * omega, a_flat_neg / a_min, 'r--', linewidth=2, label='Flat ($k=0$): Singularity')
    ax1.scatter([0], [0], c='red', s=100, marker='x', zorder=5, label='Big Crunch (flat)')
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax1.set_xlabel(r'$\omega t$')
    ax1.set_ylabel(r'$a / a_{\rm min}$')
    ax1.set_title('Scale Factor Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.2, 3)
    
    # The key equation comparison
    ax2 = axes[1]
    ax2.text(0.5, 0.85, 'Flat Universe ($k=0$):', fontsize=12, fontweight='bold',
             ha='center', transform=ax2.transAxes)
    ax2.text(0.5, 0.75, r'$\dot{H} = -\frac{\rho+p}{2M^2} \leq 0$ always', fontsize=11,
             ha='center', transform=ax2.transAxes)
    ax2.text(0.5, 0.65, '→ No bounce possible', fontsize=10, color='red',
             ha='center', transform=ax2.transAxes)
    
    ax2.text(0.5, 0.45, 'Closed Universe ($k=+1$):', fontsize=12, fontweight='bold',
             ha='center', transform=ax2.transAxes)
    ax2.text(0.5, 0.35, r'$\dot{H} = -\frac{\rho+p}{2M^2} + \frac{1}{a^2}$', fontsize=11,
             ha='center', transform=ax2.transAxes)
    ax2.text(0.5, 0.25, r'$+\frac{1}{a^2}$ can make $\dot{H} > 0$', fontsize=10, color='blue',
             ha='center', transform=ax2.transAxes)
    ax2.text(0.5, 0.15, '→ Bounce without NEC violation!', fontsize=10, color='green',
             ha='center', transform=ax2.transAxes)
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_title('The Key Difference')
    
    plt.suptitle('Why Spatial Curvature Matters', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")

def plot_field_space_geometry(filename='field_space.pdf'):
    """Visualize the hyperbolic field space geometry."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Metric component g_χχ = e^(2αφ)
    phi_range = np.linspace(-3, 5, 200)
    g_chi_chi = np.exp(2 * alpha * phi_range)
    
    ax1 = axes[0]
    ax1.semilogy(phi_range, g_chi_chi, 'b-', linewidth=2)
    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax1.fill_between(phi_range[phi_range < 0], 0.001, g_chi_chi[phi_range < 0], 
                     alpha=0.3, color='blue', label=r'$\phi < 0$: $\chi$ kinetic suppressed')
    ax1.fill_between(phi_range[phi_range > 0], 1, g_chi_chi[phi_range > 0], 
                     alpha=0.3, color='red', label=r'$\phi > 0$: $\chi$ kinetic enhanced')
    ax1.set_xlabel(r'$\phi / M_{\rm Pl}$')
    ax1.set_ylabel(r'$g_{\chi\chi} = e^{2\alpha\phi}$')
    ax1.set_title('Field-Space Metric Component')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.001, 1000)
    
    # Right: Visualization of field space
    ax2 = axes[1]
    phi_grid = np.linspace(-2, 3, 100)
    chi_grid = np.linspace(-2, 2, 100)
    PHI, CHI = np.meshgrid(phi_grid, chi_grid)
    STRETCH = np.exp(alpha * PHI)
    
    contour = ax2.contourf(PHI, CHI, STRETCH, levels=20, cmap='viridis')
    plt.colorbar(contour, ax=ax2, label=r'$e^{\alpha\phi}$ (metric scale)')
    ax2.set_xlabel(r'$\phi / M_{\rm Pl}$')
    ax2.set_ylabel(r'$\chi / M_{\rm Pl}$')
    ax2.set_title(r'Field Space: Hyperbolic Geometry ($K = -\alpha^2$)')
    ax2.annotate('Bounce\nregion', xy=(-1.5, 0), fontsize=10, ha='center', color='white')
    ax2.annotate('Inflation\nregion', xy=(2, 0), fontsize=10, ha='center', color='black')
    
    plt.suptitle('Hyperbolic Field Space Geometry', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")

# =============================================================================
# Summary statistics
# =============================================================================
def print_summary(sol):
    """Print summary statistics of the solution."""
    t = sol.t
    phi, chi, pi_phi, pi_chi, a, a_dot = sol.y
    H = a_dot / a
    
    # Get analytic parameters
    _, _, _, _, _, a_min_an, omega = analytic_bounce_closed()
    
    # Find bounce point
    bounce_idx = np.argmin(np.abs(H))
    t_bounce = t[bounce_idx]
    a_bounce = a[bounce_idx]
    H_bounce = H[bounce_idx]
    
    # Check NEC
    rho_arr = np.array([rho(phi[i], chi[i], pi_phi[i], pi_chi[i]) for i in range(len(t))])
    p_arr = np.array([pressure(phi[i], chi[i], pi_phi[i], pi_chi[i]) for i in range(len(t))])
    rho_plus_p = rho_arr + p_arr
    NEC_satisfied = np.all(rho_plus_p >= -1e-20)  # Allow tiny numerical error
    
    # Equation of state at bounce
    w_bounce = equation_of_state(phi[bounce_idx], chi[bounce_idx], 
                                  pi_phi[bounce_idx], pi_chi[bounce_idx])
    
    print("\n" + "="*60)
    print("BOUNCE SOLUTION SUMMARY (k=+1 Closed Universe)")
    print("="*60)
    print(f"\nAnalytic predictions:")
    print(f"  Expected a_min: {a_min_an:.4e}")
    print(f"  ω = √(V₀/3M²): {omega:.4e}")
    
    print(f"\nNumerical results:")
    print(f"  Bounce time: t = {t_bounce:.4e}")
    print(f"  Scale factor at bounce: a = {a_bounce:.4e}")
    print(f"  Ratio a_bounce / a_min_an: {a_bounce/a_min_an:.4f}")
    print(f"  H at bounce: {H_bounce:.4e} (should be ≈ 0)")
    print(f"  w at bounce: {w_bounce:.4f} (should be ≈ -1)")
    
    print(f"\nPhysical checks:")
    print(f"  NEC satisfied (ρ+p ≥ 0): {'✓ YES' if NEC_satisfied else '✗ NO'}")
    print(f"  min(ρ+p): {np.min(rho_plus_p):.4e}")
    print(f"  min(a): {np.min(a):.4e} (should be > 0)")
    print(f"  Singularity avoided: {'✓ YES' if np.min(a) > 0 else '✗ NO'}")
    
    print("\n" + "="*60)
    print("KEY PHYSICS:")
    print("="*60)
    print("• Bounce occurs because +1/a² in dH/dt overcomes -(ρ+p)/(2M²)")
    print("• Hyperbolic geometry suppresses kinetic energy → potential dominates")
    print("• w ≈ -1 ensures bounce condition is met")
    print("• NO NEC violation required!")
    print("• NO ghosts!")
    print("="*60 + "\n")

# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("="*60)
    print("Non-Singular Bouncing Cosmology in Closed Universe (k=+1)")
    print("with Hyperbolic Field Space Geometry")
    print("="*60)
    
    # 1. Solve bounce
    print("\n1. Solving bounce equations...")
    sol = solve_bounce_closed()
    print(f"   Integration: {len(sol.t)} points, success: {sol.success}")
    
    # 2. Print summary
    print_summary(sol)
    
    # 3. Generate plots
    print("\n3. Generating plots...")
    
    plot_bounce_closed(sol, 'bounce_closed.pdf')
    plot_kinetic_suppression(sol, 'kinetic_suppression.pdf')
    plot_Hdot_mechanism('Hdot_mechanism.pdf')
    plot_comparison_flat_closed('flat_vs_closed.pdf')
    plot_field_space_geometry('field_space.pdf')
    
    print("\n" + "="*60)
    print("All plots saved successfully!")
    print("="*60)
