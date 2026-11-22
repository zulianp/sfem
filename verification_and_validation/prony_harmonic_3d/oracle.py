"""Independent oracles for a Prony-series material driven harmonically (a virtual DMA).

The deformation is homogeneous constrained uniaxial strain, F = diag(lambda, 1, 1) with
lambda = 1 + eps(t), eps(t) = eps0 sin(omega t). The law is Mooney-Rivlin whose isochoric part
relaxes by a Prony series in Simo's form on the second Piola-Kirchhoff stress:

    S = g_inf S_iso(C) + sum_i H_i + S_vol(C),
    H_i <- alpha_i H_i + b_i (S_iso^{n+1} - S_iso^n),   alpha_i = exp(-dt/tau_i),

with b_i = g_i (1 - alpha_i)/(dt/tau_i) (linear interpolation of the stress over the step,
Taylor, Pister and Goudreau 1970) or b_i = g_i exp(-dt/(2 tau_i)) (midpoint rule, Simo 1987).

Three oracles, from exact to continuum:

- `algorithmic_reaction`: the recursion above replayed on the prescribed stretch history. For a
  homogeneous deformation it is what the discrete problem computes, at any amplitude, so the
  simulated reaction must match it to the solver's tolerance.
- `discrete_complex_modulus`: the recursion is a linear filter at small strain; its transfer
  function at frequency omega and step dt is the algorithm's own complex modulus.
- `complex_modulus` and `continuum_linear_reaction`: the generalized Maxwell model in continuous
  time, G*(omega)/G0 = g_inf + sum_i g_i i omega tau_i/(1 + i omega tau_i), and the exact response
  to eps0 sin(omega t) applied from rest, transient included. The discrete results converge to
  these as dt^2.
"""

import numpy as np


def parse_series(values):
    if isinstance(values, str):
        values = values.split(",")
    result = np.asarray([float(value) for value in values], dtype=np.float64)
    if result.ndim != 1 or not len(result) or np.any(~np.isfinite(result)):
        raise ValueError("Prony series must contain finite values")
    return result


def check_series(g, tau):
    g = parse_series(g)
    tau = parse_series(tau)
    if g.shape != tau.shape or np.sum(g) >= 1.0 or np.any(g <= 0) or np.any(tau <= 0):
        raise ValueError("invalid Prony coefficients")
    return g, tau


def step_weights(g, tau, dt, integrator="linear"):
    """alpha_i and b_i of one step of length dt."""
    g, tau = check_series(g, tau)
    x = float(dt) / tau
    alpha = np.exp(-x)
    if integrator == "linear":
        b = g * (-np.expm1(-x) / x)
    elif integrator == "midpoint":
        b = g * np.exp(-x / 2.0)
    else:
        raise ValueError(f"unknown integrator {integrator}")
    return alpha, b


def isochoric_axial_piola(stretch, c10, c01):
    """P_11 of the isochoric Mooney-Rivlin energy for F = diag(stretch, 1, 1)."""
    stretch = np.asarray(stretch, dtype=np.float64)
    d1 = 4.0 / 3.0 * (stretch ** (1.0 / 3.0) - stretch ** (-5.0 / 3.0))
    d2 = 4.0 / 3.0 * (stretch ** (-1.0 / 3.0) - stretch ** (-7.0 / 3.0))
    return float(c10) * d1 + float(c01) * d2


def axial_modulus(c10, c01, bulk_modulus=0.0):
    """dP_11/deps at the reference: the isochoric part 8/3 (C10 + C01) = 4/3 mu, and K."""
    return 8.0 / 3.0 * (float(c10) + float(c01)), float(bulk_modulus)


def algorithmic_reaction(strain, g, tau, dt, c10, c01, bulk_modulus=0.0, integrator="linear"):
    """P_11 at each state of `strain` (strain[0] the rest state, then one value per step)."""
    g, tau = check_series(g, tau)
    alpha, b = step_weights(g, tau, dt, integrator)
    g_inf = 1.0 - np.sum(g)
    stretch = 1.0 + np.asarray(strain, dtype=np.float64)
    s_iso = isochoric_axial_piola(stretch, c10, c01) / stretch
    history = np.zeros_like(g)
    reaction = np.empty_like(stretch)
    reaction[0] = stretch[0] * (g_inf * s_iso[0]) + float(bulk_modulus) * (stretch[0] - 1.0)
    for n in range(1, len(stretch)):
        history = alpha * history + b * (s_iso[n] - s_iso[n - 1])
        reaction[n] = stretch[n] * (g_inf * s_iso[n] + np.sum(history)) + float(bulk_modulus) * (stretch[n] - 1.0)
    return reaction


def complex_modulus(omega, g, tau):
    """G*(omega)/G0 of the continuous generalized Maxwell model."""
    g, tau = check_series(g, tau)
    iwt = 1j * float(omega) * tau
    return 1.0 - np.sum(g) + np.sum(g * iwt / (1.0 + iwt))


def discrete_complex_modulus(omega, g, tau, dt, integrator="linear"):
    """The recursion's transfer function at frequency omega: for a sampled S_n = Re(S e^{i omega t_n}),
    H_i = b_i (1 - z^-1)/(1 - alpha_i z^-1) S with z = e^{i omega dt}."""
    g, tau = check_series(g, tau)
    alpha, b = step_weights(g, tau, dt, integrator)
    zinv = np.exp(-1j * float(omega) * float(dt))
    return 1.0 - np.sum(g) + np.sum(b * (1.0 - zinv) / (1.0 - alpha * zinv))


def continuum_linear_reaction(times, eps0, omega, g, tau, c10, c01, bulk_modulus=0.0):
    """The small-strain response to eps0 sin(omega t) applied from rest at t = 0, transient included:
    M0 [g_inf eps + sum_i g_i eps0 omega tau_i/(1 + omega^2 tau_i^2)
        (cos omega t + omega tau_i sin omega t - exp(-t/tau_i))] + K eps."""
    g, tau = check_series(g, tau)
    t = np.asarray(times, dtype=np.float64)[:, None]
    omega = float(omega)
    m_iso, m_vol = axial_modulus(c10, c01, bulk_modulus)
    eps = float(eps0) * np.sin(omega * t[:, 0])
    wt = omega * tau[None, :]
    hereditary = g[None, :] * float(eps0) * wt / (1.0 + wt**2) * (
        np.cos(omega * t) + wt * np.sin(omega * t) - np.exp(-t / tau[None, :]))
    return m_iso * ((1.0 - np.sum(g)) * eps + np.sum(hereditary, axis=1)) + m_vol * eps


def harmonic_fit(times, reaction, eps0, omega, tau, m_iso, m_vol=0.0):
    """G'/G0 and G''/G0 from a reaction history that started from rest: the linear response is
    exactly M0 eps0 (G' sin + G'' cos) plus one decaying exponential per relaxation time (the
    transient) plus K eps, so a linear least-squares fit over the whole history needs no steady state."""
    t = np.asarray(times, dtype=np.float64)
    tau = parse_series(tau)
    basis = [np.sin(omega * t), np.cos(omega * t)] + [np.exp(-t / ti) for ti in tau]
    A = np.column_stack(basis)
    rhs = np.asarray(reaction, dtype=np.float64) - float(m_vol) * float(eps0) * np.sin(omega * t)
    coefficients, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    scale = float(m_iso) * float(eps0)
    return coefficients[0] / scale, coefficients[1] / scale


def dissipation_per_cycle(eps0, omega, g, tau, c10, c01):
    """The energy dissipated per unit volume over one steady cycle, pi eps0^2 M0 G''/G0."""
    m_iso, _ = axial_modulus(c10, c01)
    return np.pi * float(eps0) ** 2 * m_iso * complex_modulus(omega, g, tau).imag
