"""The free axial mode of a Prony-viscoelastic bar: frequency, decay and history, exact.

A bar of length L, its lateral displacements held, both ends fixed axially, starts at rest in its
reference shape with the velocity v0 sin(pi x/L). At small strain the material is the linearized
Mooney-Rivlin with a Prony series on its isochoric part,

    sigma = K eps + M0 (g_inf eps + sum_i h_i),   h_i' = -h_i/tau_i + g_i eps',
    M0 = 4/3 mu = 8/3 (C10 + C01)  (constrained uniaxial strain; K does not relax),

so the mode stays a sine and its amplitude q obeys

    rho q'' + k2 [K q + M0 (g_inf q + sum_i h_i)] = 0,   h_i' = -h_i/tau_i + g_i q',

a linear system of 2 + n states (the state-space form of a structure with exponential damping
kernels, Wagner and Adhikari, AIAA J. 41, 2003). Its eigenvalues are one damped pair
-delta +- i omega_d and n real relaxation rates.

k2 is pi^2/L^2 in the continuum. On a uniform mesh of linear elements along the bar (HEX8 with
the field constant across), the sampled sine is an exact eigenvector of the stiffness and the
consistent mass, with k2_h = 12/h^2 (1 - cos(pi h/L))/(4 + 2 cos(pi h/L)): the semi-discrete
oracle carries no spatial error, and the continuum one differs by O(h^2).

`newmark_amplitude` replays the fully discrete scheme on the mode: Newmark (beta, gamma) with
the Prony recursion H_i <- alpha_i H_i + b_i (q_{n+1} - q_n) of the material, solved for q_{n+1}.
"""

import numpy as np
from scipy.linalg import expm


def parse_series(values):
    if isinstance(values, str):
        values = values.split(",")
    result = np.asarray([float(value) for value in values], dtype=np.float64)
    if result.ndim != 1 or not len(result) or np.any(~np.isfinite(result)):
        raise ValueError("Prony series must contain finite values")
    return result


def check_series(g, tau):
    g, tau = parse_series(g), parse_series(tau)
    if g.shape != tau.shape or np.sum(g) >= 1.0 or np.any(g <= 0) or np.any(tau <= 0):
        raise ValueError("invalid Prony coefficients")
    return g, tau


def moduli(material):
    """M0 = 8/3 (C10 + C01), the isochoric constrained-uniaxial modulus, and K."""
    return 8.0 / 3.0 * (float(material["c10"]) + float(material["c01"])), float(material["bulk_modulus"])


def wave_number_squared(length, elements=None):
    """pi^2/L^2, or its value on `elements` uniform linear elements with the consistent mass."""
    k = np.pi / float(length)
    if elements is None:
        return k * k
    h = float(length) / int(elements)
    return 12.0 / h**2 * (1.0 - np.cos(k * h)) / (4.0 + 2.0 * np.cos(k * h))


def state_matrix(material, g, tau, k2):
    g, tau = check_series(g, tau)
    m0, bulk = moduli(material)
    rho = float(material["density"])
    n = len(g)
    A = np.zeros((2 + n, 2 + n))
    A[0, 1] = 1.0
    A[1, 0] = -k2 * (bulk + m0 * (1.0 - np.sum(g))) / rho
    A[1, 2:] = -k2 * m0 / rho
    A[2:, 1] = g
    A[2:, 2:] = np.diag(-1.0 / tau)
    return A


def modal_parameters(material, g, tau, k2):
    """The damped pair (decay delta, frequency omega_d) and the real relaxation rates."""
    eigenvalues = np.linalg.eigvals(state_matrix(material, g, tau, k2))
    oscillatory = eigenvalues[np.abs(eigenvalues.imag) > 1.0e-12 * np.max(np.abs(eigenvalues))]
    if len(oscillatory) != 2:
        raise ValueError("the mode is not underdamped")
    pair = oscillatory[oscillatory.imag > 0][0]
    rates = np.sort(-eigenvalues[np.abs(eigenvalues.imag) <= 1.0e-12 * np.max(np.abs(eigenvalues))].real)
    return {"decay": float(-pair.real), "omega_d": float(pair.imag), "relaxation_rates": rates.tolist()}


def amplitude(times, material, g, tau, k2, v0):
    """q(t) from rest with q'(0) = v0: x(t) = expm(A t) x0."""
    A = state_matrix(material, g, tau, k2)
    x0 = np.zeros(A.shape[0])
    x0[1] = float(v0)
    times = np.asarray(times, dtype=np.float64)
    # Equal steps: one propagator, applied repeatedly, keeps the cost linear in the samples.
    dt = np.diff(times)
    if len(times) > 1 and np.allclose(dt, dt[0], rtol=1.0e-12, atol=0.0):
        step = expm(A * dt[0])
        x = expm(A * times[0]) @ x0
        out = np.empty(len(times))
        for i in range(len(times)):
            out[i] = x[0]
            x = step @ x
        return out
    return np.asarray([(expm(A * t) @ x0)[0] for t in times])


def step_weights(g, tau, dt, integrator="linear"):
    g, tau = check_series(g, tau)
    x = float(dt) / tau
    alpha = np.exp(-x)
    if integrator == "linear":
        return alpha, g * (-np.expm1(-x) / x)
    if integrator == "midpoint":
        return alpha, g * np.exp(-x / 2.0)
    raise ValueError(f"unknown integrator {integrator}")


def newmark_amplitude(steps, dt, material, g, tau, k2, v0, integrator="linear", beta=0.25, gamma=0.5):
    """The fully discrete modal amplitude: Newmark with the material's Prony recursion."""
    g, tau = check_series(g, tau)
    m0, bulk = moduli(material)
    rho = float(material["density"])
    alpha, b = step_weights(g, tau, dt, integrator)
    g_inf = 1.0 - np.sum(g)
    q, v, a = 0.0, float(v0), 0.0
    H = np.zeros_like(g)
    out = np.empty(int(steps) + 1)
    out[0] = q
    for n in range(int(steps)):
        # rho a1 + k2 [K q1 + M0 (g_inf q1 + sum(alpha H + b (q1 - q)))] = 0, with
        # q1 = q + dt v + dt^2 ((1/2 - beta) a + beta a1)  =>  a1 = (q1 - predictor)/(beta dt^2).
        predictor = q + dt * v + dt * dt * (0.5 - beta) * a
        stiffness = k2 * (bulk + m0 * (g_inf + np.sum(b)))
        history = k2 * m0 * np.sum(alpha * H - b * q)
        q1 = (rho * predictor / (beta * dt * dt) - history) / (rho / (beta * dt * dt) + stiffness)
        a1 = (q1 - predictor) / (beta * dt * dt)
        v = v + dt * ((1.0 - gamma) * a + gamma * a1)
        H = alpha * H + b * (q1 - q)
        q, a = q1, a1
        out[n + 1] = q
    return out
