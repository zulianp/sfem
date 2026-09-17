"""Independent radial equilibrium oracle for a finite-strain spherical shell."""

import numpy as np
from scipy.integrate import quad, solve_bvp
from scipy.optimize import brentq


def nominal_stresses(operator, radial_stretch, tangential_stretch, material):
    v = np.asarray(radial_stretch)
    t = np.asarray(tangential_stretch)
    jacobian = v * t ** 2
    log_j = np.log(jacobian)
    if operator == "GeneratedNeoHookeanOgden":
        mu = float(material["mu"])
        lmbda = float(material["lambda"])
        radial = mu * v + (lmbda * log_j - mu) / v
        tangential = mu * t + (lmbda * log_j - mu) / t
    elif operator == "GeneratedModifiedMooneyRivlin":
        c1 = float(material["c1"])
        c2 = float(material["c2"])
        kappa = float(material["kappa"])
        invariant_1 = v ** 2 + 2.0 * t ** 2
        invariant_2 = 2.0 * v ** 2 * t ** 2 + t ** 4
        j_m23 = jacobian ** (-2.0 / 3.0)
        j_m43 = jacobian ** (-4.0 / 3.0)
        radial = (
            2.0 * c1 * j_m23 * (v - invariant_1 / (3.0 * v))
            + 2.0 * c2 * j_m43 * (invariant_1 * v - v ** 3 - 2.0 * invariant_2 / (3.0 * v))
            + kappa * log_j / v
        )
        tangential = (
            2.0 * c1 * j_m23 * (t - invariant_1 / (3.0 * t))
            + 2.0 * c2 * j_m43 * (invariant_1 * t - t ** 3 - 2.0 * invariant_2 / (3.0 * t))
            + kappa * log_j / t
        )
    else:
        raise ValueError(f"unsupported radial constitutive oracle: {operator}")
    return radial, tangential


def linear_moduli(operator, material):
    if operator == "GeneratedNeoHookeanOgden":
        mu = float(material["mu"])
        bulk = float(material["lambda"]) + 2.0 * mu / 3.0
    elif operator == "GeneratedModifiedMooneyRivlin":
        mu = 2.0 * (float(material["c1"]) + float(material["c2"]))
        bulk = float(material["kappa"])
    else:
        raise ValueError(f"unsupported radial constitutive oracle: {operator}")
    return mu, bulk


def small_strain_guess(radius, inner_radius, outer_radius, pressure, mu, bulk):
    a3 = inner_radius ** 3
    b3 = outer_radius ** 3
    uniform = pressure * a3 / (b3 - a3)
    inverse_cubic = pressure * a3 * b3 / (b3 - a3)
    displacement = uniform * radius / (3.0 * bulk) + inverse_cubic / (4.0 * mu * radius ** 2)
    derivative = uniform / (3.0 * bulk) - inverse_cubic / (2.0 * mu * radius ** 3)
    return radius + displacement, 1.0 + derivative


def solve_radial_shell(operator, inner_radius, outer_radius, pressure, material, tolerance=1.0e-9):
    """Solve r(R) and dr/dR with follower-pressure nominal boundary tractions."""
    a = float(inner_radius)
    b = float(outer_radius)
    p = float(pressure)
    if not (0.0 < a < b and p >= 0.0):
        raise ValueError("invalid radius or pressure for radial shell oracle")
    mu, bulk = linear_moduli(operator, material)
    radius = np.linspace(a, b, 129)
    initial_radius, initial_stretch = small_strain_guess(radius, a, b, p, mu, bulk)
    complex_step = 1.0e-25

    def ode(reference_radius, state):
        current_radius, radial_stretch = state
        tangential_stretch = current_radius / reference_radius
        radial_nominal, tangential_nominal = nominal_stresses(
            operator, radial_stretch, tangential_stretch, material,
        )
        radial_derivative = np.imag(nominal_stresses(
            operator, radial_stretch + 1j * complex_step, tangential_stretch, material,
        )[0]) / complex_step
        tangential_derivative = np.imag(nominal_stresses(
            operator, radial_stretch, tangential_stretch + 1j * complex_step, material,
        )[0]) / complex_step
        stretch_derivative = (
            -2.0 * (radial_nominal - tangential_nominal) / reference_radius
            - tangential_derivative * (radial_stretch - tangential_stretch) / reference_radius
        ) / radial_derivative
        return np.vstack((radial_stretch, stretch_derivative))

    def boundary(inner_state, outer_state):
        inner_tangential = inner_state[0] / a
        outer_tangential = outer_state[0] / b
        inner_nominal = nominal_stresses(
            operator, inner_state[1], inner_tangential, material,
        )[0]
        outer_nominal = nominal_stresses(
            operator, outer_state[1], outer_tangential, material,
        )[0]
        return np.asarray((inner_nominal + p * inner_tangential ** 2, outer_nominal))

    solution = solve_bvp(
        ode, boundary, radius, np.vstack((initial_radius, initial_stretch)),
        tol=tolerance, max_nodes=4096,
    )
    if not solution.success:
        raise RuntimeError(f"radial oracle BVP did not converge: {solution.message}")
    sampled = solution.sol(np.linspace(a, b, 257))
    if np.min(sampled[1]) <= 0.0 or np.min(sampled[0]) <= 0.0:
        raise RuntimeError("radial oracle produced a non-positive stretch or radius")
    return solution


def radial_fields(solution, reference_radius, operator, material):
    reference_radius = np.asarray(reference_radius, dtype=np.float64)
    current_radius, radial_stretch = solution.sol(reference_radius)
    tangential_stretch = current_radius / reference_radius
    radial_nominal, tangential_nominal = nominal_stresses(
        operator, radial_stretch, tangential_stretch, material,
    )
    jacobian = radial_stretch * tangential_stretch ** 2
    radial_cauchy = radial_nominal / tangential_stretch ** 2
    hoop_cauchy = tangential_nominal / (radial_stretch * tangential_stretch)
    return np.stack(
        (current_radius, radial_stretch, tangential_stretch, jacobian, radial_cauchy, hoop_cauchy),
        axis=-1,
    )


def incompressible_pressure_from_shift(operator, inner_radius, outer_radius, shift, material):
    def integrand(reference_radius):
        current_radius = (reference_radius ** 3 + shift) ** (1.0 / 3.0)
        tangential = current_radius / reference_radius
        radial = 1.0 / tangential ** 2
        radial_nominal, tangential_nominal = nominal_stresses(
            operator, radial, tangential, material,
        )
        stress_difference = tangential_nominal * tangential - radial_nominal * radial
        return 2.0 * stress_difference * radial / current_radius

    return quad(integrand, inner_radius, outer_radius, epsabs=1.0e-12, epsrel=1.0e-12)[0]


def incompressible_shift(operator, inner_radius, outer_radius, pressure, material):
    if pressure == 0.0:
        return 0.0
    upper = inner_radius ** 3
    while incompressible_pressure_from_shift(operator, inner_radius, outer_radius, upper, material) < pressure:
        upper *= 2.0
        if upper > 1000.0 * outer_radius ** 3:
            raise RuntimeError("incompressible pressure branch could not be bracketed")
    return brentq(
        lambda shift: incompressible_pressure_from_shift(
            operator, inner_radius, outer_radius, shift, material,
        ) - pressure,
        0.0, upper, xtol=1.0e-13,
    )
