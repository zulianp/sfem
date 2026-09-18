"""Independent homogeneous three-dimensional Kelvin-Voigt creep oracles."""

import numpy as np
from scipy.integrate import solve_ivp


def smooth_ramp(time, ramp_time):
    phase = np.clip(np.asarray(time, dtype=np.float64) / float(ramp_time), 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * phase))


def mode_coefficients(mode, material):
    mu = float(material["mu"])
    if mode == "axial":
        stiffness = 6.0 * mu + float(material["lambda"])
        viscosity = 4.0 * float(material["eta_s"]) / 3.0 + float(material["eta_b"])
        return stiffness, viscosity, True
    if mode == "simple_shear":
        return 4.0 * mu, float(material["eta_s"]), False
    raise ValueError(f"unsupported creep mode: {mode}")


def solve_response(times, mode, material, traction, ramp_time):
    times = np.asarray(times, dtype=np.float64)
    stiffness, viscosity, finite_axial = mode_coefficients(mode, material)

    def rhs(time, state):
        load = float(traction) * float(smooth_ramp(time, ramp_time))
        response = state[0]
        rate = (load - stiffness * response) / viscosity
        return [(1.0 + response) * rate if finite_axial else rate]

    result = solve_ivp(
        rhs, (float(times[0]), float(times[-1])), [0.0], t_eval=times,
        rtol=2.0e-12, atol=2.0e-14, method="DOP853",
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.y[0]


def small_strain_exponential(times, stiffness, viscosity, traction):
    times = np.asarray(times, dtype=np.float64)
    return float(traction) / float(stiffness) * (1.0 - np.exp(-float(stiffness) * times / float(viscosity)))
