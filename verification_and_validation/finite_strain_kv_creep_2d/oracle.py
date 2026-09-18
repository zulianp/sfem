"""Independent homogeneous plane-strain Kelvin-Voigt creep oracle."""

import numpy as np
from scipy.integrate import solve_ivp


def smooth_ramp(time, ramp_time):
    time = np.asarray(time, dtype=np.float64)
    phase = np.clip(time / float(ramp_time), 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * phase))


def elastic_axial_piola(stretch, mu, lmbda):
    return (6.0 * float(mu) + float(lmbda)) * (np.asarray(stretch) - 1.0)


def axial_viscosity(eta_s, eta_b):
    return float(eta_s) + float(eta_b)


def solve_axial(times, material, traction, ramp_time):
    times = np.asarray(times, dtype=np.float64)
    stiffness = 6.0 * float(material["mu"]) + float(material["lambda"])
    viscosity = axial_viscosity(material["eta_s"], material["eta_b"])

    def rhs(time, state):
        load = float(traction) * float(smooth_ramp(time, ramp_time))
        stretch = state[0]
        return [stretch * (load - stiffness * (stretch - 1.0)) / viscosity]

    result = solve_ivp(
        rhs,
        (float(times[0]), float(times[-1])),
        [1.0],
        t_eval=times,
        rtol=2.0e-12,
        atol=2.0e-14,
        method="DOP853",
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.y[0]


def small_strain_exponential(times, stiffness, viscosity, traction):
    times = np.asarray(times, dtype=np.float64)
    return 1.0 + float(traction) / float(stiffness) * (
        1.0 - np.exp(-float(stiffness) * times / float(viscosity))
    )
