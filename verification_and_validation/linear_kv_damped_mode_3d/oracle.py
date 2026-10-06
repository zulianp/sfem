"""Separated underdamped axial mode for the legacy Kelvin-Voigt operator."""

import numpy as np


def modal_parameters(material, length):
    wave_number = np.pi / float(length)
    axial_modulus = float(material["bulk_modulus"]) + 2.0 * float(material["shear_stiffness"]) / 3.0
    viscous_modulus = 2.0 * float(material["damping"]) / 3.0
    density = float(material["density"])
    omega_0 = wave_number * np.sqrt(axial_modulus / density)
    decay = viscous_modulus * wave_number ** 2 / (2.0 * density)
    omega_d = np.sqrt(omega_0 ** 2 - decay ** 2)
    return {"wave_number": wave_number, "axial_modulus": axial_modulus,
            "viscous_modulus": viscous_modulus, "omega_0": omega_0,
            "decay": decay, "omega_d": omega_d}


def amplitude(times, q0, v0, decay, omega_d):
    times = np.asarray(times, dtype=np.float64)
    return np.exp(-float(decay) * times) * (
        float(q0) * np.cos(float(omega_d) * times)
        + (float(v0) + float(decay) * float(q0)) / float(omega_d)
        * np.sin(float(omega_d) * times)
    )
