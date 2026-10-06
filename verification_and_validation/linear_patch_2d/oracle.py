"""Closed-form plane-strain linear-elastic oracle for affine modes."""

import numpy as np


MODES = {
    "deviatoric_extension": np.asarray(((1.02, 0.00), (0.00, 0.98))),
    "simple_shear": np.asarray(((1.00, 0.03), (0.00, 1.00))),
    "mixed_volumetric": np.asarray(((1.015, 0.010), (-0.006, 1.025))),
}


def deformation_gradients():
    return {name: value.copy() for name, value in MODES.items()}


def response(deformation, material):
    deformation = np.asarray(deformation, dtype=np.float64)
    gradient = deformation - np.eye(2)
    strain = 0.5 * (gradient + gradient.T)
    mu = float(material["mu"])
    lmbda = float(material["lambda"])
    stress = 2.0 * mu * strain + lmbda * np.trace(strain) * np.eye(2)
    energy_density = 0.5 * float(np.sum(stress * strain))
    return energy_density, stress
