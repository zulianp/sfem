"""Closed-form affine oracle for three-dimensional isotropic linear elasticity."""

import numpy as np


def deformation_gradients():
    return {
        "deviatoric_extension": np.diag((1.02, 0.99, 0.99)),
        "simple_shear": np.asarray(((1.0, 0.03, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        "triaxial_dilation": np.diag((1.012, 1.012, 1.012)),
    }


def response(deformation, material):
    displacement_gradient = np.asarray(deformation, dtype=np.float64) - np.eye(3)
    strain = 0.5 * (displacement_gradient + displacement_gradient.T)
    mu = float(material["mu"])
    lmbda = float(material["lambda"])
    stress = 2.0 * mu * strain + lmbda * np.trace(strain) * np.eye(3)
    energy = 0.5 * np.sum(stress * strain)
    return float(energy), stress
