"""Closed-form energies and first Piola stresses for 3D affine modes."""

import numpy as np

from common.hyperelastic import response as material_response


def deformation_gradients():
    return {
        "uniaxial_extension": np.diag((1.125, 1.0, 1.0)),
        "simple_shear": np.asarray(((1.0, 0.125, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        "isotropic_dilation": np.diag((1.0625, 1.0625, 1.0625)),
        "combined_nonsymmetric": np.asarray(
            ((1.0625, 0.0625, -0.03125), (0.015625, 0.96875, 0.046875), (0.03125, -0.015625, 1.03125))
        ),
    }


def response(operator, deformation, material):
    return material_response(operator, deformation, material)
