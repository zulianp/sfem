"""Independent homogeneous-deformation oracles for hyperelastic materials."""

import numpy as np


def _validated_deformation(deformation):
    deformation = np.asarray(deformation, dtype=np.float64)
    if deformation.ndim != 2 or deformation.shape[0] != deformation.shape[1]:
        raise ValueError("deformation gradient must be a square matrix")
    jacobian = float(np.linalg.det(deformation))
    if not np.isfinite(jacobian) or jacobian <= 0:
        raise ValueError(f"deformation gradient must have positive determinant, got {jacobian}")
    return deformation, jacobian


def neohookean_ogden(deformation, material):
    deformation, jacobian = _validated_deformation(deformation)
    mu = float(material["mu"])
    lmbda = float(material["lambda"])
    log_j = np.log(jacobian)
    inverse_transpose = np.linalg.inv(deformation).T
    dimension = deformation.shape[0]
    energy = 0.5 * mu * (np.sum(deformation * deformation) - dimension) - mu * log_j
    energy += 0.5 * lmbda * log_j * log_j
    first_piola = mu * deformation + (lmbda * log_j - mu) * inverse_transpose
    return float(energy), first_piola


def saint_venant_kirchhoff(deformation, material):
    deformation, _ = _validated_deformation(deformation)
    mu = float(material["mu"])
    lmbda = float(material["lambda"])
    green_strain = 0.5 * (deformation.T @ deformation - np.eye(deformation.shape[0]))
    trace = float(np.trace(green_strain))
    second_piola = lmbda * trace * np.eye(deformation.shape[0]) + 2.0 * mu * green_strain
    energy = mu * np.sum(green_strain * green_strain) + 0.5 * lmbda * trace * trace
    return float(energy), deformation @ second_piola


def modified_mooney_rivlin(deformation, material):
    deformation, _ = _validated_deformation(deformation)
    dimension = deformation.shape[0]
    if dimension == 2:
        embedded = np.eye(3)
        embedded[:2, :2] = deformation
    elif dimension == 3:
        embedded = deformation
    else:
        raise ValueError("modified Mooney-Rivlin is supported only in two or three dimensions")

    jacobian = float(np.linalg.det(embedded))
    inverse_transpose = np.linalg.inv(embedded).T
    right_cauchy_green = embedded.T @ embedded
    invariant_1 = float(np.trace(right_cauchy_green))
    invariant_2 = 0.5 * (invariant_1 * invariant_1 - float(np.sum(right_cauchy_green * right_cauchy_green)))
    c1 = float(material["c1"])
    c2 = float(material["c2"])
    kappa = float(material["kappa"])
    j_m23 = jacobian ** (-2.0 / 3.0)
    j_m43 = jacobian ** (-4.0 / 3.0)
    log_j = np.log(jacobian)

    energy = c1 * (j_m23 * invariant_1 - 3.0) + c2 * (j_m43 * invariant_2 - 3.0)
    energy += 0.5 * kappa * log_j * log_j
    first_piola = 2.0 * c1 * j_m23 * (embedded - invariant_1 * inverse_transpose / 3.0)
    first_piola += 2.0 * c2 * j_m43 * (
        invariant_1 * embedded
        - embedded @ right_cauchy_green
        - 2.0 * invariant_2 * inverse_transpose / 3.0
    )
    first_piola += kappa * log_j * inverse_transpose
    return float(energy), first_piola[:dimension, :dimension]


RESPONSES = {
    "GeneratedNeoHookeanOgden": neohookean_ogden,
    "NeoHookeanOgden": neohookean_ogden,
    "GeneratedModifiedMooneyRivlin": modified_mooney_rivlin,
    "GeneratedSaintVenantKirchhoff": saint_venant_kirchhoff,
}


def response(operator, deformation, material):
    try:
        oracle = RESPONSES[operator]
    except KeyError as error:
        raise ValueError(f"no hyperelastic oracle for operator {operator!r}") from error
    return oracle(deformation, material)
