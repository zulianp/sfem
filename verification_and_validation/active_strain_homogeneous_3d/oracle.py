"""Independent multiplicative-decomposition oracle for active strain."""

import numpy as np


def active_gradient():
    return np.diag((0.9, 1.05, 1.0 / (0.9 * 1.05)))


def deformation_gradients():
    return {"undeformed": np.eye(3), "compatible": active_gradient()}


def _neohookean(deformation, material):
    mu = float(material["mu"])
    lmbda = float(material["lambda"])
    jacobian = float(np.linalg.det(deformation))
    log_j = np.log(jacobian)
    inverse_transpose = np.linalg.inv(deformation).T
    energy = 0.5 * mu * (np.sum(deformation * deformation) - 3.0) - mu * log_j
    energy += 0.5 * lmbda * log_j * log_j
    stress = mu * deformation + (lmbda * log_j - mu) * inverse_transpose
    return float(energy), stress


def _smith_mooney_rivlin(deformation, material):
    mu = float(material["mu"])
    bulk = float(material["lambda"])
    jacobian = float(np.linalg.det(deformation))
    invariant_1 = float(np.sum(deformation * deformation))
    shift = jacobian - 1.0 - 3.0 * mu / (4.0 * bulk)
    energy = 0.5 * mu * (invariant_1 - 3.0)
    energy += 0.5 * bulk * shift * shift - 0.5 * mu * np.log(invariant_1 + 1.0)
    stress = mu * (1.0 - 1.0 / (invariant_1 + 1.0)) * deformation
    stress += bulk * shift * jacobian * np.linalg.inv(deformation).T
    return float(energy), stress


def response(operator, total_deformation, active_deformation, material):
    total_deformation = np.asarray(total_deformation, dtype=np.float64)
    active_deformation = np.asarray(active_deformation, dtype=np.float64)
    active_jacobian = float(np.linalg.det(active_deformation))
    elastic_deformation = total_deformation @ np.linalg.inv(active_deformation)
    if operator == "NeoHookeanOgdenActiveStrainPacked":
        energy, elastic_stress = _neohookean(elastic_deformation, material)
    elif operator == "MooneyRivlinActiveStrainPacked":
        energy, elastic_stress = _smith_mooney_rivlin(elastic_deformation, material)
    else:
        raise ValueError(f"unsupported active-strain operator {operator!r}")
    total_stress = active_jacobian * elastic_stress @ np.linalg.inv(active_deformation).T
    return active_jacobian * energy, total_stress
