"""Independent finite-element kinematics and integral helpers for case oracles."""

from dataclasses import dataclass
import itertools
import math

import numpy as np

from .sets import surface_geometry


@dataclass(frozen=True)
class ElementKinematics:
    locations: np.ndarray
    weights: np.ndarray
    displacement_gradient: np.ndarray
    deformation_gradient: np.ndarray
    deformation_jacobian: np.ndarray
    small_strain: np.ndarray


def _quadrature(element_type):
    if element_type == "TRI3":
        values = np.asarray(((1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),))
        gradients = np.asarray((((-1.0, -1.0), (1.0, 0.0), (0.0, 1.0)),))
        return values, gradients, np.asarray((0.5,))
    if element_type == "TET4":
        values = np.asarray(((0.25, 0.25, 0.25, 0.25),))
        gradients = np.asarray((((-1.0, -1.0, -1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                                 (0.0, 0.0, 1.0)),))
        return values, gradients, np.asarray((1.0 / 6.0,))

    gauss = 1.0 / math.sqrt(3.0)
    if element_type == "QUAD4":
        signs = np.asarray(((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)))
        quadrature_points = itertools.product((-gauss, gauss), repeat=2)
    elif element_type == "HEX8":
        signs = np.asarray(
            (
                (-1.0, -1.0, -1.0),
                (1.0, -1.0, -1.0),
                (1.0, 1.0, -1.0),
                (-1.0, 1.0, -1.0),
                (-1.0, -1.0, 1.0),
                (1.0, -1.0, 1.0),
                (1.0, 1.0, 1.0),
                (-1.0, 1.0, 1.0),
            )
        )
        quadrature_points = itertools.product((-gauss, gauss), repeat=3)
    else:
        raise ValueError(f"unsupported element type for kinematics: {element_type}")

    values = []
    gradients = []
    for point in quadrature_points:
        point = np.asarray(point)
        factors = 1.0 + signs * point
        values.append(np.prod(factors, axis=1) / (2.0 ** len(point)))
        gradient = np.empty_like(signs)
        for axis in range(len(point)):
            other = np.delete(factors, axis, axis=1)
            gradient[:, axis] = signs[:, axis] * np.prod(other, axis=1) / (2.0 ** len(point))
        gradients.append(gradient)
    return np.asarray(values), np.asarray(gradients), np.ones(len(values))


def element_kinematics(mesh, displacement):
    displacement = np.asarray(displacement, dtype=np.float64)
    if displacement.shape != mesh.points.shape or not np.all(np.isfinite(displacement)):
        raise ValueError(f"displacement must be a finite array with shape {mesh.points.shape}")

    shape_values, reference_gradients, reference_weights = _quadrature(mesh.element_type)
    coordinates = mesh.points[mesh.elements]
    local_displacement = displacement[mesh.elements]
    n_elements = mesh.n_elements
    n_quadrature = len(reference_weights)
    dimension = mesh.dimension
    locations = np.empty((n_elements, n_quadrature, dimension))
    weights = np.empty((n_elements, n_quadrature))
    gradients = np.empty((n_elements, n_quadrature, dimension, dimension))

    for q in range(n_quadrature):
        reference_gradient = reference_gradients[q]
        geometry_jacobian = np.einsum("eni,nj->eij", coordinates, reference_gradient)
        determinants = np.linalg.det(geometry_jacobian)
        if np.any(~np.isfinite(determinants)) or np.any(determinants <= np.finfo(np.float64).eps):
            raise ValueError(f"mesh has a non-positive geometry Jacobian at quadrature point {q}")
        physical_shape_gradients = np.einsum("ni,eij->enj", reference_gradient, np.linalg.inv(geometry_jacobian))
        gradients[:, q] = np.einsum("eni,enj->eij", local_displacement, physical_shape_gradients)
        locations[:, q] = np.einsum("n,end->ed", shape_values[q], coordinates)
        weights[:, q] = reference_weights[q] * determinants

    deformation = deformation_gradient(gradients)
    jacobian = deformation_jacobian(deformation)
    return ElementKinematics(
        locations=locations,
        weights=weights,
        displacement_gradient=gradients,
        deformation_gradient=deformation,
        deformation_jacobian=jacobian,
        small_strain=small_strain(gradients),
    )


def displacement_gradient(mesh, displacement):
    return element_kinematics(mesh, displacement).displacement_gradient


def deformation_gradient(displacement_gradient_values):
    gradient = np.asarray(displacement_gradient_values, dtype=np.float64)
    if gradient.ndim < 2 or gradient.shape[-1] != gradient.shape[-2]:
        raise ValueError("displacement gradient must end in square matrix dimensions")
    return gradient + np.eye(gradient.shape[-1])


def deformation_jacobian(deformation_gradient_values):
    deformation = np.asarray(deformation_gradient_values, dtype=np.float64)
    if deformation.ndim < 2 or deformation.shape[-1] != deformation.shape[-2]:
        raise ValueError("deformation gradient must end in square matrix dimensions")
    jacobian = np.linalg.det(deformation)
    if np.any(~np.isfinite(jacobian)):
        raise ValueError("deformation Jacobian is non-finite")
    return jacobian


def require_positive_jacobian(deformation_gradient_values, minimum=0.0):
    jacobian = deformation_jacobian(deformation_gradient_values)
    if np.any(jacobian <= minimum):
        raise ValueError(f"non-positive deformation Jacobian: minimum={np.min(jacobian):.6g}")
    return jacobian


def small_strain(displacement_gradient_values):
    gradient = np.asarray(displacement_gradient_values, dtype=np.float64)
    if gradient.ndim < 2 or gradient.shape[-1] != gradient.shape[-2]:
        raise ValueError("displacement gradient must end in square matrix dimensions")
    return 0.5 * (gradient + np.swapaxes(gradient, -1, -2))


def cauchy_from_first_piola(first_piola, deformation_gradient_values):
    first_piola = np.asarray(first_piola, dtype=np.float64)
    deformation = np.asarray(deformation_gradient_values, dtype=np.float64)
    jacobian = require_positive_jacobian(deformation)
    return np.matmul(first_piola, np.swapaxes(deformation, -1, -2)) / jacobian[..., None, None]


def first_piola_from_cauchy(cauchy, deformation_gradient_values):
    cauchy = np.asarray(cauchy, dtype=np.float64)
    deformation = np.asarray(deformation_gradient_values, dtype=np.float64)
    jacobian = require_positive_jacobian(deformation)
    inverse_transpose = np.swapaxes(np.linalg.inv(deformation), -1, -2)
    return jacobian[..., None, None] * np.matmul(cauchy, inverse_transpose)


def small_strain_energy_density(cauchy, strain):
    cauchy = np.asarray(cauchy, dtype=np.float64)
    strain = np.asarray(strain, dtype=np.float64)
    return 0.5 * np.sum(cauchy * strain, axis=(-2, -1))


def integrate_volume(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2 or values.shape[:2] != weights.shape:
        raise ValueError("volume values must start with the element/quadrature shape of weights")
    if np.any(weights <= 0) or not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights)):
        raise ValueError("volume values must be finite and quadrature weights positive")
    return np.einsum("eq,eq...->...", weights, values)


def integrate_strain_energy_density(energy_density, weights):
    result = integrate_volume(energy_density, weights)
    if np.ndim(result) != 0:
        raise ValueError("strain energy density must be scalar at every quadrature point")
    return float(result)


def integrate_boundary_traction(mesh, sideset, traction):
    geometry = surface_geometry(mesh, sideset)
    values = traction(geometry.centroids, geometry.normals) if callable(traction) else traction
    values = np.asarray(values, dtype=np.float64)
    if values.shape == (mesh.dimension,):
        values = np.broadcast_to(values, (sideset.size, mesh.dimension))
    if values.shape != (sideset.size, mesh.dimension) or not np.all(np.isfinite(values)):
        raise ValueError("traction must provide one finite vector per side")
    return np.sum(values * geometry.measures[:, None], axis=0)


def boundary_resultant_from_stress(mesh, sideset, stress):
    geometry = surface_geometry(mesh, sideset)
    stress = np.asarray(stress, dtype=np.float64)
    if stress.shape == (mesh.dimension, mesh.dimension):
        stress = np.broadcast_to(stress, (sideset.size, mesh.dimension, mesh.dimension))
    if stress.shape != (sideset.size, mesh.dimension, mesh.dimension) or not np.all(np.isfinite(stress)):
        raise ValueError("stress must provide one finite tensor per side")
    tractions_integrated = np.einsum("sij,sj->si", stress, geometry.area_vectors)
    return np.sum(tractions_integrated, axis=0)


def pressure_resultant(mesh, sideset, pressure):
    geometry = surface_geometry(mesh, sideset)
    pressure = np.asarray(pressure, dtype=np.float64)
    if pressure.ndim == 0:
        pressure = np.full(sideset.size, pressure)
    if pressure.shape != (sideset.size,) or not np.all(np.isfinite(pressure)):
        raise ValueError("pressure must provide one finite scalar per side")
    return -np.sum(pressure[:, None] * geometry.area_vectors, axis=0)
