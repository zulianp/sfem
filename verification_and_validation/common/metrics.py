"""Error norms with explicit near-zero normalization policy."""

import numpy as np


def _finite_array(value, name):
    array = np.asarray(value, dtype=np.float64)
    if not array.size or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def _floor(value):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("absolute_floor must be finite and positive")
    return value


def absolute_l2_error(observed, expected):
    observed = _finite_array(observed, "observed")
    expected = _finite_array(expected, "expected")
    if observed.shape != expected.shape:
        raise ValueError("observed and expected values must have the same shape")
    return float(np.linalg.norm((observed - expected).ravel()))


def relative_l2_error(observed, expected, absolute_floor=1.0e-14):
    expected = _finite_array(expected, "expected")
    numerator = absolute_l2_error(observed, expected)
    denominator = max(float(np.linalg.norm(expected.ravel())), _floor(absolute_floor))
    return numerator / denominator


def max_abs_error(observed, expected):
    observed = _finite_array(observed, "observed")
    expected = _finite_array(expected, "expected")
    if observed.shape != expected.shape:
        raise ValueError("observed and expected values must have the same shape")
    return float(np.max(np.abs(observed - expected)))


def weighted_l2_error(observed, expected, weights):
    observed = _finite_array(observed, "observed")
    expected = _finite_array(expected, "expected")
    weights = _finite_array(weights, "weights")
    if observed.shape != expected.shape or observed.ndim < 1:
        raise ValueError("observed and expected values must have the same non-scalar shape")
    if weights.shape != (observed.shape[0],) or np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("weights must be non-negative, nonzero, and match the first value dimension")
    squared = (observed - expected) ** 2
    if squared.ndim > 1:
        squared = np.sum(squared, axis=tuple(range(1, squared.ndim)))
    return float(np.sqrt(np.dot(weights, squared)))


def weighted_relative_l2_error(observed, expected, weights, absolute_floor=1.0e-14):
    expected = _finite_array(expected, "expected")
    zero = np.zeros_like(expected)
    numerator = weighted_l2_error(observed, expected, weights)
    denominator = max(weighted_l2_error(expected, zero, weights), _floor(absolute_floor))
    return numerator / denominator


def trapezoidal_weights(coordinates):
    coordinates = _finite_array(coordinates, "coordinates")
    if coordinates.ndim != 1 or len(coordinates) < 2 or np.any(np.diff(coordinates) <= 0):
        raise ValueError("coordinates must be a strictly increasing vector with at least two entries")
    spacing = np.diff(coordinates)
    weights = np.empty_like(coordinates)
    weights[0] = 0.5 * spacing[0]
    weights[-1] = 0.5 * spacing[-1]
    if len(coordinates) > 2:
        weights[1:-1] = 0.5 * (spacing[:-1] + spacing[1:])
    return weights


def curve_errors(observed_coordinates, observed_values, reference_coordinates, reference_values,
                 absolute_floor=1.0e-14):
    observed_coordinates = _finite_array(observed_coordinates, "observed_coordinates")
    reference_coordinates = _finite_array(reference_coordinates, "reference_coordinates")
    observed_values = _finite_array(observed_values, "observed_values")
    reference_values = _finite_array(reference_values, "reference_values")
    if observed_coordinates.ndim != 1 or reference_coordinates.ndim != 1:
        raise ValueError("curve coordinates must be one-dimensional")
    if len(observed_coordinates) < 2 or np.any(np.diff(observed_coordinates) <= 0):
        raise ValueError("observed curve coordinates must be strictly increasing")
    if len(reference_coordinates) < 2 or np.any(np.diff(reference_coordinates) <= 0):
        raise ValueError("reference curve coordinates must be strictly increasing")
    if observed_values.shape[0] != len(observed_coordinates) or reference_values.shape[0] != len(reference_coordinates):
        raise ValueError("curve values must match their coordinate arrays")
    if observed_values.ndim != 1 or reference_values.ndim != 1:
        raise ValueError("curve_errors currently supports scalar curves")
    if observed_coordinates[0] < reference_coordinates[0] or observed_coordinates[-1] > reference_coordinates[-1]:
        raise ValueError("observed curve lies outside the reference interpolation interval")

    expected = np.interp(observed_coordinates, reference_coordinates, reference_values)
    weights = trapezoidal_weights(observed_coordinates)
    return {
        "absolute_l2": weighted_l2_error(observed_values, expected, weights),
        "relative_l2": weighted_relative_l2_error(
            observed_values, expected, weights, absolute_floor=absolute_floor
        ),
        "max_abs": max_abs_error(observed_values, expected),
        "sample_count": len(observed_coordinates),
    }
