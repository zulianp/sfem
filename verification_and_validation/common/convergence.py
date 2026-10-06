"""Log-log spatial and temporal convergence-rate fitting."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConvergenceFit:
    rate: float
    intercept: float
    r_squared: float
    sample_count: int

    def as_dict(self):
        return {
            "rate": self.rate,
            "intercept": self.intercept,
            "r_squared": self.r_squared,
            "sample_count": self.sample_count,
        }


def fit_convergence_rate(scales, errors):
    scales = np.asarray(scales, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)
    if scales.ndim != 1 or errors.shape != scales.shape or len(scales) < 2:
        raise ValueError("convergence fit requires equally sized vectors with at least two samples")
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("convergence scales must be finite and positive")
    if np.any(~np.isfinite(errors)) or np.any(errors <= 0):
        raise ValueError("convergence errors must be finite and positive")
    if len(np.unique(scales)) != len(scales):
        raise ValueError("convergence scales must be distinct")

    x = np.log(scales)
    y = np.log(errors)
    design = np.column_stack((x, np.ones_like(x)))
    rate, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    prediction = rate * x + intercept
    residual = np.sum((y - prediction) ** 2)
    total = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 if total <= np.finfo(np.float64).eps else 1.0 - residual / total
    return ConvergenceFit(float(rate), float(intercept), float(r_squared), len(scales))


def fit_spatial_convergence(mesh_sizes, errors):
    return fit_convergence_rate(mesh_sizes, errors)


def fit_temporal_convergence(time_steps, errors):
    return fit_convergence_rate(time_steps, errors)
