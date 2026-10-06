"""Independent Prony-series and WLF relaxation formulas."""

import numpy as np


def parse_series(values):
    if isinstance(values, str):
        values = values.split(",")
    result = np.asarray([float(value) for value in values], dtype=np.float64)
    if result.ndim != 1 or not len(result) or np.any(~np.isfinite(result)):
        raise ValueError("Prony series must contain finite values")
    return result


def wlf_shift(temperature, c1, c2, reference_temperature):
    delta = float(temperature) - float(reference_temperature)
    return 10.0 ** (float(c1) * delta / (float(c2) + delta))


def relaxation_factor(times, g, tau_reference, temperature=None, wlf=None):
    times = np.asarray(times, dtype=np.float64)
    g = parse_series(g)
    tau = parse_series(tau_reference)
    if g.shape != tau.shape or np.sum(g) >= 1.0 or np.any(g <= 0) or np.any(tau <= 0):
        raise ValueError("invalid Prony coefficients")
    shift = 1.0
    if wlf is not None:
        shift = wlf_shift(temperature, wlf["C1"], wlf["C2"], wlf["T_ref"])
    tau_effective = tau / shift
    return 1.0 - np.sum(g) + np.sum(g[None, :] * np.exp(-times[:, None] / tau_effective[None, :]), axis=1)


def elastic_axial_reaction(strain, c10, c01, bulk_modulus=0.0):
    stretch = 1.0 + float(strain)
    derivative_i1 = 4.0 / 3.0 * (stretch ** (1.0 / 3.0) - stretch ** (-5.0 / 3.0))
    derivative_i2 = 4.0 / 3.0 * (stretch ** (-1.0 / 3.0) - stretch ** (-7.0 / 3.0))
    return float(c10) * derivative_i1 + float(c01) * derivative_i2 + float(bulk_modulus) * (stretch - 1.0)
