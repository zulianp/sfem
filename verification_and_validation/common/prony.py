"""Prony-series helpers shared by the Prony V&V cases (prony_relaxation_3d, prony_harmonic_3d,
prony_bar_mode_3d).

A Prony series relaxes the isochoric stress of `GeneratedMooneyRivlinProny` as

    G(t)/G0 = g_inf + sum_i g_i exp(-t/tau_i),   g_inf = 1 - sum_i g_i,

and the material advances one history variable per term with Simo's recursion over a step dt,

    H_i <- alpha_i H_i + b_i (S^{n+1} - S^n),   alpha_i = exp(-dt/tau_i).

The weight b_i depends on how the stress is interpolated over the step:

- "linear": b_i = g_i (1 - alpha_i)/(dt/tau_i), the stress linear in time over the step
  (Taylor, Pister and Goudreau 1970);
- "midpoint": b_i = g_i exp(-dt/(2 tau_i)), the increment applied at the step's midpoint
  (Simo 1987).

Both tend to g_i as dt/tau_i -> 0. `step_weights` is the oracles' copy of these weights; it is
written from the formulas above, not taken from SFEM's kernels.
"""

import numpy as np

# prony_harmonic_3d's continuum reference: the recursion replayed at this many steps per period,
# an O(dt^2) error of about 5e-8 of the hereditary stress, far below every level's. Each level's
# steps_per_period must divide it, so that every level's step lands on a reference step.
REFERENCE_STEPS_PER_PERIOD = 8192


def check_steps_per_period(levels):
    """The harmonic case's steps per period as ints, each positive and dividing the reference's."""
    levels = [int(value) for value in levels]
    for steps_per_period in levels:
        if steps_per_period <= 0 or REFERENCE_STEPS_PER_PERIOD % steps_per_period:
            raise ValueError(f"steps_per_period {steps_per_period} does not divide the reference's "
                             f"{REFERENCE_STEPS_PER_PERIOD}: the level's steps would not land on reference steps")
    return levels


def parse_series(values):
    """A comma-separated string or a list of numbers as a finite 1D float64 array."""
    if isinstance(values, str):
        values = values.split(",")
    result = np.asarray([float(value) for value in values], dtype=np.float64)
    if result.ndim != 1 or not len(result) or np.any(~np.isfinite(result)):
        raise ValueError("Prony series must contain finite values")
    return result


def check_series(g, tau):
    """g_i and tau_i as arrays, with g_i > 0, sum g_i < 1 (so g_inf > 0) and tau_i > 0."""
    g, tau = parse_series(g), parse_series(tau)
    if g.shape != tau.shape or np.sum(g) >= 1.0 or np.any(g <= 0) or np.any(tau <= 0):
        raise ValueError("invalid Prony coefficients")
    return g, tau


def step_weights(g, tau, dt, integrator="linear"):
    """alpha_i and b_i of one step of length dt (see the module docstring)."""
    g, tau = check_series(g, tau)
    x = float(dt) / tau
    alpha = np.exp(-x)
    if integrator == "linear":
        # (1 - exp(-x))/x, written with expm1 so it stays accurate for x << 1.
        b = g * (-np.expm1(-x) / x)
    elif integrator == "midpoint":
        b = g * np.exp(-x / 2.0)
    else:
        raise ValueError(f"unknown integrator {integrator}")
    return alpha, b


def material_config(config):
    """The operator YAML for a resolved Prony case: the case's Mooney-Rivlin parameters and the
    selected variant's series and integrator, the series relaxing the isochoric part only."""
    material = config["material"]
    resolution = config["selected_variant"]["resolution"]
    g, tau = check_series(str(resolution["prony_g"]), str(resolution["prony_tau"]))
    group = {"prony": [{"g": float(gi), "tau": float(ti)} for gi, ti in zip(g, tau)],
             "integrator": str(resolution["integrator"])}
    return {
        "type": "GeneratedMooneyRivlinProny",
        "C10": float(material["c10"]),
        "C01": float(material["c01"]),
        "K": float(material["bulk_modulus"]),
        "relaxation": {"iso": group},
    }
