#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

import numpy as np
import yaml

SUITE_DIR = Path(__file__).resolve().parents[1]
if str(SUITE_DIR) not in sys.path:
    sys.path.insert(0, str(SUITE_DIR))

from common.metrics import relative_l2_error
from common.prony import check_series
from common.reporting import build_verification_report, make_check, write_verification_report
from common.run_prony_harmonic import frequency_label
from common.transient import TIME_LEVELS
from oracle import (
    algorithmic_reaction,
    axial_modulus,
    complex_modulus,
    discrete_complex_modulus,
    harmonic_fit,
)

# The continuum reference: the recursion replayed at this many steps per period, an O(dt^2)
# error of about 5e-8 of the hereditary stress, far below every level's. Each level's
# steps_per_period must divide it, so that every level's step lands on a reference step.
REFERENCE_STEPS_PER_PERIOD = 8192


def reaction_history(folder, expected_times):
    """The right face's axial reaction resultant at every step. generate_bar_mesh makes a unit
    square cross-section, so under the homogeneous strain this resultant is P_11 itself."""
    path = Path(folder) / "quantities.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    entries = data.get("material_objective_history") if isinstance(data, dict) else None
    if not isinstance(entries, list) or len(entries) != len(expected_times):
        raise ValueError(f"incomplete reaction history in {path}")
    times = np.asarray([entry.get("time") for entry in entries], dtype=np.float64)
    if not np.allclose(times, expected_times, rtol=1.0e-12, atol=1.0e-12):
        raise ValueError(f"reaction times differ from the prescribed steps in {path}")
    reactions = []
    for entry in entries:
        match = [item for item in entry.get("constraint_reactions") or []
                 if item.get("condition") == 1 and item.get("component") == 0]
        if len(match) != 1:
            raise ValueError(f"missing right-face reaction in {path}")
        reactions.append(match[0].get("resultant"))
    reactions = np.asarray(reactions, dtype=np.float64)
    if np.any(~np.isfinite(reactions)):
        raise ValueError(f"non-finite reaction history in {path}")
    return times, reactions


def loop_area(strain, reaction):
    """The area of the closed reaction-strain loop over the samples given (trapezoid rule)."""
    return float(np.sum(0.5 * (reaction[1:] + reaction[:-1]) * np.diff(strain)))


def main():
    parser = argparse.ArgumentParser(description="Verify the harmonic Prony response")
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.case.read_text(encoding="utf-8"))
    resolution = config["selected_variant"]["resolution"]
    tolerances = config["verification"]["tolerances"]
    material = config["material"]
    c10, c01, K = float(material["c10"]), float(material["c01"]), float(material["bulk_modulus"])
    eps0 = float(resolution["amplitude"])
    integrator = str(resolution["integrator"])
    g, tau = check_series(str(resolution["prony_g"]), str(resolution["prony_tau"]))
    omegas = [float(value) for value in config["loading"]["omegas"]]
    levels = [int(value) for value in config["time"]["steps_per_period"]]
    cycles = int(config["time"]["cycles"])
    for steps_per_period in levels:
        if steps_per_period <= 0 or REFERENCE_STEPS_PER_PERIOD % steps_per_period:
            raise ValueError(f"steps_per_period {steps_per_period} does not divide the reference's "
                             f"{REFERENCE_STEPS_PER_PERIOD}: the level's steps would not land on reference steps")
    m_iso, m_vol = axial_modulus(c10, c01, K)
    # The continuum checks hold where the coarsest step resolves the fastest relaxation time.
    resolved = [2.0 * np.pi / omega / levels[0] <= float(np.min(tau)) * (1.0 + 1.0e-12) for omega in omegas]
    if not any(resolved):
        raise ValueError("no frequency whose coarsest step resolves the fastest relaxation time")
    oracle = {"type": "analytical", "reference": "prony_harmonic_3d/oracle.py"}

    algorithmic_errors, continuum_errors, rates, all_rates = [], [], [], {}
    discrete_modulus_errors, storage_errors, loss_errors, dissipation_errors = [], [], [], []
    rows = []
    for index, omega in enumerate(omegas):
        period = 2.0 * np.pi / omega
        # The continuum reference over the run, at every step of the finest level.
        ref_dt = period / REFERENCE_STEPS_PER_PERIOD
        ref_steps = REFERENCE_STEPS_PER_PERIOD * cycles
        ref_strain = eps0 * np.sin(omega * ref_dt * np.arange(ref_steps + 1))
        reference = algorithmic_reaction(ref_strain, g, tau, ref_dt, c10, c01, K, integrator)
        level_errors = []
        for name, steps_per_period in zip(TIME_LEVELS, levels):
            dt = period / steps_per_period
            steps = steps_per_period * cycles
            expected_times = dt * np.arange(1, steps + 1)
            times, reaction = reaction_history(args.output / "solution" / frequency_label(index) / name, expected_times)
            strain = eps0 * np.sin(omega * np.concatenate(([0.0], times)))
            # The discrete algorithm itself, replayed: exact at any amplitude.
            algorithmic = algorithmic_reaction(strain, g, tau, dt, c10, c01, K, integrator)[1:]
            scale = float(np.max(np.abs(algorithmic)))
            algorithmic_errors.append(float(np.max(np.abs(reaction - algorithmic))) / scale)
            stride = REFERENCE_STEPS_PER_PERIOD // steps_per_period
            continuum = reference[stride::stride]
            error = relative_l2_error(reaction, continuum)
            level_errors.append(error)
            # The last cycle's loop against the reference's over the same cycle.
            full_strain = np.concatenate(([0.0], eps0 * np.sin(omega * times)))
            full_reaction = np.concatenate(([0.0], reaction))
            area = loop_area(full_strain[steps - steps_per_period:], full_reaction[steps - steps_per_period:])
            ref_last = slice(ref_steps - REFERENCE_STEPS_PER_PERIOD, ref_steps + 1)
            ref_area = loop_area(ref_strain[ref_last], reference[ref_last])
            fit = None
            if "discrete_modulus_max_relative_error" in tolerances:
                storage, loss = harmonic_fit(times, reaction, eps0, omega, tau, m_iso, m_vol)
                fit = storage + 1j * loss
                discrete = discrete_complex_modulus(omega, g, tau, dt, integrator)
                discrete_modulus_errors.append(abs(fit - discrete) / abs(discrete))
            if name == TIME_LEVELS[-1] and resolved[index]:
                continuum_errors.append(error)
                dissipation_errors.append(abs(area - ref_area) / abs(ref_area))
                if fit is not None:
                    exact = complex_modulus(omega, g, tau)
                    storage_errors.append(abs(fit.real - exact.real) / abs(exact.real))
                    loss_errors.append(abs(fit.imag - exact.imag) / abs(exact.imag))
            rows.append([omega, steps_per_period, int(resolved[index]), error, algorithmic_errors[-1], area, ref_area]
                        + ([fit.real, fit.imag] if fit is not None else [np.nan, np.nan]))
        level_rates = [float(np.log2(level_errors[k] / level_errors[k + 1])) for k in range(len(level_errors) - 1)]
        all_rates[str(omega)] = level_rates
        if resolved[index]:
            rates.extend(level_rates)

    measured = {
        "algorithmic_reaction_max_relative_error": max(algorithmic_errors),
        "continuum_reaction_relative_l2": max(continuum_errors),
        "dissipation_relative_error": max(dissipation_errors),
        "temporal_rate_deficit": max(0.0, 1.8 - min(rates)),
        "discrete_modulus_max_relative_error": max(discrete_modulus_errors) if discrete_modulus_errors else 0.0,
        "storage_modulus_max_relative_error": max(storage_errors) if storage_errors else 0.0,
        "loss_modulus_max_relative_error": max(loss_errors) if loss_errors else 0.0,
    }
    checks = []
    for name in tolerances:
        value = float(measured[name])
        expected = 1.8 if name == "temporal_rate_deficit" else 0
        observed = min(rates) if name == "temporal_rate_deficit" else value
        checks.append(make_check(name, observed, expected, value, tolerances[name], "1", oracle))

    table = args.output / "harmonic_levels.csv"
    np.savetxt(table, np.asarray(rows), delimiter=",", comments="",
               header="omega,steps_per_period,resolved,continuum_l2,algorithmic_max,loop_area,reference_loop_area,storage_fit,loss_fit")
    report = build_verification_report(
        config["id"], checks,
        diagnostics={"amplitude": eps0, "integrator": integrator, "prony_g": g.tolist(), "prony_tau": tau.tolist(),
                     "omegas": omegas, "resolved": resolved, "temporal_rates": all_rates,
                     "continuum_moduli": {str(w): [complex_modulus(w, g, tau).real, complex_modulus(w, g, tau).imag]
                                          for w in omegas},
                     "measured": {k: float(v) for k, v in measured.items()}},
        artifacts={"levels": table},
    )
    write_verification_report(args.report, report, tolerances=tolerances)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
