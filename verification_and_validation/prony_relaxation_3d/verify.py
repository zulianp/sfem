#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares
import yaml

SUITE_DIR = Path(__file__).resolve().parents[1]
if str(SUITE_DIR) not in sys.path:
    sys.path.insert(0, str(SUITE_DIR))

from common.metrics import relative_l2_error
from common.reporting import build_verification_report, make_check, write_verification_report
from common.transient import TIME_LEVELS
from common.run_prony_relaxation import temperature_label
from oracle import elastic_axial_reaction, parse_series, relaxation_factor, wlf_shift


def reaction_history(folder, expected_times):
    path = Path(folder) / "quantities.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    entries = data.get("material_objective_history") if isinstance(data, dict) else None
    if not isinstance(entries, list) or len(entries) != len(expected_times):
        raise ValueError(f"incomplete reaction history in {path}")
    times = np.asarray([entry.get("time") for entry in entries], dtype=np.float64)
    if np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError(f"reaction times are not finite and strictly increasing in {path}")
    if not np.allclose(times, expected_times, rtol=0.0, atol=1.0e-11):
        raise ValueError(f"reaction time history is incomplete in {path}")
    reactions = []
    for entry in entries:
        conditions = entry.get("constraint_reactions")
        if not isinstance(conditions, list):
            raise ValueError(f"missing condition reactions in {path}")
        match = [item for item in conditions if item.get("condition") == 1 and item.get("component") == 0]
        if len(match) != 1:
            raise ValueError(f"missing right-face reaction in {path}")
        reactions.append(match[0].get("resultant"))
    reactions = np.asarray(reactions, dtype=np.float64)
    if np.any(~np.isfinite(reactions)):
        raise ValueError(f"non-finite reaction history in {path}")
    return times, reactions


def fit_taus(times, observed_factor, g, initial_taus, temperature, wlf):
    initial_taus = np.asarray(initial_taus, dtype=np.float64)

    def residual(log_tau):
        predicted = relaxation_factor(times, g, np.exp(log_tau), temperature=temperature, wlf=wlf)
        return predicted - observed_factor

    result = least_squares(residual, np.log(initial_taus), xtol=1.0e-13, ftol=1.0e-13, gtol=1.0e-13)
    return np.exp(result.x)


def main():
    parser = argparse.ArgumentParser(description="Verify Prony relaxation and WLF shifting")
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.case.read_text(encoding="utf-8"))
    resolution = config["selected_variant"]["resolution"]
    tolerances = config["verification"]["tolerances"]
    material = config["material"]
    duration = float(config["time"]["duration"])
    time_steps = np.asarray(config["time"]["time_steps"], dtype=np.float64)
    temperatures = [float(value) for value in str(resolution["temperatures"]).split(",")]
    g = parse_series(str(resolution["prony_g"]))
    tau = parse_series(str(resolution["prony_tau"]))
    use_wlf = bool(resolution["use_wlf"])
    wlf = config["wlf"] if use_wlf else None
    elastic_reaction = elastic_axial_reaction(
        config["loading"]["strain"], material["c10"], material["c01"], material["bulk_modulus"]
    )
    oracle = {"type": "analytical", "reference": "prony_relaxation_3d/oracle.py"}

    all_levels = {}
    finest_history_errors = []
    fitted_tau_errors = []
    for temperature in temperatures:
        levels = {}
        for name, dt in zip(TIME_LEVELS, time_steps):
            expected_times = np.arange(1, round(duration / dt) + 1, dtype=np.float64) * dt
            times, reaction = reaction_history(
                args.output / "solution" / temperature_label(temperature) / name, expected_times
            )
            observed_factor = reaction / elastic_reaction
            expected_factor = relaxation_factor(times, g, tau, temperature=temperature, wlf=wlf)
            error = relative_l2_error(observed_factor, expected_factor)
            levels[name] = (times, observed_factor, expected_factor, error)
        all_levels[temperature] = levels
        times, observed, expected, error = levels["fine"]
        finest_history_errors.append(error)
        fitted = fit_taus(times, observed, g, tau, temperature, wlf)
        fitted_tau_errors.extend(np.abs(fitted - tau) / tau)

    reference_temperature = temperatures[0]
    fine_times, baseline_reaction = reaction_history(
        args.output / "solution" / temperature_label(reference_temperature) / "fine",
        np.arange(1, round(duration / time_steps[-1]) + 1, dtype=np.float64) * time_steps[-1],
    )
    _, rejected_reaction = reaction_history(args.output / "solution" / "rejected_trial", fine_times)
    commit_error = relative_l2_error(rejected_reaction, baseline_reaction)

    history_error = max(finest_history_errors)
    tau_error = max(fitted_tau_errors)
    reference_factor = all_levels[reference_temperature]["fine"][1]
    long_time_error = abs(reference_factor[-1] - (1.0 - np.sum(g))) / (1.0 - np.sum(g))
    reductions = []
    for levels in all_levels.values():
        level_errors = [levels[name][3] for name in TIME_LEVELS]
        reductions.extend(level_errors[index + 1] / level_errors[index] for index in range(2))
    reduction_deficit = max(0.0, max(reductions) - 1.0)

    collapse_error = 0.0
    if len(temperatures) == 2:
        cold = temperatures[0]
        hot = temperatures[1]
        cold_times, cold_factor, _, _ = all_levels[cold]["fine"]
        hot_times, hot_factor, _, _ = all_levels[hot]["fine"]
        shift_cold = wlf_shift(cold, wlf["C1"], wlf["C2"], wlf["T_ref"])
        shift_hot = wlf_shift(hot, wlf["C1"], wlf["C2"], wlf["T_ref"])
        cold_reduced = cold_times * shift_cold
        hot_reduced = hot_times * shift_hot
        upper = min(cold_reduced[-1], hot_reduced[-1])
        mask = hot_reduced <= upper
        interpolated = np.interp(hot_reduced[mask], cold_reduced, cold_factor)
        collapse_error = relative_l2_error(hot_factor[mask], interpolated)

    checks = [
        make_check("reaction_history_relative_l2", history_error, 0, history_error,
                   tolerances["reaction_history_relative_l2"], "1", oracle),
        make_check("long_time_modulus_relative_error", reference_factor[-1], 1.0 - np.sum(g), long_time_error,
                   tolerances["long_time_modulus_relative_error"], "1", oracle),
        make_check("fitted_tau_max_relative_error", tau_error, 0, tau_error,
                   tolerances["fitted_tau_max_relative_error"], "1", oracle),
        make_check("wlf_collapse_relative_l2", collapse_error, 0, collapse_error,
                   tolerances["wlf_collapse_relative_l2"], "1", oracle),
        make_check("temporal_reduction_deficit", max(reductions), 1.0, reduction_deficit,
                   tolerances["temporal_reduction_deficit"], "1", oracle),
        make_check("rejected_trial_history_relative_l2", commit_error, 0, commit_error,
                   tolerances["rejected_trial_history_relative_l2"], "1", oracle),
    ]
    history_path = args.output / "relaxation_history.csv"
    times, observed, expected, _ = all_levels[reference_temperature]["fine"]
    np.savetxt(history_path, np.column_stack((times, observed, expected)), delimiter=",",
               header="time,normalized_reaction,oracle_relaxation", comments="")
    report = build_verification_report(
        config["id"], checks,
        diagnostics={"temperatures": temperatures, "prony_g": g.tolist(), "prony_tau": tau.tolist(),
                     "finest_history_errors": finest_history_errors, "fitted_tau_errors": fitted_tau_errors,
                     "temporal_error_ratios": reductions, "elastic_reaction": elastic_reaction},
        artifacts={"relaxation_history": history_path},
    )
    write_verification_report(args.report, report, tolerances=tolerances)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
