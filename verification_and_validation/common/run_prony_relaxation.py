#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import shutil
import sys

import yaml

SUITE_DIR = Path(__file__).resolve().parents[1]
if str(SUITE_DIR) not in sys.path:
    sys.path.insert(0, str(SUITE_DIR))

from common.transient import TIME_LEVELS, run_checked


def temperature_label(value):
    return "T" + format(float(value), ".8g").replace("-", "m").replace(".", "p")


def run_level(args, output, duration, dt, temperature, rejected_trial=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    steps = round(duration / dt)
    if abs(steps * dt - duration) > 1.0e-12:
        raise ValueError(f"duration {duration} is not divisible by dt={dt}")
    environment = os.environ.copy()
    environment.update(
        {
            "SFEM_DT": str(dt),
            "SFEM_LOAD_STEPS": str(steps),
            "SFEM_TEMPERATURE": str(temperature),
            "SFEM_EXPORT_FREQ": str(steps),
            "SFEM_VISCO_REJECT_TRIAL": "1" if rejected_trial else "0",
        }
    )
    return run_checked([args.driver, args.mesh, args.dirichlet, output], environment)


def main():
    parser = argparse.ArgumentParser(description="Run Prony relaxation and history-commit checks")
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--driver", required=True, type=Path)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--dirichlet", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.case.read_text(encoding="utf-8"))
    duration = float(config["time"]["duration"])
    time_steps = [float(value) for value in config["time"]["time_steps"]]
    temperatures = [float(value) for value in os.environ.get("SFEM_PRONY_TEMPERATURES", "20").split(",")]
    if len(time_steps) != len(TIME_LEVELS):
        raise ValueError("Prony relaxation requires exactly three time-step sizes")
    if len(temperatures) not in (1, 2):
        raise ValueError("Prony variants require one reference temperature or one temperature pair")

    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)
    for temperature in temperatures:
        root = args.output / temperature_label(temperature)
        for name, dt in zip(TIME_LEVELS, time_steps):
            run_level(args, root / name, duration, dt, temperature)
            print("SFEM_TRANSIENT_LEVEL_COMPLETE")

    reference_temperature = temperatures[0]
    run_level(
        args,
        args.output / "rejected_trial",
        duration,
        time_steps[-1],
        reference_temperature,
        rejected_trial=True,
    )
    print("SFEM_PRONY_COMMIT_CHECK_COMPLETE")
    print("SFEM_PRONY_RELAXATION_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
