#!/usr/bin/env python3

"""Run the harmonic Prony case (prony_harmonic_3d): per frequency and per step level, the bar driven
by eps0 sin(omega t) from rest, the right face's reaction written every step to quantities.yaml."""

import argparse
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import yaml

SUITE_DIR = Path(__file__).resolve().parents[1]
if str(SUITE_DIR) not in sys.path:
    sys.path.insert(0, str(SUITE_DIR))

from common import playground
from common.prony import check_steps_per_period, material_config
from common.transient import TIME_LEVELS


def frequency_label(index):
    return f"w{index}"


def run_level(args, config, output, omega, steps_per_period, cycles):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    period = 2.0 * np.pi / omega
    dt = period / steps_per_period
    steps = steps_per_period * cycles
    conditions = yaml.safe_load(args.dirichlet.read_text(encoding="utf-8"))["dirichlet_conditions"]
    driven = [c for c in conditions if isinstance(c.get("profile"), dict) and c["profile"].get("type") == "sine"]
    if len(driven) != 1:
        raise ValueError("the harmonic case needs exactly one condition with a sine profile")
    driven[0]["profile"]["period"] = float(period)
    case = {
        "mesh": {"path": str(args.mesh), "packed": False},
        "material": material_config(config),
        "dirichlet_conditions": conditions,
        "regime": "quasi_static",
        "time": {"dt": float(dt), "steps": int(steps)},
        "output": {"path": str(output), "export_freq": int(steps), "fields": ["material_reaction"], "quantities": True},
    }
    completed = playground.run(args.driver, case, output / "case.yaml", args.overrides, os.environ.copy())
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    if completed.returncode:
        raise RuntimeError(f"playground exited with status {completed.returncode}")


def main():
    parser = argparse.ArgumentParser(description="Run the harmonic Prony case")
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--driver", required=True, type=Path)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--dirichlet", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="playground case override key=value")
    args = parser.parse_args()

    config = yaml.safe_load(args.case.read_text(encoding="utf-8"))
    omegas = [float(value) for value in config["loading"]["omegas"]]
    levels = check_steps_per_period(config["time"]["steps_per_period"])
    cycles = int(config["time"]["cycles"])
    if len(levels) != len(TIME_LEVELS):
        raise ValueError("the harmonic case requires exactly three step levels")

    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)
    for index, omega in enumerate(omegas):
        for name, steps_per_period in zip(TIME_LEVELS, levels):
            run_level(args, config, args.output / frequency_label(index) / name, omega, steps_per_period, cycles)
            print("SFEM_TRANSIENT_LEVEL_COMPLETE")
    print("SFEM_PRONY_HARMONIC_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
