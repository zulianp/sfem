#!/usr/bin/env python3

"""Run three time-step levels of finite-strain Kelvin-Voigt creep on the elasticity playground."""

import argparse
import os
from pathlib import Path
import shutil
import sys

import yaml

SUITE_DIR = Path(__file__).resolve().parents[1]
if str(SUITE_DIR) not in sys.path:
    sys.path.insert(0, str(SUITE_DIR))

from common import playground
from common.transient import TIME_LEVELS


def creep_case(args, material, output, dt, steps, ramp):
    """Newmark without inertia (the viscous term needs the scheme's velocity), the Neumann load
    ramped in with zero slope at both ends, every step exported with the material reaction."""
    return {
        "mesh": {"path": str(args.mesh), "packed": False},
        "material": material,
        "dirichlet_conditions": {"file": str(args.dirichlet)},
        "loads": {
            "neumann": {"file": str(args.neumann)},
            "profile": {"type": "smooth_ramp", "start_time": 0, "end_time": ramp, "start_value": 0, "end_value": 1},
        },
        "regime": "dynamic",
        "dynamics": {"scheme": "newmark", "density": 0},
        "time": {"dt": dt, "steps": steps},
        "solver": {
            "newton": {"atol": 1.0e-11, "rtol": 0},
            "linear": {"type": "bcgs", "rtol": 1.0e-12, "atol": 1.0e-13},
        },
        "output": {"path": str(output), "export_freq": 1, "fields": ["material_reaction"]},
    }


def main():
    parser = argparse.ArgumentParser(description="Run three finite-strain Kelvin-Voigt time-step levels")
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--driver", required=True, type=Path)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--dirichlet", required=True, type=Path)
    parser.add_argument("--neumann", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--param", dest="parameters", action="append", default=[], help="material parameter name=value")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="playground case override key=value")
    args = parser.parse_args()
    material = playground.material(args.operator, args.parameters)

    config = yaml.safe_load(args.case.read_text(encoding="utf-8"))
    duration = float(config["time"]["duration"])
    time_steps = [float(value) for value in config["time"]["time_steps"]]
    ramp = float(config["loading"]["ramp_time"])
    if len(time_steps) != len(TIME_LEVELS):
        raise ValueError("finite-strain creep requires exactly three time-step sizes")

    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)
    for name, dt in zip(TIME_LEVELS, time_steps):
        steps = round(duration / dt)
        if abs(steps * dt - duration) > 1.0e-12:
            raise ValueError(f"duration {duration} is not divisible by dt={dt}")
        output = args.output / name
        case = creep_case(args, material, output, dt, steps, ramp)
        completed = playground.run(args.driver, case, output / "case.yaml", args.overrides, os.environ.copy())
        if completed.stdout:
            print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
        if completed.returncode:
            raise RuntimeError(f"playground exited with status {completed.returncode}")
        print("SFEM_TRANSIENT_LEVEL_COMPLETE")
    print("SFEM_KV_CREEP_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
