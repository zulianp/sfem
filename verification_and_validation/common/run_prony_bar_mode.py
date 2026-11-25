#!/usr/bin/env python3

"""Run the free axial mode of the Prony bar (prony_bar_mode_3d) at three time-step sizes: from rest
with the velocity v0 sin(pi x/L), Newmark with the consistent mass, every step exported."""

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
from common.mesh import read_mesh
from common.prony import material_config
from common.transient import TIME_LEVELS, write_modal_initial_fields


def main():
    parser = argparse.ArgumentParser(description="Run the Prony bar mode at three time-step sizes")
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--driver", required=True, type=Path)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--dirichlet", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="playground case override key=value")
    args = parser.parse_args()

    config = yaml.safe_load(args.case.read_text(encoding="utf-8"))
    duration = float(config["time"]["duration"])
    time_steps = [float(value) for value in config["time"]["time_steps"]]
    velocity = float(config["initial_conditions"]["modal_velocity"])
    length = float(config["geometry"]["length"])
    if len(time_steps) != len(TIME_LEVELS):
        raise ValueError("the Prony bar mode requires exactly three time-step sizes")

    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)
    mesh = read_mesh(args.mesh)
    # From rest: no displacement, no stress, no history, hence no acceleration.
    displacement, initial_velocity, acceleration = write_modal_initial_fields(
        args.output / "initial", mesh, 0.0, velocity, 0.0, length)
    for name, dt in zip(TIME_LEVELS, time_steps):
        steps = round(duration / dt)
        if abs(steps * dt - duration) > 1.0e-12:
            raise ValueError(f"duration {duration} is not divisible by dt={dt}")
        output = args.output / name
        case = {
            "mesh": {"path": str(args.mesh), "packed": False},
            "material": material_config(config),
            "dirichlet_conditions": {"file": str(args.dirichlet)},
            "regime": "dynamic",
            "dynamics": {"scheme": "newmark", "density": float(config["material"]["density"]),
                         "beta": 0.25, "gamma": 0.5, "mass": "consistent"},
            "time": {"dt": dt, "steps": steps},
            "initial_conditions": {
                "displacement": {"components": displacement.split(",")},
                "velocity": {"components": initial_velocity.split(",")},
                "acceleration": {"components": acceleration.split(",")},
            },
            "output": {"path": str(output), "export_freq": 1},
        }
        completed = playground.run(args.driver, case, output / "case.yaml", args.overrides, os.environ.copy())
        if completed.stdout:
            print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
        if completed.returncode:
            raise RuntimeError(f"playground exited with status {completed.returncode}")
        print("SFEM_TRANSIENT_LEVEL_COMPLETE")

    print("SFEM_PRONY_BAR_MODE_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
