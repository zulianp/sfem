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


def main():
    parser = argparse.ArgumentParser(description="Run three finite-strain Kelvin-Voigt time-step levels")
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--driver", required=True, type=Path)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--dirichlet", required=True, type=Path)
    parser.add_argument("--neumann", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.case.read_text(encoding="utf-8"))
    duration = float(config["time"]["duration"])
    time_steps = [float(value) for value in config["time"]["time_steps"]]
    if len(time_steps) != len(TIME_LEVELS):
        raise ValueError("finite-strain creep requires exactly three time-step sizes")

    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)
    for name, dt in zip(TIME_LEVELS, time_steps):
        steps = round(duration / dt)
        if abs(steps * dt - duration) > 1.0e-12:
            raise ValueError(f"duration {duration} is not divisible by dt={dt}")
        environment = os.environ.copy()
        environment.update(
            {
                "SFEM_DT": str(dt),
                "SFEM_T_END": str(duration),
                "SFEM_STEPS": str(steps),
                "SFEM_EXPORT_FREQ": "1",
            }
        )
        run_checked(
            [args.driver, args.mesh, args.dirichlet, args.neumann, args.output / name],
            environment,
            marker="SFEM_TRANSIENT_LEVEL_COMPLETE",
        )
    print("SFEM_KV_CREEP_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
