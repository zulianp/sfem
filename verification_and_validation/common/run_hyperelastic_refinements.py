#!/usr/bin/env python3

"""Run a quasi-static pressure solve of the elasticity playground for every structural refinement."""

import argparse
import os
from pathlib import Path
import sys

import yaml

SUITE_DIR = Path(__file__).resolve().parents[1]
if str(SUITE_DIR) not in sys.path:
    sys.path.insert(0, str(SUITE_DIR))

from common import playground
from common.run_linear_refinements import relocate_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--driver", required=True, type=Path)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--param", dest="parameters", action="append", default=[], help="material parameter name=value")
    parser.add_argument("--dt", required=True, type=float)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--ramp", type=float, default=0, help="seconds over which the pressure ramps to its value")
    parser.add_argument("--export-freq", type=int, default=1)
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="playground case override key=value")
    args = parser.parse_args()
    material = playground.material(args.operator, args.parameters)

    config = yaml.safe_load(args.case.read_text(encoding="utf-8"))
    canonical_mesh = args.mesh / config["refinements"][0]["id"]
    for level in config["refinements"]:
        level_id = level["id"]
        target_mesh = args.mesh / level_id
        level_inputs = args.input_dir / "inputs" / level_id
        level_inputs.mkdir(parents=True, exist_ok=True)
        rendered = []
        for filename in ("dirichlet.yaml", "neumann.yaml"):
            source = args.input_dir / filename
            document = yaml.safe_load(source.read_text(encoding="utf-8"))
            target = level_inputs / filename
            target.write_text(
                yaml.safe_dump(relocate_paths(document, canonical_mesh, target_mesh), sort_keys=False),
                encoding="utf-8",
            )
            rendered.append(str(target))
        solution = args.output / level_id
        solution.mkdir(parents=True, exist_ok=True)
        case = playground.load_ramp_case(
            target_mesh, material, solution, rendered[0], rendered[1], args.dt, args.steps, args.ramp, args.export_freq
        )
        print(f"--- nonlinear structural refinement: {level_id} ---", flush=True)
        completed = playground.run(args.driver, case, solution / "case.yaml", args.overrides, os.environ.copy())
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
        if completed.returncode:
            return completed.returncode
        print(f"SFEM_HYPERELASTIC_REFINEMENT_COMPLETE {level_id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
