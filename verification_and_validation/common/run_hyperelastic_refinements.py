#!/usr/bin/env python3

"""Run an SFEM BDF2 pressure solve for every structural refinement."""

import argparse
import os
from pathlib import Path
import subprocess

import yaml

from run_linear_refinements import relocate_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--driver", required=True, type=Path)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

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
        command = [str(args.driver), str(target_mesh), *rendered, str(solution)]
        print(f"--- nonlinear structural refinement: {level_id} ---", flush=True)
        completed = subprocess.run(
            command, env=os.environ.copy(), text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False,
        )
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
        if completed.returncode:
            return completed.returncode
        print(f"SFEM_HYPERELASTIC_REFINEMENT_COMPLETE {level_id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
