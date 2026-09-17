#!/usr/bin/env python3

"""Run one static linear-elastic solve per declared structural refinement."""

import argparse
import os
from pathlib import Path
import subprocess

import yaml


def relocate_paths(value, source_mesh, target_mesh):
    if isinstance(value, dict):
        return {key: relocate_paths(item, source_mesh, target_mesh) for key, item in value.items()}
    if isinstance(value, list):
        return [relocate_paths(item, source_mesh, target_mesh) for item in value]
    if isinstance(value, str) and value.startswith(str(source_mesh) + "/"):
        return str(target_mesh / Path(value).relative_to(source_mesh))
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--driver", required=True, type=Path)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dirichlet", default="dirichlet.yaml")
    parser.add_argument("--neumann")
    args = parser.parse_args()

    config = yaml.safe_load(args.case.read_text(encoding="utf-8"))
    levels = config["refinements"]
    canonical = levels[0]["id"]
    canonical_mesh = args.mesh / canonical
    for level in levels:
        level_id = level["id"]
        target_mesh = args.mesh / level_id
        level_inputs = args.input_dir / "inputs" / level_id
        level_inputs.mkdir(parents=True, exist_ok=True)
        rendered = []
        for filename in (args.dirichlet, args.neumann):
            if filename is None:
                rendered.append("NONE")
                continue
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
        command = [
            str(args.driver), str(target_mesh), rendered[0], rendered[1],
            str(args.input_dir / "operator.yaml"), str(solution),
        ]
        print(f"--- structural refinement: {level_id} ---", flush=True)
        completed = subprocess.run(
            command, env=os.environ.copy(), text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False,
        )
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
        if completed.returncode:
            return completed.returncode
        print(f"SFEM_LINEAR_REFINEMENT_COMPLETE {level_id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
