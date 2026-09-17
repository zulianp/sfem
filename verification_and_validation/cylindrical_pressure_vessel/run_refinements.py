#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import subprocess

import yaml


def _level_input(source, canonical_mesh, level_mesh, target):
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    for conditions in data.values():
        for condition in conditions:
            relative = Path(condition["path"]).relative_to(canonical_mesh)
            condition["path"] = str(level_mesh / relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return target


def main():
    parser = argparse.ArgumentParser(description="Solve pressure vessel at every generated resolution")
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--driver", required=True, type=Path)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    case = yaml.safe_load(args.case.read_text(encoding="utf-8"))
    canonical_id = next(level["id"] for level in case["refinements"] if level["role"] == "canonical")
    canonical_mesh = args.mesh / canonical_id
    canonical_symmetry = args.input_dir / "symmetry.yaml"
    canonical_pressure = args.input_dir / "inner_pressure.yaml"
    for level in case["refinements"]:
        level_id = level["id"]
        level_mesh = args.mesh / level_id
        if level_id == canonical_id:
            symmetry = canonical_symmetry
            pressure = canonical_pressure
        else:
            input_folder = args.input_dir / "inputs" / level_id
            symmetry = _level_input(canonical_symmetry, canonical_mesh, level_mesh, input_folder / "symmetry.yaml")
            pressure = _level_input(canonical_pressure, canonical_mesh, level_mesh, input_folder / "inner_pressure.yaml")
        solution = args.output / level_id
        solution.mkdir(parents=True, exist_ok=True)
        print(f"--- pressure-vessel refinement {level_id} ---", flush=True)
        result = subprocess.run(
            [str(args.driver), str(level_mesh), str(symmetry), str(pressure), str(solution)],
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        if result.returncode:
            return result.returncode
        print(f"SFEM_REFINEMENT_COMPLETE {level_id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
