#!/usr/bin/env python3

"""Run one steady solve of the elasticity playground for each affine loading mode."""

import argparse
import os
from pathlib import Path
import sys

import yaml

SUITE_DIR = Path(__file__).resolve().parents[1]
if str(SUITE_DIR) not in sys.path:
    sys.path.insert(0, str(SUITE_DIR))

from common import playground
from common.affine import MODE_COMPLETION_MARKER


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", required=True, type=Path)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--initialize",
        action="store_true",
        help="start each mode from the affine field in <mesh>/initial_values/<mode>.<component>.float64.raw",
    )
    parser.add_argument(
        "--field",
        dest="fields",
        action="append",
        default=[],
        help="name=path: an element field of the material, 9 values per element",
    )
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="playground case override key=value")
    parser.add_argument("modes", nargs="+")
    args = parser.parse_args()

    dimension = None
    if args.initialize:
        metadata = yaml.safe_load((args.mesh / "meta.yaml").read_text(encoding="utf-8"))
        dimension = int(metadata["spatial_dimension"])
    fields = []
    for field in args.fields:
        name, path = field.split("=", 1)
        fields.append({"name": name, "file": path, "values_per_element": 9})

    for mode in args.modes:
        mode_output = args.output / mode
        mode_output.mkdir(parents=True, exist_ok=True)
        case = playground.static_case(
            args.mesh,
            args.input_dir / "operator.yaml",
            mode_output,
            dirichlet=args.input_dir / f"dirichlet_{mode}.yaml",
        )
        if fields:
            case["fields"] = fields
        if args.initialize:
            case["initial_conditions"] = {
                "displacement": {
                    "components": [
                        str(args.mesh / "initial_values" / f"{mode}.{component}.float64.raw")
                        for component in range(dimension)
                    ]
                }
            }

        print(f"--- affine mode: {mode} ---", flush=True)
        completed = playground.run(args.driver, case, mode_output / "case.yaml", args.overrides, os.environ.copy())
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
        if completed.returncode:
            return completed.returncode
        print(f"{MODE_COMPLETION_MARKER} {mode}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
