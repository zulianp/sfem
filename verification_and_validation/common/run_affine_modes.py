#!/usr/bin/env python3

"""Invoke an SFEM solid driver once for each affine loading mode."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

SUITE_DIR = Path(__file__).resolve().parents[1]
if str(SUITE_DIR) not in sys.path:
    sys.path.insert(0, str(SUITE_DIR))

from common.affine import MODE_COMPLETION_MARKER


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("linear", "hyperelastic"), required=True)
    parser.add_argument("--driver", required=True, type=Path)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("modes", nargs="+")
    args = parser.parse_args()

    environment = os.environ.copy()
    for mode in args.modes:
        mode_output = args.output / mode
        mode_output.mkdir(parents=True, exist_ok=True)
        dirichlet = args.input_dir / f"dirichlet_{mode}.yaml"
        if args.kind == "linear":
            command = [
                str(args.driver),
                str(args.mesh),
                str(dirichlet),
                "NONE",
                str(args.input_dir / "operator.yaml"),
                str(mode_output),
            ]
        else:
            command = [str(args.driver), str(args.mesh), str(dirichlet), str(mode_output)]

        print(f"--- affine mode: {mode} ---", flush=True)
        completed = subprocess.run(
            command,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
        if completed.returncode:
            return completed.returncode
        print(f"{MODE_COMPLETION_MARKER} {mode}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
