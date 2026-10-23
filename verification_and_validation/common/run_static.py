#!/usr/bin/env python3

"""Run one steady solve of the elasticity playground on a V&V case's operator and conditions."""

import argparse
import os
from pathlib import Path
import sys

SUITE_DIR = Path(__file__).resolve().parents[1]
if str(SUITE_DIR) not in sys.path:
    sys.path.insert(0, str(SUITE_DIR))

from common import playground


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", required=True, type=Path)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--operator", required=True, type=Path)
    parser.add_argument("--dirichlet", type=Path)
    parser.add_argument("--neumann", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="playground case override key=value")
    args = parser.parse_args()

    case = playground.static_case(args.mesh, args.operator, args.output, dirichlet=args.dirichlet, neumann=args.neumann)
    completed = playground.run(args.driver, case, args.output / "case.yaml", args.overrides, os.environ.copy())
    print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
