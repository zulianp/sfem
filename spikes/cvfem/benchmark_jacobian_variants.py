#!/usr/bin/env python3
"""Run CVFEM TET4 Jacobian A/B variants and print a performance table."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_BENCH = HERE / "build" / "cvfem_tet4_ns_upwind_bench"
DEFAULT_KERNELS = (
    "current",
    "current_slots",
    "sympy",
    "sympy_slots",
    "sympy_direct",
    "sympy_block",
    "sympy_face",
    "sympy_simd",
    "sympy_simd_clean",
    "sympy_block_simd",
    "sympy_row_simd",
    "sympy_row_simd_fused",
    "sympy_face_simd",
)


FLOAT_RE = r"([-+0-9.eE]+)"
PATTERNS = {
    "reported_kernel": re.compile(r"^\s*kernel:\s*(\S+)\s*$", re.MULTILINE),
    "seconds_per_assemble": re.compile(r"^\s*seconds_per_assemble:\s*" + FLOAT_RE + r"\s*$", re.MULTILINE),
    "unique_mdofs": re.compile(r"^\s*MDOF/s_unique_mesh_dofs:\s*" + FLOAT_RE + r"\s*$", re.MULTILINE),
    "melem_s_assemble": re.compile(r"^\s*MELEM/s_assemble:\s*" + FLOAT_RE + r"\s*$", re.MULTILINE),
    "gflop_s_assemble": re.compile(r"^\s*GFLOP/s_assemble_model:\s*" + FLOAT_RE + r"\s*$", re.MULTILINE),
}


@dataclass
class RunResult:
    kernel: str
    trial: int
    seconds_per_assemble: float
    unique_mdofs: float
    melem_s_assemble: float
    gflop_s_assemble: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bench", type=Path, default=DEFAULT_BENCH, help=f"benchmark executable (default: {DEFAULT_BENCH})")
    p.add_argument("--n", type=int, default=48, help="cube cells per dimension (default: 48)")
    p.add_argument("--repeat", type=int, default=20, help="benchmark repeats (default: 20)")
    p.add_argument("--warmup", type=int, default=5, help="warmup repeats (default: 5)")
    p.add_argument("--layout", choices=("packed", "atomic"), default="packed", help="benchmark layout (default: packed)")
    p.add_argument("--kernels", nargs="+", default=list(DEFAULT_KERNELS), help="kernel list to run")
    p.add_argument("--trials", type=int, default=1, help="run each kernel N times and report the best unique MDOF/s")
    p.add_argument("--format", choices=("plain", "markdown", "csv", "tsv"), default="plain", help="output table format")
    p.add_argument("--show-command", action="store_true", help="print each benchmark command to stderr before running")
    p.add_argument("--build", action="store_true", help="build cvfem_tet4_ns_upwind_bench before running")
    p.add_argument("--build-dir", type=Path, default=HERE / "build", help="CMake build directory for --build")
    p.add_argument("--extra", nargs=argparse.REMAINDER, help="extra arguments appended to every benchmark command")
    args = p.parse_args()
    cwd = Path.cwd()
    if not args.bench.is_absolute():
        args.bench = (cwd / args.bench).resolve()
    if not args.build_dir.is_absolute():
        args.build_dir = (cwd / args.build_dir).resolve()
    return args


def maybe_build(args: argparse.Namespace) -> None:
    if not args.build:
        return
    cmd = ["cmake", "--build", str(args.build_dir), "--target", "cvfem_tet4_ns_upwind_bench"]
    subprocess.run(cmd, cwd=HERE, check=True)


def parse_metric(name: str, text: str) -> str:
    m = PATTERNS[name].search(text)
    if not m:
        raise ValueError(f"missing metric '{name}'")
    return m.group(1)


def run_one(args: argparse.Namespace, kernel: str, trial: int) -> RunResult:
    cmd = [
        str(args.bench),
        "--n",
        str(args.n),
        "--repeat",
        str(args.repeat),
        "--warmup",
        str(args.warmup),
        "--layout",
        args.layout,
        "--kernel",
        kernel,
        "--assemble",
    ]
    if args.extra:
        cmd.extend(args.extra)
    if args.show_command:
        print("+ " + " ".join(cmd), file=sys.stderr)

    proc = subprocess.run(cmd, cwd=HERE, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        sys.stderr.write(proc.stdout)
        raise subprocess.CalledProcessError(proc.returncode, cmd)

    text = proc.stdout
    try:
        reported = parse_metric("reported_kernel", text)
        return RunResult(
            kernel=reported,
            trial=trial,
            seconds_per_assemble=float(parse_metric("seconds_per_assemble", text)),
            unique_mdofs=float(parse_metric("unique_mdofs", text)),
            melem_s_assemble=float(parse_metric("melem_s_assemble", text)),
            gflop_s_assemble=float(parse_metric("gflop_s_assemble", text)),
        )
    except ValueError as e:
        sys.stderr.write(proc.stderr)
        sys.stderr.write(proc.stdout)
        raise RuntimeError(f"failed to parse output for kernel '{kernel}': {e}") from e


def best_result(results: list[RunResult]) -> RunResult:
    return max(results, key=lambda r: (r.unique_mdofs, -r.seconds_per_assemble))


def print_markdown(rows: list[RunResult], trials: int) -> None:
    if trials > 1:
        print(f"best of {trials} trials per kernel\n")
    print("| kernel | trial | sec/assemble | unique MDOF/s | MELEM/s | GFLOP/s |")
    print("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| `{r.kernel}` | {r.trial} | {r.seconds_per_assemble:.6e} | "
            f"{r.unique_mdofs:.3f} | {r.melem_s_assemble:.3f} | {r.gflop_s_assemble:.3f} |"
        )


def print_plain(rows: list[RunResult], trials: int) -> None:
    if trials > 1:
        print(f"best of {trials} trials per kernel")
        print()

    headers = ("kernel", "trial", "sec/assemble", "unique MDOF/s", "MELEM/s", "GFLOP/s")
    data = [
        (
            r.kernel,
            str(r.trial),
            f"{r.seconds_per_assemble:.6e}",
            f"{r.unique_mdofs:.3f}",
            f"{r.melem_s_assemble:.3f}",
            f"{r.gflop_s_assemble:.3f}",
        )
        for r in rows
    ]
    widths = [len(h) for h in headers]
    for row in data:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(row: tuple[str, ...]) -> str:
        return (
            f"{row[0]:<{widths[0]}}  "
            f"{row[1]:>{widths[1]}}  "
            f"{row[2]:>{widths[2]}}  "
            f"{row[3]:>{widths[3]}}  "
            f"{row[4]:>{widths[4]}}  "
            f"{row[5]:>{widths[5]}}"
        )

    print(fmt_row(headers))
    print(
        f"{'-' * widths[0]}  "
        f"{'-' * widths[1]}  "
        f"{'-' * widths[2]}  "
        f"{'-' * widths[3]}  "
        f"{'-' * widths[4]}  "
        f"{'-' * widths[5]}"
    )
    for row in data:
        print(fmt_row(row))


def print_delimited(rows: list[RunResult], delimiter: str) -> None:
    w = csv.writer(sys.stdout, delimiter=delimiter)
    w.writerow(["kernel", "trial", "seconds_per_assemble", "MDOF/s_unique_mesh_dofs", "MELEM/s_assemble", "GFLOP/s_assemble_model"])
    for r in rows:
        w.writerow([r.kernel, r.trial, f"{r.seconds_per_assemble:.12e}", f"{r.unique_mdofs:.6f}", f"{r.melem_s_assemble:.6f}", f"{r.gflop_s_assemble:.6f}"])


def main() -> int:
    args = parse_args()
    maybe_build(args)
    if not args.bench.exists():
        print(f"error: benchmark executable not found: {args.bench}", file=sys.stderr)
        print("hint: pass --build or run cmake --build build --target cvfem_tet4_ns_upwind_bench", file=sys.stderr)
        return 1

    rows: list[RunResult] = []
    for kernel in args.kernels:
        trials = [run_one(args, kernel, trial + 1) for trial in range(args.trials)]
        rows.append(best_result(trials))

    if args.format == "plain":
        print_plain(rows, args.trials)
    elif args.format == "markdown":
        print_markdown(rows, args.trials)
    elif args.format == "csv":
        print_delimited(rows, ",")
    else:
        print_delimited(rows, "\t")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
