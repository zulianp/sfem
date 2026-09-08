#!/usr/bin/env python3
"""Run CVFEM TET4 benchmark drivers and print a performance table."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
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
DEFAULT_DRIVERS = ("residual", "assemble", "jac-action", "bsr-apply")
DRIVER_ARGS = {
    "residual": (),
    "assemble": ("--assemble",),
    "jac-action": ("--jac-action",),
    "bsr-apply": ("--bsr-apply",),
}


FLOAT_RE = r"([-+0-9.eE]+)"
PATTERNS = {
    "reported_kernel": re.compile(r"^\s*kernel:\s*(\S+)\s*$", re.MULTILINE),
    "seconds_per_call": re.compile(r"^\s*seconds_per_call:\s*" + FLOAT_RE + r"\s*$", re.MULTILINE),
    "unique_mdofs": re.compile(r"^\s*MDOF/s_unique_mesh_dofs:\s*" + FLOAT_RE + r"\s*$", re.MULTILINE),
    "melem_s": re.compile(r"^\s*MELEM/s:\s*" + FLOAT_RE + r"\s*$", re.MULTILINE),
    "gflop_s_model": re.compile(r"^\s*GFLOP/s_model:\s*" + FLOAT_RE + r"\s*$", re.MULTILINE),
    "gflop_s_assemble": re.compile(r"^\s*GFLOP/s_assemble_model:\s*" + FLOAT_RE + r"\s*$", re.MULTILINE),
    "gflop_s_jac_action": re.compile(r"^\s*GFLOP/s_jac_action_model:\s*" + FLOAT_RE + r"\s*$", re.MULTILINE),
    "gflop_s_bsr_apply": re.compile(r"^\s*GFLOP/s_bsr_apply_model:\s*" + FLOAT_RE + r"\s*$", re.MULTILINE),
}


@dataclass
class RunResult:
    driver: str
    kernel: str
    trial: int
    seconds_per_call: float
    unique_mdofs: float
    melem_s: float
    gflop_s: float


@dataclass
class SummaryResult:
    driver: str
    kernel: str
    trials: int
    best_trial: int
    seconds_per_best: float
    unique_mdofs_worst: float
    unique_mdofs_best: float
    unique_mdofs_median: float
    unique_mdofs_average: float
    melem_s_best: float
    gflop_s_best: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bench", type=Path, default=DEFAULT_BENCH, help=f"benchmark executable (default: {DEFAULT_BENCH})")
    p.add_argument("--n", type=int, default=48, help="cube cells per dimension (default: 48)")
    p.add_argument("--repeat", type=int, default=20, help="benchmark repeats (default: 20)")
    p.add_argument("--warmup", type=int, default=5, help="warmup repeats (default: 5)")
    p.add_argument("--layout", choices=("packed", "atomic"), default="packed", help="benchmark layout (default: packed)")
    p.add_argument("--kernels", nargs="+", default=list(DEFAULT_KERNELS), help="kernel list to run")
    p.add_argument(
        "--drivers",
        nargs="+",
        default=["assemble"],
        help="driver list to run: residual, assemble, jac-action, bsr-apply, or all (default: assemble)",
    )
    p.add_argument("--trials", type=int, default=1, help="run each driver/kernel N times and report DOF-rate statistics")
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
    if args.trials < 1:
        p.error("--trials must be >= 1")
    if "all" in args.drivers:
        args.drivers = list(DEFAULT_DRIVERS)
    bad_drivers = [d for d in args.drivers if d not in DRIVER_ARGS]
    if bad_drivers:
        p.error("invalid --drivers value(s): " + ", ".join(bad_drivers))
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


def parse_optional_metric(name: str, text: str) -> str | None:
    m = PATTERNS[name].search(text)
    if not m:
        return None
    return m.group(1)


def parse_gflop_s(driver: str, text: str) -> float:
    if driver == "assemble":
        name = "gflop_s_assemble"
    elif driver == "jac-action":
        name = "gflop_s_jac_action"
    elif driver == "bsr-apply":
        name = "gflop_s_bsr_apply"
    else:
        name = "gflop_s_model"

    parsed = parse_optional_metric(name, text)
    if parsed is not None:
        return float(parsed)
    parsed = parse_optional_metric("gflop_s_model", text)
    return float(parsed) if parsed is not None else 0.0


def run_one(args: argparse.Namespace, driver: str, kernel: str, trial: int) -> RunResult:
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
    ]
    cmd.extend(DRIVER_ARGS[driver])
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
            driver=driver,
            kernel=reported,
            trial=trial,
            seconds_per_call=float(parse_metric("seconds_per_call", text)),
            unique_mdofs=float(parse_metric("unique_mdofs", text)),
            melem_s=float(parse_metric("melem_s", text)),
            gflop_s=parse_gflop_s(driver, text),
        )
    except ValueError as e:
        sys.stderr.write(proc.stderr)
        sys.stderr.write(proc.stdout)
        raise RuntimeError(f"failed to parse output for driver '{driver}', kernel '{kernel}': {e}") from e


def best_result(results: list[RunResult]) -> RunResult:
    return max(results, key=lambda r: (r.unique_mdofs, -r.seconds_per_call))


def summarize_results(results: list[RunResult]) -> SummaryResult:
    best = best_result(results)
    rates = [r.unique_mdofs for r in results]
    return SummaryResult(
        driver=best.driver,
        kernel=best.kernel,
        trials=len(results),
        best_trial=best.trial,
        seconds_per_best=best.seconds_per_call,
        unique_mdofs_worst=min(rates),
        unique_mdofs_best=max(rates),
        unique_mdofs_median=statistics.median(rates),
        unique_mdofs_average=statistics.fmean(rates),
        melem_s_best=best.melem_s,
        gflop_s_best=best.gflop_s,
    )


def print_markdown(rows: list[SummaryResult], trials: int) -> None:
    if trials > 1:
        print(f"{trials} trials per driver/kernel\n")
    print("| driver | kernel | trials | best trial | sec/best | DOF worst | DOF best | DOF median | DOF average | MELEM/s best | GFLOP/s best |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| `{r.driver}` | `{r.kernel}` | {r.trials} | {r.best_trial} | {r.seconds_per_best:.6e} | "
            f"{r.unique_mdofs_worst:.3f} | {r.unique_mdofs_best:.3f} | {r.unique_mdofs_median:.3f} | "
            f"{r.unique_mdofs_average:.3f} | {r.melem_s_best:.3f} | {r.gflop_s_best:.3f} |"
        )


def print_plain(rows: list[SummaryResult], trials: int) -> None:
    if trials > 1:
        print(f"{trials} trials per driver/kernel")
        print()

    headers = ("driver", "kernel", "trials", "best_trial", "sec/best", "DOF worst", "DOF best", "DOF median", "DOF average", "MELEM/s best", "GFLOP/s best")
    data = [
        (
            r.driver,
            r.kernel,
            str(r.trials),
            str(r.best_trial),
            f"{r.seconds_per_best:.6e}",
            f"{r.unique_mdofs_worst:.3f}",
            f"{r.unique_mdofs_best:.3f}",
            f"{r.unique_mdofs_median:.3f}",
            f"{r.unique_mdofs_average:.3f}",
            f"{r.melem_s_best:.3f}",
            f"{r.gflop_s_best:.3f}",
        )
        for r in rows
    ]
    widths = [len(h) for h in headers]
    for row in data:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(row: tuple[str, ...]) -> str:
        return "  ".join(f"{cell:<{widths[i]}}" if i < 2 else f"{cell:>{widths[i]}}" for i, cell in enumerate(row))

    print(fmt_row(headers))
    print("  ".join("-" * w for w in widths))
    for row in data:
        print(fmt_row(row))


def print_delimited(rows: list[SummaryResult], delimiter: str) -> None:
    w = csv.writer(sys.stdout, delimiter=delimiter)
    w.writerow(
        [
            "driver",
            "kernel",
            "trials",
            "best_trial",
            "seconds_per_best",
            "MDOF/s_unique_mesh_dofs_worst",
            "MDOF/s_unique_mesh_dofs_best",
            "MDOF/s_unique_mesh_dofs_median",
            "MDOF/s_unique_mesh_dofs_average",
            "MELEM/s_best",
            "GFLOP/s_best",
        ]
    )
    for r in rows:
        w.writerow(
            [
                r.driver,
                r.kernel,
                r.trials,
                r.best_trial,
                f"{r.seconds_per_best:.12e}",
                f"{r.unique_mdofs_worst:.6f}",
                f"{r.unique_mdofs_best:.6f}",
                f"{r.unique_mdofs_median:.6f}",
                f"{r.unique_mdofs_average:.6f}",
                f"{r.melem_s_best:.6f}",
                f"{r.gflop_s_best:.6f}",
            ]
        )


def main() -> int:
    args = parse_args()
    maybe_build(args)
    if not args.bench.exists():
        print(f"error: benchmark executable not found: {args.bench}", file=sys.stderr)
        print("hint: pass --build or run cmake --build build --target cvfem_tet4_ns_upwind_bench", file=sys.stderr)
        return 1

    rows: list[SummaryResult] = []
    for driver in args.drivers:
        for kernel in args.kernels:
            trials = [run_one(args, driver, kernel, trial + 1) for trial in range(args.trials)]
            rows.append(summarize_results(trials))

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
