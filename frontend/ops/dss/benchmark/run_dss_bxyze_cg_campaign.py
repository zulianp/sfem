#!/usr/bin/env python3
"""Serial, resumable CG campaign; each method/sample uses a fresh process."""

import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
BENCH = ROOT / "frontend/ops/dss/benchmark"
CPP_FIELDS = (
    "degree,Q,nx,ny,nz,nelements,n_global_dofs,n_cellwise_dofs,B_save,B_exec,threads,"
    "fusion,deterministic_blas,rtol,atol,max_it,rhs,seed,repeat,warmup,real_bytes,scalar_bytes,initial_scale,"
    "method,status,iterations,attempt_seconds,tts_s,s_per_iteration,true_relres_r0,truth_error_inf,"
    "duplicate_difference_inf,peak_solve_rss_bytes,peak_process_rss_bytes,warmup_iterations"
).split(",")
FIELDS = ["case_id", "sample", "execution_order", "recorded_at", "process_returncode"] + CPP_FIELDS


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def integers(text):
    values = list(dict.fromkeys(int(v) for v in text.split(",")))
    if not values or any(v < 1 for v in values):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return values


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path, default=ROOT / "build/sfem_DSSBXYZECGBench")
    p.add_argument("--best-csv", type=Path, default=BENCH / "dss_bxyze_affine_laplacian_global_bench_best.csv")
    p.add_argument("--additional-best-csv", type=Path, action="append", default=[])
    p.add_argument("--output-base", type=Path, default=BENCH / "dss_bxyze_cg_16_64_bench")
    p.add_argument("--levels", type=integers, default=[2, 4, 8])
    p.add_argument("--sizes", type=integers, default=[16, 64])
    p.add_argument("--threads", type=integers, default=[8, 1])
    p.add_argument("--repeat16", type=int, default=4)
    p.add_argument("--repeat64", type=int, default=2)
    p.add_argument("--warmup-iterations", type=int, default=10)
    p.add_argument("--max-it", type=int, default=1000000)
    p.add_argument("--fixed-iterations", type=int, default=0)
    p.add_argument("--fixed-iterations-csv", type=Path,
                   help="CSV with degree,size,iterations for case-specific fixed iteration counts")
    p.add_argument("--reuse-cellwise-csv", type=Path,
                   help="copy validated cellwise samples from an earlier campaign and rerun only global")
    p.add_argument("--rtol", type=float, default=1e-8)
    p.add_argument("--detach", action="store_true")
    a = p.parse_args()
    for n in (a.repeat16, a.repeat64):
        if n < 2 or n % 2:
            p.error("repeat counts must be positive and even")
    if a.max_it < 1 or a.fixed_iterations < 0 or a.warmup_iterations < 1 or not 0 < a.rtol < 1:
        p.error("invalid iteration count or tolerance")
    if a.fixed_iterations and a.fixed_iterations_csv:
        p.error("choose either --fixed-iterations or --fixed-iterations-csv")
    if a.reuse_cellwise_csv and not (a.fixed_iterations or a.fixed_iterations_csv):
        p.error("reusing cellwise samples requires fixed iterations")
    a.binary = a.binary.resolve()
    a.best_csv = a.best_csv.resolve()
    a.additional_best_csv = [path.resolve() for path in a.additional_best_csv]
    if a.fixed_iterations_csv:
        a.fixed_iterations_csv = a.fixed_iterations_csv.resolve()
    if a.reuse_cellwise_csv:
        a.reuse_cellwise_csv = a.reuse_cellwise_csv.resolve()
    a.output_base = a.output_base.resolve()
    if a.reuse_cellwise_csv == Path(str(a.output_base) + ".csv"):
        p.error("the cellwise source and output CSV must differ")
    return a


def atomic_json(path, data):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def cases(a):
    records = []
    for path in [a.best_csv, *a.additional_best_csv]:
        with path.open(newline="") as f:
            records.extend(r for r in csv.DictReader(f) if r["selection"] == "best_apply_dss")
    limits = {}
    if a.fixed_iterations_csv:
        with a.fixed_iterations_csv.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != ["degree", "size", "iterations"]:
                raise ValueError("fixed-iterations CSV requires degree,size,iterations columns")
            for row in reader:
                key = int(row["degree"]), int(row["size"])
                count = int(row["iterations"])
                if key in limits or count < 1:
                    raise ValueError(f"duplicate or invalid fixed-iteration limit for {key}")
                limits[key] = count
    result = []
    # Complete low-degree problems first; test eight threads before one.
    for degree in a.levels:
        for size in a.sizes:
            for threads in a.threads:
                matches = [r for r in records if (int(r["degree"]), int(r["nx"]), int(r["threads"])) ==
                           (degree, size, threads)]
                if len(matches) != 1:
                    raise ValueError(f"expected one best_apply_dss row for {(degree, size, threads)}")
                r = matches[0]
                case = dict(case_id=f"n{size}_p{degree}_t{threads}", degree=degree, size=size,
                            threads=threads, B_save=int(r["B_save"]), B_exec=int(r["B_exec"]),
                            repeat=a.repeat64 if size == 64 else a.repeat16)
                if a.fixed_iterations_csv:
                    if (degree, size) not in limits:
                        raise ValueError(f"missing fixed-iteration limit for {(degree, size)}")
                    case["fixed_iterations"] = limits[degree, size]
                result.append(case)
    return result


def number(row, key):
    try:
        value = float(row[key])
        return value if math.isfinite(value) else None
    except (KeyError, ValueError, TypeError):
        return None


def reused_cellwise_rows(path, plan):
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIELDS:
            raise ValueError("incompatible cellwise source CSV")
        rows = [row for row in reader if row["method"] == "cellwise"]
    expected = {(case["case_id"], sample) for case in plan for sample in range(case["repeat"])}
    actual = {(row["case_id"], int(row["sample"])) for row in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError("cellwise source does not contain exactly the planned samples")
    cases_by_id = {case["case_id"]: case for case in plan}
    for row in rows:
        case = cases_by_id[row["case_id"]]
        if (row["status"] != "fixed_iterations" or int(row["process_returncode"]) != 0 or
                int(row["degree"]) != case["degree"] or int(row["nx"]) != case["size"] or
                int(row["threads"]) != case["threads"] or int(row["max_it"]) != case["fixed_iterations"] or
                int(row["iterations"]) != case["fixed_iterations"] or
                int(row["B_save"]) != case["B_save"] or int(row["B_exec"]) != case["B_exec"]):
            raise ValueError(f"cellwise source disagrees with planned case {case['case_id']}")
    return rows


def summary(path, plan, records, fixed_mode=False):
    rows = []
    expected = "fixed_iterations" if fixed_mode else "converged"
    runtime = "attempt_seconds" if fixed_mode else "tts_s"
    for case in plan:
        row = dict(case)
        selected = [r for r in records if r["case_id"] == case["case_id"]]
        errors = any(r["status"] != expected for r in selected)
        for method in ("global", "cellwise"):
            good = [r for r in selected if r["method"] == method and r["status"] == expected]
            row[f"{method}_completed"] = len(good)
            metric_keys = (runtime, "s_per_iteration", "true_relres_r0") if fixed_mode else (runtime, "s_per_iteration")
            for key in metric_keys:
                values = [number(r, key) for r in good]
                row[f"{method}_{key}_median"] = statistics.median(values) if values else ""
            for key in ("true_relres_r0", "truth_error_inf", "peak_solve_rss_bytes", "peak_process_rss_bytes"):
                values = [number(r, key) for r in good]
                row[f"{method}_{key}_max"] = max(values) if values else ""
            its = [int(r["iterations"]) for r in good]
            row[f"{method}_iterations_min"] = min(its) if its else ""
            row[f"{method}_iterations_max"] = max(its) if its else ""
        complete = all(row[f"{m}_completed"] == case["repeat"] for m in ("global", "cellwise"))
        row["status"] = "failed" if errors else "complete" if complete else "partial" if selected else "pending"
        # A speedup is published only after the complete balanced sample set.
        for key in (runtime, "s_per_iteration"):
            g, c = row[f"global_{key}_median"], row[f"cellwise_{key}_median"]
            row[f"speedup_{key}"] = g / c if complete and not errors else ""
        if fixed_mode:
            g, c = row["global_true_relres_r0_median"], row["cellwise_true_relres_r0_median"]
            row["residual_ratio_cellwise_global"] = c / g if complete and g else ""
        rows.append(row)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return sum(r["status"] == "complete" for r in rows)


def run(a):
    fixed_mode = bool(a.fixed_iterations or a.fixed_iterations_csv)
    base = a.output_base
    directory = Path(str(base) + "_runs")
    directory.mkdir(parents=True, exist_ok=True)
    raw_path = Path(str(base) + ".csv")
    summary_path = Path(str(base) + "_summary.csv")
    status_path = Path(str(base) + "_status.json")
    with (directory / "campaign.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = cases(a)
        manifest_path = directory / "manifest.json"
        protocol = dict(cases=plan, rtol=a.rtol, max_it=a.fixed_iterations or a.max_it,
                        warmup_iterations=a.warmup_iterations,
                        x0=0, fusion=1, deterministic_blas=0, rhs="smooth", seed=42,
                        best_csvs=[str(a.best_csv), *map(str, a.additional_best_csv)],
                        binary_source=str(a.binary))
        if a.reuse_cellwise_csv:
            protocol["reuse_cellwise_csv"] = str(a.reuse_cellwise_csv)
            protocol["reuse_cellwise_sha256"] = hashlib.sha256(a.reuse_cellwise_csv.read_bytes()).hexdigest()
        if a.fixed_iterations:
            protocol["fixed_iterations"] = a.fixed_iterations
        if a.fixed_iterations_csv:
            protocol["fixed_iterations_csv"] = str(a.fixed_iterations_csv)
        snapshot = directory / a.binary.name
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest["protocol"] != protocol:
                raise ValueError("existing campaign uses another protocol; choose a new --output-base")
            if hashlib.sha256(snapshot.read_bytes()).hexdigest() != manifest["binary_sha256"]:
                raise ValueError("campaign binary snapshot was modified")
        else:
            shutil.copy2(a.binary, snapshot)
            atomic_json(manifest_path, dict(created_at=now(), protocol=protocol,
                        binary_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(), python=sys.executable))
        records = []
        if raw_path.exists():
            with raw_path.open(newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames != FIELDS:
                    raise ValueError("incompatible existing raw CSV")
                records = list(reader)
        else:
            if a.reuse_cellwise_csv:
                records = reused_cellwise_rows(a.reuse_cellwise_csv, plan)
            with raw_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(records)
        done = {(r["case_id"], int(r["sample"]), r["method"]) for r in records}
        completed = summary(summary_path, plan, records, fixed_mode)
        Path(str(base) + ".pid").write_text(f"{os.getpid()}\n")
        state = dict(worker_pid=os.getpid(), started_at=now(), phase="running", total_cases=len(plan),
                     completed_cases=completed, completed_samples=len(records), current=None)
        atomic_json(status_path, state)
        print(f"{now()} Campaign started: {len(plan)} cases; results={raw_path}", flush=True)
        env = os.environ.copy()
        env.update(PYTHONPATH="python", OMPI_MCA_btl="self", OMP_DYNAMIC="FALSE")
        for case in plan:
            for sample in range(case["repeat"]):
                order = ("global", "cellwise") if sample % 2 == 0 else ("cellwise", "global")
                for ordinal, method in enumerate(order):
                    key = case["case_id"], sample, method
                    if key in done:
                        continue
                    stem = directory / f"{case['case_id']}_r{sample}_{method}"
                    output = Path(str(stem) + ".csv")
                    logfile = Path(str(stem) + ".log")
                    command = [str(snapshot), "--method", method, "--levels", str(case["degree"]),
                               "--sizes", str(case["size"]), "--threads", str(case["threads"]),
                               "--blocks", str(case["B_save"]), "--exec-blocks", str(case["B_exec"]),
                               "--repeat", str(case["repeat"]), "--warmup", "1", "--fusions", "1",
                               "--warmup-iterations", str(a.warmup_iterations), "--initial-scale", "0",
                               "--rtol", str(a.rtol), "--max-it", str(case.get("fixed_iterations", a.fixed_iterations) or a.max_it),
                               "--deterministic", "0",
                               "--rhs", "smooth", "--seed", "42", "--output", str(output)]
                    if fixed_mode:
                        command.extend(("--fixed-iterations", str(case.get("fixed_iterations", a.fixed_iterations))))
                    current = dict(case=case, sample=sample, method=method, execution_order=ordinal,
                                   log=str(logfile), started_at=now(), command=command)
                    state.update(current=current, updated_at=now())
                    atomic_json(status_path, state)
                    print(f"{now()} START {key}, B={case['B_save']}/{case['B_exec']}", flush=True)
                    env["OMP_NUM_THREADS"] = str(case["threads"])
                    start = time.monotonic()
                    with logfile.open("w") as log:
                        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                                stdin=subprocess.DEVNULL)
                        state["current"]["child_pid"] = proc.pid
                        atomic_json(status_path, state)
                        try:
                            while True:
                                try:
                                    code = proc.wait(timeout=30)
                                    break
                                except subprocess.TimeoutExpired:
                                    tail = logfile.read_text(errors="replace").splitlines()
                                    print(f"{now()} RUNNING {key}, elapsed={time.monotonic()-start:.0f}s; "
                                          f"{tail[-1] if tail else 'setup'}", flush=True)
                                    state.update(updated_at=now(), current_elapsed_s=time.monotonic()-start)
                                    atomic_json(status_path, state)
                        except BaseException:
                            proc.terminate()
                            proc.wait()
                            raise
                    row = None
                    if output.exists():
                        with output.open(newline="") as f:
                            values = list(csv.DictReader(f))
                        if len(values) == 1 and set(values[0]) == set(CPP_FIELDS):
                            row = values[0]
                    if row is None:
                        row = dict.fromkeys(CPP_FIELDS, "")
                        row.update(degree=case["degree"], nx=case["size"], threads=case["threads"],
                                   B_save=case["B_save"], B_exec=case["B_exec"], method=method,
                                   status="process_failed", attempt_seconds=time.monotonic()-start)
                    if code != 0 and row["status"] == ("fixed_iterations" if fixed_mode else "converged"):
                        row["status"] = "process_failed"
                    row.update(case_id=case["case_id"], sample=sample, execution_order=ordinal,
                               recorded_at=now(), process_returncode=code)
                    with raw_path.open("a", newline="") as f:
                        csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
                        f.flush()
                        os.fsync(f.fileno())
                    records.append(row)
                    done.add(key)
                    completed = summary(summary_path, plan, records, fixed_mode)
                    state.update(completed_cases=completed, completed_samples=len(records), updated_at=now())
                    atomic_json(status_path, state)
                    runtime = row["attempt_seconds"] if fixed_mode else row["tts_s"]
                    label = "fixed runtime" if fixed_mode else "tts"
                    print(f"{now()} END {key}: {row['status']}, iterations={row['iterations']}, "
                          f"{label}={runtime} s; complete cases={completed}/{len(plan)}", flush=True)
        state.update(phase="complete" if completed == len(plan) else "finished_with_failures", current=None, finished_at=now())
        atomic_json(status_path, state)
        print(f"{now()} {state['phase']}: {completed}/{len(plan)} cases", flush=True)
        return 0 if completed == len(plan) else 1


def main():
    a = arguments()
    if a.detach:
        a.output_base.parent.mkdir(parents=True, exist_ok=True)
        logfile = Path(str(a.output_base) + ".log")
        command = [sys.executable, str(Path(__file__).resolve()), *[s for s in sys.argv[1:] if s != "--detach"]]
        with logfile.open("a") as log:
            proc = subprocess.Popen(command, cwd=ROOT, start_new_session=True, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT)
        print(f"Started campaign PID={proc.pid}\nLog: {logfile}\nStatus: {a.output_base}_status.json")
        return 0
    try:
        return run(a)
    except Exception as error:
        print(f"{now()} Campaign error: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
