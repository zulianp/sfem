#!/usr/bin/env python3
"""The pump-stroke runs of FF200 (workflows/ldp_containment/transient) as the report's data files.

Run on Alps, where the runs are; stdlib only. Each run folder holds the transient folder
run_transient.sh leaves (smesh.trace.csv, sim2/run.log, sim2/history.csv), moved there by the job.

    python3 collect_transient.py <out folder>

writes
  transient_runs.csv      the four strokes per time integrator, and on the refined mesh: steps,
                          failed attempts, Newton steps and the traced times of simulation 2
  transient_ablation.csv  FF200 over the first 16 or 30 ms on one GPU, each change against the
                          build before it, same columns
  transient_history.csv   the four strokes per time integrator, the start (t = 0, at rest) and every
                          converged step: time, step size, Newton and QP iterations, the valve
                          centres' vertical displacement
  transient_failures.csv  every failed attempt of those runs: the time it would have reached, its
                          step size and why it failed (from run.log; the history's attempts column
                          of the Newmark run's build counts something else)
"""

import csv
import os
import re
import struct
import sys

SCRATCH = "/ritom/scratch/cscs/zulianp"
CASES = SCRATCH + "/ldp_cases/FF200_ExodusMesh"
REFINED = SCRATCH + "/ldp_cases/FF200_mesh_refined"

# (run, case, GPUs, job, folder holding sim2/ and smesh.trace.csv, what it is)
FOUR = [
    ("newmark", "FF200_ExodusMesh", 1, "5131942", CASES + "/transient_strokes4_damped",
     "contact-stabilised Newmark, gamma 0.6, every field written"),
    ("bdf2", "FF200_ExodusMesh", 1, "5132710", CASES + "/transient_strokes4_bdf2", "BDF2, every field written"),
    ("final", "FF200_ExodusMesh", 1, "5133369", CASES + "/transient_strokes4_final",
     "BDF2 started at u_n + dt v_n, float32 tangent, MPRGP runs on the device, the displacement written"),
    ("refined", "FF200_mesh_refined", 4, "5135347", REFINED + "/transient_strokes4",
     "as final, on the refined mesh, four ranks (MPRGP's device runs are single-rank)"),
]
# (change, variant, job, folder): each change against the build before it, FF200 on one GPU.
ABLATION = [
    ("finish", "separate", "5132727", SCRATCH + "/ldp_fuse/before"),
    ("finish", "fused", "5132727", SCRATCH + "/ldp_fuse/fused"),
    ("float32", "float64", "5132754", SCRATCH + "/ldp_fuse/fused2"),
    ("float32", "float32", "5132754", SCRATCH + "/ldp_fuse/fused_f32"),
    ("runs", "steps", "5133192", SCRATCH + "/ldp_fuse/steps_f32"),
    ("runs", "runs", "5133192", SCRATCH + "/ldp_fuse/runs_f32"),
    ("start", "u_n", "5133342", SCRATCH + "/ldp_kind1/base"),
    ("start", "u_n+dt v_n", "5133342", SCRATCH + "/ldp_kind1/start"),
    ("start", "nonmonotone", "5133342", SCRATCH + "/ldp_kind1/nonmono"),
    ("start", "both", "5133342", SCRATCH + "/ldp_kind1/both"),
]
# The valve centres' probes (sim2.yaml). With several ranks the probe sampled rank 0's nearest node
# (fixed with the commit that adds this note); those runs' series are read from the written
# displacement at the node nearest each point, which a one-rank run's probe matches exactly.
PROBES = {"pv_centre_uy": (-0.011, -0.0008, 0.0), "sv_centre_uy": (0.011, -0.00095, 0.0)}
TIMERS = {"stepper_s": "Stepper::run", "newton_s": "NewtonSolver::solve",
          "qp_s": "NodeObstacleStep::solve", "export_s": "FieldExport::write"}
COLUMNS = ["steps", "failed", "newton", "linear", "t_end", "contact_nodes"] + list(TIMERS)
SERIES = ["t", "dt", "newton", "linear", "pv_centre_uy", "sv_centre_uy"]
FAILURE = re.compile(r"^step \d+ t=(\S+) dt=(\S+): attempt failed, (\w+)", re.M)


def transient(folder):
    """The folder with sim2/: the run folder itself or its FF200_ExodusMesh/transient."""
    for f in (folder, os.path.join(folder, "FF200_ExodusMesh", "transient")):
        if os.path.isfile(os.path.join(f, "sim2", "history.csv")):
            return f
    sys.exit("collect_transient: no sim2/history.csv under %s" % folder)


def from_output(folder, n_rows):
    """{column: the y displacement of every exported step} at the node nearest each probe."""
    sim2 = os.path.join(transient(folder), "sim2")
    xyz = []
    for c in "xyz":
        path = os.path.join(sim2, "mesh", c + ".float32")
        data = open(path, "rb").read()
        xyz.append(struct.unpack("<%df" % (len(data) // 4), data))
    out = {}
    for column, point in PROBES.items():
        node = min(range(len(xyz[0])), key=lambda i: sum((xyz[d][i] - point[d]) ** 2 for d in range(3)))
        series = []
        for step in range(n_rows):
            with open(os.path.join(sim2, "out", "disp.1.%09d.float64" % step), "rb") as f:
                f.seek(8 * node)
                series.append(repr(struct.unpack("<d", f.read(8))[0]))
        out[column] = series
    return out


def summary(folder):
    f = transient(folder)
    trace = {}
    for row in csv.reader(open(os.path.join(f, "smesh.trace.csv"))):
        if len(row) == 4:
            trace[row[0]] = row
    log = open(os.path.join(f, "sim2", "run.log")).read()
    every = list(csv.DictReader(open(os.path.join(f, "sim2", "history.csv"))))
    rows = [r for r in every if r["status"] == "converged"]
    contact = re.search(r"Contact: (\d+) nodes", log)
    out = {"steps": len(rows), "failed": len(re.findall(r"attempt failed", log)),
           "newton": int(trace["NodeObstacleStep::solve"][1]),
           "linear": sum(int(r["linear"]) for r in rows), "t_end": rows[-1]["t"],
           "contact_nodes": contact.group(1) if contact else ""}
    for column, timer in TIMERS.items():
        out[column] = trace[timer][2] if timer in trace else ""
    failures = FAILURE.findall(log)
    if len(failures) != out["failed"]:
        sys.exit("collect_transient: %s has failures of an unknown form" % f)
    return out, [r for r in every if r["status"] in ("initial", "converged")], failures


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "transient_runs.csv"), "w", newline="") as fr, \
         open(os.path.join(out, "transient_history.csv"), "w", newline="") as fh, \
         open(os.path.join(out, "transient_failures.csv"), "w", newline="") as ff:
        runs = csv.writer(fr)
        runs.writerow(["run", "case", "gpus", "job", "what"] + COLUMNS)
        history = csv.writer(fh)
        history.writerow(["run"] + SERIES)
        failed = csv.writer(ff)
        failed.writerow(["run", "t", "dt", "reason"])
        for run, case, gpus, job, folder, what in FOUR:
            s, rows, failures = summary(folder)
            runs.writerow([run, case, gpus, job, what] + [s[c] for c in COLUMNS])
            if run in ("final", "refined"):
                written = from_output(folder, len(rows))
                for column, series in written.items():
                    worst = max(abs(float(a) - float(r[column])) for a, r in zip(series, rows))
                    print("collect_transient: %s %s, the probe against the written displacement: %.3g" % (run, column, worst))
                    if gpus > 1:
                        for r, v in zip(rows, series):
                            r[column] = v
                    elif worst != 0:
                        sys.exit("collect_transient: %s's probe is not the written displacement" % run)
            for r in rows:
                history.writerow([run] + [r[c] for c in SERIES])
            for t, dt, why in failures:
                failed.writerow([run, t, dt, why])
    with open(os.path.join(out, "transient_ablation.csv"), "w", newline="") as fa:
        ablation = csv.writer(fa)
        ablation.writerow(["change", "variant", "job"] + COLUMNS)
        for change, variant, job, folder in ABLATION:
            s, _, _ = summary(folder)
            ablation.writerow([change, variant, job] + [s[c] for c in COLUMNS])


if __name__ == "__main__":
    main()
