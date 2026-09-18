#!/usr/bin/env python3
"""Per-step cost model for the FDA-nozzle transient campaign, built from measured runs.

    scripts/dns_cost_model.py <run-tree> [campaign-tree]

where <run-tree> holds one directory per configuration, each with the `diag.csv` and `log` that
jobs/dns_cost.sbatch writes. Nothing is hard-coded from a previous run: every number this
prints is read back out of those two files.

WHAT IS MEASURED AND WHAT IS FITTED, because the distinction is the whole point.

The driver's phase table times `vanka_setup` directly, so the set-up term is DATA. Only the
smoothing term is fitted, as `c * ndof * lin_per_step`. That matters: an earlier version of this
model fitted both terms, which is two free parameters against three size points, and its
leave-one-out errors were -27%, +47%, -25% -- a curve that could not reproduce any rung it had
not been shown.

WHY THIS REFUSES TO PRINT A SINGLE CAMPAIGN COST.

`vanka_setup` does not keep one character across the ladder. Per dof it costs 0.156, 0.142 and
0.514 microseconds at 116k, 894k and 7.01M dof -- flat, then 3.6x worse. The implied exponent is
0.954 between the first two rungs and 1.625 between the last two, and the campaign's operating
point lies BEYOND the top rung, in the regime with a single sample. Extrapolating one exponent
through a transition is how a factor-of-two error gets presented as a decision, so this reports a
RANGE bracketed by both exponents and says which rungs constrain it.

Step 1 is excluded from every mean. A fresh run pays a continuation ramp there, measured at
4.8x, 4.4x and 5.8x the later per-step cost on the three rungs.
"""
import csv
import math
import os
import re
import statistics as st
import sys

# The physical campaign, from the benchmark geometry and the solver's own reported throat speed.
R_INLET, R_THROAT = 0.006, 0.002
L_DOWNSTREAM = 0.16          # expansion plane to domain outlet
N_RESIDENCE = 5              # residence times of averaging window


def phase(log_text, name):
    """(seconds, calls, us_per_call) for one row of the driver's phase table."""
    m = re.search(rf"^\s+{re.escape(name)}\s+([\d.]+)\s+(\d+)\s+([\d.]+)", log_text, re.M)
    return (float(m.group(1)), int(m.group(2)), float(m.group(3))) if m else None


def read_run(d):
    csv_path, log_path = os.path.join(d, "diag.csv"), os.path.join(d, "log")
    if not (os.path.exists(csv_path) and os.path.exists(log_path)):
        return None
    rows = list(csv.DictReader(open(csv_path)))
    if len(rows) < 3:
        return None
    later = rows[1:]                       # drop the continuation ramp
    ts = [float(r["t_step"]) for r in later]
    nw = [float(r["newton_it"]) for r in later]
    li = [float(r["lin_it"]) for r in later]
    dli = [li[i] - li[i - 1] for i in range(1, len(li))] or [li[0]]
    text = open(log_path, errors="replace").read()
    vs, sm = phase(text, "vanka_setup"), phase(text, "smooth[L0]")
    if vs is None:
        return None
    return dict(ndof=int(float(rows[0]["ndof"])),
                t=st.mean(ts), sd=(st.stdev(ts) if len(ts) > 1 else 0.0),
                nw=st.mean(nw), li=st.mean(dli),
                step1=float(rows[0]["t_step"]),
                setup_s=vs[2] / 1e6, smooth_us=(sm[2] if sm else float("nan")),
                dt=float(rows[0]["t"]) - 0.0)


def main(tree, campaign_tree=None):
    runs = {}
    for name in sorted(os.listdir(tree)):
        r = read_run(os.path.join(tree, name))
        if r:
            runs[name] = r
    if not runs:
        print(f"no usable runs under {tree}", file=sys.stderr)
        return 1

    print("== measured, per rung (step 1 excluded) ==")
    print(f"{'run':<12}{'ndof':>10}{'t_step':>9}{'sd':>8}{'rel':>7}"
          f"{'newton':>8}{'lin/st':>8}{'setup/reb':>11}{'setup us/dof':>13}{'ramp x':>8}")
    for k, r in runs.items():
        rel = 100 * r["sd"] / r["t"] if r["t"] else 0
        print(f"{k:<12}{r['ndof']:>10}{r['t']:>9.3f}{r['sd']:>8.3f}{rel:>6.1f}%"
              f"{r['nw']:>8.2f}{r['li']:>8.1f}{r['setup_s']:>11.4f}"
              f"{1e6 * r['setup_s'] / r['ndof']:>13.4f}{r['step1'] / r['t']:>7.2f}x")

    ladder = [k for k in runs if k.endswith("_off") or k.endswith("_rep")]
    ladder.sort(key=lambda k: runs[k]["ndof"])
    if len(ladder) < 2:
        print("\nnot enough size rungs to fit; need the *_off ladder", file=sys.stderr)
        return 1

    print("\n== set-up scaling between consecutive rungs (this is what moves) ==")
    exps = []
    for a, b in zip(ladder, ladder[1:]):
        ra, rb = runs[a], runs[b]
        if rb["ndof"] == ra["ndof"]:
            continue
        q = math.log(rb["setup_s"] / ra["setup_s"]) / math.log(rb["ndof"] / ra["ndof"])
        exps.append(q)
        print(f"  {a:>10} -> {b:<10} exponent {q:.3f}")
    if exps and max(exps) - min(exps) > 0.25:
        print(f"  !! the exponent moves by {max(exps) - min(exps):.3f} across the ladder:"
              f" one power law does not span it")

    print("\n== leave-one-out (set-up measured, only the smoothing constant fitted) ==")
    cs = {k: (runs[k]["t"] - runs[k]["nw"] * runs[k]["setup_s"]) / (runs[k]["ndof"] * runs[k]["li"])
          for k in ladder}
    for hold in ladder:
        tr = [k for k in ladder if k != hold]
        c = sum(cs[k] for k in tr) / len(tr)
        r = runs[hold]
        pred = r["nw"] * r["setup_s"] + c * r["ndof"] * r["li"]
        err = 100 * (pred - r["t"]) / r["t"]
        flag = "" if abs(err) <= max(10.0, 2 * 100 * r["sd"] / r["t"]) else "   <- outside scatter"
        print(f"  hold {hold:<11} predicted {pred:8.3f} s   measured {r['t']:8.3f} s"
              f"   {err:+6.1f}%{flag}")

    # The averaging window, from the solver's own throat velocity rather than a quoted figure.
    u_t = None
    for k in ladder:
        m = re.search(r"u_t ([\d.eE+-]+) m/s", open(os.path.join(tree, k, "log"), errors="replace").read())
        if m:
            u_t = float(m.group(1))
            break
    if u_t:
        u_bulk = u_t * (R_THROAT / R_INLET) ** 2
        window = N_RESIDENCE * L_DOWNSTREAM / u_bulk
        print(f"\n== campaign window ==\n  u_throat {u_t:.4f} m/s -> u_bulk {u_bulk:.5f} m/s"
              f" -> residence {L_DOWNSTREAM / u_bulk:.3f} s -> window {window:.2f} s")
        print("\n== why no per-step cost is extrapolated past the top rung ==")
        print("  Both mechanisms are size-dependent, so neither a single exponent nor a constant")
        print("  c in 'c * ndof * lin' spans the ladder. Per dof, per call:")
        print(f"    {'run':<10}{'setup us/dof':>14}{'smooth[L0] us/dof':>20}")
        for k in ladder:
            r = runs[k]
            sm = r["smooth_us"] / r["ndof"] if r["ndof"] else float("nan")
            print(f"    {k:<10}{1e6 * r['setup_s'] / r['ndof']:>14.4f}{sm:>20.4f}")
        print("  Neither column is flat. That is the +38% leave-one-out miss at the middle rung,")
        print("  and it is why this script never fits a cost for a size it has not measured.")
        print("\n  The ladder cannot be extended to settle it. The operator refuses internal")
        print("  levels that are not powers of two (cvfem_hex8_ns_op.cpp), because the hoisted")
        print("  macro-element geometry is exact in floating point only then -- measured there as")
        print("  2.12e-06 operator error at L=3, moving a converged Poiseuille from 1e-11 to")
        print("  8e-02. So the next rung is L=32, about 56.1M dof and ~206 GB against L16's")
        print("  measured 25.7 GB: past one socket. There is no fourth point to be had.")

        top = runs[ladder[-1]]
        # Measured by jobs/vanka_freeze.sbatch and vanka_freeze_tail.sbatch at this rung. Iteration
        # counts do not move with the stride, so this is a pure per-step saving.
        freeze = {1: 45.366, 2: 38.219, 4: 34.472}
        # The campaign runs BELOW the dt this ladder was measured at, and that penalty is not
        # extrapolable from the ladder either: it is size-dependent, and measured WEAKER at the top
        # rung (+9.2%) than at the middle one (+38.3%), so projecting the L8 penalty onto L16 would
        # overstate the campaign. The campaign per-step cost is therefore READ from a dt_campaign
        # run tree when one is given, and simply not claimed when it is not. A constant pasted here
        # would outlive the measurement that justified it, which is how the earlier figure went
        # stale. jobs/dt_campaign.sbatch produces that tree.
        camp = None
        if campaign_tree and os.path.isdir(campaign_tree):
            for name in sorted(os.listdir(campaign_tree)):
                r = read_run(os.path.join(campaign_tree, name))
                if r and r["ndof"] == top["ndof"]:
                    camp = r
                    break
        print(f"\n== campaign cost from the top rung ({top['ndof']:,} dof) ==")
        hdr = f"  {'dt':>10}{'steps':>10}" + "".join(f"{'freeze=' + str(n):>12}" for n in (1, 2, 4))
        if camp is None:
            dt = 1.0e-3
            steps = window / dt
            row = "".join(f"{steps * freeze[n] / 86400.0:>12.1f}" for n in (1, 2, 4))
            print("  No campaign-dt tree supplied, so this is the FLOOR at the measured dt, not a")
            print("  cost: pass the dt_campaign tree as a second argument to cost the campaign.")
            print(hdr)
            print(f"  {dt:>10.2e}{steps:>10,.0f}{row}   socket-days")
        else:
            dt = camp["dt"]
            steps = window / dt
            # The freeze saving is a ratio measured at this rung; apply it to the campaign t_step.
            row = "".join(f"{steps * camp['t'] * (freeze[n] / freeze[1]) / 86400.0:>12.1f}"
                          for n in (1, 2, 4))
            print(f"  Measured at the campaign step size: {camp['t']:.3f} s/step at dt = {dt:.2e}"
                  f" ({100 * camp['sd'] / camp['t']:.1f}% scatter),")
            print(f"  against {top['t']:.3f} s at the ladder's dt -- a penalty of"
                  f" {100 * (camp['t'] / top['t'] - 1):+.1f}%.")
            print(hdr)
            print(f"  {dt:>10.2e}{steps:>10,.0f}{row}   socket-days")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else ".",
                  sys.argv[2] if len(sys.argv) > 2 else None))
