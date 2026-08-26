#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "invertible_mooney_rivlin_plane_strain_2d.py")


def load_model():
    spec = importlib.util.spec_from_file_location("inv_mr_model", MODEL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def one_run(args):
    seed, amp, nx, ny, C10, C01, kappa, Jc, Jmin = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    m = load_model()
    t0 = time.time()
    X, tris, anchor_right = m.structured_square_mesh(nx, ny)
    x0, Js0 = m.random_deformed_initial_state(X, tris, amp, seed)
    n_inv0 = int(np.count_nonzero(Js0 < 0.0))

    try:
        x, hist, converged = m.run_homotopy(
            X, tris, x0, anchor_right, C10, C01, kappa, Jc, Jmin, verbose=False
        )
        E, g, _, Js = m.assemble(x, X, tris, C10, C01, kappa, Jc, Jmin)
        fixed = np.array([0, 1, 2 * anchor_right + 1], dtype=int)
        free = np.setdiff1d(np.arange(2 * len(X)), fixed)
        gnorm = float(np.linalg.norm(g[free]))
        xerr = float(np.linalg.norm(x - X))
        nit = int(sum(len(hh) for _, hh in hist))
        success = bool(
            converged
            and np.isfinite(E)
            and E < 1e-8
            and xerr < 1e-5
            and Js.min() > 0.9999
            and Js.max() < 1.0001
        )
        if success:
            mode = "success"
        elif not converged:
            mode = "solver_not_converged"
        elif np.any(Js <= 0.0):
            mode = "final_inversion"
        elif E >= 1e-8 or xerr >= 1e-5:
            mode = "nonreference_local_minimum"
        else:
            mode = "other"
        row = dict(
            seed=seed,
            amplitude=amp,
            nx=nx,
            ny=ny,
            C10=C10,
            C01=C01,
            kappa=kappa,
            initial_J_min=float(Js0.min()),
            initial_J_max=float(Js0.max()),
            initial_inverted=n_inv0,
            initial_inverted_fraction=n_inv0 / len(Js0),
            converged=bool(converged),
            success=success,
            failure_mode=mode,
            final_energy=float(E),
            final_J_min=float(Js.min()),
            final_J_max=float(Js.max()),
            final_grad_norm=gnorm,
            final_x_error=xerr,
            iterations=nit,
            runtime_s=time.time() - t0,
        )
    except Exception as exc:
        row = dict(
            seed=seed,
            amplitude=amp,
            nx=nx,
            ny=ny,
            C10=C10,
            C01=C01,
            kappa=kappa,
            initial_J_min=float(Js0.min()),
            initial_J_max=float(Js0.max()),
            initial_inverted=n_inv0,
            initial_inverted_fraction=n_inv0 / len(Js0),
            converged=False,
            success=False,
            failure_mode="exception:" + type(exc).__name__,
            final_energy=np.nan,
            final_J_min=np.nan,
            final_J_max=np.nan,
            final_grad_norm=np.nan,
            final_x_error=np.nan,
            iterations=0,
            runtime_s=time.time() - t0,
        )
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=8)
    ap.add_argument("--ny", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=60)
    ap.add_argument("--amplitudes", type=float, nargs="+", default=[0.5, 0.75, 1.0])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--C10", type=float, default=0.35)
    ap.add_argument("--C01", type=float, default=0.15)
    ap.add_argument("--kappa", type=float, default=500.0)
    ap.add_argument("--Jc", type=float, default=0.2)
    ap.add_argument("--Jmin", type=float, default=-1.0)
    ap.add_argument("--csv", default="inversion_mooney_rivlin_stats.csv")
    args = ap.parse_args()

    jobs = [
        (seed, amp, args.nx, args.ny, args.C10, args.C01, args.kappa, args.Jc, args.Jmin)
        for amp in args.amplitudes
        for seed in range(args.seeds)
    ]

    rows = []
    if args.workers <= 1:
        for k, job in enumerate(jobs, 1):
            row = one_run(job)
            rows.append(row)
            print(
                f"[{k:3d}/{len(jobs)}] amp={row['amplitude']:.2f} "
                f"seed={row['seed']:3d} success={row['success']} "
                f"mode={row['failure_mode']} "
                f"J0min={row['initial_J_min']:.3f} "
                f"inv0={row['initial_inverted']:3d} "
                f"E={row['final_energy']:.3e} "
                f"iters={row['iterations']}"
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(one_run, j): j for j in jobs}
            for k, fut in enumerate(as_completed(futs), 1):
                row = fut.result()
                rows.append(row)
                print(
                    f"[{k:3d}/{len(jobs)}] amp={row['amplitude']:.2f} "
                    f"seed={row['seed']:3d} success={row['success']} "
                    f"mode={row['failure_mode']} "
                    f"J0min={row['initial_J_min']:.3f} "
                    f"inv0={row['initial_inverted']:3d} "
                    f"E={row['final_energy']:.3e} "
                    f"iters={row['iterations']}"
                )

    rows.sort(key=lambda r: (r["amplitude"], r["seed"]))
    fields = list(rows[0].keys())
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print("\nFAILURES")
    for amp in args.amplitudes:
        bad = [r["seed"] for r in rows if r["amplitude"] == amp and not r["success"]]
        ok = sum(r["success"] for r in rows if r["amplitude"] == amp)
        n = sum(1 for r in rows if r["amplitude"] == amp)
        print(f"amp={amp:.2f}: success {ok}/{n}; unsuccessful seeds={bad}")


if __name__ == "__main__":
    main()
