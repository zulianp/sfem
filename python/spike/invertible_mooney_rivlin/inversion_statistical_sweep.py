#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

_cache_root = os.path.join(tempfile.gettempdir(), "sfem_matplotlib_cache")
_xdg_root = os.path.join(tempfile.gettempdir(), "sfem_xdg_cache")
os.makedirs(_cache_root, exist_ok=True)
os.makedirs(os.path.join(_xdg_root, "fontconfig"), exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _cache_root)
os.environ.setdefault("XDG_CACHE_HOME", _xdg_root)

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ELASTIC_MODEL = os.path.join(HERE, "invertible_mooney_rivlin_plane_strain_2d.py")
KV_MODEL = os.path.join(HERE, "invertible_mooney_rivlin_kelvin_voigt_plane_strain_2d.py")


class RunTimeout(Exception):
    pass


def raise_timeout(signum, frame):
    raise RunTimeout()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.modules[os.path.splitext(os.path.basename(path))[0]] = module
    spec.loader.exec_module(module)
    return module


def run_elastic_homotopy_capped(
    m,
    X,
    triangles,
    x0,
    anchor_right,
    C10,
    C01,
    kappa,
    Jc_target,
    Jmin,
    stage_max_iter,
    final_max_iter,
    grad_tol,
    final_grad_tol,
):
    fixed = np.array([0, 1, 2 * anchor_right + 1], dtype=int)
    x = x0.copy()
    stages = [0.50, 0.35, 0.25, max(Jc_target, 0.20), Jc_target]
    all_hist = []
    use_numba = getattr(m, "NUMBA_AVAILABLE", False)
    mesh_data = m.prepare_mesh_data(X, triangles) if use_numba else None

    for Jc in stages:
        x, hist, ok = m.solve_stage(
            x,
            X,
            triangles,
            fixed,
            C10,
            C01,
            kappa,
            Jc,
            Jmin,
            max_iter=stage_max_iter,
            grad_tol=grad_tol,
            mesh_data=mesh_data,
            use_numba=use_numba,
            verbose=False,
        )
        all_hist.append((Jc, hist))
        if not ok:
            return x, all_hist, False

    x, hist, ok = m.solve_stage(
        x,
        X,
        triangles,
        fixed,
        C10,
        C01,
        kappa,
        Jc_target,
        Jmin,
        max_iter=final_max_iter,
        grad_tol=final_grad_tol,
        mesh_data=mesh_data,
        use_numba=use_numba,
        verbose=False,
    )
    all_hist.append((Jc_target, hist))
    return x, all_hist, ok


def one_run(args):
    (
        solver,
        seed,
        amp,
        initial_state,
        fold_noise_amplitude,
        fold_inversion_amplitude,
        nx,
        ny,
        C10,
        C01,
        kappa,
        Jc,
        Jmin,
        eta_s,
        eta_b,
        dt,
        steps,
        residual_tol,
        line_search_c,
        regularization_start,
        regularization_attempts,
        elastic_stage_max_iter,
        elastic_final_max_iter,
        elastic_grad_tol,
        elastic_final_grad_tol,
        max_runtime_s,
        energy_tol,
        x_error_tol,
        J_tol,
        max_inversion_ratio,
        max_inverted_fraction,
    ) = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    m = load_module(ELASTIC_MODEL, "inv_mr_model")
    t0 = time.time()
    X, tris, anchor_right = m.structured_square_mesh(nx, ny)
    if initial_state == "folded-random":
        x0, Js0, initial_deformation_scale = m.folded_random_initial_state(
            X,
            tris,
            amp,
            fold_inversion_amplitude,
            fold_noise_amplitude,
            seed,
            max_inversion_ratio=max_inversion_ratio,
            max_inverted_fraction=max_inverted_fraction,
            return_scale=True,
        )
    else:
        x0, Js0, initial_deformation_scale = m.random_deformed_initial_state(
            X,
            tris,
            amp,
            seed,
            max_inversion_ratio=max_inversion_ratio,
            max_inverted_fraction=max_inverted_fraction,
            return_scale=True,
        )
    n_inv0 = int(np.count_nonzero(Js0 < 0.0))
    inverted = Js0 < 0.0
    inversion_amplitudes = -Js0[inverted]
    if n_inv0:
        initial_inversion_amplitude_min = float(inversion_amplitudes.min())
        initial_inversion_amplitude_mean = float(inversion_amplitudes.mean())
        initial_inversion_amplitude_max = float(inversion_amplitudes.max())
    else:
        initial_inversion_amplitude_min = 0.0
        initial_inversion_amplitude_mean = 0.0
        initial_inversion_amplitude_max = 0.0
    timer_set = False
    old_handler = None

    try:
        if max_runtime_s > 0.0:
            import signal

            old_handler = signal.signal(signal.SIGALRM, raise_timeout)
            signal.setitimer(signal.ITIMER_REAL, max_runtime_s)
            timer_set = True

        if solver == "elastic":
            x, hist, converged = run_elastic_homotopy_capped(
                m,
                X,
                tris,
                x0,
                anchor_right,
                C10,
                C01,
                kappa,
                Jc,
                Jmin,
                elastic_stage_max_iter,
                elastic_final_max_iter,
                elastic_grad_tol,
                elastic_final_grad_tol,
            )
            nit = int(sum(len(hh) for _, hh in hist))
        else:
            kv = load_module(KV_MODEL, "inv_mr_kv_model")
            x, hist, _, converged = kv.run_dynamic_relaxation(
                X,
                tris,
                x0,
                anchor_right,
                C10,
                C01,
                kappa,
                eta_s,
                eta_b,
                dt,
                steps,
                Jc,
                Jmin,
                residual_tol,
                line_search_c,
                regularization_start,
                regularization_attempts,
                collect_snapshots=False,
                verbose=False,
            )
            nit = int(sum(len(hh) for _, hh in hist))

        use_numba = getattr(m, "NUMBA_AVAILABLE", False)
        mesh_data = m.prepare_mesh_data(X, tris) if use_numba else None
        E, g, _, Js = m.assemble(
            x,
            X,
            tris,
            C10,
            C01,
            kappa,
            Jc,
            Jmin,
            mesh_data=mesh_data,
            use_numba=use_numba,
        )
        fixed = np.array([0, 1, 2 * anchor_right + 1], dtype=int)
        free = np.setdiff1d(np.arange(2 * len(X)), fixed)
        gnorm = float(np.linalg.norm(g[free]))
        xerr = float(np.linalg.norm(x - X))
        recovered = bool(
            np.isfinite(E)
            and E < energy_tol
            and xerr < x_error_tol
            and Js.min() > 1.0 - J_tol
            and Js.max() < 1.0 + J_tol
        )
        success = recovered
        if recovered and converged:
            mode = "success"
        elif recovered:
            mode = "state_recovered_iter_limit"
        elif not converged:
            mode = "solver_not_converged"
        elif np.any(Js <= 0.0):
            mode = "final_inversion"
        elif E >= 1e-8 or xerr >= 1e-5:
            mode = "nonreference_local_minimum"
        else:
            mode = "other"
        row = dict(
            solver=solver,
            seed=seed,
            amplitude=amp,
            initial_state=initial_state,
            fold_noise_amplitude=fold_noise_amplitude if initial_state == "folded-random" else 0.0,
            fold_inversion_amplitude=fold_inversion_amplitude if initial_state == "folded-random" else 0.0,
            nx=nx,
            ny=ny,
            C10=C10,
            C01=C01,
            kappa=kappa,
            eta_s=eta_s if solver == "kelvin-voigt" else 0.0,
            eta_b=eta_b if solver == "kelvin-voigt" else 0.0,
            dt=dt if solver == "kelvin-voigt" else 0.0,
            pseudo_steps=steps if solver == "kelvin-voigt" else 0,
            initial_J_min=float(Js0.min()),
            initial_J_max=float(Js0.max()),
            initial_inverted=n_inv0,
            initial_inverted_fraction=n_inv0 / len(Js0),
            max_initial_inversion_ratio=max_inversion_ratio,
            max_initial_inverted_fraction=max_inverted_fraction,
            initial_deformation_scale=initial_deformation_scale,
            initial_inversion_amplitude_min=initial_inversion_amplitude_min,
            initial_inversion_amplitude_mean=initial_inversion_amplitude_mean,
            initial_inversion_amplitude_max=initial_inversion_amplitude_max,
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
    except RunTimeout:
        row = dict(
            solver=solver,
            seed=seed,
            amplitude=amp,
            initial_state=initial_state,
            fold_noise_amplitude=fold_noise_amplitude if initial_state == "folded-random" else 0.0,
            fold_inversion_amplitude=fold_inversion_amplitude if initial_state == "folded-random" else 0.0,
            nx=nx,
            ny=ny,
            C10=C10,
            C01=C01,
            kappa=kappa,
            eta_s=eta_s if solver == "kelvin-voigt" else 0.0,
            eta_b=eta_b if solver == "kelvin-voigt" else 0.0,
            dt=dt if solver == "kelvin-voigt" else 0.0,
            pseudo_steps=steps if solver == "kelvin-voigt" else 0,
            initial_J_min=float(Js0.min()),
            initial_J_max=float(Js0.max()),
            initial_inverted=n_inv0,
            initial_inverted_fraction=n_inv0 / len(Js0),
            max_initial_inversion_ratio=max_inversion_ratio,
            max_initial_inverted_fraction=max_inverted_fraction,
            initial_deformation_scale=initial_deformation_scale,
            initial_inversion_amplitude_min=initial_inversion_amplitude_min,
            initial_inversion_amplitude_mean=initial_inversion_amplitude_mean,
            initial_inversion_amplitude_max=initial_inversion_amplitude_max,
            converged=False,
            success=False,
            failure_mode="timeout",
            final_energy=np.nan,
            final_J_min=np.nan,
            final_J_max=np.nan,
            final_grad_norm=np.nan,
            final_x_error=np.nan,
            iterations=0,
            runtime_s=time.time() - t0,
        )
    except Exception as exc:
        row = dict(
            solver=solver,
            seed=seed,
            amplitude=amp,
            initial_state=initial_state,
            fold_noise_amplitude=fold_noise_amplitude if initial_state == "folded-random" else 0.0,
            fold_inversion_amplitude=fold_inversion_amplitude if initial_state == "folded-random" else 0.0,
            nx=nx,
            ny=ny,
            C10=C10,
            C01=C01,
            kappa=kappa,
            eta_s=eta_s if solver == "kelvin-voigt" else 0.0,
            eta_b=eta_b if solver == "kelvin-voigt" else 0.0,
            dt=dt if solver == "kelvin-voigt" else 0.0,
            pseudo_steps=steps if solver == "kelvin-voigt" else 0,
            initial_J_min=float(Js0.min()),
            initial_J_max=float(Js0.max()),
            initial_inverted=n_inv0,
            initial_inverted_fraction=n_inv0 / len(Js0),
            max_initial_inversion_ratio=max_inversion_ratio,
            max_initial_inverted_fraction=max_inverted_fraction,
            initial_deformation_scale=initial_deformation_scale,
            initial_inversion_amplitude_min=initial_inversion_amplitude_min,
            initial_inversion_amplitude_mean=initial_inversion_amplitude_mean,
            initial_inversion_amplitude_max=initial_inversion_amplitude_max,
            converged=False,
            success=False,
            failure_mode="exception:" + type(exc).__name__ + ":" + str(exc),
            final_energy=np.nan,
            final_J_min=np.nan,
            final_J_max=np.nan,
            final_grad_norm=np.nan,
            final_x_error=np.nan,
            iterations=0,
            runtime_s=time.time() - t0,
        )
    finally:
        if timer_set:
            import signal

            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, old_handler)
    return row


def percentile(values, p):
    vals = [float(v) for v in values if np.isfinite(v)]
    if not vals:
        return np.nan
    return float(np.percentile(np.asarray(vals), p))


def mean(values):
    vals = [float(v) for v in values if np.isfinite(v)]
    if not vals:
        return np.nan
    return float(np.mean(vals))


def group_rows(rows, key):
    groups = {}
    for row in rows:
        groups.setdefault(key(row), []).append(row)
    return groups


def inversion_bin(frac, bins):
    for i in range(len(bins) - 1):
        if bins[i] <= frac < bins[i + 1]:
            return i
    if frac == bins[-1]:
        return len(bins) - 2
    return None


def summarize_group(rows, label):
    n = len(rows)
    ok = sum(bool(r["success"]) for r in rows)
    inv_frac = [r["initial_inverted_fraction"] for r in rows]
    iters = [r["iterations"] for r in rows if r["success"]]
    runtime = [r["runtime_s"] for r in rows]
    return dict(
        group=label,
        runs=n,
        successes=ok,
        success_rate=ok / n if n else np.nan,
        mean_initial_inverted_fraction=mean(inv_frac),
        median_initial_J_min=percentile([r["initial_J_min"] for r in rows], 50),
        median_iterations=percentile(iters, 50),
        p90_iterations=percentile(iters, 90),
        mean_runtime_s=mean(runtime),
        max_runtime_s=percentile(runtime, 100),
    )


def write_summary_csv(path, rows, bins):
    summaries = []
    for amp, group in sorted(group_rows(rows, lambda r: r["amplitude"]).items()):
        summaries.append(summarize_group(group, f"amplitude={amp:g}"))

    for i in range(len(bins) - 1):
        lo = bins[i]
        hi = bins[i + 1]
        group = [
            r
            for r in rows
            if (idx := inversion_bin(r["initial_inverted_fraction"], bins)) is not None
            and idx == i
        ]
        if group:
            summaries.append(summarize_group(group, f"inverted_fraction=[{lo:g},{hi:g})"))

    fields = list(summaries[0].keys()) if summaries else []
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summaries)


def print_summary(rows, amplitudes, bins):
    print("\nAMPLITUDE SUMMARY")
    print("amp      runs  success  mean inv%  median Jmin  median iters  p90 iters")
    for amp in amplitudes:
        group = [r for r in rows if r["amplitude"] == amp]
        s = summarize_group(group, f"amplitude={amp:g}")
        print(
            f"{amp:7.3f} {s['runs']:5d} "
            f"{s['successes']:3d}/{s['runs']:<3d} "
            f"{100.0 * s['mean_initial_inverted_fraction']:8.2f} "
            f"{s['median_initial_J_min']:11.3f} "
            f"{s['median_iterations']:13.1f} "
            f"{s['p90_iterations']:10.1f}"
        )

    print("\nINITIAL INVERSION FRACTION SUMMARY")
    print("bin inv%        runs  success  median Jmin  median iters  p90 iters")
    for i in range(len(bins) - 1):
        lo = bins[i]
        hi = bins[i + 1]
        group = [
            r
            for r in rows
            if (idx := inversion_bin(r["initial_inverted_fraction"], bins)) is not None
            and idx == i
        ]
        if not group:
            continue
        s = summarize_group(group, f"inverted_fraction=[{lo:g},{hi:g})")
        print(
            f"[{100*lo:5.1f},{100*hi:5.1f}) "
            f"{s['runs']:5d} {s['successes']:3d}/{s['runs']:<3d} "
            f"{s['median_initial_J_min']:11.3f} "
            f"{s['median_iterations']:13.1f} "
            f"{s['p90_iterations']:10.1f}"
        )

    failures = [r for r in rows if not r["success"]]
    print("\nFAILURES")
    if not failures:
        print("none")
    else:
        for r in failures:
            print(
                f"amp={r['amplitude']:.3f} seed={r['seed']} "
                f"mode={r['failure_mode']} inv0={r['initial_inverted']} "
                f"J0min={r['initial_J_min']:.3f} E={r['final_energy']:.3e} "
                f"iters={r['iterations']}"
            )


def write_plot(path, rows, bins):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = np.asarray([bool(r["success"]) for r in rows])
    timeout = np.asarray([r["failure_mode"] == "timeout" for r in rows])
    inv = np.asarray([r["initial_inverted_fraction"] for r in rows])
    failed = ~ok
    amplitude_fields = (
        ("initial_inversion_amplitude_min", "minimum"),
        ("initial_inversion_amplitude_mean", "mean"),
        ("initial_inversion_amplitude_max", "maximum"),
    )
    amplitudes = [np.asarray([r[field] for r in rows], dtype=float) for field, _ in amplitude_fields]
    fraction_caps = [
        float(r["max_initial_inverted_fraction"])
        for r in rows
        if r.get("max_initial_inverted_fraction") is not None
    ]
    x_limit = min(fraction_caps) if fraction_caps else 1.0
    x_limit = max(x_limit, float(inv.max()))
    x_padding = max(0.01, 0.02 * x_limit)

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.8), sharex=True)
    for ax, values, (_, label) in zip(axs, amplitudes, amplitude_fields):
        ax.scatter(
            inv[ok],
            values[ok],
            s=50,
            c="#1b7f3a",
            edgecolors="white",
            linewidths=0.6,
            label="recovered",
            zorder=3,
        )
        if np.any(failed):
            ax.scatter(
                inv[failed],
                values[failed],
                s=52,
                c="#b42318",
                marker="x",
                linewidths=1.6,
                label="not recovered",
                zorder=4,
        )
        ax.set_title(f"{label.capitalize()} inversion amplitude")
        ax.set_xlabel("initial inverted-element fraction")
        ax.set_xlim(-x_padding, x_limit + x_padding)
        panel_ymax = max(1.0e-12, 1.05 * float(values.max()))
        ax.set_ylim(-0.02 * panel_ymax, panel_ymax)
        ax.set_ylabel(r"initial inversion amplitude $a=-J_0$ ($J_0<0$)")
        ax.grid(True, alpha=0.35)

    axs[0].legend(loc="upper left")
    fig.suptitle("Mooney-Rivlin inversion-recovery outcomes")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", choices=["elastic", "kelvin-voigt"], default="elastic")
    ap.add_argument(
        "--initial-state",
        choices=["random", "folded-random"],
        default="random",
    )
    ap.add_argument("--fold-noise-amplitude", type=float, default=0.04)
    ap.add_argument(
        "--fold-inversion-amplitudes", type=float, nargs="+", default=[1.0]
    )
    ap.add_argument("--nx", type=int, default=8)
    ap.add_argument("--ny", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=60)
    ap.add_argument(
        "--amplitudes",
        type=float,
        nargs="+",
        default=[0.25, 0.50, 0.75, 1.00, 1.25],
    )
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--C10", type=float, default=0.35)
    ap.add_argument("--C01", type=float, default=0.15)
    ap.add_argument("--kappa", type=float, default=500.0)
    ap.add_argument("--eta-s", type=float, default=0.05)
    ap.add_argument("--eta-b", type=float, default=0.005)
    ap.add_argument("--dt", type=float, default=0.25)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--residual-tol", type=float, default=1.0e-3)
    ap.add_argument("--line-search-c", type=float, default=1.0e-4)
    ap.add_argument("--regularization-start", type=float, default=1.0e-10)
    ap.add_argument("--regularization-attempts", type=int, default=8)
    ap.add_argument("--elastic-stage-max-iter", type=int, default=80)
    ap.add_argument("--elastic-final-max-iter", type=int, default=120)
    ap.add_argument("--elastic-grad-tol", type=float, default=1.0e-9)
    ap.add_argument("--elastic-final-grad-tol", type=float, default=1.0e-11)
    ap.add_argument("--max-runtime-s", type=float, default=0.0)
    ap.add_argument("--Jc", type=float, default=0.2)
    ap.add_argument("--Jmin", type=float, default=-1.0)
    ap.add_argument("--csv", default="inversion_mooney_rivlin_stats.csv")
    ap.add_argument("--summary-csv", default="inversion_mooney_rivlin_summary.csv")
    ap.add_argument("--plot", default="inversion_mooney_rivlin_robustness.png")
    ap.add_argument(
        "--inversion-bins",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.15, 0.30, 0.50, 0.75, 1.01],
    )
    ap.add_argument("--energy-tol", type=float, default=1.0e-6)
    ap.add_argument("--x-error-tol", type=float, default=1.0e-3)
    ap.add_argument("--J-tol", type=float, default=1.0e-3)
    ap.add_argument(
        "--max-inversion-ratio",
        type=float,
        default=10.0,
        help="cap initial physical inversion magnitude at -J <= this value; use 0 to disable",
    )
    ap.add_argument(
        "--max-inverted-fraction",
        type=float,
        default=0.5,
        help="cap the initial fraction of inverted elements; use 1 to disable",
    )
    args = ap.parse_args()

    bins = sorted(args.inversion_bins)
    if len(bins) < 2 or bins[0] > 0.0 or bins[-1] < 1.0:
        raise SystemExit("--inversion-bins must cover [0, 1]")
    if args.max_inversion_ratio < 0.0:
        raise SystemExit("--max-inversion-ratio must be nonnegative")
    if not 0.0 < args.max_inverted_fraction <= 1.0:
        raise SystemExit("--max-inverted-fraction must lie in (0, 1]")
    max_inversion_ratio = args.max_inversion_ratio if args.max_inversion_ratio > 0.0 else None
    max_inverted_fraction = (
        args.max_inverted_fraction if args.max_inverted_fraction < 1.0 else None
    )

    jobs = [
        (
            args.solver,
            seed,
            amp,
            args.initial_state,
            args.fold_noise_amplitude,
            fold_inversion_amplitude,
            args.nx,
            args.ny,
            args.C10,
            args.C01,
            args.kappa,
            args.Jc,
            args.Jmin,
            args.eta_s,
            args.eta_b,
            args.dt,
            args.steps,
            args.residual_tol,
            args.line_search_c,
            args.regularization_start,
            args.regularization_attempts,
            args.elastic_stage_max_iter,
            args.elastic_final_max_iter,
            args.elastic_grad_tol,
            args.elastic_final_grad_tol,
            args.max_runtime_s,
            args.energy_tol,
            args.x_error_tol,
            args.J_tol,
            max_inversion_ratio,
            max_inverted_fraction,
        )
        for amp in args.amplitudes
        for fold_inversion_amplitude in args.fold_inversion_amplitudes
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
                f"iters={row['iterations']}",
                flush=True,
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
                    f"iters={row['iterations']}",
                    flush=True,
                )

    rows.sort(key=lambda r: (r["amplitude"], r["seed"]))
    fields = list(rows[0].keys())
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    write_summary_csv(args.summary_csv, rows, bins)
    if args.plot:
        write_plot(args.plot, rows, bins)

    print_summary(rows, args.amplitudes, bins)
    print(f"\nWrote raw CSV: {args.csv}")
    print(f"Wrote summary CSV: {args.summary_csv}")
    if args.plot:
        print(f"Wrote plot: {args.plot}")


if __name__ == "__main__":
    main()
