#!/usr/bin/env python3
"""Dual-pass unbiased augmented-Lagrangian contact with Neo-Hookean MMG.

This variant keeps the nonlinear multigrid framework from
``mmg_nitsche_neohookean.py`` but replaces the Nitsche contact law with a
Solberg-Puso-style dual-pass normal constraint.  Contact rows are integrated on
both surfaces with equal weights in the two-body case, and each directed row
stores a positive compressive augmented-Lagrangian multiplier.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

import numpy as np
from scipy import sparse

_SPIKE = os.path.dirname(os.path.abspath(__file__))
if _SPIKE not in sys.path:
    sys.path.insert(0, _SPIKE)

import mmg_nitsche as mg
import mmg_nitsche_neohookean as neo
import nitsche_contact as nc


def _row_weight(level, surface_id):
    return float(level.theta_b if surface_id == 0 else level.theta_o)


def _surface_rows(
    level,
    u,
    surface_id,
    frozen_edges=None,
):
    if surface_id == 0:
        edges = level.edges_b
        parent_elems = level.elems_b
        parent_block = level.bid_b
        other_edges = level.edges_o
        mu = level.mu_b
        lam = level.lam_b
        snap_self_circle = False
    else:
        edges = level.edges_o
        parent_elems = level.elems_o
        parent_block = level.bid_o
        other_edges = level.edges_b
        mu = level.mu_o
        lam = level.lam_o
        snap_self_circle = True

    rows = []
    X, Y = level.X, level.Y
    for ie, (edge, e_parent) in enumerate(zip(edges, parent_elems)):
        n0, n1 = int(edge[0]), int(edge[1])
        length, nx0, ny0 = nc.edge_geometry(X, Y, n0, n1)[:3]
        if length <= 1e-16:
            continue
        parent_nodes = nc.tri3_parent_nodes(level.ps, level.mesh, parent_block, int(e_parent))
        px = np.array([X[i] for i in parent_nodes], dtype=np.float64)
        py = np.array([Y[i] for i in parent_nodes], dtype=np.float64)
        u_elem = np.empty(6, dtype=np.float64)
        for a, node in enumerate(parent_nodes):
            u_elem[2 * a] = u[2 * node]
            u_elem[2 * a + 1] = u[2 * node + 1]
        if frozen_edges is not None and ie in frozen_edges:
            samples = nc.eval_frozen_edge_qps(u, n0, n1, frozen_edges[ie])
        else:
            samples = nc.collect_edge_qps(
                X,
                Y,
                u,
                n0,
                n1,
                length,
                nx0,
                ny0,
                other_edges,
                level.radius,
                snap_self_circle,
            )
        if not samples:
            continue

        w_int = 0.0
        g_int = 0.0
        dg_bar = {}
        mid = samples[0]

        def add_dg(dof, val):
            if val != 0.0:
                dg_bar[dof] = dg_bar.get(dof, 0.0) + val

        for w, xi, g, nx, ny, _tx, _ty, Na, Nb, m0, m1, Nm0, Nm1, _xref in samples:
            w_int += w
            g_int += w * g
            if abs(xi - 0.5) < abs(mid[1] - 0.5):
                mid = (w, xi, g, nx, ny, _tx, _ty, Na, Nb, m0, m1, Nm0, Nm1, _xref)
            add_dg(2 * n0, w * (-Na * nx))
            add_dg(2 * n0 + 1, w * (-Na * ny))
            add_dg(2 * n1, w * (-Nb * nx))
            add_dg(2 * n1 + 1, w * (-Nb * ny))
            add_dg(2 * m0, w * (Nm0 * nx))
            add_dg(2 * m0 + 1, w * (Nm0 * ny))
            add_dg(2 * m1, w * (Nm1 * nx))
            add_dg(2 * m1 + 1, w * (Nm1 * ny))

        inv_w = 1.0 / w_int
        g_bar = g_int * inv_w
        for dof in list(dg_bar.keys()):
            dg_bar[dof] *= inv_w

        nx, ny = float(mid[3]), float(mid[4])
        gamma = neo.contact_penalty_gamma(
            length,
            px,
            py,
            u_elem,
            mu,
            lam,
            level.al_penalty,
            nx,
            ny,
            level.fd_eps,
            level.contact_penalty_scaling,
        )
        sn = neo.tri3_neo_sigma_n(px, py, u_elem, mu, lam, nx, ny)
        key = (surface_id, ie, 0)
        rows.append(
            {
                "key": key,
                "x": float(0.5 * (X[n0] + X[n1])),
                "g": float(g_bar),
                "dg": dg_bar,
                "w": float(w_int),
                "gamma": float(gamma),
                "sn": float(sn),
            }
        )
    return rows


def al_contact_residual_tangent(level, u_vec, forced_active=None, frozen_geom=None):
    r_elastic, K_elastic = level.elastic_residual_tangent(u_vec)
    g_contact = np.zeros(level.ndofs, dtype=np.float64)
    coo = []
    new_active = set()
    n_rows = 0

    for surface_id in (0, 1):
        weight = _row_weight(level, surface_id)
        if weight <= 0.0:
            continue
        frozen_edges = None if frozen_geom is None else frozen_geom.get(surface_id)
        rows = _surface_rows(level, u_vec, surface_id, frozen_edges)
        for row in rows:
            key = row["key"]
            lam_old = float(level.al_multipliers.get(key, 0.0))
            pressure_trial = lam_old - row["gamma"] * row["g"]
            active = pressure_trial > 0.0 if forced_active is None else key in forced_active
            pressure = max(pressure_trial, 0.0)
            if not active:
                continue
            new_active.add(key)
            n_rows += 1
            scale = weight * row["w"]
            for dof, dgd in row["dg"].items():
                g_contact[dof] -= scale * pressure * dgd
            for di, dgi in row["dg"].items():
                for dj, dgj in row["dg"].items():
                    val = scale * row["gamma"] * dgi * dgj
                    if val != 0.0:
                        coo.append((di, dj, val))

    if coo:
        ii, jj, vv = zip(*coo)
        K_contact = sparse.coo_matrix((vv, (ii, jj)), shape=(level.ndofs, level.ndofs)).tocsr()
    else:
        K_contact = sparse.csr_matrix((level.ndofs, level.ndofs))
    residual = r_elastic + g_contact
    Ksys = K_elastic + K_contact
    Ksys, residual = nc.apply_dirichlet_system(
        Ksys, residual, level.constrained, u_vec, level.u_bc
    )
    return residual, Ksys, g_contact, K_contact, n_rows, new_active


def update_al_multipliers(level, u_vec, relaxation):
    changed2 = 0.0
    lambda2 = 0.0
    active = set()
    max_violation = 0.0
    rows_seen = 0
    old = dict(level.al_multipliers)
    new = {}
    for surface_id in (0, 1):
        if _row_weight(level, surface_id) <= 0.0:
            continue
        for row in _surface_rows(level, u_vec, surface_id):
            rows_seen += 1
            key = row["key"]
            lam_old = float(old.get(key, 0.0))
            projected = max(0.0, lam_old - row["gamma"] * row["g"])
            lam_new = (1.0 - relaxation) * lam_old + relaxation * projected
            if lam_new > level.al_drop_tol:
                new[key] = lam_new
                active.add(key)
            diff = lam_new - lam_old
            changed2 += diff * diff
            lambda2 += lam_new * lam_new
            max_violation = max(max_violation, max(0.0, -row["g"]))

    for key, lam_old in old.items():
        if key not in new:
            changed2 += lam_old * lam_old
    level.al_multipliers = new
    return {
        "lambda_change": float(np.sqrt(changed2)),
        "lambda_norm": float(np.sqrt(lambda2)),
        "max_violation": float(max_violation),
        "n_lambda": int(len(new)),
        "n_rows": int(rows_seen),
        "active": active,
    }


def build_level(ps, args):
    level = neo.build_level(ps, args)
    level.al_penalty = float(getattr(args, "al_penalty", args.gamma0))
    level.al_drop_tol = float(getattr(args, "al_drop_tol", 1e-14))
    level.al_multipliers = {}
    level.include_sigma = False
    level.residual_tangent = lambda u, forced_active=None, frozen_geom=None: (
        al_contact_residual_tangent(level, u, forced_active, frozen_geom)
    )
    return level


def solve_mmg(ps, args, initial_u=None):
    if args.max_inner_it < 1:
        raise ValueError("--max-inner-it must be positive")
    if args.nlsmooth_steps < 0 or args.mg_pre < 0 or args.mg_post < 0:
        raise ValueError("smoothing step counts must be non-negative")
    relaxation = float(getattr(args, "al_relaxation", 1.0))
    if not (0.0 < relaxation <= 1.0):
        raise ValueError("--al-relaxation must be in (0, 1]")

    sizes = mg.hierarchy_sizes(args.nx, args.ny, args.levels)
    levels = []
    for nxk, nyk, nyb in sizes:
        a = copy.copy(args)
        a.nx, a.ny, a.ny_block = nxk, nyk, nyb
        levels.append(build_level(ps, a))
        print(
            f"level nx={nxk} ny={nyk} ny_block={nyb} nodes={levels[-1].mesh.n_nodes()} "
            f"dofs={levels[-1].ndofs}"
        )
    prolong = [mg.hertz_prolongation(levels[k + 1], levels[k]) for k in range(len(levels) - 1)]
    fine = levels[0]
    if initial_u is None:
        u = fine.u_bc.copy()
    else:
        u = np.asarray(initial_u, dtype=np.float64).copy()
        if u.size != fine.ndofs:
            raise ValueError(f"initial displacement has {u.size} dofs, expected {fine.ndofs}")
        u[fine.constrained] = fine.u_bc[fine.constrained]

    r_hist = []
    n_active_hist = []
    n_qp_hist = []
    vcycle_hist = []
    direct_hist = []
    al_hist = []
    residual, A, g_c, Kn, n_qp, active, contact_mask, masks = mg.filtered_hierarchy(
        levels, prolong, fine, u
    )
    residual_norm_0 = max(float(np.linalg.norm(residual)), 1e-30)
    count_inner_iter = 0
    count_smoothing_steps = 0
    sweep_factor = 2 if args.smoother == "sgs" else 1
    sweeps_per_cycle = args.nlsmooth_steps * (args.mg_pre + args.mg_post) * sweep_factor

    for it in range(args.max_iter):
        rnorm_previous = 1e300
        inner_used = 0
        stagnation = False
        for inner in range(args.max_inner_it):
            residual, A, g_c, Kn, n_qp, active, contact_mask, masks = mg.nonlinear_cycle(
                levels, prolong, fine, u, args
            )
            count_inner_iter += 1
            inner_used += 1
            count_smoothing_steps += sweeps_per_cycle
            rnorm = float(np.linalg.norm(residual))
            rel = rnorm / residual_norm_0
            stagnation = abs(rnorm / rnorm_previous) > args.stagnation_threshold
            rnorm_previous = rnorm
            n_filt = [int(np.count_nonzero(m)) for m in masks]
            n_contact = int(np.count_nonzero(contact_mask & ~mg.dirichlet_mask(fine)))
            print(
                f"al-mmg[{it:02d}:{inner:02d}] ||r||={rnorm:.3e}  ||r/r0||={rel:.3e}  "
                f"|A|={len(active)}  al_rows={n_qp}  contact_dofs={n_contact}  "
                f"filt_dofs={n_filt}  contact_nnz={Kn.nnz}"
            )
            if ((rnorm < args.atol or rel < args.rtol) and inner != 0) or stagnation:
                break

        al_stats = update_al_multipliers(fine, u, relaxation)
        residual, A, g_c, Kn, n_qp, active, contact_mask, masks = mg.filtered_hierarchy(
            levels, prolong, fine, u
        )
        rnorm = float(np.linalg.norm(residual))
        rel = rnorm / residual_norm_0
        norm_pen = mg.contact_penetration_norm(fine, u)
        r_hist.append(rnorm)
        n_active_hist.append(len(active))
        n_qp_hist.append(n_qp)
        vcycle_hist.append(int(inner_used))
        direct_hist.append(False)
        al_hist.append(al_stats)
        print(
            f"          inner={count_inner_iter}  smooth={count_smoothing_steps}  "
            f"|g_-|={norm_pen:.3e}  |lambda|={al_stats['lambda_norm']:.3e}  "
            f"|d_lambda|={al_stats['lambda_change']:.3e}  max_pen={al_stats['max_violation']:.3e}  "
            f"stagnation={stagnation}"
        )
        residual_converged = rnorm < args.atol or rel < args.rtol
        lambda_converged = al_stats["lambda_change"] < float(args.al_lambda_atol)
        penetration_converged = norm_pen < args.ptol
        if residual_converged and lambda_converged and penetration_converged:
            break

    residual, A, g_c, Kn, n_qp, active = fine.residual_tangent(u, forced_active=None)
    r_final = float(np.linalg.norm(residual))
    if not r_hist or abs(r_hist[-1] - r_final) > 1e-30:
        r_hist.append(r_final)
        n_active_hist.append(len(active))
        n_qp_hist.append(n_qp)
        vcycle_hist.append(0)
        direct_hist.append(False)
        print(f"al-mmg[final] ||r||={r_final:.3e}  |A|={len(active)}")

    result = pack_result(
        fine, args, u, r_hist, n_active_hist, n_qp_hist, vcycle_hist, direct_hist
    )
    result["al_hist"] = al_hist
    print(
        f"nodes={fine.mesh.n_nodes()} dofs={fine.ndofs} F={result['F']:.4e} "
        f"a_hertz={result['a']:.4e} p0={result['p0']:.4e} |g_-|={result['penetration']:.3e}  "
        f"al_gap_l2={result['al_gap_l2']:.3e} al_min_gap={result['al_min_gap']:.3e}  "
        f"dual_pass_weights=({fine.theta_b:g},{fine.theta_o:g})  n_active={result['n_active']}  "
        f"|lambda|={result['al_lambda_norm']:.3e}"
    )
    return result


def contact_trace_al(level, u, surface_id):
    xs, gs, sns, pns, active, xis, ws = [], [], [], [], [], [], []
    for row in _surface_rows(level, u, surface_id):
        lam_old = float(level.al_multipliers.get(row["key"], 0.0))
        pressure = max(0.0, lam_old - row["gamma"] * row["g"])
        xs.append(row["x"])
        gs.append(row["g"])
        sns.append(row["sn"])
        pns.append(-pressure)
        active.append(pressure > 0.0)
        xis.append(0.5)
        ws.append(row["w"])
    return (
        np.asarray(xs),
        np.asarray(gs),
        np.asarray(sns),
        np.asarray(pns),
        np.asarray(active, dtype=bool),
        np.asarray(xis),
        np.asarray(ws),
    )


def pack_result(fine, args, u, r_hist, n_active_hist, n_qp_hist, vcycle_hist=None, direct_hist=None):
    result = neo.pack_result(fine, args, u, r_hist, n_active_hist, n_qp_hist, vcycle_hist, direct_hist)
    tr_x, tr_g, tr_sn, tr_pn, tr_on, tr_xi, tr_w = contact_trace_al(fine, u, 0)
    tr_x_o, tr_g_o, tr_sn_o, tr_pn_o, tr_on_o, _, tr_w_o = contact_trace_al(fine, u, 1)
    p_applied = np.maximum(-tr_pn, 0.0)
    g_minus = np.minimum(tr_g, 0.0)
    g_minus_o = np.minimum(tr_g_o, 0.0)
    al_gap_l2 = 0.0
    if tr_w.size == g_minus.size:
        al_gap_l2 += float(np.dot(tr_w, g_minus * g_minus))
    if tr_w_o.size == g_minus_o.size:
        al_gap_l2 += float(np.dot(tr_w_o, g_minus_o * g_minus_o))
    min_al_gap = float("nan")
    if tr_g.size or tr_g_o.size:
        min_al_gap = float(np.min(np.concatenate([tr_g, tr_g_o])))
    result.update(
        {
            "formulation": "solberg-puso-dual-pass-augmented-lagrangian",
            "formulation_label": "Compressible Neo-Hookean dual-pass AL contact",
            "qp_x": tr_x,
            "qp_g": tr_g,
            "qp_sn": tr_sn,
            "qp_pn": tr_pn,
            "qp_active": tr_on,
            "qp_xi": tr_xi,
            "qp_w": tr_w,
            "p_cauchy": -tr_sn,
            "p_cauchy_o": -tr_sn_o,
            "qp_x_o": tr_x_o,
            "qp_sn_o": tr_sn_o,
            "p_applied": p_applied,
            "F_int": float(np.sum(p_applied * tr_w)) if tr_w.size else 0.0,
            "n_active": int(np.count_nonzero(tr_on) + np.count_nonzero(tr_on_o)),
            "al_penalty": float(fine.al_penalty),
            "al_lambda_norm": float(
                np.sqrt(sum(v * v for v in fine.al_multipliers.values()))
            ),
            "al_n_lambda": int(len(fine.al_multipliers)),
            "al_gap_l2": float(np.sqrt(max(al_gap_l2, 0.0))),
            "al_min_gap": min_al_gap,
            "al_lambdas": {str(k): float(v) for k, v in fine.al_multipliers.items()},
        }
    )
    return result


def parse_args(argv=None):
    al_parser = argparse.ArgumentParser(add_help=False)
    al_parser.add_argument(
        "--al-penalty",
        type=float,
        help="Augmented-Lagrangian normal penalty gamma0. Defaults to --gamma0.",
    )
    al_parser.add_argument(
        "--al-relaxation",
        type=float,
        default=1.0,
        help="Relaxation for lambda <- max(0, lambda - gamma g).",
    )
    al_parser.add_argument(
        "--al-lambda-atol",
        type=float,
        default=1e-10,
        help="Absolute stopping tolerance for the AL multiplier update norm.",
    )
    al_parser.add_argument(
        "--al-drop-tol",
        type=float,
        default=1e-14,
        help="Drop multipliers below this magnitude from the sparse row map.",
    )
    al_args, remaining = al_parser.parse_known_args(argv)
    args = neo.parse_args(remaining)
    if al_args.al_penalty is None:
        al_args.al_penalty = float(args.gamma0)
    for key, val in vars(al_args).items():
        setattr(args, key, val)
    return args


def main(argv=None):
    args = parse_args(argv)
    mg.build_level = build_level
    mg.pack_result = pack_result
    mg.solve_mmg = solve_mmg
    ps = nc._load_pysfem()
    ps.init()
    try:
        if args.check:
            args.nx = 8
            args.ny = 4
            args.levels = 2
            args.max_iter = 20
            args.max_inner_it = 3
            args.nlsmooth_steps = 3
            args.indent = 0.02
            args.plot = False
            result = solve_mmg(ps, args)
            mg.check_mmg(result, args)
            print(
                f"check ok  F={result['F']:.4e}  n_active={result['n_active']}  "
                f"|lambda|={result['al_lambda_norm']:.3e}"
            )
        else:
            if args.load_steps > 1:
                if not (0.0 < args.load_reduction < 1.0):
                    raise ValueError("--load-reduction must be in (0, 1)")
                result = neo.solve_incremental_load(ps, args)
            else:
                result = solve_mmg(ps, args)
            if args.plot:
                neo.plot_result(result, args.plot_output)
    finally:
        ps.finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
