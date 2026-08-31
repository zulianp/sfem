#!/usr/bin/env python3
"""Solberg-Puso dual-pass stabilized mortar contact with Neo-Hookean MMG.

This variant keeps the nonlinear multigrid framework from
``mmg_nitsche_neohookean.py`` but replaces the Nitsche contact law with a
Solberg-Puso-style localized nodal multiplier space.  Each contact side has its
own P1 scalar pressure field; nodal normals turn scalar pressures into vector
tractions, and the dual-pass overlap quadrature assembles the displacement
coupling and traction-jump stabilization.
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
import solberg_puso_contact as spc


def _row_weight(level, surface_id):
    return float(level.theta_b if surface_id == 0 else level.theta_o)


def _integrated_contact_force(pressure, mass, theta):
    pressure = np.asarray(pressure, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)
    if pressure.size == 0 or mass.size == 0 or abs(theta) <= 1e-30:
        return 0.0
    n = min(pressure.size, mass.size)
    return float(np.sum(np.maximum(pressure[:n], 0.0) * mass[:n]) / theta)


def _surface_sigma_n(level, u, side):
    surfaces = getattr(level, "sp_surfaces", None)
    if surfaces is None:
        return np.zeros(0, dtype=np.float64)
    s = surfaces[side]
    out = np.zeros(len(s.nodes), dtype=np.float64)
    parent_ie = {}
    for ie, edge in enumerate(s.edges):
        parent_ie.setdefault(int(edge[0]), ie)
        parent_ie.setdefault(int(edge[1]), ie)
    for i, node in enumerate(s.nodes):
        ie = parent_ie.get(int(node))
        if ie is None:
            continue
        e_parent = int(s.elems[ie])
        parent_nodes = nc.tri3_parent_nodes(level.ps, level.mesh, s.bid, e_parent)
        px = np.array([level.X[j] for j in parent_nodes], dtype=np.float64)
        py = np.array([level.Y[j] for j in parent_nodes], dtype=np.float64)
        u_elem = np.empty(6, dtype=np.float64)
        for a, pn in enumerate(parent_nodes):
            u_elem[2 * a] = u[2 * int(pn)]
            u_elem[2 * a + 1] = u[2 * int(pn) + 1]
        nx, ny = float(s.normals[i, 0]), float(s.normals[i, 1])
        try:
            out[i] = neo.tri3_neo_sigma_n(px, py, u_elem, s.mu, s.lam, nx, ny)
        except (FloatingPointError, RuntimeError, ValueError):
            out[i] = 0.0
    return out


def al_contact_residual_tangent(level, u_vec, forced_active=None, frozen_geom=None):
    r_elastic, K_elastic = level.elastic_residual_tangent(u_vec)
    g_contact = np.zeros(level.ndofs, dtype=np.float64)
    coo = []
    new_active = set()
    assembly = spc.assemble(level, u_vec, neo)

    for row in assembly.rows:
        idx = int(row["idx"])
        pressure = float(assembly.pressure[idx])
        active = pressure > level.al_drop_tol if forced_active is None else idx in forced_active
        if not active:
            continue
        new_active.add(idx)
        scale = float(row["mass"])
        gamma = float(row["gamma"])
        for dof, dgd in row["dg"].items():
            g_contact[dof] -= scale * pressure * dgd
        for di, dgi in row["dg"].items():
            for dj, dgj in row["dg"].items():
                val = scale * gamma * dgi * dgj
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
    return residual, Ksys, g_contact, K_contact, assembly.n_qp, new_active


def _row_multiplier_update(level, assembly, old):
    trial = old - assembly.gamma * assembly.stabilized_gap
    projected = np.maximum(0.0, trial)
    projected[assembly.mass <= 1e-16] = 0.0
    return projected


def _projected_jacobi(H, rhs, x0, sweeps, omega=1.0):
    """Inexact dual step: x <- max(0, x + ω D^{-1}(rhs - H x)). Sparse matvecs only."""
    H = H.tocsr() if sparse.issparse(H) else sparse.csr_matrix(np.asarray(H, dtype=np.float64))
    rhs = np.asarray(rhs, dtype=np.float64).reshape(-1)
    n = int(rhs.size)
    if n == 0:
        return rhs
    diag = np.asarray(H.diagonal(), dtype=np.float64)
    diag = np.where(np.abs(diag) > 1e-30, diag, 1.0)
    if x0 is None:
        x = np.zeros(n, dtype=np.float64)
    else:
        x = np.maximum(np.asarray(x0, dtype=np.float64).reshape(-1), 0.0)
        if x.size != n:
            raise ValueError(f"x0 length {x.size} does not match rhs length {n}")
    omega = float(omega)
    for _ in range(max(1, int(sweeps))):
        x = np.maximum(0.0, x + omega * (rhs - H @ x) / diag)
    return x


def _check_coupled_multiplier_qp():
    H = np.array(((2.0, -1.0), (-1.0, 2.0)), dtype=np.float64)
    rhs = np.array((1.0, -2.0), dtype=np.float64)
    x = _projected_jacobi(H, rhs, None, 8)
    expected = np.array((0.5, 0.0), dtype=np.float64)
    if float(np.linalg.norm(x - expected)) > 1e-8:
        raise RuntimeError(f"projected Jacobi dual step regression failed: x={x}")


def _coupled_multiplier_update(level, assembly, old):
    good = (assembly.mass > 1e-16) & (assembly.gamma > 1e-30)
    rows = np.flatnonzero(good)
    projected = np.zeros_like(old)
    if rows.size == 0:
        return projected

    mass = np.asarray(assembly.mass[rows], dtype=np.float64)
    gamma = np.asarray(assembly.gamma[rows], dtype=np.float64)
    Jrr = assembly.J[rows][:, rows].tocsr()
    H = (sparse.diags(mass / gamma) + Jrr).tocsr()
    rhs = mass * (old[rows] / gamma - assembly.physical_gap[rows])
    sweeps = int(getattr(level, "sp_multiplier_max_iter", 8))
    projected[rows] = _projected_jacobi(H, rhs, old[rows], sweeps)
    return projected


def update_al_multipliers(level, u_vec, relaxation):
    assembly = spc.assemble(level, u_vec, neo)
    old = np.asarray(level.sp_multipliers, dtype=np.float64).copy()
    update = getattr(level, "sp_multiplier_update", "coupled")
    if update == "coupled":
        projected = _coupled_multiplier_update(level, assembly, old)
    elif update == "row":
        projected = _row_multiplier_update(level, assembly, old)
    else:
        raise ValueError(f"unknown Solberg-Puso multiplier update '{update}'")
    new = (1.0 - relaxation) * old + relaxation * projected
    new[new < level.al_drop_tol] = 0.0
    level.sp_multipliers = new
    post = spc.assemble(level, u_vec, neo)
    diff = new - old
    active = set(int(i) for i in np.flatnonzero(new > level.al_drop_tol))
    return {
        "lambda_change": float(np.linalg.norm(diff)),
        "lambda_norm": float(np.linalg.norm(new)),
        "max_violation": float(np.max(np.maximum(0.0, -post.physical_gap)))
        if post.physical_gap.size
        else 0.0,
        "max_stabilized_violation": float(np.max(np.maximum(0.0, -post.stabilized_gap)))
        if post.stabilized_gap.size
        else 0.0,
        "n_lambda": int(np.count_nonzero(new > level.al_drop_tol)),
        "n_rows": int(post.n_qp),
        "active": active,
        "traction_jump_norm": float(post.traction_jump_norm),
        "side_pressure_mismatch": float(post.side_pressure_mismatch),
    }


def build_level(ps, args):
    level = neo.build_level(ps, args)
    level.al_penalty = float(getattr(args, "al_penalty", args.gamma0))
    level.al_drop_tol = float(getattr(args, "al_drop_tol", 1e-14))
    level.sp_stabilization = float(getattr(args, "sp_stabilization", 1.0))
    level.sp_filter_stabilization_neighbors = not bool(
        getattr(args, "sp_no_filter_stabilization_neighbors", False)
    )
    level.sp_multiplier_update = getattr(args, "sp_multiplier_update", "coupled")
    level.sp_multiplier_max_iter = int(getattr(args, "sp_multiplier_max_iter", 8))
    level.sp_multipliers = np.zeros(
        nc.unique_nodes_from_edges(level.edges_b).size
        + nc.unique_nodes_from_edges(level.edges_o).size,
        dtype=np.float64,
    )
    level.include_sigma = False
    level.residual_tangent = lambda u, forced_active=None, frozen_geom=None: (
        al_contact_residual_tangent(level, u, forced_active, frozen_geom)
    )
    return level


def solve_mmg(ps, args, initial_u=None, initial_multipliers=None):
    if args.max_inner_it < 1:
        raise ValueError("--max-inner-it must be positive")
    if args.nlsmooth_steps < 0 or args.mg_pre < 0 or args.mg_post < 0:
        raise ValueError("smoothing step counts must be non-negative")
    relaxation = float(getattr(args, "al_relaxation", 1.0))
    if not (0.0 < relaxation <= 1.0):
        raise ValueError("--al-relaxation must be in (0, 1]")
    tail_relaxation_arg = float(getattr(args, "al_tail_relaxation", relaxation))
    if not (0.0 < tail_relaxation_arg <= 1.0):
        raise ValueError("--al-tail-relaxation must be in (0, 1]")

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
    if initial_multipliers is not None:
        lam0 = np.asarray(initial_multipliers, dtype=np.float64).reshape(-1)
        if lam0.size != fine.sp_multipliers.size:
            raise ValueError(
                f"initial multipliers have {lam0.size} entries, expected {fine.sp_multipliers.size}"
            )
        fine.sp_multipliers = np.maximum(lam0, 0.0)
        fine.sp_multipliers[fine.sp_multipliers < fine.al_drop_tol] = 0.0

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
    rnorm = float(np.linalg.norm(residual))
    rel = rnorm / residual_norm_0
    count_inner_iter = 0
    count_smoothing_steps = 0
    sweep_factor = 2 if args.smoother == "sgs" else 1
    sweeps_per_cycle = args.nlsmooth_steps * (args.mg_pre + args.mg_post) * sweep_factor
    converged = False

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
                f"|A|={len(active)}  n_qp={n_qp}  contact_dofs={n_contact}  "
                f"filt_dofs={n_filt}  contact_nnz={Kn.nnz}"
            )
            if ((rnorm < args.atol or rel < args.rtol) and inner != 0) or stagnation:
                break

        effective_relaxation = relaxation
        tail_atol = float(getattr(args, "al_tail_atol", 1e-8))
        tail_rtol = float(getattr(args, "al_tail_rtol", 1e-8))
        tail_relaxation = float(getattr(args, "al_tail_relaxation", relaxation))
        if rnorm < tail_atol or rel < tail_rtol:
            effective_relaxation = min(effective_relaxation, tail_relaxation)
        al_stats = update_al_multipliers(fine, u, effective_relaxation)
        residual, A, g_c, Kn, n_qp, active, contact_mask, masks = mg.filtered_hierarchy(
            levels, prolong, fine, u
        )
        rnorm = float(np.linalg.norm(residual))
        rel = rnorm / residual_norm_0
        if hasattr(fine, "sp_last_assembly"):
            gap_neg = np.minimum(fine.sp_last_assembly.physical_gap, 0.0)
            norm_pen = float(
                np.sqrt(max(float(np.dot(fine.sp_last_assembly.mass, gap_neg * gap_neg)), 0.0))
            )
        else:
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
            f"al_relax={effective_relaxation:.3e}  "
            f"max_stab_pen={al_stats['max_stabilized_violation']:.3e}  "
            f"|jump|={al_stats['traction_jump_norm']:.3e}  "
            f"side_mis={al_stats['side_pressure_mismatch']:.3e}  "
            f"stagnation={stagnation}"
        )
        residual_converged = rnorm < args.atol or rel < args.rtol
        lambda_converged = al_stats["lambda_change"] < float(args.al_lambda_atol)
        penetration_converged = norm_pen < args.ptol
        if residual_converged and lambda_converged and penetration_converged:
            converged = True
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
    result["al_converged"] = bool(converged)
    result["load_step_ok"] = bool(converged)
    print(
        f"nodes={fine.mesh.n_nodes()} dofs={fine.ndofs} F={result['F']:.4e} "
        f"a_hertz={result['a']:.4e} p0={result['p0']:.4e} |g_-|={result['penetration']:.3e}  "
        f"al_gap_l2={result['al_gap_l2']:.3e} al_min_gap={result['al_min_gap']:.3e}  "
        f"dual_pass_weights=({fine.theta_b:g},{fine.theta_o:g})  n_active={result['n_active']}  "
        f"|lambda|={result['al_lambda_norm']:.3e}"
    )
    return result


def contact_trace_al(level, u, surface_id):
    if not hasattr(level, "sp_last_assembly"):
        spc.assemble(level, u, neo)
    return spc.traces(level, surface_id)


def pack_result(fine, args, u, r_hist, n_active_hist, n_qp_hist, vcycle_hist=None, direct_hist=None):
    result = neo.pack_result(fine, args, u, r_hist, n_active_hist, n_qp_hist, vcycle_hist, direct_hist)
    tr_x, tr_g, _, tr_pn, tr_on, tr_xi, tr_w = contact_trace_al(fine, u, 0)
    tr_x_o, tr_g_o, _, tr_pn_o, tr_on_o, _, tr_w_o = contact_trace_al(fine, u, 1)
    tr_sn = _surface_sigma_n(fine, u, 0)
    tr_sn_o = _surface_sigma_n(fine, u, 1)
    surfaces = getattr(fine, "sp_surfaces", None)
    if surfaces is not None:
        result["nodes_b"] = surfaces[0].nodes.astype(np.int32)
        result["gap"] = tr_g.copy()
    p_applied = np.maximum(-tr_pn, 0.0)
    F_int = _integrated_contact_force(p_applied, tr_w, _row_weight(fine, 0))
    if F_int == 0.0:
        F_int = _integrated_contact_force(
            np.maximum(-tr_pn_o, 0.0), tr_w_o, _row_weight(fine, 1)
        )
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
            "formulation": "solberg-puso-localized-stabilized-augmented-lagrangian",
            "formulation_label": "Compressible Neo-Hookean Solberg-Puso stabilized AL contact",
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
            "F_int": F_int,
            "n_active": int(np.count_nonzero(tr_on) + np.count_nonzero(tr_on_o)),
            "al_penalty": float(fine.al_penalty),
            "sp_stabilization": float(fine.sp_stabilization),
            "al_lambda_norm": float(np.linalg.norm(fine.sp_multipliers)),
            "al_n_lambda": int(np.count_nonzero(fine.sp_multipliers > fine.al_drop_tol)),
            "penetration": float(np.linalg.norm(g_minus)),
            "al_gap_l2": float(np.sqrt(max(al_gap_l2, 0.0))),
            "al_min_gap": min_al_gap,
            "sp_stabilized_gap_l2": float(
                np.sqrt(
                    max(
                        float(
                            np.dot(
                                fine.sp_last_assembly.mass,
                                np.minimum(fine.sp_last_assembly.stabilized_gap, 0.0) ** 2,
                            )
                        ),
                        0.0,
                    )
                )
            )
            if hasattr(fine, "sp_last_assembly")
            else float("nan"),
            "sp_traction_jump_norm": float(fine.sp_last_assembly.traction_jump_norm)
            if hasattr(fine, "sp_last_assembly")
            else float("nan"),
            "sp_side_pressure_mismatch": float(fine.sp_last_assembly.side_pressure_mismatch)
            if hasattr(fine, "sp_last_assembly")
            else float("nan"),
            "al_lambdas": [float(v) for v in fine.sp_multipliers],
        }
    )
    return result


def sp_sample_contact_geometry(level, u):
    assembly = getattr(level, "sp_last_assembly", None)
    if assembly is None:
        return spc.assemble(level, u, neo)
    return assembly


def sp_frozen_touch_dofs(level, geom, active):
    if hasattr(level, "sp_last_assembly"):
        return spc.active_dof_mask(level, active)
    return np.zeros(level.ndofs, dtype=bool)


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
        help="Relaxation for the projected multiplier update.",
    )
    al_parser.add_argument(
        "--al-tail-relaxation",
        type=float,
        default=0.25,
        help="Multiplier relaxation used after the displacement residual enters the tail.",
    )
    al_parser.add_argument(
        "--al-tail-atol",
        type=float,
        default=1e-8,
        help="Absolute residual threshold for switching to --al-tail-relaxation.",
    )
    al_parser.add_argument(
        "--al-tail-rtol",
        type=float,
        default=1e-8,
        help="Relative residual threshold for switching to --al-tail-relaxation.",
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
    al_parser.add_argument(
        "--mortar-quadrature",
        choices=("edge-overlap",),
        default="edge-overlap",
        help="Contact quadrature for the Solberg-Puso multiplier space.",
    )
    al_parser.add_argument(
        "--common-multiplier-degree",
        choices=("p1",),
        default="p1",
        help="Localized multiplier degree. Puso-Solberg uses the contact trace basis; TRI3 gives P1.",
    )
    al_parser.add_argument(
        "--sp-stabilization",
        type=float,
        default=1.0,
        help="Dimensionless Solberg-Puso traction-jump stabilization alpha.",
    )
    al_parser.add_argument(
        "--sp-no-filter-stabilization-neighbors",
        action="store_true",
        help="Filter only active multiplier support, not stabilization-connected neighbors.",
    )
    al_parser.add_argument(
        "--sp-multiplier-update",
        choices=("row", "coupled"),
        default="coupled",
        help=(
            "row: lagged Uzawa λ←max(0, λ-γĝ), one J matvec, no dual iteration. "
            "coupled: projected Jacobi sweeps on sparse (M/γ+J); no factorization."
        ),
    )
    al_parser.add_argument(
        "--sp-multiplier-max-iter",
        type=int,
        default=8,
        help="Projected Jacobi sweeps used by --sp-multiplier-update coupled.",
    )
    al_args, remaining = al_parser.parse_known_args(argv)
    args = neo.parse_args(remaining)
    if al_args.al_penalty is None:
        al_args.al_penalty = float(args.gamma0)
    for key, val in vars(al_args).items():
        setattr(args, key, val)
    argv_list = list(sys.argv[1:] if argv is None else argv)
    user_set_plot = any(a == "--plot-output" or a.startswith("--plot-output=") for a in argv_list)
    if not user_set_plot:
        args.plot_output = os.path.join(_SPIKE, "mmg_solberg_puso_al_neohookean.png")
    return args


def solve_incremental_load(ps, args):
    last_lambda = [None]
    inner = solve_mmg

    def wrapped(ps2, trial_args, initial_u=None):
        result = inner(
            ps2, trial_args, initial_u=initial_u, initial_multipliers=last_lambda[0]
        )
        last_lambda[0] = result.get("al_lambdas")
        return result

    saved = mg.solve_mmg
    mg.solve_mmg = wrapped
    try:
        return neo.solve_incremental_load(ps, args)
    finally:
        mg.solve_mmg = saved


def main(argv=None):
    args = parse_args(argv)
    mg.build_level = build_level
    mg.pack_result = pack_result
    mg.solve_mmg = solve_mmg
    mg.sample_contact_geometry = sp_sample_contact_geometry
    mg.frozen_touch_dofs = sp_frozen_touch_dofs
    ps = nc._load_pysfem()
    ps.init()
    try:
        if args.check:
            args.nx = 16
            args.ny = 8
            args.levels = 2
            args.max_iter = 30
            args.max_inner_it = 3
            args.nlsmooth_steps = 3
            args.indent = 0.02
            args.plot = False
            _check_coupled_multiplier_qp()
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
                result = solve_incremental_load(ps, args)
            else:
                result = solve_mmg(ps, args)
            if args.plot:
                neo.plot_result(result, args.plot_output)
    finally:
        ps.finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
