#!/usr/bin/env python3
"""Filtered unbiased Nitsche monotone multigrid (jump-free coarse space).

Fine level: Mlika–Renard–Chouly Nitsche residual/tangent from
``nitsche_contact.py``. Contact geometry (pairing, normals, weights) is
resampled every outer iteration, as in Newton. The active set follows
[P]_- at the current iterate.

Each outer iteration rebuilds the Nitsche tangent and takes a filtered
V-cycle step. Coarse correction is Galerkin ``P^T A P`` with jump-free
elimination of every displacement DOF touched by the current active
contact.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

_SPIKE = os.path.dirname(os.path.abspath(__file__))
if _SPIKE not in sys.path:
    sys.path.insert(0, _SPIKE)

import nitsche_contact as nc


@dataclass
class NitscheLevel:
    nx: int
    ny: int
    ny_block: int
    n_block: int
    mesh: object
    X: np.ndarray
    Y: np.ndarray
    ndofs: int
    Ke: sparse.csr_matrix
    constrained: np.ndarray
    u_bc: np.ndarray
    edges_b: np.ndarray
    edges_o: np.ndarray
    elems_b: np.ndarray
    elems_o: np.ndarray
    bid_b: int
    bid_o: int
    mu_b: float
    lam_b: float
    mu_o: float
    lam_o: float
    theta_b: float
    theta_o: float
    include_sigma: bool
    radius: float
    ps: object
    residual_tangent: object
    contact_nodes: np.ndarray
    hx: float


def nodal_prolongation(nx_c, ny_c, nx_f, ny_f):
    if nx_f != 2 * nx_c or ny_f != 2 * ny_c:
        raise ValueError(
            f"nested refinement required: coarse ({nx_c},{ny_c}) fine ({nx_f},{ny_f})"
        )
    nc = (nx_c + 1) * (ny_c + 1)
    nf = (nx_f + 1) * (ny_f + 1)
    rows, cols, data = [], [], []

    def cid(i, j):
        return i + j * (nx_c + 1)

    def fid(i, j):
        return i + j * (nx_f + 1)

    for jf in range(ny_f + 1):
        for i_f in range(nx_f + 1):
            f = fid(i_f, jf)
            ie, io = divmod(i_f, 2)
            je, jo = divmod(jf, 2)
            if io == 0 and jo == 0:
                rows.append(f)
                cols.append(cid(ie, je))
                data.append(1.0)
            elif io == 1 and jo == 0:
                rows.extend((f, f))
                cols.extend((cid(ie, je), cid(ie + 1, je)))
                data.extend((0.5, 0.5))
            elif io == 0 and jo == 1:
                rows.extend((f, f))
                cols.extend((cid(ie, je), cid(ie, je + 1)))
                data.extend((0.5, 0.5))
            else:
                rows.extend((f, f, f, f))
                cols.extend(
                    (
                        cid(ie, je),
                        cid(ie + 1, je),
                        cid(ie, je + 1),
                        cid(ie + 1, je + 1),
                    )
                )
                data.extend((0.25, 0.25, 0.25, 0.25))
    return sparse.csr_matrix((data, (rows, cols)), shape=(nf, nc))


def vector_prolongation(p_nodes):
    return sparse.kron(p_nodes, sparse.eye(2, format="csr"), format="csr")


def hertz_prolongation(coarse: NitscheLevel, fine: NitscheLevel):
    p_block = vector_prolongation(
        nodal_prolongation(coarse.nx, coarse.ny_block, fine.nx, fine.ny_block)
    )
    p_obs = vector_prolongation(
        nodal_prolongation(coarse.nx, coarse.ny, fine.nx, fine.ny)
    )
    return sparse.block_diag((p_block, p_obs), format="csr")


def hierarchy_sizes(nx, ny, n_levels):
    sizes = []
    nxk, nyk = int(nx), int(ny)
    for k in range(int(n_levels)):
        nyb = max(1, nyk // 2)
        sizes.append((nxk, nyk, nyb))
        if k + 1 >= int(n_levels):
            break
        if nxk % 2 or nyk % 2 or nyb % 2:
            break
        nxk //= 2
        nyk //= 2
    if len(sizes) < 2:
        raise RuntimeError(
            f"need at least 2 nested levels from nx={nx} ny={ny} n_levels={n_levels}"
        )
    return sizes


def build_level(ps, args) -> NitscheLevel:
    radius = args.radius
    ny_block = int(getattr(args, "ny_block", max(1, args.ny // 2)))
    mesh, n_block = nc.make_hertz_mesh(
        ps,
        args.nx,
        args.ny,
        radius,
        args.r_inner * radius,
        args.width,
        args.height,
        args.gap,
        ny_block=ny_block,
    )
    X, Y = nc.coords(ps, mesh)
    space = ps.FunctionSpace(mesh, 2)
    op = ps.create_op(space, "LinearElasticity")
    op.initialize()
    E_b, nu_b = args.E_block, args.nu
    E_o, nu_o = args.E_obstacle, args.nu
    if args.rigid_block:
        E_b *= args.rigid_stiffness
    if args.rigid_obstacle:
        E_o *= args.rigid_stiffness
    lam_b, mu_b = nc.lame(E_b, nu_b)
    lam_o, mu_o = nc.lame(E_o, nu_o)
    op.set_value_in_block("block", "mu", mu_b)
    op.set_value_in_block("block", "lambda", lam_b)
    op.set_value_in_block("obstacle", "mu", mu_o)
    op.set_value_in_block("obstacle", "lambda", lam_o)
    fun = ps.Function(space)
    fun.add_operator(op)
    rowptr, colidx, values = ps.assemble_csr(fun)
    ndofs = int(space.n_dofs())
    Ke = sparse.csr_matrix(
        (np.array(values), np.array(colidx), np.array(rowptr)), shape=(ndofs, ndofs)
    )
    tol = 1e-3 * max(args.width, radius) / max(args.nx, 1)
    y_top = float(np.max(Y[:n_block]))
    y_bot = float(np.min(Y[:n_block]))
    y_diam = float(np.min(Y[n_block:]))
    ss_top = nc.sidesets_from_selector(ps, mesh, lambda x, y, z: y > y_top - tol, ["block"])
    ss_left = nc.sidesets_from_selector(
        ps, mesh, lambda x, y, z: x < -0.5 * args.width + tol, ["block"]
    )
    ss_right = nc.sidesets_from_selector(
        ps, mesh, lambda x, y, z: x > 0.5 * args.width - tol, ["block"]
    )
    ss_contact_block = nc.sidesets_from_selector(
        ps, mesh, lambda x, y, z: y < y_bot + 4 * tol, ["block"]
    )
    ss_contact_obs = nc.sidesets_from_selector(
        ps,
        mesh,
        lambda x, y, z: bool(np.hypot(x, y) > 0.85 * radius) and bool(y > 0.05 * radius),
        ["obstacle"],
    )
    ss_diam = nc.sidesets_from_selector(
        ps, mesh, lambda x, y, z: y < y_diam + 4 * tol, ["obstacle"]
    )
    conditions = [
        nc.dirichlet_condition(ps, ss_top, 1, -args.indent),
        nc.dirichlet_condition(ps, ss_top, 0, 0.0),
        nc.dirichlet_condition(ps, ss_left, 0, 0.0),
        nc.dirichlet_condition(ps, ss_right, 0, 0.0),
        nc.dirichlet_condition(ps, ss_diam, 0, 0.0),
        nc.dirichlet_condition(ps, ss_diam, 1, 0.0),
    ]
    if args.rigid_obstacle:
        ss_all_obs = nc.sidesets_from_selector(ps, mesh, lambda x, y, z: True, ["obstacle"])
        conditions.append(nc.dirichlet_condition(ps, ss_all_obs, 0, 0.0))
        conditions.append(nc.dirichlet_condition(ps, ss_all_obs, 1, 0.0))
    if args.rigid_block:
        ss_all_block = nc.sidesets_from_selector(ps, mesh, lambda x, y, z: True, ["block"])
        conditions.append(nc.dirichlet_condition(ps, ss_all_block, 0, 0.0))
        conditions.append(nc.dirichlet_condition(ps, ss_all_block, 1, -args.indent))
    bcs = ps.create_dirichlet_conditions(space, conditions, ps.ExecutionSpace.EXECUTION_SPACE_HOST)
    fun.add_constraint(bcs)
    u_buf = ps.create_real_buffer(ndofs)
    g_buf = ps.create_real_buffer(ndofs)
    u = nc.numpy_view(ps, u_buf)
    g = nc.numpy_view(ps, g_buf)
    u[:] = 0.0
    ps.apply_constraints(fun, u_buf)
    u_bc = u.copy()
    mask = np.zeros(ndofs, dtype=bool)
    ps.apply_zero_constraints(fun, g_buf)
    g[:] = 1.0
    ps.apply_zero_constraints(fun, g_buf)
    mask[g == 0.0] = True
    constrained = np.where(mask)[0]
    contact_block = nc.flatten_sidesets([ss_contact_block])
    contact_obs = nc.flatten_sidesets([ss_contact_obs])
    if not contact_block or not contact_obs:
        raise RuntimeError("empty contact sidesets")
    edges_b, elems_b, bid_b = nc.sideset_edges(ps, mesh, contact_block[0])
    edges_o, elems_o, bid_o = nc.sideset_edges(ps, mesh, contact_obs[0])
    for extra in contact_block[1:]:
        e, p, _ = nc.sideset_edges(ps, mesh, extra)
        edges_b = np.vstack([edges_b, e])
        elems_b = np.concatenate([elems_b, p])
    for extra in contact_obs[1:]:
        e, p, _ = nc.sideset_edges(ps, mesh, extra)
        edges_o = np.vstack([edges_o, e])
        elems_o = np.concatenate([elems_o, p])
    dr = (1.0 - args.r_inner) * radius / max(args.ny, 1)
    r_tol = min(0.35 * dr, 0.02 * radius)
    x_face = 0.5 * args.width + 4 * tol

    def is_block_bottom(n0, n1):
        return Y[n0] < y_bot + 4 * tol and Y[n1] < y_bot + 4 * tol

    def is_outer_arc(n0, n1):
        r0 = float(np.hypot(X[n0], Y[n0]))
        r1 = float(np.hypot(X[n1], Y[n1]))
        if abs(r0 - radius) > r_tol or abs(r1 - radius) > r_tol:
            return False
        if abs(r0 - r1) > 0.5 * r_tol:
            return False
        if abs(X[n0]) > x_face or abs(X[n1]) > x_face:
            return False
        if Y[n0] < 0.5 * radius or Y[n1] < 0.5 * radius:
            return False
        return True

    edges_b, elems_b = nc.filter_edges(X, Y, edges_b, elems_b, is_block_bottom)
    edges_o, elems_o = nc.filter_edges(X, Y, edges_o, elems_o, is_outer_arc)
    edges_b, elems_b = nc.unique_oriented_edges(edges_b, elems_b)
    edges_o, elems_o = nc.unique_oriented_edges(edges_o, elems_o)
    if edges_b.size == 0 or edges_o.size == 0:
        raise RuntimeError("empty contact edges after filter")
    edges_b = nc.orient_edges(X, Y, edges_b, "down")
    edges_o = nc.orient_edges(X, Y, edges_o, "outward")
    if args.rigid_block and not args.rigid_obstacle:
        theta_b, theta_o = 0.0, 1.0
    elif args.rigid_obstacle or args.biased or args.penalty or args.lagrange:
        theta_b, theta_o = 1.0, 0.0
    else:
        theta_b, theta_o = 0.5, 0.5
    include_sigma = not args.penalty and not args.lagrange

    def residual_tangent(u_vec, forced_active=None, frozen_geom=None):
        g_contact = np.zeros(ndofs)
        coo = []
        new_active = set()
        n_qp = 0
        frozen_b = None if frozen_geom is None else frozen_geom.get(0)
        frozen_o = None if frozen_geom is None else frozen_geom.get(1)
        if theta_b > 0:
            n_qp += nc.surface_contrib(
                X, Y, u_vec, edges_b, elems_b, bid_b, edges_o, ps, mesh, mu_b, lam_b,
                args.gamma0, theta_b, False, args.mu_f, g_contact, coo, radius,
                0, new_active, False, None, None, True, include_sigma, forced_active,
                None, None, frozen_b,
            )
        if theta_o > 0:
            n_qp += nc.surface_contrib(
                X, Y, u_vec, edges_o, elems_o, bid_o, edges_b, ps, mesh, mu_o, lam_o,
                args.gamma0, theta_o, False, args.mu_f, g_contact, coo, radius,
                1, new_active, True, None, None, True, include_sigma, forced_active,
                None, None, frozen_o,
            )
        resid = Ke @ u_vec + g_contact
        if coo:
            ii, jj, vv = zip(*coo)
            Kn = sparse.coo_matrix((vv, (ii, jj)), shape=(ndofs, ndofs)).tocsr()
        else:
            Kn = sparse.csr_matrix((ndofs, ndofs))
        Ksys = Ke + Kn
        Ksys, resid = nc.apply_dirichlet_system(Ksys, resid, constrained, u_vec, u_bc)
        return resid, Ksys, g_contact, Kn, n_qp, new_active

    contact_nodes = np.unique(
        np.concatenate([nc.unique_nodes_from_edges(edges_b), nc.unique_nodes_from_edges(edges_o)])
    )
    hx = 0.0
    for edges in (edges_b, edges_o):
        for n0, n1 in edges:
            hx = max(hx, float(np.hypot(X[n0] - X[n1], Y[n0] - Y[n1])))
    return NitscheLevel(
        nx=int(args.nx),
        ny=int(args.ny),
        ny_block=ny_block,
        n_block=n_block,
        mesh=mesh,
        X=X,
        Y=Y,
        ndofs=ndofs,
        Ke=Ke.tocsr(),
        constrained=constrained,
        u_bc=u_bc,
        edges_b=edges_b,
        edges_o=edges_o,
        elems_b=elems_b,
        elems_o=elems_o,
        bid_b=bid_b,
        bid_o=bid_o,
        mu_b=mu_b,
        lam_b=lam_b,
        mu_o=mu_o,
        lam_o=lam_o,
        theta_b=theta_b,
        theta_o=theta_o,
        include_sigma=include_sigma,
        radius=radius,
        ps=ps,
        residual_tangent=residual_tangent,
        contact_nodes=contact_nodes.astype(np.int32),
        hx=max(hx, 1e-12),
    )


def sample_contact_geometry(level: NitscheLevel, u):
    frozen = {0: {}, 1: {}}
    specs = (
        (0, level.edges_b, level.edges_o, False),
        (1, level.edges_o, level.edges_b, True),
    )
    if level.theta_b <= 0:
        specs = (specs[1],)
    if level.theta_o <= 0:
        specs = tuple(s for s in specs if s[0] != 1)
    for sid, edges, other, snap in specs:
        for ie, edge in enumerate(edges):
            n0, n1 = int(edge[0]), int(edge[1])
            length, nx0, ny0 = nc.edge_geometry(level.X, level.Y, n0, n1)[:3]
            if length <= 1e-16:
                continue
            frozen[sid][ie] = nc.sample_edge_qps(
                level.X, level.Y, u, n0, n1, length, nx0, ny0, other, level.radius, snap
            )
    return frozen


def frozen_touch_dofs(level: NitscheLevel, frozen, active):
    mask = np.zeros(level.ndofs, dtype=bool)
    for key in active:
        sid, ie = int(key[0]), int(key[1])
        edges = level.edges_b if sid == 0 else level.edges_o
        n0, n1 = int(edges[ie, 0]), int(edges[ie, 1])
        for node in (n0, n1):
            mask[2 * node] = True
            mask[2 * node + 1] = True
        for s in frozen.get(sid, {}).get(ie, ()):
            m0, m1 = int(s[8]), int(s[9])
            mask[2 * m0] = True
            mask[2 * m0 + 1] = True
            mask[2 * m1] = True
            mask[2 * m1 + 1] = True
    return mask


def dirichlet_mask(level: NitscheLevel):
    mask = np.zeros(level.ndofs, dtype=bool)
    mask[level.constrained] = True
    return mask


def restrict_dof_mask(P, fine_mask):
    touched = np.abs(P.T) @ fine_mask.astype(np.float64)
    return touched > 1e-14


def filter_hierarchy(levels, prolong, fine_contact_mask):
    """Fine: Dirichlet only (contact is smoothed). Coarse: Dirichlet plus
    the variational restriction of the active-contact support."""
    masks = [dirichlet_mask(levels[0])]
    current = fine_contact_mask.copy()
    for k, P in enumerate(prolong):
        coarse_contact = restrict_dof_mask(P, current)
        masks.append(dirichlet_mask(levels[k + 1]) | coarse_contact)
        current = coarse_contact
    return masks


def elastic_solve(level: NitscheLevel):
    u = level.u_bc.copy()
    r = np.asarray(level.Ke @ u, dtype=np.float64)
    K, r = nc.apply_dirichlet_system(level.Ke.copy(), r, level.constrained, u, level.u_bc)
    du = spsolve(K, -r)
    u = u + np.asarray(du, dtype=np.float64)
    u[level.constrained] = level.u_bc[level.constrained]
    return u


def jacobi(A, rhs, x, mask, steps, omega):
    if steps <= 0:
        return x
    allowed = ~mask
    diag = A.diagonal().copy()
    diag[np.abs(diag) < 1e-30] = 1.0
    diag[mask] = 1.0
    for _ in range(steps):
        res = rhs - A @ x
        x[allowed] += omega * res[allowed] / diag[allowed]
        x[mask] = 0.0
    return x


def masked_direct_solve(A, rhs, mask):
    x = np.zeros_like(rhs)
    allowed = np.flatnonzero(~mask)
    if allowed.size == 0:
        return x
    x[allowed] = spsolve(A[np.ix_(allowed, allowed)].tocsc(), rhs[allowed])
    x[mask] = 0.0
    return np.asarray(x, dtype=np.float64)


def patch_solve(A, rhs, x, patch, exclude):
    """Exact multiplicative Schwarz step on the active-contact patch."""
    allowed = np.flatnonzero(patch & ~exclude)
    if allowed.size == 0:
        return x
    res = rhs - A @ x
    x = np.asarray(x, dtype=np.float64).copy()
    x[allowed] += np.asarray(
        spsolve(A[np.ix_(allowed, allowed)].tocsc(), res[allowed]), dtype=np.float64
    )
    x[exclude] = 0.0
    return x


def filtered_vcycle(operators, prolong, ell, rhs, masks, pre, post, omega, contact_patch=None):
    A = operators[ell]
    mask = masks[ell]
    rhs_eff = rhs.copy()
    rhs_eff[mask] = 0.0
    if ell == len(operators) - 1:
        return masked_direct_solve(A, rhs_eff, mask)
    x = np.zeros_like(rhs_eff)
    x = jacobi(A, rhs_eff, x, mask, pre, omega)
    if ell == 0 and contact_patch is not None:
        x = patch_solve(A, rhs_eff, x, contact_patch, mask)
    res = rhs_eff - A @ x
    res[mask] = 0.0
    if ell == 0 and contact_patch is not None:
        res[contact_patch] = 0.0
    coarse_rhs = np.asarray(prolong[ell].T @ res).ravel()
    coarse_error = filtered_vcycle(
        operators, prolong, ell + 1, coarse_rhs, masks, pre, post, omega, contact_patch
    )
    x = x + prolong[ell] @ coarse_error
    x[mask] = 0.0
    if ell == 0 and contact_patch is not None:
        x = patch_solve(A, rhs_eff, x, contact_patch, mask)
    x = jacobi(A, rhs_eff, x, mask, post, omega)
    if ell == 0 and contact_patch is not None:
        x = patch_solve(A, rhs_eff, x, contact_patch, mask)
    return np.asarray(x, dtype=np.float64)


def coarse_operators(levels, prolong, fine_A):
    """Inherited coarse models: A_H = P^T A_h^N P. Contact is not rediscretized."""
    ops = [fine_A.tocsr()]
    for P in prolong:
        ops.append((P.T @ ops[-1] @ P).tocsr())
    return ops


def mg_linear_solve(
    levels, prolong, operators, rhs, masks, cycles, rtol, pre, post, omega, contact_patch=None
):
    x = np.zeros_like(rhs)
    A = operators[0]
    rhs_norm = max(float(np.linalg.norm(rhs)), 1e-30)
    rel = float(np.linalg.norm(rhs - A @ x)) / rhs_norm
    used = 0
    for _ in range(cycles):
        if rel <= rtol:
            break
        x = x + filtered_vcycle(
            operators, prolong, 0, rhs - A @ x, masks, pre, post, omega, contact_patch
        )
        x[masks[0]] = 0.0
        used += 1
        rel = float(np.linalg.norm(rhs - A @ x)) / rhs_norm
    return x, used, rel


def pack_result(fine: NitscheLevel, args, u, r_hist, n_active_hist, n_qp_hist, vcycle_hist=None):
    X, Y = fine.X, fine.Y
    n_block = fine.n_block
    residual, K, g_c, Kn, n_qp, A = fine.residual_tangent(u, forced_active=None)
    nodes_b = nc.unique_nodes_from_edges(fine.edges_b)
    gap = np.empty(nodes_b.size)
    for i, node in enumerate(nodes_b):
        node = int(node)
        px = float(X[node] + u[2 * node])
        py = float(Y[node] + u[2 * node + 1])
        m0, m1, t, qx, qy = nc.master_on_circle(px, py, X, Y, fine.edges_o, fine.radius)
        rn = float(np.hypot(qx, qy))
        if rn <= 1e-30:
            gap[i] = 0.0
            continue
        nx, ny = -qx / rn, -qy / rn
        uqx = (1.0 - t) * u[2 * m0] + t * u[2 * m1]
        uqy = (1.0 - t) * u[2 * m0 + 1] + t * u[2 * m1 + 1]
        gap[i] = (qx + uqx - px) * nx + (qy + uqy - py) * ny
    penetration = float(np.linalg.norm(np.minimum(gap, 0.0)))
    r_full = np.asarray(fine.Ke @ u + g_c, dtype=np.float64)
    tol = 1e-3 * max(args.width, args.radius) / max(args.nx, 1)
    y_top = float(np.max(Y[:n_block]))
    top = np.flatnonzero(np.abs(Y[:n_block] - y_top) <= 4.0 * tol)
    F = abs(float(np.sum(r_full[2 * top + 1]))) if top.size else 0.0
    E_o = args.E_obstacle if not args.rigid_obstacle else 1e300
    E_star = nc.effective_modulus(args.E_block, args.nu, E_o, args.nu)
    xc = 0.5 * (X[fine.edges_b[:, 0]] + X[fine.edges_b[:, 1]])
    p_hertz, a, p0 = nc.hertz_pressure(xc, F, args.radius, E_star)
    tr_x, tr_g, tr_sn, tr_pn, tr_on, tr_xi, tr_w = nc.contact_trace(
        X, Y, u, fine.edges_b, fine.elems_b, fine.bid_b, fine.edges_o, fine.ps, fine.mesh,
        fine.mu_b, fine.lam_b, args.gamma0, fine.theta_b if fine.theta_b > 0 else 1.0,
        fine.radius, fine.include_sigma, False, 0.0,
    )
    tr_x_o, tr_g_o, tr_sn_o, tr_pn_o, tr_on_o, _, tr_w_o = nc.contact_trace(
        X, Y, u, fine.edges_o, fine.elems_o, fine.bid_o, fine.edges_b, fine.ps, fine.mesh,
        fine.mu_o, fine.lam_o, args.gamma0, fine.theta_o if fine.theta_o > 0 else 1.0,
        fine.radius, fine.include_sigma, True, 0.0,
    )
    th = fine.theta_b if fine.theta_b > 0.0 else 1.0
    p_applied = np.where(tr_on, np.maximum(-tr_pn / th, 0.0), 0.0)
    n_active = int(np.count_nonzero(tr_on))
    return {
        "mesh": fine.mesh,
        "X": X,
        "Y": Y,
        "u": u.copy(),
        "F": F,
        "a": a,
        "p0": p0,
        "gap": gap,
        "nodes_b": nodes_b,
        "xc": xc,
        "p_hertz": p_hertz,
        "ndofs": fine.ndofs,
        "penetration": penetration,
        "r_hist": r_hist,
        "n_active_hist": n_active_hist,
        "n_qp_hist": n_qp_hist,
        "vcycle_hist": [] if vcycle_hist is None else list(vcycle_hist),
        "rtol": float(args.rtol),
        "n_block": n_block,
        "tris_b": nc.block_triangles(fine.ps, fine.mesh, fine.bid_b),
        "tris_o": nc.block_triangles(fine.ps, fine.mesh, fine.bid_o),
        "rigid_obstacle": bool(args.rigid_obstacle),
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
        "n_active": n_active,
        "include_sigma": fine.include_sigma,
        "theta_b": float(fine.theta_b),
        "theta_o": float(fine.theta_o),
        "indent": float(args.indent),
    }


def solve_mmg(ps, args):
    sizes = hierarchy_sizes(args.nx, args.ny, args.levels)
    levels = []
    for nxk, nyk, nyb in sizes:
        a = copy.copy(args)
        a.nx, a.ny, a.ny_block = nxk, nyk, nyb
        levels.append(build_level(ps, a))
        print(
            f"level nx={nxk} ny={nyk} ny_block={nyb} nodes={levels[-1].mesh.n_nodes()} "
            f"dofs={levels[-1].ndofs}"
        )
    prolong = [hertz_prolongation(levels[k + 1], levels[k]) for k in range(len(levels) - 1)]
    fine = levels[0]
    u = fine.u_bc.copy()

    r_hist = []
    n_active_hist = []
    n_qp_hist = []
    vcycle_hist = []
    active = set()
    n_qp = 0
    last_cycles = 0
    for it in range(args.max_iter):
        residual, A, g_c, Kn, n_qp, active = fine.residual_tangent(
            u, forced_active=None
        )
        rnorm = float(np.linalg.norm(residual))
        r_hist.append(rnorm)
        n_active_hist.append(len(active))
        n_qp_hist.append(n_qp)
        vcycle_hist.append(int(last_cycles))
        geom = sample_contact_geometry(fine, u)
        contact_mask = frozen_touch_dofs(fine, geom, active)
        masks = filter_hierarchy(levels, prolong, contact_mask)
        n_filt = [int(np.count_nonzero(m)) for m in masks]
        n_contact = int(np.count_nonzero(contact_mask & ~masks[0]))
        print(
            f"mmg[{it:02d}] ||r||={rnorm:.3e}  |A|={len(active)}  nitsche_qp={n_qp}  "
            f"contact_dofs={n_contact}  filt_dofs={n_filt}  contact_nnz={Kn.nnz}"
        )
        if rnorm < args.rtol:
            break
        operators = coarse_operators(levels, prolong, A)
        du, cycles, lin_rel = mg_linear_solve(
            levels,
            prolong,
            operators,
            -residual,
            masks,
            args.mg_cycles,
            args.mg_linear_rtol,
            args.mg_pre,
            args.mg_post,
            args.mg_omega,
            contact_mask,
        )
        du[fine.constrained] = 0.0
        if (not np.all(np.isfinite(du))) or lin_rel > 0.25:
            du = np.asarray(spsolve(A, -residual), dtype=np.float64)
            du[fine.constrained] = 0.0
            cycles = 0
            lin_rel = float(np.linalg.norm(residual + A @ du)) / max(rnorm, 1e-30)
        if not np.all(np.isfinite(du)):
            raise RuntimeError("MMG increment is not finite")
        u = u + du
        u[fine.constrained] = fine.u_bc[fine.constrained]
        last_cycles = int(cycles)
        print(f"          V-cycles={cycles}  lin_rel={lin_rel:.3e}")
    residual, A, g_c, Kn, n_qp, active = fine.residual_tangent(u, forced_active=None)
    r_final = float(np.linalg.norm(residual))
    if not r_hist or abs(r_hist[-1] - r_final) > 1e-30:
        r_hist.append(r_final)
        n_active_hist.append(len(active))
        n_qp_hist.append(n_qp)
        vcycle_hist.append(int(last_cycles))
        print(f"mmg[final] ||r||={r_final:.3e}  |A|={len(active)}")
    result = pack_result(fine, args, u, r_hist, n_active_hist, n_qp_hist, vcycle_hist)
    print(
        f"nodes={fine.mesh.n_nodes()} dofs={fine.ndofs} F={result['F']:.4e} "
        f"a_hertz={result['a']:.4e} p0={result['p0']:.4e} |g_-|={result['penetration']:.3e}  "
        f"theta=({fine.theta_b:g},{fine.theta_o:g})  n_active={result['n_active']}"
    )
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nx", type=int, default=8)
    p.add_argument("--ny", type=int, default=4)
    p.add_argument("--levels", type=int, default=3)
    p.add_argument("--radius", type=float, default=1.0)
    p.add_argument("--r-inner", type=float, default=0.4)
    p.add_argument("--width", type=float, default=1.6)
    p.add_argument("--height", type=float, default=0.4)
    p.add_argument("--gap", type=float, default=0.0)
    p.add_argument("--indent", type=float, default=0.02)
    p.add_argument("--E-block", type=float, default=1.0)
    p.add_argument("--E-obstacle", type=float, default=1.0)
    p.add_argument("--nu", type=float, default=0.3)
    p.add_argument("--gamma0", type=float, default=50.0)
    p.add_argument("--max-iter", type=int, default=40)
    p.add_argument("--rtol", type=float, default=1e-8)
    p.add_argument("--mg-cycles", type=int, default=2)
    p.add_argument("--mg-pre", type=int, default=4)
    p.add_argument("--mg-post", type=int, default=4)
    p.add_argument("--mg-omega", type=float, default=0.4)
    p.add_argument("--mg-linear-rtol", type=float, default=1e-2)
    p.add_argument("--mu-f", type=float, default=0.3)
    p.add_argument("--rigid-block", action="store_true")
    p.add_argument("--rigid-obstacle", action="store_true")
    p.add_argument("--rigid-stiffness", type=float, default=1e4)
    p.add_argument("--biased", action="store_true")
    p.add_argument("--unbiased", action="store_true")
    p.add_argument("--penalty", action="store_true")
    p.add_argument("--lagrange", action="store_true")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--check", action="store_true")
    return p.parse_args(argv)


def check_mmg(result, args):
    r_hist = np.asarray(result.get("r_hist", []), dtype=float)
    if r_hist.size < 2 or not np.all(np.isfinite(r_hist)):
        raise SystemExit("mmg check failed: non-finite residual history")
    if r_hist[-1] > 1e-4 * r_hist[0] and r_hist[-1] > args.rtol:
        raise SystemExit(
            f"mmg check failed: linearized residual stalled "
            f"{r_hist[0]:.3e} -> {r_hist[-1]:.3e}"
        )
    if result["n_active"] < 1:
        raise SystemExit("mmg check failed: empty active set")
    if result["F"] <= 0.0:
        raise SystemExit("mmg check failed: non-positive contact force")
    if not np.isfinite(result["penetration"]):
        raise SystemExit("mmg check failed: non-finite penetration")
    gap = np.asarray(result["gap"])
    nodes = result["nodes_b"]
    i_min = int(np.argmin(gap))
    x_min = float(result["X"][nodes[i_min]])
    if abs(x_min) > 0.25:
        raise SystemExit(f"mmg check failed: deepest gap at x={x_min:.3g}, expected near x=0")
    xg = result["X"][nodes]
    far = np.abs(xg) > 0.45 * args.width
    if np.any(far) and float(np.min(gap[far])) < -1e-4:
        raise SystemExit("mmg check failed: far-field overlap (not a Hertz patch)")
    if float(np.min(gap)) < -0.25 * abs(args.indent):
        raise SystemExit(
            f"mmg check failed: contact face sucked in "
            f"(min gap {float(np.min(gap)):.3e} vs indent {args.indent:g})"
        )
    uy_c = np.asarray(result["u"])[2 * np.asarray(nodes) + 1]
    if float(np.min(uy_c)) < -1.25 * abs(args.indent):
        raise SystemExit(
            f"mmg check failed: contact uy={float(np.min(uy_c)):.3e} "
            f"past indent {args.indent:g}"
        )


def main(argv=None):
    args = parse_args(argv)
    ps = nc._load_pysfem()
    ps.init()
    try:
        if args.check:
            args.nx = 8
            args.ny = 4
            args.levels = 2
            args.max_iter = 20
            args.indent = 0.02
            args.plot = False
            args.mg_cycles = 2
            result = solve_mmg(ps, args)
            check_mmg(result, args)
            print(f"check ok  F={result['F']:.4e}  n_active={result['n_active']}")
        else:
            result = solve_mmg(ps, args)
            if args.plot:
                nc.plot_solution(result)
    finally:
        ps.finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
