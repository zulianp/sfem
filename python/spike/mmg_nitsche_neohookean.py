#!/usr/bin/env python3
"""Filtered Nitsche multigrid with compressible Neo-Hookean elasticity.

This script is a hyperelastic variant of ``mmg_nitsche.py``.  It keeps the
same nonlinear multigrid skeleton and jump-free coarse active-space filter,
but replaces the small-strain elastic residual/tangent and Nitsche stress
diagnostic with a compressible Neo-Hookean plane-strain model.

The hyperelastic tangent can be assembled by a closed-form TRI3 tangent or by
local finite differences for audit runs before moving the kernel to generated
or matrix-free code.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

import matplotlib
import numpy as np
from scipy import sparse

matplotlib.use("Agg")

_SPIKE = os.path.dirname(os.path.abspath(__file__))
if _SPIKE not in sys.path:
    sys.path.insert(0, _SPIKE)

import mmg_nitsche as mg
import nitsche_contact as nc


_LINEAR_BUILD_LEVEL = mg.build_level


def tri3_kinematics(px, py, u_elem):
    x0, x1, x2 = px
    y0, y1, y2 = py
    det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    if det <= 1e-30:
        raise RuntimeError(f"invalid TRI3 reference area det={det:.3e}")
    inv = 1.0 / det
    dNdx = np.array((y1 - y2, y2 - y0, y0 - y1), dtype=np.float64) * inv
    dNdy = np.array((x2 - x1, x0 - x2, x1 - x0), dtype=np.float64) * inv
    ux = u_elem[0::2]
    uy = u_elem[1::2]
    F = np.array(
        (
            (1.0 + float(dNdx @ ux), float(dNdy @ ux)),
            (float(dNdx @ uy), 1.0 + float(dNdy @ uy)),
        ),
        dtype=np.float64,
    )
    return 0.5 * det, dNdx, dNdy, F


def neo_pk1(F, mu, lam):
    J = float(np.linalg.det(F))
    if J <= 1e-12:
        raise FloatingPointError(f"inverted hyperelastic element J={J:.3e}")
    FinvT = np.linalg.inv(F).T
    return mu * (F - FinvT) + lam * np.log(J) * FinvT


def neo_cauchy(F, mu, lam):
    J = float(np.linalg.det(F))
    if J <= 1e-12:
        raise FloatingPointError(f"inverted hyperelastic element J={J:.3e}")
    b = F @ F.T
    return (mu / J) * (b - np.eye(2)) + (lam * np.log(J) / J) * np.eye(2)


def tri3_neo_residual(px, py, u_elem, mu, lam):
    area, dNdx, dNdy, F = tri3_kinematics(px, py, u_elem)
    P = neo_pk1(F, mu, lam)
    re = np.empty(6, dtype=np.float64)
    for a in range(3):
        grad = np.array((dNdx[a], dNdy[a]), dtype=np.float64)
        force = area * (P @ grad)
        re[2 * a] = force[0]
        re[2 * a + 1] = force[1]
    return re


def tri3_neo_tangent_fd(px, py, u_elem, mu, lam, fd_eps):
    ue = np.asarray(u_elem, dtype=np.float64)
    k = np.empty((6, 6), dtype=np.float64)
    r0 = tri3_neo_residual(px, py, ue, mu, lam)
    scale = max(1.0, float(np.linalg.norm(ue, ord=np.inf)))
    base_eps = max(float(fd_eps) * scale, 1e-10)
    for j in range(6):
        eps = base_eps * max(1.0, abs(float(ue[j])))
        up = ue.copy()
        um = ue.copy()
        up[j] += eps
        um[j] -= eps
        rp = rm = None
        try:
            rp = tri3_neo_residual(px, py, up, mu, lam)
        except FloatingPointError:
            pass
        try:
            rm = tri3_neo_residual(px, py, um, mu, lam)
        except FloatingPointError:
            pass
        if rp is not None and rm is not None:
            k[:, j] = (rp - rm) / (2.0 * eps)
        elif rp is not None:
            k[:, j] = (rp - r0) / eps
        elif rm is not None:
            k[:, j] = (r0 - rm) / eps
        else:
            k[:, j] = 0.0
    return k


def tri3_neo_tangent_analytic(px, py, u_elem, mu, lam):
    area, dNdx, dNdy, F = tri3_kinematics(px, py, u_elem)
    J = float(np.linalg.det(F))
    if J <= 1e-12:
        raise FloatingPointError(f"inverted hyperelastic element J={J:.3e}")
    G = np.linalg.inv(F).T
    coeff = lam * np.log(J) - mu
    k = np.empty((6, 6), dtype=np.float64)
    grads = ((dNdx[0], dNdy[0]), (dNdx[1], dNdy[1]), (dNdx[2], dNdy[2]))
    for a in range(3):
        for i in range(2):
            row = 2 * a + i
            for b in range(3):
                for m in range(2):
                    col = 2 * b + m
                    val = 0.0
                    for Jidx in range(2):
                        grad_a = grads[a][Jidx]
                        for Lidx in range(2):
                            A = (
                                (mu if i == m and Jidx == Lidx else 0.0)
                                + lam * G[m, Lidx] * G[i, Jidx]
                                - coeff * G[m, Jidx] * G[i, Lidx]
                            )
                            val += grad_a * A * grads[b][Lidx]
                    k[row, col] = area * val
    return k


def tri3_neo_sigma_n(px, py, u_elem, mu, lam, nx, ny):
    _, _, _, F = tri3_kinematics(px, py, u_elem)
    sigma = neo_cauchy(F, mu, lam)
    n = np.array((nx, ny), dtype=np.float64)
    return float(n @ sigma @ n)


def tri3_neo_sigma_n_fd(px, py, u_elem, mu, lam, nx, ny, fd_eps):
    ue = np.asarray(u_elem, dtype=np.float64)
    sn0 = tri3_neo_sigma_n(px, py, ue, mu, lam, nx, ny)
    dsn = np.empty(6, dtype=np.float64)
    scale = max(1.0, float(np.linalg.norm(ue, ord=np.inf)))
    base_eps = max(float(fd_eps) * scale, 1e-10)
    for j in range(6):
        eps = base_eps * max(1.0, abs(float(ue[j])))
        up = ue.copy()
        um = ue.copy()
        up[j] += eps
        um[j] -= eps
        sp = sm = None
        try:
            sp = tri3_neo_sigma_n(px, py, up, mu, lam, nx, ny)
        except FloatingPointError:
            pass
        try:
            sm = tri3_neo_sigma_n(px, py, um, mu, lam, nx, ny)
        except FloatingPointError:
            pass
        if sp is not None and sm is not None:
            dsn[j] = (sp - sm) / (2.0 * eps)
        elif sp is not None:
            dsn[j] = (sp - sn0) / eps
        elif sm is not None:
            dsn[j] = (sn0 - sm) / eps
        else:
            dsn[j] = 0.0
    return sn0, dsn


def neo_normal_penalty_modulus(px, py, u_elem, mu, lam, nx, ny, fd_eps):
    _, _, _, F = tri3_kinematics(px, py, u_elem)
    n = np.array((nx, ny), dtype=np.float64)
    nn = float(np.linalg.norm(n))
    if nn <= 1e-30:
        return float(mu)
    n *= 1.0 / nn
    dF = np.outer(n, n)
    eps = max(float(fd_eps), 1e-7)

    def sigma_nn(F_eval):
        sigma = neo_cauchy(F_eval, mu, lam)
        return float(n @ sigma @ n)

    s0 = sigma_nn(F)
    try:
        sp = sigma_nn(F + eps * dF)
        sm = sigma_nn(F - eps * dF)
        kn = (sp - sm) / (2.0 * eps)
    except FloatingPointError:
        sp = sigma_nn(F + eps * dF)
        kn = (sp - s0) / eps
    if not np.isfinite(kn) or kn <= 0.0:
        return float(mu)
    return float(max(kn, mu))


def contact_penalty_gamma(length, px, py, u_elem, mu, lam, gamma0, nx, ny, fd_eps, scaling):
    if scaling == "normal-tangent":
        modulus = neo_normal_penalty_modulus(px, py, u_elem, mu, lam, nx, ny, fd_eps)
    elif scaling == "shear":
        modulus = mu
    else:
        raise ValueError(f"unknown contact penalty scaling '{scaling}'")
    return gamma0 * modulus / length


def assemble_neo_elastic(level, u_vec, assemble_tangent=True):
    ndofs = level.ndofs
    residual = np.zeros(ndofs, dtype=np.float64)
    rows, cols, data = [], [], []
    blocks = (
        (level.tris_b, level.mu_b, level.lam_b),
        (level.tris_o, level.mu_o, level.lam_o),
    )
    for tris, mu, lam in blocks:
        for nodes in tris:
            nodes = np.asarray(nodes, dtype=np.int32)
            px = level.X[nodes]
            py = level.Y[nodes]
            dofs = np.empty(6, dtype=np.int64)
            for a, node in enumerate(nodes):
                dofs[2 * a] = 2 * int(node)
                dofs[2 * a + 1] = 2 * int(node) + 1
            ue = u_vec[dofs]
            re = tri3_neo_residual(px, py, ue, mu, lam)
            residual[dofs] += re
            if assemble_tangent:
                if getattr(level, "material_tangent", "analytic") == "fd":
                    ke = tri3_neo_tangent_fd(px, py, ue, mu, lam, level.fd_eps)
                else:
                    ke = tri3_neo_tangent_analytic(px, py, ue, mu, lam)
                for i in range(6):
                    for j in range(6):
                        val = ke[i, j]
                        if val != 0.0:
                            rows.append(int(dofs[i]))
                            cols.append(int(dofs[j]))
                            data.append(float(val))
    if assemble_tangent:
        K = sparse.coo_matrix((data, (rows, cols)), shape=(ndofs, ndofs)).tocsr()
    else:
        K = None
    return residual, K


def min_neo_element_jacobian(level, u_vec):
    j_min = np.inf
    for tris in (level.tris_b, level.tris_o):
        for nodes in tris:
            nodes = np.asarray(nodes, dtype=np.int32)
            px = level.X[nodes]
            py = level.Y[nodes]
            dofs = np.empty(6, dtype=np.int64)
            for a, node in enumerate(nodes):
                dofs[2 * a] = 2 * int(node)
                dofs[2 * a + 1] = 2 * int(node) + 1
            _, _, _, F = tri3_kinematics(px, py, u_vec[dofs])
            j_min = min(j_min, float(np.linalg.det(F)))
    return j_min


def neo_surface_contrib(
    X,
    Y,
    u,
    edges,
    parent_elems,
    parent_block,
    other_edges,
    ps,
    mesh,
    mu,
    lam,
    gamma0,
    theta,
    friction,
    mu_f,
    residual,
    coo,
    circle_R=None,
    surface_id=0,
    new_active=None,
    snap_self_circle=False,
    energy_acc=None,
    qp_out=None,
    assemble=True,
    include_sigma=True,
    forced_active=None,
    status_out=None,
    gap_rows_out=None,
    frozen_edges=None,
    fd_eps=1e-7,
    penalty_scaling="shear",
):
    if friction:
        raise NotImplementedError("friction is not implemented in the Neo-Hookean spike")
    if new_active is None:
        new_active = set()
    n_active = 0
    for ie, (edge, e_parent) in enumerate(zip(edges, parent_elems)):
        n0, n1 = int(edge[0]), int(edge[1])
        length, nx0, ny0 = nc.edge_geometry(X, Y, n0, n1)[:3]
        if length <= 1e-16:
            continue
        parent_nodes = nc.tri3_parent_nodes(ps, mesh, parent_block, int(e_parent))
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
                X, Y, u, n0, n1, length, nx0, ny0, other_edges, circle_R, snap_self_circle
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

        for w, xi, g, nx, ny, tx, ty, Na, Nb, m0, m1, Nm0, Nm1, xref in samples:
            w_int += w
            g_int += w * g
            if abs(xi - 0.5) < abs(mid[1] - 0.5):
                mid = (w, xi, g, nx, ny, tx, ty, Na, Nb, m0, m1, Nm0, Nm1, xref)
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
        sn, dsn = tri3_neo_sigma_n_fd(px, py, u_elem, mu, lam, mid[3], mid[4], fd_eps)
        gamma = contact_penalty_gamma(
            length, px, py, u_elem, mu, lam, gamma0, mid[3], mid[4], fd_eps, penalty_scaling
        )
        Pn = (theta * sn if include_sigma else 0.0) + gamma * g_bar
        key = (surface_id, ie, 0)
        xref_mid = 0.5 * (X[n0] + X[n1])
        if gap_rows_out is not None:
            gap_rows_out.append((key, float(g_bar), float(xref_mid), dict(dg_bar), float(w_int)))
        if status_out is not None:
            status_out[key] = (g_bar, Pn, xref_mid)
        on = Pn < 0.0 if forced_active is None else key in forced_active
        if not on:
            continue
        new_active.add(key)
        n_active += 1
        scale = (1.0 / theta) * (1.0 / gamma) * w_int
        if energy_acc is not None:
            energy_acc[0] += 0.5 * scale * Pn * Pn
        if qp_out is not None:
            qp_out.append((xref_mid, g_bar, Pn))
        if not assemble:
            continue
        pv = {}
        for dof, dgd in dg_bar.items():
            val = gamma * dgd
            if val != 0.0:
                pv[dof] = pv.get(dof, 0.0) + val
        if include_sigma:
            for a, node in enumerate(parent_nodes):
                node = int(node)
                pv[2 * node] = pv.get(2 * node, 0.0) + theta * dsn[2 * a]
                pv[2 * node + 1] = pv.get(2 * node + 1, 0.0) + theta * dsn[2 * a + 1]
        for dof, pvi in pv.items():
            residual[dof] += scale * Pn * pvi
        for di, pi in pv.items():
            for dj, pj in pv.items():
                val = scale * pi * pj
                if val != 0.0:
                    coo.append((di, dj, val))
    return n_active


def contact_trace_neo(
    X,
    Y,
    u,
    edges,
    parent_elems,
    parent_block,
    other_edges,
    ps,
    mesh,
    mu,
    lam,
    gamma0,
    theta,
    circle_R,
    include_sigma,
    snap_self_circle=False,
    g_open=0.0,
    fd_eps=1e-7,
    penalty_scaling="shear",
):
    xs, gs, sns, pns, active, xis, ws = [], [], [], [], [], [], []
    for edge, e_parent in zip(edges, parent_elems):
        n0, n1 = int(edge[0]), int(edge[1])
        length, nx0, ny0 = nc.edge_geometry(X, Y, n0, n1)[:3]
        if length <= 1e-16:
            continue
        parent_nodes = nc.tri3_parent_nodes(ps, mesh, parent_block, int(e_parent))
        px = np.array([X[i] for i in parent_nodes], dtype=np.float64)
        py = np.array([Y[i] for i in parent_nodes], dtype=np.float64)
        u_elem = np.empty(6, dtype=np.float64)
        for a, node in enumerate(parent_nodes):
            u_elem[2 * a] = u[2 * node]
            u_elem[2 * a + 1] = u[2 * node + 1]
        samples = nc.collect_edge_qps(
            X, Y, u, n0, n1, length, nx0, ny0, other_edges, circle_R, snap_self_circle
        )
        if not samples:
            continue
        w_int = 0.0
        g_int = 0.0
        mid = samples[0]
        for s in samples:
            w_int += s[0]
            g_int += s[0] * s[2]
            if abs(s[1] - 0.5) < abs(mid[1] - 0.5):
                mid = s
        g_bar = g_int / w_int
        sn = tri3_neo_sigma_n(px, py, u_elem, mu, lam, mid[3], mid[4])
        gamma = contact_penalty_gamma(
            length, px, py, u_elem, mu, lam, gamma0, mid[3], mid[4], fd_eps, penalty_scaling
        )
        Pn = (theta * sn if include_sigma else 0.0) + gamma * g_bar
        on = Pn < 0.0
        xs.append(0.5 * (X[n0] + X[n1]))
        gs.append(g_bar)
        sns.append(sn)
        pns.append(Pn if on else 0.0)
        active.append(on)
        xis.append(0.5)
        ws.append(w_int)
    return (
        np.asarray(xs),
        np.asarray(gs),
        np.asarray(sns),
        np.asarray(pns),
        np.asarray(active, dtype=bool),
        np.asarray(xis),
        np.asarray(ws),
    )


def build_level(ps, args):
    level = _LINEAR_BUILD_LEVEL(ps, args)
    level.tris_b = nc.block_triangles(ps, level.mesh, level.bid_b)
    level.tris_o = nc.block_triangles(ps, level.mesh, level.bid_o)
    level.fd_eps = float(args.fd_eps)
    level.material_tangent = getattr(args, "material_tangent", "analytic")
    level.material_linearization = getattr(args, "material_linearization", "every-call")
    level.contact_penalty_scaling = getattr(args, "contact_penalty_scaling", "shear")
    level.cached_K_elastic = None

    def refresh_material_tangent(u_vec):
        _, K_elastic = assemble_neo_elastic(level, u_vec, assemble_tangent=True)
        level.cached_K_elastic = K_elastic.tocsr()
        return level.cached_K_elastic

    def begin_vcycle(u_vec):
        if level.material_linearization == "every-vcycle":
            refresh_material_tangent(u_vec)

    def elastic_residual_tangent(u_vec, force_refresh=False):
        if level.material_linearization == "every-call" or force_refresh:
            return assemble_neo_elastic(level, u_vec, assemble_tangent=True)
        r_elastic, _ = assemble_neo_elastic(level, u_vec, assemble_tangent=False)
        if level.cached_K_elastic is None:
            refresh_material_tangent(u_vec)
        return r_elastic, level.cached_K_elastic

    def elastic_residual(u_vec):
        return assemble_neo_elastic(level, u_vec, assemble_tangent=False)[0]

    def min_element_jacobian(u_vec):
        return min_neo_element_jacobian(level, u_vec)

    def residual_tangent(u_vec, forced_active=None, frozen_geom=None):
        r_elastic, K_elastic = elastic_residual_tangent(u_vec)
        g_contact = np.zeros(level.ndofs, dtype=np.float64)
        coo = []
        new_active = set()
        n_qp = 0
        frozen_b = None if frozen_geom is None else frozen_geom.get(0)
        frozen_o = None if frozen_geom is None else frozen_geom.get(1)
        if level.theta_b > 0:
            n_qp += neo_surface_contrib(
                level.X, level.Y, u_vec, level.edges_b, level.elems_b, level.bid_b,
                level.edges_o, ps, level.mesh, level.mu_b, level.lam_b,
                args.gamma0, level.theta_b, False, args.mu_f, g_contact, coo, level.radius,
                0, new_active, False, None, None, True, level.include_sigma, forced_active,
                None, None, frozen_b, level.fd_eps, level.contact_penalty_scaling,
            )
        if level.theta_o > 0:
            n_qp += neo_surface_contrib(
                level.X, level.Y, u_vec, level.edges_o, level.elems_o, level.bid_o,
                level.edges_b, ps, level.mesh, level.mu_o, level.lam_o,
                args.gamma0, level.theta_o, False, args.mu_f, g_contact, coo, level.radius,
                1, new_active, True, None, None, True, level.include_sigma, forced_active,
                None, None, frozen_o, level.fd_eps, level.contact_penalty_scaling,
            )
        if coo:
            ii, jj, vv = zip(*coo)
            Kn = sparse.coo_matrix((vv, (ii, jj)), shape=(level.ndofs, level.ndofs)).tocsr()
        else:
            Kn = sparse.csr_matrix((level.ndofs, level.ndofs))
        residual = r_elastic + g_contact
        Ksys = K_elastic + Kn
        Ksys, residual = nc.apply_dirichlet_system(
            Ksys, residual, level.constrained, u_vec, level.u_bc
        )
        return residual, Ksys, g_contact, Kn, n_qp, new_active

    reference_u = np.zeros_like(level.u_bc)
    _, K0 = elastic_residual_tangent(reference_u, force_refresh=True)
    level.Ke = K0.tocsr()
    level.refresh_material_tangent = refresh_material_tangent
    level.begin_vcycle = begin_vcycle
    level.elastic_residual = elastic_residual
    level.elastic_residual_tangent = elastic_residual_tangent
    level.min_element_jacobian = min_element_jacobian
    level.residual_tangent = residual_tangent
    return level


def pack_result(fine, args, u, r_hist, n_active_hist, n_qp_hist, vcycle_hist=None, direct_hist=None):
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
    r_full = np.asarray(fine.elastic_residual(u) + g_c, dtype=np.float64)
    tol = 1e-3 * max(args.width, args.radius) / max(args.nx, 1)
    y_top = float(np.max(Y[:n_block]))
    top = np.flatnonzero(np.abs(Y[:n_block] - y_top) <= 4.0 * tol)
    F = abs(float(np.sum(r_full[2 * top + 1]))) if top.size else 0.0
    E_b = args.E_block * (args.rigid_stiffness if args.rigid_block else 1.0)
    E_o = args.E_obstacle * (args.rigid_stiffness if args.rigid_obstacle else 1.0)
    E_o_hertz = E_o if not args.rigid_obstacle else 1e300
    E_star = nc.effective_modulus(E_b, args.nu, E_o_hertz, args.nu)
    xc = 0.5 * (X[fine.edges_b[:, 0]] + X[fine.edges_b[:, 1]])
    p_hertz, a, p0 = nc.hertz_pressure(xc, F, args.radius, E_star)
    tr_x, tr_g, tr_sn, tr_pn, tr_on, tr_xi, tr_w = contact_trace_neo(
        X, Y, u, fine.edges_b, fine.elems_b, fine.bid_b, fine.edges_o, fine.ps, fine.mesh,
        fine.mu_b, fine.lam_b, args.gamma0, fine.theta_b if fine.theta_b > 0 else 1.0,
        fine.radius, fine.include_sigma, False, 0.0, fine.fd_eps, fine.contact_penalty_scaling,
    )
    tr_x_o, tr_g_o, tr_sn_o, tr_pn_o, tr_on_o, _, tr_w_o = contact_trace_neo(
        X, Y, u, fine.edges_o, fine.elems_o, fine.bid_o, fine.edges_b, fine.ps, fine.mesh,
        fine.mu_o, fine.lam_o, args.gamma0, fine.theta_o if fine.theta_o > 0 else 1.0,
        fine.radius, fine.include_sigma, True, 0.0, fine.fd_eps, fine.contact_penalty_scaling,
    )
    th = fine.theta_b if fine.theta_b > 0.0 else 1.0
    p_applied = np.where(tr_on, np.maximum(-tr_pn / th, 0.0), 0.0)
    n_active = int(np.count_nonzero(tr_on))
    min_J = min_neo_element_jacobian(fine, u)
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
        "direct_hist": [] if direct_hist is None else [bool(v) for v in direct_hist],
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
        "min_J": float(min_J),
        "gamma0": float(args.gamma0),
        "contact_penalty_scaling": str(fine.contact_penalty_scaling),
    }


def plot_result(result, path):
    import matplotlib.pyplot as plt

    X = np.asarray(result["X"], dtype=np.float64)
    Y = np.asarray(result["Y"], dtype=np.float64)
    u = np.asarray(result["u"], dtype=np.float64)
    xd = X + u[0::2]
    yd = Y + u[1::2]
    uy = u[1::2]
    tris_b = np.asarray(result["tris_b"], dtype=np.int32)
    tris_o = np.asarray(result["tris_o"], dtype=np.int32)

    fig, axs = plt.subplots(2, 2, figsize=(13.0, 8.5), constrained_layout=True)
    ax_mesh, ax_gap, ax_stress, ax_conv = axs.ravel()

    vmax = max(float(np.max(np.abs(uy))), 1e-14)
    for tris in (tris_b, tris_o):
        tpc = ax_mesh.tripcolor(
            xd, yd, tris, uy, shading="gouraud", cmap="coolwarm", vmin=-vmax, vmax=vmax
        )
        ax_mesh.triplot(xd, yd, tris, color="k", lw=0.22, alpha=0.45)
    fig.colorbar(tpc, ax=ax_mesh, label=r"$u_y$")
    ax_mesh.set_aspect("equal", adjustable="box")
    ax_mesh.set_xlabel("x")
    ax_mesh.set_ylabel("y")
    ax_mesh.set_title("deformed solution")

    nodes_b = np.asarray(result["nodes_b"], dtype=np.int32)
    gap = np.asarray(result["gap"], dtype=np.float64)
    order_gap = np.argsort(X[nodes_b])
    ax_gap.plot(X[nodes_b][order_gap], gap[order_gap], "o-", ms=3.5, lw=1.4, label="nodal gap")
    tr_x = np.asarray(result.get("qp_x", []), dtype=np.float64)
    tr_g = np.asarray(result.get("qp_g", []), dtype=np.float64)
    qp_active = np.asarray(result.get("qp_active", []), dtype=bool)
    if tr_x.size and tr_g.size:
        order = np.argsort(tr_x)
        ax_gap.plot(tr_x[order], tr_g[order], "s-", ms=3.0, lw=1.2, label="edge mean gap")
        if qp_active.size == tr_x.size:
            ax_gap.scatter(
                tr_x[qp_active], tr_g[qp_active], s=42, marker="o",
                facecolors="none", edgecolors="C3", linewidths=1.5, label="active rows"
            )
    ax_gap.axhline(0.0, color="0.35", lw=0.8)
    ax_gap.set_xlabel("x")
    ax_gap.set_ylabel("gap")
    ax_gap.set_title("contact gap")
    ax_gap.grid(True, ls=":", alpha=0.35)
    ax_gap.legend(loc="best", fontsize=8)

    p_active = np.asarray(result.get("p_applied", []), dtype=np.float64)
    p_block = np.asarray(result.get("p_cauchy", []), dtype=np.float64)
    p_obs = np.asarray(result.get("p_cauchy_o", []), dtype=np.float64)
    tr_x_o = np.asarray(result.get("qp_x_o", []), dtype=np.float64)
    xc = np.asarray(result.get("xc", []), dtype=np.float64)
    p_hertz = np.asarray(result.get("p_hertz", []), dtype=np.float64)
    if tr_x.size and p_active.size:
        order = np.argsort(tr_x)
        ax_stress.plot(
            tr_x[order], p_active[order], "o-", color="C3", lw=2.2, ms=3.2,
            label=r"active $-P_n/\theta$"
        )
    if tr_x.size and p_block.size:
        order = np.argsort(tr_x)
        ax_stress.plot(
            tr_x[order], p_block[order], drawstyle="steps-mid", color="C0", lw=1.5,
            label=r"block $-\sigma_n$"
        )
    if tr_x_o.size and p_obs.size:
        order = np.argsort(tr_x_o)
        ax_stress.plot(
            tr_x_o[order], p_obs[order], drawstyle="steps-mid", color="C2", lw=1.5,
            label=r"obstacle $-\sigma_n$"
        )
    if xc.size and p_hertz.size:
        order = np.argsort(xc)
        ax_stress.plot(xc[order], p_hertz[order], "--", color="C1", lw=1.4, label="Hertz")
    if tr_x.size and qp_active.size == tr_x.size:
        for x_active in tr_x[qp_active]:
            ax_stress.axvline(x_active, color="C3", lw=0.8, alpha=0.18)
    ax_stress.axhline(0.0, color="0.35", lw=0.8)
    ax_stress.set_xlabel("x")
    ax_stress.set_ylabel("pressure / normal stress")
    ax_stress.set_title("contact stresses")
    ax_stress.grid(True, ls=":", alpha=0.35)
    ax_stress.legend(loc="best", fontsize=8)

    rh = np.asarray(result.get("r_hist", []), dtype=np.float64)
    if rh.size:
        xs = np.arange(1, rh.size + 1)
        ax_conv.semilogy(xs, np.maximum(rh, 1e-30), "o-", color="C4", ms=3.5, label=r"$\|r\|$")
        ax_conv.axhline(1e-10, color="0.35", ls="--", lw=0.9, label=r"$10^{-10}$")
        ah = np.asarray(result.get("n_active_hist", []), dtype=np.float64)
        if ah.size == rh.size:
            ax_active = ax_conv.twinx()
            ax_active.plot(xs, ah, "s--", color="C5", ms=3.0, label=r"$|A|$")
            ax_active.set_ylabel("active rows")
            ax_active.set_ylim(bottom=-0.2)
        ax_conv.set_xlabel("outer iteration record")
        ax_conv.set_ylabel(r"$\|r\|$")
        ax_conv.set_title("convergence")
        ax_conv.grid(True, which="both", ls=":", alpha=0.35)
        lines, labels = ax_conv.get_legend_handles_labels()
        if "ax_active" in locals():
            lines2, labels2 = ax_active.get_legend_handles_labels()
            lines += lines2
            labels += labels2
        ax_conv.legend(lines, labels, loc="best", fontsize=8)

    fig.suptitle(
        "Compressible Neo-Hookean Nitsche contact: "
        f"F={result['F']:.4e}, |g_-|={result['penetration']:.3e}, "
        f"active={result['n_active']}, final ||r||={rh[-1]:.3e}"
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"saved {path}")


def load_step_plot_path(path, step_index):
    root, ext = os.path.splitext(path)
    if not ext:
        ext = ".png"
    return f"{root}_load{step_index:03d}{ext}"


def solve_incremental_load(ps, args):
    target_indent = float(args.indent)
    n_steps = max(1, int(args.load_steps))
    max_accepted = max(n_steps, int(getattr(args, "load_max_accepted", 100)))
    max_attempts = max(max_accepted, int(getattr(args, "load_max_attempts", 4 * max_accepted)))
    min_delta = abs(target_indent) * float(args.load_min_fraction)
    if target_indent == 0.0:
        min_delta = float(args.load_min_fraction)
    if min_delta <= 0.0:
        raise ValueError("--load-min-fraction must be positive")

    current_indent = 0.0
    step_delta = target_indent / n_steps
    previous_u = None
    last_result = None
    accepted = 0
    attempts = 0
    load_history = []

    while abs(target_indent - current_indent) > 1e-15:
        if accepted >= max_accepted:
            raise RuntimeError(
                f"incremental load stopped after {accepted} accepted steps at "
                f"indent={current_indent:.6e}, target={target_indent:.6e}"
            )
        if attempts >= max_attempts:
            raise RuntimeError(
                f"incremental load stopped after {attempts} attempts at "
                f"indent={current_indent:.6e}, target={target_indent:.6e}"
            )
        remaining = target_indent - current_indent
        if abs(step_delta) > abs(remaining):
            step_delta = remaining
        trial_indent = current_indent + step_delta
        trial_args = copy.copy(args)
        trial_args.indent = trial_indent
        trial_args.plot = False
        attempts += 1
        print(
            f"load_step trial={attempts} accepted={accepted} indent={trial_indent:.6e} "
            f"delta={step_delta:.6e}"
        )
        try:
            result = mg.solve_mmg(ps, trial_args, initial_u=previous_u)
            u = result["u"]
            j_min = float(result.get("min_J", np.inf))
            min_j = float(args.coarse_linesearch_min_j)
            if not np.isfinite(j_min) or j_min <= min_j:
                raise FloatingPointError(f"load step produced min J={j_min:.6e}")
            r_hist = np.asarray(result.get("r_hist", []), dtype=np.float64)
            final_residual = float(r_hist[-1]) if r_hist.size else float("inf")
            if not np.isfinite(final_residual) or final_residual > float(args.atol):
                raise RuntimeError(
                    f"load step did not reach absolute residual tolerance: "
                    f"||r||={final_residual:.6e}, atol={float(args.atol):.6e}"
                )
        except (FloatingPointError, RuntimeError, ValueError) as exc:
            load_history.append(
                {
                    "accepted": False,
                    "attempt": attempts,
                    "from_indent": float(current_indent),
                    "trial_indent": float(trial_indent),
                    "delta": float(step_delta),
                    "reason": str(exc),
                }
            )
            if abs(step_delta) <= min_delta:
                raise RuntimeError(
                    f"load step failed at indent={trial_indent:.6e} "
                    f"with minimum allowed delta={min_delta:.6e}"
                ) from exc
            step_delta *= float(args.load_reduction)
            print(f"load_step rejected: {exc}; reducing delta to {step_delta:.6e}")
            continue

        accepted += 1
        current_indent = trial_indent
        previous_u = u.copy()
        last_result = result
        load_history.append(
            {
                "accepted": True,
                "attempt": attempts,
                "accepted_index": accepted,
                "from_indent": float(current_indent - step_delta),
                "trial_indent": float(current_indent),
                "delta": float(step_delta),
                "min_J": float(j_min),
                "final_residual": float(result["r_hist"][-1]),
                "n_active": int(result["n_active"]),
                "penetration": float(result["penetration"]),
            }
        )
        print(
            f"load_step accepted={accepted} indent={current_indent:.6e} "
            f"min_J={j_min:.6e} final ||r||={result['r_hist'][-1]:.3e}"
        )
        if args.plot_load_steps:
            plot_result(result, load_step_plot_path(args.plot_output, accepted))
        if abs(target_indent - current_indent) <= 1e-15:
            break
        remaining_steps = max(1, n_steps - accepted)
        step_delta = (target_indent - current_indent) / remaining_steps

    if last_result is None:
        raise RuntimeError("incremental load did not accept any load step")
    last_result["load_history"] = load_history
    last_result["target_indent"] = float(target_indent)
    last_result["reached_target"] = bool(abs(target_indent - current_indent) <= 1e-15)
    return last_result


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
    p.add_argument(
        "--load-steps",
        type=int,
        default=1,
        help="Apply indentation in this many external load steps.",
    )
    p.add_argument(
        "--load-reduction",
        type=float,
        default=0.5,
        help="Factor used to reduce a failed external load increment.",
    )
    p.add_argument(
        "--load-min-fraction",
        type=float,
        default=1e-3,
        help="Minimum external load increment as a fraction of the target indentation.",
    )
    p.add_argument(
        "--plot-load-steps",
        action="store_true",
        help="Write one plot for every accepted external load step.",
    )
    p.add_argument(
        "--load-max-accepted",
        type=int,
        default=100,
        help="Maximum accepted external load steps after adaptive reductions.",
    )
    p.add_argument(
        "--load-max-attempts",
        type=int,
        default=400,
        help="Maximum attempted external load steps, including rejected steps.",
    )
    p.add_argument("--E-block", type=float, default=1.0)
    p.add_argument("--E-obstacle", type=float, default=1.0)
    p.add_argument("--nu", type=float, default=0.3)
    p.add_argument("--gamma0", type=float, default=50.0)
    p.add_argument(
        "--contact-penalty-scaling",
        choices=("shear", "normal-tangent"),
        default="shear",
        help=(
            "Scale Nitsche gamma with mu/h or with a local finite-deformation "
            "normal tangent modulus divided by h."
        ),
    )
    p.add_argument("--fd-eps", type=float, default=1e-7)
    p.add_argument(
        "--material-tangent",
        choices=("analytic", "fd"),
        default="analytic",
        help="Use the closed-form Neo-Hookean element tangent or finite differences.",
    )
    p.add_argument(
        "--material-linearization",
        choices=("every-call", "every-vcycle"),
        default="every-call",
        help="Refresh the Neo-Hookean elastic tangent at every residual/tangent call or once at the start of each V-cycle.",
    )
    p.add_argument("--max-iter", type=int, default=40)
    p.add_argument("--max-inner-it", type=int, default=3)
    p.add_argument("--nlsmooth-steps", type=int, default=3)
    p.add_argument("--cycle-type", type=int, default=1, choices=(1, 2))
    p.add_argument("--atol", type=float, default=1e-10)
    p.add_argument("--ptol", type=float, default=float("inf"))
    p.add_argument("--rtol", type=float, default=1e-10)
    p.add_argument("--mg-pre", type=int, default=8)
    p.add_argument("--mg-post", type=int, default=8)
    p.add_argument("--mg-omega", type=float, default=0.25)
    p.add_argument("--smoother", choices=("jacobi", "sgs", "gs", "block", "scalar"), default="jacobi")
    p.add_argument(
        "--coarse-tangent",
        choices=("galerkin", "rediscretized"),
        default="galerkin",
        help="Use inherited Galerkin coarse operators or assemble each coarse tangent at projected displacement.",
    )
    p.add_argument(
        "--coarse-displacement-projection",
        choices=("injection", "l2"),
        default="injection",
        help="Projection used for rediscretized coarse tangents.",
    )
    p.add_argument("--coarse-linesearch", action="store_true")
    p.add_argument(
        "--coarse-linesearch-mode",
        choices=("residual", "inversion"),
        default="residual",
        help="Backtrack coarse corrections by residual decrease or only by positive element Jacobian.",
    )
    p.add_argument("--coarse-linesearch-reduction", type=float, default=0.5)
    p.add_argument("--coarse-linesearch-min-alpha", type=float, default=1e-3)
    p.add_argument("--coarse-linesearch-c1", type=float, default=0.0)
    p.add_argument("--coarse-linesearch-min-j", type=float, default=1e-10)
    p.add_argument("--smooth-linesearch", action="store_true")
    p.add_argument("--smooth-linesearch-reduction", type=float, default=0.5)
    p.add_argument("--smooth-linesearch-min-alpha", type=float, default=1e-3)
    p.add_argument("--smooth-linesearch-min-j", type=float, default=1e-10)
    p.add_argument("--stagnation-threshold", type=float, default=0.999)
    p.add_argument("--skip-coarse", action="store_true")
    p.add_argument("--mu-f", type=float, default=0.3)
    p.add_argument("--rigid-block", action="store_true")
    p.add_argument("--rigid-obstacle", action="store_true")
    p.add_argument("--rigid-stiffness", type=float, default=1e4)
    p.add_argument("--biased", action="store_true")
    p.add_argument("--unbiased", action="store_true")
    p.add_argument("--penalty", action="store_true")
    p.add_argument("--lagrange", action="store_true")
    p.add_argument("--plot", action="store_true")
    p.add_argument(
        "--plot-output",
        default=os.path.join(_SPIKE, "mmg_nitsche_neohookean.png"),
        help="PNG path used when --plot is enabled.",
    )
    p.add_argument("--check", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    mg.build_level = build_level
    mg.pack_result = pack_result
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
            result = mg.solve_mmg(ps, args)
            mg.check_mmg(result, args)
            print(f"check ok  F={result['F']:.4e}  n_active={result['n_active']}")
        else:
            if args.load_steps > 1:
                if not (0.0 < args.load_reduction < 1.0):
                    raise ValueError("--load-reduction must be in (0, 1)")
                result = solve_incremental_load(ps, args)
            else:
                result = mg.solve_mmg(ps, args)
            if args.plot:
                plot_result(result, args.plot_output)
    finally:
        ps.finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
