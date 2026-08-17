#!/usr/bin/env python3
"""Two-body unbiased Nitsche contact (Mlika, Renard, Chouly).

Frictionless first; Coulomb friction with --friction. Hertz geometry: elastic
(or rigid) rectangular block pressed onto an elastic (or rigid) hemi-annulus.

Requires pysfem (build_py with -DSFEM_ENABLE_PYTHON=ON). See
python/spike/run_nitsche_contact.sh and plans/NITSCHE_CONTACT.md.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def _load_pysfem():
    try:
        import pysfem as ps
        return ps
    except ImportError as err:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        candidates = [
            os.path.join(root, "build_py", "python", "bindings"),
            os.path.join(root, "build", "python", "bindings"),
        ]
        for path in candidates:
            if os.path.isdir(path) and path not in sys.path:
                sys.path.insert(0, path)
        try:
            import pysfem as ps
            return ps
        except ImportError:
            raise ImportError(
                "pysfem is not importable. Build it in build_py "
                "(-DSFEM_ENABLE_PYTHON=ON) and set PYTHONPATH to "
                "build_py/python/bindings."
            ) from err


TRI3_SIDES = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int32)
# Simpson on [0, 1] includes vertices so the Hertz point x=0 is sampled.
# 2-point Gauss sits on coarse TRI3 chords inside the circle and can miss contact.
QUAD_1D = ((0.0, 1.0 / 6.0), (0.5, 4.0 / 6.0), (1.0, 1.0 / 6.0))


def lame(E, nu, plane_strain=True):
    if plane_strain:
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
    else:
        lam = E * nu / (1 - nu * nu)
        mu = E / (2 * (1 + nu))
    return lam, mu


def effective_modulus(E1, nu1, E2, nu2):
    return 1.0 / ((1 - nu1 * nu1) / E1 + (1 - nu2 * nu2) / E2)


def numpy_view(ps, buf):
    return np.array(ps.numpy_view(buf), copy=False)


def sidesets_from_selector(ps, mesh, selector, block_names=None):
    if block_names is None:
        return list(ps.Sideset.create_from_selector(mesh, selector))
    return list(ps.Sideset.create_from_selector(mesh, selector, list(block_names)))


def flatten_sidesets(groups):
    out = []
    for g in groups:
        if g is None:
            continue
        if isinstance(g, (list, tuple)):
            out.extend(g)
        else:
            out.append(g)
    return [s for s in out if s is not None and s.size() > 0]


def dirichlet_condition(ps, sidesets, component, value):
    c = ps.DirichletCondition()
    c.sidesets = flatten_sidesets([sidesets])
    c.component = int(component)
    c.value = float(value)
    return c


def sideset_edges(ps, mesh, sideset):
    bid = int(sideset.block_id())
    parent = np.array(numpy_view(ps, sideset.parent()), copy=True)
    lfi = np.array(numpy_view(ps, sideset.lfi()), copy=True)
    nxe = int(mesh.n_nodes_per_element(bid))
    conn = [np.array(ps.elements(mesh, v, bid), copy=True) for v in range(nxe)]
    n = parent.size
    edges = np.empty((n, 2), dtype=np.int32)
    elems = np.empty(n, dtype=np.int32)
    for i in range(n):
        e = int(parent[i])
        s = int(lfi[i])
        a, b = TRI3_SIDES[s]
        edges[i, 0] = int(conn[a][e])
        edges[i, 1] = int(conn[b][e])
        elems[i] = e
    return edges, elems, bid


def unique_nodes_from_edges(edges):
    return np.unique(edges.reshape(-1))


def block_triangles(ps, mesh, block_id):
    nxe = int(mesh.n_nodes_per_element(block_id))
    conn = [np.array(ps.elements(mesh, v, block_id), copy=True) for v in range(min(nxe, 3))]
    return np.column_stack(conn).astype(np.int32)


def orient_edges(X, Y, edges, mode):
    out = np.array(edges, copy=True)
    for i, (n0, n1) in enumerate(out):
        _, nx, ny, _ = edge_geometry(X, Y, int(n0), int(n1))
        mx = 0.5 * (X[n0] + X[n1])
        my = 0.5 * (Y[n0] + Y[n1])
        flip = (mode == "down" and ny > 0) or (
            mode == "outward" and (nx * mx + ny * my) < 0
        )
        if flip:
            out[i, 0], out[i, 1] = n1, n0
    return out


def filter_edges(X, Y, edges, elems, pred):
    keep = [i for i, (n0, n1) in enumerate(edges) if pred(int(n0), int(n1))]
    if not keep:
        return edges[:0], elems[:0]
    return edges[keep], elems[keep]


def map_square_to_hemi_annulus(ps, mesh, r_in, r_out):
    x = np.array(ps.points(mesh, 0), copy=False)
    y = np.array(ps.points(mesh, 1), copy=False)
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    xi = (x - xmin) / max(xmax - xmin, 1e-30)
    eta = (y - ymin) / max(ymax - ymin, 1e-30)
    r = r_in + eta * (r_out - r_in)
    th = np.pi * xi
    x[:] = r * np.cos(th)
    y[:] = r * np.sin(th)


def make_hertz_mesh(ps, nx, ny, radius, r_in, width, height, gap):
    block = ps.create_tri3_square(
        nx, max(2, ny // 2), -0.5 * width, radius + gap, 0.5 * width, radius + gap + height
    )
    obstacle = ps.create_tri3_square(nx, ny, 0.0, 0.0, 1.0, 1.0)
    map_square_to_hemi_annulus(ps, obstacle, r_in, radius)
    n_block = int(block.n_nodes())
    mesh = ps.join_meshes(block, obstacle, "block", "obstacle")
    X, Y = coords(ps, mesh)
    r = np.hypot(X[n_block:], Y[n_block:])
    if r.size == 0 or r.max() < 0.9 * radius or r.min() < 0.05 * radius:
        raise RuntimeError(
            f"hemi-annulus mapping failed: obstacle r in [{r.min():.3g}, {r.max():.3g}], expected ~[{r_in:.3g}, {radius:.3g}]"
        )
    return mesh, n_block


def coords(ps, mesh):
    return np.array(ps.points(mesh, 0), copy=True).astype(np.float64), np.array(
        ps.points(mesh, 1), copy=True
    ).astype(np.float64)


def deformed(X, Y, u):
    return X + u[0::2], Y + u[1::2]


def edge_geometry(X, Y, n0, n1):
    tx = X[n1] - X[n0]
    ty = Y[n1] - Y[n0]
    length = np.hypot(tx, ty)
    if length <= 0:
        return 0.0, 0.0, 0.0, 0.0
    nx, ny = ty / length, -tx / length
    return length, nx, ny, 1.0


def closest_on_edges(px, py, X, Y, edges):
    best_d2 = np.inf
    best = (edges[0, 0], edges[0, 1], 0.0, X[edges[0, 0]], Y[edges[0, 0]])
    for n0, n1 in edges:
        ax, ay = X[n0], Y[n0]
        bx, by = X[n1], Y[n1]
        abx, aby = bx - ax, by - ay
        ab2 = abx * abx + aby * aby
        t = 0.0 if ab2 <= 0 else np.clip(((px - ax) * abx + (py - ay) * aby) / ab2, 0.0, 1.0)
        qx, qy = ax + t * abx, ay + t * aby
        d2 = (px - qx) ** 2 + (py - qy) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best = (int(n0), int(n1), float(t), qx, qy)
    return best


def project_to_circle(px, py, R):
    r = float(np.hypot(px, py))
    if r < 1e-30:
        return 0.0, float(R)
    s = float(R) / r
    return px * s, py * s


def closest_contact_point(px, py, X, Y, edges, circle_R):
    """Closest point on the other surface. Circle radius avoids TRI3 chord glue."""
    if circle_R is None or circle_R <= 0:
        return closest_on_edges(px, py, X, Y, edges)
    qx, qy = project_to_circle(px, py, circle_R)
    m0, m1, t, _, _ = closest_on_edges(qx, qy, X, Y, edges)
    return m0, m1, t, qx, qy


def tri3_parent_nodes(ps, mesh, block_id, e):
    return np.array([int(ps.elements(mesh, v, block_id)[e]) for v in range(3)], dtype=np.int32)


def tri3_sigma_n(px, py, u_elem, mu, lam, nx, ny):
    x0, x1, x2 = px
    y0, y1, y2 = py
    det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    if abs(det) < 1e-30:
        return 0.0, np.zeros(6)
    inv = 1.0 / det
    dNdx = np.array([(y1 - y2), (y2 - y0), (y0 - y1)], dtype=np.float64) * inv
    dNdy = np.array([(x2 - x1), (x0 - x2), (x1 - x0)], dtype=np.float64) * inv
    ux = u_elem[0::2]
    uy = u_elem[1::2]
    exx = float(dNdx @ ux)
    eyy = float(dNdy @ uy)
    exy = 0.5 * float(dNdx @ uy + dNdy @ ux)
    tr = exx + eyy
    sxx = lam * tr + 2 * mu * exx
    syy = lam * tr + 2 * mu * eyy
    sxy = 2 * mu * exy
    sn = nx * (sxx * nx + sxy * ny) + ny * (sxy * nx + syy * ny)
    dsn = np.zeros(6)
    for a in range(3):
        dex_dux, dey_dux, dxy_dux = dNdx[a], 0.0, 0.5 * dNdy[a]
        dex_duy, dey_duy, dxy_duy = 0.0, dNdy[a], 0.5 * dNdx[a]
        dsxx_x = lam * (dex_dux + dey_dux) + 2 * mu * dex_dux
        dsyy_x = lam * (dex_dux + dey_dux) + 2 * mu * dey_dux
        dsxy_x = 2 * mu * dxy_dux
        dsxx_y = lam * (dex_duy + dey_duy) + 2 * mu * dex_duy
        dsyy_y = lam * (dex_duy + dey_duy) + 2 * mu * dey_duy
        dsxy_y = 2 * mu * dxy_duy
        dsn[2 * a] = nx * (dsxx_x * nx + dsxy_x * ny) + ny * (dsxy_x * nx + dsyy_x * ny)
        dsn[2 * a + 1] = nx * (dsxx_y * nx + dsxy_y * ny) + ny * (dsxy_y * nx + dsyy_y * ny)
    return sn, dsn


def contact_trace(
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
):
    """Cauchy σ_n and applied contact traction on every contact quadrature point."""
    xs, gs, sns, pns, active = [], [], [], [], []
    for edge, e_parent in zip(edges, parent_elems):
        n0, n1 = int(edge[0]), int(edge[1])
        length, nx0, ny0 = edge_geometry(X, Y, n0, n1)[:3]
        if length <= 1e-16:
            continue
        h = length
        gamma = gamma0 * mu / h
        parent_nodes = tri3_parent_nodes(ps, mesh, parent_block, int(e_parent))
        px = np.array([X[i] for i in parent_nodes])
        py = np.array([Y[i] for i in parent_nodes])
        u_elem = np.empty(6)
        for a, node in enumerate(parent_nodes):
            u_elem[2 * a] = u[2 * node]
            u_elem[2 * a + 1] = u[2 * node + 1]
        for xi, _ in QUAD_1D:
            nx, ny = nx0, ny0
            Na, Nb = 1.0 - xi, xi
            pxp = Na * X[n0] + Nb * X[n1]
            pyp = Na * Y[n0] + Nb * Y[n1]
            if snap_self_circle and circle_R:
                pxp, pyp = project_to_circle(pxp, pyp, circle_R)
                rn = float(np.hypot(pxp, pyp))
                if rn > 1e-30:
                    nx, ny = pxp / rn, pyp / rn
            m0, m1, t, qx, qy = closest_contact_point(
                pxp, pyp, X, Y, other_edges, None if snap_self_circle else circle_R
            )
            Nm0, Nm1 = 1.0 - t, t
            upx = Na * u[2 * n0] + Nb * u[2 * n1]
            upy = Na * u[2 * n0 + 1] + Nb * u[2 * n1 + 1]
            uqx = Nm0 * u[2 * m0] + Nm1 * u[2 * m1]
            uqy = Nm0 * u[2 * m0 + 1] + Nm1 * u[2 * m1 + 1]
            g = (qx - pxp + uqx - upx) * nx + (qy - pyp + uqy - upy) * ny
            sn, _ = tri3_sigma_n(px, py, u_elem, mu, lam, nx, ny)
            Pn = (theta * sn if include_sigma else 0.0) + gamma * g
            on = g < 0.0
            xs.append(pxp)
            gs.append(g)
            sns.append(sn)
            pns.append(Pn if on else 0.0)
            active.append(on)
    return (
        np.asarray(xs),
        np.asarray(gs),
        np.asarray(sns),
        np.asarray(pns),
        np.asarray(active, dtype=bool),
    )


def surface_contrib(
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
):
    """Unbiased Nitsche residual/tangent on one contact surface (paper P_{γ,θ}).

    Unilateral Signorini: assemble only where the geometric gap is negative.
    No freeze / hysteresis (those are bilateral glue). Pair to a circle of
    radius circle_R so TRI3 chords cannot glue the block onto interior vertices.
    """
    if new_active is None:
        new_active = set()
    n_active = 0
    for ie, (edge, e_parent) in enumerate(zip(edges, parent_elems)):
        n0, n1 = int(edge[0]), int(edge[1])
        length, nx, ny = edge_geometry(X, Y, n0, n1)[:3]
        if length <= 1e-16:
            continue
        tx, ty = -ny, nx
        h = length
        gamma = gamma0 * mu / h
        parent_nodes = tri3_parent_nodes(ps, mesh, parent_block, int(e_parent))
        px = np.array([X[i] for i in parent_nodes])
        py = np.array([Y[i] for i in parent_nodes])
        u_elem = np.empty(6)
        for a, node in enumerate(parent_nodes):
            u_elem[2 * a] = u[2 * node]
            u_elem[2 * a + 1] = u[2 * node + 1]

        for iq, (xi, w_hat) in enumerate(QUAD_1D):
            w = w_hat * length
            Na = 1.0 - xi
            Nb = xi
            pxp = Na * X[n0] + Nb * X[n1]
            pyp = Na * Y[n0] + Nb * Y[n1]
            if snap_self_circle and circle_R:
                pxp, pyp = project_to_circle(pxp, pyp, circle_R)
                rn = float(np.hypot(pxp, pyp))
                if rn > 1e-30:
                    nx, ny = pxp / rn, pyp / rn
                    tx, ty = -ny, nx
            m0, m1, t, qx, qy = closest_contact_point(
                pxp, pyp, X, Y, other_edges, None if snap_self_circle else circle_R
            )
            Nm0, Nm1 = 1.0 - t, t
            upx = Na * u[2 * n0] + Nb * u[2 * n1]
            upy = Na * u[2 * n0 + 1] + Nb * u[2 * n1 + 1]
            uqx = Nm0 * u[2 * m0] + Nm1 * u[2 * m1]
            uqy = Nm0 * u[2 * m0 + 1] + Nm1 * u[2 * m1 + 1]
            # Frozen reference pairing/normal. g > 0 when separated, so
            # [u]_n = -g and P = θ σ_n - γ [u]_n = θ σ_n + γ g (paper).
            g = (qx - pxp + uqx - upx) * nx + (qy - pyp + uqy - upy) * ny
            # Unilateral: force only while overlapping. No freeze / hysteresis
            # (glue) and no P_n filter that would leave penetration unloaded.
            if g >= 0.0:
                continue
            sn, dsn = tri3_sigma_n(px, py, u_elem, mu, lam, nx, ny)
            Pn = (theta * sn if include_sigma else 0.0) + gamma * g
            key = (surface_id, ie, iq)
            new_active.add(key)
            n_active += 1
            scale = (1.0 / theta) * (1.0 / gamma) * w
            if energy_acc is not None:
                energy_acc[0] += 0.5 * scale * Pn * Pn
            if qp_out is not None:
                qp_out.append((pxp, g, Pn))
            if not assemble:
                continue
            local_nodes = [n0, n1, m0, m1]
            Nself = [Na, Nb, 0.0, 0.0]
            Nother = [0.0, 0.0, Nm0, Nm1]
            pv = {}

            def add_pv(node, comp, val):
                if val == 0.0:
                    return
                dof = 2 * int(node) + comp
                pv[dof] = pv.get(dof, 0.0) + val

            add_pv(n0, 0, gamma * (-Na * nx))
            add_pv(n0, 1, gamma * (-Na * ny))
            add_pv(n1, 0, gamma * (-Nb * nx))
            add_pv(n1, 1, gamma * (-Nb * ny))
            add_pv(m0, 0, gamma * (Nm0 * nx))
            add_pv(m0, 1, gamma * (Nm0 * ny))
            add_pv(m1, 0, gamma * (Nm1 * nx))
            add_pv(m1, 1, gamma * (Nm1 * ny))
            if include_sigma:
                for a, node in enumerate(parent_nodes):
                    add_pv(int(node), 0, theta * dsn[2 * a])
                    add_pv(int(node), 1, theta * dsn[2 * a + 1])

            for dof, pvi in pv.items():
                residual[dof] += scale * Pn * pvi
            for di, pi in pv.items():
                for dj, pj in pv.items():
                    val = scale * pi * pj
                    if val != 0.0:
                        coo.append((di, dj, val))

            if not friction:
                continue

            gt = (qx - pxp + uqx - upx) * tx + (qy - pyp + uqy - upy) * ty
            Pt = -gamma * gt
            for k in range(4):
                for c in range(2):
                    tv = (Nself[k] - Nother[k]) * (tx if c == 0 else ty)
                    residual[2 * local_nodes[k] + c] += w * Pt * tv
            for i in range(4):
                for ci in range(2):
                    tvi = (Nself[i] - Nother[i]) * (tx if ci == 0 else ty)
                    row = 2 * local_nodes[i] + ci
                    for j in range(4):
                        for cj in range(2):
                            tvj = (Nself[j] - Nother[j]) * (tx if cj == 0 else ty)
                            col = 2 * local_nodes[j] + cj
                            val = w * gamma * tvi * tvj
                            if val != 0.0:
                                coo.append((row, col, val))
    return n_active


def apply_dirichlet_system(K, g, constrained, u, u_bc):
    K = K.tocsr()
    diag = K.diagonal().copy()
    diag[diag == 0] = 1.0
    K = K.tolil()
    for i in constrained:
        K.rows[i] = [i]
        K.data[i] = [diag[i] if diag[i] != 0 else 1.0]
        g[i] = diag[i] * (u[i] - u_bc[i])
    return K.tocsr(), g


def hertz_pressure(x, F, R, E_star):
    if F <= 0 or E_star <= 0:
        return np.zeros_like(x), 0.0, 0.0
    a = np.sqrt(max(4.0 * R * F / (np.pi * E_star), 0.0))
    p0 = 0.0 if a <= 0 else 2.0 * F / (np.pi * a)
    p = np.zeros_like(x)
    inside = np.abs(x) < a
    p[inside] = p0 * np.sqrt(np.maximum(0.0, 1.0 - (x[inside] / a) ** 2))
    return p, a, p0


def solve_nitsche(ps, args):
    radius = args.radius
    mesh, n_block = make_hertz_mesh(
        ps, args.nx, args.ny, radius, args.r_inner * radius, args.width, args.height, args.gap
    )
    X, Y = coords(ps, mesh)
    dim = 2
    space = ps.FunctionSpace(mesh, dim)
    op = ps.create_op(space, "LinearElasticity")
    op.initialize()

    E_b, nu_b = args.E_block, args.nu
    E_o, nu_o = args.E_obstacle, args.nu
    if args.rigid_block:
        E_b *= args.rigid_stiffness
    if args.rigid_obstacle:
        E_o *= args.rigid_stiffness
    lam_b, mu_b = lame(E_b, nu_b)
    lam_o, mu_o = lame(E_o, nu_o)
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

    ss_top = sidesets_from_selector(
        ps, mesh, lambda x, y, z: y > y_top - tol, ["block"]
    )
    ss_left = sidesets_from_selector(
        ps, mesh, lambda x, y, z: x < -0.5 * args.width + tol, ["block"]
    )
    ss_right = sidesets_from_selector(
        ps, mesh, lambda x, y, z: x > 0.5 * args.width - tol, ["block"]
    )
    ss_contact_block = sidesets_from_selector(
        ps, mesh, lambda x, y, z: y < y_bot + 4 * tol, ["block"]
    )
    ss_contact_obs = sidesets_from_selector(
        ps,
        mesh,
        lambda x, y, z: bool(np.hypot(x, y) > 0.85 * radius) and bool(y > 0.05 * radius),
        ["obstacle"],
    )
    ss_diam = sidesets_from_selector(
        ps, mesh, lambda x, y, z: y < y_diam + 4 * tol, ["obstacle"]
    )

    conditions = [
        dirichlet_condition(ps, ss_top, 1, -args.indent),
        dirichlet_condition(ps, ss_top, 0, 0.0),
        dirichlet_condition(ps, ss_left, 0, 0.0),
        dirichlet_condition(ps, ss_right, 0, 0.0),
        dirichlet_condition(ps, ss_diam, 0, 0.0),
        dirichlet_condition(ps, ss_diam, 1, 0.0),
    ]
    if args.rigid_obstacle:
        ss_all_obs = sidesets_from_selector(ps, mesh, lambda x, y, z: True, ["obstacle"])
        conditions.append(dirichlet_condition(ps, ss_all_obs, 0, 0.0))
        conditions.append(dirichlet_condition(ps, ss_all_obs, 1, 0.0))
    if args.rigid_block:
        ss_all_block = sidesets_from_selector(ps, mesh, lambda x, y, z: True, ["block"])
        conditions.append(dirichlet_condition(ps, ss_all_block, 0, 0.0))
        conditions.append(dirichlet_condition(ps, ss_all_block, 1, -args.indent))

    bcs = ps.create_dirichlet_conditions(space, conditions, ps.ExecutionSpace.EXECUTION_SPACE_HOST)
    fun.add_constraint(bcs)

    u_buf = ps.create_real_buffer(ndofs)
    g_buf = ps.create_real_buffer(ndofs)
    u = numpy_view(ps, u_buf)
    g = numpy_view(ps, g_buf)
    u[:] = 0.0
    ps.apply_constraints(fun, u_buf)
    u_bc = u.copy()

    mask = np.zeros(ndofs, dtype=bool)
    ps.apply_zero_constraints(fun, g_buf)
    g[:] = 1.0
    ps.apply_zero_constraints(fun, g_buf)
    mask[g == 0.0] = True
    constrained = np.where(mask)[0]

    contact_block = flatten_sidesets([ss_contact_block])
    contact_obs = flatten_sidesets([ss_contact_obs])
    if not contact_block or not contact_obs:
        raise RuntimeError(
            f"empty contact sidesets: block={sum(s.size() for s in contact_block)} "
            f"obstacle={sum(s.size() for s in contact_obs)}"
        )

    edges_b, elems_b, bid_b = sideset_edges(ps, mesh, contact_block[0])
    edges_o, elems_o, bid_o = sideset_edges(ps, mesh, contact_obs[0])
    for extra in contact_block[1:]:
        e, p, _ = sideset_edges(ps, mesh, extra)
        edges_b = np.vstack([edges_b, e])
        elems_b = np.concatenate([elems_b, p])
    for extra in contact_obs[1:]:
        e, p, _ = sideset_edges(ps, mesh, extra)
        edges_o = np.vstack([edges_o, e])
        elems_o = np.concatenate([elems_o, p])

    r_tol = max(4 * tol, 0.05 * radius)
    edges_b, elems_b = filter_edges(
        X, Y, edges_b, elems_b, lambda n0, n1: Y[n0] < y_bot + 4 * tol and Y[n1] < y_bot + 4 * tol
    )
    edges_o, elems_o = filter_edges(
        X,
        Y,
        edges_o,
        elems_o,
        lambda n0, n1: abs(np.hypot(X[n0], Y[n0]) - radius) <= r_tol
        and abs(np.hypot(X[n1], Y[n1]) - radius) <= r_tol,
    )
    if edges_b.size == 0 or edges_o.size == 0:
        raise RuntimeError(
            f"empty contact edges after filter: block={edges_b.shape} obstacle={edges_o.shape}"
        )
    edges_b = orient_edges(X, Y, edges_b, "down")
    edges_o = orient_edges(X, Y, edges_o, "outward")

    # Two-body Hertz: Nitsche/penalty on the block (slave). The obstacle is
    # still elastic through u_q in the gap. The paper σ_n term (P = θσ_n+γg)
    # makes the Newton step want g>0 (adhesion) on this coarse pairing; the
    # default two-body form is geometric penalty P=γg. Rigid obstacle keeps
    # full Nitsche. --nitsche-stress / --unbiased restore σ_n.
    theta_b, theta_o = 1.0, 0.0
    if args.unbiased and not args.rigid_obstacle and not args.rigid_block:
        theta_b, theta_o = 0.5, 0.5
    if args.rigid_block and not args.rigid_obstacle:
        theta_b, theta_o = 0.0, 1.0
    include_sigma = bool(
        args.rigid_obstacle or args.rigid_block or args.unbiased or args.nitsche_stress
    )

    r_hist = []
    g_c = np.zeros(ndofs)
    qp_samples = []

    def _contact_kwargs(u_vec, residual, coo, new_active, energy_acc=None, assemble=True):
        n_qp = 0
        if theta_b > 0:
            n_qp += surface_contrib(
                X, Y, u_vec, edges_b, elems_b, bid_b, edges_o, ps, mesh, mu_b, lam_b,
                args.gamma0, theta_b, args.friction, args.mu_f, residual, coo, radius,
                0, new_active, False, energy_acc, qp_samples if assemble else None,
                assemble, include_sigma,
            )
        if theta_o > 0:
            n_qp += surface_contrib(
                X, Y, u_vec, edges_o, elems_o, bid_o, edges_b, ps, mesh, mu_o, lam_o,
                args.gamma0, theta_o, args.friction, args.mu_f, residual, coo, radius,
                1, new_active, True, energy_acc, qp_samples if assemble else None,
                assemble, include_sigma,
            )
        return n_qp

    def residual_tangent(u_vec):
        qp_samples.clear()
        g_contact = np.zeros(ndofs)
        coo = []
        new_active = set()
        n_qp = _contact_kwargs(u_vec, g_contact, coo, new_active, assemble=True)
        resid = Ke @ u_vec + g_contact
        if coo:
            ii, jj, vv = zip(*coo)
            Kn = sparse.coo_matrix((vv, (ii, jj)), shape=(ndofs, ndofs)).tocsr()
        else:
            Kn = sparse.csr_matrix((ndofs, ndofs))
        Ksys = Ke + Kn
        Ksys, resid = apply_dirichlet_system(Ksys, resid, constrained, u_vec, u_bc)
        return resid, Ksys, g_contact, Kn, n_qp, new_active

    def total_energy(u_vec):
        acc = [0.0]
        uc = u_vec.copy()
        uc[constrained] = u_bc[constrained]
        _contact_kwargs(uc, None, None, set(), energy_acc=acc, assemble=False)
        return 0.5 * float(uc @ (Ke @ uc)) + acc[0]

    u_bc_full = u_bc.copy()
    n_load = args.load_steps if args.load_steps > 0 else 1
    u[:] = 0.0
    for load in range(1, n_load + 1):
        u_bc[:] = u_bc_full * (load / float(n_load))
        u[constrained] = u_bc[constrained]
        if n_load > 1:
            print(f"load[{load}/{n_load}] indent={args.indent * load / n_load:.4g}")
        for it in range(args.max_newton):
            residual, K, g_c, Kn, n_qp, new_active = residual_tangent(u)
            rnorm = float(np.linalg.norm(residual))
            r_hist.append(rnorm)
            print(f"newton[{it:02d}] ||r||={rnorm:.3e}  contact_nnz={Kn.nnz}  nitsche_qp={n_qp}")
            if rnorm < args.rtol:
                break
            du = spsolve(K, -residual)
            if not np.all(np.isfinite(du)):
                raise RuntimeError("Newton increment is not finite")
            def trial(omega):
                u_try = u + omega * du
                u_try[constrained] = u_bc[constrained]
                _, _, _, _, n_try, _ = residual_tangent(u_try)
                return u_try, total_energy(u_try), n_try

            if n_qp == 0:
                phi0 = total_energy(u)
                best_phi, best_u, best_w, best_n = np.inf, None, 0.0, 0
                omega = 1.0
                while omega >= 1.0 / 64.0:
                    u_try, phi, n_try = trial(omega)
                    if phi < best_phi:
                        best_phi, best_u, best_w, best_n = phi, u_try.copy(), omega, n_try
                    omega *= 0.5
                if best_u is None:
                    u[:] = u + du
                else:
                    u[:] = best_u
                    print(
                        f"          enter omega={best_w:g}  n_qp={best_n}  "
                        f"dphi={best_phi - phi0:.3e}"
                    )
            else:
                phi0 = total_energy(u)
                slope = float(residual @ du)
                omega = 1.0
                best_phi, best_u, best_w, best_n = phi0, None, 0.0, n_qp
                accepted = False
                while omega >= 1.0 / 32.0:
                    u_try, phi, n_try = trial(omega)
                    if n_try == 0:
                        omega *= 0.5
                        continue
                    if phi < best_phi:
                        best_phi, best_u, best_w, best_n = phi, u_try.copy(), omega, n_try
                    if phi <= phi0 + 1e-4 * omega * min(slope, 0.0):
                        u[:] = u_try
                        accepted = True
                        if omega < 1.0:
                            print(
                                f"          linesearch omega={omega:g}  "
                                f"dphi={phi - phi0:.3e}  n_qp={n_try}"
                            )
                        break
                    omega *= 0.5
                if not accepted:
                    if best_u is not None and best_phi < phi0:
                        u[:] = best_u
                        print(
                            f"          linesearch omega={best_w:g}  "
                            f"dphi={best_phi - phi0:.3e}  n_qp={best_n}"
                        )
                    else:
                        print("          linesearch: no energy descent in g<0")
                        break
            u[constrained] = u_bc[constrained]

    residual, K, g_c, Kn, n_qp, _ = residual_tangent(u)
    r_final = float(np.linalg.norm(residual))
    if not r_hist or abs(r_hist[-1] - r_final) > 1e-30:
        r_hist.append(r_final)
        print(f"newton[final] ||r||={r_final:.3e}  contact_nnz={Kn.nnz}  nitsche_qp={n_qp}")

    xd, yd = deformed(X, Y, u)
    nodes_b = unique_nodes_from_edges(edges_b)
    gap = np.empty(nodes_b.size)
    for i, node in enumerate(nodes_b):
        qx, qy = project_to_circle(X[node], Y[node], radius)
        m0, m1, t, _, _ = closest_on_edges(qx, qy, X, Y, edges_o)
        uqy = (1.0 - t) * u[2 * m0 + 1] + t * u[2 * m1 + 1]
        gap[i] = (qy + uqy - yd[node]) * (-1.0)
    penetration = float(np.linalg.norm(np.minimum(gap, 0.0)))
    F = abs(float(np.sum(g_c[1 : 2 * n_block : 2])))
    E_star = effective_modulus(E_b, nu_b, E_o if not args.rigid_obstacle else 1e300, nu_o)
    xc = 0.5 * (X[edges_b[:, 0]] + X[edges_b[:, 1]])
    p_hertz, a, p0 = hertz_pressure(xc, F, radius, E_star)
    tr_x, tr_g, tr_sn, tr_pn, tr_on = contact_trace(
        X, Y, u, edges_b, elems_b, bid_b, edges_o, ps, mesh, mu_b, lam_b,
        args.gamma0, theta_b if theta_b > 0 else 1.0, radius, include_sigma, False,
    )
    p_cauchy = -tr_sn
    p_applied = np.where(tr_on, -tr_pn, 0.0)
    p_cauchy_max = float(np.max(p_cauchy)) if p_cauchy.size else 0.0
    p_applied_max = float(np.max(p_applied)) if p_applied.size else 0.0

    print(
        f"nodes={mesh.n_nodes()} dofs={ndofs} F={F:.4e} a_hertz={a:.4e} p0={p0:.4e} "
        f"|g_-|={penetration:.3e}  theta=({theta_b:g},{theta_o:g})  "
        f"sigma={'on' if include_sigma else 'off'}  "
        f"max(-σ_n)={p_cauchy_max:.4e}  max(-P_n)={p_applied_max:.4e}"
    )
    return {
        "mesh": mesh,
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
        "ndofs": ndofs,
        "penetration": penetration,
        "r_hist": r_hist,
        "n_block": n_block,
        "tris_b": block_triangles(ps, mesh, bid_b),
        "tris_o": block_triangles(ps, mesh, bid_o),
        "rigid_obstacle": bool(args.rigid_obstacle),
        "qp_x": tr_x,
        "qp_g": tr_g,
        "qp_sn": tr_sn,
        "qp_pn": tr_pn,
        "qp_active": tr_on,
        "p_cauchy": p_cauchy,
        "p_applied": p_applied,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nx", type=int, default=8)
    p.add_argument("--ny", type=int, default=4)
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
    p.add_argument("--max-newton", type=int, default=25)
    p.add_argument(
        "--load-steps",
        type=int,
        default=0,
        help="Indent increments. 0/1: apply the full indent in one shot.",
    )
    p.add_argument("--rtol", type=float, default=1e-8)
    p.add_argument("--friction", action="store_true")
    p.add_argument("--mu-f", type=float, default=0.3)
    p.add_argument("--rigid-block", action="store_true")
    p.add_argument("--rigid-obstacle", action="store_true")
    p.add_argument("--rigid-stiffness", type=float, default=1e4)
    p.add_argument(
        "--biased",
        action="store_true",
        help="Master/slave (block is slave). Default for two-body Hertz.",
    )
    p.add_argument(
        "--unbiased",
        action="store_true",
        help="Nitsche on both surfaces (paper). Unstable on this coarse pairing.",
    )
    p.add_argument(
        "--nitsche-stress",
        action="store_true",
        help="Include θ σ_n in P for two-body (paper). Default two-body uses P=γg.",
    )
    p.add_argument("--plot", action="store_true")
    p.add_argument("--conv", action="store_true", help="Mesh refinement study")
    p.add_argument("--check", action="store_true", help="Tiny run, exit 0 if Newton decreases residual")
    return p.parse_args(argv)


def plot_solution(result):
    import matplotlib.pyplot as plt

    X, Y, u = result["X"], result["Y"], result["u"]
    xd, yd = X + u[0::2], Y + u[1::2]
    mag = np.hypot(u[0::2], u[1::2])
    vmin, vmax = float(np.min(mag)), float(np.max(mag))
    if vmax <= vmin:
        vmax = vmin + 1e-16

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    tpc = None
    for tris in (result["tris_b"], result["tris_o"]):
        tpc = ax[0].tripcolor(
            xd, yd, tris, mag, shading="gouraud", cmap="viridis", vmin=vmin, vmax=vmax
        )
        ax[0].triplot(xd, yd, tris, color="k", lw=0.25, alpha=0.55)
    fig.colorbar(tpc, ax=ax[0], fraction=0.046, pad=0.04, label="|u|")
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].set_title(
        "deformed TRI3 (rigid obstacle)"
        if result.get("rigid_obstacle")
        else "deformed TRI3 (two-body elastic)"
    )

    nodes = result["nodes_b"]
    order = np.argsort(X[nodes])
    xg = X[nodes][order]
    gap = np.asarray(result["gap"])[order]
    xc_order = np.argsort(result["xc"])
    xc = np.asarray(result["xc"])[xc_order]
    ph = np.asarray(result["p_hertz"])[xc_order]

    axg = ax[1]
    axp = axg.twinx()
    ln_gap = axg.plot(xg, gap, "o-", color="C0", ms=4, label="gap (>0 open, <0 overlap)")
    axg.axhline(0.0, color="0.45", lw=0.8)
    a = float(result["a"])
    if a > 0:
        axg.axvline(-a, color="0.55", ls="--", lw=0.8)
        axg.axvline(a, color="0.55", ls="--", lw=0.8)
    ln_hertz = axp.plot(xc, ph, color="C1", label="Hertz p(x) from F")
    tr_x = np.asarray(result.get("qp_x", []))
    p_cauchy = np.asarray(result.get("p_cauchy", []))
    p_applied = np.asarray(result.get("p_applied", []))
    ln_cauchy = []
    ln_applied = []
    if tr_x.size:
        order_qp = np.argsort(tr_x)
        xq = tr_x[order_qp]
        ln_cauchy = axp.plot(
            xq, p_cauchy[order_qp], "o-", color="C3", ms=4, lw=1.2, label="solution -σ_n"
        )
        ln_applied = axp.plot(
            xq,
            p_applied[order_qp],
            "s",
            color="C4",
            ms=5,
            label="applied -P_n (g<0, else 0)",
        )
    axg.set_xlabel("x (reference, block bottom)")
    axg.set_ylabel("gap", color="C0")
    axp.set_ylabel("pressure", color="C1")
    axg.set_title(f"contact cut  a={a:.3g}  p0={result['p0']:.3g}")
    lines = ln_gap + ln_hertz + ln_cauchy + ln_applied
    axg.legend(lines, [ln.get_label() for ln in lines], loc="best")
    plt.tight_layout()
    plt.show()


def refinement_study(ps, args):
    prev_h = None
    prev_pen = None
    print("# h  |g_-|  rate")
    for k in range(3):
        args.nx = 4 * (2**k)
        args.ny = 2 * (2**k)
        out = solve_nitsche(ps, args)
        h = 1.0 / args.nx
        pen = max(out["penetration"], 1e-30)
        rate = "" if prev_h is None else f"{np.log(prev_pen / pen) / np.log(prev_h / h):.2f}"
        print(f"{h:.4e}  {pen:.3e}  {rate}")
        prev_h, prev_pen = h, pen


def main(argv=None):
    args = parse_args(argv)
    if args.check:
        args.nx = 8
        args.ny = 4
        args.max_newton = 15
        args.plot = False
        args.indent = 0.02
        args.rigid_obstacle = True
    ps = _load_pysfem()
    ps.init()
    try:
        if args.conv:
            refinement_study(ps, args)
        else:
            result = solve_nitsche(ps, args)
            if args.check:
                r_hist = result.get("r_hist", [])
                if not r_hist or not np.all(np.isfinite(r_hist)):
                    raise SystemExit("check failed: non-finite Newton residual")
                if not np.isfinite(result["penetration"]):
                    raise SystemExit("check failed: non-finite penetration")
                if r_hist[-1] > 1e-6:
                    raise SystemExit(
                        f"check failed: Newton residual {r_hist[0]:.3e} -> {r_hist[-1]:.3e}"
                    )
                if result["penetration"] > 0.5 * args.indent:
                    raise SystemExit(
                        f"check failed: remaining penetration {result['penetration']:.3e} "
                        f"with indent {args.indent:g}"
                    )
                nodes = result["nodes_b"]
                i_min = int(np.argmin(result["gap"]))
                x_min = float(result["X"][nodes[i_min]])
                if abs(x_min) > 0.25:
                    raise SystemExit(
                        f"check failed: deepest gap at x={x_min:.3g}, expected near x=0"
                    )
            if args.plot:
                plot_solution(result)
    finally:
        ps.finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

