#!/usr/bin/env python3
"""Two-body unbiased Nitsche contact (Mlika, Renard, Chouly).

Frictionless first; Coulomb friction with --friction. Hertz geometry: elastic
(or rigid) rectangular block pressed onto an elastic (or rigid) hemi-annulus.

Requires pysfem (build_py with -DSFEM_ENABLE_PYTHON=ON). See
python/spike/run_nitsche_contact.sh and plans/NITSCHE_CONTACT.md.
"""

from __future__ import annotations

import argparse
import copy
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
# 3-point Gauss–Legendre on [0, 1] (weights sum to 1). Used to average the
# gap to one P0 traction per EDGE2 (staircase). Collocating γg at each QP
# checkerboards a P1 trace. --lagrange still uses one midpoint (edge_rows).
_GL3 = np.sqrt(3.0 / 5.0)
QUAD_1D = (
    (0.5 * (1.0 - _GL3), 5.0 / 18.0),
    (0.5, 8.0 / 18.0),
    (0.5 * (1.0 + _GL3), 5.0 / 18.0),
)


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


def unique_oriented_edges(edges, elems):
    seen = {}
    keep = []
    for i, (n0, n1) in enumerate(edges):
        key = (min(int(n0), int(n1)), max(int(n0), int(n1)))
        if key in seen:
            continue
        seen[key] = i
        keep.append(i)
    if not keep:
        return edges[:0], elems[:0]
    idx = np.asarray(keep, dtype=np.int32)
    return edges[idx], elems[idx]


def symmetrize_tri3_square_diagonals(ps, mesh, nx, ny, block_id=0):
    """Mirror TRI3 diagonals about the parametric midline so P0 σ_n is even in x."""
    c0 = np.array(ps.elements(mesh, 0, block_id), copy=False)
    c1 = np.array(ps.elements(mesh, 1, block_id), copy=False)
    c2 = np.array(ps.elements(mesh, 2, block_id), copy=False)
    mid = (int(nx) + 1) // 2
    for yi in range(int(ny)):
        for xi in range(mid, int(nx)):
            e = 2 * (yi * int(nx) + xi)
            i0, i1, i3 = int(c0[e]), int(c1[e]), int(c2[e])
            i2 = int(c1[e + 1])
            c0[e], c1[e], c2[e] = i0, i1, i2
            c0[e + 1], c1[e + 1], c2[e + 1] = i0, i2, i3


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
    # θ = π(1−ξ) keeps TRI3 orientation. θ=πξ has Jacobian −π r Δr and
    # flips the obstacle stiffness sign, so contact tension is cheaper
    # than compression and the block is sucked into the cylinder.
    th = np.pi * (1.0 - xi)
    x[:] = r * np.cos(th)
    y[:] = r * np.sin(th)


def tri3_signed_areas(tris, X, Y):
    x0, y0 = X[tris[:, 0]], Y[tris[:, 0]]
    x1, y1 = X[tris[:, 1]], Y[tris[:, 1]]
    x2, y2 = X[tris[:, 2]], Y[tris[:, 2]]
    return 0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))


def make_hertz_mesh(ps, nx, ny, radius, r_in, width, height, gap, ny_block=None):
    ny_b = int(ny_block) if ny_block is not None else max(2, ny // 2)
    block = ps.create_tri3_square(
        nx, ny_b, -0.5 * width, radius + gap, 0.5 * width, radius + gap + height
    )
    obstacle = ps.create_tri3_square(nx, ny, 0.0, 0.0, 1.0, 1.0)
    symmetrize_tri3_square_diagonals(ps, block, nx, ny_b)
    symmetrize_tri3_square_diagonals(ps, obstacle, nx, ny)
    map_square_to_hemi_annulus(ps, obstacle, r_in, radius)
    n_block = int(block.n_nodes())
    mesh = ps.join_meshes(block, obstacle, "block", "obstacle")
    X, Y = coords(ps, mesh)
    r = np.hypot(X[n_block:], Y[n_block:])
    if r.size == 0 or r.max() < 0.9 * radius or r.min() < 0.05 * radius:
        raise RuntimeError(
            f"hemi-annulus mapping failed: obstacle r in [{r.min():.3g}, {r.max():.3g}], expected ~[{r_in:.3g}, {radius:.3g}]"
        )
    for bid, name in enumerate(("block", "obstacle")):
        areas = tri3_signed_areas(block_triangles(ps, mesh, bid), X, Y)
        amin = float(np.min(areas)) if areas.size else 0.0
        if areas.size == 0 or amin <= 0.0:
            raise RuntimeError(f"{name} TRI3 inverted (min area {amin:.3g})")
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


def pair_on_x(x_query, X, Y, edges):
    """Opposing point on a nearly-horizontal trace, parametrized by x.

    Euclidean closest-point from the cylinder equator snaps to the block
    corners and reports a bogus negative gap, so unbiased Nitsche glues
    the sides of the obstacle to the block.
    """
    best = None
    best_span = np.inf
    for n0, n1 in edges:
        n0 = int(n0)
        n1 = int(n1)
        x0 = float(X[n0])
        x1 = float(X[n1])
        lo, hi = (x0, x1) if x0 <= x1 else (x1, x0)
        if x_query < lo - 1e-14 or x_query > hi + 1e-14:
            continue
        span = hi - lo
        if span > best_span:
            continue
        t = 0.0 if abs(x1 - x0) < 1e-30 else (x_query - x0) / (x1 - x0)
        t = float(np.clip(t, 0.0, 1.0))
        qx = x0 + t * (x1 - x0)
        qy = float(Y[n0]) + t * (float(Y[n1]) - float(Y[n0]))
        best = (n0, n1, t, qx, qy)
        best_span = span
    return best


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


def master_on_circle(px, py, X, Y, edges, R):
    """Pair to the analytic circle; interpolate u on the arc by angle.

    Euclidean closest-point on TRI3 chords snaps to vertices (chords lie
    inside the circle), so every near-crown slave QP dumps its force onto
    the single crown node and the mesh pinches into a V.
    """
    qx, qy = project_to_circle(px, py, R)
    th = float(np.arctan2(qy, qx))
    best = None
    best_span = np.inf
    for n0, n1 in edges:
        n0 = int(n0)
        n1 = int(n1)
        t0 = float(np.arctan2(Y[n0], X[n0]))
        t1 = float(np.arctan2(Y[n1], X[n1]))
        lo, hi = (t0, t1) if t0 <= t1 else (t1, t0)
        span = hi - lo
        if span > np.pi or span <= 0.0:
            continue
        if th < lo - 1e-14 or th > hi + 1e-14:
            continue
        if span > best_span:
            continue
        if abs(t1 - t0) < 1e-30:
            t = 0.0
        else:
            t = (th - t0) / (t1 - t0)
        best = (n0, n1, float(np.clip(t, 0.0, 1.0)), qx, qy)
        best_span = span
    if best is None:
        m0, m1, t, _, _ = closest_on_edges(qx, qy, X, Y, edges)
        return m0, m1, t, qx, qy
    return best


def closest_contact_point(px, py, X, Y, edges, circle_R):
    """Closest point on the other surface. Circle radius avoids TRI3 chord glue."""
    if circle_R is None or circle_R <= 0:
        return closest_on_edges(px, py, X, Y, edges)
    return master_on_circle(px, py, X, Y, edges, circle_R)


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


def collect_edge_qps(X, Y, u, n0, n1, length, nx0, ny0, other_edges, circle_R, snap_self_circle):
    """Quadrature samples on one EDGE2. Pairing may skip a QP with no opposite face."""
    samples = []
    for xi, w_hat in QUAD_1D:
        w = w_hat * length
        Na, Nb = 1.0 - xi, xi
        xref = Na * X[n0] + Nb * X[n1]
        yref = Na * Y[n0] + Nb * Y[n1]
        pxp = xref + Na * u[2 * n0] + Nb * u[2 * n1]
        pyp = yref + Na * u[2 * n0 + 1] + Nb * u[2 * n1 + 1]
        nx, ny = nx0, ny0
        tx, ty = -ny, nx
        if snap_self_circle:
            rn = float(np.hypot(xref, yref))
            if rn > 1e-30:
                nx, ny = xref / rn, yref / rn
                tx, ty = -ny, nx
            paired = pair_on_x(xref, X, Y, other_edges)
            if paired is None:
                continue
            m0, m1, t, qx, qy = paired
        else:
            m0, m1, t, qx, qy = closest_contact_point(
                pxp, pyp, X, Y, other_edges, circle_R
            )
            if circle_R:
                rn = float(np.hypot(qx, qy))
                if rn > 1e-30:
                    nx, ny = -qx / rn, -qy / rn
                    tx, ty = -ny, nx
        Nm0, Nm1 = 1.0 - t, t
        uqx = Nm0 * u[2 * m0] + Nm1 * u[2 * m1]
        uqy = Nm0 * u[2 * m0 + 1] + Nm1 * u[2 * m1 + 1]
        g = (qx + uqx - pxp) * nx + (qy + uqy - pyp) * ny
        samples.append(
            (w, xi, g, nx, ny, tx, ty, Na, Nb, int(m0), int(m1), Nm0, Nm1, xref)
        )
    return samples


def sample_edge_qps(X, Y, u, n0, n1, length, nx0, ny0, other_edges, circle_R, snap_self_circle):
    """Frozen pairing/normals/weights at the current configuration."""
    frozen = []
    for xi, w_hat in QUAD_1D:
        w = w_hat * length
        Na, Nb = 1.0 - xi, xi
        xref = Na * X[n0] + Nb * X[n1]
        yref = Na * Y[n0] + Nb * Y[n1]
        pxp = xref + Na * u[2 * n0] + Nb * u[2 * n1]
        pyp = yref + Na * u[2 * n0 + 1] + Nb * u[2 * n1 + 1]
        nx, ny = nx0, ny0
        tx, ty = -ny, nx
        if snap_self_circle:
            rn = float(np.hypot(xref, yref))
            if rn > 1e-30:
                nx, ny = xref / rn, yref / rn
                tx, ty = -ny, nx
            paired = pair_on_x(xref, X, Y, other_edges)
            if paired is None:
                continue
            m0, m1, t, qx, qy = paired
        else:
            m0, m1, t, qx, qy = closest_contact_point(
                pxp, pyp, X, Y, other_edges, circle_R
            )
            if circle_R:
                rn = float(np.hypot(qx, qy))
                if rn > 1e-30:
                    nx, ny = -qx / rn, -qy / rn
                    tx, ty = -ny, nx
        Nm0, Nm1 = 1.0 - t, t
        frozen.append(
            (
                w, xi, nx, ny, tx, ty, Na, Nb,
                int(m0), int(m1), Nm0, Nm1, xref, yref, qx, qy,
            )
        )
    return frozen


def eval_frozen_edge_qps(u, n0, n1, frozen):
    samples = []
    for w, xi, nx, ny, tx, ty, Na, Nb, m0, m1, Nm0, Nm1, xref, yref, qx, qy in frozen:
        pxp = xref + Na * u[2 * n0] + Nb * u[2 * n1]
        pyp = yref + Na * u[2 * n0 + 1] + Nb * u[2 * n1 + 1]
        uqx = Nm0 * u[2 * m0] + Nm1 * u[2 * m1]
        uqy = Nm0 * u[2 * m0 + 1] + Nm1 * u[2 * m1 + 1]
        g = (qx + uqx - pxp) * nx + (qy + uqy - pyp) * ny
        samples.append(
            (w, xi, g, nx, ny, tx, ty, Na, Nb, int(m0), int(m1), Nm0, Nm1, xref)
        )
    return samples


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
    g_open=0.0,
):
    """Cauchy σ_n and applied contact traction, one P0 value per EDGE2."""
    xs, gs, sns, pns, active, xis, ws = [], [], [], [], [], [], []
    for edge, e_parent in zip(edges, parent_elems):
        n0, n1 = int(edge[0]), int(edge[1])
        length, nx0, ny0 = edge_geometry(X, Y, n0, n1)[:3]
        if length <= 1e-16:
            continue
        gamma = gamma0 * mu / length
        parent_nodes = tri3_parent_nodes(ps, mesh, parent_block, int(e_parent))
        px = np.array([X[i] for i in parent_nodes])
        py = np.array([Y[i] for i in parent_nodes])
        u_elem = np.empty(6)
        for a, node in enumerate(parent_nodes):
            u_elem[2 * a] = u[2 * node]
            u_elem[2 * a + 1] = u[2 * node + 1]
        samples = collect_edge_qps(
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
        sn, _ = tri3_sigma_n(px, py, u_elem, mu, lam, mid[3], mid[4])
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
    forced_active=None,
    status_out=None,
    gap_rows_out=None,
    frozen_edges=None,
):
    """Unbiased Nitsche residual/tangent on one contact surface (paper P_{γ,θ}).

    Unilateral: assemble [P]_- . Gap is sampled with Gauss–Legendre then averaged
    to one P0 P=θσ_n+γḡ per EDGE2 (staircase, not a P1 checkerboard).
    """
    if new_active is None:
        new_active = set()
    n_active = 0
    for ie, (edge, e_parent) in enumerate(zip(edges, parent_elems)):
        n0, n1 = int(edge[0]), int(edge[1])
        length, nx0, ny0 = edge_geometry(X, Y, n0, n1)[:3]
        if length <= 1e-16:
            continue
        gamma = gamma0 * mu / length
        parent_nodes = tri3_parent_nodes(ps, mesh, parent_block, int(e_parent))
        px = np.array([X[i] for i in parent_nodes])
        py = np.array([Y[i] for i in parent_nodes])
        u_elem = np.empty(6)
        for a, node in enumerate(parent_nodes):
            u_elem[2 * a] = u[2 * node]
            u_elem[2 * a + 1] = u[2 * node + 1]
        if frozen_edges is not None and ie in frozen_edges:
            samples = eval_frozen_edge_qps(u, n0, n1, frozen_edges[ie])
        else:
            samples = collect_edge_qps(
                X, Y, u, n0, n1, length, nx0, ny0, other_edges, circle_R, snap_self_circle
            )
        if not samples:
            continue
        w_int = 0.0
        g_int = 0.0
        gt_int = 0.0
        dg_bar = {}
        mid = samples[0]

        def add_dg(dof, val):
            if val == 0.0:
                return
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
            if friction:
                pxp = (1.0 - xi) * (X[n0] + u[2 * n0]) + xi * (X[n1] + u[2 * n1])
                pyp = (1.0 - xi) * (Y[n0] + u[2 * n0 + 1]) + xi * (Y[n1] + u[2 * n1 + 1])
                qdx = Nm0 * (X[m0] + u[2 * m0]) + Nm1 * (X[m1] + u[2 * m1]) - pxp
                qdy = Nm0 * (Y[m0] + u[2 * m0 + 1]) + Nm1 * (Y[m1] + u[2 * m1 + 1]) - pyp
                gt_int += w * (qdx * tx + qdy * ty)
        inv_w = 1.0 / w_int
        g_bar = g_int * inv_w
        for dof in list(dg_bar.keys()):
            dg_bar[dof] *= inv_w
        sn, dsn = tri3_sigma_n(px, py, u_elem, mu, lam, mid[3], mid[4])
        Pn = (theta * sn if include_sigma else 0.0) + gamma * g_bar
        key = (surface_id, ie, 0)
        xref_mid = 0.5 * (X[n0] + X[n1])
        if gap_rows_out is not None:
            gap_rows_out.append((key, float(g_bar), float(xref_mid), dict(dg_bar), float(w_int)))
        if status_out is not None:
            status_out[key] = (g_bar, Pn, xref_mid)
        if forced_active is None:
            on = Pn < 0.0
        else:
            on = key in forced_active
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

        if not friction:
            continue
        gt_bar = gt_int * inv_w
        Pt = -gamma * gt_bar
        tx, ty = mid[5], mid[6]
        local_nodes = [n0, n1, mid[9], mid[10]]
        Nself = [0.5, 0.5, 0.0, 0.0]
        Nother = [0.0, 0.0, 0.5, 0.5]
        for k in range(4):
            for c in range(2):
                tv = (Nself[k] - Nother[k]) * (tx if c == 0 else ty)
                residual[2 * local_nodes[k] + c] += w_int * Pt * tv
        for i in range(4):
            for ci in range(2):
                tvi = (Nself[i] - Nother[i]) * (tx if ci == 0 else ty)
                row = 2 * local_nodes[i] + ci
                for j in range(4):
                    for cj in range(2):
                        tvj = (Nself[j] - Nother[j]) * (tx if cj == 0 else ty)
                        col = 2 * local_nodes[j] + cj
                        val = w_int * gamma * tvi * tvj
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
        ps,
        args.nx,
        args.ny,
        radius,
        args.r_inner * radius,
        args.width,
        args.height,
        args.gap,
        ny_block=getattr(args, "ny_block", None),
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

    edges_b, elems_b = filter_edges(X, Y, edges_b, elems_b, is_block_bottom)
    edges_o, elems_o = filter_edges(X, Y, edges_o, elems_o, is_outer_arc)
    edges_b, elems_b = unique_oriented_edges(edges_b, elems_b)
    edges_o, elems_o = unique_oriented_edges(edges_o, elems_o)
    if edges_b.size == 0 or edges_o.size == 0:
        raise RuntimeError(
            f"empty contact edges after filter: block={edges_b.shape} obstacle={edges_o.shape}"
        )
    edges_b = orient_edges(X, Y, edges_b, "down")
    edges_o = orient_edges(X, Y, edges_o, "outward")

    # Default two-body: unbiased Nitsche (Mlika–Renard–Chouly), θ=1/2 on both
    # surfaces, P = θ σ_n + γ g. Rigid obstacle/block is one-sided Nitsche.
    # `--biased` / `--penalty` / `--lagrange` opt out.
    if args.rigid_block and not args.rigid_obstacle:
        theta_b, theta_o = 0.0, 1.0
    elif args.rigid_obstacle or args.biased or args.penalty or args.lagrange:
        theta_b, theta_o = 1.0, 0.0
    else:
        theta_b, theta_o = 0.5, 0.5
    include_sigma = not args.penalty and not args.lagrange

    r_hist = []
    n_active_hist = []
    n_qp_hist = []
    g_c = np.zeros(ndofs)
    qp_samples = []

    def record_newton(rnorm, n_active, n_qp=0):
        r_hist.append(float(rnorm))
        n_active_hist.append(int(n_active))
        n_qp_hist.append(int(n_qp))

    def _contact_kwargs(
        u_vec, residual, coo, new_active, energy_acc=None, assemble=True,
        forced_active=None, status_out=None, gap_rows_out=None,
    ):
        n_qp = 0
        if theta_b > 0:
            n_qp += surface_contrib(
                X, Y, u_vec, edges_b, elems_b, bid_b, edges_o, ps, mesh, mu_b, lam_b,
                args.gamma0, theta_b, args.friction, args.mu_f, residual, coo, radius,
                0, new_active, False, energy_acc, qp_samples if assemble else None,
                assemble, include_sigma, forced_active, status_out, gap_rows_out,
            )
        if theta_o > 0:
            n_qp += surface_contrib(
                X, Y, u_vec, edges_o, elems_o, bid_o, edges_b, ps, mesh, mu_o, lam_o,
                args.gamma0, theta_o, args.friction, args.mu_f, residual, coo, radius,
                1, new_active, True, energy_acc, qp_samples if assemble else None,
                assemble, include_sigma, forced_active, status_out, gap_rows_out,
            )
        return n_qp

    def residual_tangent(u_vec, forced_active=None):
        qp_samples.clear()
        g_contact = np.zeros(ndofs)
        coo = []
        new_active = set()
        n_qp = _contact_kwargs(
            u_vec, g_contact, coo, new_active, assemble=True, forced_active=forced_active
        )
        resid = Ke @ u_vec + g_contact
        if coo:
            ii, jj, vv = zip(*coo)
            Kn = sparse.coo_matrix((vv, (ii, jj)), shape=(ndofs, ndofs)).tocsr()
        else:
            Kn = sparse.csr_matrix((ndofs, ndofs))
        Ksys = Ke + Kn
        Ksys, resid = apply_dirichlet_system(Ksys, resid, constrained, u_vec, u_bc)
        return resid, Ksys, g_contact, Kn, n_qp, new_active

    x_cmax = 1.25 * float(np.sqrt(max(2.0 * radius * abs(args.indent), 0.0)))
    g_tol = 1e-8 * max(radius, 1.0)

    def qp_status(u_vec):
        status = {}
        uc = u_vec.copy()
        uc[constrained] = u_bc[constrained]
        _contact_kwargs(
            uc, None, None, set(), assemble=False, forced_active=set(), status_out=status
        )
        return status

    def overlapping(status):
        return {
            k
            for k, (g, Pn, xp) in status.items()
            if Pn < 0.0 and abs(xp) <= x_cmax
        }

    def elastic_dirichlet(u_vec):
        resid = Ke @ u_vec
        Ksys, resid = apply_dirichlet_system(
            Ke.copy(), resid, constrained, u_vec, u_bc
        )
        return resid, Ksys

    def edge_rows(u_vec):
        """One geometric gap per EDGE2 midpoint (P0 traction, P1-stable)."""
        rows = []
        uc = u_vec.copy()
        uc[constrained] = u_bc[constrained]
        for ie in range(int(edges_b.shape[0])):
            n0, n1 = int(edges_b[ie, 0]), int(edges_b[ie, 1])
            xp = 0.5 * (float(X[n0]) + float(X[n1]))
            if abs(xp) > x_cmax:
                continue
            px = 0.5 * (X[n0] + uc[2 * n0] + X[n1] + uc[2 * n1])
            py = 0.5 * (Y[n0] + uc[2 * n0 + 1] + Y[n1] + uc[2 * n1 + 1])
            m0, m1, t, qx, qy = master_on_circle(px, py, X, Y, edges_o, radius)
            rn = float(np.hypot(qx, qy))
            if rn <= 1e-30:
                continue
            nx, ny = -qx / rn, -qy / rn
            Nm0, Nm1 = 1.0 - t, t
            uqx = Nm0 * uc[2 * m0] + Nm1 * uc[2 * m1]
            uqy = Nm0 * uc[2 * m0 + 1] + Nm1 * uc[2 * m1 + 1]
            g = (qx + uqx - px) * nx + (qy + uqy - py) * ny
            dg = {}

            def add(dof, val):
                if val == 0.0:
                    return
                dg[dof] = dg.get(dof, 0.0) + val

            add(2 * n0, -0.5 * nx)
            add(2 * n0 + 1, -0.5 * ny)
            add(2 * n1, -0.5 * nx)
            add(2 * n1 + 1, -0.5 * ny)
            add(2 * m0, Nm0 * nx)
            add(2 * m0 + 1, Nm0 * ny)
            add(2 * m1, Nm1 * nx)
            add(2 * m1 + 1, Nm1 * ny)
            rows.append((ie, float(g), xp, dg))
        return rows

    def edge_overlapping(rows):
        return {ie for ie, g, xp, _dg in rows if g < 0.0}

    def solve_kkt(u_vec, keys):
        r_el, Kel = elastic_dirichlet(u_vec)
        rows = [r for r in edge_rows(u_vec) if r[0] in keys]
        if not rows:
            du = spsolve(Kel, -r_el)
            return du, {}, [], r_el, sparse.csr_matrix((0, ndofs)), np.zeros(0)
        nA = len(rows)
        G = sparse.lil_matrix((nA, ndofs))
        gvec = np.zeros(nA)
        order = []
        for i, (ie, gval, _xp, dg) in enumerate(rows):
            order.append(ie)
            gvec[i] = gval
            for dof, val in dg.items():
                if mask[dof]:
                    continue
                G[i, dof] += val
        G = G.tocsr()
        # Signorini: min ½ u·Ke u s.t. g≥0. Lagrangian ½ u·Ke u − λ g, λ≥0.
        # Stationarity Ke u − Gᵀ λ = 0. λ is the edge force; p = λ/L is P0.
        Kaug = sparse.bmat(
            [
                [Kel, -G.T],
                [G, sparse.eye(nA, format="csr") * (-1e-16)],
            ],
            format="csr",
        )
        sol = spsolve(Kaug, np.concatenate([-r_el, -gvec]))
        du = np.asarray(sol[:ndofs], dtype=np.float64)
        lam = np.asarray(sol[ndofs:], dtype=np.float64)
        lam_map = {order[i]: float(lam[i]) for i in range(nA)}
        return du, lam_map, order, r_el, G, lam

    use_kkt = bool(getattr(args, "lagrange", False)) and not bool(
        args.rigid_obstacle or args.rigid_block
    )
    u_bc_full = u_bc.copy()
    n_load = args.load_steps if args.load_steps > 0 else 1
    u[:] = 0.0
    lam_by_key = {}
    for load in range(1, n_load + 1):
        u_bc[:] = u_bc_full * (load / float(n_load))
        u[constrained] = u_bc[constrained]
        if n_load > 1:
            print(f"load[{load}/{n_load}] indent={args.indent * load / n_load:.4g}")
        A = set()
        for it in range(args.max_newton):
            if not use_kkt:
                status = qp_status(u)
                A_nat = overlapping(status)
                residual, K, g_c, Kn, n_qp, A = residual_tangent(u, forced_active=None)
                rnorm = float(np.linalg.norm(residual))
                record_newton(rnorm, len(A), n_qp)
                print(
                    f"newton[{it:02d}] ||r||={rnorm:.3e}  contact_nnz={Kn.nnz}  "
                    f"nitsche_qp={n_qp}  |A|={len(A)}  |A_nat|={len(A_nat)}"
                )
                if rnorm < args.rtol:
                    break
                du = spsolve(K, -residual)
                if not np.all(np.isfinite(du)):
                    raise RuntimeError("Newton increment is not finite")
                u[:] = u + du
                u[constrained] = u_bc[constrained]
                st = qp_status(u)
                A_nat = overlapping(st)
                extra = ""
                if A_nat:
                    gs = sorted((st[k][2], st[k][0]) for k in A_nat if k in st)
                    if len(gs) <= 12:
                        extra += "  A=[" + ", ".join(f"{x:.2f}:{g:.2e}" for x, g in gs) + "]"
                    elif gs:
                        extra += (
                            f"  A=[{gs[0][0]:.2f}:{gs[0][1]:.2e} ... "
                            f"{gs[-1][0]:.2f}:{gs[-1][1]:.2e}]"
                        )
                print(f"          accept |A|={len(A_nat)}{extra}")
                continue

            rows = edge_rows(u)
            A_nat = edge_overlapping(rows)
            if not A:
                r_el, Kel = elastic_dirichlet(u)
                rnorm = float(np.linalg.norm(r_el))
                record_newton(rnorm, 0, 0)
                print(
                    f"newton[{it:02d}] ||r||={rnorm:.3e}  contact_nnz=0  "
                    f"nitsche_qp=0  |A|=0  |A_nat|={len(A_nat)}"
                )
                if rnorm < args.rtol and not A_nat:
                    break
                du = spsolve(Kel, -r_el)
                if not np.all(np.isfinite(du)):
                    raise RuntimeError("Newton increment is not finite")
                u[:] = u + du
                u[constrained] = u_bc[constrained]
                rows = edge_rows(u)
                A_nat = edge_overlapping(rows)
                A = set(A_nat)
                extra = f"  added={len(A)}"
                if A:
                    xg = [(xp, g) for ie, g, xp, _d in rows if ie in A]
                    extra += "  A=[" + ", ".join(f"{x:.2f}:{g:.2e}" for x, g in xg) + "]"
                print(f"          accept |A|={len(A)}{extra}")
                continue

            du, lam_map, order, r_el, Gm, lam = solve_kkt(u, A)
            if not np.all(np.isfinite(du)) or not np.all(np.isfinite(lam)):
                raise RuntimeError("Newton increment is not finite")
            u[:] = u + du
            u[constrained] = u_bc[constrained]
            lam_by_key = {k: v for k, v in lam_map.items() if v > 0.0}
            g_c = np.asarray(-Gm.T.dot(lam), dtype=np.float64) if lam.size else np.zeros(ndofs)
            r_el, _ = elastic_dirichlet(u)
            residual = r_el + g_c
            residual[constrained] = 0.0
            rnorm = float(np.linalg.norm(residual))
            record_newton(rnorm, len(A), len(order))
            rows = edge_rows(u)
            A_nat = edge_overlapping(rows)
            dropped = {k for k, lv in lam_map.items() if lv < -1e-12}
            A_new = {k for k in order if k not in dropped}
            outside = A_nat - A_new
            added = set(outside)
            if added:
                A_new |= added
            if not A_new and A_nat:
                added = set(A_nat)
                A_new = set(added)
            opened = A - A_new
            gmin = min((g for _ie, g, _x, _d in rows), default=0.0)
            print(
                f"newton[{it:02d}] ||r||={rnorm:.3e}  contact_nnz={Gm.nnz}  "
                f"nitsche_qp={len(order)}  |A|={len(A)}  |A_nat|={len(A_nat)}"
            )
            extra = ""
            if added:
                extra += f"  added={len(added)}"
            if opened:
                extra += f"  opened={len(opened)}"
            if dropped:
                extra += f"  tensile={len(dropped)}"
            if lam.size:
                extra += f"  λ∈[{lam.min():.2e},{lam.max():.2e}]"
            extra += f"  gmin={gmin:.2e}"
            if A_new:
                xg = sorted((xp, g) for ie, g, xp, _d in rows if ie in A_new)
                if len(xg) <= 12:
                    extra += "  A=[" + ", ".join(f"{x:.2f}:{g:.2e}" for x, g in xg) + "]"
                elif xg:
                    extra += (
                        f"  A=[{xg[0][0]:.2f}:{xg[0][1]:.2e} ... "
                        f"{xg[-1][0]:.2f}:{xg[-1][1]:.2e}]"
                    )
            print(f"          accept |A|={len(A_new)}{extra}")
            if rnorm < args.rtol and not added and not dropped and gmin >= -g_tol:
                A = A_new
                break
            A = A_new

    if use_kkt:
        r_el, _ = elastic_dirichlet(u)
        g_c = np.zeros(ndofs)
        if A:
            _du, lam_map, _order, _r, Gm, lam = solve_kkt(u, A)
            lam_by_key = {k: v for k, v in lam_map.items() if v > 0.0}
            if lam.size:
                g_c = np.asarray(-Gm.T.dot(lam), dtype=np.float64)
        residual = r_el + g_c
        residual[constrained] = 0.0
        Kn = sparse.csr_matrix((ndofs, ndofs))
        n_qp = len(A)
    else:
        residual, K, g_c, Kn, n_qp, A = residual_tangent(u, forced_active=None)
    r_final = float(np.linalg.norm(residual))
    if not r_hist or abs(r_hist[-1] - r_final) > 1e-30:
        record_newton(r_final, len(A) if A is not None else 0, n_qp)
        print(f"newton[final] ||r||={r_final:.3e}  contact_nnz={getattr(Kn, 'nnz', 0)}  nitsche_qp={n_qp}")

    nodes_b = unique_nodes_from_edges(edges_b)
    gap = np.empty(nodes_b.size)
    for i, node in enumerate(nodes_b):
        node = int(node)
        px = float(X[node] + u[2 * node])
        py = float(Y[node] + u[2 * node + 1])
        m0, m1, t, qx, qy = master_on_circle(px, py, X, Y, edges_o, radius)
        rn = float(np.hypot(qx, qy))
        if rn <= 1e-30:
            gap[i] = 0.0
            continue
        nx, ny = -qx / rn, -qy / rn
        uqx = (1.0 - t) * u[2 * m0] + t * u[2 * m1]
        uqy = (1.0 - t) * u[2 * m0 + 1] + t * u[2 * m1 + 1]
        gap[i] = (qx + uqx - px) * nx + (qy + uqy - py) * ny
    penetration = float(np.linalg.norm(np.minimum(gap, 0.0)))
    r_full = np.asarray(Ke @ u + g_c, dtype=np.float64)
    top = np.flatnonzero(np.abs(Y[:n_block] - y_top) <= 4.0 * tol)
    F = abs(float(np.sum(r_full[2 * top + 1]))) if top.size else abs(
        float(np.sum(g_c[1 : 2 * n_block : 2]))
    )
    E_star = effective_modulus(E_b, nu_b, E_o if not args.rigid_obstacle else 1e300, nu_o)
    xc = 0.5 * (X[edges_b[:, 0]] + X[edges_b[:, 1]])
    p_hertz, a, p0 = hertz_pressure(xc, F, radius, E_star)
    tr_x, tr_g, tr_sn, tr_pn, tr_on, tr_xi, tr_w = contact_trace(
        X, Y, u, edges_b, elems_b, bid_b, edges_o, ps, mesh, mu_b, lam_b,
        args.gamma0, theta_b if theta_b > 0 else 1.0, radius, include_sigma, False, 0.0,
    )
    tr_x_o, tr_g_o, tr_sn_o, tr_pn_o, tr_on_o, _, tr_w_o = contact_trace(
        X, Y, u, edges_o, elems_o, bid_o, edges_b, ps, mesh, mu_o, lam_o,
        args.gamma0, theta_o if theta_o > 0 else 1.0, radius, include_sigma, True, 0.0,
    )
    p_cauchy = -tr_sn
    p_cauchy_o = -tr_sn_o
    th = theta_b if theta_b > 0.0 else 1.0
    p_applied = np.where(tr_on, np.maximum(-tr_pn / th, 0.0), 0.0)
    p_cauchy_max = float(np.max(p_cauchy)) if p_cauchy.size else 0.0
    p_cauchy_o_max = float(np.max(p_cauchy_o)) if p_cauchy_o.size else 0.0
    p_applied_max = float(np.max(p_applied)) if p_applied.size else 0.0
    F_int = float(np.sum(p_applied * tr_w)) if tr_w.size else 0.0
    n_active = int(np.count_nonzero(tr_on))
    if use_kkt:
        n_active = len(lam_by_key)

    x_sym = max(1.25 * a, 0.2) if a > 0.0 else 0.45

    def even_l2(x, p, xmax):
        x = np.asarray(x, dtype=float)
        p = np.asarray(p, dtype=float)
        m = np.abs(x) <= xmax
        x, p = x[m], p[m]
        if x.size < 2:
            return float("nan")
        o = np.argsort(x)
        x, p = x[o], p[o]
        num = float(np.sqrt(np.mean((p - np.interp(-x, x, p)) ** 2)))
        den = max(float(np.sqrt(np.mean(p ** 2))), 1e-30)
        return num / den

    def pair_l2(x0, p0, x1, p1, xmax):
        x0 = np.asarray(x0, dtype=float)
        p0 = np.asarray(p0, dtype=float)
        x1 = np.asarray(x1, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        m0 = np.abs(x0) <= xmax
        x0, p0 = x0[m0], p0[m0]
        if x0.size < 2 or x1.size < 2:
            return float("nan")
        o1 = np.argsort(x1)
        p1_on_0 = np.interp(x0, x1[o1], p1[o1])
        num = float(np.sqrt(np.mean((p0 - p1_on_0) ** 2)))
        den = max(float(np.sqrt(np.mean(p0 ** 2))), 1e-30)
        return num / den

    even_b = even_l2(tr_x, p_cauchy, x_sym)
    even_o = even_l2(tr_x_o, p_cauchy_o, x_sym)
    react = pair_l2(tr_x, p_cauchy, tr_x_o, p_cauchy_o, x_sym)

    print(
        f"nodes={mesh.n_nodes()} dofs={ndofs} F={F:.4e} F_int={F_int:.4e} "
        f"a_hertz={a:.4e} p0={p0:.4e} |g_-|={penetration:.3e}  "
        f"theta=({theta_b:g},{theta_o:g})  sigma={'on' if include_sigma else 'off'}  "
        f"n_active={n_active}  max(-σ_n^b)={p_cauchy_max:.4e}  max(-σ_n^o)={p_cauchy_o_max:.4e}  "
        f"max(-P_n)={p_applied_max:.4e}  even_b={even_b:.3f}  even_o={even_o:.3f}  "
        f"σ_n b↔o={react:.3f}"
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
        "n_active_hist": n_active_hist,
        "n_qp_hist": n_qp_hist,
        "rtol": float(args.rtol),
        "n_block": n_block,
        "tris_b": block_triangles(ps, mesh, bid_b),
        "tris_o": block_triangles(ps, mesh, bid_o),
        "rigid_obstacle": bool(args.rigid_obstacle),
        "qp_x": tr_x,
        "qp_g": tr_g,
        "qp_sn": tr_sn,
        "qp_pn": tr_pn,
        "qp_active": tr_on,
        "qp_xi": tr_xi,
        "qp_w": tr_w,
        "p_cauchy": p_cauchy,
        "p_cauchy_o": p_cauchy_o,
        "qp_x_o": tr_x_o,
        "qp_sn_o": tr_sn_o,
        "p_applied": p_applied,
        "F_int": F_int,
        "n_active": n_active,
        "include_sigma": include_sigma,
        "theta_b": float(theta_b),
        "theta_o": float(theta_o),
        "indent": float(args.indent),
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
    p.add_argument("--max-newton", type=int, default=40)
    p.add_argument(
        "--load-steps",
        type=int,
        default=0,
        help="Indent increments. 0/1: full indent in one shot.",
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
        help="One-sided Nitsche on the block (θ=1). Default is unbiased (θ=1/2 both).",
    )
    p.add_argument(
        "--unbiased",
        action="store_true",
        help="Nitsche on both surfaces (paper). Default for two-body Hertz.",
    )
    p.add_argument(
        "--nitsche-stress",
        action="store_true",
        help="Kept for compatibility; σ_n is already in P unless --penalty.",
    )
    p.add_argument(
        "--lagrange",
        action="store_true",
        help="Two-body: nodal/edge Lagrange (γ=∞), not Nitsche.",
    )
    p.add_argument(
        "--penalty",
        action="store_true",
        help="Geometric penalty P=γg (no σ_n). Not Nitsche.",
    )
    p.add_argument("--plot", action="store_true")
    p.add_argument("--conv", action="store_true", help="Mesh refinement study")
    p.add_argument("--check", action="store_true", help="Tiny run, exit 0 if Newton decreases residual")
    return p.parse_args(argv)


def plot_solution(result):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    X, Y, u = result["X"], result["Y"], result["u"]
    xd, yd = X + u[0::2], Y + u[1::2]
    uy = u[1::2]
    vmin, vmax = float(np.min(uy)), float(np.max(uy))
    if vmax <= vmin:
        vmax = vmin + 1e-16

    pad = 0.04 * max(float(np.ptp(xd)), float(np.ptp(yd)), 1e-12)
    x0, x1 = float(np.min(xd)) - pad, float(np.max(xd)) + pad
    y0, y1 = float(np.min(yd)) - pad, float(np.max(yd)) + pad
    fig = plt.figure(figsize=(14.5, 7.2), layout="constrained")
    gs = GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[1.7, 0.055, 1.0],
        height_ratios=[1.15, 1.0],
    )
    ax0 = fig.add_subplot(gs[:, 0])
    cax = fig.add_subplot(gs[:, 1])
    axg = fig.add_subplot(gs[0, 2])
    axc = fig.add_subplot(gs[1, 2])
    tpc = None
    for tris in (result["tris_b"], result["tris_o"]):
        tpc = ax0.tripcolor(
            xd, yd, tris, uy, shading="gouraud", cmap="coolwarm", vmin=vmin, vmax=vmax
        )
        ax0.triplot(xd, yd, tris, color="k", lw=0.25, alpha=0.55)
    ax0.set_xlim(x0, x1)
    ax0.set_ylim(y0, y1)
    ax0.set_box_aspect((y1 - y0) / (x1 - x0))
    ax0.set_aspect("equal")
    fig.colorbar(tpc, cax=cax, label="u_y")
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    if result.get("rigid_obstacle"):
        mesh_title = "deformed TRI3 (Nitsche vs rigid obstacle)"
    elif result.get("theta_o", 0.0) > 0.0 and result.get("include_sigma"):
        mesh_title = "deformed TRI3 (unbiased Nitsche)"
    elif result.get("include_sigma"):
        mesh_title = "deformed TRI3 (biased Nitsche)"
    else:
        mesh_title = "deformed TRI3 (penalty)"
    ax0.set_title(mesh_title)

    nodes = result["nodes_b"]
    order = np.argsort(X[nodes])
    xg = X[nodes][order]
    gap = np.asarray(result["gap"])[order]
    xc_order = np.argsort(result["xc"])
    xc = np.asarray(result["xc"])[xc_order]
    ph = np.asarray(result["p_hertz"])[xc_order]

    axp = axg.twinx()
    tr_x = np.asarray(result.get("qp_x", []))
    tr_g = np.asarray(result.get("qp_g", []))
    if tr_x.size and tr_g.size:
        order_g = np.argsort(tr_x)
        ln_gap = axg.plot(
            tr_x[order_g],
            tr_g[order_g],
            "o-",
            color="C0",
            ms=3,
            label="gap (edge mean, >0 open, <0 overlap)",
        )
    else:
        ln_gap = axg.plot(xg, gap, "o-", color="C0", ms=4, label="gap (>0 open, <0 overlap)")
    axg.axhline(0.0, color="0.45", lw=0.8)
    a = float(result["a"])
    if a > 0:
        axg.axvline(-a, color="0.55", ls="--", lw=0.8)
        axg.axvline(a, color="0.55", ls="--", lw=0.8)
    ln_hertz = axp.plot(xc, ph, color="C1", label="Hertz p(x) from F")
    p_cauchy = np.asarray(result.get("p_cauchy", []))
    p_cauchy_o = np.asarray(result.get("p_cauchy_o", []))
    tr_x_o = np.asarray(result.get("qp_x_o", []))
    ln_cauchy = []
    ln_cauchy_o = []
    if tr_x.size and p_cauchy.size:
        order_c = np.argsort(tr_x)
        ln_cauchy = axp.plot(
            tr_x[order_c],
            p_cauchy[order_c],
            drawstyle="steps-mid",
            color="C3",
            lw=1.6,
            label="block $-\\sigma_n$ (P0)",
        )
    if tr_x_o.size and p_cauchy_o.size:
        order_o = np.argsort(tr_x_o)
        ln_cauchy_o = axp.plot(
            tr_x_o[order_o],
            p_cauchy_o[order_o],
            drawstyle="steps-mid",
            color="C2",
            lw=1.6,
            label="obstacle $-\\sigma_n$ (P0)",
        )
    axg.set_xlabel("x (reference)")
    axg.set_ylabel("gap", color="C0")
    axp.set_ylabel("pressure", color="C1")
    axg.set_title(f"contact cut  a={a:.3g}  p0={result['p0']:.3g}  indent={result.get('indent', '')}")
    xpad = max(1.6 * a, 0.4) if a > 0.0 else 0.5
    axg.set_xlim(-xpad, xpad)
    lines = ln_gap + ln_hertz + ln_cauchy + ln_cauchy_o
    axg.legend(lines, [ln.get_label() for ln in lines], loc="best")

    rh = np.asarray(result.get("r_hist", []), dtype=float)
    if rh.size:
        rh = np.maximum(rh, 1e-30)
        vh = np.asarray(result.get("vcycle_hist", []), dtype=float)
        use_vcycles = vh.size == rh.size
        if use_vcycles:
            xs = np.cumsum(vh)
            xlabel = "V-cycles"
            n_vc = int(np.sum(vh))
            conv_title = f"solver convergence  ({n_vc} V-cycles)"
        else:
            xs = np.arange(rh.size, dtype=float)
            xlabel = "Newton iteration"
            conv_title = "solver convergence"
        ln_r = axc.semilogy(xs, rh, "o-", color="C4", ms=4, label=r"$\|r\|$")
        dh = np.asarray(result.get("direct_hist", []), dtype=bool)
        ln_dir = []
        n_direct = 0
        if dh.size == rh.size and np.any(dh):
            n_direct = int(np.count_nonzero(dh))
            ln_dir = axc.semilogy(
                xs[dh],
                rh[dh],
                "x",
                color="k",
                ms=8,
                mew=1.8,
                label=f"spsolve ({n_direct})",
                zorder=5,
            )
        if use_vcycles:
            n_vc = int(np.sum(vh))
            extra = f", {n_direct} spsolve" if n_direct else ""
            conv_title = f"solver convergence  ({n_vc} V-cycles{extra})"
            for x, y, dv, is_d in zip(xs, rh, vh, dh if dh.size == rh.size else [False] * rh.size):
                if is_d:
                    axc.annotate(
                        "D",
                        (x, y),
                        textcoords="offset points",
                        xytext=(4, -10),
                        fontsize=8,
                        color="k",
                        fontweight="bold",
                    )
                elif dv > 0:
                    axc.annotate(
                        f"{int(dv)}",
                        (x, y),
                        textcoords="offset points",
                        xytext=(4, 4),
                        fontsize=8,
                        color="C4",
                    )
        rtol = result.get("rtol")
        ln_tol = []
        if rtol is not None and float(rtol) > 0.0:
            ln_tol = axc.axhline(
                float(rtol), color="0.45", ls="--", lw=0.9, label=f"rtol={float(rtol):g}"
            )
            ln_tol = [ln_tol]
        axc.set_xlabel(xlabel)
        axc.set_ylabel(r"$\|r\|$")
        axc.set_title(conv_title)
        axc.grid(True, which="both", ls=":", alpha=0.45)
        if not use_vcycles:
            axc.set_xticks(xs)
        conv_lines = list(ln_r) + list(ln_dir) + ln_tol
        ah = np.asarray(result.get("n_active_hist", []), dtype=float)
        qh = np.asarray(result.get("n_qp_hist", []), dtype=float)
        if ah.size == rh.size:
            axa = axc.twinx()
            ln_a = axa.plot(xs, ah, "s--", color="C5", ms=4, label="$|A|$")
            conv_lines += ln_a
            if qh.size == rh.size and not np.allclose(qh, ah):
                ln_q = axa.plot(xs, qh, "^:", color="C6", ms=4, label="nitsche qp")
                conv_lines += ln_q
            if use_vcycles:
                ln_v = axa.plot(
                    xs, vh, "v-", color="C8", ms=5, label="V-cycles / step"
                )
                conv_lines += ln_v
            axa.set_ylabel("active set" + (" / V-cycles" if use_vcycles else ""))
            axa.set_ylim(bottom=-0.2)
        axc.legend(conv_lines, [ln.get_label() for ln in conv_lines], loc="best")
    else:
        axc.set_axis_off()
        axc.set_title("solver convergence (no history)")

    out = os.path.join(os.path.dirname(__file__), "nitsche_contact.png")
    fig.savefig(out, dpi=140)
    print(f"saved {out}")
    try:
        plt.show()
    except Exception:
        pass


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


def _fail(msg):
    raise SystemExit(msg)


def check_rigid(result, args):
    r_hist = result.get("r_hist", [])
    if not r_hist or not np.all(np.isfinite(r_hist)):
        _fail("rigid check failed: non-finite Newton residual")
    if not np.isfinite(result["penetration"]):
        _fail("rigid check failed: non-finite penetration")
    if r_hist[-1] > 1e-6:
        _fail(f"rigid check failed: Newton residual {r_hist[0]:.3e} -> {r_hist[-1]:.3e}")
    if result["penetration"] > 0.5 * args.indent:
        _fail(
            f"rigid check failed: remaining penetration {result['penetration']:.3e} "
            f"with indent {args.indent:g}"
        )
    nodes = result["nodes_b"]
    i_min = int(np.argmin(result["gap"]))
    x_min = float(result["X"][nodes[i_min]])
    if abs(x_min) > 0.25:
        _fail(f"rigid check failed: deepest gap at x={x_min:.3g}, expected near x=0")
    if result["F"] <= 0.0:
        _fail("rigid check failed: non-positive contact force")


def check_twobody(result, args, F_rigid):
    r_hist = result.get("r_hist", [])
    if not r_hist or not np.all(np.isfinite(r_hist)):
        _fail("two-body check failed: non-finite Newton residual")
    if r_hist[-1] > 1e-6:
        _fail(f"two-body check failed: Newton residual {r_hist[0]:.3e} -> {r_hist[-1]:.3e}")
    if result["n_active"] < 1:
        _fail(f"two-body check failed: only {result['n_active']} active quadrature points")
    if result["F"] < 0.15 * F_rigid:
        _fail(
            f"two-body check failed: F={result['F']:.3e} "
            f"is too small vs rigid F={F_rigid:.3e}"
        )
    if result["F"] > 1.5 * F_rigid:
        _fail(
            f"two-body check failed: F={result['F']:.3e} "
            f"exceeds rigid F={F_rigid:.3e}"
        )
    nodes = result["nodes_b"]
    gap = np.asarray(result["gap"])
    i_min = int(np.argmin(gap))
    x_min = float(result["X"][nodes[i_min]])
    if abs(x_min) > 0.25:
        _fail(f"two-body check failed: deepest gap at x={x_min:.3g}, expected near x=0")
    xg = result["X"][nodes]
    far = np.abs(xg) > 0.45 * args.width
    if np.any(far) and np.min(gap[far]) < -1e-4:
        _fail("two-body check failed: far-field overlap (not a Hertz patch)")
    if float(np.min(gap)) < -0.25 * abs(args.indent):
        _fail(
            f"two-body check failed: contact face sucked in "
            f"(min gap {float(np.min(gap)):.3e} vs indent {args.indent:g})"
        )
    uy_c = np.asarray(result["u"])[2 * np.asarray(nodes) + 1]
    if float(np.min(uy_c)) < -1.25 * abs(args.indent):
        _fail(
            f"two-body check failed: contact uy={float(np.min(uy_c)):.3e} "
            f"past indent {args.indent:g} (stretch / suction)"
        )
    p_cauchy = np.asarray(result.get("p_cauchy", []))
    if p_cauchy.size and float(np.max(p_cauchy)) <= 0.0:
        _fail("two-body check failed: Cauchy -σ_n is not compressive")
    p_applied = np.asarray(result["p_applied"])
    gqp = np.asarray(result["qp_g"])
    if gqp.size and p_applied.size:
        overlap = gqp < 0.0
        pn = np.asarray(result["qp_pn"])
        if np.any(overlap) and pn.size == gqp.size and float(np.mean(pn[overlap])) > 1e-10:
            _fail("two-body check failed: net tensile P on overlap (adhesion)")


def main(argv=None):
    args = parse_args(argv)
    ps = _load_pysfem()
    ps.init()
    try:
        if args.conv:
            refinement_study(ps, args)
        elif args.check:
            rigid_args = copy.copy(args)
            rigid_args.nx = 8
            rigid_args.ny = 4
            rigid_args.max_newton = 20
            rigid_args.plot = False
            rigid_args.indent = 0.02
            rigid_args.rigid_obstacle = True
            rigid_args.load_steps = 1
            rigid = solve_nitsche(ps, rigid_args)
            check_rigid(rigid, rigid_args)
            two_args = copy.copy(rigid_args)
            two_args.rigid_obstacle = False
            two_args.load_steps = 1
            two_args.max_newton = 80
            two = solve_nitsche(ps, two_args)
            check_twobody(two, two_args, rigid["F"])
            print(
                f"check ok  rigid F={rigid['F']:.4e}  two-body F={two['F']:.4e}  "
                f"n_active={two['n_active']}"
            )
        else:
            result = solve_nitsche(ps, args)
            if args.plot:
                plot_solution(result)
    finally:
        ps.finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

