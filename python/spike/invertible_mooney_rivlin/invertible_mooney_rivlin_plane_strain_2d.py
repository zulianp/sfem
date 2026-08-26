#!/usr/bin/env python3
"""
Tether-free inversion recovery test for a nearly incompressible
plane-strain Mooney-Rivlin material.

Physical branch, J >= Jc:
    W(F) = C10 * (J^(-2/3) * I1 - 3)
         + C01 * (J^(-4/3) * I2 - 3)
         + kappa/2 * (J - 1)^2

with a 3D plane-strain embedding F3 = diag(F, 1):
    I1 = F:F + 1
    I2 = F:F + J^2

For J < Jc, both singular powers J^(-2/3) and J^(-4/3) are replaced by
globally C2 quartic continuations with constant extension for J <= Jmin.
The nonlinear solve uses absolute-eigenvalue projected Newton and an Armijo
line search with no inversion barrier.
"""

import argparse
import os
import tempfile

import numpy as np

_cache_root = os.path.join(tempfile.gettempdir(), "sfem_matplotlib_cache")
_xdg_root = os.path.join(tempfile.gettempdir(), "sfem_xdg_cache")
os.makedirs(_cache_root, exist_ok=True)
os.makedirs(os.path.join(_xdg_root, "fontconfig"), exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _cache_root)
os.environ.setdefault("XDG_CACHE_HOME", _xdg_root)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def continued_power_coeffs(Jc, Jmin, m):
    """
    h(J) continues J^-m.

    p(z) = a0 + a1 z + a2 z^2 + a3 z^3 + a4 z^4,
    z = (J - Jc) / (Jc - Jmin).
    Match value/slope/curvature at Jc and flatten slope/curvature at Jmin.
    """
    delta = Jc - Jmin
    sc = Jc ** (-m)
    sp = -m * Jc ** (-m - 1.0)
    spp = m * (m + 1.0) * Jc ** (-m - 2.0)

    a0 = sc
    a1 = delta * sp
    a2 = 0.5 * delta * delta * spp
    a3 = (4.0 / 3.0) * a2 - a1
    a4 = 0.5 * (a2 - a1)
    return a0, a1, a2, a3, a4


def continued_power(J, Jc, Jmin, m):
    if J >= Jc:
        h = J ** (-m)
        hp = -m * J ** (-m - 1.0)
        hpp = m * (m + 1.0) * J ** (-m - 2.0)
        return h, hp, hpp

    a0, a1, a2, a3, a4 = continued_power_coeffs(Jc, Jmin, m)
    delta = Jc - Jmin

    if J <= Jmin:
        z = -1.0
        h = a0 + z * (a1 + z * (a2 + z * (a3 + z * a4)))
        return h, 0.0, 0.0

    z = (J - Jc) / delta
    h = a0 + z * (a1 + z * (a2 + z * (a3 + z * a4)))
    hz = a1 + z * (2.0 * a2 + z * (3.0 * a3 + z * 4.0 * a4))
    hzz = 2.0 * a2 + z * (6.0 * a3 + z * 12.0 * a4)
    return h, hz / delta, hzz / (delta * delta)


def cofactor_2d(F):
    return np.array([[F[1, 1], -F[1, 0]], [-F[0, 1], F[0, 0]]], dtype=float)


def structured_square_mesh(nx, ny):
    X = np.array(
        [[i / nx, j / ny] for j in range(ny + 1) for i in range(nx + 1)],
        dtype=float,
    )

    def node(i, j):
        return j * (nx + 1) + i

    tris = []
    for j in range(ny):
        for i in range(nx):
            n00 = node(i, j)
            n10 = node(i + 1, j)
            n01 = node(i, j + 1)
            n11 = node(i + 1, j + 1)
            if (i + j) & 1:
                tris.append([n00, n10, n01])
                tris.append([n10, n11, n01])
            else:
                tris.append([n00, n10, n11])
                tris.append([n00, n11, n01])

    return X, np.asarray(tris, dtype=int), node(nx, 0)


def element_kinematics(xe, Xe):
    Dm = np.column_stack((Xe[1] - Xe[0], Xe[2] - Xe[0]))
    inv_Dm = np.linalg.inv(Dm)
    Ds = np.column_stack((xe[1] - xe[0], xe[2] - xe[0]))
    F = Ds @ inv_Dm
    A0 = 0.5 * abs(np.linalg.det(Dm))
    return F, inv_Dm, A0


def element_B(inv_Dm):
    B = np.zeros((4, 6))
    for j in range(6):
        dxe = np.zeros((3, 2))
        dxe.flat[j] = 1.0
        dDs = np.column_stack((dxe[1] - dxe[0], dxe[2] - dxe[0]))
        dF = dDs @ inv_Dm
        B[:, j] = dF.ravel()
    return B


def mooney_rivlin_energy_gradient_hessian(F, C10, C01, kappa, Jc, Jmin):
    J = np.linalg.det(F)
    I = np.sum(F * F)
    I1 = I + 1.0
    I2 = I + J * J

    h1, h1p, h1pp = continued_power(J, Jc, Jmin, 2.0 / 3.0)
    h2, h2p, h2pp = continued_power(J, Jc, Jmin, 4.0 / 3.0)

    W = C10 * (h1 * I1 - 3.0) + C01 * (h2 * I2 - 3.0)
    W += 0.5 * kappa * (J - 1.0) * (J - 1.0)

    G = cofactor_2d(F)
    a = 2.0 * (C10 * h1 + C01 * h2)
    q = C10 * h1p * I1
    q += C01 * (2.0 * J * h2 + h2p * I2)
    q += kappa * (J - 1.0)
    P = a * F + q * G

    HF = np.zeros((4, 4))
    dq_dJ_coeff = C10 * h1pp * I1
    dq_dJ_coeff += C01 * (2.0 * h2 + 4.0 * J * h2p + h2pp * I2)
    dq_dJ_coeff += kappa
    dq_dI_coeff = 2.0 * (C10 * h1p + C01 * h2p)
    da_dJ_coeff = 2.0 * (C10 * h1p + C01 * h2p)

    for j in range(4):
        dF = np.zeros((2, 2))
        dF.flat[j] = 1.0

        dJ = np.sum(G * dF)
        FdF = np.sum(F * dF)
        dG = cofactor_2d(dF)

        da = da_dJ_coeff * dJ
        dq = dq_dJ_coeff * dJ + dq_dI_coeff * FdF
        dP = a * dF + da * F + dq * G + q * dG
        HF[:, j] = dP.ravel()

    return W, P, 0.5 * (HF + HF.T), J


def element_energy_gradient_hessian(xe, Xe, C10, C01, kappa, Jc, Jmin):
    F, inv_Dm, A0 = element_kinematics(xe, Xe)
    W, P, HF, J = mooney_rivlin_energy_gradient_hessian(
        F, C10, C01, kappa, Jc, Jmin
    )
    B = element_B(inv_Dm)
    ge = A0 * (B.T @ P.ravel())
    He = A0 * (B.T @ HF @ B)
    return A0 * W, ge, He, J


def projected_hessian(H, floor=1e-10):
    lam, Q = np.linalg.eigh(H)
    lam_mod = np.maximum(np.abs(lam), floor)
    return (Q * lam_mod) @ Q.T


def assemble(x, X, triangles, C10, C01, kappa, Jc, Jmin):
    ndof = 2 * len(X)
    E = 0.0
    g = np.zeros(ndof)
    H = np.zeros((ndof, ndof))
    Js = []

    for tri in triangles:
        Ee, ge, He, J = element_energy_gradient_hessian(
            x[tri], X[tri], C10, C01, kappa, Jc, Jmin
        )
        E += Ee
        Js.append(J)
        idx = np.array([[2 * i, 2 * i + 1] for i in tri]).ravel()
        g[idx] += ge
        H[np.ix_(idx, idx)] += projected_hessian(He)

    return E, g, 0.5 * (H + H.T), np.asarray(Js)


def solve_stage(
    x,
    X,
    triangles,
    fixed,
    C10,
    C01,
    kappa,
    Jc,
    Jmin,
    max_iter=140,
    grad_tol=1e-9,
    verbose=True,
):
    ndof = 2 * len(X)
    free = np.setdiff1d(np.arange(ndof), fixed)
    history = []

    for it in range(max_iter):
        E, g, H, Js = assemble(x, X, triangles, C10, C01, kappa, Jc, Jmin)
        gf = g[free]
        Hf = H[np.ix_(free, free)]
        gn = np.linalg.norm(gf)
        history.append((E, gn, Js.min(), Js.max()))

        if verbose:
            print(
                f"{it:3d} E={E: .8e} |g|={gn: .3e} "
                f"J=[{Js.min(): .5f},{Js.max(): .5f}]"
            )

        if gn < grad_tol:
            return x, history, True

        try:
            p = np.linalg.solve(Hf, -gf)
        except np.linalg.LinAlgError:
            p = -gf

        gd = float(gf @ p)
        if gd >= 0.0:
            p = -gf
            gd = -float(gf @ gf)

        flat = x.ravel().copy()
        alpha = 1.0
        accepted = False
        for _ in range(80):
            tf = flat.copy()
            tf[free] += alpha * p
            xt = tf.reshape(x.shape)
            Et, _, _, _ = assemble(xt, X, triangles, C10, C01, kappa, Jc, Jmin)
            if np.isfinite(Et) and Et <= E + 1.0e-4 * alpha * gd:
                x = xt
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            return x, history, False

    return x, history, False


def run_homotopy(
    X, triangles, x0, anchor_right, C10, C01, kappa, Jc_target, Jmin, verbose=True
):
    fixed = np.array([0, 1, 2 * anchor_right + 1], dtype=int)
    x = x0.copy()
    stages = [0.50, 0.35, 0.25, max(Jc_target, 0.20), Jc_target]
    all_hist = []

    for s, Jc in enumerate(stages):
        if verbose:
            print(f"\n--- stage {s}: Jc={Jc:g}, no inversion barrier ---")
        x, hist, ok = solve_stage(
            x,
            X,
            triangles,
            fixed,
            C10,
            C01,
            kappa,
            Jc,
            Jmin,
            max_iter=180,
            grad_tol=1e-9,
            verbose=verbose,
        )
        all_hist.append((Jc, hist))
        if not ok:
            return x, all_hist, False

    if verbose:
        print(f"\n--- final exact stage: Jc={Jc_target:g} ---")
    x, hist, ok = solve_stage(
        x,
        X,
        triangles,
        fixed,
        C10,
        C01,
        kappa,
        Jc_target,
        Jmin,
        max_iter=240,
        grad_tol=1e-11,
        verbose=verbose,
    )
    all_hist.append((Jc_target, hist))
    return x, all_hist, ok


def _mesh_bounds(*meshes):
    pts = np.vstack(meshes)
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    span = np.maximum(hi - lo, 1.0e-12)
    pad = 0.08 * max(span[0], span[1])
    return lo - pad, hi + pad


def _svg_mesh_lines(x, tris, xoff, yoff, width, height, lo, hi, color):
    span = hi - lo
    sx = width / span[0]
    sy = height / span[1]
    scale = min(sx, sy)
    ox = xoff + 0.5 * (width - scale * span[0])
    oy = yoff + 0.5 * (height - scale * span[1])
    lines = []
    for tri in tris:
        pts = x[np.asarray([tri[0], tri[1], tri[2], tri[0]])]
        coords = []
        for p in pts:
            px = ox + scale * (p[0] - lo[0])
            py = oy + height - scale * (p[1] - lo[1])
            coords.append(f"{px:.3f},{py:.3f}")
        lines.append(
            f'<polyline points="{" ".join(coords)}" '
            f'fill="none" stroke="{color}" stroke-width="0.65"/>'
        )
    return lines


def write_svg_summary(path, x0, x, tris, hist):
    lo, hi = _mesh_bounds(x0, x)
    width = 1200
    height = 410
    panel_w = 360
    panel_h = 300
    top = 55
    lefts = [25, 420, 815]

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:sans-serif;font-size:14px}'
        '.title{font-size:18px;font-weight:600}.axis{stroke:#333;stroke-width:1}'
        '</style>',
        f'<text class="title" x="{lefts[0]}" y="28">initial random deformation</text>',
        f'<text class="title" x="{lefts[1]}" y="28">final</text>',
        f'<text class="title" x="{lefts[2]}" y="28">energy and determinant range</text>',
    ]
    lines += _svg_mesh_lines(x0, tris, lefts[0], top, panel_w, panel_h, lo, hi, "#555")
    lines += _svg_mesh_lines(x, tris, lefts[1], top, panel_w, panel_h, lo, hi, "#0a5")

    chart_x = lefts[2]
    chart_y = top
    chart_w = panel_w
    chart_h = panel_h
    values = []
    jmins = []
    jmaxs = []
    for _, hh in hist:
        for E0, _, jmn, jmx in hh:
            values.append(max(E0, 1.0e-16))
            jmins.append(jmn)
            jmaxs.append(jmx)

    if values:
        n = len(values)
        loge = np.log10(np.asarray(values))
        emin = float(loge.min())
        emax = float(loge.max())
        if emax <= emin:
            emax = emin + 1.0
        jall = np.asarray(jmins + jmaxs)
        jlo = float(jall.min())
        jhi = float(jall.max())
        if jhi <= jlo:
            jhi = jlo + 1.0

        def cx(i):
            return chart_x + (chart_w * i / max(n - 1, 1))

        def cy_log(v):
            return chart_y + chart_h * (1.0 - (np.log10(v) - emin) / (emax - emin))

        def cy_j(v):
            return chart_y + chart_h * (1.0 - (v - jlo) / (jhi - jlo))

        epts = " ".join(f"{cx(i):.3f},{cy_log(v):.3f}" for i, v in enumerate(values))
        jmpts = " ".join(f"{cx(i):.3f},{cy_j(v):.3f}" for i, v in enumerate(jmins))
        jxpts = " ".join(f"{cx(i):.3f},{cy_j(v):.3f}" for i, v in enumerate(jmaxs))
        lines += [
            f'<rect x="{chart_x}" y="{chart_y}" width="{chart_w}" height="{chart_h}" '
            'fill="none" stroke="#333" stroke-width="1"/>',
            f'<polyline points="{epts}" fill="none" stroke="#06c" stroke-width="1.8"/>',
            f'<polyline points="{jmpts}" fill="none" stroke="#b42" stroke-width="1.4" '
            'stroke-dasharray="5 4"/>',
            f'<polyline points="{jxpts}" fill="none" stroke="#666" stroke-width="1.2" '
            'stroke-dasharray="2 4"/>',
            f'<text x="{chart_x}" y="{chart_y + chart_h + 24}">iterations: {n}</text>',
            f'<text x="{chart_x}" y="{chart_y + chart_h + 43}">'
            f'log10(E): [{emin:.2f}, {emax:.2f}]</text>',
            f'<text x="{chart_x + 180}" y="{chart_y + chart_h + 43}">'
            f'J: [{jlo:.3f}, {jhi:.3f}]</text>',
            f'<text x="{chart_x + 265}" y="{chart_y + 19}" fill="#06c">energy</text>',
            f'<text x="{chart_x + 265}" y="{chart_y + 39}" fill="#b42">min J</text>',
            f'<text x="{chart_x + 265}" y="{chart_y + 59}" fill="#666">max J</text>',
        ]

    lines.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_mesh(ax, x, tris, title):
    for tri in tris:
        pts = np.vstack((x[tri], x[tri[0]]))
        ax.plot(pts[:, 0], pts[:, 1], "-", linewidth=0.6)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)


def write_matplotlib_summary(path, x0, x, tris, hist):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    plot_mesh(axs[0], x0, tris, "initial random deformation")
    plot_mesh(axs[1], x, tris, "final")

    stage_ids = []
    energies = []
    jmins = []
    jmaxs = []
    k = 0
    for _, hh in hist:
        for E0, _, jmn, jmx in hh:
            stage_ids.append(k)
            energies.append(max(E0, 1e-16))
            jmins.append(jmn)
            jmaxs.append(jmx)
            k += 1

    axs[2].semilogy(stage_ids, energies, label="energy")
    ax2 = axs[2].twinx()
    ax2.plot(stage_ids, jmins, "--", label="min J")
    ax2.plot(stage_ids, jmaxs, ":", label="max J")
    axs[2].set_title("energy and determinant range")
    axs[2].set_xlabel("global iteration")
    axs[2].set_ylabel("energy")
    ax2.set_ylabel("J")
    axs[2].grid(True)

    l1, lab1 = axs[2].get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    axs[2].legend(l1 + l2, lab1 + lab2, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def random_deformed_initial_state(X, triangles, amplitude, seed):
    rng = np.random.default_rng(seed)
    ux = np.unique(X[:, 0])
    uy = np.unique(X[:, 1])
    hx = np.min(np.diff(ux)) if len(ux) > 1 else 1.0
    hy = np.min(np.diff(uy)) if len(uy) > 1 else 1.0
    h = min(hx, hy)
    x0 = X + amplitude * h * rng.standard_normal(X.shape)

    lower_left = np.argmin(np.sum(X * X, axis=1))
    lower_right = np.argmax(X[:, 0] - 1000.0 * np.abs(X[:, 1]))
    x0[lower_left] = X[lower_left]
    x0[lower_right, 1] = X[lower_right, 1]

    Js = []
    for tri in triangles:
        F, _, _ = element_kinematics(x0[tri], X[tri])
        Js.append(np.linalg.det(F))
    return x0, np.asarray(Js)


def check_material_derivatives(C10, C01, kappa, Jc, Jmin):
    rng = np.random.default_rng(13)
    F = np.array([[1.10, 0.08], [-0.04, 0.93]])
    W, P, HF, _ = mooney_rivlin_energy_gradient_hessian(
        F, C10, C01, kappa, Jc, Jmin
    )

    dF = rng.standard_normal((2, 2))
    dF /= np.linalg.norm(dF)
    eps = 1e-6
    Wp, Pp, _, _ = mooney_rivlin_energy_gradient_hessian(
        F + eps * dF, C10, C01, kappa, Jc, Jmin
    )
    Wm, Pm, _, _ = mooney_rivlin_energy_gradient_hessian(
        F - eps * dF, C10, C01, kappa, Jc, Jmin
    )
    dW_fd = (Wp - Wm) / (2.0 * eps)
    dW = float(np.sum(P * dF))
    dP_fd = ((Pp - Pm) / (2.0 * eps)).ravel()
    dP = HF @ dF.ravel()
    return abs(dW - dW_fd), np.linalg.norm(dP - dP_fd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=8)
    ap.add_argument("--ny", type=int, default=8)
    ap.add_argument("--C10", type=float, default=0.35)
    ap.add_argument("--C01", type=float, default=0.15)
    ap.add_argument("--kappa", type=float, default=500.0)
    ap.add_argument("--Jc", type=float, default=0.2)
    ap.add_argument("--Jmin", type=float, default=-1.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--amplitude", type=float, default=1.0)
    ap.add_argument("--plot", default="invertible_mooney_rivlin_plane_strain_2d.png")
    ap.add_argument("--check-derivatives", action="store_true")
    args = ap.parse_args()

    if args.check_derivatives:
        egrad, ehess = check_material_derivatives(
            args.C10, args.C01, args.kappa, args.Jc, args.Jmin
        )
        print(f"material_gradient_directional_error: {egrad:.6e}")
        print(f"material_hessian_directional_error: {ehess:.6e}")

    X, tris, anchor_right = structured_square_mesh(args.nx, args.ny)
    x0, Js0 = random_deformed_initial_state(X, tris, args.amplitude, args.seed)
    n_inv0 = int(np.count_nonzero(Js0 < 0.0))

    print(f"Initial random deformation: seed={args.seed}, amplitude={args.amplitude}")
    print(f"Initial J range: [{Js0.min():.6f}, {Js0.max():.6f}]")
    print(f"Initially inverted elements: {n_inv0}/{len(Js0)}")
    print(
        "Material: "
        f"C10={args.C10:g}, C01={args.C01:g}, kappa={args.kappa:g}, "
        f"Jc={args.Jc:g}, Jmin={args.Jmin:g}"
    )

    x, hist, ok = run_homotopy(
        X,
        tris,
        x0,
        anchor_right,
        args.C10,
        args.C01,
        args.kappa,
        args.Jc,
        args.Jmin,
        verbose=True,
    )

    E, g, _, Js = assemble(x, X, tris, args.C10, args.C01, args.kappa, args.Jc, args.Jmin)
    fixed = np.array([0, 1, 2 * anchor_right + 1], dtype=int)
    free = np.setdiff1d(np.arange(2 * len(X)), fixed)

    print("\nConverged:", ok)
    print(f"Mesh: {len(X)} vertices, {len(tris)} triangles")
    print("Final J range:", (Js.min(), Js.max()))
    print("Final energy:", E)
    print("Final free-gradient norm:", np.linalg.norm(g[free]))
    print("||x-X||_F:", np.linalg.norm(x - X))

    if args.plot:
        if args.plot.lower().endswith(".svg"):
            write_svg_summary(args.plot, x0, x, tris, hist)
        else:
            write_matplotlib_summary(args.plot, x0, x, tris, hist)
        print("Saved plot:", args.plot)

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
