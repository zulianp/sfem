"""Emit the element-specific parts of the compressed quadrature-point partial assembly.

    python make_qp_kernels.py ELEMENT GENERATED_INLINE_HEADER OUT_DIR

Writes into OUT_DIR:

  qp_element.hpp  the reference-element stages of the application, as straight-line code:
                  `stage_in` forms the R gradient vectors the stored tensors contract with
                  (the point gradients on HEX8, the vertex gradients on TET10 and TET4),
                  `stage_out` is its transpose, and `contract` applies one packed tangent.
                  Also the constants the set-up needs: the stored tensors' pairing (a, b),
                  their reference weights X_j, and the moment factors per point.
  qp_setup.hpp    the set-up kernel.  It is the generated P0 tangent kernel with its
                  quadrature sum replaced: each point's weighted tangent is written to a
                  block scratch (points) or accumulated into the P2 moments (TET10), and the
                  block is then handed to the store's encoder.  Everything that evaluates the
                  material -- gather, gradients, pullback -- is the generated code unchanged.
  qp_point.hpp    the same kernel's pointwise body alone, the weighted point tangent from the
                  reference gradient and the affine geometry, for the GPU set-up (cuda/).
  qp_device_tables.cuh  the element's tables in constant memory, for the GPU kernels.

The stages and the contraction are __host__ __device__ under nvcc (QP_HD), so the GPU kernels run
the same straight-line code with one lane.

This is the prototype the spike optimises by hand; what eventually ships is the emitter change
in the code generator (docs/compressed_partial_assembly, Section "The set-up and application
kernels").
"""

import itertools
import math
import re
import sys
from fractions import Fraction as Fr

D = 3
NC = 45

# __host__ __device__ under nvcc: the GPU kernels call the stages with one lane.
QPHD = """#ifndef QP_HD
#if defined(__CUDACC__)
#define QP_HD __host__ __device__
#else
#define QP_HD
#endif
#endif"""


def slot(r, c):
    """Packed index of entry (r, c) of the symmetric 9x9 tangent; r = i*3+n, c = k*3+m."""
    if r > c:
        r, c = c, r
    return r * 9 - r * (r - 1) // 2 + (c - r)


# ------------------------------------------------------------------------------- elements


def lam(p):
    x, y, z = p
    return [1 - x - y - z, x, y, z]


def tet4():
    grads = [(-1, -1, -1), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    D_ = [[[Fr(g[m]) for g in grads] for m in range(3)]]  # one "point": constant gradients
    return dict(N=4, R=1, D=D_, pairs=[(0, 0)], X=[Fr(1, 6)], nq=1, kvol=Fr(1, 6),
                mult=[1], moment_factor=None, kind="points")


def tet10_basis_grad(p):
    """Reference gradients of SFEM's TET10 basis (vertices, then edges (0,1) (1,2) (0,2) (0,3)
    (1,3) (2,3)) at barycentric point p, as exact fractions when p is rational."""
    l = lam(p)
    gl = [(-1, -1, -1), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    out = []
    for a in range(4):
        out.append([(4 * l[a] - 1) * gl[a][m] for m in range(3)])
    for a, b in [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]:
        out.append([4 * (l[b] * gl[a][m] + l[a] * gl[b][m]) for m in range(3)])
    return out  # out[j][m]


KEAST = None


def keast11():
    # The rule's order matches the generated reference tables (fem/reference.py, order 4).
    a, b = 0.7857142857142857, 0.07142857142857142
    c, d = 0.3994035761667992, 0.1005964238332008
    pts = [(0.25, 0.25, 0.25), (b, b, b), (a, b, b), (b, a, b), (b, b, a),
           (c, c, d), (c, d, c), (d, c, c), (c, d, d), (d, c, d), (d, d, c)]
    w = [-0.013155555555555556] + [0.007622222222222222] * 4 + [0.024888888888888887] * 6
    return pts, w


def tet10_q4():
    """TET10 on the 4-point degree-2 rule: what the affine residual and exact apply integrate with,
    so the Jacobian Newton needs.  4 < dim P2 = 10, so the point values are the minimal store."""
    a, b = 0.5854101966249685, 0.1381966011250105
    pts = [(b, b, b), (a, b, b), (b, a, b), (b, b, a)]
    D_ = []
    for p in pts:
        g = tet10_basis_grad(p)
        D_.append([[g[j][m] for j in range(10)] for m in range(3)])
    return dict(N=10, R=4, D=D_, pairs=[(q, q) for q in range(4)], X=[1.0 / 24.0] * 4, nq=4,
                kvol=1.0 / 6.0, mult=[1] * 4, moment_factor=None, kind="points")


def tet10_p11():
    """TET10 on the 11-point rule stored as point values: the control for the moment store,
    which must reproduce it to round-off."""
    pts, w = keast11()
    D_ = []
    for p in pts:
        g = tet10_basis_grad(p)
        D_.append([[g[j][m] for j in range(10)] for m in range(3)])
    return dict(N=10, R=11, D=D_, pairs=[(q, q) for q in range(11)], X=list(w), nq=11, kvol=sum(w),
                mult=[1] * 11, moment_factor=None, kind="points")


def tet10():
    verts = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    D_ = []
    for v in verts:
        g = tet10_basis_grad(tuple(Fr(x) for x in v))
        D_.append([[g[j][m] for j in range(10)] for m in range(3)])
    pairs = [(a, b) for a in range(4) for b in range(a, 4)]
    pts, w = keast11()
    # Moment factor of point q for pair (a, b): lambda_a lambda_b at the point.
    factor = [[lam(p)[a] * lam(p)[b] for (a, b) in pairs] for p in pts]
    X = [sum(w[q] * factor[q][j] for q in range(len(pts))) for j in range(len(pairs))]
    mult = [1 if a == b else 2 for (a, b) in pairs]
    return dict(N=10, R=4, D=D_, pairs=pairs, X=X, nq=11, kvol=sum(w), mult=mult,
                moment_factor=factor, kind="moments")


def hex8():
    g = [(1 - 1 / math.sqrt(3)) / 2, (1 + 1 / math.sqrt(3)) / 2]
    pts = [(g[i], g[j], g[k]) for k in (0, 1) for j in (0, 1) for i in (0, 1)]

    def l1(i, t):
        return t if i else 1 - t

    def dl1(i):
        return 1.0 if i else -1.0

    D_ = []
    for (x, y, z) in pts:
        rows = [[], [], []]
        for kk, jj, ii in itertools.product((0, 1), repeat=3):
            rows[0].append(dl1(ii) * l1(jj, y) * l1(kk, z))
            rows[1].append(l1(ii, x) * dl1(jj) * l1(kk, z))
            rows[2].append(l1(ii, x) * l1(jj, y) * dl1(kk))
        D_.append(rows)
    pairs = [(q, q) for q in range(8)]
    return dict(N=8, R=8, D=D_, pairs=pairs, X=[0.125] * 8, nq=8, kvol=1.0, mult=[1] * 8,
                moment_factor=None, kind="points")


# TET10 integrates with the 11-point rule its residual, matrix-free and BSR operators use, and
# stores the P2 moments; TET10Q4 is the 4-point rule of the affine apply, stored as point values.
def gauss4():
    a = math.sqrt(3 / 7 - 2 / 7 * math.sqrt(6 / 5))
    b = math.sqrt(3 / 7 + 2 / 7 * math.sqrt(6 / 5))
    wa, wb = (18 + math.sqrt(30)) / 36, (18 - math.sqrt(30)) / 36
    pts = [(1 - b) / 2, (1 - a) / 2, (1 + a) / 2, (1 + b) / 2]
    return pts, [wb / 2, wa / 2, wa / 2, wb / 2]


def p2_line(x):
    """Lagrange P2 on nodes 0, 1/2, 1 (the lexicographic order): values and derivatives."""
    return ([2 * (x - 0.5) * (x - 1), -4 * x * (x - 1), 2 * x * (x - 0.5)], [4 * x - 3, -8 * x + 4, 4 * x - 1])


def proteus_hex27():
    """PROTEUS_HEX27 on the 4^3 Gauss rule, which its isoparametric residual, matrix-free apply
    and BSR assembly integrate with.  Point values are the minimal store (the product spaces of
    the gradient functions exceed 64)."""
    g, w = gauss4()
    B = [p2_line(x)[0] for x in g]
    Dd = [p2_line(x)[1] for x in g]
    D_ = []
    X = []
    for qz in range(4):
        for qy in range(4):
            for qx in range(4):
                rows = [[], [], []]
                for sz in range(3):
                    for sy in range(3):
                        for sx in range(3):
                            rows[0].append(Dd[qx][sx] * B[qy][sy] * B[qz][sz])
                            rows[1].append(B[qx][sx] * Dd[qy][sy] * B[qz][sz])
                            rows[2].append(B[qx][sx] * B[qy][sy] * Dd[qz][sz])
                D_.append(rows)
                X.append(w[qx] * w[qy] * w[qz])
    return dict(N=27, R=64, D=D_, pairs=[(q, q) for q in range(64)], X=X, nq=64, kvol=1.0, mult=[1] * 64,
                moment_factor=None, kind="points", B1=B, D1=Dd)


def tensor_stages(n1, q1, B, Dd):
    """Sum-factorised stages for a tensor-product element with n1 nodes and q1 points per direction:
    the reference gradient at every point (stage_in) and its transpose (stage_out).  Node
    s = sx + n1 (sy + n1 sz), point q = qx + q1 (qy + q1 qz).

    Lane form: one lane loop per 1D sweep, the intermediates in small arrays (L1) rather than live
    temporaries, so no sweep holds more than a few values per lane.  The sweeps run in the
    contraction's precision on the increment relative to the element's first node: every sweep
    contains a derivative factor, which annihilates the constant, and the rounding is then relative
    to the variation of the increment over the element."""
    N, NQ = n1 ** 3, q1 ** 3
    lit_ = lambda c: repr(float(c))
    node = lambda sx, sy, sz: sx + n1 * (sy + n1 * sz)
    L = []
    L.append("  template <typename h_t, typename t_out, int VS>")
    L.append(f"  static QP_HD SFEM_INLINE void stage_in(const int ne, const h_t h[3][{N}][VS], t_out P[{NQ}][9][VS]) {{")
    L.append("    using t = t_out;")
    L.append(f"    t u[{N}][VS], v[{q1}][{n1}][{n1}][VS], d[{q1}][{n1}][{n1}][VS];")
    L.append(f"    t vv[{q1}][{q1}][{n1}][VS], dv[{q1}][{q1}][{n1}][VS], vd[{q1}][{q1}][{n1}][VS];")
    L.append("    for (int k = 0; k < 3; ++k) {")
    L.append("#pragma omp simd")
    L.append("      for (int lane = 0; lane < ne; ++lane) {")
    for s_ in range(N):
        L.append(f"        u[{s_}][lane] = t(h[k][{s_}][lane] - h[k][0][lane]);")
    L.append("      }")
    L.append("#pragma omp simd")
    L.append("      for (int lane = 0; lane < ne; ++lane) {")
    for sz in range(n1):
        for sy in range(n1):
            for qx in range(q1):
                us = [f"u[{node(sx, sy, sz)}][lane]" for sx in range(n1)]
                L.append(f"        v[{qx}][{sy}][{sz}][lane] = " + " + ".join(f"t({lit_(B[qx][sx])})*{us[sx]}" for sx in range(n1)) + ";")
                L.append(f"        d[{qx}][{sy}][{sz}][lane] = " + " + ".join(f"t({lit_(Dd[qx][sx])})*{us[sx]}" for sx in range(n1)) + ";")
    L.append("      }")
    L.append("#pragma omp simd")
    L.append("      for (int lane = 0; lane < ne; ++lane) {")
    for sz in range(n1):
        for qx in range(q1):
            for qy in range(q1):
                L.append(f"        vv[{qx}][{qy}][{sz}][lane] = " + " + ".join(f"t({lit_(B[qy][sy])})*v[{qx}][{sy}][{sz}][lane]" for sy in range(n1)) + ";")
                L.append(f"        dv[{qx}][{qy}][{sz}][lane] = " + " + ".join(f"t({lit_(B[qy][sy])})*d[{qx}][{sy}][{sz}][lane]" for sy in range(n1)) + ";")
                L.append(f"        vd[{qx}][{qy}][{sz}][lane] = " + " + ".join(f"t({lit_(Dd[qy][sy])})*v[{qx}][{sy}][{sz}][lane]" for sy in range(n1)) + ";")
    L.append("      }")
    L.append("#pragma omp simd")
    L.append("      for (int lane = 0; lane < ne; ++lane) {")
    for qz in range(q1):
        for qy in range(q1):
            for qx in range(q1):
                q = qx + q1 * (qy + q1 * qz)
                gx = " + ".join(f"t({lit_(B[qz][sz])})*dv[{qx}][{qy}][{sz}][lane]" for sz in range(n1))
                gy = " + ".join(f"t({lit_(B[qz][sz])})*vd[{qx}][{qy}][{sz}][lane]" for sz in range(n1))
                gz = " + ".join(f"t({lit_(Dd[qz][sz])})*vv[{qx}][{qy}][{sz}][lane]" for sz in range(n1))
                L.append(f"        P[{q}][k * 3 + 0][lane] = {gx};")
                L.append(f"        P[{q}][k * 3 + 1][lane] = {gy};")
                L.append(f"        P[{q}][k * 3 + 2][lane] = {gz};")
    L.append("      }")
    L.append("    }")
    L.append("  }")
    L.append("")
    L.append("  template <typename y_t, typename out_t, int VS>")
    L.append(f"  static QP_HD SFEM_INLINE void stage_out(const int ne, const y_t Y[{NQ}][9][VS], out_t out[3][{N}][VS]) {{")
    L.append("    using t = y_t;")
    L.append(f"    t ax[{q1}][{q1}][{n1}][VS], ay[{q1}][{q1}][{n1}][VS], az[{q1}][{q1}][{n1}][VS];")
    L.append(f"    t bx[{q1}][{n1}][{n1}][VS], c[{q1}][{n1}][{n1}][VS];")
    L.append("    for (int i = 0; i < 3; ++i) {")
    Yq = lambda qx, qy, qz, n: f"Y[{qx + q1 * (qy + q1 * qz)}][i * 3 + {n}][lane]"
    L.append("#pragma omp simd")
    L.append("      for (int lane = 0; lane < ne; ++lane) {")
    for qx in range(q1):
        for qy in range(q1):
            for sz in range(n1):
                L.append(f"        ax[{qx}][{qy}][{sz}][lane] = " + " + ".join(f"t({lit_(B[qz][sz])})*{Yq(qx, qy, qz, 0)}" for qz in range(q1)) + ";")
                L.append(f"        ay[{qx}][{qy}][{sz}][lane] = " + " + ".join(f"t({lit_(B[qz][sz])})*{Yq(qx, qy, qz, 1)}" for qz in range(q1)) + ";")
                L.append(f"        az[{qx}][{qy}][{sz}][lane] = " + " + ".join(f"t({lit_(Dd[qz][sz])})*{Yq(qx, qy, qz, 2)}" for qz in range(q1)) + ";")
    L.append("      }")
    L.append("#pragma omp simd")
    L.append("      for (int lane = 0; lane < ne; ++lane) {")
    for qx in range(q1):
        for sy in range(n1):
            for sz in range(n1):
                L.append(f"        bx[{qx}][{sy}][{sz}][lane] = " + " + ".join(f"t({lit_(B[qy][sy])})*ax[{qx}][{qy}][{sz}][lane]" for qy in range(q1)) + ";")
                L.append(f"        c[{qx}][{sy}][{sz}][lane] = " + " + ".join(f"t({lit_(Dd[qy][sy])})*ay[{qx}][{qy}][{sz}][lane] + t({lit_(B[qy][sy])})*az[{qx}][{qy}][{sz}][lane]" for qy in range(q1)) + ";")
    L.append("      }")
    L.append("#pragma omp simd")
    L.append("      for (int lane = 0; lane < ne; ++lane) {")
    for sz in range(n1):
        for sy in range(n1):
            for sx in range(n1):
                expr = " + ".join(f"t({lit_(Dd[qx][sx])})*bx[{qx}][{sy}][{sz}][lane] + t({lit_(B[qx][sx])})*c[{qx}][{sy}][{sz}][lane]" for qx in range(q1))
                L.append(f"        out[i][{node(sx, sy, sz)}][lane] = out_t({expr});")
    L.append("      }")
    L.append("    }")
    L.append("  }")
    return L


ELEMENTS = {"PROTEUS_HEX27": proteus_hex27, "TET4": tet4, "TET10": tet10, "TET10P11": tet10_p11, "TET10Q4": tet10_q4, "HEX8": hex8}


# --------------------------------------------------------------------------- emission


def lit(c):
    c = float(c)
    if c == int(c) and abs(c) < 1e6:
        return f"{int(c)}"
    return repr(c)


def linear_combination(terms):
    """terms: list of (coefficient, symbol); emits a sum skipping zeros and folding +-1."""
    parts = []
    for c, s in terms:
        c = float(c)
        if abs(c) < 1e-15:
            continue
        if c == 1.0:
            parts.append(f"+ {s}")
        elif c == -1.0:
            parts.append(f"- {s}")
        elif c < 0:
            parts.append(f"- t({lit(-c)})*{s}")
        else:
            parts.append(f"+ t({lit(c)})*{s}")
    if not parts:
        return "t(0)"
    text = " ".join(parts)
    return text[2:] if text.startswith("+ ") else "-" + text[2:]


def hex8_sum_factorised_stages():
    """HEX8 on 2x2x2 Gauss points by one-dimensional sweeps.  With two linear shape functions
    the derivative along an axis is the difference of the two nodal values, constant along that
    axis, and interpolation to the two points is u0 + g_q (u1 - u0); both facts are written out
    so the elimination is in the code rather than left to the compiler.  Node s = sx + 2 sy + 4 sz,
    point q = qx + 2 qy + 4 qz (lexicographic, as the generated kernels number them)."""
    g = [(1 - 1 / math.sqrt(3)) / 2, (1 + 1 / math.sqrt(3)) / 2]
    G0, G1 = repr(g[0]), repr(g[1])
    L = []
    L.append("  template <typename h_t, typename t_out, int VS>")
    L.append("  static QP_HD SFEM_INLINE void stage_in(const int ne, const h_t h[3][8][VS], t_out P[8][9][VS]) {")
    L.append("    using t = h_t;")
    L.append(f"    const t g0 = t({G0}), g1 = t({G1});")
    L.append("#pragma omp simd")
    L.append("    for (int lane = 0; lane < ne; ++lane) {")
    for k in range(3):
        u = lambda sx, sy, sz: f"h[{k}][{sx + 2 * sy + 4 * sz}][lane]"
        p = f"c{k}_"
        # x sweep: difference along x (the x-derivative, constant in x) and values at qx
        for sy in (0, 1):
            for sz in (0, 1):
                L.append(f"      const t {p}dx{sy}{sz} = {u(1, sy, sz)} - {u(0, sy, sz)};")
                for qx, gq in ((0, "g0"), (1, "g1")):
                    L.append(f"      const t {p}vx{qx}{sy}{sz} = {u(0, sy, sz)} + {gq}*{p}dx{sy}{sz};")
        # y sweep
        for sz in (0, 1):
            L.append(f"      const t {p}ddx{sz} = {p}dx1{sz} - {p}dx0{sz};")
            for qy, gq in ((0, "g0"), (1, "g1")):
                L.append(f"      const t {p}dxy{qy}{sz} = {p}dx0{sz} + {gq}*{p}ddx{sz};")
            for qx in (0, 1):
                L.append(f"      const t {p}dy{qx}{sz} = {p}vx{qx}1{sz} - {p}vx{qx}0{sz};")
                for qy, gq in ((0, "g0"), (1, "g1")):
                    L.append(f"      const t {p}vxy{qx}{qy}{sz} = {p}vx{qx}0{sz} + {gq}*{p}dy{qx}{sz};")
        # z sweep, written straight into the gradient at every point
        for qz, gq in ((0, "g0"), (1, "g1")):
            for qy in (0, 1):
                for qx in (0, 1):
                    q = qx + 2 * qy + 4 * qz
                    gx = f"{p}dxy{qy}0 + {gq}*({p}dxy{qy}1 - {p}dxy{qy}0)"
                    gy = f"{p}dy{qx}0 + {gq}*({p}dy{qx}1 - {p}dy{qx}0)"
                    gz = f"{p}vxy{qx}{qy}1 - {p}vxy{qx}{qy}0"
                    L.append(f"      P[{q}][{k * 3 + 0}][lane] = t_out({gx});")
                    L.append(f"      P[{q}][{k * 3 + 1}][lane] = t_out({gy});")
                    L.append(f"      P[{q}][{k * 3 + 2}][lane] = t_out({gz});")
    L.append("    }")
    L.append("  }")
    # Transpose: out[i][s] = sum_q (dphi_s/dx Y_qx + dphi_s/dy Y_qy + dphi_s/dz Y_qz)
    L.append("")
    L.append("  template <typename y_t, typename out_t, int VS>")
    L.append("  static QP_HD SFEM_INLINE void stage_out(const int ne, const y_t Y[8][9][VS], out_t out[3][8][VS]) {")
    L.append("    using t = out_t;")
    L.append(f"    const t g0 = t({G0}), g1 = t({G1});")
    L.append("#pragma omp simd")
    L.append("    for (int lane = 0; lane < ne; ++lane) {")

    def ti(a0, a1, name):
        # transpose of interpolation: n1 = g0 a0 + g1 a1, n0 = (a0 + a1) - n1
        return [f"      const t {name}1 = g0*{a0} + g1*{a1};",
                f"      const t {name}0 = ({a0} + {a1}) - {name}1;"]

    for i in range(3):
        p = f"o{i}_"
        Yv = lambda qx, qy, qz, n: f"t(Y[{qx + 2 * qy + 4 * qz}][{i * 3 + n}][lane])"
        acc = {s_: [] for s_ in range(8)}
        # x-derivative part: sum over qx, transpose-interpolate z then y, then +-1 along x
        for qy in (0, 1):
            for qz in (0, 1):
                L.append(f"      const t {p}ax{qy}{qz} = {Yv(0, qy, qz, 0)} + {Yv(1, qy, qz, 0)};")
        for qy in (0, 1):
            L += ti(f"{p}ax{qy}0", f"{p}ax{qy}1", f"{p}bx{qy}_")
        for sz in (0, 1):
            L += ti(f"{p}bx0_{sz}", f"{p}bx1_{sz}", f"{p}cx{sz}_")
        for sy in (0, 1):
            for sz in (0, 1):
                c = f"{p}cx{sz}_{sy}"
                acc[0 + 2 * sy + 4 * sz].append(f"- {c}")
                acc[1 + 2 * sy + 4 * sz].append(f"+ {c}")
        # y-derivative part: sum over qy, transpose z then x, then +-1 along y
        for qx in (0, 1):
            for qz in (0, 1):
                L.append(f"      const t {p}ay{qx}{qz} = {Yv(qx, 0, qz, 1)} + {Yv(qx, 1, qz, 1)};")
        for qx in (0, 1):
            L += ti(f"{p}ay{qx}0", f"{p}ay{qx}1", f"{p}by{qx}_")
        for sz in (0, 1):
            L += ti(f"{p}by0_{sz}", f"{p}by1_{sz}", f"{p}cy{sz}_")
        for sx in (0, 1):
            for sz in (0, 1):
                c = f"{p}cy{sz}_{sx}"
                acc[sx + 0 + 4 * sz].append(f"- {c}")
                acc[sx + 2 + 4 * sz].append(f"+ {c}")
        # z-derivative part: sum over qz, transpose y then x, then +-1 along z
        for qx in (0, 1):
            for qy in (0, 1):
                L.append(f"      const t {p}az{qx}{qy} = {Yv(qx, qy, 0, 2)} + {Yv(qx, qy, 1, 2)};")
        for qx in (0, 1):
            L += ti(f"{p}az{qx}0", f"{p}az{qx}1", f"{p}bz{qx}_")
        for sy in (0, 1):
            L += ti(f"{p}bz0_{sy}", f"{p}bz1_{sy}", f"{p}cz{sy}_")
        for sx in (0, 1):
            for sy in (0, 1):
                c = f"{p}cz{sy}_{sx}"
                acc[sx + 2 * sy + 0].append(f"- {c}")
                acc[sx + 2 * sy + 4].append(f"+ {c}")
        for s_ in range(8):
            expr = " ".join(acc[s_])
            expr = expr[2:] if expr.startswith("+ ") else "-" + expr[2:]
            L.append(f"      out[{i}][{s_}][lane] = {expr};")
    L.append("    }")
    L.append("  }")
    return L


def setup_weights(e):
    if e["kind"] == "moments":
        return keast11()[1]
    return e["X"]


def setup_stage(e):
    """`setup_in`: the reference gradient at the set-up's points, G[q][k*3+m].  The stored points
    are the set-up's own except for the moments (TET10), which integrate over the 11-point rule."""
    N = e["N"]
    L = ["", "  // The reference gradient at the set-up's quadrature points: G[q][k*3+m].",
         "  template <typename h_t, typename t_out, int VS>",
         f"  static QP_HD SFEM_INLINE void setup_in(const int ne, const h_t h[3][{N}][VS], t_out G[NQ][9][VS]) {{"]
    if e["kind"] != "moments":
        L += ["    stage_in<h_t, t_out, VS>(ne, h, G);", "  }"]
        return L
    pts = keast11()[0]
    L += ["    using t = h_t;", "#pragma omp simd", "    for (int lane = 0; lane < ne; ++lane) {"]
    for k in range(3):
        for j in range(N):
            L.append(f"      const t h{k}_{j} = h[{k}][{j}][lane];")
    for q, p in enumerate(pts):
        g = tet10_basis_grad(p)
        for k in range(3):
            for m in range(3):
                expr = linear_combination([(g[j][m], f"h{k}_{j}") for j in range(N)])
                L.append(f"      G[{q}][{k * 3 + m}][lane] = t_out({expr});")
    L += ["    }", "  }"]
    return L


def emit_device_tables(name, e):
    """The element's tables in constant memory, for the GPU kernels (one translation unit)."""
    N, R, D_ = e["N"], e["R"], e["D"]
    NV, NQ = len(e["pairs"]), e["nq"]
    NP = N * (N - 1) // 2
    pairs = [(a, b) for a in range(N) for b in range(a + 1, N)]
    flat = lambda xs: ", ".join(xs)
    L = [f"// Emitted by make_qp_kernels.py for {name}; do not edit.", "#pragma once",
         "// The element's tables for the GPU kernels: reads uniform across a warp come from the constant cache.",
         "namespace qp {", "namespace dev {"]
    L.append(f"static __constant__ double DT[{R * 3 * N}] = {{{flat(lit(D_[r][m][j]) for r in range(R) for m in range(3) for j in range(N))}}};")
    L.append(f"static __constant__ int PA[{NV}] = {{{flat(str(a) for a, _ in e['pairs'])}}};")
    L.append(f"static __constant__ int PB[{NV}] = {{{flat(str(b) for _, b in e['pairs'])}}};")
    L.append(f"static __constant__ double X[{NV}] = {{{flat(lit(x) for x in e['X'])}}};")
    L.append(f"static __constant__ double MULT[{NV}] = {{{flat(lit(x) for x in e['mult'])}}};")
    L.append(f"static __constant__ double QW[{NQ}] = {{{flat(lit(w) for w in setup_weights(e))}}};")
    if e["moment_factor"] is not None:
        L.append(f"static __constant__ double FACTOR[{NQ * NV}] = {{{flat(lit(v) for row in e['moment_factor'] for v in row)}}};")
    else:
        L.append("static __constant__ double FACTOR[1] = {0};  // points: no moments, never read")
    # The reference gradients at the set-up's points, SG[q][m][j]: the stored points' own (DT) except
    # for the moments, which integrate over the 11-point rule.
    if e["kind"] == "moments":
        sg = [tet10_basis_grad(p) for p in keast11()[0]]
        L.append("constexpr bool SG_IS_DT = false;")
        L.append(f"static __constant__ double SG[{NQ * 3 * N}] = {{{flat(lit(sg[q][j][m]) for q in range(NQ) for m in range(3) for j in range(N))}}};")
    else:
        L.append("constexpr bool SG_IS_DT = true;")
        L.append("static __constant__ double SG[1] = {0};  // the set-up's points are DT's")
    L.append(f"static __constant__ int SLOT[81] = {{{flat(str(slot(r, c)) for r in range(9) for c in range(9))}}};")
    # Tensor-product elements: the one-dimensional values and derivatives at the points, B1[q][s] and
    # D1[q][s], from which the point-parallel kernels form the reference gradients.
    if "B1" in e:
        q1, n1 = len(e["B1"]), len(e["B1"][0])
        L.append(f"constexpr int Q1 = {q1}, S1 = {n1};")
        L.append(f"static __constant__ double B1[{q1 * n1}] = {{{flat(lit(v) for row in e['B1'] for v in row)}}};")
        L.append(f"static __constant__ double D1[{q1 * n1}] = {{{flat(lit(v) for row in e['D1'] for v in row)}}};")
    else:
        L.append("constexpr int Q1 = 1, S1 = 1;  // not a tensor-product element: never read")
        L.append("static __constant__ double B1[1] = {0};")
        L.append("static __constant__ double D1[1] = {0};")
    L.append(f"static __constant__ int PAIR_P[{NP}] = {{{flat(str(a) for a, _ in pairs)}}};")
    L.append(f"static __constant__ int PAIR_Q[{NP}] = {{{flat(str(b) for _, b in pairs)}}};")
    L += ["}  // namespace dev", "}  // namespace qp"]
    return "\n".join(L) + "\n"


def emit_element(name, e):
    N, R, D_ = e["N"], e["R"], e["D"]
    NV = len(e["pairs"])
    L = []
    L.append(f"// Emitted by make_qp_kernels.py for {name}; do not edit.")
    L.append("#pragma once")
    L += QPHD.split("\n")
    L.append("namespace qp {")
    L.append(f"struct Element{name} {{")
    L.append(f'  static constexpr const char *name = "{name}";')
    L.append(f"  static constexpr int N = {N};      // nodes")
    L.append(f"  static constexpr int R = {R};      // gradient vectors the store contracts with")
    L.append(f"  static constexpr int NV = {NV};     // stored tensors per element")
    L.append(f"  static constexpr int NQ = {e['nq']};     // quadrature points of the set-up")
    L.append(f'  static constexpr bool moments = {"true" if e["kind"] == "moments" else "false"};')
    L.append(f"  static constexpr double KVOL = {lit(e['kvol'])};")
    # The reference derivative tables the stages fold in: DT[r][m][j] = dphi_j/dxi_m at gradient
    # point r.  Read only by the element-matrix set-up, which builds K_e from the stored tensors.
    rows = ", ".join("{" + ", ".join("{" + ", ".join(lit(D_[r][m][j]) for j in range(N)) + "}" for m in range(3)) + "}"
                     for r in range(R))
    L.append(f"  static constexpr double DT[R][3][N] = {{{rows}}};")
    pa = ", ".join(str(a) for a, _ in e["pairs"])
    pb = ", ".join(str(b) for _, b in e["pairs"])
    L.append(f"  static constexpr int PA[NV] = {{{pa}}};")
    L.append(f"  static constexpr int PB[NV] = {{{pb}}};")
    L.append(f"  static constexpr double X[NV] = {{{', '.join(lit(x) for x in e['X'])}}};")
    L.append(f"  static constexpr double MULT[NV] = {{{', '.join(lit(x) for x in e['mult'])}}};")
    if e["moment_factor"] is not None:
        rows = ", ".join("{" + ", ".join(lit(v) for v in row) + "}" for row in e["moment_factor"])
        L.append(f"  static constexpr double FACTOR[NQ][NV] = {{{rows}}};")
    # The set-up's quadrature weights: the stored points' own (X) or, for the moments, the rule's.
    L.append(f"  static constexpr double QW[NQ] = {{{', '.join(lit(w) for w in setup_weights(e))}}};")
    if name in ("HEX8", "PROTEUS_HEX27"):
        L.append("")
        L += hex8_sum_factorised_stages() if name == "HEX8" else tensor_stages(3, 4, e["B1"], e["D1"])
        L += setup_stage(e)
        L.append("};")
        L.append("")
        L.append(f"using Element = Element{name};")
        L.append("}  // namespace qp")
        return "\n".join(L) + "\n"
    # stage_in: P[r][k*3+m] = sum_j D[r][m][j] h[k][j]
    L.append("")
    L.append("  // P[r][k*3+m] = sum_j dphi_j/dxi_m (at r) h[k][j]: in the vector's precision, then")
    L.append("  // rounded to the arithmetic type, so differences of nodal values are formed first.")
    L.append("  template <typename h_t, typename t_out, int VS>")
    L.append(f"  static QP_HD SFEM_INLINE void stage_in(const int ne, const h_t h[3][{N}][VS], t_out P[{R}][9][VS]) {{")
    L.append("    using t = h_t;")
    L.append("#pragma omp simd")
    L.append("    for (int lane = 0; lane < ne; ++lane) {")
    for k in range(3):
        for j in range(N):
            L.append(f"      const t h{k}_{j} = h[{k}][{j}][lane];")
    for r in range(R):
        for k in range(3):
            for m in range(3):
                expr = linear_combination([(D_[r][m][j], f"h{k}_{j}") for j in range(N)])
                L.append(f"      P[{r}][{k * 3 + m}][lane] = t_out({expr});")
    L.append("    }")
    L.append("  }")
    # stage_out: out[i][p] += sum_r sum_n D[r][n][p] Y[r][i*3+n]
    L.append("")
    L.append("  // out[i][p] = sum_r sum_n dphi_p/dxi_n (at r) Y[r][i*3+n], accumulated in out_t.")
    L.append("  template <typename y_t, typename out_t, int VS>")
    L.append(f"  static QP_HD SFEM_INLINE void stage_out(const int ne, const y_t Y[{R}][9][VS], out_t out[3][{N}][VS]) {{")
    L.append("    using t = out_t;")
    L.append("#pragma omp simd")
    L.append("    for (int lane = 0; lane < ne; ++lane) {")
    for r in range(R):
        for c in range(9):
            L.append(f"      const t y{r}_{c} = t(Y[{r}][{c}][lane]);")
    for i in range(3):
        for p in range(N):
            expr = linear_combination([(D_[r][n][p], f"y{r}_{i * 3 + n}") for r in range(R) for n in range(3)])
            L.append(f"      out[{i}][{p}][lane] = {expr};")
    L.append("    }")
    L.append("  }")
    L += setup_stage(e)
    L.append("};")
    L.append("")
    L.append(f"using Element = Element{name};")
    L.append("}  // namespace qp")
    return "\n".join(L) + "\n"


def emit_contract():
    L = ["// Emitted by make_qp_kernels.py; do not edit.", "#pragma once"] + QPHD.split("\n") + ["namespace qp {",
         "// y[r] += sum_c S[slot(r,c)] g[c] for the packed symmetric 9x9 tangent: 81 multiply-adds.",
         "template <typename t>",
         "static QP_HD SFEM_INLINE void contract(const t *const S, const t *const g, t *const y) {"]
    for r in range(9):
        terms = " + ".join(f"S[{slot(r, c)}]*g[{c}]" for c in range(9))
        L.append(f"  y[{r}] += {terms};")
    L.append("}")
    L.append("")
    L.append("// The same for a block of lanes, lane loop innermost: Y[r] += sum_c S[slot(r,c)] G[c].")
    L.append("template <typename t, int VS>")
    L.append("static QP_HD SFEM_INLINE void contract_lanes(const int ne, const t S[45][VS], const t G[9][VS], t Y[9][VS]) {")
    for r in range(9):
        terms = " + ".join(f"S[{slot(r, c)}][lane]*G[{c}][lane]" for c in range(9))
        L.append("#pragma omp simd")
        L.append(f"  for (int lane = 0; lane < ne; ++lane) Y[{r}][lane] += {terms};")
    L.append("}")
    L.append("")
    L.append("// Row r, column c of the packed tangent, for the set-up.")
    L.append("static constexpr int SLOT[9][9] = {" +
             ", ".join("{" + ", ".join(str(slot(r, c)) for c in range(9)) + "}" for r in range(9)) + "};")
    L.append("}  // namespace qp")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------- set-up kernel


def hex8_to_hex27(text):
    """The PROTEUS_HEX8 tangent kernel turned into PROTEUS_HEX27's on the 4^3 rule.  The pointwise
    body -- deformation gradient, material law, pullback -- reads only the reference gradient and
    the (affine) geometry and is the same for both; the gather, the rule and the gradient offsets
    are the element's."""
    text = (text.replace("static constexpr int NQ = 8;", "static constexpr int NQ = 64;")
            .replace("static constexpr int NQ1 = 2;", "static constexpr int NQ1 = 4;")
            .replace("static constexpr int NS = 8;", "static constexpr int NS = 27;")
            .replace("ref_line_p1_q2", "ref_line_p2_q4").replace("quad_line_q2", "quad_line_q4")
            .replace('#include "line_p1_q2.hpp"', '#include "line_p2_q4.hpp"')
            .replace('#include "quad_line_q2.hpp"', '#include "quad_line_q4.hpp"'))
    m = re.search(r"\n    idx_t bev0\[VS\];.*?(?=\n    tensor_gradient_contiguous)", text, re.S)
    assert m, "HEX8 gather block not found"
    g = ["", *[f"    idx_t bev{s}[VS];" for s in range(27)], "    #pragma omp simd",
         "    for (int lane = 0; lane < ne; ++lane) {",
         *[f"      bev{s}[lane] = elements[{s}][evb + lane];" for s in range(27)], "    }",
         "    s_t bu_data[NS * 3][VS];", "    s_t gu_ref_q[NQ * 9 * VS];", "    #pragma omp simd",
         "    for (int lane = 0; lane < ne; ++lane) {"]
    for s_ in range(27):
        for c, u in enumerate(("ux", "uy", "uz")):
            g.append(f"      bu_data[{3 * s_ + c}][lane] = {u}[bev{s_}[lane] * u_stride];")
    g.append("    }")
    text = text[:m.start()] + "\n".join(g) + text[m.end():]
    # gradient storage [component][point][direction]: offsets of 3 * NQ per component
    text = text.replace("&gu_ref_q[24 * VS]", "&gu_ref_q[192 * VS]").replace("&gu_ref_q[48 * VS]", "&gu_ref_q[384 * VS]")
    for old, new in ((24, 192), (25, 193), (26, 194), (48, 384), (49, 385), (50, 386)):
        text = text.replace(f"gu_ref_q[({old} + q * 3) * VS]", f"gu_ref_q[({new} + q * 3) * VS]")
    return text


def emit_setup(name, header_text):
    """The generated P0 tangent kernel, with its quadrature sum turned into per-point (or
    per-moment) writes and its store turned into a call to the encoder."""
    if name == "PROTEUS_HEX27":
        header_text = hex8_to_hex27(header_text)
    if name == "TET10Q4":
        # The P0 kernel averages over the 11-point rule; the affine apply integrates with the
        # 4-point one, and so must the tangent compared with it.
        header_text = (header_text.replace("ref_tet10_q11", "ref_tet10_q4").replace("quad_tet_q11", "quad_tet_q4")
                       .replace("static constexpr int NQ = 11;", "static constexpr int NQ = 4;")
                       .replace('#include "kernel_math.hpp"', '#include "kernel_math.hpp"\n#include "tet10_q4.hpp"\n#include "quad_tet_q4.hpp"'))
    m = re.search(r"template <typename s_t, typename g_t, typename tangent_t, int VS>\n"
                  r"static SFEM_INLINE int (\w+)_inexact_apply_tangent_a_msoa_impl\((.*?)\n^}\n",
                  header_text, re.S | re.M)
    if not m:
        raise SystemExit("no tangent kernel in the generated header")
    body = m.group(0)
    body = body.replace("typename tangent_t", "typename Store", 1)
    body = re.sub(r"int \w+_inexact_apply_tangent_a_msoa_impl\(", "int qp_setup_impl(", body, count=1)
    body, n = re.subn(r"const ptrdiff_t tangent_component_stride,\n\s*tangent_t \*const RSTR tangent\n",
                      "Store &store\n", body)
    assert n == 1, "tangent parameters not found"
    body, n = re.subn(r"\n\s*tangent_t \*const RSTR btangent\d+ = [^\n]*;", "", body)
    assert n == NC, f"expected 45 store pointers, found {n}"
    # The weight: the true quadrature weight, not the one normalised to an average.
    body = re.sub(r"(const s_t qw = [^;]*?) \* QMEASURE;", r"\1;", body)
    scratch = "s_t bT[Store::NV][45][VS];"
    if "btangent_acc" in body:
        body = body.replace("s_t btangent_acc[45][VS];", scratch, 1)
        body, n = re.subn(r"btangent_acc\[(\d+)\]\[lane\] = s_t\(0\);",
                          r"for (int j = 0; j < Store::NV; ++j) bT[j][\1][lane] = s_t(0);", body)
        assert n == NC
        body, n = re.subn(r"btangent_acc\[(\d+)\]\[lane\] \+= qw \* (integrand\d+);",
                          r"Store::accumulate(q, bT, \1, lane, qw * \2);", body)
        assert n == NC, f"accumulation lines: {n}"
        # The P0 store loop goes; the block is encoded after the quadrature loop instead.
        body, n = re.subn(r"\n\s*#pragma omp simd\n\s*for \(int lane = 0; lane < ne; \+\+lane\) \{\n"
                          r"(\s*btangent\d+\[lane\] = tangent_t\(btangent_acc\[\d+\]\[lane\]\);\n)+\s*\}",
                          "", body)
        assert n == 1, "P0 store loop not found"
    else:
        # One point (TET4): the tangent is computed once, straight-line, per lane.
        body = re.sub(r"(const int ne = [^\n]*\n)", r"\1    " + scratch + "\n", body, count=1)
        body, n = re.subn(r"btangent(\d+)\[lane\] = tangent_t\((tangent\d+)\);",
                          r"Store::accumulate(0, bT, \1, lane, s_t(Store::Element::KVOL) * \2);", body)
        assert n == NC
    body, n = re.subn(r"\n  \}\n\n  return SFEM_SUCCESS;",
                      "\n    store.encode_block(evb, ne, bT);\n  }\n\n  return SFEM_SUCCESS;", body)
    assert n == 1, "end of the element-block loop not found"
    includes = re.findall(r'^#include "[^"]+"\n', header_text, re.M)
    out = ["// Emitted by make_qp_kernels.py from the generated P0 tangent kernel; do not edit.",
           "#pragma once"] + [i.rstrip() for i in includes] + [
        "// In the generated kernels' namespace, which holds the math helpers the body calls.",
        "namespace sfem {", "namespace codegen {", body, "}  // namespace codegen", "}  // namespace sfem",
        "namespace qp {", "using sfem::codegen::qp_setup_impl;", "}  // namespace qp"]
    return "\n".join(out) + "\n"


HELPERS = {"pow_m1", "pow_m2", "pow_m3", "pow_2", "pow_3", "pow_4", "sfem_log1p", "sqrt", "log", "exp", "s_t",
           "const", "fabs", "pow"}


def emit_point(name, header_text):
    """The tangent kernel's pointwise body: from the reference gradient (after its last component is
    formed) to the 45th tangent entry, minus the quadrature weight.  Its free names must be the
    gradient, the geometry and the material parameters, which the function takes as arguments."""
    if name == "PROTEUS_HEX27":
        header_text = hex8_to_hex27(header_text)
    m = re.search(r"template <typename s_t, typename g_t, typename tangent_t, int VS>\n"
                  r"static SFEM_INLINE int (\w+)_inexact_apply_tangent_a_msoa_impl\((.*?)\n^}\n",
                  header_text, re.S | re.M)
    if not m:
        raise SystemExit("no tangent kernel in the generated header")
    lines = m.group(0).split("\n")
    i0 = next(i for i, l in enumerate(lines) if re.match(r"\s*const s_t gu_ref_2_2 = ", l))
    i1 = next(i for i in range(i0, len(lines)) if re.match(r"\s*const s_t (integrand|tangent)44 = ", lines[i]))
    out = re.match(r"\s*const s_t (integrand|tangent)44 = ", lines[i1]).group(1)
    seg = [l.strip() for l in lines[i0 + 1:i1 + 1] if not re.match(r"\s*const s_t qw = ", l)]
    assert not any("[lane]" in l or "qw" in l for l in seg), "pointwise body reads lanes or the weight"
    declared = {re.match(r"const s_t (\w+) = ", l).group(1) for l in seg}
    used = set(re.findall(r"\b[A-Za-z_]\w*\b", " ".join(re.sub(r"^const s_t \w+ = ", "", l) for l in seg)))
    args = {f"gu_ref_{k}_{c}" for k in range(3) for c in range(3)} | {f"adjugate{c}" for c in range(9)} | {"determinant", "lmbda", "mu"}
    free = used - declared - args - HELPERS
    assert not free, f"pointwise body has free names {sorted(free)}"
    L = ["// Emitted by make_qp_kernels.py from the generated P0 tangent kernel's pointwise body; do not edit.",
         "#pragma once"] + QPHD.split("\n") + [
         "// The tangent at one point, unweighted: T[c], c the packed index of the symmetric 9x9 tangent.",
         "// The includer supplies the math helpers in sfem::codegen (kernel_math.hpp, or .cuh on the device).",
         "namespace sfem {", "namespace codegen {", "template <typename s_t>",
         "static QP_HD SFEM_INLINE void qp_point_tangent(const s_t *const RSTR gu_ref, const s_t *const RSTR adj,",
         "                                               const s_t determinant, const s_t lmbda, const s_t mu,",
         "                                               s_t *const RSTR T) {"]
    L += [f"  const s_t gu_ref_{k}_{c} = gu_ref[{k * 3 + c}];" for k in range(3) for c in range(3)]
    L += [f"  const s_t adjugate{c} = adj[{c}];" for c in range(9)]
    L += ["  " + l for l in seg]
    L += [f"  T[{c}] = {out}{c};" for c in range(NC)]
    L += ["}", "}  // namespace codegen", "}  // namespace sfem", "namespace qp {",
          "using sfem::codegen::qp_point_tangent;", "}  // namespace qp"]
    return "\n".join(L) + "\n"


def main():
    name, header, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    import os
    os.makedirs(out_dir, exist_ok=True)
    e = ELEMENTS[name]()
    with open(os.path.join(out_dir, "qp_element.hpp"), "w") as f:
        f.write(emit_element(name, e))
    with open(os.path.join(out_dir, "qp_contract.hpp"), "w") as f:
        f.write(emit_contract())
    with open(header) as f:
        text = f.read()
    with open(os.path.join(out_dir, "qp_setup.hpp"), "w") as f:
        f.write(emit_setup(name, text))
    with open(os.path.join(out_dir, "qp_point.hpp"), "w") as f:
        f.write(emit_point(name, text))
    with open(os.path.join(out_dir, "qp_device_tables.cuh"), "w") as f:
        f.write(emit_device_tables(name, e))


if __name__ == "__main__":
    main()
