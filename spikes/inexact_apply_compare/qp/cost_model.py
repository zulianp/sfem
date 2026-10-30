"""Cost model for the per-quadrature-point compressed partial assembly.

Every number the formulation document quotes for store sizes, operation counts
and BSR footprints is computed here, so the document can be checked against it:

    venv/bin/python spikes/inexact_apply_compare/qp/cost_model.py docs/compressed_partial_assembly/tables

Operation counts are taken after common-subexpression elimination, as in
docs/partial_assembly.tex, with the stored tangent entering as opaque symbols.
BSR footprints are counted on periodic versions of the spike's structured meshes (the Freudenthal
split for TET4/TET10, the uniform cube grid for HEX8 and HEX27).
"""

import itertools
import sys

import numpy as np
import sympy as sp

X = sp.symbols("x0 x1 x2")

# ---------------------------------------------------------------- elements


def tet4_basis():
    x, y, z = X
    return [1 - x - y - z, x, y, z]


def tet10_basis():
    x, y, z = X
    l = [1 - x - y - z, x, y, z]
    v = [li * (2 * li - 1) for li in l]
    # SFEM edge order (0,1) (1,2) (0,2) (0,3) (1,3) (2,3)
    e = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]
    return v + [4 * l[a] * l[b] for a, b in e]


def hex8_basis():
    x, y, z = X
    f = []
    for k, j, i in itertools.product((0, 1), repeat=3):
        f.append((x if i else 1 - x) * (y if j else 1 - y) * (z if k else 1 - z))
    return f


def hex27_basis():
    # Lagrange triquadratic on [0,1]^3, nodes at 0, 1/2, 1 in every direction.
    x, y, z = X
    l = lambda t: [2 * (t - sp.Rational(1, 2)) * (t - 1), -4 * t * (t - 1), 2 * t * (t - sp.Rational(1, 2))]
    return [a * b * c for c in l(z) for b in l(y) for a in l(x)]


def keast11():
    a = (1 + sp.sqrt(sp.Rational(5, 14))) / 4
    b = (1 - sp.sqrt(sp.Rational(5, 14))) / 4
    pts = [(sp.Rational(1, 4),) * 3]
    w = [sp.Rational(-74, 5625)]
    c, d = sp.Rational(1, 14), sp.Rational(11, 14)
    for p in [(c, c, c), (d, c, c), (c, d, c), (c, c, d)]:
        pts.append(p)
        w.append(sp.Rational(343, 45000))
    for p in [(a, a, b), (a, b, a), (a, b, b), (b, a, a), (b, a, b), (b, b, a)]:
        pts.append(p)
        w.append(sp.Rational(56, 2250))
    return pts, w


def gauss2_hex():
    g = [(1 - 1 / sp.sqrt(3)) / 2, (1 + 1 / sp.sqrt(3)) / 2]
    pts = [(g[i], g[j], g[k]) for k in (0, 1) for j in (0, 1) for i in (0, 1)]
    return pts, [sp.Rational(1, 8)] * 8


def gauss4_hex():
    # The 4x4x4 Gauss rule the isoparametric PROTEUS_HEX27 kernels integrate with.
    g, w = np.polynomial.legendre.leggauss(4)
    g, w = [(1 + t) / 2 for t in g], [t / 2 for t in w]
    pts = [(g[i], g[j], g[k]) for k in range(4) for j in range(4) for i in range(4)]
    return pts, [w[i] * w[j] * w[k] for k in range(4) for j in range(4) for i in range(4)]


ELEMENTS = {
    "TET4": dict(basis=tet4_basis, rule=lambda: ([(sp.Rational(1, 4),) * 3], [sp.Rational(1, 6)])),
    "TET10": dict(basis=tet10_basis, rule=keast11),
    "HEX8": dict(basis=hex8_basis, rule=gauss2_hex),
    "HEX27": dict(basis=hex27_basis, rule=gauss4_hex),
}

# Stored tangent components (major symmetry, energy materials) and the
# pairing used by docs/partial_assembly.tex: row (i,n), column (k,m).
D = 3
NC = (D * D) * (D * D + 1) // 2


def tangent_slot(i, k, m, n):
    r, c = i * D + n, k * D + m
    if r > c:
        r, c = c, r
    return r * (D * D) - r * (r - 1) // 2 + (c - r)


# ---------------------------------------------------------------- counting


def ops_after_cse(exprs):
    exprs = [sp.sympify(e) for e in exprs]
    reps, red = sp.cse(exprs, optimizations="basic")
    total = sum(sp.count_ops(rhs) for _, rhs in reps) + sum(sp.count_ops(e) for e in red)
    return int(total)


def grad_table(basis, pts):
    """B[q][m][j] = d phi_j / d xi_m at point q, numerically exact (floats)."""
    return [[[float(sp.diff(f, X[m]).subs(dict(zip(X, p)))) for f in basis] for m in range(D)] for p in pts]


def nz_coeff(c):
    return abs(c) > 1e-14


def coef(c):
    # Rounded so that equal tabulated values are recognised as equal by the elimination.
    return sp.Float(round(c, 12), 15)


def count_direct(name):
    """Per point: gradient of h (3x3), y = S^q g, scatter with B^T.  Weights folded into S^q."""
    el = ELEMENTS[name]
    basis = el["basis"]()
    pts, _ = el["rule"]()
    B = grad_table(basis, pts)
    N = len(basis)
    h = sp.symbols(f"h0:{D}_0:{N}")
    h = [[h[k * N + j] for j in range(N)] for k in range(D)]
    nq = len(pts)
    # Gradient at every point, eliminated jointly so sharing across points counts.
    g = [sum(coef(B[q][m][j]) * h[k][j] for j in range(N) if nz_coeff(B[q][m][j]))
         for q in range(nq) for k in range(D) for m in range(D)]
    flops_grad = ops_after_cse(g)
    # Contraction with 45 opaque symbols per point: y_in = sum_km S_ikmn g_km.
    flops_contract = nq * D * D * (2 * D * D - 1)
    # Scatter, summed over the points inside each output.
    ys = sp.symbols(f"y0:{nq * D * D}")
    sc = [sum(coef(B[q][n][p]) * ys[(q * D + i) * D + n] for q in range(nq) for n in range(D)
              if nz_coeff(B[q][n][p])) for i in range(D) for p in range(N)]
    flops_scatter = ops_after_cse(sc)
    return dict(grad=flops_grad, contract=flops_contract, scatter=flops_scatter,
                total=flops_grad + flops_contract + flops_scatter)


def count_modal_per_point(name):
    """TET10: P = vertex gradients, g_q = sum_a l_a(q) P_a, y_q = S^q g_q,
    Q_b = sum_q l_b(q) y_q, out = stage-1 transpose."""
    el = ELEMENTS[name]
    pts, _ = el["rule"]()
    nq = len(pts)
    base = count_modal(name)
    lam = lambda p: [1 - sum(p), p[0], p[1], p[2]]
    Pa = sp.symbols(f"P0:{4 * D * D}")
    g = [sum(coef(float(lam(p)[a])) * Pa[a * D * D + c] for a in range(4)) for p in pts for c in range(D * D)]
    ys = sp.symbols(f"y0:{nq * D * D}")
    Q = [sum(coef(float(lam(p)[b])) * ys[q * D * D + c] for q, p in enumerate(pts)) for b in range(4) for c in range(D * D)]
    interp, accum = ops_after_cse(g), ops_after_cse(Q)
    contract = nq * D * D * (2 * D * D - 1)
    return dict(stage1=base["stage1"], interp=interp, contract=contract, accum=accum, stage4=base["stage4"],
                total=base["stage1"] + interp + contract + accum + base["stage4"])


def gradient_span(name):
    """Rank and a basis of span{d phi_j / d xi_m}: the modal space of the staged schedule."""
    el = ELEMENTS[name]
    basis = el["basis"]()
    funcs = [sp.expand(sp.diff(f, X[m])) for f in basis for m in range(D)]
    monos = sorted({mono for f in funcs for mono in sp.Poly(f, *X).monoms()})
    M = sp.Matrix([[sp.Poly(f, *X).coeff_monomial(mono) for mono in monos] for f in funcs])
    return M.rank(), monos


def product_space_dim(name):
    """dim span{dphi_a/dxi_m * dphi_b/dxi_n}: what the operator sees of S (per component)."""
    el = ELEMENTS[name]
    basis = el["basis"]()
    grads = [[sp.expand(sp.diff(f, X[m])) for f in basis] for m in range(D)]
    dims = {}
    for m in range(D):
        for n in range(D):
            prods = [sp.expand(a * b) for a in grads[m] for b in grads[n]]
            monos = sorted({mono for f in prods for mono in sp.Poly(f, *X).monoms()})
            rows = [[sp.Poly(f, *X).coeff_monomial(mono) for mono in monos] for f in prods]
            # Exact rank for the small elements; HEX27's 729 x 125 rational matrix in floats.
            dims[(m, n)] = (sp.Matrix(rows).rank() if len(basis) <= 10
                            else int(np.linalg.matrix_rank(np.array(rows, dtype=float))))
    return dims


def moment_store(name):
    """Values per element if each stored component keeps only what the quadrature can see:
    min(nq, dim of the product space of its (m, n) pair), summed over the 45 components."""
    el = ELEMENTS[name]
    pts, _ = el["rule"]()
    nq = len(pts)
    dims = product_space_dim(name)
    total = 0
    seen = set()
    for i, k, m, n in itertools.product(range(D), repeat=4):
        s = tangent_slot(i, k, m, n)
        if s in seen:
            continue
        seen.add(s)
        total += min(nq, dims[(m, n)])
    return total, dims


def count_modal(name):
    """TET10: P = vertex gradients (gradient space is P1), moments M^{ab} = sum_q w_q l_a l_b S^q.
    out_ip = sum_{b,n} dl_b-structure ... counted as: stage 1 (h -> P), r^2 blocks of 81 FMA,
    stage 4 (Y -> out)."""
    el = ELEMENTS[name]
    basis = el["basis"]()
    N = len(basis)
    r, _ = gradient_span(name)
    # stage 1: gradient of h at the r Lagrange points of the gradient space (TET10: vertices)
    verts = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    B = grad_table(basis, verts)
    h = sp.symbols(f"h0:{D}_0:{N}")
    h = [[h[k * N + j] for j in range(N)] for k in range(D)]
    P = [sum(coef(B[a][m][j]) * h[k][j] for j in range(N) if nz_coeff(B[a][m][j]))
         for a in range(r) for k in range(D) for m in range(D)]
    s1 = ops_after_cse(P)
    # stage 2: Y_{b,i,n} = sum_a sum_{k,m} M^{ab}_{ikmn} P_{a,k,m}
    s2 = r * r * D * D * (2 * D * D) - r * D * D  # r^2 blocks of 9x9 FMAs, minus first adds
    # stage 4: out_ip = sum_b sum_n B[b][n][p] Y_{b,i,n}   (transpose of stage 1)
    ys = sp.symbols(f"y0:{r * D * D}")
    out = [sum(coef(B[b][n][p]) * ys[(b * D + i) * D + n] for b in range(r) for n in range(D)
               if nz_coeff(B[b][n][p])) for i in range(D) for p in range(N)]
    s4 = ops_after_cse(out)
    return dict(stage1=s1, contract=s2, stage4=s4, total=s1 + s2 + s4)


def count_sumfact(n1, nq1):
    """HEX27: gradients by 1D sweeps, n1 nodes and nq1 points per direction.  Per scalar field,
    x: B and D on n1^2 lines; y: B(Bx), D(Bx), B(Dx) on nq1 n1 lines; z: B(DxBy), B(BxDy), D(BxBy)
    on nq1^2 lines; each 1D map is nq1 outputs of n1 terms.  The scatter is the transpose."""
    line = nq1 * (2 * n1 - 1)
    grad = 3 * (2 * n1 * n1 + 3 * nq1 * n1 + 3 * nq1 * nq1) * line
    contract = nq1 ** 3 * D * D * (2 * D * D - 1)
    return dict(grad=grad, contract=contract, scatter=grad, total=2 * grad + contract)


def count_dense(name):
    """Direct schedule without elimination: nonzeros of the tabulated gradients."""
    el = ELEMENTS[name]
    basis = el["basis"]()
    pts, _ = el["rule"]()
    B = grad_table(basis, pts)
    nnz = sum(sum(1 for v in B[q][m] if nz_coeff(v)) for q in range(len(pts)) for m in range(D))
    grad = D * (2 * nnz - D * len(pts))
    contract = len(pts) * D * D * (2 * D * D - 1)
    return dict(grad=grad, contract=contract, scatter=grad, total=2 * grad + contract)


# ---------------------------------------------------------------- BSR


def mesh_graph(name, n):
    """The spike's meshes, made periodic so every node is an interior node: the counts are the
    asymptotic ones a large problem sees, without a boundary layer that shrinks with n."""
    nid = lambda i, j, k: ((k % n) * n + (j % n)) * n + (i % n)
    elems = []
    if name == "HEX8":
        for k, j, i in itertools.product(range(n), repeat=3):
            elems.append([nid(i + a, j + b, k + c) for c in (0, 1) for b in (0, 1) for a in (0, 1)])
        nnodes = n ** 3
    elif name == "HEX27":
        # One triquadratic element per 2x2x2 block of a (2n)^3 node grid.
        m = 2 * n
        nid2 = lambda i, j, k: ((k % m) * m + (j % m)) * m + (i % m)
        for k, j, i in itertools.product(range(n), repeat=3):
            elems.append([nid2(2 * i + a, 2 * j + b, 2 * k + c) for c in range(3) for b in range(3) for a in range(3)])
        nnodes = m ** 3
    else:
        tets = [[0, 1, 3, 7], [0, 1, 5, 7], [0, 4, 5, 7], [0, 4, 6, 7], [0, 2, 6, 7], [0, 2, 3, 7]]
        mid = {}
        nxt = n ** 3
        for k, j, i in itertools.product(range(n), repeat=3):
            for t in tets:
                v = [nid(i + (b & 1), j + ((b >> 1) & 1), k + ((b >> 2) & 1)) for b in t]
                if name == "TET10":
                    for a, b in [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]:
                        key = (min(v[a], v[b]), max(v[a], v[b]))
                        if key not in mid:
                            mid[key] = nxt
                            nxt += 1
                        v.append(mid[key])
                elems.append(v)
        nnodes = nxt
    return elems, nnodes


def bsr_blocks(name, n=6):
    elems, nnodes = mesh_graph(name, n)
    pairs = {(a, b) for e in elems for a in e for b in e}
    return len(pairs), nnodes, len(elems)


# ---------------------------------------------------------------- model

# Grace (one GH200 superchip, 72 Neoverse-V2 cores).  Bandwidth: what the BSR product sustains
# on this machine (403 GB/s on 2738019 dof, spikes/inexact_apply_compare/RESULTS.md, "Is the BSR
# implementation any good?").  Peaks are nominal, from the microarchitecture.
BANDWIDTH = 403e9
# Grace's nominal LPDDR5X bandwidth, the 100% that BANDWIDTH is a fraction of.
NOMINAL_BANDWIDTH = 500e9
PEAK_F64 = 3571e9
PEAK_F32 = 2 * PEAK_F64
IDX_BYTES = 4
VEC_BYTES_PER_DOF = 16  # read the input, write the output, once each per dof

# The HEX8 mesh of the BSR measurement below (RESULTS.md, same section as BANDWIDTH).
HEX8_DOF = 2738019
# The BSR product on the same mesh and machine, f64 and f32 values (same section).
BSR_HEX8_SECONDS = {"f64": 4.758e-3, "f32": 2.661e-3}


def codecs(nvals):
    """Bytes per element of each store, for nvals stored numbers."""
    return {
        "f64": nvals * 8,
        "f32": nvals * 4,
        "f16, scaled": nvals * 2 + 4,
        "int8, per element": nvals + 4,
        "int8, per component": nvals + 4 * NC,
        # The element mean (the P0 store) in f32, and every deviation from it in int8.  The
        # deviations sum to zero against the mean's weights, so one block could be implied, but
        # reconstructing it would sum the rounding of all the others.
        "int8, mean + deviation": NC * 4 + nvals + 4,
    }


def analyse():
    out = {}
    for name in ELEMENTS:
        pts, _ = ELEMENTS[name]["rule"]()
        nq = len(pts)
        r, _ = gradient_span(name)
        minimal, dims = moment_store(name)
        # HEX27: elimination over 64 points is out of reach; the direct count is the dense one,
        # and the schedule the sum-factorised one.
        direct = count_dense(name)["total"] if name == "HEX27" else count_direct(name)["total"]
        modal = count_modal_per_point(name)["total"] if name == "TET10" else None
        moments = count_modal(name)["total"] if name == "TET10" else None
        nb, nnodes, ne = bsr_blocks(name)
        ndof = 3 * nnodes
        el_per_dof = ne / ndof
        N = len(ELEMENTS[name]["basis"]())
        # Selected representation: the smaller exact store; its schedule.
        if minimal < NC * nq:
            store, flops = minimal, moments
        elif name == "HEX27":
            store, flops = NC * nq, count_sumfact(3, 4)["total"]
        else:
            store, flops = NC * nq, min(x for x in (direct, modal) if x is not None)
        bsr = {lab: (nb * (9 * vb + IDX_BYTES) + nnodes * 8) / ndof for lab, vb in (("f64", 8), ("f32", 4))}
        pa = {c: b * el_per_dof for c, b in codecs(store).items()}
        conn = N * IDX_BYTES * el_per_dof
        # Element matrices: the symmetric (3N)^2 matrix as its upper triangle, or only its
        # off-diagonal 3x3 node blocks (difference form, diagonal implied by the zero row sums).
        n = 3 * N
        ea_full, ea_diff = n * (n + 1) // 2, 9 * N * (N - 1) // 2
        # Full: n^2 multiply-adds.  Difference: per node pair 3 differences and 2 x 9 multiply-adds.
        ea_full_flops, ea_diff_flops = 2 * n * n, (N * (N - 1) // 2) * (3 + 36)
        # Pair-scaled int8 difference form: one byte per node pair and the element scale.
        ea_pair_bytes = ea_diff + N * (N - 1) // 2 + 4
        out_ea = dict(ea_full=ea_full, ea_diff=ea_diff, ea_full_flops=ea_full_flops, ea_diff_flops=ea_diff_flops,
                      ea_pair_bytes=ea_pair_bytes)
        out[name] = dict(**out_ea, nq=nq, r=r, dims=sorted(set(dims.values())), per_point=NC * nq, minimal=minimal,
                         store=store, direct=direct, modal=modal, moments=moments, flops=flops,
                         blocks=nb / nnodes, el_per_dof=el_per_dof, bsr=bsr, pa=pa,
                         conn=conn, flop_per_dof=flops * el_per_dof)
    return out


def fmt(x, nd=0):
    return f"{x:,.{nd}f}".replace(",", "{,}")


def write_tables(res, outdir):
    import os
    os.makedirs(outdir, exist_ok=True)
    E = ("TET4", "TET10", "HEX8", "HEX27")
    with open(os.path.join(outdir, "element_matrix.tex"), "w") as f:
        for e in E:
            d = res[e]
            epd = d["el_per_dof"]
            f16 = lambda v: fmt((v * 2 + 4) * epd)
            f.write(f"{e} & {d['store']} & {d['ea_full']} & {d['ea_diff']} & {d['flops']} & {d['ea_full_flops']} & "
                    f"{d['ea_diff_flops']} & {f16(d['store'])} & {f16(d['ea_diff'])} & "
                    f"{fmt(d['ea_pair_bytes'] * epd)} & {fmt(d['bsr']['f32'])} \\\\\n")
    with open(os.path.join(outdir, "stores.tex"), "w") as f:
        for e in E:
            d = res[e]
            dims = "/".join(str(x) for x in d["dims"])
            f.write(f"{e} & {d['nq']} & {d['r']} & {dims} & {d['per_point']} & {d['minimal']} \\\\\n")
    with open(os.path.join(outdir, "flops.tex"), "w") as f:
        for e in E:
            d = res[e]
            dash = "---"
            f.write(f"{e} & {d['direct']} & {d['modal'] or dash} & {d['moments'] or dash} & "
                    f"\\textbf{{{d['flops']}}} & {fmt(d['flop_per_dof'])} \\\\\n")
    with open(os.path.join(outdir, "footprint.tex"), "w") as f:
        for e in E:
            d = res[e]
            row = [fmt(d["bsr"]["f64"]), fmt(d["bsr"]["f32"])]
            # With one point there is nothing to share a scale across and no deviation from the
            # mean: the per-component and mean + deviation stores are not meaningful on TET4.
            single = d["nq"] == 1
            row += ["---" if single and ("component" in c or "mean" in c) else fmt(v) for c, v in d["pa"].items()]
            f.write(f"{e} & " + " & ".join(row) + " \\\\\n")
    with open(os.path.join(outdir, "roofline.tex"), "w") as f:
        for e in E:
            d = res[e]
            bsr64 = BANDWIDTH / (d["bsr"]["f64"] + VEC_BYTES_PER_DOF) / 1e6
            bsr32 = BANDWIDTH / (d["bsr"]["f32"] + VEC_BYTES_PER_DOF) / 1e6
            traffic = {c: d["pa"][c] + d["conn"] + VEC_BYTES_PER_DOF for c in ("f16, scaled", "int8, per element")}
            pa_bw = {c: BANDWIDTH / t / 1e6 for c, t in traffic.items()}
            need = d["flop_per_dof"] * bsr32 * 1e6
            f.write(f"{e} & {fmt(bsr64)} & {fmt(bsr32)} & {fmt(pa_bw['f16, scaled'])} & "
                    f"{fmt(pa_bw['int8, per element'])} & {fmt(need / 1e9)} & {fmt(100 * need / PEAK_F64)}\\% & "
                    f"{fmt(100 * need / PEAK_F32)}\\% \\\\\n")
    with open(os.path.join(outdir, "constants.tex"), "w") as f:
        f.write(f"\\newcommand{{\\MachineBandwidth}}{{{fmt(BANDWIDTH / 1e9)}}}\n")
        f.write(f"\\newcommand{{\\MachineNominalBandwidth}}{{{fmt(NOMINAL_BANDWIDTH / 1e9)}}}\n")
        f.write(f"\\newcommand{{\\MachineBandwidthPct}}{{{fmt(100 * BANDWIDTH / NOMINAL_BANDWIDTH)}}}\n")
        f.write(f"\\newcommand{{\\MachinePeakFsixtyfour}}{{{fmt(PEAK_F64 / 1e9)}}}\n")
        f.write(f"\\newcommand{{\\MachinePeakFthirtytwo}}{{{fmt(PEAK_F32 / 1e9)}}}\n")
        f.write(f"\\newcommand{{\\MeasuredDof}}{{{HEX8_DOF / 1e6:.2f}\\times10^{{6}}}}\n")
        h = res["HEX8"]
        for lab, tag in (("f64", "Sixtyfour"), ("f32", "Thirtytwo")):
            f.write(f"\\newcommand{{\\BsrMeasured{tag}}}{{{fmt(HEX8_DOF / BSR_HEX8_SECONDS[lab] / 1e6)}}}\n")
            f.write(f"\\newcommand{{\\BsrBound{tag}}}{{{fmt(BANDWIDTH / (h['bsr'][lab] + VEC_BYTES_PER_DOF) / 1e6)}}}\n")
        for e, tag in (("TET4", "TetFour"), ("TET10", "TetTen"), ("HEX8", "HexEight"), ("HEX27", "HexTwentySeven")):
            d = res[e]
            for key in ("nq", "r", "per_point", "minimal", "store", "flops", "direct"):
                macro = key.replace("_", "")
                f.write(f"\\newcommand{{\\{macro}{tag}}}{{{d[key] if d[key] is not None else '---'}}}\n")
            f.write(f"\\newcommand{{\\blocks{tag}}}{{{fmt(d['blocks'], 0)}}}\n")
            f.write(f"\\newcommand{{\\bsrsixtyfour{tag}}}{{{fmt(d['bsr']['f64'])}}}\n")
            f.write(f"\\newcommand{{\\bsrthirtytwo{tag}}}{{{fmt(d['bsr']['f32'])}}}\n")
            f.write(f"\\newcommand{{\\painteight{tag}}}{{{fmt(d['pa']['int8, per element'])}}}\n")
            f.write(f"\\newcommand{{\\pafsixteen{tag}}}{{{fmt(d['pa']['f16, scaled'])}}}\n")
            f.write(f"\\newcommand{{\\ratioint{tag}}}{{{d['bsr']['f64'] / d['pa']['int8, per element']:.1f}}}\n")
            f.write(f"\\newcommand{{\\ratiohalf{tag}}}{{{d['bsr']['f64'] / d['pa']['f16, scaled']:.1f}}}\n")


def main():
    res = analyse()
    for e, d in res.items():
        print(e, {k: v for k, v in d.items()})
    if len(sys.argv) > 1:
        write_tables(res, sys.argv[1])
        print("tables written to", sys.argv[1])


if __name__ == "__main__":
    main()
