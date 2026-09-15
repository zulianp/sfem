#!/usr/bin/env python3
"""TikZ figures and counts for docs/CVFEM_Operator_Locality.tex.

A two-dimensional analogue of the semi-structured CVFEM mesh: an unstructured macro mesh of
quadrilaterals (three macro-elements around a valence-3 vertex), each refined uniformly into
L x L micro-cells. Every operator's row reach is computed from the connectivity through global
node ids -- no stencil is assumed anywhere -- so the figures show what the operators couple on
an unstructured macro mesh, and the regular-lattice counts quoted beside them are computed the
same way rather than stated.

  python3 python/cvfem_locality_figures.py [outdir]     # default: docs/figures
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "docs", "figures")


def bilinear(q, s, t):
    a, b, c, d = q  # (0,0), (1,0), (1,1), (0,1)
    return tuple((1 - s) * (1 - t) * a[k] + s * (1 - t) * b[k] + s * t * c[k] + (1 - s) * t * d[k]
                 for k in range(len(a)))


class Mesh2D:
    """Macro quads refined L x L; nodes deduplicated across macro-elements by position."""

    def __init__(self, macros, L):
        self.L, self.macros = L, macros
        key, self.xy, self.gid = {}, [], {}
        for e, q in enumerate(macros):
            for y in range(L + 1):
                for x in range(L + 1):
                    p = bilinear(q, x / L, y / L)
                    k = (round(p[0], 5) + 0.0, round(p[1], 5) + 0.0)
                    if k not in key:
                        key[k] = len(self.xy)
                        self.xy.append(p)
                    self.gid[(e, x, y)] = key[k]
        self.cells = []  # (macro, x, y, [g0, g1, g2, g3] counter-clockwise in the macro lattice)
        for e in range(len(macros)):
            for y in range(L):
                for x in range(L):
                    self.cells.append((e, x, y, [self.gid[(e, x, y)], self.gid[(e, x + 1, y)],
                                                 self.gid[(e, x + 1, y + 1)], self.gid[(e, x, y + 1)]]))
        self.cells_of = [[] for _ in self.xy]
        for c, cell in enumerate(self.cells):
            for g in cell[3]:
                self.cells_of[g].append(c)

    def macros_of(self, n):
        return {self.cells[c][0] for c in self.cells_of[n]}

    # Frozen Jacobian: every face term reads only the nodes of its own micro-cell.
    def frozen_row(self, i):
        return {g for c in self.cells_of[i] for g in self.cells[c][3]}

    # G: the nodal gradient at n averages the cell gradients of the cells around n.
    def grad_row(self, n):
        return self.frozen_row(n)

    # K: the faces of node i's control volume each join i to an edge neighbour j and read
    # avg(qg_i, qg_j), so row i reads the gradient at i and at its edge neighbours.
    def k_row(self, i):
        s = {i}
        for c in self.cells_of[i]:
            g = self.cells[c][3]
            k = g.index(i)
            s.add(g[(k + 1) % 4])
            s.add(g[(k + 3) % 4])
        return s

    # K W^-1 G: node m, with the gradient nodes n it is reached through.
    def exact_row(self, i):
        via = {}
        for n in self.k_row(i):
            for m in self.grad_row(n):
                via.setdefault(m, set()).add(n)
        return via

    def storable(self, i, m):
        """An entry a sum of macro-element matrices can hold: some macro contains both nodes."""
        return bool(self.macros_of(i) & self.macros_of(m))


def prolongation_support(mesh, Lc):
    """Fine nodes where coarse node I's bilinear basis is non-zero, per coarse node (fine gid)."""
    q = mesh.L // Lc
    supp, cmacros = {}, {}
    for e in range(len(mesh.macros)):
        for Y in range(Lc + 1):
            for X in range(Lc + 1):
                I = mesh.gid[(e, q * X, q * Y)]
                s = supp.setdefault(I, set())
                cmacros.setdefault(I, set()).add(e)
                for y in range(mesh.L + 1):
                    for x in range(mesh.L + 1):
                        if abs(x / q - X) < 1 and abs(y / q - Y) < 1:
                            s.add(mesh.gid[(e, x, y)])
    return supp, cmacros


def coarse_row(supp, I, rowfn):
    reach = set()
    for f in supp[I]:
        reach |= rowfn(f)
    return {J for J, s in supp.items() if s & reach}


# ---------------------------------------------------------------------------------------
# Regular lattices: the counts a node away from any macro boundary sees, in 2D and 3D.
def regular_counts(dim, N=13):
    import itertools
    shape = [N] * dim
    idx = lambda p: sum(p[k] * N ** k for k in range(dim))
    inside = lambda p: all(0 <= p[k] < N for k in range(dim))
    box = [o for o in itertools.product((-1, 0, 1), repeat=dim)]
    star = [o for o in box if sum(abs(v) for v in o) <= 1]
    add = lambda p, o: tuple(p[k] + o[k] for k in range(dim))

    def frozen(p):
        return {add(p, o) for o in box if inside(add(p, o))}

    def exact(p):
        s = set()
        for o in star:
            n = add(p, o)
            if inside(n):
                s |= frozen(n)
        return s

    c = tuple([N // 2] * dim)
    fine = (len(frozen(c)), len(exact(c)))
    # coarse, ratio 2: coarse nodes at even indices, bilinear/trilinear support |d| <= 1 fine step
    C = N // 2 if (N // 2) % 2 == 0 else N // 2 - 1
    I = tuple([C] * dim)
    cnodes = [p for p in itertools.product(range(0, N, 2), repeat=dim)]
    supp = lambda J: {add(J, o) for o in box if inside(add(J, o))}

    def crow(fn):
        reach = set()
        for f in supp(I):
            reach |= fn(f)
        return sum(1 for J in cnodes if supp(J) & reach)

    return fine, (crow(frozen), crow(exact))


# ---------------------------------------------------------------------------------------
# TikZ
def P(p):
    return "(%.3f,%.3f)" % (p[0], p[1])


def mid(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def centre(m, c):
    g = m.cells[c][3]
    return (sum(m.xy[k][0] for k in g) / 4, sum(m.xy[k][1] for k in g) / 4)


class Pic:
    def __init__(self, scale):
        self.body = []
        self.scale = scale

    def __iadd__(self, s):
        self.body.append(s)
        return self

    def text(self):
        return "\\begin{tikzpicture}[scale=%s]\n%s\n\\end{tikzpicture}\n" % (self.scale, "\n".join(self.body))


def draw_mesh(pic, m, dual=False, coarse_q=None, faint=False, macro_style="macro"):
    seen = set()
    style = "microfaint" if faint else "micro"
    for (_, _, _, g) in m.cells:
        for k in range(4):
            a, b = g[k], g[(k + 1) % 4]
            if (min(a, b), max(a, b)) in seen:
                continue
            seen.add((min(a, b), max(a, b)))
            pic += "\\draw[%s] %s -- %s;" % (style, P(m.xy[a]), P(m.xy[b]))
    if dual:
        for c, (_, _, _, g) in enumerate(m.cells):
            cc = centre(m, c)
            for k in range(4):
                pic += "\\draw[dual] %s -- %s;" % (P(cc), P(mid(m.xy[g[k]], m.xy[g[(k + 1) % 4]])))
    if coarse_q:
        for e, q in enumerate(m.macros):
            for j in range(0, m.L + 1, coarse_q):
                pic += "\\draw[coarse] %s -- %s;" % (P(m.xy[m.gid[(e, 0, j)]]), P(m.xy[m.gid[(e, m.L, j)]]))
                pic += "\\draw[coarse] %s -- %s;" % (P(m.xy[m.gid[(e, j, 0)]]), P(m.xy[m.gid[(e, j, m.L)]]))
    for q in m.macros:
        pic += "\\draw[%s] %s -- %s -- %s -- %s -- cycle;" % ((macro_style,) + tuple(P(v) for v in q))


def shade_cells(pic, m, cells, style):
    for c in cells:
        g = m.cells[c][3]
        pic += "\\fill[%s] %s -- %s -- %s -- %s -- cycle;" % ((style,) + tuple(P(m.xy[k]) for k in g))


def shade_cv(pic, m, i, style):
    for c in m.cells_of[i]:
        g = m.cells[c][3]
        k = g.index(i)
        a, b = m.xy[g[(k + 1) % 4]], m.xy[g[(k + 3) % 4]]
        pic += "\\fill[%s] %s -- %s -- %s -- %s -- cycle;" % (style, P(m.xy[i]), P(mid(m.xy[i], a)),
                                                             P(centre(m, c)), P(mid(m.xy[i], b)))


def cv_faces(pic, m, i, style):
    for c in m.cells_of[i]:
        g = m.cells[c][3]
        k = g.index(i)
        cc = centre(m, c)
        for j in (g[(k + 1) % 4], g[(k + 3) % 4]):
            pic += "\\draw[%s] %s -- %s;" % (style, P(cc), P(mid(m.xy[i], m.xy[j])))


def dots(pic, m, nodes, style):
    for n in sorted(nodes):
        pic += "\\node[%s] at %s {};" % (style, P(m.xy[n]))


def label(pic, p, text, where="above right"):
    pic += "\\node[nodelabel, %s] at %s {%s};" % (where, P(p), text)


def arrow(pic, a, b, style):
    pic += "\\draw[%s] %s -- %s;" % (style, P(a), P(b))


def face_of(m, c, i, j):
    """The sub-control-volume face of cell c separating edge neighbours i and j: centre -> edge midpoint."""
    return centre(m, c), mid(m.xy[i], m.xy[j])


def panel_label(pic, text, where=(0.0, 3.65)):
    pic += "\\node[panel, anchor=south] at %s {%s};" % (P(where), text)


def main():
    os.makedirs(OUT, exist_ok=True)
    # A triangle tiled by three kite-shaped quadrilaterals: centroid, edge midpoint, corner, edge
    # midpoint. The centroid is a valence-3 vertex. (Parallelograms around a hexagon read as an
    # isometric cube; kites on a triangle cannot be mistaken for depth.)
    R = 3.4
    T = [(R * math.cos(math.radians(90 + 120 * k)), R * math.sin(math.radians(90 + 120 * k))) for k in range(3)]
    O = (0.0, 0.0)
    macros = [(O, mid(T[k], T[(k + 1) % 3]), T[k], mid(T[k], T[(k + 2) % 3])) for k in range(3)]
    m = Mesh2D(macros, 4)

    i_int = m.gid[(0, 2, 2)]    # centre of macro-element 0
    i_near = m.gid[(0, 2, 1)]   # one step inside a shared macro edge
    i_corner = m.gid[(0, 1, 1)]  # one step inside, next to the valence-3 vertex
    n_edge = m.gid[(0, 2, 0)]   # on the edge macro-element 0 shares with one neighbour
    counts = {}
    (nbr,) = m.macros_of(n_edge) - {0}
    nbr_name = "$E_%d$" % nbr

    # Figure 1: the two meshes on one set of nodes -- elements (micro-cells) and control volumes
    c_elem = [c for c, cell in enumerate(m.cells) if cell[:3] == (0, 1, 2)][0]
    pic = Pic(0.95)
    shade_cells(pic, m, [c_elem], "cellfill")
    shade_cv(pic, m, i_near, "cvfill")
    draw_mesh(pic, m, dual=True)
    cv_faces(pic, m, i_near, "kface")
    dots(pic, m, [i_near], "rownode")
    pic += "\\node[elemlabel] at %s {$\\Omega_c$};" % P(centre(m, c_elem))
    other = [c for c in m.cells_of[i_near] if m.cells[c][2] == 1 and m.cells[c][1] == 2][0]
    pic += "\\node[cvlabel, anchor=south west] at %s {$V_i$};" % P(mid(m.xy[i_near], centre(m, other)))
    for e, q in enumerate(macros):
        pic += "\\node[macrolabel] at %s {$E_%d$};" % (P(bilinear(q, 0.625, 0.625)), e)
    pic += "\\node[note, anchor=north] at (0,-0.2) {valence 3};"
    pic += "\\node[notedot] at (0,0) {};"
    open(os.path.join(OUT, "locality_mesh.tex"), "w").write(pic.text())

    # Figure: the anatomy of one frozen entry and of one path through K W^-1 G, zoomed.
    j_near = m.gid[(0, 3, 1)]  # an edge neighbour of i inside E_0
    via = m.exact_row(i_near)
    cands = sorted(k for k in via if n_edge in via[k] and not m.storable(i_near, k))
    m_node = cands[len(cands) // 2]
    cells_nm = [c for c in m.cells_of[n_edge] if m_node in m.cells[c][3]]

    def zoom(pic, pts, pad=0.75):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        pic += "\\clip %s rectangle %s;" % (P((min(xs) - pad, min(ys) - pad)), P((max(xs) + pad, max(ys) + pad)))

    def normal_arrow(pic, f0, f1, i, j, length=0.22):
        fp = mid(f0, f1)
        tx, ty = f1[0] - f0[0], f1[1] - f0[1]
        nx, ny = -ty, tx
        dx, dy = m.xy[j][0] - m.xy[i][0], m.xy[j][1] - m.xy[i][1]
        if nx * dx + ny * dy < 0:
            nx, ny = -nx, -ny
        s = length / math.hypot(nx, ny)
        arrow(pic, fp, (fp[0] + s * nx, fp[1] + s * ny), "avec")
        return fp

    # (a) a frozen entry: the faces of V_i inside one cell c read only c's corners
    c_ij = [c for c in m.cells_of[i_near] if j_near in m.cells[c][3]][0]
    pic = Pic(3.6)
    zoom(pic, [m.xy[g] for g in m.cells[c_ij][3]], pad=0.3)
    shade_cells(pic, m, [c_ij], "cellfill")
    shade_cv(pic, m, i_near, "cvfill")
    draw_mesh(pic, m, dual=True)
    g = m.cells[c_ij][3]
    k = g.index(i_near)
    for j in (g[(k + 1) % 4], g[(k + 3) % 4]):
        f0, f1 = face_of(m, c_ij, i_near, j)
        pic += "\\draw[kface] %s -- %s;" % (P(f0), P(f1))
    f0, f1 = face_of(m, c_ij, i_near, j_near)
    fp = normal_arrow(pic, f0, f1, i_near, j_near)
    label(pic, fp, "$\\cf{\\boldsymbol a_s}$", "left")
    arrow(pic, m.xy[i_near], m.xy[j_near], "dvec")
    dots(pic, m, g, "reach")
    dots(pic, m, [i_near], "rownode")
    label(pic, m.xy[i_near], "$i$", "below left")
    label(pic, m.xy[j_near], "$j$", "below right")
    # Towards the corner opposite i, clear of the faces of V_i and their labels.
    pic += "\\node[elemlabel] at %s {$\\Omega_c$};" % P(mid(centre(m, c_ij), m.xy[g[(k + 2) % 4]]))
    c_oth = [c for c in m.cells_of[i_near] if c != c_ij and j_near not in m.cells[c][3]][0]
    pic += "\\node[cvlabel] at %s {$V_i$};" % P(mid(m.xy[i_near], centre(m, c_oth)))
    label(pic, mid(m.xy[i_near], m.xy[j_near]), "$\\boldsymbol d_s$", "right")
    open(os.path.join(OUT, "locality_anatomy_frozen.tex"), "w").write(pic.text())

    # (b) one path i -> n -> m through K W^-1 G, crossing the macro-element boundary
    pic = Pic(3.0)
    zoom(pic, [m.xy[i_near], m.xy[n_edge], m.xy[m_node]], pad=0.4)
    shade_cells(pic, m, m.cells_of[n_edge], "gradfill")
    shade_cells(pic, m, cells_nm, "gradstrong")
    shade_cv(pic, m, i_near, "cvfill")
    draw_mesh(pic, m, dual=True)
    c_in = [c for c in m.cells_of[i_near] if n_edge in m.cells[c][3]]
    for c in c_in:
        f0, f1 = face_of(m, c, i_near, n_edge)
        pic += "\\draw[kface] %s -- %s;" % (P(f0), P(f1))
    f0, f1 = face_of(m, c_in[0], i_near, n_edge)
    label(pic, mid(f0, f1), "$\\cf{s}$", "right")
    arrow(pic, m.xy[i_near], m.xy[n_edge], "kpath")
    arrow(pic, m.xy[n_edge], m.xy[m_node], "gpath")
    dots(pic, m, [n_edge], "gradnode")
    dots(pic, m, [m_node], "cross")
    dots(pic, m, [i_near], "rownode")
    label(pic, m.xy[i_near], "$i$", "above left")
    label(pic, m.xy[n_edge], "$\\cg{n}$", "left")
    label(pic, m.xy[m_node], "$\\cx{m}$", "below right")
    label(pic, centre(m, cells_nm[0]), "$\\cg{\\Omega_{c'}}$", "above")
    c_oth = [c for c in m.cells_of[i_near] if n_edge not in m.cells[c][3]][0]
    pic += "\\node[cvlabel] at %s {$V_i$};" % P(mid(m.xy[i_near], centre(m, c_oth)))
    open(os.path.join(OUT, "locality_anatomy_path.tex"), "w").write(pic.text())
    counts["pathCellsN"] = len(m.cells_of[n_edge])
    counts["pathCellsNM"] = len(cells_nm)
    assert m.macros_of(n_edge) > {0} and not (m.macros_of(i_near) & m.macros_of(m_node))

    # Figure 2: the three factors, one panel each
    A = m.frozen_row(i_near)
    pic = Pic(0.72)
    shade_cells(pic, m, m.cells_of[i_near], "cellfill")
    draw_mesh(pic, m)
    cv_faces(pic, m, i_near, "cvline")
    dots(pic, m, A, "reach")
    dots(pic, m, [i_near], "rownode")
    label(pic, m.xy[i_near], "$i$", "above left")
    panel_label(pic, "(a) $\\cv{A_{\\mathrm{frozen}}}$, row $i$")
    open(os.path.join(OUT, "locality_frozen.tex"), "w").write(pic.text())
    counts["frozenNear"] = len(A)

    Gn = m.grad_row(n_edge)
    pic = Pic(0.72)
    shade_cells(pic, m, m.cells_of[n_edge], "gradfill")
    draw_mesh(pic, m)
    dots(pic, m, Gn, "reach")
    dots(pic, m, [n_edge], "gradnode")
    label(pic, m.xy[n_edge], "$\\cg{n}$", "below left")
    panel_label(pic, "(b) $\\cg{W^{-1}G}$, row $\\cg{n}$")
    open(os.path.join(OUT, "locality_grad.tex"), "w").write(pic.text())
    counts["gradEdge"] = len(Gn)
    counts["gradEdgeCells"] = len(m.cells_of[n_edge])

    Kn = m.k_row(i_near)
    pic = Pic(0.72)
    shade_cv(pic, m, i_near, "cvfill")
    draw_mesh(pic, m, dual=True)
    cv_faces(pic, m, i_near, "kface")
    dots(pic, m, Kn, "gradnode")
    dots(pic, m, [i_near], "rownode")
    label(pic, m.xy[i_near], "$i$", "above left")
    panel_label(pic, "(c) $\\cf{K}$, row $i$")
    open(os.path.join(OUT, "locality_k.tex"), "w").write(pic.text())
    counts["kNear"] = len(Kn)

    # Figure 3: the exact term's row, coloured by whether a macro-element matrix could hold it
    for tag, i, title in (("int", i_int, "(a) $i$ inside $E_0$"), ("near", i_near, "(b) $i$ next to $E_0$ | %s" % nbr_name),
                          ("corner", i_corner, "(c) $i$ next to the valence-3 vertex")):
        via = m.exact_row(i)
        local = {n for n in via if m.storable(i, n)}
        cross = set(via) - local
        pic = Pic(0.72)
        draw_mesh(pic, m)
        cv_faces(pic, m, i, "cvline")
        dots(pic, m, local, "reach")
        dots(pic, m, cross, "cross")
        dots(pic, m, m.k_row(i), "gradring")
        dots(pic, m, [i], "rownode")
        panel_label(pic, title)
        open(os.path.join(OUT, "locality_exact_%s.tex" % tag), "w").write(pic.text())
        counts["exact" + tag.capitalize()] = len(via)
        counts["cross" + tag.capitalize()] = len(cross)
        counts["frozen" + tag.capitalize()] = len(m.frozen_row(i))

    # Figure 5: the same geometry as a standard quad4 mesh -- every micro-cell its own element,
    # so an element matrix holds only the four nodes of one cell.
    flat = Mesh2D([tuple(m.xy[g] for g in cell[3]) for cell in m.cells], 1)
    to_flat = lambda g: min(range(len(flat.xy)),
                            key=lambda k: (flat.xy[k][0] - m.xy[g][0]) ** 2 + (flat.xy[k][1] - m.xy[g][1]) ** 2)
    for tag, i, title in (("frozen", to_flat(i_int), "(a) $\\cv{A_{\\mathrm{frozen}}}$, row $i$"),
                          ("int", to_flat(i_int), "(b) $\\cf{K}\\cg{W^{-1}G}$, row $i$"),
                          ("corner", to_flat(i_corner), "(c) $\\cf{K}\\cg{W^{-1}G}$, next to the valence-3 vertex")):
        row = flat.frozen_row(i) if tag == "frozen" else set(flat.exact_row(i))
        local = {n for n in row if flat.storable(i, n)}
        cross = row - local
        pic = Pic(0.72)
        if tag == "frozen":
            shade_cells(pic, flat, flat.cells_of[i], "cellfill")
        draw_mesh(pic, flat, macro_style="element")
        dots(pic, flat, local, "reach")
        dots(pic, flat, cross, "cross")
        if tag != "frozen":
            dots(pic, flat, flat.k_row(i), "gradring")
        dots(pic, flat, [i], "rownode")
        panel_label(pic, title)
        open(os.path.join(OUT, "locality_quad4_%s.tex" % tag), "w").write(pic.text())
        counts["quadRow" + tag.capitalize()] = len(row)
        counts["quadCross" + tag.capitalize()] = len(cross)

    # Figure 4: one coarse level (ratio 2), frozen and exact Galerkin rows of one coarse node
    supp, cmac = prolongation_support(m, 2)
    for tag, I in (("int", m.gid[(0, 2, 2)]), ("edge", m.gid[(0, 2, 0)])):
        for kind, fn in (("frozen", m.frozen_row), ("exact", lambda f: set(m.exact_row(f)) | m.frozen_row(f))):
            row = coarse_row(supp, I, fn)
            local = {J for J in row if cmac[I] & cmac[J]}
            cross = row - local
            pic = Pic(0.72)
            draw_mesh(pic, m, coarse_q=2, faint=True)
            dots(pic, m, supp[I], "support")
            dots(pic, m, local, "reachc")
            dots(pic, m, cross, "crossc")
            dots(pic, m, [I], "rownode")
            panel_label(pic, "(%s) %s" % ("a" if kind == "frozen" else "b",
                                          "$P^{\\mathsf T}\\cv{A_{\\mathrm{frozen}}}P$" if kind == "frozen"
                                          else "$P^{\\mathsf T}J_{\\mathrm{exact}}P$"))
            open(os.path.join(OUT, "locality_coarse_%s_%s.tex" % (tag, kind)), "w").write(pic.text())
            counts["coarse" + kind.capitalize() + tag.capitalize()] = len(row)
            counts["coarseCross" + kind.capitalize() + tag.capitalize()] = len(cross)

    (f2, c2), (f3, c3) = regular_counts(2), regular_counts(3)
    counts.update(regTwoFrozen=f2[0], regTwoExact=f2[1], regTwoCoarseFrozen=c2[0], regTwoCoarseExact=c2[1],
                  regThreeFrozen=f3[0], regThreeExact=f3[1], regThreeCoarseFrozen=c3[0], regThreeCoarseExact=c3[1])

    with open(os.path.join(OUT, "locality_counts.tex"), "w") as fh:
        fh.write("%% generated by python/cvfem_locality_figures.py\n")
        fh.write("\\newcommand{\\nbr}{%s}\n" % nbr_name)
        for k, v in sorted(counts.items()):
            fh.write("\\newcommand{\\n%s}{%d}\n" % (k, v))
    for k, v in sorted(counts.items()):
        print("%-28s %d" % (k, v))


if __name__ == "__main__":
    main()
