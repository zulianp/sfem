#!/usr/bin/env python3
"""TikZ figures for the packed mesh format (paper figures F1-F4).

Stdlib only, following spikes/cvfem/python/cvfem_locality_figures.py: the Alps uenv has no
numpy or matplotlib, and a figure generator that cannot run where the measurements run is a
figure that goes stale.

WHAT THIS MODELS, AND WHY IT IS NOT A DRAWING.

The figures are not hand-placed. This module reimplements the *documented* packing rules from
spikes/cvfem/docs/PACKED_FORMAT.md on a small 2D UNSTRUCTURED mesh -- jittered points, Delaunay
triangulated, ordered along a Hilbert curve -- and draws the result, so the pictures cannot
disagree with the contract they illustrate. The mesh is unstructured because the figure has to
show what a pack boundary is: on a grid the packs come out as rectangular blocks and read as a
decomposition somebody chose, when in fact a pack is wherever a contiguous range of the element
order happens to end. The rules implemented:

  * a pack is a contiguous range of element indices, p = e // elements_per_pack (spec S1);
  * each node has exactly one owning pack, and the owned ranges partition [0, nnodes) in
    monotone order (S2);
  * within a pack's owned range, ids are ordered non-shared first then shared (S3);
  * a pack's ghost list holds the global ids it touches but does not own, deduplicated, in
    first-appearance order of a node-major element-minor sweep, NOT sorted (S4);
  * the ghost reduction is a CSR gather whose builder sorts (global_id, entry_index) pairs, so
    each destination appears in exactly one row (S5).

--selftest asserts every one of those against the model. That is the point of having a model
rather than a drawing: the invariants the paper claims are checked, not asserted.

TWO HONEST LIMITS, stated here so they end up in the captions rather than being discovered by a
reviewer. (a) The mesh is 2D and the format is used on 3D HEX8; 2D is chosen because a readable
figure of a 3D pack decomposition does not exist. The structure -- ownership, the three zones,
ghosts, the reduction graph -- is dimension-independent, the node counts are not. (b) The real
builder derives ownership inside smesh; this reproduces its documented *result*, not its code
path.

Usage:
    python3 packed_format_figures.py --out ../figures
    python3 packed_format_figures.py --selftest
"""

import argparse
import os
import sys

# ---------------------------------------------------------------------------------------------
# The model: an unstructured 2D mesh, spatially ordered, then packed.
# ---------------------------------------------------------------------------------------------


class _Lcg:
    """A fixed linear congruential generator, so the point set is decided by this source file.

    `random` would do, and its Mersenne Twister is stable across CPython versions, but the whole
    directory rests on a figure regenerating byte for byte; pinning the generator here removes
    the interpreter from that guarantee entirely. Numerical Recipes' constants.
    """

    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def next(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return self.state / 4294967296.0

    def between(self, lo, hi):
        return lo + (hi - lo) * self.next()


def _circumcircle(a, b, c):
    """Centre and squared radius, or None for a degenerate triangle."""
    (ax, ay), (bx, by), (cx, cy) = a, b, c
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        return None
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    return (ux, uy), (ux - ax) ** 2 + (uy - ay) ** 2


class TriMesh:
    """An unstructured triangular mesh: jittered points, Delaunay-triangulated.

    Unstructured rather than a grid because the figure has to show what a pack boundary actually
    is. On a structured mesh the packs come out as rectangular blocks and read as a decomposition
    somebody chose; the boundary of a pack is in fact wherever a contiguous range of the
    space-filling element order happens to end, which on a real mesh is a ragged line that
    follows nothing. That is only visible on a mesh whose elements are not in rows.

    Bowyer-Watson, which is short enough to keep this module stdlib-only and dependency-free.
    """

    nodes_per_element = 3

    def __init__(self, nx=9, ny=7, jitter=0.34, seed=20260928):
        rng = _Lcg(seed)
        pts = []
        # A jittered lattice rather than uniform sampling: it keeps the element sizes within a
        # narrow band, so no triangle is too small to see, while leaving the connectivity
        # irregular. Boundary points are jittered only along their edge, which keeps the domain
        # a clean rectangle and stops the triangulation growing slivers at the hull.
        for j in range(ny + 1):
            for i in range(nx + 1):
                x, y = float(i), float(j)
                on_x = i in (0, nx)
                on_y = j in (0, ny)
                if not on_x:
                    x += rng.between(-jitter, jitter)
                if not on_y:
                    y += rng.between(-jitter, jitter)
                pts.append((x, y))
        self.points = pts
        self.nnodes = len(pts)
        self.tris = self._delaunay(pts)
        self.nelements = len(self.tris)

    @staticmethod
    def _delaunay(pts):
        # A super-triangle large enough to contain every point; its vertices carry negative ids
        # so they are trivially identifiable and discarded at the end.
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        mx, my = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
        span = max(max(xs) - min(xs), max(ys) - min(ys)) * 10.0
        sup = [(mx - span, my - span), (mx + span, my - span), (mx, my + span)]

        def pt(i):
            return sup[-1 - i] if i < 0 else pts[i]

        tris = [(-1, -2, -3)]
        for i, p in enumerate(pts):
            bad, keep = [], []
            for t in tris:
                cc = _circumcircle(pt(t[0]), pt(t[1]), pt(t[2]))
                if cc and (p[0] - cc[0][0]) ** 2 + (p[1] - cc[0][1]) ** 2 < cc[1] * (1 - 1e-12):
                    bad.append(t)
                else:
                    keep.append(t)
            # The cavity boundary: edges of the bad triangles that are not shared by two of them.
            count = {}
            for t in bad:
                for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                    count[tuple(sorted(e))] = count.get(tuple(sorted(e)), 0) + 1
            tris = keep
            for e, c in sorted(count.items()):
                if c == 1:
                    tris.append((e[0], e[1], i))
        out = []
        for t in tris:
            if min(t) < 0:
                continue
            # Counter-clockwise, the orientation the element kernels assume.
            (ax, ay), (bx, by), (cx, cy) = pts[t[0]], pts[t[1]], pts[t[2]]
            if (bx - ax) * (cy - ay) - (by - ay) * (cx - ax) < 0:
                t = (t[0], t[2], t[1])
            out.append(t)
        return sorted(out)

    def coord(self, n):
        return self.points[n]

    def element_nodes(self, e):
        return list(self.tris[e])

    def element_centre(self, e):
        ps = [self.points[n] for n in self.tris[e]]
        return (sum(p[0] for p in ps) / 3.0, sum(p[1] for p in ps) / 3.0)


def morton_order(mesh):
    """A space-filling element order. The format REQUIRES spatial reordering before packing --
    it is a precondition, not an optimisation: without it a pack's elements are scattered and
    its node set is huge. Measured consequence at pack_size 2048 on the real mesh: 2735 nodes
    per pack when space-filling against 4373 when lexicographic, and 1.25x in throughput.

    Keyed on the element CENTROID, quantised to a fine lattice, so it applies to a mesh whose
    elements are not in rows -- which is also what makes the pack boundaries in the figure fall
    where the curve leaves rather than along a straight line.

    HILBERT rather than Morton. Morton is shorter but its curve jumps, so a pack of contiguous
    indices comes out as several disconnected shards -- true of Morton, misleading as a picture of
    a pack, and not what production does: smesh::SFC walks a Hilbert curve, whose packs are
    compact and simply connected. The figure detaches two packs and they have to look like
    subdomains, because that is what they are."""
    cs = [mesh.element_centre(e) for e in range(mesh.nelements)]
    x0 = min(c[0] for c in cs)
    y0 = min(c[1] for c in cs)
    span = max(max(c[0] for c in cs) - x0, max(c[1] for c in cs) - y0) or 1.0
    res = 1 << 10

    def hilbert_d(x, y, order=10):
        """Distance along a Hilbert curve of side 2**order. The standard xy-to-d rotation."""
        rx = ry = 0
        d = 0
        side = 1 << (order - 1)
        while side > 0:
            rx = 1 if (x & side) > 0 else 0
            ry = 1 if (y & side) > 0 else 0
            d += side * side * ((3 * rx) ^ ry)
            # Rotate the quadrant so the curve stays continuous across it.
            if ry == 0:
                if rx == 1:
                    x = side - 1 - x
                    y = side - 1 - y
                x, y = y, x
            side >>= 1
        return d

    def key(e):
        cx, cy = cs[e]
        x = min(res - 1, int((cx - x0) / span * res))
        y = min(res - 1, int((cy - y0) / span * res))
        return (hilbert_d(x, y), e)

    return sorted(range(mesh.nelements), key=key)


class Packing:
    """The packed layout of a mesh under a given element order and pack size.

    Attributes mirror the names in PackedData (spikes/cvfem/src/hex8/cvfem_hex8_pack_common.hpp)
    so the figure labels and the code agree:
        n_packs, elements_per_pack
        owned_nodes_ptr[p]      prefix sum; pack p owns globals [ptr[p], ptr[p+1])
        n_shared[p]             how many of p's owned nodes another pack also touches
        ghost_ptr / ghost_idx   CSR of the global ids p touches but does not own
        ghost_reduce_{ptr,idx,dest}
        new_of_old / old_of_new the in-place renumbering
    """

    def __init__(self, mesh, order, elements_per_pack):
        self.mesh = mesh
        self.order = order
        self.elements_per_pack = elements_per_pack
        self.n_packs = (mesh.nelements + elements_per_pack - 1) // elements_per_pack

        # Which packs touch each node, in pack order.
        touch = [[] for _ in range(mesh.nnodes)]
        for slot, e in enumerate(order):
            p = slot // elements_per_pack
            for n in mesh.element_nodes(e):
                if not touch[n] or touch[n][-1] != p:
                    if p not in touch[n]:
                        touch[n].append(p)

        # Ownership: the lowest-indexed pack that touches the node. Any single-valued rule
        # gives monotone contiguous ranges once the renumbering below sorts by (owner, shared);
        # taking the minimum is what makes the owned ranges appear in pack order.
        self.owner = [min(t) for t in touch]
        self.shared = [len(t) > 1 for t in touch]

        # The renumbering. Sort by (owning pack, shared-last) so that:
        #   * owned ranges are contiguous and monotone in p                      (spec S2)
        #   * within a pack, non-shared ids precede shared ones                  (spec S3)
        # The third key keeps the order deterministic for a fixed input, which is the whole
        # spirit of the format.
        old = sorted(range(mesh.nnodes), key=lambda n: (self.owner[n], self.shared[n], n))
        self.old_of_new = old
        self.new_of_old = [0] * mesh.nnodes
        for new, o in enumerate(old):
            self.new_of_old[o] = new

        # Renumbered ownership arrays.
        self.owner_new = [0] * mesh.nnodes
        self.shared_new = [False] * mesh.nnodes
        for new, o in enumerate(old):
            self.owner_new[new] = self.owner[o]
            self.shared_new[new] = self.shared[o]

        self.owned_nodes_ptr = [0] * (self.n_packs + 1)
        for n in range(mesh.nnodes):
            self.owned_nodes_ptr[self.owner_new[n] + 1] += 1
        for p in range(self.n_packs):
            self.owned_nodes_ptr[p + 1] += self.owned_nodes_ptr[p]

        self.n_shared = [0] * self.n_packs
        for n in range(mesh.nnodes):
            if self.shared_new[n]:
                self.n_shared[self.owner_new[n]] += 1

        # Ghost lists: first appearance in a node-major, element-minor sweep, deduplicated,
        # NOT sorted (spec S4). The sweep order is what fixes the ids, so it is reproduced
        # rather than approximated by a set.
        self.ghost_ptr = [0]
        self.ghost_idx = []
        self.pack_elements = []
        for p in range(self.n_packs):
            elems = order[p * elements_per_pack:(p + 1) * elements_per_pack]
            self.pack_elements.append(elems)
            seen, ghosts = set(), []
            for v in range(mesh.nodes_per_element):  # node-major
                for e in elems:  # element-minor
                    n = self.new_of_old[self.mesh.element_nodes(e)[v]]
                    if self.owner_new[n] != p and n not in seen:
                        seen.add(n)
                        ghosts.append(n)
            self.ghost_idx.extend(ghosts)
            self.ghost_ptr.append(len(self.ghost_idx))

        # The reduction graph (spec S5): sort (destination, entry) pairs and group, so each
        # destination is exactly one row and rows are ordered by destination.
        pairs = sorted((self.ghost_idx[k], k) for k in range(len(self.ghost_idx)))
        self.ghost_reduce_dest, self.ghost_reduce_ptr, self.ghost_reduce_idx = [], [0], []
        for dest, entry in pairs:
            if not self.ghost_reduce_dest or self.ghost_reduce_dest[-1] != dest:
                self.ghost_reduce_dest.append(dest)
                self.ghost_reduce_ptr.append(self.ghost_reduce_ptr[-1])
            self.ghost_reduce_idx.append(entry)
            self.ghost_reduce_ptr[-1] += 1

    # -- the format's own accessors, so the figures use the same arithmetic the kernels do ----

    def n_contiguous(self, p):
        return self.owned_nodes_ptr[p + 1] - self.owned_nodes_ptr[p]

    def n_pack_nodes(self, p):
        return self.n_contiguous(p) + (self.ghost_ptr[p + 1] - self.ghost_ptr[p])

    def local_to_global(self, p, l):
        """The reference resolution, cvfem_hex8_best_common.hpp::pack_local_to_global."""
        nc = self.n_contiguous(p)
        if l < nc:
            return self.owned_nodes_ptr[p] + l
        return self.ghost_idx[self.ghost_ptr[p] + (l - nc)]

    def zone(self, p, l):
        """Which of the three zones a pack-local id falls in."""
        nc, ns = self.n_contiguous(p), self.n_shared[p]
        if l < nc - ns:
            return "exclusive"
        if l < nc:
            return "shared"
        return "ghost"


# ---------------------------------------------------------------------------------------------
# TikZ emission
# ---------------------------------------------------------------------------------------------

PACK_COLORS = ["PackA", "PackB", "PackC", "PackD", "PackE", "PackF"]

PREAMBLE = r"""% Generated by wip/paper/python/packed_format_figures.py -- do not edit.
% Regenerate with: make figures
"""

COLOR_DEFS = r"""\definecolor{PackA}{HTML}{4C72B0}
\definecolor{PackB}{HTML}{DD8452}
\definecolor{PackC}{HTML}{55A868}
\definecolor{PackD}{HTML}{C44E52}
\definecolor{PackE}{HTML}{8172B3}
\definecolor{PackF}{HTML}{937860}
"""


def adjacent_pair(pk):
    """The two packs worth detaching.

    Ghosts flow one way between any two packs: ownership is first-touch in pack order, so of two
    packs sharing a node the LOWER always owns it and the higher always holds the ghost. A pair
    therefore shows arcs in one direction only, and that is a property of the format rather than
    an artefact of the choice.

    Chosen to maximise, in order, the number of DISTINCT owners appearing in the two packs' ghost
    lists, then the number of arcs between them. Diversity first because the point most easily
    missed is that one pack's ghost list spans several owners -- a pair whose ghosts all come
    from each other would suggest the ghost list is a per-neighbour thing, which it is not.
    """
    best, bestkey = (0, min(1, pk.n_packs - 1)), (-1, -1)
    for a in range(pk.n_packs):
        for b in range(a + 1, pk.n_packs):
            ga = pk.ghost_idx[pk.ghost_ptr[a]:pk.ghost_ptr[a + 1]]
            gb = pk.ghost_idx[pk.ghost_ptr[b]:pk.ghost_ptr[b + 1]]
            owned_a = range(pk.owned_nodes_ptr[a], pk.owned_nodes_ptr[a + 1])
            owned_b = range(pk.owned_nodes_ptr[b], pk.owned_nodes_ptr[b + 1])
            cross = (len(set(ga) & set(owned_b)) + len(set(gb) & set(owned_a)))
            owners = {pk.owner_new[n] for n in ga} | {pk.owner_new[n] for n in gb}
            key = (len(owners), cross)
            if key > bestkey:
                best, bestkey = (a, b), key
    return best


def pack_node_ids(pk, p):
    """Every global id pack p addresses: its owned range then its ghost list."""
    owned = list(range(pk.owned_nodes_ptr[p], pk.owned_nodes_ptr[p + 1]))
    ghosts = pk.ghost_idx[pk.ghost_ptr[p]:pk.ghost_ptr[p + 1]]
    return owned, list(ghosts)


def pack_label_point(pk, p, dx=0.0, dy=0.0):
    """A point well inside pack p, for its label.

    The centroid of the pack's element centres is the obvious choice and the wrong one: a pack is
    a contiguous range of a space-filling order, so its region is often L-shaped or notched, and
    the centroid then drifts toward an edge or out of the pack altogether. This takes the discrete
    pole of inaccessibility instead -- the pack's own element whose centre is farthest from any
    element the pack does not contain -- which is inside the region by construction and away from
    its boundary.
    """
    m = pk.mesh
    mine = pk.pack_elements[p]
    # What the label must keep away from: the other packs AND the domain boundary. Omitting the
    # boundary put every label in a corner of the mesh, which is indeed as far from the other
    # packs as one can get and is not the middle of anything.
    avoid = [m.element_centre(e) for q in range(pk.n_packs) if q != p
             for e in pk.pack_elements[q]]
    seen = {}
    for e in range(m.nelements):
        nn = m.element_nodes(e)
        for u, v in zip(nn, nn[1:] + nn[:1]):
            key = (min(u, v), max(u, v))
            seen[key] = seen.get(key, 0) + 1
    for (u, v), c in seen.items():
        if c == 1:
            avoid.append(((m.points[u][0] + m.points[v][0]) / 2.0,
                          (m.points[u][1] + m.points[v][1]) / 2.0))
    best, bestd = m.element_centre(mine[0]), -1.0
    for e in mine:
        cx, cy = m.element_centre(e)
        d = min(((cx - ox) ** 2 + (cy - oy) ** 2 for ox, oy in avoid), default=1e9)
        if d > bestd:
            best, bestd = (cx, cy), d
    return (best[0] + dx, best[1] + dy)


def _node_xy(pk, n):
    """A node's coordinates. Coordinates belong to the node, so they follow the renumbering."""
    return pk.mesh.coord(pk.old_of_new[n])


def fig_decomposition(pk, scale=0.78, gap=1.6):
    """F1: the decomposition, and two packs detached so the interface is visible.

    Three rules carry the whole figure and are stated in the caption rather than a legend:
    element fill is the pack, node colour is the OWNER, and opacity is ownership. A faded node
    drawn in another pack's colour is therefore a ghost, and says whose it is without a label.

    The detached pair is drawn from the model's own arrays -- owned ranges for the solid nodes,
    ghost_idx for the faded ones -- so the picture cannot show a relationship the format does
    not have.
    """
    m = pk.mesh
    a, b = adjacent_pair(pk)
    out = [PREAMBLE, COLOR_DEFS, r"\begin{tikzpicture}[scale=%.2f]" % scale]

    xs = [p[0] for p in m.points]
    width = max(xs) - min(xs)

    def poly(e, fill, opacity=1.0, draw="black!45", lw="0.3pt", dx=0.0, dy=0.0):
        pts = " -- ".join("(%.3f,%.3f)" % (m.points[n][0] + dx, m.points[n][1] + dy)
                          for n in m.element_nodes(e))
        return (r"  \filldraw[fill=%s,draw=%s,line width=%s,fill opacity=%.2f] %s -- cycle;"
                % (fill, draw, lw, opacity, pts))

    # ---- context: the whole mesh, tinted by pack -------------------------------------------
    for p in range(pk.n_packs):
        col = PACK_COLORS[p % len(PACK_COLORS)]
        for e in pk.pack_elements[p]:
            out.append(poly(e, "%s!30" % col))
    for p in range(pk.n_packs):
        col = PACK_COLORS[p % len(PACK_COLORS)]
        cx, cy = pack_label_point(pk, p)
        out.append(r"  \node[%s,font=\bfseries\scriptsize,fill=white,inner sep=1.6pt,"
                   r"rounded corners=1pt,draw=%s!60] at (%.2f,%.2f) {$P_%d$};" % (col, col, cx, cy, p))
    out.append(r"  \node[anchor=north,font=\scriptsize\itshape,black!65] at (%.2f,%.2f) "
               r"{pack boundaries follow the element order, not the geometry};"
               % ((min(xs) + max(xs)) / 2.0, min(q[1] for q in m.points) - 0.25))

    # ---- the two detached packs ------------------------------------------------------------
    # Laid out from each pack's OWN bounding box rather than at fractions of the mesh width: the
    # packs are compact but not equal in extent, and placing them by mesh fraction overlapped them.
    ymin = min(q[1] for q in m.points)

    def bbox(p):
        ns = set()
        for e in pk.pack_elements[p]:
            ns.update(m.element_nodes(e))
        ps = [m.points[n] for n in ns]
        return (min(q[0] for q in ps), min(q[1] for q in ps),
                max(q[0] for q in ps), max(q[1] for q in ps))

    bxa, bxb = bbox(a), bbox(b)
    wa, wb = bxa[2] - bxa[0], bxb[2] - bxb[0]
    ha, hb = bxa[3] - bxa[1], bxb[3] - bxb[1]
    drop = 1.25
    # Centre the pair under the mesh, with a gap wide enough for the arcs to read.
    total = wa + gap + wb
    x_start = min(xs) + (width - total) / 2.0
    top_row = ymin - drop
    offs = {
        a: (x_start - bxa[0], top_row - bxa[3]),
        b: (x_start + wa + gap - bxb[0], top_row - bxb[3]),
    }
    label_y = top_row - max(ha, hb) - 0.30

    arcs = []
    for p in (a, b):
        col = PACK_COLORS[p % len(PACK_COLORS)]
        dx, dy = offs[p]
        for e in pk.pack_elements[p]:
            out.append(poly(e, "%s!38" % col, draw="%s!70" % col, lw="0.4pt", dx=dx, dy=dy))
        owned, ghosts = pack_node_ids(pk, p)
        for n in owned:
            x, y = _node_xy(pk, n)
            x, y = x + dx, y + dy
            if pk.shared_new[n]:
                out.append(r"  \draw[%s,fill=%s,line width=0.5pt] (%.3f,%.3f) circle (0.135);"
                           % (col, col, x, y))
                out.append(r"  \draw[black,line width=0.4pt] (%.3f,%.3f) circle (0.205);" % (x, y))
            else:
                out.append(r"  \fill[%s] (%.3f,%.3f) circle (0.115);" % (col, x, y))
        # Ghosts: the OWNER's colour, faded, dashed outline. The dashes are the redundant cue
        # that survives a greyscale print, where the opacity alone would not.
        for n in ghosts:
            ocol = PACK_COLORS[pk.owner_new[n] % len(PACK_COLORS)]
            x, y = _node_xy(pk, n)
            x, y = x + dx, y + dy
            out.append(r"  \filldraw[fill=%s,draw=%s,fill opacity=0.30,draw opacity=0.85,"
                       r"dashed,line width=0.45pt] (%.3f,%.3f) circle (0.135);" % (ocol, ocol, x, y))
            if pk.owner_new[n] in (a, b):
                odx, ody = offs[pk.owner_new[n]]
                ox, oy = _node_xy(pk, n)
                arcs.append(((x, y), (ox + odx, oy + ody), ocol))
        lx, ly = pack_label_point(pk, p, dx, dy)
        out.append(r"  \node[%s,font=\bfseries\scriptsize,fill=white,inner sep=1.6pt,"
                   r"rounded corners=1pt,draw=%s!60] at (%.2f,%.2f) {$P_%d$};" % (col, col, lx, ly, p))

    # ---- the relation: each ghost to the node it duplicates --------------------------------
    for (x0, y0), (x1, y1), col in arcs:
        out.append(r"  \draw[%s,dashed,line width=0.4pt,draw opacity=0.75,->,>=stealth,"
                   r"bend left=14] (%.3f,%.3f) to[bend left=14] (%.3f,%.3f);"
                   % (col, x0, y0, x1, y1))

    out.append(r"\end{tikzpicture}")
    return "\n".join(out) + "\n"


def representative_pack(pk):
    """A pack that exhibits all three zones, which the extreme packs do not.

    Ownership is first-touch in pack order, and that makes both ends degenerate:
      * the LOWEST pack owns every node it touches, so it has no ghosts;
      * the HIGHEST pack owns only nodes no earlier pack reached, so none of what it owns is
        shared and its shared zone is empty.
    Either would draw a two-zone id space and quietly contradict the text beside it. A middle
    pack has both, so the choice maximises the smaller of the two counts and falls back to the
    largest ghost list only if no pack has both."""
    def zones(p):
        nc, ns = pk.n_contiguous(p), pk.n_shared[p]
        return (nc - ns, ns, pk.ghost_ptr[p + 1] - pk.ghost_ptr[p])

    # Maximise the SMALLEST of the three zones, so none of them is a sliver: the bar is drawn to
    # scale and a two-node zone is too narrow to carry its own label, which is what a rule keyed
    # only on shared and ghost counts produced.
    both = [p for p in range(pk.n_packs) if min(zones(p)) > 0]
    if both:
        return max(both, key=lambda p: (min(zones(p)), sum(zones(p))))
    return max(range(pk.n_packs), key=lambda p: pk.ghost_ptr[p + 1] - pk.ghost_ptr[p])


def fig_id_space(pk, p, scale=1.0):
    """F2: one pack's local id space, the three zones, and how an id resolves.

    This is the least obvious property of the format (spec S3) and the one a reader needs in
    order to follow the flush: the first zone can be written without synchronisation by any
    implementation, the second only by one where a pack is a single thread, and the third never
    directly."""
    nc, ns = pk.n_contiguous(p), pk.n_shared[p]
    ng = pk.ghost_ptr[p + 1] - pk.ghost_ptr[p]
    assert ng > 0 and ns > 0, ("fig_id_space needs a pack with both a shared zone and "
                               "ghosts; see representative_pack()")
    total = nc + ng
    w = 12.0
    unit = w / total
    col = PACK_COLORS[p % len(PACK_COLORS)]

    out = [PREAMBLE, COLOR_DEFS, r"\begin{tikzpicture}[scale=%.2f]" % scale]
    x = 0.0
    for label, count, style in (
        (r"exclusively owned", nc - ns, r"%s!45" % col),
        (r"owned, shared", ns, r"%s!85" % col),
        (r"ghost", ng, r"black!12"),
    ):
        if count <= 0:
            continue
        out.append(r"  \fill[%s] (%.3f,0) rectangle (%.3f,0.8);" % (style, x, x + count * unit))
        out.append(r"  \draw[black!55] (%.3f,0) rectangle (%.3f,0.8);" % (x, x + count * unit))
        out.append(r"  \node[font=\scriptsize,align=center] at (%.3f,0.4) {%s\\($%d$)};"
                   % (x + count * unit / 2, label, count))
        x += count * unit

    # The two boundaries that matter, named as the code names them. Anchored east/west at the
    # ends so the outer labels cannot collide with the interior ones when a zone is narrow.
    for at, lab in ((nc - ns, r"$n_c-n_s$"), (nc, r"$n_c$")):
        out.append(r"  \draw[black,line width=0.6pt] (%.3f,-0.12) -- (%.3f,0.92);" % (at * unit, at * unit))
        out.append(r"  \node[font=\scriptsize,below] at (%.3f,-0.14) {%s};" % (at * unit, lab))
    out.append(r"  \node[font=\scriptsize,below right] at (0,-0.14) {$0$};")
    out.append(r"  \node[font=\scriptsize,below left] at (%.3f,-0.14) {$n_{\mathrm{pack}}$};" % w)

    out.append(r"  \node[font=\scriptsize,anchor=west,align=left] at (0,1.55) {"
               r"$\ell < n_c$:\ \ global $=$ \texttt{owned\_nodes\_ptr}$[p]+\ell$\\"
               r"$\ell \ge n_c$:\ \ global $=$ \texttt{ghost\_idx}$[\,$\texttt{ghost\_ptr}$[p]+\ell-n_c\,]$};")
    out.append(r"\end{tikzpicture}")
    return "\n".join(out) + "\n"


def fig_reduction(pk, scale=0.92, max_rows=4):
    """F3: the apply's two phases, framed.

    Both phases end in the same global field, which is the point of drawing them together: the
    owned rows go there directly and the staged ones go there through the gather. The figure
    labels objects and leaves the properties to the text -- what the reduction graph buys is
    argued in the prose, and repeating it on the drawing is words a reader has to step over.
    """
    out = [PREAMBLE, COLOR_DEFS,
           r"\begin{tikzpicture}[scale=%.2f,font=\scriptsize]" % scale]

    # EVERY pack, not just the pair Figure 1 detaches: the reduction graph spans the whole mesh,
    # and a buffer showing one pair's entries would suggest the gather is a per-neighbour exchange.
    # In ghost_buf order, which is pack-major by construction.
    packs = list(range(pk.n_packs))
    entries = []
    for p in packs:
        for k in range(pk.ghost_ptr[p], pk.ghost_ptr[p + 1]):
            entries.append((k, p, pk.ghost_idx[k]))
    rows = []
    for r in range(len(pk.ghost_reduce_dest)):
        ks = pk.ghost_reduce_idx[pk.ghost_reduce_ptr[r]:pk.ghost_reduce_ptr[r + 1]]
        if len(ks) >= 2:
            rows.append((r, ks))
    rows = sorted(rows, key=lambda rk: -len(rk[1]))[:max_rows]
    rows.sort()
    shown = [k for _r, ks in rows for k in ks]

    LANE = 0.0                          # the owned-rows bypass runs down this lane
    L, R = 0.62, 9.0                    # the frames' left and right edges
    y1t, y1b = 0.0, -1.15               # phase 1 band
    ybuf = -1.95                        # the staging buffer, between the phases
    y2t, y2b = -2.75, -4.45             # phase 2 band
    yglob = -4.95                       # the global field, where both phases land

    def frame(yt, yb, label):
        out.append(r"  \draw[black!30,rounded corners=3pt,line width=0.5pt,dash pattern=on 2pt off 2pt] "
                   r"(%.2f,%.2f) rectangle (%.2f,%.2f);" % (L, yb, R, yt))
        out.append(r"  \node[anchor=west,font=\scriptsize\bfseries,black!55,fill=white,"
                   r"inner xsep=3pt] at (%.2f,%.2f) {%s};" % (L + 0.28, yt, label))

    frame(y1t, y1b, "Phase 1")
    frame(y2t, y2b, "Phase 2")

    # ---- phase 1: a private accumulator per pack ------------------------------------------
    bw = (R - L - 1.0) / len(packs) - 0.2
    bx = {}
    # The box's own extent, named once: the label goes at ITS centre. Deriving the label's y from
    # the band instead put it a quarter of the box height above centre, which reads as a caption
    # stuck to the top edge rather than as the box's name.
    box_b, box_t = y1b + 0.28, y1t - 0.34
    for i, p in enumerate(packs):
        col = PACK_COLORS[p % len(PACK_COLORS)]
        x = L + 0.5 + i * (bw + 0.2)
        bx[p] = x + bw / 2.0
        out.append(r"  \draw[%s,fill=%s!18,rounded corners=2pt,line width=0.6pt] "
                   r"(%.2f,%.2f) rectangle (%.2f,%.2f);" % (col, col, x, box_b, x + bw, box_t))
        out.append(r"  \node[%s,font=\bfseries] at (%.2f,%.2f) {$P_%d$};"
                   % (col, x + bw / 2.0, (box_b + box_t) / 2.0, p))
    out.append(r"  \node[anchor=east,font=\tiny,black!55] at (%.2f,%.2f) "
               r"{pack-private accumulators};" % (R - 0.14, y1t - 0.20))

    # ---- the staging buffer ----------------------------------------------------------------
    cw = min(0.52, (R - L - 1.6) / max(1, len(entries)))
    bufx = L + 1.15
    out.append(r"  \node[anchor=east,font=\tiny\ttfamily] at (%.2f,%.2f) {ghost\_buf};"
               % (bufx - 0.12, ybuf + 0.17))
    xof = {}
    for i, (k, p, _dest) in enumerate(entries):
        col = PACK_COLORS[p % len(PACK_COLORS)]
        x = bufx + i * cw
        xof[k] = x + cw / 2.0
        hi = k in shown
        out.append(r"  \draw[%s,fill=%s!%d,line width=0.4pt] (%.3f,%.2f) rectangle (%.3f,%.2f);"
                   % (col, col, 45 if hi else 16, x, ybuf, x + cw, ybuf + 0.34))

    # Ghost contributions leave each pack for its slots; one arrow per pack, to the middle of
    # the run of slots it staged, rather than one per slot -- which drew a thicket.
    for p in packs:
        lo, hi_ = pk.ghost_ptr[p], pk.ghost_ptr[p + 1]
        if lo == hi_:
            continue
        col = PACK_COLORS[p % len(PACK_COLORS)]
        mid = (xof[lo] + xof[hi_ - 1]) / 2.0
        out.append(r"  \draw[->,>=stealth,%s,line width=0.55pt,draw opacity=0.9] "
                   r"(%.2f,%.2f) to[out=-90,in=90] (%.2f,%.2f);"
                   % (col, bx[p], box_b, mid, ybuf + 0.36))
    out.append(r"  \node[anchor=west,font=\tiny,black!60] at (%.2f,%.2f) {ghost contributions};"
               % (bufx + len(entries) * cw + 0.12, ybuf + 0.17))

    # ---- phase 2: one row of the reduction graph per destination ---------------------------
    n = len(rows)
    dxs = []
    for i, (r, ks) in enumerate(rows):
        dest = pk.ghost_reduce_dest[r]
        dcol = PACK_COLORS[pk.owner_new[dest] % len(PACK_COLORS)]
        dw = 1.45
        dx = L + 0.8 + i * ((R - L - 1.6 - dw) / max(1, n - 1) if n > 1 else 0)
        # The drop to the global field leaves from the box's right, so it does not run through
        # the row label centred beneath it.
        dxs.append(dx + dw - 0.18)
        out.append(r"  \draw[%s,fill=%s!22,rounded corners=1.5pt,line width=0.5pt] "
                   r"(%.2f,%.2f) rectangle (%.2f,%.2f);" % (dcol, dcol, dx, y2b + 0.42, dx + dw, y2b + 0.92))
        out.append(r"  \node[font=\tiny] at (%.2f,%.2f) {node %d};" % (dx + dw / 2.0, y2b + 0.67, dest))
        # Set left of the box, not under its centre: the label is about as wide as the box, and
        # the drop to the global field leaves from the box's right end, so anything centred here
        # runs into it. The offset is what keeps the two apart.
        out.append(r"  \node[font=\tiny,black!50,anchor=north west] at (%.2f,%.2f) "
                   r"{row %d: \texttt{ptr}[%d..%d)};"
                   % (dx - 0.38, y2b + 0.38, r, pk.ghost_reduce_ptr[r], pk.ghost_reduce_ptr[r + 1]))
        for k in ks:
            if k in xof:
                out.append(r"  \draw[->,>=stealth,%s,line width=0.45pt,draw opacity=0.8] "
                           r"(%.3f,%.2f) to[out=-90,in=90] (%.2f,%.2f);"
                           % (dcol, xof[k], ybuf - 0.02, dx + dw / 2.0, y2b + 0.94))

    # ---- the global field, reached by both phases ------------------------------------------
    out.append(r"  \draw[black!45,fill=black!5,rounded corners=2pt,line width=0.6pt] "
               r"(%.2f,%.2f) rectangle (%.2f,%.2f);" % (L + 0.5, yglob, R - 0.5, yglob + 0.42))
    out.append(r"  \node[font=\tiny,black!70] at (%.2f,%.2f) {global field};"
               % ((L + R) / 2.0, yglob + 0.21))
    # Owned rows bypass the gather entirely: down the left lane, outside both frames, so the one
    # path that needs no reduction is also the one that touches nothing on its way.
    out.append(r"  \draw[->,>=stealth,black!55,line width=0.7pt] "
               r"(%.2f,%.2f) to[out=180,in=90] (%.2f,%.2f) to[out=-90,in=180] (%.2f,%.2f);"
               % (L + 0.5, (box_b + box_t) / 2.0, LANE, (y1b + yglob) / 2.0, L + 0.52, yglob + 0.21))
    out.append(r"  \node[font=\tiny,black!60,rotate=90,anchor=south] at (%.2f,%.2f) {owned rows};"
               % (LANE - 0.06, (y1b + yglob) / 2.0))
    for dx in dxs:
        out.append(r"  \draw[->,>=stealth,black!45,line width=0.5pt] (%.2f,%.2f) -- (%.2f,%.2f);"
                   % (dx, y2b + 0.40, dx, yglob + 0.44))

    out.append(r"\end{tikzpicture}")
    return "\n".join(out) + "\n"


def fig_scatters(scale=1.0):
    """F4: the three scatter strategies, and what each fixes and costs.

    Deliberately schematic: this is a claim about synchronisation structure, not about any one
    mesh. The measured trade-off lives in the results section; the numbers quoted in the
    annotations are the ones from the layout comparison at 8,586,756 dof."""
    out = [PREAMBLE, COLOR_DEFS, r"\begin{tikzpicture}[scale=%.2f,font=\scriptsize]" % scale]
    panels = [
        ("atomic", 0.0, [
            r"element sweep, flat",
            r"\textcolor{PackD}{\texttt{atomic} per entry}",
            r"global array pre-zeroed",
        ], r"no order fixed $\Rightarrow$ \textcolor{PackD}{not reproducible}"),
        ("packed", 4.4, [
            r"pack-private buffer",
            r"plain \texttt{+=}, no atomic",
            r"owned rows \emph{written}",
            r"ghosts $\to$ CSR gather",
        ], r"order fixed by indices $\Rightarrow$ \textcolor{PackC}{reproducible}"),
        ("colored", 8.8, [
            r"direct global write",
            r"no atomic, no reduction",
            r"\textcolor{PackD}{barrier per colour}",
        ], r"race-free; barriers dominate for vectors"),
    ]
    for name, x0, lines, note in panels:
        out.append(r"  \node[anchor=south,font=\small\bfseries] at (%.2f,2.80) {\texttt{%s}};" % (x0 + 1.95, name))
        out.append(r"  \draw[black!45,rounded corners=2pt] (%.2f,0.25) rectangle (%.2f,2.70);" % (x0, x0 + 3.9))
        y = 2.32
        for ln in lines:
            out.append(r"  \node[anchor=west,align=left] at (%.2f,%.2f) {%s};" % (x0 + 0.18, y, ln))
            y -= 0.42
        out.append(r"  \node[anchor=north,align=center,text width=4.3cm,font=\scriptsize] at (%.2f,0.18) {%s};" % (x0 + 1.95, note))
    out.append(r"\end{tikzpicture}")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------------------------

FIGURES = ("packed_decomposition", "packed_id_space", "packed_reduction", "packed_scatters")


def build(out_dir, nx=9, ny=7, eper=42):
    mesh = TriMesh(nx, ny)
    pk = Packing(mesh, morton_order(mesh), eper)
    os.makedirs(out_dir, exist_ok=True)
    figs = {
        "packed_decomposition": fig_decomposition(pk),
        "packed_id_space": fig_id_space(pk, representative_pack(pk)),
        "packed_reduction": fig_reduction(pk),
        "packed_scatters": fig_scatters(),
    }
    for name, body in figs.items():
        with open(os.path.join(out_dir, name + ".tex"), "w") as fh:
            fh.write(body)
    # The palette as a standalone file. Each figure also carries its own copy so it can be
    # compiled on its own, but the paper inputs this in the preamble: the performance figures
    # use the same colours, and a float may be typeset before the figure that would otherwise
    # have defined them -- which failed the build with "I do not know the key '/tikz/PackA'".
    with open(os.path.join(out_dir, "colors.tex"), "w") as fh:
        fh.write(PREAMBLE + COLOR_DEFS)

    # Counts the paper's prose quotes, as macros, so no number is typed twice.
    with open(os.path.join(out_dir, "packed_counts.tex"), "w") as fh:
        fh.write(PREAMBLE)
        # Element and node counts, not grid dimensions: the mesh is unstructured and has none.
        fh.write(r"\newcommand{\figPackNElems}{%d}" % mesh.nelements + "\n")
        fh.write(r"\newcommand{\figPackElemsPerPack}{%d}" % eper + "\n")
        fh.write(r"\newcommand{\figPackNPacks}{%d}" % pk.n_packs + "\n")
        fh.write(r"\newcommand{\figPackNNodes}{%d}" % mesh.nnodes + "\n")
        fh.write(r"\newcommand{\figPackGhostEntries}{%d}" % len(pk.ghost_idx) + "\n")
        fh.write(r"\newcommand{\figPackReduceRows}{%d}" % len(pk.ghost_reduce_dest) + "\n")
        a, b = adjacent_pair(pk)
        fh.write(r"\newcommand{\figPackDetachedA}{%d}" % a + "\n")
        fh.write(r"\newcommand{\figPackDetachedB}{%d}" % b + "\n")
    return pk, len(figs) + 2


# ---------------------------------------------------------------------------------------------
# Selftest: the format's documented invariants, checked against the model.
# ---------------------------------------------------------------------------------------------

def selftest():
    fails = []

    def check(ok, what, detail=""):
        print("%-62s %s%s" % (what, "OK" if ok else "FAIL", ("  " + detail) if detail and not ok else ""))
        if not ok:
            fails.append(what)

    # Several shapes, including a pack size that does not divide the element count and a
    # lexicographic order, because the invariants must not depend on either.
    for nx, ny, eper, order_name in ((9, 7, 22, "morton"), (9, 7, 25, "morton"), (7, 6, 16, "lex"),
                                     (11, 5, 19, "morton"), (4, 4, 40, "morton")):
        m = TriMesh(nx, ny)
        order = morton_order(m) if order_name == "morton" else list(range(m.nelements))
        pk = Packing(m, order, eper)
        tag = "%dx%d/%d/%s" % (nx, ny, eper, order_name)

        # S2: the owned ranges partition [0, nnodes), monotone, no gaps or overlaps.
        ok = (pk.owned_nodes_ptr[0] == 0 and pk.owned_nodes_ptr[-1] == m.nnodes
              and all(pk.owned_nodes_ptr[i] <= pk.owned_nodes_ptr[i + 1] for i in range(pk.n_packs)))
        check(ok, "[%s] owned ranges partition [0,nnodes) monotonically" % tag)

        # S2: every node is owned by exactly the pack whose range contains it.
        ok = all(pk.owned_nodes_ptr[pk.owner_new[n]] <= n < pk.owned_nodes_ptr[pk.owner_new[n] + 1]
                 for n in range(m.nnodes))
        check(ok, "[%s] each node lies in its owner's range" % tag)

        # S2: local->global round-trips over the whole local id space.
        bad = 0
        for p in range(pk.n_packs):
            for l in range(pk.n_pack_nodes(p)):
                g = pk.local_to_global(p, l)
                if not (0 <= g < m.nnodes):
                    bad += 1
                if l < pk.n_contiguous(p) and pk.owner_new[g] != p:
                    bad += 1
                if l >= pk.n_contiguous(p) and pk.owner_new[g] == p:
                    bad += 1
        check(bad == 0, "[%s] pack_local_to_global resolves every id correctly" % tag, "%d bad" % bad)

        # S3: within an owned range, non-shared ids strictly precede shared ones.
        bad = 0
        for p in range(pk.n_packs):
            base, nc, ns = pk.owned_nodes_ptr[p], pk.n_contiguous(p), pk.n_shared[p]
            for l in range(nc):
                want_shared = l >= nc - ns
                if pk.shared_new[base + l] != want_shared:
                    bad += 1
        check(bad == 0, "[%s] non-shared ids precede shared ones in every pack" % tag, "%d bad" % bad)

        # S3: the zone classification agrees with the arrays.
        ok = all(pk.zone(p, l) == "ghost"
                 for p in range(pk.n_packs)
                 for l in range(pk.n_contiguous(p), pk.n_pack_nodes(p)))
        check(ok, "[%s] ids at or past n_contiguous classify as ghost" % tag)

        # S4: ghost lists are deduplicated, and never contain an owned node.
        ok = True
        for p in range(pk.n_packs):
            g = pk.ghost_idx[pk.ghost_ptr[p]:pk.ghost_ptr[p + 1]]
            ok = ok and len(g) == len(set(g)) and all(pk.owner_new[n] != p for n in g)
        check(ok, "[%s] ghost lists deduplicated and exclude owned nodes" % tag)

        # S4/S5 together: every (pack, node) incidence is reachable, i.e. a pack can address
        # every node its elements touch. This is the property the element sweep depends on.
        bad = 0
        for p in range(pk.n_packs):
            addressable = set(range(pk.owned_nodes_ptr[p], pk.owned_nodes_ptr[p + 1]))
            addressable |= set(pk.ghost_idx[pk.ghost_ptr[p]:pk.ghost_ptr[p + 1]])
            for e in pk.pack_elements[p]:
                for o in m.element_nodes(e):
                    if pk.new_of_old[o] not in addressable:
                        bad += 1
        check(bad == 0, "[%s] every node a pack touches is addressable by it" % tag, "%d bad" % bad)

        # S5: each destination appears in exactly one row, rows ordered by destination.
        d = pk.ghost_reduce_dest
        check(len(d) == len(set(d)), "[%s] each reduction destination appears in exactly one row" % tag)
        check(d == sorted(d), "[%s] reduction rows are ordered by destination" % tag)

        # S5: the reduction consumes every staged ghost entry exactly once. If it did not, a
        # contribution would be silently dropped -- which is the failure mode that would make
        # the operator wrong rather than merely slow.
        consumed = sorted(pk.ghost_reduce_idx)
        check(consumed == list(range(len(pk.ghost_idx))),
              "[%s] the reduction consumes every ghost entry exactly once" % tag)

        # The renumbering is a bijection.
        check(sorted(pk.old_of_new) == list(range(m.nnodes)), "[%s] the renumbering is a bijection" % tag)

    # Determinism of the model itself: same input, same arrays. A generator that is not
    # reproducible cannot illustrate reproducibility.
    m = TriMesh(9, 7)
    a = Packing(m, morton_order(m), 6)
    b = Packing(m, morton_order(m), 6)
    check(a.owned_nodes_ptr == b.owned_nodes_ptr and a.ghost_idx == b.ghost_idx
          and a.ghost_reduce_idx == b.ghost_reduce_idx, "the packing is reproducible")

    # The SFC precondition, as a measurable statement: a space-filling order gives smaller
    # pack node sets than a lexicographic one. This is the figure's justification for saying
    # reordering is a precondition rather than a tuning knob.
    m = TriMesh(16, 14)
    sfc = Packing(m, morton_order(m), 16)
    lex = Packing(m, list(range(m.nelements)), 16)
    mean_sfc = sum(sfc.n_pack_nodes(p) for p in range(sfc.n_packs)) / sfc.n_packs
    mean_lex = sum(lex.n_pack_nodes(p) for p in range(lex.n_packs)) / lex.n_packs
    check(mean_sfc < mean_lex,
          "space-filling order gives smaller pack node sets than lexicographic",
          "sfc %.1f vs lex %.1f" % (mean_sfc, mean_lex))
    print("    (mean nodes per pack: space-filling %.1f, lexicographic %.1f)" % (mean_sfc, mean_lex))

    # Every figure emits non-trivial TikZ.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        pk, n = build(td)
        for name in FIGURES:
            path = os.path.join(td, name + ".tex")
            body = open(path).read()
            check(os.path.exists(path) and r"\begin{tikzpicture}" in body and len(body) > 200,
                  "figure %s emits a tikzpicture" % name)
        check(os.path.exists(os.path.join(td, "packed_counts.tex")), "the counts macros are emitted")

    # ---- the mesh itself -----------------------------------------------------------------
    # The figure argues from a mesh, so a malformed one would undermine it quietly. Delaunay by
    # Bowyer-Watson can degenerate on nearly-cocircular points; these catch that if the point set
    # is ever changed.
    tm = TriMesh()
    edge_count = {}
    for e in range(tm.nelements):
        nn = tm.element_nodes(e)
        for u, v in ((nn[0], nn[1]), (nn[1], nn[2]), (nn[2], nn[0])):
            edge_count[(min(u, v), max(u, v))] = edge_count.get((min(u, v), max(u, v)), 0) + 1
    check(all(c in (1, 2) for c in edge_count.values()),
          "every edge is shared by one or two triangles")
    check(len({tuple(sorted(t)) for t in tm.tris}) == tm.nelements,
          "the triangulation has no duplicate triangles")
    check(tm.nnodes - len(edge_count) + tm.nelements == 1,
          "Euler characteristic is that of a disc",
          "V-E+F = %d" % (tm.nnodes - len(edge_count) + tm.nelements))
    used = {n for e in range(tm.nelements) for n in tm.element_nodes(e)}
    check(len(used) == tm.nnodes, "every node belongs to at least one triangle")
    areas = []
    for e in range(tm.nelements):
        (ax, ay), (bx, by), (cx, cy) = [tm.points[n] for n in tm.element_nodes(e)]
        areas.append(0.5 * ((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)))
    check(all(a > 1e-6 for a in areas),
          "every triangle is counter-clockwise and non-degenerate",
          "min area %.3g" % min(areas))
    check([tuple(round(c, 12) for c in q) for q in TriMesh().points] ==
          [tuple(round(c, 12) for c in q) for q in TriMesh().points],
          "the point set is reproducible")

    # ---- the figures say what the model says ----------------------------------------------
    fm = TriMesh()
    fpk = Packing(fm, morton_order(fm), 22)
    pa, pbb = adjacent_pair(fpk)
    check(pa != pbb and 0 <= pa < fpk.n_packs and 0 <= pbb < fpk.n_packs,
          "the detached pair is two distinct packs")
    # Every arc F1 draws is a real ghost relation, and every such relation is drawn.
    want = set()
    for p in (pa, pbb):
        for k in range(fpk.ghost_ptr[p], fpk.ghost_ptr[p + 1]):
            n = fpk.ghost_idx[k]
            if fpk.owner_new[n] in (pa, pbb):
                want.add((p, n))
    dec = fig_decomposition(fpk)
    check(dec.count("to[bend left") == len(want),
          "F1 draws exactly one arc per ghost relation between the detached packs",
          "arcs %d, relations %d" % (dec.count("to[bend left"), len(want)))
    # Faded nodes: one per ghost id of each detached pack, whatever its owner.
    nghost = sum(fpk.ghost_ptr[p + 1] - fpk.ghost_ptr[p] for p in (pa, pbb))
    check(dec.count("fill opacity=0.30") == nghost,
          "F1 draws exactly one faded node per ghost id",
          "faded %d, ghosts %d" % (dec.count("fill opacity=0.30"), nghost))
    # Every row F3 draws is a real reduction row, drawn with the model's own pointers, and every
    # row it draws sums at least two terms -- a one-term row would illustrate nothing. Counted
    # rather than short-circuited: a tag that never matches would make this check vacuous, which
    # is exactly how the first version of it passed while testing nothing.
    red = fig_reduction(fpk)
    drawn = 0
    for r in range(len(fpk.ghost_reduce_dest)):
        tag = "row %d: " % r
        if tag not in red:
            continue
        drawn += 1
        ks = fpk.ghost_reduce_idx[fpk.ghost_reduce_ptr[r]:fpk.ghost_reduce_ptr[r + 1]]
        check(len(ks) >= 2, "F3 row %d sums several terms" % r)
        check(("[%d..%d)" % (fpk.ghost_reduce_ptr[r], fpk.ghost_reduce_ptr[r + 1])) in red,
              "F3 row %d carries the model's own pointer range" % r)
    check(drawn > 0, "F3 draws at least one reduction row", "drew %d" % drawn)

    print()
    if fails:
        print("%d check(s) FAILED" % len(fails))
        return 1
    print("all packed-format figure checks passed")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "figures"))
    ap.add_argument("--selftest", action="store_true")
    # nx/ny are the jittered lattice the points come from, not a grid of elements: the mesh is
    # an unstructured triangulation of those points.
    ap.add_argument("--nx", type=int, default=9)
    ap.add_argument("--ny", type=int, default=7)
    ap.add_argument("--elements-per-pack", type=int, default=42)
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())

    pk, n = build(args.out, args.nx, args.ny, args.elements_per_pack)
    print("wrote %d figure inputs to %s" % (n, os.path.abspath(args.out)))
    print("  %d packs, %d nodes, %d ghost entries, %d reduction rows"
          % (pk.n_packs, pk.mesh.nnodes, len(pk.ghost_idx), len(pk.ghost_reduce_dest)))


if __name__ == "__main__":
    main()
