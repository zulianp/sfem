#!/usr/bin/env python3
"""TikZ figures for the packed mesh format (paper figures F1-F4).

Stdlib only, following spikes/cvfem/python/cvfem_locality_figures.py: the Alps uenv has no
numpy or matplotlib, and a figure generator that cannot run where the measurements run is a
figure that goes stale.

WHAT THIS MODELS, AND WHY IT IS NOT A DRAWING.

The figures are not hand-placed. This module reimplements the *documented* packing rules from
spikes/cvfem/docs/PACKED_FORMAT.md on a small 2D quadrilateral mesh and draws the result, so
the pictures cannot disagree with the contract they illustrate. The rules implemented:

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
# The model: a structured 2D quad mesh, spatially ordered, then packed.
# ---------------------------------------------------------------------------------------------


class QuadMesh:
    """nx x ny quadrilateral mesh. Nodes are (nx+1) x (ny+1), numbered lexicographically
    before packing; packing renumbers them, exactly as PackedMesh::create does in place."""

    def __init__(self, nx, ny):
        self.nx, self.ny = nx, ny
        self.nnx, self.nny = nx + 1, ny + 1
        self.nnodes = self.nnx * self.nny
        self.nelements = nx * ny

    def node(self, i, j):
        return j * self.nnx + i

    def coord(self, n):
        return (n % self.nnx, n // self.nnx)

    def element_nodes(self, e):
        """Counter-clockwise, the ordering the element kernels assume."""
        i, j = e % self.nx, e // self.nx
        return [self.node(i, j), self.node(i + 1, j), self.node(i + 1, j + 1), self.node(i, j + 1)]

    def element_centre(self, e):
        i, j = e % self.nx, e // self.nx
        return (i + 0.5, j + 0.5)


def morton_order(mesh):
    """A space-filling element order. The format REQUIRES spatial reordering before packing --
    it is a precondition, not an optimisation: without it a pack's elements are scattered and
    its node set is huge. Measured consequence at pack_size 2048 on the real mesh: 2735 nodes
    per pack when space-filling against 4373 when lexicographic, and 1.25x in throughput.

    Morton rather than Hilbert because it is four lines and the figure only needs the order to
    be spatially coherent; the production path uses smesh::SFC."""

    def key(e):
        x, y = e % mesh.nx, e // mesh.nx
        k = 0
        for b in range(16):
            k |= ((x >> b) & 1) << (2 * b)
            k |= ((y >> b) & 1) << (2 * b + 1)
        return k

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
            for v in range(4):  # node-major
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


def fig_decomposition(pk, scale=0.92):
    """F1: the pack decomposition, and the three kinds of node.

    The figure the format section hangs on. Elements are tinted by pack; nodes are drawn by
    role: exclusively owned (solid, in the owner's colour), owned but shared (solid with a
    ring, because it is the node another pack will ghost), and -- in the inset -- what one
    pack sees as a ghost."""
    m = pk.mesh
    out = [PREAMBLE, COLOR_DEFS, r"\begin{tikzpicture}[scale=%.2f]" % scale]

    for p in range(pk.n_packs):
        col = PACK_COLORS[p % len(PACK_COLORS)]
        for e in pk.pack_elements[p]:
            i, j = e % m.nx, e // m.nx
            out.append(r"  \fill[%s!28] (%d,%d) rectangle (%d,%d);" % (col, i, j, i + 1, j + 1))

    out.append(r"  \draw[black!25,very thin] (0,0) grid (%d,%d);" % (m.nx, m.ny))

    # Pack labels at the centroid of each pack's elements, on a white plate: unplated they sat
    # under the node markers and were unreadable.
    for p in range(pk.n_packs):
        col = PACK_COLORS[p % len(PACK_COLORS)]
        cs = [m.element_centre(e) for e in pk.pack_elements[p]]
        cx = sum(c[0] for c in cs) / len(cs)
        cy = sum(c[1] for c in cs) / len(cs)
        out.append(r"  \node[%s,font=\bfseries,fill=white,inner sep=3pt,"
                   r"rounded corners=1pt,draw=%s!60] at (%.2f,%.2f) {$P_%d$};" % (col, col, cx, cy, p))

    for n in range(m.nnodes):
        # Coordinates are a property of the node, so they follow the renumbering.
        x, y = m.coord(pk.old_of_new[n])
        col = PACK_COLORS[pk.owner_new[n] % len(PACK_COLORS)]
        if pk.shared_new[n]:
            out.append(r"  \draw[%s,fill=%s,line width=0.7pt] (%d,%d) circle (0.155);" % (col, col, x, y))
            out.append(r"  \draw[black,line width=0.5pt] (%d,%d) circle (0.235);" % (x, y))
        else:
            out.append(r"  \fill[%s] (%d,%d) circle (0.135);" % (col, x, y))

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
    both = [p for p in range(pk.n_packs)
            if pk.n_shared[p] > 0 and pk.ghost_ptr[p + 1] - pk.ghost_ptr[p] > 0]
    if both:
        return max(both, key=lambda p: min(pk.n_shared[p], pk.ghost_ptr[p + 1] - pk.ghost_ptr[p]))
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


def fig_reduction(pk, scale=1.0, max_rows=4):
    """F3: the ghost reduction as a CSR gather.

    Each destination is exactly one row, so one thread owns it and no atomic is needed -- and
    the row's terms are visited in a fixed index order, which is what makes the reduction
    bit-deterministic rather than merely race-free."""
    rows = min(max_rows, len(pk.ghost_reduce_dest))
    out = [PREAMBLE, COLOR_DEFS, r"\begin{tikzpicture}[scale=%.2f,font=\scriptsize]" % scale]

    out.append(r"  \node[anchor=south,font=\scriptsize\bfseries] at (1.1,%.2f) "
               r"{\texttt{ghost\_buf}};" % (rows * 0.50 + 0.15))
    out.append(r"  \node[anchor=south,font=\scriptsize\bfseries] at (7.4,%.2f) "
               r"{global field};" % (rows * 0.50 + 0.15))

    # Entries, grouped by the row that consumes them, so the gather is visible.
    ypos = {}
    y = rows * 0.50 - 0.3
    for r in range(rows):
        beg, end = pk.ghost_reduce_ptr[r], pk.ghost_reduce_ptr[r + 1]
        for j in range(beg, end):
            entry = pk.ghost_reduce_idx[j]
            # Which pack staged this entry: the one whose ghost range contains it.
            p = max(q for q in range(pk.n_packs) if pk.ghost_ptr[q] <= entry)
            col = PACK_COLORS[p % len(PACK_COLORS)]
            out.append(r"  \draw[%s,fill=%s!25] (0.2,%.2f) rectangle (2.0,%.2f);" % (col, col, y, y + 0.36))
            out.append(r"  \node at (1.1,%.2f) {\texttt{[%d]} from $P_%d$};" % (y + 0.21, entry, p))
            ypos[j] = y + 0.18
            y -= 0.50
        y -= 0.10

    y = rows * 0.50 - 0.3
    for r in range(rows):
        beg, end = pk.ghost_reduce_ptr[r], pk.ghost_reduce_ptr[r + 1]
        mid = sum(ypos[j] for j in range(beg, end)) / (end - beg)
        dest = pk.ghost_reduce_dest[r]
        dcol = PACK_COLORS[pk.owner_new[dest] % len(PACK_COLORS)]
        out.append(r"  \draw[%s,fill=%s!20] (6.4,%.2f) rectangle (8.4,%.2f);" % (dcol, dcol, mid - 0.21, mid + 0.21))
        out.append(r"  \node at (7.4,%.2f) {node $%d$ $\mathrel{+}=$};" % (mid, dest))
        for j in range(beg, end):
            out.append(r"  \draw[->,black!65,line width=0.5pt] (2.05,%.2f) -- (6.35,%.2f);" % (ypos[j], mid))
        y = mid

    # The caption sits below the LOWEST drawn entry, computed rather than guessed -- a fixed
    # offset overlapped the last rows whenever a row had more than one term.
    lowest = min(ypos.values()) if ypos else 0.0
    out.append(r"  \node[anchor=north,align=center,font=\scriptsize] at (4.2,%.2f) {"
               r"one row per destination $\Rightarrow$ no atomic, and\\a fixed summation order "
               r"$\Rightarrow$ bit-deterministic};" % (lowest - 0.45))
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


def build(out_dir, nx=8, ny=8, eper=16):
    mesh = QuadMesh(nx, ny)
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
        fh.write(r"\newcommand{\figPackNx}{%d}" % nx + "\n")
        fh.write(r"\newcommand{\figPackNy}{%d}" % ny + "\n")
        fh.write(r"\newcommand{\figPackElemsPerPack}{%d}" % eper + "\n")
        fh.write(r"\newcommand{\figPackNPacks}{%d}" % pk.n_packs + "\n")
        fh.write(r"\newcommand{\figPackNNodes}{%d}" % mesh.nnodes + "\n")
        fh.write(r"\newcommand{\figPackGhostEntries}{%d}" % len(pk.ghost_idx) + "\n")
        fh.write(r"\newcommand{\figPackReduceRows}{%d}" % len(pk.ghost_reduce_dest) + "\n")
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
    for nx, ny, eper, order_name in ((8, 6, 6, "morton"), (8, 6, 7, "morton"), (6, 6, 4, "lex"),
                                     (10, 4, 5, "morton"), (3, 3, 9, "morton")):
        m = QuadMesh(nx, ny)
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
    m = QuadMesh(8, 6)
    a = Packing(m, morton_order(m), 6)
    b = Packing(m, morton_order(m), 6)
    check(a.owned_nodes_ptr == b.owned_nodes_ptr and a.ghost_idx == b.ghost_idx
          and a.ghost_reduce_idx == b.ghost_reduce_idx, "the packing is reproducible")

    # The SFC precondition, as a measurable statement: a space-filling order gives smaller
    # pack node sets than a lexicographic one. This is the figure's justification for saying
    # reordering is a precondition rather than a tuning knob.
    m = QuadMesh(16, 16)
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
    ap.add_argument("--nx", type=int, default=8)
    ap.add_argument("--ny", type=int, default=8)
    ap.add_argument("--elements-per-pack", type=int, default=16)
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())

    pk, n = build(args.out, args.nx, args.ny, args.elements_per_pack)
    print("wrote %d figure inputs to %s" % (n, os.path.abspath(args.out)))
    print("  %d packs, %d nodes, %d ghost entries, %d reduction rows"
          % (pk.n_packs, pk.mesh.nnodes, len(pk.ghost_idx), len(pk.ghost_reduce_dest)))


if __name__ == "__main__":
    main()
