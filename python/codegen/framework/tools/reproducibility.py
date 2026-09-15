"""Input-output reproducibility for the generated kernels.

Every step of the layering work has been gated on generated source coming back
byte-identical.  That is the strongest possible statement while it holds, and it
settles performance by construction -- identical text compiles to identical
kernels.  It also stops being available the moment a change is *meant* to alter
the text: removing the dead temporaries of OP 11, hoisting the graph validation
of OP 10, reconciling the assembly plan names of OP 13, and every loop-structure
pass the IR is supposed to hold.  For those, byte-identity is not the proof, it
is the thing that must break.

This is the gate that replaces it, and it asks the only question that survives a
deliberate output change: **given the same inputs, does the kernel still produce
the same numbers?**

It differs from ``apply_bench`` in scope rather than in kind.  That tool proves
nine matrix-free apply variants of one material agree with each other and with a
recorded answer, which is a parity check between kernels.  This one drives every
entry point it can bind, across every maintained material, and records what each
one returns -- so a change that alters one kernel's arithmetic is caught even
when there is no sibling kernel to disagree with it.

How a kernel is driven
----------------------

Nothing here knows anything about any particular material.  The generated
manifest lists every C ABI entry point with its full declaration, and the
parameter *names* in those declarations are the ABI that ``plans/streams.py``
defines: ``nelements``, ``elements``, ``points``, ``g_adj<i>``,
``<role>_stride``, ``<field>``, ``<field>_old``, ``<field>_direction``,
``<field>_out``.  Binding a call is therefore a matter of reading names, not of
knowing physics, and a material added tomorrow is driven without touching this
file.

An entry point whose parameters cannot all be bound -- one that wants a sparsity
pattern, a packed partition, or a geometry metric this harness does not build --
is skipped and reported as skipped.  That list is part of the output on purpose:
it says exactly how much of the surface is covered, rather than leaving the
uncovered part invisible.

Inputs are a pure function of the parameter's name and the index, so two runs on
two machines fill the same arrays.  The mesh is a Cartesian grid of hexahedra
whose map is affine, so the affine and isoparametric kernels see the same
geometry and are expected to agree; that is inherited from ``apply_bench`` and
is a real check on geometry routing rather than a convenience.

Usage::

    python -m codegen.framework.tools.reproducibility --material laplace
    python -m codegen.framework.tools.reproducibility --all
    python -m codegen.framework.tools.reproducibility --all --record

``--record`` rewrites the committed baseline and is a deliberate act: it is what
you run *after* deciding that an output change is intended, and the diff it
produces is the reviewable statement of which kernels moved.
"""

import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys

from codegen.framework.plans import conventions
import tempfile


BASELINE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "reproducibility.json"
)

#: Digests are compared as a relative change.  The kernels are deterministic, so
#: a rerun on the same machine reproduces bit-for-bit; this tolerance covers
#: compiler and libm differences across machines, not genuine drift.
TOLERANCE = 1e-9

#: How many entry points each material is allowed to leave undriven, by the
#: reason `_bind` gives.  A kernel it cannot bind is skipped and its digest is
#: never checked, so an unbounded skipped list is a hole in the gate that grows
#: quietly: a renamed parameter, a new type spelling, a signature the binder
#: stops recognising, and the kernel simply leaves the run.  Shrink-only, in the
#: shape of the framework's other ratchets -- lower an entry when a kernel
#: becomes bindable, and never raise one to make a change fit.
SKIPPED_BUDGET = {}

#: The materials the snapshot maintains.  Kept as a list rather than discovered
#: so that a material silently failing to generate is a visible absence.
MATERIALS = (
    "laplace",
    "linear_elasticity",
    "mooney_rivlin_kelvin_voigt_newmark",
    "neohookean_ogden",
    "two_phase_flow",
)

#: Element the grid is built from, and the elements that must be generated with
#: it because their kernels are aliases forwarding to another's.
ELEMENT = "HEX8"
ALIAS_TARGETS = ("PROTEUS_HEX8",)

#: Materials whose kernels are not enabled for HEX8 and the element they take
#: instead.  Taylor-Hood pairs a quadratic velocity mesh with a linear pressure
#: mesh on the same cells, so the grid is the HEX27 one and the pressure nodes
#: are its corners.
ELEMENT_BY_MATERIAL = {
    "poro_hyperelasticity": ("HEX27_HEX8", ()),
    "stokes": ("HEX27_HEX8", ()),
}


#: Set from --element, when a run is checking a family the default grid for a
#: material cannot reach.
ELEMENT_OVERRIDE = None


def _elements_for(material):
    if ELEMENT_OVERRIDE:
        return ELEMENT_OVERRIDE, ()
    element, aliases = ELEMENT_BY_MATERIAL.get(material, (ELEMENT, ALIAS_TARGETS))
    return element, aliases


#: The local node index of each lexicographic position in a HEX27 element,
#: taken from fem.reference.sfem_tensor_hex_shape_index rather than guessed --
#: a wrong ordering would make the isoparametric geometry disagree with the
#: affine adjugate the harness supplies, which the parity check would catch.
HEX27_FROM_LEXICOGRAPHIC = (
    0, 8, 1, 11, 24, 9, 3, 10, 2,
    16, 20, 17, 23, 26, 21, 19, 22, 18,
    4, 12, 5, 15, 25, 13, 7, 14, 6,
)

#: Materials this harness cannot drive, and why.  These are facts about the
#: material rather than defects in the tool, so they are stated rather than
#: left to surface as a failure -- and they are the honest edge of the
#: coverage, which is the number that matters when this gate is used to justify
#: an intended output change.
NOT_COVERED = {}


def _repo_root():
    path = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.isdir(os.path.join(path, "base")) and os.path.isdir(
            os.path.join(path, "python")
        ):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            raise RuntimeError("could not locate the repository root")
        path = parent


def _compiler():
    for candidate in ("mpic++", "mpicxx", "c++"):
        try:
            subprocess.run(
                [candidate, "--version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            return candidate
        except (OSError, subprocess.CalledProcessError):
            continue
    return None


def _include_flags(root, generated_dir):
    dirs = [
        generated_dir,
        # `packed_layout.inc` sits beside this module, shared with the
        # inexact-apply spike.
        os.path.dirname(os.path.abspath(__file__)),
        os.path.join(root, "base"),
        # The simplex CPU kernels reach into SFEM proper: tet4_inline_cpu.hpp
        # includes sortreduce.hpp, which lives here.  Without it the harness
        # cannot drive TET4 at all, which is how a whole kernel family came to
        # be outside every gate -- see ARCHITECTURE.html OP 17.
        os.path.join(root, "algebra"),
        os.path.join(root, "build"),
        os.path.join(root, "build", "external", "smesh"),
        os.path.join(root, "external", "smesh", "src"),
    ]
    for material_dir in sorted(glob.glob(os.path.join(generated_dir, "*"))):
        if os.path.isdir(material_dir):
            dirs.append(material_dir)
            for current, subdirs, _files in os.walk(material_dir):
                subdirs[:] = [d for d in subdirs if d not in ("cuda", "CMakeFiles")]
                dirs.append(current)
    operators_root = os.path.join(root, "operators")
    if os.path.isdir(operators_root):
        for current, subdirs, _files in os.walk(operators_root):
            subdirs[:] = [d for d in subdirs if d not in ("cuda", "CMakeFiles")]
            dirs.append(current)
    smesh_src = os.path.join(root, "external", "smesh", "src")
    if os.path.isdir(smesh_src):
        for current, subdirs, _files in os.walk(smesh_src):
            subdirs[:] = [d for d in subdirs if d not in ("cuda", "CMakeFiles")]
            dirs.append(current)
    return ["-I" + d for d in dirs if os.path.isdir(d)]


# --------------------------------------------------------------------------
# Reading the manifest
# --------------------------------------------------------------------------

#: A parameter is a type, a name, and optionally an array extent.  The extent
#: matters: a Taylor-Hood kernel takes its velocity components as
#: ``const double *const RSTR u_data[3]`` -- an array of pointers, not
#: a pointer -- and calling it needs three buffers and a stack array holding
#: their addresses.
_PARAM = re.compile(
    r"^\s*(?P<type>.+?)\s*(?P<name>[A-Za-z_]\w*)\s*(?:\[(?P<extent>\d+)\])?\s*$"
)


def _parameters(declaration):
    """The (type, name, extent) triples of one declaration, in order.

    ``extent`` is 0 for a plain parameter and the array length for one declared
    as ``name[N]``.
    """
    inner = declaration[declaration.index("(") + 1 : declaration.rindex(")")]
    params = []
    for part in inner.split(","):
        part = " ".join(part.split())
        if not part:
            continue
        match = _PARAM.match(part)
        if match is None:
            return None
        extent = match.group("extent")
        params.append(
            (match.group("type").strip(), match.group("name"), int(extent or 0))
        )
    return params


def _manifest_entries(generated_dir, material):
    pattern = os.path.join(generated_dir, material, "op", "*_manifest.json")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return []
    with open(matches[0], encoding="utf-8") as handle:
        manifest = json.load(handle)
    return manifest.get("c_abi", [])


# --------------------------------------------------------------------------
# Binding a call
# --------------------------------------------------------------------------

#: Scalar element types the harness can pass a value for.
SCALARS = ("const double", "const float", "const real_t", "const int")

#: How many trial step lengths a `value_steps` kernel is driven with.  More than
#: one, so the per-step layout is exercised rather than just the first slot.
#: Elements per pack.  Small enough that a pack's node set fits the uint16_t
#: the ABI indexes with, and large enough that the gather and scatter through
#: thread scratch are worth their setup.  SFEM's own packed meshes use a few
#: thousand; this is sized for the refinement this harness drives.
PACK_SIZE = 512

N_STEPS = 3

#: The restrict qualifier the generator emits.  Read from the naming table
#: rather than spelled here: these tables match parameter declarations by exact
#: text, so a qualifier that moves without them turns every kernel into a
#: `skipped` entry -- which is a hole in the gate, not a failure of it.
_RSTR = conventions.QUALIFIERS["restrict"]

#: Geometry parameters are bound structurally -- to mesh data, not to a seeded
#: value -- so their names are free to change.  What is not free is this binder
#: falling behind them: a parameter it stops recognising becomes a `skipped`
#: kernel whose digest is no longer checked.  The patterns are built from the
#: naming table so the two cannot drift.
_ADJ_PATTERN = r"%s(\d+)" % re.escape(conventions.abi_geometry_name("adj").rstrip("0123456789"))
_MET_PATTERN = r"%s(\d+)" % re.escape(conventions.abi_geometry_name("geom_metric").rstrip("0123456789"))
_MET_AOS = conventions.abi_geometry_name("geom_metric")

#: Pointer element types that name a field the kernel reads or writes.
#: The most components an element tangent can have: 3x3x3x3 with no symmetry.
#: The store is over-allocated to this rather than sized from the plan, because
#: the plan's answer depends on the material's flux form and this harness reads
#: only emitted text.  Over-allocating is safe -- the kernel touches the slots it
#: was generated for -- and the untouched tail is deterministic zeros, so the
#: digest stays stable and still moves when the kernel does.
TANGENT_SLOTS = 81

#: The stored element tangent, which is element-indexed rather than node-indexed
#: and therefore sized and seeded differently from every other buffer here.  The
#: compressed store is deliberately absent: it is `half_t`, which this driver has
#: no arithmetic for, and no Op calls that entry point.
STORE_IN_FIELDS = ("const metric_tensor_t *const %s" % _RSTR,)
STORE_OUT_FIELDS = ("metric_tensor_t *const %s" % _RSTR,)

IN_FIELDS = (
    "const double *const %s" % _RSTR,
    "const float *const %s" % _RSTR,
    "const real_t *const %s" % _RSTR,
    # A runtime-typed entry point takes its buffers as void and is told their
    # type by a separate parameter.  The harness drives those at
    # SMESH_FLOAT64, so the buffer it allocates is a double one and the digest
    # is directly comparable with the baseline recorded before the ABI carried
    # its type at run time.  See ARCHITECTURE.html OP 17.
    "const void *const %s" % _RSTR,
)
OUT_FIELDS = (
    "double *const %s" % _RSTR,
    "float *const %s" % _RSTR,
    "real_t *const %s" % _RSTR,
    "void *const %s" % _RSTR,
)

#: What the harness asks a runtime-typed entry point for.
RUNTIME_TYPE_CTYPE = "const enum smesh::PrimitiveType"
RUNTIME_TYPE_VALUE = "smesh::SMESH_FLOAT64"


def _field_argument(buffer, ctype, extent, const):
    """How the harness hands one field buffer to the kernel.

    An array-of-pointer parameter takes the array; everything else takes
    ``.data()``.  A void parameter additionally needs the cast the type system
    no longer does implicitly for an array of pointers.
    """
    value = buffer if extent else "%s.data()" % buffer
    if "void" not in ctype:
        return value
    qualifier = "const " if const else ""
    if extent:
        return "(%svoid *const *)%s" % (qualifier, value)
    return "(%svoid *)%s" % (qualifier, value)


def _scalar_type(ctype):
    if "void" in ctype:
        # Driven at SMESH_FLOAT64, so the buffer behind the void is a double.
        return "double"
    # The store's own type, before the generic scalars: `metric_tensor_t` is a
    # `float` typedef and matching on the word would size the buffer right and
    # name the type wrong.
    for name in ("metric_tensor_t",):
        if name in ctype:
            return name
    for name in ("double", "float", "real_t"):
        if name in ctype:
            return name
    return None


class Unbindable(Exception):
    """A parameter this harness has no deterministic value for."""


def _bind(params, element, components, block_values=None):
    """One C argument expression per parameter, plus the outputs to digest.

    ``components`` is how many field values each node carries in this entry
    point's layout: one for a structure-of-arrays kernel, which takes a stride
    and one pointer per field, and the field count for an array-of-structures
    kernel, which takes a single interleaved array.  Getting it wrong is not a
    wrong answer but a buffer overrun, which is how the first version of this
    crashed on the vector-valued materials.

    Raises ``Unbindable`` naming the first parameter it cannot supply, which is
    what puts an entry point on the skipped list instead of into the baseline.
    """
    args = []
    inputs = []
    outputs = []
    scratch = []
    # A packed kernel is driven over the renumbered mesh the packed layout
    # implies, so every mesh-shaped argument comes from there instead.  Which
    # one it is, is read off the signature rather than guessed: `n_packs` is a
    # parameter no unpacked kernel has.
    packed = any(name == "n_packs" for _t, name, _e in params)
    mesh = "packed.mesh" if packed else "mesh"
    for ctype, name, extent in params:
        if name == "element_type":
            args.append("smesh::ElemType::%s" % element)
        elif name == "nelements":
            args.append("%s.nelements" % mesh)
        elif name == "nnodes":
            args.append("%s.nnodes" % mesh)
        elif name == "n_packs":
            args.append("packed.n_packs")
        elif name == "n_elements_per_pack":
            args.append("packed.n_elements_per_pack")
        elif name == "max_nodes_per_pack":
            args.append("packed.max_nodes_per_pack")
        elif name == "owned_nodes_ptr":
            args.append("packed.owned_nodes_ptr.data()")
        elif name == "n_shared_nodes":
            args.append("packed.n_shared_nodes.data()")
        elif name == "ghost_ptr":
            args.append("packed.ghost_ptr.data()")
        elif name == "ghost_idx":
            args.append("packed.ghost_idx.data()")
        elif name == "n_ghost_entries":
            args.append("packed.n_ghost_entries")
        elif name == "n_ghost_reduce_rows":
            args.append("packed.n_ghost_reduce_rows")
        elif name == "ghost_reduce_ptr":
            args.append("packed.ghost_reduce_ptr.data()")
        elif name == "ghost_reduce_idx":
            args.append("packed.ghost_reduce_idx.data()")
        elif name == "ghost_reduce_dest":
            args.append("packed.ghost_reduce_dest.data()")
        elif name == "ghost_buf":
            # Scratch, not an answer: the two-pass apply stages each pack's
            # ghost contributions here and reduces them into the output in a
            # second pass.  Sized `components * n_ghost_entries`, which is how
            # the kernel strides it, and deliberately not digested -- what it
            # holds afterwards is an implementation detail, and the output it
            # was reduced into is the thing to compare.
            scalar = _scalar_type(ctype)
            buffer = "scratch_%s_%s" % (scalar, name)
            scratch.append((buffer, scalar, components))
            args.append(
                "(%s *)%s.data()" % (scalar, buffer)
                if ctype.startswith("void")
                else "%s.data()" % buffer
            )
        elif name == "elements" and ctype.startswith("uint16_t **"):
            args.append("packed.element_ptrs.data()")
        elif name == "elements" and ctype.startswith("idx_t **"):
            args.append("%s.element_ptrs.data()" % mesh)
        elif name == "points" and "*const *const" in ctype:
            args.append("%s.point_ptrs.data()" % mesh)
        elif re.fullmatch(_ADJ_PATTERN, name or ""):
            index = int(re.fullmatch(_ADJ_PATTERN, name).group(1))
            args.append("%s.adjugate[%d].data()" % (mesh, index))
        elif name == conventions.abi_geometry_name("det0"):
            args.append("%s.determinant.data()" % mesh)
        elif re.fullmatch(_MET_PATTERN, name or ""):
            index = int(re.fullmatch(_MET_PATTERN, name).group(1))
            args.append("%s.metric[%d].data()" % (mesh, index))
        elif name == _MET_AOS:
            args.append("%s.metric_aos.data()" % mesh)
        elif "PrimitiveType" in ctype:
            args.append(RUNTIME_TYPE_VALUE)
        elif name == "tangent_component_stride" and ctype == "const ptrdiff_t":
            # Not 1.  The store is component-major over the elements, so this is
            # the distance between one component's run and the next; binding it
            # like every other stride would land all 81 components on top of each
            # other and the kernel would still run.
            args.append("%s.nelements" % mesh)
        elif name.endswith("_stride") and ctype == "const ptrdiff_t":
            args.append("1")
        elif name == "nsteps" and ctype == "const int":
            # The line-search step count.  It fell through to `material_scalar`,
            # which is a seeded double: truncated to an int it came out zero, so
            # every `objective_steps` kernel returned before its first loop and
            # digested as exactly 0.0 -- for every material, since this session
            # began.  The 0-form was generated, compiled, and never once
            # checked against a number.
            args.append(str(N_STEPS))
        elif ctype in SCALARS:
            args.append("material_scalar(\"%s\")" % name)
        elif ctype in OUT_FIELDS and name == "values":
            # A matrix value array, not a field.  How many entries it holds is
            # the assembly format's decision: BSR stores one n_fields x n_fields
            # block per graph entry, which the mesh's own adjacency graph gives.
            # The other formats index it differently -- block-diagonal-symmetric
            # writes dim*(dim+1)/2 per node -- and sizing one of those as if it
            # were another is not a wrong answer but a buffer overrun, so only
            # the layout this harness actually models is accepted.
            if block_values is None:
                raise Unbindable("matrix value array '%s': layout not modelled" % name)
            scalar = _scalar_type(ctype)
            buffer = "out_%s_%s" % (scalar, name)
            outputs.append((buffer, scalar, name, 0, 0, block_values))
            args.append("%s.data()" % buffer)
        elif name == "rowptr" and "count_t" in ctype:
            args.append("graph.rowptr.data()")
        elif name == "colidx" and "idx_t" in ctype:
            args.append("graph.colidx.data()")
        elif ctype in STORE_OUT_FIELDS:
            # Element-indexed, so its length is the element count and not the
            # node count, and it is seeded by index rather than through the
            # packed permutation -- packing renumbers nodes and never elements,
            # so element e is element e in both layouts.
            scalar = _scalar_type(ctype)
            buffer = "out_%s_%s" % (scalar, name)
            length = "(size_t)%s.nelements * %d" % (mesh, TANGENT_SLOTS)
            outputs.append((buffer, scalar, name, components, extent, length))
            args.append("%s.data()" % buffer)
        elif ctype in STORE_IN_FIELDS:
            scalar = _scalar_type(ctype)
            buffer = "in_%s_%s" % (scalar, name)
            length = "(size_t)%s.nelements * %d" % (mesh, TANGENT_SLOTS)
            inputs.append((buffer, scalar, name, components, extent, length))
            args.append("%s.data()" % buffer)
        elif ctype in OUT_FIELDS:
            scalar = _scalar_type(ctype)
            buffer = "out_%s_%s" % (scalar, name)
            outputs.append((buffer, scalar, name, components, extent, None))
            args.append(
                _field_argument(buffer, ctype, extent, const=False)
            )
        elif ctype in IN_FIELDS:
            scalar = _scalar_type(ctype)
            buffer = "in_%s_%s" % (scalar, name)
            inputs.append((buffer, scalar, name, components, extent, None))
            args.append(
                _field_argument(buffer, ctype, extent, const=True)
            )
        else:
            raise Unbindable("%s %s" % (ctype, name))
    if not outputs:
        raise Unbindable("no output to digest")
    return args, inputs, outputs, scratch, packed


# --------------------------------------------------------------------------
# The driver
# --------------------------------------------------------------------------

DRIVER_HEAD = r"""
// Generated by codegen.framework.tools.reproducibility.  Do not edit.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "sfem_base.hpp"
#include "smesh_elem_type.hpp"

// The generated declarations this driver links against use the generator's
// short restrict qualifier; the driver is standalone C++ and has to agree.
#ifndef RSTR
#define RSTR SFEM_RESTRICT
#endif
#include "smesh_types.hpp"
#include "tet4_inline_cpu.hpp"
#include "tri3_inline_cpu.hpp"

// The grid, its affine geometry, and the deterministic fills are shared with
// codegen.framework.tools.apply_bench: a Cartesian grid of hexahedra whose map
// is affine, so the affine and isoparametric kernels see the same geometry.
struct Mesh {
    ptrdiff_t nelements = 0, nnodes = 0;
    double h = 0.0;
    std::vector<std::vector<idx_t>> elements;
    std::vector<std::vector<geom_t>> points;
    std::vector<idx_t *> element_ptrs;
    std::vector<const geom_t *> point_ptrs;
    std::vector<std::vector<geom_t>> adjugate;
    std::vector<geom_t> determinant;
    // The symmetric gradient metric SFEM calls fff, in both layouts the
    // generated kernels ask for: six separate component arrays, and one
    // interleaved array indexed [element * 6 + component].  Filled by calling
    // SFEM's own tet4_fff rather than by reimplementing it here, so the
    // convention is theirs and cannot drift from the kernels being checked.
    std::vector<std::vector<geom_t>> metric;
    std::vector<geom_t> metric_aos;
};

static Mesh build_grid(int n) {
    Mesh m;
    const int nn = n + 1;
    m.h = 1.0 / (double)n;
    m.nnodes = (ptrdiff_t)nn * nn * nn;
    m.nelements = (ptrdiff_t)n * n * n;
    m.elements.assign(8, std::vector<idx_t>(m.nelements));
    m.points.assign(3, std::vector<geom_t>(m.nnodes));
    auto nid = [&](int i, int j, int k) { return (idx_t)((k * nn + j) * nn + i); };
    for (int k = 0; k < nn; ++k)
        for (int j = 0; j < nn; ++j)
            for (int i = 0; i < nn; ++i) {
                const idx_t id = nid(i, j, k);
                m.points[0][id] = (geom_t)(i * m.h);
                m.points[1][id] = (geom_t)(j * m.h);
                m.points[2][id] = (geom_t)(k * m.h);
            }
    ptrdiff_t e = 0;
    for (int k = 0; k < n; ++k)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i, ++e) {
                m.elements[0][e] = nid(i, j, k);
                m.elements[1][e] = nid(i + 1, j, k);
                m.elements[2][e] = nid(i + 1, j + 1, k);
                m.elements[3][e] = nid(i, j + 1, k);
                m.elements[4][e] = nid(i, j, k + 1);
                m.elements[5][e] = nid(i + 1, j, k + 1);
                m.elements[6][e] = nid(i + 1, j + 1, k + 1);
                m.elements[7][e] = nid(i, j + 1, k + 1);
            }
    for (auto &row : m.elements) m.element_ptrs.push_back(row.data());
    for (auto &row : m.points) m.point_ptrs.push_back(row.data());
    m.adjugate.assign(9, std::vector<geom_t>(m.nelements, (geom_t)0));
    m.determinant.assign(m.nelements, (geom_t)(m.h * m.h * m.h));
    for (ptrdiff_t i = 0; i < m.nelements; ++i) {
        m.adjugate[0][i] = (geom_t)(m.h * m.h);
        m.adjugate[4][i] = (geom_t)(m.h * m.h);
        m.adjugate[8][i] = (geom_t)(m.h * m.h);
    }
    return m;
}

// The quadratic grid: the same cells, with nodes on a lattice of twice the
// resolution.  The cell map is still affine, so the adjugate and determinant
// are the same constants, and the isoparametric kernels must still agree with
// the affine ones -- which is what checks the local node ordering below.
static Mesh build_grid_hex27(int n) {
    Mesh m;
    static const int local_of[27] = {
        0, 8, 1, 11, 24, 9, 3, 10, 2,
        16, 20, 17, 23, 26, 21, 19, 22, 18,
        4, 12, 5, 15, 25, 13, 7, 14, 6
    };
    const int nn = 2 * n + 1;
    m.h = 1.0 / (double)n;
    const double hh = m.h / 2.0;
    m.nnodes = (ptrdiff_t)nn * nn * nn;
    m.nelements = (ptrdiff_t)n * n * n;
    m.elements.assign(27, std::vector<idx_t>(m.nelements));
    m.points.assign(3, std::vector<geom_t>(m.nnodes));
    auto nid = [&](int i, int j, int k) { return (idx_t)((k * nn + j) * nn + i); };
    for (int k = 0; k < nn; ++k)
        for (int j = 0; j < nn; ++j)
            for (int i = 0; i < nn; ++i) {
                const idx_t id = nid(i, j, k);
                m.points[0][id] = (geom_t)(i * hh);
                m.points[1][id] = (geom_t)(j * hh);
                m.points[2][id] = (geom_t)(k * hh);
            }
    ptrdiff_t e = 0;
    for (int k = 0; k < n; ++k)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i, ++e)
                for (int sz = 0; sz < 3; ++sz)
                    for (int sy = 0; sy < 3; ++sy)
                        for (int sx = 0; sx < 3; ++sx) {
                            const int local = local_of[sx + 3 * (sy + 3 * sz)];
                            m.elements[local][e] = nid(2 * i + sx, 2 * j + sy, 2 * k + sz);
                        }
    for (auto &row : m.elements) m.element_ptrs.push_back(row.data());
    for (auto &row : m.points) m.point_ptrs.push_back(row.data());
    m.adjugate.assign(9, std::vector<geom_t>(m.nelements, (geom_t)0));
    m.determinant.assign(m.nelements, (geom_t)(m.h * m.h * m.h));
    for (ptrdiff_t i = 0; i < m.nelements; ++i) {
        m.adjugate[0][i] = (geom_t)(m.h * m.h);
        m.adjugate[4][i] = (geom_t)(m.h * m.h);
        m.adjugate[8][i] = (geom_t)(m.h * m.h);
    }
    return m;
}

// A tetrahedral grid: the same cube lattice, each cell split into six
// tetrahedra by the Freudenthal subdivision about the main diagonal.  Unlike
// the hexahedral grids the cells are not all alike, so the adjugate and
// determinant are computed per element from its own vertices rather than being
// the same constant everywhere -- which makes this a stronger check of the
// geometry paths, not a weaker one.
static Mesh build_grid_tet4(int n) {
    Mesh m;
    const int nn = n + 1;
    m.h = 1.0 / (double)n;
    m.nnodes = (ptrdiff_t)nn * nn * nn;
    m.nelements = (ptrdiff_t)n * n * n * 6;
    m.elements.assign(4, std::vector<idx_t>(m.nelements));
    m.points.assign(3, std::vector<geom_t>(m.nnodes));
    auto nid = [&](int i, int j, int k) { return (idx_t)((k * nn + j) * nn + i); };
    for (int k = 0; k < nn; ++k)
        for (int j = 0; j < nn; ++j)
            for (int i = 0; i < nn; ++i) {
                const idx_t id = nid(i, j, k);
                m.points[0][id] = (geom_t)(i * m.h);
                m.points[1][id] = (geom_t)(j * m.h);
                m.points[2][id] = (geom_t)(k * m.h);
            }
    // The six tetrahedra of a cube, as corner bit patterns (bit0=x, 1=y, 2=z).
    static const int tets[6][4] = {
        {0, 1, 3, 7}, {0, 1, 5, 7}, {0, 4, 5, 7},
        {0, 4, 6, 7}, {0, 2, 6, 7}, {0, 2, 3, 7},
    };
    m.adjugate.assign(9, std::vector<geom_t>(m.nelements, (geom_t)0));
    m.determinant.assign(m.nelements, (geom_t)0);
    ptrdiff_t e = 0;
    for (int k = 0; k < n; ++k)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                for (int t = 0; t < 6; ++t, ++e) {
                    idx_t v[4];
                    for (int c = 0; c < 4; ++c) {
                        const int bits = tets[t][c];
                        v[c] = nid(i + (bits & 1), j + ((bits >> 1) & 1), k + ((bits >> 2) & 1));
                    }
                    double J[9];
                    for (int c = 0; c < 3; ++c)
                        for (int d = 0; d < 3; ++d)
                            J[d * 3 + c] = (double)m.points[d][v[c + 1]] - (double)m.points[d][v[0]];
                    double det = J[0] * (J[4] * J[8] - J[5] * J[7])
                               - J[1] * (J[3] * J[8] - J[5] * J[6])
                               + J[2] * (J[3] * J[7] - J[4] * J[6]);
                    if (det < 0.0) {           // keep every element positively oriented
                        const idx_t swap = v[1]; v[1] = v[2]; v[2] = swap;
                        for (int c = 0; c < 3; ++c)
                            for (int d = 0; d < 3; ++d)
                                J[d * 3 + c] = (double)m.points[d][v[c + 1]] - (double)m.points[d][v[0]];
                        det = -det;
                    }
                    for (int c = 0; c < 4; ++c) m.elements[c][e] = v[c];
                    // adj(J) = det(J) * J^-1, written row-major.
                    m.adjugate[0][e] = (geom_t)(J[4] * J[8] - J[5] * J[7]);
                    m.adjugate[1][e] = (geom_t)(J[2] * J[7] - J[1] * J[8]);
                    m.adjugate[2][e] = (geom_t)(J[1] * J[5] - J[2] * J[4]);
                    m.adjugate[3][e] = (geom_t)(J[5] * J[6] - J[3] * J[8]);
                    m.adjugate[4][e] = (geom_t)(J[0] * J[8] - J[2] * J[6]);
                    m.adjugate[5][e] = (geom_t)(J[2] * J[3] - J[0] * J[5]);
                    m.adjugate[6][e] = (geom_t)(J[3] * J[7] - J[4] * J[6]);
                    m.adjugate[7][e] = (geom_t)(J[1] * J[6] - J[0] * J[7]);
                    m.adjugate[8][e] = (geom_t)(J[0] * J[4] - J[1] * J[3]);
                    m.determinant[e] = (geom_t)det;
                }
    m.metric.assign(6, std::vector<geom_t>(m.nelements, (geom_t)0));
    m.metric_aos.assign((size_t)m.nelements * 6, (geom_t)0);
    for (ptrdiff_t e = 0; e < m.nelements; ++e) {
        geom_t fff[6];
        tet4_fff(m.points[0][m.elements[0][e]], m.points[0][m.elements[1][e]],
                 m.points[0][m.elements[2][e]], m.points[0][m.elements[3][e]],
                 m.points[1][m.elements[0][e]], m.points[1][m.elements[1][e]],
                 m.points[1][m.elements[2][e]], m.points[1][m.elements[3][e]],
                 m.points[2][m.elements[0][e]], m.points[2][m.elements[1][e]],
                 m.points[2][m.elements[2][e]], m.points[2][m.elements[3][e]],
                 fff);
        for (int k = 0; k < 6; ++k) {
            m.metric[k][e] = fff[k];
            m.metric_aos[(size_t)e * 6 + k] = fff[k];
        }
    }
    for (auto &row : m.elements) m.element_ptrs.push_back(row.data());
    for (auto &row : m.points) m.point_ptrs.push_back(row.data());
    return m;
}

// The two-dimensional grids.  Everything above builds a cube, so a kernel for
// a triangle or a quadrilateral had no mesh to run on: `--element TRI3` fell
// through to the hexahedral builder, was handed a `metric` array that builder
// never fills, and segmentation-faulted before its first element.  Half the
// elements the framework generates for were outside the gate because of that,
// which is why the 2D families could not be changed with evidence.
//
// The layout mirrors the 3D builders exactly -- `points` has two rows, the
// adjugate has four components and the symmetric metric three -- because the
// binding code indexes them by position and knows nothing about dimension.

static void fill_metric_tri3(Mesh &m) {
    m.metric.assign(3, std::vector<geom_t>(m.nelements, (geom_t)0));
    m.metric_aos.assign((size_t)m.nelements * 3, (geom_t)0);
    for (ptrdiff_t e = 0; e < m.nelements; ++e) {
        jacobian_t fff[3];
        // SFEM's own tri3_fff, for the same reason the tetrahedral builder
        // calls tet4_fff: the convention is theirs and cannot drift from the
        // kernels being checked.
        tri3_fff(m.points[0][m.elements[0][e]], m.points[0][m.elements[1][e]],
                 m.points[0][m.elements[2][e]],
                 m.points[1][m.elements[0][e]], m.points[1][m.elements[1][e]],
                 m.points[1][m.elements[2][e]],
                 fff);
        // The two references do not agree on one factor: tet4_fff folds the
        // reference tetrahedron's measure into what it returns -- the 1/6
        // visible in its body -- and tri3_fff returns J^-1 J^-T det(J) with no
        // reference area at all.  The generated kernels take the weighted
        // form, so the triangle's 1/2 is applied here.  This is not a guess:
        // without it the affine kernels answered exactly twice their
        // isoparametric counterparts, which compute their geometry from the
        // node coordinates alone, and the parity check is what says so.
        for (int k = 0; k < 3; ++k) {
            const geom_t weighted = (geom_t)(0.5 * (double)fff[k]);
            m.metric[k][e] = weighted;
            m.metric_aos[(size_t)e * 3 + k] = weighted;
        }
    }
}

// A triangular grid: the unit square split into n x n squares, each cut along
// its rising diagonal into two triangles.  As with the tetrahedra the cells
// are not all alike under the affine map, so the adjugate and determinant are
// computed per element.
static Mesh build_grid_tri3(int n) {
    Mesh m;
    const int nn = n + 1;
    m.h = 1.0 / (double)n;
    m.nnodes = (ptrdiff_t)nn * nn;
    m.nelements = (ptrdiff_t)n * n * 2;
    m.elements.assign(3, std::vector<idx_t>(m.nelements));
    m.points.assign(2, std::vector<geom_t>(m.nnodes));
    auto nid = [&](int i, int j) { return (idx_t)(j * nn + i); };
    for (int j = 0; j < nn; ++j)
        for (int i = 0; i < nn; ++i) {
            const idx_t id = nid(i, j);
            m.points[0][id] = (geom_t)(i * m.h);
            m.points[1][id] = (geom_t)(j * m.h);
        }
    static const int tris[2][3][2] = {
        {{0, 0}, {1, 0}, {1, 1}},
        {{0, 0}, {1, 1}, {0, 1}},
    };
    m.adjugate.assign(4, std::vector<geom_t>(m.nelements, (geom_t)0));
    m.determinant.assign(m.nelements, (geom_t)0);
    ptrdiff_t e = 0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            for (int t = 0; t < 2; ++t, ++e) {
                idx_t v[3];
                for (int c = 0; c < 3; ++c)
                    v[c] = nid(i + tris[t][c][0], j + tris[t][c][1]);
                double J[4];
                for (int c = 0; c < 2; ++c)
                    for (int d = 0; d < 2; ++d)
                        J[d * 2 + c] = (double)m.points[d][v[c + 1]] - (double)m.points[d][v[0]];
                double det = J[0] * J[3] - J[1] * J[2];
                if (det < 0.0) {           // keep every element positively oriented
                    const idx_t swap = v[1]; v[1] = v[2]; v[2] = swap;
                    for (int c = 0; c < 2; ++c)
                        for (int d = 0; d < 2; ++d)
                            J[d * 2 + c] = (double)m.points[d][v[c + 1]] - (double)m.points[d][v[0]];
                    det = -det;
                }
                for (int c = 0; c < 3; ++c) m.elements[c][e] = v[c];
                // adj(J) = det(J) * J^-1, written row-major.
                m.adjugate[0][e] = (geom_t)J[3];
                m.adjugate[1][e] = (geom_t)(-J[1]);
                m.adjugate[2][e] = (geom_t)(-J[2]);
                m.adjugate[3][e] = (geom_t)J[0];
                m.determinant[e] = (geom_t)det;
            }
    fill_metric_tri3(m);
    for (auto &row : m.elements) m.element_ptrs.push_back(row.data());
    for (auto &row : m.points) m.point_ptrs.push_back(row.data());
    return m;
}

// A quadrilateral grid: the same squares, uncut.  The map is affine and the
// same everywhere, so the adjugate and determinant are constants -- the 2D
// counterpart of build_grid.
static Mesh build_grid_quad4(int n) {
    Mesh m;
    const int nn = n + 1;
    m.h = 1.0 / (double)n;
    m.nnodes = (ptrdiff_t)nn * nn;
    m.nelements = (ptrdiff_t)n * n;
    m.elements.assign(4, std::vector<idx_t>(m.nelements));
    m.points.assign(2, std::vector<geom_t>(m.nnodes));
    auto nid = [&](int i, int j) { return (idx_t)(j * nn + i); };
    for (int j = 0; j < nn; ++j)
        for (int i = 0; i < nn; ++i) {
            const idx_t id = nid(i, j);
            m.points[0][id] = (geom_t)(i * m.h);
            m.points[1][id] = (geom_t)(j * m.h);
        }
    ptrdiff_t e = 0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i, ++e) {
            m.elements[0][e] = nid(i, j);
            m.elements[1][e] = nid(i + 1, j);
            m.elements[2][e] = nid(i + 1, j + 1);
            m.elements[3][e] = nid(i, j + 1);
        }
    m.adjugate.assign(4, std::vector<geom_t>(m.nelements, (geom_t)0));
    m.determinant.assign(m.nelements, (geom_t)(m.h * m.h));
    for (ptrdiff_t i = 0; i < m.nelements; ++i) {
        m.adjugate[0][i] = (geom_t)m.h;
        m.adjugate[3][i] = (geom_t)m.h;
    }
    // On this grid the cell map is J = h*I, so the symmetric metric
    // J^-1 J^-T det(J) is the identity exactly.  Written out rather than left
    // empty: a kernel that asks for the metric on a quadrilateral must be
    // handed the value its own geometry implies, and an unfilled array is how
    // `--element TRI3` used to crash.
    m.metric.assign(3, std::vector<geom_t>(m.nelements, (geom_t)0));
    m.metric_aos.assign((size_t)m.nelements * 3, (geom_t)0);
    for (ptrdiff_t i = 0; i < m.nelements; ++i) {
        // The reference square has unit measure, so unlike the triangle there
        // is no factor to fold in.
        const geom_t diag = (geom_t)1;
        m.metric[0][i] = diag;
        m.metric[2][i] = diag;
        m.metric_aos[(size_t)i * 3 + 0] = diag;
        m.metric_aos[(size_t)i * 3 + 2] = diag;
    }
    for (auto &row : m.elements) m.element_ptrs.push_back(row.data());
    for (auto &row : m.points) m.point_ptrs.push_back(row.data());
    return m;
}

// The packed layout.  Shared with the inexact-apply spike rather than written
// twice: both need it without linking smesh, and two copies of a contract this
// exact is two chances to get it subtly different.
//
// The shared file computes the partition from the connectivity; rebuilding a
// mesh under the new numbering is left here, because only this driver knows what
// else its `Mesh` carries per node.
#include "packed_layout.inc"

struct Packed : PackedLayout {
    Mesh mesh;
};

static Packed build_packed(const Mesh &source, ptrdiff_t elements_per_pack) {
    Packed p;
    static_cast<PackedLayout &>(p) =
        build_packed_layout(source.elements, source.nelements, source.nnodes, elements_per_pack);

    // Geometry is per element and so is untouched; only the connectivity and
    // the coordinates move.
    p.mesh.nelements = source.nelements;
    p.mesh.nnodes = source.nnodes;
    p.mesh.h = source.h;
    p.mesh.adjugate = source.adjugate;
    p.mesh.determinant = source.determinant;
    p.mesh.metric = source.metric;
    p.mesh.metric_aos = source.metric_aos;
    p.mesh.elements = p.renumbered;
    p.mesh.points.assign(source.points.size(), std::vector<geom_t>((size_t)source.nnodes));
    for (size_t d = 0; d < source.points.size(); ++d)
        for (ptrdiff_t node = 0; node < source.nnodes; ++node)
            p.mesh.points[d][p.to_new[node]] = source.points[d][node];
    for (auto &row : p.mesh.elements) p.mesh.element_ptrs.push_back(row.data());
    for (auto &row : p.mesh.points) p.mesh.point_ptrs.push_back(row.data());
    return p;
}

// The node adjacency graph of the mesh, in CRS form: node i is adjacent to
// every node sharing an element with it, itself included.  This is the sparsity
// pattern the assembly kernels expect, built from the same connectivity they
// are handed -- which is the point.  A kernel assembling into a graph built
// from its own mesh finds every entry it looks for, and that is precisely the
// precondition those kernels used to re-establish once per element.
struct Graph {
    std::vector<count_t> rowptr;
    std::vector<idx_t> colidx;
    ptrdiff_t nnz = 0;
};

static Graph build_graph(const Mesh &m) {
    const int n_shape = (int)m.elements.size();
    std::vector<std::vector<idx_t>> adjacency((size_t)m.nnodes);
    for (ptrdiff_t e = 0; e < m.nelements; ++e)
        for (int a = 0; a < n_shape; ++a)
            for (int b = 0; b < n_shape; ++b)
                adjacency[(size_t)m.elements[a][e]].push_back(m.elements[b][e]);
    Graph g;
    g.rowptr.assign((size_t)m.nnodes + 1, 0);
    for (ptrdiff_t i = 0; i < m.nnodes; ++i) {
        auto &row = adjacency[(size_t)i];
        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
        g.rowptr[(size_t)i + 1] = g.rowptr[(size_t)i] + (count_t)row.size();
    }
    g.nnz = (ptrdiff_t)g.rowptr[(size_t)m.nnodes];
    g.colidx.reserve((size_t)g.nnz);
    for (ptrdiff_t i = 0; i < m.nnodes; ++i)
        for (idx_t c : adjacency[(size_t)i]) g.colidx.push_back(c);
    return g;
}

// A stable hash of the parameter name, so every field is filled differently and
// the same way on every machine.  Two kernels reading the same named field read
// the same numbers, which is what lets their digests be compared.
static double name_seed(const char *name) {
    unsigned long h = 1469598103934665603UL;
    for (const char *p = name; *p; ++p) {
        h ^= (unsigned char)*p;
        h *= 1099511628211UL;
    }
    return (double)(h % 1000u) / 1000.0;
}

// Fields are filled small.  The amplitude used to reach 1.25, which is a fine
// state for a Laplacian and a ruinous one for a hyperelastic material: a
// displacement that large turns elements inside out, `det(I + grad u)` goes
// through zero, and every neohookean and Mooney-Rivlin kernel answered NaN.
// Those materials had no digest at all as a result -- their only baseline
// entries were two `objective_steps` numbers that were finite because the
// kernel returned without computing.  A twentieth of that is a state every
// material here can survive, and it is still large enough that a kernel
// dropping a term shows up in the digest.
static constexpr double FIELD_AMPLITUDE = 0.05;

template <typename T>
static void fill_field(std::vector<T> &v, const char *name) {
    const double s = name_seed(name);
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = (T)(std::sin(0.5 + s + 0.125 * (double)i) * (0.25 + s) * FIELD_AMPLITUDE);
    }
}

// The same values, placed through the packed renumbering.  Seeding by index
// would make the packed mesh a different problem from the unpacked one, and
// then their digests could not be compared; seeding through `to_old` makes it
// the same problem relabelled, and the l1/l2 digests are permutation-invariant,
// so the two must agree exactly.
template <typename T>
static void fill_field_permuted(std::vector<T> &v,
                                const char *name,
                                const std::vector<idx_t> &to_old,
                                const int components) {
    const double s = name_seed(name);
    for (size_t node = 0; node < to_old.size(); ++node)
        for (int c = 0; c < components; ++c) {
            const size_t old_slot = (size_t)to_old[node] * (size_t)components + (size_t)c;
            v[node * (size_t)components + (size_t)c] =
                    (T)(std::sin(0.5 + s + 0.125 * (double)old_slot) * (0.25 + s) * FIELD_AMPLITUDE);
        }
}

// Material parameters are positive and distinct, and never 1, so a kernel that
// drops one is visible in the digest rather than silently correct.
static double material_scalar(const std::string &name) {
    return 0.5 + name_seed(name.c_str());
}

struct Digest {
    double l1 = 0.0, l2 = 0.0;
};

template <typename T>
static void accumulate(Digest &d, const std::vector<T> &v) {
    for (size_t i = 0; i < v.size(); ++i) {
        const double x = (double)v[i];
        d.l1 += std::fabs(x);
        d.l2 += x * x;
    }
}

static void report(const char *name, const Digest &d) {
    std::printf("kernel %s l1=%.12e l2=%.12e\n", name, d.l1, std::sqrt(d.l2));
}

// Throughput, in millions of node-degrees-of-freedom per second, over the
// fastest of several repeats.  Reported rather than gated: it is meaningful
// compared against the same machine before and after a change, and meaningless
// compared against a number recorded somewhere else.
static void report_rate(const char *name, double seconds, ptrdiff_t nnodes) {
    if (seconds <= 0.0) return;
    std::printf("rate %s %.4f\n", name, (double)nnodes * 1e-6 / seconds);
}
"""


def _fill_call(packed, buffer, param, components):
    """How one input buffer is seeded.

    A packed kernel is driven over a renumbered mesh, so its inputs are placed
    through the permutation; that keeps the two layouts the same problem and
    makes their digests comparable.
    """
    if packed:
        return 'fill_field_permuted(%s, "%s", packed.to_old, %d)' % (
            buffer,
            param,
            components,
        )
    return 'fill_field(%s, "%s")' % (buffer, param)


def _call_block(name, args, inputs, outputs, repeats=0, packed=False, scratch=()):
    # Announce the kernel on stderr before running it.  A driver that dies
    # silently -- and one did, with SIGABRT and no message -- otherwise names
    # only the last kernel that *finished*, which is the one before the
    # problem.  stderr rather than stdout so the digest parser is unaffected.
    lines = [
        "    {",
        '        std::fprintf(stderr, "running %s\\n");' % name,
    ]
    for buffer, scalar, param, components, extent, length in inputs:
        if length is not None:
            lines.append(
                "        std::vector<%s> %s(%s); fill_field(%s, \"%s\");"
                % (scalar, buffer, length, buffer, param)
            )
        elif extent:
            for slot in range(extent):
                lines.append(
                    "        std::vector<%s> %s_%d(nodes * %d); %s;"
                    % (
                        scalar,
                        buffer,
                        slot,
                        components,
                        _fill_call(
                            packed, "%s_%d" % (buffer, slot), "%s_%d" % (param, slot), components
                        ),
                    )
                )
            lines.append(
                "        const %s *const %s[%d] = {%s};"
                % (
                    scalar,
                    buffer,
                    extent,
                    ", ".join("%s_%d.data()" % (buffer, s) for s in range(extent)),
                )
            )
        else:
            lines.append(
                "        std::vector<%s> %s(nodes * %d); %s;"
                % (scalar, buffer, components, _fill_call(packed, buffer, param, components))
            )
    for buffer, scalar, components in scratch:
        lines.append(
            "        std::vector<%s> %s((size_t)packed.n_ghost_entries * %d, (%s)0);"
            % (scalar, buffer, components, scalar)
        )
    for buffer, scalar, _param, components, extent, length in outputs:
        if length is not None:
            lines.append(
                "        std::vector<%s> %s(%s, (%s)0);" % (scalar, buffer, length, scalar)
            )
        elif extent:
            for slot in range(extent):
                lines.append(
                    "        std::vector<%s> %s_%d(nodes * %d, (%s)0);"
                    % (scalar, buffer, slot, components, scalar)
                )
            lines.append(
                "        %s *const %s[%d] = {%s};"
                % (
                    scalar,
                    buffer,
                    extent,
                    ", ".join("%s_%d.data()" % (buffer, s) for s in range(extent)),
                )
            )
        else:
            # Sized for the larger of one entry per node and one per element,
            # because the signature does not say which the kernel writes.  A
            # residual scatters to nodes; an objective accumulates per element
            # -- `value[evbegin + lane] += ...` -- and sizing that by nodes
            # overruns the buffer whenever a mesh has more elements than nodes.
            # A hexahedral grid never does, so this was invisible until a
            # tetrahedral one was driven: six tets per cell against one hex,
            # 162 elements against 64 nodes, and the heap corruption showed up
            # as a silent SIGABRT in the *next* kernel's allocation.
            # A stepped objective writes `value[step * nelements + element]`,
            # so its output is `N_STEPS` times what a plain one needs.  Sizing
            # by the larger for every kernel costs a few thousand doubles and
            # removes a way to overrun this buffer.
            lines.append(
                "        std::vector<%s> %s(std::max<size_t>((size_t)nodes * %d, "
                "(size_t)mesh.nelements * %d), (%s)0);"
                % (scalar, buffer, components, N_STEPS, scalar)
            )
    call = [
        "        %s(" % name,
        "            " + ",\n            ".join(args),
        "        );",
    ]
    lines.extend(call)

    # Digest the *first* call's output, before any timing repeats.  These
    # kernels accumulate -- `out[i] += ...` -- into a buffer nothing re-zeroes
    # between calls, so digesting after the repeat loop made l1 and l2 scale by
    # exactly `repeats + 1`.  A baseline recorded during a throughput run was
    # therefore incomparable with one recorded plainly, silently and by a clean
    # integer factor: the refine=6 bucket carries linear_elasticity digests
    # exactly 20x the plain values, from a `--repeats 19` run.  A digest that
    # depends on how many times you asked for the timing is not a digest.
    lines.append("        Digest d;")
    for buffer, _scalar, _param, _components, extent, _length in outputs:
        if extent:
            for slot in range(extent):
                lines.append("        accumulate(d, %s_%d);" % (buffer, slot))
        else:
            lines.append("        accumulate(d, %s);" % buffer)
    lines.append('        report("%s", d);' % name)

    if repeats:
        lines.append("        {")
        lines.append("            double best = 1e30;")
        lines.append("            for (int rep = 0; rep < %d; ++rep) {" % repeats)
        lines.append("                const auto t0 = std::chrono::steady_clock::now();")
        lines.extend("    " + line for line in call)
        lines.append("                const auto t1 = std::chrono::steady_clock::now();")
        lines.append(
            "                const double s = std::chrono::duration<double>(t1 - t0).count();"
        )
        lines.append("                if (s < best) best = s;")
        lines.append("            }")
        lines.append('            report_rate("%s", best, mesh.nnodes);' % name)
        lines.append("        }")
    lines.append("    }")
    return "\n".join(lines)


def _driver_source(declarations, blocks, refine, builder):
    return "\n".join(
        [
            DRIVER_HEAD,
            'extern "C" {',
            "\n".join(declarations),
            "}",
            "",
            "int main() {",
            "    Mesh mesh = %s(%d);  // not const: the kernels take idx_t **const"
            % (builder, refine),
            "    const Graph graph = build_graph(mesh);",
            "    const size_t nodes = (size_t)mesh.nnodes;",
            "    // The same mesh in the layout the packed kernels require.  Built",
            "    // unconditionally: it costs one pass over the connectivity, and a",
            "    // driver that builds it only when something asks would need to know",
            "    // which kernels those are before it has bound them.",
            "    Packed packed = build_packed(mesh, %d);" % PACK_SIZE,
            "",
            "\n\n".join(blocks),
            "    return 0;",
            "}",
            "",
        ]
    )


# --------------------------------------------------------------------------
# Driving one material
# --------------------------------------------------------------------------


def _generate(root, generated, material):
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        os.path.join(root, "python") + os.pathsep + env.get("PYTHONPATH", "")
    )
    element, aliases = _elements_for(material)
    elements = (element,) + aliases
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "codegen.framework.generators.%s" % material,
            "--out-dir",
            os.path.join(generated, material),
            *sum((["--element", e] for e in elements), []),
        ],
        cwd=os.path.join(root, "python"),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed


class _KeptDirectory:
    """A working directory that survives the run, for when the driver dies.

    The driver is generated, compiled and thrown away, which is right until it
    aborts: then the one artifact that would explain it is the thing that just
    got deleted.  Set SFEM_REPRODUCIBILITY_KEEP=1 to keep it and print where.
    """

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        return self.path

    def __exit__(self, *exc_info):
        sys.stderr.write("kept the driver workspace at %s\n" % self.path)
        return False


def run_material(root, generated, material, refine, compiler, verbose=False, repeats=0):
    """Generate, compile and drive one material.

    Returns ``(digests, rates, skipped)``.  ``repeats`` greater than zero also
    times each kernel; the optimisation level is raised to match, because a
    throughput number from a -O2 build says nothing about the -O3 one that
    ships.
    """
    completed = _generate(root, generated, material)
    if completed.returncode != 0:
        sys.stderr.write(completed.stdout.decode("utf-8", "replace"))
        raise RuntimeError("generator failed for %s" % material)

    entries = _manifest_entries(generated, material)
    if not entries:
        raise RuntimeError("no manifest for %s" % material)

    element, _aliases = _elements_for(material)
    # A Taylor-Hood pair names its velocity element; the grid is that element's.
    grid_element = element.split("_")[0] if "_" in element else element
    builder = {
        "HEX27": "build_grid_hex27",
        "TET4": "build_grid_tet4",
        "TET10": "build_grid_tet4",
        "TRI3": "build_grid_tri3",
        "TRI6": "build_grid_tri3",
        "QUAD4": "build_grid_quad4",
    }.get(grid_element, "build_grid")

    # How many fields the system carries, read off the structure-of-arrays
    # kernels: those take one output pointer per field.  The interleaved
    # array-of-structures kernels take a single pointer holding all of them, so
    # this is what their buffers have to be sized by.
    n_fields = 1
    for entry in entries:
        params = _parameters(entry["declaration"])
        if params is None or not any(n.endswith("_stride") for _t, n, _e in params):
            continue
        outs = sum(1 for t, _n, _e in params if t in OUT_FIELDS)
        n_fields = max(n_fields, outs)

    declarations, blocks, skipped = [], [], []
    for entry in sorted(entries, key=lambda e: e["name"]):
        params = _parameters(entry["declaration"])
        if params is None:
            skipped.append((entry["name"], "unparsed declaration"))
            continue
        interleaved = not any(name.endswith("_stride") for _t, name, _e in params)
        try:
            # element_type is a smesh::ElemType, so a Taylor-Hood pair passes
            # its velocity element rather than the pair label the generator
            # names the kernels after.
            # BSR stores one n_fields x n_fields block per graph entry.  Only
            # that layout is modelled, so only BSR assembly is driven; the rest
            # stay on the skipped list with their reason.
            block_values = (
                "(size_t)graph.nnz * %d" % (n_fields * n_fields)
                if "_bsr_" in entry["name"]
                else None
            )
            args, inputs, outputs, scratch, packed = _bind(
                params,
                grid_element,
                n_fields if interleaved else 1,
                block_values=block_values,
            )
        except Unbindable as reason:
            skipped.append((entry["name"], str(reason)))
            continue
        declarations.append(entry["declaration"].replace('extern "C" ', "").strip())
        blocks.append(
            _call_block(
                entry["name"], args, inputs, outputs, repeats, packed, scratch
            )
        )

    if not blocks:
        return {}, {}, skipped

    workdir = os.path.join(generated, "_driver_%s" % material)
    os.makedirs(workdir, exist_ok=True)
    driver_path = os.path.join(workdir, "driver.cpp")
    with open(driver_path, "w", encoding="utf-8") as handle:
        handle.write(_driver_source(declarations, blocks, refine, builder))

    operators = sorted(
        os.path.join(directory, name)
        for directory, _dirs, files in os.walk(os.path.join(generated, material))
        for name in files
        # The per-element operators, plus the dispatch units that define the
        # element-generic entry points.  Both are self-contained: the dispatch
        # translation unit includes only the C ABI header, so none of this
        # needs the SFEM library to link.
        if name.endswith("_operator.cpp") or name.endswith("_dispatch.cpp")
    )
    binary = os.path.join(workdir, "driver")
    optimisation = "-O3" if repeats else "-O2"
    command = (
        [compiler, optimisation, "-std=c++17"]
        + (os.environ.get("SFEM_REPRODUCIBILITY_CXXFLAGS", "").split())
        + ["-o", binary, driver_path]
        + operators
        + _include_flags(root, generated)
    )
    build = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if build.returncode != 0:
        sys.stderr.write(build.stdout.decode("utf-8", "replace")[-4000:])
        raise RuntimeError("driver failed to compile for %s" % material)

    run = subprocess.run([binary], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = run.stdout.decode("utf-8", "replace")
    if run.returncode != 0:
        sys.stderr.write(output)
        # A driver killed by a signal reports a negative return code and prints
        # nothing, so "failed to run" on its own says only that something went
        # wrong somewhere in a few hundred kernels.  Naming the signal and the
        # last kernel that did report turns that into a place to look.
        signal_name = ""
        if run.returncode < 0:
            try:
                import signal as _signal

                signal_name = " (%s)" % _signal.Signals(-run.returncode).name
            except (ValueError, AttributeError):
                signal_name = ""
        last = ""
        for line in output.splitlines():
            if line.startswith("kernel "):
                last = line.split()[1]
        raise RuntimeError(
            "driver failed to run for %s: exit %d%s%s"
            % (
                material,
                run.returncode,
                signal_name,
                "; last kernel to report was %s" % last if last else "",
            )
        )
    if verbose:
        print(output)

    digests = {}
    for match in re.finditer(
        r"kernel (\S+) l1=([-\d.e+]+) l2=([-\d.e+]+)", output
    ):
        digests[match.group(1)] = {
            "l1": float(match.group(2)),
            "l2": float(match.group(3)),
        }
    rates = {
        match.group(1): float(match.group(2))
        for match in re.finditer(r"rate (\S+) ([\d.]+)", output)
    }
    return digests, rates, skipped


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------


#: An affine grid means the isoparametric geometry *is* the affine one, so a
#: kernel pair differing only in geometry mode has to agree.  This is free --
#: both are driven anyway -- and it is the check that catches a bug in the
#: adjugate routing, which a digest alone would happily record as the new
#: correct answer.
#:
#: They agree to geometry precision rather than exactly, because the two derive
#: the metric differently and ``geom_t`` is single precision: the affine kernels
#: read an adjugate the harness rounds to float, the isoparametric ones compute
#: one in double from float coordinates.  ``apply_bench`` classifies the same
#: comparison the same way and for the same reason.
#:
#: The first version of this used 1e-12 and passed, which was luck rather than
#: agreement: on a HEX8 grid at refine=4 the spacing is 0.25 and every geometry
#: value is exactly representable, so no rounding happened.  The quadratic grid,
#: whose spacing is 1/3, showed the difference at 1e-8 immediately.
PARITY_TOLERANCE = {"double": 1e-5, "float": 1e-5}

#: Materials whose two geometry modes integrate at different quadrature orders,
#: and the tolerance that difference costs.
#:
#: The check asks whether the affine shortcut agrees with the general path on a
#: mesh where it is valid.  It assumes both integrate the same way, which holds
#: while the integrand is polynomial: a Laplacian or a linear elasticity is
#: exact under either rule and the two agree to rounding.  A hyperelastic
#: integrand is not polynomial, and poro's affine kernels use four quadrature
#: points against the isoparametric eleven, so the paths differ by the
#: quadrature error rather than by anything about geometry.  Holding them to
#: 1e-5 measures the rule, not the claim.
PARITY_QUADRATURE_EXCEPTIONS = {"poro_hyperelasticity_solid_": 1e-4}


def _parity_tolerance(name):
    for prefix, tolerance in PARITY_QUADRATURE_EXCEPTIONS.items():
        if name.startswith(prefix):
            return tolerance
    return PARITY_TOLERANCE["float" if name.endswith("_float") else "double"]


def _geometry_parity(measured):
    """Pairs that differ only in geometry mode and disagree anyway."""
    disagreements = []
    for name in sorted(measured):
        if "_affine_" not in name:
            continue
        twin = name.replace("_affine_", "_isoparametric_")
        if twin not in measured:
            continue
        tolerance = _parity_tolerance(name)
        worst = 0.0
        for key in ("l1", "l2"):
            a, b = measured[name][key], measured[twin][key]
            scale = max(abs(a), abs(b), 1e-30)
            worst = max(worst, abs(a - b) / scale)
        if worst > tolerance:
            disagreements.append((name, twin, worst, tolerance))
    return disagreements


def _unpacked_twin(name):
    """The standard-layout kernel a packed one mirrors, or `None`.

    Built from the naming table rather than by deleting the text `_packed_`.
    That spelling only ever matched the one-pass kernels: a two-pass name
    carries `_packed_two_pass_`, and removing `_packed_` from it leaves
    `_two_pass_`, which names nothing -- so every two-pass kernel in the tree
    silently paired with nothing and went unchecked. The qualifier slot can also
    hold a store *and* a traversal, as a packed inexact apply does, and only the
    traversal is what the twin lacks.
    """
    try:
        traversal = conventions.abi_traversal(conventions.abi_qualifier(name))
    except ValueError:
        return None
    if not traversal:
        return None
    return name.replace("_%s_" % traversal, "_", 1)


def _packed_parity(measured):
    """Packed kernels that disagree with the unpacked kernel they mirror.

    A packed kernel is the same mathematics over a mesh partitioned into packs,
    gathered through thread-local scratch and scattered back.  Driven over the
    renumbered mesh with its input placed through the same permutation, it is
    the same problem relabelled, and the l1/l2 digests are permutation-invariant
    -- so it must reproduce the unpacked answer, not merely be close to it.

    This is what makes the packed layout the harness builds checkable at all.
    Nothing else validates the partition, the ghost lists or the pack-local
    indices; a wrong ghost list still runs and still returns a number.
    """
    disagreements = []
    for name in sorted(measured):
        twin = _unpacked_twin(name)
        if twin is None or twin not in measured:
            continue
        tolerance = _parity_tolerance(name)
        worst = 0.0
        for key in ("l1", "l2"):
            a, b = measured[name][key], measured[twin][key]
            scale = max(abs(a), abs(b), 1e-30)
            worst = max(worst, abs(a - b) / scale)
        if worst > tolerance:
            disagreements.append((name, twin, worst, tolerance))
    return disagreements


def _is_maintained(name, digest):
    """Whether this baseline entry belongs to a material that still generates.

    An entry for a material outside `MATERIALS` can never be refreshed, because
    nothing produces the kernel; counting it as missing would make the gate
    permanently red for a scope decision taken deliberately elsewhere.
    """
    owned = digest.get("material") if isinstance(digest, dict) else None
    if owned is not None:
        return owned in MATERIALS
    return any(name.startswith(material) for material in MATERIALS)


def _compare(recorded, measured):
    """Kernels that moved, that appeared, and that went missing."""
    moved, appeared, missing = [], [], []
    for name, digest in sorted(measured.items()):
        expected = recorded.get(name)
        if expected is None:
            appeared.append(name)
            continue
        worst = 0.0
        for key in ("l1", "l2"):
            scale = max(abs(expected[key]), abs(digest[key]), 1e-30)
            worst = max(worst, abs(expected[key] - digest[key]) / scale)
        if worst > TOLERANCE:
            moved.append((name, worst, expected, digest))
    for name in sorted(recorded):
        if name not in measured:
            missing.append(name)
    return moved, appeared, missing


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--material", action="append", default=None)
    parser.add_argument("--all", action="store_true", help="every maintained material")
    parser.add_argument("--refine", type=int, default=6, help="cells per side")
    parser.add_argument(
        "--element",
        help="drive this element instead of the material's default, so a "
             "kernel family the default grid cannot reach can still be "
             "checked and timed",
    )
    parser.add_argument("--record", action="store_true", help="rewrite the baseline")
    parser.add_argument(
        "--rename-map",
        default=None,
        help=(
            "JSON file mapping old kernel name -> new kernel name.  Rewrites the "
            "baseline keys through it and requires every digest to be unchanged, "
            "which is how a rename is proved rather than re-recorded."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--repeats",
        type=int,
        default=0,
        help=(
            "also time each kernel, taking the fastest of this many runs. "
            "Throughput is reported, never gated: it is meaningful against the "
            "same machine before and after a change and meaningless against a "
            "number recorded elsewhere"
        ),
    )
    args = parser.parse_args(argv)

    if args.element:
        globals()["ELEMENT_OVERRIDE"] = args.element
    materials = args.material or (list(MATERIALS) if args.all else ["laplace"])

    compiler = _compiler()
    if compiler is None:
        sys.stderr.write("no C++ compiler available\n")
        return 2

    root = _repo_root()
    # A digest is a property of the kernel *and* the mesh it ran on, so the
    # baseline is bucketed by refinement.  Keying by kernel name alone meant a
    # run at a different --refine compared against another mesh's numbers and
    # reported every kernel as moved -- a gate that cries wolf is worth about as
    # much as one that stays silent.
    #
    # It is also a property of the element, and that was missing: a HEX8 run
    # and a TET4 run generate the same kernel names -- both are `_3d_` -- so a
    # recorded TET4 answer would overwrite the HEX8 one under the same key and
    # the next `--all` run would report every kernel moved.  The default bucket
    # therefore keeps its meaning, "each material on its own default grid", and
    # an explicit `--element` gets a bucket of its own.
    bucket = (
        "element=%s refine=%d" % (ELEMENT_OVERRIDE, args.refine)
        if ELEMENT_OVERRIDE
        else "refine=%d" % args.refine
    )
    baseline = {}
    if os.path.exists(BASELINE_PATH):
        with open(BASELINE_PATH, encoding="utf-8") as handle:
            baseline = json.load(handle)
    recorded = baseline.get(bucket, {})

    # An entry that does not say which material owns it cannot be scoped to a
    # run, so it is never compared and never dropped -- it just sits there
    # looking like coverage.  Sixteen did, for kernels deleted when laplace was
    # rewritten as an energy and when the precision suffixes were collapsed.
    # The owner is recorded on every entry now, so an entry without one is a
    # baseline that predates that and has to be re-recorded, not ignored.
    unattributable = sorted(
        name
        for name, digest in recorded.items()
        if not (isinstance(digest, dict) and digest.get("material"))
    )
    if unattributable:
        sys.stderr.write(
            "%d baseline entries at %s name no material, so nothing can compare "
            "them:\n%s\nRe-record the bucket, or delete them if the kernels are "
            "gone.\n"
            % (len(unattributable), bucket, "\n".join("    " + n for n in unattributable[:20]))
        )
        return 1

    measured, measured_rates, skipped_all, failures = {}, {}, {}, []
    owner = {}
    keep = bool(os.environ.get("SFEM_REPRODUCIBILITY_KEEP"))
    context = (
        _KeptDirectory(tempfile.mkdtemp(prefix="sfem_reproducibility_"))
        if keep
        else tempfile.TemporaryDirectory(prefix="sfem_reproducibility_")
    )
    with context as workdir:
        generated = os.path.join(workdir, "generated")
        os.makedirs(generated)
        for material in materials:
            print("==> %s" % material)
            if material in NOT_COVERED:
                print("    not driven: %s" % NOT_COVERED[material])
                continue
            try:
                digests, rates, skipped = run_material(
                    root,
                    generated,
                    material,
                    args.refine,
                    compiler,
                    args.verbose,
                    repeats=args.repeats,
                )
            except RuntimeError as error:
                failures.append("%s: %s" % (material, error))
                continue
            measured.update(digests)
            # Which material a kernel belongs to is known exactly here and
            # nowhere else.  It used to be re-derived downstream as
            # `name.startswith(material)`, which is a guess that poro-elasticity
            # falsifies: its material is "poro_elasticity" and its kernels are
            # prefixed "poro_hyperelasticity_", so none of its 28 digests ever
            # matched their own material.  They were reported as new on every
            # single run, and `--record` never pruned them, so a renamed or
            # deleted poro kernel would have sat in the baseline forever.
            owner.update(dict.fromkeys(digests, material))
            measured_rates.update(rates)
            skipped_all[material] = skipped
            print(
                "    %d kernels driven, %d skipped" % (len(digests), len(skipped))
            )

    over_budget = []
    for material, skipped in sorted(skipped_all.items()):
        reasons = {}
        for _name, reason in skipped:
            reasons[reason] = reasons.get(reason, 0) + 1
        for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1])[:4]:
            print("    %s skipped %d for: %s" % (material, count, reason))
        allowed = SKIPPED_BUDGET.get(material)
        if allowed is not None and len(skipped) > allowed:
            over_budget.append((material, len(skipped), allowed))
    if over_budget:
        sys.stderr.write(
            "kernels left the run that the budget does not allow; each one is a "
            "digest that is no longer checked:\n"
        )
        for material, count, allowed in over_budget:
            sys.stderr.write(
                "    %s skipped %d, budget %d\n" % (material, count, allowed)
            )
        sys.stderr.write(
            "\nIf a kernel legitimately stopped binding, lower nothing -- work out "
            "why the binder no longer recognises it.  SKIPPED_BUDGET only ever "
            "shrinks.\n"
        )
        return 1

    if failures:
        for failure in failures:
            sys.stderr.write("FAILED %s\n" % failure)
        return 1

    key_scope = set()
    for material in materials:
        key_scope.update(name for name, m in owner.items() if m == material)
    # A baseline entry for one of this run's materials belongs in scope even
    # when nothing measured it, because that is exactly the case worth
    # reporting: the kernel used to exist and does not any more.  Scoping only
    # by what was measured this run made a vanished kernel invisible -- 74 of
    # them went missing when the precision suffixes were collapsed and the gate
    # stayed green, which is the failure it exists to prevent.  The owner
    # recorded alongside the digest answers this; entries from a baseline
    # written before owners were recorded fall back to the kernel prefixes seen
    # this run.
    prefixes = {name.rsplit("_", 1)[0] for name in owner}
    for name, digest in recorded.items():
        recorded_owner = digest.get("material") if isinstance(digest, dict) else None
        if recorded_owner is not None:
            if recorded_owner in materials:
                key_scope.add(name)
        elif name in owner or any(name.startswith(prefix) for prefix in prefixes):
            key_scope.add(name)

    if measured_rates:
        print("\nthroughput, MDOF/s (same-machine comparison only):")
        for name, rate in sorted(measured_rates.items(), key=lambda kv: -kv[1]):
            print("    %-72s %8.2f" % (name, rate))

    disagreements = _geometry_parity(measured)
    if disagreements:
        sys.stderr.write(
            "%d kernel pairs differ by geometry mode alone on an affine mesh:\n"
            % len(disagreements)
        )
        for name, twin, worst, tolerance in disagreements:
            sys.stderr.write(
                "    %s vs %s  rel=%.3e  tolerance=%.0e\n"
                % (name, twin, worst, tolerance)
            )
        return 1
    parity_pairs = sum(
        1
        for name in measured
        if "_affine_" in name and name.replace("_affine_", "_isoparametric_") in measured
    )
    if parity_pairs:
        print("affine/isoparametric parity holds for %d pairs" % parity_pairs)

    packed_disagreements = _packed_parity(measured)
    if packed_disagreements:
        sys.stderr.write(
            "%d packed kernels disagree with the unpacked kernel they mirror:\n"
            % len(packed_disagreements)
        )
        for name, twin, worst, tolerance in packed_disagreements:
            sys.stderr.write(
                "    %s vs %s  rel=%.3e  tolerance=%.0e\n"
                % (name, twin, worst, tolerance)
            )
        return 1
    # Counted the same way the check pairs them.  These were two different
    # expressions, so the number reported was not the number checked -- it
    # undercounted by every two-pass kernel in the tree.
    packed_pairs = sum(
        1
        for name in measured
        if (_unpacked_twin(name) or "") in measured
    )
    if packed_pairs:
        print("packed/unpacked parity holds for %d pairs" % packed_pairs)

    if args.record and args.rename_map:
        sys.stderr.write(
            "--record and --rename-map are different claims and cannot be combined.\n"
            "--record asserts the new digests are correct; --rename-map proves the\n"
            "old ones survived the rename, and rewrites the baseline keys itself.\n"
        )
        return 1

    if args.record:
        merged = dict(recorded)
        # Also drop entries owned by a material the snapshot no longer
        # maintains.  Five materials left the default generation set with the
        # scope reduction, and their digests cannot be refreshed by any run --
        # nothing generates them -- so they would report as `missing` forever
        # now that a changed population fails.  A baseline entry for a material
        # that is not generated is history, not coverage.
        unmaintained = []
        for name, digest in merged.items():
            owned = digest.get("material") if isinstance(digest, dict) else None
            if owned is not None:
                if owned not in MATERIALS:
                    unmaintained.append(name)
            elif not any(name.startswith(m) for m in MATERIALS):
                unmaintained.append(name)
        for name in unmaintained:
            del merged[name]
        if unmaintained:
            print(
                "dropped %d entries for materials the snapshot no longer maintains"
                % len(unmaintained)
            )
        # Drop this run's materials wholesale before re-adding what was
        # measured, so a kernel that no longer exists leaves the baseline.
        #
        # `key_scope` is the right set and is already computed above: it is
        # every entry this run's materials own, whether or not anything measured
        # it.  The previous rule matched on kernel-name *prefixes* seen in this
        # run, which cannot see a verb change -- when laplace was rewritten as an
        # energy its `laplace_residual_*` and `laplace_jacobian_action_*` entries
        # matched no surviving prefix, so eleven dead entries stayed in the
        # baseline through every subsequent --record.  They were invisible
        # because the comparison forgave `missing`; both halves are fixed here.
        for name in list(merged):
            if name in key_scope or name in measured:
                del merged[name]
        merged.update(
            {
                name: dict(digest, material=owner[name])
                if isinstance(digest, dict) and name in owner
                else digest
                for name, digest in measured.items()
            }
        )
        baseline[bucket] = merged
        with open(BASELINE_PATH, "w", encoding="utf-8") as handle:
            json.dump(baseline, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print("recorded %d kernel digests for %s" % (len(measured), bucket))
        return 0

    if args.rename_map:
        # Carry each recorded digest to the kernel's new name and compare as
        # normal.  The point is that nothing else changes: if the rename was
        # only a rename, every digest matches under the new key and `appeared`
        # and `missing` are both empty.  Re-recording would assert the same
        # thing by fiat; this proves it.
        with open(args.rename_map, encoding="utf-8") as handle:
            renames = json.load(handle)
        unknown = sorted(set(renames) - set(recorded))
        if unknown:
            sys.stderr.write(
                "%d names in the rename map are not in the baseline:\n" % len(unknown)
            )
            for name in unknown[:20]:
                sys.stderr.write("    %s\n" % name)
            return 1
        collisions = sorted(set(renames.values()) & (set(recorded) - set(renames)))
        if collisions:
            sys.stderr.write(
                "%d renamed kernels collide with a baseline name that is not "
                "being renamed:\n" % len(collisions)
            )
            for name in collisions[:20]:
                sys.stderr.write("    %s\n" % name)
            return 1
        recorded = {renames.get(name, name): digest for name, digest in recorded.items()}
        key_scope = {renames.get(name, name) for name in key_scope}
        print("rename map applied to %d baseline entries" % len(renames))

    scoped = {
        name: digest
        for name, digest in recorded.items()
        if name in key_scope and _is_maintained(name, digest)
    }
    moved, appeared, missing = _compare(scoped, measured)

    # A kernel that appears or vanishes is a change to *what is checked*, and
    # until now it was reported and then forgiven: only `moved` returned
    # non-zero, so a run in which every recorded kernel went missing and an
    # equal number of new ones appeared printed "all N kernel answers match"
    # and exited 0.  That is precisely the shape of a rename, which is the one
    # change this gate most needs to survive -- and `key_scope` above was
    # already built to make a vanished kernel visible, so the scoping half of
    # the fix was done and the exit code was not.
    # Report a moved answer before a changed population: a digest that moved is
    # the stronger signal and must not be hidden behind a rename that happened
    # in the same run.
    if moved:
        sys.stderr.write("%d kernel answers moved:\n" % len(moved))
        for name, worst, expected, digest in moved[:20]:
            sys.stderr.write(
                "    %s  rel=%.3e  was l1=%.12e l2=%.12e  now l1=%.12e l2=%.12e\n"
                % (name, worst, expected["l1"], expected["l2"], digest["l1"], digest["l2"])
            )
        sys.stderr.write(
            "\nIf the change was intended, re-record with --record and review the "
            "baseline diff: it is the statement of which kernels moved.\n"
        )

    if appeared or missing:
        if appeared:
            sys.stderr.write(
                "%d kernels are not in the baseline:\n" % len(appeared)
            )
            for name in appeared[:20]:
                sys.stderr.write("    %s\n" % name)
        if missing:
            sys.stderr.write(
                "%d kernels in the baseline no longer generate:\n" % len(missing)
            )
            for name in missing[:20]:
                sys.stderr.write("    %s\n" % name)
        sys.stderr.write(
            "\nThe set of kernels changed, so the digests below cover a different\n"
            "population than the baseline records.  If a kernel was renamed, use\n"
            "--rename-map to carry its digest across and prove the answer did not\n"
            "move.  If it was genuinely added or removed, re-record with --record.\n"
        )
        return 1
    if moved:
        return 1

    if not scoped:
        print(
            "no baseline recorded for these materials at %s yet; run with --record"
            % bucket
        )
        return 0
    if args.rename_map:
        baseline[bucket] = recorded
        with open(BASELINE_PATH, "w", encoding="utf-8") as handle:
            json.dump(baseline, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(
            "renamed %d baseline keys at %s; every digest survived unchanged"
            % (len(json.load(open(args.rename_map, encoding="utf-8"))), bucket)
        )
    print(
        "all %d kernel answers match the recorded baseline at %s"
        % (len(measured), bucket)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
