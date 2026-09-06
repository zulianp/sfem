"""Which matrix-free apply variants a kernel gets.

The matrix-free action -- applying the Jacobian or Hessian to a vector without
assembling it -- is the framework's primary product. It is 219 of the 520
generated files, and it is the kernel whose arithmetic intensity the whole
design exists to protect.

Each kernel is emitted several times over, once per combination of:

    mesh traversal   standard, packed, or packed two-pass
    geometry         affine (precomputed adjugate) or isoparametric
    element data     SoA or AoS
    precision        the kernel scalar type, or float

Not every combination is emitted, and the rule for which ones are is real:

    AoS only ever appears with isoparametric geometry.  Affine geometry is
    handed precomputed adjugate and determinant streams, so there are no
    element coordinates to read in either layout -- an affine AoS variant would
    be the same kernel under a different name.

    AoS only appears for equal-order formulations.  This one is a capability
    gap, not a design rule, and the distinction matters: the mixed-order
    emitter (``generate_mixed_residual_sfem_files``) simply has no AoS code
    path.  The two residual emitters share 419 of 9,290 lines -- 4.5% -- so the
    mixed path is a parallel implementation that never grew several of the
    coupled path's capabilities:

        AoS dispatch    coupled only
        packed apply    coupled only
        CRS assembly    coupled only
        DIA assembly    coupled only
        COO assembly    both
        BSR assembly    neither

    An earlier version of this file asserted that AoS is meaningless for
    mixed-order kernels because they read a different element type per field.
    That was inferred from the emitted output, not established; the cell
    geometry a mixed kernel reads is the high-order element's, so an AoS
    coordinate path is not obviously impossible.  What is established is that
    nobody wrote one.

    Packed traversal is Laplace-only today.  It is the nearly-optimal reference
    implementation the PRD points at, not yet generalised.  Combined with the
    line above, that means the Taylor-Hood formulations -- poro-hyperelasticity
    and Stokes -- get no packed matrix-free apply at all.

    Packed applies to the Jacobian action only, never to the residual.  It is a
    matrix-free apply optimisation; there is nothing for it to do in a residual
    evaluation.

    Packed isoparametric variants are skipped for affine-equivalent linear
    simplices -- TRI3 and TET4.  Their isoparametric geometry *is* affine (one
    geometry point per element, as pinned in test_geometry_plan_agrees), so a
    packed isoparametric kernel would duplicate the packed affine one.

    Packed is always SoA, and packed two-pass is always isoparametric.

That rule was implicit in the emitters -- readable only by generating the
output and looking at what came out. This module states it. The variant set is
a planning decision: it fixes how many kernels exist, what each is called, and
which data layout each reads, all of which are structure rather than syntax.
"""

from dataclasses import dataclass


class MeshTraversal:
    STANDARD = "standard"
    PACKED = "packed"
    PACKED_TWO_PASS = "packed_two_pass"


class Geometry:
    AFFINE = "affine"
    ISOPARAMETRIC = "isoparametric"


class ElementLayout:
    SOA = "soa"
    AOS = "aos"


class Precision:
    SCALAR = "scalar"
    FLOAT = "float"


#: Emitted for every kernel, in this order.
BASE_VARIANTS = (
    (MeshTraversal.STANDARD, Geometry.AFFINE, ElementLayout.SOA),
    (MeshTraversal.STANDARD, Geometry.ISOPARAMETRIC, ElementLayout.SOA),
)

#: Emitted only for equal-order formulations.
EQUAL_ORDER_VARIANTS = (
    (MeshTraversal.STANDARD, Geometry.ISOPARAMETRIC, ElementLayout.AOS),
)

#: Emitted wherever packed traversal is supported.
PACKED_AFFINE_VARIANTS = (
    (MeshTraversal.PACKED, Geometry.AFFINE, ElementLayout.SOA),
)

#: Additionally emitted where the element's geometry is genuinely curved.
PACKED_ISOPARAMETRIC_VARIANTS = (
    (MeshTraversal.PACKED, Geometry.ISOPARAMETRIC, ElementLayout.SOA),
    (MeshTraversal.PACKED_TWO_PASS, Geometry.ISOPARAMETRIC, ElementLayout.SOA),
)

#: The two-pass affine traversal, which only the energy path publishes.
PACKED_TWO_PASS_AFFINE_VARIANTS = (
    (MeshTraversal.PACKED_TWO_PASS, Geometry.AFFINE, ElementLayout.SOA),
)

PRECISIONS = (Precision.SCALAR, Precision.FLOAT)


#: The scalar types a single entry point can be asked for at run time, and the
#: ``smesh::PrimitiveType`` value that names each one.
#:
#: This is the same axis as ``PRECISIONS``, read for the opposite purpose.  As
#: ``PRECISIONS`` it multiplies symbols: every kernel touching scalar data is
#: emitted once per entry, the second copy differing only in ``using scalar_t =
#: float`` and its parameter types.  As runtime cases it multiplies nothing --
#: one entry point takes ``void *`` buffers and a ``const enum
#: smesh::PrimitiveType``, switches, casts, and calls a template.
#:
#: The form and the values are SFEM's, not new: ``cu_tet4_laplacian_apply`` in
#: ``operators/tet4/cuda/`` switches over exactly these three and falls through
#: to ``SFEM_ERROR``.  ``SMESH_DEFAULT`` is first because it is the default a
#: caller gets -- ``GPULaplacian`` declares ``real_type{smesh::SMESH_DEFAULT}``
#: and carries it through ``clone()`` -- so it is the common path, not a
#: fallback, and it resolves to ``real_t`` rather than to a fixed width.
#:
#: See ARCHITECTURE.html OP 17.
RUNTIME_SCALAR_CASES = (
    ("smesh::SMESH_DEFAULT", "real_t"),
    ("smesh::SMESH_FLOAT32", "float"),
    ("smesh::SMESH_FLOAT64", "double"),
)

#: The parameter a runtime-typed entry point carries, spelled as SFEM spells it.
RUNTIME_SCALAR_TYPE_PARAMETER = "const enum smesh::PrimitiveType real_type"

#: What the entry point names that parameter.
RUNTIME_SCALAR_TYPE_ARGUMENT = "real_type"


def runtime_scalar_cases():
    """``(enum_value, c_type)`` for each case a runtime-typed kernel accepts."""
    return RUNTIME_SCALAR_CASES


@dataclass(frozen=True)
class ApplyVariant:
    """One matrix-free apply kernel."""

    traversal: str
    geometry: str
    layout: str
    precision: str

    def __post_init__(self):
        if self.layout == ElementLayout.AOS and self.geometry != Geometry.ISOPARAMETRIC:
            raise ValueError(
                "an AoS apply variant is only meaningful with isoparametric "
                "geometry; affine geometry reads precomputed adjugate streams"
            )
        if self.traversal != MeshTraversal.STANDARD and self.layout != ElementLayout.SOA:
            raise ValueError("packed traversal is SoA only")
        # The two-pass traversal was isoparametric-only when the residual path
        # was the only one that published it: the second pass exists to reduce
        # ghost contributions without atomics, which is a property of the mesh
        # partition rather than of the geometry.  The energy path publishes the
        # affine form too, and it is generated, compiled and driven -- so the
        # restriction was a statement about one emitter, not about the variant,
        # and refusing to represent it would have made the plan disagree with
        # the output rather than describe it.

    @property
    def suffix(self):
        """The name fragment this variant contributes, e.g. ``packed_affine_mesh_soa``."""
        traversal = "" if self.traversal == MeshTraversal.STANDARD else "%s_" % self.traversal
        precision = "" if self.precision == Precision.SCALAR else "_float"
        return "%s%s_mesh_%s%s" % (traversal, self.geometry, self.layout, precision)

    def to_dict(self):
        return {
            "traversal": self.traversal,
            "geometry": self.geometry,
            "layout": self.layout,
            "precision": self.precision,
            "suffix": self.suffix,
        }


@dataclass(frozen=True)
class ApplyVariantPlan:
    """The full set of matrix-free apply variants for one kernel."""

    variants: tuple

    def suffixes(self):
        return tuple(variant.suffix for variant in self.variants)

    def to_dict(self):
        return {"variants": [variant.to_dict() for variant in self.variants]}


def apply_variant_plan(
    *,
    mixed_order,
    supports_packed,
    is_jacobian_action=True,
    affine_equivalent_element=False,
    from_energy=False,
):
    """The variants a kernel gets.

    ``mixed_order``               Taylor-Hood style: fields use different
                                  element types on the same cell.
    ``supports_packed``           the packed mesh traversal is implemented here.
    ``is_jacobian_action``        this is the matrix-free apply rather than a
                                  residual evaluation; on the residual path,
                                  packed applies only to the former.
    ``affine_equivalent_element`` a linear simplex, whose isoparametric geometry
                                  is constant over the cell.
    ``from_energy``               the material is written as an energy.

    The last is a difference between the two emitters, stated here rather than
    left for a reader to discover from the output.  Measured on every energy
    material in the tree -- `laplace`, `scalar_potential`, `linear_elasticity`
    -- the energy path publishes the packed traversals for its 1-form as well
    as its 2-form action, publishes the packed isoparametric and two-pass forms
    even on an affine-equivalent element, and publishes no array-of-structures
    variant at all.  The residual path does the opposite on each count.

    That the two differ is a fact about where the work stands, not a design:
    the prescription is one lowering below the form layer, and this is the
    shape of what has yet to converge.  Writing it down is what lets the
    difference shrink under a test instead of drifting.
    """
    shapes = list(BASE_VARIANTS)
    if from_energy:
        if supports_packed:
            shapes.extend(PACKED_AFFINE_VARIANTS)
            shapes.extend(PACKED_TWO_PASS_AFFINE_VARIANTS)
            shapes.extend(PACKED_ISOPARAMETRIC_VARIANTS)
    else:
        if not mixed_order:
            shapes.extend(EQUAL_ORDER_VARIANTS)
        if supports_packed and is_jacobian_action:
            shapes.extend(PACKED_AFFINE_VARIANTS)
            if not affine_equivalent_element:
                shapes.extend(PACKED_ISOPARAMETRIC_VARIANTS)
    return ApplyVariantPlan(
        variants=tuple(
            ApplyVariant(traversal, geometry, layout, precision)
            for traversal, geometry, layout in shapes
            for precision in PRECISIONS
        )
    )


#: The scalar type each precision is emitted with.  ``double`` is the default
#: kernel scalar type; the ``float`` variants exist so a solver can apply the
#: operator in reduced precision, which is a bandwidth decision and therefore
#: this layer's to make.
PRECISION_SCALAR_TYPES = {
    Precision.SCALAR: "double",
    Precision.FLOAT: "float",
}

#: The name fragment each precision contributes.
PRECISION_SUFFIXES = {
    Precision.SCALAR: "",
    Precision.FLOAT: "_float",
}


def precision_axis():
    """``(scalar_type, name_suffix)`` for each emitted precision, in order.

    Every matrix-free apply kernel is emitted once per entry.  The emitters
    used to carry this as a literal tuple, repeated seventeen times in the
    residual emitter alone, so adding or removing a precision meant editing
    seventeen places and hoping none was missed.
    """
    return tuple(
        (PRECISION_SCALAR_TYPES[precision], PRECISION_SUFFIXES[precision])
        for precision in PRECISIONS
    )
