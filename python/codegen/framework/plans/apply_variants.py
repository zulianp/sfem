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

    AoS only appears for equal-order formulations.  A mixed-order kernel
    (Taylor-Hood: poro-hyperelasticity, Stokes) reads a different element type
    per field, and the AoS coordinate path assumes one element type per cell.

    Packed traversal is Laplace-only today.  It is the nearly-optimal reference
    implementation the PRD points at, not yet generalised.

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

PRECISIONS = (Precision.SCALAR, Precision.FLOAT)


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
        if (
            self.traversal == MeshTraversal.PACKED_TWO_PASS
            and self.geometry != Geometry.ISOPARAMETRIC
        ):
            raise ValueError("packed two-pass traversal is isoparametric only")

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
):
    """The variants a kernel gets.

    ``mixed_order``               Taylor-Hood style: fields use different
                                  element types on the same cell.
    ``supports_packed``           the packed mesh traversal is implemented here.
    ``is_jacobian_action``        this is the matrix-free apply rather than a
                                  residual evaluation; packed applies only to
                                  the former.
    ``affine_equivalent_element`` a linear simplex, whose isoparametric geometry
                                  is constant over the cell.
    """
    shapes = list(BASE_VARIANTS)
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
