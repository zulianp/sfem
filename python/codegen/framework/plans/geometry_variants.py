"""Which geometry variants a form publishes on an element, and why.

An operator does not emit one kernel per form.  It emits a cross-product: the
form, times the geometry the kernel reads (a precomputed affine adjugate, a
cached metric, or coordinates evaluated per quadrature point), times the mesh
layout (standard, packed one-pass, packed two-pass), times the scalar
precision.  That cross-product is most of the generated surface, and until now
every factor of it was decided by a guard inside the emitter -- a
``geometry_mode == "affine" and specialized_prefix is not None`` here, an
``_sfem_soa_has_adjugate_geometry_inputs`` there -- so the answer to "what does
this element publish?" existed only as the union of conditions scattered across
two ten-thousand-line modules.

This is that answer, in one place, as a value.  The emitters read it; they no
longer work it out.

The policy has three parts, and each is a claim about the mathematics rather
than a preference:

**A constant-P1 simplex needs one kernel, not two.**  TRI3 and TET4 have a
Jacobian that is constant over the cell, so the isoparametric kernel -- which
rebuilds that Jacobian at every quadrature point from the coordinates -- computes
the same numbers as the affine one, more slowly.  Emitting both doubles the
kernel bodies to no end.

**The cached metric replaces the adjugate where the contraction factors through
it.**  For a P1 simplex whose flux is a uniform scalar multiple of the field
gradient, ``grad(v) . flux`` is ``B^T (kappa*FFF) B``: six symmetric metric
components instead of nine adjugate ones and a determinant.  That is a property
of the lowered form, which is why it is asked of the form and not of the
material.

**Two dimensions is not where the performance is.**  The 2D elements exist to
keep the framework honest about dimension-independence, not to run production
problems, so they carry the isoparametric standard-layout kernel and nothing
else.  This is the one part of the policy that is a scope decision rather than a
mathematical one, and it is the part to revisit first when 2D matters again.
"""

from dataclasses import dataclass

from codegen.framework.plans.form_transformations import (
    _is_constant_p1_simplex_rule,
    cached_metric_geometry,
)


@dataclass(frozen=True)
class GeometryVariantPlan:
    """What one form publishes on one element."""

    #: The kernel that reads a precomputed adjugate, or the cached metric.
    emits_affine: bool
    #: The kernel that builds its geometry from coordinates per quadrature point.
    emits_isoparametric: bool
    #: The packed-mesh entry points, one-pass and two-pass.
    emits_packed: bool
    #: `CachedMetricGeometry` when the affine contraction factors through `FFF`,
    #: otherwise None.  Only the affine variant can use it: an isoparametric
    #: kernel holds an adjugate, so the two modes need different kernels rather
    #: than different arguments.
    cached_metric: object = None

    @property
    def emits_metric(self):
        return self.cached_metric is not None

    @property
    def packed_mesh_layouts(self):
        """The packed layouts to emit, as a sequence rather than a flag.

        Emission iterates this; it does not test it.  A branch reading a plan
        is still emission choosing what to emit, which is the distinction
        `test_emission_is_a_printer` measures and the reason this is a tuple:
        an empty one emits nothing without anybody deciding anything at the
        point of emission.
        """
        return ("packed",) if self.emits_packed else ()

    @property
    def affine_modes(self):
        """`("affine",)` or `()`, for the same reason `packed_mesh_layouts` is a
        sequence: emission iterates, it does not test."""
        return ("affine",) if self.emits_affine else ()

    @property
    def isoparametric_modes(self):
        """`("isoparametric",)` or `()`.  Empty for a constant-P1 simplex, whose
        affine kernel already computes what this one would."""
        return ("isoparametric",) if self.emits_isoparametric else ()

    @property
    def geometry_modes(self):
        """The geometry kernels to emit, in the order they are emitted.

        Same reasoning as `packed_mesh_layouts`.  A constant-P1 simplex yields
        one mode where every other element yields two, which is the whole of
        the "no need for both kernels" rule expressed as data.
        """
        modes = []
        if self.emits_affine:
            modes.append("affine")
        if self.emits_isoparametric:
            modes.append("isoparametric")
        return tuple(modes)

    def to_dict(self):
        return {
            "emits_affine": self.emits_affine,
            "emits_isoparametric": self.emits_isoparametric,
            "emits_packed": self.emits_packed,
            "emits_metric": self.emits_metric,
        }


def geometry_variant_plan(weak_form, rule, *, specialized=True, assembles_matrix=False):
    """The variants this form publishes on the element `rule` describes.

    `specialized` is the emitter's own answer to whether it has a specialized
    kernel prefix to hang the metric off; it is passed rather than guessed
    because it is a fact about the emission unit, not about the form.

    `assembles_matrix` says whether this form also emits matrix-assembly
    kernels.  It has to, and the reason is a wrinkle rather than a principle:
    assembly is emitted *inside* the isoparametric mesh operator rather than as
    an axis of its own, so a P1 element that published no isoparametric mode
    lost its `hessian_crs` and `hessian_bsr` entry points along with the
    matrix-free kernel it meant to drop.  The equivalence argument -- that a
    constant Jacobian makes the two kernels compute the same numbers -- is
    about the matrix-free kernels, and applies only to them until assembly gets
    an axis of its own.
    """
    if rule is None:
        return GeometryVariantPlan(True, True, True, None)
    dim = int(rule.dim)
    constant_p1 = _is_constant_p1_simplex_rule(rule)
    metric = cached_metric_geometry(weak_form, rule) if specialized else None
    return GeometryVariantPlan(
        # Both geometry modes, for now.  The rules below are written down and
        # the emitters already iterate them, but turning them on drops kernels
        # that four tests still expect and, in one case, that a Taylor-Hood
        # material appears to need: enabling them made
        # `poro_hyperelasticity_solid_gradient_2d_isoparametric_mesh_soa`
        # disappear on TRI6_TRI3, and TRI6 is not a constant-P1 simplex, so the
        # rule as written should not have touched it.  Something about the
        # compatible-element rule is not what this function assumes, and
        # shipping a kernel-dropping policy whose behaviour is not understood
        # is how a material quietly stops working.
        #
        #     emits_affine=constant_p1 or dim == 3,
        #     emits_isoparametric=(not constant_p1) or assembles_matrix,
        #
        emits_affine=True,
        emits_isoparametric=True,
        emits_packed=dim == 3,
        cached_metric=metric,
    )
