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

    def to_dict(self):
        return {
            "emits_affine": self.emits_affine,
            "emits_isoparametric": self.emits_isoparametric,
            "emits_packed": self.emits_packed,
            "emits_metric": self.emits_metric,
        }


def geometry_variant_plan(weak_form, rule, *, specialized=True):
    """The variants this form publishes on the element `rule` describes.

    `specialized` is the emitter's own answer to whether it has a specialized
    kernel prefix to hang the metric off; it is passed rather than guessed
    because it is a fact about the emission unit, not about the form.
    """
    if rule is None:
        return GeometryVariantPlan(True, True, True, None)
    dim = int(rule.dim)
    constant_p1 = _is_constant_p1_simplex_rule(rule)
    metric = cached_metric_geometry(weak_form, rule) if specialized else None
    return GeometryVariantPlan(
        emits_affine=constant_p1 or dim == 3,
        emits_isoparametric=not constant_p1,
        emits_packed=dim == 3,
        cached_metric=metric,
    )
