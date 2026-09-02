"""The geometry plan and the element specialization must not drift apart.

Two objects currently describe the same geometry facts.  ``GeometryPlanNode``
states the decision -- affine or isoparametric, how the Jacobian is evaluated,
how many geometry streams cross the boundary and how many geometry points each
element carries.  ``SfemSoAElementSpecialization`` produces the concrete stream
inputs the kernel is handed.  Today the emitters read the second and ignore the
first, so nothing forces the two to agree, and a change to either could silently
contradict the other.

That matters for the migration rather than for correctness today: the point of
moving geometry into the plan is that the plan becomes the authority, and it can
only become the authority if it already tells the truth.  These tests establish
that it does, for every supported element, before anything starts depending on
it.

The comparison has one subtlety worth stating, because getting it wrong looks
exactly like a bug.  ``adjugate_geometry_inputs()`` returns three streams:
``grad_ref``, ``jacobian_adjugate`` and ``jacobian_determinant``.  The first is
reference basis data, not geometry, and the plan node rightly does not count it.
Comparing all three against ``geometry_stream_count`` reports a mismatch on every
element and means nothing.

The points-per-element rule is likewise not the obvious one.  It is not
``n_shape``: a tensor-product element does carry geometry at every node, but a
*linear* simplex is affine and carries a single geometry point even in the
isoparametric plan, and a higher-order simplex carries geometry at its vertices
only.  Asserting ``points == n_shape`` fails on TRI3 and TET4 and says nothing
about the elements it passes on.
"""

import unittest

from codegen.framework.fem.geometry import GeometryMode
from codegen.framework.fem.reference import sfem_supported_element_types
from codegen.framework.plans.emission import emission_plan_for_element


#: Streams that are geometry.  `grad_ref` is reference basis data and is excluded.
GEOMETRY_STREAM_NAMES = ("jacobian_adjugate", "jacobian_determinant")

VECTOR_SIZE = 8


def _emission_plans():
    for element_type in sfem_supported_element_types():
        try:
            yield element_type, emission_plan_for_element(element_type, VECTOR_SIZE, None)
        except (ValueError, TypeError, KeyError):
            # Elements with no affine/isoparametric pair are out of scope here.
            continue


class GeometryPlanAgreesTest(unittest.TestCase):
    def test_affine_stream_count_matches_the_specialization(self):
        checked = 0
        for element_type, plan in _emission_plans():
            streams = plan.affine_specialization.adjugate_geometry_inputs()
            counted = sum(
                int(stream.components)
                for stream in streams
                if stream.name in GEOMETRY_STREAM_NAMES
            )
            self.assertEqual(
                plan.affine_geometry.geometry_stream_count,
                counted,
                "%s: the geometry plan says %d affine geometry streams, the "
                "specialization produces %d"
                % (
                    element_type,
                    plan.affine_geometry.geometry_stream_count,
                    counted,
                ),
            )
            checked += 1
        self.assertGreater(checked, 0, "no element produced an emission plan")

    def test_affine_geometry_carries_one_point_per_element(self):
        """Affine geometry is constant over the cell; that is what makes it affine."""
        for element_type, plan in _emission_plans():
            self.assertEqual(
                plan.affine_geometry.geometry_points_per_element,
                1,
                "%s: affine geometry claims %d points per element"
                % (element_type, plan.affine_geometry.geometry_points_per_element),
            )
            self.assertEqual(plan.affine_geometry.jacobian_scope, "element")

    def test_isoparametric_geometry_is_evaluated_per_quadrature_point(self):
        for element_type, plan in _emission_plans():
            geometry = plan.isoparametric_geometry
            self.assertIs(geometry.mode, GeometryMode.ISOPARAMETRIC)
            self.assertEqual(
                geometry.jacobian_scope,
                "quadrature_point",
                "%s: isoparametric geometry is not evaluated per quadrature point"
                % element_type,
            )
            n_shape = plan.isoparametric_specialization.n_shape
            dim = plan.isoparametric_specialization.dim
            points = geometry.geometry_points_per_element
            if plan.family == "tensor_product":
                # Every node carries geometry, so the map is evaluated from all
                # of them by sum factorization.
                self.assertEqual(
                    points,
                    n_shape,
                    "%s: tensor-product geometry should use all %d nodes, uses %d"
                    % (element_type, n_shape, points),
                )
            elif points == 1:
                # A linear simplex is affine: the Jacobian is constant, so one
                # geometry point describes the whole cell even in the
                # isoparametric plan.
                self.assertEqual(
                    n_shape,
                    dim + 1,
                    "%s: only a linear simplex may carry a single geometry point"
                    % element_type,
                )
            else:
                # A higher-order simplex carries geometry at its vertices.
                self.assertEqual(
                    points,
                    dim + 1,
                    "%s: simplex geometry should use its %d vertices, uses %d"
                    % (element_type, dim + 1, points),
                )
                self.assertGreater(n_shape, dim + 1)

    def test_sum_factorization_is_claimed_only_for_tensor_product_elements(self):
        for element_type, plan in _emission_plans():
            if plan.isoparametric_geometry.uses_sum_factorization:
                self.assertEqual(
                    plan.family,
                    "tensor_product",
                    "%s: geometry claims sum factorization on a %s element"
                    % (element_type, plan.family),
                )


if __name__ == "__main__":
    unittest.main()
