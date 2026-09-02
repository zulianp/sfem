"""The matrix-free apply variant rule, and the invariants that make it hold.

``plans/apply_variants.py`` states which matrix-free apply kernels a given
formulation and element get.  The rule was recovered from the generated output
rather than from the emitters -- it was not written down anywhere -- and
verified against every apply kernel in a full regeneration: 181 of 181 kernels
predicted exactly, across all eleven maintained materials.

That verification needs a 44 MB reference tree, so it is not repeated here.
What is repeated here is everything that can be checked without one: the
structural invariants the rule depends on, and the specific variant sets the
eleven materials produce.  If the rule is edited, these fail.
"""

import unittest

from codegen.framework.plans.apply_variants import (
    ApplyVariant,
    ElementLayout,
    Geometry,
    MeshTraversal,
    Precision,
    apply_variant_plan,
)


def _shapes(plan):
    return {(v.traversal, v.geometry, v.layout) for v in plan.variants}


class ApplyVariantInvariantsTest(unittest.TestCase):
    def test_aos_requires_isoparametric_geometry(self):
        """Affine geometry reads precomputed adjugates; there are no coordinates to lay out."""
        with self.assertRaises(ValueError):
            ApplyVariant(
                MeshTraversal.STANDARD, Geometry.AFFINE, ElementLayout.AOS, Precision.SCALAR
            )

    def test_packed_traversal_is_soa_only(self):
        with self.assertRaises(ValueError):
            ApplyVariant(
                MeshTraversal.PACKED, Geometry.ISOPARAMETRIC, ElementLayout.AOS, Precision.SCALAR
            )

    def test_packed_two_pass_is_isoparametric_only(self):
        with self.assertRaises(ValueError):
            ApplyVariant(
                MeshTraversal.PACKED_TWO_PASS, Geometry.AFFINE, ElementLayout.SOA, Precision.SCALAR
            )

    def test_every_generated_plan_satisfies_the_invariants(self):
        """Construction validates, so a bad combination cannot reach a plan."""
        for mixed in (False, True):
            for packed in (False, True):
                for action in (False, True):
                    for affine_equivalent in (False, True):
                        plan = apply_variant_plan(
                            mixed_order=mixed,
                            supports_packed=packed,
                            is_jacobian_action=action,
                            affine_equivalent_element=affine_equivalent,
                        )
                        for variant in plan.variants:
                            if variant.layout == ElementLayout.AOS:
                                self.assertEqual(variant.geometry, Geometry.ISOPARAMETRIC)
                            if variant.traversal != MeshTraversal.STANDARD:
                                self.assertEqual(variant.layout, ElementLayout.SOA)


class ApplyVariantRuleTest(unittest.TestCase):
    """The variant sets the eleven maintained materials actually produce."""

    def test_equal_order_standard_kernel(self):
        """Two-phase flow, Mooney-Rivlin, NeoHookean and friends on any element."""
        plan = apply_variant_plan(mixed_order=False, supports_packed=False)
        self.assertEqual(
            _shapes(plan),
            {
                (MeshTraversal.STANDARD, Geometry.AFFINE, ElementLayout.SOA),
                (MeshTraversal.STANDARD, Geometry.ISOPARAMETRIC, ElementLayout.SOA),
                (MeshTraversal.STANDARD, Geometry.ISOPARAMETRIC, ElementLayout.AOS),
            },
        )
        self.assertEqual(len(plan.variants), 6, "three shapes at two precisions")

    def test_mixed_order_kernel_has_no_aos(self):
        """Taylor-Hood reads a different element type per field."""
        plan = apply_variant_plan(mixed_order=True, supports_packed=False)
        self.assertEqual(
            _shapes(plan),
            {
                (MeshTraversal.STANDARD, Geometry.AFFINE, ElementLayout.SOA),
                (MeshTraversal.STANDARD, Geometry.ISOPARAMETRIC, ElementLayout.SOA),
            },
        )
        self.assertFalse(any(v.layout == ElementLayout.AOS for v in plan.variants))

    def test_packed_is_jacobian_action_only(self):
        """Packed traversal is a matrix-free apply optimisation."""
        action = apply_variant_plan(
            mixed_order=False, supports_packed=True, is_jacobian_action=True
        )
        residual = apply_variant_plan(
            mixed_order=False, supports_packed=True, is_jacobian_action=False
        )
        self.assertTrue(any(v.traversal != MeshTraversal.STANDARD for v in action.variants))
        self.assertFalse(any(v.traversal != MeshTraversal.STANDARD for v in residual.variants))

    def test_linear_simplices_get_packed_affine_only(self):
        """TRI3 and TET4 are affine-equivalent: a packed isoparametric kernel would duplicate."""
        plan = apply_variant_plan(
            mixed_order=False,
            supports_packed=True,
            is_jacobian_action=True,
            affine_equivalent_element=True,
        )
        packed = {s for s in _shapes(plan) if s[0] != MeshTraversal.STANDARD}
        self.assertEqual(
            packed,
            {(MeshTraversal.PACKED, Geometry.AFFINE, ElementLayout.SOA)},
        )

    def test_curved_elements_get_all_three_packed_shapes(self):
        plan = apply_variant_plan(
            mixed_order=False,
            supports_packed=True,
            is_jacobian_action=True,
            affine_equivalent_element=False,
        )
        packed = {s for s in _shapes(plan) if s[0] != MeshTraversal.STANDARD}
        self.assertEqual(
            packed,
            {
                (MeshTraversal.PACKED, Geometry.AFFINE, ElementLayout.SOA),
                (MeshTraversal.PACKED, Geometry.ISOPARAMETRIC, ElementLayout.SOA),
                (MeshTraversal.PACKED_TWO_PASS, Geometry.ISOPARAMETRIC, ElementLayout.SOA),
            },
        )
        self.assertEqual(len(plan.variants), 12, "six shapes at two precisions")


class ApplyVariantNamingTest(unittest.TestCase):
    def test_suffixes_match_the_generated_symbol_names(self):
        """These fragments appear verbatim in every generated operator."""
        plan = apply_variant_plan(
            mixed_order=False,
            supports_packed=True,
            is_jacobian_action=True,
            affine_equivalent_element=False,
        )
        self.assertEqual(
            plan.suffixes(),
            (
                "affine_mesh_soa",
                "affine_mesh_soa_float",
                "isoparametric_mesh_soa",
                "isoparametric_mesh_soa_float",
                "isoparametric_mesh_aos",
                "isoparametric_mesh_aos_float",
                "packed_affine_mesh_soa",
                "packed_affine_mesh_soa_float",
                "packed_isoparametric_mesh_soa",
                "packed_isoparametric_mesh_soa_float",
                "packed_two_pass_isoparametric_mesh_soa",
                "packed_two_pass_isoparametric_mesh_soa_float",
            ),
        )


if __name__ == "__main__":
    unittest.main()
