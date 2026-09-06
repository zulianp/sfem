"""The wrapper takes its affine geometry from the kernels, not from a guess.

Which geometry an affine kernel wants -- the symmetric gradient metric or the
full Jacobian adjugate and determinant -- is a property of the form being
lowered, decided once in ``plans.form_transformations.cached_metric_geometry``
and visible afterwards in the kernel's own signature.  The wrapper reads it
there.  It used to spell the adjugate unconditionally, and a metric kernel then
got five geometry arguments where it took three.

Two elements of one dimension can also disagree: an affine simplex contracts
through the metric where a hexahedron needs the adjugate.  That is two ABIs
under one dispatch name, and the second used to be dropped silently, leaving
the metric elements with no affine entry point and a runtime failure as the
only sign.  They are kept apart by name now.
"""

import unittest

from codegen.framework.package.op_wrappers import (
    _affine_dispatch_uses_metric,
    _affine_geometry_call_args,
    _geometry_qualified_dispatch_name,
    _metric_dispatch_name,
)


METRIC_KERNEL = '''
extern "C" int demo_gradient_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const geom_t *const SFEM_RESTRICT g_geom_metric0,
        const geom_t *const SFEM_RESTRICT g_geom_metric5
);
'''

ADJUGATE_KERNEL = '''
extern "C" int demo_apply_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0
);
'''

# An element kernel is declared in its own header without `extern "C"`; it has
# to answer here too, because the per-element call sites ask about it.
ELEMENT_KERNEL = '''
int demo_tet4_gradient_affine_mesh_soa(
        const ptrdiff_t nelements,
        const geom_t *const SFEM_RESTRICT g_geom_metric0
);
'''

SOURCES = {
    "a.hpp": METRIC_KERNEL,
    "b.hpp": ADJUGATE_KERNEL,
    "c.hpp": ELEMENT_KERNEL,
}


class AffineGeometryFollowsTheKernelTest(unittest.TestCase):
    def test_metric_kernel_is_handed_the_metric(self):
        self.assertTrue(
            _affine_dispatch_uses_metric(SOURCES, "demo_gradient_3d_affine_mesh_soa")
        )
        self.assertEqual(
            _affine_geometry_call_args(SOURCES, "demo_gradient_3d_affine_mesh_soa", 3),
            tuple("geom_metric[%d]" % index for index in range(6)),
        )

    def test_adjugate_kernel_is_handed_the_adjugate_and_determinant(self):
        self.assertFalse(
            _affine_dispatch_uses_metric(SOURCES, "demo_apply_3d_affine_mesh_soa")
        )
        self.assertEqual(
            _affine_geometry_call_args(SOURCES, "demo_apply_3d_affine_mesh_soa", 3),
            tuple(["adjugate[%d]" % index for index in range(9)] + ["determinant"]),
        )

    def test_two_dimensions_get_their_own_component_counts(self):
        self.assertEqual(
            len(
                _affine_geometry_call_args(
                    SOURCES, "demo_gradient_3d_affine_mesh_soa", 2
                )
            ),
            3,
        )

    def test_an_element_kernel_answers_without_extern_c(self):
        self.assertTrue(
            _affine_dispatch_uses_metric(SOURCES, "demo_tet4_gradient_affine_mesh_soa")
        )

    def test_an_undeclared_kernel_falls_back_to_the_general_geometry(self):
        self.assertFalse(_affine_dispatch_uses_metric(SOURCES, "demo_absent"))

    def test_a_call_site_is_not_mistaken_for_a_declaration(self):
        caller = {"d.cpp": "return demo_gradient_3d_affine_mesh_soa(a, b, c);"}
        self.assertIsNone(
            __import__(
                "codegen.framework.package.op_wrappers", fromlist=["x"]
            )._affine_dispatch_parameters(caller, "demo_gradient_3d_affine_mesh_soa")
        )

    def test_colliding_dispatch_signatures_are_named_apart(self):
        metric = ("const geom_t *const SFEM_RESTRICT g_geom_metric0",)
        adjugate = ("const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0",)
        self.assertEqual(
            _geometry_qualified_dispatch_name("demo_gradient_3d_affine_mesh_soa", metric),
            "demo_gradient_3d_affine_metric_mesh_soa",
        )
        # The adjugate keeps the plain name: it is the shape every element can
        # be handed, so it is the one a caller finds without asking.
        self.assertEqual(
            _geometry_qualified_dispatch_name("demo_gradient_3d_affine_mesh_soa", adjugate),
            "demo_gradient_3d_affine_mesh_soa",
        )

    def test_the_metric_sibling_is_derivable_from_the_plain_name(self):
        self.assertEqual(
            _metric_dispatch_name("demo_gradient_3d_affine_mesh_soa"),
            "demo_gradient_3d_affine_metric_mesh_soa",
        )


if __name__ == "__main__":
    unittest.main()
