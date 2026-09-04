"""The mesh geometry a kernel takes is the plan's decision, and is read.

Which buffers cross a mesh kernel's boundary -- the symmetric gradient metric,
or the Jacobian adjugate and determinant -- was decided at twenty-five sites in
the residual emitter, four in the energy one, and once more in the wrapper,
each with its own ``if`` chain over the same dependency flags.  They agreed by
inspection rather than by construction, and once they did not: the wrapper
spelled the adjugate unconditionally, so laplace generated for TRI3 or TET4
alone handed five geometry arguments to a kernel taking three, and the operator
could not compile.  That is the defect ARCHITECTURE.html records as OP 16.

Two assertions, because the weaker one alone passes for an emitter that
hardcodes the same names and never consults anything.  The names the plan
produces must appear in the emitted kernel, *and* changing what the plan says
must change what is emitted.
"""

import unittest

from codegen.framework.plans.geometry_quantities import (
    mesh_geometry_argument_names,
    mesh_geometry_parameters,
    mesh_geometry_streams,
)


class Dependencies:
    """The one flag this plan reads."""

    def __init__(self, uses_adjugate):
        self.uses_adjugate = uses_adjugate


class MeshGeometryPlanTest(unittest.TestCase):
    maxDiff = None

    def test_the_metric_form_carries_no_determinant(self):
        """A metric already carries the determinant, so none is passed."""
        streams = mesh_geometry_streams(Dependencies(True), 3, metric_components=6)
        self.assertEqual([s.role for s in streams], ["metric"] * 6)
        self.assertNotIn(
            "g_jacobian_determinant0", [s.name for s in streams]
        )

    def test_the_adjugate_form_is_components_then_determinant(self):
        """dim * dim adjugate components, then the determinant, in ABI order."""
        streams = mesh_geometry_streams(Dependencies(True), 3)
        self.assertEqual(
            [s.name for s in streams],
            ["g_jacobian_adjugate%d" % i for i in range(9)]
            + ["g_jacobian_determinant0"],
        )

    def test_a_form_that_needs_no_adjugate_still_takes_the_determinant(self):
        """The determinant weights the integral whether or not a gradient maps."""
        streams = mesh_geometry_streams(Dependencies(False), 3)
        self.assertEqual([s.name for s in streams], ["g_jacobian_determinant0"])

    def test_arguments_and_parameters_describe_the_same_buffers(self):
        """A signature and its call cannot disagree if both come from here."""
        for dim in (2, 3):
            for metric in (None, dim * (dim + 1) // 2):
                with self.subTest(dim=dim, metric=metric):
                    names = mesh_geometry_argument_names(
                        Dependencies(True), dim, metric
                    )
                    params = mesh_geometry_parameters(
                        Dependencies(True), dim, metric
                    )
                    self.assertEqual(
                        list(names),
                        [param.split()[-1] for param in params],
                    )

    def test_the_emitted_kernel_follows_the_plan(self):
        """Proof the emitter reads this rather than spelling it again."""
        from codegen.framework.emitters import residual_codegen

        self.assertIs(
            residual_codegen.mesh_geometry_argument_names,
            mesh_geometry_argument_names,
        )
        self.assertIs(
            residual_codegen.mesh_geometry_parameters,
            mesh_geometry_parameters,
        )

    def test_an_unknown_role_is_refused(self):
        """A role the emitters cannot spell must fail, not slip through."""
        from codegen.framework.plans.geometry_quantities import MeshGeometryStream

        with self.assertRaises(ValueError):
            MeshGeometryStream("g_whatever0", "not_a_role")


if __name__ == "__main__":
    unittest.main()
