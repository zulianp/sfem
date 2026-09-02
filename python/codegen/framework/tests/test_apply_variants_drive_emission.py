"""The apply-variant plan drives emission, rather than describing it.

There is a real difference between a plan that records what the emitters happen
to do and a plan that decides it, and it is easy to claim the second while
shipping the first.  The difference is observable: change the plan, and the
generated kernel must change to match.

These tests change the precision axis of the matrix-free apply variant matrix
and assert the emitted source follows.  If the emitters ever go back to
carrying their own literal, the kernels stop responding and these fail --
which is what the byte-identity gate alone cannot tell you, because a plan that
is merely consulted and a plan that is ignored produce identical output.
"""

import unittest
from unittest import mock

import sympy as sp

from codegen.framework.emitters import residual_codegen
from codegen.framework.emitters.residual_codegen import (
    generate_coupled_residual_sfem_files,
)
from codegen.framework.plans.apply_variants import precision_axis
from codegen.framework.plans.emission import emission_plan_for_element
from codegen.framework.plans.residual_model import residual_emission_model_from_system
from codegen.framework.symbolic.residual import CoupledResidualSystem


def _diffusion_system(dim=2):
    system = CoupledResidualSystem(dim=dim)
    u = system.add_field("u")
    k = sp.Symbol("k")
    system.add_parameters(k)
    system.add_residual(
        u,
        u.value * u.test_value
        + k * sum(u.gradient[d] * u.test_gradient[d] for d in range(dim)),
    )
    return system


def _emit(prefix="probe"):
    system = _diffusion_system()
    return generate_coupled_residual_sfem_files(
        residual_emission_model_from_system(system),
        prefix=prefix,
        emission_plan=emission_plan_for_element("TRI3", 16, None),
    )


def _operator_source(files):
    for generated in files:
        if generated.path.endswith("_operator.cpp"):
            return generated.source
    raise AssertionError("no mesh operator source was emitted")


class ApplyVariantsDriveEmissionTest(unittest.TestCase):
    def test_the_default_axis_emits_both_precisions(self):
        source = _operator_source(_emit())
        for scalar_type, suffix in precision_axis():
            self.assertIn(
                "_mesh_soa%s(" % suffix,
                source,
                "the %s precision variant was not emitted" % (scalar_type,),
            )

    def test_removing_a_precision_removes_its_kernels(self):
        """The clearest evidence the plan decides: drop float, lose the float kernels."""
        with mock.patch.object(
            residual_codegen, "precision_axis", lambda: (("double", ""),)
        ):
            source = _operator_source(_emit())
        self.assertIn("_mesh_soa(", source)
        self.assertNotIn(
            "_mesh_soa_float(",
            source,
            "float kernels were still emitted after the plan dropped that "
            "precision, so the emitter is not reading the plan",
        )

    def test_renaming_a_precision_renames_its_kernels(self):
        """The suffix in the emitted symbol comes from the plan, not a literal."""
        with mock.patch.object(
            residual_codegen,
            "precision_axis",
            lambda: (("double", ""), ("float", "_reduced")),
        ):
            source = _operator_source(_emit())
        self.assertIn("_mesh_soa_reduced(", source)
        self.assertNotIn("_mesh_soa_float(", source)

    def test_a_third_precision_produces_a_third_set_of_kernels(self):
        """Adding a row to the variant matrix is a plan change, not an emitter change."""
        with mock.patch.object(
            residual_codegen,
            "precision_axis",
            lambda: (("double", ""), ("float", "_float"), ("long double", "_extended")),
        ):
            source = _operator_source(_emit())
        for suffix in ("", "_float", "_extended"):
            self.assertIn("_mesh_soa%s(" % suffix, source)
        self.assertIn(
            "long double",
            source,
            "the added precision's scalar type never reached the emitted source",
        )


if __name__ == "__main__":
    unittest.main()
