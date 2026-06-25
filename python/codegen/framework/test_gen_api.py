import os
import tempfile
import unittest

from sfem import gen

from .materials.neohookean_ogden import material as neohookean_ogden
from .materials.two_phase_flow import material as two_phase_flow


class GenApiTest(unittest.TestCase):
    def test_generates_hyperelastic_material(self):
        with tempfile.TemporaryDirectory() as out_dir:
            result = gen.generate(
                neohookean_ogden,
                out_dir,
                elements=("TRI3",),
            )
            names = {os.path.basename(path) for path in result.sources}
            self.assertIn(
                "generated_neohookean_ogden_tri3_operator.cpp",
                names,
            )
            self.assertIn("kernel_diagnostics.hpp", names)
            self.assertIn("sfem_GeneratedNeoHookeanOgden.cpp", names)
            wrapper = os.path.join(
                out_dir,
                "op",
                "sfem_GeneratedNeoHookeanOgden.cpp",
            )
            with open(wrapper, encoding="utf-8") as stream:
                source = stream.read()
            header = os.path.join(
                out_dir,
                "op",
                "sfem_GeneratedNeoHookeanOgden.hpp",
            )
            with open(header, encoding="utf-8") as stream:
                self.assertIn("public Op", stream.read())
            self.assertIn(
                "generated_neohookean_ogden_tri3_tri3_gradient_isoparametric_mesh_soa",
                source,
            )

    def test_generates_coupled_residual_material(self):
        with tempfile.TemporaryDirectory() as out_dir:
            result = gen.generate(
                two_phase_flow,
                out_dir,
                elements=("TRI3",),
            )
            names = {os.path.basename(path) for path in result.sources}
            self.assertIn(
                "generated_two_phase_flow_tri3_operator.cpp",
                names,
            )
            self.assertIn(
                "generated_two_phase_flow_d2_simplex_local.hpp",
                names,
            )
            self.assertIn("sfem_GeneratedTwoPhaseFlow.cpp", names)

    def test_rejects_elements_outside_material_contract(self):
        with tempfile.TemporaryDirectory() as out_dir:
            with self.assertRaisesRegex(ValueError, "not enabled"):
                gen.generate(
                    two_phase_flow,
                    out_dir,
                    elements=("HEX27",),
                )


if __name__ == "__main__":
    unittest.main()
