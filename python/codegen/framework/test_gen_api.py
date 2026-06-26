import os
import shutil
import subprocess
import tempfile
import unittest

from sfem import gen

from .materials.neohookean_ogden import material as neohookean_ogden
from .materials.poro_hyperelasticity import material as poro_hyperelasticity
from .materials.two_phase_flow import material as two_phase_flow


class GenApiTest(unittest.TestCase):
    def test_equation_system_accepts_vector_scalar_energy_and_residual_equations(self):
        system = gen.EquationSystem(3)
        displacement = system.vector_field("u", family="displacement")
        pressure = system.scalar_field("p", family="pressure")

        energy = system.energy(
            "solid",
            lambda F: F[0, 0],
            fields=(displacement,),
        )
        residual = system.residual(
            "flow",
            lambda residual_system: None,
            fields=(displacement, pressure),
        )

        self.assertTrue(displacement.is_vector)
        self.assertTrue(pressure.is_scalar)
        self.assertEqual(energy.form, gen.EquationForm.ENERGY)
        self.assertEqual(residual.form, gen.EquationForm.RESIDUAL)
        self.assertEqual(system.equations, (energy, residual))

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

    def test_poro_hyperelastic_material_uses_taylor_hood_elements(self):
        names = tuple(element.name for element in poro_hyperelasticity.elements)
        self.assertEqual(names, ("TRI6_TRI3", "TET10_TET4", "HEX27_HEX8"))

    def test_rejects_equal_order_poro_hyperelastic_element(self):
        with tempfile.TemporaryDirectory() as out_dir:
            with self.assertRaisesRegex(ValueError, "not enabled"):
                gen.generate(
                    poro_hyperelasticity,
                    out_dir,
                    elements=("TRI3",),
                )

    def test_generates_taylor_hood_poro_hyperelastic_material(self):
        with tempfile.TemporaryDirectory() as out_dir:
            result = gen.generate(
                poro_hyperelasticity,
                out_dir,
                elements=("TRI6_TRI3",),
            )
            names = {os.path.basename(path) for path in result.sources}
            self.assertIn(
                "generated_poro_hyperelasticity_solid_tri6_operator.cpp",
                names,
            )
            self.assertIn(
                "generated_poro_hyperelasticity_poro_tri6_tri3_operator.cpp",
                names,
            )

    def test_compiles_taylor_hood_poro_residual_operator(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("c++ compiler is not available")
        with tempfile.TemporaryDirectory() as out_dir:
            gen.generate(
                poro_hyperelasticity,
                out_dir,
                elements=("TRI6_TRI3",),
            )
            source = os.path.join(
                out_dir,
                "generated_poro_hyperelasticity_poro_tri6_tri3_operator.cpp",
            )
            subprocess.run(
                [
                    compiler,
                    "-std=c++14",
                    "-O3",
                    "-fopenmp-simd",
                    "-Werror",
                    "-c",
                    source,
                    "-I",
                    out_dir,
                    "-o",
                    os.path.join(out_dir, "poro_residual.o"),
                ],
                check=True,
            )

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
