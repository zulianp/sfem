import os
import shutil
import subprocess
import tempfile
import unittest

import sympy as sp

from .residual import CoupledResidualSystem
from .residual_codegen import (
    coupled_residual_weak_coefficients,
    generate_coupled_residual_sfem_files,
    weak_residual_coefficients,
)
from .symbolic import ExpressionRole


def two_field_diffusion_system(dim=2):
    system = CoupledResidualSystem(dim=dim)
    u = system.add_field("u")
    v = system.add_field("v")
    dt, k_u, k_v, coupling = sp.symbols("dt k_u k_v coupling")
    system.add_parameters(dt, k_u, k_v, coupling)
    system.set_residual(
        u,
        (u.value - u.previous_value) * u.test_value / dt
        + k_u * sum(u.gradient[d] * u.test_gradient[d] for d in range(dim))
        + coupling * (u.value - v.value) * u.test_value,
    )
    system.set_residual(
        v,
        (v.value - v.previous_value) * v.test_value / dt
        + k_v * sum(v.gradient[d] * v.test_gradient[d] for d in range(dim))
        + coupling * (v.value - u.value) * v.test_value,
    )
    return system, u, v


class CoupledResidualSystemTest(unittest.TestCase):
    def test_decomposes_residual_and_action_into_test_coefficients(self):
        system, u, v = two_field_diffusion_system()
        dt, k_u, _, coupling = system.parameters

        residual = coupled_residual_weak_coefficients(system)
        action = coupled_residual_weak_coefficients(system, jacobian_action=True)

        self.assertEqual(
            sp.simplify(
                residual[0].value
                - ((u.value - u.previous_value) / dt
                   + coupling * (u.value - v.value))
            ),
            0,
        )
        self.assertEqual(
            tuple(sp.simplify(value) for value in residual[0].gradient),
            tuple(k_u * value for value in u.gradient),
        )
        self.assertEqual(
            sp.simplify(
                action[0].value
                - ((u.direction_value / dt)
                   + coupling * (u.direction_value - v.direction_value))
            ),
            0,
        )
        self.assertEqual(
            tuple(sp.simplify(value) for value in action[0].gradient),
            tuple(k_u * value for value in u.direction_gradient),
        )

        with self.assertRaisesRegex(ValueError, "linear"):
            weak_residual_coefficients(
                system,
                u.value * u.test_value**2,
                u.name,
            )

    def test_registers_complete_field_semantics(self):
        system = CoupledResidualSystem(3)
        field = system.add_field("pressure")

        self.assertEqual(field.dim, 3)
        self.assertEqual(len(field.gradient), 3)
        self.assertEqual(len(field.previous_gradient), 3)
        self.assertEqual(len(field.test_gradient), 3)
        self.assertEqual(len(field.direction_gradient), 3)
        self.assertEqual(field.variables, (field.value,) + field.gradient)
        self.assertEqual(field.directions, (field.direction_value,) + field.direction_gradient)

    def test_preserves_residual_and_block_identity(self):
        system, _, _ = two_field_diffusion_system()
        residual_graph = system.build_residual_graph()
        action_graph = system.build_jacobian_action_graph(include_blocks=True)

        self.assertEqual(
            tuple(output.name for output in residual_graph.outputs),
            ("residual_u", "residual_v"),
        )
        self.assertEqual(
            tuple(output.role for output in residual_graph.outputs),
            (ExpressionRole.RESIDUAL, ExpressionRole.RESIDUAL),
        )
        self.assertEqual(
            tuple(output.name for output in action_graph.outputs),
            (
                "jacobian_u_u",
                "jacobian_u_v",
                "jacobian_v_u",
                "jacobian_v_v",
                "jacobian_action_u",
                "jacobian_action_v",
            ),
        )
        self.assertEqual(
            tuple(
                (block.row_field, block.column_field)
                for block in system.jacobian_blocks()
            ),
            (("u", "u"), ("u", "v"), ("v", "u"), ("v", "v")),
        )

    def test_generates_and_compiles_material_agnostic_kernels(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("c++ compiler is not available")

        system, _, _ = two_field_diffusion_system()
        kernels = system.generate_cpp_kernels("coupled_diffusion")
        self.assertIn('extern "C" void coupled_diffusion_residual', kernels.residual.source)
        self.assertIn(
            'extern "C" void coupled_diffusion_jacobian_action',
            kernels.jacobian_action.source,
        )
        self.assertNotIn("two_phase", kernels.residual.source)
        self.assertNotIn("two_phase", kernels.jacobian_action.source)

        with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir:
            for generated in (kernels.residual, kernels.jacobian_action):
                source = os.path.join(tmpdir, "%s.cpp" % generated.function_name)
                output = os.path.join(tmpdir, "%s.o" % generated.function_name)
                with open(source, "w", encoding="utf-8") as stream:
                    stream.write(generated.source)
                subprocess.run(
                    [compiler, "-std=c++11", "-O3", "-c", source, "-o", output],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

    def test_generates_vectorized_element_local_residual_kernels(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("c++ compiler is not available")

        with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir:
            generated_sources = []
            for element, dim in (
                ("TRI3", 2),
                ("TET4", 3),
                ("QUAD4", 2),
                ("HEX8", 3),
            ):
                system, _, _ = two_field_diffusion_system(dim)
                files = generate_coupled_residual_sfem_files(
                    system,
                    prefix="coupled_diffusion",
                    element_type=element,
                )
                for generated in files:
                    path = os.path.join(tmpdir, generated.path)
                    with open(path, "w", encoding="utf-8") as stream:
                        stream.write(generated.source)
                local_source = files[1].source
                operator_source = files[2].source
                self.assertIn("for (int q = 0; q < N_QP; ++q)", local_source)
                self.assertIn("#pragma omp simd", local_source)
                self.assertNotIn("two_phase", local_source)
                self.assertIn(
                    "coupled_diffusion_%s_residual_element_soa_float"
                    % element.lower(),
                    operator_source,
                )
                self.assertIn(
                    "coupled_diffusion_%s_jacobian_action_element_soa"
                    % element.lower(),
                    operator_source,
                )
                if element in ("QUAD4", "HEX8"):
                    self.assertIn("_tensor_evaluate", local_source)
                    self.assertIn("_tensor_integrate", local_source)
                else:
                    self.assertNotIn("_tensor_evaluate", local_source)
                generated_sources.append(
                    os.path.join(tmpdir, files[2].path)
                )

            for index, source in enumerate(generated_sources):
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
                        tmpdir,
                        "-o",
                        os.path.join(tmpdir, "element_%d.o" % index),
                    ],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

            version = subprocess.run(
                [compiler, "--version"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            ).stdout.lower()
            if "clang" in version:
                report = subprocess.run(
                    [
                        compiler,
                        "-std=c++14",
                        "-O3",
                        "-fopenmp-simd",
                        "-Rpass=loop-vectorize",
                        "-c",
                        generated_sources[-1],
                        "-I",
                        tmpdir,
                        "-o",
                        os.path.join(tmpdir, "vectorized.o"),
                    ],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                ).stderr
                self.assertIn("vectorized loop", report)

    def test_reports_invalid_fields_dimensions_and_symbols(self):
        with self.assertRaisesRegex(ValueError, "dimension"):
            CoupledResidualSystem(4)

        system = CoupledResidualSystem(2)
        u = system.add_field("u")
        with self.assertRaisesRegex(ValueError, "already registered"):
            system.add_field("u")
        with self.assertRaisesRegex(ValueError, "unknown residual field"):
            system.field("v")
        with self.assertRaisesRegex(ValueError, "must be scalar"):
            system.set_residual(u, sp.Matrix([u.value, u.value]))
        unknown = sp.Symbol("unknown")
        with self.assertRaisesRegex(ValueError, "unregistered symbols"):
            system.set_residual(u, u.value + unknown)
        with self.assertRaisesRegex(ValueError, "missing residual equations"):
            system.build_residual_graph()


if __name__ == "__main__":
    unittest.main()
