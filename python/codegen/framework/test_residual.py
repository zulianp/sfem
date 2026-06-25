import ctypes
import os
import re
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
from .symbolic import ExpressionRole, sfem_element_quadrature_rule


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


def _reference_coordinates(element):
    if element == "TRI3":
        return ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    if element == "HEX8":
        return (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 1.0),
        )
    raise ValueError(element)


def _deform_coordinates(coords):
    if len(coords[0]) == 2:
        return tuple((1.2 * x + 0.1 * y, -0.05 * x + 0.9 * y) for x, y in coords)
    return tuple(
        (
            1.1 * x + 0.08 * y,
            -0.04 * x + 0.95 * y + 0.05 * z,
            0.03 * x + 1.15 * z,
        )
        for x, y, z in coords
    )


def _shape_values(rule, q):
    if rule.element_type == "TRI3":
        return (1.0 / 3.0,) * 3
    if rule.element_type == "TET4":
        return (0.25,) * 4
    q1 = rule.tensor_product_n_qp_1d
    s1 = rule.tensor_product_n_shape_1d
    qx = q % q1
    qy = (q // q1) % q1
    qz = q // (q1 * q1)
    values = rule.tensor_product_shape_values_1d
    if rule.dim == 2:
        shape_indices = ((0, 0), (1, 0), (1, 1), (0, 1))
        return tuple(
            values[qx * s1 + sx] * values[qy * s1 + sy]
            for sx, sy in shape_indices
        )
    shape_indices = (
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
    )
    return tuple(
        values[qx * s1 + sx]
        * values[qy * s1 + sy]
        * values[qz * s1 + sz]
        for sx, sy, sz in shape_indices
    )


def _adjugate_and_determinant(matrix):
    if len(matrix) == 2:
        a, b = matrix[0]
        c, d = matrix[1]
        return ((d, -b), (-c, a)), a * d - b * c
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    adj = (
        (e * i - f * h, c * h - b * i, b * f - c * e),
        (f * g - d * i, a * i - c * g, c * d - a * f),
        (d * h - e * g, b * g - a * h, a * e - b * d),
    )
    det = a * adj[0][0] + b * adj[1][0] + c * adj[2][0]
    return adj, det


def _diffusion_reference(rule, coords, current, previous, direction=None):
    dim = rule.dim
    n_shape = rule.n_shape
    dt, k_u, k_v, coupling = (0.7, 1.3, 0.8, 0.25)
    output = [[0.0] * n_shape for _ in range(2)]
    accumulation_integral = [0.0, 0.0]
    for q, weight in enumerate(rule.weights):
        shape = _shape_values(rule, q)
        grad_ref = [
            rule.reference_gradients[(q * n_shape + s) * dim:
                                     (q * n_shape + s + 1) * dim]
            for s in range(n_shape)
        ]
        jacobian = [
            [
                sum(coords[s][i] * grad_ref[s][j] for s in range(n_shape))
                for j in range(dim)
            ]
            for i in range(dim)
        ]
        adjugate, determinant = _adjugate_and_determinant(jacobian)
        grad = [
            [
                sum(grad_ref[s][k] * adjugate[k][d] for k in range(dim))
                / determinant
                for d in range(dim)
            ]
            for s in range(n_shape)
        ]
        source = direction if direction is not None else current
        values = [
            sum(source[f][s] * shape[s] for s in range(n_shape))
            for f in range(2)
        ]
        gradients = [
            [
                sum(source[f][s] * grad[s][d] for s in range(n_shape))
                for d in range(dim)
            ]
            for f in range(2)
        ]
        if direction is None:
            old_values = [
                sum(previous[f][s] * shape[s] for s in range(n_shape))
                for f in range(2)
            ]
            value_coeff = (
                (values[0] - old_values[0]) / dt
                + coupling * (values[0] - values[1]),
                (values[1] - old_values[1]) / dt
                + coupling * (values[1] - values[0]),
            )
        else:
            value_coeff = (
                values[0] / dt + coupling * (values[0] - values[1]),
                values[1] / dt + coupling * (values[1] - values[0]),
            )
        grad_coeff = (
            tuple(k_u * value for value in gradients[0]),
            tuple(k_v * value for value in gradients[1]),
        )
        measure = weight * determinant
        for field in range(2):
            accumulation_integral[field] += measure * value_coeff[field]
            for test in range(n_shape):
                output[field][test] += measure * (
                    value_coeff[field] * shape[test]
                    + sum(
                        grad_coeff[field][d] * grad[test][d]
                        for d in range(dim)
                    )
                )
    return output, accumulation_integral


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
        unused = sp.Symbol("unused")
        system.add_parameters(unused)
        residual_dependencies = system.residual_dependencies()
        action_dependencies = system.jacobian_action_dependencies()
        self.assertTrue(residual_dependencies.current)
        self.assertTrue(residual_dependencies.previous)
        self.assertFalse(residual_dependencies.direction)
        self.assertFalse(action_dependencies.current)
        self.assertFalse(action_dependencies.previous)
        self.assertTrue(action_dependencies.direction)
        self.assertNotIn(unused, residual_dependencies.parameters)
        self.assertNotIn(unused, action_dependencies.parameters)
        self.assertNotIn(u.previous_value, system.jacobian_action_data_symbols())

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
                diagnostics_source = files[1].source
                local_source = files[2].source
                operator_source = files[3].source
                family = (
                    "tensor_product"
                    if element in ("QUAD4", "HEX8")
                    else "simplex"
                )
                self.assertEqual(
                    files[2].path,
                    "coupled_diffusion_d%d_%s_local.hpp" % (dim, family),
                )
                self.assertEqual(
                    files[3].path,
                    "coupled_diffusion_%s_operator.cpp" % element.lower(),
                )
                self.assertIn(
                    '#include "coupled_diffusion_d%d_%s_local.hpp"'
                    % (dim, family),
                    operator_source,
                )
                self.assertIn(
                    '#include "kernel_diagnostics.hpp"',
                    operator_source,
                )
                self.assertIn(
                    "struct KernelDiagnostics",
                    diagnostics_source,
                )
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
                self.assertIn(
                    "coupled_diffusion_%s_residual_affine_mesh_soa"
                    % element.lower(),
                    operator_source,
                )
                self.assertIn(
                    "coupled_diffusion_%s_jacobian_action_affine_mesh_soa"
                    % element.lower(),
                    operator_source,
                )
                self.assertIn(
                    "coupled_diffusion_%s_residual_isoparametric_mesh_soa"
                    % element.lower(),
                    operator_source,
                )
                self.assertIn(
                    "coupled_diffusion_%s_jacobian_action_isoparametric_mesh_soa"
                    % element.lower(),
                    operator_source,
                )
                self.assertIn(
                    "block_adjugate_data[%d][N_QP * VECTOR_SIZE]"
                    % (dim * dim),
                    operator_source,
                )
                self.assertIn(
                    "idx_t **const SFEM_RESTRICT elements",
                    operator_source,
                )
                self.assertIn(
                    "block_current[N_FIELDS * N_SHAPE][VECTOR_SIZE]",
                    operator_source,
                )
                action_local = local_source.split(
                    "static SFEM_INLINE void "
                    "coupled_diffusion_d%d_%s_jacobian_action_block"
                    % (dim, family),
                    1,
                )[1]
                action_signature = action_local.split(") {", 1)[0]
                self.assertNotIn(" current[", action_signature)
                self.assertNotIn(" previous[", action_signature)
                self.assertIn(" direction[", action_signature)
                self.assertIn("#pragma omp atomic update", operator_source)
                self.assertIn(
                    "coupled_diffusion_%s_residual_element_soa_diagnostics"
                    % element.lower(),
                    operator_source,
                )
                self.assertIn(
                    "coupled_diffusion_%s_jacobian_u_v_diagnostics"
                    % element.lower(),
                    operator_source,
                )
                self.assertIn(
                    "coupled_diffusion_%s_jacobian_action_element_soa_arithmetic_intensity"
                    % element.lower(),
                    operator_source,
                )
                self.assertIn(
                    "KernelDiagnostics_print_rate",
                    diagnostics_source,
                )
                self.assertIn(
                    'extern "C" void '
                    "coupled_diffusion_%s_residual_affine_mesh_soa_print_rate"
                    % element.lower(),
                    operator_source,
                )
                self.assertIn(
                    'extern "C" void '
                    "coupled_diffusion_%s_jacobian_action_isoparametric_mesh_soa_float_print_rate"
                    % element.lower(),
                    operator_source,
                )
                regenerated = generate_coupled_residual_sfem_files(
                    system,
                    prefix="coupled_diffusion",
                    element_type=element,
                )
                self.assertEqual(
                    tuple(file.source for file in files),
                    tuple(file.source for file in regenerated),
                )
                residual_cost = system.build_residual_graph().cost
                diagnostic_match = re.search(
                    r"coupled_diffusion_%s_residual_element_soa_diagnostics_data = \{"
                    r".*?\"%s\",\s*%d,\s*\d+,\s*\d+,\s*16,\s*\d+,"
                    r"\s*(\d+),\s*(\d+),\s*(\d+),"
                    % (element.lower(), element, dim),
                    operator_source,
                    re.DOTALL,
                )
                self.assertIsNotNone(diagnostic_match)
                self.assertEqual(
                    tuple(map(int, diagnostic_match.groups())),
                    (
                        residual_cost.adds,
                        residual_cost.muls,
                        residual_cost.divs,
                    ),
                )
                if element in ("QUAD4", "HEX8"):
                    self.assertIn("_tensor_evaluate", local_source)
                    self.assertIn("_tensor_integrate", local_source)
                    for form in ("residual", "jacobian_action"):
                        marker = (
                            "static SFEM_INLINE int "
                            "coupled_diffusion_%s_%s_isoparametric_mesh_soa_impl"
                            % (element.lower(), form)
                        )
                        section = operator_source.split(marker, 1)[1].split(
                            'extern "C" int coupled_diffusion_%s_%s_isoparametric_mesh_soa'
                            % (element.lower(), form),
                            1,
                        )[0]
                        self.assertIn(
                            "coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE]",
                            section,
                        )
                        self.assertIn(
                            "_tensor_evaluate<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM>",
                            section,
                        )
                        self.assertNotIn("geometry_grad_ref", section)
                        self.assertNotIn(
                            "block_coordinates[0][lane] *",
                            section,
                        )
                else:
                    self.assertNotIn("_tensor_evaluate", local_source)
                generated_sources.append(
                    os.path.join(tmpdir, files[3].path)
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

    def test_hex27_isoparametric_geometry_uses_sum_factorization(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("c++ compiler is not available")

        system, _, _ = two_field_diffusion_system(3)
        files = generate_coupled_residual_sfem_files(
            system,
            prefix="coupled_diffusion_hex27",
            element_type="HEX27",
        )
        local_source = files[2].source
        operator_source = files[3].source
        marker = (
            "static SFEM_INLINE int "
            "coupled_diffusion_hex27_hex27_residual_isoparametric_mesh_soa_impl"
        )
        section = operator_source.split(marker, 1)[1].split(
            'extern "C" int '
            "coupled_diffusion_hex27_hex27_residual_isoparametric_mesh_soa",
            1,
        )[0]
        self.assertIn(
            "_tensor_evaluate<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM>",
            section,
        )
        self.assertIn("static constexpr int N_SHAPE = 27;", section)
        self.assertNotIn("geometry_grad_ref", section)
        self.assertNotIn("tensor_index", local_source)
        self.assertIn(
            "const int s = sx + S * (sy + S * sz);",
            local_source,
        )
        self.assertIn(
            "block_coordinates[0], block_coordinates[1], block_coordinates[2], "
            "block_coordinates[24], block_coordinates[25], block_coordinates[26], "
            "block_coordinates[3], block_coordinates[4], block_coordinates[5]",
            section,
        )

        with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir:
            for generated in files:
                with open(
                    os.path.join(tmpdir, generated.path),
                    "w",
                    encoding="utf-8",
                ) as stream:
                    stream.write(generated.source)
            subprocess.run(
                [
                    compiler,
                    "-std=c++14",
                    "-O3",
                    "-fopenmp-simd",
                    "-Werror",
                    "-c",
                    os.path.join(tmpdir, files[3].path),
                    "-I",
                    tmpdir,
                    "-o",
                    os.path.join(tmpdir, "hex27.o"),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

    def test_isoparametric_mesh_residual_and_action_match_python_reference(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("c++ compiler is not available")

        for element in ("TRI3", "HEX8"):
            rule = sfem_element_quadrature_rule(element)
            system, _, _ = two_field_diffusion_system(rule.dim)
            files = generate_coupled_residual_sfem_files(
                system,
                prefix="coupled_diffusion",
                element_type=element,
                vector_size=16,
            )
            with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir:
                for generated in files:
                    with open(
                        os.path.join(tmpdir, generated.path),
                        "w",
                        encoding="utf-8",
                    ) as stream:
                        stream.write(generated.source)
                library_path = os.path.join(
                    tmpdir,
                    "libresidual.%s"
                    % ("dylib" if os.uname().sysname == "Darwin" else "so"),
                )
                command = [
                    compiler,
                    "-std=c++14",
                    "-O3",
                    "-fPIC",
                    os.path.join(
                        tmpdir,
                        "coupled_diffusion_%s_operator.cpp" % element.lower(),
                    ),
                    "-o",
                    library_path,
                ]
                command.insert(
                    -2,
                    "-dynamiclib"
                    if os.uname().sysname == "Darwin"
                    else "-shared",
                )
                subprocess.run(
                    command,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                library = ctypes.CDLL(library_path)
                n_shape = rule.n_shape
                current = (
                    tuple(1.0 + 0.12 * s for s in range(n_shape)),
                    tuple(0.7 - 0.05 * s for s in range(n_shape)),
                )
                previous = (
                    tuple(value - 0.03 for value in current[0]),
                    tuple(value + 0.02 for value in current[1]),
                )
                direction = (
                    tuple(0.04 * (s + 1) for s in range(n_shape)),
                    tuple(-0.025 * (s + 1) for s in range(n_shape)),
                )
                for coords in (
                    _reference_coordinates(element),
                    _deform_coordinates(_reference_coordinates(element)),
                ):
                    for form, trial in (
                        ("residual", None),
                        ("jacobian_action", direction),
                    ):
                        expected, accumulation = _diffusion_reference(
                            rule,
                            coords,
                            current,
                            previous,
                            trial,
                        )
                        actual = self._call_isoparametric_mesh_kernel(
                            library,
                            element,
                            form,
                            coords,
                            current,
                            previous,
                            trial,
                        )
                        actual_affine = self._call_affine_mesh_kernel(
                            library,
                            element,
                            form,
                            coords,
                            current,
                            previous,
                            trial,
                        )
                        for field in range(2):
                            for value, reference in zip(actual[field], expected[field]):
                                self.assertAlmostEqual(value, reference, places=11)
                            for value, reference in zip(
                                actual_affine[field],
                                expected[field],
                            ):
                                self.assertAlmostEqual(value, reference, places=11)
                            self.assertAlmostEqual(
                                sum(actual[field]),
                                accumulation[field],
                                places=11,
                            )

    def _call_isoparametric_mesh_kernel(
        self,
        library,
        element,
        form,
        coords,
        current,
        previous,
        direction,
    ):
        scalar = ctypes.c_double
        index = ctypes.c_long
        n_shape = len(coords)
        dim = len(coords[0])
        element_storage = [(index * 1)(shape) for shape in range(n_shape)]
        elements = (ctypes.POINTER(index) * n_shape)(
            *(ctypes.cast(values, ctypes.POINTER(index)) for values in element_storage)
        )
        point_storage = [
            (scalar * n_shape)(*(coords[s][d] for s in range(n_shape)))
            for d in range(dim)
        ]
        points = (ctypes.POINTER(scalar) * dim)(
            *(ctypes.cast(values, ctypes.POINTER(scalar)) for values in point_storage)
        )
        current_storage = [(scalar * n_shape)(*values) for values in current]
        previous_storage = [(scalar * n_shape)(*values) for values in previous]
        output_storage = [(scalar * n_shape)(*([0.0] * n_shape)) for _ in range(2)]
        function = getattr(
            library,
            "coupled_diffusion_%s_%s_isoparametric_mesh_soa"
            % (element.lower(), form),
        )
        args = [
            ctypes.c_long(1),
            ctypes.c_long(n_shape),
            elements,
            points,
            scalar(0.7),
            scalar(1.3),
            scalar(0.8),
            scalar(0.25),
        ]
        if form == "residual":
            args.extend(
                (
                    ctypes.c_long(1),
                    current_storage[0],
                    current_storage[1],
                    ctypes.c_long(1),
                    previous_storage[0],
                    previous_storage[1],
                )
            )
        if direction is not None:
            direction_storage = [
                (scalar * n_shape)(*values) for values in direction
            ]
            args.extend(
                (
                    ctypes.c_long(1),
                    direction_storage[0],
                    direction_storage[1],
                )
            )
        args.extend(
            (
                ctypes.c_long(1),
                output_storage[0],
                output_storage[1],
            )
        )
        self.assertEqual(function(*args), 0)
        return tuple(
            tuple(values[shape] for shape in range(n_shape))
            for values in output_storage
        )

    def _call_affine_mesh_kernel(
        self,
        library,
        element,
        form,
        coords,
        current,
        previous,
        direction,
    ):
        scalar = ctypes.c_double
        index = ctypes.c_long
        rule = sfem_element_quadrature_rule(element)
        n_shape = rule.n_shape
        dim = rule.dim
        grad_ref = [
            rule.reference_gradients[s * dim:(s + 1) * dim]
            for s in range(n_shape)
        ]
        jacobian = [
            [
                sum(coords[s][i] * grad_ref[s][j] for s in range(n_shape))
                for j in range(dim)
            ]
            for i in range(dim)
        ]
        adjugate, determinant = _adjugate_and_determinant(jacobian)
        geometry_storage = [
            (scalar * 1)(adjugate[i][j])
            for i in range(dim)
            for j in range(dim)
        ]
        determinant_storage = (scalar * 1)(determinant)
        element_storage = [(index * 1)(shape) for shape in range(n_shape)]
        elements = (ctypes.POINTER(index) * n_shape)(
            *(ctypes.cast(values, ctypes.POINTER(index)) for values in element_storage)
        )
        current_storage = [(scalar * n_shape)(*values) for values in current]
        previous_storage = [(scalar * n_shape)(*values) for values in previous]
        output_storage = [(scalar * n_shape)(*([0.0] * n_shape)) for _ in range(2)]
        function = getattr(
            library,
            "coupled_diffusion_%s_%s_affine_mesh_soa"
            % (element.lower(), form),
        )
        args = [
            ctypes.c_long(1),
            ctypes.c_long(n_shape),
            elements,
        ]
        args.extend(geometry_storage)
        args.extend(
            (
                determinant_storage,
                scalar(0.7),
                scalar(1.3),
                scalar(0.8),
                scalar(0.25),
            )
        )
        if form == "residual":
            args.extend(
                (
                    ctypes.c_long(1),
                    current_storage[0],
                    current_storage[1],
                    ctypes.c_long(1),
                    previous_storage[0],
                    previous_storage[1],
                )
            )
        if direction is not None:
            direction_storage = [
                (scalar * n_shape)(*values) for values in direction
            ]
            args.extend(
                (
                    ctypes.c_long(1),
                    direction_storage[0],
                    direction_storage[1],
                )
            )
        args.extend(
            (
                ctypes.c_long(1),
                output_storage[0],
                output_storage[1],
            )
        )
        self.assertEqual(function(*args), 0)
        return tuple(
            tuple(values[shape] for shape in range(n_shape))
            for values in output_storage
        )

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
