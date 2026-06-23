import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import sympy as sp

sys.path.insert(0, os.path.dirname(__file__))

from symbolic import (
    DeformationGradient,
    DimensionSpecialization,
    ExpressionRole,
    KernelExpressions,
    KernelTemplateParameter,
    LayoutKind,
    ReferenceShapeGradients,
    ScopeKind,
    data_layout,
    displacement_gradient_from_reference,
    execution_scope,
    generate_cpp_kernel,
    generate_openmp_cpp_kernel,
    generate_sfem_soa_cpp_files_for_element,
    hessian_action_from_energy,
    matrix_inner,
    residual_from_energy,
    sfem_element_quadrature_rule,
    sfem_supported_element_types,
    sfem_soa_element_specialization,
    sfem_soa_element_specializations,
    sfem_soa_kernel_form,
    sfem_soa_weak_form,
    vector_symbols,
)


def neohookean_ogden_energy(F, mu, lmbda):
    dim = F.shape[0]
    J = F.det()
    I1 = matrix_inner(F, F)
    logJ = sp.log(J)
    return mu * sp.Rational(1, 2) * (I1 - dim) - mu * logJ + (
        lmbda * sp.Rational(1, 2) * logJ * logJ
    )


def compiler_vectorization_flags(compiler):
    version = subprocess.run(
        [compiler, "--version"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.lower()
    if "clang" in version:
        return ["-O3", "-Rpass=loop-vectorize"], "clang"
    if "gcc" in version or "g++" in version:
        return ["-O3", "-fopt-info-vec-optimized"], "gcc"
    return None, "unknown"


def assert_generated_lane_loops_vectorized(test_case, compiler, source_path, object_path):
    flags, compiler_kind = compiler_vectorization_flags(compiler)
    if flags is None:
        test_case.skipTest("compiler does not expose a supported vectorization report")

    completed = subprocess.run(
        [
            compiler,
            "-std=c++11",
            *flags,
            "-c",
            source_path,
            "-o",
            object_path,
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    report = completed.stdout + completed.stderr
    if compiler_kind == "clang":
        pattern = r"generated_neohookean_ogden_local\.hpp:\d+:\d+: remark: vectorized loop"
    else:
        pattern = r"generated_neohookean_ogden_local\.hpp:.*loop vectorized"

    matches = re.findall(pattern, report)
    test_case.assertGreaterEqual(
        len(matches),
        3,
        "expected objective, gradient, and apply local lane loops to be vectorized; "
        "compiler report was:\n%s" % report,
    )


class NeoHookeanOgdenFrameworkTest(unittest.TestCase):
    def test_sfem_element_specialization_api_covers_relevant_elements(self):
        required = ("TET4", "HEX8", "HEX27", "TET10", "QUAD4", "TRI3", "TRI6")
        self.assertEqual(set(sfem_supported_element_types()), set(required))

        specializations = sfem_soa_element_specializations(required, vector_size=16)
        by_type = {specialization.element_type: specialization for specialization in specializations}

        expected_shape = {
            "TRI3": (2, 3, 1),
            "TRI6": (2, 6, 3),
            "QUAD4": (2, 4, 4),
            "TET4": (3, 4, 1),
            "TET10": (3, 10, 4),
            "HEX8": (3, 8, 8),
            "HEX27": (3, 27, 27),
        }
        for element_type, (dim, n_shape, n_qp) in expected_shape.items():
            specialization = by_type[element_type]
            self.assertEqual(specialization.dim, dim)
            self.assertEqual(specialization.n_shape, n_shape)
            self.assertEqual(specialization.n_qp, n_qp)
            self.assertEqual(specialization.vector_size, 16)
            self.assertEqual(
                len(specialization.quadrature_rule.reference_gradients),
                dim * n_shape * n_qp,
            )

    def test_tensor_product_elements_expose_1d_quadrature_and_shapes(self):
        quad4 = sfem_element_quadrature_rule("QUAD4")
        self.assertTrue(quad4.is_tensor_product)
        self.assertEqual(quad4.tensor_product_dim, 2)
        self.assertEqual(quad4.tensor_product_n_qp_1d, 2)
        self.assertEqual(quad4.tensor_product_n_shape_1d, 2)
        self.assertEqual(len(quad4.tensor_product_weights_1d), 2)
        self.assertEqual(len(quad4.tensor_product_shape_values_1d), 4)
        self.assertEqual(
            quad4.tensor_product_shape_gradients_1d,
            (-1.0, 1.0, -1.0, 1.0),
        )
        self.assertEqual(len(quad4.weights), 4)

        hex8 = sfem_element_quadrature_rule("HEX8")
        self.assertTrue(hex8.is_tensor_product)
        self.assertEqual(hex8.tensor_product_dim, 3)
        self.assertEqual(hex8.tensor_product_n_qp_1d, 2)
        self.assertEqual(hex8.tensor_product_n_shape_1d, 2)
        self.assertEqual(len(hex8.tensor_product_weights_1d), 2)
        self.assertEqual(len(hex8.tensor_product_shape_values_1d), 4)
        self.assertEqual(
            hex8.tensor_product_shape_gradients_1d,
            (-1.0, 1.0, -1.0, 1.0),
        )
        self.assertEqual(len(hex8.weights), 8)

        hex27 = sfem_element_quadrature_rule("HEX27")
        self.assertTrue(hex27.is_tensor_product)
        self.assertEqual(hex27.tensor_product_dim, 3)
        self.assertEqual(hex27.tensor_product_n_qp_1d, 3)
        self.assertEqual(hex27.tensor_product_n_shape_1d, 3)
        self.assertEqual(len(hex27.tensor_product_weights_1d), 3)
        self.assertEqual(len(hex27.tensor_product_shape_values_1d), 9)
        self.assertEqual(len(hex27.tensor_product_shape_gradients_1d), 9)
        self.assertEqual(hex27.tensor_product_shape_values_1d[3:6], (0.0, 1.0, 0.0))
        self.assertEqual(hex27.tensor_product_shape_gradients_1d[3:6], (-1.0, 0.0, 1.0))
        self.assertEqual(len(hex27.weights), 27)

    def test_generated_quad4_soa_kernel_uses_tensor_product_reference_data(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("c++ compiler is not available")

        specialization = sfem_soa_element_specialization("QUAD4", vector_size=8)
        dim = specialization.dim
        n_nodes = specialization.n_shape
        grad_ref = ReferenceShapeGradients("grad_ref", n_nodes=n_nodes, dim=dim)
        displacement = vector_symbols("u", n_nodes * dim)
        mu, lmbda, qw = sp.symbols("mu lmbda qw")

        energy = qw * sum(
            displacement[node * dim + d] * grad_ref.gradient(node, d)
            for node in range(n_nodes)
            for d in range(dim)
        )
        graph = (
            KernelExpressions()
            .add(ExpressionRole.OPERATOR_EVALUATION, energy)
            .build_graph(
                data_symbols=tuple(displacement) + (qw,),
                symbolic_objects=(grad_ref,),
                temporary_prefix="quad4_tp_tmp",
            )
        )

        generated_files = generate_sfem_soa_cpp_files_for_element(
            (
                sfem_soa_kernel_form(
                    "objective",
                    graph,
                ),
            ),
            prefix="generated_quad4_tensor_product",
            specialization=specialization,
        )

        source_by_path = {generated.path: generated.source for generated in generated_files}
        operator_source = source_by_path["generated_quad4_tensor_product_operator.cpp"]
        local_source = source_by_path["generated_quad4_tensor_product_local.hpp"]

        self.assertIn("generated_quad4_tensor_product_quad4_shape_1d", operator_source)
        self.assertIn("generated_quad4_tensor_product_quad4_grad_1d", operator_source)
        self.assertIn("generated_quad4_tensor_product_quad4_q_weight_1d", operator_source)
        self.assertNotIn("generated_quad4_tensor_product_quad4_grad_ref", operator_source)
        self.assertNotIn("generated_quad4_tensor_product_quad4_q_weight[", operator_source)
        self.assertNotIn("GRAD_REF_NCOMPONENTS", operator_source)
        self.assertNotIn("GRAD_REF_NCOMPONENTS", local_source)
        self.assertIn("const scalar_t *const SFEM_RESTRICT shape_1d", operator_source)
        self.assertIn("const scalar_t *const SFEM_RESTRICT grad_1d", operator_source)
        self.assertIn("const scalar_t *const SFEM_RESTRICT q_weight_1d", operator_source)
        self.assertIn("const int qx = q % N_QP_1D;", operator_source)
        self.assertIn("const int qy = q / N_QP_1D;", operator_source)
        self.assertIn(
            "const scalar_t tensor_q_weight = q_weight_1d[qx] * q_weight_1d[qy];",
            operator_source,
        )
        self.assertIn(
            "return generated_quad4_tensor_product_quad4_objective_soa_impl<4, 4, 8>",
            operator_source,
        )
        self.assertIn("const scalar_t *const SFEM_RESTRICT shape_1d", local_source)
        self.assertIn("const scalar_t *const SFEM_RESTRICT grad_1d", local_source)
        self.assertNotIn("grad_ref_data", local_source)
        self.assertIn("static constexpr int N_QP_1D = 2;", local_source)
        self.assertIn(
            "grad_ref[0] = grad_1d[qx * N_SHAPE_1D + 0] * shape_1d[qy * N_SHAPE_1D + 0];",
            local_source,
        )
        self.assertIn(
            "grad_ref[7] = shape_1d[qx * N_SHAPE_1D + 0] * grad_1d[qy * N_SHAPE_1D + 1];",
            local_source,
        )

        with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir:
            for generated in generated_files:
                with open(os.path.join(tmpdir, generated.path), "w", encoding="utf-8") as output:
                    output.write(generated.source)

            subprocess.run(
                [
                    compiler,
                    "-std=c++11",
                    "-O3",
                    "-c",
                    os.path.join(tmpdir, "generated_quad4_tensor_product_operator.cpp"),
                    "-o",
                    os.path.join(tmpdir, "generated_quad4_tensor_product_operator.o"),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

    def test_generated_hex8_soa_kernel_uses_tensor_product_reference_data(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("c++ compiler is not available")

        specialization = sfem_soa_element_specialization("HEX8", vector_size=8)
        dim = specialization.dim
        n_nodes = specialization.n_shape
        grad_ref = ReferenceShapeGradients("grad_ref", n_nodes=n_nodes, dim=dim)
        displacement = vector_symbols("u", n_nodes * dim)
        qw = sp.symbols("qw")

        energy = qw * sum(
            displacement[node * dim + d] * grad_ref.gradient(node, d)
            for node in range(n_nodes)
            for d in range(dim)
        )
        graph = (
            KernelExpressions()
            .add(ExpressionRole.OPERATOR_EVALUATION, energy)
            .build_graph(
                data_symbols=tuple(displacement) + (qw,),
                symbolic_objects=(grad_ref,),
                temporary_prefix="hex8_tp_tmp",
            )
        )

        generated_files = generate_sfem_soa_cpp_files_for_element(
            (
                sfem_soa_kernel_form(
                    "objective",
                    graph,
                ),
            ),
            prefix="generated_hex8_tensor_product",
            specialization=specialization,
        )

        source_by_path = {generated.path: generated.source for generated in generated_files}
        operator_source = source_by_path["generated_hex8_tensor_product_operator.cpp"]
        local_source = source_by_path["generated_hex8_tensor_product_local.hpp"]

        self.assertIn("generated_hex8_tensor_product_hex8_shape_1d", operator_source)
        self.assertIn("generated_hex8_tensor_product_hex8_grad_1d", operator_source)
        self.assertIn("generated_hex8_tensor_product_hex8_q_weight_1d", operator_source)
        self.assertNotIn("generated_hex8_tensor_product_hex8_grad_ref", operator_source)
        self.assertNotIn("GRAD_REF_NCOMPONENTS", operator_source)
        self.assertNotIn("GRAD_REF_NCOMPONENTS", local_source)
        self.assertIn("const int qx = q % N_QP_1D;", operator_source)
        self.assertIn("const int qy = (q / N_QP_1D) % N_QP_1D;", operator_source)
        self.assertIn("const int qz = q / (N_QP_1D * N_QP_1D);", operator_source)
        self.assertIn(
            "const scalar_t tensor_q_weight = q_weight_1d[qx] * q_weight_1d[qy] * q_weight_1d[qz];",
            operator_source,
        )
        self.assertIn(
            "return generated_hex8_tensor_product_hex8_objective_soa_impl<8, 8, 8>",
            operator_source,
        )
        self.assertNotIn("grad_ref_data", local_source)
        self.assertIn(
            "grad_ref[23] = shape_1d[qx * N_SHAPE_1D + 0] * shape_1d[qy * N_SHAPE_1D + 1] * grad_1d[qz * N_SHAPE_1D + 1];",
            local_source,
        )

        with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir:
            for generated in generated_files:
                with open(os.path.join(tmpdir, generated.path), "w", encoding="utf-8") as output:
                    output.write(generated.source)

            subprocess.run(
                [
                    compiler,
                    "-std=c++11",
                    "-O3",
                    "-c",
                    os.path.join(tmpdir, "generated_hex8_tensor_product_operator.cpp"),
                    "-o",
                    os.path.join(tmpdir, "generated_hex8_tensor_product_operator.o"),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

    def test_generated_weak_form_kernel_uses_shape_loops_and_loperand(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("c++ compiler is not available")

        specialization = sfem_soa_element_specialization("TRI3", vector_size=8)
        dim = specialization.dim
        F = sp.Matrix(
            dim,
            dim,
            tuple(sp.symbols("F[%d]" % i) for i in range(dim * dim)),
        )
        weak_form = sfem_soa_weak_form(neohookean_ogden_energy(F, *sp.symbols("mu lmbda")), F)

        generated_files = generate_sfem_soa_cpp_files_for_element(
            (
                sfem_soa_kernel_form(
                    "objective",
                    weak_form=weak_form,
                    output_mode="accumulate",
                ),
                sfem_soa_kernel_form(
                    "gradient",
                    weak_form=weak_form,
                    output_mode="accumulate",
                ),
                sfem_soa_kernel_form(
                    "apply",
                    weak_form=weak_form,
                    has_direction=True,
                    output_mode="accumulate",
                ),
            ),
            prefix="generated_weak_neohookean",
            specialization=specialization,
        )

        source_by_path = {generated.path: generated.source for generated in generated_files}
        local_source = source_by_path["generated_weak_neohookean_local.hpp"]
        operator_source = source_by_path["generated_weak_neohookean_operator.cpp"]

        self.assertIn("for (int shape = 0; shape < N_SHAPE; ++shape)", local_source)
        self.assertNotIn("scalar_t grad_ref", local_source)
        self.assertNotIn("grad_ref[shape", local_source)
        self.assertNotIn("grad_ref_data", local_source)
        self.assertIn("generated_weak_neohookean_tri3_grad_ref_x", operator_source)
        self.assertIn("generated_weak_neohookean_tri3_grad_ref_y", operator_source)
        self.assertIn(
            "grad_u_ref[0] += u[shape * 2 + 0] * grad_ref_x[q * N_SHAPE + shape];",
            local_source,
        )
        self.assertIn(
            "grad_h_ref[0] += du[shape * 2 + 0] * grad_ref_x[q * N_SHAPE + shape];",
            local_source,
        )
        self.assertIn("scalar_t trial_grad[4];", local_source)
        self.assertIn("scalar_t material[4];", local_source)
        self.assertIn("scalar_t loperand[4];", local_source)
        self.assertIn("scalar_t element_vector[N_SHAPE * 2];", local_source)
        self.assertIn(
            "element_vector[shape * 2 + 0] = loperand[0] * grad_ref_x[q * N_SHAPE + shape]",
            local_source,
        )
        self.assertIn("outx0[lane] += element_vector[0];", local_source)
        self.assertIn("generated_weak_neohookean_tri3_apply_soa_impl<1, 3, 8>", operator_source)

        with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir:
            for generated in generated_files:
                with open(os.path.join(tmpdir, generated.path), "w", encoding="utf-8") as output:
                    output.write(generated.source)

            subprocess.run(
                [
                    compiler,
                    "-std=c++11",
                    "-O3",
                    "-c",
                    os.path.join(tmpdir, "generated_weak_neohookean_operator.cpp"),
                    "-o",
                    os.path.join(tmpdir, "generated_weak_neohookean_operator.o"),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

    def test_generated_tensor_product_weak_form_uses_1d_gradients_directly(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("c++ compiler is not available")

        specialization = sfem_soa_element_specialization("QUAD4", vector_size=8)
        dim = specialization.dim
        F = sp.Matrix(
            dim,
            dim,
            tuple(sp.symbols("F[%d]" % i) for i in range(dim * dim)),
        )
        weak_form = sfem_soa_weak_form(neohookean_ogden_energy(F, *sp.symbols("mu lmbda")), F)

        generated_files = generate_sfem_soa_cpp_files_for_element(
            (
                sfem_soa_kernel_form(
                    "gradient",
                    weak_form=weak_form,
                    output_mode="accumulate",
                ),
                sfem_soa_kernel_form(
                    "apply",
                    weak_form=weak_form,
                    has_direction=True,
                    output_mode="accumulate",
                ),
            ),
            prefix="generated_quad4_weak_neohookean",
            specialization=specialization,
        )

        source_by_path = {generated.path: generated.source for generated in generated_files}
        local_source = source_by_path["generated_quad4_weak_neohookean_local.hpp"]

        self.assertIn("const scalar_t *const SFEM_RESTRICT shape_1d", local_source)
        self.assertIn("const scalar_t *const SFEM_RESTRICT grad_1d", local_source)
        self.assertNotIn("scalar_t grad_ref", local_source)
        self.assertNotIn("grad_ref[", local_source)
        self.assertNotIn("grad_ref_data", local_source)
        self.assertIn("const int sx = ((shape + 1) >> 1) & 1;", local_source)
        self.assertIn("const int sy = shape >> 1;", local_source)
        self.assertIn(
            "grad_u_ref[0] += u[shape * 2 + 0] * grad_1d[qx * N_SHAPE_1D + sx] * shape_1d[qy * N_SHAPE_1D + sy];",
            local_source,
        )
        self.assertIn(
            "element_vector[shape * 2 + 0] = loperand[0] * grad_1d[qx * N_SHAPE_1D + sx] * shape_1d[qy * N_SHAPE_1D + sy]",
            local_source,
        )

        with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir:
            for generated in generated_files:
                with open(os.path.join(tmpdir, generated.path), "w", encoding="utf-8") as output:
                    output.write(generated.source)

            subprocess.run(
                [
                    compiler,
                    "-std=c++11",
                    "-O3",
                    "-c",
                    os.path.join(tmpdir, "generated_quad4_weak_neohookean_operator.cpp"),
                    "-o",
                    os.path.join(tmpdir, "generated_quad4_weak_neohookean_operator.o"),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

    def test_generated_hex27_weak_form_uses_q2_tensor_product_api(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("c++ compiler is not available")

        specialization = sfem_soa_element_specialization("HEX27", vector_size=8)
        dim = specialization.dim
        F = sp.Matrix(
            dim,
            dim,
            tuple(sp.symbols("F[%d]" % i) for i in range(dim * dim)),
        )
        weak_form = sfem_soa_weak_form(neohookean_ogden_energy(F, *sp.symbols("mu lmbda")), F)

        generated_files = generate_sfem_soa_cpp_files_for_element(
            (
                sfem_soa_kernel_form(
                    "gradient",
                    weak_form=weak_form,
                    output_mode="accumulate",
                ),
                sfem_soa_kernel_form(
                    "apply",
                    weak_form=weak_form,
                    has_direction=True,
                    output_mode="accumulate",
                ),
            ),
            prefix="generated_hex27_weak_neohookean",
            specialization=specialization,
        )

        source_by_path = {generated.path: generated.source for generated in generated_files}
        local_source = source_by_path["generated_hex27_weak_neohookean_local.hpp"]
        operator_source = source_by_path["generated_hex27_weak_neohookean_operator.cpp"]

        self.assertIn("generated_hex27_weak_neohookean_hex27_shape_1d", operator_source)
        self.assertIn("generated_hex27_weak_neohookean_hex27_grad_1d", operator_source)
        self.assertIn("generated_hex27_weak_neohookean_hex27_q_weight_1d", operator_source)
        self.assertIn("generated_hex27_weak_neohookean_hex27_apply_soa_impl<27, 27, 8>", operator_source)
        self.assertIn("static constexpr int N_QP_1D = 3;", local_source)
        self.assertIn("static constexpr int N_SHAPE_1D = 3;", local_source)
        self.assertIn("const int sx = shape % N_SHAPE_1D;", local_source)
        self.assertIn("const int sy = (shape / N_SHAPE_1D) % N_SHAPE_1D;", local_source)
        self.assertIn("const int sz = shape / (N_SHAPE_1D * N_SHAPE_1D);", local_source)
        self.assertNotIn("scalar_t grad_ref", local_source)
        self.assertNotIn("grad_ref[", local_source)
        self.assertNotIn("grad_ref_data", local_source)
        self.assertIn(
            "grad_u_ref[0] += u[shape * 3 + 0] * grad_1d[qx * N_SHAPE_1D + sx] * shape_1d[qy * N_SHAPE_1D + sy] * shape_1d[qz * N_SHAPE_1D + sz];",
            local_source,
        )
        self.assertIn(
            "element_vector[shape * 3 + 0] = loperand[0] * grad_1d[qx * N_SHAPE_1D + sx] * shape_1d[qy * N_SHAPE_1D + sy] * shape_1d[qz * N_SHAPE_1D + sz]",
            local_source,
        )

        with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir:
            for generated in generated_files:
                with open(os.path.join(tmpdir, generated.path), "w", encoding="utf-8") as output:
                    output.write(generated.source)

            subprocess.run(
                [
                    compiler,
                    "-std=c++11",
                    "-O3",
                    "-c",
                    os.path.join(tmpdir, "generated_hex27_weak_neohookean_operator.cpp"),
                    "-o",
                    os.path.join(tmpdir, "generated_hex27_weak_neohookean_operator.o"),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

    def test_passes_neohookean_ogden_strain_energy_to_framework(self):
        mu, lmbda = sp.symbols("mu lambda")
        q = sp.symbols("q")
        F_obj = DeformationGradient("F", 3)
        F = F_obj.as_matrix()
        variables = F_obj.entries
        directions = vector_symbols("dF", len(variables))

        energy = neohookean_ogden_energy(F, mu, lmbda)
        residual = residual_from_energy(energy, variables)
        hessian_action = hessian_action_from_energy(energy, variables, directions)

        graph = (
            KernelExpressions()
            .energy(energy)
            .residual(residual)
            .hessian_action(hessian_action)
            .build_graph(
                symbolic_objects=(F_obj,),
                scopes=(execution_scope(ScopeKind.QUADRATURE, (q,)),),
                temporary_prefix="nh_tmp",
            )
        )

        roles = [expr.role for expr in graph.outputs]
        self.assertEqual(roles.count(ExpressionRole.ENERGY), 1)
        self.assertEqual(roles.count(ExpressionRole.RESIDUAL), 9)
        self.assertEqual(roles.count(ExpressionRole.HESSIAN_ACTION), 9)
        self.assertGreater(graph.cost.flops, 0)
        self.assertGreaterEqual(len(graph.evaluation_plan.statements), 19)
        self.assertGreater(graph.cost.estimated_registers, 0)

    def test_neohookean_ogden_first_derivative_on_diagonal_F(self):
        mu, lmbda = sp.symbols("mu lambda")
        a, b, c = sp.symbols("a b c", positive=True)
        F_obj = DeformationGradient("F", 3)
        F = F_obj.as_matrix()
        energy = neohookean_ogden_energy(F, mu, lmbda)
        residual = residual_from_energy(energy, F_obj.entries)

        substitutions = {
            F[0, 0]: a,
            F[0, 1]: 0,
            F[0, 2]: 0,
            F[1, 0]: 0,
            F[1, 1]: b,
            F[1, 2]: 0,
            F[2, 0]: 0,
            F[2, 1]: 0,
            F[2, 2]: c,
        }
        actual = sp.simplify(residual[0].subs(substitutions))
        expected = mu * a + (lmbda * sp.log(a * b * c) - mu) / a

        self.assertEqual(sp.simplify(actual - expected), 0)

    def test_neohookean_ogden_from_reference_gradients_and_displacement_coeffs(self):
        mu, lmbda, qw = sp.symbols("mu lambda qw")
        q = sp.symbols("q")
        dim = 2
        n_nodes = 2

        grad_ref = ReferenceShapeGradients(
            "grad_ref",
            n_nodes=n_nodes,
            dim=dim,
            layout=data_layout(LayoutKind.AOS),
        )
        displacement = vector_symbols("u", n_nodes * dim)
        trial_direction = vector_symbols("du", n_nodes * dim)

        reference_gradients = []
        for node in range(n_nodes):
            for row in range(dim):
                reference_gradients.append(grad_ref.tensor_gradient(node, row))

        disp_grad = displacement_gradient_from_reference(
            displacement,
            reference_gradients,
            sp.eye(dim),
        )
        F = sp.eye(dim) + disp_grad
        energy = neohookean_ogden_energy(F, mu, lmbda) * qw
        residual = residual_from_energy(energy, displacement)
        hessian_action = hessian_action_from_energy(
            energy,
            displacement,
            trial_direction,
        )

        graph = (
            KernelExpressions()
            .energy(energy)
            .residual(residual)
            .hessian_action(hessian_action)
            .build_graph(
                data_symbols=tuple(displacement) + tuple(trial_direction) + (mu, lmbda, qw),
                symbolic_objects=(grad_ref,),
                scopes=(execution_scope(ScopeKind.QUADRATURE, (qw, q)),),
                temporary_prefix="nh_ref_tmp",
            )
        )

        roles = [expr.role for expr in graph.outputs]
        self.assertEqual(roles.count(ExpressionRole.ENERGY), 1)
        self.assertEqual(roles.count(ExpressionRole.RESIDUAL), n_nodes * dim)
        self.assertEqual(roles.count(ExpressionRole.HESSIAN_ACTION), n_nodes * dim)

        grad_symbol = grad_ref.gradient(node=1, component=0)
        grad_node = graph.graph.nodes[grad_symbol]
        self.assertEqual(grad_node["layout_kind"], LayoutKind.AOS)
        self.assertEqual(grad_node["node"], 1)
        self.assertEqual(grad_node["dim_component"], 0)
        self.assertEqual(grad_node["layout_offset"], 4 * grad_node["layout_index"] + 2)

        self.assertIn(ScopeKind.QUADRATURE, graph.evaluation_plan.outputs[0].scopes)
        self.assertEqual(graph.evaluation_plan.outputs[0].hoist_scope, ScopeKind.QUADRATURE)
        self.assertEqual(graph.specialization, DimensionSpecialization(dim, "grad_ref"))
        self.assertIn(
            KernelTemplateParameter("grad_ref_n_nodes", n_nodes, "grad_ref"),
            graph.template_parameters,
        )
        self.assertIn(
            KernelTemplateParameter("grad_ref_dim", dim, "grad_ref"),
            graph.template_parameters,
        )

    def test_compiles_generated_neohookean_ogden_cpp_kernel(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("c++ compiler is not available")

        mu, lmbda, qw = sp.symbols("mu lmbda qw")
        quadrature_rule = sfem_element_quadrature_rule("TRI3")
        dim = quadrature_rule.dim
        n_nodes = quadrature_rule.n_shape
        grad_ref = ReferenceShapeGradients("grad_ref", n_nodes=n_nodes, dim=dim)
        displacement = vector_symbols("u", n_nodes * dim)

        reference_gradients = []
        for node in range(n_nodes):
            for row in range(dim):
                reference_gradients.append(grad_ref.tensor_gradient(node, row))

        disp_grad = displacement_gradient_from_reference(
            displacement,
            reference_gradients,
            sp.eye(dim),
        )
        energy = neohookean_ogden_energy(sp.eye(dim) + disp_grad, mu, lmbda) * qw
        residual = residual_from_energy(energy, displacement)
        graph = (
            KernelExpressions()
            .energy(energy)
            .residual(residual)
            .build_graph(
                data_symbols=tuple(displacement) + (mu, lmbda, qw),
                symbolic_objects=(grad_ref,),
                temporary_prefix="nh_compile_tmp",
            )
        )
        generated = generate_cpp_kernel(
            graph,
            function_name="generic_expression_kernel",
        )

        with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir:
            source_path = os.path.join(tmpdir, "generic_expression_kernel.cpp")
            object_path = os.path.join(tmpdir, "generic_expression_kernel.o")
            with open(source_path, "w", encoding="utf-8") as source_file:
                source_file.write(generated.source)

            subprocess.run(
                [
                    compiler,
                    "-std=c++11",
                    "-O2",
                    "-c",
                    source_path,
                    "-o",
                    object_path,
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

        self.assertIn('extern "C" void generic_expression_kernel', generated.source)

    def test_compiles_generated_neohookean_ogden_openmp_kernel_with_wrapper(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("c++ compiler is not available")

        mu, lmbda, qw = sp.symbols("mu lmbda qw")
        quadrature_rule = sfem_element_quadrature_rule("TRI3")
        dim = quadrature_rule.dim
        n_nodes = quadrature_rule.n_shape
        grad_ref = ReferenceShapeGradients("grad_ref", n_nodes=n_nodes, dim=dim)
        displacement = vector_symbols("u", n_nodes * dim)

        reference_gradients = []
        for node in range(n_nodes):
            for row in range(dim):
                reference_gradients.append(grad_ref.tensor_gradient(node, row))

        disp_grad = displacement_gradient_from_reference(
            displacement,
            reference_gradients,
            sp.eye(dim),
        )
        energy = neohookean_ogden_energy(sp.eye(dim) + disp_grad, mu, lmbda) * qw
        residual = residual_from_energy(energy, displacement)
        graph = (
            KernelExpressions()
            .energy(energy)
            .residual(residual)
            .build_graph(
                data_symbols=tuple(displacement) + (mu, lmbda, qw),
                symbolic_objects=(grad_ref,),
                temporary_prefix="nh_omp_tmp",
            )
        )
        generated = generate_openmp_cpp_kernel(
            graph,
            function_name="generic_expression_openmp_kernel",
            wrapper_name="GenericExpressionOpenMPOperator",
        )

        with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir:
            source_path = os.path.join(tmpdir, "generic_expression_openmp_kernel.cpp")
            object_path = os.path.join(tmpdir, "generic_expression_openmp_kernel.o")
            with open(source_path, "w", encoding="utf-8") as source_file:
                source_file.write(generated.source)

            subprocess.run(
                [
                    compiler,
                    "-std=c++11",
                    "-O2",
                    "-c",
                    source_path,
                    "-o",
                    object_path,
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

        self.assertIn("#pragma omp parallel for", generated.source)
        self.assertIn("struct GenericExpressionOpenMPOperator", generated.source)
        self.assertIn('extern "C" void generic_expression_openmp_kernel', generated.source)

    def test_compiles_generated_neohookean_ogden_sfem_soa_kernel(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("c++ compiler is not available")

        mu, lmbda, qw = sp.symbols("mu lmbda qw")
        specialization = sfem_soa_element_specialization("TRI3", vector_size=8)
        quadrature_rule = specialization.quadrature_rule
        dim = quadrature_rule.dim
        n_nodes = quadrature_rule.n_shape
        grad_ref = ReferenceShapeGradients("grad_ref", n_nodes=n_nodes, dim=dim)
        jacobian_adjugate = sp.Matrix(
            dim,
            dim,
            tuple(
                sp.symbols("jacobian_adjugate[%d]" % i)
                for i in range(dim * dim)
            ),
        )
        jacobian_determinant = sp.symbols("jacobian_determinant[0]")
        displacement = vector_symbols("u", n_nodes * dim)
        trial_direction = vector_symbols("du", n_nodes * dim)

        reference_gradients = []
        for node in range(n_nodes):
            for row in range(dim):
                reference_gradients.append(grad_ref.tensor_gradient(node, row))

        disp_grad = displacement_gradient_from_reference(
            displacement,
            reference_gradients,
            jacobian_adjugate / jacobian_determinant,
        )
        energy = (
            neohookean_ogden_energy(sp.eye(dim) + disp_grad, mu, lmbda)
            * qw
            * jacobian_determinant
        )
        residual = residual_from_energy(energy, displacement)
        hessian_action = hessian_action_from_energy(
            energy,
            displacement,
            trial_direction,
        )

        geometry_data = tuple(jacobian_adjugate) + (jacobian_determinant,)
        common_data = tuple(displacement) + geometry_data + (mu, lmbda, qw)
        apply_data = tuple(displacement) + tuple(trial_direction) + geometry_data + (mu, lmbda, qw)

        def expression_graph(expression, data_symbols, prefix):
            return (
                KernelExpressions()
                .add(ExpressionRole.OPERATOR_EVALUATION, expression)
                .build_graph(
                    data_symbols=data_symbols,
                    symbolic_objects=(grad_ref,),
                    temporary_prefix=prefix,
                )
            )

        generated_files = generate_sfem_soa_cpp_files_for_element(
            (
                sfem_soa_kernel_form(
                    "objective",
                    expression_graph(energy, common_data, "nh_obj_tmp"),
                ),
                sfem_soa_kernel_form(
                    "gradient",
                    expression_graph(residual, common_data, "nh_grad_tmp"),
                ),
                sfem_soa_kernel_form(
                    "apply",
                    expression_graph(hessian_action, apply_data, "nh_apply_tmp"),
                    has_direction=True,
                ),
            ),
            prefix="generated_neohookean_ogden",
            specialization=specialization,
        )

        source_by_path = {generated.path: generated.source for generated in generated_files}
        operator_source = source_by_path["generated_neohookean_ogden_operator.cpp"]
        local_source = source_by_path["generated_neohookean_ogden_local.hpp"]

        self.assertIn("template <int N_QP, int N_SHAPE, int VECTOR_SIZE>", operator_source)
        self.assertIn("generated_neohookean_ogden_tri3_grad_ref", operator_source)
        self.assertIn("generated_neohookean_ogden_tri3_q_weight", operator_source)
        self.assertIn("#ifndef SFEM_KERNEL_DIAGNOSTICS_DEFINED", operator_source)
        self.assertIn("struct SfemKernelDiagnostics", operator_source)
        self.assertIn("add_instructions_per_qp_lane", operator_source)
        self.assertIn("mul_instructions_per_qp_lane", operator_source)
        self.assertIn("div_instructions_per_qp_lane", operator_source)
        self.assertIn("sqrt_instructions_per_qp_lane", operator_source)
        self.assertIn("pow_instructions_per_qp_lane", operator_source)
        self.assertIn("load_instructions_per_qp_lane", operator_source)
        self.assertIn("store_instructions_per_qp_lane", operator_source)
        self.assertIn("double add_cpi", operator_source)
        self.assertIn("double div_cpi", operator_source)
        self.assertIn("int vector_size", operator_source)
        self.assertIn("geometry_streams", operator_source)
        self.assertIn("reference_scalars", operator_source)
        self.assertIn("output_reads_per_element", operator_source)
        self.assertIn(
            'extern "C" const SfemKernelDiagnostics *generated_neohookean_ogden_tri3_apply_soa_diagnostics',
            operator_source,
        )
        self.assertIn(
            'extern "C" double generated_neohookean_ogden_tri3_apply_soa_arithmetic_intensity',
            operator_source,
        )
        self.assertIn("SfemKernelDiagnostics_total_bytes", operator_source)
        self.assertIn("static SFEM_INLINE int generated_neohookean_ogden_tri3_apply_soa_impl", operator_source)
        self.assertIn('extern "C" int generated_neohookean_ogden_tri3_apply_soa', operator_source)
        self.assertIn(
            "return generated_neohookean_ogden_tri3_apply_soa_impl<1, 3, 8>",
            operator_source,
        )
        self.assertIn("generated_neohookean_ogden_tri3_grad_ref", operator_source)
        self.assertIn("generated_neohookean_ogden_tri3_q_weight", operator_source)
        self.assertIn("static_assert(N_QP == 1", operator_source)
        self.assertIn("static_assert(N_SHAPE == 3", operator_source)
        self.assertIn("for (int q = 0; q < N_QP; ++q)", operator_source)
        self.assertIn("block_ux0[VECTOR_SIZE]", operator_source)
        self.assertIn("block_jacobian_adjugate0[VECTOR_SIZE]", operator_source)
        self.assertIn("block_jacobian_determinant0[VECTOR_SIZE]", operator_source)
        self.assertIn(
            "block_jacobian_adjugate0[lane] = jacobian_adjugate0[(ptrdiff_t)q * nelements + evbegin + lane];",
            operator_source,
        )
        apply_wrapper_source = operator_source.split(
            'extern "C" int generated_neohookean_ogden_tri3_apply_soa',
            1,
        )[1]
        apply_wrapper_source = apply_wrapper_source.split(
            "return generated_neohookean_ogden_tri3_apply_soa_impl",
            1,
        )[0]
        self.assertNotIn("grad_ref", apply_wrapper_source)
        self.assertNotIn("qw", apply_wrapper_source)
        self.assertIn("const real_t *const SFEM_RESTRICT ux0", operator_source)
        self.assertIn("#pragma omp simd", local_source)
        self.assertIn("template <int N_QP, int N_SHAPE, int VECTOR_SIZE>", local_source)
        self.assertIn("generated_neohookean_ogden_apply_block", local_source)
        self.assertIn("const int q", local_source)
        self.assertIn("const scalar_t *const SFEM_RESTRICT grad_ref_data", local_source)
        self.assertNotIn("GRAD_REF_NCOMPONENTS", local_source)
        self.assertIn("scalar_t grad_ref[N_SHAPE * 2];", local_source)
        self.assertIn("scalar_t u[N_SHAPE * 2];", local_source)
        self.assertIn(
            "grad_ref[0] = grad_ref_data[(q * N_SHAPE + 0) * 2 + 0];",
            local_source,
        )
        self.assertIn(
            "grad_ref[5] = grad_ref_data[(q * N_SHAPE + 2) * 2 + 1];",
            local_source,
        )
        self.assertIn("jacobian_adjugate[0]", local_source)
        self.assertIn("jacobian_determinant[0]", local_source)

        with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir:
            for generated in generated_files:
                with open(os.path.join(tmpdir, generated.path), "w", encoding="utf-8") as output:
                    output.write(generated.source)

            source_path = os.path.join(tmpdir, "generated_neohookean_ogden_operator.cpp")
            object_path = os.path.join(tmpdir, "generated_neohookean_ogden_operator.o")
            assert_generated_lane_loops_vectorized(
                self,
                compiler,
                source_path,
                object_path,
            )


if __name__ == "__main__":
    unittest.main()
