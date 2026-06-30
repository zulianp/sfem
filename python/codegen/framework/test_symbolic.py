import unittest
import os
import sys

import sympy as sp
import sympy.codegen.ast as ast

sys.path.insert(0, os.path.dirname(__file__))

from symbolic import (
    DeformationGradient,
    DimensionSpecialization,
    DisplacementGradient,
    ExpressionRole,
    FirstPiolaStress,
    GeometricAdjugate,
    GeometricJacobian,
    KernelExpressions,
    KernelTemplateParameter,
    LayoutKind,
    LinearizedTransformedFirstPiola,
    PatternKind,
    ReferenceShapeGradient,
    ReferenceShapeGradients,
    ReferenceShapeValues,
    ScopeKind,
    TransformedFirstPiola,
    build_expression_graph,
    data_layout,
    dimension_specialization,
    directional_derivative,
    displacement_gradient_from_reference,
    execution_scope,
    generate_cpp_kernel,
    generate_cuda_kernel,
    gradient_from_energy,
    hessian_action_from_energy,
    jacobian_action_from_residual,
    kernel_template_parameter,
    linear_elastic_energy,
    linear_elastic_first_piola,
    linearized_first_piola,
    linearized_transformed_first_piola,
    layout_offset,
    matrix_inner,
    matrix_symbols,
    residual_from_energy,
    transformed_first_piola,
    weak_gradient_from_transformed_first_piola,
    weak_hessian_action_from_linearized_transformed_first_piola,
)
from forms import (
    FormKind,
    FormOrder,
    PipelineStage,
    StandardFormName,
    energy_form_pipeline,
    residual_form_pipeline,
)
from targets import CUDATarget, ExecutionModel, OpenMPTarget, TargetLanguage


class SymbolicFrameworkTest(unittest.TestCase):
    def test_unified_energy_and_residual_form_pipelines(self):
        u0, u1, du0, du1 = sp.symbols("u0 u1 du0 du1")
        energy = u0 * u0 + u0 * u1
        residual = sp.Matrix([u0 + u1, u0 - u1])

        energy_evaluation = energy_form_pipeline(
            energy,
            (u0, u1),
            (du0, du1),
        ).evaluate()
        residual_evaluation = residual_form_pipeline(
            residual,
            (u0, u1),
            (du0, du1),
        ).evaluate()
        energy_forms = energy_evaluation.forms
        residual_forms = residual_evaluation.forms

        self.assertEqual(energy_evaluation.stage, PipelineStage.FORM_EVALUATION)
        self.assertEqual(residual_evaluation.stage, PipelineStage.FORM_EVALUATION)

        self.assertEqual(
            [(form.kind, form.order, form.role, form.name) for form in energy_forms],
            [
                (FormKind.ENERGY, FormOrder.ZERO, ExpressionRole.ENERGY, "energy"),
                (FormKind.ENERGY, FormOrder.ONE, ExpressionRole.GRADIENT, "gradient"),
                (
                    FormKind.ENERGY,
                    FormOrder.TWO,
                    ExpressionRole.HESSIAN_ACTION,
                    "hessian_action",
                ),
            ],
        )
        self.assertEqual(
            [(form.kind, form.order, form.role, form.name) for form in residual_forms],
            [
                (FormKind.RESIDUAL, FormOrder.ZERO, ExpressionRole.MERIT, "merit"),
                (FormKind.RESIDUAL, FormOrder.ONE, ExpressionRole.RESIDUAL, "residual"),
                (
                    FormKind.RESIDUAL,
                    FormOrder.TWO,
                    ExpressionRole.JACOBIAN_ACTION,
                    "jacobian_action",
                ),
            ],
        )
        self.assertEqual(energy_forms[1].expression, sp.Matrix([2 * u0 + u1, u0]))
        self.assertEqual(residual_forms[0].expression, u0**2 + u1**2)
        self.assertEqual(
            tuple(form.standard_name for form in energy_forms),
            ("form_0", "form_1", "form_2"),
        )
        self.assertEqual(
            tuple(form.standard_name for form in residual_forms),
            ("form_0", "form_1", "form_2"),
        )
        self.assertIs(
            energy_evaluation.standard_form(StandardFormName.ONE),
            energy_forms[1],
        )
        self.assertIs(
            residual_evaluation.standard_form("form_1"),
            residual_forms[1],
        )
        self.assertEqual(
            tuple(energy_evaluation.standard_forms()),
            ("form_0", "form_1", "form_2"),
        )

    def test_target_platform_classes(self):
        openmp = OpenMPTarget()
        cuda = CUDATarget()

        self.assertEqual(openmp.generated_language, "c++")
        self.assertEqual(openmp.parallel_for_pragma(), "#pragma omp parallel for")
        self.assertEqual(openmp.parallel_for_pragma("static"), "#pragma omp parallel for schedule(static)")
        self.assertEqual(openmp.function_qualifier(), "static SFEM_INLINE")
        self.assertEqual(openmp.restrict_qualifier(), "SFEM_RESTRICT")
        self.assertEqual(openmp.vectorize_pragma(), "#pragma omp simd")
        self.assertEqual(openmp.atomic_update_pragma(), "#pragma omp atomic update")
        self.assertEqual(openmp.alignment_assumption("x"), "__builtin_assume_aligned(x, 64)")
        self.assertEqual(openmp.math_header(), "kernel_math.hpp")
        self.assertEqual(openmp.math_helper_name("pow", 2), "pow_2")
        self.assertEqual(openmp.math_helper_name("pow", -2), "pow_m2")
        self.assertEqual(openmp.diagnostics_header(), "kernel_diagnostics.hpp")
        self.assertEqual(openmp.diagnostic_print_function(), "sfem::codegen::KernelDiagnostics_print_rate")
        self.assertEqual(openmp.kernel_launch_style(), "host_function")
        self.assertEqual(openmp.wrapper_style(), "c_abi")
        self.assertFalse(openmp.supports_device_kernels)
        openmp_loop = openmp.loop_lowering_policy()
        self.assertEqual(openmp_loop.execution_model, ExecutionModel.VECTOR_LANES)
        self.assertTrue(openmp_loop.emits_lane_loop)
        self.assertFalse(openmp_loop.maps_lane_to_thread)
        self.assertTrue(openmp_loop.vectorize_lane_loop)
        self.assertTrue(openmp_loop.parallel_element_loop)
        self.assertEqual(openmp_loop.lane_index, "lane")
        self.assertEqual(openmp_loop.vector_size_symbol, "VECTOR_SIZE")
        self.assertEqual(cuda.language, TargetLanguage.CUDA)
        self.assertEqual(cuda.function_qualifier(), "__device__ __forceinline__")
        self.assertEqual(cuda.restrict_qualifier(), "__restrict__")
        self.assertIsNone(cuda.parallel_for_pragma())
        self.assertIsNone(cuda.vectorize_pragma())
        self.assertEqual(cuda.alignment_assumption("x"), "__builtin_assume_aligned(x, 16)")
        self.assertEqual(cuda.math_helper_name("pow", 3), "pow_3")
        self.assertEqual(cuda.kernel_launch_style(), "cuda_grid_stride")
        self.assertEqual(cuda.wrapper_style(), "cuda_launcher")
        self.assertTrue(cuda.supports_device_kernels)
        cuda_loop = cuda.loop_lowering_policy()
        self.assertEqual(cuda_loop.execution_model, ExecutionModel.SIMT_THREADS)
        self.assertFalse(cuda_loop.emits_lane_loop)
        self.assertTrue(cuda_loop.maps_lane_to_thread)
        self.assertFalse(cuda_loop.vectorize_lane_loop)
        self.assertFalse(cuda_loop.parallel_element_loop)
        self.assertTrue(cuda_loop.supports_shared_memory)

    def test_accepts_all_m1_expression_roles(self):
        x, y, z = sp.symbols("x y z")
        out = sp.symbols("out[0]")

        expressions = (
            KernelExpressions()
            .energy(x * x + y)
            .residual(sp.Matrix([x + y, x - y]))
            .gradient([x * z, y * z])
            .jacobian_action(ast.Assignment(out, (x + y) * z))
            .hessian_action(x / y)
            .merit(sp.sqrt(x * x + y * y))
        )

        graph = expressions.build_graph(data_symbols=(x, y, z))
        roles = [expr.role for expr in graph.outputs]

        self.assertIn(ExpressionRole.ENERGY, roles)
        self.assertIn(ExpressionRole.RESIDUAL, roles)
        self.assertIn(ExpressionRole.GRADIENT, roles)
        self.assertIn(ExpressionRole.JACOBIAN_ACTION, roles)
        self.assertIn(ExpressionRole.HESSIAN_ACTION, roles)
        self.assertIn(ExpressionRole.MERIT, roles)
        self.assertEqual(len(graph.outputs), 8)

    def test_builds_dependency_graph_with_cse_intermediates(self):
        x, y = sp.symbols("x y")
        repeated = x + y

        graph = build_expression_graph(
            [
                (ExpressionRole.GRADIENT, repeated * repeated + repeated),
                (ExpressionRole.RESIDUAL, repeated * repeated - repeated),
            ],
            data_symbols=(x, y),
        )

        self.assertGreaterEqual(len(graph.intermediates), 1)
        self.assertTrue(graph.graph.has_edge(x, graph.intermediates[0][0]))
        self.assertTrue(graph.graph.has_edge(y, graph.intermediates[0][0]))
        self.assertGreaterEqual(graph.cost.temporaries, 1)
        self.assertGreaterEqual(graph.cost.estimated_registers, graph.cost.temporaries)

    def test_preserves_assignment_outputs(self):
        x, y = sp.symbols("x y")
        out = sp.symbols("element_vector[0]")
        assignment = ast.Assignment(out, x * y + y)

        graph = KernelExpressions().residual(assignment).build_graph(data_symbols=(x, y))

        self.assertIsInstance(graph.reduced_outputs[0], ast.Assignment)
        self.assertEqual(graph.reduced_outputs[0].lhs, out)
        self.assertEqual(graph.cost.stores, 1)

    def test_expression_cost_counts_log_and_trigonometric_functions(self):
        x, y, z = sp.symbols("x y z")

        graph = (
            KernelExpressions()
            .add(
                ExpressionRole.OPERATOR_EVALUATION,
                sp.log(x)
                + sp.sin(y)
                + sp.cos(x * y)
                + sp.atan(z)
                + sp.tanh(x + z)
            )
            .build_graph(data_symbols=(x, y, z))
        )

        self.assertEqual(graph.cost.logs, 1)
        self.assertEqual(graph.cost.trigs, 4)
        self.assertGreaterEqual(graph.cost.flops, 20 + 4 * 24)

    def test_expression_cost_counts_exponential_separately(self):
        x = sp.symbols("x")
        graph = (
            KernelExpressions()
            .add(ExpressionRole.OPERATOR_EVALUATION, sp.exp(x))
            .build_graph(data_symbols=(x,))
        )

        self.assertEqual(graph.cost.exps, 1)
        self.assertEqual(graph.cost.pows, 0)
        self.assertEqual(graph.cost.flops, 20)

    def test_generated_cpp_uses_specialized_pow_helpers_for_integer_exponents(self):
        x, y = sp.symbols("x y")
        graph = (
            KernelExpressions()
            .add(ExpressionRole.OPERATOR_EVALUATION, (x + y) ** 2 + x ** 3 + y ** -2)
            .build_graph(data_symbols=(x, y))
        )

        generated = generate_cpp_kernel(
            graph,
            function_name="pow_specialized_kernel",
        )

        self.assertIn("static SFEM_INLINE T pow_2", generated.source)
        self.assertIn("static SFEM_INLINE T pow_3", generated.source)
        self.assertIn("static SFEM_INLINE T pow_m2", generated.source)
        self.assertIn("pow_2(", generated.source)
        self.assertIn("pow_3(", generated.source)
        self.assertIn("pow_m2(", generated.source)
        self.assertNotIn("pow(x", generated.source)
        self.assertNotIn("pow(y", generated.source)

    def test_generated_cuda_kernel_uses_simt_grid_stride_lowering(self):
        x, y = sp.symbols("x[0] y[0]")
        graph = (
            KernelExpressions()
            .add(ExpressionRole.OPERATOR_EVALUATION, (x + y) ** 2)
            .build_graph(data_symbols=(x, y))
        )

        generated = generate_cuda_kernel(
            graph,
            function_name="cuda_plan_kernel",
        )

        self.assertEqual(generated.language, "cuda")
        self.assertIn("#include <cuda_runtime.h>", generated.source)
        self.assertIn("__device__ __forceinline__", generated.source)
        self.assertIn('extern "C" __global__ void cuda_plan_kernel_global', generated.source)
        self.assertIn("blockIdx.x * blockDim.x + threadIdx.x", generated.source)
        self.assertIn("e += blockDim.x * gridDim.x", generated.source)
        self.assertIn("cuda_plan_kernel_global<<<grid_size, block_size>>>", generated.source)
        self.assertIn("pow_2(", generated.source)
        self.assertNotIn("#pragma omp", generated.source)
        self.assertNotIn("ptrdiff_t lane", generated.source)

    def test_tags_loop_symbols(self):
        i, x = sp.symbols("i x")

        graph = KernelExpressions().gradient(i * x).build_graph(
            data_symbols=(x,),
            loop_symbols={"quadrature": (i,)},
        )

        self.assertEqual(graph.graph.nodes[i]["kind"], "loop_index")
        self.assertEqual(graph.graph.nodes[i]["scope"], "quadrature")
        self.assertEqual(graph.graph.nodes[i]["scope_kind"], ScopeKind.QUADRATURE)
        self.assertEqual(graph.graph.nodes[x]["kind"], "data")

    def test_represents_all_m2_execution_scopes(self):
        mesh, patch, elem, q, trial, test, lane, warp, thread = sp.symbols(
            "mesh patch elem q trial test lane warp thread"
        )
        scoped_symbols = (mesh, patch, elem, q, trial, test, lane, warp, thread)
        scopes = (
            execution_scope(ScopeKind.MESH, (mesh,)),
            execution_scope(ScopeKind.PATCH, (patch,)),
            execution_scope(ScopeKind.ELEMENT, (elem,)),
            execution_scope(ScopeKind.QUADRATURE, (q,)),
            execution_scope(ScopeKind.TRIAL, (trial,)),
            execution_scope(ScopeKind.TEST, (test,)),
            execution_scope(ScopeKind.VECTOR_LANE, (lane,)),
            execution_scope(ScopeKind.WARP, (warp,)),
            execution_scope(ScopeKind.THREAD, (thread,)),
        )

        graph = KernelExpressions().gradient(sum(scoped_symbols)).build_graph(scopes=scopes)
        statement = graph.evaluation_plan.outputs[0]

        self.assertEqual(tuple(scope.kind for scope in graph.scopes), tuple(ScopeKind))
        self.assertEqual(statement.scopes, tuple(ScopeKind))
        for kind, symbol in zip(ScopeKind, scoped_symbols):
            self.assertEqual(graph.scope_symbols(kind), (symbol,))

    def test_scope_aliases_work_with_legacy_loop_symbols(self):
        m, lane = sp.symbols("m lane")

        graph = KernelExpressions().gradient(m + lane).build_graph(
            loop_symbols={"mesh_wide": (m,), "vector-lane": (lane,)},
        )

        self.assertEqual(graph.graph.nodes[m]["scope_kind"], ScopeKind.MESH)
        self.assertEqual(graph.graph.nodes[lane]["scope_kind"], ScopeKind.VECTOR_LANE)

    def test_statement_hoist_scope_uses_direct_scope_dependencies(self):
        elem, q, trial = sp.symbols("elem q trial")
        scopes = (
            execution_scope(ScopeKind.ELEMENT, (elem,)),
            execution_scope(ScopeKind.QUADRATURE, (q,)),
            execution_scope(ScopeKind.TRIAL, (trial,)),
        )

        graph = KernelExpressions().gradient(elem + q + trial).build_graph(scopes=scopes)
        statement = graph.evaluation_plan.outputs[0]

        self.assertEqual(statement.scopes, (ScopeKind.ELEMENT, ScopeKind.QUADRATURE, ScopeKind.TRIAL))
        self.assertEqual(statement.hoist_scope, ScopeKind.TRIAL)

    def test_statement_hoist_scope_propagates_through_cse_temporaries(self):
        elem, q, trial, test = sp.symbols("elem q trial test")
        repeated = elem + q
        scopes = (
            execution_scope(ScopeKind.ELEMENT, (elem,)),
            execution_scope(ScopeKind.QUADRATURE, (q,)),
            execution_scope(ScopeKind.TRIAL, (trial,)),
            execution_scope(ScopeKind.TEST, (test,)),
        )

        graph = KernelExpressions().gradient(
            [repeated * trial, repeated * test]
        ).build_graph(scopes=scopes, temporary_prefix="tmp")

        tmp_stmt = graph.evaluation_plan.intermediates[0]
        trial_stmt = graph.evaluation_plan.outputs[0]
        test_stmt = graph.evaluation_plan.outputs[1]

        self.assertEqual(tmp_stmt.scopes, (ScopeKind.ELEMENT, ScopeKind.QUADRATURE))
        self.assertEqual(tmp_stmt.hoist_scope, ScopeKind.QUADRATURE)
        self.assertEqual(
            trial_stmt.scopes,
            (ScopeKind.ELEMENT, ScopeKind.QUADRATURE, ScopeKind.TRIAL),
        )
        self.assertEqual(trial_stmt.hoist_scope, ScopeKind.TRIAL)
        self.assertEqual(
            test_stmt.scopes,
            (ScopeKind.ELEMENT, ScopeKind.QUADRATURE, ScopeKind.TEST),
        )
        self.assertEqual(test_stmt.hoist_scope, ScopeKind.TEST)

    def test_scope_free_statement_is_mesh_hoistable(self):
        x, y = sp.symbols("x y")

        graph = KernelExpressions().gradient(x + y).build_graph(data_symbols=(x, y))
        statement = graph.evaluation_plan.outputs[0]

        self.assertEqual(statement.scopes, ())
        self.assertEqual(statement.hoist_scope, ScopeKind.MESH)

    def test_symbolic_objects_default_to_soa_layout(self):
        grad_u = DisplacementGradient("grad_u", 2)

        self.assertEqual(grad_u.layout.kind, LayoutKind.SOA)

    def test_graph_data_nodes_carry_symbolic_object_layout(self):
        aos = data_layout(LayoutKind.AOS, components=2)
        grad_u = DisplacementGradient("grad_u", 2, layout=aos)
        G = grad_u.as_matrix()

        graph = KernelExpressions().gradient(G[0, 0] + G[1, 1]).build_graph(
            symbolic_objects=(grad_u,),
        )

        self.assertEqual(graph.graph.nodes[G[0, 0]]["layout_kind"], LayoutKind.AOS)
        self.assertEqual(graph.graph.nodes[G[0, 0]]["layout"].components, 2)
        self.assertEqual(graph.graph.nodes[G[0, 0]]["symbolic_object"], "grad_u")

    def test_aosoa_layout_requires_positive_block_size(self):
        with self.assertRaises(ValueError):
            data_layout(LayoutKind.AOSOA)

        with self.assertRaises(ValueError):
            data_layout(LayoutKind.AOSOA, block_size=0)

    def test_aosoa_layout_is_preserved_on_derived_operator_objects(self):
        layout = data_layout(LayoutKind.AOSOA, block_size=8, components=4)
        grad_u = DisplacementGradient("grad_u", 2, layout=layout)
        mu, lmbda = sp.symbols("mu lambda")

        P = FirstPiolaStress.from_linear_elasticity("P", grad_u, mu, lmbda)

        self.assertEqual(P.layout.kind, LayoutKind.AOSOA)
        self.assertEqual(P.layout.block_size, 8)

    def test_layout_offset_for_soa(self):
        i, stride = sp.symbols("i stride")

        offset = layout_offset(data_layout(LayoutKind.SOA), 3, i, components=9, stride=stride)

        self.assertEqual(offset, 3 * stride + i)

    def test_layout_offset_for_aos(self):
        i = sp.symbols("i")

        offset = layout_offset(data_layout(LayoutKind.AOS), 3, i, components=9)

        self.assertEqual(offset, 9 * i + 3)

    def test_layout_offset_for_aosoa(self):
        i = sp.symbols("i", integer=True, nonnegative=True)

        offset = layout_offset(
            data_layout(LayoutKind.AOSOA, block_size=8),
            3,
            i,
            components=9,
        )

        expected = sp.floor(i / 8) * 72 + 24 + sp.Mod(i, 8)
        self.assertEqual(offset, expected)

    def test_symbolic_object_layout_offset_uses_entry_component(self):
        i = sp.symbols("i")
        grad_u = DisplacementGradient("grad_u", 2, layout=data_layout(LayoutKind.AOS))
        G = grad_u.as_matrix()

        self.assertEqual(grad_u.component_index(G[1, 0]), 2)
        self.assertEqual(grad_u.layout_offset(G[1, 0], i), 4 * i + 2)

    def test_graph_data_node_carries_layout_offset(self):
        grad_u = DisplacementGradient("grad_u", 2, layout=data_layout(LayoutKind.AOS))
        G = grad_u.as_matrix()

        graph = KernelExpressions().gradient(G[1, 0]).build_graph(
            symbolic_objects=(grad_u,),
        )
        node = graph.graph.nodes[G[1, 0]]

        self.assertEqual(node["component"], 2)
        self.assertEqual(node["layout_offset"], 4 * node["layout_index"] + 2)

    def test_reference_shape_values_are_linear_arrays(self):
        shape = ReferenceShapeValues("shape", n_nodes=4)

        self.assertEqual(shape.shape, (4,))
        self.assertEqual(shape.value(2), shape.entries[2])
        self.assertEqual(shape.component_index(shape.value(2)), 2)
        self.assertEqual(
            shape.template_parameters,
            (KernelTemplateParameter("shape_n_nodes", 4, "shape"),),
        )

    def test_reference_shape_gradients_are_linear_arrays(self):
        grad = ReferenceShapeGradients("grad_ref", n_nodes=4, dim=3)

        self.assertEqual(grad.shape, (4, 3))
        self.assertEqual(grad.gradient(2, 1), grad.entries[7])
        self.assertEqual(grad.node_gradient(2), sp.Matrix([grad.entries[6], grad.entries[7], grad.entries[8]]))
        self.assertEqual(grad.tensor_gradient(2, 1)[1, 2], grad.gradient(2, 2))
        self.assertEqual(
            grad.template_parameters,
            (
                KernelTemplateParameter("grad_ref_n_nodes", 4, "grad_ref"),
                KernelTemplateParameter("grad_ref_dim", 3, "grad_ref"),
            ),
        )

    def test_reference_shape_gradient_graph_metadata(self):
        grad = ReferenceShapeGradients(
            "grad_ref",
            n_nodes=4,
            dim=3,
            layout=data_layout(LayoutKind.AOS),
        )
        expr = grad.gradient(2, 1) + grad.gradient(0, 2)

        graph = KernelExpressions().gradient(expr).build_graph(symbolic_objects=(grad,))
        node = graph.graph.nodes[grad.gradient(2, 1)]

        self.assertEqual(node["layout_kind"], LayoutKind.AOS)
        self.assertEqual(node["component"], 7)
        self.assertEqual(node["node"], 2)
        self.assertEqual(node["dim_component"], 1)
        self.assertEqual(node["layout_offset"], 12 * node["layout_index"] + 7)

    def test_expression_graph_exposes_kernel_template_parameters(self):
        grad = ReferenceShapeGradients("grad_ref", n_nodes=4, dim=2)
        n_qp = kernel_template_parameter("n_qp", 5, "quadrature")

        graph = KernelExpressions().gradient(grad.gradient(0, 0)).build_graph(
            symbolic_objects=(grad,),
            template_parameters=(n_qp,),
        )

        self.assertEqual(
            graph.template_parameters,
            (
                KernelTemplateParameter("grad_ref_n_nodes", 4, "grad_ref"),
                KernelTemplateParameter("grad_ref_dim", 2, "grad_ref"),
                n_qp,
            ),
        )
        self.assertEqual(graph.graph.graph["template_parameters"], graph.template_parameters)

    def test_conflicting_template_parameter_values_are_rejected(self):
        grad = ReferenceShapeGradients("grad_ref", n_nodes=4, dim=2)

        with self.assertRaises(ValueError):
            KernelExpressions().gradient(grad.gradient(0, 0)).build_graph(
                symbolic_objects=(grad,),
                template_parameters=(kernel_template_parameter("grad_ref_dim", 3),),
            )

    def test_expression_graph_derives_dimension_specialization(self):
        for dim in (1, 2, 3):
            grad = ReferenceShapeGradients("grad_ref", n_nodes=4, dim=dim)

            graph = KernelExpressions().gradient(grad.gradient(0, 0)).build_graph(
                symbolic_objects=(grad,),
            )

            self.assertEqual(graph.specialization, DimensionSpecialization(dim, "grad_ref"))
            self.assertEqual(graph.graph.graph["specialization"], graph.specialization)

    def test_expression_graph_accepts_explicit_dimension_specialization(self):
        x = sp.symbols("x")

        graph = KernelExpressions().gradient(x).build_graph(
            data_symbols=(x,),
            specialization=dimension_specialization(3, "user"),
        )

        self.assertEqual(graph.specialization, DimensionSpecialization(3, "user"))

    def test_conflicting_dimension_specializations_are_rejected(self):
        grad = ReferenceShapeGradients("grad_ref", n_nodes=4, dim=2)

        with self.assertRaises(ValueError):
            KernelExpressions().gradient(grad.gradient(0, 0)).build_graph(
                symbolic_objects=(grad,),
                specialization=dimension_specialization(3, "user"),
            )

    def test_dimension_specialization_is_limited_to_supported_dims(self):
        with self.assertRaises(ValueError):
            dimension_specialization(4)

    def test_names_alone_do_not_define_patterns(self):
        mu, qw = sp.symbols("mu qw")
        jac0 = sp.symbols("jac[0]")
        disp_grad0 = sp.symbols("disp_grad[0]")

        graph = (
            KernelExpressions()
            .energy(mu * disp_grad0 * disp_grad0 * jac0 * qw)
            .build_graph(data_symbols=(mu, disp_grad0, jac0, qw))
        )

        self.assertEqual(len(graph.patterns_by_kind(PatternKind.DISPLACEMENT_GRADIENT)), 0)
        self.assertEqual(len(graph.patterns_by_kind(PatternKind.GEOMETRIC_JACOBIAN)), 0)

    def test_detects_explicit_gradient_and_jacobian_objects(self):
        grad_u = DisplacementGradient("any_gradient_name", 2)
        J = GeometricJacobian("any_geometry_name", 2)

        expr = grad_u.as_matrix()[0, 1] * J.as_matrix()[1, 0]
        graph = KernelExpressions().energy(expr).build_graph(
            symbolic_objects=(grad_u, J)
        )

        self.assertEqual(len(graph.patterns_by_kind(PatternKind.DISPLACEMENT_GRADIENT)), 1)
        self.assertEqual(len(graph.patterns_by_kind(PatternKind.GEOMETRIC_JACOBIAN)), 1)

    def test_detects_derived_deformation_gradient_structurally(self):
        grad_u = DisplacementGradient("du", 2)
        F = DeformationGradient.from_displacement_gradient("F", grad_u)
        Fm = F.as_matrix()

        expr = Fm[0, 0] * Fm[1, 1] - Fm[0, 1] * Fm[1, 0]
        graph = KernelExpressions().energy(expr).build_graph(symbolic_objects=(F,))

        patterns = graph.patterns_by_kind(PatternKind.DEFORMATION_GRADIENT)
        self.assertEqual(len(patterns), 1)
        self.assertGreaterEqual(len(patterns[0].matched_expressions), 1)

    def test_detects_geometric_adjugate_from_jacobian(self):
        J = GeometricJacobian("J", 2)
        adjJ = GeometricAdjugate.from_jacobian("adjJ", J)

        expr = adjJ.as_matrix()[0, 0] + adjJ.as_matrix()[1, 1]
        graph = KernelExpressions().gradient(expr).build_graph(symbolic_objects=(adjJ,))

        patterns = graph.patterns_by_kind(PatternKind.GEOMETRIC_ADJUGATE)
        self.assertEqual(len(patterns), 1)
        self.assertEqual(patterns[0].source, "adjJ")

    def test_marks_cse_intermediates_as_repeated_subexpressions(self):
        x, y = sp.symbols("x y")
        repeated = x + y

        graph = KernelExpressions().residual(
            [repeated * repeated, repeated * repeated + repeated]
        ).build_graph(data_symbols=(x, y))

        repeated_patterns = graph.patterns_by_kind(PatternKind.REPEATED_SUBEXPRESSION)
        self.assertGreaterEqual(len(repeated_patterns), 1)
        self.assertEqual(repeated_patterns[0].source, "sympy_cse")
        self.assertIn("patterns", graph.graph.nodes[repeated_patterns[0].node])

    def test_evaluation_plan_uses_stable_temporary_prefix(self):
        x, y = sp.symbols("x y")
        repeated = x + y

        graph = KernelExpressions().gradient(
            [repeated * repeated, repeated * repeated + repeated]
        ).build_graph(data_symbols=(x, y), temporary_prefix="sfem_tmp")

        self.assertGreaterEqual(len(graph.evaluation_plan.intermediates), 1)
        self.assertEqual(str(graph.evaluation_plan.temporary_symbols[0]), "sfem_tmp0")
        self.assertEqual(graph.evaluation_plan.statements[0].kind, "intermediate")
        self.assertEqual(graph.evaluation_plan.outputs[-1].kind, "output")

    def test_evaluation_plan_accepts_custom_temporary_symbols(self):
        x, y = sp.symbols("x y")
        custom0, custom1 = sp.symbols("qreuse0 qreuse1")
        repeated = x + y

        graph = KernelExpressions().residual(
            [repeated * repeated, repeated * repeated + x]
        ).build_graph(
            data_symbols=(x, y),
            temporary_symbols=iter((custom0, custom1)),
        )

        self.assertGreaterEqual(len(graph.evaluation_plan.temporary_symbols), 1)
        self.assertEqual(graph.evaluation_plan.temporary_symbols[0], custom0)

    def test_evaluation_plan_records_dependencies_and_statement_cost(self):
        x, y, z = sp.symbols("x y z")
        out = sp.symbols("out[0]")
        assignment = ast.Assignment(out, x * y + z)

        graph = KernelExpressions().residual(assignment).build_graph(
            data_symbols=(x, y, z)
        )
        statement = graph.evaluation_plan.outputs[0]

        self.assertEqual(statement.target, out)
        self.assertEqual(statement.role, ExpressionRole.RESIDUAL)
        self.assertEqual(statement.dependencies, (x, y, z))
        self.assertEqual(statement.cost.muls, 1)
        self.assertEqual(statement.cost.adds, 1)
        self.assertEqual(statement.cost.stores, 1)

    def test_liveness_tracks_temporary_last_use(self):
        x, y, z, w = sp.symbols("x y z w")
        repeated = x + y

        graph = KernelExpressions().gradient(
            [repeated * z, repeated * w]
        ).build_graph(data_symbols=(x, y, z, w), temporary_prefix="tmp")

        tmp0 = graph.evaluation_plan.temporary_symbols[0]
        liveness = graph.evaluation_plan.metrics.liveness

        self.assertEqual(liveness[0].live_temporaries_after, (tmp0,))
        self.assertEqual(liveness[1].live_temporaries_after, (tmp0,))
        self.assertEqual(liveness[2].live_temporaries_after, ())
        self.assertEqual(graph.evaluation_plan.metrics.peak_live_temporaries, 1)

    def test_register_pressure_uses_liveness_estimate(self):
        x, y, z, w = sp.symbols("x y z w")
        repeated = x + y

        graph = KernelExpressions().gradient(
            [repeated * z, repeated * w]
        ).build_graph(data_symbols=(x, y, z, w), temporary_prefix="tmp")

        self.assertEqual(graph.evaluation_plan.metrics.peak_registers, 3)
        self.assertEqual(graph.cost.estimated_registers, 3)
        self.assertEqual(graph.cost.stores, graph.evaluation_plan.metrics.total_stores)

    def test_derives_residual_gradient_from_energy(self):
        u0, u1, k = sp.symbols("u0 u1 k")
        energy = sp.Rational(1, 2) * k * (u0 * u0 + u1 * u1) + u0 * u1

        residual = residual_from_energy(energy, (u0, u1))
        gradient = gradient_from_energy(energy, sp.Matrix([u0, u1]))

        expected = sp.Matrix([k * u0 + u1, k * u1 + u0])
        self.assertEqual(residual, expected)
        self.assertEqual(gradient, expected)

    def test_derives_jacobian_action_from_residual(self):
        u0, u1, du0, du1 = sp.symbols("u0 u1 du0 du1")
        residual = sp.Matrix([u0 * u0 + u1, u0 * u1])

        action = jacobian_action_from_residual(
            residual,
            (u0, u1),
            (du0, du1),
        )

        self.assertEqual(action, sp.Matrix([2 * u0 * du0 + du1, u1 * du0 + u0 * du1]))

    def test_derives_hessian_action_from_energy(self):
        u0, u1, du0, du1, k = sp.symbols("u0 u1 du0 du1 k")
        energy = sp.Rational(1, 2) * k * (u0 * u0 + u1 * u1) + u0 * u1

        action = hessian_action_from_energy(
            energy,
            (u0, u1),
            (du0, du1),
        )

        self.assertEqual(action, sp.Matrix([k * du0 + du1, du0 + k * du1]))

    def test_directional_derivative_validates_lengths(self):
        u0, u1, du0 = sp.symbols("u0 u1 du0")

        with self.assertRaises(ValueError):
            directional_derivative(u0 + u1, (u0, u1), (du0,))

    def test_energy_derivative_requires_scalar_energy(self):
        u0, u1 = sp.symbols("u0 u1")

        with self.assertRaises(TypeError):
            residual_from_energy(sp.Matrix([u0, u1]), (u0, u1))

    def test_kernel_expressions_adds_derived_forms(self):
        u0, u1, du0, du1 = sp.symbols("u0 u1 du0 du1")
        energy = sp.Rational(1, 2) * (u0 * u0 + u1 * u1)

        graph = (
            KernelExpressions()
            .energy(energy)
            .residual_from_energy(energy, (u0, u1))
            .hessian_action_from_energy(energy, (u0, u1), (du0, du1))
            .build_graph(data_symbols=(u0, u1, du0, du1))
        )

        roles = [expr.role for expr in graph.outputs]
        self.assertEqual(roles.count(ExpressionRole.ENERGY), 1)
        self.assertEqual(roles.count(ExpressionRole.RESIDUAL), 2)
        self.assertEqual(roles.count(ExpressionRole.HESSIAN_ACTION), 2)

    def test_linear_elastic_first_piola_matches_energy_derivative(self):
        mu, lmbda = sp.symbols("mu lambda")
        grad_u = DisplacementGradient("grad_u", 2)
        G = grad_u.as_matrix()

        energy = linear_elastic_energy(G, mu, lmbda)
        P = linear_elastic_first_piola(G, mu, lmbda)

        expected = sp.Matrix(
            2,
            2,
            [sp.diff(energy, G[i, j]) for i in range(2) for j in range(2)],
        )
        self.assertEqual(sp.simplify(P - expected), sp.zeros(2, 2))

    def test_displacement_gradient_from_reference_matches_old_gpu_path(self):
        u0, u1 = sp.symbols("u0 u1")
        Jinv = matrix_symbols("Jinv", 2, 2)
        grad0 = matrix_symbols("grad0", 2, 2)
        grad1 = matrix_symbols("grad1", 2, 2)

        actual = displacement_gradient_from_reference(
            (u0, u1),
            (grad0, grad1),
            Jinv,
        )

        self.assertEqual(actual, (u0 * grad0 + u1 * grad1) * Jinv)

    def test_transformed_first_piola_matches_p_tx_jinv_t_operand(self):
        mu, lmbda, measure = sp.symbols("mu lambda measure")
        grad_u = DisplacementGradient("grad_u", 2)
        Jinv = matrix_symbols("Jinv", 2, 2)

        P = linear_elastic_first_piola(grad_u, mu, lmbda)
        operand = transformed_first_piola(P, Jinv, measure)

        self.assertEqual(operand, P * Jinv.T * measure)

    def test_weak_gradient_uses_transformed_operand_and_reference_gradients(self):
        P_tXJinv_t = matrix_symbols("P_tXJinv_t", 2, 2)
        grad0 = matrix_symbols("grad0", 2, 2)
        grad1 = matrix_symbols("grad1", 2, 2)

        gradient = weak_gradient_from_transformed_first_piola(
            P_tXJinv_t,
            (grad0, grad1),
        )

        self.assertEqual(gradient[0], matrix_inner(P_tXJinv_t, grad0))
        self.assertEqual(gradient[1], matrix_inner(P_tXJinv_t, grad1))

    def test_transformed_first_piola_object_is_detected_structurally(self):
        mu, lmbda, measure = sp.symbols("mu lambda measure")
        grad_u = DisplacementGradient("grad_u", 2)
        Jinv = matrix_symbols("Jinv", 2, 2)
        P = FirstPiolaStress.from_linear_elasticity("P", grad_u, mu, lmbda)
        operand = TransformedFirstPiola.from_first_piola(
            "P_tXJinv_t",
            P,
            Jinv,
            measure,
        )

        ref_grad = ReferenceShapeGradient("ref_grad0", 2)
        gradient = weak_gradient_from_transformed_first_piola(
            operand,
            (ref_grad,),
        )
        graph = (
            KernelExpressions()
            .operator_evaluation(P)
            .operator_evaluation(operand)
            .gradient(gradient)
            .build_graph(
                symbolic_objects=(P, operand, ref_grad),
            )
        )

        self.assertGreaterEqual(len(graph.patterns_by_kind(PatternKind.FIRST_PIOLA_STRESS)), 1)
        self.assertGreaterEqual(
            len(graph.patterns_by_kind(PatternKind.TRANSFORMED_FIRST_PIOLA)),
            1,
        )
        self.assertGreaterEqual(
            len(graph.patterns_by_kind(PatternKind.REFERENCE_SHAPE_GRADIENT)),
            1,
        )
        self.assertEqual(
            operand.definition_matrix(),
            P.as_matrix() * Jinv.T * measure,
        )
        operator_targets = tuple(
            statement.target
            for statement in graph.evaluation_plan.outputs
            if statement.role == ExpressionRole.OPERATOR_EVALUATION
        )
        for entry in operand.entries:
            self.assertIn(entry, operator_targets)

    def test_linearized_first_piola_matches_directional_derivative(self):
        mu, lmbda = sp.symbols("mu lambda")
        grad_u = DisplacementGradient("grad_u", 2)
        direction = matrix_symbols("trial_grad", 2, 2)
        G = grad_u.as_matrix()
        P = linear_elastic_first_piola(G, mu, lmbda)

        dP = linearized_first_piola(P, G, direction)
        expected = sp.Matrix(
            2,
            2,
            [
                sum(
                    sp.diff(P[i, j], G[k, l]) * direction[k, l]
                    for k in range(2)
                    for l in range(2)
                )
                for i in range(2)
                for j in range(2)
            ],
        )

        self.assertEqual(sp.simplify(dP - expected), sp.zeros(2, 2))

    def test_linearized_transformed_operand_matches_direct_gradient_derivative(self):
        mu, lmbda, measure = sp.symbols("mu lambda measure")
        grad_u = DisplacementGradient("grad_u", 2)
        G = grad_u.as_matrix()
        Jinv = matrix_symbols("Jinv", 2, 2)
        trial_ref = matrix_symbols("trial_ref", 2, 2)
        test_ref = matrix_symbols("test_ref", 2, 2)

        P = linear_elastic_first_piola(G, mu, lmbda)
        transformed = transformed_first_piola(P, Jinv, measure)
        optimized_grad = matrix_inner(transformed, test_ref)
        direction = trial_ref * Jinv
        direct = sum(
            sp.diff(optimized_grad, G[i, j]) * direction[i, j]
            for i in range(2)
            for j in range(2)
        )

        lin_operand = linearized_transformed_first_piola(
            P,
            G,
            trial_ref,
            Jinv,
            measure,
        )
        optimized = matrix_inner(lin_operand, test_ref)

        self.assertEqual(sp.simplify(optimized - direct), 0)

    def test_linearized_transformed_first_piola_object_evaluates_hessian_action(self):
        mu, lmbda, measure = sp.symbols("mu lambda measure")
        grad_u = DisplacementGradient("grad_u", 2)
        Jinv = matrix_symbols("Jinv", 2, 2)
        P = FirstPiolaStress.from_linear_elasticity("P", grad_u, mu, lmbda)
        trial_ref = ReferenceShapeGradient("trial_ref", 2)
        lin = LinearizedTransformedFirstPiola.from_first_piola(
            "lin_P_tXJinv_t",
            P,
            grad_u,
            trial_ref,
            Jinv,
            measure,
        )
        test_ref = ReferenceShapeGradient("test_ref", 2)
        self.assertEqual(
            lin.definition_matrix(),
            linearized_transformed_first_piola(
                P,
                grad_u,
                trial_ref,
                Jinv,
                measure,
            ),
        )

        action = weak_hessian_action_from_linearized_transformed_first_piola(
            lin,
            (test_ref,),
        )
        graph = (
            KernelExpressions()
            .operator_evaluation(P)
            .operator_evaluation(lin)
            .hessian_action(action)
            .build_graph(symbolic_objects=(P, trial_ref, test_ref, lin))
        )

        self.assertGreaterEqual(
            len(graph.patterns_by_kind(PatternKind.LINEARIZED_TRANSFORMED_FIRST_PIOLA)),
            1,
        )
        operator_targets = tuple(
            statement.target
            for statement in graph.evaluation_plan.outputs
            if statement.role == ExpressionRole.OPERATOR_EVALUATION
        )
        for entry in lin.entries:
            self.assertIn(entry, operator_targets)


if __name__ == "__main__":
    unittest.main()
