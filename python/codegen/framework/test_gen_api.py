import json
import os
import shutil
import subprocess
import tempfile
import unittest

import sympy as sp

from sfem import gen

from .materials.neohookean_ogden import material as neohookean_ogden
from .materials.poro_hyperelasticity import material as poro_hyperelasticity
from .materials.stokes import material as stokes
from .materials.two_phase_flow import material as two_phase_flow
from .generate_stokes_files import validate_m6_4 as validate_stokes_m6_4
from .tensor_product_geometry import (
    tensor_product_evaluated_isoparametric_geometry_lines,
    tensor_product_gradient_isoparametric_geometry_lines,
    tensor_product_ordered_coordinate_streams,
)


def _relative_sources(result, out_dir):
    return {os.path.relpath(path, out_dir) for path in result.sources}


class GenApiTest(unittest.TestCase):
    def test_symbolic_scalar_field_is_sympy_compatible(self):
        p = gen.scalar_field("p", family="pressure")
        expression = p * p + 2 * p + 1

        self.assertIsInstance(p, gen.ScalarField)
        self.assertEqual(p.value, sp.Symbol("p"))
        self.assertEqual(sp.sympify(p), sp.Symbol("p"))
        self.assertEqual(sp.diff(expression, p.symbol), 2 * p.symbol + 2)
        self.assertEqual(p.family, "pressure")

    def test_symbolic_vector_and_tensor_fields_expose_sympy_values(self):
        u = gen.vector_field("u", 3, family="displacement")
        F = gen.tensor_field("F", (3, 3), metadata={"qualifier": "deformation"})

        self.assertIsInstance(u, gen.VectorField)
        self.assertEqual(u.shape, (3,))
        self.assertEqual(tuple(u), tuple(sp.Symbol("u[%d]" % i) for i in range(3)))
        self.assertEqual(u.as_matrix().shape, (3, 1))
        self.assertEqual(u[2], sp.Symbol("u[2]"))

        self.assertIsInstance(F, gen.TensorField)
        self.assertEqual(F.shape, (3, 3))
        self.assertEqual(F.as_matrix().shape, (3, 3))
        self.assertEqual(F[1, 2], sp.Symbol("F[5]"))
        self.assertEqual(F.metadata["qualifier"], "deformation")

    def test_symbolic_field_validation(self):
        with self.assertRaisesRegex(ValueError, "valid identifier"):
            gen.scalar_field("not valid")
        with self.assertRaisesRegex(ValueError, "positive"):
            gen.vector_field("u", 0)
        with self.assertRaisesRegex(ValueError, "rank at least 2"):
            gen.tensor_field("T", (3,))

    def test_symbolic_test_and_trial_functions_follow_field_shape(self):
        p = gen.scalar_field("p", family="pressure")
        q = gen.test_function(p)
        dp = gen.trial_function(p)
        expression = q * (p + dp)

        self.assertIsInstance(q, gen.TestFunction)
        self.assertIsInstance(dp, gen.TrialFunction)
        self.assertEqual(q.field, p)
        self.assertEqual(dp.field, p)
        self.assertEqual(q.role, gen.TEST_ARGUMENT)
        self.assertEqual(dp.role, gen.TRIAL_ARGUMENT)
        self.assertEqual(q.value, sp.Symbol("p_test"))
        self.assertEqual(dp.value, sp.Symbol("p_trial"))
        self.assertEqual(
            sp.diff(expression, dp.value),
            sp.Symbol("p_test"),
        )

    def test_symbolic_vector_test_function_exposes_matrix(self):
        u = gen.vector_field("u", 2, family="displacement")
        v = gen.test_function(u, name="v")
        du = gen.trial_function(u, name="du")

        self.assertEqual(v.shape, (2,))
        self.assertEqual(v.family, "displacement")
        self.assertEqual(tuple(v), (sp.Symbol("v[0]"), sp.Symbol("v[1]")))
        self.assertEqual(v.as_matrix().shape, (2, 1))
        self.assertEqual(du.as_matrix()[1, 0], sp.Symbol("du[1]"))

    def test_ufl_style_function_spaces_and_arguments(self):
        V = gen.FunctionSpace(
            gen.VectorElement("Lagrange", degree=2),
            dim=3,
        )
        Q = gen.FunctionSpace(
            gen.FiniteElement("Lagrange", degree=1),
            dim=3,
        )
        W = gen.MixedFunctionSpace(V, Q)

        u = gen.Function(V, "u", qualifier=gen.DISPLACEMENT)
        p = gen.Function(Q, "p", qualifier=gen.PRESSURE)
        v = gen.TestFunction(V, name="v")
        du = gen.TrialFunction(V, name="du")

        self.assertEqual(len(W), 2)
        self.assertEqual(u.shape, (3,))
        self.assertEqual(u.family, "displacement")
        self.assertEqual(p.shape, ())
        self.assertEqual(p.family, "pressure")
        self.assertEqual(v.shape, (3,))
        self.assertEqual(tuple(v), tuple(sp.Symbol("v[%d]" % i) for i in range(3)))
        self.assertEqual(du[2], sp.Symbol("du[2]"))

    def test_geometric_vector_space_uses_generation_dimension_context(self):
        V = gen.FunctionSpace(
            gen.VectorElement("Lagrange", degree=2)
        )

        with gen.geometric_dimension_context(2):
            u = gen.Function(V, "u", qualifier=gen.DISPLACEMENT)
            v = gen.TestFunction(V, name="v")
            du = gen.TrialFunction(V, name="du")

        self.assertEqual(u.shape, (2,))
        self.assertEqual(u.metadata["dim"], 2)
        self.assertEqual(v.shape, (2,))
        self.assertEqual(du.shape, (2,))
        self.assertEqual(tuple(v), (sp.Symbol("v[0]"), sp.Symbol("v[1]")))

    def test_ufl_style_operators_return_sympy_expressions(self):
        p = gen.scalar_field("p")
        u = gen.vector_field("u", 2)
        F = gen.tensor_field("F", (2, 2))

        grad_p = gen.grad(p, dim=2)
        grad_u = gen.grad(u)
        deformation = gen.deformation_gradient(u)

        self.assertEqual(gen.value(p), sp.Symbol("p"))
        self.assertEqual(grad_p, sp.Matrix([sp.Symbol("p_grad[0]"), sp.Symbol("p_grad[1]")]))
        self.assertEqual(grad_u.shape, (2, 2))
        self.assertEqual(gen.div(u), sp.Symbol("u_grad[0]") + sp.Symbol("u_grad[3]"))
        self.assertEqual(deformation, sp.eye(2) + grad_u)
        self.assertEqual(gen.inner(u.value, u.value), sp.Symbol("u[0]")**2 + sp.Symbol("u[1]")**2)
        self.assertEqual(gen.det(F), F.as_matrix().det())
        self.assertEqual(gen.inv(F), F.as_matrix().inv())
        self.assertEqual(gen.adjugate(F), F.as_matrix().adjugate())

    def test_codegen_qualifiers_and_material_parameters(self):
        mu = gen.material_parameter("mu", default=2.0)
        p = gen.scalar_field("p")
        u = gen.vector_field("u", 2)

        expression = mu * p * p
        F = gen.qualify(
            gen.deformation_gradient(u),
            gen.DEFORMATION_GRADIENT,
        )

        self.assertIsInstance(mu, gen.MaterialParameter)
        self.assertEqual(mu.value, sp.Symbol("mu"))
        self.assertEqual(mu.default, 2.0)
        self.assertIn(gen.MATERIAL_PARAMETER, mu.qualifiers)
        self.assertEqual(sp.diff(expression, mu.symbol), p.value**2)

        self.assertIsInstance(F, gen.QualifiedExpression)
        self.assertEqual(F.value, gen.deformation_gradient(u))
        self.assertEqual(gen.qualifiers(F), (gen.DEFORMATION_GRADIENT,))

    def test_equation_system_accepts_vector_scalar_energy_and_residual_equations(self):
        system = gen.EquationSystem(3)
        displacement = system.vector_field("u", family="displacement")
        pressure = system.scalar_field("p", family="pressure")
        variables = sp.symbols("F[0:9]")

        energy = system.add_energy(
            "solid",
            variables[0],
            fields=(displacement,),
            variables=(variables,),
        )
        residual = system.add_residual(
            "flow",
            lambda residual_system: None,
            fields=(displacement, pressure),
        )

        self.assertTrue(displacement.is_vector)
        self.assertTrue(pressure.is_scalar)
        self.assertEqual(energy.form, gen.EquationForm.ENERGY)
        self.assertEqual(residual.form, gen.EquationForm.RESIDUAL)
        self.assertEqual(system.equations, (energy, residual))

    def test_equation_system_builder_maps_symbolic_fields_to_equations(self):
        builder = gen.EquationSystemBuilder(3)
        displacement = builder.vector_field("u", family="displacement")
        pressure = builder.scalar_field("p", family="pressure")
        F = gen.variable(
            gen.Identity(3) + gen.grad(displacement),
            name="F",
            qualifier=gen.DEFORMATION_GRADIENT,
        )

        builder.add_energy("solid", F.value[0, 0], fields=(displacement,), variables=(F,))
        builder.add_residual("flow", lambda residual_system: None, fields=(displacement, pressure))

        system = builder.build()
        self.assertIsInstance(displacement, gen.VectorField)
        self.assertIsInstance(pressure, gen.ScalarField)
        self.assertEqual(tuple(field.name for field in system.fields), ("u", "p"))
        self.assertEqual(tuple(field.components for field in system.fields), (3, 1))
        self.assertEqual(
            tuple(field.family for field in system.equations[1].fields),
            ("displacement", "pressure"),
        )

    def test_equation_system_builder_derives_energy_and_merit_forms(self):
        builder = gen.EquationSystemBuilder(1)
        p = builder.scalar_field("p")

        energy = builder.derive_energy_forms(p * p, variables=p)
        merit = builder.derive_merit_forms(p * p + p, variables=p)

        self.assertEqual(tuple(energy.standard_forms()), ("form_0", "form_1", "form_2"))
        self.assertIs(energy.standard_form(gen.StandardFormName.TWO), energy.forms[2])
        self.assertEqual(energy.forms[0].expression, p.value**2)
        self.assertEqual(energy.forms[1].expression, sp.Matrix([2 * p.value]))
        self.assertEqual(
            energy.forms[2].expression,
            sp.Matrix([2 * sp.Symbol("p_trial")]),
        )
        self.assertEqual(merit.forms[1].expression, sp.Matrix([2 * p.value + 1]))

    def test_energy_equation_uses_explicit_differentiation_variables(self):
        builder = gen.EquationSystemBuilder(1)
        p = builder.scalar_field("p")

        equation = builder.add_energy("stored", p * p, fields=(p,), variables=(p,))
        evaluated = gen._evaluate_equation(1, equation)

        self.assertEqual(equation.variables, (p.value,))
        self.assertEqual(
            evaluated.form_evaluation.form(gen.FormOrder.ONE).expression,
            sp.Matrix([2 * p.value]),
        )
        self.assertEqual(
            evaluated.form_evaluation.form(gen.FormOrder.TWO).expression,
            sp.Matrix([2 * sp.Symbol("p_trial")]),
        )

    def test_equation_system_owns_standard_form_collections(self):
        builder = gen.EquationSystemBuilder(1)
        p = builder.scalar_field("p", family="pressure")
        q = gen.test_function(p)

        energy = builder.add_energy("stored", p * p, fields=(p,), variables=(p,))
        residual = builder.add_residual("flow", p * q, fields=(p,))
        system = builder.build()

        energy_forms = system.form_collection(energy)
        residual_forms = system.form_collection("flow")

        self.assertIsInstance(energy_forms, gen.FormCollection)
        self.assertIsInstance(residual_forms, gen.FormCollection)
        self.assertEqual(energy_forms.equation_name, "stored")
        self.assertEqual(residual_forms.equation_name, "flow")
        self.assertEqual(tuple(energy_forms.standard_forms()), ("form_0", "form_1", "form_2"))
        self.assertEqual(tuple(residual_forms.standard_forms()), ("form_0", "form_1", "form_2"))
        self.assertEqual(energy_forms.form(gen.FormOrder.ONE).expression, sp.Matrix([2 * p.value]))
        self.assertEqual(
            residual_forms.form(gen.FormOrder.ONE).expression,
            sp.Matrix([p.value * q.value]),
        )
        self.assertEqual(
            residual_forms.source.residual_expression(residual_forms.source.fields[0]),
            p.value * q.value,
        )
        residual_metadata = residual_forms.form_metadata(gen.FormOrder.ONE)
        action_metadata = residual_forms.form_metadata(gen.FormOrder.TWO)
        self.assertIsInstance(residual_metadata, gen.FormMetadata)
        self.assertEqual(residual_metadata.coefficients[0].row_field, "p")
        self.assertEqual(residual_metadata.coefficients[0].value, p.value)
        self.assertTrue(residual_metadata.dependencies.current)
        self.assertFalse(residual_metadata.dependencies.direction)
        self.assertIn(gen.FormQualifier("p", "field_family", "pressure"), energy_forms.qualifiers)
        self.assertIn(gen.FormQualifier("p", "field_family", "pressure"), residual_forms.qualifiers)
        self.assertEqual(
            energy_forms.form_metadata(gen.FormOrder.ONE).dependencies,
            (p.value,),
        )
        self.assertEqual(action_metadata.coefficients[0].value, sp.Symbol("p_direction"))
        self.assertTrue(action_metadata.dependencies.direction)
        self.assertFalse(action_metadata.dependencies.previous)
        self.assertTrue(residual_metadata.blocks)
        self.assertTrue(action_metadata.blocks)
        residual_block = residual_forms.block(gen.FormOrder.ONE, "p")
        action_block = residual_forms.block(gen.FormOrder.TWO, "p", "p")
        self.assertIsInstance(residual_block, gen.FormBlock)
        self.assertIsInstance(action_block, gen.FormBlock)
        self.assertEqual(residual_block.expression, p.value * q.value)
        self.assertEqual(action_block.expression, sp.Symbol("p_direction") * q.value)
        self.assertEqual(action_metadata.blocks, residual_forms.blocks_for(gen.FormOrder.TWO))
        self.assertEqual(residual_forms.block_matrix(gen.FormOrder.TWO), ((action_block,),))
        residual_unit = gen._residual_codegen_unit(
            "material",
            1,
            gen._evaluate_equation(1, residual, residual_forms),
        )
        self.assertIs(residual_unit.form_collection, residual_forms)
        self.assertIsNone(residual_unit.payload)
        self.assertIs(system.form_collection(energy), energy_forms)
        self.assertEqual(system.form_collections(), (energy_forms, residual_forms))

    def test_hyperelastic_energy_records_deformation_gradient_explicitly(self):
        builder = gen.EquationSystemBuilder(2)
        u = builder.vector_field("u", family="displacement")
        F = gen.variable(
            gen.Identity(2) + gen.grad(u),
            name="F",
            qualifier=gen.DEFORMATION_GRADIENT,
        )

        equation = builder.add_energy("solid", gen.inner(F, F), fields=(u,), variables=(F,))

        self.assertEqual(equation.variables, tuple(F.value))

    def test_code_generator_requires_explicit_equation_systems(self):
        builder = gen.EquationSystemBuilder(2)
        p = builder.scalar_field("p")
        builder.add_energy("", p * p, fields=(p,), variables=(p,))
        systems = gen.EquationSystems(builder.build())

        material = gen.CodeGenerator("explicit_system", systems, elements=("TRI3",))

        self.assertEqual(material.systems.dims, (2,))
        with self.assertRaisesRegex(TypeError, "not a callback"):
            gen.CodeGenerator("callback_system", lambda system: None, elements=("TRI3",))

    def test_equation_system_builder_derives_residual_and_jacobian_action_forms(self):
        builder = gen.EquationSystemBuilder(2)
        u = builder.vector_field("u", 2)
        residual = sp.Matrix([u[0] + u[1], u[0] - u[1]])

        forms = builder.derive_residual_forms(residual, fields=(u,))

        self.assertEqual(tuple(forms.standard_forms()), ("form_0", "form_1", "form_2"))
        self.assertIs(forms.standard_form("form_1"), forms.forms[1])
        self.assertEqual(forms.forms[0].expression, u[0] ** 2 + u[1] ** 2)
        self.assertEqual(forms.forms[1].expression, residual)
        self.assertEqual(
            forms.forms[2].expression,
            sp.Matrix([
                sp.Symbol("u_trial[0]") + sp.Symbol("u_trial[1]"),
                sp.Symbol("u_trial[0]") - sp.Symbol("u_trial[1]"),
            ]),
        )

    def test_equation_system_builder_handles_mixed_scalar_vector_forms(self):
        builder = gen.EquationSystemBuilder(2)
        u = builder.vector_field("u", family="displacement")
        p = builder.scalar_field("p", family="pressure")
        residual = sp.Matrix([u[0] + p.value, u[1] - p.value, p.value + u[0] - u[1]])

        forms = builder.derive_residual_forms(residual, fields=(u, p))

        self.assertEqual(forms.forms[1].expression, residual)
        self.assertEqual(
            forms.forms[2].expression,
            sp.Matrix([
                sp.Symbol("u_trial[0]") + sp.Symbol("p_trial"),
                sp.Symbol("u_trial[1]") - sp.Symbol("p_trial"),
                sp.Symbol("p_trial") + sp.Symbol("u_trial[0]") - sp.Symbol("u_trial[1]"),
            ]),
        )
        self.assertEqual(
            tuple(field.family for field in builder.build().fields),
            ("displacement", "pressure"),
        )

    def test_residual_form_collection_exposes_coupled_block_matrix(self):
        builder = gen.EquationSystemBuilder(2)
        u = builder.scalar_field("u")
        v = builder.scalar_field("v")
        q_u = gen.test_function(u)
        q_v = gen.test_function(v)
        builder.add_residual(
            "coupled",
            (u.value + 2 * v.value) * q_u + (3 * u.value - v.value) * q_v,
            fields=(u, v),
        )
        forms = builder.build().form_collection("coupled")

        residual_blocks = forms.blocks_for(gen.FormOrder.ONE)
        action_blocks = forms.blocks_for(gen.FormOrder.TWO)
        self.assertEqual(tuple(block.row_field for block in residual_blocks), ("u", "v"))
        self.assertEqual(len(action_blocks), 4)

        matrix = forms.block_matrix(gen.FormOrder.TWO)
        self.assertEqual(
            tuple(tuple(block.name for block in row) for row in matrix),
            (
                ("form_2_u_u", "form_2_u_v"),
                ("form_2_v_u", "form_2_v_v"),
            ),
        )
        self.assertEqual(matrix[0][1].expression, 2 * sp.Symbol("v_direction") * q_u.value)
        self.assertEqual(matrix[1][0].expression, 3 * sp.Symbol("u_direction") * q_v.value)
        self.assertTrue(matrix[0][1].is_coupling)
        self.assertTrue(matrix[1][1].is_diagonal)
        self.assertEqual(matrix[0][1].coefficients[0].row_field, "u")
        self.assertTrue(matrix[0][1].dependencies.direction)

    def test_generates_hyperelastic_material(self):
        with tempfile.TemporaryDirectory() as out_dir:
            result = gen.generate(
                neohookean_ogden,
                out_dir,
                elements=("TRI3",),
            )
            names = _relative_sources(result, out_dir)
            self.assertIn(
                "d2/tri3/neohookean_ogden_tri3_operator.cpp",
                names,
            )
            self.assertIn("d2/neohookean_ogden_d2_simplex_local.hpp", names)
            self.assertIn("kernel_diagnostics.hpp", names)
            self.assertIn("op/sfem_GeneratedNeoHookeanOgden.cpp", names)
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
                "neohookean_ogden_tri3_tri3_gradient_isoparametric_mesh_soa",
                source,
            )
            self.assertNotIn("generated_neohookean_ogden", source)

    def test_material_runner_writes_generation_plan_dump(self):
        with tempfile.TemporaryDirectory() as out_dir:
            plan_path = os.path.join(out_dir, "inspect", "neo_plan.json")
            result = gen.run(
                neohookean_ogden,
                out_dir,
                argv=(
                    "--element",
                    "TRI3",
                    "--plan-out",
                    plan_path,
                ),
            )
            self.assertIsInstance(result.plan, gen.GenerationPlan)
            self.assertEqual(result.plan_dump, plan_path)
            self.assertTrue(os.path.exists(plan_path))
            with open(plan_path, encoding="utf-8") as stream:
                dump = json.load(stream)
            self.assertEqual(dump["stage"], gen.PipelineStage.SPECIALIZED_FORM_MANIPULATION.value)
            self.assertEqual(dump["n_monolithic_kernels"], 1)
            self.assertEqual(dump["kernels"][0]["name"], "neohookean_ogden")
            self.assertEqual(dump["kernels"][0]["mesh_phases"], ["geometry", "local_call", "scatter"])

    def test_generates_coupled_residual_material(self):
        with tempfile.TemporaryDirectory() as out_dir:
            result = gen.generate(
                two_phase_flow,
                out_dir,
                elements=("TRI3",),
            )
            names = _relative_sources(result, out_dir)
            self.assertIn(
                "d2/tri3/two_phase_flow_tri3_operator.cpp",
                names,
            )
            self.assertIn(
                "d2/two_phase_flow_d2_simplex_local.hpp",
                names,
            )
            self.assertIn("op/sfem_GeneratedTwoPhaseFlow.cpp", names)
            wrapper = os.path.join(out_dir, "op", "sfem_GeneratedTwoPhaseFlow.cpp")
            with open(wrapper, encoding="utf-8") as stream:
                source = stream.read()
            self.assertLess(
                source.index('parameters.require_real_value("C_ka1")'),
                source.index('parameters.require_real_value("porosity")'),
            )
            self.assertIn("two_phase_flow_tri3_residual_isoparametric_mesh_aos", source)
            self.assertIn("two_phase_flow_tri3_jacobian_action_isoparametric_mesh_aos", source)
            self.assertNotIn("generated_two_phase_flow", source)
            self.assertNotIn('parameters.require_real_value("K_" + std::to_string(i))', source)

    def test_poro_hyperelastic_material_uses_taylor_hood_elements(self):
        names = tuple(element.name for element in poro_hyperelasticity.elements)
        self.assertEqual(names, ("TRI6_TRI3", "TET10_TET4", "HEX27_HEX8"))

    def test_taylor_hood_elements_are_detected_from_mixed_function_spaces(self):
        system = poro_hyperelasticity.systems.for_dim(3)
        names = tuple(
            element.name
            for element in gen.sfem_detect_taylor_hood_element_types(system.fields)
        )
        self.assertEqual(names, ("TET10_TET4", "HEX27_HEX8"))

        material = gen.CodeGenerator("auto_taylor_hood", gen.EquationSystems(system))
        self.assertEqual(
            tuple(element.name for element in material.elements),
            ("TET10_TET4", "HEX27_HEX8"),
        )

    def test_stokes_material_uses_taylor_hood_elements(self):
        names = tuple(element.name for element in stokes.elements)
        self.assertEqual(names, ("TRI6_TRI3", "TET10_TET4", "HEX27_HEX8"))

        system = stokes.systems.for_dim(2)
        self.assertEqual(tuple(field.name for field in system.fields), ("u", "p"))
        self.assertEqual(tuple(field.components for field in system.fields), (2, 1))
        self.assertEqual(tuple(field.family for field in system.fields), ("velocity", "pressure"))
        forms = system.form_collection("")
        self.assertEqual(
            tuple(block.row_field for block in forms.blocks_for(gen.FormOrder.ONE)),
            ("u", "p"),
        )
        self.assertEqual(
            tuple(
                (block.row_field, block.column_field)
                for block in forms.blocks_for(gen.FormOrder.TWO)
            ),
            (("u", "u"), ("u", "p"), ("p", "u")),
        )
        self.assertNotIn(
            ("p", "p"),
            tuple(
                (block.row_field, block.column_field)
                for block in forms.blocks_for(gen.FormOrder.TWO)
            ),
        )

    def test_mixed_isoparametric_taylor_hood_uses_higher_quadrature(self):
        tri = gen.sfem_fem_policy(stokes.elements[0])
        tet = gen.sfem_fem_policy(stokes.elements[1])
        hex27 = gen.sfem_fem_policy(stokes.elements[2])

        self.assertEqual((tri.quadrature_rule.order, tri.quadrature_rule.n_qp), (4, 6))
        self.assertEqual((tet.quadrature_rule.order, tet.quadrature_rule.n_qp), (4, 11))
        self.assertEqual((hex27.quadrature_rule.order, hex27.quadrature_rule.n_qp), (4, 64))

        self.assertEqual(
            gen.sfem_default_quadrature_order("TRI6"),
            2,
        )
        self.assertEqual(
            gen.sfem_default_quadrature_order("TRI6", "isoparametric_mixed"),
            4,
        )
        self.assertEqual(
            gen.sfem_default_quadrature_order("TRI6", "affine_mixed"),
            2,
        )

    def test_current_materials_select_case_specific_quadrature(self):
        def context_orders(material, elements):
            stage = gen.UserInputStage.create(material, elements, 16, None)
            return {
                context.label.upper(): (
                    context.specialization.quadrature_rule.order,
                    context.specialization.quadrature_rule.n_qp,
                )
                for context in stage.element_contexts
            }

        def affine_context_orders(material, elements):
            stage = gen.UserInputStage.create(material, elements, 16, None)
            return {
                context.label.upper(): (
                    context.affine_specialization.quadrature_rule.order,
                    context.affine_specialization.quadrature_rule.n_qp,
                )
                for context in stage.element_contexts
            }

        neo = context_orders(
            neohookean_ogden,
            ("TRI3", "TRI6", "TET4", "TET10", "QUAD4", "HEX8", "HEX27"),
        )
        self.assertEqual(neo["TRI3"], (1, 1))
        self.assertEqual(neo["TET4"], (1, 1))
        self.assertEqual(neo["QUAD4"], (2, 4))
        self.assertEqual(neo["HEX8"], (2, 8))
        self.assertEqual(neo["TRI6"], (4, 6))
        self.assertEqual(neo["TET10"], (4, 11))
        self.assertEqual(neo["HEX27"], (4, 64))

        neo_affine = affine_context_orders(
            neohookean_ogden,
            ("TRI3", "TRI6", "TET4", "TET10", "QUAD4", "HEX8", "HEX27"),
        )
        self.assertEqual(neo_affine["TRI3"], (1, 1))
        self.assertEqual(neo_affine["TET4"], (1, 1))
        self.assertEqual(neo_affine["QUAD4"], (2, 4))
        self.assertEqual(neo_affine["HEX8"], (2, 8))
        self.assertEqual(neo_affine["TRI6"], (2, 3))
        self.assertEqual(neo_affine["TET10"], (2, 4))
        self.assertEqual(neo_affine["HEX27"], (3, 27))

        two_phase = context_orders(
            two_phase_flow,
            ("TRI3", "TET4", "QUAD4", "HEX8"),
        )
        self.assertEqual(two_phase["TRI3"], (4, 6))
        self.assertEqual(two_phase["TET4"], (4, 11))
        self.assertEqual(two_phase["QUAD4"], (4, 16))
        self.assertEqual(two_phase["HEX8"], (4, 64))

        two_phase_affine = affine_context_orders(
            two_phase_flow,
            ("TRI3", "TET4", "QUAD4", "HEX8"),
        )
        self.assertEqual(two_phase_affine, two_phase)

        stokes_orders = context_orders(
            stokes,
            stokes.elements,
        )
        self.assertEqual(stokes_orders["TRI6_TRI3"], (4, 6))
        self.assertEqual(stokes_orders["TET10_TET4"], (4, 11))
        self.assertEqual(stokes_orders["HEX27_HEX8"], (4, 64))

        stokes_affine = affine_context_orders(
            stokes,
            stokes.elements,
        )
        self.assertEqual(stokes_affine["TRI6_TRI3"], (2, 3))
        self.assertEqual(stokes_affine["TET10_TET4"], (2, 4))
        self.assertEqual(stokes_affine["HEX27_HEX8"], (3, 27))

        poro_orders = context_orders(
            poro_hyperelasticity,
            poro_hyperelasticity.elements,
        )
        self.assertEqual(poro_orders, stokes_orders)

        poro_affine = affine_context_orders(
            poro_hyperelasticity,
            poro_hyperelasticity.elements,
        )
        self.assertEqual(poro_affine, stokes_affine)

    def test_material_quadrature_settings_override_case_defaults(self):
        material = gen.CodeGenerator(
            "two_phase_custom_quadrature",
            two_phase_flow.systems,
            elements=("TRI3",),
            quadrature_settings=(("TRI3", "value_residual", 2),),
        )
        stage = gen.UserInputStage.create(material, ("TRI3",), 16, None)
        rule = stage.element_contexts[0].specialization.quadrature_rule
        self.assertEqual((rule.order, rule.n_qp), (2, 3))

        stage = gen.UserInputStage.create(material, ("TRI3",), 16, 3)
        rule = stage.element_contexts[0].specialization.quadrature_rule
        self.assertEqual((rule.order, rule.n_qp), (3, 4))

    def test_material_quadrature_settings_can_target_affine_and_isoparametric_cases(self):
        material = gen.CodeGenerator(
            "neo_custom_quadrature",
            neohookean_ogden.systems,
            elements=("TRI6",),
            quadrature_settings=(
                ("TRI6", "isoparametric_energy", 4),
                ("TRI6", "affine_energy", 3),
            ),
        )
        stage = gen.UserInputStage.create(material, ("TRI6",), 16, None)
        context = stage.element_contexts[0]

        isoparametric_rule = context.specialization.quadrature_rule
        affine_rule = context.affine_specialization.quadrature_rule
        self.assertEqual((isoparametric_rule.order, isoparametric_rule.n_qp), (4, 6))
        self.assertEqual((affine_rule.order, affine_rule.n_qp), (3, 4))

    def test_hyperelastic_affine_and_isoparametric_mesh_use_separate_quadrature(self):
        with tempfile.TemporaryDirectory() as out_dir:
            gen.generate(
                neohookean_ogden,
                out_dir,
                elements=("TRI6",),
            )
            source = os.path.join(
                out_dir,
                "d2",
                "tri6",
                "neohookean_ogden_tri6_operator.cpp",
            )
            with open(source, encoding="utf-8") as stream:
                contents = stream.read()

        affine = contents.index(
            "neohookean_ogden_tri6_tri6_gradient_affine_mesh_soa_impl"
        )
        isoparametric = contents.index(
            "neohookean_ogden_tri6_tri6_gradient_isoparametric_mesh_soa_impl"
        )
        self.assertIn("static constexpr int N_QP = 3;", contents[affine:isoparametric])
        self.assertIn("const scalar_t *const affine_grad_ref_x", contents[affine:isoparametric])
        self.assertIn("const scalar_t *const affine_q_weight", contents[affine:isoparametric])
        self.assertIn(
            "neohookean_ogden_tri6_affine_reference_data<scalar_t>::grad_ref_x()",
            contents[affine:isoparametric],
        )
        self.assertIn("static constexpr int N_QP = 6;", contents[isoparametric:])
        self.assertIn("const scalar_t *const isoparametric_grad_ref_x", contents[isoparametric:])
        self.assertIn("const scalar_t *const isoparametric_q_weight", contents[isoparametric:])
        self.assertIn(
            "neohookean_ogden_tri6_isoparametric_reference_data<scalar_t>::grad_ref_x()",
            contents[isoparametric:],
        )

    def test_low_level_specialization_accepts_integration_case(self):
        standard = gen.sfem_soa_element_specialization("HEX27").quadrature_rule
        energy = gen.sfem_soa_element_specialization(
            "HEX27",
            integration_case="energy",
        ).quadrature_rule
        explicit = gen.sfem_soa_element_specialization(
            "HEX27",
            quadrature_order=3,
            integration_case="energy",
        ).quadrature_rule

        self.assertEqual((standard.order, standard.n_qp), (3, 27))
        self.assertEqual((energy.order, energy.n_qp), (4, 64))
        self.assertEqual((explicit.order, explicit.n_qp), (3, 27))

    def test_linear_value_residual_does_not_use_nonlinear_quadrature(self):
        builder = gen.EquationSystemBuilder(2)
        p = builder.scalar_field("p", family="pressure")
        q = gen.test_function(p)
        builder.add_residual("mass", p * q, fields=(p,))
        material = gen.CodeGenerator(
            "linear_mass",
            gen.EquationSystems(builder.build()),
            elements=("TRI3", "QUAD4"),
        )
        stage = gen.UserInputStage.create(material, material.elements, 16, None)
        orders = {
            context.label.upper(): (
                context.specialization.quadrature_rule.order,
                context.specialization.quadrature_rule.n_qp,
            )
            for context in stage.element_contexts
        }
        self.assertEqual(orders["TRI3"], (2, 3))
        self.assertEqual(orders["QUAD4"], (2, 4))

    def test_fem_policy_describes_basis_quadrature_and_field_compatibility(self):
        element = poro_hyperelasticity.elements[2]
        policy = gen.sfem_fem_policy(element)

        self.assertIsInstance(policy, gen.SfemFEMPolicy)
        self.assertEqual(policy.label, "hex27_hex8")
        self.assertEqual(policy.cell_element_type, "HEX27")
        self.assertEqual(policy.family, "tensor_product")
        self.assertEqual(policy.basis.cell, "hexahedron")
        self.assertEqual(policy.basis.degree, 2)
        self.assertEqual(policy.quadrature_rule.order, 4)
        self.assertEqual(policy.quadrature_rule.n_qp, 64)
        self.assertEqual(policy.element_for_family("displacement"), "HEX27")
        self.assertEqual(policy.element_for_family("velocity"), "HEX27")
        self.assertEqual(policy.element_for_family("pressure"), "HEX8")
        self.assertTrue(policy.is_mixed_order)

        builder = gen.EquationSystemBuilder(3)
        u = builder.vector_field("u", family="displacement")
        p = builder.scalar_field("p", family="pressure")
        mapping = {
            field.name: element
            for field, element in policy.field_element_types_for(builder.build().fields)
        }
        self.assertEqual(mapping, {"u": "HEX27", "p": "HEX8"})

        reference_data = {item.name: item.values for item in gen.sfem_reference_data(policy.quadrature_rule)}
        self.assertEqual(set(reference_data), {"shape_1d", "grad_1d", "q_weight_1d"})
        self.assertEqual(len(reference_data["shape_1d"]), 12)
        self.assertEqual(len(reference_data["grad_1d"]), 12)
        self.assertEqual(len(reference_data["q_weight_1d"]), 4)

        hex27_data = {
            item.name: item.values
            for item in gen.sfem_tensor_product_field_reference_data(
                "HEX27",
                policy.quadrature_rule,
                "hex27",
            )
        }
        hex8_data = {
            item.name: item.values
            for item in gen.sfem_tensor_product_field_reference_data(
                "HEX8",
                policy.quadrature_rule,
                "hex8",
            )
        }
        self.assertEqual(set(hex27_data), {"hex27_shape_1d", "hex27_grad_1d"})
        self.assertEqual(set(hex8_data), {"hex8_shape_1d", "hex8_grad_1d"})
        self.assertEqual(len(hex27_data["hex27_shape_1d"]), 12)
        self.assertEqual(len(hex8_data["hex8_shape_1d"]), 8)

        context = gen.ElementGenerationContext.create(
            "poro_hyperelasticity",
            element,
            16,
            None,
        )
        self.assertIs(context.fem_policy.compatible_element, element)
        self.assertEqual(context.family, "tensor_product")
        self.assertTrue(context.is_mixed_order)

    def test_geometry_policy_nodes_describe_affine_and_isoparametric_paths(self):
        for element, expected_dim, tensor_product in (
            ("TRI3", 2, False),
            ("TET4", 3, False),
            ("QUAD4", 2, True),
            ("HEX8", 3, True),
        ):
            context = gen.ElementGenerationContext.create(
                "test_material",
                element,
                16,
                None,
            )
            affine = context.geometry_plan(gen.GeometryMode.AFFINE)
            iso = context.geometry_plan(gen.GeometryMode.ISOPARAMETRIC)

            self.assertTrue(affine.is_affine)
            self.assertEqual(affine.element_type, element)
            self.assertEqual(affine.dim, expected_dim)
            self.assertEqual(affine.input_layout, gen.GeometryInputLayout.ADJUGATE_DETERMINANT_SOA)
            self.assertEqual(affine.evaluation, gen.GeometryEvaluation.ROUTE_PRECOMPUTED_AFFINE)
            self.assertEqual(affine.jacobian_scope, "element")
            self.assertEqual(affine.geometry_points_per_element, 1)
            self.assertEqual(affine.geometry_stream_count, expected_dim * expected_dim + 1)
            self.assertTrue(affine.requires_adjugate_determinant_streams)
            self.assertFalse(affine.uses_sum_factorization)

            self.assertTrue(iso.is_isoparametric)
            self.assertEqual(iso.element_type, element)
            self.assertEqual(iso.dim, expected_dim)
            self.assertEqual(iso.input_layout, gen.GeometryInputLayout.COORDINATE_AOS)
            self.assertEqual(iso.jacobian_scope, "quadrature_point")
            self.assertEqual(iso.geometry_points_per_element, context.specialization.n_qp)
            self.assertEqual(iso.geometry_stream_count, expected_dim * expected_dim + 1)
            self.assertTrue(iso.requires_coordinates)
            if tensor_product:
                self.assertEqual(iso.evaluation, gen.GeometryEvaluation.TENSOR_PRODUCT_SUM_FACTOR)
                self.assertTrue(iso.uses_sum_factorization)
                self.assertIsInstance(iso.sum_factorization_plan, gen.TensorProductSumFactorizationPlan)
                self.assertTrue(iso.sum_factorization_plan.evaluates_geometry_jacobian)
                self.assertEqual(
                    iso.sum_factorization_plan.operations,
                    (gen.TensorProductOperation.GEOMETRY_JACOBIAN,),
                )
            else:
                self.assertEqual(iso.evaluation, gen.GeometryEvaluation.SIMPLEX_REFERENCE)
                self.assertFalse(iso.uses_sum_factorization)
                self.assertIsNone(iso.sum_factorization_plan)

        mixed_context = gen.ElementGenerationContext.create(
            "test_material",
            poro_hyperelasticity.elements[2],
            16,
            None,
        )
        mixed_iso = mixed_context.geometry_plan("isoparametric")
        self.assertEqual(mixed_iso.element_type, "HEX27")
        self.assertEqual(mixed_iso.n_shape, 27)
        self.assertTrue(mixed_iso.uses_sum_factorization)

    def test_basis_policy_nodes_describe_simplex_tensor_and_mixed_fields(self):
        tri_context = gen.ElementGenerationContext.create(
            "test_material",
            "TRI3",
            16,
            None,
        )
        tri_basis = tri_context.basis_plan()
        self.assertIsInstance(tri_basis, gen.BasisPlanNode)
        self.assertEqual(tri_basis.family, gen.BasisFamily.SIMPLEX)
        self.assertEqual(tri_basis.evaluation, gen.BasisEvaluation.DIRECT_REFERENCE)
        self.assertEqual(tri_basis.data_layout, gen.BasisDataLayout.QP_SHAPE)
        self.assertEqual(tri_basis.n_shape, 3)
        self.assertEqual(tri_basis.n_qp, 1)
        self.assertEqual(tri_basis.reference_shape_size, 3)
        self.assertEqual(tri_basis.reference_gradient_size, 1 * 3 * 2)
        self.assertEqual(tri_basis.scatter_n_shape, 3)
        self.assertFalse(tri_basis.uses_sum_factorization)

        hex_context = gen.ElementGenerationContext.create(
            "test_material",
            "HEX27",
            16,
            None,
        )
        hex_basis = hex_context.basis_plan("cell")
        self.assertEqual(hex_basis.family, gen.BasisFamily.TENSOR_PRODUCT)
        self.assertEqual(hex_basis.evaluation, gen.BasisEvaluation.TENSOR_PRODUCT_SUM_FACTOR)
        self.assertEqual(hex_basis.data_layout, gen.BasisDataLayout.TENSOR_PRODUCT_1D)
        self.assertEqual(hex_basis.n_shape, 27)
        self.assertEqual(hex_basis.n_qp, 27)
        self.assertEqual(hex_basis.n_shape_1d, 3)
        self.assertEqual(hex_basis.n_qp_1d, 3)
        self.assertEqual(hex_basis.reference_shape_size, 9)
        self.assertEqual(hex_basis.reference_gradient_size, 9)
        self.assertTrue(hex_basis.uses_sum_factorization)
        self.assertEqual(len(hex_basis.sum_factorization_plans), 2)
        self.assertTrue(hex_basis.field_evaluation_sum_factorization.evaluates_values)
        self.assertTrue(hex_basis.field_evaluation_sum_factorization.evaluates_gradients)
        self.assertTrue(hex_basis.test_contraction_sum_factorization.contracts_tests)
        self.assertTrue(hex_basis.test_contraction_sum_factorization.uses_1d_basis)

        mixed_context = gen.ElementGenerationContext.create(
            "test_material",
            poro_hyperelasticity.elements[2],
            16,
            None,
        )
        builder = gen.EquationSystemBuilder(3)
        u = builder.vector_field("u", family="displacement")
        p = builder.scalar_field("p", family="pressure")
        displacement_basis, pressure_basis = mixed_context.field_basis_plans((u, p))
        self.assertEqual(displacement_basis.role, "field:u")
        self.assertEqual(displacement_basis.element_type, "HEX27")
        self.assertEqual(displacement_basis.cell_element_type, "HEX27")
        self.assertEqual(displacement_basis.n_shape, 27)
        self.assertEqual(displacement_basis.n_qp, 64)
        self.assertEqual(displacement_basis.n_qp_1d, 4)
        self.assertTrue(displacement_basis.uses_sum_factorization)
        self.assertEqual(pressure_basis.role, "field:p")
        self.assertEqual(pressure_basis.element_type, "HEX8")
        self.assertEqual(pressure_basis.cell_element_type, "HEX27")
        self.assertEqual(pressure_basis.n_shape, 8)
        self.assertEqual(pressure_basis.n_qp, 64)
        self.assertEqual(pressure_basis.n_shape_1d, 2)
        self.assertEqual(pressure_basis.n_qp_1d, 4)
        self.assertEqual(pressure_basis.scatter_n_shape, 8)
        self.assertTrue(pressure_basis.uses_sum_factorization)
        self.assertEqual(
            pressure_basis.field_evaluation_sum_factorization.operations,
            (
                gen.TensorProductOperation.FIELD_VALUE,
                gen.TensorProductOperation.FIELD_GRADIENT,
            ),
        )
        self.assertEqual(
            pressure_basis.test_contraction_sum_factorization.operations,
            (
                gen.TensorProductOperation.TEST_VALUE_CONTRACTION,
                gen.TensorProductOperation.TEST_GRADIENT_CONTRACTION,
            ),
        )

        simplex_mixed_context = gen.ElementGenerationContext.create(
            "test_material",
            poro_hyperelasticity.elements[0],
            16,
            None,
        )
        pressure_basis = simplex_mixed_context.field_basis_plan(p)
        self.assertEqual(pressure_basis.element_type, "TRI3")
        self.assertEqual(pressure_basis.cell_element_type, "TRI6")
        self.assertEqual(pressure_basis.family, gen.BasisFamily.SIMPLEX)
        self.assertEqual(pressure_basis.n_shape, 3)
        self.assertEqual(pressure_basis.n_qp, 6)
        self.assertEqual(pressure_basis.reference_shape_size, 6 * 3)
        self.assertEqual(pressure_basis.reference_gradient_size, 6 * 3 * 2)

    def test_shared_tensor_product_geometry_emitters_cover_residual_and_hyperelastic_paths(self):
        coordinate_streams = tensor_product_ordered_coordinate_streams(
            3,
            8,
            tuple(range(24)),
            lambda stream: "block_coordinates[%d]" % stream,
        )
        self.assertEqual(coordinate_streams[:6], ("block_coordinates[0]", "block_coordinates[1]", "block_coordinates[2]", "block_coordinates[3]", "block_coordinates[4]", "block_coordinates[5]"))
        self.assertEqual(coordinate_streams[6:12], ("block_coordinates[9]", "block_coordinates[10]", "block_coordinates[11]", "block_coordinates[6]", "block_coordinates[7]", "block_coordinates[8]"))

        residual_lines = "\n".join(
            tensor_product_evaluated_isoparametric_geometry_lines(
                dim=3,
                n_shape=8,
                n_qp=8,
                local_prefix="residual",
                coordinate_streams=coordinate_streams,
                adjugate_target=lambda component, index: "adj%d[%s]" % (component, index),
                determinant_target=lambda index: "det[%s]" % index,
            )
        )
        self.assertIn("tensor_evaluate<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, DIM>", residual_lines)
        self.assertIn("scalar_t coordinate_value[DIM * N_QP * VECTOR_SIZE];", residual_lines)
        self.assertIn("adj0[q * VECTOR_SIZE + lane] = J11 * J22 - J12 * J21;", residual_lines)
        self.assertIn("det[q * VECTOR_SIZE + lane] = J00 * (J11 * J22 - J12 * J21)", residual_lines)

        hyper_lines = "\n".join(
            tensor_product_gradient_isoparametric_geometry_lines(
                dim=3,
                n_shape=8,
                n_qp=8,
                local_prefix="hyper",
                coordinate_streams=coordinate_streams,
                adjugate_target=lambda component, index: "adj%d[%s]" % (component, index),
                determinant_target=lambda index: "det[%s]" % index,
            )
        )
        self.assertIn("tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>", hyper_lines)
        self.assertIn("coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE", hyper_lines)
        self.assertNotIn("coordinate_value", hyper_lines)

    def test_generation_plan_schema_validates_kernel_blocks_and_streams(self):
        context = gen.ElementGenerationContext.create("test_material", "TRI3", 16, None)
        user_input = gen.UserInputStage.create(neohookean_ogden, ("TRI3",), 16, None)
        form_evaluation = gen._evaluate_forms(user_input)
        form_collection = form_evaluation.by_dim[2].units[0].form_evaluation
        stream = gen.DataStreamPlan(
            "u",
            gen.DataStreamRole.FIELD,
            gen.DataStreamLayout.SOA,
            components=2,
            n_items=context.basis_plan().n_shape,
        )
        geometry = gen.GeometryPlan(context.geometry_plan("affine"), (stream,))
        block = gen.BlockPlan(
            "u_u",
            "u",
            "u",
            gen.FormOrder.TWO,
            (
                gen.LocalPhase.EVALUATE_TRIAL,
                gen.LocalPhase.EVALUATE_MATERIAL,
                gen.LocalPhase.CONTRACT_TEST,
            ),
            (stream,),
            (context.basis_plan(),),
        )
        kernel = gen.KernelPlan(
            "test_material_solid",
            "energy",
            form_collection,
            2,
            (
                gen.MeshPhase.GEOMETRY,
                gen.MeshPhase.LOCAL_CALL,
                gen.MeshPhase.SCATTER,
            ),
            geometry,
            (block,),
            (stream,),
        )
        plan = gen.GenerationPlan((kernel,))

        self.assertEqual(plan.stage, gen.PipelineStage.SPECIALIZED_FORM_MANIPULATION)
        self.assertEqual(plan.units_for_context(context), (kernel,))
        self.assertEqual(plan.monolithic_kernels, (kernel,))
        self.assertEqual(plan.block_kernels, ())
        self.assertTrue(block.is_diagonal)
        self.assertEqual(geometry.mode, gen.GeometryMode.AFFINE)
        self.assertTrue(all(isinstance(phase, gen.MeshPhasePlan) for phase in kernel.mesh_phase_plans))
        self.assertEqual(
            tuple(phase.phase for phase in kernel.mesh_phase_plans),
            (
                gen.MeshPhase.GEOMETRY,
                gen.MeshPhase.LOCAL_CALL,
                gen.MeshPhase.SCATTER,
            ),
        )
        self.assertIs(kernel.mesh_phase_plans[0].geometry, geometry)
        self.assertEqual(kernel.mesh_phase_plans[1].blocks, (block,))
        self.assertEqual(kernel.mesh_phase_plans[2].streams, (stream,))
        self.assertTrue(all(isinstance(phase, gen.LocalPhasePlan) for phase in block.local_phase_plans))
        self.assertEqual(
            tuple(phase.phase for phase in block.local_phase_plans),
            (
                gen.LocalPhase.EVALUATE_TRIAL,
                gen.LocalPhase.EVALUATE_MATERIAL,
                gen.LocalPhase.CONTRACT_TEST,
            ),
        )
        plan.validate_for_context(context)
        self.assertEqual(plan.emission_kernels_for_context(context), (kernel,))

    def test_codegen_stage_uses_shared_openmp_soa_backend(self):
        self.assertIsInstance(gen.OPENMP_SOA_BACKEND, gen.OpenMPSoABackend)

    def test_openmp_backend_consumes_kernel_phase_plan(self):
        user_input = gen.UserInputStage.create(neohookean_ogden, ("TRI3",), 16, None)
        plan = gen.SpecializedFormManipulationStage(
            user_input,
            gen._evaluate_forms(user_input),
        ).run()
        unit = plan.units[0]
        context = user_input.element_contexts[0]
        invalid_unit = gen.CodeGenerationUnit(
            name=unit.name,
            kind=unit.kind,
            form_collection=unit.form_collection,
            dim=unit.dim,
            mesh_phase_plans=(
                gen.MeshPhasePlan(gen.MeshPhase.GEOMETRY),
                gen.MeshPhasePlan(gen.MeshPhase.LOCAL_CALL),
            ),
            target=unit.target,
            payload=unit.payload,
            coupling=unit.coupling,
            material_name=unit.material_name,
            unit_name=unit.unit_name,
        )
        with self.assertRaisesRegex(ValueError, "mesh phase plan"):
            gen.OPENMP_SOA_BACKEND.emit(invalid_unit, context)

    def test_specialized_form_manipulation_returns_generation_plan(self):
        energy_input = gen.UserInputStage.create(neohookean_ogden, ("TRI3",), 16, None)
        energy_plan = gen.SpecializedFormManipulationStage(
            energy_input,
            gen._evaluate_forms(energy_input),
        ).run()
        self.assertIsInstance(energy_plan, gen.GenerationPlan)
        self.assertTrue(all(isinstance(unit, gen.KernelPlan) for unit in energy_plan.units))
        self.assertEqual(
            energy_plan.units[0].mesh_phases,
            (
                gen.MeshPhase.GEOMETRY,
                gen.MeshPhase.LOCAL_CALL,
                gen.MeshPhase.SCATTER,
            ),
        )
        self.assertTrue(all(isinstance(phase, gen.MeshPhasePlan) for phase in energy_plan.units[0].mesh_phase_plans))
        self.assertTrue(energy_plan.units[0].mesh_phase_plans[0].is_geometry)
        self.assertTrue(energy_plan.units[0].mesh_phase_plans[1].is_local_call)
        self.assertTrue(energy_plan.units[0].mesh_phase_plans[2].is_scatter)

        residual_input = gen.UserInputStage.create(two_phase_flow, ("TRI3",), 16, None)
        residual_plan = gen.SpecializedFormManipulationStage(
            residual_input,
            gen._evaluate_forms(residual_input),
        ).run()
        self.assertIsInstance(residual_plan, gen.GenerationPlan)
        self.assertEqual(
            residual_plan.units[0].mesh_phases,
            (
                gen.MeshPhase.GATHER,
                gen.MeshPhase.GEOMETRY,
                gen.MeshPhase.LOCAL_CALL,
                gen.MeshPhase.SCATTER,
            ),
        )
        self.assertTrue(residual_plan.units[0].mesh_phase_plans[0].is_gather)
        self.assertTrue(residual_plan.units[0].mesh_phase_plans[1].is_geometry)
        self.assertTrue(residual_plan.units[0].mesh_phase_plans[2].is_local_call)
        self.assertTrue(residual_plan.units[0].mesh_phase_plans[3].is_scatter)
        self.assertTrue(residual_plan.units[0].is_monolithic)
        self.assertTrue(residual_plan.units[0].is_complete_system)
        self.assertEqual(residual_plan.monolithic_kernels, residual_plan.units)
        self.assertEqual(residual_plan.complete_system_kernels, residual_plan.units)
        self.assertEqual(len(residual_plan.units[0].blocks), 6)
        self.assertEqual(len(residual_plan.units[0].block_kernels), 6)
        self.assertEqual(residual_plan.block_kernels, residual_plan.units[0].block_kernels)
        self.assertTrue(all(kernel.is_block for kernel in residual_plan.block_kernels))
        self.assertTrue(all(kernel.coupling is gen.KernelCoupling.BLOCK for kernel in residual_plan.block_kernels))
        self.assertEqual(
            tuple(kernel.block.name for kernel in residual_plan.block_kernels),
            (
                "form_1_p_w",
                "form_1_p_c",
                "form_2_p_w_p_w",
                "form_2_p_w_p_c",
                "form_2_p_c_p_w",
                "form_2_p_c_p_c",
            ),
        )
        self.assertEqual(
            tuple(kernel.mesh_phase_plans[2].blocks for kernel in residual_plan.block_kernels),
            tuple((kernel.block,) for kernel in residual_plan.block_kernels),
        )
        self.assertEqual(
            tuple(phase.phase for phase in residual_plan.block_kernels[0].block.local_phase_plans),
            (
                gen.LocalPhase.EVALUATE_TRIAL,
                gen.LocalPhase.TRANSFORM_REFERENCE,
                gen.LocalPhase.EVALUATE_MATERIAL,
                gen.LocalPhase.CONTRACT_TEST,
            ),
        )
        self.assertEqual(
            tuple(kernel.emission for kernel in residual_plan.block_kernels),
            (gen.KernelEmission.FILES,) * len(residual_plan.block_kernels),
        )
        self.assertEqual(
            residual_plan.emission_kernels_for_context(residual_input.element_contexts[0]),
            residual_plan.units + residual_plan.block_kernels,
        )

    def test_generation_plan_validation_rejects_unsupported_combinations(self):
        context = gen.ElementGenerationContext.create("test_material", "TRI3", 16, None)
        user_input = gen.UserInputStage.create(neohookean_ogden, ("TRI3",), 16, None)
        form_collection = gen._evaluate_forms(user_input).by_dim[2].units[0].form_evaluation

        with self.assertRaisesRegex(ValueError, "target 'cuda' is not supported"):
            gen.GenerationPlan((
                gen.KernelPlan(
                    "cuda_kernel",
                    "energy",
                    form_collection,
                    2,
                    (gen.MeshPhase.LOCAL_CALL,),
                    target=gen.KernelTarget.CUDA,
                ),
            )).validate_for_context(context)

        with self.assertRaisesRegex(ValueError, "mesh phases are not in canonical order"):
            gen.GenerationPlan((
                gen.KernelPlan(
                    "bad_order",
                    "energy",
                    form_collection,
                    2,
                    (
                        gen.MeshPhase.SCATTER,
                        gen.MeshPhase.LOCAL_CALL,
                    ),
                ),
            )).validate_for_context(context)

        bad_block = gen.BlockPlan(
            "bad_field",
            "missing",
            "",
            gen.FormOrder.ONE,
            (gen.LocalPhase.EVALUATE_TRIAL,),
        )
        with self.assertRaisesRegex(ValueError, "row field 'missing'"):
            gen.GenerationPlan((
                gen.KernelPlan(
                    "bad_block",
                    "energy",
                    form_collection,
                    2,
                    (gen.MeshPhase.LOCAL_CALL,),
                    blocks=(bad_block,),
                ),
            )).validate_for_context(context)

    def test_local_kernel_plan_names_by_dimension_and_family(self):
        simplex = gen.LocalKernelPlan("generated_material", 2, "simplex")
        tensor_product = gen.LocalKernelPlan("generated_material", 3, "tensor_product")
        mixed = gen.LocalKernelPlan("generated_material", 3, "tensor_product", "_mixed")

        self.assertEqual(simplex.name, "generated_material_d2_simplex")
        self.assertEqual(simplex.header, "generated_material_d2_simplex_local.hpp")
        self.assertEqual(tensor_product.name, "generated_material_d3_tensor_product")
        self.assertEqual(mixed.name, "generated_material_d3_tensor_product_mixed")
        self.assertEqual(mixed.header, "generated_material_d3_tensor_product_mixed_local.hpp")
        with self.assertRaisesRegex(ValueError, "unsupported local kernel family"):
            gen.LocalKernelPlan("generated_material", 2, "hex8")

    def test_mesh_kernel_plan_names_by_element_or_compatible_element(self):
        tri3 = gen.MeshKernelPlan("generated_material", "TRI3")
        taylor_hood = gen.MeshKernelPlan("generated_material", "TRI6_TRI3")

        self.assertEqual(tri3.name, "generated_material_tri3")
        self.assertEqual(tri3.source, "generated_material_tri3_operator.cpp")
        self.assertEqual(taylor_hood.name, "generated_material_tri6_tri3")
        self.assertEqual(taylor_hood.source, "generated_material_tri6_tri3_operator.cpp")
        with self.assertRaisesRegex(ValueError, "element label"):
            gen.MeshKernelPlan("generated_material", "tri6-tri3")

    def test_generation_plan_dumps_json_for_inspection(self):
        residual_input = gen.UserInputStage.create(two_phase_flow, ("TRI3",), 16, None)
        residual_plan = gen.SpecializedFormManipulationStage(
            residual_input,
            gen._evaluate_forms(residual_input),
        ).run()
        dump = residual_plan.to_dict()
        parsed = json.loads(residual_plan.to_json())

        self.assertEqual(dump["stage"], gen.PipelineStage.SPECIALIZED_FORM_MANIPULATION.value)
        self.assertEqual(parsed["n_monolithic_kernels"], 1)
        self.assertEqual(parsed["n_block_kernels"], 6)
        self.assertEqual(parsed["n_complete_system_kernels"], 1)
        self.assertEqual(parsed["kernels"][0]["scope"], gen.KernelScope.MONOLITHIC.value)
        self.assertEqual(parsed["kernels"][0]["coupling"], gen.KernelCoupling.COMPLETE_SYSTEM.value)
        self.assertEqual(parsed["kernels"][0]["mesh_phases"], ["gather", "geometry", "local_call", "scatter"])
        self.assertEqual(
            tuple(block["name"] for block in parsed["kernels"][0]["blocks"]),
            (
                "form_1_p_w",
                "form_1_p_c",
                "form_2_p_w_p_w",
                "form_2_p_w_p_c",
                "form_2_p_c_p_w",
                "form_2_p_c_p_c",
            ),
        )
        self.assertEqual(
            tuple(kernel["selected_block"] for kernel in parsed["kernels"][0]["block_kernels"]),
            (
                "form_1_p_w",
                "form_1_p_c",
                "form_2_p_w_p_w",
                "form_2_p_w_p_c",
                "form_2_p_c_p_w",
                "form_2_p_c_p_c",
            ),
        )

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
                elements=("TRI6",),
            )
            names = _relative_sources(result, out_dir)
            self.assertIn(
                "d2/tri6/poro_hyperelasticity_solid_tri6_operator.cpp",
                names,
            )
            self.assertIn(
                "d2/tri6_tri3/poro_hyperelasticity_poro_tri6_tri3_operator.cpp",
                names,
            )

    def test_generates_taylor_hood_stokes_material(self):
        with tempfile.TemporaryDirectory() as out_dir:
            result = gen.generate(
                stokes,
                out_dir,
                elements=("TRI6_TRI3",),
            )
            names = _relative_sources(result, out_dir)
            self.assertIn("d2/tri6_tri3/stokes_tri6_tri3_operator.cpp", names)
            self.assertIn("d2/stokes_d2_simplex_mixed_local.hpp", names)
            self.assertIn("kernel_diagnostics.hpp", names)
            self.assertIsInstance(result.plan, gen.GenerationPlan)
            self.assertEqual(result.plan.units[0].name, "stokes")
            self.assertTrue(result.plan.units[0].is_complete_system)
            self.assertEqual(result.plan.complete_system_kernels, result.plan.units)
            self.assertEqual(
                tuple(block.name for block in result.plan.units[0].blocks),
                ("form_1_u", "form_1_p", "form_2_u_u", "form_2_u_p", "form_2_p_u"),
            )
            source = os.path.join(
                out_dir,
                "d2",
                "tri6_tri3",
                "stokes_tri6_tri3_operator.cpp",
            )
            with open(source) as input_file:
                contents = input_file.read()
            self.assertIn('#include "../stokes_d2_simplex_mixed_local.hpp"', contents)
            self.assertIn("stokes_tri6_tri3_residual_isoparametric_mesh_soa", contents)
            self.assertIn("stokes_tri6_tri3_jacobian_action_isoparametric_mesh_soa", contents)
            self.assertIn(
                "d2/tri6_tri3/stokes_form_2_u_p_tri6_tri3_operator.cpp",
                names,
            )
            self.assertIn(
                "d2/tri6/stokes_form_2_u_u_tri6_operator.cpp",
                names,
            )
            self.assertNotIn(
                "d2/tri6_tri3/stokes_form_2_u0_p_tri6_tri3_operator.cpp",
                names,
            )
            self.assertNotIn(
                "d2/tri6_tri3/stokes_form_2_u_u_tri6_tri3_operator.cpp",
                names,
            )
            self.assertNotIn(
                "d2/tri6_tri3/stokes_form_2_p_p_tri6_tri3_operator.cpp",
                names,
            )
            for field in result.plan.units[0].form_collection.fields:
                for component in range(int(field.components)):
                    name = "%s%d" % (field.name, component) if int(field.components) > 1 else field.name
                    self.assertIn("%s_out" % name, contents)
            validate_stokes_m6_4(result)

    def test_stokes_validation_handles_multiple_dimensions(self):
        with tempfile.TemporaryDirectory() as out_dir:
            result = gen.generate(
                stokes,
                out_dir,
                elements=("TRI6_TRI3", "TET10_TET4"),
            )
            validate_stokes_m6_4(result)

    def test_stokes_tensor_product_local_uses_sum_factorization(self):
        with tempfile.TemporaryDirectory() as out_dir:
            gen.generate(
                stokes,
                out_dir,
                elements=("HEX27_HEX8",),
                compile=True,
            )
            local = os.path.join(
                out_dir,
                "d3",
                "stokes_d3_tensor_product_mixed_local.hpp",
            )
            with open(local) as input_file:
                contents = input_file.read()
            self.assertIn("field_shape_1d", contents)
            self.assertIn("field_grad_1d", contents)
            self.assertIn("tensor_evaluate<", contents)
            self.assertIn("tensor_integrate<", contents)
            self.assertIn("N_QP_1D", contents)
            self.assertNotIn("field_shape[", contents)
            self.assertNotIn("field_grad_ref[", contents)
            self.assertNotIn("for (int trial", contents)
            self.assertNotIn("for (int test", contents)

    def test_stokes_tensor_product_operator_deduplicates_reference_data(self):
        with tempfile.TemporaryDirectory() as out_dir:
            gen.generate(
                stokes,
                out_dir,
                elements=("HEX27_HEX8",),
            )
            operator = os.path.join(
                out_dir,
                "d3",
                "hex27_hex8",
                "stokes_hex27_hex8_operator.cpp",
            )
            with open(operator) as input_file:
                contents = input_file.read()
            self.assertIn("struct stokes_isoparametric_reference_data", contents)
            self.assertIn("static const scalar_t *hex27_shape_1d()", contents)
            self.assertIn("static const scalar_t *hex27_grad_1d()", contents)
            self.assertIn("static const scalar_t *hex8_shape_1d()", contents)
            self.assertIn("static const scalar_t *hex8_grad_1d()", contents)
            self.assertIn("static const scalar_t data[", contents)
            self.assertIn("scalar_t(", contents)
            self.assertIn(
                "field_shape_1d[N_FIELDS] = {sfem::codegen::stokes_isoparametric_reference_data<scalar_t>::hex27_shape_1d(), "
                "sfem::codegen::stokes_isoparametric_reference_data<scalar_t>::hex27_shape_1d(), "
                "sfem::codegen::stokes_isoparametric_reference_data<scalar_t>::hex27_shape_1d(), "
                "sfem::codegen::stokes_isoparametric_reference_data<scalar_t>::hex8_shape_1d()}",
                contents,
            )
            self.assertNotIn("stokes_reference_data", contents)
            self.assertNotIn("stokes_cell_grad_ref", contents)
            self.assertNotIn("stokes_u0_shape_", contents)
            self.assertNotIn("stokes_u1_shape_", contents)
            self.assertNotIn("stokes_u2_shape_", contents)
            self.assertNotIn("stokes_u0_grad_ref", contents)
            self.assertNotIn("static const scalar_t shape_1d[", contents)
            self.assertNotIn("static const scalar_t grad_1d[", contents)

    def test_stokes_simplex_operator_deduplicates_reference_data(self):
        with tempfile.TemporaryDirectory() as out_dir:
            gen.generate(
                stokes,
                out_dir,
                elements=("TRI6_TRI3", "TET10_TET4"),
            )
            tri_source = os.path.join(
                out_dir,
                "d2",
                "tri6_tri3",
                "stokes_tri6_tri3_operator.cpp",
            )
            tet_source = os.path.join(
                out_dir,
                "d3",
                "tet10_tet4",
                "stokes_tet10_tet4_operator.cpp",
            )
            with open(tri_source) as input_file:
                tri = input_file.read()
            with open(tet_source) as input_file:
                tet = input_file.read()
            with open(
                os.path.join(
                    out_dir,
                    "d2",
                    "stokes_d2_simplex_mixed_local.hpp",
                )
            ) as input_file:
                tri_local = input_file.read()
            with open(
                os.path.join(
                    out_dir,
                    "d3",
                    "stokes_d3_simplex_mixed_local.hpp",
                )
            ) as input_file:
                tet_local = input_file.read()

            self.assertIn("struct stokes_isoparametric_reference_data", tri)
            self.assertIn("static const scalar_t *tri6_shape()", tri)
            self.assertIn("static const scalar_t *tri6_grad_ref_x()", tri)
            self.assertIn("static const scalar_t *tri3_shape()", tri)
            self.assertIn("static const scalar_t *tri3_grad_ref_y()", tri)
            self.assertIn("static const scalar_t data[", tri)
            self.assertIn("scalar_t(", tri)
            self.assertIn(
                "field_shape[N_FIELDS] = {sfem::codegen::stokes_isoparametric_reference_data<scalar_t>::tri6_shape(), "
                "sfem::codegen::stokes_isoparametric_reference_data<scalar_t>::tri6_shape(), "
                "sfem::codegen::stokes_isoparametric_reference_data<scalar_t>::tri3_shape()}",
                tri,
            )
            self.assertIn(
                "isoparametric_cell_grad_ref_0 = sfem::codegen::stokes_isoparametric_reference_data<scalar_t>::tri6_grad_ref_x()",
                tri,
            )

            self.assertIn("struct stokes_isoparametric_reference_data", tet)
            self.assertIn("static const scalar_t *tet10_shape()", tet)
            self.assertIn("static const scalar_t *tet10_grad_ref_z()", tet)
            self.assertIn("static const scalar_t *tet4_shape()", tet)
            self.assertIn("static const scalar_t *tet4_grad_ref_z()", tet)
            self.assertIn("static const scalar_t data[", tet)
            self.assertIn("scalar_t(", tet)
            self.assertIn(
                "field_shape[N_FIELDS] = {sfem::codegen::stokes_isoparametric_reference_data<scalar_t>::tet10_shape(), "
                "sfem::codegen::stokes_isoparametric_reference_data<scalar_t>::tet10_shape(), "
                "sfem::codegen::stokes_isoparametric_reference_data<scalar_t>::tet10_shape(), "
                "sfem::codegen::stokes_isoparametric_reference_data<scalar_t>::tet4_shape()}",
                tet,
            )
            self.assertIn(
                "isoparametric_cell_grad_ref_2 = sfem::codegen::stokes_isoparametric_reference_data<scalar_t>::tet10_grad_ref_z()",
                tet,
            )

            for contents in (tri, tet):
                self.assertNotIn("stokes_reference_data", contents)
                self.assertNotIn("stokes_cell_grad_ref", contents)
                self.assertNotIn("stokes_u0_shape_", contents)
                self.assertNotIn("stokes_u1_shape_", contents)
                self.assertNotIn("stokes_u2_shape_", contents)
                self.assertNotIn("stokes_u0_grad_ref", contents)
                self.assertNotIn("stokes_u1_grad_ref", contents)
                self.assertNotIn("stokes_u2_grad_ref", contents)
            for contents in (tri_local, tet_local):
                self.assertNotIn("for (int trial", contents)
                self.assertNotIn("for (int test", contents)

    def test_compiles_taylor_hood_stokes_operator(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("c++ compiler is not available")
        with tempfile.TemporaryDirectory() as out_dir:
            gen.generate(
                stokes,
                out_dir,
                elements=("TRI6_TRI3",),
            )
            source = os.path.join(
                out_dir,
                "d2",
                "tri6_tri3",
                "stokes_tri6_tri3_operator.cpp",
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
                    os.path.join(out_dir, "stokes.o"),
                ],
                check=True,
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
                "d2",
                "tri6_tri3",
                "poro_hyperelasticity_poro_tri6_tri3_operator.cpp",
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
