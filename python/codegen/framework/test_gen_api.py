import os
import shutil
import subprocess
import tempfile
import unittest

import sympy as sp

from sfem import gen

from .materials.neohookean_ogden import material as neohookean_ogden
from .materials.poro_hyperelasticity import material as poro_hyperelasticity
from .materials.two_phase_flow import material as two_phase_flow


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
