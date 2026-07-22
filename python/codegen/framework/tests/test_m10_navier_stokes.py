import os
import shutil
import tempfile
import unittest
from pathlib import Path

import sympy as sp

from sfem import gen

from codegen.framework.materials.navier_stokes import material as navier_stokes


def _element_name(element):
    return getattr(element, "name", str(element).upper())


def _selected_element(material, element_name):
    by_name = {_element_name(element): element for element in material.elements}
    return by_name[str(element_name).upper()]


def _generation_plan(material, element_name):
    stage = gen.UserInputStage.create(
        material,
        (_selected_element(material, element_name),),
        gen.DEFAULT_VECTOR_SIZE,
        None,
    )
    plan = gen.SpecializedFormManipulationStage(stage, gen._evaluate_forms(stage)).run()
    return stage, plan


def _expression_plan(unit, order):
    return next(plan for plan in unit.expression_plans if plan.form_order is order)


def _coefficients_by_row(expression_plan):
    return {coefficient.row_field: coefficient for coefficient in expression_plan.coefficients}


def _symbol(name):
    return sp.Symbol(name)


def _assert_sympy_equal(test_case, actual, expected):
    test_case.assertEqual(sp.simplify(actual - expected), sp.S.Zero)


def _navier_stokes_reference_coefficients(dim, direction=False):
    rho = _symbol("rho")
    nu = _symbol("nu")
    dt = _symbol("dt")
    convection_scale = _symbol("convection_scale")
    p = _symbol("p_direction" if direction else "p")
    grad_suffix = "_direction_grad" if direction else "_grad"
    values = {}
    for row in range(dim):
        convection = sum(
            _symbol("u%d_old" % col) * _symbol("u%d%s_%d" % (row, grad_suffix, col))
            for col in range(dim)
        )
        if direction:
            value = rho * _symbol("u%d_direction" % row) / dt + rho * convection_scale * convection
        else:
            value = rho * (
                (_symbol("u%d" % row) - _symbol("u%d_old" % row)) / dt
                + convection_scale * convection
                - _symbol("f%d" % row)
            )
        gradient = []
        for col in range(dim):
            component = nu * rho * (
                _symbol("u%d%s_%d" % (row, grad_suffix, col))
                + _symbol("u%d%s_%d" % (col, grad_suffix, row))
            )
            if row == col:
                component -= p
            gradient.append(component)
        values["u%d" % row] = (value, tuple(gradient))
    values["p"] = (
        sum(_symbol("u%d%s_%d" % (d, grad_suffix, d)) for d in range(dim)),
        tuple(sp.S.Zero for _ in range(dim)),
    )
    return values


def _assert_coefficients(test_case, expression_plan, expected):
    actual_by_row = _coefficients_by_row(expression_plan)
    test_case.assertEqual(set(actual_by_row), set(expected))
    for row, (expected_value, expected_gradient) in expected.items():
        coefficient = actual_by_row[row]
        _assert_sympy_equal(test_case, coefficient.value, expected_value)
        test_case.assertEqual(len(coefficient.gradient), len(expected_gradient))
        for actual, expected_component in zip(coefficient.gradient, expected_gradient):
            _assert_sympy_equal(test_case, actual, expected_component)


class M10NavierStokesApplicationTest(unittest.TestCase):
    def test_incompressible_navier_stokes_material_matches_hardcoded_reference(self):
        for element in ("TRI6_TRI3", "TET10_TET4"):
            with self.subTest(element=element):
                stage, plan = _generation_plan(navier_stokes, element)
                context = stage.element_contexts[0]
                unit = next(
                    unit
                    for unit in plan.emission_kernels_for_context(context)
                    if unit.name == "navier_stokes"
                )
                dim = context.specialization.dim
                _assert_coefficients(
                    self,
                    _expression_plan(unit, gen.FormOrder.ONE),
                    _navier_stokes_reference_coefficients(dim, direction=False),
                )
                _assert_coefficients(
                    self,
                    _expression_plan(unit, gen.FormOrder.TWO),
                    _navier_stokes_reference_coefficients(dim, direction=True),
                )

    def test_generated_navier_stokes_operator_compiles(self):
        if not (shutil.which("mpic++") or shutil.which("mpicxx") or shutil.which("c++")):
            self.skipTest("C++ compiler is not available")
        with tempfile.TemporaryDirectory() as out_dir:
            result = gen.generate(
                navier_stokes,
                out_dir,
                elements=("TRI6_TRI3",),
                compile=True,
                clean=True,
                dump_plan=True,
            )
            names = {os.path.relpath(path, out_dir) for path in result.sources}
            self.assertTrue(result.objects)
            self.assertIn("op/sfem_GeneratedNavierStokes.cpp", names)
            self.assertIn("op/sfem_GeneratedNavierStokes_manifest.json", names)
            self.assertIn("d2/tri6_tri3/navier_stokes_tri6_tri3_operator.cpp", names)
            self.assertTrue(os.path.exists(result.plan_dump))
            wrapper_source = (
                Path(out_dir) / "op/sfem_GeneratedNavierStokes.cpp"
            ).read_text()
            registration_source = (
                Path(out_dir) / "op/sfem_GeneratedNavierStokes_registration.cpp"
            ).read_text()
            self.assertIn("constexpr int MAX_PARAMETERS = 7;", wrapper_source)
            self.assertIn(
                'Factory::register_op("ss:GeneratedNavierStokes"',
                registration_source,
            )
            self.assertIn(
                "storage[0], storage[1], storage[4], storage[5], FIELD_STRIDE",
                wrapper_source,
            )
            self.assertNotIn(
                "storage[0], storage[1], storage[2], storage[3], storage[4], storage[5], FIELD_STRIDE, u_old_data",
                wrapper_source,
            )

    def test_wall_mounted_hump_sources_cover_mesh_markers_and_restart_output(self):
        root = Path(__file__).resolve().parents[4]
        mesh_header = (root / "external/smesh/src/frontend/smesh_mesh.hpp").read_text()
        mesh_source = (root / "external/smesh/src/frontend/smesh_mesh.cpp").read_text()
        driver_source = (root / "drivers/simulations/wall_mounted_hump.cpp").read_text()
        high_re_script = (
            root / "drivers/simulations/run_wall_mounted_hump_high_re.sh"
        ).read_text()

        self.assertIn("create_wall_mounted_hump", mesh_header)
        self.assertIn("Mesh::create_wall_mounted_hump", mesh_source)
        self.assertIn("PROTEUS_HEX27", mesh_source)
        self.assertIn("MARKER_INLET", driver_source)
        self.assertIn("MARKER_OUTLET", driver_source)
        self.assertIn("MARKER_WALL", driver_source)
        self.assertIn("FunctionSpace::create(mesh, 4", driver_source)
        self.assertIn("GeneratedNavierStokes", driver_source)
        self.assertIn("supports_generated_navier_stokes_solver", driver_source)
        self.assertIn("prepare_mesh_for_generated_navier_stokes", driver_source)
        self.assertIn("PROTEUS_HEX27", driver_source)
        self.assertIn("Use HEX27 or PROTEUS_HEX27", driver_source)
        self.assertIn("DirichletConditions", driver_source)
        self.assertIn("create_linear_operator", driver_source)
        self.assertIn("create_bcgs", driver_source)
        self.assertIn("function->add_operator(op)", driver_source)
        self.assertIn("function->gradient", driver_source)
        self.assertIn("function->apply", driver_source)
        self.assertIn("write_time_step(\"state\"", driver_source)
        self.assertIn("write_nodal(\"u0\"", driver_source)
        self.assertIn("write_nodal(\"p\"", driver_source)
        self.assertIn("solve_stages.csv", driver_source)
        self.assertIn("residual_norm", driver_source)
        self.assertIn("SFEM_REYNOLDS_NUMBER", high_re_script)
        self.assertIn("SFEM_NU", high_re_script)
        self.assertIn("SFEM_HUMP_INLET_U", high_re_script)
        self.assertIn("SFEM_HUMP_BODY_LENGTH", high_re_script)
        self.assertIn("SFEM_REYNOLDS_NUMBER:=2000", high_re_script)
        self.assertIn("u * L / re", high_re_script)
        self.assertIn("run_wall_mounted_hump.sh", high_re_script)


if __name__ == "__main__":
    unittest.main()
