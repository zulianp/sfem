import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import sympy as sp

from sfem import gen

from codegen.framework.materials.mooney_rivlin import material as mooney_rivlin
from codegen.framework.materials.neohookean_ogden import material as neohookean_ogden
from codegen.framework.materials.neumann import material as neumann
from codegen.framework.materials.neumann_general import material as neumann_general
from codegen.framework.materials.poro_hyperelasticity import material as poro_hyperelasticity
from codegen.framework.materials.stokes import material as stokes
from codegen.framework.materials.two_phase_flow import material as two_phase_flow


def _element_name(element):
    return getattr(element, "name", str(element).upper())


def _selected_elements(material, element_names):
    by_name = {_element_name(element): element for element in material.elements}
    return tuple(by_name.get(str(element).upper(), element) for element in element_names)


def _generation_plan(material, element_name):
    elements = _selected_elements(material, (element_name,))
    stage = gen.UserInputStage.create(
        material,
        elements,
        gen.DEFAULT_VECTOR_SIZE,
        None,
    )
    plan = gen.SpecializedFormManipulationStage(stage, gen._evaluate_forms(stage)).run()
    return stage, plan


def _expression_plan(unit, order):
    return next(plan for plan in unit.expression_plans if plan.form_order is order)


def _coefficients_by_row(expression_plan):
    return {coefficient.row_field: coefficient for coefficient in expression_plan.coefficients}


def _assert_sympy_equal(test_case, actual, expected):
    test_case.assertEqual(sp.simplify(actual - expected), sp.S.Zero)


def _symbol(name):
    return sp.Symbol(name)


def _stokes_reference_coefficients(dim, direction=False):
    mu = _symbol("mu")
    p = _symbol("p_direction" if direction else "p")
    grad_suffix = "_direction_grad" if direction else "_grad"
    values = {}
    for row in range(dim):
        gradient = []
        for col in range(dim):
            value = mu * (
                _symbol("u%d%s_%d" % (row, grad_suffix, col))
                + _symbol("u%d%s_%d" % (col, grad_suffix, row))
            )
            if row == col:
                value -= p
            gradient.append(value)
        values["u%d" % row] = (sp.S.Zero, tuple(gradient))
    values["p"] = (
        sum(_symbol("u%d%s_%d" % (d, grad_suffix, d)) for d in range(dim)),
        tuple(sp.S.Zero for _ in range(dim)),
    )
    return values


def _poro_reference_coefficients(dim, direction=False):
    alpha = _symbol("alpha")
    dt = _symbol("dt")
    hydraulic_conductivity = _symbol("hydraulic_conductivity")
    storage = _symbol("storage")
    p = _symbol("p_direction" if direction else "p")
    grad_suffix = "_direction_grad" if direction else "_grad"
    values = {}
    for row in range(dim):
        gradient = []
        for col in range(dim):
            gradient.append(-alpha * p if row == col else sp.S.Zero)
        values["u%d" % row] = (sp.S.Zero, tuple(gradient))
    if direction:
        pressure_value = (
            alpha
            * sum(_symbol("u%d_direction_grad_%d" % (d, d)) for d in range(dim))
            + storage * _symbol("p_direction")
        ) / dt
        pressure_gradient = tuple(
            hydraulic_conductivity * _symbol("p_direction_grad_%d" % d)
            for d in range(dim)
        )
    else:
        pressure_value = (
            alpha
            * sum(
                _symbol("u%d_grad_%d" % (d, d))
                - _symbol("u%d_old_grad_%d" % (d, d))
                for d in range(dim)
            )
            + storage * (_symbol("p") - _symbol("p_old"))
        ) / dt
        pressure_gradient = tuple(
            hydraulic_conductivity * _symbol("p_grad_%d" % d) for d in range(dim)
        )
    values["p"] = (pressure_value, pressure_gradient)
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


def _repo_root():
    return Path(__file__).resolve().parents[4]


def _operator_compiler():
    requested = os.environ.get("CXX")
    if requested:
        return shutil.which(requested) or requested
    return shutil.which("mpic++") or shutil.which("mpicxx") or shutil.which("c++")


def _wrapper_compile_include_dirs(out_dir, manifest):
    root = _repo_root()
    include_dirs = [
        Path(out_dir),
        root / "frontend",
        root / "frontend" / "ops",
        root / "base",
        root / "external" / "smesh" / "src",
        root / "external" / "smesh" / "src" / "base",
        root / "external" / "smesh" / "src" / "frontend",
        root / "external" / "smesh" / "src" / "io",
        root / "external" / "smesh" / "src" / "mesh",
        root / "external" / "smesh" / "src" / "mesh" / "geometry",
        root / "external" / "smesh" / "src" / "mesh" / "sets",
        root / "external" / "smesh" / "src" / "mesh" / "semistructured",
        root / "external" / "smesh" / "src" / "mesh" / "semistructured" / "graph",
        root / "external" / "smesh" / "src" / "utils",
        root / "external" / "smesh" / "src" / "graph",
        root / "external" / "smesh" / "src" / "profile",
        root / "external" / "smesh" / "src" / "sorting",
        root / "external" / "smesh" / "src" / "arrays",
        root / "external" / "smesh" / "src" / "quadrature",
    ]
    include_dirs.extend(Path(out_dir) / path for path in manifest["generated_include_paths"])
    include_dirs.extend(Path(path) for path in gen._compile_config_include_dirs(str(root)))
    include_dirs.extend(path.parent for path in root.glob("build*/_deps/ryml-src/src/ryml.hpp"))
    include_dirs.extend(path.parents[1] for path in root.glob("build*/_deps/ryml-src/ext/c4core/src/c4/substr.hpp"))
    seen = set()
    unique = []
    for include_dir in include_dirs:
        include_dir = Path(include_dir)
        key = str(include_dir)
        if key not in seen and include_dir.is_dir():
            unique.append(include_dir)
            seen.add(key)
    return unique


def _assert_wrapper_syntax_compiles(test_case, out_dir, manifest):
    compiler = _operator_compiler()
    if compiler is None:
        test_case.skipTest("C++ compiler is not available")
    source = Path(out_dir) / manifest["wrapper"]["source"]
    include_flags = []
    for include_dir in _wrapper_compile_include_dirs(out_dir, manifest):
        include_flags.extend(("-I", str(include_dir)))
    command = [
        compiler,
        "-std=c++17",
        "-O0",
        "-fopenmp-simd",
        "-Werror",
        "-Wno-unused-command-line-argument",
        "-fsyntax-only",
        str(source),
        *include_flags,
    ]
    result = subprocess.run(
        command,
        cwd=str(_repo_root()),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode == 0:
        return
    if "ryml.hpp" in result.stderr:
        test_case.skipTest("wrapper syntax compile requires ryml.hpp")
    test_case.fail(result.stderr)


class M9ReferenceRegressionTest(unittest.TestCase):
    def test_taylor_hood_stokes_residual_and_action_match_hardcoded_reference(self):
        for element in ("TRI6_TRI3", "TET10_TET4", "HEX27_HEX8"):
            with self.subTest(element=element):
                stage, plan = _generation_plan(stokes, element)
                context = stage.element_contexts[0]
                unit = next(unit for unit in plan.emission_kernels_for_context(context) if unit.name == "stokes")
                dim = context.specialization.dim
                _assert_coefficients(
                    self,
                    _expression_plan(unit, gen.FormOrder.ONE),
                    _stokes_reference_coefficients(dim, direction=False),
                )
                _assert_coefficients(
                    self,
                    _expression_plan(unit, gen.FormOrder.TWO),
                    _stokes_reference_coefficients(dim, direction=True),
                )

    def test_coupled_poro_hyperelastic_monolithic_path_matches_hardcoded_reference(self):
        stage, plan = _generation_plan(poro_hyperelasticity, "TRI6_TRI3")
        context = stage.element_contexts[0]
        unit = next(
            unit
            for unit in plan.emission_kernels_for_context(context)
            if unit.name == "poro_hyperelasticity_poro" and unit.is_monolithic
        )
        dim = context.specialization.dim
        _assert_coefficients(
            self,
            _expression_plan(unit, gen.FormOrder.ONE),
            _poro_reference_coefficients(dim, direction=False),
        )
        _assert_coefficients(
            self,
            _expression_plan(unit, gen.FormOrder.TWO),
            _poro_reference_coefficients(dim, direction=True),
        )

    def test_generated_coupled_block_kernel_matches_hardcoded_reference(self):
        stage, plan = _generation_plan(stokes, "TET10_TET4")
        context = stage.element_contexts[0]
        block_unit = next(
            unit
            for unit in plan.emission_kernels_for_context(context)
            if unit.is_block and unit.block.name == "form_2_u_p"
        )
        dim = context.specialization.dim
        expected = {
            "u%d" % row: (
                sp.S.Zero,
                tuple(-_symbol("p_direction") if row == col else sp.S.Zero for col in range(dim)),
            )
            for row in range(dim)
        }
        _assert_coefficients(
            self,
            _expression_plan(block_unit, gen.FormOrder.TWO),
            expected,
        )


class M9GeneratedArtifactRegressionTest(unittest.TestCase):
    MAINTAINED = (
        ("neohookean_ogden", neohookean_ogden, ("TRI3",), ("objective", "gradient", "apply")),
        ("mooney_rivlin", mooney_rivlin, ("TRI3",), ("objective", "gradient", "apply")),
        ("two_phase_flow", two_phase_flow, ("TRI3",), ("residual", "jacobian_action")),
        ("stokes", stokes, ("TRI6_TRI3",), ("residual", "jacobian_action")),
        ("poro_hyperelasticity", poro_hyperelasticity, ("TRI6_TRI3",), ("gradient", "residual", "jacobian_action")),
        ("neumann", neumann, ("TRI3",), ("boundary_residual",)),
        ("neumann_general", neumann_general, ("TRI3",), ("boundary_residual",)),
    )

    def test_maintained_materials_regenerate_compile_and_emit_wrapper_metadata(self):
        if not (shutil.which("mpic++") or shutil.which("mpicxx") or shutil.which("c++")):
            self.skipTest("C++ compiler is not available")
        for name, material, elements, operations in self.MAINTAINED:
            with self.subTest(material=name), tempfile.TemporaryDirectory() as out_dir:
                result = gen.generate(
                    material,
                    out_dir,
                    elements=elements,
                    compile=True,
                    clean=True,
                    dump_plan=True,
                )
                self.assertTrue(result.objects)
                self.assertTrue(os.path.exists(result.plan_dump))
                source_names = {os.path.relpath(path, out_dir) for path in result.sources}
                self.assertIn("kernel_math.hpp", source_names)
                self.assertIn("kernel_diagnostics.hpp", source_names)
                manifest_path = os.path.join(out_dir, "op", "sfem_%s_manifest.json" % material.op_name)
                self.assertIn(os.path.relpath(manifest_path, out_dir), source_names)
                with open(manifest_path, encoding="utf-8") as input_file:
                    manifest = json.load(input_file)
                self.assertEqual(manifest["material"], name)
                self.assertEqual(manifest["op_name"], material.op_name)
                self.assertTrue(manifest["c_abi"])
                self.assertTrue(manifest["generated_include_paths"])
                self.assertTrue(manifest["wrapper"]["source"].endswith(".cpp"))
                self.assertTrue(manifest["wrapper"]["header"].endswith(".hpp"))
                self.assertTrue(manifest["wrapper"]["c_abi_header"].endswith(".hpp"))
                self.assertTrue(manifest["registration"]["source"].endswith(".cpp"))
                self.assertEqual(manifest["registration"]["operator_name"], material.op_name)
                self.assertTrue(manifest["factory"]["class"].endswith(material.op_name))
                self.assertTrue(manifest["factory"]["create"].endswith("%s::create" % material.op_name))
                self.assertTrue(
                    manifest["factory"]["create_from_yaml"].endswith("%s::create_from_yaml" % material.op_name)
                )
                c_abi_names = {entry["name"] for entry in manifest["c_abi"]}
                runtime_operations = {
                    operation["name"] for operation in manifest["runtime_operations"]
                }
                for operation in operations:
                    self.assertIn(operation, runtime_operations)
                wrapper_source = os.path.join(out_dir, manifest["wrapper"]["source"])
                with open(wrapper_source, encoding="utf-8") as input_file:
                    wrapper = input_file.read()
                for operation in manifest["runtime_operations"]:
                    self.assertTrue(operation["variants"])
                    real_t_functions = []
                    for variant in operation["variants"]:
                        self.assertIn(variant["function"], c_abi_names)
                        self.assertIn(variant["variant"], ("affine", "affine_aos", "isoparametric", "sideset"))
                        self.assertIn(variant["scalar_type"], ("real_t", "float"))
                        self.assertTrue(variant["target"])
                        if variant["scalar_type"] == "real_t":
                            real_t_functions.append(variant["function"])
                    if operation["name"] in operations:
                        self.assertTrue(any(function in wrapper for function in real_t_functions))

    def test_generated_op_wrappers_syntax_compile_when_frontend_headers_are_available(self):
        if _operator_compiler() is None:
            self.skipTest("C++ compiler is not available")
        for name, material, elements, _ in self.MAINTAINED:
            with self.subTest(material=name), tempfile.TemporaryDirectory() as out_dir:
                gen.generate(
                    material,
                    out_dir,
                    elements=elements,
                    compile=True,
                    clean=True,
                    dump_plan=True,
                )
                manifest_path = os.path.join(out_dir, "op", "sfem_%s_manifest.json" % material.op_name)
                with open(manifest_path, encoding="utf-8") as input_file:
                    manifest = json.load(input_file)
                _assert_wrapper_syntax_compiles(self, out_dir, manifest)

    def test_plan_dump_schema_and_specialized_plan_metadata_for_all_maintained_materials(self):
        for name, material, elements, _ in self.MAINTAINED:
            with self.subTest(material=name), tempfile.TemporaryDirectory() as out_dir:
                result = gen.generate(
                    material,
                    out_dir,
                    elements=elements,
                    clean=True,
                    dump_plan=True,
                )
                with open(result.plan_dump, encoding="utf-8") as input_file:
                    dump = json.load(input_file)
                self.assertEqual(dump["stage"], gen.PipelineStage.SPECIALIZED_FORM_MANIPULATION.value)
                self.assertGreater(dump["n_kernels"], 0)
                self.assertTrue(dump["kernels"])
                for kernel in dump["kernels"]:
                    for key in (
                        "form_collection",
                        "mesh_phase_plans",
                        "blocks",
                        "streams",
                        "expression_plans",
                    ):
                        self.assertIn(key, kernel)

                selected_elements = _selected_elements(material, elements)
                stage = gen.UserInputStage.create(
                    material,
                    selected_elements,
                    gen.DEFAULT_VECTOR_SIZE,
                    None,
                )
                for context in stage.element_contexts:
                    specialized_units = result.plan.emission_kernels_for_context(context)
                    self.assertTrue(specialized_units)
                    for unit in specialized_units:
                        geometry_phases = [
                            phase
                            for phase in unit.mesh_phase_plans
                            if phase.phase is gen.MeshPhase.GEOMETRY
                        ]
                        self.assertTrue(geometry_phases)
                        self.assertTrue(geometry_phases[0].geometries)
                        local_call_phases = [
                            phase
                            for phase in unit.mesh_phase_plans
                            if phase.phase is gen.MeshPhase.LOCAL_CALL
                        ]
                        self.assertTrue(local_call_phases)
                        self.assertTrue(unit.expression_plans)
                        for expression_plan in unit.expression_plans:
                            if expression_plan.form_order is not gen.FormOrder.ZERO:
                                self.assertIsNotNone(expression_plan.dependencies)
                        for block in unit.blocks:
                            self.assertTrue(block.local_phase_plans)
                            for local_phase in block.local_phase_plans:
                                self.assertIn(local_phase.phase, tuple(gen.LocalPhase))
                        if unit.blocks:
                            self.assertTrue(
                                any(block.basis_plans or any(phase.basis_plans for phase in block.local_phase_plans)
                                    for block in unit.blocks)
                                or unit.kind.value == "boundary_residual_soa"
                            )


if __name__ == "__main__":
    unittest.main()
