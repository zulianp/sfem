import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from sfem import gen

from codegen.framework.materials.laplace import material as laplace
from codegen.framework.materials.navier_stokes import material as navier_stokes


def _element_name(element):
    return getattr(element, "name", str(element).upper())


def _selected_element(material, element_name):
    by_name = {_element_name(element): element for element in material.elements}
    return by_name[str(element_name).upper()]


def _generation_plan(material, element_name, matrix_format_plan):
    stage = gen.UserInputStage.create(
        material,
        (_selected_element(material, element_name),),
        gen.DEFAULT_VECTOR_SIZE,
        None,
        matrix_format_plan,
    )
    plan = gen.SpecializedFormManipulationStage(stage, gen._evaluate_forms(stage)).run()
    return stage, plan


class M11MatrixFormatAssemblyTest(unittest.TestCase):
    def test_generated_neohookean_hex8_hessian_assembly_reuses_sum_factorization(self):
        generated_root = (
            Path(__file__).resolve().parents[4]
            / "frontend"
            / "ops"
            / "generated"
            / "neohookean_ogden"
        )
        source = (
            generated_root
            / "d3"
            / "hex8"
            / "neohookean_ogden_hex8_operator.cpp"
        ).read_text()
        hessian_begin = source.index(
            "neohookean_ogden_hex8_hex8_hessian_isoparametric_mesh_soa_assemble_impl"
        )
        hessian_end = source.index(
            'extern "C" int neohookean_ogden_hex8_hex8_hessian_crs_isoparametric_mesh_soa',
            hessian_begin,
        )
        hessian_source = source[hessian_begin:hessian_end]

        self.assertNotIn("TENSOR_SHAPE_INDEX", hessian_source)
        self.assertNotIn("STREAM_SHAPE_ORDER", hessian_source)
        self.assertIn(
            "neohookean_ogden_d3_tensor_product_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>",
            hessian_source,
        )
        self.assertIn(
            "neohookean_ogden_d3_tensor_product_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>",
            source,
        )
        self.assertIn("static constexpr int VECTOR_SIZE = 1;", hessian_source)
        self.assertIn("tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>", hessian_source)
        self.assertIn("neohookean_ogden_hex8_hex8_hessian_crs_isoparametric_mesh_soa", source)
        self.assertIn("neohookean_ogden_hex8_hex8_hessian_bsr_isoparametric_mesh_soa", source)
        self.assertIn(
            "GeneratedNeoHookeanOgden::hessian_bsr",
            (
                generated_root
                / "op"
                / "sfem_GeneratedNeoHookeanOgden.cpp"
            ).read_text(),
        )
        self.assertIn(
            "neohookean_ogden_tet4_tet4_hessian_crs_isoparametric_mesh_soa",
            (
                generated_root
                / "op"
                / "sfem_GeneratedNeoHookeanOgden_c_abi.hpp"
            ).read_text(),
        )

    def test_matrix_format_request_specializes_simplex_diagnostics(self):
        matrix_plan = gen.matrix_format_plan_from_request(
            ("crs", "bsr", "dia", "coo", "patch"),
            ("standard", "packed"),
            ("one_pass", "two_pass"),
            patch_node_index_filter=True,
        )
        stage, plan = _generation_plan(laplace, "TRI3", matrix_plan)
        unit = next(
            unit
            for unit in plan.emission_kernels_for_context(stage.element_contexts[0])
            if unit.name == "laplace"
        )

        variants = {variant.name: variant for variant in unit.matrix_format_plan.variants}
        self.assertEqual(len(variants), 15)
        self.assertIn("crs_standard", variants)
        self.assertIn("bsr_packed_one_pass", variants)
        self.assertIn("dia_packed_two_pass", variants)
        self.assertIn("coo_standard", variants)
        self.assertIn("patch_standard_indexed", variants)
        self.assertIn("patch_packed_two_pass_indexed", variants)

        for variant in variants.values():
            self.assertEqual(variant.row_dofs_per_element, 3)
            self.assertEqual(variant.column_dofs_per_element, 3)
            self.assertEqual(variant.entries_per_element, 9)
            self.assertGreater(variant.expected_flops_per_element, 0)
            self.assertGreater(variant.expected_bytes_per_element, 0)

        self.assertFalse(variants["crs_standard"].format_aware_apply)
        self.assertTrue(variants["bsr_standard"].format_aware_apply)
        self.assertTrue(variants["dia_standard"].format_aware_apply)
        self.assertTrue(variants["patch_standard_indexed"].node_index_filter)

    def test_matrix_format_plan_specializes_mixed_taylor_hood_blocks(self):
        matrix_plan = gen.matrix_format_plan_from_request(("crs",), ("standard",))
        stage, plan = _generation_plan(navier_stokes, "TRI6_TRI3", matrix_plan)
        units = {
            unit.name: unit
            for unit in plan.emission_kernels_for_context(stage.element_contexts[0])
        }

        monolithic = units["navier_stokes"].matrix_format_plan.variants[0]
        self.assertEqual(monolithic.row_dofs_per_element, 15)
        self.assertEqual(monolithic.column_dofs_per_element, 15)
        self.assertEqual(monolithic.entries_per_element, 225)

        velocity_block = units["navier_stokes_form_2_u_u"].matrix_format_plan.variants[0]
        self.assertEqual(velocity_block.row_dofs_per_element, 12)
        self.assertEqual(velocity_block.column_dofs_per_element, 12)
        self.assertEqual(velocity_block.entries_per_element, 144)

        pressure_velocity_block = units["navier_stokes_form_2_p_u"].matrix_format_plan.variants[0]
        self.assertEqual(pressure_velocity_block.row_dofs_per_element, 3)
        self.assertEqual(pressure_velocity_block.column_dofs_per_element, 12)
        self.assertEqual(pressure_velocity_block.entries_per_element, 36)

    def test_generated_matrix_format_metadata_sources_compile(self):
        if not (shutil.which("mpic++") or shutil.which("mpicxx") or shutil.which("c++")):
            self.skipTest("C++ compiler is not available")
        with tempfile.TemporaryDirectory() as out_dir:
            result = gen.generate(
                laplace,
                out_dir,
                elements=("TRI3",),
                clean=True,
                compile=True,
                dump_plan=True,
                matrix_formats=("crs", "bsr", "dia", "coo", "patch"),
                matrix_mesh_layouts=("standard", "packed"),
                matrix_packed_passes=("one_pass", "two_pass"),
                matrix_patch_node_index_filter=True,
            )
            names = {os.path.relpath(path, out_dir) for path in result.sources}
            self.assertIn("matrix_formats.hpp", names)
            self.assertIn("d2/tri3/laplace_tri3_matrix_format_operator.cpp", names)
            self.assertIn("d2/tri3/laplace_tri3_matrix_format_operator.o", {os.path.relpath(path, out_dir) for path in result.objects})

            source = (Path(out_dir) / "d2/tri3/laplace_tri3_matrix_format_operator.cpp").read_text()
            header = (Path(out_dir) / "matrix_formats.hpp").read_text()
            self.assertIn("laplace_tri3_crs_standard_matrix_assembly_diagnostics_data", source)
            self.assertIn("laplace_tri3_bsr_packed_one_pass_matrix_assembly_diagnostics_data", source)
            self.assertIn("laplace_tri3_dia_packed_two_pass_matrix_assembly_diagnostics_data", source)
            self.assertIn("laplace_tri3_patch_standard_indexed_matrix_assembly_diagnostics_data", source)
            self.assertIn("MatrixAssemblyDiagnostics_arithmetic_intensity", header)

            plan = Path(result.plan_dump).read_text()
            self.assertIn('"matrix_format_plan"', plan)
            self.assertIn('"format": "crs"', plan)
            self.assertIn('"mesh_layout": "packed"', plan)
            self.assertIn('"packed_pass": "two_pass"', plan)

            manifest = json.loads(
                (Path(out_dir) / "op/sfem_GeneratedLaplace_manifest.json").read_text()
            )
            self.assertIn(
                {
                    "header": "matrix_formats.hpp",
                    "source": "d2/tri3/laplace_tri3_matrix_format_operator.cpp",
                },
                manifest["matrix_formats"],
            )


if __name__ == "__main__":
    unittest.main()
