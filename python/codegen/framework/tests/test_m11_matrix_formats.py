import csv
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from sfem import gen

from codegen.framework.materials.laplace import material as laplace
from codegen.framework.materials.linear_elasticity import material as linear_elasticity
from codegen.framework.materials.mooney_rivlin import material as mooney_rivlin
from codegen.framework.materials.navier_stokes import material as navier_stokes
from codegen.framework.materials.neohookean_ogden import material as neohookean_ogden
from codegen.framework.materials.neumann import material as neumann
from codegen.framework.materials.neumann_general import material as neumann_general
from codegen.framework.materials.poro_hyperelasticity import material as poro_hyperelasticity
from codegen.framework.materials.stokes import material as stokes
from codegen.framework.materials.two_phase_flow import material as two_phase_flow
from codegen.framework.plans.matrix_formats import (
    BlockDiagSymAssemblyPlan,
    BSRAssemblyPlan,
    COOAssemblyPlan,
    CRSAssemblyPlan,
    DIAAssemblyPlan,
    PatchAssemblyPlan,
)
from codegen.framework.scripts import matrix_format_benchmark_report


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


def _hessian_assembly_body(source, function_name):
    begin = source.index(function_name)
    end = source.index('extern "C" int', begin)
    return source[begin:end]


def _manifest_runtime_variants(metadata, operation):
    for runtime_operation in metadata["runtime_operations"]:
        if runtime_operation["name"] == operation:
            return runtime_operation["variants"]
    raise AssertionError("missing runtime operation %s" % operation)


def _static_function_body(source, signature):
    begin = source.index(signature)
    next_function = source.find("template <typename scalar_t>", begin + len(signature))
    if next_function < 0:
        return source[begin:]
    return source[begin:next_function]


class M11MatrixFormatAssemblyTest(unittest.TestCase):
    MAINTAINED_MATRIX_FORMAT_MATERIALS = (
        ("neohookean_ogden", neohookean_ogden, ("TRI3",)),
        ("mooney_rivlin", mooney_rivlin, ("TRI3",)),
        ("two_phase_flow", two_phase_flow, ("TRI3",)),
        ("stokes", stokes, ("TRI6_TRI3",)),
        ("poro_hyperelasticity", poro_hyperelasticity, ("TRI6_TRI3",)),
        ("neumann", neumann, ("TRI3",)),
        ("neumann_general", neumann_general, ("TRI3",)),
    )

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
            / "proteus_hex8"
            / "neohookean_ogden_proteus_hex8_operator.cpp"
        ).read_text()
        self.assertIn("#include <cstdio>", source)
        hessian_begin = source.index(
            "neohookean_ogden_proteus_hex8_hessian_isoparametric_mesh_soa_assemble_impl"
        )
        hessian_end = source.index(
            'extern "C" int neohookean_ogden_proteus_hex8_hessian_bsr_isoparametric_mesh_soa',
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
        self.assertNotIn("ordered_shape_index", hessian_source)
        self.assertNotIn("matrix_coordinate_streams", hessian_source)
        self.assertNotIn("block_coordinate_streams", hessian_source)
        self.assertNotIn("coordinate_value", hessian_source)
        self.assertIn(
            "tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>",
            hessian_source,
        )
        self.assertIn("isoparametric_grad_1d, block_coordinate_data,", hessian_source)
        self.assertNotIn("neohookean_ogden_proteus_hex8_hessian_crs_isoparametric_mesh_soa", source)
        self.assertIn("neohookean_ogden_proteus_hex8_hessian_bsr_isoparametric_mesh_soa", source)
        bsr_scatter = _static_function_body(
            source,
            "static SFEM_INLINE int neohookean_ogden_proteus_hex8_hessian_isoparametric_mesh_soa_scatter_bsr",
        )
        self.assertIn("count_t entries[N_SHAPE * N_SHAPE];", bsr_scatter)
        self.assertIn("bool valid_block_graph = true;", bsr_scatter)
        self.assertIn("missing block graph entry", bsr_scatter)
        self.assertIn("if (!valid_block_graph) return SFEM_FAILURE;", bsr_scatter)
        self.assertIn("return SFEM_SUCCESS;", bsr_scatter)
        self.assertIn("neohookean_ogden_proteus_hex8_hessian_isoparametric_mesh_soa_find_cols(ev, cols, lenrow, ks);", bsr_scatter)
        self.assertIn("entries[i * N_SHAPE + j] = row_begin + ks[j];", bsr_scatter)
        self.assertIn("scalar_t *const block = &values[entries[i * N_SHAPE + j] * DIM * DIM];", bsr_scatter)
        self.assertIn("block[bi * DIM + bj] += element_matrix[row * (DIM * N_SHAPE) + col];", bsr_scatter)
        self.assertLess(
            bsr_scatter.index("entries[i * N_SHAPE + j] = row_begin + ks[j];"),
            bsr_scatter.index("scalar_t *const block = &values[entries[i * N_SHAPE + j] * DIM * DIM];"),
        )
        self.assertNotIn("std::vector", bsr_scatter)
        self.assertIn("int invalid_matrix_graph = 0;", source)
        self.assertIn("reduction(|:invalid_matrix_graph)", source)
        self.assertIn("return invalid_matrix_graph ? SFEM_FAILURE : SFEM_SUCCESS;", source)
        self.assertIn("invalid_matrix_graph |= (neohookean_ogden_proteus_hex8_hessian_isoparametric_mesh_soa_scatter_bsr", source)
        self.assertIn("values[entries[i * N_SHAPE + j] * DIM * DIM]", source)
        for matrix_format in ("crs", "dia", "coo", "patch"):
            self.assertNotIn(
                "neohookean_ogden_proteus_hex8_hessian_isoparametric_mesh_soa_scatter_%s"
                % matrix_format,
                source,
            )
            self.assertNotIn(
                "neohookean_ogden_proteus_hex8_hessian_%s_isoparametric_mesh_soa"
                % matrix_format,
                source,
            )
        self.assertNotIn(
            "neohookean_ogden_proteus_hex8_hessian_coo_triplet_isoparametric_mesh_soa",
            source,
        )
        self.assertIn(
            "GeneratedNeoHookeanOgden::hessian_bsr",
            (
                generated_root
                / "op"
                / "sfem_GeneratedNeoHookeanOgden.cpp"
            ).read_text(),
        )
        wrapper_source = (
            generated_root
            / "op"
            / "sfem_GeneratedNeoHookeanOgden.cpp"
        ).read_text()
        wrapper_header = (
            generated_root
            / "op"
            / "sfem_GeneratedNeoHookeanOgden.hpp"
        ).read_text()
        self.assertIn("GeneratedNeoHookeanOgden::hessian_dia", wrapper_source)
        self.assertIn("GeneratedNeoHookeanOgden::hessian_coo", wrapper_source)
        self.assertIn("GeneratedNeoHookeanOgden::hessian_patch", wrapper_source)
        self.assertIn("int hessian_dia", wrapper_header)
        self.assertIn("int hessian_coo", wrapper_header)
        self.assertIn("int hessian_patch", wrapper_header)
        with tempfile.TemporaryDirectory() as out_dir:
            gen.generate(
                neohookean_ogden,
                out_dir,
                elements=("HEX8", "TET4"),
                clean=True,
                matrix_formats=("crs", "coo"),
            )
            c_abi_header = (
                Path(out_dir)
                / "op"
                / "sfem_GeneratedNeoHookeanOgden_c_abi.hpp"
            ).read_text()
            self.assertIn("neohookean_ogden_hessian_coo_triplet_3d_isoparametric_mesh_soa", c_abi_header)
            self.assertIn("neohookean_ogden_hessian_crs_3d_isoparametric_mesh_soa", c_abi_header)
            self.assertIn("const smesh::ElemType element_type", c_abi_header)
            self.assertNotIn("neohookean_ogden_hex8_hessian_coo_triplet_isoparametric_mesh_soa", c_abi_header)
            self.assertNotIn("neohookean_ogden_tet4_hessian_crs_isoparametric_mesh_soa", c_abi_header)

    def test_generated_neohookean_hex_hessian_assembly_wraps_proteus(self):
        generated_root = (
            Path(__file__).resolve().parents[4]
            / "frontend"
            / "ops"
            / "generated"
            / "neohookean_ogden"
        )
        quad4_source = (
            generated_root
            / "d2"
            / "quad4"
            / "neohookean_ogden_quad4_operator.cpp"
        ).read_text()
        proteus_quad4_source = (
            generated_root
            / "d2"
            / "proteus_quad4"
            / "neohookean_ogden_proteus_quad4_operator.cpp"
        ).read_text()
        proteus_quad4_hessian = _hessian_assembly_body(
            proteus_quad4_source,
            "neohookean_ogden_proteus_quad4_hessian_isoparametric_mesh_soa_assemble_impl",
        )
        hex8_source = (
            generated_root
            / "d3"
            / "hex8"
            / "neohookean_ogden_hex8_operator.cpp"
        ).read_text()
        proteus_hex8_source = (
            generated_root
            / "d3"
            / "proteus_hex8"
            / "neohookean_ogden_proteus_hex8_operator.cpp"
        ).read_text()
        proteus_hex8_hessian = _hessian_assembly_body(
            proteus_hex8_source,
            "neohookean_ogden_proteus_hex8_hessian_isoparametric_mesh_soa_assemble_impl",
        )
        hex27_source = (
            generated_root
            / "d3"
            / "hex27"
            / "neohookean_ogden_hex27_operator.cpp"
        ).read_text()
        proteus_source = (
            generated_root
            / "d3"
            / "proteus_hex27"
            / "neohookean_ogden_proteus_hex27_operator.cpp"
        ).read_text()
        proteus_hessian = _hessian_assembly_body(
            proteus_source,
            "neohookean_ogden_proteus_hex27_hessian_isoparametric_mesh_soa_assemble_impl",
        )

        self.assertNotIn("neohookean_ogden_quad4_hessian_isoparametric_mesh_soa_assemble_impl", quad4_source)
        self.assertIn("idx_t *proteus_elements[4] = {", quad4_source)
        self.assertIn("elements[3],", quad4_source)
        self.assertIn("neohookean_ogden_proteus_quad4_hessian_bsr_isoparametric_mesh_soa", quad4_source)
        self.assertNotIn("block_u_streams_ordered_shape_index", quad4_source)
        self.assertNotIn("block_h_streams_ordered_shape_index", quad4_source)
        self.assertNotIn("block_out_streams_ordered_shape_index", quad4_source)
        self.assertNotIn("for (ptrdiff_t element = 0; element < nelements; ++element)", quad4_source)
        self.assertNotIn("STREAM_SHAPE_ORDER", proteus_quad4_hessian)
        self.assertNotIn("TENSOR_SHAPE_INDEX", proteus_quad4_hessian)
        self.assertNotIn("block_coordinate_streams[N_SHAPE * DIM] = {", proteus_quad4_hessian)
        self.assertNotIn("block_coordinate_streams_ordered_shape_index", proteus_quad4_hessian)
        self.assertNotIn("matrix_coordinate_streams", proteus_quad4_hessian)
        self.assertNotIn("coordinate_value", proteus_quad4_hessian)
        self.assertNotIn("neohookean_ogden_hex8_hessian_isoparametric_mesh_soa_assemble_impl", hex8_source)
        self.assertIn("idx_t *proteus_elements[8] = {", hex8_source)
        self.assertIn("elements[3],", hex8_source)
        self.assertIn("neohookean_ogden_proteus_hex8_hessian_bsr_isoparametric_mesh_soa", hex8_source)
        self.assertNotIn("block_u_streams_ordered_shape_index", hex8_source)
        self.assertNotIn("block_h_streams_ordered_shape_index", hex8_source)
        self.assertNotIn("block_out_streams_ordered_shape_index", hex8_source)
        self.assertNotIn("for (ptrdiff_t element = 0; element < nelements; ++element)", hex8_source)
        self.assertNotIn("STREAM_SHAPE_ORDER", proteus_hex8_hessian)
        self.assertNotIn("TENSOR_SHAPE_INDEX", proteus_hex8_hessian)
        self.assertNotIn("block_coordinate_streams[N_SHAPE * DIM] = {", proteus_hex8_hessian)
        self.assertNotIn("block_coordinate_streams_ordered_shape_index", proteus_hex8_hessian)
        self.assertNotIn("matrix_coordinate_streams[stream] = block_coordinate_streams[stream]", proteus_hex8_hessian)
        self.assertNotIn("matrix_coordinate_streams", proteus_hex8_hessian)
        self.assertNotIn("coordinate_value", proteus_hex8_hessian)
        self.assertNotIn("neohookean_ogden_hex27_hessian_isoparametric_mesh_soa_assemble_impl", hex27_source)
        self.assertIn("idx_t *proteus_elements[27] = {", hex27_source)
        self.assertIn("elements[8],", hex27_source)
        self.assertIn("neohookean_ogden_proteus_hex27_hessian_bsr_isoparametric_mesh_soa", hex27_source)
        self.assertNotIn("block_u_streams_ordered_shape_index", hex27_source)
        self.assertNotIn("block_h_streams_ordered_shape_index", hex27_source)
        self.assertNotIn("block_out_streams_ordered_shape_index", hex27_source)
        self.assertNotIn("for (ptrdiff_t element = 0; element < nelements; ++element)", hex27_source)
        self.assertNotIn("STREAM_SHAPE_ORDER", proteus_hessian)
        self.assertNotIn("TENSOR_SHAPE_INDEX", proteus_hessian)
        self.assertNotIn("block_coordinate_streams[N_SHAPE * DIM] = {", proteus_hessian)
        self.assertNotIn("block_coordinate_streams_ordered_shape_index", proteus_hessian)
        self.assertNotIn("matrix_coordinate_streams[stream] = block_coordinate_streams[stream]", proteus_hessian)
        self.assertNotIn("matrix_coordinate_streams", proteus_hessian)
        self.assertNotIn("coordinate_value", proteus_hessian)
        self.assertIn(
            "tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>",
            proteus_hessian,
        )
        self.assertIn("isoparametric_grad_1d, block_coordinate_data,", proteus_hessian)
        self.assertIn("for (int shape = 0; shape < N_SHAPE; ++shape)", proteus_hessian)

    def test_generated_block_diag_sym_hessian_for_neohookean_and_linear_elasticity(self):
        if not (shutil.which("mpic++") or shutil.which("mpicxx") or shutil.which("c++")):
            self.skipTest("C++ compiler is not available")
        for material_name, material, op_name in (
            ("neohookean_ogden", neohookean_ogden, "GeneratedNeoHookeanOgden"),
            ("linear_elasticity", linear_elasticity, "GeneratedLinearElasticity"),
        ):
            with self.subTest(material=material_name):
                with tempfile.TemporaryDirectory() as out_dir:
                    result = gen.generate(
                        material,
                        out_dir,
                        elements=("TET4",),
                        clean=True,
                        compile=True,
                        dump_plan=True,
                        matrix_formats=("block_diag_sym",),
                    )
                    root = Path(out_dir)
                    source = (
                        root
                        / "d3"
                        / "tet4"
                        / ("%s_tet4_operator.cpp" % material_name)
                    ).read_text()
                    wrapper_source = (
                        root / "op" / ("sfem_%s.cpp" % op_name)
                    ).read_text()
                    wrapper_header = (
                        root / "op" / ("sfem_%s.hpp" % op_name)
                    ).read_text()
                    c_abi_header = (
                        root / "op" / ("sfem_%s_c_abi.hpp" % op_name)
                    ).read_text()
                    manifest = json.loads(
                        (root / "op" / ("sfem_%s_manifest.json" % op_name)).read_text()
                    )

                    assembly_base = "%s_tet4_hessian_isoparametric_mesh_soa" % material_name
                    public_name = "%s_hessian_block_diag_sym_3d_isoparametric_mesh_soa" % material_name
                    self.assertIn("%s_scatter_block_diag_sym" % assembly_base, source)
                    self.assertIn("static constexpr int SYM_DIM = (DIM * (DIM + 1)) / 2;", source)
                    self.assertIn("values[(ptrdiff_t)ev[i] * SYM_DIM]", source)
                    self.assertIn("for (int bj = bi; bj < DIM; ++bj)", source)
                    self.assertIn("block[sym++] += element_matrix[row * NDOFS + col];", source)
                    self.assertNotIn("%s_scatter_bsr" % assembly_base, source)
                    self.assertIn(public_name, c_abi_header)
                    self.assertIn("real_t *const values) override", wrapper_header)
                    self.assertIn("%s::hessian_block_diag_sym" % op_name, wrapper_source)
                    self.assertIn("%s(domain.element_type" % public_name, wrapper_source)
                    method_begin = wrapper_source.index("%s::hessian_block_diag_sym" % op_name)
                    method_end = wrapper_source.index("void %s::set_option" % op_name, method_begin)
                    block_diag_method = wrapper_source[method_begin:method_end]
                    if material_name == "linear_elasticity":
                        self.assertIn("(void)x;", block_diag_method)
                        self.assertNotIn("requires a current state", block_diag_method)
                    else:
                        self.assertIn("const real_t *const current = x;", block_diag_method)
                        self.assertIn("requires a current state", block_diag_method)

                    runtime_variants = _manifest_runtime_variants(
                        manifest,
                        "hessian_block_diag_sym",
                    )
                    self.assertTrue(
                        any(variant["function"] == public_name for variant in runtime_variants)
                    )
                    plan = Path(result.plan_dump).read_text()
                    self.assertIn('"format": "block_diag_sym"', plan)
                    self.assertIn('"kind": "block_diag_sym"', plan)

    def test_block_diag_sym_plan_specializes_vector_block(self):
        matrix_plan = gen.matrix_format_plan_from_request(("block_diag_sym",), ("standard",))
        stage, plan = _generation_plan(linear_elasticity, "TET4", matrix_plan)
        unit = next(
            unit
            for unit in plan.emission_kernels_for_context(stage.element_contexts[0])
            if unit.name == "linear_elasticity"
        )
        variant = unit.matrix_format_plan.variants[0]
        self.assertEqual(variant.name, "block_diag_sym_standard")
        self.assertEqual(variant.row_dofs_per_element, 12)
        self.assertEqual(variant.column_dofs_per_element, 12)
        self.assertIsInstance(variant.assembly_plan, BlockDiagSymAssemblyPlan)
        self.assertEqual(variant.assembly_plan.block_size, 3)
        self.assertEqual(variant.assembly_plan.symmetric_entries_per_node, 6)
        self.assertEqual(variant.value_writes_per_element, 24)
        self.assertEqual(variant.assembly_plan.value_writes_per_element, 24)
        self.assertEqual(variant.assembly_plan.value_layout, "node_major_upper_symmetric_aos")

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
        self.assertIn("patch_standard", variants)
        self.assertIn("patch_packed_two_pass", variants)

        for variant in variants.values():
            self.assertEqual(variant.row_dofs_per_element, 3)
            self.assertEqual(variant.column_dofs_per_element, 3)
            self.assertEqual(variant.entries_per_element, 9)
            self.assertGreater(variant.expected_flops_per_element, 0)
            self.assertGreater(variant.expected_bytes_per_element, 0)

        removed_apply_key = "format" + "_aware" + "_apply"
        self.assertNotIn(removed_apply_key, variants["crs_standard"].to_dict())
        self.assertNotIn(removed_apply_key, variants["bsr_standard"].to_dict())
        self.assertNotIn(removed_apply_key, variants["dia_standard"].to_dict())
        self.assertFalse(variants["patch_standard"].node_index_filter)
        self.assertIsInstance(variants["crs_standard"].assembly_plan, CRSAssemblyPlan)
        self.assertEqual(variants["crs_standard"].assembly_plan.row_pointer, "rowptr")
        self.assertEqual(variants["crs_standard"].assembly_plan.mesh_access, "standard_block_elements")
        self.assertEqual(variants["crs_standard"].assembly_plan.accumulation_policy, "add_scatter")
        self.assertEqual(variants["crs_standard"].assembly_plan.reduction_policy, "atomic_add")
        self.assertEqual(variants["crs_packed_one_pass"].assembly_plan.mesh_access, "FunctionSpace::PackedMesh")
        self.assertEqual(
            variants["crs_packed_one_pass"].assembly_plan.element_connectivity,
            "packed->elements(block)->data()",
        )
        self.assertEqual(
            variants["crs_packed_one_pass"].assembly_plan.pack_index_type,
            "FunctionSpace::PackedIdxType",
        )
        self.assertEqual(
            variants["crs_packed_one_pass"].assembly_plan.pack_partition,
            "n_packs/n_elements_per_pack/max_nodes_per_pack",
        )
        self.assertEqual(
            variants["crs_packed_one_pass"].assembly_plan.packed_node_partition,
            "owned_nodes_ptr/n_shared/ghost_ptr/ghost_idx",
        )
        self.assertEqual(
            variants["crs_packed_one_pass"].assembly_plan.value_mapping,
            "PackedMesh::map_to_packed/map_to_unpacked",
        )
        self.assertIsInstance(variants["bsr_standard"].assembly_plan, BSRAssemblyPlan)
        self.assertEqual(variants["bsr_standard"].assembly_plan.block_size, 1)
        self.assertEqual(variants["bsr_standard"].assembly_plan.block_entries_per_element, 9)
        self.assertEqual(variants["bsr_standard"].assembly_plan.structural_compatibility, "requires_node_block_graph")
        self.assertIsInstance(variants["dia_standard"].assembly_plan, DIAAssemblyPlan)
        self.assertEqual(variants["dia_standard"].assembly_plan.values_per_element, 3)
        self.assertEqual(
            variants["dia_standard"].assembly_plan.structural_compatibility,
            "stable_simplex_affine_diagonal_offsets",
        )
        self.assertEqual(
            variants["dia_standard"].assembly_plan.stencil_compatibility,
            "stable_simplex_affine_diagonal_offsets",
        )
        self.assertIsInstance(variants["coo_standard"].assembly_plan, COOAssemblyPlan)
        self.assertEqual(
            variants["coo_standard"].assembly_plan.duplicate_policy,
            "deterministic_element_order_external_reduction",
        )
        self.assertEqual(
            variants["coo_standard"].assembly_plan.sort_policy,
            "external_stable_sort_or_existing_sfem_coo_reduce",
        )
        self.assertEqual(
            variants["coo_standard"].assembly_plan.reduction_phase,
            "non_hot_setup_phase",
        )
        self.assertEqual(variants["coo_standard"].assembly_plan.structural_compatibility, "allows_duplicates")
        self.assertIsInstance(variants["patch_standard"].assembly_plan, PatchAssemblyPlan)
        self.assertFalse(variants["patch_standard"].assembly_plan.node_index_filter)
        self.assertEqual(
            variants["patch_standard"].assembly_plan.structural_compatibility,
            "requires_full_graph",
        )
        self.assertEqual(variants["patch_standard"].assembly_plan.patch_graph, "rowptr_colidx")

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

    def test_bsr_layout_selects_component_block_sizes_for_mixed_blocks(self):
        matrix_plan = gen.matrix_format_plan_from_request(("bsr",), ("standard",))
        stage, plan = _generation_plan(navier_stokes, "TRI6_TRI3", matrix_plan)
        units = {
            unit.name: unit
            for unit in plan.emission_kernels_for_context(stage.element_contexts[0])
        }

        velocity_plan = units["navier_stokes_form_2_u_u"].matrix_format_plan.variants[0].assembly_plan
        self.assertIsInstance(velocity_plan, BSRAssemblyPlan)
        self.assertEqual(velocity_plan.block_size, 2)
        self.assertEqual(velocity_plan.row_block_size, 2)
        self.assertEqual(velocity_plan.column_block_size, 2)
        self.assertEqual(velocity_plan.block_rows_per_element, 6)
        self.assertEqual(velocity_plan.block_columns_per_element, 6)
        self.assertTrue(velocity_plan.compatible_block_size)

        pressure_velocity_plan = units["navier_stokes_form_2_p_u"].matrix_format_plan.variants[0].assembly_plan
        self.assertIsInstance(pressure_velocity_plan, BSRAssemblyPlan)
        self.assertEqual(pressure_velocity_plan.row_block_size, 1)
        self.assertEqual(pressure_velocity_plan.column_block_size, 2)
        self.assertEqual(pressure_velocity_plan.block_size, 0)
        self.assertFalse(pressure_velocity_plan.compatible_block_size)

    def test_dia_layout_detects_stable_and_unsupported_structures(self):
        matrix_plan = gen.matrix_format_plan_from_request(("dia",), ("standard",))
        stage, plan = _generation_plan(laplace, "HEX8", matrix_plan)
        units = {
            unit.name: unit
            for unit in plan.emission_kernels_for_context(stage.element_contexts[0])
        }
        tensor_plan = units["laplace"].matrix_format_plan.variants[0].assembly_plan
        self.assertIsInstance(tensor_plan, DIAAssemblyPlan)
        self.assertEqual(tensor_plan.structural_compatibility, "stable_tensor_product_diagonal_offsets")
        self.assertEqual(tensor_plan.stencil_compatibility, "stable_tensor_product_diagonal_offsets")

        stage, plan = _generation_plan(navier_stokes, "TRI6_TRI3", matrix_plan)
        units = {
            unit.name: unit
            for unit in plan.emission_kernels_for_context(stage.element_contexts[0])
        }
        pressure_velocity_plan = units["navier_stokes_form_2_p_u"].matrix_format_plan.variants[0].assembly_plan
        self.assertIsInstance(pressure_velocity_plan, DIAAssemblyPlan)
        self.assertEqual(
            pressure_velocity_plan.structural_compatibility,
            "unsupported_mixed_or_asymmetric_diagonal_structure",
        )
        self.assertEqual(pressure_velocity_plan.reduction_policy, "not_emitted")

    def test_bsr_coverage_includes_vector_action_and_taylor_hood_velocity_plan(self):
        matrix_test_source = (
            Path(__file__).resolve().parents[4]
            / "frontend"
            / "tests"
            / "sfem_MatrixFromatsTest.cpp"
        ).read_text()
        self.assertIn('sfem::create_op(space, "GeneratedNeoHookeanOgden"', matrix_test_source)
        self.assertIn("neohookean_ogden_apply_packed_3d_isoparametric_mesh_soa", matrix_test_source)
        self.assertIn('assert_close_action("generated NeoHookean packed hessian action"', matrix_test_source)
        frontend_api_source = (
            Path(__file__).resolve().parents[4] / "frontend" / "sfem_API.hpp"
        ).read_text()
        self.assertIn("#include \"sfem_DIA.hpp\"", frontend_api_source)
        self.assertIn("hessian_dia(f, u, es)", frontend_api_source)

        matrix_plan = gen.matrix_format_plan_from_request(("bsr",), ("standard",))
        stage, plan = _generation_plan(stokes, "TRI6_TRI3", matrix_plan)
        units = {
            unit.name: unit
            for unit in plan.emission_kernels_for_context(stage.element_contexts[0])
        }
        velocity_plan = units["stokes_form_2_u_u"].matrix_format_plan.variants[0].assembly_plan
        self.assertIsInstance(velocity_plan, BSRAssemblyPlan)
        self.assertEqual(velocity_plan.block_size, 2)
        self.assertEqual(velocity_plan.block_rows_per_element, 6)
        self.assertEqual(velocity_plan.block_columns_per_element, 6)
        self.assertTrue(velocity_plan.compatible_block_size)

    def test_mixed_residual_coo_triplet_emits_monolithic_and_block_shapes(self):
        with tempfile.TemporaryDirectory() as out_dir:
            result = gen.generate(
                stokes,
                out_dir,
                elements=("TRI6_TRI3",),
                clean=True,
                matrix_formats=("coo",),
            )

            manifest = json.loads(
                (Path(out_dir) / "op/sfem_GeneratedStokes_manifest.json").read_text()
            )
            coo_triplet_variants = _manifest_runtime_variants(manifest, "hessian_coo_triplet")
            self.assertEqual(len(coo_triplet_variants), 8)

            c_abi_header = (Path(out_dir) / "op/sfem_GeneratedStokes_c_abi.hpp").read_text()
            self.assertIn("stokes_hessian_coo_triplet_2d_isoparametric_mesh_soa", c_abi_header)
            self.assertIn("stokes_form_2_p_u_hessian_coo_triplet_2d_isoparametric_mesh_soa", c_abi_header)
            self.assertIn("stokes_form_2_u_p_hessian_coo_triplet_2d_isoparametric_mesh_soa", c_abi_header)
            self.assertNotIn("stokes_tri6_tri3_hessian_coo_triplet_isoparametric_mesh_soa", c_abi_header)

            monolithic = (Path(out_dir) / "d2/tri6_tri3/stokes_tri6_tri3_operator.cpp").read_text()
            pressure_velocity = (
                Path(out_dir) / "d2/tri6_tri3/stokes_form_2_p_u_tri6_tri3_operator.cpp"
            ).read_text()
            velocity_pressure = (
                Path(out_dir) / "d2/tri6_tri3/stokes_form_2_u_p_tri6_tri3_operator.cpp"
            ).read_text()

            self.assertIn("static constexpr int N_ROW_STREAMS = 15;", monolithic)
            self.assertIn("static constexpr int N_COL_STREAMS = 15;", monolithic)
            self.assertIn("static constexpr int N_ROW_STREAMS = 3;", pressure_velocity)
            self.assertIn("static constexpr int N_COL_STREAMS = 12;", pressure_velocity)
            self.assertIn("static constexpr int N_ROW_STREAMS = 12;", velocity_pressure)
            self.assertIn("static constexpr int N_COL_STREAMS = 3;", velocity_pressure)
            self.assertIn("const ptrdiff_t element_offset = element * N_ROW_STREAMS * N_COL_STREAMS;", monolithic)
            self.assertIn("rows[entry] = global_row;", monolithic)
            self.assertIn("cols[entry] = col_node * out_stride + COL_COMPONENT[col_stream];", monolithic)
            for source in (monolithic, pressure_velocity, velocity_pressure):
                self.assertNotIn("ROW_STREAMS[", source)
                self.assertNotIn("COL_STREAMS[", source)
                self.assertNotIn("ROW_TENSOR_STREAMS[", source)
                self.assertNotIn("COL_TENSOR_STREAMS[", source)
            self.assertNotIn("#pragma omp atomic", _static_function_body(
                monolithic,
                "static SFEM_INLINE void stokes_tri6_tri3_hessian_coo_triplet_isoparametric_mesh_soa_scatter_coo_triplets",
            ))
            self.assertNotIn("find_col", monolithic)

    def test_state_dependent_compatible_residual_coo_triplet_emits_two_phase_blocks(self):
        with tempfile.TemporaryDirectory() as out_dir:
            gen.generate(
                two_phase_flow,
                out_dir,
                elements=("TRI3",),
                clean=True,
                matrix_formats=("coo",),
                compile=True,
            )

            manifest = json.loads(
                (Path(out_dir) / "op/sfem_GeneratedTwoPhaseFlow_manifest.json").read_text()
            )
            coo_triplet_variants = _manifest_runtime_variants(manifest, "hessian_coo_triplet")
            self.assertEqual(len(coo_triplet_variants), 10)

            c_abi_header = (Path(out_dir) / "op/sfem_GeneratedTwoPhaseFlow_c_abi.hpp").read_text()
            self.assertIn(
                "two_phase_flow_hessian_coo_triplet_2d_isoparametric_mesh_soa",
                c_abi_header,
            )
            self.assertIn(
                "two_phase_flow_form_2_p_w_p_w_hessian_coo_triplet_2d_isoparametric_mesh_soa",
                c_abi_header,
            )
            self.assertIn(
                "two_phase_flow_form_2_p_c_p_w_hessian_coo_triplet_2d_isoparametric_mesh_soa",
                c_abi_header,
            )
            declaration_begin = c_abi_header.index(
                "extern \"C\" int two_phase_flow_form_2_p_w_p_w_hessian_coo_triplet_2d_isoparametric_mesh_soa"
            )
            declaration_end = c_abi_header.index(");", declaration_begin)
            triplet_declaration = c_abi_header[declaration_begin:declaration_end]

            source = (
                Path(out_dir)
                / "d2/tri3/two_phase_flow_form_2_p_w_p_w_tri3_operator.cpp"
            ).read_text()
            triplet_body = _static_function_body(
                source,
                "static SFEM_INLINE void two_phase_flow_form_2_p_w_p_w_tri3_hessian_coo_triplet_isoparametric_mesh_soa_scatter_coo_triplets",
            )
            self.assertIn("static constexpr int N_FIELDS = 2;", triplet_body)
            self.assertIn("static constexpr int N_STREAMS = N_FIELDS * N_SHAPE;", source)
            self.assertIn("const ptrdiff_t current_stride", source)
            self.assertIn("block_current[0 * N_SHAPE + shape][0] = p_w[node * current_stride];", source)
            self.assertIn("block_current[1 * N_SHAPE + shape][0] = p_c[node * current_stride];", source)
            self.assertIn("block_direction[tensor_trial][0] = scalar_t(1);", source)
            self.assertIn("cols[entry] = ev[col_shape] * out_stride + col_field;", source)
            self.assertNotIn("const ptrdiff_t direction_stride", triplet_declaration)
            self.assertNotIn("#pragma omp atomic", triplet_body)
            self.assertNotIn("find_col", source)

    def test_state_dependent_compatible_residual_crs_bsr_emits_two_phase_blocks(self):
        with tempfile.TemporaryDirectory() as out_dir:
            gen.generate(
                two_phase_flow,
                out_dir,
                elements=("TRI3",),
                clean=True,
                matrix_formats=("crs", "bsr"),
                compile=True,
            )

            manifest = json.loads(
                (Path(out_dir) / "op/sfem_GeneratedTwoPhaseFlow_manifest.json").read_text()
            )
            self.assertEqual(len(_manifest_runtime_variants(manifest, "hessian_crs")), 10)
            self.assertEqual(len(_manifest_runtime_variants(manifest, "hessian_bsr")), 10)

            c_abi_header = (Path(out_dir) / "op/sfem_GeneratedTwoPhaseFlow_c_abi.hpp").read_text()
            self.assertIn(
                "two_phase_flow_form_2_p_w_p_w_hessian_bsr_2d_isoparametric_mesh_soa",
                c_abi_header,
            )
            declaration_begin = c_abi_header.index(
                "extern \"C\" int two_phase_flow_form_2_p_w_p_w_hessian_bsr_2d_isoparametric_mesh_soa"
            )
            declaration_end = c_abi_header.index(");", declaration_begin)
            bsr_declaration = c_abi_header[declaration_begin:declaration_end]
            self.assertNotIn("const ptrdiff_t direction_stride", bsr_declaration)

            source = (
                Path(out_dir)
                / "d2/tri3/two_phase_flow_form_2_p_w_p_w_tri3_operator.cpp"
            ).read_text()
            wrapper = (Path(out_dir) / "op/sfem_GeneratedTwoPhaseFlow.cpp").read_text()
            self.assertIn("static constexpr int N_FIELDS = 2;", source)
            self.assertIn("static constexpr int N_ROW_STREAMS = 3;", source)
            self.assertIn("static constexpr int N_COL_STREAMS = 3;", source)
            self.assertIn("block_current[0 * N_SHAPE + shape][0] = p_w[node * current_stride];", source)
            self.assertIn("block_current[1 * N_SHAPE + shape][0] = p_c[node * current_stride];", source)
            self.assertIn("block[bi * N_FIELDS + bj] += element_matrix[row_stream * N_COL_STREAMS + col_stream];", source)
            self.assertNotIn("ROW_STREAMS[", source)
            self.assertNotIn("COL_STREAMS[", source)
            self.assertNotIn("ROW_TENSOR_STREAMS[", source)
            self.assertNotIn("COL_TENSOR_STREAMS[", source)
            self.assertIn("const real_t *const current = state ? state : impl_->current;", wrapper)
            self.assertIn("static constexpr ptrdiff_t FIELD_STRIDE = 2;", wrapper)
            self.assertIn("FIELD_STRIDE, p_w_data, p_c_data, rowptr, colidx, values", wrapper)
            self.assertNotIn("TENSOR_SHAPE_INDEX", source)
            self.assertNotIn("STREAM_SHAPE_ORDER", source)

    def test_matrix_format_plan_specializes_tensor_product_mixed_taylor_hood_blocks(self):
        matrix_plan = gen.matrix_format_plan_from_request(("crs",), ("standard",))
        stage, plan = _generation_plan(navier_stokes, "HEX27_HEX8", matrix_plan)
        units = {
            unit.name: unit
            for unit in plan.emission_kernels_for_context(stage.element_contexts[0])
        }

        monolithic = units["navier_stokes"].matrix_format_plan.variants[0]
        self.assertEqual(monolithic.row_dofs_per_element, 89)
        self.assertEqual(monolithic.column_dofs_per_element, 89)
        self.assertEqual(monolithic.entries_per_element, 7921)

        velocity_block = units["navier_stokes_form_2_u_u"].matrix_format_plan.variants[0]
        self.assertEqual(velocity_block.row_dofs_per_element, 81)
        self.assertEqual(velocity_block.column_dofs_per_element, 81)
        self.assertEqual(velocity_block.entries_per_element, 6561)

        pressure_velocity_block = units["navier_stokes_form_2_p_u"].matrix_format_plan.variants[0]
        self.assertEqual(pressure_velocity_block.row_dofs_per_element, 8)
        self.assertEqual(pressure_velocity_block.column_dofs_per_element, 81)
        self.assertEqual(pressure_velocity_block.entries_per_element, 648)

    def test_matrix_format_plan_rejects_missing_mixed_field_element_mapping(self):
        matrix_plan = gen.matrix_format_plan_from_request(("crs",), ("standard",))
        broken_element = gen.SfemCompatibleElement(
            "BROKEN_TRI6_TRI3",
            "TRI6",
            (("pressure", "TRI3"),),
        )

        stage = gen.UserInputStage.create(
            navier_stokes,
            (broken_element,),
            gen.DEFAULT_VECTOR_SIZE,
            None,
            matrix_plan,
        )
        with self.assertRaisesRegex(ValueError, "missing matrix field-element mapping"):
            plan = gen.SpecializedFormManipulationStage(stage, gen._evaluate_forms(stage)).run()
            plan.emission_kernels_for_context(stage.element_contexts[0])

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
            operator_source = (Path(out_dir) / "d2/tri3/laplace_tri3_operator.cpp").read_text()
            header = (Path(out_dir) / "matrix_formats.hpp").read_text()
            c_abi_header = (Path(out_dir) / "op/sfem_GeneratedLaplace_c_abi.hpp").read_text()
            self.assertIn("laplace_tri3_crs_standard_matrix_assembly_diagnostics_data", source)
            self.assertIn("laplace_tri3_hessian_crs_isoparametric_mesh_soa", operator_source)
            self.assertIn("laplace_tri3_hessian_crs_packed_one_pass_isoparametric_mesh_soa", operator_source)
            self.assertIn("laplace_tri3_hessian_crs_packed_two_pass_isoparametric_mesh_soa", operator_source)
            self.assertIn("laplace_tri3_hessian_bsr_isoparametric_mesh_soa", operator_source)
            self.assertIn("laplace_tri3_hessian_dia_isoparametric_mesh_soa", operator_source)
            self.assertIn("laplace_tri3_hessian_coo_triplet_isoparametric_mesh_soa", operator_source)
            self.assertIn("laplace_d2_simplex_jacobian_action_block_contiguous", operator_source)
            self.assertIn("laplace_tri3_hessian_crs_isoparametric_mesh_soa_scatter_crs", operator_source)
            self.assertIn("laplace_tri3_hessian_crs_isoparametric_mesh_soa_packed_global_node", operator_source)
            self.assertIn("laplace_tri3_hessian_crs_isoparametric_mesh_soa_discover_packed_crs_entries", operator_source)
            self.assertIn("laplace_tri3_hessian_crs_isoparametric_mesh_soa_scatter_packed_crs_entries", operator_source)
            self.assertIn("packed_element_entries[element * N_SHAPE * N_SHAPE]", operator_source)
            self.assertIn("const int graph_status = sfem::codegen::laplace_tri3_hessian_crs_isoparametric_mesh_soa_packed_discover_impl", operator_source)
            self.assertIn("laplace_tri3_hessian_dia_isoparametric_mesh_soa_scatter_dia", operator_source)
            self.assertIn("laplace_tri3_hessian_coo_triplet_isoparametric_mesh_soa_scatter_coo_triplets", operator_source)
            self.assertIn("bool valid_graph = true;", operator_source)
            self.assertIn("bool valid_diagonal_offsets = true;", operator_source)
            self.assertIn("missing diagonal offset", operator_source)
            self.assertIn("return invalid_matrix_graph ? SFEM_FAILURE : SFEM_SUCCESS;", operator_source)
            self.assertIn(
                "laplace_tri3_hessian_crs_isoparametric_mesh_soa_impl<double>",
                operator_source,
            )
            self.assertIn("rows[entry] = global_row;", operator_source)
            self.assertIn("cols[entry] = ev[j];", operator_source)
            self.assertIn("values[entry] = element_matrix[i * N_SHAPE + j];", operator_source)
            packed_fill_begin = operator_source.index(
                "laplace_tri3_hessian_crs_isoparametric_mesh_soa_packed_fill_impl"
            )
            packed_fill_end = operator_source.index(
                'extern "C" int laplace_tri3_hessian_crs_isoparametric_mesh_soa',
                packed_fill_begin,
            )
            packed_fill = operator_source[packed_fill_begin:packed_fill_end]
            self.assertIn("sfem::codegen::thread_scratch<scalar_t>", packed_fill)
            self.assertNotIn("std::malloc", packed_fill)
            self.assertNotIn("std::free", packed_fill)
            self.assertIn("laplace_d2_simplex_jacobian_action_block_contiguous", packed_fill)
            self.assertIn("scatter_packed_crs_entries(element_matrix, entries, values);", packed_fill)
            self.assertNotIn("find_col", packed_fill)
            self.assertIn("laplace_tri3_bsr_packed_one_pass_matrix_assembly_diagnostics_data", source)
            self.assertIn("laplace_tri3_dia_packed_two_pass_matrix_assembly_diagnostics_data", source)
            self.assertIn("laplace_tri3_patch_standard_matrix_assembly_diagnostics_data", source)
            self.assertIn('extern "C" int laplace_tri3_matrix_assembly_variant_count()', source)
            self.assertIn('extern "C" const sfem_MatrixAssemblyDiagnostics *laplace_tri3_matrix_assembly_variant', source)
            self.assertIn("MatrixAssemblyDiagnostics_arithmetic_intensity", header)
            self.assertIn("struct sfem_MatrixAssemblyDiagnostics", header)
            self.assertIn("const char *assembly_kind", header)
            self.assertIn("const char *mesh_access", header)
            self.assertIn("const char *index_policy", header)
            self.assertIn("const char *structural_compatibility", header)
            self.assertIn("const char *reduction_policy", header)
            self.assertIn("int block_size", header)
            self.assertIn('#include "../matrix_formats.hpp"', c_abi_header)
            self.assertIn("laplace_tri3_matrix_assembly_variant_count", c_abi_header)
            self.assertIn("laplace_hessian_crs_2d_isoparametric_mesh_soa", c_abi_header)
            self.assertIn("laplace_hessian_crs_packed_one_pass_2d_isoparametric_mesh_soa", c_abi_header)
            self.assertIn("laplace_hessian_crs_packed_two_pass_2d_isoparametric_mesh_soa", c_abi_header)
            self.assertIn("uint16_t **const SFEM_RESTRICT elements", c_abi_header)
            self.assertIn("laplace_hessian_bsr_2d_isoparametric_mesh_soa", c_abi_header)
            self.assertIn("laplace_hessian_dia_2d_isoparametric_mesh_soa", c_abi_header)
            self.assertIn("laplace_hessian_coo_triplet_2d_isoparametric_mesh_soa", c_abi_header)
            self.assertIn("const smesh::ElemType element_type", c_abi_header)
            self.assertNotIn("extern \"C\" int laplace_tri3_hessian_crs_isoparametric_mesh_soa", c_abi_header)
            self.assertIn('"rowptr_colidx"', source)
            self.assertIn('"FunctionSpace::PackedMesh"', source)
            self.assertIn('"diagonal_offsets"', source)
            self.assertIn('"rowidx_colidx"', source)
            self.assertIn('"rowptr_colidx"', source)
            self.assertIn('"stable_simplex_affine_diagonal_offsets"', source)
            self.assertIn('"deterministic_element_order_external_reduction"', source)
            self.assertIn('"requires_full_graph"', source)

            plan = Path(result.plan_dump).read_text()
            self.assertIn('"matrix_format_plan"', plan)
            self.assertIn('"schema": "sfem.matrix_format_plan"', plan)
            self.assertIn('"schema_version": 3', plan)
            self.assertIn('"format": "crs"', plan)
            self.assertIn('"mesh_layout": "packed"', plan)
            self.assertIn('"packed_pass": "two_pass"', plan)
            self.assertIn('"mesh_access": "FunctionSpace::PackedMesh"', plan)
            self.assertIn('"element_connectivity": "packed->elements(block)->data()"', plan)
            self.assertIn('"pack_index_type": "FunctionSpace::PackedIdxType"', plan)
            self.assertIn('"pack_partition": "n_packs/n_elements_per_pack/max_nodes_per_pack"', plan)
            self.assertIn('"packed_node_partition": "owned_nodes_ptr/n_shared/ghost_ptr/ghost_idx"', plan)
            self.assertIn('"value_mapping": "PackedMesh::map_to_packed/map_to_unpacked"', plan)
            self.assertIn('"assembly_plan"', plan)
            self.assertIn('"structural_compatibility"', plan)
            self.assertIn('"reduction_policy"', plan)
            self.assertIn('"sort_policy": "external_stable_sort_or_existing_sfem_coo_reduce"', plan)
            self.assertIn('"reduction_phase": "non_hot_setup_phase"', plan)
            self.assertIn('"kind": "crs"', plan)
            self.assertIn('"kind": "bsr"', plan)
            self.assertIn('"kind": "dia"', plan)
            self.assertIn('"kind": "coo"', plan)
            self.assertIn('"kind": "patch"', plan)

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
            self.assertTrue(_manifest_runtime_variants(manifest, "hessian_crs"))
            hessian_crs_functions = {
                variant["function"]
                for variant in _manifest_runtime_variants(manifest, "hessian_crs")
            }
            self.assertIn(
                "laplace_hessian_crs_packed_one_pass_2d_isoparametric_mesh_soa",
                hessian_crs_functions,
            )
            self.assertIn(
                "laplace_hessian_crs_packed_two_pass_2d_isoparametric_mesh_soa",
                hessian_crs_functions,
            )
            self.assertTrue(_manifest_runtime_variants(manifest, "hessian_bsr"))
            self.assertTrue(_manifest_runtime_variants(manifest, "hessian_dia"))
            self.assertTrue(_manifest_runtime_variants(manifest, "hessian_coo_triplet"))
            wrapper = (Path(out_dir) / "op/sfem_GeneratedLaplace.cpp").read_text()
            wrapper_header = (Path(out_dir) / "op/sfem_GeneratedLaplace.hpp").read_text()
            self.assertIn("GeneratedLaplace::hessian_crs", wrapper)
            self.assertIn("GeneratedLaplace::hessian_bsr", wrapper)
            self.assertIn("GeneratedLaplace::hessian_dia", wrapper)
            self.assertIn("laplace_hessian_crs_2d_isoparametric_mesh_soa", wrapper)
            self.assertIn("laplace_hessian_bsr_2d_isoparametric_mesh_soa", wrapper)
            self.assertIn("laplace_hessian_dia_2d_isoparametric_mesh_soa", wrapper)
            hessian_crs_body = wrapper[
                wrapper.index("int GeneratedLaplace::hessian_crs") :
                wrapper.index("int GeneratedLaplace::hessian_bsr")
            ]
            hessian_bsr_body = wrapper[
                wrapper.index("int GeneratedLaplace::hessian_bsr") :
                wrapper.index("int GeneratedLaplace::hessian_dia")
            ]
            hessian_dia_body = wrapper[
                wrapper.index("int GeneratedLaplace::hessian_dia") :
                wrapper.index("int GeneratedLaplace::value")
            ]
            self.assertNotIn("switch (domain.element_type)", hessian_crs_body)
            self.assertNotIn("switch (domain.element_type)", hessian_bsr_body)
            self.assertNotIn("switch (domain.element_type)", hessian_dia_body)
            for removed_query in (
                "n_matrix_format_variants",
                "matrix_format_variant",
                "supports_matrix_format",
            ):
                self.assertNotIn(removed_query, wrapper)
                self.assertNotIn(removed_query, wrapper_header)
                self.assertNotIn(removed_query, c_abi_header)
                self.assertNotIn(removed_query, json.dumps(manifest))

    def test_matrix_format_hessian_c_abi_is_manifest_runtime_metadata(self):
        with tempfile.TemporaryDirectory() as out_dir:
            result = gen.generate(
                neohookean_ogden,
                out_dir,
                elements=("TRI3",),
                clean=True,
                matrix_formats=("crs", "bsr", "dia", "coo", "patch"),
                matrix_mesh_layouts=("standard", "packed"),
                matrix_packed_passes=("one_pass", "two_pass"),
            )

            manifest_path = Path(out_dir) / "op/sfem_GeneratedNeoHookeanOgden_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            files = gen.generate_op_registration_files([manifest])
            self.assertIn("register_GeneratedNeoHookeanOgden_generated_op();", files["sfem_generated_ops_registration.cpp"])

            expected_operations = (
                "hessian_crs",
                "hessian_bsr",
                "hessian_dia",
                "hessian_coo",
                "hessian_coo_triplet",
                "hessian_patch",
            )
            c_abi_names = {entry["name"] for entry in manifest["c_abi"]}
            for operation in expected_operations:
                variants = _manifest_runtime_variants(manifest, operation)
                self.assertTrue(variants)
                self.assertTrue(
                    any(
                        variant["variant"] == "isoparametric"
                        and variant["scalar_type"] == "real_t"
                        and variant["function"] in c_abi_names
                        for variant in variants
                    )
                )

            operator_source = (
                Path(out_dir)
                / "d2"
                / "tri3"
                / "neohookean_ogden_tri3_operator.cpp"
            ).read_text()
            for matrix_format in ("bsr", "dia", "patch"):
                self.assertNotIn("_%s_apply_" % matrix_format, operator_source)
            self.assertIn(
                'extern "C" int neohookean_ogden_tri3_apply_isoparametric_mesh_soa(',
                operator_source,
            )
            packed_apply_begin = operator_source.index(
                "neohookean_ogden_tri3_apply_packed_isoparametric_mesh_soa"
            )
            packed_apply_end = operator_source.index(
                'extern "C" int neohookean_ogden_tri3_apply_packed_isoparametric_mesh_soa_float',
                packed_apply_begin,
            )
            packed_apply = operator_source[packed_apply_begin:packed_apply_end]
            self.assertIn("sfem::codegen::thread_scratch<scalar_t>", packed_apply)
            self.assertNotIn("std::malloc", packed_apply)
            self.assertNotIn("std::free", packed_apply)
            self.assertIn(
                "neohookean_ogden_tri3_hessian_crs_packed_one_pass_isoparametric_mesh_soa",
                operator_source,
            )
            self.assertIn(
                "neohookean_ogden_tri3_hessian_crs_packed_two_pass_isoparametric_mesh_soa",
                operator_source,
            )
            self.assertIn(
                "neohookean_ogden_tri3_hessian_isoparametric_mesh_soa_packed_global_node",
                operator_source,
            )
            self.assertIn(
                "neohookean_ogden_tri3_hessian_isoparametric_mesh_soa_scatter_packed_crs_entries",
                operator_source,
            )
            packed_fill_begin = operator_source.index(
                "neohookean_ogden_tri3_hessian_isoparametric_mesh_soa_packed_fill_impl"
            )
            packed_fill_end = operator_source.index(
                'extern "C" int neohookean_ogden_tri3_hessian_crs_isoparametric_mesh_soa',
                packed_fill_begin,
            )
            packed_fill = operator_source[packed_fill_begin:packed_fill_end]
            self.assertIn("sfem::codegen::thread_scratch<scalar_t>", packed_fill)
            self.assertNotIn("std::malloc", packed_fill)
            self.assertNotIn("std::free", packed_fill)
            self.assertIn("neohookean_ogden_d2_simplex_tri3_apply_block", packed_fill)
            self.assertIn("scatter_packed_crs_entries(element_matrix, entries, values);", packed_fill)
            self.assertNotIn("find_col", packed_fill)
            hessian_crs_functions = {
                variant["function"]
                for variant in _manifest_runtime_variants(manifest, "hessian_crs")
            }
            self.assertIn(
                "neohookean_ogden_hessian_crs_packed_one_pass_2d_isoparametric_mesh_soa",
                hessian_crs_functions,
            )
            self.assertIn(
                "neohookean_ogden_hessian_crs_packed_two_pass_2d_isoparametric_mesh_soa",
                hessian_crs_functions,
            )

            broken = json.loads(json.dumps(manifest))
            hessian_crs = _manifest_runtime_variants(broken, "hessian_crs")
            hessian_crs[0]["function"] = "missing_hessian_crs_c_abi"
            with self.assertRaisesRegex(ValueError, "not declared in c_abi"):
                gen.generate_op_registration_files([broken])

    def test_matrix_format_user_documentation_covers_cli_and_api_requests(self):
        doc = (
            Path(__file__).resolve().parents[1]
            / "docs"
            / "matrix_formats.md"
        ).read_text()

        self.assertIn("--matrix-format crs", doc)
        self.assertIn("--matrix-format all", doc)
        self.assertIn("--matrix-layout all", doc)
        self.assertIn("matrix_formats=(\"crs\", \"bsr\", \"dia\", \"coo\", \"patch\")", doc)
        self.assertIn("matrix_formats=\"crs,bsr,dia,coo,patch\"", doc)
        self.assertIn("matrix_format_benchmark_report", doc)
        self.assertIn("--elapsed-seconds", doc)
        self.assertIn("achieved GFLOP/s", doc)

    def test_matrix_format_benchmark_report_reads_plan_dump(self):
        with tempfile.TemporaryDirectory() as out_dir:
            result = gen.generate(
                laplace,
                out_dir,
                elements=("TRI3",),
                clean=True,
                dump_plan=True,
                matrix_formats=("crs",),
            )

            plan = json.loads(Path(result.plan_dump).read_text())
            rows = list(
                matrix_format_benchmark_report.iter_matrix_format_rows(
                    result.plan_dump,
                    plan,
                    nelements=7,
                    elapsed_seconds=0.014,
                    repeat=2,
                )
            )
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["format"], "crs")
            self.assertEqual(row["assembly_kind"], "crs")
            self.assertEqual(row["index_policy"], "rowptr_colidx")
            self.assertEqual(row["value_layout"], "scalar_element_matrix")
            self.assertEqual(row["accumulation_policy"], "add_scatter")
            self.assertEqual(row["structural_compatibility"], "requires_full_graph")
            self.assertEqual(row["reduction_policy"], "atomic_add")
            self.assertEqual(row["nelements"], 7)
            self.assertEqual(row["repeat"], 2)
            self.assertAlmostEqual(float(row["seconds_per_call"]), 0.007)
            self.assertGreater(float(row["total_flops"]), 0.0)
            self.assertGreater(int(row["total_bytes"]), 0)
            self.assertGreater(float(row["arithmetic_intensity"]), 0.0)
            self.assertGreater(float(row["bandwidth_gb_s"]), 0.0)
            self.assertGreater(float(row["achieved_gflop_s"]), 0.0)

            output = io.StringIO()
            matrix_format_benchmark_report.write_csv(rows, output)
            output.seek(0)
            parsed = list(csv.DictReader(output))
            self.assertEqual(parsed[0]["format"], "crs")
            self.assertEqual(parsed[0]["structural_compatibility"], "requires_full_graph")
            self.assertIn("achieved_gflop_s", parsed[0])

    def test_clean_output_regenerates_all_maintained_materials_with_matrix_formats(self):
        for name, material, elements in self.MAINTAINED_MATRIX_FORMAT_MATERIALS:
            with self.subTest(material=name), tempfile.TemporaryDirectory() as out_dir:
                result = gen.generate(
                    material,
                    out_dir,
                    elements=elements,
                    clean=True,
                    dump_plan=True,
                    matrix_formats=("crs", "bsr", "dia", "coo", "patch"),
                )

                source_names = {os.path.relpath(path, out_dir) for path in result.sources}
                self.assertIn("matrix_formats.hpp", source_names)
                matrix_sources = sorted(
                    source
                    for source in source_names
                    if source.endswith("_matrix_format_operator.cpp")
                )
                self.assertTrue(matrix_sources)

                manifest = json.loads(
                    (
                        Path(out_dir)
                        / "op"
                        / ("sfem_%s_manifest.json" % material.op_name)
                    ).read_text()
                )
                self.assertTrue(manifest["matrix_formats"])
                self.assertEqual(
                    {entry["source"] for entry in manifest["matrix_formats"]},
                    set(matrix_sources),
                )
                if name in (
                    "neohookean_ogden",
                    "mooney_rivlin",
                    "two_phase_flow",
                    "stokes",
                    "poro_hyperelasticity",
                ):
                    self.assertTrue(_manifest_runtime_variants(manifest, "hessian_coo_triplet"))
                if name == "two_phase_flow":
                    self.assertEqual(
                        10,
                        len(_manifest_runtime_variants(manifest, "hessian_coo_triplet")),
                    )

                dump = json.loads(Path(result.plan_dump).read_text())
                variants = [
                    variant
                    for kernel in dump["kernels"]
                    for variant in (kernel.get("matrix_format_plan") or {}).get("variants", ())
                ]
                self.assertTrue(variants)
                self.assertEqual(
                    {"crs", "bsr", "dia", "coo", "patch"},
                    {variant["format"] for variant in variants},
                )
                for variant in variants:
                    self.assertGreater(variant["row_dofs_per_element"], 0)
                    self.assertGreater(variant["column_dofs_per_element"], 0)
                    self.assertGreater(variant["entries_per_element"], 0)
                    self.assertGreater(variant["expected_bytes_per_element"], 0)


if __name__ == "__main__":
    unittest.main()
