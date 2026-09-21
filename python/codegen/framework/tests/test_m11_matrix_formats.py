import csv
import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from sfem import gen

from codegen.framework.materials.laplace import material as laplace
from codegen.framework.materials.linear_elasticity import material as linear_elasticity
from codegen.framework.materials.navier_stokes import material as navier_stokes
from codegen.framework.materials.neohookean_ogden import material as neohookean_ogden
from codegen.framework.materials.neumann import material as neumann
from codegen.framework.materials.neumann_general import material as neumann_general
from codegen.framework.materials.two_phase_flow import material as two_phase_flow
from codegen.framework.plans.matrix_formats import (
    BlockDiagSymAssemblyPlan,
    BSRAssemblyPlan,
    CRSAssemblyPlan,
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
    next_function = source.find("template <typename s_t>", begin + len(signature))
    if next_function < 0:
        return source[begin:]
    return source[begin:next_function]


class M11MatrixFormatAssemblyTest(unittest.TestCase):
    MAINTAINED_MATRIX_FORMAT_MATERIALS = (
        ("neohookean_ogden", neohookean_ogden, ("TRI3",)),
        ("two_phase_flow", two_phase_flow, ("TRI3",)),
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
            "neohookean_ogden_proteus_hex8_hessian_i_msoa_assemble_impl"
        )
        hessian_end = source.index(
            'extern "C" int neohookean_ogden_proteus_hex8_hessian_bsr_i_msoa',
            hessian_begin,
        )
        hessian_source = source[hessian_begin:hessian_end]

        self.assertNotIn("TENSOR_SHAPE_INDEX", hessian_source)
        self.assertNotIn("STREAM_SHAPE_ORDER", hessian_source)
        # The assembly forms the element matrix directly.  It used to call the
        # apply once per trial degree of freedom and keep a column, which is
        # twenty-four applies per element for a HEX8; the apply block belongs to
        # the apply and must not appear in an assembly again.
        self.assertNotIn(
            "neohookean_ogden_d3_tensor_product_apply_block<s_t, NQ, NS, VS>",
            hessian_source,
        )
        self.assertIn(
            "neohookean_ogden_d3_tensor_product_direct_hessian_tensor_product_element_matrix<s_t, NQ, NS, VS>",
            hessian_source,
        )
        self.assertIn(
            "neohookean_ogden_d3_tensor_product_apply_block<s_t, NQ, NS, VS>",
            source,
        )
        self.assertIn("static constexpr int VS = 1;", hessian_source)
        self.assertNotIn("ordered_shape_index", hessian_source)
        self.assertNotIn("matrix_coordinate_streams", hessian_source)
        self.assertNotIn("bcoordinate_streams", hessian_source)
        self.assertNotIn("coordinate_value", hessian_source)
        self.assertIn(
            "tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>",
            hessian_source,
        )
        self.assertIn("isoparametric_grad_1d, bcoordinate_data,", hessian_source)
        self.assertNotIn("neohookean_ogden_proteus_hex8_hessian_crs_i_msoa", source)
        self.assertIn("neohookean_ogden_proteus_hex8_hessian_bsr_i_msoa", source)
        bsr_scatter = _static_function_body(
            source,
            "static SFEM_INLINE void neohookean_ogden_proteus_hex8_hessian_i_msoa_scatter_bsr",
        )
        self.assertIn("count_t entries[NS * NS];", bsr_scatter)
        # The scatter no longer reports a malformed graph: it returns void, and
        # the graph is settled once at Op::initialize instead of being retested
        # per element.  What is pinned here is that the check has not crept back
        # into the element loop.
        self.assertNotIn("valid_block_graph", bsr_scatter)
        self.assertNotIn("missing block graph entry", bsr_scatter)
        self.assertIn("neohookean_ogden_proteus_hex8_hessian_i_msoa_find_cols(ev, cols, lenrow, ks);", bsr_scatter)
        self.assertIn("entries[i * NS + j] = row_begin + ks[j];", bsr_scatter)
        self.assertIn("s_t *const block = &values[entries[i * NS + j] * NC * NC];", bsr_scatter)
        self.assertIn("block[bi * NC + bj] += element_matrix[row * (NC * NS) + col];", bsr_scatter)
        self.assertLess(
            bsr_scatter.index("entries[i * NS + j] = row_begin + ks[j];"),
            bsr_scatter.index("s_t *const block = &values[entries[i * NS + j] * NC * NC];"),
        )
        self.assertNotIn("std::vector", bsr_scatter)
        # The requested matrix format is settled at compile time: the kernel
        # instantiates one scatter and refuses the instantiation outright when
        # FORMAT names a scatter it does not have.  No runtime flag, and above
        # all no reduction clause on the element loop.
        self.assertIn(
            'static_assert(FORMAT == 1,\n'
            '                "this kernel has no scatter for the requested matrix format");',
            source,
        )
        self.assertNotIn("unsupported_matrix_format", source)
        self.assertNotIn("reduction(|:", source)
        self.assertIn("values[entries[i * NS + j] * NC * NC]", source)
        for matrix_format in ("crs", "dia", "coo", "patch"):
            self.assertNotIn(
                "neohookean_ogden_proteus_hex8_hessian_i_msoa_scatter_%s"
                % matrix_format,
                source,
            )
            self.assertNotIn(
                "neohookean_ogden_proteus_hex8_hessian_%s_i_msoa"
                % matrix_format,
                source,
            )
        self.assertNotIn(
            "neohookean_ogden_proteus_hex8_hessian_coo_triplet_i_msoa",
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
        with tempfile.TemporaryDirectory() as out_dir:
            gen.generate(
                neohookean_ogden,
                out_dir,
                elements=("HEX8", "TET4"),
                clean=True,
                matrix_formats=("crs", "bsr"),
            )
            c_abi_header = (
                Path(out_dir)
                / "op"
                / "sfem_GeneratedNeoHookeanOgden_c_abi.hpp"
            ).read_text()
            self.assertIn("neohookean_ogden_hessian_crs_3d_i_msoa", c_abi_header)
            self.assertIn("const smesh::ElemType element_type", c_abi_header)
            self.assertNotIn("neohookean_ogden_hex8_hessian_coo_triplet_i_msoa", c_abi_header)
            self.assertNotIn("neohookean_ogden_tet4_hessian_crs_i_msoa", c_abi_header)

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

                    assembly_base = "%s_tet4_hessian_i_msoa" % material_name
                    public_name = "%s_hessian_block_diag_sym_3d_i_msoa" % material_name
                    self.assertIn("%s_scatter_block_diag_sym" % assembly_base, source)
                    self.assertIn("static constexpr int SYM_DIM = (NC * (NC + 1)) / 2;", source)
                    self.assertIn("values[(ptrdiff_t)ev[i] * SYM_DIM]", source)
                    self.assertIn("for (int bj = bi; bj < NC; ++bj)", source)
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
                        # A linear material's own tangent does not depend on the
                        # state, but the method forwards that state to the time
                        # scheme's term before assembling, so the parameter is
                        # named and genuinely read.  What the rule was always
                        # about is still checked: a name with a `(void)x;` to
                        # apologise for it would mean nothing reads it.
                        self.assertIn(
                            "hessian_block_diag_sym(const real_t *const x,",
                            block_diag_method,
                        )
                        self.assertIn("term->hessian_block_diag_sym(x, values)", block_diag_method)
                        self.assertNotIn("(void)x;", block_diag_method)
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
            ("crs", "bsr", "block_diag_sym"),
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
        self.assertEqual(len(variants), 9)
        self.assertIn("crs_standard", variants)
        self.assertIn("bsr_packed_one_pass", variants)
        self.assertIn("block_diag_sym_packed_two_pass", variants)

        for variant in variants.values():
            self.assertEqual(variant.row_dofs_per_element, 3)
            self.assertEqual(variant.column_dofs_per_element, 3)
            self.assertEqual(variant.entries_per_element, 9)
            self.assertGreater(variant.expected_flops_per_element, 0)
            self.assertGreater(variant.expected_bytes_per_element, 0)

        removed_apply_key = "format" + "_aware" + "_apply"
        self.assertNotIn(removed_apply_key, variants["crs_standard"].to_dict())
        self.assertNotIn(removed_apply_key, variants["bsr_standard"].to_dict())
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

    def test_bsr_coverage_includes_vector_action_and_taylor_hood_velocity_plan(self):
        matrix_test_source = (
            Path(__file__).resolve().parents[4]
            / "frontend"
            / "tests"
            / "sfem_MatrixFromatsTest.cpp"
        ).read_text()
        self.assertIn('sfem::create_op(space, "GeneratedNeoHookeanOgden"', matrix_test_source)
        self.assertIn("neohookean_ogden_apply_packed_3d_i_msoa", matrix_test_source)
        self.assertIn('assert_close_action("generated NeoHookean packed hessian action"', matrix_test_source)
        frontend_api_source = (
            Path(__file__).resolve().parents[4] / "frontend" / "sfem_API.hpp"
        ).read_text()
        self.assertIn("#include \"sfem_DIA.hpp\"", frontend_api_source)


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
            self.assertEqual(len(_manifest_runtime_variants(manifest, "hessian_crs")), 5)
            self.assertEqual(len(_manifest_runtime_variants(manifest, "hessian_bsr")), 5)

            c_abi_header = (Path(out_dir) / "op/sfem_GeneratedTwoPhaseFlow_c_abi.hpp").read_text()
            self.assertIn(
                "two_phase_flow_form_2_p_w_p_w_hessian_bsr_2d_i_msoa",
                c_abi_header,
            )
            declaration_begin = c_abi_header.index(
                "extern \"C\" int two_phase_flow_form_2_p_w_p_w_hessian_bsr_2d_i_msoa"
            )
            declaration_end = c_abi_header.index(");", declaration_begin)
            bsr_declaration = c_abi_header[declaration_begin:declaration_end]
            self.assertNotIn("const ptrdiff_t direction_stride", bsr_declaration)

            source = (
                Path(out_dir)
                / "d2/tri3/two_phase_flow_form_2_p_w_p_w_tri3_operator.cpp"
            ).read_text()
            wrapper = (Path(out_dir) / "op/sfem_GeneratedTwoPhaseFlow.cpp").read_text()
            self.assertIn("static constexpr int NC = 2;", source)
            self.assertIn("static constexpr int N_ROW_STREAMS = 3;", source)
            self.assertIn("static constexpr int N_COL_STREAMS = 3;", source)
            self.assertIn("bcurrent[0 * NS + shape][0] = p_w[node * current_stride];", source)
            self.assertIn("bcurrent[1 * NS + shape][0] = p_c[node * current_stride];", source)
            self.assertIn("block[bi * NC + bj] += element_matrix[row_stream * N_COL_STREAMS + col_stream];", source)
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
        """The requested formats generate, compile, and publish a C ABI.

        This used to pin about forty internal kernel names and in-kernel
        variables -- `valid_graph`, `invalid_matrix_graph`, the exact index
        arithmetic of the packed entry table.  Every one of them was an
        implementation detail the emitters are free to change, and every one of
        them broke when they did, which is the opposite of what a test should
        do: it made the emitted text harder to improve without telling anyone
        whether the operator still worked.  What is pinned now is the contract
        a caller can actually depend on -- the files that appear, the public C
        ABI entry points for the formats that were asked for, and that the whole
        thing compiles.  Kernel bodies are the snapshot gate's job.
        """
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
                matrix_formats=("crs", "bsr", "block_diag_sym"),
                matrix_mesh_layouts=("standard", "packed"),
                matrix_packed_passes=("one_pass", "two_pass"),
            )
            names = {os.path.relpath(path, out_dir) for path in result.sources}
            self.assertIn("matrix_formats.hpp", names)
            self.assertIn("d2/tri3/laplace_tri3_matrix_format_operator.cpp", names)

            c_abi_header = (Path(out_dir) / "op/sfem_GeneratedLaplace_c_abi.hpp").read_text()
            for matrix_format in ("crs", "bsr", "block_diag_sym"):
                with self.subTest(matrix_format=matrix_format):
                    self.assertIn(
                        "laplace_hessian_%s_2d_i_msoa" % matrix_format,
                        c_abi_header,
                    )
            # The formats that were removed must not come back by accident.
            for matrix_format in ("dia", "coo", "coo_triplet", "patch"):
                with self.subTest(removed=matrix_format):
                    self.assertNotIn("hessian_%s" % matrix_format, c_abi_header)

            plan = json.loads(Path(result.plan_dump).read_text())
            self.assertTrue(json.dumps(plan))

    def test_matrix_format_hessian_c_abi_is_manifest_runtime_metadata(self):
        with tempfile.TemporaryDirectory() as out_dir:
            result = gen.generate(
                neohookean_ogden,
                out_dir,
                # TET10: a simplex, so the packed internals this checks are the
                # same ones, and not constant-P1, so it still publishes an
                # isoparametric kernel for the ABI assertions below.
                elements=("TET10",),
                clean=True,
                matrix_formats=("crs", "bsr", "block_diag_sym"),
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
                / "d3"
                / "tet10"
                / "neohookean_ogden_tet10_operator.cpp"
            ).read_text()
            for matrix_format in ("bsr",):
                self.assertNotIn("_%s_apply_" % matrix_format, operator_source)
            self.assertIn(
                'extern "C" int neohookean_ogden_tet10_apply_i_msoa(',
                operator_source,
            )
            packed_apply_begin = operator_source.index(
                "neohookean_ogden_tet10_apply_packed_i_msoa"
            )
            # The next entry point, whatever it is: the packed apply used to be
            # delimited by its own `_float` twin, and there is no twin now.
            packed_apply_end = operator_source.index(
                '\nextern "C" ',
                operator_source.index("{", packed_apply_begin),
            )
            packed_apply = operator_source[packed_apply_begin:packed_apply_end]
            self.assertIn("sfem::codegen::thread_scratch<s_t>", packed_apply)
            self.assertNotIn("std::malloc", packed_apply)
            self.assertNotIn("std::free", packed_apply)
            self.assertIn(
                "neohookean_ogden_tet10_hessian_crs_packed_one_pass_i_msoa",
                operator_source,
            )
            self.assertIn(
                "neohookean_ogden_tet10_hessian_crs_packed_two_pass_i_msoa",
                operator_source,
            )
            self.assertIn(
                "neohookean_ogden_tet10_hessian_i_msoa_packed_global_node",
                operator_source,
            )
            self.assertIn(
                "neohookean_ogden_tet10_hessian_i_msoa_scatter_packed_crs_entries",
                operator_source,
            )
            packed_fill_begin = operator_source.index(
                "neohookean_ogden_tet10_hessian_i_msoa_packed_fill_impl"
            )
            packed_fill_end = operator_source.index(
                'extern "C" int neohookean_ogden_tet10_hessian_crs_i_msoa',
                packed_fill_begin,
            )
            packed_fill = operator_source[packed_fill_begin:packed_fill_end]
            self.assertIn("sfem::codegen::thread_scratch<s_t>", packed_fill)
            self.assertNotIn("std::malloc", packed_fill)
            self.assertNotIn("std::free", packed_fill)
            # Which local kernel the packed fill reuses is the closed-form
            # simplex path's business and TET10 does not take it; what this
            # test is about is the C ABI and the manifest.
            self.assertIn("scatter_packed_crs_entries(element_matrix, entries, values);", packed_fill)
            self.assertNotIn("find_col", packed_fill)
            hessian_crs_functions = {
                variant["function"]
                for variant in _manifest_runtime_variants(manifest, "hessian_crs")
            }
            self.assertIn(
                "neohookean_ogden_hessian_crs_packed_one_pass_3d_i_msoa",
                hessian_crs_functions,
            )
            self.assertIn(
                "neohookean_ogden_hessian_crs_packed_two_pass_3d_i_msoa",
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
        self.assertIn("matrix_formats=(\"crs\", \"bsr\", \"block_diag_sym\")", doc)
        self.assertIn("matrix_formats=\"crs,bsr,block_diag_sym\"", doc)
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
                    matrix_formats=("crs", "bsr", "block_diag_sym"),
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

                dump = json.loads(Path(result.plan_dump).read_text())
                variants = [
                    variant
                    for kernel in dump["kernels"]
                    for variant in (kernel.get("matrix_format_plan") or {}).get("variants", ())
                ]
                self.assertTrue(variants)
                self.assertEqual(
                    {"crs", "bsr", "block_diag_sym"},
                    {variant["format"] for variant in variants},
                )
                for variant in variants:
                    self.assertGreater(variant["row_dofs_per_element"], 0)
                    self.assertGreater(variant["column_dofs_per_element"], 0)
                    self.assertGreater(variant["entries_per_element"], 0)
                    self.assertGreater(variant["expected_bytes_per_element"], 0)


class NoProbingAssemblyRatchetTest(unittest.TestCase):
    """An element matrix is computed, not discovered by probing.

    Setting a unit basis vector, applying the operator and keeping the column
    costs one apply per trial degree of freedom -- thirty per element on TET10.
    The energy path was converted and
    `test_generated_direct_hessian_assembly_does_not_call_the_apply_block`
    above forbids its return, but it forbids it by naming one material's apply
    block, so nothing stopped the residual path from keeping the pattern.

    This asserts the shape instead of the name, over the whole shipped tree, so
    a path that acquires an assembly cannot acquire this with it.  The five that
    remain are listed rather than excused: the list is the work left, and it may
    only shrink.

    Read from the shipped tree rather than generated here on purpose.  This is a
    property of what the build compiles, it costs nothing to check, and the tree
    is the artifact `codegen_snapshot check-tree` already holds to be current.
    """

    #: Kernels that still build an element matrix by probing.  Only ever fewer.
    #:
    #: All of them are the Mooney-Rivlin Kelvin-Voigt Newmark *viscous* unit,
    #: which is written as a residual.  The same material's *elastic* unit,
    #: written as an energy, assembles directly on the same elements -- which is
    #: what says this is the path and not the mathematics.
    #:
    #: Empty, and it stays empty.  Every element in the tree now builds its
    #: matrix from the substituted flux, in the shape its own evaluation
    #: strategy asks for: entry by entry with no loop where the element
    #: evaluates in closed form, a quadrature point at a time on a higher-order
    #: simplex, a column at a time through the factorised contraction on a
    #: tensor-product element.
    #:
    #: The list is kept rather than the test simplified to `assertEqual(found,
    #: {})`, because what it says is not "there are none" but "these are the
    #: ones left and they may only get fewer".  A material added tomorrow gets
    #: the same answer with no edit here.
    PROBING = {}

    @staticmethod
    def _shipped_tree():
        """Asked of the tool that owns the question, not recomputed here.

        `codegen_snapshot.shipped_tree()` is what `check-tree` compares against,
        so this reads the same directory the build compiles rather than a second
        path expression that can drift from it -- and a first attempt at one
        promptly did, silently finding nothing.
        """
        from codegen.framework.tools.codegen_snapshot import shipped_tree

        return Path(shipped_tree())

    def _probing_assemblies(self):
        """file -> applies per element, for every assembly built by probing."""
        import re

        unit_direction = re.compile(r"bdirection\[trial\]\[0\] = s_t\(1\);")
        trial_loop = re.compile(
            r"for \(int trial_local = 0; trial_local < (\d+); \+\+trial_local\)"
        )
        found = {}
        for path in sorted(self._shipped_tree().rglob("*_operator.cpp")):
            source = path.read_text()
            if not unit_direction.search(source):
                continue
            bounds = [int(bound) for bound in trial_loop.findall(source)]
            found[path.name] = max(bounds) if bounds else 0
        return found

    def test_no_assembly_probes_for_its_columns(self):
        found = self._probing_assemblies()
        self.assertEqual(
            sorted(found),
            sorted(self.PROBING),
            "the set of assemblies built by probing changed.  An element matrix "
            "is computed directly; if one of these was converted, remove it from "
            "PROBING in this commit, and if a new one appeared, it must not ship",
        )
        for name in sorted(self.PROBING):
            with self.subTest(kernel=name):
                self.assertLessEqual(
                    found[name],
                    self.PROBING[name],
                    "%s now costs %d applies per element, up from %d"
                    % (name, found[name], self.PROBING[name]),
                )
                self.assertEqual(
                    found[name],
                    self.PROBING[name],
                    "%s is down to %d applies per element and the list still "
                    "says %d -- lower it to lock the improvement in"
                    % (name, found[name], self.PROBING[name]),
                )

    def test_the_energy_path_assembles_directly_on_the_same_elements(self):
        """The counter-example, pinned, because it is what makes the case.

        Mooney-Rivlin Kelvin-Voigt Newmark carries both an energy unit and a
        residual one.  If the elastic unit ever started probing too, the
        argument that this is fixable would have quietly stopped being true.
        """
        import re

        direct = re.compile(r"direct_hessian[a-z_]*element_matrix<")
        for path in sorted(self._shipped_tree().rglob("*elastic*_operator.cpp")):
            source = path.read_text()
            if "element_matrix" not in source:
                continue
            with self.subTest(kernel=path.name):
                self.assertTrue(
                    direct.search(source),
                    "%s assembles without calling a direct element-matrix "
                    "kernel" % path.name,
                )
                self.assertNotIn("bdirection[trial][0] = s_t(1);", source)


class ElementMatrixAgreesWithTheApplyTest(unittest.TestCase):
    """The element matrix and the operator it assembles are one bilinear form.

    `A * v == apply(v)` for every `v`.  Probing had that property by
    construction -- the column *was* an apply -- and the substitution has to earn
    it, so it is checked rather than argued.

    It is the gate that matters for the assembly rework, and the reproducibility
    harness is not: that harness drives each material on a HEX8 grid, so a digest
    at `refine=6` says nothing about a TET4 or a TET10 kernel, and its TET10 run
    cannot start at all.  This compares the assembly against the operator it is
    supposed to *be*, rather than against an older copy of itself.

    All four shapes are checked, because they are four pieces of code: TET4
    writes its entries with no loop, TET10 accumulates a quadrature point at a
    time, and the two tensor-product elements build a column at a time through
    the factorised contraction.
    """

    #: What each case needs to instantiate the driver below.  The apply is the
    #: same element's Jacobian-action block at the same rule, which is the only
    #: kernel that can settle the question.
    CASES = (
        {
            'element': 'TET4',
            'header': 'mooney_rivlin_kelvin_voigt_viscous_d3_simplex_local.hpp',
            'dim': 3,
            'n_fields': 3,
            'n_qp': 1,
            'n_shape': 4,
            'assembly': 'mooney_rivlin_kelvin_voigt_viscous_d3_simplex_tet4_hessian_block',
            'apply': 'mooney_rivlin_kelvin_voigt_viscous_d3_simplex_tet4_jacobian_action_block_contiguous',
            'reference': 'ref_tet4_q1',
            'quadrature': 'quad_tet_q1',
            'includes': ('tet4_q1.hpp', 'quad_tet_q1.hpp'),
        },
        {
            'element': 'TET10',
            'header': 'mooney_rivlin_kelvin_voigt_viscous_d3_simplex_local.hpp',
            'dim': 3,
            'n_fields': 3,
            'n_qp': 11,
            'n_shape': 10,
            'assembly': 'mooney_rivlin_kelvin_voigt_viscous_d3_simplex_hessian_block',
            'apply': 'mooney_rivlin_kelvin_voigt_viscous_d3_simplex_jacobian_action_block_contiguous',
            'reference': 'ref_tet10_q11',
            'quadrature': 'quad_tet_q11',
            'includes': ('tet10_q11.hpp', 'quad_tet_q11.hpp'),
        },
        {
            'element': 'PROTEUS_HEX8',
            'header': 'mooney_rivlin_kelvin_voigt_viscous_d3_tensor_product_local.hpp',
            'dim': 3,
            'n_fields': 3,
            'n_qp': 8,
            'n_shape': 8,
            'assembly': 'mooney_rivlin_kelvin_voigt_viscous_d3_tensor_product_hessian_block',
            'apply': 'mooney_rivlin_kelvin_voigt_viscous_d3_tensor_product_jacobian_action_block_contiguous',
            'reference': 'ref_line_p1_q2',
            'quadrature': 'quad_line_q2',
            'includes': ('line_p1_q2.hpp', 'quad_line_q2.hpp'),
        },
        {
            'element': 'PROTEUS_QUAD4',
            'header': 'mooney_rivlin_kelvin_voigt_viscous_d2_tensor_product_local.hpp',
            'dim': 2,
            'n_fields': 2,
            'n_qp': 4,
            'n_shape': 4,
            'assembly': 'mooney_rivlin_kelvin_voigt_viscous_d2_tensor_product_hessian_block',
            'apply': 'mooney_rivlin_kelvin_voigt_viscous_d2_tensor_product_jacobian_action_block_contiguous',
            'reference': 'ref_line_p1_q2',
            'quadrature': 'quad_line_q2',
            'includes': ('line_p1_q2.hpp', 'quad_line_q2.hpp'),
        },
    )

    DRIVER = '// Does the element matrix agree with the operator it assembles?\n//\n// The matrix is built from the flux with a trial basis function substituted for\n// the direction; the apply evaluates the same flux with a real direction in it.\n// They are one bilinear form, so `A * v` must equal `apply(v)` for every `v`.\n// That is the property probing had by construction -- the column *was* an apply\n// -- and the property the substitution has to earn.\n//\n// Checked over every basis direction and one dense one, on a perturbed identity\n// geometry and a random state, so a defect in one column cannot hide behind the\n// zeros of the next.\n#include <cstdio>\n#include <cmath>\n#include <cstdlib>\n#include <cmath>\n#include "@LOCAL_HEADER@"\n@INCLUDES@\n\nusing s_t = double;\nstatic constexpr int NQ = @NQ@;\nstatic constexpr int NS = @NS@;\nstatic constexpr int NC = @NC@;\nstatic constexpr int ND = @ND@;\nstatic constexpr int VS = 1;\n\n// Which fields this matrix has rows and columns for.  A coupled system\n// publishes one matrix per Jacobian block as well as the whole square, and the\n// two are the same check with different field lists.\nstatic constexpr int ROW_FIELDS[] = {@ROW_FIELDS@};\nstatic constexpr int COL_FIELDS[] = {@COLUMN_FIELDS@};\nstatic constexpr int N_ROWS = (int)(sizeof(ROW_FIELDS) / sizeof(int)) * NS;\nstatic constexpr int N_COLS = (int)(sizeof(COL_FIELDS) / sizeof(int)) * NS;\n\n// A matrix row or column is `field position * NS + shape`; a kernel stream is\n// `shape * NC + field`.  Both orders appear below because the apply speaks\n// streams and the matrix speaks its own rows and columns.\nstatic int row_stream(int row) { return (row % NS) * NC + ROW_FIELDS[row / NS]; }\nstatic int col_stream(int col) { return (col % NS) * NC + COL_FIELDS[col / NS]; }\n\nstatic double rnd() { return (double)rand() / RAND_MAX - 0.5; }\n\nint main() {\n  srand(20260915);\n  // The material\'s parameters, in the order its kernels take them.  Values\n  // chosen to be ordinary rather than special: a zero or a one can hide a term.\n@PARAM_DECLS@\n\n  s_t det[NQ * VS], adj_data[ND * ND][NQ * VS];\n  const s_t *adj[ND * ND];\n  for (int c = 0; c < ND * ND; ++c) adj[c] = adj_data[c];\n  for (int q = 0; q < NQ; ++q) {\n    for (int c = 0; c < ND * ND; ++c)\n      adj_data[c][q] = (c % (ND + 1) == 0 ? 1.0 : 0.0) + 0.2 * rnd();\n    det[q] = 1.0 + 0.1 * rnd();\n  }\n\n  // Both state roles are filled whether or not the kernels take both; which\n  // ones cross the boundary is the form\'s answer and `@STATE@` carries it.\n  s_t current[NC * NS][VS], previous[NC * NS][VS];\n  for (int i = 0; i < NC * NS; ++i) {\n    // `field` is which of the system\'s fields this stream carries; a material\n    // whose state has to satisfy an inequality reads it.\n    const int field = i % NC;\n    (void)field;\n    current[i][0] = @STATE_INIT@;\n    previous[i][0] = @STATE_INIT@;\n  }\n\n  s_t element_matrix[N_ROWS * N_COLS];\n  sfem::codegen::@ASSEMBLY@<s_t, NQ, NS, VS>(\n      1, 1, det, adj, @ASSEMBLY_ARGS@element_matrix);\n\n  // Checked before anything is compared.  `fmax` returns its non-NaN operand,\n  // so a NaN entry would leave `worst` at zero and read as perfect agreement --\n  // which is how a state outside a material\'s domain, where a fractional power\n  // of a negative number is taken, would pass this test silently.\n  for (int i = 0; i < N_ROWS * N_COLS; ++i) {\n    if (!std::isfinite(element_matrix[i])) {\n      printf("nan %d\\n", i);\n      return 1;\n    }\n  }\n\n  double worst = 0.0, scale = 0.0;\n  for (int i = 0; i < N_ROWS * N_COLS; ++i) scale = fmax(scale, fabs(element_matrix[i]));\n\n  for (int trial_local = 0; trial_local <= N_COLS; ++trial_local) {\n    s_t direction[NC * NS][VS], out[NC * NS][VS];\n    for (int i = 0; i < NC * NS; ++i) { direction[i][0] = 0.0; out[i][0] = 0.0; }\n    if (trial_local < N_COLS) {\n      direction[col_stream(trial_local)][0] = 1.0;\n    } else {\n      // A dense direction, confined to the columns this matrix has: anything\n      // outside them is not in the matrix and the apply would answer for it.\n      for (int col = 0; col < N_COLS; ++col) direction[col_stream(col)][0] = rnd();\n    }\n    sfem::codegen::@APPLY@<s_t, NQ, NS, VS>(\n        1, 1, det, adj, @APPLY_ARGS@out);\n\n    for (int test_local = 0; test_local < N_ROWS; ++test_local) {\n      double from_matrix = 0.0;\n      for (int j = 0; j < N_COLS; ++j) {\n        from_matrix += element_matrix[test_local * N_COLS + j]\n                     * direction[col_stream(j)][0];\n      }\n      worst = fmax(worst, fabs(from_matrix - out[row_stream(test_local)][0]));\n    }\n  }\n  printf("%.17e %.17e\\n", worst, scale);\n  return 0;\n}\n'

    @staticmethod
    def _shipped_tree():
        from codegen.framework.tools.codegen_snapshot import shipped_tree

        return Path(shipped_tree())

    def test_the_matrix_applies_like_the_operator(self):
        compiler = shutil.which("c++") or shutil.which("g++")
        if compiler is None:
            self.skipTest("C++ compiler is not available")
        tree = self._shipped_tree()
        material = tree / "mooney_rivlin_kelvin_voigt"
        if not material.is_dir():
            self.skipTest("the material is not in the shipped tree")

        from codegen.framework.materials import mooney_rivlin_kelvin_voigt

        for case in self.CASES:
            with self.subTest(element=case["element"]):
                self._check(
                    case,
                    material / ("d%d" % case["dim"]),
                    tree,
                    mooney_rivlin_kelvin_voigt.material,
                )

    @staticmethod
    def _parameter_names(header, kernel):
        """One kernel's arguments, in its order, read from its signature.

        Everything this driver passes between the adjugate and the element
        matrix is taken from here rather than transcribed into the case: the
        reference tables a kernel reads, the state roles it takes, and its
        material parameters.

        That is not tidiness.  A hard-coded list goes stale silently and looks
        like a defect in the kernel when it does -- the first version of this
        test spelled a `shape` argument, a later change correctly stopped the
        quadrature kernel taking a table it never read, and the test then failed
        as though the kernel were wrong.
        """
        import re

        signature = header.read_text().split("void %s(" % kernel, 1)[1]
        signature = signature.split(") {", 1)[0]
        names = []
        for line in signature.split("\n"):
            # `const s_t *const RSTR adjugate[9],` and `const s_t eta_b,` and
            # `const s_t current[3 * NS][VS],` all end in the name, once the
            # array extents are off.
            declaration = re.sub(r"\[.*$", "", line.strip().rstrip(",")).strip()
            if declaration:
                names.append(declaration.split()[-1].lstrip("*"))
        return tuple(names)

    #: What each argument name is spelled as at the call.  Reference tables come
    #: from the element's own structs, which the case names; a state role is the
    #: local array of that name; anything else is a material parameter.
    REFERENCE_ACCESSORS = {
        "shape": "%(ref)s<s_t>::shape()",
        "grad_ref_x": "%(ref)s<s_t>::grad_ref_x()",
        "grad_ref_y": "%(ref)s<s_t>::grad_ref_y()",
        "grad_ref_z": "%(ref)s<s_t>::grad_ref_z()",
        "q_weight": "%(quad)s<s_t>::q_weight()",
        "shape_1d": "%(ref)s<s_t>::shape_1d()",
        "grad_1d": "%(ref)s<s_t>::grad_1d()",
        "q_weight_1d": "%(quad)s<s_t>::q_weight_1d()",
    }
    STATE_ARGUMENTS = ("current", "previous", "direction")

    @classmethod
    def _call_arguments(cls, header, kernel, case, defaults):
        """The argument list this kernel is called with, and the values it needs."""
        spelling = {"ref": case["reference"], "quad": case["quadrature"]}
        arguments, parameters = [], []
        # The first four and the last are what the driver spells itself: the
        # work-item count, the geometry stride, the geometry, and whatever the
        # kernel writes -- `element_matrix` for an assembly, `output` for an
        # apply, which is why the last is dropped by position and not by name.
        for name in cls._parameter_names(header, kernel)[4:-1]:
            if name in cls.REFERENCE_ACCESSORS:
                arguments.append(
                    "sfem::codegen::" + cls.REFERENCE_ACCESSORS[name] % spelling
                )
            elif name in cls.STATE_ARGUMENTS:
                arguments.append(name)
            else:
                arguments.append(name)
                parameters.append((name, repr(float(defaults[name]))))
        return arguments, parameters

    def _check(self, case, headers, tree, material):
        """Build and run the driver for one case, and hold it to the claim."""
        compiler = shutil.which("c++") or shutil.which("g++")
        header = headers / case["header"]
        defaults = dict(material.parameter_defaults)
        assembly_args, parameters = self._call_arguments(
            header, case["assembly"], case, defaults
        )
        apply_args, _ = self._call_arguments(header, case["apply"], case, defaults)
        source = self.DRIVER
        for key, value in (
            ("@LOCAL_HEADER@", case["header"]),
            ("@INCLUDES@",
             "\n".join('#include "%s"' % name for name in case["includes"])),
            ("@NQ@", str(case["n_qp"])),
            ("@NS@", str(case["n_shape"])),
            ("@NC@", str(case["n_fields"])),
            ("@ND@", str(case["dim"])),
            ("@ROW_FIELDS@",
             ", ".join(str(f) for f in case.get(
                 "row_fields", range(case["n_fields"])))),
            ("@COLUMN_FIELDS@",
             ", ".join(str(f) for f in case.get(
                 "column_fields", range(case["n_fields"])))),
            ("@STATE_INIT@", case.get("state_init", "0.05 * rnd()")),
            ("@PARAM_DECLS@",
             "\n".join(
                 "  const s_t %s = %s;" % (name, value)
                 for name, value in parameters
             )),
            ("@ASSEMBLY@", case["assembly"]),
            ("@ASSEMBLY_ARGS@", "".join("%s, " % a for a in assembly_args)),
            ("@APPLY@", case["apply"]),
            ("@APPLY_ARGS@", "".join("%s, " % a for a in apply_args)),
        ):
            source = source.replace(key, value)
        with tempfile.TemporaryDirectory() as work:
            driver = Path(work) / "check.cpp"
            driver.write_text(source)
            binary = Path(work) / "check"
            build = subprocess.run(
                [
                    compiler, "-O1", "-std=c++17", "-Wno-unknown-pragmas",
                    "-I", str(headers),
                    "-I", str(tree / "reference"),
                    "-I", str(tree),
                    "-o", str(binary), str(driver),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(build.returncode, 0, build.stderr[-4000:])
            run = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr[-4000:])
            worst, scale = (float(x) for x in run.stdout.split())
            self.assertGreater(scale, 0.0, "the matrix is all zeros")
            self.assertLess(
                worst / scale,
                1e-12,
                "the assembled matrix and the apply disagree by "
                + ("%.3e" % (worst / scale))
                + " relative to the largest entry",
            )


    #: One Jacobian block of a coupled system, which is a different matrix from
    #: the whole square: its rows are one field's, its columns another's, and
    #: its stride is its own.  Two-phase flow is the case that exercises it, and
    #: it exercises two more things with it -- its flux contracts the
    #: *direction's value* as well as its gradient, and TRI3 is a closed-form
    #: element whose shape cannot supply a trial value, so the block falls back
    #: to the quadrature shape rather than to probing.
    BLOCK_CASE = {
        'element': 'TRI3',
        'header': 'two_phase_flow_form_2_p_w_p_w_d2_simplex_local.hpp',
        'dim': 2,
        'n_fields': 2,
        'n_qp': 6,
        'n_shape': 3,
        'row_fields': (0,),
        'column_fields': (0,),
        'assembly': 'two_phase_flow_form_2_p_w_p_w_d2_simplex_hessian_block',
        'apply': 'two_phase_flow_form_2_p_w_p_w_d2_simplex_jacobian_action_block_contiguous',
        'reference': 'ref_tri3_q6',
        'quadrature': 'quad_tri_q6',
        'includes': ('tri3_q6.hpp', 'quad_tri_q6.hpp'),
        'state_init': '(field == 1 ? 1.2 + 0.02 * rnd() : 1.0 + 0.02 * rnd())',
    }

    def test_a_jacobian_block_applies_like_the_operator(self):
        """The block is generated here: no material ships one in the tree."""
        if shutil.which("c++") is None and shutil.which("g++") is None:
            self.skipTest("C++ compiler is not available")
        from codegen.framework.materials import two_phase_flow

        with tempfile.TemporaryDirectory() as out_dir:
            gen.generate(
                two_phase_flow.material,
                out_dir,
                elements=("TRI3",),
                clean=True,
                matrix_formats=("crs", "bsr"),
            )
            tree = Path(out_dir)
            self._check(
                self.BLOCK_CASE, tree / "d2", tree, two_phase_flow.material
            )

class ClosedFormAssemblyShapeTest(unittest.TestCase):
    """A closed-form element matrix has no loops, because that is the point.

    SFEM's hand-written `tet4_linear_elasticity_crs_adj` writes 144 entries with
    178 temporaries shared across all of them and not one loop.  The sharing is
    what the shape buys: a loop over trial functions can eliminate only within
    one column and re-derives for every column what the columns have in common,
    which on a constant-basis element is nearly everything.

    So this asserts the shape rather than the absence of probing -- the ratchet
    above already forbids probing, and a kernel could obey it while still being
    a loop nest that shares nothing.
    """

    @staticmethod
    def _shipped_tree():
        from codegen.framework.tools.codegen_snapshot import shipped_tree

        return Path(shipped_tree())

    def _closed_form_kernels(self):
        """Every generated closed-form element-matrix kernel, as its body.

        Told apart from the quadrature one by what it *takes*, not by its name:
        a closed-form kernel has its basis gradients and its quadrature weight
        folded into the arithmetic, so no `q_weight` crosses its boundary.  A
        name test would have to know the specialisation's spelling; this reads
        the property the shape is defined by.
        """
        import re

        opening = re.compile(r"^static SFEM_INLINE void (\S+_hessian_block)\($", re.M)
        found = {}
        for path in sorted(self._shipped_tree().rglob("*_local.hpp")):
            lines = path.read_text().split("\n")
            for index, line in enumerate(lines):
                match = opening.match(line)
                if match is None:
                    continue
                for end in range(index + 1, len(lines)):
                    if lines[end] == "}":
                        body = lines[index:end]
                        signature = body[: body.index(") {")] if ") {" in body else body
                        if not any("q_weight" in line for line in signature):
                            found[match.group(1)] = body
                        break
        return found

    def test_a_closed_form_element_matrix_opens_no_loop_of_its_own(self):
        kernels = self._closed_form_kernels()
        self.assertTrue(
            kernels,
            "no closed-form element-matrix kernel is shipped; if the last one "
            "was withdrawn, say so here rather than leaving this vacuous",
        )
        for name, body in sorted(kernels.items()):
            loops = [line.strip() for line in body if "for (" in line]
            with self.subTest(kernel=name):
                # The work-item loop is the target's, not the kernel's: under
                # CUDA it is not a loop at all.  Anything else would be a
                # quadrature, shape or component loop this shape does not have.
                self.assertEqual(
                    [loop for loop in loops if "lane" not in loop],
                    [],
                    "%s opens a loop of its own" % name,
                )

    def test_every_entry_is_written_exactly_once(self):
        """Written, not accumulated -- there is nothing to accumulate over."""
        import re

        entry = re.compile(r"element_matrix\[(\d+)\] = ")
        accumulated = re.compile(r"element_matrix\[[^]]*\] \+= ")
        for name, body in sorted(self._closed_form_kernels().items()):
            source = "\n".join(body)
            written = [int(index) for index in entry.findall(source)]
            with self.subTest(kernel=name):
                self.assertEqual(
                    sorted(written),
                    list(range(len(written))),
                    "%s does not write each entry of a dense square once" % name,
                )
                self.assertIsNone(accumulated.search(source))


if __name__ == "__main__":
    unittest.main()
