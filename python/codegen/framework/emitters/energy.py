from codegen.framework.plans.conventions import restrict_prelude
from dataclasses import dataclass

from codegen.framework.ir.kernel_ast import (
    BufferDeclNode,
    LoopHeaderNode,
    LoopKind,
    LoopNode,
    add_assign_increment,
    expr_ref,
    iteration_range,
    iterator,
)
from codegen.framework.emitters.ast_printer import render_kernel_ast_lines
from codegen.framework.emitters.tensor_product_geometry import sfem_geometry_kernels_header_source
from codegen.framework.targets import CUDATarget, OpenMPTarget


def _join_lines(lines):
    return "\n".join(line for line in lines if line != "")


def _cuda_geometry_header_source():
    return _join_lines(
        [
            "#ifndef SFEM_CODEGEN_GEOMETRY_KERNELS_CUH",
            "#define SFEM_CODEGEN_GEOMETRY_KERNELS_CUH",
            "",
            "#include <stddef.h>",
            "",
            *restrict_prelude(),
            "",
            "namespace sfem {",
            "namespace codegen {",
            "",
            "template <typename s_t, int ND, int NQ, int VS>",
            "struct GeometryJacobianAdjugateDeterminant;",
            "",
            "template <typename s_t>",
            "static __host__ __device__ __forceinline__ void geometry_jacobian_adjugate_and_determinant_2(",
            "    const s_t J00,",
            "    const s_t J01,",
            "    const s_t J10,",
            "    const s_t J11,",
            "    s_t *const *const RSTR adjugate,",
            "    s_t *const RSTR determinant,",
            "    const ptrdiff_t offset) {",
            "  adjugate[0][offset] = J11;",
            "  adjugate[1][offset] = -J01;",
            "  adjugate[2][offset] = -J10;",
            "  adjugate[3][offset] = J00;",
            "  determinant[offset] = J00 * J11 - J01 * J10;",
            "}",
            "",
            "template <typename s_t>",
            "static __host__ __device__ __forceinline__ void geometry_jacobian_adjugate_and_determinant_3(",
            "    const s_t J00,",
            "    const s_t J01,",
            "    const s_t J02,",
            "    const s_t J10,",
            "    const s_t J11,",
            "    const s_t J12,",
            "    const s_t J20,",
            "    const s_t J21,",
            "    const s_t J22,",
            "    s_t *const *const RSTR adjugate,",
            "    s_t *const RSTR determinant,",
            "    const ptrdiff_t offset) {",
            "  adjugate[0][offset] = J11 * J22 - J12 * J21;",
            "  adjugate[1][offset] = J02 * J21 - J01 * J22;",
            "  adjugate[2][offset] = J01 * J12 - J02 * J11;",
            "  adjugate[3][offset] = J12 * J20 - J10 * J22;",
            "  adjugate[4][offset] = J00 * J22 - J02 * J20;",
            "  adjugate[5][offset] = J02 * J10 - J00 * J12;",
            "  adjugate[6][offset] = J10 * J21 - J11 * J20;",
            "  adjugate[7][offset] = J01 * J20 - J00 * J21;",
            "  adjugate[8][offset] = J00 * J11 - J01 * J10;",
            "  determinant[offset] = J00 * (J11 * J22 - J12 * J21)",
            "      - J01 * (J10 * J22 - J12 * J20)",
            "      + J02 * (J10 * J21 - J11 * J20);",
            "}",
            "",
            "template <typename s_t, int NQ, int VS>",
            "struct GeometryJacobianAdjugateDeterminant<s_t, 2, NQ, VS> {",
            "  static __host__ __device__ __forceinline__ void eval(",
            "      const int ne,",
            "      const s_t *const RSTR coordinate_grad_ref,",
            "      s_t *const *const RSTR adjugate,",
            "      s_t *const RSTR determinant) {",
            "    for (int q = 0; q < NQ; ++q) {",
            "      {",
            "        const ptrdiff_t offset = q * VS;",
            "        const s_t J00 = coordinate_grad_ref[((0 * NQ + q) * 2 + 0) * VS];",
            "        const s_t J01 = coordinate_grad_ref[((0 * NQ + q) * 2 + 1) * VS];",
            "        const s_t J10 = coordinate_grad_ref[((1 * NQ + q) * 2 + 0) * VS];",
            "        const s_t J11 = coordinate_grad_ref[((1 * NQ + q) * 2 + 1) * VS];",
            "        geometry_jacobian_adjugate_and_determinant_2<s_t>(",
            "            J00, J01, J10, J11, adjugate, determinant, offset);",
            "      }",
            "    }",
            "  }",
            "};",
            "",
            "template <typename s_t, int NQ, int VS>",
            "struct GeometryJacobianAdjugateDeterminant<s_t, 3, NQ, VS> {",
            "  static __host__ __device__ __forceinline__ void eval(",
            "      const int ne,",
            "      const s_t *const RSTR coordinate_grad_ref,",
            "      s_t *const *const RSTR adjugate,",
            "      s_t *const RSTR determinant) {",
            "    for (int q = 0; q < NQ; ++q) {",
            "      {",
            "        const ptrdiff_t offset = q * VS;",
            "        const s_t J00 = coordinate_grad_ref[((0 * NQ + q) * 3 + 0) * VS];",
            "        const s_t J01 = coordinate_grad_ref[((0 * NQ + q) * 3 + 1) * VS];",
            "        const s_t J02 = coordinate_grad_ref[((0 * NQ + q) * 3 + 2) * VS];",
            "        const s_t J10 = coordinate_grad_ref[((1 * NQ + q) * 3 + 0) * VS];",
            "        const s_t J11 = coordinate_grad_ref[((1 * NQ + q) * 3 + 1) * VS];",
            "        const s_t J12 = coordinate_grad_ref[((1 * NQ + q) * 3 + 2) * VS];",
            "        const s_t J20 = coordinate_grad_ref[((2 * NQ + q) * 3 + 0) * VS];",
            "        const s_t J21 = coordinate_grad_ref[((2 * NQ + q) * 3 + 1) * VS];",
            "        const s_t J22 = coordinate_grad_ref[((2 * NQ + q) * 3 + 2) * VS];",
            "        geometry_jacobian_adjugate_and_determinant_3<s_t>(",
            "            J00, J01, J02, J10, J11, J12, J20, J21, J22,",
            "            adjugate, determinant, offset);",
            "      }",
            "    }",
            "  }",
            "};",
            "",
            "template <typename s_t, int ND, int NQ, int VS>",
            "static __host__ __device__ __forceinline__ void geometry_jacobian_adjugate_and_determinant(",
            "    const int ne,",
            "    const s_t *const RSTR coordinate_grad_ref,",
            "    s_t *const *const RSTR adjugate,",
            "    s_t *const RSTR determinant) {",
            "  GeometryJacobianAdjugateDeterminant<s_t, ND, NQ, VS>::eval(",
            "      ne, coordinate_grad_ref, adjugate, determinant);",
            "}",
            "",
            "} // namespace codegen",
            "} // namespace sfem",
            "",
            "#endif",
            "",
        ]
    )


@dataclass(frozen=True)
class OpenMPEnergySoASourceBuilder:
    operator_extension: str = "cpp"
    emit_objective_steps: bool = True
    target: object = OpenMPTarget()

    def header_name(self, stem):
        return "%s.hpp" % stem

    def header_guard_suffix(self):
        return "HPP"

    def inline_qualifier(self):
        return self.target.inline_qualifier()

    def local_header_preamble_lines(self, math_name, tensor_product_name, basis_family):
        return (
            '#include "%s"' % math_name,
            '#include "%s"' % tensor_product_name,
            "",
            *self.target.inline_definition_lines(),
            "",
            *restrict_prelude(""),
        )

    def operator_preamble_lines(self, local_name, geometry_name, diagnostics_name, extra_headers=()):
        return (
            "#include <cstdio>",
            "#include <type_traits>",
            '#include "%s"' % local_name,
            *('#include "%s"' % header for header in extra_headers),
            '#include "%s"' % geometry_name,
            '#include "%s"' % diagnostics_name,
            *self.target.includes(),
        )

    def geometry_header_source(self):
        return sfem_geometry_kernels_header_source(
            inline_qualifier=self.inline_qualifier(),
            define_sfem_inline=True,
            restrict_definition="",
            work_item_index=self.work_item_index(),
            simd_lines=self.simd_lines(),
            single_work_item=False,
            header_guard_suffix=self.header_guard_suffix(),
        )

    def emits_tensor_product_header(self, basis_family):
        return True

    def tensor_product_header_source(self):
        from codegen.framework.emitters.tensor_product_kernels import sfem_tensor_product_kernels_header_source

        return sfem_tensor_product_kernels_header_source()

    def simd_lines(self):
        pragma = self.target.vectorize_pragma()
        return () if pragma is None else (pragma,)

    def work_item_index(self):
        return self.target.work_item_index()

    def work_item_name(self, name, component):
        return self.target.work_item_name(name, component)

    def diagnostic_work_item(self):
        return self.target.diagnostic_work_item()

    def work_item_loop_lines(self, indent):
        return self.target.work_item_loop_lines(indent)

    def parallel_for_lines(self, reduction=None):
        return self.target.parallel_element_loop_lines("static", reduction)

    def atomic_update_lines(self):
        """The atomic-update pragma, or nothing where the target has none.

        A tuple rather than a string so a target without the concept -- CUDA,
        which uses ``atomicAdd`` instead -- contributes no line, matching
        ``simd_lines``.
        """
        pragma = self.target.atomic_update_pragma()
        return () if pragma is None else (pragma,)

    def effective_vector_size(self, vector_size):
        return int(vector_size)

    def mesh_loop_lines(self):
        tile_iterator = iterator("evb", "ptrdiff_t")
        lines = render_kernel_ast_lines(
            "openmp_mesh_tile_loop",
            (
                LoopHeaderNode(
                    LoopNode(
                        LoopKind.TILE,
                        tile_iterator,
                        iteration_range(0, expr_ref("nelements", "element_count")),
                        add_assign_increment(
                            tile_iterator,
                            expr_ref("VS", "vector_width"),
                        ),
                    )
                ),
                BufferDeclNode(
                    "const int",
                    "ne",
                    (),
                    "(int)MIN((ptrdiff_t)VS, nelements - evb)",
                ),
            ),
        )
        return ("  %s" % lines[0], "    %s" % lines[1])

    def mesh_template_line(self, geometry_mode, extra_template_params=()):
        """The mesh kernel's template head.

        `extra_template_params` carries the extents the body would otherwise
        declare as `static constexpr` -- today the vector width.  It is a
        parameter rather than a constant in the body because the width is a
        tuning choice, and it carries no default because a caller must choose:
        the generated entry point spells the number it wants at the point where
        it names the concrete kernel.
        """
        return "template <typename s_t, typename g_t%s>" % "".join(
            ", %s" % parameter for parameter in extra_template_params
        )

    def mesh_function_line(self, implementation_name):
        return "%s int %s(" % (self.target.function_qualifier(), implementation_name)

    def success_return_lines(self):
        return ("  return SFEM_SUCCESS;",)

    def wrapper_call_lines(self, implementation_name, scalar_type, extra_template_args, wrapper_args):
        return (
            "  return sfem::codegen::%s<%s%s>(%s);"
            % (implementation_name, scalar_type, extra_template_args, ", ".join(wrapper_args)),
        )

    def scatter_add_lines(self, lhs, rhs, indent):
        return self.target.scatter_add_lines(lhs, rhs, indent)


@dataclass(frozen=True)
class CUDAEnergySoASourceBuilder:
    operator_extension: str = "cu"
    emit_objective_steps: bool = False
    target: object = CUDATarget()

    def header_name(self, stem):
        return "%s.cuh" % stem

    def header_guard_suffix(self):
        return "CUH"

    def inline_qualifier(self):
        return self.target.inline_qualifier()

    def local_header_preamble_lines(self, math_name, tensor_product_name, basis_family):
        tensor_include = (
            ('#include "%s"' % tensor_product_name,)
            if str(basis_family) == "tensor_product"
            else ()
        )
        return (
            '#include "%s"' % math_name,
            *tensor_include,
            "",
            *restrict_prelude(),
        )

    def operator_preamble_lines(self, local_name, geometry_name, diagnostics_name, extra_headers=()):
        return (
            "#include <type_traits>",
            *self.target.includes(),
            '#include "%s"' % local_name,
            *('#include "%s"' % header for header in extra_headers),
            '#include "%s"' % geometry_name,
            '#include "%s"' % diagnostics_name,
        )

    def geometry_header_source(self):
        return _cuda_geometry_header_source()

    def emits_tensor_product_header(self, basis_family):
        return str(basis_family) == "tensor_product"

    def tensor_product_header_source(self):
        from codegen.framework.emitters.tensor_product_kernels import sfem_tensor_product_kernels_header_source

        return sfem_tensor_product_kernels_header_source(
            inline_qualifier=self.inline_qualifier(),
            define_sfem_inline=False,
            restrict_definition="__restrict__",
            work_item_index=self.work_item_index(),
            simd_lines=(),
            single_work_item=True,
            header_guard_suffix=self.header_guard_suffix(),
        )

    def simd_lines(self):
        return ()

    def work_item_index(self):
        return self.target.work_item_index()

    def work_item_name(self, name, component):
        return self.target.work_item_name(name, component)

    def diagnostic_work_item(self):
        return self.target.diagnostic_work_item()

    def work_item_loop_lines(self, indent):
        return self.target.work_item_loop_lines(indent)

    def parallel_for_lines(self, reduction=None):
        return ()

    def atomic_update_lines(self):
        """Nothing: CUDA scatters with ``atomicAdd``, not a pragma."""
        return ()

    def effective_vector_size(self, vector_size):
        return 1

    def mesh_loop_lines(self):
        kernel_iterator = iterator("evb", "ptrdiff_t")
        lines = render_kernel_ast_lines(
            "cuda_mesh_grid_stride_loop",
            (
                LoopHeaderNode(
                    LoopNode(
                        LoopKind.KERNEL,
                        kernel_iterator,
                        iteration_range(
                            expr_ref(
                                "(ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x",
                                "cuda_thread_start",
                            ),
                            expr_ref("nelements", "element_count"),
                        ),
                        add_assign_increment(
                            kernel_iterator,
                            expr_ref("(ptrdiff_t)blockDim.x * gridDim.x", "cuda_grid_stride"),
                        ),
                    )
                ),
                BufferDeclNode("const int", "ne", (), "1"),
            ),
        )
        return ("  %s" % lines[0], "    %s" % lines[1])

    def mesh_template_line(self, geometry_mode, extra_template_params=()):
        """The mesh kernel's template head.

        `extra_template_params` carries the extents the body would otherwise
        declare as `static constexpr` -- today the vector width.  It is a
        parameter rather than a constant in the body because the width is a
        tuning choice, and it carries no default because a caller must choose:
        the generated entry point spells the number it wants at the point where
        it names the concrete kernel.
        """
        return "template <typename s_t, typename g_t%s>" % "".join(
            ", %s" % parameter for parameter in extra_template_params
        )

    def mesh_function_line(self, implementation_name):
        return "__global__ void %s(" % implementation_name

    def success_return_lines(self):
        return ()

    def wrapper_call_lines(self, implementation_name, scalar_type, extra_template_args, wrapper_args):
        template_args = "%s%s" % (scalar_type, extra_template_args)
        return (
            "  const int block_size = 256;",
            "  const int grid_size = (int)((nelements + block_size - 1) / block_size);",
            "  sfem::codegen::%s<%s><<<grid_size, block_size>>>(%s);"
            % (implementation_name, template_args, ", ".join(wrapper_args)),
            "  return SFEM_SUCCESS;",
        )

    def scatter_add_lines(self, lhs, rhs, indent):
        return self.target.scatter_add_lines(lhs, rhs, indent)


@dataclass(frozen=True)
class OpenMPEnergySoAEmitter:
    """Opaque OpenMP emitter: consume an energy-SoA emission plan, emit files."""

    supports_op_wrapper: bool = True
    target: object = OpenMPTarget()

    def emit_plan(self, plan):
        from codegen.framework.emitters.energy_codegen import generate_sfem_soa_cpp_files_for_element

        return generate_sfem_soa_cpp_files_for_element(
            plan.forms,
            prefix=plan.prefix,
            local_prefix=plan.local_prefix,
            emission_plan=plan.emission_plan,
            reference_data_plan=plan.reference_data_plan,
            diagnostics_plan=plan.diagnostics_plan,
            matrix_format_plan=plan.matrix_format_plan,
            source_builder=OpenMPEnergySoASourceBuilder(target=self.target),
        )


@dataclass(frozen=True)
class CUDAEnergySoAEmitter:
    """Opaque CUDA emitter: consume an energy-SoA emission plan, emit files."""

    supports_op_wrapper: bool = False
    target: object = CUDATarget()
    operator_extension: str = "cu"

    def emit_plan(self, plan):
        from codegen.framework.emitters.energy_codegen import generate_sfem_soa_cpp_files_for_element

        return generate_sfem_soa_cpp_files_for_element(
            plan.forms,
            prefix=plan.prefix,
            local_prefix=plan.local_prefix,
            emission_plan=plan.emission_plan,
            reference_data_plan=plan.reference_data_plan,
            diagnostics_plan=plan.diagnostics_plan,
            source_builder=CUDAEnergySoASourceBuilder(
                operator_extension=self.operator_extension,
                target=self.target,
            ),
        )
