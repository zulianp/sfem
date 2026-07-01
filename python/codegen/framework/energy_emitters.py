from dataclasses import dataclass

try:
    from .tensor_product_geometry import sfem_geometry_kernels_header_source
    from .targets import CUDATarget, OpenMPTarget
except ImportError:
    from tensor_product_geometry import sfem_geometry_kernels_header_source
    from targets import CUDATarget, OpenMPTarget


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
            "#ifndef SFEM_RESTRICT",
            "#define SFEM_RESTRICT __restrict__",
            "#endif",
            "",
            "namespace sfem {",
            "namespace codegen {",
            "",
            "template <typename scalar_t, int DIM, int N_QP, int VECTOR_SIZE>",
            "struct GeometryJacobianAdjugateDeterminant;",
            "",
            "template <typename scalar_t>",
            "static __host__ __device__ __forceinline__ void geometry_jacobian_adjugate_and_determinant_2(",
            "        const scalar_t J00,",
            "        const scalar_t J01,",
            "        const scalar_t J10,",
            "        const scalar_t J11,",
            "        scalar_t *const *const SFEM_RESTRICT adjugate,",
            "        scalar_t *const SFEM_RESTRICT determinant,",
            "        const ptrdiff_t offset) {",
            "    adjugate[0][offset] = J11;",
            "    adjugate[1][offset] = -J01;",
            "    adjugate[2][offset] = -J10;",
            "    adjugate[3][offset] = J00;",
            "    determinant[offset] = J00 * J11 - J01 * J10;",
            "}",
            "",
            "template <typename scalar_t>",
            "static __host__ __device__ __forceinline__ void geometry_jacobian_adjugate_and_determinant_3(",
            "        const scalar_t J00,",
            "        const scalar_t J01,",
            "        const scalar_t J02,",
            "        const scalar_t J10,",
            "        const scalar_t J11,",
            "        const scalar_t J12,",
            "        const scalar_t J20,",
            "        const scalar_t J21,",
            "        const scalar_t J22,",
            "        scalar_t *const *const SFEM_RESTRICT adjugate,",
            "        scalar_t *const SFEM_RESTRICT determinant,",
            "        const ptrdiff_t offset) {",
            "    adjugate[0][offset] = J11 * J22 - J12 * J21;",
            "    adjugate[1][offset] = J02 * J21 - J01 * J22;",
            "    adjugate[2][offset] = J01 * J12 - J02 * J11;",
            "    adjugate[3][offset] = J12 * J20 - J10 * J22;",
            "    adjugate[4][offset] = J00 * J22 - J02 * J20;",
            "    adjugate[5][offset] = J02 * J10 - J00 * J12;",
            "    adjugate[6][offset] = J10 * J21 - J11 * J20;",
            "    adjugate[7][offset] = J01 * J20 - J00 * J21;",
            "    adjugate[8][offset] = J00 * J11 - J01 * J10;",
            "    determinant[offset] = J00 * (J11 * J22 - J12 * J21)",
            "            - J01 * (J10 * J22 - J12 * J20)",
            "            + J02 * (J10 * J21 - J11 * J20);",
            "}",
            "",
            "template <typename scalar_t, int N_QP, int VECTOR_SIZE>",
            "struct GeometryJacobianAdjugateDeterminant<scalar_t, 2, N_QP, VECTOR_SIZE> {",
            "    static __host__ __device__ __forceinline__ void eval(",
            "            const int nelems,",
            "            const scalar_t *const SFEM_RESTRICT coordinate_grad_ref,",
            "            scalar_t *const *const SFEM_RESTRICT adjugate,",
            "            scalar_t *const SFEM_RESTRICT determinant) {",
            "        for (int q = 0; q < N_QP; ++q) {",
            "            {",
            "                const ptrdiff_t offset = q * VECTOR_SIZE;",
            "                const scalar_t J00 = coordinate_grad_ref[((0 * N_QP + q) * 2 + 0) * VECTOR_SIZE];",
            "                const scalar_t J01 = coordinate_grad_ref[((0 * N_QP + q) * 2 + 1) * VECTOR_SIZE];",
            "                const scalar_t J10 = coordinate_grad_ref[((1 * N_QP + q) * 2 + 0) * VECTOR_SIZE];",
            "                const scalar_t J11 = coordinate_grad_ref[((1 * N_QP + q) * 2 + 1) * VECTOR_SIZE];",
            "                geometry_jacobian_adjugate_and_determinant_2<scalar_t>(",
            "                        J00, J01, J10, J11, adjugate, determinant, offset);",
            "            }",
            "        }",
            "    }",
            "};",
            "",
            "template <typename scalar_t, int N_QP, int VECTOR_SIZE>",
            "struct GeometryJacobianAdjugateDeterminant<scalar_t, 3, N_QP, VECTOR_SIZE> {",
            "    static __host__ __device__ __forceinline__ void eval(",
            "            const int nelems,",
            "            const scalar_t *const SFEM_RESTRICT coordinate_grad_ref,",
            "            scalar_t *const *const SFEM_RESTRICT adjugate,",
            "            scalar_t *const SFEM_RESTRICT determinant) {",
            "        for (int q = 0; q < N_QP; ++q) {",
            "            {",
            "                const ptrdiff_t offset = q * VECTOR_SIZE;",
            "                const scalar_t J00 = coordinate_grad_ref[((0 * N_QP + q) * 3 + 0) * VECTOR_SIZE];",
            "                const scalar_t J01 = coordinate_grad_ref[((0 * N_QP + q) * 3 + 1) * VECTOR_SIZE];",
            "                const scalar_t J02 = coordinate_grad_ref[((0 * N_QP + q) * 3 + 2) * VECTOR_SIZE];",
            "                const scalar_t J10 = coordinate_grad_ref[((1 * N_QP + q) * 3 + 0) * VECTOR_SIZE];",
            "                const scalar_t J11 = coordinate_grad_ref[((1 * N_QP + q) * 3 + 1) * VECTOR_SIZE];",
            "                const scalar_t J12 = coordinate_grad_ref[((1 * N_QP + q) * 3 + 2) * VECTOR_SIZE];",
            "                const scalar_t J20 = coordinate_grad_ref[((2 * N_QP + q) * 3 + 0) * VECTOR_SIZE];",
            "                const scalar_t J21 = coordinate_grad_ref[((2 * N_QP + q) * 3 + 1) * VECTOR_SIZE];",
            "                const scalar_t J22 = coordinate_grad_ref[((2 * N_QP + q) * 3 + 2) * VECTOR_SIZE];",
            "                geometry_jacobian_adjugate_and_determinant_3<scalar_t>(",
            "                        J00, J01, J02, J10, J11, J12, J20, J21, J22,",
            "                        adjugate, determinant, offset);",
            "            }",
            "        }",
            "    }",
            "};",
            "",
            "template <typename scalar_t, int DIM, int N_QP, int VECTOR_SIZE>",
            "static __host__ __device__ __forceinline__ void geometry_jacobian_adjugate_and_determinant(",
            "        const int nelems,",
            "        const scalar_t *const SFEM_RESTRICT coordinate_grad_ref,",
            "        scalar_t *const *const SFEM_RESTRICT adjugate,",
            "        scalar_t *const SFEM_RESTRICT determinant) {",
            "    GeometryJacobianAdjugateDeterminant<scalar_t, DIM, N_QP, VECTOR_SIZE>::eval(",
            "            nelems, coordinate_grad_ref, adjugate, determinant);",
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
            "#ifndef SFEM_RESTRICT",
            "#define SFEM_RESTRICT",
            "#endif",
        )

    def operator_preamble_lines(self, local_name, geometry_name, diagnostics_name):
        return (
            "#include <type_traits>",
            '#include "%s"' % local_name,
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
        try:
            from .tensor_product_kernels import sfem_tensor_product_kernels_header_source
        except ImportError:
            from tensor_product_kernels import sfem_tensor_product_kernels_header_source

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

    def parallel_for_lines(self):
        return self.target.parallel_element_loop_lines("static")

    def effective_vector_size(self, vector_size):
        return int(vector_size)

    def mesh_loop_lines(self):
        return (
            "    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {",
            "        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);",
        )

    def mesh_template_line(self, geometry_mode):
        return (
            "template <typename scalar_t, typename geometry_t>"
            if geometry_mode == "isoparametric"
            else "template <typename scalar_t, typename jacobian_t>"
        )

    def mesh_function_line(self, implementation_name):
        return "%s int %s(" % (self.target.function_qualifier(), implementation_name)

    def success_return_lines(self):
        return ("    return SFEM_SUCCESS;",)

    def wrapper_call_lines(self, implementation_name, scalar_type, extra_template_args, wrapper_args):
        return (
            "    return sfem::codegen::%s<%s%s>(%s);"
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
            "#ifndef SFEM_RESTRICT",
            "#define SFEM_RESTRICT __restrict__",
            "#endif",
        )

    def operator_preamble_lines(self, local_name, geometry_name, diagnostics_name):
        return (
            "#include <type_traits>",
            *self.target.includes(),
            '#include "%s"' % local_name,
            '#include "%s"' % geometry_name,
            '#include "%s"' % diagnostics_name,
        )

    def geometry_header_source(self):
        return _cuda_geometry_header_source()

    def emits_tensor_product_header(self, basis_family):
        return str(basis_family) == "tensor_product"

    def tensor_product_header_source(self):
        try:
            from .tensor_product_kernels import sfem_tensor_product_kernels_header_source
        except ImportError:
            from tensor_product_kernels import sfem_tensor_product_kernels_header_source

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

    def parallel_for_lines(self):
        return ()

    def effective_vector_size(self, vector_size):
        return 1

    def mesh_loop_lines(self):
        return (
            "    for (ptrdiff_t evbegin = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; evbegin < nelements; evbegin += (ptrdiff_t)blockDim.x * gridDim.x) {",
            "        const int nelems = 1;",
        )

    def mesh_template_line(self, geometry_mode):
        return (
            "template <typename scalar_t, typename geometry_t>"
            if geometry_mode == "isoparametric"
            else "template <typename scalar_t, typename jacobian_t>"
        )

    def mesh_function_line(self, implementation_name):
        return "__global__ void %s(" % implementation_name

    def success_return_lines(self):
        return ()

    def wrapper_call_lines(self, implementation_name, scalar_type, extra_template_args, wrapper_args):
        template_args = "%s%s" % (scalar_type, extra_template_args)
        return (
            "    const int block_size = 256;",
            "    const int grid_size = (int)((nelements + block_size - 1) / block_size);",
            "    sfem::codegen::%s<%s><<<grid_size, block_size>>>(%s);"
            % (implementation_name, template_args, ", ".join(wrapper_args)),
            "    return SFEM_SUCCESS;",
        )

    def scatter_add_lines(self, lhs, rhs, indent):
        return self.target.scatter_add_lines(lhs, rhs, indent)


@dataclass(frozen=True)
class OpenMPEnergySoAEmitter:
    """Opaque OpenMP emitter: consume an energy-SoA emission plan, emit files."""

    supports_op_wrapper: bool = True

    def emit_plan(self, plan):
        from .energy_codegen import generate_sfem_soa_cpp_files_for_element

        return generate_sfem_soa_cpp_files_for_element(
            plan.forms,
            prefix=plan.prefix,
            local_prefix=plan.local_prefix,
            emission_plan=plan.emission_plan,
            reference_data_plan=plan.reference_data_plan,
            diagnostics_plan=plan.diagnostics_plan,
            source_builder=OpenMPEnergySoASourceBuilder(),
        )


@dataclass(frozen=True)
class CUDAEnergySoAEmitter:
    """Opaque CUDA emitter: consume an energy-SoA emission plan, emit files."""

    supports_op_wrapper: bool = False

    def emit_plan(self, plan):
        from .energy_codegen import generate_sfem_soa_cpp_files_for_element

        return generate_sfem_soa_cpp_files_for_element(
            plan.forms,
            prefix=plan.prefix,
            local_prefix=plan.local_prefix,
            emission_plan=plan.emission_plan,
            reference_data_plan=plan.reference_data_plan,
            diagnostics_plan=plan.diagnostics_plan,
            source_builder=CUDAEnergySoASourceBuilder(),
        )
