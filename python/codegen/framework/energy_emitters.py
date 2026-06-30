from dataclasses import dataclass

try:
    from .tensor_product_geometry import sfem_geometry_kernels_header_source
except ImportError:
    from tensor_product_geometry import sfem_geometry_kernels_header_source


def _join_lines(lines):
    return "\n".join(line for line in lines if line != "")


def _cuda_geometry_header_source():
    return _join_lines(
        [
            "#ifndef SFEM_CODEGEN_GEOMETRY_KERNELS_HPP",
            "#define SFEM_CODEGEN_GEOMETRY_KERNELS_HPP",
            "",
            "#include <stddef.h>",
            "",
            "#ifndef SFEM_INLINE",
            "#define SFEM_INLINE __host__ __device__ __forceinline__",
            "#endif",
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
            "static SFEM_INLINE void geometry_jacobian_adjugate_and_determinant_2(",
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
            "static SFEM_INLINE void geometry_jacobian_adjugate_and_determinant_3(",
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
            "    static SFEM_INLINE void eval(",
            "            const ptrdiff_t nelems,",
            "            const scalar_t *const SFEM_RESTRICT coordinate_grad_ref,",
            "            scalar_t *const *const SFEM_RESTRICT adjugate,",
            "            scalar_t *const SFEM_RESTRICT determinant) {",
            "        for (int q = 0; q < N_QP; ++q) {",
            "            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {",
            "                const ptrdiff_t offset = q * VECTOR_SIZE + lane;",
            "                const scalar_t J00 = coordinate_grad_ref[((0 * N_QP + q) * 2 + 0) * VECTOR_SIZE + lane];",
            "                const scalar_t J01 = coordinate_grad_ref[((0 * N_QP + q) * 2 + 1) * VECTOR_SIZE + lane];",
            "                const scalar_t J10 = coordinate_grad_ref[((1 * N_QP + q) * 2 + 0) * VECTOR_SIZE + lane];",
            "                const scalar_t J11 = coordinate_grad_ref[((1 * N_QP + q) * 2 + 1) * VECTOR_SIZE + lane];",
            "                geometry_jacobian_adjugate_and_determinant_2<scalar_t>(",
            "                        J00, J01, J10, J11, adjugate, determinant, offset);",
            "            }",
            "        }",
            "    }",
            "};",
            "",
            "template <typename scalar_t, int N_QP, int VECTOR_SIZE>",
            "struct GeometryJacobianAdjugateDeterminant<scalar_t, 3, N_QP, VECTOR_SIZE> {",
            "    static SFEM_INLINE void eval(",
            "            const ptrdiff_t nelems,",
            "            const scalar_t *const SFEM_RESTRICT coordinate_grad_ref,",
            "            scalar_t *const *const SFEM_RESTRICT adjugate,",
            "            scalar_t *const SFEM_RESTRICT determinant) {",
            "        for (int q = 0; q < N_QP; ++q) {",
            "            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {",
            "                const ptrdiff_t offset = q * VECTOR_SIZE + lane;",
            "                const scalar_t J00 = coordinate_grad_ref[((0 * N_QP + q) * 3 + 0) * VECTOR_SIZE + lane];",
            "                const scalar_t J01 = coordinate_grad_ref[((0 * N_QP + q) * 3 + 1) * VECTOR_SIZE + lane];",
            "                const scalar_t J02 = coordinate_grad_ref[((0 * N_QP + q) * 3 + 2) * VECTOR_SIZE + lane];",
            "                const scalar_t J10 = coordinate_grad_ref[((1 * N_QP + q) * 3 + 0) * VECTOR_SIZE + lane];",
            "                const scalar_t J11 = coordinate_grad_ref[((1 * N_QP + q) * 3 + 1) * VECTOR_SIZE + lane];",
            "                const scalar_t J12 = coordinate_grad_ref[((1 * N_QP + q) * 3 + 2) * VECTOR_SIZE + lane];",
            "                const scalar_t J20 = coordinate_grad_ref[((2 * N_QP + q) * 3 + 0) * VECTOR_SIZE + lane];",
            "                const scalar_t J21 = coordinate_grad_ref[((2 * N_QP + q) * 3 + 1) * VECTOR_SIZE + lane];",
            "                const scalar_t J22 = coordinate_grad_ref[((2 * N_QP + q) * 3 + 2) * VECTOR_SIZE + lane];",
            "                geometry_jacobian_adjugate_and_determinant_3<scalar_t>(",
            "                        J00, J01, J02, J10, J11, J12, J20, J21, J22,",
            "                        adjugate, determinant, offset);",
            "            }",
            "        }",
            "    }",
            "};",
            "",
            "template <typename scalar_t, int DIM, int N_QP, int VECTOR_SIZE>",
            "static SFEM_INLINE void geometry_jacobian_adjugate_and_determinant(",
            "        const ptrdiff_t nelems,",
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
class EnergySoAKernelEmissionPlan:
    unit: object
    context: object
    forms: tuple
    prefix: str
    local_prefix: str
    local_kernel: object
    mesh_kernel: object
    emission_plan: object
    reference_data_plan: object = None
    diagnostics_plan: object = None
    local_signatures: tuple = ()
    mesh_signature: object = None


def energy_soa_kernel_emission_plan(unit, context):
    from .diagnostics_plan import kernel_diagnostics_plan_from_plan
    from .emission_plan import emission_plan_from_unit_context
    from .generation_plan import mesh_kernel_plan_from_context
    from .kernel_signature import (
        local_kernel_signatures_from_plan,
        mesh_kernel_signature_from_plan,
    )
    from .reference_data_plan import reference_data_plan_from_emission_plan

    prefix = unit.name
    local_kernel = unit.local_kernel_plan(context, prefix)
    mesh_kernel = mesh_kernel_plan_from_context(unit, context, prefix)
    element_plan = emission_plan_from_unit_context(unit, context)
    _validate_element_emission_plan(unit.name, element_plan)
    reference_data_plan = reference_data_plan_from_emission_plan(
        unit,
        context,
        element_plan,
        mesh_kernel.name,
    )
    local_signatures = local_kernel_signatures_from_plan(
        unit,
        element_plan,
        local_kernel.name,
        "energy_soa",
    )
    mesh_signature = mesh_kernel_signature_from_plan(
        unit,
        element_plan,
        mesh_kernel.name,
        "energy_soa",
    )
    diagnostics_plan = kernel_diagnostics_plan_from_plan(
        unit,
        element_plan,
        mesh_kernel.name,
        "energy_soa",
        reference_data_plan,
        mesh_signature,
        local_signatures,
    )
    return EnergySoAKernelEmissionPlan(
        unit=unit,
        context=context,
        forms=_energy_kernel_forms(unit),
        prefix=mesh_kernel.name,
        local_prefix=local_kernel.name,
        local_kernel=local_kernel,
        mesh_kernel=mesh_kernel,
        emission_plan=element_plan,
        reference_data_plan=reference_data_plan,
        diagnostics_plan=diagnostics_plan,
        local_signatures=local_signatures,
        mesh_signature=mesh_signature,
    )


def _energy_kernel_forms(unit):
    kernel_forms = tuple(
        expression_plan.source
        for expression_plan in unit.expression_plans
        if expression_plan.source is not None
    )
    if not kernel_forms:
        raise ValueError("energy kernel plan '%s' has no expression-plan kernel forms" % unit.name)
    return kernel_forms


def _validate_element_emission_plan(kernel_name, element_plan):
    _validate_geometry_specialization(
        kernel_name,
        element_plan.affine_geometry,
        element_plan.affine_specialization,
    )
    _validate_geometry_specialization(
        kernel_name,
        element_plan.isoparametric_geometry,
        element_plan.isoparametric_specialization,
    )


def _validate_geometry_specialization(kernel_name, geometry, specialization):
    rule = specialization.quadrature_rule
    if geometry.node.n_shape != rule.n_shape or geometry.node.n_qp != rule.n_qp:
        raise ValueError(
            "kernel plan '%s' geometry mode '%s' has (%d shapes, %d qp), "
            "but specialization has (%d shapes, %d qp)"
            % (
                kernel_name,
                geometry.mode.value,
                geometry.node.n_shape,
                geometry.node.n_qp,
                rule.n_shape,
                rule.n_qp,
            )
        )


@dataclass(frozen=True)
class OpenMPEnergySoASourceBuilder:
    operator_extension: str = "cpp"
    emit_objective_steps: bool = True

    def local_header_preamble_lines(self, math_name, tensor_product_name, basis_family):
        return (
            '#include "%s"' % math_name,
            '#include "%s"' % tensor_product_name,
            "",
            "#ifndef SFEM_INLINE",
            "#define SFEM_INLINE inline",
            "#endif",
            "",
            "#ifndef SFEM_RESTRICT",
            "#define SFEM_RESTRICT",
            "#endif",
        )

    def operator_preamble_lines(self, local_name, geometry_name, diagnostics_name):
        return (
            '#include "%s"' % local_name,
            '#include "%s"' % geometry_name,
            '#include "%s"' % diagnostics_name,
            "#ifdef _OPENMP",
            "#include <omp.h>",
            "#endif",
        )

    def geometry_header_source(self):
        return sfem_geometry_kernels_header_source()

    def emits_tensor_product_header(self, basis_family):
        return True

    def simd_lines(self):
        return ("#pragma omp simd",)

    def parallel_for_lines(self):
        return ("#pragma omp parallel for schedule(static)",)

    def effective_vector_size(self, vector_size):
        return int(vector_size)

    def mesh_loop_lines(self):
        return (
            "    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {",
            "        const ptrdiff_t nelems = MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);",
        )

    def mesh_template_line(self, geometry_mode):
        return (
            "template <typename scalar_t, typename geometry_t>"
            if geometry_mode == "isoparametric"
            else "template <typename scalar_t>"
        )

    def mesh_function_line(self, implementation_name):
        return "static SFEM_INLINE int %s(" % implementation_name

    def success_return_lines(self):
        return ("    return SFEM_SUCCESS;",)

    def wrapper_call_lines(self, implementation_name, scalar_type, extra_template_args, wrapper_args):
        return (
            "    return sfem::codegen::%s<%s%s>(%s);"
            % (implementation_name, scalar_type, extra_template_args, ", ".join(wrapper_args)),
        )

    def scatter_add_lines(self, lhs, rhs, indent):
        return (
            "%s#pragma omp atomic update" % indent,
            "%s%s += %s;" % (indent, lhs, rhs),
        )


@dataclass(frozen=True)
class CUDAEnergySoASourceBuilder:
    operator_extension: str = "cu"
    emit_objective_steps: bool = False

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
            "#ifndef SFEM_INLINE",
            "#define SFEM_INLINE __host__ __device__ __forceinline__",
            "#endif",
            "",
            "#ifndef SFEM_RESTRICT",
            "#define SFEM_RESTRICT __restrict__",
            "#endif",
        )

    def operator_preamble_lines(self, local_name, geometry_name, diagnostics_name):
        return (
            "#include <cuda_runtime.h>",
            '#include "%s"' % local_name,
            '#include "%s"' % geometry_name,
            '#include "%s"' % diagnostics_name,
        )

    def geometry_header_source(self):
        return _cuda_geometry_header_source()

    def emits_tensor_product_header(self, basis_family):
        return str(basis_family) == "tensor_product"

    def simd_lines(self):
        return ()

    def parallel_for_lines(self):
        return ()

    def effective_vector_size(self, vector_size):
        return 1

    def mesh_loop_lines(self):
        return (
            "    for (ptrdiff_t evbegin = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; evbegin < nelements; evbegin += (ptrdiff_t)blockDim.x * gridDim.x) {",
            "        const ptrdiff_t nelems = 1;",
        )

    def mesh_template_line(self, geometry_mode):
        return (
            "template <typename scalar_t, typename geometry_t>"
            if geometry_mode == "isoparametric"
            else "template <typename scalar_t>"
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
        return ("%satomicAdd(&(%s), %s);" % (indent, lhs, rhs),)


@dataclass(frozen=True)
class OpenMPEnergySoAEmitter:
    """Opaque OpenMP emitter: consume an energy-SoA emission context, emit files."""

    supports_op_wrapper: bool = True

    def plan(self, unit, context):
        return energy_soa_kernel_emission_plan(unit, context)

    def emit(self, unit, context):
        return self.emit_plan(self.plan(unit, context))

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
    """Opaque CUDA emitter: consume an energy-SoA emission context, emit files."""

    supports_op_wrapper: bool = False

    def plan(self, unit, context):
        return energy_soa_kernel_emission_plan(unit, context)

    def emit(self, unit, context):
        return self.emit_plan(self.plan(unit, context))

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
