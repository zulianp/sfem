#include "../../../op/cuda/sfem_GeneratedBodyForce_cuda_c_abi.hpp"

extern "C" int cu_body_force_proteus_quad4_residual_i_maos(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        void *const RSTR output,
        void *const stream
);
extern "C" int cu_body_force_proteus_quad4_residual_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t density,
        const real_t g0,
        const real_t g1,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const stream
);
extern "C" int cu_body_force_proteus_quad4_residual_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_det0,
        const real_t density,
        const real_t g0,
        const real_t g1,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const stream
);
extern "C" const sfem::codegen::KernelDiagnostics * cu_body_force_proteus_quad4_jacobian_action_esoa_diagnostics(
        void
);
extern "C" const sfem::codegen::KernelDiagnostics * cu_body_force_proteus_quad4_residual_esoa_diagnostics(
        void
);

extern "C" int cu_body_force_quad4_residual_i_maos(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        void *const RSTR output,
        void *const stream
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return cu_body_force_proteus_quad4_residual_i_maos(scalar_bytes, nelements, nnodes, proteus_elements, points, parameters, output, stream);
}

extern "C" int cu_body_force_quad4_residual_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t density,
        const real_t g0,
        const real_t g1,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const stream
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return cu_body_force_proteus_quad4_residual_i_msoa(scalar_bytes, nelements, nnodes, proteus_elements, points, density, g0, g1, out_stride, u0_out, u1_out, stream);
}

extern "C" int cu_body_force_quad4_residual_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_det0,
        const real_t density,
        const real_t g0,
        const real_t g1,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const stream
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return cu_body_force_proteus_quad4_residual_a_msoa(scalar_bytes, nelements, nnodes, proteus_elements, g_det0, density, g0, g1, out_stride, u0_out, u1_out, stream);
}

extern "C" const sfem::codegen::KernelDiagnostics * cu_body_force_quad4_jacobian_action_esoa_diagnostics(
        void
) {
    return cu_body_force_proteus_quad4_jacobian_action_esoa_diagnostics();
}

extern "C" const sfem::codegen::KernelDiagnostics * cu_body_force_quad4_residual_esoa_diagnostics(
        void
) {
    return cu_body_force_proteus_quad4_residual_esoa_diagnostics();
}
