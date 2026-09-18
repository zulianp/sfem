#include "../../op/sfem_GeneratedBodyForce_c_abi.hpp"

extern "C" int body_force_proteus_hex8_residual_i_maos(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        void *const RSTR output
);
extern "C" int body_force_proteus_hex8_residual_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t density,
        const real_t g0,
        const real_t g1,
        const real_t g2,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const RSTR u2_out
);
extern "C" int body_force_proteus_hex8_residual_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_det0,
        const real_t density,
        const real_t g0,
        const real_t g1,
        const real_t g2,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const RSTR u2_out
);
extern "C" const sfem::codegen::KernelDiagnostics * body_force_proteus_hex8_jacobian_action_esoa_diagnostics(
        void
);
extern "C" const sfem::codegen::KernelDiagnostics * body_force_proteus_hex8_residual_esoa_diagnostics(
        void
);

extern "C" int body_force_hex8_residual_i_maos(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        void *const RSTR output
) {
    idx_t *proteus_elements[8] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2],
        elements[4],
        elements[5],
        elements[7],
        elements[6]
    };
    return body_force_proteus_hex8_residual_i_maos(scalar_bytes, nelements, nnodes, proteus_elements, points, parameters, output);
}

extern "C" int body_force_hex8_residual_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t density,
        const real_t g0,
        const real_t g1,
        const real_t g2,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const RSTR u2_out
) {
    idx_t *proteus_elements[8] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2],
        elements[4],
        elements[5],
        elements[7],
        elements[6]
    };
    return body_force_proteus_hex8_residual_i_msoa(scalar_bytes, nelements, nnodes, proteus_elements, points, density, g0, g1, g2, out_stride, u0_out, u1_out, u2_out);
}

extern "C" int body_force_hex8_residual_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_det0,
        const real_t density,
        const real_t g0,
        const real_t g1,
        const real_t g2,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const RSTR u2_out
) {
    idx_t *proteus_elements[8] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2],
        elements[4],
        elements[5],
        elements[7],
        elements[6]
    };
    return body_force_proteus_hex8_residual_a_msoa(scalar_bytes, nelements, nnodes, proteus_elements, g_det0, density, g0, g1, g2, out_stride, u0_out, u1_out, u2_out);
}

extern "C" const sfem::codegen::KernelDiagnostics * body_force_hex8_jacobian_action_esoa_diagnostics(
        void
) {
    return body_force_proteus_hex8_jacobian_action_esoa_diagnostics();
}

extern "C" const sfem::codegen::KernelDiagnostics * body_force_hex8_residual_esoa_diagnostics(
        void
) {
    return body_force_proteus_hex8_residual_esoa_diagnostics();
}
