#include "../../../op/cuda/sfem_GeneratedLaplace_cuda_c_abi.hpp"

extern "C" int cu_laplace_proteus_hex8_apply_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t kappa,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const stream
);
extern "C" int cu_laplace_proteus_hex8_gradient_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const stream
);
extern "C" int cu_laplace_proteus_hex8_objective_steps_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const int nsteps,
        const void *const RSTR steps,
        void *const RSTR value,
        void *const stream
);
extern "C" int cu_laplace_proteus_hex8_apply_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const real_t kappa,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const stream
);
extern "C" int cu_laplace_proteus_hex8_gradient_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const stream
);
extern "C" int cu_laplace_proteus_hex8_objective_steps_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const int nsteps,
        const void *const RSTR steps,
        void *const RSTR value,
        void *const stream
);
extern "C" const sfem::codegen::KernelDiagnostics * cu_laplace_proteus_hex8_apply_soa_diagnostics(
        void
);
extern "C" const sfem::codegen::KernelDiagnostics * cu_laplace_proteus_hex8_gradient_soa_diagnostics(
        void
);
extern "C" const sfem::codegen::KernelDiagnostics * cu_laplace_proteus_hex8_objective_soa_diagnostics(
        void
);

extern "C" int cu_laplace_hex8_apply_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t kappa,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const stream
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
    return cu_laplace_proteus_hex8_apply_i_msoa(scalar_bytes, nelements, nnodes, proteus_elements, points, kappa, h_stride, hx, out_stride, outx, stream);
}

extern "C" int cu_laplace_hex8_gradient_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const stream
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
    return cu_laplace_proteus_hex8_gradient_i_msoa(scalar_bytes, nelements, nnodes, proteus_elements, points, kappa, u_stride, ux, out_stride, outx, stream);
}

extern "C" int cu_laplace_hex8_objective_steps_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const int nsteps,
        const void *const RSTR steps,
        void *const RSTR value,
        void *const stream
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
    return cu_laplace_proteus_hex8_objective_steps_i_msoa(scalar_bytes, nelements, nnodes, proteus_elements, points, kappa, u_stride, ux, h_stride, hx, nsteps, steps, value, stream);
}

extern "C" int cu_laplace_hex8_apply_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const real_t kappa,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const stream
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
    return cu_laplace_proteus_hex8_apply_a_msoa(scalar_bytes, nelements, nnodes, proteus_elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, kappa, h_stride, hx, out_stride, outx, stream);
}

extern "C" int cu_laplace_hex8_gradient_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const stream
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
    return cu_laplace_proteus_hex8_gradient_a_msoa(scalar_bytes, nelements, nnodes, proteus_elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, kappa, u_stride, ux, out_stride, outx, stream);
}

extern "C" int cu_laplace_hex8_objective_steps_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const int nsteps,
        const void *const RSTR steps,
        void *const RSTR value,
        void *const stream
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
    return cu_laplace_proteus_hex8_objective_steps_a_msoa(scalar_bytes, nelements, nnodes, proteus_elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, kappa, u_stride, ux, h_stride, hx, nsteps, steps, value, stream);
}

extern "C" const sfem::codegen::KernelDiagnostics * cu_laplace_hex8_apply_soa_diagnostics(
        void
) {
    return cu_laplace_proteus_hex8_apply_soa_diagnostics();
}

extern "C" const sfem::codegen::KernelDiagnostics * cu_laplace_hex8_gradient_soa_diagnostics(
        void
) {
    return cu_laplace_proteus_hex8_gradient_soa_diagnostics();
}

extern "C" const sfem::codegen::KernelDiagnostics * cu_laplace_hex8_objective_soa_diagnostics(
        void
) {
    return cu_laplace_proteus_hex8_objective_soa_diagnostics();
}
