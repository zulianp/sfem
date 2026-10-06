#include "../../../op/cuda/sfem_GeneratedLaplace_cuda_c_abi.hpp"

extern "C" int cu_laplace_proteus_quad4_apply_i_msoa(
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
extern "C" int cu_laplace_proteus_quad4_gradient_i_msoa(
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
extern "C" int cu_laplace_proteus_quad4_objective_steps_i_msoa(
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
extern "C" const sfem::codegen::KernelDiagnostics * cu_laplace_proteus_quad4_apply_soa_diagnostics(
        void
);
extern "C" const sfem::codegen::KernelDiagnostics * cu_laplace_proteus_quad4_gradient_soa_diagnostics(
        void
);
extern "C" const sfem::codegen::KernelDiagnostics * cu_laplace_proteus_quad4_objective_soa_diagnostics(
        void
);

extern "C" int cu_laplace_quad4_apply_i_msoa(
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
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return cu_laplace_proteus_quad4_apply_i_msoa(scalar_bytes, nelements, nnodes, proteus_elements, points, kappa, h_stride, hx, out_stride, outx, stream);
}

extern "C" int cu_laplace_quad4_gradient_i_msoa(
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
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return cu_laplace_proteus_quad4_gradient_i_msoa(scalar_bytes, nelements, nnodes, proteus_elements, points, kappa, u_stride, ux, out_stride, outx, stream);
}

extern "C" int cu_laplace_quad4_objective_steps_i_msoa(
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
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return cu_laplace_proteus_quad4_objective_steps_i_msoa(scalar_bytes, nelements, nnodes, proteus_elements, points, kappa, u_stride, ux, h_stride, hx, nsteps, steps, value, stream);
}

extern "C" const sfem::codegen::KernelDiagnostics * cu_laplace_quad4_apply_soa_diagnostics(
        void
) {
    return cu_laplace_proteus_quad4_apply_soa_diagnostics();
}

extern "C" const sfem::codegen::KernelDiagnostics * cu_laplace_quad4_gradient_soa_diagnostics(
        void
) {
    return cu_laplace_proteus_quad4_gradient_soa_diagnostics();
}

extern "C" const sfem::codegen::KernelDiagnostics * cu_laplace_quad4_objective_soa_diagnostics(
        void
) {
    return cu_laplace_proteus_quad4_objective_soa_diagnostics();
}
