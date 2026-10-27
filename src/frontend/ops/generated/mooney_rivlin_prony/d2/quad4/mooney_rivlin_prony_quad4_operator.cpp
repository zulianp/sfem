#include "sfem_GeneratedMooneyRivlinProny_c_abi.hpp"

extern "C" int mooney_rivlin_prony_proteus_quad4_apply_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C01,
        const real_t C10,
        const real_t K,
        const real_t iso_relaxation_gamma,
        const void *const RSTR iso_history,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy
);
extern "C" int mooney_rivlin_prony_proteus_quad4_gradient_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C01,
        const real_t C10,
        const real_t K,
        const real_t iso_relaxation_gamma,
        const void *const RSTR iso_history,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy
);
extern "C" int mooney_rivlin_prony_proteus_quad4_hessian_bsr_i_msoa(
        const int scalar_bytes,
        const int values_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C01,
        const real_t C10,
        const real_t K,
        const real_t iso_relaxation_gamma,
        const void *const RSTR iso_history,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        void *const RSTR values
);
extern "C" int mooney_rivlin_prony_proteus_quad4_objective_steps_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C01,
        const real_t C10,
        const real_t K,
        const real_t iso_relaxation_gamma,
        const void *const RSTR iso_history,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const int nsteps,
        const void *const RSTR steps,
        void *const RSTR value
);
extern "C" int mooney_rivlin_prony_proteus_quad4_quadrature_output_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C01,
        const real_t C10,
        void *const RSTR iso_stress,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy
);
extern "C" const sfem::codegen::KernelDiagnostics * mooney_rivlin_prony_proteus_quad4_apply_soa_diagnostics(
        void
);
extern "C" const sfem::codegen::KernelDiagnostics * mooney_rivlin_prony_proteus_quad4_gradient_soa_diagnostics(
        void
);
extern "C" const sfem::codegen::KernelDiagnostics * mooney_rivlin_prony_proteus_quad4_objective_soa_diagnostics(
        void
);
extern "C" const sfem::codegen::KernelDiagnostics * mooney_rivlin_prony_proteus_quad4_quadrature_output_soa_diagnostics(
        void
);

extern "C" int mooney_rivlin_prony_quad4_apply_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C01,
        const real_t C10,
        const real_t K,
        const real_t iso_relaxation_gamma,
        const void *const RSTR iso_history,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return mooney_rivlin_prony_proteus_quad4_apply_i_msoa(scalar_bytes, nelements, nnodes, proteus_elements, points, C01, C10, K, iso_relaxation_gamma, iso_history, u_stride, ux, uy, h_stride, hx, hy, out_stride, outx, outy);
}

extern "C" int mooney_rivlin_prony_quad4_gradient_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C01,
        const real_t C10,
        const real_t K,
        const real_t iso_relaxation_gamma,
        const void *const RSTR iso_history,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return mooney_rivlin_prony_proteus_quad4_gradient_i_msoa(scalar_bytes, nelements, nnodes, proteus_elements, points, C01, C10, K, iso_relaxation_gamma, iso_history, u_stride, ux, uy, out_stride, outx, outy);
}

extern "C" int mooney_rivlin_prony_quad4_hessian_bsr_i_msoa(
        const int scalar_bytes,
        const int values_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C01,
        const real_t C10,
        const real_t K,
        const real_t iso_relaxation_gamma,
        const void *const RSTR iso_history,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        void *const RSTR values
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return mooney_rivlin_prony_proteus_quad4_hessian_bsr_i_msoa(scalar_bytes, values_bytes, nelements, nnodes, proteus_elements, points, C01, C10, K, iso_relaxation_gamma, iso_history, u_stride, ux, uy, rowptr, colidx, values);
}

extern "C" int mooney_rivlin_prony_quad4_objective_steps_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C01,
        const real_t C10,
        const real_t K,
        const real_t iso_relaxation_gamma,
        const void *const RSTR iso_history,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const int nsteps,
        const void *const RSTR steps,
        void *const RSTR value
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return mooney_rivlin_prony_proteus_quad4_objective_steps_i_msoa(scalar_bytes, nelements, nnodes, proteus_elements, points, C01, C10, K, iso_relaxation_gamma, iso_history, u_stride, ux, uy, h_stride, hx, hy, nsteps, steps, value);
}

extern "C" int mooney_rivlin_prony_quad4_quadrature_output_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C01,
        const real_t C10,
        void *const RSTR iso_stress,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return mooney_rivlin_prony_proteus_quad4_quadrature_output_i_msoa(scalar_bytes, nelements, nnodes, proteus_elements, points, C01, C10, iso_stress, u_stride, ux, uy);
}

extern "C" const sfem::codegen::KernelDiagnostics * mooney_rivlin_prony_quad4_apply_soa_diagnostics(
        void
) {
    return mooney_rivlin_prony_proteus_quad4_apply_soa_diagnostics();
}

extern "C" const sfem::codegen::KernelDiagnostics * mooney_rivlin_prony_quad4_gradient_soa_diagnostics(
        void
) {
    return mooney_rivlin_prony_proteus_quad4_gradient_soa_diagnostics();
}

extern "C" const sfem::codegen::KernelDiagnostics * mooney_rivlin_prony_quad4_objective_soa_diagnostics(
        void
) {
    return mooney_rivlin_prony_proteus_quad4_objective_soa_diagnostics();
}

extern "C" const sfem::codegen::KernelDiagnostics * mooney_rivlin_prony_quad4_quadrature_output_soa_diagnostics(
        void
) {
    return mooney_rivlin_prony_proteus_quad4_quadrature_output_soa_diagnostics();
}
