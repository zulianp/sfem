#include "../../op/sfem_GeneratedLaplace_c_abi.hpp"

extern "C" int laplace_proteus_quad4_apply_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const ptrdiff_t out_stride,
        double *const RSTR outx
);
extern "C" int laplace_proteus_quad4_apply_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const ptrdiff_t out_stride,
        float *const RSTR outx
);
extern "C" int laplace_proteus_quad4_gradient_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const ptrdiff_t out_stride,
        double *const RSTR outx
);
extern "C" int laplace_proteus_quad4_gradient_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const ptrdiff_t out_stride,
        float *const RSTR outx
);
extern "C" int laplace_proteus_quad4_hessian_bsr_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        double *const RSTR values
);
extern "C" int laplace_proteus_quad4_hessian_bsr_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        float *const RSTR values
);
extern "C" int laplace_proteus_quad4_hessian_crs_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        double *const RSTR values
);
extern "C" int laplace_proteus_quad4_hessian_crs_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        float *const RSTR values
);
extern "C" int laplace_proteus_quad4_objective_steps_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const int nsteps,
        const double *const RSTR steps,
        double *const RSTR value
);
extern "C" int laplace_proteus_quad4_objective_steps_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const int nsteps,
        const float *const RSTR steps,
        float *const RSTR value
);
extern "C" const sfem::codegen::KernelDiagnostics * laplace_proteus_quad4_apply_soa_diagnostics(
        void
);
extern "C" const sfem::codegen::KernelDiagnostics * laplace_proteus_quad4_gradient_soa_diagnostics(
        void
);
extern "C" const sfem::codegen::KernelDiagnostics * laplace_proteus_quad4_objective_soa_diagnostics(
        void
);

extern "C" int laplace_quad4_apply_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const ptrdiff_t out_stride,
        double *const RSTR outx
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return laplace_proteus_quad4_apply_i_msoa(nelements, nnodes, proteus_elements, points, kappa, h_stride, hx, out_stride, outx);
}

extern "C" int laplace_quad4_apply_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const ptrdiff_t out_stride,
        float *const RSTR outx
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return laplace_proteus_quad4_apply_i_msoa_float(nelements, nnodes, proteus_elements, points, kappa, h_stride, hx, out_stride, outx);
}

extern "C" int laplace_quad4_gradient_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const ptrdiff_t out_stride,
        double *const RSTR outx
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return laplace_proteus_quad4_gradient_i_msoa(nelements, nnodes, proteus_elements, points, kappa, u_stride, ux, out_stride, outx);
}

extern "C" int laplace_quad4_gradient_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const ptrdiff_t out_stride,
        float *const RSTR outx
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return laplace_proteus_quad4_gradient_i_msoa_float(nelements, nnodes, proteus_elements, points, kappa, u_stride, ux, out_stride, outx);
}

extern "C" int laplace_quad4_hessian_bsr_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        double *const RSTR values
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return laplace_proteus_quad4_hessian_bsr_i_msoa(nelements, nnodes, proteus_elements, points, kappa, rowptr, colidx, values);
}

extern "C" int laplace_quad4_hessian_bsr_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        float *const RSTR values
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return laplace_proteus_quad4_hessian_bsr_i_msoa_float(nelements, nnodes, proteus_elements, points, kappa, rowptr, colidx, values);
}

extern "C" int laplace_quad4_hessian_crs_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        double *const RSTR values
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return laplace_proteus_quad4_hessian_crs_i_msoa(nelements, nnodes, proteus_elements, points, kappa, rowptr, colidx, values);
}

extern "C" int laplace_quad4_hessian_crs_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        float *const RSTR values
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return laplace_proteus_quad4_hessian_crs_i_msoa_float(nelements, nnodes, proteus_elements, points, kappa, rowptr, colidx, values);
}

extern "C" int laplace_quad4_objective_steps_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const int nsteps,
        const double *const RSTR steps,
        double *const RSTR value
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return laplace_proteus_quad4_objective_steps_i_msoa(nelements, nnodes, proteus_elements, points, kappa, u_stride, ux, h_stride, hx, nsteps, steps, value);
}

extern "C" int laplace_quad4_objective_steps_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const int nsteps,
        const float *const RSTR steps,
        float *const RSTR value
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return laplace_proteus_quad4_objective_steps_i_msoa_float(nelements, nnodes, proteus_elements, points, kappa, u_stride, ux, h_stride, hx, nsteps, steps, value);
}

extern "C" const sfem::codegen::KernelDiagnostics * laplace_quad4_apply_soa_diagnostics(
        void
) {
    return laplace_proteus_quad4_apply_soa_diagnostics();
}

extern "C" const sfem::codegen::KernelDiagnostics * laplace_quad4_gradient_soa_diagnostics(
        void
) {
    return laplace_proteus_quad4_gradient_soa_diagnostics();
}

extern "C" const sfem::codegen::KernelDiagnostics * laplace_quad4_objective_soa_diagnostics(
        void
) {
    return laplace_proteus_quad4_objective_soa_diagnostics();
}
