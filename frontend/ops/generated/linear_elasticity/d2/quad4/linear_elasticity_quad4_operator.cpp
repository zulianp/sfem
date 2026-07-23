#include "../../op/sfem_GeneratedLinearElasticity_c_abi.hpp"

extern "C" int linear_elasticity_quad4_quad4_apply_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double mu,
        const double lmbda,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return linear_elasticity_proteus_quad4_proteus_quad4_apply_affine_mesh_soa(nelements, nnodes, proteus_elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, mu, lmbda, h_stride, hx, hy, out_stride, outx, outy);
}

extern "C" int linear_elasticity_quad4_quad4_apply_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float mu,
        const float lmbda,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return linear_elasticity_proteus_quad4_proteus_quad4_apply_affine_mesh_soa_float(nelements, nnodes, proteus_elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, mu, lmbda, h_stride, hx, hy, out_stride, outx, outy);
}

extern "C" void linear_elasticity_quad4_quad4_apply_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    linear_elasticity_proteus_quad4_proteus_quad4_apply_affine_mesh_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void linear_elasticity_quad4_quad4_apply_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    linear_elasticity_proteus_quad4_proteus_quad4_apply_affine_mesh_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" int linear_elasticity_quad4_quad4_apply_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double mu,
        const double lmbda,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return linear_elasticity_proteus_quad4_proteus_quad4_apply_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, mu, lmbda, h_stride, hx, hy, out_stride, outx, outy);
}

extern "C" int linear_elasticity_quad4_quad4_apply_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float mu,
        const float lmbda,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return linear_elasticity_proteus_quad4_proteus_quad4_apply_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, mu, lmbda, h_stride, hx, hy, out_stride, outx, outy);
}

extern "C" void linear_elasticity_quad4_quad4_apply_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    linear_elasticity_proteus_quad4_proteus_quad4_apply_isoparametric_mesh_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void linear_elasticity_quad4_quad4_apply_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    linear_elasticity_proteus_quad4_proteus_quad4_apply_isoparametric_mesh_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" double linear_elasticity_quad4_quad4_apply_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return linear_elasticity_proteus_quad4_proteus_quad4_apply_soa_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * linear_elasticity_quad4_quad4_apply_soa_diagnostics(
        void
) {
    return linear_elasticity_proteus_quad4_proteus_quad4_apply_soa_diagnostics();
}

extern "C" void linear_elasticity_quad4_quad4_apply_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    linear_elasticity_proteus_quad4_proteus_quad4_apply_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void linear_elasticity_quad4_quad4_apply_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    linear_elasticity_proteus_quad4_proteus_quad4_apply_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" int linear_elasticity_quad4_quad4_gradient_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double mu,
        const double lmbda,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return linear_elasticity_proteus_quad4_proteus_quad4_gradient_affine_mesh_soa(nelements, nnodes, proteus_elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, out_stride, outx, outy);
}

extern "C" int linear_elasticity_quad4_quad4_gradient_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float mu,
        const float lmbda,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return linear_elasticity_proteus_quad4_proteus_quad4_gradient_affine_mesh_soa_float(nelements, nnodes, proteus_elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, out_stride, outx, outy);
}

extern "C" void linear_elasticity_quad4_quad4_gradient_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    linear_elasticity_proteus_quad4_proteus_quad4_gradient_affine_mesh_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void linear_elasticity_quad4_quad4_gradient_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    linear_elasticity_proteus_quad4_proteus_quad4_gradient_affine_mesh_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" int linear_elasticity_quad4_quad4_gradient_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double mu,
        const double lmbda,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return linear_elasticity_proteus_quad4_proteus_quad4_gradient_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, mu, lmbda, u_stride, ux, uy, out_stride, outx, outy);
}

extern "C" int linear_elasticity_quad4_quad4_gradient_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float mu,
        const float lmbda,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return linear_elasticity_proteus_quad4_proteus_quad4_gradient_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, mu, lmbda, u_stride, ux, uy, out_stride, outx, outy);
}

extern "C" void linear_elasticity_quad4_quad4_gradient_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    linear_elasticity_proteus_quad4_proteus_quad4_gradient_isoparametric_mesh_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void linear_elasticity_quad4_quad4_gradient_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    linear_elasticity_proteus_quad4_proteus_quad4_gradient_isoparametric_mesh_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" double linear_elasticity_quad4_quad4_gradient_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return linear_elasticity_proteus_quad4_proteus_quad4_gradient_soa_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * linear_elasticity_quad4_quad4_gradient_soa_diagnostics(
        void
) {
    return linear_elasticity_proteus_quad4_proteus_quad4_gradient_soa_diagnostics();
}

extern "C" void linear_elasticity_quad4_quad4_gradient_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    linear_elasticity_proteus_quad4_proteus_quad4_gradient_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void linear_elasticity_quad4_quad4_gradient_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    linear_elasticity_proteus_quad4_proteus_quad4_gradient_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" int linear_elasticity_quad4_quad4_objective_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double mu,
        const double lmbda,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        double *const SFEM_RESTRICT value
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return linear_elasticity_proteus_quad4_proteus_quad4_objective_affine_mesh_soa(nelements, nnodes, proteus_elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, value);
}

extern "C" int linear_elasticity_quad4_quad4_objective_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float mu,
        const float lmbda,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        float *const SFEM_RESTRICT value
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return linear_elasticity_proteus_quad4_proteus_quad4_objective_affine_mesh_soa_float(nelements, nnodes, proteus_elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, value);
}

extern "C" void linear_elasticity_quad4_quad4_objective_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    linear_elasticity_proteus_quad4_proteus_quad4_objective_affine_mesh_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void linear_elasticity_quad4_quad4_objective_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    linear_elasticity_proteus_quad4_proteus_quad4_objective_affine_mesh_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" int linear_elasticity_quad4_quad4_objective_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double mu,
        const double lmbda,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        double *const SFEM_RESTRICT value
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return linear_elasticity_proteus_quad4_proteus_quad4_objective_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, mu, lmbda, u_stride, ux, uy, value);
}

extern "C" int linear_elasticity_quad4_quad4_objective_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float mu,
        const float lmbda,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        float *const SFEM_RESTRICT value
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return linear_elasticity_proteus_quad4_proteus_quad4_objective_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, mu, lmbda, u_stride, ux, uy, value);
}

extern "C" void linear_elasticity_quad4_quad4_objective_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    linear_elasticity_proteus_quad4_proteus_quad4_objective_isoparametric_mesh_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void linear_elasticity_quad4_quad4_objective_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    linear_elasticity_proteus_quad4_proteus_quad4_objective_isoparametric_mesh_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" double linear_elasticity_quad4_quad4_objective_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return linear_elasticity_proteus_quad4_proteus_quad4_objective_soa_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * linear_elasticity_quad4_quad4_objective_soa_diagnostics(
        void
) {
    return linear_elasticity_proteus_quad4_proteus_quad4_objective_soa_diagnostics();
}

extern "C" void linear_elasticity_quad4_quad4_objective_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    linear_elasticity_proteus_quad4_proteus_quad4_objective_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void linear_elasticity_quad4_quad4_objective_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    linear_elasticity_proteus_quad4_proteus_quad4_objective_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" int linear_elasticity_quad4_quad4_objective_steps_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double mu,
        const double lmbda,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return linear_elasticity_proteus_quad4_proteus_quad4_objective_steps_affine_mesh_soa(nelements, nnodes, proteus_elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, h_stride, hx, hy, nsteps, steps, value);
}

extern "C" int linear_elasticity_quad4_quad4_objective_steps_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float mu,
        const float lmbda,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return linear_elasticity_proteus_quad4_proteus_quad4_objective_steps_affine_mesh_soa_float(nelements, nnodes, proteus_elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, h_stride, hx, hy, nsteps, steps, value);
}

extern "C" int linear_elasticity_quad4_quad4_objective_steps_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double mu,
        const double lmbda,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return linear_elasticity_proteus_quad4_proteus_quad4_objective_steps_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, mu, lmbda, u_stride, ux, uy, h_stride, hx, hy, nsteps, steps, value);
}

extern "C" int linear_elasticity_quad4_quad4_objective_steps_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float mu,
        const float lmbda,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return linear_elasticity_proteus_quad4_proteus_quad4_objective_steps_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, mu, lmbda, u_stride, ux, uy, h_stride, hx, hy, nsteps, steps, value);
}
