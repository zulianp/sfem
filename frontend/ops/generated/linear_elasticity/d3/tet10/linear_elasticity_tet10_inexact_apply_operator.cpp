#include "../../op/sfem_GeneratedLinearElasticity_c_abi.hpp"

#include "linear_elasticity_tet10_inexact_apply_inline.hpp"

extern "C" int linear_elasticity_tet10_inexact_apply_tangent_affine_mesh_soa(
        const ptrdiff_t nelements,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t tangent_element_stride,
        const ptrdiff_t tangent_component_stride,
        metric_tensor_t *const SFEM_RESTRICT tangent
) {
    return sfem::codegen::linear_elasticity_tet10_inexact_apply_tangent_affine_mesh_soa_impl<double, geom_t, metric_tensor_t>(
            nelements, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0,
            lmbda, mu,
            u_stride, ux, uy, uz,
            tangent_element_stride, tangent_component_stride, tangent);
}

extern "C" int linear_elasticity_tet10_inexact_apply_stored_affine_mesh_soa(
        const ptrdiff_t nelements,
        idx_t **const SFEM_RESTRICT elements,
        const ptrdiff_t tangent_element_stride,
        const ptrdiff_t tangent_component_stride,
        const metric_tensor_t *const SFEM_RESTRICT tangent,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
) {
    return sfem::codegen::linear_elasticity_tet10_inexact_apply_stored_affine_mesh_soa_impl<double, metric_tensor_t>(
            nelements, elements,
            tangent_element_stride, tangent_component_stride, tangent,
            h_stride, hx, hy, hz,
            out_stride, outx, outy, outz);
}

extern "C" int linear_elasticity_tet10_inexact_apply_compressed_affine_mesh_soa(
        const ptrdiff_t nelements,
        idx_t **const SFEM_RESTRICT elements,
        const ptrdiff_t tangent_element_stride,
        const ptrdiff_t tangent_component_stride,
        const compressed_t *const SFEM_RESTRICT tangent,
        const scaling_t *const SFEM_RESTRICT scaling,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
) {
    return sfem::codegen::linear_elasticity_tet10_inexact_apply_compressed_affine_mesh_soa_impl<double, compressed_t, scaling_t>(
            nelements, elements,
            tangent_element_stride, tangent_component_stride, tangent, scaling,
            h_stride, hx, hy, hz,
            out_stride, outx, outy, outz);
}

extern "C" int linear_elasticity_tet10_inexact_apply_tangent_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t tangent_element_stride,
        const ptrdiff_t tangent_component_stride,
        metric_tensor_t *const SFEM_RESTRICT tangent
) {
    return sfem::codegen::linear_elasticity_tet10_inexact_apply_tangent_affine_mesh_soa_impl<float, geom_t, metric_tensor_t>(
            nelements, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0,
            lmbda, mu,
            u_stride, ux, uy, uz,
            tangent_element_stride, tangent_component_stride, tangent);
}

extern "C" int linear_elasticity_tet10_inexact_apply_stored_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        idx_t **const SFEM_RESTRICT elements,
        const ptrdiff_t tangent_element_stride,
        const ptrdiff_t tangent_component_stride,
        const metric_tensor_t *const SFEM_RESTRICT tangent,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
) {
    return sfem::codegen::linear_elasticity_tet10_inexact_apply_stored_affine_mesh_soa_impl<float, metric_tensor_t>(
            nelements, elements,
            tangent_element_stride, tangent_component_stride, tangent,
            h_stride, hx, hy, hz,
            out_stride, outx, outy, outz);
}

extern "C" int linear_elasticity_tet10_inexact_apply_compressed_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        idx_t **const SFEM_RESTRICT elements,
        const ptrdiff_t tangent_element_stride,
        const ptrdiff_t tangent_component_stride,
        const compressed_t *const SFEM_RESTRICT tangent,
        const scaling_t *const SFEM_RESTRICT scaling,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
) {
    return sfem::codegen::linear_elasticity_tet10_inexact_apply_compressed_affine_mesh_soa_impl<float, compressed_t, scaling_t>(
            nelements, elements,
            tangent_element_stride, tangent_component_stride, tangent, scaling,
            h_stride, hx, hy, hz,
            out_stride, outx, outy, outz);
}
