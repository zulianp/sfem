#include "../../../kernel_math.hpp"

namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t, typename tangent_t>
static SFEM_INLINE int linear_elasticity_tri3_inexact_apply_tangent_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        idx_t **const SFEM_RESTRICT elements,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const scalar_t lmbda,
        const scalar_t mu,
        const ptrdiff_t u_stride,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const ptrdiff_t tangent_element_stride,
        const ptrdiff_t tangent_component_stride,
        tangent_t *const SFEM_RESTRICT tangent
) {
    #pragma omp parallel for schedule(static)
    for (ptrdiff_t element = 0; element < nelements; ++element) {
        const idx_t ev0 = elements[0][element];
        const idx_t ev1 = elements[1][element];
        const idx_t ev2 = elements[2][element];
        const scalar_t adjugate0 = scalar_t(g_jacobian_adjugate0[element]);
        const scalar_t adjugate1 = scalar_t(g_jacobian_adjugate1[element]);
        const scalar_t adjugate2 = scalar_t(g_jacobian_adjugate2[element]);
        const scalar_t adjugate3 = scalar_t(g_jacobian_adjugate3[element]);
        const scalar_t determinant = scalar_t(g_jacobian_determinant0[element]);
        const scalar_t tangent_t0 = pow_m1(determinant);
        const scalar_t tangent_t1 = pow_2(adjugate1);
        const scalar_t tangent_t2 = pow_2(adjugate0);
        const scalar_t tangent_t3 = lmbda + scalar_t(2)*mu;
        const scalar_t tangent_t4 = adjugate1*mu;
        const scalar_t tangent_t5 = adjugate0*adjugate2;
        const scalar_t tangent_t6 = adjugate0*lmbda;
        const scalar_t tangent_t7 = pow_2(adjugate3);
        const scalar_t tangent_t8 = pow_2(adjugate2);
        const scalar_t tangent_t9 = adjugate3*mu;
        const scalar_t tangent_t10 = adjugate2*lmbda;
        const scalar_t tangent0 = tangent_t0*(mu*tangent_t1 + tangent_t2*tangent_t3);
        const scalar_t tangent1 = tangent_t0*(adjugate3*tangent_t4 + tangent_t3*tangent_t5);
        const scalar_t tangent2 = tangent_t0*(adjugate0*tangent_t4 + adjugate1*tangent_t6);
        const scalar_t tangent3 = tangent_t0*(adjugate2*tangent_t4 + adjugate3*tangent_t6);
        const scalar_t tangent4 = tangent_t0*(mu*tangent_t7 + tangent_t3*tangent_t8);
        const scalar_t tangent5 = tangent_t0*(adjugate0*tangent_t9 + adjugate1*tangent_t10);
        const scalar_t tangent6 = tangent_t0*(adjugate2*tangent_t9 + adjugate3*tangent_t10);
        const scalar_t tangent7 = tangent_t0*(mu*tangent_t2 + tangent_t1*tangent_t3);
        const scalar_t tangent8 = tangent_t0*(adjugate1*adjugate3*tangent_t3 + mu*tangent_t5);
        const scalar_t tangent9 = tangent_t0*(mu*tangent_t8 + tangent_t3*tangent_t7);
        tangent[element * tangent_element_stride + 0 * tangent_component_stride] = tangent_t(tangent0);
        tangent[element * tangent_element_stride + 1 * tangent_component_stride] = tangent_t(tangent1);
        tangent[element * tangent_element_stride + 2 * tangent_component_stride] = tangent_t(tangent2);
        tangent[element * tangent_element_stride + 3 * tangent_component_stride] = tangent_t(tangent3);
        tangent[element * tangent_element_stride + 4 * tangent_component_stride] = tangent_t(tangent4);
        tangent[element * tangent_element_stride + 5 * tangent_component_stride] = tangent_t(tangent5);
        tangent[element * tangent_element_stride + 6 * tangent_component_stride] = tangent_t(tangent6);
        tangent[element * tangent_element_stride + 7 * tangent_component_stride] = tangent_t(tangent7);
        tangent[element * tangent_element_stride + 8 * tangent_component_stride] = tangent_t(tangent8);
        tangent[element * tangent_element_stride + 9 * tangent_component_stride] = tangent_t(tangent9);
    }

    return SFEM_SUCCESS;
}

template <typename scalar_t, typename tangent_t>
static SFEM_INLINE int linear_elasticity_tri3_inexact_apply_stored_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        idx_t **const SFEM_RESTRICT elements,
        const ptrdiff_t tangent_element_stride,
        const ptrdiff_t tangent_component_stride,
        const tangent_t *const SFEM_RESTRICT tangent,
        const ptrdiff_t h_stride,
        const scalar_t *const SFEM_RESTRICT hx,
        const scalar_t *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT outx,
        scalar_t *const SFEM_RESTRICT outy
) {
    #pragma omp parallel for schedule(static)
    for (ptrdiff_t element = 0; element < nelements; ++element) {
        const idx_t ev0 = elements[0][element];
        const idx_t ev1 = elements[1][element];
        const idx_t ev2 = elements[2][element];
        const scalar_t hx_0 = hx[ev0 * h_stride];
        const scalar_t hx_1 = hx[ev1 * h_stride];
        const scalar_t hx_2 = hx[ev2 * h_stride];
        const scalar_t hy_0 = hy[ev0 * h_stride];
        const scalar_t hy_1 = hy[ev1 * h_stride];
        const scalar_t hy_2 = hy[ev2 * h_stride];
        const scalar_t tangent0 = scalar_t(tangent[element * tangent_element_stride + 0 * tangent_component_stride]);
        const scalar_t tangent1 = scalar_t(tangent[element * tangent_element_stride + 1 * tangent_component_stride]);
        const scalar_t tangent2 = scalar_t(tangent[element * tangent_element_stride + 2 * tangent_component_stride]);
        const scalar_t tangent3 = scalar_t(tangent[element * tangent_element_stride + 3 * tangent_component_stride]);
        const scalar_t tangent4 = scalar_t(tangent[element * tangent_element_stride + 4 * tangent_component_stride]);
        const scalar_t tangent5 = scalar_t(tangent[element * tangent_element_stride + 5 * tangent_component_stride]);
        const scalar_t tangent6 = scalar_t(tangent[element * tangent_element_stride + 6 * tangent_component_stride]);
        const scalar_t tangent7 = scalar_t(tangent[element * tangent_element_stride + 7 * tangent_component_stride]);
        const scalar_t tangent8 = scalar_t(tangent[element * tangent_element_stride + 8 * tangent_component_stride]);
        const scalar_t tangent9 = scalar_t(tangent[element * tangent_element_stride + 9 * tangent_component_stride]);
        const scalar_t compressed_increment_t0 = ((scalar_t(1) / scalar_t(2)))*hx_0;
        const scalar_t compressed_increment_t1 = ((scalar_t(1) / scalar_t(2)))*hy_0;
        const scalar_t pa_p0_0_0 = compressed_increment_t0 - (scalar_t(1) / scalar_t(2))*hx_1;
        const scalar_t pa_p0_0_1 = compressed_increment_t0 - (scalar_t(1) / scalar_t(2))*hx_2;
        const scalar_t pa_p1_0_0 = compressed_increment_t1 - (scalar_t(1) / scalar_t(2))*hy_1;
        const scalar_t pa_p1_0_1 = compressed_increment_t1 - (scalar_t(1) / scalar_t(2))*hy_2;
        const scalar_t pa_y0_0_0 = pa_p0_0_0*tangent0 + pa_p0_0_1*tangent1 + pa_p1_0_0*tangent2 + pa_p1_0_1*tangent3;
        const scalar_t pa_y0_0_1 = pa_p0_0_0*tangent1 + pa_p0_0_1*tangent4 + pa_p1_0_0*tangent5 + pa_p1_0_1*tangent6;
        const scalar_t pa_y1_0_0 = pa_p0_0_0*tangent2 + pa_p0_0_1*tangent5 + pa_p1_0_0*tangent7 + pa_p1_0_1*tangent8;
        const scalar_t pa_y1_0_1 = pa_p0_0_0*tangent3 + pa_p0_0_1*tangent6 + pa_p1_0_0*tangent8 + pa_p1_0_1*tangent9;
        const scalar_t pa_q0_0_0 = scalar_t(2)*pa_y0_0_0;
        const scalar_t pa_q0_0_1 = scalar_t(2)*pa_y0_0_1;
        const scalar_t pa_q1_0_0 = scalar_t(2)*pa_y1_0_0;
        const scalar_t pa_q1_0_1 = scalar_t(2)*pa_y1_0_1;
        const scalar_t output_t0 = ((scalar_t(1) / scalar_t(2)))*pa_q0_0_0;
        const scalar_t output_t1 = ((scalar_t(1) / scalar_t(2)))*pa_q0_0_1;
        const scalar_t output_t2 = ((scalar_t(1) / scalar_t(2)))*pa_q1_0_0;
        const scalar_t output_t3 = ((scalar_t(1) / scalar_t(2)))*pa_q1_0_1;
        const scalar_t element_out0_0 = output_t0 + output_t1;
        const scalar_t element_out0_1 = -output_t0;
        const scalar_t element_out0_2 = -output_t1;
        const scalar_t element_out1_0 = output_t2 + output_t3;
        const scalar_t element_out1_1 = -output_t2;
        const scalar_t element_out1_2 = -output_t3;
        #pragma omp atomic update
        outx[ev0 * out_stride] += element_out0_0;
        #pragma omp atomic update
        outx[ev1 * out_stride] += element_out0_1;
        #pragma omp atomic update
        outx[ev2 * out_stride] += element_out0_2;
        #pragma omp atomic update
        outy[ev0 * out_stride] += element_out1_0;
        #pragma omp atomic update
        outy[ev1 * out_stride] += element_out1_1;
        #pragma omp atomic update
        outy[ev2 * out_stride] += element_out1_2;
    }

    return SFEM_SUCCESS;
}

template <typename scalar_t, typename tangent_t, typename scale_t>
static SFEM_INLINE int linear_elasticity_tri3_inexact_apply_compressed_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        idx_t **const SFEM_RESTRICT elements,
        const ptrdiff_t tangent_element_stride,
        const ptrdiff_t tangent_component_stride,
        const tangent_t *const SFEM_RESTRICT tangent,
        const scale_t *const SFEM_RESTRICT scaling,
        const ptrdiff_t h_stride,
        const scalar_t *const SFEM_RESTRICT hx,
        const scalar_t *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT outx,
        scalar_t *const SFEM_RESTRICT outy
) {
    #pragma omp parallel for schedule(static)
    for (ptrdiff_t element = 0; element < nelements; ++element) {
        const idx_t ev0 = elements[0][element];
        const idx_t ev1 = elements[1][element];
        const idx_t ev2 = elements[2][element];
        const scalar_t hx_0 = hx[ev0 * h_stride];
        const scalar_t hx_1 = hx[ev1 * h_stride];
        const scalar_t hx_2 = hx[ev2 * h_stride];
        const scalar_t hy_0 = hy[ev0 * h_stride];
        const scalar_t hy_1 = hy[ev1 * h_stride];
        const scalar_t hy_2 = hy[ev2 * h_stride];
        const scalar_t scale = scalar_t(scaling[element]);
        const scalar_t tangent0 = scalar_t(tangent[element * tangent_element_stride + 0 * tangent_component_stride]);
        const scalar_t tangent1 = scalar_t(tangent[element * tangent_element_stride + 1 * tangent_component_stride]);
        const scalar_t tangent2 = scalar_t(tangent[element * tangent_element_stride + 2 * tangent_component_stride]);
        const scalar_t tangent3 = scalar_t(tangent[element * tangent_element_stride + 3 * tangent_component_stride]);
        const scalar_t tangent4 = scalar_t(tangent[element * tangent_element_stride + 4 * tangent_component_stride]);
        const scalar_t tangent5 = scalar_t(tangent[element * tangent_element_stride + 5 * tangent_component_stride]);
        const scalar_t tangent6 = scalar_t(tangent[element * tangent_element_stride + 6 * tangent_component_stride]);
        const scalar_t tangent7 = scalar_t(tangent[element * tangent_element_stride + 7 * tangent_component_stride]);
        const scalar_t tangent8 = scalar_t(tangent[element * tangent_element_stride + 8 * tangent_component_stride]);
        const scalar_t tangent9 = scalar_t(tangent[element * tangent_element_stride + 9 * tangent_component_stride]);
        const scalar_t compressed_increment_t0 = ((scalar_t(1) / scalar_t(2)))*hx_0;
        const scalar_t compressed_increment_t1 = ((scalar_t(1) / scalar_t(2)))*hy_0;
        const scalar_t pa_p0_0_0 = compressed_increment_t0 - (scalar_t(1) / scalar_t(2))*hx_1;
        const scalar_t pa_p0_0_1 = compressed_increment_t0 - (scalar_t(1) / scalar_t(2))*hx_2;
        const scalar_t pa_p1_0_0 = compressed_increment_t1 - (scalar_t(1) / scalar_t(2))*hy_1;
        const scalar_t pa_p1_0_1 = compressed_increment_t1 - (scalar_t(1) / scalar_t(2))*hy_2;
        const scalar_t pa_y0_0_0 = pa_p0_0_0*tangent0 + pa_p0_0_1*tangent1 + pa_p1_0_0*tangent2 + pa_p1_0_1*tangent3;
        const scalar_t pa_y0_0_1 = pa_p0_0_0*tangent1 + pa_p0_0_1*tangent4 + pa_p1_0_0*tangent5 + pa_p1_0_1*tangent6;
        const scalar_t pa_y1_0_0 = pa_p0_0_0*tangent2 + pa_p0_0_1*tangent5 + pa_p1_0_0*tangent7 + pa_p1_0_1*tangent8;
        const scalar_t pa_y1_0_1 = pa_p0_0_0*tangent3 + pa_p0_0_1*tangent6 + pa_p1_0_0*tangent8 + pa_p1_0_1*tangent9;
        const scalar_t pa_q0_0_0 = scalar_t(2)*pa_y0_0_0;
        const scalar_t pa_q0_0_1 = scalar_t(2)*pa_y0_0_1;
        const scalar_t pa_q1_0_0 = scalar_t(2)*pa_y1_0_0;
        const scalar_t pa_q1_0_1 = scalar_t(2)*pa_y1_0_1;
        const scalar_t output_t0 = ((scalar_t(1) / scalar_t(2)))*pa_q0_0_0;
        const scalar_t output_t1 = ((scalar_t(1) / scalar_t(2)))*pa_q0_0_1;
        const scalar_t output_t2 = ((scalar_t(1) / scalar_t(2)))*pa_q1_0_0;
        const scalar_t output_t3 = ((scalar_t(1) / scalar_t(2)))*pa_q1_0_1;
        const scalar_t element_out0_0 = output_t0 + output_t1;
        const scalar_t element_out0_1 = -output_t0;
        const scalar_t element_out0_2 = -output_t1;
        const scalar_t element_out1_0 = output_t2 + output_t3;
        const scalar_t element_out1_1 = -output_t2;
        const scalar_t element_out1_2 = -output_t3;
        #pragma omp atomic update
        outx[ev0 * out_stride] += scale * element_out0_0;
        #pragma omp atomic update
        outx[ev1 * out_stride] += scale * element_out0_1;
        #pragma omp atomic update
        outx[ev2 * out_stride] += scale * element_out0_2;
        #pragma omp atomic update
        outy[ev0 * out_stride] += scale * element_out1_0;
        #pragma omp atomic update
        outy[ev1 * out_stride] += scale * element_out1_1;
        #pragma omp atomic update
        outy[ev2 * out_stride] += scale * element_out1_2;
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem
