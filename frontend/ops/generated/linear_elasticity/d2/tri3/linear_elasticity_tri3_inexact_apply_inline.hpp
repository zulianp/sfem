#include "../../../kernel_math.hpp"

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t, typename tangent_t>
static SFEM_INLINE int linear_elasticity_tri3_inexact_apply_tangent_affine_mesh_soa_impl(
    const ptrdiff_t nelements,
    idx_t **const RSTR elements,
    const g_t *const RSTR g_adj0,
    const g_t *const RSTR g_adj1,
    const g_t *const RSTR g_adj2,
    const g_t *const RSTR g_adj3,
    const g_t *const RSTR g_det0,
    const s_t lmbda,
    const s_t mu,
    const ptrdiff_t u_stride,
    const s_t *const RSTR ux,
    const s_t *const RSTR uy,
    const ptrdiff_t tangent_element_stride,
    const ptrdiff_t tangent_component_stride,
    tangent_t *const RSTR tangent
) {
  #pragma omp parallel for schedule(static)
  for (ptrdiff_t element = 0; element < nelements; ++element) {
    const idx_t ev0 = elements[0][element];
    const idx_t ev1 = elements[1][element];
    const idx_t ev2 = elements[2][element];
    const s_t adjugate0 = s_t(g_adj0[element]);
    const s_t adjugate1 = s_t(g_adj1[element]);
    const s_t adjugate2 = s_t(g_adj2[element]);
    const s_t adjugate3 = s_t(g_adj3[element]);
    const s_t determinant = s_t(g_det0[element]);
    const s_t tangent_t0 = pow_m1(determinant);
    const s_t tangent_t1 = pow_2(adjugate1);
    const s_t tangent_t2 = pow_2(adjugate0);
    const s_t tangent_t3 = lmbda + s_t(2)*mu;
    const s_t tangent_t4 = adjugate1*mu;
    const s_t tangent_t5 = adjugate0*adjugate2;
    const s_t tangent_t6 = adjugate0*lmbda;
    const s_t tangent_t7 = pow_2(adjugate3);
    const s_t tangent_t8 = pow_2(adjugate2);
    const s_t tangent_t9 = adjugate3*mu;
    const s_t tangent_t10 = adjugate2*lmbda;
    const s_t tangent0 = tangent_t0*(mu*tangent_t1 + tangent_t2*tangent_t3);
    const s_t tangent1 = tangent_t0*(adjugate3*tangent_t4 + tangent_t3*tangent_t5);
    const s_t tangent2 = tangent_t0*(adjugate0*tangent_t4 + adjugate1*tangent_t6);
    const s_t tangent3 = tangent_t0*(adjugate2*tangent_t4 + adjugate3*tangent_t6);
    const s_t tangent4 = tangent_t0*(mu*tangent_t7 + tangent_t3*tangent_t8);
    const s_t tangent5 = tangent_t0*(adjugate0*tangent_t9 + adjugate1*tangent_t10);
    const s_t tangent6 = tangent_t0*(adjugate2*tangent_t9 + adjugate3*tangent_t10);
    const s_t tangent7 = tangent_t0*(mu*tangent_t2 + tangent_t1*tangent_t3);
    const s_t tangent8 = tangent_t0*(adjugate1*adjugate3*tangent_t3 + mu*tangent_t5);
    const s_t tangent9 = tangent_t0*(mu*tangent_t8 + tangent_t3*tangent_t7);
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

template <typename s_t, typename tangent_t>
static SFEM_INLINE int linear_elasticity_tri3_inexact_apply_stored_affine_mesh_soa_impl(
    const ptrdiff_t nelements,
    idx_t **const RSTR elements,
    const ptrdiff_t tangent_element_stride,
    const ptrdiff_t tangent_component_stride,
    const tangent_t *const RSTR tangent,
    const ptrdiff_t h_stride,
    const s_t *const RSTR hx,
    const s_t *const RSTR hy,
    const ptrdiff_t out_stride,
    s_t *const RSTR outx,
    s_t *const RSTR outy
) {
  #pragma omp parallel for schedule(static)
  for (ptrdiff_t element = 0; element < nelements; ++element) {
    const idx_t ev0 = elements[0][element];
    const idx_t ev1 = elements[1][element];
    const idx_t ev2 = elements[2][element];
    const s_t hx_0 = hx[ev0 * h_stride];
    const s_t hx_1 = hx[ev1 * h_stride];
    const s_t hx_2 = hx[ev2 * h_stride];
    const s_t hy_0 = hy[ev0 * h_stride];
    const s_t hy_1 = hy[ev1 * h_stride];
    const s_t hy_2 = hy[ev2 * h_stride];
    const s_t tangent0 = s_t(tangent[element * tangent_element_stride + 0 * tangent_component_stride]);
    const s_t tangent1 = s_t(tangent[element * tangent_element_stride + 1 * tangent_component_stride]);
    const s_t tangent2 = s_t(tangent[element * tangent_element_stride + 2 * tangent_component_stride]);
    const s_t tangent3 = s_t(tangent[element * tangent_element_stride + 3 * tangent_component_stride]);
    const s_t tangent4 = s_t(tangent[element * tangent_element_stride + 4 * tangent_component_stride]);
    const s_t tangent5 = s_t(tangent[element * tangent_element_stride + 5 * tangent_component_stride]);
    const s_t tangent6 = s_t(tangent[element * tangent_element_stride + 6 * tangent_component_stride]);
    const s_t tangent7 = s_t(tangent[element * tangent_element_stride + 7 * tangent_component_stride]);
    const s_t tangent8 = s_t(tangent[element * tangent_element_stride + 8 * tangent_component_stride]);
    const s_t tangent9 = s_t(tangent[element * tangent_element_stride + 9 * tangent_component_stride]);
    const s_t compressed_increment_t0 = ((s_t(1) / s_t(2)))*hx_0;
    const s_t compressed_increment_t1 = ((s_t(1) / s_t(2)))*hy_0;
    const s_t pa_p0_0_0 = compressed_increment_t0 - (s_t(1) / s_t(2))*hx_1;
    const s_t pa_p0_0_1 = compressed_increment_t0 - (s_t(1) / s_t(2))*hx_2;
    const s_t pa_p1_0_0 = compressed_increment_t1 - (s_t(1) / s_t(2))*hy_1;
    const s_t pa_p1_0_1 = compressed_increment_t1 - (s_t(1) / s_t(2))*hy_2;
    const s_t pa_y0_0_0 = pa_p0_0_0*tangent0 + pa_p0_0_1*tangent1 + pa_p1_0_0*tangent2 + pa_p1_0_1*tangent3;
    const s_t pa_y0_0_1 = pa_p0_0_0*tangent1 + pa_p0_0_1*tangent4 + pa_p1_0_0*tangent5 + pa_p1_0_1*tangent6;
    const s_t pa_y1_0_0 = pa_p0_0_0*tangent2 + pa_p0_0_1*tangent5 + pa_p1_0_0*tangent7 + pa_p1_0_1*tangent8;
    const s_t pa_y1_0_1 = pa_p0_0_0*tangent3 + pa_p0_0_1*tangent6 + pa_p1_0_0*tangent8 + pa_p1_0_1*tangent9;
    const s_t pa_q0_0_0 = s_t(2)*pa_y0_0_0;
    const s_t pa_q0_0_1 = s_t(2)*pa_y0_0_1;
    const s_t pa_q1_0_0 = s_t(2)*pa_y1_0_0;
    const s_t pa_q1_0_1 = s_t(2)*pa_y1_0_1;
    const s_t output_t0 = ((s_t(1) / s_t(2)))*pa_q0_0_0;
    const s_t output_t1 = ((s_t(1) / s_t(2)))*pa_q0_0_1;
    const s_t output_t2 = ((s_t(1) / s_t(2)))*pa_q1_0_0;
    const s_t output_t3 = ((s_t(1) / s_t(2)))*pa_q1_0_1;
    const s_t element_out0_0 = output_t0 + output_t1;
    const s_t element_out0_1 = -output_t0;
    const s_t element_out0_2 = -output_t1;
    const s_t element_out1_0 = output_t2 + output_t3;
    const s_t element_out1_1 = -output_t2;
    const s_t element_out1_2 = -output_t3;
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

template <typename s_t, typename tangent_t, typename scale_t>
static SFEM_INLINE int linear_elasticity_tri3_inexact_apply_compressed_affine_mesh_soa_impl(
    const ptrdiff_t nelements,
    idx_t **const RSTR elements,
    const ptrdiff_t tangent_element_stride,
    const ptrdiff_t tangent_component_stride,
    const tangent_t *const RSTR tangent,
    const scale_t *const RSTR scaling,
    const ptrdiff_t h_stride,
    const s_t *const RSTR hx,
    const s_t *const RSTR hy,
    const ptrdiff_t out_stride,
    s_t *const RSTR outx,
    s_t *const RSTR outy
) {
  #pragma omp parallel for schedule(static)
  for (ptrdiff_t element = 0; element < nelements; ++element) {
    const idx_t ev0 = elements[0][element];
    const idx_t ev1 = elements[1][element];
    const idx_t ev2 = elements[2][element];
    const s_t hx_0 = hx[ev0 * h_stride];
    const s_t hx_1 = hx[ev1 * h_stride];
    const s_t hx_2 = hx[ev2 * h_stride];
    const s_t hy_0 = hy[ev0 * h_stride];
    const s_t hy_1 = hy[ev1 * h_stride];
    const s_t hy_2 = hy[ev2 * h_stride];
    const s_t scale = s_t(scaling[element]);
    const s_t tangent0 = s_t(tangent[element * tangent_element_stride + 0 * tangent_component_stride]);
    const s_t tangent1 = s_t(tangent[element * tangent_element_stride + 1 * tangent_component_stride]);
    const s_t tangent2 = s_t(tangent[element * tangent_element_stride + 2 * tangent_component_stride]);
    const s_t tangent3 = s_t(tangent[element * tangent_element_stride + 3 * tangent_component_stride]);
    const s_t tangent4 = s_t(tangent[element * tangent_element_stride + 4 * tangent_component_stride]);
    const s_t tangent5 = s_t(tangent[element * tangent_element_stride + 5 * tangent_component_stride]);
    const s_t tangent6 = s_t(tangent[element * tangent_element_stride + 6 * tangent_component_stride]);
    const s_t tangent7 = s_t(tangent[element * tangent_element_stride + 7 * tangent_component_stride]);
    const s_t tangent8 = s_t(tangent[element * tangent_element_stride + 8 * tangent_component_stride]);
    const s_t tangent9 = s_t(tangent[element * tangent_element_stride + 9 * tangent_component_stride]);
    const s_t compressed_increment_t0 = ((s_t(1) / s_t(2)))*hx_0;
    const s_t compressed_increment_t1 = ((s_t(1) / s_t(2)))*hy_0;
    const s_t pa_p0_0_0 = compressed_increment_t0 - (s_t(1) / s_t(2))*hx_1;
    const s_t pa_p0_0_1 = compressed_increment_t0 - (s_t(1) / s_t(2))*hx_2;
    const s_t pa_p1_0_0 = compressed_increment_t1 - (s_t(1) / s_t(2))*hy_1;
    const s_t pa_p1_0_1 = compressed_increment_t1 - (s_t(1) / s_t(2))*hy_2;
    const s_t pa_y0_0_0 = pa_p0_0_0*tangent0 + pa_p0_0_1*tangent1 + pa_p1_0_0*tangent2 + pa_p1_0_1*tangent3;
    const s_t pa_y0_0_1 = pa_p0_0_0*tangent1 + pa_p0_0_1*tangent4 + pa_p1_0_0*tangent5 + pa_p1_0_1*tangent6;
    const s_t pa_y1_0_0 = pa_p0_0_0*tangent2 + pa_p0_0_1*tangent5 + pa_p1_0_0*tangent7 + pa_p1_0_1*tangent8;
    const s_t pa_y1_0_1 = pa_p0_0_0*tangent3 + pa_p0_0_1*tangent6 + pa_p1_0_0*tangent8 + pa_p1_0_1*tangent9;
    const s_t pa_q0_0_0 = s_t(2)*pa_y0_0_0;
    const s_t pa_q0_0_1 = s_t(2)*pa_y0_0_1;
    const s_t pa_q1_0_0 = s_t(2)*pa_y1_0_0;
    const s_t pa_q1_0_1 = s_t(2)*pa_y1_0_1;
    const s_t output_t0 = ((s_t(1) / s_t(2)))*pa_q0_0_0;
    const s_t output_t1 = ((s_t(1) / s_t(2)))*pa_q0_0_1;
    const s_t output_t2 = ((s_t(1) / s_t(2)))*pa_q1_0_0;
    const s_t output_t3 = ((s_t(1) / s_t(2)))*pa_q1_0_1;
    const s_t element_out0_0 = output_t0 + output_t1;
    const s_t element_out0_1 = -output_t0;
    const s_t element_out0_2 = -output_t1;
    const s_t element_out1_0 = output_t2 + output_t3;
    const s_t element_out1_1 = -output_t2;
    const s_t element_out1_2 = -output_t3;
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
