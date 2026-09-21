#pragma once
#include "../../../kernel_math.hpp"
#include "../../../reference/line_p1_q2.hpp"
#include "../../../reference/quad_line_q2.hpp"
#include "../../../tensor_product_kernels.hpp"

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t, typename tangent_t, int VS>
static SFEM_INLINE int linear_elasticity_proteus_quad4_inexact_apply_tangent_a_msoa_impl(
    const ptrdiff_t nelements,
    const g_t *const RSTR g_adj0,
    const g_t *const RSTR g_adj1,
    const g_t *const RSTR g_adj2,
    const g_t *const RSTR g_adj3,
    const g_t *const RSTR g_det0,
    const s_t lmbda,
    const s_t mu,
    const ptrdiff_t tangent_component_stride,
    tangent_t *const RSTR tangent
) {
  #pragma omp parallel for schedule(static)
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)((nelements - evb) < (ptrdiff_t)VS ? (nelements - evb) : (ptrdiff_t)VS);
    static constexpr int NQ = 4;
    static constexpr int NQ1 = 2;
    const s_t *const RSTR q_weight_1d = sfem::codegen::quad_line_q2<s_t>::q_weight_1d();
    static constexpr s_t QMEASURE = s_t(1);
    s_t btangent_acc[10][VS];
    const g_t *const RSTR bg_adj0 = g_adj0 + evb;
    const g_t *const RSTR bg_adj1 = g_adj1 + evb;
    const g_t *const RSTR bg_adj2 = g_adj2 + evb;
    const g_t *const RSTR bg_adj3 = g_adj3 + evb;
    const g_t *const RSTR bg_det0 = g_det0 + evb;
    tangent_t *const RSTR btangent0 = tangent + evb + 0 * tangent_component_stride;
    tangent_t *const RSTR btangent1 = tangent + evb + 1 * tangent_component_stride;
    tangent_t *const RSTR btangent2 = tangent + evb + 2 * tangent_component_stride;
    tangent_t *const RSTR btangent3 = tangent + evb + 3 * tangent_component_stride;
    tangent_t *const RSTR btangent4 = tangent + evb + 4 * tangent_component_stride;
    tangent_t *const RSTR btangent5 = tangent + evb + 5 * tangent_component_stride;
    tangent_t *const RSTR btangent6 = tangent + evb + 6 * tangent_component_stride;
    tangent_t *const RSTR btangent7 = tangent + evb + 7 * tangent_component_stride;
    tangent_t *const RSTR btangent8 = tangent + evb + 8 * tangent_component_stride;
    tangent_t *const RSTR btangent9 = tangent + evb + 9 * tangent_component_stride;
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
        btangent_acc[0][lane] = s_t(0);
        btangent_acc[1][lane] = s_t(0);
        btangent_acc[2][lane] = s_t(0);
        btangent_acc[3][lane] = s_t(0);
        btangent_acc[4][lane] = s_t(0);
        btangent_acc[5][lane] = s_t(0);
        btangent_acc[6][lane] = s_t(0);
        btangent_acc[7][lane] = s_t(0);
        btangent_acc[8][lane] = s_t(0);
        btangent_acc[9][lane] = s_t(0);
    }
    for (int q = 0; q < NQ; ++q) {
      const int qx = q % NQ1;
      const int qy = q / NQ1;
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
      const s_t adjugate0 = s_t(bg_adj0[lane]);
      const s_t adjugate1 = s_t(bg_adj1[lane]);
      const s_t adjugate2 = s_t(bg_adj2[lane]);
      const s_t adjugate3 = s_t(bg_adj3[lane]);
      const s_t determinant = s_t(bg_det0[lane]);
        const s_t qw = q_weight_1d[qx] * q_weight_1d[qy] * QMEASURE;
            const s_t integrand_t0 = pow_m1(determinant);
            const s_t integrand_t1 = pow_2(adjugate1);
            const s_t integrand_t2 = pow_2(adjugate0);
            const s_t integrand_t3 = lmbda + s_t(2)*mu;
            const s_t integrand_t4 = adjugate1*mu;
            const s_t integrand_t5 = adjugate0*adjugate2;
            const s_t integrand_t6 = adjugate0*lmbda;
            const s_t integrand_t7 = pow_2(adjugate3);
            const s_t integrand_t8 = pow_2(adjugate2);
            const s_t integrand_t9 = adjugate3*mu;
            const s_t integrand_t10 = adjugate2*lmbda;
            const s_t integrand0 = integrand_t0*(integrand_t1*mu + integrand_t2*integrand_t3);
            const s_t integrand1 = integrand_t0*(adjugate3*integrand_t4 + integrand_t3*integrand_t5);
            const s_t integrand2 = integrand_t0*(adjugate0*integrand_t4 + adjugate1*integrand_t6);
            const s_t integrand3 = integrand_t0*(adjugate2*integrand_t4 + adjugate3*integrand_t6);
            const s_t integrand4 = integrand_t0*(integrand_t3*integrand_t8 + integrand_t7*mu);
            const s_t integrand5 = integrand_t0*(adjugate0*integrand_t9 + adjugate1*integrand_t10);
            const s_t integrand6 = integrand_t0*(adjugate2*integrand_t9 + adjugate3*integrand_t10);
            const s_t integrand7 = integrand_t0*(integrand_t1*integrand_t3 + integrand_t2*mu);
            const s_t integrand8 = integrand_t0*(adjugate1*adjugate3*integrand_t3 + integrand_t5*mu);
            const s_t integrand9 = integrand_t0*(integrand_t3*integrand_t7 + integrand_t8*mu);
        btangent_acc[0][lane] += qw * integrand0;
        btangent_acc[1][lane] += qw * integrand1;
        btangent_acc[2][lane] += qw * integrand2;
        btangent_acc[3][lane] += qw * integrand3;
        btangent_acc[4][lane] += qw * integrand4;
        btangent_acc[5][lane] += qw * integrand5;
        btangent_acc[6][lane] += qw * integrand6;
        btangent_acc[7][lane] += qw * integrand7;
        btangent_acc[8][lane] += qw * integrand8;
        btangent_acc[9][lane] += qw * integrand9;
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
        btangent0[lane] = tangent_t(btangent_acc[0][lane]);
        btangent1[lane] = tangent_t(btangent_acc[1][lane]);
        btangent2[lane] = tangent_t(btangent_acc[2][lane]);
        btangent3[lane] = tangent_t(btangent_acc[3][lane]);
        btangent4[lane] = tangent_t(btangent_acc[4][lane]);
        btangent5[lane] = tangent_t(btangent_acc[5][lane]);
        btangent6[lane] = tangent_t(btangent_acc[6][lane]);
        btangent7[lane] = tangent_t(btangent_acc[7][lane]);
        btangent8[lane] = tangent_t(btangent_acc[8][lane]);
        btangent9[lane] = tangent_t(btangent_acc[9][lane]);
    }
  }

  return SFEM_SUCCESS;
}

template <typename s_t, typename tangent_t, int VS>
static SFEM_INLINE int linear_elasticity_proteus_quad4_inexact_apply_stored_a_msoa_impl(
    const ptrdiff_t nelements,
    idx_t **const RSTR elements,
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
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)((nelements - evb) < (ptrdiff_t)VS ? (nelements - evb) : (ptrdiff_t)VS);
    idx_t bev0[VS];
    idx_t bev1[VS];
    idx_t bev2[VS];
    idx_t bev3[VS];
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      bev0[lane] = elements[0][evb + lane];
      bev1[lane] = elements[1][evb + lane];
      bev2[lane] = elements[2][evb + lane];
      bev3[lane] = elements[3][evb + lane];
    }
    s_t bhx_0[VS];
    s_t bhx_1[VS];
    s_t bhx_2[VS];
    s_t bhx_3[VS];
    s_t bhy_0[VS];
    s_t bhy_1[VS];
    s_t bhy_2[VS];
    s_t bhy_3[VS];
    s_t bout0_0[VS];
    s_t bout0_1[VS];
    s_t bout0_2[VS];
    s_t bout0_3[VS];
    s_t bout1_0[VS];
    s_t bout1_1[VS];
    s_t bout1_2[VS];
    s_t bout1_3[VS];
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      bhx_0[lane] = hx[bev0[lane] * h_stride];
      bhx_1[lane] = hx[bev1[lane] * h_stride];
      bhx_2[lane] = hx[bev2[lane] * h_stride];
      bhx_3[lane] = hx[bev3[lane] * h_stride];
      bhy_0[lane] = hy[bev0[lane] * h_stride];
      bhy_1[lane] = hy[bev1[lane] * h_stride];
      bhy_2[lane] = hy[bev2[lane] * h_stride];
      bhy_3[lane] = hy[bev3[lane] * h_stride];
    }
    const tangent_t *const RSTR btangent0 = tangent + evb + 0 * tangent_component_stride;
    const tangent_t *const RSTR btangent1 = tangent + evb + 1 * tangent_component_stride;
    const tangent_t *const RSTR btangent2 = tangent + evb + 2 * tangent_component_stride;
    const tangent_t *const RSTR btangent3 = tangent + evb + 3 * tangent_component_stride;
    const tangent_t *const RSTR btangent4 = tangent + evb + 4 * tangent_component_stride;
    const tangent_t *const RSTR btangent5 = tangent + evb + 5 * tangent_component_stride;
    const tangent_t *const RSTR btangent6 = tangent + evb + 6 * tangent_component_stride;
    const tangent_t *const RSTR btangent7 = tangent + evb + 7 * tangent_component_stride;
    const tangent_t *const RSTR btangent8 = tangent + evb + 8 * tangent_component_stride;
    const tangent_t *const RSTR btangent9 = tangent + evb + 9 * tangent_component_stride;
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      const s_t hx_0 = bhx_0[lane];
      const s_t hx_1 = bhx_1[lane];
      const s_t hx_2 = bhx_2[lane];
      const s_t hx_3 = bhx_3[lane];
      const s_t hy_0 = bhy_0[lane];
      const s_t hy_1 = bhy_1[lane];
      const s_t hy_2 = bhy_2[lane];
      const s_t hy_3 = bhy_3[lane];
      const s_t tangent0 = s_t(btangent0[lane]);
      const s_t tangent1 = s_t(btangent1[lane]);
      const s_t tangent2 = s_t(btangent2[lane]);
      const s_t tangent3 = s_t(btangent3[lane]);
      const s_t tangent4 = s_t(btangent4[lane]);
      const s_t tangent5 = s_t(btangent5[lane]);
      const s_t tangent6 = s_t(btangent6[lane]);
      const s_t tangent7 = s_t(btangent7[lane]);
      const s_t tangent8 = s_t(btangent8[lane]);
      const s_t tangent9 = s_t(btangent9[lane]);
      const s_t reference_product_t0 = ((s_t(1) / s_t(3)))*hx_1;
      const s_t reference_product_t1 = ((s_t(1) / s_t(6)))*hx_2;
      const s_t reference_product_t2 = ((s_t(1) / s_t(3)))*hx_0 - (s_t(1) / s_t(6))*hx_3;
      const s_t reference_product_t3 = -reference_product_t0 + reference_product_t1 + reference_product_t2;
      const s_t reference_product_t4 = ((s_t(1) / s_t(4)))*hx_1;
      const s_t reference_product_t5 = ((s_t(1) / s_t(4)))*hx_2;
      const s_t reference_product_t6 = ((s_t(1) / s_t(4)))*hx_0 - (s_t(1) / s_t(4))*hx_3;
      const s_t reference_product_t7 = -reference_product_t4 + reference_product_t5 + reference_product_t6;
      const s_t reference_product_t8 = ((s_t(1) / s_t(6)))*hx_1;
      const s_t reference_product_t9 = ((s_t(1) / s_t(3)))*hx_2;
      const s_t reference_product_t10 = ((s_t(1) / s_t(6)))*hx_0 - (s_t(1) / s_t(3))*hx_3;
      const s_t reference_product_t11 = reference_product_t10 - reference_product_t8 + reference_product_t9;
      const s_t reference_product_t12 = -reference_product_t7;
      const s_t reference_product_t13 = reference_product_t4 - reference_product_t5 + reference_product_t6;
      const s_t reference_product_t14 = reference_product_t2 + reference_product_t8 - reference_product_t9;
      const s_t reference_product_t15 = -reference_product_t13;
      const s_t reference_product_t16 = reference_product_t0 - reference_product_t1 + reference_product_t10;
      const s_t reference_product_t17 = ((s_t(1) / s_t(3)))*hy_1;
      const s_t reference_product_t18 = ((s_t(1) / s_t(6)))*hy_2;
      const s_t reference_product_t19 = ((s_t(1) / s_t(3)))*hy_0 - (s_t(1) / s_t(6))*hy_3;
      const s_t reference_product_t20 = -reference_product_t17 + reference_product_t18 + reference_product_t19;
      const s_t reference_product_t21 = ((s_t(1) / s_t(4)))*hy_1;
      const s_t reference_product_t22 = ((s_t(1) / s_t(4)))*hy_2;
      const s_t reference_product_t23 = ((s_t(1) / s_t(4)))*hy_0 - (s_t(1) / s_t(4))*hy_3;
      const s_t reference_product_t24 = -reference_product_t21 + reference_product_t22 + reference_product_t23;
      const s_t reference_product_t25 = ((s_t(1) / s_t(6)))*hy_1;
      const s_t reference_product_t26 = ((s_t(1) / s_t(3)))*hy_2;
      const s_t reference_product_t27 = ((s_t(1) / s_t(6)))*hy_0 - (s_t(1) / s_t(3))*hy_3;
      const s_t reference_product_t28 = -reference_product_t25 + reference_product_t26 + reference_product_t27;
      const s_t reference_product_t29 = -reference_product_t24;
      const s_t reference_product_t30 = reference_product_t21 - reference_product_t22 + reference_product_t23;
      const s_t reference_product_t31 = reference_product_t19 + reference_product_t25 - reference_product_t26;
      const s_t reference_product_t32 = -reference_product_t30;
      const s_t reference_product_t33 = reference_product_t17 - reference_product_t18 + reference_product_t27;
      const s_t pa_g0_0_1_0 = -reference_product_t3;
      const s_t pa_g0_0_3_0 = -reference_product_t11;
      const s_t pa_g0_1_2_1 = -reference_product_t14;
      const s_t pa_g0_1_3_1 = -reference_product_t16;
      const s_t pa_g1_0_1_0 = -reference_product_t20;
      const s_t pa_g1_0_3_0 = -reference_product_t28;
      const s_t pa_g1_1_2_1 = -reference_product_t31;
      const s_t pa_g1_1_3_1 = -reference_product_t33;
      const s_t output_t0 = reference_product_t24*tangent5 + reference_product_t7*tangent1;
      const s_t output_t1 = reference_product_t13*tangent1 + reference_product_t30*tangent3;
      const s_t output_t2 = reference_product_t15*tangent1 + reference_product_t32*tangent3;
      const s_t output_t3 = reference_product_t12*tangent1 + reference_product_t29*tangent5;
      const s_t output_t4 = reference_product_t24*tangent8 + reference_product_t7*tangent3;
      const s_t output_t5 = reference_product_t13*tangent5 + reference_product_t30*tangent8;
      const s_t output_t6 = reference_product_t15*tangent5 + reference_product_t32*tangent8;
      const s_t output_t7 = reference_product_t12*tangent3 + reference_product_t29*tangent8;
      const s_t element_out0_0 = output_t0 + output_t1 + reference_product_t14*tangent4 + reference_product_t20*tangent2 + reference_product_t3*tangent0 + reference_product_t31*tangent6;
      const s_t element_out0_1 = output_t0 + output_t2 + pa_g0_0_1_0*tangent0 + pa_g1_0_1_0*tangent2 + reference_product_t16*tangent4 + reference_product_t33*tangent6;
      const s_t element_out0_2 = output_t1 + output_t3 + pa_g0_1_2_1*tangent4 + pa_g1_1_2_1*tangent6 + reference_product_t11*tangent0 + reference_product_t28*tangent2;
      const s_t element_out0_3 = output_t2 + output_t3 + pa_g0_0_3_0*tangent0 + pa_g0_1_3_1*tangent4 + pa_g1_0_3_0*tangent2 + pa_g1_1_3_1*tangent6;
      const s_t element_out1_0 = output_t4 + output_t5 + reference_product_t14*tangent6 + reference_product_t20*tangent7 + reference_product_t3*tangent2 + reference_product_t31*tangent9;
      const s_t element_out1_1 = output_t4 + output_t6 + pa_g0_0_1_0*tangent2 + pa_g1_0_1_0*tangent7 + reference_product_t16*tangent6 + reference_product_t33*tangent9;
      const s_t element_out1_2 = output_t5 + output_t7 + pa_g0_1_2_1*tangent6 + pa_g1_1_2_1*tangent9 + reference_product_t11*tangent2 + reference_product_t28*tangent7;
      const s_t element_out1_3 = output_t6 + output_t7 + pa_g0_0_3_0*tangent2 + pa_g0_1_3_1*tangent6 + pa_g1_0_3_0*tangent7 + pa_g1_1_3_1*tangent9;
      bout0_0[lane] = element_out0_0;
      bout0_1[lane] = element_out0_1;
      bout0_2[lane] = element_out0_2;
      bout0_3[lane] = element_out0_3;
      bout1_0[lane] = element_out1_0;
      bout1_1[lane] = element_out1_1;
      bout1_2[lane] = element_out1_2;
      bout1_3[lane] = element_out1_3;
    }
    for (int lane = 0; lane < ne; ++lane) {
      #pragma omp atomic update
      outx[bev0[lane] * out_stride] += bout0_0[lane];
    }
    for (int lane = 0; lane < ne; ++lane) {
      #pragma omp atomic update
      outx[bev1[lane] * out_stride] += bout0_1[lane];
    }
    for (int lane = 0; lane < ne; ++lane) {
      #pragma omp atomic update
      outx[bev2[lane] * out_stride] += bout0_2[lane];
    }
    for (int lane = 0; lane < ne; ++lane) {
      #pragma omp atomic update
      outx[bev3[lane] * out_stride] += bout0_3[lane];
    }
    for (int lane = 0; lane < ne; ++lane) {
      #pragma omp atomic update
      outy[bev0[lane] * out_stride] += bout1_0[lane];
    }
    for (int lane = 0; lane < ne; ++lane) {
      #pragma omp atomic update
      outy[bev1[lane] * out_stride] += bout1_1[lane];
    }
    for (int lane = 0; lane < ne; ++lane) {
      #pragma omp atomic update
      outy[bev2[lane] * out_stride] += bout1_2[lane];
    }
    for (int lane = 0; lane < ne; ++lane) {
      #pragma omp atomic update
      outy[bev3[lane] * out_stride] += bout1_3[lane];
    }
  }

  return SFEM_SUCCESS;
}

template <typename s_t, typename tangent_t, typename scale_t>
static SFEM_INLINE int linear_elasticity_proteus_quad4_inexact_apply_compressed_a_msoa_impl(
    const ptrdiff_t nelements,
    idx_t **const RSTR elements,
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
    const idx_t ev3 = elements[3][element];
    const s_t hx_0 = hx[ev0 * h_stride];
    const s_t hx_1 = hx[ev1 * h_stride];
    const s_t hx_2 = hx[ev2 * h_stride];
    const s_t hx_3 = hx[ev3 * h_stride];
    const s_t hy_0 = hy[ev0 * h_stride];
    const s_t hy_1 = hy[ev1 * h_stride];
    const s_t hy_2 = hy[ev2 * h_stride];
    const s_t hy_3 = hy[ev3 * h_stride];
    const s_t scale = s_t(scaling[element]);
    const s_t tangent0 = s_t(tangent[element + 0 * tangent_component_stride]);
    const s_t tangent1 = s_t(tangent[element + 1 * tangent_component_stride]);
    const s_t tangent2 = s_t(tangent[element + 2 * tangent_component_stride]);
    const s_t tangent3 = s_t(tangent[element + 3 * tangent_component_stride]);
    const s_t tangent4 = s_t(tangent[element + 4 * tangent_component_stride]);
    const s_t tangent5 = s_t(tangent[element + 5 * tangent_component_stride]);
    const s_t tangent6 = s_t(tangent[element + 6 * tangent_component_stride]);
    const s_t tangent7 = s_t(tangent[element + 7 * tangent_component_stride]);
    const s_t tangent8 = s_t(tangent[element + 8 * tangent_component_stride]);
    const s_t tangent9 = s_t(tangent[element + 9 * tangent_component_stride]);
    const s_t reference_product_t0 = ((s_t(1) / s_t(3)))*hx_1;
    const s_t reference_product_t1 = ((s_t(1) / s_t(6)))*hx_2;
    const s_t reference_product_t2 = ((s_t(1) / s_t(3)))*hx_0 - (s_t(1) / s_t(6))*hx_3;
    const s_t reference_product_t3 = -reference_product_t0 + reference_product_t1 + reference_product_t2;
    const s_t reference_product_t4 = ((s_t(1) / s_t(4)))*hx_1;
    const s_t reference_product_t5 = ((s_t(1) / s_t(4)))*hx_2;
    const s_t reference_product_t6 = ((s_t(1) / s_t(4)))*hx_0 - (s_t(1) / s_t(4))*hx_3;
    const s_t reference_product_t7 = -reference_product_t4 + reference_product_t5 + reference_product_t6;
    const s_t reference_product_t8 = ((s_t(1) / s_t(6)))*hx_1;
    const s_t reference_product_t9 = ((s_t(1) / s_t(3)))*hx_2;
    const s_t reference_product_t10 = ((s_t(1) / s_t(6)))*hx_0 - (s_t(1) / s_t(3))*hx_3;
    const s_t reference_product_t11 = reference_product_t10 - reference_product_t8 + reference_product_t9;
    const s_t reference_product_t12 = -reference_product_t7;
    const s_t reference_product_t13 = reference_product_t4 - reference_product_t5 + reference_product_t6;
    const s_t reference_product_t14 = reference_product_t2 + reference_product_t8 - reference_product_t9;
    const s_t reference_product_t15 = -reference_product_t13;
    const s_t reference_product_t16 = reference_product_t0 - reference_product_t1 + reference_product_t10;
    const s_t reference_product_t17 = ((s_t(1) / s_t(3)))*hy_1;
    const s_t reference_product_t18 = ((s_t(1) / s_t(6)))*hy_2;
    const s_t reference_product_t19 = ((s_t(1) / s_t(3)))*hy_0 - (s_t(1) / s_t(6))*hy_3;
    const s_t reference_product_t20 = -reference_product_t17 + reference_product_t18 + reference_product_t19;
    const s_t reference_product_t21 = ((s_t(1) / s_t(4)))*hy_1;
    const s_t reference_product_t22 = ((s_t(1) / s_t(4)))*hy_2;
    const s_t reference_product_t23 = ((s_t(1) / s_t(4)))*hy_0 - (s_t(1) / s_t(4))*hy_3;
    const s_t reference_product_t24 = -reference_product_t21 + reference_product_t22 + reference_product_t23;
    const s_t reference_product_t25 = ((s_t(1) / s_t(6)))*hy_1;
    const s_t reference_product_t26 = ((s_t(1) / s_t(3)))*hy_2;
    const s_t reference_product_t27 = ((s_t(1) / s_t(6)))*hy_0 - (s_t(1) / s_t(3))*hy_3;
    const s_t reference_product_t28 = -reference_product_t25 + reference_product_t26 + reference_product_t27;
    const s_t reference_product_t29 = -reference_product_t24;
    const s_t reference_product_t30 = reference_product_t21 - reference_product_t22 + reference_product_t23;
    const s_t reference_product_t31 = reference_product_t19 + reference_product_t25 - reference_product_t26;
    const s_t reference_product_t32 = -reference_product_t30;
    const s_t reference_product_t33 = reference_product_t17 - reference_product_t18 + reference_product_t27;
    const s_t pa_g0_0_1_0 = -reference_product_t3;
    const s_t pa_g0_0_3_0 = -reference_product_t11;
    const s_t pa_g0_1_2_1 = -reference_product_t14;
    const s_t pa_g0_1_3_1 = -reference_product_t16;
    const s_t pa_g1_0_1_0 = -reference_product_t20;
    const s_t pa_g1_0_3_0 = -reference_product_t28;
    const s_t pa_g1_1_2_1 = -reference_product_t31;
    const s_t pa_g1_1_3_1 = -reference_product_t33;
    const s_t output_t0 = reference_product_t24*tangent5 + reference_product_t7*tangent1;
    const s_t output_t1 = reference_product_t13*tangent1 + reference_product_t30*tangent3;
    const s_t output_t2 = reference_product_t15*tangent1 + reference_product_t32*tangent3;
    const s_t output_t3 = reference_product_t12*tangent1 + reference_product_t29*tangent5;
    const s_t output_t4 = reference_product_t24*tangent8 + reference_product_t7*tangent3;
    const s_t output_t5 = reference_product_t13*tangent5 + reference_product_t30*tangent8;
    const s_t output_t6 = reference_product_t15*tangent5 + reference_product_t32*tangent8;
    const s_t output_t7 = reference_product_t12*tangent3 + reference_product_t29*tangent8;
    const s_t element_out0_0 = output_t0 + output_t1 + reference_product_t14*tangent4 + reference_product_t20*tangent2 + reference_product_t3*tangent0 + reference_product_t31*tangent6;
    const s_t element_out0_1 = output_t0 + output_t2 + pa_g0_0_1_0*tangent0 + pa_g1_0_1_0*tangent2 + reference_product_t16*tangent4 + reference_product_t33*tangent6;
    const s_t element_out0_2 = output_t1 + output_t3 + pa_g0_1_2_1*tangent4 + pa_g1_1_2_1*tangent6 + reference_product_t11*tangent0 + reference_product_t28*tangent2;
    const s_t element_out0_3 = output_t2 + output_t3 + pa_g0_0_3_0*tangent0 + pa_g0_1_3_1*tangent4 + pa_g1_0_3_0*tangent2 + pa_g1_1_3_1*tangent6;
    const s_t element_out1_0 = output_t4 + output_t5 + reference_product_t14*tangent6 + reference_product_t20*tangent7 + reference_product_t3*tangent2 + reference_product_t31*tangent9;
    const s_t element_out1_1 = output_t4 + output_t6 + pa_g0_0_1_0*tangent2 + pa_g1_0_1_0*tangent7 + reference_product_t16*tangent6 + reference_product_t33*tangent9;
    const s_t element_out1_2 = output_t5 + output_t7 + pa_g0_1_2_1*tangent6 + pa_g1_1_2_1*tangent9 + reference_product_t11*tangent2 + reference_product_t28*tangent7;
    const s_t element_out1_3 = output_t6 + output_t7 + pa_g0_0_3_0*tangent2 + pa_g0_1_3_1*tangent6 + pa_g1_0_3_0*tangent7 + pa_g1_1_3_1*tangent9;
    #pragma omp atomic update
    outx[ev0 * out_stride] += scale * element_out0_0;
    #pragma omp atomic update
    outx[ev1 * out_stride] += scale * element_out0_1;
    #pragma omp atomic update
    outx[ev2 * out_stride] += scale * element_out0_2;
    #pragma omp atomic update
    outx[ev3 * out_stride] += scale * element_out0_3;
    #pragma omp atomic update
    outy[ev0 * out_stride] += scale * element_out1_0;
    #pragma omp atomic update
    outy[ev1 * out_stride] += scale * element_out1_1;
    #pragma omp atomic update
    outy[ev2 * out_stride] += scale * element_out1_2;
    #pragma omp atomic update
    outy[ev3 * out_stride] += scale * element_out1_3;
  }

  return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem
