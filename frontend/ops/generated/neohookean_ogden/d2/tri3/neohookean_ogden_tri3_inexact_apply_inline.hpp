#pragma once
#include "../../../kernel_math.hpp"

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t, typename tangent_t, int VS>
static SFEM_INLINE int neohookean_ogden_tri3_inexact_apply_tangent_a_msoa_impl(
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
    const ptrdiff_t tangent_component_stride,
    tangent_t *const RSTR tangent
) {
  #pragma omp parallel for schedule(static)
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)((nelements - evb) < (ptrdiff_t)VS ? (nelements - evb) : (ptrdiff_t)VS);
    idx_t bev0[VS];
    idx_t bev1[VS];
    idx_t bev2[VS];
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      bev0[lane] = elements[0][evb + lane];
      bev1[lane] = elements[1][evb + lane];
      bev2[lane] = elements[2][evb + lane];
    }
    s_t bux_0[VS];
    s_t bux_1[VS];
    s_t bux_2[VS];
    s_t buy_0[VS];
    s_t buy_1[VS];
    s_t buy_2[VS];
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      bux_0[lane] = ux[bev0[lane] * u_stride];
      bux_1[lane] = ux[bev1[lane] * u_stride];
      bux_2[lane] = ux[bev2[lane] * u_stride];
      buy_0[lane] = uy[bev0[lane] * u_stride];
      buy_1[lane] = uy[bev1[lane] * u_stride];
      buy_2[lane] = uy[bev2[lane] * u_stride];
    }
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
      const s_t adjugate0 = s_t(bg_adj0[lane]);
      const s_t adjugate1 = s_t(bg_adj1[lane]);
      const s_t adjugate2 = s_t(bg_adj2[lane]);
      const s_t adjugate3 = s_t(bg_adj3[lane]);
      const s_t determinant = s_t(bg_det0[lane]);
      const s_t ux_0 = bux_0[lane];
      const s_t ux_1 = bux_1[lane];
      const s_t ux_2 = bux_2[lane];
      const s_t uy_0 = buy_0[lane];
      const s_t uy_1 = buy_1[lane];
      const s_t uy_2 = buy_2[lane];
      const s_t gu_ref_0_0 = -ux_0 + ux_1;
      const s_t gu_ref_0_1 = -ux_0 + ux_2;
      const s_t gu_ref_1_0 = -uy_0 + uy_1;
      const s_t gu_ref_1_1 = -uy_0 + uy_2;
      const s_t gradient1_t0 = pow_m1(determinant);
      const s_t gradient1_t1 = gradient1_t0*gu_ref_0_0;
      const s_t gradient1_t2 = gradient1_t0*gu_ref_0_1;
      const s_t gradient1_t3 = gradient1_t0*gu_ref_1_0;
      const s_t gradient1_t4 = gradient1_t0*gu_ref_1_1;
      const s_t gu_0_0 = adjugate0*gradient1_t1 + adjugate2*gradient1_t2;
      const s_t gu_0_1 = adjugate1*gradient1_t1 + adjugate3*gradient1_t2;
      const s_t gu_1_0 = adjugate0*gradient1_t3 + adjugate2*gradient1_t4;
      const s_t gu_1_1 = adjugate1*gradient1_t3 + adjugate3*gradient1_t4;
      const s_t tangent_t0 = pow_m1(determinant);
      const s_t tangent_t1 = pow_2(adjugate1);
      const s_t tangent_t2 = gu_0_1*gu_1_0;
      const s_t tangent_t3 = gu_0_0 + s_t(1);
      const s_t tangent_t4 = gu_1_1 + s_t(1);
      const s_t tangent_t5 = -tangent_t2 + tangent_t3*tangent_t4;
      const s_t tangent_t6 = pow_m2(tangent_t5);
      const s_t tangent_t7 = pow_2(gu_1_0)*tangent_t6;
      const s_t tangent_t8 = lmbda*tangent_t7;
      const s_t tangent_t9 = log(tangent_t5);
      const s_t tangent_t10 = mu*tangent_t7 + mu - tangent_t8*tangent_t9 + tangent_t8;
      const s_t tangent_t11 = adjugate0*adjugate1;
      const s_t tangent_t12 = tangent_t4*tangent_t6;
      const s_t tangent_t13 = gu_1_0*tangent_t12;
      const s_t tangent_t14 = gu_1_0*lmbda*tangent_t4*tangent_t6*tangent_t9 - lmbda*tangent_t13 - mu*tangent_t13;
      const s_t tangent_t15 = s_t(2)*tangent_t14;
      const s_t tangent_t16 = pow_2(adjugate0);
      const s_t tangent_t17 = pow_2(tangent_t4)*tangent_t6;
      const s_t tangent_t18 = lmbda*tangent_t17;
      const s_t tangent_t19 = mu*tangent_t17 + mu - tangent_t18*tangent_t9 + tangent_t18;
      const s_t tangent_t20 = adjugate1*adjugate3;
      const s_t tangent_t21 = adjugate0*adjugate3;
      const s_t tangent_t22 = adjugate1*adjugate2;
      const s_t tangent_t23 = adjugate0*adjugate2;
      const s_t tangent_t24 = gu_0_1*tangent_t12;
      const s_t tangent_t25 = gu_0_1*lmbda*tangent_t4*tangent_t6*tangent_t9 - lmbda*tangent_t24 - mu*tangent_t24;
      const s_t tangent_t26 = lmbda*tangent_t6;
      const s_t tangent_t27 = gu_1_0*tangent_t3;
      const s_t tangent_t28 = mu*tangent_t6;
      const s_t tangent_t29 = gu_1_0*lmbda*tangent_t3*tangent_t6*tangent_t9 - tangent_t26*tangent_t27 - tangent_t27*tangent_t28;
      const s_t tangent_t30 = pow_m1(tangent_t5);
      const s_t tangent_t31 = mu*tangent_t30;
      const s_t tangent_t32 = lmbda*tangent_t9;
      const s_t tangent_t33 = tangent_t30*tangent_t32;
      const s_t tangent_t34 = tangent_t32*tangent_t6;
      const s_t tangent_t35 = tangent_t2*tangent_t26 + tangent_t2*tangent_t28 - tangent_t2*tangent_t34 + tangent_t31 - tangent_t33;
      const s_t tangent_t36 = tangent_t3*tangent_t4;
      const s_t tangent_t37 = tangent_t26*tangent_t36 + tangent_t28*tangent_t36 - tangent_t31 + tangent_t33 - tangent_t34*tangent_t36;
      const s_t tangent_t38 = tangent_t20*tangent_t29 + tangent_t23*tangent_t25;
      const s_t tangent_t39 = pow_2(adjugate3);
      const s_t tangent_t40 = adjugate2*adjugate3;
      const s_t tangent_t41 = pow_2(adjugate2);
      const s_t tangent_t42 = pow_2(gu_0_1)*tangent_t6;
      const s_t tangent_t43 = lmbda*tangent_t42 + mu*tangent_t42 + mu - tangent_t32*tangent_t42;
      const s_t tangent_t44 = gu_0_1*tangent_t3;
      const s_t tangent_t45 = gu_0_1*lmbda*tangent_t3*tangent_t6*tangent_t9 - tangent_t26*tangent_t44 - tangent_t28*tangent_t44;
      const s_t tangent_t46 = s_t(2)*tangent_t45;
      const s_t tangent_t47 = pow_2(tangent_t3)*tangent_t6;
      const s_t tangent_t48 = lmbda*tangent_t47 + mu*tangent_t47 + mu - tangent_t32*tangent_t47;
      const s_t tangent0 = tangent_t0*(tangent_t1*tangent_t10 + tangent_t11*tangent_t15 + tangent_t16*tangent_t19);
      const s_t tangent1 = tangent_t0*(tangent_t10*tangent_t20 + tangent_t14*tangent_t21 + tangent_t14*tangent_t22 + tangent_t19*tangent_t23);
      const s_t tangent2 = tangent_t0*(tangent_t1*tangent_t29 + tangent_t11*tangent_t35 + tangent_t11*tangent_t37 + tangent_t16*tangent_t25);
      const s_t tangent3 = tangent_t0*(tangent_t21*tangent_t37 + tangent_t22*tangent_t35 + tangent_t38);
      const s_t tangent4 = tangent_t0*(tangent_t10*tangent_t39 + tangent_t15*tangent_t40 + tangent_t19*tangent_t41);
      const s_t tangent5 = tangent_t0*(tangent_t21*tangent_t35 + tangent_t22*tangent_t37 + tangent_t38);
      const s_t tangent6 = tangent_t0*(tangent_t25*tangent_t41 + tangent_t29*tangent_t39 + tangent_t35*tangent_t40 + tangent_t37*tangent_t40);
      const s_t tangent7 = tangent_t0*(tangent_t1*tangent_t48 + tangent_t11*tangent_t46 + tangent_t16*tangent_t43);
      const s_t tangent8 = tangent_t0*(tangent_t20*tangent_t48 + tangent_t21*tangent_t45 + tangent_t22*tangent_t45 + tangent_t23*tangent_t43);
      const s_t tangent9 = tangent_t0*(tangent_t39*tangent_t48 + tangent_t40*tangent_t46 + tangent_t41*tangent_t43);
      btangent0[lane] = tangent_t(tangent0);
      btangent1[lane] = tangent_t(tangent1);
      btangent2[lane] = tangent_t(tangent2);
      btangent3[lane] = tangent_t(tangent3);
      btangent4[lane] = tangent_t(tangent4);
      btangent5[lane] = tangent_t(tangent5);
      btangent6[lane] = tangent_t(tangent6);
      btangent7[lane] = tangent_t(tangent7);
      btangent8[lane] = tangent_t(tangent8);
      btangent9[lane] = tangent_t(tangent9);
    }
  }

  return SFEM_SUCCESS;
}

template <typename s_t, typename tangent_t, int VS>
static SFEM_INLINE int neohookean_ogden_tri3_inexact_apply_stored_a_msoa_impl(
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
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      bev0[lane] = elements[0][evb + lane];
      bev1[lane] = elements[1][evb + lane];
      bev2[lane] = elements[2][evb + lane];
    }
    s_t bhx_0[VS];
    s_t bhx_1[VS];
    s_t bhx_2[VS];
    s_t bhy_0[VS];
    s_t bhy_1[VS];
    s_t bhy_2[VS];
    s_t bout0_0[VS];
    s_t bout0_1[VS];
    s_t bout0_2[VS];
    s_t bout1_0[VS];
    s_t bout1_1[VS];
    s_t bout1_2[VS];
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      bhx_0[lane] = hx[bev0[lane] * h_stride];
      bhx_1[lane] = hx[bev1[lane] * h_stride];
      bhx_2[lane] = hx[bev2[lane] * h_stride];
      bhy_0[lane] = hy[bev0[lane] * h_stride];
      bhy_1[lane] = hy[bev1[lane] * h_stride];
      bhy_2[lane] = hy[bev2[lane] * h_stride];
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
      const s_t hy_0 = bhy_0[lane];
      const s_t hy_1 = bhy_1[lane];
      const s_t hy_2 = bhy_2[lane];
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
      const s_t element_out1_0 = output_t2 + output_t3;
      bout0_0[lane] = element_out0_0;
      bout0_1[lane] = element_out0_1;
      bout0_2[lane] = element_out0_2;
      bout1_0[lane] = element_out1_0;
      bout1_1[lane] = element_out1_1;
      bout1_2[lane] = element_out1_2;
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
  }

  return SFEM_SUCCESS;
}

template <typename s_t, typename tangent_t, typename scale_t>
static SFEM_INLINE int neohookean_ogden_tri3_inexact_apply_compressed_a_msoa_impl(
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
    const s_t hx_0 = hx[ev0 * h_stride];
    const s_t hx_1 = hx[ev1 * h_stride];
    const s_t hx_2 = hx[ev2 * h_stride];
    const s_t hy_0 = hy[ev0 * h_stride];
    const s_t hy_1 = hy[ev1 * h_stride];
    const s_t hy_2 = hy[ev2 * h_stride];
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
    const s_t element_out1_0 = output_t2 + output_t3;
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
