#pragma once
#include "../../../kernel_math.hpp"

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t, typename tangent_t, int VS>
static SFEM_INLINE int neohookean_ogden_quad4_inexact_apply_tangent_a_msoa_impl(
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
    static constexpr int NQ = 4;
    static constexpr s_t QGRAD[32] = {s_t(-0.78867513459481287), s_t(-0.78867513459481287), s_t(0.78867513459481287), s_t(-0.21132486540518711), s_t(0.21132486540518711), s_t(0.21132486540518711), s_t(-0.21132486540518711), s_t(0.78867513459481287), s_t(-0.78867513459481287), s_t(-0.21132486540518711), s_t(0.78867513459481287), s_t(-0.78867513459481287), s_t(0.21132486540518711), s_t(0.78867513459481287), s_t(-0.21132486540518711), s_t(0.21132486540518711), s_t(-0.21132486540518711), s_t(-0.78867513459481287), s_t(0.21132486540518711), s_t(-0.21132486540518711), s_t(0.78867513459481287), s_t(0.21132486540518711), s_t(-0.78867513459481287), s_t(0.78867513459481287), s_t(-0.21132486540518711), s_t(-0.21132486540518711), s_t(0.21132486540518711), s_t(-0.78867513459481287), s_t(0.78867513459481287), s_t(0.78867513459481287), s_t(-0.78867513459481287), s_t(0.21132486540518711)};
    static constexpr s_t QWEIGHT[4] = {s_t(0.25), s_t(0.25), s_t(0.25), s_t(0.25)};
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
    s_t bux_0[VS];
    s_t bux_1[VS];
    s_t bux_2[VS];
    s_t bux_3[VS];
    s_t buy_0[VS];
    s_t buy_1[VS];
    s_t buy_2[VS];
    s_t buy_3[VS];
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      bux_0[lane] = ux[bev0[lane] * u_stride];
      bux_1[lane] = ux[bev1[lane] * u_stride];
      bux_2[lane] = ux[bev2[lane] * u_stride];
      bux_3[lane] = ux[bev3[lane] * u_stride];
      buy_0[lane] = uy[bev0[lane] * u_stride];
      buy_1[lane] = uy[bev1[lane] * u_stride];
      buy_2[lane] = uy[bev2[lane] * u_stride];
      buy_3[lane] = uy[bev3[lane] * u_stride];
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
      const s_t ux_0 = bux_0[lane];
      const s_t ux_1 = bux_1[lane];
      const s_t ux_2 = bux_2[lane];
      const s_t ux_3 = bux_3[lane];
      const s_t uy_0 = buy_0[lane];
      const s_t uy_1 = buy_1[lane];
      const s_t uy_2 = buy_2[lane];
      const s_t uy_3 = buy_3[lane];
      const s_t adjugate0 = s_t(bg_adj0[lane]);
      const s_t adjugate1 = s_t(bg_adj1[lane]);
      const s_t adjugate2 = s_t(bg_adj2[lane]);
      const s_t adjugate3 = s_t(bg_adj3[lane]);
      const s_t determinant = s_t(bg_det0[lane]);
      s_t tangent0 = s_t(0);
      s_t tangent1 = s_t(0);
      s_t tangent2 = s_t(0);
      s_t tangent3 = s_t(0);
      s_t tangent4 = s_t(0);
      s_t tangent5 = s_t(0);
      s_t tangent6 = s_t(0);
      s_t tangent7 = s_t(0);
      s_t tangent8 = s_t(0);
      s_t tangent9 = s_t(0);
      for (int q = 0; q < NQ; ++q) {
        const s_t gref_0_0 = QGRAD[q * 8 + 0];
        const s_t gref_0_1 = QGRAD[q * 8 + 1];
        const s_t gref_1_0 = QGRAD[q * 8 + 2];
        const s_t gref_1_1 = QGRAD[q * 8 + 3];
        const s_t gref_2_0 = QGRAD[q * 8 + 4];
        const s_t gref_2_1 = QGRAD[q * 8 + 5];
        const s_t gref_3_0 = QGRAD[q * 8 + 6];
        const s_t gref_3_1 = QGRAD[q * 8 + 7];
        const s_t qw = QWEIGHT[q];
            const s_t integrand_t0 = pow_m1(determinant);
            const s_t integrand_t1 = pow_2(adjugate1);
            const s_t integrand_t2 = adjugate1*integrand_t0;
            const s_t integrand_t3 = adjugate3*integrand_t0;
            const s_t integrand_t4 = gref_0_0*integrand_t2 + gref_0_1*integrand_t3;
            const s_t integrand_t5 = gref_1_0*integrand_t2 + gref_1_1*integrand_t3;
            const s_t integrand_t6 = gref_2_0*integrand_t2 + gref_2_1*integrand_t3;
            const s_t integrand_t7 = gref_3_0*integrand_t2 + gref_3_1*integrand_t3;
            const s_t integrand_t8 = integrand_t4*ux_0 + integrand_t5*ux_1 + integrand_t6*ux_2 + integrand_t7*ux_3;
            const s_t integrand_t9 = adjugate0*integrand_t0;
            const s_t integrand_t10 = adjugate2*integrand_t0;
            const s_t integrand_t11 = gref_0_0*integrand_t9 + gref_0_1*integrand_t10;
            const s_t integrand_t12 = gref_1_0*integrand_t9 + gref_1_1*integrand_t10;
            const s_t integrand_t13 = gref_2_0*integrand_t9 + gref_2_1*integrand_t10;
            const s_t integrand_t14 = gref_3_0*integrand_t9 + gref_3_1*integrand_t10;
            const s_t integrand_t15 = integrand_t11*uy_0 + integrand_t12*uy_1 + integrand_t13*uy_2 + integrand_t14*uy_3;
            const s_t integrand_t16 = integrand_t15*integrand_t8;
            const s_t integrand_t17 = integrand_t11*ux_0 + integrand_t12*ux_1 + integrand_t13*ux_2 + integrand_t14*ux_3 + s_t(1);
            const s_t integrand_t18 = integrand_t4*uy_0 + integrand_t5*uy_1 + integrand_t6*uy_2 + integrand_t7*uy_3 + s_t(1);
            const s_t integrand_t19 = -integrand_t16 + integrand_t17*integrand_t18;
            const s_t integrand_t20 = pow_m2(integrand_t19);
            const s_t integrand_t21 = pow_2(integrand_t15)*integrand_t20;
            const s_t integrand_t22 = integrand_t21*lmbda;
            const s_t integrand_t23 = log(integrand_t19);
            const s_t integrand_t24 = integrand_t21*mu - integrand_t22*integrand_t23 + integrand_t22 + mu;
            const s_t integrand_t25 = pow_2(adjugate0);
            const s_t integrand_t26 = pow_2(integrand_t18)*integrand_t20;
            const s_t integrand_t27 = integrand_t26*lmbda;
            const s_t integrand_t28 = -integrand_t23*integrand_t27 + integrand_t26*mu + integrand_t27 + mu;
            const s_t integrand_t29 = adjugate0*adjugate1;
            const s_t integrand_t30 = integrand_t18*integrand_t20;
            const s_t integrand_t31 = integrand_t15*integrand_t30;
            const s_t integrand_t32 = integrand_t15*integrand_t18*integrand_t20*integrand_t23*lmbda - integrand_t31*lmbda - integrand_t31*mu;
            const s_t integrand_t33 = s_t(2)*integrand_t32;
            const s_t integrand_t34 = adjugate1*adjugate3;
            const s_t integrand_t35 = adjugate0*adjugate2;
            const s_t integrand_t36 = adjugate0*adjugate3;
            const s_t integrand_t37 = adjugate1*adjugate2;
            const s_t integrand_t38 = integrand_t30*integrand_t8;
            const s_t integrand_t39 = integrand_t18*integrand_t20*integrand_t23*integrand_t8*lmbda - integrand_t38*lmbda - integrand_t38*mu;
            const s_t integrand_t40 = integrand_t20*lmbda;
            const s_t integrand_t41 = integrand_t15*integrand_t17;
            const s_t integrand_t42 = integrand_t20*mu;
            const s_t integrand_t43 = integrand_t15*integrand_t17*integrand_t20*integrand_t23*lmbda - integrand_t40*integrand_t41 - integrand_t41*integrand_t42;
            const s_t integrand_t44 = pow_m1(integrand_t19);
            const s_t integrand_t45 = integrand_t44*mu;
            const s_t integrand_t46 = integrand_t23*lmbda;
            const s_t integrand_t47 = integrand_t44*integrand_t46;
            const s_t integrand_t48 = integrand_t20*integrand_t46;
            const s_t integrand_t49 = integrand_t16*integrand_t40 + integrand_t16*integrand_t42 - integrand_t16*integrand_t48 + integrand_t45 - integrand_t47;
            const s_t integrand_t50 = integrand_t17*integrand_t18;
            const s_t integrand_t51 = integrand_t40*integrand_t50 + integrand_t42*integrand_t50 - integrand_t45 + integrand_t47 - integrand_t48*integrand_t50;
            const s_t integrand_t52 = integrand_t34*integrand_t43 + integrand_t35*integrand_t39;
            const s_t integrand_t53 = pow_2(adjugate3);
            const s_t integrand_t54 = pow_2(adjugate2);
            const s_t integrand_t55 = adjugate2*adjugate3;
            const s_t integrand_t56 = integrand_t20*pow_2(integrand_t8);
            const s_t integrand_t57 = -integrand_t46*integrand_t56 + integrand_t56*lmbda + integrand_t56*mu + mu;
            const s_t integrand_t58 = pow_2(integrand_t17)*integrand_t20;
            const s_t integrand_t59 = -integrand_t46*integrand_t58 + integrand_t58*lmbda + integrand_t58*mu + mu;
            const s_t integrand_t60 = integrand_t17*integrand_t8;
            const s_t integrand_t61 = integrand_t17*integrand_t20*integrand_t23*integrand_t8*lmbda - integrand_t40*integrand_t60 - integrand_t42*integrand_t60;
            const s_t integrand_t62 = s_t(2)*integrand_t61;
            const s_t integrand0 = integrand_t0*(integrand_t1*integrand_t24 + integrand_t25*integrand_t28 + integrand_t29*integrand_t33);
            const s_t integrand1 = integrand_t0*(integrand_t24*integrand_t34 + integrand_t28*integrand_t35 + integrand_t32*integrand_t36 + integrand_t32*integrand_t37);
            const s_t integrand2 = integrand_t0*(integrand_t1*integrand_t43 + integrand_t25*integrand_t39 + integrand_t29*integrand_t49 + integrand_t29*integrand_t51);
            const s_t integrand3 = integrand_t0*(integrand_t36*integrand_t51 + integrand_t37*integrand_t49 + integrand_t52);
            const s_t integrand4 = integrand_t0*(integrand_t24*integrand_t53 + integrand_t28*integrand_t54 + integrand_t33*integrand_t55);
            const s_t integrand5 = integrand_t0*(integrand_t36*integrand_t49 + integrand_t37*integrand_t51 + integrand_t52);
            const s_t integrand6 = integrand_t0*(integrand_t39*integrand_t54 + integrand_t43*integrand_t53 + integrand_t49*integrand_t55 + integrand_t51*integrand_t55);
            const s_t integrand7 = integrand_t0*(integrand_t1*integrand_t59 + integrand_t25*integrand_t57 + integrand_t29*integrand_t62);
            const s_t integrand8 = integrand_t0*(integrand_t34*integrand_t59 + integrand_t35*integrand_t57 + integrand_t36*integrand_t61 + integrand_t37*integrand_t61);
            const s_t integrand9 = integrand_t0*(integrand_t53*integrand_t59 + integrand_t54*integrand_t57 + integrand_t55*integrand_t62);
        tangent0 += qw * integrand0;
        tangent1 += qw * integrand1;
        tangent2 += qw * integrand2;
        tangent3 += qw * integrand3;
        tangent4 += qw * integrand4;
        tangent5 += qw * integrand5;
        tangent6 += qw * integrand6;
        tangent7 += qw * integrand7;
        tangent8 += qw * integrand8;
        tangent9 += qw * integrand9;
      }
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
static SFEM_INLINE int neohookean_ogden_quad4_inexact_apply_stored_a_msoa_impl(
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
      const s_t reference_product_t1 = ((s_t(1) / s_t(6)))*hx_3;
      const s_t reference_product_t2 = ((s_t(1) / s_t(3)))*hx_0 - (s_t(1) / s_t(6))*hx_2;
      const s_t reference_product_t3 = -reference_product_t0 + reference_product_t1 + reference_product_t2;
      const s_t reference_product_t4 = ((s_t(1) / s_t(4)))*hx_1;
      const s_t reference_product_t5 = ((s_t(1) / s_t(4)))*hx_3;
      const s_t reference_product_t6 = ((s_t(1) / s_t(4)))*hx_0 - (s_t(1) / s_t(4))*hx_2;
      const s_t reference_product_t7 = -reference_product_t4 + reference_product_t5 + reference_product_t6;
      const s_t reference_product_t8 = ((s_t(1) / s_t(6)))*hx_1;
      const s_t reference_product_t9 = ((s_t(1) / s_t(3)))*hx_3;
      const s_t reference_product_t10 = ((s_t(1) / s_t(6)))*hx_0 - (s_t(1) / s_t(3))*hx_2;
      const s_t reference_product_t11 = reference_product_t10 - reference_product_t8 + reference_product_t9;
      const s_t reference_product_t12 = -reference_product_t7;
      const s_t reference_product_t13 = reference_product_t4 - reference_product_t5 + reference_product_t6;
      const s_t reference_product_t14 = reference_product_t2 + reference_product_t8 - reference_product_t9;
      const s_t reference_product_t15 = -reference_product_t13;
      const s_t reference_product_t16 = reference_product_t0 - reference_product_t1 + reference_product_t10;
      const s_t reference_product_t17 = ((s_t(1) / s_t(3)))*hy_1;
      const s_t reference_product_t18 = ((s_t(1) / s_t(6)))*hy_3;
      const s_t reference_product_t19 = ((s_t(1) / s_t(3)))*hy_0 - (s_t(1) / s_t(6))*hy_2;
      const s_t reference_product_t20 = -reference_product_t17 + reference_product_t18 + reference_product_t19;
      const s_t reference_product_t21 = ((s_t(1) / s_t(4)))*hy_1;
      const s_t reference_product_t22 = ((s_t(1) / s_t(4)))*hy_3;
      const s_t reference_product_t23 = ((s_t(1) / s_t(4)))*hy_0 - (s_t(1) / s_t(4))*hy_2;
      const s_t reference_product_t24 = -reference_product_t21 + reference_product_t22 + reference_product_t23;
      const s_t reference_product_t25 = ((s_t(1) / s_t(6)))*hy_1;
      const s_t reference_product_t26 = ((s_t(1) / s_t(3)))*hy_3;
      const s_t reference_product_t27 = ((s_t(1) / s_t(6)))*hy_0 - (s_t(1) / s_t(3))*hy_2;
      const s_t reference_product_t28 = -reference_product_t25 + reference_product_t26 + reference_product_t27;
      const s_t reference_product_t29 = -reference_product_t24;
      const s_t reference_product_t30 = reference_product_t21 - reference_product_t22 + reference_product_t23;
      const s_t reference_product_t31 = reference_product_t19 + reference_product_t25 - reference_product_t26;
      const s_t reference_product_t32 = -reference_product_t30;
      const s_t reference_product_t33 = reference_product_t17 - reference_product_t18 + reference_product_t27;
      const s_t pa_g0_0_0_0 = reference_product_t3;
      const s_t pa_g0_0_0_1 = reference_product_t7;
      const s_t pa_g0_0_1_0 = -reference_product_t3;
      const s_t pa_g0_0_1_1 = reference_product_t7;
      const s_t pa_g0_0_2_0 = -reference_product_t11;
      const s_t pa_g0_0_2_1 = reference_product_t12;
      const s_t pa_g0_0_3_0 = reference_product_t11;
      const s_t pa_g0_0_3_1 = reference_product_t12;
      const s_t pa_g0_1_0_0 = reference_product_t13;
      const s_t pa_g0_1_0_1 = reference_product_t14;
      const s_t pa_g0_1_1_0 = reference_product_t15;
      const s_t pa_g0_1_1_1 = reference_product_t16;
      const s_t pa_g0_1_2_0 = reference_product_t15;
      const s_t pa_g0_1_2_1 = -reference_product_t16;
      const s_t pa_g0_1_3_0 = reference_product_t13;
      const s_t pa_g0_1_3_1 = -reference_product_t14;
      const s_t pa_g1_0_0_0 = reference_product_t20;
      const s_t pa_g1_0_0_1 = reference_product_t24;
      const s_t pa_g1_0_1_0 = -reference_product_t20;
      const s_t pa_g1_0_1_1 = reference_product_t24;
      const s_t pa_g1_0_2_0 = -reference_product_t28;
      const s_t pa_g1_0_2_1 = reference_product_t29;
      const s_t pa_g1_0_3_0 = reference_product_t28;
      const s_t pa_g1_0_3_1 = reference_product_t29;
      const s_t pa_g1_1_0_0 = reference_product_t30;
      const s_t pa_g1_1_0_1 = reference_product_t31;
      const s_t pa_g1_1_1_0 = reference_product_t32;
      const s_t pa_g1_1_1_1 = reference_product_t33;
      const s_t pa_g1_1_2_0 = reference_product_t32;
      const s_t pa_g1_1_2_1 = -reference_product_t33;
      const s_t pa_g1_1_3_0 = reference_product_t30;
      const s_t pa_g1_1_3_1 = -reference_product_t31;
      const s_t element_out0_0 = pa_g0_0_0_0*tangent0 + pa_g0_0_0_1*tangent1 + pa_g0_1_0_0*tangent1 + pa_g0_1_0_1*tangent4 + pa_g1_0_0_0*tangent2 + pa_g1_0_0_1*tangent5 + pa_g1_1_0_0*tangent3 + pa_g1_1_0_1*tangent6;
      const s_t element_out0_1 = pa_g0_0_1_0*tangent0 + pa_g0_0_1_1*tangent1 + pa_g0_1_1_0*tangent1 + pa_g0_1_1_1*tangent4 + pa_g1_0_1_0*tangent2 + pa_g1_0_1_1*tangent5 + pa_g1_1_1_0*tangent3 + pa_g1_1_1_1*tangent6;
      const s_t element_out0_2 = pa_g0_0_2_0*tangent0 + pa_g0_0_2_1*tangent1 + pa_g0_1_2_0*tangent1 + pa_g0_1_2_1*tangent4 + pa_g1_0_2_0*tangent2 + pa_g1_0_2_1*tangent5 + pa_g1_1_2_0*tangent3 + pa_g1_1_2_1*tangent6;
      const s_t element_out0_3 = pa_g0_0_3_0*tangent0 + pa_g0_0_3_1*tangent1 + pa_g0_1_3_0*tangent1 + pa_g0_1_3_1*tangent4 + pa_g1_0_3_0*tangent2 + pa_g1_0_3_1*tangent5 + pa_g1_1_3_0*tangent3 + pa_g1_1_3_1*tangent6;
      const s_t element_out1_0 = pa_g0_0_0_0*tangent2 + pa_g0_0_0_1*tangent3 + pa_g0_1_0_0*tangent5 + pa_g0_1_0_1*tangent6 + pa_g1_0_0_0*tangent7 + pa_g1_0_0_1*tangent8 + pa_g1_1_0_0*tangent8 + pa_g1_1_0_1*tangent9;
      const s_t element_out1_1 = pa_g0_0_1_0*tangent2 + pa_g0_0_1_1*tangent3 + pa_g0_1_1_0*tangent5 + pa_g0_1_1_1*tangent6 + pa_g1_0_1_0*tangent7 + pa_g1_0_1_1*tangent8 + pa_g1_1_1_0*tangent8 + pa_g1_1_1_1*tangent9;
      const s_t element_out1_2 = pa_g0_0_2_0*tangent2 + pa_g0_0_2_1*tangent3 + pa_g0_1_2_0*tangent5 + pa_g0_1_2_1*tangent6 + pa_g1_0_2_0*tangent7 + pa_g1_0_2_1*tangent8 + pa_g1_1_2_0*tangent8 + pa_g1_1_2_1*tangent9;
      const s_t element_out1_3 = pa_g0_0_3_0*tangent2 + pa_g0_0_3_1*tangent3 + pa_g0_1_3_0*tangent5 + pa_g0_1_3_1*tangent6 + pa_g1_0_3_0*tangent7 + pa_g1_0_3_1*tangent8 + pa_g1_1_3_0*tangent8 + pa_g1_1_3_1*tangent9;
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
static SFEM_INLINE int neohookean_ogden_quad4_inexact_apply_compressed_a_msoa_impl(
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
    const s_t reference_product_t1 = ((s_t(1) / s_t(6)))*hx_3;
    const s_t reference_product_t2 = ((s_t(1) / s_t(3)))*hx_0 - (s_t(1) / s_t(6))*hx_2;
    const s_t reference_product_t3 = -reference_product_t0 + reference_product_t1 + reference_product_t2;
    const s_t reference_product_t4 = ((s_t(1) / s_t(4)))*hx_1;
    const s_t reference_product_t5 = ((s_t(1) / s_t(4)))*hx_3;
    const s_t reference_product_t6 = ((s_t(1) / s_t(4)))*hx_0 - (s_t(1) / s_t(4))*hx_2;
    const s_t reference_product_t7 = -reference_product_t4 + reference_product_t5 + reference_product_t6;
    const s_t reference_product_t8 = ((s_t(1) / s_t(6)))*hx_1;
    const s_t reference_product_t9 = ((s_t(1) / s_t(3)))*hx_3;
    const s_t reference_product_t10 = ((s_t(1) / s_t(6)))*hx_0 - (s_t(1) / s_t(3))*hx_2;
    const s_t reference_product_t11 = reference_product_t10 - reference_product_t8 + reference_product_t9;
    const s_t reference_product_t12 = -reference_product_t7;
    const s_t reference_product_t13 = reference_product_t4 - reference_product_t5 + reference_product_t6;
    const s_t reference_product_t14 = reference_product_t2 + reference_product_t8 - reference_product_t9;
    const s_t reference_product_t15 = -reference_product_t13;
    const s_t reference_product_t16 = reference_product_t0 - reference_product_t1 + reference_product_t10;
    const s_t reference_product_t17 = ((s_t(1) / s_t(3)))*hy_1;
    const s_t reference_product_t18 = ((s_t(1) / s_t(6)))*hy_3;
    const s_t reference_product_t19 = ((s_t(1) / s_t(3)))*hy_0 - (s_t(1) / s_t(6))*hy_2;
    const s_t reference_product_t20 = -reference_product_t17 + reference_product_t18 + reference_product_t19;
    const s_t reference_product_t21 = ((s_t(1) / s_t(4)))*hy_1;
    const s_t reference_product_t22 = ((s_t(1) / s_t(4)))*hy_3;
    const s_t reference_product_t23 = ((s_t(1) / s_t(4)))*hy_0 - (s_t(1) / s_t(4))*hy_2;
    const s_t reference_product_t24 = -reference_product_t21 + reference_product_t22 + reference_product_t23;
    const s_t reference_product_t25 = ((s_t(1) / s_t(6)))*hy_1;
    const s_t reference_product_t26 = ((s_t(1) / s_t(3)))*hy_3;
    const s_t reference_product_t27 = ((s_t(1) / s_t(6)))*hy_0 - (s_t(1) / s_t(3))*hy_2;
    const s_t reference_product_t28 = -reference_product_t25 + reference_product_t26 + reference_product_t27;
    const s_t reference_product_t29 = -reference_product_t24;
    const s_t reference_product_t30 = reference_product_t21 - reference_product_t22 + reference_product_t23;
    const s_t reference_product_t31 = reference_product_t19 + reference_product_t25 - reference_product_t26;
    const s_t reference_product_t32 = -reference_product_t30;
    const s_t reference_product_t33 = reference_product_t17 - reference_product_t18 + reference_product_t27;
    const s_t pa_g0_0_0_0 = reference_product_t3;
    const s_t pa_g0_0_0_1 = reference_product_t7;
    const s_t pa_g0_0_1_0 = -reference_product_t3;
    const s_t pa_g0_0_1_1 = reference_product_t7;
    const s_t pa_g0_0_2_0 = -reference_product_t11;
    const s_t pa_g0_0_2_1 = reference_product_t12;
    const s_t pa_g0_0_3_0 = reference_product_t11;
    const s_t pa_g0_0_3_1 = reference_product_t12;
    const s_t pa_g0_1_0_0 = reference_product_t13;
    const s_t pa_g0_1_0_1 = reference_product_t14;
    const s_t pa_g0_1_1_0 = reference_product_t15;
    const s_t pa_g0_1_1_1 = reference_product_t16;
    const s_t pa_g0_1_2_0 = reference_product_t15;
    const s_t pa_g0_1_2_1 = -reference_product_t16;
    const s_t pa_g0_1_3_0 = reference_product_t13;
    const s_t pa_g0_1_3_1 = -reference_product_t14;
    const s_t pa_g1_0_0_0 = reference_product_t20;
    const s_t pa_g1_0_0_1 = reference_product_t24;
    const s_t pa_g1_0_1_0 = -reference_product_t20;
    const s_t pa_g1_0_1_1 = reference_product_t24;
    const s_t pa_g1_0_2_0 = -reference_product_t28;
    const s_t pa_g1_0_2_1 = reference_product_t29;
    const s_t pa_g1_0_3_0 = reference_product_t28;
    const s_t pa_g1_0_3_1 = reference_product_t29;
    const s_t pa_g1_1_0_0 = reference_product_t30;
    const s_t pa_g1_1_0_1 = reference_product_t31;
    const s_t pa_g1_1_1_0 = reference_product_t32;
    const s_t pa_g1_1_1_1 = reference_product_t33;
    const s_t pa_g1_1_2_0 = reference_product_t32;
    const s_t pa_g1_1_2_1 = -reference_product_t33;
    const s_t pa_g1_1_3_0 = reference_product_t30;
    const s_t pa_g1_1_3_1 = -reference_product_t31;
    const s_t element_out0_0 = pa_g0_0_0_0*tangent0 + pa_g0_0_0_1*tangent1 + pa_g0_1_0_0*tangent1 + pa_g0_1_0_1*tangent4 + pa_g1_0_0_0*tangent2 + pa_g1_0_0_1*tangent5 + pa_g1_1_0_0*tangent3 + pa_g1_1_0_1*tangent6;
    const s_t element_out0_1 = pa_g0_0_1_0*tangent0 + pa_g0_0_1_1*tangent1 + pa_g0_1_1_0*tangent1 + pa_g0_1_1_1*tangent4 + pa_g1_0_1_0*tangent2 + pa_g1_0_1_1*tangent5 + pa_g1_1_1_0*tangent3 + pa_g1_1_1_1*tangent6;
    const s_t element_out0_2 = pa_g0_0_2_0*tangent0 + pa_g0_0_2_1*tangent1 + pa_g0_1_2_0*tangent1 + pa_g0_1_2_1*tangent4 + pa_g1_0_2_0*tangent2 + pa_g1_0_2_1*tangent5 + pa_g1_1_2_0*tangent3 + pa_g1_1_2_1*tangent6;
    const s_t element_out0_3 = pa_g0_0_3_0*tangent0 + pa_g0_0_3_1*tangent1 + pa_g0_1_3_0*tangent1 + pa_g0_1_3_1*tangent4 + pa_g1_0_3_0*tangent2 + pa_g1_0_3_1*tangent5 + pa_g1_1_3_0*tangent3 + pa_g1_1_3_1*tangent6;
    const s_t element_out1_0 = pa_g0_0_0_0*tangent2 + pa_g0_0_0_1*tangent3 + pa_g0_1_0_0*tangent5 + pa_g0_1_0_1*tangent6 + pa_g1_0_0_0*tangent7 + pa_g1_0_0_1*tangent8 + pa_g1_1_0_0*tangent8 + pa_g1_1_0_1*tangent9;
    const s_t element_out1_1 = pa_g0_0_1_0*tangent2 + pa_g0_0_1_1*tangent3 + pa_g0_1_1_0*tangent5 + pa_g0_1_1_1*tangent6 + pa_g1_0_1_0*tangent7 + pa_g1_0_1_1*tangent8 + pa_g1_1_1_0*tangent8 + pa_g1_1_1_1*tangent9;
    const s_t element_out1_2 = pa_g0_0_2_0*tangent2 + pa_g0_0_2_1*tangent3 + pa_g0_1_2_0*tangent5 + pa_g0_1_2_1*tangent6 + pa_g1_0_2_0*tangent7 + pa_g1_0_2_1*tangent8 + pa_g1_1_2_0*tangent8 + pa_g1_1_2_1*tangent9;
    const s_t element_out1_3 = pa_g0_0_3_0*tangent2 + pa_g0_0_3_1*tangent3 + pa_g0_1_3_0*tangent5 + pa_g0_1_3_1*tangent6 + pa_g1_0_3_0*tangent7 + pa_g1_0_3_1*tangent8 + pa_g1_1_3_0*tangent8 + pa_g1_1_3_1*tangent9;
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
