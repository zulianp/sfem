#pragma once
#include "../../../kernel_math.hpp"
#include "../../../packed_thread_scratch.hpp"

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t, typename tangent_t, int VS>
static SFEM_INLINE int linear_elasticity_tet4_inexact_apply_tangent_a_msoa_impl(
    const ptrdiff_t nelements,
    const g_t *const RSTR g_adj0,
    const g_t *const RSTR g_adj1,
    const g_t *const RSTR g_adj2,
    const g_t *const RSTR g_adj3,
    const g_t *const RSTR g_adj4,
    const g_t *const RSTR g_adj5,
    const g_t *const RSTR g_adj6,
    const g_t *const RSTR g_adj7,
    const g_t *const RSTR g_adj8,
    const g_t *const RSTR g_det0,
    const s_t lmbda,
    const s_t mu,
    const ptrdiff_t tangent_component_stride,
    tangent_t *const RSTR tangent
) {
  #pragma omp parallel for schedule(static)
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)((nelements - evb) < (ptrdiff_t)VS ? (nelements - evb) : (ptrdiff_t)VS);
    const g_t *const RSTR bg_adj0 = g_adj0 + evb;
    const g_t *const RSTR bg_adj1 = g_adj1 + evb;
    const g_t *const RSTR bg_adj2 = g_adj2 + evb;
    const g_t *const RSTR bg_adj3 = g_adj3 + evb;
    const g_t *const RSTR bg_adj4 = g_adj4 + evb;
    const g_t *const RSTR bg_adj5 = g_adj5 + evb;
    const g_t *const RSTR bg_adj6 = g_adj6 + evb;
    const g_t *const RSTR bg_adj7 = g_adj7 + evb;
    const g_t *const RSTR bg_adj8 = g_adj8 + evb;
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
    tangent_t *const RSTR btangent10 = tangent + evb + 10 * tangent_component_stride;
    tangent_t *const RSTR btangent11 = tangent + evb + 11 * tangent_component_stride;
    tangent_t *const RSTR btangent12 = tangent + evb + 12 * tangent_component_stride;
    tangent_t *const RSTR btangent13 = tangent + evb + 13 * tangent_component_stride;
    tangent_t *const RSTR btangent14 = tangent + evb + 14 * tangent_component_stride;
    tangent_t *const RSTR btangent15 = tangent + evb + 15 * tangent_component_stride;
    tangent_t *const RSTR btangent16 = tangent + evb + 16 * tangent_component_stride;
    tangent_t *const RSTR btangent17 = tangent + evb + 17 * tangent_component_stride;
    tangent_t *const RSTR btangent18 = tangent + evb + 18 * tangent_component_stride;
    tangent_t *const RSTR btangent19 = tangent + evb + 19 * tangent_component_stride;
    tangent_t *const RSTR btangent20 = tangent + evb + 20 * tangent_component_stride;
    tangent_t *const RSTR btangent21 = tangent + evb + 21 * tangent_component_stride;
    tangent_t *const RSTR btangent22 = tangent + evb + 22 * tangent_component_stride;
    tangent_t *const RSTR btangent23 = tangent + evb + 23 * tangent_component_stride;
    tangent_t *const RSTR btangent24 = tangent + evb + 24 * tangent_component_stride;
    tangent_t *const RSTR btangent25 = tangent + evb + 25 * tangent_component_stride;
    tangent_t *const RSTR btangent26 = tangent + evb + 26 * tangent_component_stride;
    tangent_t *const RSTR btangent27 = tangent + evb + 27 * tangent_component_stride;
    tangent_t *const RSTR btangent28 = tangent + evb + 28 * tangent_component_stride;
    tangent_t *const RSTR btangent29 = tangent + evb + 29 * tangent_component_stride;
    tangent_t *const RSTR btangent30 = tangent + evb + 30 * tangent_component_stride;
    tangent_t *const RSTR btangent31 = tangent + evb + 31 * tangent_component_stride;
    tangent_t *const RSTR btangent32 = tangent + evb + 32 * tangent_component_stride;
    tangent_t *const RSTR btangent33 = tangent + evb + 33 * tangent_component_stride;
    tangent_t *const RSTR btangent34 = tangent + evb + 34 * tangent_component_stride;
    tangent_t *const RSTR btangent35 = tangent + evb + 35 * tangent_component_stride;
    tangent_t *const RSTR btangent36 = tangent + evb + 36 * tangent_component_stride;
    tangent_t *const RSTR btangent37 = tangent + evb + 37 * tangent_component_stride;
    tangent_t *const RSTR btangent38 = tangent + evb + 38 * tangent_component_stride;
    tangent_t *const RSTR btangent39 = tangent + evb + 39 * tangent_component_stride;
    tangent_t *const RSTR btangent40 = tangent + evb + 40 * tangent_component_stride;
    tangent_t *const RSTR btangent41 = tangent + evb + 41 * tangent_component_stride;
    tangent_t *const RSTR btangent42 = tangent + evb + 42 * tangent_component_stride;
    tangent_t *const RSTR btangent43 = tangent + evb + 43 * tangent_component_stride;
    tangent_t *const RSTR btangent44 = tangent + evb + 44 * tangent_component_stride;
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      const s_t adjugate0 = s_t(bg_adj0[lane]);
      const s_t adjugate1 = s_t(bg_adj1[lane]);
      const s_t adjugate2 = s_t(bg_adj2[lane]);
      const s_t adjugate3 = s_t(bg_adj3[lane]);
      const s_t adjugate4 = s_t(bg_adj4[lane]);
      const s_t adjugate5 = s_t(bg_adj5[lane]);
      const s_t adjugate6 = s_t(bg_adj6[lane]);
      const s_t adjugate7 = s_t(bg_adj7[lane]);
      const s_t adjugate8 = s_t(bg_adj8[lane]);
      const s_t determinant = s_t(bg_det0[lane]);
      const s_t tangent_t0 = pow_m1(determinant);
      const s_t tangent_t1 = pow_2(adjugate1);
      const s_t tangent_t2 = mu*tangent_t1;
      const s_t tangent_t3 = pow_2(adjugate2);
      const s_t tangent_t4 = mu*tangent_t3;
      const s_t tangent_t5 = pow_2(adjugate0);
      const s_t tangent_t6 = lmbda + s_t(2)*mu;
      const s_t tangent_t7 = adjugate1*mu;
      const s_t tangent_t8 = adjugate4*tangent_t7;
      const s_t tangent_t9 = adjugate2*mu;
      const s_t tangent_t10 = adjugate5*tangent_t9;
      const s_t tangent_t11 = adjugate0*tangent_t6;
      const s_t tangent_t12 = adjugate7*tangent_t7;
      const s_t tangent_t13 = adjugate8*tangent_t9;
      const s_t tangent_t14 = adjugate0*lmbda;
      const s_t tangent_t15 = pow_2(adjugate4);
      const s_t tangent_t16 = mu*tangent_t15;
      const s_t tangent_t17 = pow_2(adjugate5);
      const s_t tangent_t18 = mu*tangent_t17;
      const s_t tangent_t19 = pow_2(adjugate3);
      const s_t tangent_t20 = adjugate4*mu;
      const s_t tangent_t21 = adjugate7*tangent_t20;
      const s_t tangent_t22 = adjugate5*mu;
      const s_t tangent_t23 = adjugate8*tangent_t22;
      const s_t tangent_t24 = adjugate3*adjugate6;
      const s_t tangent_t25 = adjugate3*lmbda;
      const s_t tangent_t26 = pow_2(adjugate7);
      const s_t tangent_t27 = mu*tangent_t26;
      const s_t tangent_t28 = pow_2(adjugate8);
      const s_t tangent_t29 = mu*tangent_t28;
      const s_t tangent_t30 = pow_2(adjugate6);
      const s_t tangent_t31 = adjugate7*mu;
      const s_t tangent_t32 = adjugate6*lmbda;
      const s_t tangent_t33 = adjugate8*mu;
      const s_t tangent_t34 = mu*tangent_t5;
      const s_t tangent_t35 = adjugate0*mu;
      const s_t tangent_t36 = adjugate3*tangent_t35;
      const s_t tangent_t37 = adjugate1*tangent_t6;
      const s_t tangent_t38 = adjugate6*tangent_t35;
      const s_t tangent_t39 = adjugate1*lmbda;
      const s_t tangent_t40 = mu*tangent_t19;
      const s_t tangent_t41 = mu*tangent_t24;
      const s_t tangent_t42 = adjugate4*lmbda;
      const s_t tangent_t43 = mu*tangent_t30;
      const s_t tangent_t44 = adjugate7*lmbda;
      const s_t tangent_t45 = adjugate2*tangent_t6;
      const s_t tangent0 = tangent_t0*(tangent_t2 + tangent_t4 + tangent_t5*tangent_t6);
      const s_t tangent1 = tangent_t0*(adjugate3*tangent_t11 + tangent_t10 + tangent_t8);
      const s_t tangent2 = tangent_t0*(adjugate6*tangent_t11 + tangent_t12 + tangent_t13);
      const s_t tangent3 = tangent_t0*(adjugate0*tangent_t7 + adjugate1*tangent_t14);
      const s_t tangent4 = tangent_t0*(adjugate3*tangent_t7 + adjugate4*tangent_t14);
      const s_t tangent5 = tangent_t0*(adjugate6*tangent_t7 + adjugate7*tangent_t14);
      const s_t tangent6 = tangent_t0*(adjugate0*tangent_t9 + adjugate2*tangent_t14);
      const s_t tangent7 = tangent_t0*(adjugate3*tangent_t9 + adjugate5*tangent_t14);
      const s_t tangent8 = tangent_t0*(adjugate6*tangent_t9 + adjugate8*tangent_t14);
      const s_t tangent9 = tangent_t0*(tangent_t16 + tangent_t18 + tangent_t19*tangent_t6);
      const s_t tangent10 = tangent_t0*(tangent_t21 + tangent_t23 + tangent_t24*tangent_t6);
      const s_t tangent11 = tangent_t0*(adjugate0*tangent_t20 + adjugate1*tangent_t25);
      const s_t tangent12 = tangent_t0*(adjugate3*tangent_t20 + adjugate4*tangent_t25);
      const s_t tangent13 = tangent_t0*(adjugate6*tangent_t20 + adjugate7*tangent_t25);
      const s_t tangent14 = tangent_t0*(adjugate0*tangent_t22 + adjugate2*tangent_t25);
      const s_t tangent15 = tangent_t0*(adjugate3*tangent_t22 + adjugate5*tangent_t25);
      const s_t tangent16 = tangent_t0*(adjugate6*tangent_t22 + adjugate8*tangent_t25);
      const s_t tangent17 = tangent_t0*(tangent_t27 + tangent_t29 + tangent_t30*tangent_t6);
      const s_t tangent18 = tangent_t0*(adjugate0*tangent_t31 + adjugate1*tangent_t32);
      const s_t tangent19 = tangent_t0*(adjugate3*tangent_t31 + adjugate4*tangent_t32);
      const s_t tangent20 = tangent_t0*(adjugate6*tangent_t31 + adjugate7*tangent_t32);
      const s_t tangent21 = tangent_t0*(adjugate0*tangent_t33 + adjugate2*tangent_t32);
      const s_t tangent22 = tangent_t0*(adjugate3*tangent_t33 + adjugate5*tangent_t32);
      const s_t tangent23 = tangent_t0*(adjugate6*tangent_t33 + adjugate8*tangent_t32);
      const s_t tangent24 = tangent_t0*(tangent_t1*tangent_t6 + tangent_t34 + tangent_t4);
      const s_t tangent25 = tangent_t0*(adjugate4*tangent_t37 + tangent_t10 + tangent_t36);
      const s_t tangent26 = tangent_t0*(adjugate7*tangent_t37 + tangent_t13 + tangent_t38);
      const s_t tangent27 = tangent_t0*(adjugate2*tangent_t39 + adjugate2*tangent_t7);
      const s_t tangent28 = tangent_t0*(adjugate4*tangent_t9 + adjugate5*tangent_t39);
      const s_t tangent29 = tangent_t0*(adjugate7*tangent_t9 + adjugate8*tangent_t39);
      const s_t tangent30 = tangent_t0*(tangent_t15*tangent_t6 + tangent_t18 + tangent_t40);
      const s_t tangent31 = tangent_t0*(adjugate4*adjugate7*tangent_t6 + tangent_t23 + tangent_t41);
      const s_t tangent32 = tangent_t0*(adjugate2*tangent_t42 + adjugate5*tangent_t7);
      const s_t tangent33 = tangent_t0*(adjugate5*tangent_t20 + adjugate5*tangent_t42);
      const s_t tangent34 = tangent_t0*(adjugate7*tangent_t22 + adjugate8*tangent_t42);
      const s_t tangent35 = tangent_t0*(tangent_t26*tangent_t6 + tangent_t29 + tangent_t43);
      const s_t tangent36 = tangent_t0*(adjugate2*tangent_t44 + adjugate8*tangent_t7);
      const s_t tangent37 = tangent_t0*(adjugate5*tangent_t44 + adjugate8*tangent_t20);
      const s_t tangent38 = tangent_t0*(adjugate8*tangent_t31 + adjugate8*tangent_t44);
      const s_t tangent39 = tangent_t0*(tangent_t2 + tangent_t3*tangent_t6 + tangent_t34);
      const s_t tangent40 = tangent_t0*(adjugate5*tangent_t45 + tangent_t36 + tangent_t8);
      const s_t tangent41 = tangent_t0*(adjugate8*tangent_t45 + tangent_t12 + tangent_t38);
      const s_t tangent42 = tangent_t0*(tangent_t16 + tangent_t17*tangent_t6 + tangent_t40);
      const s_t tangent43 = tangent_t0*(adjugate5*adjugate8*tangent_t6 + tangent_t21 + tangent_t41);
      const s_t tangent44 = tangent_t0*(tangent_t27 + tangent_t28*tangent_t6 + tangent_t43);
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
      btangent10[lane] = tangent_t(tangent10);
      btangent11[lane] = tangent_t(tangent11);
      btangent12[lane] = tangent_t(tangent12);
      btangent13[lane] = tangent_t(tangent13);
      btangent14[lane] = tangent_t(tangent14);
      btangent15[lane] = tangent_t(tangent15);
      btangent16[lane] = tangent_t(tangent16);
      btangent17[lane] = tangent_t(tangent17);
      btangent18[lane] = tangent_t(tangent18);
      btangent19[lane] = tangent_t(tangent19);
      btangent20[lane] = tangent_t(tangent20);
      btangent21[lane] = tangent_t(tangent21);
      btangent22[lane] = tangent_t(tangent22);
      btangent23[lane] = tangent_t(tangent23);
      btangent24[lane] = tangent_t(tangent24);
      btangent25[lane] = tangent_t(tangent25);
      btangent26[lane] = tangent_t(tangent26);
      btangent27[lane] = tangent_t(tangent27);
      btangent28[lane] = tangent_t(tangent28);
      btangent29[lane] = tangent_t(tangent29);
      btangent30[lane] = tangent_t(tangent30);
      btangent31[lane] = tangent_t(tangent31);
      btangent32[lane] = tangent_t(tangent32);
      btangent33[lane] = tangent_t(tangent33);
      btangent34[lane] = tangent_t(tangent34);
      btangent35[lane] = tangent_t(tangent35);
      btangent36[lane] = tangent_t(tangent36);
      btangent37[lane] = tangent_t(tangent37);
      btangent38[lane] = tangent_t(tangent38);
      btangent39[lane] = tangent_t(tangent39);
      btangent40[lane] = tangent_t(tangent40);
      btangent41[lane] = tangent_t(tangent41);
      btangent42[lane] = tangent_t(tangent42);
      btangent43[lane] = tangent_t(tangent43);
      btangent44[lane] = tangent_t(tangent44);
    }
  }

  return SFEM_SUCCESS;
}

template <typename s_t, typename tangent_t, int VS>
static SFEM_INLINE int linear_elasticity_tet4_inexact_apply_stored_a_msoa_impl(
    const ptrdiff_t nelements,
    idx_t **const RSTR elements,
    const ptrdiff_t tangent_component_stride,
    const tangent_t *const RSTR tangent,
    const ptrdiff_t h_stride,
    const s_t *const RSTR hx,
    const s_t *const RSTR hy,
    const s_t *const RSTR hz,
    const ptrdiff_t out_stride,
    s_t *const RSTR outx,
    s_t *const RSTR outy,
    s_t *const RSTR outz
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
    s_t bhz_0[VS];
    s_t bhz_1[VS];
    s_t bhz_2[VS];
    s_t bhz_3[VS];
    s_t bout0_0[VS];
    s_t bout0_1[VS];
    s_t bout0_2[VS];
    s_t bout0_3[VS];
    s_t bout1_0[VS];
    s_t bout1_1[VS];
    s_t bout1_2[VS];
    s_t bout1_3[VS];
    s_t bout2_0[VS];
    s_t bout2_1[VS];
    s_t bout2_2[VS];
    s_t bout2_3[VS];
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
      bhz_0[lane] = hz[bev0[lane] * h_stride];
      bhz_1[lane] = hz[bev1[lane] * h_stride];
      bhz_2[lane] = hz[bev2[lane] * h_stride];
      bhz_3[lane] = hz[bev3[lane] * h_stride];
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
    const tangent_t *const RSTR btangent10 = tangent + evb + 10 * tangent_component_stride;
    const tangent_t *const RSTR btangent11 = tangent + evb + 11 * tangent_component_stride;
    const tangent_t *const RSTR btangent12 = tangent + evb + 12 * tangent_component_stride;
    const tangent_t *const RSTR btangent13 = tangent + evb + 13 * tangent_component_stride;
    const tangent_t *const RSTR btangent14 = tangent + evb + 14 * tangent_component_stride;
    const tangent_t *const RSTR btangent15 = tangent + evb + 15 * tangent_component_stride;
    const tangent_t *const RSTR btangent16 = tangent + evb + 16 * tangent_component_stride;
    const tangent_t *const RSTR btangent17 = tangent + evb + 17 * tangent_component_stride;
    const tangent_t *const RSTR btangent18 = tangent + evb + 18 * tangent_component_stride;
    const tangent_t *const RSTR btangent19 = tangent + evb + 19 * tangent_component_stride;
    const tangent_t *const RSTR btangent20 = tangent + evb + 20 * tangent_component_stride;
    const tangent_t *const RSTR btangent21 = tangent + evb + 21 * tangent_component_stride;
    const tangent_t *const RSTR btangent22 = tangent + evb + 22 * tangent_component_stride;
    const tangent_t *const RSTR btangent23 = tangent + evb + 23 * tangent_component_stride;
    const tangent_t *const RSTR btangent24 = tangent + evb + 24 * tangent_component_stride;
    const tangent_t *const RSTR btangent25 = tangent + evb + 25 * tangent_component_stride;
    const tangent_t *const RSTR btangent26 = tangent + evb + 26 * tangent_component_stride;
    const tangent_t *const RSTR btangent27 = tangent + evb + 27 * tangent_component_stride;
    const tangent_t *const RSTR btangent28 = tangent + evb + 28 * tangent_component_stride;
    const tangent_t *const RSTR btangent29 = tangent + evb + 29 * tangent_component_stride;
    const tangent_t *const RSTR btangent30 = tangent + evb + 30 * tangent_component_stride;
    const tangent_t *const RSTR btangent31 = tangent + evb + 31 * tangent_component_stride;
    const tangent_t *const RSTR btangent32 = tangent + evb + 32 * tangent_component_stride;
    const tangent_t *const RSTR btangent33 = tangent + evb + 33 * tangent_component_stride;
    const tangent_t *const RSTR btangent34 = tangent + evb + 34 * tangent_component_stride;
    const tangent_t *const RSTR btangent35 = tangent + evb + 35 * tangent_component_stride;
    const tangent_t *const RSTR btangent36 = tangent + evb + 36 * tangent_component_stride;
    const tangent_t *const RSTR btangent37 = tangent + evb + 37 * tangent_component_stride;
    const tangent_t *const RSTR btangent38 = tangent + evb + 38 * tangent_component_stride;
    const tangent_t *const RSTR btangent39 = tangent + evb + 39 * tangent_component_stride;
    const tangent_t *const RSTR btangent40 = tangent + evb + 40 * tangent_component_stride;
    const tangent_t *const RSTR btangent41 = tangent + evb + 41 * tangent_component_stride;
    const tangent_t *const RSTR btangent42 = tangent + evb + 42 * tangent_component_stride;
    const tangent_t *const RSTR btangent43 = tangent + evb + 43 * tangent_component_stride;
    const tangent_t *const RSTR btangent44 = tangent + evb + 44 * tangent_component_stride;
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
      const s_t hz_0 = bhz_0[lane];
      const s_t hz_1 = bhz_1[lane];
      const s_t hz_2 = bhz_2[lane];
      const s_t hz_3 = bhz_3[lane];
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
      const s_t tangent10 = s_t(btangent10[lane]);
      const s_t tangent11 = s_t(btangent11[lane]);
      const s_t tangent12 = s_t(btangent12[lane]);
      const s_t tangent13 = s_t(btangent13[lane]);
      const s_t tangent14 = s_t(btangent14[lane]);
      const s_t tangent15 = s_t(btangent15[lane]);
      const s_t tangent16 = s_t(btangent16[lane]);
      const s_t tangent17 = s_t(btangent17[lane]);
      const s_t tangent18 = s_t(btangent18[lane]);
      const s_t tangent19 = s_t(btangent19[lane]);
      const s_t tangent20 = s_t(btangent20[lane]);
      const s_t tangent21 = s_t(btangent21[lane]);
      const s_t tangent22 = s_t(btangent22[lane]);
      const s_t tangent23 = s_t(btangent23[lane]);
      const s_t tangent24 = s_t(btangent24[lane]);
      const s_t tangent25 = s_t(btangent25[lane]);
      const s_t tangent26 = s_t(btangent26[lane]);
      const s_t tangent27 = s_t(btangent27[lane]);
      const s_t tangent28 = s_t(btangent28[lane]);
      const s_t tangent29 = s_t(btangent29[lane]);
      const s_t tangent30 = s_t(btangent30[lane]);
      const s_t tangent31 = s_t(btangent31[lane]);
      const s_t tangent32 = s_t(btangent32[lane]);
      const s_t tangent33 = s_t(btangent33[lane]);
      const s_t tangent34 = s_t(btangent34[lane]);
      const s_t tangent35 = s_t(btangent35[lane]);
      const s_t tangent36 = s_t(btangent36[lane]);
      const s_t tangent37 = s_t(btangent37[lane]);
      const s_t tangent38 = s_t(btangent38[lane]);
      const s_t tangent39 = s_t(btangent39[lane]);
      const s_t tangent40 = s_t(btangent40[lane]);
      const s_t tangent41 = s_t(btangent41[lane]);
      const s_t tangent42 = s_t(btangent42[lane]);
      const s_t tangent43 = s_t(btangent43[lane]);
      const s_t tangent44 = s_t(btangent44[lane]);
      const s_t compressed_increment_t0 = ((s_t(1) / s_t(6)))*hx_0;
      const s_t compressed_increment_t1 = ((s_t(1) / s_t(6)))*hy_0;
      const s_t compressed_increment_t2 = ((s_t(1) / s_t(6)))*hz_0;
      const s_t pa_p0_0_0 = compressed_increment_t0 - (s_t(1) / s_t(6))*hx_1;
      const s_t pa_p0_0_1 = compressed_increment_t0 - (s_t(1) / s_t(6))*hx_2;
      const s_t pa_p0_0_2 = compressed_increment_t0 - (s_t(1) / s_t(6))*hx_3;
      const s_t pa_p1_0_0 = compressed_increment_t1 - (s_t(1) / s_t(6))*hy_1;
      const s_t pa_p1_0_1 = compressed_increment_t1 - (s_t(1) / s_t(6))*hy_2;
      const s_t pa_p1_0_2 = compressed_increment_t1 - (s_t(1) / s_t(6))*hy_3;
      const s_t pa_p2_0_0 = compressed_increment_t2 - (s_t(1) / s_t(6))*hz_1;
      const s_t pa_p2_0_1 = compressed_increment_t2 - (s_t(1) / s_t(6))*hz_2;
      const s_t pa_p2_0_2 = compressed_increment_t2 - (s_t(1) / s_t(6))*hz_3;
      const s_t pa_y0_0_0 = pa_p0_0_0*tangent0 + pa_p0_0_1*tangent1 + pa_p0_0_2*tangent2 + pa_p1_0_0*tangent3 + pa_p1_0_1*tangent4 + pa_p1_0_2*tangent5 + pa_p2_0_0*tangent6 + pa_p2_0_1*tangent7 + pa_p2_0_2*tangent8;
      const s_t pa_y0_0_1 = pa_p0_0_0*tangent1 + pa_p0_0_1*tangent9 + pa_p0_0_2*tangent10 + pa_p1_0_0*tangent11 + pa_p1_0_1*tangent12 + pa_p1_0_2*tangent13 + pa_p2_0_0*tangent14 + pa_p2_0_1*tangent15 + pa_p2_0_2*tangent16;
      const s_t pa_y0_0_2 = pa_p0_0_0*tangent2 + pa_p0_0_1*tangent10 + pa_p0_0_2*tangent17 + pa_p1_0_0*tangent18 + pa_p1_0_1*tangent19 + pa_p1_0_2*tangent20 + pa_p2_0_0*tangent21 + pa_p2_0_1*tangent22 + pa_p2_0_2*tangent23;
      const s_t pa_y1_0_0 = pa_p0_0_0*tangent3 + pa_p0_0_1*tangent11 + pa_p0_0_2*tangent18 + pa_p1_0_0*tangent24 + pa_p1_0_1*tangent25 + pa_p1_0_2*tangent26 + pa_p2_0_0*tangent27 + pa_p2_0_1*tangent28 + pa_p2_0_2*tangent29;
      const s_t pa_y1_0_1 = pa_p0_0_0*tangent4 + pa_p0_0_1*tangent12 + pa_p0_0_2*tangent19 + pa_p1_0_0*tangent25 + pa_p1_0_1*tangent30 + pa_p1_0_2*tangent31 + pa_p2_0_0*tangent32 + pa_p2_0_1*tangent33 + pa_p2_0_2*tangent34;
      const s_t pa_y1_0_2 = pa_p0_0_0*tangent5 + pa_p0_0_1*tangent13 + pa_p0_0_2*tangent20 + pa_p1_0_0*tangent26 + pa_p1_0_1*tangent31 + pa_p1_0_2*tangent35 + pa_p2_0_0*tangent36 + pa_p2_0_1*tangent37 + pa_p2_0_2*tangent38;
      const s_t pa_y2_0_0 = pa_p0_0_0*tangent6 + pa_p0_0_1*tangent14 + pa_p0_0_2*tangent21 + pa_p1_0_0*tangent27 + pa_p1_0_1*tangent32 + pa_p1_0_2*tangent36 + pa_p2_0_0*tangent39 + pa_p2_0_1*tangent40 + pa_p2_0_2*tangent41;
      const s_t pa_y2_0_1 = pa_p0_0_0*tangent7 + pa_p0_0_1*tangent15 + pa_p0_0_2*tangent22 + pa_p1_0_0*tangent28 + pa_p1_0_1*tangent33 + pa_p1_0_2*tangent37 + pa_p2_0_0*tangent40 + pa_p2_0_1*tangent42 + pa_p2_0_2*tangent43;
      const s_t pa_y2_0_2 = pa_p0_0_0*tangent8 + pa_p0_0_1*tangent16 + pa_p0_0_2*tangent23 + pa_p1_0_0*tangent29 + pa_p1_0_1*tangent34 + pa_p1_0_2*tangent38 + pa_p2_0_0*tangent41 + pa_p2_0_1*tangent43 + pa_p2_0_2*tangent44;
      const s_t pa_q0_0_0 = s_t(6)*pa_y0_0_0;
      const s_t pa_q0_0_1 = s_t(6)*pa_y0_0_1;
      const s_t pa_q0_0_2 = s_t(6)*pa_y0_0_2;
      const s_t pa_q1_0_0 = s_t(6)*pa_y1_0_0;
      const s_t pa_q1_0_1 = s_t(6)*pa_y1_0_1;
      const s_t pa_q1_0_2 = s_t(6)*pa_y1_0_2;
      const s_t pa_q2_0_0 = s_t(6)*pa_y2_0_0;
      const s_t pa_q2_0_1 = s_t(6)*pa_y2_0_1;
      const s_t pa_q2_0_2 = s_t(6)*pa_y2_0_2;
      const s_t output_t0 = ((s_t(1) / s_t(6)))*pa_q0_0_0;
      const s_t output_t1 = ((s_t(1) / s_t(6)))*pa_q0_0_1;
      const s_t output_t2 = ((s_t(1) / s_t(6)))*pa_q0_0_2;
      const s_t output_t3 = ((s_t(1) / s_t(6)))*pa_q1_0_0;
      const s_t output_t4 = ((s_t(1) / s_t(6)))*pa_q1_0_1;
      const s_t output_t5 = ((s_t(1) / s_t(6)))*pa_q1_0_2;
      const s_t output_t6 = ((s_t(1) / s_t(6)))*pa_q2_0_0;
      const s_t output_t7 = ((s_t(1) / s_t(6)))*pa_q2_0_1;
      const s_t output_t8 = ((s_t(1) / s_t(6)))*pa_q2_0_2;
      const s_t element_out0_0 = output_t0 + output_t1 + output_t2;
      const s_t element_out0_1 = -output_t0;
      const s_t element_out0_2 = -output_t1;
      const s_t element_out0_3 = -output_t2;
      const s_t element_out1_0 = output_t3 + output_t4 + output_t5;
      const s_t element_out1_1 = -output_t3;
      const s_t element_out1_2 = -output_t4;
      const s_t element_out1_3 = -output_t5;
      const s_t element_out2_0 = output_t6 + output_t7 + output_t8;
      const s_t element_out2_1 = -output_t6;
      const s_t element_out2_2 = -output_t7;
      const s_t element_out2_3 = -output_t8;
      bout0_0[lane] = element_out0_0;
      bout0_1[lane] = element_out0_1;
      bout0_2[lane] = element_out0_2;
      bout0_3[lane] = element_out0_3;
      bout1_0[lane] = element_out1_0;
      bout1_1[lane] = element_out1_1;
      bout1_2[lane] = element_out1_2;
      bout1_3[lane] = element_out1_3;
      bout2_0[lane] = element_out2_0;
      bout2_1[lane] = element_out2_1;
      bout2_2[lane] = element_out2_2;
      bout2_3[lane] = element_out2_3;
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
    for (int lane = 0; lane < ne; ++lane) {
      #pragma omp atomic update
      outz[bev0[lane] * out_stride] += bout2_0[lane];
    }
    for (int lane = 0; lane < ne; ++lane) {
      #pragma omp atomic update
      outz[bev1[lane] * out_stride] += bout2_1[lane];
    }
    for (int lane = 0; lane < ne; ++lane) {
      #pragma omp atomic update
      outz[bev2[lane] * out_stride] += bout2_2[lane];
    }
    for (int lane = 0; lane < ne; ++lane) {
      #pragma omp atomic update
      outz[bev3[lane] * out_stride] += bout2_3[lane];
    }
  }

  return SFEM_SUCCESS;
}

template <typename s_t, typename tangent_t, int VS>
static SFEM_INLINE int linear_elasticity_tet4_inexact_apply_stored_packed_two_pass_a_msoa_impl(
    const ptrdiff_t n_packs,
    const ptrdiff_t n_elements_per_pack,
    const ptrdiff_t nelements,
    const ptrdiff_t max_nodes_per_pack,
    uint16_t **const RSTR elements,
    const ptrdiff_t *const RSTR owned_nodes_ptr,
    const ptrdiff_t n_ghost_entries,
    const ptrdiff_t n_ghost_reduce_rows,
    const ptrdiff_t *const RSTR ghost_ptr,
    const idx_t *const RSTR ghost_idx,
    const ptrdiff_t *const RSTR ghost_reduce_ptr,
    const ptrdiff_t *const RSTR ghost_reduce_idx,
    const idx_t *const RSTR ghost_reduce_dest,
    s_t *const RSTR ghost_buf,
    const ptrdiff_t tangent_component_stride,
    const tangent_t *const RSTR tangent,
    const ptrdiff_t h_stride,
    const s_t *const RSTR hx,
    const s_t *const RSTR hy,
    const s_t *const RSTR hz,
    const ptrdiff_t out_stride,
    s_t *const RSTR outx,
    s_t *const RSTR outy,
    s_t *const RSTR outz
) {
  static constexpr int NC = 3;
  const s_t *const h_components[NC] = {hx, hy, hz};
  s_t *const out_components[NC] = {outx, outy, outz};

  #pragma omp parallel
  {
    s_t *const RSTR pk_h = sfem::codegen::thread_scratch<s_t>(2, (size_t)NC * (size_t)max_nodes_per_pack);
    s_t *const RSTR pk_out = sfem::codegen::thread_scratch<s_t>(3, (size_t)NC * (size_t)max_nodes_per_pack);
    #pragma omp for schedule(static)
    for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
      const ptrdiff_t e_start = pack * n_elements_per_pack;
      const ptrdiff_t e_end = (nelements < (pack + 1) * n_elements_per_pack)
                                  ? nelements
                                  : (pack + 1) * n_elements_per_pack;
      const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
      const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
      const ptrdiff_t ghost_off = ghost_ptr[pack];
      const idx_t *const RSTR ghosts = &ghost_idx[ghost_off];

      for (int d = 0; d < NC; ++d) {
        s_t *const RSTR pk_h_component = pk_h + d * max_nodes_per_pack;
        s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
        const s_t *const RSTR h_component = h_components[d];
        for (ptrdiff_t k = 0; k < n_contiguous + n_ghost; ++k) {
          pk_component_out[k] = s_t(0);
        }
        for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
          pk_h_component[k] = h_component[(owned_nodes_ptr[pack] + k) * h_stride];
        }
        for (ptrdiff_t k = 0; k < n_ghost; ++k) {
          pk_h_component[n_contiguous + k] = h_component[ghosts[k] * h_stride];
        }
      }

      for (ptrdiff_t evb = e_start; evb < e_end; evb += VS) {
        const int ne = (int)((e_end - evb) < (ptrdiff_t)VS ? (e_end - evb) : (ptrdiff_t)VS);
        uint16_t bev0[VS];
        uint16_t bev1[VS];
        uint16_t bev2[VS];
        uint16_t bev3[VS];
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
        s_t bhz_0[VS];
        s_t bhz_1[VS];
        s_t bhz_2[VS];
        s_t bhz_3[VS];
        s_t bout0_0[VS];
        s_t bout0_1[VS];
        s_t bout0_2[VS];
        s_t bout0_3[VS];
        s_t bout1_0[VS];
        s_t bout1_1[VS];
        s_t bout1_2[VS];
        s_t bout1_3[VS];
        s_t bout2_0[VS];
        s_t bout2_1[VS];
        s_t bout2_2[VS];
        s_t bout2_3[VS];
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          bhx_0[lane] = pk_h[0 * max_nodes_per_pack + bev0[lane]];
          bhx_1[lane] = pk_h[0 * max_nodes_per_pack + bev1[lane]];
          bhx_2[lane] = pk_h[0 * max_nodes_per_pack + bev2[lane]];
          bhx_3[lane] = pk_h[0 * max_nodes_per_pack + bev3[lane]];
          bhy_0[lane] = pk_h[1 * max_nodes_per_pack + bev0[lane]];
          bhy_1[lane] = pk_h[1 * max_nodes_per_pack + bev1[lane]];
          bhy_2[lane] = pk_h[1 * max_nodes_per_pack + bev2[lane]];
          bhy_3[lane] = pk_h[1 * max_nodes_per_pack + bev3[lane]];
          bhz_0[lane] = pk_h[2 * max_nodes_per_pack + bev0[lane]];
          bhz_1[lane] = pk_h[2 * max_nodes_per_pack + bev1[lane]];
          bhz_2[lane] = pk_h[2 * max_nodes_per_pack + bev2[lane]];
          bhz_3[lane] = pk_h[2 * max_nodes_per_pack + bev3[lane]];
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
        const tangent_t *const RSTR btangent10 = tangent + evb + 10 * tangent_component_stride;
        const tangent_t *const RSTR btangent11 = tangent + evb + 11 * tangent_component_stride;
        const tangent_t *const RSTR btangent12 = tangent + evb + 12 * tangent_component_stride;
        const tangent_t *const RSTR btangent13 = tangent + evb + 13 * tangent_component_stride;
        const tangent_t *const RSTR btangent14 = tangent + evb + 14 * tangent_component_stride;
        const tangent_t *const RSTR btangent15 = tangent + evb + 15 * tangent_component_stride;
        const tangent_t *const RSTR btangent16 = tangent + evb + 16 * tangent_component_stride;
        const tangent_t *const RSTR btangent17 = tangent + evb + 17 * tangent_component_stride;
        const tangent_t *const RSTR btangent18 = tangent + evb + 18 * tangent_component_stride;
        const tangent_t *const RSTR btangent19 = tangent + evb + 19 * tangent_component_stride;
        const tangent_t *const RSTR btangent20 = tangent + evb + 20 * tangent_component_stride;
        const tangent_t *const RSTR btangent21 = tangent + evb + 21 * tangent_component_stride;
        const tangent_t *const RSTR btangent22 = tangent + evb + 22 * tangent_component_stride;
        const tangent_t *const RSTR btangent23 = tangent + evb + 23 * tangent_component_stride;
        const tangent_t *const RSTR btangent24 = tangent + evb + 24 * tangent_component_stride;
        const tangent_t *const RSTR btangent25 = tangent + evb + 25 * tangent_component_stride;
        const tangent_t *const RSTR btangent26 = tangent + evb + 26 * tangent_component_stride;
        const tangent_t *const RSTR btangent27 = tangent + evb + 27 * tangent_component_stride;
        const tangent_t *const RSTR btangent28 = tangent + evb + 28 * tangent_component_stride;
        const tangent_t *const RSTR btangent29 = tangent + evb + 29 * tangent_component_stride;
        const tangent_t *const RSTR btangent30 = tangent + evb + 30 * tangent_component_stride;
        const tangent_t *const RSTR btangent31 = tangent + evb + 31 * tangent_component_stride;
        const tangent_t *const RSTR btangent32 = tangent + evb + 32 * tangent_component_stride;
        const tangent_t *const RSTR btangent33 = tangent + evb + 33 * tangent_component_stride;
        const tangent_t *const RSTR btangent34 = tangent + evb + 34 * tangent_component_stride;
        const tangent_t *const RSTR btangent35 = tangent + evb + 35 * tangent_component_stride;
        const tangent_t *const RSTR btangent36 = tangent + evb + 36 * tangent_component_stride;
        const tangent_t *const RSTR btangent37 = tangent + evb + 37 * tangent_component_stride;
        const tangent_t *const RSTR btangent38 = tangent + evb + 38 * tangent_component_stride;
        const tangent_t *const RSTR btangent39 = tangent + evb + 39 * tangent_component_stride;
        const tangent_t *const RSTR btangent40 = tangent + evb + 40 * tangent_component_stride;
        const tangent_t *const RSTR btangent41 = tangent + evb + 41 * tangent_component_stride;
        const tangent_t *const RSTR btangent42 = tangent + evb + 42 * tangent_component_stride;
        const tangent_t *const RSTR btangent43 = tangent + evb + 43 * tangent_component_stride;
        const tangent_t *const RSTR btangent44 = tangent + evb + 44 * tangent_component_stride;
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
          const s_t hz_0 = bhz_0[lane];
          const s_t hz_1 = bhz_1[lane];
          const s_t hz_2 = bhz_2[lane];
          const s_t hz_3 = bhz_3[lane];
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
          const s_t tangent10 = s_t(btangent10[lane]);
          const s_t tangent11 = s_t(btangent11[lane]);
          const s_t tangent12 = s_t(btangent12[lane]);
          const s_t tangent13 = s_t(btangent13[lane]);
          const s_t tangent14 = s_t(btangent14[lane]);
          const s_t tangent15 = s_t(btangent15[lane]);
          const s_t tangent16 = s_t(btangent16[lane]);
          const s_t tangent17 = s_t(btangent17[lane]);
          const s_t tangent18 = s_t(btangent18[lane]);
          const s_t tangent19 = s_t(btangent19[lane]);
          const s_t tangent20 = s_t(btangent20[lane]);
          const s_t tangent21 = s_t(btangent21[lane]);
          const s_t tangent22 = s_t(btangent22[lane]);
          const s_t tangent23 = s_t(btangent23[lane]);
          const s_t tangent24 = s_t(btangent24[lane]);
          const s_t tangent25 = s_t(btangent25[lane]);
          const s_t tangent26 = s_t(btangent26[lane]);
          const s_t tangent27 = s_t(btangent27[lane]);
          const s_t tangent28 = s_t(btangent28[lane]);
          const s_t tangent29 = s_t(btangent29[lane]);
          const s_t tangent30 = s_t(btangent30[lane]);
          const s_t tangent31 = s_t(btangent31[lane]);
          const s_t tangent32 = s_t(btangent32[lane]);
          const s_t tangent33 = s_t(btangent33[lane]);
          const s_t tangent34 = s_t(btangent34[lane]);
          const s_t tangent35 = s_t(btangent35[lane]);
          const s_t tangent36 = s_t(btangent36[lane]);
          const s_t tangent37 = s_t(btangent37[lane]);
          const s_t tangent38 = s_t(btangent38[lane]);
          const s_t tangent39 = s_t(btangent39[lane]);
          const s_t tangent40 = s_t(btangent40[lane]);
          const s_t tangent41 = s_t(btangent41[lane]);
          const s_t tangent42 = s_t(btangent42[lane]);
          const s_t tangent43 = s_t(btangent43[lane]);
          const s_t tangent44 = s_t(btangent44[lane]);
          const s_t compressed_increment_t0 = ((s_t(1) / s_t(6)))*hx_0;
          const s_t compressed_increment_t1 = ((s_t(1) / s_t(6)))*hy_0;
          const s_t compressed_increment_t2 = ((s_t(1) / s_t(6)))*hz_0;
          const s_t pa_p0_0_0 = compressed_increment_t0 - (s_t(1) / s_t(6))*hx_1;
          const s_t pa_p0_0_1 = compressed_increment_t0 - (s_t(1) / s_t(6))*hx_2;
          const s_t pa_p0_0_2 = compressed_increment_t0 - (s_t(1) / s_t(6))*hx_3;
          const s_t pa_p1_0_0 = compressed_increment_t1 - (s_t(1) / s_t(6))*hy_1;
          const s_t pa_p1_0_1 = compressed_increment_t1 - (s_t(1) / s_t(6))*hy_2;
          const s_t pa_p1_0_2 = compressed_increment_t1 - (s_t(1) / s_t(6))*hy_3;
          const s_t pa_p2_0_0 = compressed_increment_t2 - (s_t(1) / s_t(6))*hz_1;
          const s_t pa_p2_0_1 = compressed_increment_t2 - (s_t(1) / s_t(6))*hz_2;
          const s_t pa_p2_0_2 = compressed_increment_t2 - (s_t(1) / s_t(6))*hz_3;
          const s_t pa_y0_0_0 = pa_p0_0_0*tangent0 + pa_p0_0_1*tangent1 + pa_p0_0_2*tangent2 + pa_p1_0_0*tangent3 + pa_p1_0_1*tangent4 + pa_p1_0_2*tangent5 + pa_p2_0_0*tangent6 + pa_p2_0_1*tangent7 + pa_p2_0_2*tangent8;
          const s_t pa_y0_0_1 = pa_p0_0_0*tangent1 + pa_p0_0_1*tangent9 + pa_p0_0_2*tangent10 + pa_p1_0_0*tangent11 + pa_p1_0_1*tangent12 + pa_p1_0_2*tangent13 + pa_p2_0_0*tangent14 + pa_p2_0_1*tangent15 + pa_p2_0_2*tangent16;
          const s_t pa_y0_0_2 = pa_p0_0_0*tangent2 + pa_p0_0_1*tangent10 + pa_p0_0_2*tangent17 + pa_p1_0_0*tangent18 + pa_p1_0_1*tangent19 + pa_p1_0_2*tangent20 + pa_p2_0_0*tangent21 + pa_p2_0_1*tangent22 + pa_p2_0_2*tangent23;
          const s_t pa_y1_0_0 = pa_p0_0_0*tangent3 + pa_p0_0_1*tangent11 + pa_p0_0_2*tangent18 + pa_p1_0_0*tangent24 + pa_p1_0_1*tangent25 + pa_p1_0_2*tangent26 + pa_p2_0_0*tangent27 + pa_p2_0_1*tangent28 + pa_p2_0_2*tangent29;
          const s_t pa_y1_0_1 = pa_p0_0_0*tangent4 + pa_p0_0_1*tangent12 + pa_p0_0_2*tangent19 + pa_p1_0_0*tangent25 + pa_p1_0_1*tangent30 + pa_p1_0_2*tangent31 + pa_p2_0_0*tangent32 + pa_p2_0_1*tangent33 + pa_p2_0_2*tangent34;
          const s_t pa_y1_0_2 = pa_p0_0_0*tangent5 + pa_p0_0_1*tangent13 + pa_p0_0_2*tangent20 + pa_p1_0_0*tangent26 + pa_p1_0_1*tangent31 + pa_p1_0_2*tangent35 + pa_p2_0_0*tangent36 + pa_p2_0_1*tangent37 + pa_p2_0_2*tangent38;
          const s_t pa_y2_0_0 = pa_p0_0_0*tangent6 + pa_p0_0_1*tangent14 + pa_p0_0_2*tangent21 + pa_p1_0_0*tangent27 + pa_p1_0_1*tangent32 + pa_p1_0_2*tangent36 + pa_p2_0_0*tangent39 + pa_p2_0_1*tangent40 + pa_p2_0_2*tangent41;
          const s_t pa_y2_0_1 = pa_p0_0_0*tangent7 + pa_p0_0_1*tangent15 + pa_p0_0_2*tangent22 + pa_p1_0_0*tangent28 + pa_p1_0_1*tangent33 + pa_p1_0_2*tangent37 + pa_p2_0_0*tangent40 + pa_p2_0_1*tangent42 + pa_p2_0_2*tangent43;
          const s_t pa_y2_0_2 = pa_p0_0_0*tangent8 + pa_p0_0_1*tangent16 + pa_p0_0_2*tangent23 + pa_p1_0_0*tangent29 + pa_p1_0_1*tangent34 + pa_p1_0_2*tangent38 + pa_p2_0_0*tangent41 + pa_p2_0_1*tangent43 + pa_p2_0_2*tangent44;
          const s_t pa_q0_0_0 = s_t(6)*pa_y0_0_0;
          const s_t pa_q0_0_1 = s_t(6)*pa_y0_0_1;
          const s_t pa_q0_0_2 = s_t(6)*pa_y0_0_2;
          const s_t pa_q1_0_0 = s_t(6)*pa_y1_0_0;
          const s_t pa_q1_0_1 = s_t(6)*pa_y1_0_1;
          const s_t pa_q1_0_2 = s_t(6)*pa_y1_0_2;
          const s_t pa_q2_0_0 = s_t(6)*pa_y2_0_0;
          const s_t pa_q2_0_1 = s_t(6)*pa_y2_0_1;
          const s_t pa_q2_0_2 = s_t(6)*pa_y2_0_2;
          const s_t output_t0 = ((s_t(1) / s_t(6)))*pa_q0_0_0;
          const s_t output_t1 = ((s_t(1) / s_t(6)))*pa_q0_0_1;
          const s_t output_t2 = ((s_t(1) / s_t(6)))*pa_q0_0_2;
          const s_t output_t3 = ((s_t(1) / s_t(6)))*pa_q1_0_0;
          const s_t output_t4 = ((s_t(1) / s_t(6)))*pa_q1_0_1;
          const s_t output_t5 = ((s_t(1) / s_t(6)))*pa_q1_0_2;
          const s_t output_t6 = ((s_t(1) / s_t(6)))*pa_q2_0_0;
          const s_t output_t7 = ((s_t(1) / s_t(6)))*pa_q2_0_1;
          const s_t output_t8 = ((s_t(1) / s_t(6)))*pa_q2_0_2;
          const s_t element_out0_0 = output_t0 + output_t1 + output_t2;
          const s_t element_out0_1 = -output_t0;
          const s_t element_out0_2 = -output_t1;
          const s_t element_out0_3 = -output_t2;
          const s_t element_out1_0 = output_t3 + output_t4 + output_t5;
          const s_t element_out1_1 = -output_t3;
          const s_t element_out1_2 = -output_t4;
          const s_t element_out1_3 = -output_t5;
          const s_t element_out2_0 = output_t6 + output_t7 + output_t8;
          const s_t element_out2_1 = -output_t6;
          const s_t element_out2_2 = -output_t7;
          const s_t element_out2_3 = -output_t8;
          bout0_0[lane] = element_out0_0;
          bout0_1[lane] = element_out0_1;
          bout0_2[lane] = element_out0_2;
          bout0_3[lane] = element_out0_3;
          bout1_0[lane] = element_out1_0;
          bout1_1[lane] = element_out1_1;
          bout1_2[lane] = element_out1_2;
          bout1_3[lane] = element_out1_3;
          bout2_0[lane] = element_out2_0;
          bout2_1[lane] = element_out2_1;
          bout2_2[lane] = element_out2_2;
          bout2_3[lane] = element_out2_3;
        }
        for (int lane = 0; lane < ne; ++lane) {
          pk_out[0 * max_nodes_per_pack + bev0[lane]] += bout0_0[lane];
        }
        for (int lane = 0; lane < ne; ++lane) {
          pk_out[0 * max_nodes_per_pack + bev1[lane]] += bout0_1[lane];
        }
        for (int lane = 0; lane < ne; ++lane) {
          pk_out[0 * max_nodes_per_pack + bev2[lane]] += bout0_2[lane];
        }
        for (int lane = 0; lane < ne; ++lane) {
          pk_out[0 * max_nodes_per_pack + bev3[lane]] += bout0_3[lane];
        }
        for (int lane = 0; lane < ne; ++lane) {
          pk_out[1 * max_nodes_per_pack + bev0[lane]] += bout1_0[lane];
        }
        for (int lane = 0; lane < ne; ++lane) {
          pk_out[1 * max_nodes_per_pack + bev1[lane]] += bout1_1[lane];
        }
        for (int lane = 0; lane < ne; ++lane) {
          pk_out[1 * max_nodes_per_pack + bev2[lane]] += bout1_2[lane];
        }
        for (int lane = 0; lane < ne; ++lane) {
          pk_out[1 * max_nodes_per_pack + bev3[lane]] += bout1_3[lane];
        }
        for (int lane = 0; lane < ne; ++lane) {
          pk_out[2 * max_nodes_per_pack + bev0[lane]] += bout2_0[lane];
        }
        for (int lane = 0; lane < ne; ++lane) {
          pk_out[2 * max_nodes_per_pack + bev1[lane]] += bout2_1[lane];
        }
        for (int lane = 0; lane < ne; ++lane) {
          pk_out[2 * max_nodes_per_pack + bev2[lane]] += bout2_2[lane];
        }
        for (int lane = 0; lane < ne; ++lane) {
          pk_out[2 * max_nodes_per_pack + bev3[lane]] += bout2_3[lane];
        }
      }

      for (int d = 0; d < NC; ++d) {
        s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
        s_t *const RSTR global_out = out_components[d];
        s_t *const RSTR ghost_component = ghost_buf + d * n_ghost_entries;
        for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
          global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pk_component_out[k];
        }
        for (ptrdiff_t k = 0; k < n_ghost; ++k) {
          ghost_component[ghost_off + k] = pk_component_out[n_contiguous + k];
        }
      }
    }
  }

  #pragma omp parallel for schedule(static)
  for (ptrdiff_t row = 0; row < n_ghost_reduce_rows; ++row) {
    const idx_t dest = ghost_reduce_dest[row];
    const ptrdiff_t begin = ghost_reduce_ptr[row];
    const ptrdiff_t end = ghost_reduce_ptr[row + 1];
    for (int d = 0; d < NC; ++d) {
      const s_t *const RSTR ghost_component = ghost_buf + d * n_ghost_entries;
      s_t sum = s_t(0);
      for (ptrdiff_t j = begin; j < end; ++j) {
        sum += ghost_component[ghost_reduce_idx[j]];
      }
      out_components[d][dest * out_stride] += sum;
    }
  }
  return SFEM_SUCCESS;
}

template <typename s_t, typename tangent_t, typename scale_t>
static SFEM_INLINE int linear_elasticity_tet4_inexact_apply_compressed_a_msoa_impl(
    const ptrdiff_t nelements,
    idx_t **const RSTR elements,
    const ptrdiff_t tangent_component_stride,
    const tangent_t *const RSTR tangent,
    const scale_t *const RSTR scaling,
    const ptrdiff_t h_stride,
    const s_t *const RSTR hx,
    const s_t *const RSTR hy,
    const s_t *const RSTR hz,
    const ptrdiff_t out_stride,
    s_t *const RSTR outx,
    s_t *const RSTR outy,
    s_t *const RSTR outz
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
    const s_t hz_0 = hz[ev0 * h_stride];
    const s_t hz_1 = hz[ev1 * h_stride];
    const s_t hz_2 = hz[ev2 * h_stride];
    const s_t hz_3 = hz[ev3 * h_stride];
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
    const s_t tangent10 = s_t(tangent[element + 10 * tangent_component_stride]);
    const s_t tangent11 = s_t(tangent[element + 11 * tangent_component_stride]);
    const s_t tangent12 = s_t(tangent[element + 12 * tangent_component_stride]);
    const s_t tangent13 = s_t(tangent[element + 13 * tangent_component_stride]);
    const s_t tangent14 = s_t(tangent[element + 14 * tangent_component_stride]);
    const s_t tangent15 = s_t(tangent[element + 15 * tangent_component_stride]);
    const s_t tangent16 = s_t(tangent[element + 16 * tangent_component_stride]);
    const s_t tangent17 = s_t(tangent[element + 17 * tangent_component_stride]);
    const s_t tangent18 = s_t(tangent[element + 18 * tangent_component_stride]);
    const s_t tangent19 = s_t(tangent[element + 19 * tangent_component_stride]);
    const s_t tangent20 = s_t(tangent[element + 20 * tangent_component_stride]);
    const s_t tangent21 = s_t(tangent[element + 21 * tangent_component_stride]);
    const s_t tangent22 = s_t(tangent[element + 22 * tangent_component_stride]);
    const s_t tangent23 = s_t(tangent[element + 23 * tangent_component_stride]);
    const s_t tangent24 = s_t(tangent[element + 24 * tangent_component_stride]);
    const s_t tangent25 = s_t(tangent[element + 25 * tangent_component_stride]);
    const s_t tangent26 = s_t(tangent[element + 26 * tangent_component_stride]);
    const s_t tangent27 = s_t(tangent[element + 27 * tangent_component_stride]);
    const s_t tangent28 = s_t(tangent[element + 28 * tangent_component_stride]);
    const s_t tangent29 = s_t(tangent[element + 29 * tangent_component_stride]);
    const s_t tangent30 = s_t(tangent[element + 30 * tangent_component_stride]);
    const s_t tangent31 = s_t(tangent[element + 31 * tangent_component_stride]);
    const s_t tangent32 = s_t(tangent[element + 32 * tangent_component_stride]);
    const s_t tangent33 = s_t(tangent[element + 33 * tangent_component_stride]);
    const s_t tangent34 = s_t(tangent[element + 34 * tangent_component_stride]);
    const s_t tangent35 = s_t(tangent[element + 35 * tangent_component_stride]);
    const s_t tangent36 = s_t(tangent[element + 36 * tangent_component_stride]);
    const s_t tangent37 = s_t(tangent[element + 37 * tangent_component_stride]);
    const s_t tangent38 = s_t(tangent[element + 38 * tangent_component_stride]);
    const s_t tangent39 = s_t(tangent[element + 39 * tangent_component_stride]);
    const s_t tangent40 = s_t(tangent[element + 40 * tangent_component_stride]);
    const s_t tangent41 = s_t(tangent[element + 41 * tangent_component_stride]);
    const s_t tangent42 = s_t(tangent[element + 42 * tangent_component_stride]);
    const s_t tangent43 = s_t(tangent[element + 43 * tangent_component_stride]);
    const s_t tangent44 = s_t(tangent[element + 44 * tangent_component_stride]);
    const s_t compressed_increment_t0 = ((s_t(1) / s_t(6)))*hx_0;
    const s_t compressed_increment_t1 = ((s_t(1) / s_t(6)))*hy_0;
    const s_t compressed_increment_t2 = ((s_t(1) / s_t(6)))*hz_0;
    const s_t pa_p0_0_0 = compressed_increment_t0 - (s_t(1) / s_t(6))*hx_1;
    const s_t pa_p0_0_1 = compressed_increment_t0 - (s_t(1) / s_t(6))*hx_2;
    const s_t pa_p0_0_2 = compressed_increment_t0 - (s_t(1) / s_t(6))*hx_3;
    const s_t pa_p1_0_0 = compressed_increment_t1 - (s_t(1) / s_t(6))*hy_1;
    const s_t pa_p1_0_1 = compressed_increment_t1 - (s_t(1) / s_t(6))*hy_2;
    const s_t pa_p1_0_2 = compressed_increment_t1 - (s_t(1) / s_t(6))*hy_3;
    const s_t pa_p2_0_0 = compressed_increment_t2 - (s_t(1) / s_t(6))*hz_1;
    const s_t pa_p2_0_1 = compressed_increment_t2 - (s_t(1) / s_t(6))*hz_2;
    const s_t pa_p2_0_2 = compressed_increment_t2 - (s_t(1) / s_t(6))*hz_3;
    const s_t pa_y0_0_0 = pa_p0_0_0*tangent0 + pa_p0_0_1*tangent1 + pa_p0_0_2*tangent2 + pa_p1_0_0*tangent3 + pa_p1_0_1*tangent4 + pa_p1_0_2*tangent5 + pa_p2_0_0*tangent6 + pa_p2_0_1*tangent7 + pa_p2_0_2*tangent8;
    const s_t pa_y0_0_1 = pa_p0_0_0*tangent1 + pa_p0_0_1*tangent9 + pa_p0_0_2*tangent10 + pa_p1_0_0*tangent11 + pa_p1_0_1*tangent12 + pa_p1_0_2*tangent13 + pa_p2_0_0*tangent14 + pa_p2_0_1*tangent15 + pa_p2_0_2*tangent16;
    const s_t pa_y0_0_2 = pa_p0_0_0*tangent2 + pa_p0_0_1*tangent10 + pa_p0_0_2*tangent17 + pa_p1_0_0*tangent18 + pa_p1_0_1*tangent19 + pa_p1_0_2*tangent20 + pa_p2_0_0*tangent21 + pa_p2_0_1*tangent22 + pa_p2_0_2*tangent23;
    const s_t pa_y1_0_0 = pa_p0_0_0*tangent3 + pa_p0_0_1*tangent11 + pa_p0_0_2*tangent18 + pa_p1_0_0*tangent24 + pa_p1_0_1*tangent25 + pa_p1_0_2*tangent26 + pa_p2_0_0*tangent27 + pa_p2_0_1*tangent28 + pa_p2_0_2*tangent29;
    const s_t pa_y1_0_1 = pa_p0_0_0*tangent4 + pa_p0_0_1*tangent12 + pa_p0_0_2*tangent19 + pa_p1_0_0*tangent25 + pa_p1_0_1*tangent30 + pa_p1_0_2*tangent31 + pa_p2_0_0*tangent32 + pa_p2_0_1*tangent33 + pa_p2_0_2*tangent34;
    const s_t pa_y1_0_2 = pa_p0_0_0*tangent5 + pa_p0_0_1*tangent13 + pa_p0_0_2*tangent20 + pa_p1_0_0*tangent26 + pa_p1_0_1*tangent31 + pa_p1_0_2*tangent35 + pa_p2_0_0*tangent36 + pa_p2_0_1*tangent37 + pa_p2_0_2*tangent38;
    const s_t pa_y2_0_0 = pa_p0_0_0*tangent6 + pa_p0_0_1*tangent14 + pa_p0_0_2*tangent21 + pa_p1_0_0*tangent27 + pa_p1_0_1*tangent32 + pa_p1_0_2*tangent36 + pa_p2_0_0*tangent39 + pa_p2_0_1*tangent40 + pa_p2_0_2*tangent41;
    const s_t pa_y2_0_1 = pa_p0_0_0*tangent7 + pa_p0_0_1*tangent15 + pa_p0_0_2*tangent22 + pa_p1_0_0*tangent28 + pa_p1_0_1*tangent33 + pa_p1_0_2*tangent37 + pa_p2_0_0*tangent40 + pa_p2_0_1*tangent42 + pa_p2_0_2*tangent43;
    const s_t pa_y2_0_2 = pa_p0_0_0*tangent8 + pa_p0_0_1*tangent16 + pa_p0_0_2*tangent23 + pa_p1_0_0*tangent29 + pa_p1_0_1*tangent34 + pa_p1_0_2*tangent38 + pa_p2_0_0*tangent41 + pa_p2_0_1*tangent43 + pa_p2_0_2*tangent44;
    const s_t pa_q0_0_0 = s_t(6)*pa_y0_0_0;
    const s_t pa_q0_0_1 = s_t(6)*pa_y0_0_1;
    const s_t pa_q0_0_2 = s_t(6)*pa_y0_0_2;
    const s_t pa_q1_0_0 = s_t(6)*pa_y1_0_0;
    const s_t pa_q1_0_1 = s_t(6)*pa_y1_0_1;
    const s_t pa_q1_0_2 = s_t(6)*pa_y1_0_2;
    const s_t pa_q2_0_0 = s_t(6)*pa_y2_0_0;
    const s_t pa_q2_0_1 = s_t(6)*pa_y2_0_1;
    const s_t pa_q2_0_2 = s_t(6)*pa_y2_0_2;
    const s_t output_t0 = ((s_t(1) / s_t(6)))*pa_q0_0_0;
    const s_t output_t1 = ((s_t(1) / s_t(6)))*pa_q0_0_1;
    const s_t output_t2 = ((s_t(1) / s_t(6)))*pa_q0_0_2;
    const s_t output_t3 = ((s_t(1) / s_t(6)))*pa_q1_0_0;
    const s_t output_t4 = ((s_t(1) / s_t(6)))*pa_q1_0_1;
    const s_t output_t5 = ((s_t(1) / s_t(6)))*pa_q1_0_2;
    const s_t output_t6 = ((s_t(1) / s_t(6)))*pa_q2_0_0;
    const s_t output_t7 = ((s_t(1) / s_t(6)))*pa_q2_0_1;
    const s_t output_t8 = ((s_t(1) / s_t(6)))*pa_q2_0_2;
    const s_t element_out0_0 = output_t0 + output_t1 + output_t2;
    const s_t element_out0_1 = -output_t0;
    const s_t element_out0_2 = -output_t1;
    const s_t element_out0_3 = -output_t2;
    const s_t element_out1_0 = output_t3 + output_t4 + output_t5;
    const s_t element_out1_1 = -output_t3;
    const s_t element_out1_2 = -output_t4;
    const s_t element_out1_3 = -output_t5;
    const s_t element_out2_0 = output_t6 + output_t7 + output_t8;
    const s_t element_out2_1 = -output_t6;
    const s_t element_out2_2 = -output_t7;
    const s_t element_out2_3 = -output_t8;
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
    #pragma omp atomic update
    outz[ev0 * out_stride] += scale * element_out2_0;
    #pragma omp atomic update
    outz[ev1 * out_stride] += scale * element_out2_1;
    #pragma omp atomic update
    outz[ev2 * out_stride] += scale * element_out2_2;
    #pragma omp atomic update
    outz[ev3 * out_stride] += scale * element_out2_3;
  }

  return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem
