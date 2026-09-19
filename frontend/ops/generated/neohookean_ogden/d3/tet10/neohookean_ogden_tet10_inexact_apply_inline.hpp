#pragma once
#include "../../../kernel_math.hpp"
#include "../../../packed_thread_scratch.hpp"
#include "../../../reference/quad_tet_q11.hpp"
#include "../../../reference/tet10_q11.hpp"

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t, typename tangent_t, int VS>
static SFEM_INLINE int neohookean_ogden_tet10_inexact_apply_tangent_a_msoa_impl(
    const ptrdiff_t nelements,
    idx_t **const RSTR elements,
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
    const ptrdiff_t u_stride,
    const s_t *const RSTR ux,
    const s_t *const RSTR uy,
    const s_t *const RSTR uz,
    const ptrdiff_t tangent_component_stride,
    tangent_t *const RSTR tangent
) {
  #pragma omp parallel for schedule(static)
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)((nelements - evb) < (ptrdiff_t)VS ? (nelements - evb) : (ptrdiff_t)VS);
    static constexpr int NQ = 11;
    const s_t *const RSTR qgrad_x = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_x();
    const s_t *const RSTR qgrad_y = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_y();
    const s_t *const RSTR qgrad_z = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_z();
    const s_t *const RSTR qweight = sfem::codegen::quad_tet_q11<s_t>::q_weight();
    static constexpr s_t QMEASURE = s_t(6.0000000000000018);
    s_t btangent_acc[45][VS];
    idx_t bev0[VS];
    idx_t bev1[VS];
    idx_t bev2[VS];
    idx_t bev3[VS];
    idx_t bev4[VS];
    idx_t bev5[VS];
    idx_t bev6[VS];
    idx_t bev7[VS];
    idx_t bev8[VS];
    idx_t bev9[VS];
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      bev0[lane] = elements[0][evb + lane];
      bev1[lane] = elements[1][evb + lane];
      bev2[lane] = elements[2][evb + lane];
      bev3[lane] = elements[3][evb + lane];
      bev4[lane] = elements[4][evb + lane];
      bev5[lane] = elements[5][evb + lane];
      bev6[lane] = elements[6][evb + lane];
      bev7[lane] = elements[7][evb + lane];
      bev8[lane] = elements[8][evb + lane];
      bev9[lane] = elements[9][evb + lane];
    }
    s_t bux_0[VS];
    s_t bux_1[VS];
    s_t bux_2[VS];
    s_t bux_3[VS];
    s_t bux_4[VS];
    s_t bux_5[VS];
    s_t bux_6[VS];
    s_t bux_7[VS];
    s_t bux_8[VS];
    s_t bux_9[VS];
    s_t buy_0[VS];
    s_t buy_1[VS];
    s_t buy_2[VS];
    s_t buy_3[VS];
    s_t buy_4[VS];
    s_t buy_5[VS];
    s_t buy_6[VS];
    s_t buy_7[VS];
    s_t buy_8[VS];
    s_t buy_9[VS];
    s_t buz_0[VS];
    s_t buz_1[VS];
    s_t buz_2[VS];
    s_t buz_3[VS];
    s_t buz_4[VS];
    s_t buz_5[VS];
    s_t buz_6[VS];
    s_t buz_7[VS];
    s_t buz_8[VS];
    s_t buz_9[VS];
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      bux_0[lane] = ux[bev0[lane] * u_stride];
      bux_1[lane] = ux[bev1[lane] * u_stride];
      bux_2[lane] = ux[bev2[lane] * u_stride];
      bux_3[lane] = ux[bev3[lane] * u_stride];
      bux_4[lane] = ux[bev4[lane] * u_stride];
      bux_5[lane] = ux[bev5[lane] * u_stride];
      bux_6[lane] = ux[bev6[lane] * u_stride];
      bux_7[lane] = ux[bev7[lane] * u_stride];
      bux_8[lane] = ux[bev8[lane] * u_stride];
      bux_9[lane] = ux[bev9[lane] * u_stride];
      buy_0[lane] = uy[bev0[lane] * u_stride];
      buy_1[lane] = uy[bev1[lane] * u_stride];
      buy_2[lane] = uy[bev2[lane] * u_stride];
      buy_3[lane] = uy[bev3[lane] * u_stride];
      buy_4[lane] = uy[bev4[lane] * u_stride];
      buy_5[lane] = uy[bev5[lane] * u_stride];
      buy_6[lane] = uy[bev6[lane] * u_stride];
      buy_7[lane] = uy[bev7[lane] * u_stride];
      buy_8[lane] = uy[bev8[lane] * u_stride];
      buy_9[lane] = uy[bev9[lane] * u_stride];
      buz_0[lane] = uz[bev0[lane] * u_stride];
      buz_1[lane] = uz[bev1[lane] * u_stride];
      buz_2[lane] = uz[bev2[lane] * u_stride];
      buz_3[lane] = uz[bev3[lane] * u_stride];
      buz_4[lane] = uz[bev4[lane] * u_stride];
      buz_5[lane] = uz[bev5[lane] * u_stride];
      buz_6[lane] = uz[bev6[lane] * u_stride];
      buz_7[lane] = uz[bev7[lane] * u_stride];
      buz_8[lane] = uz[bev8[lane] * u_stride];
      buz_9[lane] = uz[bev9[lane] * u_stride];
    }
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
        btangent_acc[10][lane] = s_t(0);
        btangent_acc[11][lane] = s_t(0);
        btangent_acc[12][lane] = s_t(0);
        btangent_acc[13][lane] = s_t(0);
        btangent_acc[14][lane] = s_t(0);
        btangent_acc[15][lane] = s_t(0);
        btangent_acc[16][lane] = s_t(0);
        btangent_acc[17][lane] = s_t(0);
        btangent_acc[18][lane] = s_t(0);
        btangent_acc[19][lane] = s_t(0);
        btangent_acc[20][lane] = s_t(0);
        btangent_acc[21][lane] = s_t(0);
        btangent_acc[22][lane] = s_t(0);
        btangent_acc[23][lane] = s_t(0);
        btangent_acc[24][lane] = s_t(0);
        btangent_acc[25][lane] = s_t(0);
        btangent_acc[26][lane] = s_t(0);
        btangent_acc[27][lane] = s_t(0);
        btangent_acc[28][lane] = s_t(0);
        btangent_acc[29][lane] = s_t(0);
        btangent_acc[30][lane] = s_t(0);
        btangent_acc[31][lane] = s_t(0);
        btangent_acc[32][lane] = s_t(0);
        btangent_acc[33][lane] = s_t(0);
        btangent_acc[34][lane] = s_t(0);
        btangent_acc[35][lane] = s_t(0);
        btangent_acc[36][lane] = s_t(0);
        btangent_acc[37][lane] = s_t(0);
        btangent_acc[38][lane] = s_t(0);
        btangent_acc[39][lane] = s_t(0);
        btangent_acc[40][lane] = s_t(0);
        btangent_acc[41][lane] = s_t(0);
        btangent_acc[42][lane] = s_t(0);
        btangent_acc[43][lane] = s_t(0);
        btangent_acc[44][lane] = s_t(0);
    }
    for (int q = 0; q < NQ; ++q) {
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
      const s_t ux_0 = bux_0[lane];
      const s_t ux_1 = bux_1[lane];
      const s_t ux_2 = bux_2[lane];
      const s_t ux_3 = bux_3[lane];
      const s_t ux_4 = bux_4[lane];
      const s_t ux_5 = bux_5[lane];
      const s_t ux_6 = bux_6[lane];
      const s_t ux_7 = bux_7[lane];
      const s_t ux_8 = bux_8[lane];
      const s_t ux_9 = bux_9[lane];
      const s_t uy_0 = buy_0[lane];
      const s_t uy_1 = buy_1[lane];
      const s_t uy_2 = buy_2[lane];
      const s_t uy_3 = buy_3[lane];
      const s_t uy_4 = buy_4[lane];
      const s_t uy_5 = buy_5[lane];
      const s_t uy_6 = buy_6[lane];
      const s_t uy_7 = buy_7[lane];
      const s_t uy_8 = buy_8[lane];
      const s_t uy_9 = buy_9[lane];
      const s_t uz_0 = buz_0[lane];
      const s_t uz_1 = buz_1[lane];
      const s_t uz_2 = buz_2[lane];
      const s_t uz_3 = buz_3[lane];
      const s_t uz_4 = buz_4[lane];
      const s_t uz_5 = buz_5[lane];
      const s_t uz_6 = buz_6[lane];
      const s_t uz_7 = buz_7[lane];
      const s_t uz_8 = buz_8[lane];
      const s_t uz_9 = buz_9[lane];
      const s_t gref_0_0 = qgrad_x[q * 10 + 0];
      const s_t gref_0_1 = qgrad_y[q * 10 + 0];
      const s_t gref_0_2 = qgrad_z[q * 10 + 0];
      const s_t gref_1_0 = qgrad_x[q * 10 + 1];
      const s_t gref_1_1 = qgrad_y[q * 10 + 1];
      const s_t gref_1_2 = qgrad_z[q * 10 + 1];
      const s_t gref_2_0 = qgrad_x[q * 10 + 2];
      const s_t gref_2_1 = qgrad_y[q * 10 + 2];
      const s_t gref_2_2 = qgrad_z[q * 10 + 2];
      const s_t gref_3_0 = qgrad_x[q * 10 + 3];
      const s_t gref_3_1 = qgrad_y[q * 10 + 3];
      const s_t gref_3_2 = qgrad_z[q * 10 + 3];
      const s_t gref_4_0 = qgrad_x[q * 10 + 4];
      const s_t gref_4_1 = qgrad_y[q * 10 + 4];
      const s_t gref_4_2 = qgrad_z[q * 10 + 4];
      const s_t gref_5_0 = qgrad_x[q * 10 + 5];
      const s_t gref_5_1 = qgrad_y[q * 10 + 5];
      const s_t gref_5_2 = qgrad_z[q * 10 + 5];
      const s_t gref_6_0 = qgrad_x[q * 10 + 6];
      const s_t gref_6_1 = qgrad_y[q * 10 + 6];
      const s_t gref_6_2 = qgrad_z[q * 10 + 6];
      const s_t gref_7_0 = qgrad_x[q * 10 + 7];
      const s_t gref_7_1 = qgrad_y[q * 10 + 7];
      const s_t gref_7_2 = qgrad_z[q * 10 + 7];
      const s_t gref_8_0 = qgrad_x[q * 10 + 8];
      const s_t gref_8_1 = qgrad_y[q * 10 + 8];
      const s_t gref_8_2 = qgrad_z[q * 10 + 8];
      const s_t gref_9_0 = qgrad_x[q * 10 + 9];
      const s_t gref_9_1 = qgrad_y[q * 10 + 9];
      const s_t gref_9_2 = qgrad_z[q * 10 + 9];
        const s_t qw = qweight[q] * QMEASURE;
            const s_t gu_ref_0_0 = gref_0_0*ux_0 + gref_1_0*ux_1 + gref_2_0*ux_2 + gref_3_0*ux_3 + gref_4_0*ux_4 + gref_5_0*ux_5 + gref_6_0*ux_6 + gref_7_0*ux_7 + gref_8_0*ux_8 + gref_9_0*ux_9;
            const s_t gu_ref_0_1 = gref_0_1*ux_0 + gref_1_1*ux_1 + gref_2_1*ux_2 + gref_3_1*ux_3 + gref_4_1*ux_4 + gref_5_1*ux_5 + gref_6_1*ux_6 + gref_7_1*ux_7 + gref_8_1*ux_8 + gref_9_1*ux_9;
            const s_t gu_ref_0_2 = gref_0_2*ux_0 + gref_1_2*ux_1 + gref_2_2*ux_2 + gref_3_2*ux_3 + gref_4_2*ux_4 + gref_5_2*ux_5 + gref_6_2*ux_6 + gref_7_2*ux_7 + gref_8_2*ux_8 + gref_9_2*ux_9;
            const s_t gu_ref_1_0 = gref_0_0*uy_0 + gref_1_0*uy_1 + gref_2_0*uy_2 + gref_3_0*uy_3 + gref_4_0*uy_4 + gref_5_0*uy_5 + gref_6_0*uy_6 + gref_7_0*uy_7 + gref_8_0*uy_8 + gref_9_0*uy_9;
            const s_t gu_ref_1_1 = gref_0_1*uy_0 + gref_1_1*uy_1 + gref_2_1*uy_2 + gref_3_1*uy_3 + gref_4_1*uy_4 + gref_5_1*uy_5 + gref_6_1*uy_6 + gref_7_1*uy_7 + gref_8_1*uy_8 + gref_9_1*uy_9;
            const s_t gu_ref_1_2 = gref_0_2*uy_0 + gref_1_2*uy_1 + gref_2_2*uy_2 + gref_3_2*uy_3 + gref_4_2*uy_4 + gref_5_2*uy_5 + gref_6_2*uy_6 + gref_7_2*uy_7 + gref_8_2*uy_8 + gref_9_2*uy_9;
            const s_t gu_ref_2_0 = gref_0_0*uz_0 + gref_1_0*uz_1 + gref_2_0*uz_2 + gref_3_0*uz_3 + gref_4_0*uz_4 + gref_5_0*uz_5 + gref_6_0*uz_6 + gref_7_0*uz_7 + gref_8_0*uz_8 + gref_9_0*uz_9;
            const s_t gu_ref_2_1 = gref_0_1*uz_0 + gref_1_1*uz_1 + gref_2_1*uz_2 + gref_3_1*uz_3 + gref_4_1*uz_4 + gref_5_1*uz_5 + gref_6_1*uz_6 + gref_7_1*uz_7 + gref_8_1*uz_8 + gref_9_1*uz_9;
            const s_t gu_ref_2_2 = gref_0_2*uz_0 + gref_1_2*uz_1 + gref_2_2*uz_2 + gref_3_2*uz_3 + gref_4_2*uz_4 + gref_5_2*uz_5 + gref_6_2*uz_6 + gref_7_2*uz_7 + gref_8_2*uz_8 + gref_9_2*uz_9;
            const s_t gradient1_t0 = pow_m1(determinant);
            const s_t gradient1_t1 = gradient1_t0*gu_ref_0_0;
            const s_t gradient1_t2 = gradient1_t0*gu_ref_0_1;
            const s_t gradient1_t3 = gradient1_t0*gu_ref_0_2;
            const s_t gradient1_t4 = gradient1_t0*gu_ref_1_0;
            const s_t gradient1_t5 = gradient1_t0*gu_ref_1_1;
            const s_t gradient1_t6 = gradient1_t0*gu_ref_1_2;
            const s_t gradient1_t7 = gradient1_t0*gu_ref_2_0;
            const s_t gradient1_t8 = gradient1_t0*gu_ref_2_1;
            const s_t gradient1_t9 = gradient1_t0*gu_ref_2_2;
            const s_t gu_0_0 = adjugate0*gradient1_t1 + adjugate3*gradient1_t2 + adjugate6*gradient1_t3;
            const s_t gu_0_1 = adjugate1*gradient1_t1 + adjugate4*gradient1_t2 + adjugate7*gradient1_t3;
            const s_t gu_0_2 = adjugate2*gradient1_t1 + adjugate5*gradient1_t2 + adjugate8*gradient1_t3;
            const s_t gu_1_0 = adjugate0*gradient1_t4 + adjugate3*gradient1_t5 + adjugate6*gradient1_t6;
            const s_t gu_1_1 = adjugate1*gradient1_t4 + adjugate4*gradient1_t5 + adjugate7*gradient1_t6;
            const s_t gu_1_2 = adjugate2*gradient1_t4 + adjugate5*gradient1_t5 + adjugate8*gradient1_t6;
            const s_t gu_2_0 = adjugate0*gradient1_t7 + adjugate3*gradient1_t8 + adjugate6*gradient1_t9;
            const s_t gu_2_1 = adjugate1*gradient1_t7 + adjugate4*gradient1_t8 + adjugate7*gradient1_t9;
            const s_t gu_2_2 = adjugate2*gradient1_t7 + adjugate5*gradient1_t8 + adjugate8*gradient1_t9;
            const s_t integrand_t0 = pow_m1(determinant);
            const s_t integrand_t1 = pow_2(adjugate1);
            const s_t integrand_t2 = gu_2_2 + s_t(1);
            const s_t integrand_t3 = gu_1_0*integrand_t2 - gu_1_2*gu_2_0;
            const s_t integrand_t4 = -integrand_t3;
            const s_t integrand_t5 = gu_0_1*gu_1_0;
            const s_t integrand_t6 = gu_1_1 + s_t(1);
            const s_t integrand_t7 = gu_0_2*gu_2_0;
            const s_t integrand_t8 = gu_0_0 + s_t(1);
            const s_t integrand_t9 = gu_1_2*gu_2_1;
            const s_t integrand_t10 = gu_0_1*gu_1_2*gu_2_0 + gu_0_2*gu_1_0*gu_2_1 - integrand_t2*integrand_t5 + integrand_t2*integrand_t6*integrand_t8 - integrand_t6*integrand_t7 - integrand_t8*integrand_t9;
            const s_t integrand_t11 = pow_m2(integrand_t10);
            const s_t integrand_t12 = integrand_t11*lmbda;
            const s_t integrand_t13 = integrand_t11*mu;
            const s_t integrand_t14 = integrand_t13*integrand_t3;
            const s_t integrand_t15 = integrand_t12*integrand_t4;
            const s_t integrand_t16 = gu_0_0*gu_1_1;
            const s_t integrand_t17 = gu_1_0*gu_2_1;
            const s_t integrand_t18 = sfem_log1p(gu_0_0*gu_2_2 - gu_0_0*integrand_t9 + gu_0_0 + gu_0_1*gu_1_2*gu_2_0 + gu_0_2*integrand_t17 + gu_1_1*gu_2_2 - gu_1_1*integrand_t7 + gu_1_1 + gu_2_2*integrand_t16 - gu_2_2*integrand_t5 + gu_2_2 + integrand_t16 - integrand_t5 - integrand_t7 - integrand_t9);
            const s_t integrand_t19 = integrand_t18*integrand_t3;
            const s_t integrand_t20 = integrand_t12*pow_2(integrand_t4) - integrand_t14*integrand_t4 + integrand_t15*integrand_t19 + mu;
            const s_t integrand_t21 = pow_2(adjugate2);
            const s_t integrand_t22 = -gu_2_0*integrand_t6 + integrand_t17;
            const s_t integrand_t23 = -integrand_t22;
            const s_t integrand_t24 = integrand_t13*integrand_t22;
            const s_t integrand_t25 = integrand_t12*integrand_t22;
            const s_t integrand_t26 = integrand_t18*integrand_t25;
            const s_t integrand_t27 = integrand_t12*pow_2(integrand_t22) - integrand_t23*integrand_t24 + integrand_t23*integrand_t26 + mu;
            const s_t integrand_t28 = integrand_t15*integrand_t22;
            const s_t integrand_t29 = -integrand_t14*integrand_t22 + integrand_t19*integrand_t25 + integrand_t28;
            const s_t integrand_t30 = adjugate1*adjugate2;
            const s_t integrand_t31 = integrand_t13*integrand_t4;
            const s_t integrand_t32 = integrand_t15*integrand_t18;
            const s_t integrand_t33 = -integrand_t23*integrand_t31 + integrand_t23*integrand_t32 + integrand_t28;
            const s_t integrand_t34 = pow_2(adjugate0);
            const s_t integrand_t35 = -integrand_t2*integrand_t6 + integrand_t9;
            const s_t integrand_t36 = -integrand_t35;
            const s_t integrand_t37 = integrand_t35*integrand_t36;
            const s_t integrand_t38 = integrand_t12*integrand_t18;
            const s_t integrand_t39 = integrand_t12*pow_2(integrand_t36) - integrand_t13*integrand_t37 + integrand_t37*integrand_t38 + mu;
            const s_t integrand_t40 = integrand_t12*integrand_t36;
            const s_t integrand_t41 = integrand_t4*integrand_t40;
            const s_t integrand_t42 = -integrand_t31*integrand_t35 + integrand_t32*integrand_t35 + integrand_t41;
            const s_t integrand_t43 = adjugate0*adjugate1;
            const s_t integrand_t44 = -integrand_t14*integrand_t36 + integrand_t19*integrand_t40 + integrand_t41;
            const s_t integrand_t45 = integrand_t22*integrand_t40;
            const s_t integrand_t46 = -integrand_t24*integrand_t35 + integrand_t26*integrand_t35 + integrand_t45;
            const s_t integrand_t47 = adjugate0*adjugate2;
            const s_t integrand_t48 = integrand_t13*integrand_t36;
            const s_t integrand_t49 = integrand_t18*integrand_t40;
            const s_t integrand_t50 = -integrand_t23*integrand_t48 + integrand_t23*integrand_t49 + integrand_t45;
            const s_t integrand_t51 = adjugate1*integrand_t20;
            const s_t integrand_t52 = adjugate2*integrand_t27;
            const s_t integrand_t53 = adjugate1*integrand_t29;
            const s_t integrand_t54 = adjugate2*integrand_t33;
            const s_t integrand_t55 = adjugate0*integrand_t39;
            const s_t integrand_t56 = adjugate0*integrand_t42;
            const s_t integrand_t57 = adjugate0*integrand_t46;
            const s_t integrand_t58 = adjugate1*integrand_t44;
            const s_t integrand_t59 = adjugate2*integrand_t50;
            const s_t integrand_t60 = gu_0_1*gu_2_0 - gu_2_1*integrand_t8;
            const s_t integrand_t61 = -integrand_t60;
            const s_t integrand_t62 = -integrand_t24*integrand_t61 + integrand_t25*integrand_t60 + integrand_t26*integrand_t61;
            const s_t integrand_t63 = gu_0_1*integrand_t2 - gu_0_2*gu_2_1;
            const s_t integrand_t64 = -integrand_t63;
            const s_t integrand_t65 = integrand_t40*integrand_t64 - integrand_t48*integrand_t63 + integrand_t49*integrand_t63;
            const s_t integrand_t66 = -integrand_t2*integrand_t8 + integrand_t7;
            const s_t integrand_t67 = -integrand_t66;
            const s_t integrand_t68 = integrand_t15*integrand_t67 - integrand_t31*integrand_t66 + integrand_t32*integrand_t66;
            const s_t integrand_t69 = pow_m1(integrand_t10);
            const s_t integrand_t70 = integrand_t69*mu;
            const s_t integrand_t71 = gu_2_1*integrand_t70;
            const s_t integrand_t72 = integrand_t18*integrand_t69*lmbda;
            const s_t integrand_t73 = gu_2_1*integrand_t72;
            const s_t integrand_t74 = -integrand_t24*integrand_t63 + integrand_t25*integrand_t64 + integrand_t26*integrand_t63 - integrand_t71 + integrand_t73;
            const s_t integrand_t75 = gu_2_0*integrand_t70;
            const s_t integrand_t76 = gu_2_0*integrand_t72;
            const s_t integrand_t77 = integrand_t15*integrand_t60 - integrand_t31*integrand_t61 + integrand_t32*integrand_t61 - integrand_t75 + integrand_t76;
            const s_t integrand_t78 = integrand_t2*integrand_t70;
            const s_t integrand_t79 = integrand_t2*integrand_t72;
            const s_t integrand_t80 = integrand_t15*integrand_t64 - integrand_t31*integrand_t63 + integrand_t32*integrand_t63 + integrand_t78 - integrand_t79;
            const s_t integrand_t81 = integrand_t40*integrand_t60 - integrand_t48*integrand_t61 + integrand_t49*integrand_t61 + integrand_t71 - integrand_t73;
            const s_t integrand_t82 = -integrand_t24*integrand_t66 + integrand_t25*integrand_t67 + integrand_t26*integrand_t66 + integrand_t75 - integrand_t76;
            const s_t integrand_t83 = integrand_t40*integrand_t67 - integrand_t48*integrand_t66 + integrand_t49*integrand_t66 - integrand_t78 + integrand_t79;
            const s_t integrand_t84 = adjugate0*integrand_t83;
            const s_t integrand_t85 = adjugate0*integrand_t81;
            const s_t integrand_t86 = adjugate1*integrand_t80;
            const s_t integrand_t87 = adjugate1*integrand_t77;
            const s_t integrand_t88 = adjugate2*integrand_t74;
            const s_t integrand_t89 = adjugate2*integrand_t82;
            const s_t integrand_t90 = adjugate0*integrand_t65;
            const s_t integrand_t91 = adjugate1*integrand_t68;
            const s_t integrand_t92 = adjugate2*integrand_t62;
            const s_t integrand_t93 = adjugate3*integrand_t90 + adjugate4*integrand_t91 + adjugate5*integrand_t92;
            const s_t integrand_t94 = adjugate6*integrand_t90 + adjugate7*integrand_t91 + adjugate8*integrand_t92;
            const s_t integrand_t95 = gu_0_2*gu_1_0 - gu_1_2*integrand_t8;
            const s_t integrand_t96 = -integrand_t95;
            const s_t integrand_t97 = integrand_t15*integrand_t95 - integrand_t31*integrand_t96 + integrand_t32*integrand_t96;
            const s_t integrand_t98 = gu_0_1*gu_1_2 - gu_0_2*integrand_t6;
            const s_t integrand_t99 = -integrand_t98;
            const s_t integrand_t100 = integrand_t40*integrand_t98 - integrand_t48*integrand_t99 + integrand_t49*integrand_t99;
            const s_t integrand_t101 = integrand_t5 - integrand_t6*integrand_t8;
            const s_t integrand_t102 = -integrand_t101;
            const s_t integrand_t103 = -integrand_t101*integrand_t24 + integrand_t101*integrand_t26 + integrand_t102*integrand_t25;
            const s_t integrand_t104 = gu_1_2*integrand_t70;
            const s_t integrand_t105 = gu_1_2*integrand_t72;
            const s_t integrand_t106 = -integrand_t104 + integrand_t105 + integrand_t15*integrand_t98 - integrand_t31*integrand_t99 + integrand_t32*integrand_t99;
            const s_t integrand_t107 = gu_1_0*integrand_t70;
            const s_t integrand_t108 = gu_1_0*integrand_t72;
            const s_t integrand_t109 = -integrand_t107 + integrand_t108 - integrand_t24*integrand_t96 + integrand_t25*integrand_t95 + integrand_t26*integrand_t96;
            const s_t integrand_t110 = integrand_t6*integrand_t70;
            const s_t integrand_t111 = integrand_t6*integrand_t72;
            const s_t integrand_t112 = integrand_t110 - integrand_t111 - integrand_t24*integrand_t99 + integrand_t25*integrand_t98 + integrand_t26*integrand_t99;
            const s_t integrand_t113 = integrand_t104 - integrand_t105 + integrand_t40*integrand_t95 - integrand_t48*integrand_t96 + integrand_t49*integrand_t96;
            const s_t integrand_t114 = -integrand_t101*integrand_t31 + integrand_t101*integrand_t32 + integrand_t102*integrand_t15 + integrand_t107 - integrand_t108;
            const s_t integrand_t115 = -integrand_t101*integrand_t48 + integrand_t101*integrand_t49 + integrand_t102*integrand_t40 - integrand_t110 + integrand_t111;
            const s_t integrand_t116 = adjugate0*integrand_t113;
            const s_t integrand_t117 = adjugate0*integrand_t115;
            const s_t integrand_t118 = adjugate1*integrand_t106;
            const s_t integrand_t119 = adjugate1*integrand_t114;
            const s_t integrand_t120 = adjugate2*integrand_t112;
            const s_t integrand_t121 = adjugate2*integrand_t109;
            const s_t integrand_t122 = adjugate0*integrand_t100;
            const s_t integrand_t123 = adjugate1*integrand_t97;
            const s_t integrand_t124 = adjugate2*integrand_t103;
            const s_t integrand_t125 = adjugate3*integrand_t122 + adjugate4*integrand_t123 + adjugate5*integrand_t124;
            const s_t integrand_t126 = adjugate6*integrand_t122 + adjugate7*integrand_t123 + adjugate8*integrand_t124;
            const s_t integrand_t127 = pow_2(adjugate4);
            const s_t integrand_t128 = pow_2(adjugate5);
            const s_t integrand_t129 = adjugate4*adjugate5;
            const s_t integrand_t130 = pow_2(adjugate3);
            const s_t integrand_t131 = adjugate3*adjugate4;
            const s_t integrand_t132 = adjugate3*adjugate5;
            const s_t integrand_t133 = adjugate4*adjugate7;
            const s_t integrand_t134 = adjugate5*adjugate8;
            const s_t integrand_t135 = adjugate4*adjugate8;
            const s_t integrand_t136 = adjugate5*adjugate7;
            const s_t integrand_t137 = adjugate3*adjugate6;
            const s_t integrand_t138 = adjugate3*adjugate7;
            const s_t integrand_t139 = adjugate3*adjugate8;
            const s_t integrand_t140 = adjugate4*adjugate6;
            const s_t integrand_t141 = adjugate5*adjugate6;
            const s_t integrand_t142 = adjugate0*adjugate4;
            const s_t integrand_t143 = adjugate0*adjugate5;
            const s_t integrand_t144 = adjugate1*adjugate3;
            const s_t integrand_t145 = adjugate1*adjugate5;
            const s_t integrand_t146 = adjugate2*adjugate3;
            const s_t integrand_t147 = adjugate2*adjugate4;
            const s_t integrand_t148 = integrand_t133*integrand_t68 + integrand_t134*integrand_t62 + integrand_t137*integrand_t65;
            const s_t integrand_t149 = integrand_t100*integrand_t137 + integrand_t103*integrand_t134 + integrand_t133*integrand_t97;
            const s_t integrand_t150 = pow_2(adjugate7);
            const s_t integrand_t151 = pow_2(adjugate8);
            const s_t integrand_t152 = adjugate7*adjugate8;
            const s_t integrand_t153 = pow_2(adjugate6);
            const s_t integrand_t154 = adjugate6*adjugate7;
            const s_t integrand_t155 = adjugate6*adjugate8;
            const s_t integrand_t156 = adjugate0*adjugate7;
            const s_t integrand_t157 = adjugate0*adjugate8;
            const s_t integrand_t158 = adjugate1*adjugate6;
            const s_t integrand_t159 = adjugate1*adjugate8;
            const s_t integrand_t160 = adjugate2*adjugate6;
            const s_t integrand_t161 = adjugate2*adjugate7;
            const s_t integrand_t162 = integrand_t13*integrand_t63;
            const s_t integrand_t163 = integrand_t12*integrand_t64;
            const s_t integrand_t164 = integrand_t18*integrand_t63;
            const s_t integrand_t165 = integrand_t12*pow_2(integrand_t64) - integrand_t162*integrand_t64 + integrand_t163*integrand_t164 + mu;
            const s_t integrand_t166 = integrand_t13*integrand_t61;
            const s_t integrand_t167 = integrand_t12*integrand_t60;
            const s_t integrand_t168 = integrand_t18*integrand_t61;
            const s_t integrand_t169 = integrand_t12*pow_2(integrand_t60) - integrand_t166*integrand_t60 + integrand_t167*integrand_t168 + mu;
            const s_t integrand_t170 = integrand_t163*integrand_t60;
            const s_t integrand_t171 = -integrand_t162*integrand_t60 + integrand_t164*integrand_t167 + integrand_t170;
            const s_t integrand_t172 = integrand_t163*integrand_t168 - integrand_t166*integrand_t64 + integrand_t170;
            const s_t integrand_t173 = integrand_t13*integrand_t66;
            const s_t integrand_t174 = integrand_t12*integrand_t67;
            const s_t integrand_t175 = integrand_t18*integrand_t66;
            const s_t integrand_t176 = integrand_t12*pow_2(integrand_t67) - integrand_t173*integrand_t67 + integrand_t174*integrand_t175 + mu;
            const s_t integrand_t177 = integrand_t163*integrand_t67;
            const s_t integrand_t178 = integrand_t163*integrand_t175 - integrand_t173*integrand_t64 + integrand_t177;
            const s_t integrand_t179 = -integrand_t162*integrand_t67 + integrand_t164*integrand_t174 + integrand_t177;
            const s_t integrand_t180 = integrand_t174*integrand_t60;
            const s_t integrand_t181 = integrand_t167*integrand_t175 - integrand_t173*integrand_t60 + integrand_t180;
            const s_t integrand_t182 = -integrand_t166*integrand_t67 + integrand_t168*integrand_t174 + integrand_t180;
            const s_t integrand_t183 = adjugate0*integrand_t165;
            const s_t integrand_t184 = adjugate2*integrand_t169;
            const s_t integrand_t185 = adjugate1*integrand_t176;
            const s_t integrand_t186 = integrand_t13*integrand_t99;
            const s_t integrand_t187 = integrand_t18*integrand_t99;
            const s_t integrand_t188 = integrand_t163*integrand_t187 + integrand_t163*integrand_t98 - integrand_t186*integrand_t64;
            const s_t integrand_t189 = integrand_t13*integrand_t96;
            const s_t integrand_t190 = integrand_t18*integrand_t96;
            const s_t integrand_t191 = integrand_t174*integrand_t190 + integrand_t174*integrand_t95 - integrand_t189*integrand_t67;
            const s_t integrand_t192 = integrand_t101*integrand_t13;
            const s_t integrand_t193 = integrand_t101*integrand_t18;
            const s_t integrand_t194 = integrand_t102*integrand_t167 + integrand_t167*integrand_t193 - integrand_t192*integrand_t60;
            const s_t integrand_t195 = gu_0_2*integrand_t70;
            const s_t integrand_t196 = gu_0_2*integrand_t72;
            const s_t integrand_t197 = integrand_t163*integrand_t190 + integrand_t163*integrand_t95 - integrand_t189*integrand_t64 - integrand_t195 + integrand_t196;
            const s_t integrand_t198 = gu_0_1*integrand_t70;
            const s_t integrand_t199 = gu_0_1*integrand_t72;
            const s_t integrand_t200 = integrand_t167*integrand_t187 + integrand_t167*integrand_t98 - integrand_t186*integrand_t60 - integrand_t198 + integrand_t199;
            const s_t integrand_t201 = integrand_t70*integrand_t8;
            const s_t integrand_t202 = integrand_t72*integrand_t8;
            const s_t integrand_t203 = integrand_t167*integrand_t190 + integrand_t167*integrand_t95 - integrand_t189*integrand_t60 + integrand_t201 - integrand_t202;
            const s_t integrand_t204 = integrand_t174*integrand_t187 + integrand_t174*integrand_t98 - integrand_t186*integrand_t67 + integrand_t195 - integrand_t196;
            const s_t integrand_t205 = integrand_t102*integrand_t163 + integrand_t163*integrand_t193 - integrand_t192*integrand_t64 + integrand_t198 - integrand_t199;
            const s_t integrand_t206 = integrand_t102*integrand_t174 + integrand_t174*integrand_t193 - integrand_t192*integrand_t67 - integrand_t201 + integrand_t202;
            const s_t integrand_t207 = adjugate0*integrand_t188;
            const s_t integrand_t208 = adjugate1*integrand_t191;
            const s_t integrand_t209 = adjugate2*integrand_t194;
            const s_t integrand_t210 = adjugate3*integrand_t207 + adjugate4*integrand_t208 + adjugate5*integrand_t209;
            const s_t integrand_t211 = adjugate6*integrand_t207 + adjugate7*integrand_t208 + adjugate8*integrand_t209;
            const s_t integrand_t212 = integrand_t133*integrand_t191 + integrand_t134*integrand_t194 + integrand_t137*integrand_t188;
            const s_t integrand_t213 = integrand_t12*integrand_t98;
            const s_t integrand_t214 = integrand_t12*pow_2(integrand_t98) - integrand_t186*integrand_t98 + integrand_t187*integrand_t213 + mu;
            const s_t integrand_t215 = integrand_t12*integrand_t95;
            const s_t integrand_t216 = integrand_t12*pow_2(integrand_t95) - integrand_t189*integrand_t95 + integrand_t190*integrand_t215 + mu;
            const s_t integrand_t217 = integrand_t213*integrand_t95;
            const s_t integrand_t218 = -integrand_t186*integrand_t95 + integrand_t187*integrand_t215 + integrand_t217;
            const s_t integrand_t219 = -integrand_t189*integrand_t98 + integrand_t190*integrand_t213 + integrand_t217;
            const s_t integrand_t220 = integrand_t102*integrand_t38;
            const s_t integrand_t221 = integrand_t101*integrand_t220 + pow_2(integrand_t102)*integrand_t12 - integrand_t102*integrand_t192 + mu;
            const s_t integrand_t222 = integrand_t102*integrand_t213;
            const s_t integrand_t223 = -integrand_t192*integrand_t98 + integrand_t193*integrand_t213 + integrand_t222;
            const s_t integrand_t224 = -integrand_t102*integrand_t186 + integrand_t220*integrand_t99 + integrand_t222;
            const s_t integrand_t225 = integrand_t102*integrand_t215;
            const s_t integrand_t226 = -integrand_t192*integrand_t95 + integrand_t193*integrand_t215 + integrand_t225;
            const s_t integrand_t227 = -integrand_t102*integrand_t189 + integrand_t220*integrand_t96 + integrand_t225;
            const s_t integrand_t228 = adjugate0*integrand_t214;
            const s_t integrand_t229 = adjugate1*integrand_t216;
            const s_t integrand_t230 = adjugate2*integrand_t221;
            const s_t integrand0 = integrand_t0*(integrand_t1*integrand_t20 + integrand_t21*integrand_t27 + integrand_t29*integrand_t30 + integrand_t30*integrand_t33 + integrand_t34*integrand_t39 + integrand_t42*integrand_t43 + integrand_t43*integrand_t44 + integrand_t46*integrand_t47 + integrand_t47*integrand_t50);
            const s_t integrand1 = integrand_t0*(adjugate3*integrand_t55 + adjugate3*integrand_t58 + adjugate3*integrand_t59 + adjugate4*integrand_t51 + adjugate4*integrand_t54 + adjugate4*integrand_t56 + adjugate5*integrand_t52 + adjugate5*integrand_t53 + adjugate5*integrand_t57);
            const s_t integrand2 = integrand_t0*(adjugate6*integrand_t55 + adjugate6*integrand_t58 + adjugate6*integrand_t59 + adjugate7*integrand_t51 + adjugate7*integrand_t54 + adjugate7*integrand_t56 + adjugate8*integrand_t52 + adjugate8*integrand_t53 + adjugate8*integrand_t57);
            const s_t integrand3 = integrand_t0*(integrand_t1*integrand_t68 + integrand_t21*integrand_t62 + integrand_t30*integrand_t77 + integrand_t30*integrand_t82 + integrand_t34*integrand_t65 + integrand_t43*integrand_t80 + integrand_t43*integrand_t83 + integrand_t47*integrand_t74 + integrand_t47*integrand_t81);
            const s_t integrand4 = integrand_t0*(adjugate3*integrand_t86 + adjugate3*integrand_t88 + adjugate4*integrand_t84 + adjugate4*integrand_t89 + adjugate5*integrand_t85 + adjugate5*integrand_t87 + integrand_t93);
            const s_t integrand5 = integrand_t0*(adjugate6*integrand_t86 + adjugate6*integrand_t88 + adjugate7*integrand_t84 + adjugate7*integrand_t89 + adjugate8*integrand_t85 + adjugate8*integrand_t87 + integrand_t94);
            const s_t integrand6 = integrand_t0*(integrand_t1*integrand_t97 + integrand_t100*integrand_t34 + integrand_t103*integrand_t21 + integrand_t106*integrand_t43 + integrand_t109*integrand_t30 + integrand_t112*integrand_t47 + integrand_t113*integrand_t43 + integrand_t114*integrand_t30 + integrand_t115*integrand_t47);
            const s_t integrand7 = integrand_t0*(adjugate3*integrand_t118 + adjugate3*integrand_t120 + adjugate4*integrand_t116 + adjugate4*integrand_t121 + adjugate5*integrand_t117 + adjugate5*integrand_t119 + integrand_t125);
            const s_t integrand8 = integrand_t0*(adjugate6*integrand_t118 + adjugate6*integrand_t120 + adjugate7*integrand_t116 + adjugate7*integrand_t121 + adjugate8*integrand_t117 + adjugate8*integrand_t119 + integrand_t126);
            const s_t integrand9 = integrand_t0*(integrand_t127*integrand_t20 + integrand_t128*integrand_t27 + integrand_t129*integrand_t29 + integrand_t129*integrand_t33 + integrand_t130*integrand_t39 + integrand_t131*integrand_t42 + integrand_t131*integrand_t44 + integrand_t132*integrand_t46 + integrand_t132*integrand_t50);
            const s_t integrand10 = integrand_t0*(integrand_t133*integrand_t20 + integrand_t134*integrand_t27 + integrand_t135*integrand_t29 + integrand_t136*integrand_t33 + integrand_t137*integrand_t39 + integrand_t138*integrand_t42 + integrand_t139*integrand_t46 + integrand_t140*integrand_t44 + integrand_t141*integrand_t50);
            const s_t integrand11 = integrand_t0*(integrand_t142*integrand_t80 + integrand_t143*integrand_t74 + integrand_t144*integrand_t83 + integrand_t145*integrand_t82 + integrand_t146*integrand_t81 + integrand_t147*integrand_t77 + integrand_t93);
            const s_t integrand12 = integrand_t0*(integrand_t127*integrand_t68 + integrand_t128*integrand_t62 + integrand_t129*integrand_t77 + integrand_t129*integrand_t82 + integrand_t130*integrand_t65 + integrand_t131*integrand_t80 + integrand_t131*integrand_t83 + integrand_t132*integrand_t74 + integrand_t132*integrand_t81);
            const s_t integrand13 = integrand_t0*(integrand_t135*integrand_t77 + integrand_t136*integrand_t82 + integrand_t138*integrand_t83 + integrand_t139*integrand_t81 + integrand_t140*integrand_t80 + integrand_t141*integrand_t74 + integrand_t148);
            const s_t integrand14 = integrand_t0*(integrand_t106*integrand_t142 + integrand_t109*integrand_t145 + integrand_t112*integrand_t143 + integrand_t113*integrand_t144 + integrand_t114*integrand_t147 + integrand_t115*integrand_t146 + integrand_t125);
            const s_t integrand15 = integrand_t0*(integrand_t100*integrand_t130 + integrand_t103*integrand_t128 + integrand_t106*integrand_t131 + integrand_t109*integrand_t129 + integrand_t112*integrand_t132 + integrand_t113*integrand_t131 + integrand_t114*integrand_t129 + integrand_t115*integrand_t132 + integrand_t127*integrand_t97);
            const s_t integrand16 = integrand_t0*(integrand_t106*integrand_t140 + integrand_t109*integrand_t136 + integrand_t112*integrand_t141 + integrand_t113*integrand_t138 + integrand_t114*integrand_t135 + integrand_t115*integrand_t139 + integrand_t149);
            const s_t integrand17 = integrand_t0*(integrand_t150*integrand_t20 + integrand_t151*integrand_t27 + integrand_t152*integrand_t29 + integrand_t152*integrand_t33 + integrand_t153*integrand_t39 + integrand_t154*integrand_t42 + integrand_t154*integrand_t44 + integrand_t155*integrand_t46 + integrand_t155*integrand_t50);
            const s_t integrand18 = integrand_t0*(integrand_t156*integrand_t80 + integrand_t157*integrand_t74 + integrand_t158*integrand_t83 + integrand_t159*integrand_t82 + integrand_t160*integrand_t81 + integrand_t161*integrand_t77 + integrand_t94);
            const s_t integrand19 = integrand_t0*(integrand_t135*integrand_t82 + integrand_t136*integrand_t77 + integrand_t138*integrand_t80 + integrand_t139*integrand_t74 + integrand_t140*integrand_t83 + integrand_t141*integrand_t81 + integrand_t148);
            const s_t integrand20 = integrand_t0*(integrand_t150*integrand_t68 + integrand_t151*integrand_t62 + integrand_t152*integrand_t77 + integrand_t152*integrand_t82 + integrand_t153*integrand_t65 + integrand_t154*integrand_t80 + integrand_t154*integrand_t83 + integrand_t155*integrand_t74 + integrand_t155*integrand_t81);
            const s_t integrand21 = integrand_t0*(integrand_t106*integrand_t156 + integrand_t109*integrand_t159 + integrand_t112*integrand_t157 + integrand_t113*integrand_t158 + integrand_t114*integrand_t161 + integrand_t115*integrand_t160 + integrand_t126);
            const s_t integrand22 = integrand_t0*(integrand_t106*integrand_t138 + integrand_t109*integrand_t135 + integrand_t112*integrand_t139 + integrand_t113*integrand_t140 + integrand_t114*integrand_t136 + integrand_t115*integrand_t141 + integrand_t149);
            const s_t integrand23 = integrand_t0*(integrand_t100*integrand_t153 + integrand_t103*integrand_t151 + integrand_t106*integrand_t154 + integrand_t109*integrand_t152 + integrand_t112*integrand_t155 + integrand_t113*integrand_t154 + integrand_t114*integrand_t152 + integrand_t115*integrand_t155 + integrand_t150*integrand_t97);
            const s_t integrand24 = integrand_t0*(integrand_t1*integrand_t176 + integrand_t165*integrand_t34 + integrand_t169*integrand_t21 + integrand_t171*integrand_t47 + integrand_t172*integrand_t47 + integrand_t178*integrand_t43 + integrand_t179*integrand_t43 + integrand_t181*integrand_t30 + integrand_t182*integrand_t30);
            const s_t integrand25 = integrand_t0*(adjugate3*integrand_t183 + adjugate4*integrand_t185 + adjugate5*integrand_t184 + integrand_t142*integrand_t179 + integrand_t143*integrand_t171 + integrand_t144*integrand_t178 + integrand_t145*integrand_t181 + integrand_t146*integrand_t172 + integrand_t147*integrand_t182);
            const s_t integrand26 = integrand_t0*(adjugate6*integrand_t183 + adjugate7*integrand_t185 + adjugate8*integrand_t184 + integrand_t156*integrand_t179 + integrand_t157*integrand_t171 + integrand_t158*integrand_t178 + integrand_t159*integrand_t181 + integrand_t160*integrand_t172 + integrand_t161*integrand_t182);
            const s_t integrand27 = integrand_t0*(integrand_t1*integrand_t191 + integrand_t188*integrand_t34 + integrand_t194*integrand_t21 + integrand_t197*integrand_t43 + integrand_t200*integrand_t47 + integrand_t203*integrand_t30 + integrand_t204*integrand_t43 + integrand_t205*integrand_t47 + integrand_t206*integrand_t30);
            const s_t integrand28 = integrand_t0*(integrand_t142*integrand_t197 + integrand_t143*integrand_t205 + integrand_t144*integrand_t204 + integrand_t145*integrand_t206 + integrand_t146*integrand_t200 + integrand_t147*integrand_t203 + integrand_t210);
            const s_t integrand29 = integrand_t0*(integrand_t156*integrand_t197 + integrand_t157*integrand_t205 + integrand_t158*integrand_t204 + integrand_t159*integrand_t206 + integrand_t160*integrand_t200 + integrand_t161*integrand_t203 + integrand_t211);
            const s_t integrand30 = integrand_t0*(integrand_t127*integrand_t176 + integrand_t128*integrand_t169 + integrand_t129*integrand_t181 + integrand_t129*integrand_t182 + integrand_t130*integrand_t165 + integrand_t131*integrand_t178 + integrand_t131*integrand_t179 + integrand_t132*integrand_t171 + integrand_t132*integrand_t172);
            const s_t integrand31 = integrand_t0*(integrand_t133*integrand_t176 + integrand_t134*integrand_t169 + integrand_t135*integrand_t181 + integrand_t136*integrand_t182 + integrand_t137*integrand_t165 + integrand_t138*integrand_t179 + integrand_t139*integrand_t171 + integrand_t140*integrand_t178 + integrand_t141*integrand_t172);
            const s_t integrand32 = integrand_t0*(integrand_t142*integrand_t204 + integrand_t143*integrand_t200 + integrand_t144*integrand_t197 + integrand_t145*integrand_t203 + integrand_t146*integrand_t205 + integrand_t147*integrand_t206 + integrand_t210);
            const s_t integrand33 = integrand_t0*(integrand_t127*integrand_t191 + integrand_t128*integrand_t194 + integrand_t129*integrand_t203 + integrand_t129*integrand_t206 + integrand_t130*integrand_t188 + integrand_t131*integrand_t197 + integrand_t131*integrand_t204 + integrand_t132*integrand_t200 + integrand_t132*integrand_t205);
            const s_t integrand34 = integrand_t0*(integrand_t135*integrand_t206 + integrand_t136*integrand_t203 + integrand_t138*integrand_t197 + integrand_t139*integrand_t205 + integrand_t140*integrand_t204 + integrand_t141*integrand_t200 + integrand_t212);
            const s_t integrand35 = integrand_t0*(integrand_t150*integrand_t176 + integrand_t151*integrand_t169 + integrand_t152*integrand_t181 + integrand_t152*integrand_t182 + integrand_t153*integrand_t165 + integrand_t154*integrand_t178 + integrand_t154*integrand_t179 + integrand_t155*integrand_t171 + integrand_t155*integrand_t172);
            const s_t integrand36 = integrand_t0*(integrand_t156*integrand_t204 + integrand_t157*integrand_t200 + integrand_t158*integrand_t197 + integrand_t159*integrand_t203 + integrand_t160*integrand_t205 + integrand_t161*integrand_t206 + integrand_t211);
            const s_t integrand37 = integrand_t0*(integrand_t135*integrand_t203 + integrand_t136*integrand_t206 + integrand_t138*integrand_t204 + integrand_t139*integrand_t200 + integrand_t140*integrand_t197 + integrand_t141*integrand_t205 + integrand_t212);
            const s_t integrand38 = integrand_t0*(integrand_t150*integrand_t191 + integrand_t151*integrand_t194 + integrand_t152*integrand_t203 + integrand_t152*integrand_t206 + integrand_t153*integrand_t188 + integrand_t154*integrand_t197 + integrand_t154*integrand_t204 + integrand_t155*integrand_t200 + integrand_t155*integrand_t205);
            const s_t integrand39 = integrand_t0*(integrand_t1*integrand_t216 + integrand_t21*integrand_t221 + integrand_t214*integrand_t34 + integrand_t218*integrand_t43 + integrand_t219*integrand_t43 + integrand_t223*integrand_t47 + integrand_t224*integrand_t47 + integrand_t226*integrand_t30 + integrand_t227*integrand_t30);
            const s_t integrand40 = integrand_t0*(adjugate3*integrand_t228 + adjugate4*integrand_t229 + adjugate5*integrand_t230 + integrand_t142*integrand_t218 + integrand_t143*integrand_t224 + integrand_t144*integrand_t219 + integrand_t145*integrand_t227 + integrand_t146*integrand_t223 + integrand_t147*integrand_t226);
            const s_t integrand41 = integrand_t0*(adjugate6*integrand_t228 + adjugate7*integrand_t229 + adjugate8*integrand_t230 + integrand_t156*integrand_t218 + integrand_t157*integrand_t224 + integrand_t158*integrand_t219 + integrand_t159*integrand_t227 + integrand_t160*integrand_t223 + integrand_t161*integrand_t226);
            const s_t integrand42 = integrand_t0*(integrand_t127*integrand_t216 + integrand_t128*integrand_t221 + integrand_t129*integrand_t226 + integrand_t129*integrand_t227 + integrand_t130*integrand_t214 + integrand_t131*integrand_t218 + integrand_t131*integrand_t219 + integrand_t132*integrand_t223 + integrand_t132*integrand_t224);
            const s_t integrand43 = integrand_t0*(integrand_t133*integrand_t216 + integrand_t134*integrand_t221 + integrand_t135*integrand_t227 + integrand_t136*integrand_t226 + integrand_t137*integrand_t214 + integrand_t138*integrand_t218 + integrand_t139*integrand_t224 + integrand_t140*integrand_t219 + integrand_t141*integrand_t223);
            const s_t integrand44 = integrand_t0*(integrand_t150*integrand_t216 + integrand_t151*integrand_t221 + integrand_t152*integrand_t226 + integrand_t152*integrand_t227 + integrand_t153*integrand_t214 + integrand_t154*integrand_t218 + integrand_t154*integrand_t219 + integrand_t155*integrand_t223 + integrand_t155*integrand_t224);
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
        btangent_acc[10][lane] += qw * integrand10;
        btangent_acc[11][lane] += qw * integrand11;
        btangent_acc[12][lane] += qw * integrand12;
        btangent_acc[13][lane] += qw * integrand13;
        btangent_acc[14][lane] += qw * integrand14;
        btangent_acc[15][lane] += qw * integrand15;
        btangent_acc[16][lane] += qw * integrand16;
        btangent_acc[17][lane] += qw * integrand17;
        btangent_acc[18][lane] += qw * integrand18;
        btangent_acc[19][lane] += qw * integrand19;
        btangent_acc[20][lane] += qw * integrand20;
        btangent_acc[21][lane] += qw * integrand21;
        btangent_acc[22][lane] += qw * integrand22;
        btangent_acc[23][lane] += qw * integrand23;
        btangent_acc[24][lane] += qw * integrand24;
        btangent_acc[25][lane] += qw * integrand25;
        btangent_acc[26][lane] += qw * integrand26;
        btangent_acc[27][lane] += qw * integrand27;
        btangent_acc[28][lane] += qw * integrand28;
        btangent_acc[29][lane] += qw * integrand29;
        btangent_acc[30][lane] += qw * integrand30;
        btangent_acc[31][lane] += qw * integrand31;
        btangent_acc[32][lane] += qw * integrand32;
        btangent_acc[33][lane] += qw * integrand33;
        btangent_acc[34][lane] += qw * integrand34;
        btangent_acc[35][lane] += qw * integrand35;
        btangent_acc[36][lane] += qw * integrand36;
        btangent_acc[37][lane] += qw * integrand37;
        btangent_acc[38][lane] += qw * integrand38;
        btangent_acc[39][lane] += qw * integrand39;
        btangent_acc[40][lane] += qw * integrand40;
        btangent_acc[41][lane] += qw * integrand41;
        btangent_acc[42][lane] += qw * integrand42;
        btangent_acc[43][lane] += qw * integrand43;
        btangent_acc[44][lane] += qw * integrand44;
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
        btangent10[lane] = tangent_t(btangent_acc[10][lane]);
        btangent11[lane] = tangent_t(btangent_acc[11][lane]);
        btangent12[lane] = tangent_t(btangent_acc[12][lane]);
        btangent13[lane] = tangent_t(btangent_acc[13][lane]);
        btangent14[lane] = tangent_t(btangent_acc[14][lane]);
        btangent15[lane] = tangent_t(btangent_acc[15][lane]);
        btangent16[lane] = tangent_t(btangent_acc[16][lane]);
        btangent17[lane] = tangent_t(btangent_acc[17][lane]);
        btangent18[lane] = tangent_t(btangent_acc[18][lane]);
        btangent19[lane] = tangent_t(btangent_acc[19][lane]);
        btangent20[lane] = tangent_t(btangent_acc[20][lane]);
        btangent21[lane] = tangent_t(btangent_acc[21][lane]);
        btangent22[lane] = tangent_t(btangent_acc[22][lane]);
        btangent23[lane] = tangent_t(btangent_acc[23][lane]);
        btangent24[lane] = tangent_t(btangent_acc[24][lane]);
        btangent25[lane] = tangent_t(btangent_acc[25][lane]);
        btangent26[lane] = tangent_t(btangent_acc[26][lane]);
        btangent27[lane] = tangent_t(btangent_acc[27][lane]);
        btangent28[lane] = tangent_t(btangent_acc[28][lane]);
        btangent29[lane] = tangent_t(btangent_acc[29][lane]);
        btangent30[lane] = tangent_t(btangent_acc[30][lane]);
        btangent31[lane] = tangent_t(btangent_acc[31][lane]);
        btangent32[lane] = tangent_t(btangent_acc[32][lane]);
        btangent33[lane] = tangent_t(btangent_acc[33][lane]);
        btangent34[lane] = tangent_t(btangent_acc[34][lane]);
        btangent35[lane] = tangent_t(btangent_acc[35][lane]);
        btangent36[lane] = tangent_t(btangent_acc[36][lane]);
        btangent37[lane] = tangent_t(btangent_acc[37][lane]);
        btangent38[lane] = tangent_t(btangent_acc[38][lane]);
        btangent39[lane] = tangent_t(btangent_acc[39][lane]);
        btangent40[lane] = tangent_t(btangent_acc[40][lane]);
        btangent41[lane] = tangent_t(btangent_acc[41][lane]);
        btangent42[lane] = tangent_t(btangent_acc[42][lane]);
        btangent43[lane] = tangent_t(btangent_acc[43][lane]);
        btangent44[lane] = tangent_t(btangent_acc[44][lane]);
    }
  }

  return SFEM_SUCCESS;
}

template <typename s_t, typename tangent_t, int VS>
static SFEM_INLINE int neohookean_ogden_tet10_inexact_apply_stored_a_msoa_impl(
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
    idx_t bev4[VS];
    idx_t bev5[VS];
    idx_t bev6[VS];
    idx_t bev7[VS];
    idx_t bev8[VS];
    idx_t bev9[VS];
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      bev0[lane] = elements[0][evb + lane];
      bev1[lane] = elements[1][evb + lane];
      bev2[lane] = elements[2][evb + lane];
      bev3[lane] = elements[3][evb + lane];
      bev4[lane] = elements[4][evb + lane];
      bev5[lane] = elements[5][evb + lane];
      bev6[lane] = elements[6][evb + lane];
      bev7[lane] = elements[7][evb + lane];
      bev8[lane] = elements[8][evb + lane];
      bev9[lane] = elements[9][evb + lane];
    }
    s_t bhx_0[VS];
    s_t bhx_1[VS];
    s_t bhx_2[VS];
    s_t bhx_3[VS];
    s_t bhx_4[VS];
    s_t bhx_5[VS];
    s_t bhx_6[VS];
    s_t bhx_7[VS];
    s_t bhx_8[VS];
    s_t bhx_9[VS];
    s_t bhy_0[VS];
    s_t bhy_1[VS];
    s_t bhy_2[VS];
    s_t bhy_3[VS];
    s_t bhy_4[VS];
    s_t bhy_5[VS];
    s_t bhy_6[VS];
    s_t bhy_7[VS];
    s_t bhy_8[VS];
    s_t bhy_9[VS];
    s_t bhz_0[VS];
    s_t bhz_1[VS];
    s_t bhz_2[VS];
    s_t bhz_3[VS];
    s_t bhz_4[VS];
    s_t bhz_5[VS];
    s_t bhz_6[VS];
    s_t bhz_7[VS];
    s_t bhz_8[VS];
    s_t bhz_9[VS];
    s_t bout0_0[VS];
    s_t bout0_1[VS];
    s_t bout0_2[VS];
    s_t bout0_3[VS];
    s_t bout0_4[VS];
    s_t bout0_5[VS];
    s_t bout0_6[VS];
    s_t bout0_7[VS];
    s_t bout0_8[VS];
    s_t bout0_9[VS];
    s_t bout1_0[VS];
    s_t bout1_1[VS];
    s_t bout1_2[VS];
    s_t bout1_3[VS];
    s_t bout1_4[VS];
    s_t bout1_5[VS];
    s_t bout1_6[VS];
    s_t bout1_7[VS];
    s_t bout1_8[VS];
    s_t bout1_9[VS];
    s_t bout2_0[VS];
    s_t bout2_1[VS];
    s_t bout2_2[VS];
    s_t bout2_3[VS];
    s_t bout2_4[VS];
    s_t bout2_5[VS];
    s_t bout2_6[VS];
    s_t bout2_7[VS];
    s_t bout2_8[VS];
    s_t bout2_9[VS];
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      bhx_0[lane] = hx[bev0[lane] * h_stride];
      bhx_1[lane] = hx[bev1[lane] * h_stride];
      bhx_2[lane] = hx[bev2[lane] * h_stride];
      bhx_3[lane] = hx[bev3[lane] * h_stride];
      bhx_4[lane] = hx[bev4[lane] * h_stride];
      bhx_5[lane] = hx[bev5[lane] * h_stride];
      bhx_6[lane] = hx[bev6[lane] * h_stride];
      bhx_7[lane] = hx[bev7[lane] * h_stride];
      bhx_8[lane] = hx[bev8[lane] * h_stride];
      bhx_9[lane] = hx[bev9[lane] * h_stride];
      bhy_0[lane] = hy[bev0[lane] * h_stride];
      bhy_1[lane] = hy[bev1[lane] * h_stride];
      bhy_2[lane] = hy[bev2[lane] * h_stride];
      bhy_3[lane] = hy[bev3[lane] * h_stride];
      bhy_4[lane] = hy[bev4[lane] * h_stride];
      bhy_5[lane] = hy[bev5[lane] * h_stride];
      bhy_6[lane] = hy[bev6[lane] * h_stride];
      bhy_7[lane] = hy[bev7[lane] * h_stride];
      bhy_8[lane] = hy[bev8[lane] * h_stride];
      bhy_9[lane] = hy[bev9[lane] * h_stride];
      bhz_0[lane] = hz[bev0[lane] * h_stride];
      bhz_1[lane] = hz[bev1[lane] * h_stride];
      bhz_2[lane] = hz[bev2[lane] * h_stride];
      bhz_3[lane] = hz[bev3[lane] * h_stride];
      bhz_4[lane] = hz[bev4[lane] * h_stride];
      bhz_5[lane] = hz[bev5[lane] * h_stride];
      bhz_6[lane] = hz[bev6[lane] * h_stride];
      bhz_7[lane] = hz[bev7[lane] * h_stride];
      bhz_8[lane] = hz[bev8[lane] * h_stride];
      bhz_9[lane] = hz[bev9[lane] * h_stride];
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
      const s_t hx_4 = bhx_4[lane];
      const s_t hx_5 = bhx_5[lane];
      const s_t hx_6 = bhx_6[lane];
      const s_t hx_7 = bhx_7[lane];
      const s_t hx_8 = bhx_8[lane];
      const s_t hx_9 = bhx_9[lane];
      const s_t hy_0 = bhy_0[lane];
      const s_t hy_1 = bhy_1[lane];
      const s_t hy_2 = bhy_2[lane];
      const s_t hy_3 = bhy_3[lane];
      const s_t hy_4 = bhy_4[lane];
      const s_t hy_5 = bhy_5[lane];
      const s_t hy_6 = bhy_6[lane];
      const s_t hy_7 = bhy_7[lane];
      const s_t hy_8 = bhy_8[lane];
      const s_t hy_9 = bhy_9[lane];
      const s_t hz_0 = bhz_0[lane];
      const s_t hz_1 = bhz_1[lane];
      const s_t hz_2 = bhz_2[lane];
      const s_t hz_3 = bhz_3[lane];
      const s_t hz_4 = bhz_4[lane];
      const s_t hz_5 = bhz_5[lane];
      const s_t hz_6 = bhz_6[lane];
      const s_t hz_7 = bhz_7[lane];
      const s_t hz_8 = bhz_8[lane];
      const s_t hz_9 = bhz_9[lane];
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
      const s_t compressed_increment_t0 = -(s_t(2) / s_t(15))*hx_4;
      const s_t compressed_increment_t1 = ((s_t(1) / s_t(30)))*hx_1;
      const s_t compressed_increment_t2 = ((s_t(1) / s_t(30)))*hx_7;
      const s_t compressed_increment_t3 = ((s_t(1) / s_t(10)))*hx_0;
      const s_t compressed_increment_t4 = ((s_t(1) / s_t(30)))*hx_5;
      const s_t compressed_increment_t5 = -compressed_increment_t2 + compressed_increment_t3 + compressed_increment_t4;
      const s_t compressed_increment_t6 = ((s_t(1) / s_t(30)))*hx_6;
      const s_t compressed_increment_t7 = ((s_t(1) / s_t(30)))*hx_8;
      const s_t compressed_increment_t8 = -compressed_increment_t6 + compressed_increment_t7;
      const s_t compressed_increment_t9 = -(s_t(2) / s_t(15))*hx_6;
      const s_t compressed_increment_t10 = ((s_t(1) / s_t(30)))*hx_2;
      const s_t compressed_increment_t11 = ((s_t(1) / s_t(30)))*hx_4;
      const s_t compressed_increment_t12 = ((s_t(1) / s_t(30)))*hx_9;
      const s_t compressed_increment_t13 = -compressed_increment_t11 + compressed_increment_t12;
      const s_t compressed_increment_t14 = -(s_t(2) / s_t(15))*hx_7;
      const s_t compressed_increment_t15 = ((s_t(1) / s_t(30)))*hx_3;
      const s_t compressed_increment_t16 = -compressed_increment_t7;
      const s_t compressed_increment_t17 = ((s_t(1) / s_t(30)))*hx_0;
      const s_t compressed_increment_t18 = compressed_increment_t17 + compressed_increment_t2 - compressed_increment_t4;
      const s_t compressed_increment_t19 = compressed_increment_t0 + ((s_t(1) / s_t(10)))*hx_1;
      const s_t compressed_increment_t20 = -compressed_increment_t10 + compressed_increment_t17;
      const s_t compressed_increment_t21 = -compressed_increment_t12;
      const s_t compressed_increment_t22 = compressed_increment_t21 - (s_t(1) / s_t(10))*hx_4;
      const s_t compressed_increment_t23 = compressed_increment_t2 + ((s_t(1) / s_t(10)))*hx_5;
      const s_t compressed_increment_t24 = -compressed_increment_t15 + compressed_increment_t17;
      const s_t compressed_increment_t25 = compressed_increment_t16 - (s_t(1) / s_t(10))*hx_6;
      const s_t compressed_increment_t26 = -(s_t(4) / s_t(15))*hx_4 + ((s_t(2) / s_t(15)))*hx_9;
      const s_t compressed_increment_t27 = -(s_t(2) / s_t(15))*hy_4;
      const s_t compressed_increment_t28 = ((s_t(1) / s_t(30)))*hy_1;
      const s_t compressed_increment_t29 = ((s_t(1) / s_t(30)))*hy_7;
      const s_t compressed_increment_t30 = ((s_t(1) / s_t(10)))*hy_0;
      const s_t compressed_increment_t31 = ((s_t(1) / s_t(30)))*hy_5;
      const s_t compressed_increment_t32 = -compressed_increment_t29 + compressed_increment_t30 + compressed_increment_t31;
      const s_t compressed_increment_t33 = ((s_t(1) / s_t(30)))*hy_6;
      const s_t compressed_increment_t34 = ((s_t(1) / s_t(30)))*hy_8;
      const s_t compressed_increment_t35 = -compressed_increment_t33 + compressed_increment_t34;
      const s_t compressed_increment_t36 = -(s_t(2) / s_t(15))*hy_6;
      const s_t compressed_increment_t37 = ((s_t(1) / s_t(30)))*hy_2;
      const s_t compressed_increment_t38 = ((s_t(1) / s_t(30)))*hy_4;
      const s_t compressed_increment_t39 = ((s_t(1) / s_t(30)))*hy_9;
      const s_t compressed_increment_t40 = -compressed_increment_t38 + compressed_increment_t39;
      const s_t compressed_increment_t41 = -(s_t(2) / s_t(15))*hy_7;
      const s_t compressed_increment_t42 = ((s_t(1) / s_t(30)))*hy_3;
      const s_t compressed_increment_t43 = -compressed_increment_t34;
      const s_t compressed_increment_t44 = ((s_t(1) / s_t(30)))*hy_0;
      const s_t compressed_increment_t45 = compressed_increment_t29 - compressed_increment_t31 + compressed_increment_t44;
      const s_t compressed_increment_t46 = compressed_increment_t27 + ((s_t(1) / s_t(10)))*hy_1;
      const s_t compressed_increment_t47 = -compressed_increment_t37 + compressed_increment_t44;
      const s_t compressed_increment_t48 = -compressed_increment_t39;
      const s_t compressed_increment_t49 = compressed_increment_t48 - (s_t(1) / s_t(10))*hy_4;
      const s_t compressed_increment_t50 = compressed_increment_t29 + ((s_t(1) / s_t(10)))*hy_5;
      const s_t compressed_increment_t51 = -compressed_increment_t42 + compressed_increment_t44;
      const s_t compressed_increment_t52 = compressed_increment_t43 - (s_t(1) / s_t(10))*hy_6;
      const s_t compressed_increment_t53 = -(s_t(4) / s_t(15))*hy_4 + ((s_t(2) / s_t(15)))*hy_9;
      const s_t compressed_increment_t54 = -(s_t(2) / s_t(15))*hz_4;
      const s_t compressed_increment_t55 = ((s_t(1) / s_t(30)))*hz_1;
      const s_t compressed_increment_t56 = ((s_t(1) / s_t(30)))*hz_7;
      const s_t compressed_increment_t57 = ((s_t(1) / s_t(10)))*hz_0;
      const s_t compressed_increment_t58 = ((s_t(1) / s_t(30)))*hz_5;
      const s_t compressed_increment_t59 = -compressed_increment_t56 + compressed_increment_t57 + compressed_increment_t58;
      const s_t compressed_increment_t60 = ((s_t(1) / s_t(30)))*hz_6;
      const s_t compressed_increment_t61 = ((s_t(1) / s_t(30)))*hz_8;
      const s_t compressed_increment_t62 = -compressed_increment_t60 + compressed_increment_t61;
      const s_t compressed_increment_t63 = -(s_t(2) / s_t(15))*hz_6;
      const s_t compressed_increment_t64 = ((s_t(1) / s_t(30)))*hz_2;
      const s_t compressed_increment_t65 = ((s_t(1) / s_t(30)))*hz_4;
      const s_t compressed_increment_t66 = ((s_t(1) / s_t(30)))*hz_9;
      const s_t compressed_increment_t67 = -compressed_increment_t65 + compressed_increment_t66;
      const s_t compressed_increment_t68 = -(s_t(2) / s_t(15))*hz_7;
      const s_t compressed_increment_t69 = ((s_t(1) / s_t(30)))*hz_3;
      const s_t compressed_increment_t70 = -compressed_increment_t61;
      const s_t compressed_increment_t71 = ((s_t(1) / s_t(30)))*hz_0;
      const s_t compressed_increment_t72 = compressed_increment_t56 - compressed_increment_t58 + compressed_increment_t71;
      const s_t compressed_increment_t73 = compressed_increment_t54 + ((s_t(1) / s_t(10)))*hz_1;
      const s_t compressed_increment_t74 = -compressed_increment_t64 + compressed_increment_t71;
      const s_t compressed_increment_t75 = -compressed_increment_t66;
      const s_t compressed_increment_t76 = compressed_increment_t75 - (s_t(1) / s_t(10))*hz_4;
      const s_t compressed_increment_t77 = compressed_increment_t56 + ((s_t(1) / s_t(10)))*hz_5;
      const s_t compressed_increment_t78 = -compressed_increment_t69 + compressed_increment_t71;
      const s_t compressed_increment_t79 = compressed_increment_t70 - (s_t(1) / s_t(10))*hz_6;
      const s_t compressed_increment_t80 = -(s_t(4) / s_t(15))*hz_4 + ((s_t(2) / s_t(15)))*hz_9;
      const s_t pa_p0_0_0 = compressed_increment_t0 + compressed_increment_t1 + compressed_increment_t5 + compressed_increment_t8;
      const s_t pa_p0_0_1 = compressed_increment_t10 + compressed_increment_t13 + compressed_increment_t5 + compressed_increment_t9;
      const s_t pa_p0_0_2 = compressed_increment_t13 + compressed_increment_t14 + compressed_increment_t15 + compressed_increment_t3 + compressed_increment_t8;
      const s_t pa_p0_1_0 = compressed_increment_t16 + compressed_increment_t18 + compressed_increment_t19 + compressed_increment_t6;
      const s_t pa_p0_1_1 = compressed_increment_t20 + compressed_increment_t22 + compressed_increment_t23;
      const s_t pa_p0_1_2 = compressed_increment_t22 + compressed_increment_t24 + compressed_increment_t6 + ((s_t(1) / s_t(10)))*hx_8;
      const s_t pa_p0_2_0 = -compressed_increment_t1 + compressed_increment_t17 + compressed_increment_t23 + compressed_increment_t25;
      const s_t pa_p0_2_1 = compressed_increment_t11 + compressed_increment_t18 + compressed_increment_t21 + compressed_increment_t9 + ((s_t(1) / s_t(10)))*hx_2;
      const s_t pa_p0_2_2 = compressed_increment_t11 + compressed_increment_t24 + compressed_increment_t25 + ((s_t(1) / s_t(10)))*hx_9;
      const s_t pa_p0_3_0 = -compressed_increment_t14 - compressed_increment_t17 - compressed_increment_t19 - compressed_increment_t9 - (s_t(2) / s_t(15))*hx_5 - (s_t(2) / s_t(15))*hx_8;
      const s_t pa_p0_3_1 = -compressed_increment_t14 - compressed_increment_t20 - compressed_increment_t26 - (s_t(4) / s_t(15))*hx_5;
      const s_t pa_p0_3_2 = -compressed_increment_t24 - compressed_increment_t26 - compressed_increment_t9 - (s_t(4) / s_t(15))*hx_8;
      const s_t pa_p1_0_0 = compressed_increment_t27 + compressed_increment_t28 + compressed_increment_t32 + compressed_increment_t35;
      const s_t pa_p1_0_1 = compressed_increment_t32 + compressed_increment_t36 + compressed_increment_t37 + compressed_increment_t40;
      const s_t pa_p1_0_2 = compressed_increment_t30 + compressed_increment_t35 + compressed_increment_t40 + compressed_increment_t41 + compressed_increment_t42;
      const s_t pa_p1_1_0 = compressed_increment_t33 + compressed_increment_t43 + compressed_increment_t45 + compressed_increment_t46;
      const s_t pa_p1_1_1 = compressed_increment_t47 + compressed_increment_t49 + compressed_increment_t50;
      const s_t pa_p1_1_2 = compressed_increment_t33 + compressed_increment_t49 + compressed_increment_t51 + ((s_t(1) / s_t(10)))*hy_8;
      const s_t pa_p1_2_0 = -compressed_increment_t28 + compressed_increment_t44 + compressed_increment_t50 + compressed_increment_t52;
      const s_t pa_p1_2_1 = compressed_increment_t36 + compressed_increment_t38 + compressed_increment_t45 + compressed_increment_t48 + ((s_t(1) / s_t(10)))*hy_2;
      const s_t pa_p1_2_2 = compressed_increment_t38 + compressed_increment_t51 + compressed_increment_t52 + ((s_t(1) / s_t(10)))*hy_9;
      const s_t pa_p1_3_0 = -compressed_increment_t36 - compressed_increment_t41 - compressed_increment_t44 - compressed_increment_t46 - (s_t(2) / s_t(15))*hy_5 - (s_t(2) / s_t(15))*hy_8;
      const s_t pa_p1_3_1 = -compressed_increment_t41 - compressed_increment_t47 - compressed_increment_t53 - (s_t(4) / s_t(15))*hy_5;
      const s_t pa_p1_3_2 = -compressed_increment_t36 - compressed_increment_t51 - compressed_increment_t53 - (s_t(4) / s_t(15))*hy_8;
      const s_t pa_p2_0_0 = compressed_increment_t54 + compressed_increment_t55 + compressed_increment_t59 + compressed_increment_t62;
      const s_t pa_p2_0_1 = compressed_increment_t59 + compressed_increment_t63 + compressed_increment_t64 + compressed_increment_t67;
      const s_t pa_p2_0_2 = compressed_increment_t57 + compressed_increment_t62 + compressed_increment_t67 + compressed_increment_t68 + compressed_increment_t69;
      const s_t pa_p2_1_0 = compressed_increment_t60 + compressed_increment_t70 + compressed_increment_t72 + compressed_increment_t73;
      const s_t pa_p2_1_1 = compressed_increment_t74 + compressed_increment_t76 + compressed_increment_t77;
      const s_t pa_p2_1_2 = compressed_increment_t60 + compressed_increment_t76 + compressed_increment_t78 + ((s_t(1) / s_t(10)))*hz_8;
      const s_t pa_p2_2_0 = -compressed_increment_t55 + compressed_increment_t71 + compressed_increment_t77 + compressed_increment_t79;
      const s_t pa_p2_2_1 = compressed_increment_t63 + compressed_increment_t65 + compressed_increment_t72 + compressed_increment_t75 + ((s_t(1) / s_t(10)))*hz_2;
      const s_t pa_p2_2_2 = compressed_increment_t65 + compressed_increment_t78 + compressed_increment_t79 + ((s_t(1) / s_t(10)))*hz_9;
      const s_t pa_p2_3_0 = -compressed_increment_t63 - compressed_increment_t68 - compressed_increment_t71 - compressed_increment_t73 - (s_t(2) / s_t(15))*hz_5 - (s_t(2) / s_t(15))*hz_8;
      const s_t pa_p2_3_1 = -compressed_increment_t68 - compressed_increment_t74 - compressed_increment_t80 - (s_t(4) / s_t(15))*hz_5;
      const s_t pa_p2_3_2 = -compressed_increment_t63 - compressed_increment_t78 - compressed_increment_t80 - (s_t(4) / s_t(15))*hz_8;
      const s_t pa_y0_0_0 = pa_p0_0_0*tangent0 + pa_p0_0_1*tangent1 + pa_p0_0_2*tangent2 + pa_p1_0_0*tangent3 + pa_p1_0_1*tangent4 + pa_p1_0_2*tangent5 + pa_p2_0_0*tangent6 + pa_p2_0_1*tangent7 + pa_p2_0_2*tangent8;
      const s_t pa_y0_0_1 = pa_p0_0_0*tangent1 + pa_p0_0_1*tangent9 + pa_p0_0_2*tangent10 + pa_p1_0_0*tangent11 + pa_p1_0_1*tangent12 + pa_p1_0_2*tangent13 + pa_p2_0_0*tangent14 + pa_p2_0_1*tangent15 + pa_p2_0_2*tangent16;
      const s_t pa_y0_0_2 = pa_p0_0_0*tangent2 + pa_p0_0_1*tangent10 + pa_p0_0_2*tangent17 + pa_p1_0_0*tangent18 + pa_p1_0_1*tangent19 + pa_p1_0_2*tangent20 + pa_p2_0_0*tangent21 + pa_p2_0_1*tangent22 + pa_p2_0_2*tangent23;
      const s_t pa_y0_1_0 = pa_p0_1_0*tangent0 + pa_p0_1_1*tangent1 + pa_p0_1_2*tangent2 + pa_p1_1_0*tangent3 + pa_p1_1_1*tangent4 + pa_p1_1_2*tangent5 + pa_p2_1_0*tangent6 + pa_p2_1_1*tangent7 + pa_p2_1_2*tangent8;
      const s_t pa_y0_1_1 = pa_p0_1_0*tangent1 + pa_p0_1_1*tangent9 + pa_p0_1_2*tangent10 + pa_p1_1_0*tangent11 + pa_p1_1_1*tangent12 + pa_p1_1_2*tangent13 + pa_p2_1_0*tangent14 + pa_p2_1_1*tangent15 + pa_p2_1_2*tangent16;
      const s_t pa_y0_1_2 = pa_p0_1_0*tangent2 + pa_p0_1_1*tangent10 + pa_p0_1_2*tangent17 + pa_p1_1_0*tangent18 + pa_p1_1_1*tangent19 + pa_p1_1_2*tangent20 + pa_p2_1_0*tangent21 + pa_p2_1_1*tangent22 + pa_p2_1_2*tangent23;
      const s_t pa_y0_2_0 = pa_p0_2_0*tangent0 + pa_p0_2_1*tangent1 + pa_p0_2_2*tangent2 + pa_p1_2_0*tangent3 + pa_p1_2_1*tangent4 + pa_p1_2_2*tangent5 + pa_p2_2_0*tangent6 + pa_p2_2_1*tangent7 + pa_p2_2_2*tangent8;
      const s_t pa_y0_2_1 = pa_p0_2_0*tangent1 + pa_p0_2_1*tangent9 + pa_p0_2_2*tangent10 + pa_p1_2_0*tangent11 + pa_p1_2_1*tangent12 + pa_p1_2_2*tangent13 + pa_p2_2_0*tangent14 + pa_p2_2_1*tangent15 + pa_p2_2_2*tangent16;
      const s_t pa_y0_2_2 = pa_p0_2_0*tangent2 + pa_p0_2_1*tangent10 + pa_p0_2_2*tangent17 + pa_p1_2_0*tangent18 + pa_p1_2_1*tangent19 + pa_p1_2_2*tangent20 + pa_p2_2_0*tangent21 + pa_p2_2_1*tangent22 + pa_p2_2_2*tangent23;
      const s_t pa_y0_3_0 = pa_p0_3_0*tangent0 + pa_p0_3_1*tangent1 + pa_p0_3_2*tangent2 + pa_p1_3_0*tangent3 + pa_p1_3_1*tangent4 + pa_p1_3_2*tangent5 + pa_p2_3_0*tangent6 + pa_p2_3_1*tangent7 + pa_p2_3_2*tangent8;
      const s_t pa_y0_3_1 = pa_p0_3_0*tangent1 + pa_p0_3_1*tangent9 + pa_p0_3_2*tangent10 + pa_p1_3_0*tangent11 + pa_p1_3_1*tangent12 + pa_p1_3_2*tangent13 + pa_p2_3_0*tangent14 + pa_p2_3_1*tangent15 + pa_p2_3_2*tangent16;
      const s_t pa_y0_3_2 = pa_p0_3_0*tangent2 + pa_p0_3_1*tangent10 + pa_p0_3_2*tangent17 + pa_p1_3_0*tangent18 + pa_p1_3_1*tangent19 + pa_p1_3_2*tangent20 + pa_p2_3_0*tangent21 + pa_p2_3_1*tangent22 + pa_p2_3_2*tangent23;
      const s_t pa_y1_0_0 = pa_p0_0_0*tangent3 + pa_p0_0_1*tangent11 + pa_p0_0_2*tangent18 + pa_p1_0_0*tangent24 + pa_p1_0_1*tangent25 + pa_p1_0_2*tangent26 + pa_p2_0_0*tangent27 + pa_p2_0_1*tangent28 + pa_p2_0_2*tangent29;
      const s_t pa_y1_0_1 = pa_p0_0_0*tangent4 + pa_p0_0_1*tangent12 + pa_p0_0_2*tangent19 + pa_p1_0_0*tangent25 + pa_p1_0_1*tangent30 + pa_p1_0_2*tangent31 + pa_p2_0_0*tangent32 + pa_p2_0_1*tangent33 + pa_p2_0_2*tangent34;
      const s_t pa_y1_0_2 = pa_p0_0_0*tangent5 + pa_p0_0_1*tangent13 + pa_p0_0_2*tangent20 + pa_p1_0_0*tangent26 + pa_p1_0_1*tangent31 + pa_p1_0_2*tangent35 + pa_p2_0_0*tangent36 + pa_p2_0_1*tangent37 + pa_p2_0_2*tangent38;
      const s_t pa_y1_1_0 = pa_p0_1_0*tangent3 + pa_p0_1_1*tangent11 + pa_p0_1_2*tangent18 + pa_p1_1_0*tangent24 + pa_p1_1_1*tangent25 + pa_p1_1_2*tangent26 + pa_p2_1_0*tangent27 + pa_p2_1_1*tangent28 + pa_p2_1_2*tangent29;
      const s_t pa_y1_1_1 = pa_p0_1_0*tangent4 + pa_p0_1_1*tangent12 + pa_p0_1_2*tangent19 + pa_p1_1_0*tangent25 + pa_p1_1_1*tangent30 + pa_p1_1_2*tangent31 + pa_p2_1_0*tangent32 + pa_p2_1_1*tangent33 + pa_p2_1_2*tangent34;
      const s_t pa_y1_1_2 = pa_p0_1_0*tangent5 + pa_p0_1_1*tangent13 + pa_p0_1_2*tangent20 + pa_p1_1_0*tangent26 + pa_p1_1_1*tangent31 + pa_p1_1_2*tangent35 + pa_p2_1_0*tangent36 + pa_p2_1_1*tangent37 + pa_p2_1_2*tangent38;
      const s_t pa_y1_2_0 = pa_p0_2_0*tangent3 + pa_p0_2_1*tangent11 + pa_p0_2_2*tangent18 + pa_p1_2_0*tangent24 + pa_p1_2_1*tangent25 + pa_p1_2_2*tangent26 + pa_p2_2_0*tangent27 + pa_p2_2_1*tangent28 + pa_p2_2_2*tangent29;
      const s_t pa_y1_2_1 = pa_p0_2_0*tangent4 + pa_p0_2_1*tangent12 + pa_p0_2_2*tangent19 + pa_p1_2_0*tangent25 + pa_p1_2_1*tangent30 + pa_p1_2_2*tangent31 + pa_p2_2_0*tangent32 + pa_p2_2_1*tangent33 + pa_p2_2_2*tangent34;
      const s_t pa_y1_2_2 = pa_p0_2_0*tangent5 + pa_p0_2_1*tangent13 + pa_p0_2_2*tangent20 + pa_p1_2_0*tangent26 + pa_p1_2_1*tangent31 + pa_p1_2_2*tangent35 + pa_p2_2_0*tangent36 + pa_p2_2_1*tangent37 + pa_p2_2_2*tangent38;
      const s_t pa_y1_3_0 = pa_p0_3_0*tangent3 + pa_p0_3_1*tangent11 + pa_p0_3_2*tangent18 + pa_p1_3_0*tangent24 + pa_p1_3_1*tangent25 + pa_p1_3_2*tangent26 + pa_p2_3_0*tangent27 + pa_p2_3_1*tangent28 + pa_p2_3_2*tangent29;
      const s_t pa_y1_3_1 = pa_p0_3_0*tangent4 + pa_p0_3_1*tangent12 + pa_p0_3_2*tangent19 + pa_p1_3_0*tangent25 + pa_p1_3_1*tangent30 + pa_p1_3_2*tangent31 + pa_p2_3_0*tangent32 + pa_p2_3_1*tangent33 + pa_p2_3_2*tangent34;
      const s_t pa_y1_3_2 = pa_p0_3_0*tangent5 + pa_p0_3_1*tangent13 + pa_p0_3_2*tangent20 + pa_p1_3_0*tangent26 + pa_p1_3_1*tangent31 + pa_p1_3_2*tangent35 + pa_p2_3_0*tangent36 + pa_p2_3_1*tangent37 + pa_p2_3_2*tangent38;
      const s_t pa_y2_0_0 = pa_p0_0_0*tangent6 + pa_p0_0_1*tangent14 + pa_p0_0_2*tangent21 + pa_p1_0_0*tangent27 + pa_p1_0_1*tangent32 + pa_p1_0_2*tangent36 + pa_p2_0_0*tangent39 + pa_p2_0_1*tangent40 + pa_p2_0_2*tangent41;
      const s_t pa_y2_0_1 = pa_p0_0_0*tangent7 + pa_p0_0_1*tangent15 + pa_p0_0_2*tangent22 + pa_p1_0_0*tangent28 + pa_p1_0_1*tangent33 + pa_p1_0_2*tangent37 + pa_p2_0_0*tangent40 + pa_p2_0_1*tangent42 + pa_p2_0_2*tangent43;
      const s_t pa_y2_0_2 = pa_p0_0_0*tangent8 + pa_p0_0_1*tangent16 + pa_p0_0_2*tangent23 + pa_p1_0_0*tangent29 + pa_p1_0_1*tangent34 + pa_p1_0_2*tangent38 + pa_p2_0_0*tangent41 + pa_p2_0_1*tangent43 + pa_p2_0_2*tangent44;
      const s_t pa_y2_1_0 = pa_p0_1_0*tangent6 + pa_p0_1_1*tangent14 + pa_p0_1_2*tangent21 + pa_p1_1_0*tangent27 + pa_p1_1_1*tangent32 + pa_p1_1_2*tangent36 + pa_p2_1_0*tangent39 + pa_p2_1_1*tangent40 + pa_p2_1_2*tangent41;
      const s_t pa_y2_1_1 = pa_p0_1_0*tangent7 + pa_p0_1_1*tangent15 + pa_p0_1_2*tangent22 + pa_p1_1_0*tangent28 + pa_p1_1_1*tangent33 + pa_p1_1_2*tangent37 + pa_p2_1_0*tangent40 + pa_p2_1_1*tangent42 + pa_p2_1_2*tangent43;
      const s_t pa_y2_1_2 = pa_p0_1_0*tangent8 + pa_p0_1_1*tangent16 + pa_p0_1_2*tangent23 + pa_p1_1_0*tangent29 + pa_p1_1_1*tangent34 + pa_p1_1_2*tangent38 + pa_p2_1_0*tangent41 + pa_p2_1_1*tangent43 + pa_p2_1_2*tangent44;
      const s_t pa_y2_2_0 = pa_p0_2_0*tangent6 + pa_p0_2_1*tangent14 + pa_p0_2_2*tangent21 + pa_p1_2_0*tangent27 + pa_p1_2_1*tangent32 + pa_p1_2_2*tangent36 + pa_p2_2_0*tangent39 + pa_p2_2_1*tangent40 + pa_p2_2_2*tangent41;
      const s_t pa_y2_2_1 = pa_p0_2_0*tangent7 + pa_p0_2_1*tangent15 + pa_p0_2_2*tangent22 + pa_p1_2_0*tangent28 + pa_p1_2_1*tangent33 + pa_p1_2_2*tangent37 + pa_p2_2_0*tangent40 + pa_p2_2_1*tangent42 + pa_p2_2_2*tangent43;
      const s_t pa_y2_2_2 = pa_p0_2_0*tangent8 + pa_p0_2_1*tangent16 + pa_p0_2_2*tangent23 + pa_p1_2_0*tangent29 + pa_p1_2_1*tangent34 + pa_p1_2_2*tangent38 + pa_p2_2_0*tangent41 + pa_p2_2_1*tangent43 + pa_p2_2_2*tangent44;
      const s_t pa_y2_3_0 = pa_p0_3_0*tangent6 + pa_p0_3_1*tangent14 + pa_p0_3_2*tangent21 + pa_p1_3_0*tangent27 + pa_p1_3_1*tangent32 + pa_p1_3_2*tangent36 + pa_p2_3_0*tangent39 + pa_p2_3_1*tangent40 + pa_p2_3_2*tangent41;
      const s_t pa_y2_3_1 = pa_p0_3_0*tangent7 + pa_p0_3_1*tangent15 + pa_p0_3_2*tangent22 + pa_p1_3_0*tangent28 + pa_p1_3_1*tangent33 + pa_p1_3_2*tangent37 + pa_p2_3_0*tangent40 + pa_p2_3_1*tangent42 + pa_p2_3_2*tangent43;
      const s_t pa_y2_3_2 = pa_p0_3_0*tangent8 + pa_p0_3_1*tangent16 + pa_p0_3_2*tangent23 + pa_p1_3_0*tangent29 + pa_p1_3_1*tangent34 + pa_p1_3_2*tangent38 + pa_p2_3_0*tangent41 + pa_p2_3_1*tangent43 + pa_p2_3_2*tangent44;
      const s_t mixed_t0 = ((s_t(15) / s_t(2)))*pa_y0_1_0;
      const s_t mixed_t1 = ((s_t(15) / s_t(2)))*pa_y0_2_0;
      const s_t mixed_t2 = ((s_t(15) / s_t(2)))*pa_y0_1_1;
      const s_t mixed_t3 = ((s_t(15) / s_t(2)))*pa_y0_2_1;
      const s_t mixed_t4 = ((s_t(15) / s_t(2)))*pa_y0_1_2;
      const s_t mixed_t5 = ((s_t(15) / s_t(2)))*pa_y0_2_2;
      const s_t mixed_t6 = -(s_t(15) / s_t(2))*pa_y0_0_0;
      const s_t mixed_t7 = s_t(6)*pa_y0_3_0;
      const s_t mixed_t8 = -(s_t(15) / s_t(2))*pa_y0_0_1;
      const s_t mixed_t9 = s_t(6)*pa_y0_3_1;
      const s_t mixed_t10 = -(s_t(15) / s_t(2))*pa_y0_0_2;
      const s_t mixed_t11 = s_t(6)*pa_y0_3_2;
      const s_t mixed_t12 = ((s_t(15) / s_t(2)))*pa_y1_1_0;
      const s_t mixed_t13 = ((s_t(15) / s_t(2)))*pa_y1_2_0;
      const s_t mixed_t14 = ((s_t(15) / s_t(2)))*pa_y1_1_1;
      const s_t mixed_t15 = ((s_t(15) / s_t(2)))*pa_y1_2_1;
      const s_t mixed_t16 = ((s_t(15) / s_t(2)))*pa_y1_1_2;
      const s_t mixed_t17 = ((s_t(15) / s_t(2)))*pa_y1_2_2;
      const s_t mixed_t18 = -(s_t(15) / s_t(2))*pa_y1_0_0;
      const s_t mixed_t19 = s_t(6)*pa_y1_3_0;
      const s_t mixed_t20 = -(s_t(15) / s_t(2))*pa_y1_0_1;
      const s_t mixed_t21 = s_t(6)*pa_y1_3_1;
      const s_t mixed_t22 = -(s_t(15) / s_t(2))*pa_y1_0_2;
      const s_t mixed_t23 = s_t(6)*pa_y1_3_2;
      const s_t mixed_t24 = ((s_t(15) / s_t(2)))*pa_y2_1_0;
      const s_t mixed_t25 = ((s_t(15) / s_t(2)))*pa_y2_2_0;
      const s_t mixed_t26 = ((s_t(15) / s_t(2)))*pa_y2_1_1;
      const s_t mixed_t27 = ((s_t(15) / s_t(2)))*pa_y2_2_1;
      const s_t mixed_t28 = ((s_t(15) / s_t(2)))*pa_y2_1_2;
      const s_t mixed_t29 = ((s_t(15) / s_t(2)))*pa_y2_2_2;
      const s_t mixed_t30 = -(s_t(15) / s_t(2))*pa_y2_0_0;
      const s_t mixed_t31 = s_t(6)*pa_y2_3_0;
      const s_t mixed_t32 = -(s_t(15) / s_t(2))*pa_y2_0_1;
      const s_t mixed_t33 = s_t(6)*pa_y2_3_1;
      const s_t mixed_t34 = -(s_t(15) / s_t(2))*pa_y2_0_2;
      const s_t mixed_t35 = s_t(6)*pa_y2_3_2;
      const s_t pa_q0_0_0 = -mixed_t0 - mixed_t1 + s_t(15)*pa_y0_0_0;
      const s_t pa_q0_0_1 = -mixed_t2 - mixed_t3 + s_t(15)*pa_y0_0_1;
      const s_t pa_q0_0_2 = -mixed_t4 - mixed_t5 + s_t(15)*pa_y0_0_2;
      const s_t pa_q0_1_0 = mixed_t1 + mixed_t6 + mixed_t7 + s_t(21)*pa_y0_1_0;
      const s_t pa_q0_1_1 = mixed_t3 + mixed_t8 + mixed_t9 + s_t(21)*pa_y0_1_1;
      const s_t pa_q0_1_2 = mixed_t10 + mixed_t11 + mixed_t5 + s_t(21)*pa_y0_1_2;
      const s_t pa_q0_2_0 = mixed_t0 + mixed_t6 + s_t(15)*pa_y0_2_0;
      const s_t pa_q0_2_1 = mixed_t2 + mixed_t8 + s_t(15)*pa_y0_2_1;
      const s_t pa_q0_2_2 = mixed_t10 + mixed_t4 + s_t(15)*pa_y0_2_2;
      const s_t pa_q0_3_0 = mixed_t7 + s_t(6)*pa_y0_1_0;
      const s_t pa_q0_3_1 = mixed_t9 + s_t(6)*pa_y0_1_1;
      const s_t pa_q0_3_2 = mixed_t11 + s_t(6)*pa_y0_1_2;
      const s_t pa_q1_0_0 = -mixed_t12 - mixed_t13 + s_t(15)*pa_y1_0_0;
      const s_t pa_q1_0_1 = -mixed_t14 - mixed_t15 + s_t(15)*pa_y1_0_1;
      const s_t pa_q1_0_2 = -mixed_t16 - mixed_t17 + s_t(15)*pa_y1_0_2;
      const s_t pa_q1_1_0 = mixed_t13 + mixed_t18 + mixed_t19 + s_t(21)*pa_y1_1_0;
      const s_t pa_q1_1_1 = mixed_t15 + mixed_t20 + mixed_t21 + s_t(21)*pa_y1_1_1;
      const s_t pa_q1_1_2 = mixed_t17 + mixed_t22 + mixed_t23 + s_t(21)*pa_y1_1_2;
      const s_t pa_q1_2_0 = mixed_t12 + mixed_t18 + s_t(15)*pa_y1_2_0;
      const s_t pa_q1_2_1 = mixed_t14 + mixed_t20 + s_t(15)*pa_y1_2_1;
      const s_t pa_q1_2_2 = mixed_t16 + mixed_t22 + s_t(15)*pa_y1_2_2;
      const s_t pa_q1_3_0 = mixed_t19 + s_t(6)*pa_y1_1_0;
      const s_t pa_q1_3_1 = mixed_t21 + s_t(6)*pa_y1_1_1;
      const s_t pa_q1_3_2 = mixed_t23 + s_t(6)*pa_y1_1_2;
      const s_t pa_q2_0_0 = -mixed_t24 - mixed_t25 + s_t(15)*pa_y2_0_0;
      const s_t pa_q2_0_1 = -mixed_t26 - mixed_t27 + s_t(15)*pa_y2_0_1;
      const s_t pa_q2_0_2 = -mixed_t28 - mixed_t29 + s_t(15)*pa_y2_0_2;
      const s_t pa_q2_1_0 = mixed_t25 + mixed_t30 + mixed_t31 + s_t(21)*pa_y2_1_0;
      const s_t pa_q2_1_1 = mixed_t27 + mixed_t32 + mixed_t33 + s_t(21)*pa_y2_1_1;
      const s_t pa_q2_1_2 = mixed_t29 + mixed_t34 + mixed_t35 + s_t(21)*pa_y2_1_2;
      const s_t pa_q2_2_0 = mixed_t24 + mixed_t30 + s_t(15)*pa_y2_2_0;
      const s_t pa_q2_2_1 = mixed_t26 + mixed_t32 + s_t(15)*pa_y2_2_1;
      const s_t pa_q2_2_2 = mixed_t28 + mixed_t34 + s_t(15)*pa_y2_2_2;
      const s_t pa_q2_3_0 = mixed_t31 + s_t(6)*pa_y2_1_0;
      const s_t pa_q2_3_1 = mixed_t33 + s_t(6)*pa_y2_1_1;
      const s_t pa_q2_3_2 = mixed_t35 + s_t(6)*pa_y2_1_2;
      const s_t output_t0 = ((s_t(1) / s_t(30)))*pa_q0_3_1;
      const s_t output_t1 = ((s_t(1) / s_t(30)))*pa_q0_3_2;
      const s_t output_t2 = ((s_t(1) / s_t(30)))*pa_q0_1_0;
      const s_t output_t3 = ((s_t(1) / s_t(30)))*pa_q0_2_0;
      const s_t output_t4 = ((s_t(1) / s_t(30)))*pa_q0_2_2;
      const s_t output_t5 = output_t2 + output_t3 + output_t4;
      const s_t output_t6 = ((s_t(1) / s_t(30)))*pa_q0_1_1;
      const s_t output_t7 = ((s_t(1) / s_t(30)))*pa_q0_1_2;
      const s_t output_t8 = ((s_t(1) / s_t(30)))*pa_q0_2_1;
      const s_t output_t9 = output_t6 + output_t7 + output_t8;
      const s_t output_t10 = ((s_t(1) / s_t(30)))*pa_q0_0_0;
      const s_t output_t11 = ((s_t(1) / s_t(30)))*pa_q0_0_1;
      const s_t output_t12 = -output_t4;
      const s_t output_t13 = ((s_t(1) / s_t(30)))*pa_q0_0_2;
      const s_t output_t14 = output_t13 - output_t7;
      const s_t output_t15 = ((s_t(2) / s_t(15)))*pa_q0_3_0;
      const s_t output_t16 = -output_t15;
      const s_t output_t17 = ((s_t(4) / s_t(15)))*pa_q0_3_2;
      const s_t output_t18 = ((s_t(1) / s_t(10)))*pa_q0_1_2;
      const s_t output_t19 = output_t11 - output_t8 + ((s_t(1) / s_t(10)))*pa_q0_1_1 - (s_t(4) / s_t(15))*pa_q0_3_1;
      const s_t output_t20 = output_t10 + output_t16 - output_t2 + ((s_t(1) / s_t(10)))*pa_q0_2_0;
      const s_t output_t21 = ((s_t(2) / s_t(15)))*pa_q0_3_2;
      const s_t output_t22 = ((s_t(1) / s_t(10)))*pa_q0_2_2;
      const s_t output_t23 = -output_t10 + output_t15;
      const s_t output_t24 = -output_t11 + ((s_t(2) / s_t(15)))*pa_q0_3_1;
      const s_t output_t25 = -output_t13;
      const s_t output_t26 = ((s_t(1) / s_t(30)))*pa_q1_3_1;
      const s_t output_t27 = ((s_t(1) / s_t(30)))*pa_q1_3_2;
      const s_t output_t28 = ((s_t(1) / s_t(30)))*pa_q1_1_0;
      const s_t output_t29 = ((s_t(1) / s_t(30)))*pa_q1_2_0;
      const s_t output_t30 = ((s_t(1) / s_t(30)))*pa_q1_2_2;
      const s_t output_t31 = output_t28 + output_t29 + output_t30;
      const s_t output_t32 = ((s_t(1) / s_t(30)))*pa_q1_1_1;
      const s_t output_t33 = ((s_t(1) / s_t(30)))*pa_q1_1_2;
      const s_t output_t34 = ((s_t(1) / s_t(30)))*pa_q1_2_1;
      const s_t output_t35 = output_t32 + output_t33 + output_t34;
      const s_t output_t36 = ((s_t(1) / s_t(30)))*pa_q1_0_0;
      const s_t output_t37 = ((s_t(1) / s_t(30)))*pa_q1_0_1;
      const s_t output_t38 = -output_t30;
      const s_t output_t39 = ((s_t(1) / s_t(30)))*pa_q1_0_2;
      const s_t output_t40 = -output_t33 + output_t39;
      const s_t output_t41 = ((s_t(2) / s_t(15)))*pa_q1_3_0;
      const s_t output_t42 = -output_t41;
      const s_t output_t43 = ((s_t(4) / s_t(15)))*pa_q1_3_2;
      const s_t output_t44 = ((s_t(1) / s_t(10)))*pa_q1_1_2;
      const s_t output_t45 = -output_t34 + output_t37 + ((s_t(1) / s_t(10)))*pa_q1_1_1 - (s_t(4) / s_t(15))*pa_q1_3_1;
      const s_t output_t46 = -output_t28 + output_t36 + output_t42 + ((s_t(1) / s_t(10)))*pa_q1_2_0;
      const s_t output_t47 = ((s_t(2) / s_t(15)))*pa_q1_3_2;
      const s_t output_t48 = ((s_t(1) / s_t(10)))*pa_q1_2_2;
      const s_t output_t49 = -output_t36 + output_t41;
      const s_t output_t50 = -output_t37 + ((s_t(2) / s_t(15)))*pa_q1_3_1;
      const s_t output_t51 = -output_t39;
      const s_t output_t52 = ((s_t(1) / s_t(30)))*pa_q2_3_1;
      const s_t output_t53 = ((s_t(1) / s_t(30)))*pa_q2_3_2;
      const s_t output_t54 = ((s_t(1) / s_t(30)))*pa_q2_1_0;
      const s_t output_t55 = ((s_t(1) / s_t(30)))*pa_q2_2_0;
      const s_t output_t56 = ((s_t(1) / s_t(30)))*pa_q2_2_2;
      const s_t output_t57 = output_t54 + output_t55 + output_t56;
      const s_t output_t58 = ((s_t(1) / s_t(30)))*pa_q2_1_1;
      const s_t output_t59 = ((s_t(1) / s_t(30)))*pa_q2_1_2;
      const s_t output_t60 = ((s_t(1) / s_t(30)))*pa_q2_2_1;
      const s_t output_t61 = output_t58 + output_t59 + output_t60;
      const s_t output_t62 = ((s_t(1) / s_t(30)))*pa_q2_0_0;
      const s_t output_t63 = ((s_t(1) / s_t(30)))*pa_q2_0_1;
      const s_t output_t64 = -output_t56;
      const s_t output_t65 = ((s_t(1) / s_t(30)))*pa_q2_0_2;
      const s_t output_t66 = -output_t59 + output_t65;
      const s_t output_t67 = ((s_t(2) / s_t(15)))*pa_q2_3_0;
      const s_t output_t68 = -output_t67;
      const s_t output_t69 = ((s_t(4) / s_t(15)))*pa_q2_3_2;
      const s_t output_t70 = ((s_t(1) / s_t(10)))*pa_q2_1_2;
      const s_t output_t71 = -output_t60 + output_t63 + ((s_t(1) / s_t(10)))*pa_q2_1_1 - (s_t(4) / s_t(15))*pa_q2_3_1;
      const s_t output_t72 = -output_t54 + output_t62 + output_t68 + ((s_t(1) / s_t(10)))*pa_q2_2_0;
      const s_t output_t73 = ((s_t(2) / s_t(15)))*pa_q2_3_2;
      const s_t output_t74 = ((s_t(1) / s_t(10)))*pa_q2_2_2;
      const s_t output_t75 = -output_t62 + output_t67;
      const s_t output_t76 = -output_t63 + ((s_t(2) / s_t(15)))*pa_q2_3_1;
      const s_t output_t77 = -output_t65;
      const s_t element_out0_0 = -output_t0 - output_t1 + output_t5 + output_t9 + ((s_t(1) / s_t(10)))*pa_q0_0_0 + ((s_t(1) / s_t(10)))*pa_q0_0_1 + ((s_t(1) / s_t(10)))*pa_q0_0_2 - (s_t(1) / s_t(30))*pa_q0_3_0;
      const s_t element_out0_1 = output_t10 - output_t3 + ((s_t(1) / s_t(10)))*pa_q0_1_0 - (s_t(1) / s_t(10))*pa_q0_3_0;
      const s_t element_out0_2 = output_t0 + output_t11 - output_t6 + ((s_t(1) / s_t(10)))*pa_q0_2_1;
      const s_t element_out0_3 = output_t1 + output_t12 + output_t14;
      const s_t element_out0_4 = -output_t12 - output_t13 - output_t16 + output_t17 - output_t18 - output_t19 - (s_t(2) / s_t(15))*pa_q0_0_0 - (s_t(2) / s_t(15))*pa_q0_1_0;
      const s_t element_out0_5 = output_t19 + output_t20;
      const s_t element_out0_6 = -output_t14 - output_t20 + output_t21 - output_t22 - (s_t(2) / s_t(15))*pa_q0_0_1 - (s_t(2) / s_t(15))*pa_q0_2_1;
      const s_t element_out0_7 = output_t2 + output_t23 + output_t24 + output_t3 + output_t6 + output_t8 - (s_t(2) / s_t(15))*pa_q0_0_2;
      const s_t element_out0_8 = -output_t17 + output_t18 - output_t23 - output_t25 - output_t5;
      const s_t element_out0_9 = -output_t21 + output_t22 - output_t24 - output_t25 - output_t9;
      const s_t element_out1_0 = -output_t26 - output_t27 + output_t31 + output_t35 + ((s_t(1) / s_t(10)))*pa_q1_0_0 + ((s_t(1) / s_t(10)))*pa_q1_0_1 + ((s_t(1) / s_t(10)))*pa_q1_0_2 - (s_t(1) / s_t(30))*pa_q1_3_0;
      const s_t element_out1_1 = -output_t29 + output_t36 + ((s_t(1) / s_t(10)))*pa_q1_1_0 - (s_t(1) / s_t(10))*pa_q1_3_0;
      const s_t element_out1_2 = output_t26 - output_t32 + output_t37 + ((s_t(1) / s_t(10)))*pa_q1_2_1;
      const s_t element_out1_3 = output_t27 + output_t38 + output_t40;
      const s_t element_out1_4 = -output_t38 - output_t39 - output_t42 + output_t43 - output_t44 - output_t45 - (s_t(2) / s_t(15))*pa_q1_0_0 - (s_t(2) / s_t(15))*pa_q1_1_0;
      const s_t element_out1_5 = output_t45 + output_t46;
      const s_t element_out1_6 = -output_t40 - output_t46 + output_t47 - output_t48 - (s_t(2) / s_t(15))*pa_q1_0_1 - (s_t(2) / s_t(15))*pa_q1_2_1;
      const s_t element_out1_7 = output_t28 + output_t29 + output_t32 + output_t34 + output_t49 + output_t50 - (s_t(2) / s_t(15))*pa_q1_0_2;
      const s_t element_out1_8 = -output_t31 - output_t43 + output_t44 - output_t49 - output_t51;
      const s_t element_out1_9 = -output_t35 - output_t47 + output_t48 - output_t50 - output_t51;
      const s_t element_out2_0 = -output_t52 - output_t53 + output_t57 + output_t61 + ((s_t(1) / s_t(10)))*pa_q2_0_0 + ((s_t(1) / s_t(10)))*pa_q2_0_1 + ((s_t(1) / s_t(10)))*pa_q2_0_2 - (s_t(1) / s_t(30))*pa_q2_3_0;
      const s_t element_out2_1 = -output_t55 + output_t62 + ((s_t(1) / s_t(10)))*pa_q2_1_0 - (s_t(1) / s_t(10))*pa_q2_3_0;
      const s_t element_out2_2 = output_t52 - output_t58 + output_t63 + ((s_t(1) / s_t(10)))*pa_q2_2_1;
      const s_t element_out2_3 = output_t53 + output_t64 + output_t66;
      const s_t element_out2_4 = -output_t64 - output_t65 - output_t68 + output_t69 - output_t70 - output_t71 - (s_t(2) / s_t(15))*pa_q2_0_0 - (s_t(2) / s_t(15))*pa_q2_1_0;
      const s_t element_out2_5 = output_t71 + output_t72;
      const s_t element_out2_6 = -output_t66 - output_t72 + output_t73 - output_t74 - (s_t(2) / s_t(15))*pa_q2_0_1 - (s_t(2) / s_t(15))*pa_q2_2_1;
      const s_t element_out2_7 = output_t54 + output_t55 + output_t58 + output_t60 + output_t75 + output_t76 - (s_t(2) / s_t(15))*pa_q2_0_2;
      const s_t element_out2_8 = -output_t57 - output_t69 + output_t70 - output_t75 - output_t77;
      const s_t element_out2_9 = -output_t61 - output_t73 + output_t74 - output_t76 - output_t77;
      bout0_0[lane] = element_out0_0;
      bout0_1[lane] = element_out0_1;
      bout0_2[lane] = element_out0_2;
      bout0_3[lane] = element_out0_3;
      bout0_4[lane] = element_out0_4;
      bout0_5[lane] = element_out0_5;
      bout0_6[lane] = element_out0_6;
      bout0_7[lane] = element_out0_7;
      bout0_8[lane] = element_out0_8;
      bout0_9[lane] = element_out0_9;
      bout1_0[lane] = element_out1_0;
      bout1_1[lane] = element_out1_1;
      bout1_2[lane] = element_out1_2;
      bout1_3[lane] = element_out1_3;
      bout1_4[lane] = element_out1_4;
      bout1_5[lane] = element_out1_5;
      bout1_6[lane] = element_out1_6;
      bout1_7[lane] = element_out1_7;
      bout1_8[lane] = element_out1_8;
      bout1_9[lane] = element_out1_9;
      bout2_0[lane] = element_out2_0;
      bout2_1[lane] = element_out2_1;
      bout2_2[lane] = element_out2_2;
      bout2_3[lane] = element_out2_3;
      bout2_4[lane] = element_out2_4;
      bout2_5[lane] = element_out2_5;
      bout2_6[lane] = element_out2_6;
      bout2_7[lane] = element_out2_7;
      bout2_8[lane] = element_out2_8;
      bout2_9[lane] = element_out2_9;
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
      outx[bev4[lane] * out_stride] += bout0_4[lane];
    }
    for (int lane = 0; lane < ne; ++lane) {
      #pragma omp atomic update
      outx[bev5[lane] * out_stride] += bout0_5[lane];
    }
    for (int lane = 0; lane < ne; ++lane) {
      #pragma omp atomic update
      outx[bev6[lane] * out_stride] += bout0_6[lane];
    }
    for (int lane = 0; lane < ne; ++lane) {
      #pragma omp atomic update
      outx[bev7[lane] * out_stride] += bout0_7[lane];
    }
    for (int lane = 0; lane < ne; ++lane) {
      #pragma omp atomic update
      outx[bev8[lane] * out_stride] += bout0_8[lane];
    }
    for (int lane = 0; lane < ne; ++lane) {
      #pragma omp atomic update
      outx[bev9[lane] * out_stride] += bout0_9[lane];
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
      outy[bev4[lane] * out_stride] += bout1_4[lane];
    }
    for (int lane = 0; lane < ne; ++lane) {
      #pragma omp atomic update
      outy[bev5[lane] * out_stride] += bout1_5[lane];
    }
    for (int lane = 0; lane < ne; ++lane) {
      #pragma omp atomic update
      outy[bev6[lane] * out_stride] += bout1_6[lane];
    }
    for (int lane = 0; lane < ne; ++lane) {
      #pragma omp atomic update
      outy[bev7[lane] * out_stride] += bout1_7[lane];
    }
    for (int lane = 0; lane < ne; ++lane) {
      #pragma omp atomic update
      outy[bev8[lane] * out_stride] += bout1_8[lane];
    }
    for (int lane = 0; lane < ne; ++lane) {
      #pragma omp atomic update
      outy[bev9[lane] * out_stride] += bout1_9[lane];
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
    for (int lane = 0; lane < ne; ++lane) {
      #pragma omp atomic update
      outz[bev4[lane] * out_stride] += bout2_4[lane];
    }
    for (int lane = 0; lane < ne; ++lane) {
      #pragma omp atomic update
      outz[bev5[lane] * out_stride] += bout2_5[lane];
    }
    for (int lane = 0; lane < ne; ++lane) {
      #pragma omp atomic update
      outz[bev6[lane] * out_stride] += bout2_6[lane];
    }
    for (int lane = 0; lane < ne; ++lane) {
      #pragma omp atomic update
      outz[bev7[lane] * out_stride] += bout2_7[lane];
    }
    for (int lane = 0; lane < ne; ++lane) {
      #pragma omp atomic update
      outz[bev8[lane] * out_stride] += bout2_8[lane];
    }
    for (int lane = 0; lane < ne; ++lane) {
      #pragma omp atomic update
      outz[bev9[lane] * out_stride] += bout2_9[lane];
    }
  }

  return SFEM_SUCCESS;
}

template <typename s_t, typename tangent_t, int VS>
static SFEM_INLINE int neohookean_ogden_tet10_inexact_apply_stored_packed_two_pass_a_msoa_impl(
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
        uint16_t bev4[VS];
        uint16_t bev5[VS];
        uint16_t bev6[VS];
        uint16_t bev7[VS];
        uint16_t bev8[VS];
        uint16_t bev9[VS];
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          bev0[lane] = elements[0][evb + lane];
          bev1[lane] = elements[1][evb + lane];
          bev2[lane] = elements[2][evb + lane];
          bev3[lane] = elements[3][evb + lane];
          bev4[lane] = elements[4][evb + lane];
          bev5[lane] = elements[5][evb + lane];
          bev6[lane] = elements[6][evb + lane];
          bev7[lane] = elements[7][evb + lane];
          bev8[lane] = elements[8][evb + lane];
          bev9[lane] = elements[9][evb + lane];
        }
        s_t bhx_0[VS];
        s_t bhx_1[VS];
        s_t bhx_2[VS];
        s_t bhx_3[VS];
        s_t bhx_4[VS];
        s_t bhx_5[VS];
        s_t bhx_6[VS];
        s_t bhx_7[VS];
        s_t bhx_8[VS];
        s_t bhx_9[VS];
        s_t bhy_0[VS];
        s_t bhy_1[VS];
        s_t bhy_2[VS];
        s_t bhy_3[VS];
        s_t bhy_4[VS];
        s_t bhy_5[VS];
        s_t bhy_6[VS];
        s_t bhy_7[VS];
        s_t bhy_8[VS];
        s_t bhy_9[VS];
        s_t bhz_0[VS];
        s_t bhz_1[VS];
        s_t bhz_2[VS];
        s_t bhz_3[VS];
        s_t bhz_4[VS];
        s_t bhz_5[VS];
        s_t bhz_6[VS];
        s_t bhz_7[VS];
        s_t bhz_8[VS];
        s_t bhz_9[VS];
        s_t bout0_0[VS];
        s_t bout0_1[VS];
        s_t bout0_2[VS];
        s_t bout0_3[VS];
        s_t bout0_4[VS];
        s_t bout0_5[VS];
        s_t bout0_6[VS];
        s_t bout0_7[VS];
        s_t bout0_8[VS];
        s_t bout0_9[VS];
        s_t bout1_0[VS];
        s_t bout1_1[VS];
        s_t bout1_2[VS];
        s_t bout1_3[VS];
        s_t bout1_4[VS];
        s_t bout1_5[VS];
        s_t bout1_6[VS];
        s_t bout1_7[VS];
        s_t bout1_8[VS];
        s_t bout1_9[VS];
        s_t bout2_0[VS];
        s_t bout2_1[VS];
        s_t bout2_2[VS];
        s_t bout2_3[VS];
        s_t bout2_4[VS];
        s_t bout2_5[VS];
        s_t bout2_6[VS];
        s_t bout2_7[VS];
        s_t bout2_8[VS];
        s_t bout2_9[VS];
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          bhx_0[lane] = pk_h[0 * max_nodes_per_pack + bev0[lane]];
          bhx_1[lane] = pk_h[0 * max_nodes_per_pack + bev1[lane]];
          bhx_2[lane] = pk_h[0 * max_nodes_per_pack + bev2[lane]];
          bhx_3[lane] = pk_h[0 * max_nodes_per_pack + bev3[lane]];
          bhx_4[lane] = pk_h[0 * max_nodes_per_pack + bev4[lane]];
          bhx_5[lane] = pk_h[0 * max_nodes_per_pack + bev5[lane]];
          bhx_6[lane] = pk_h[0 * max_nodes_per_pack + bev6[lane]];
          bhx_7[lane] = pk_h[0 * max_nodes_per_pack + bev7[lane]];
          bhx_8[lane] = pk_h[0 * max_nodes_per_pack + bev8[lane]];
          bhx_9[lane] = pk_h[0 * max_nodes_per_pack + bev9[lane]];
          bhy_0[lane] = pk_h[1 * max_nodes_per_pack + bev0[lane]];
          bhy_1[lane] = pk_h[1 * max_nodes_per_pack + bev1[lane]];
          bhy_2[lane] = pk_h[1 * max_nodes_per_pack + bev2[lane]];
          bhy_3[lane] = pk_h[1 * max_nodes_per_pack + bev3[lane]];
          bhy_4[lane] = pk_h[1 * max_nodes_per_pack + bev4[lane]];
          bhy_5[lane] = pk_h[1 * max_nodes_per_pack + bev5[lane]];
          bhy_6[lane] = pk_h[1 * max_nodes_per_pack + bev6[lane]];
          bhy_7[lane] = pk_h[1 * max_nodes_per_pack + bev7[lane]];
          bhy_8[lane] = pk_h[1 * max_nodes_per_pack + bev8[lane]];
          bhy_9[lane] = pk_h[1 * max_nodes_per_pack + bev9[lane]];
          bhz_0[lane] = pk_h[2 * max_nodes_per_pack + bev0[lane]];
          bhz_1[lane] = pk_h[2 * max_nodes_per_pack + bev1[lane]];
          bhz_2[lane] = pk_h[2 * max_nodes_per_pack + bev2[lane]];
          bhz_3[lane] = pk_h[2 * max_nodes_per_pack + bev3[lane]];
          bhz_4[lane] = pk_h[2 * max_nodes_per_pack + bev4[lane]];
          bhz_5[lane] = pk_h[2 * max_nodes_per_pack + bev5[lane]];
          bhz_6[lane] = pk_h[2 * max_nodes_per_pack + bev6[lane]];
          bhz_7[lane] = pk_h[2 * max_nodes_per_pack + bev7[lane]];
          bhz_8[lane] = pk_h[2 * max_nodes_per_pack + bev8[lane]];
          bhz_9[lane] = pk_h[2 * max_nodes_per_pack + bev9[lane]];
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
          const s_t hx_4 = bhx_4[lane];
          const s_t hx_5 = bhx_5[lane];
          const s_t hx_6 = bhx_6[lane];
          const s_t hx_7 = bhx_7[lane];
          const s_t hx_8 = bhx_8[lane];
          const s_t hx_9 = bhx_9[lane];
          const s_t hy_0 = bhy_0[lane];
          const s_t hy_1 = bhy_1[lane];
          const s_t hy_2 = bhy_2[lane];
          const s_t hy_3 = bhy_3[lane];
          const s_t hy_4 = bhy_4[lane];
          const s_t hy_5 = bhy_5[lane];
          const s_t hy_6 = bhy_6[lane];
          const s_t hy_7 = bhy_7[lane];
          const s_t hy_8 = bhy_8[lane];
          const s_t hy_9 = bhy_9[lane];
          const s_t hz_0 = bhz_0[lane];
          const s_t hz_1 = bhz_1[lane];
          const s_t hz_2 = bhz_2[lane];
          const s_t hz_3 = bhz_3[lane];
          const s_t hz_4 = bhz_4[lane];
          const s_t hz_5 = bhz_5[lane];
          const s_t hz_6 = bhz_6[lane];
          const s_t hz_7 = bhz_7[lane];
          const s_t hz_8 = bhz_8[lane];
          const s_t hz_9 = bhz_9[lane];
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
          const s_t compressed_increment_t0 = -(s_t(2) / s_t(15))*hx_4;
          const s_t compressed_increment_t1 = ((s_t(1) / s_t(30)))*hx_1;
          const s_t compressed_increment_t2 = ((s_t(1) / s_t(30)))*hx_7;
          const s_t compressed_increment_t3 = ((s_t(1) / s_t(10)))*hx_0;
          const s_t compressed_increment_t4 = ((s_t(1) / s_t(30)))*hx_5;
          const s_t compressed_increment_t5 = -compressed_increment_t2 + compressed_increment_t3 + compressed_increment_t4;
          const s_t compressed_increment_t6 = ((s_t(1) / s_t(30)))*hx_6;
          const s_t compressed_increment_t7 = ((s_t(1) / s_t(30)))*hx_8;
          const s_t compressed_increment_t8 = -compressed_increment_t6 + compressed_increment_t7;
          const s_t compressed_increment_t9 = -(s_t(2) / s_t(15))*hx_6;
          const s_t compressed_increment_t10 = ((s_t(1) / s_t(30)))*hx_2;
          const s_t compressed_increment_t11 = ((s_t(1) / s_t(30)))*hx_4;
          const s_t compressed_increment_t12 = ((s_t(1) / s_t(30)))*hx_9;
          const s_t compressed_increment_t13 = -compressed_increment_t11 + compressed_increment_t12;
          const s_t compressed_increment_t14 = -(s_t(2) / s_t(15))*hx_7;
          const s_t compressed_increment_t15 = ((s_t(1) / s_t(30)))*hx_3;
          const s_t compressed_increment_t16 = -compressed_increment_t7;
          const s_t compressed_increment_t17 = ((s_t(1) / s_t(30)))*hx_0;
          const s_t compressed_increment_t18 = compressed_increment_t17 + compressed_increment_t2 - compressed_increment_t4;
          const s_t compressed_increment_t19 = compressed_increment_t0 + ((s_t(1) / s_t(10)))*hx_1;
          const s_t compressed_increment_t20 = -compressed_increment_t10 + compressed_increment_t17;
          const s_t compressed_increment_t21 = -compressed_increment_t12;
          const s_t compressed_increment_t22 = compressed_increment_t21 - (s_t(1) / s_t(10))*hx_4;
          const s_t compressed_increment_t23 = compressed_increment_t2 + ((s_t(1) / s_t(10)))*hx_5;
          const s_t compressed_increment_t24 = -compressed_increment_t15 + compressed_increment_t17;
          const s_t compressed_increment_t25 = compressed_increment_t16 - (s_t(1) / s_t(10))*hx_6;
          const s_t compressed_increment_t26 = -(s_t(4) / s_t(15))*hx_4 + ((s_t(2) / s_t(15)))*hx_9;
          const s_t compressed_increment_t27 = -(s_t(2) / s_t(15))*hy_4;
          const s_t compressed_increment_t28 = ((s_t(1) / s_t(30)))*hy_1;
          const s_t compressed_increment_t29 = ((s_t(1) / s_t(30)))*hy_7;
          const s_t compressed_increment_t30 = ((s_t(1) / s_t(10)))*hy_0;
          const s_t compressed_increment_t31 = ((s_t(1) / s_t(30)))*hy_5;
          const s_t compressed_increment_t32 = -compressed_increment_t29 + compressed_increment_t30 + compressed_increment_t31;
          const s_t compressed_increment_t33 = ((s_t(1) / s_t(30)))*hy_6;
          const s_t compressed_increment_t34 = ((s_t(1) / s_t(30)))*hy_8;
          const s_t compressed_increment_t35 = -compressed_increment_t33 + compressed_increment_t34;
          const s_t compressed_increment_t36 = -(s_t(2) / s_t(15))*hy_6;
          const s_t compressed_increment_t37 = ((s_t(1) / s_t(30)))*hy_2;
          const s_t compressed_increment_t38 = ((s_t(1) / s_t(30)))*hy_4;
          const s_t compressed_increment_t39 = ((s_t(1) / s_t(30)))*hy_9;
          const s_t compressed_increment_t40 = -compressed_increment_t38 + compressed_increment_t39;
          const s_t compressed_increment_t41 = -(s_t(2) / s_t(15))*hy_7;
          const s_t compressed_increment_t42 = ((s_t(1) / s_t(30)))*hy_3;
          const s_t compressed_increment_t43 = -compressed_increment_t34;
          const s_t compressed_increment_t44 = ((s_t(1) / s_t(30)))*hy_0;
          const s_t compressed_increment_t45 = compressed_increment_t29 - compressed_increment_t31 + compressed_increment_t44;
          const s_t compressed_increment_t46 = compressed_increment_t27 + ((s_t(1) / s_t(10)))*hy_1;
          const s_t compressed_increment_t47 = -compressed_increment_t37 + compressed_increment_t44;
          const s_t compressed_increment_t48 = -compressed_increment_t39;
          const s_t compressed_increment_t49 = compressed_increment_t48 - (s_t(1) / s_t(10))*hy_4;
          const s_t compressed_increment_t50 = compressed_increment_t29 + ((s_t(1) / s_t(10)))*hy_5;
          const s_t compressed_increment_t51 = -compressed_increment_t42 + compressed_increment_t44;
          const s_t compressed_increment_t52 = compressed_increment_t43 - (s_t(1) / s_t(10))*hy_6;
          const s_t compressed_increment_t53 = -(s_t(4) / s_t(15))*hy_4 + ((s_t(2) / s_t(15)))*hy_9;
          const s_t compressed_increment_t54 = -(s_t(2) / s_t(15))*hz_4;
          const s_t compressed_increment_t55 = ((s_t(1) / s_t(30)))*hz_1;
          const s_t compressed_increment_t56 = ((s_t(1) / s_t(30)))*hz_7;
          const s_t compressed_increment_t57 = ((s_t(1) / s_t(10)))*hz_0;
          const s_t compressed_increment_t58 = ((s_t(1) / s_t(30)))*hz_5;
          const s_t compressed_increment_t59 = -compressed_increment_t56 + compressed_increment_t57 + compressed_increment_t58;
          const s_t compressed_increment_t60 = ((s_t(1) / s_t(30)))*hz_6;
          const s_t compressed_increment_t61 = ((s_t(1) / s_t(30)))*hz_8;
          const s_t compressed_increment_t62 = -compressed_increment_t60 + compressed_increment_t61;
          const s_t compressed_increment_t63 = -(s_t(2) / s_t(15))*hz_6;
          const s_t compressed_increment_t64 = ((s_t(1) / s_t(30)))*hz_2;
          const s_t compressed_increment_t65 = ((s_t(1) / s_t(30)))*hz_4;
          const s_t compressed_increment_t66 = ((s_t(1) / s_t(30)))*hz_9;
          const s_t compressed_increment_t67 = -compressed_increment_t65 + compressed_increment_t66;
          const s_t compressed_increment_t68 = -(s_t(2) / s_t(15))*hz_7;
          const s_t compressed_increment_t69 = ((s_t(1) / s_t(30)))*hz_3;
          const s_t compressed_increment_t70 = -compressed_increment_t61;
          const s_t compressed_increment_t71 = ((s_t(1) / s_t(30)))*hz_0;
          const s_t compressed_increment_t72 = compressed_increment_t56 - compressed_increment_t58 + compressed_increment_t71;
          const s_t compressed_increment_t73 = compressed_increment_t54 + ((s_t(1) / s_t(10)))*hz_1;
          const s_t compressed_increment_t74 = -compressed_increment_t64 + compressed_increment_t71;
          const s_t compressed_increment_t75 = -compressed_increment_t66;
          const s_t compressed_increment_t76 = compressed_increment_t75 - (s_t(1) / s_t(10))*hz_4;
          const s_t compressed_increment_t77 = compressed_increment_t56 + ((s_t(1) / s_t(10)))*hz_5;
          const s_t compressed_increment_t78 = -compressed_increment_t69 + compressed_increment_t71;
          const s_t compressed_increment_t79 = compressed_increment_t70 - (s_t(1) / s_t(10))*hz_6;
          const s_t compressed_increment_t80 = -(s_t(4) / s_t(15))*hz_4 + ((s_t(2) / s_t(15)))*hz_9;
          const s_t pa_p0_0_0 = compressed_increment_t0 + compressed_increment_t1 + compressed_increment_t5 + compressed_increment_t8;
          const s_t pa_p0_0_1 = compressed_increment_t10 + compressed_increment_t13 + compressed_increment_t5 + compressed_increment_t9;
          const s_t pa_p0_0_2 = compressed_increment_t13 + compressed_increment_t14 + compressed_increment_t15 + compressed_increment_t3 + compressed_increment_t8;
          const s_t pa_p0_1_0 = compressed_increment_t16 + compressed_increment_t18 + compressed_increment_t19 + compressed_increment_t6;
          const s_t pa_p0_1_1 = compressed_increment_t20 + compressed_increment_t22 + compressed_increment_t23;
          const s_t pa_p0_1_2 = compressed_increment_t22 + compressed_increment_t24 + compressed_increment_t6 + ((s_t(1) / s_t(10)))*hx_8;
          const s_t pa_p0_2_0 = -compressed_increment_t1 + compressed_increment_t17 + compressed_increment_t23 + compressed_increment_t25;
          const s_t pa_p0_2_1 = compressed_increment_t11 + compressed_increment_t18 + compressed_increment_t21 + compressed_increment_t9 + ((s_t(1) / s_t(10)))*hx_2;
          const s_t pa_p0_2_2 = compressed_increment_t11 + compressed_increment_t24 + compressed_increment_t25 + ((s_t(1) / s_t(10)))*hx_9;
          const s_t pa_p0_3_0 = -compressed_increment_t14 - compressed_increment_t17 - compressed_increment_t19 - compressed_increment_t9 - (s_t(2) / s_t(15))*hx_5 - (s_t(2) / s_t(15))*hx_8;
          const s_t pa_p0_3_1 = -compressed_increment_t14 - compressed_increment_t20 - compressed_increment_t26 - (s_t(4) / s_t(15))*hx_5;
          const s_t pa_p0_3_2 = -compressed_increment_t24 - compressed_increment_t26 - compressed_increment_t9 - (s_t(4) / s_t(15))*hx_8;
          const s_t pa_p1_0_0 = compressed_increment_t27 + compressed_increment_t28 + compressed_increment_t32 + compressed_increment_t35;
          const s_t pa_p1_0_1 = compressed_increment_t32 + compressed_increment_t36 + compressed_increment_t37 + compressed_increment_t40;
          const s_t pa_p1_0_2 = compressed_increment_t30 + compressed_increment_t35 + compressed_increment_t40 + compressed_increment_t41 + compressed_increment_t42;
          const s_t pa_p1_1_0 = compressed_increment_t33 + compressed_increment_t43 + compressed_increment_t45 + compressed_increment_t46;
          const s_t pa_p1_1_1 = compressed_increment_t47 + compressed_increment_t49 + compressed_increment_t50;
          const s_t pa_p1_1_2 = compressed_increment_t33 + compressed_increment_t49 + compressed_increment_t51 + ((s_t(1) / s_t(10)))*hy_8;
          const s_t pa_p1_2_0 = -compressed_increment_t28 + compressed_increment_t44 + compressed_increment_t50 + compressed_increment_t52;
          const s_t pa_p1_2_1 = compressed_increment_t36 + compressed_increment_t38 + compressed_increment_t45 + compressed_increment_t48 + ((s_t(1) / s_t(10)))*hy_2;
          const s_t pa_p1_2_2 = compressed_increment_t38 + compressed_increment_t51 + compressed_increment_t52 + ((s_t(1) / s_t(10)))*hy_9;
          const s_t pa_p1_3_0 = -compressed_increment_t36 - compressed_increment_t41 - compressed_increment_t44 - compressed_increment_t46 - (s_t(2) / s_t(15))*hy_5 - (s_t(2) / s_t(15))*hy_8;
          const s_t pa_p1_3_1 = -compressed_increment_t41 - compressed_increment_t47 - compressed_increment_t53 - (s_t(4) / s_t(15))*hy_5;
          const s_t pa_p1_3_2 = -compressed_increment_t36 - compressed_increment_t51 - compressed_increment_t53 - (s_t(4) / s_t(15))*hy_8;
          const s_t pa_p2_0_0 = compressed_increment_t54 + compressed_increment_t55 + compressed_increment_t59 + compressed_increment_t62;
          const s_t pa_p2_0_1 = compressed_increment_t59 + compressed_increment_t63 + compressed_increment_t64 + compressed_increment_t67;
          const s_t pa_p2_0_2 = compressed_increment_t57 + compressed_increment_t62 + compressed_increment_t67 + compressed_increment_t68 + compressed_increment_t69;
          const s_t pa_p2_1_0 = compressed_increment_t60 + compressed_increment_t70 + compressed_increment_t72 + compressed_increment_t73;
          const s_t pa_p2_1_1 = compressed_increment_t74 + compressed_increment_t76 + compressed_increment_t77;
          const s_t pa_p2_1_2 = compressed_increment_t60 + compressed_increment_t76 + compressed_increment_t78 + ((s_t(1) / s_t(10)))*hz_8;
          const s_t pa_p2_2_0 = -compressed_increment_t55 + compressed_increment_t71 + compressed_increment_t77 + compressed_increment_t79;
          const s_t pa_p2_2_1 = compressed_increment_t63 + compressed_increment_t65 + compressed_increment_t72 + compressed_increment_t75 + ((s_t(1) / s_t(10)))*hz_2;
          const s_t pa_p2_2_2 = compressed_increment_t65 + compressed_increment_t78 + compressed_increment_t79 + ((s_t(1) / s_t(10)))*hz_9;
          const s_t pa_p2_3_0 = -compressed_increment_t63 - compressed_increment_t68 - compressed_increment_t71 - compressed_increment_t73 - (s_t(2) / s_t(15))*hz_5 - (s_t(2) / s_t(15))*hz_8;
          const s_t pa_p2_3_1 = -compressed_increment_t68 - compressed_increment_t74 - compressed_increment_t80 - (s_t(4) / s_t(15))*hz_5;
          const s_t pa_p2_3_2 = -compressed_increment_t63 - compressed_increment_t78 - compressed_increment_t80 - (s_t(4) / s_t(15))*hz_8;
          const s_t pa_y0_0_0 = pa_p0_0_0*tangent0 + pa_p0_0_1*tangent1 + pa_p0_0_2*tangent2 + pa_p1_0_0*tangent3 + pa_p1_0_1*tangent4 + pa_p1_0_2*tangent5 + pa_p2_0_0*tangent6 + pa_p2_0_1*tangent7 + pa_p2_0_2*tangent8;
          const s_t pa_y0_0_1 = pa_p0_0_0*tangent1 + pa_p0_0_1*tangent9 + pa_p0_0_2*tangent10 + pa_p1_0_0*tangent11 + pa_p1_0_1*tangent12 + pa_p1_0_2*tangent13 + pa_p2_0_0*tangent14 + pa_p2_0_1*tangent15 + pa_p2_0_2*tangent16;
          const s_t pa_y0_0_2 = pa_p0_0_0*tangent2 + pa_p0_0_1*tangent10 + pa_p0_0_2*tangent17 + pa_p1_0_0*tangent18 + pa_p1_0_1*tangent19 + pa_p1_0_2*tangent20 + pa_p2_0_0*tangent21 + pa_p2_0_1*tangent22 + pa_p2_0_2*tangent23;
          const s_t pa_y0_1_0 = pa_p0_1_0*tangent0 + pa_p0_1_1*tangent1 + pa_p0_1_2*tangent2 + pa_p1_1_0*tangent3 + pa_p1_1_1*tangent4 + pa_p1_1_2*tangent5 + pa_p2_1_0*tangent6 + pa_p2_1_1*tangent7 + pa_p2_1_2*tangent8;
          const s_t pa_y0_1_1 = pa_p0_1_0*tangent1 + pa_p0_1_1*tangent9 + pa_p0_1_2*tangent10 + pa_p1_1_0*tangent11 + pa_p1_1_1*tangent12 + pa_p1_1_2*tangent13 + pa_p2_1_0*tangent14 + pa_p2_1_1*tangent15 + pa_p2_1_2*tangent16;
          const s_t pa_y0_1_2 = pa_p0_1_0*tangent2 + pa_p0_1_1*tangent10 + pa_p0_1_2*tangent17 + pa_p1_1_0*tangent18 + pa_p1_1_1*tangent19 + pa_p1_1_2*tangent20 + pa_p2_1_0*tangent21 + pa_p2_1_1*tangent22 + pa_p2_1_2*tangent23;
          const s_t pa_y0_2_0 = pa_p0_2_0*tangent0 + pa_p0_2_1*tangent1 + pa_p0_2_2*tangent2 + pa_p1_2_0*tangent3 + pa_p1_2_1*tangent4 + pa_p1_2_2*tangent5 + pa_p2_2_0*tangent6 + pa_p2_2_1*tangent7 + pa_p2_2_2*tangent8;
          const s_t pa_y0_2_1 = pa_p0_2_0*tangent1 + pa_p0_2_1*tangent9 + pa_p0_2_2*tangent10 + pa_p1_2_0*tangent11 + pa_p1_2_1*tangent12 + pa_p1_2_2*tangent13 + pa_p2_2_0*tangent14 + pa_p2_2_1*tangent15 + pa_p2_2_2*tangent16;
          const s_t pa_y0_2_2 = pa_p0_2_0*tangent2 + pa_p0_2_1*tangent10 + pa_p0_2_2*tangent17 + pa_p1_2_0*tangent18 + pa_p1_2_1*tangent19 + pa_p1_2_2*tangent20 + pa_p2_2_0*tangent21 + pa_p2_2_1*tangent22 + pa_p2_2_2*tangent23;
          const s_t pa_y0_3_0 = pa_p0_3_0*tangent0 + pa_p0_3_1*tangent1 + pa_p0_3_2*tangent2 + pa_p1_3_0*tangent3 + pa_p1_3_1*tangent4 + pa_p1_3_2*tangent5 + pa_p2_3_0*tangent6 + pa_p2_3_1*tangent7 + pa_p2_3_2*tangent8;
          const s_t pa_y0_3_1 = pa_p0_3_0*tangent1 + pa_p0_3_1*tangent9 + pa_p0_3_2*tangent10 + pa_p1_3_0*tangent11 + pa_p1_3_1*tangent12 + pa_p1_3_2*tangent13 + pa_p2_3_0*tangent14 + pa_p2_3_1*tangent15 + pa_p2_3_2*tangent16;
          const s_t pa_y0_3_2 = pa_p0_3_0*tangent2 + pa_p0_3_1*tangent10 + pa_p0_3_2*tangent17 + pa_p1_3_0*tangent18 + pa_p1_3_1*tangent19 + pa_p1_3_2*tangent20 + pa_p2_3_0*tangent21 + pa_p2_3_1*tangent22 + pa_p2_3_2*tangent23;
          const s_t pa_y1_0_0 = pa_p0_0_0*tangent3 + pa_p0_0_1*tangent11 + pa_p0_0_2*tangent18 + pa_p1_0_0*tangent24 + pa_p1_0_1*tangent25 + pa_p1_0_2*tangent26 + pa_p2_0_0*tangent27 + pa_p2_0_1*tangent28 + pa_p2_0_2*tangent29;
          const s_t pa_y1_0_1 = pa_p0_0_0*tangent4 + pa_p0_0_1*tangent12 + pa_p0_0_2*tangent19 + pa_p1_0_0*tangent25 + pa_p1_0_1*tangent30 + pa_p1_0_2*tangent31 + pa_p2_0_0*tangent32 + pa_p2_0_1*tangent33 + pa_p2_0_2*tangent34;
          const s_t pa_y1_0_2 = pa_p0_0_0*tangent5 + pa_p0_0_1*tangent13 + pa_p0_0_2*tangent20 + pa_p1_0_0*tangent26 + pa_p1_0_1*tangent31 + pa_p1_0_2*tangent35 + pa_p2_0_0*tangent36 + pa_p2_0_1*tangent37 + pa_p2_0_2*tangent38;
          const s_t pa_y1_1_0 = pa_p0_1_0*tangent3 + pa_p0_1_1*tangent11 + pa_p0_1_2*tangent18 + pa_p1_1_0*tangent24 + pa_p1_1_1*tangent25 + pa_p1_1_2*tangent26 + pa_p2_1_0*tangent27 + pa_p2_1_1*tangent28 + pa_p2_1_2*tangent29;
          const s_t pa_y1_1_1 = pa_p0_1_0*tangent4 + pa_p0_1_1*tangent12 + pa_p0_1_2*tangent19 + pa_p1_1_0*tangent25 + pa_p1_1_1*tangent30 + pa_p1_1_2*tangent31 + pa_p2_1_0*tangent32 + pa_p2_1_1*tangent33 + pa_p2_1_2*tangent34;
          const s_t pa_y1_1_2 = pa_p0_1_0*tangent5 + pa_p0_1_1*tangent13 + pa_p0_1_2*tangent20 + pa_p1_1_0*tangent26 + pa_p1_1_1*tangent31 + pa_p1_1_2*tangent35 + pa_p2_1_0*tangent36 + pa_p2_1_1*tangent37 + pa_p2_1_2*tangent38;
          const s_t pa_y1_2_0 = pa_p0_2_0*tangent3 + pa_p0_2_1*tangent11 + pa_p0_2_2*tangent18 + pa_p1_2_0*tangent24 + pa_p1_2_1*tangent25 + pa_p1_2_2*tangent26 + pa_p2_2_0*tangent27 + pa_p2_2_1*tangent28 + pa_p2_2_2*tangent29;
          const s_t pa_y1_2_1 = pa_p0_2_0*tangent4 + pa_p0_2_1*tangent12 + pa_p0_2_2*tangent19 + pa_p1_2_0*tangent25 + pa_p1_2_1*tangent30 + pa_p1_2_2*tangent31 + pa_p2_2_0*tangent32 + pa_p2_2_1*tangent33 + pa_p2_2_2*tangent34;
          const s_t pa_y1_2_2 = pa_p0_2_0*tangent5 + pa_p0_2_1*tangent13 + pa_p0_2_2*tangent20 + pa_p1_2_0*tangent26 + pa_p1_2_1*tangent31 + pa_p1_2_2*tangent35 + pa_p2_2_0*tangent36 + pa_p2_2_1*tangent37 + pa_p2_2_2*tangent38;
          const s_t pa_y1_3_0 = pa_p0_3_0*tangent3 + pa_p0_3_1*tangent11 + pa_p0_3_2*tangent18 + pa_p1_3_0*tangent24 + pa_p1_3_1*tangent25 + pa_p1_3_2*tangent26 + pa_p2_3_0*tangent27 + pa_p2_3_1*tangent28 + pa_p2_3_2*tangent29;
          const s_t pa_y1_3_1 = pa_p0_3_0*tangent4 + pa_p0_3_1*tangent12 + pa_p0_3_2*tangent19 + pa_p1_3_0*tangent25 + pa_p1_3_1*tangent30 + pa_p1_3_2*tangent31 + pa_p2_3_0*tangent32 + pa_p2_3_1*tangent33 + pa_p2_3_2*tangent34;
          const s_t pa_y1_3_2 = pa_p0_3_0*tangent5 + pa_p0_3_1*tangent13 + pa_p0_3_2*tangent20 + pa_p1_3_0*tangent26 + pa_p1_3_1*tangent31 + pa_p1_3_2*tangent35 + pa_p2_3_0*tangent36 + pa_p2_3_1*tangent37 + pa_p2_3_2*tangent38;
          const s_t pa_y2_0_0 = pa_p0_0_0*tangent6 + pa_p0_0_1*tangent14 + pa_p0_0_2*tangent21 + pa_p1_0_0*tangent27 + pa_p1_0_1*tangent32 + pa_p1_0_2*tangent36 + pa_p2_0_0*tangent39 + pa_p2_0_1*tangent40 + pa_p2_0_2*tangent41;
          const s_t pa_y2_0_1 = pa_p0_0_0*tangent7 + pa_p0_0_1*tangent15 + pa_p0_0_2*tangent22 + pa_p1_0_0*tangent28 + pa_p1_0_1*tangent33 + pa_p1_0_2*tangent37 + pa_p2_0_0*tangent40 + pa_p2_0_1*tangent42 + pa_p2_0_2*tangent43;
          const s_t pa_y2_0_2 = pa_p0_0_0*tangent8 + pa_p0_0_1*tangent16 + pa_p0_0_2*tangent23 + pa_p1_0_0*tangent29 + pa_p1_0_1*tangent34 + pa_p1_0_2*tangent38 + pa_p2_0_0*tangent41 + pa_p2_0_1*tangent43 + pa_p2_0_2*tangent44;
          const s_t pa_y2_1_0 = pa_p0_1_0*tangent6 + pa_p0_1_1*tangent14 + pa_p0_1_2*tangent21 + pa_p1_1_0*tangent27 + pa_p1_1_1*tangent32 + pa_p1_1_2*tangent36 + pa_p2_1_0*tangent39 + pa_p2_1_1*tangent40 + pa_p2_1_2*tangent41;
          const s_t pa_y2_1_1 = pa_p0_1_0*tangent7 + pa_p0_1_1*tangent15 + pa_p0_1_2*tangent22 + pa_p1_1_0*tangent28 + pa_p1_1_1*tangent33 + pa_p1_1_2*tangent37 + pa_p2_1_0*tangent40 + pa_p2_1_1*tangent42 + pa_p2_1_2*tangent43;
          const s_t pa_y2_1_2 = pa_p0_1_0*tangent8 + pa_p0_1_1*tangent16 + pa_p0_1_2*tangent23 + pa_p1_1_0*tangent29 + pa_p1_1_1*tangent34 + pa_p1_1_2*tangent38 + pa_p2_1_0*tangent41 + pa_p2_1_1*tangent43 + pa_p2_1_2*tangent44;
          const s_t pa_y2_2_0 = pa_p0_2_0*tangent6 + pa_p0_2_1*tangent14 + pa_p0_2_2*tangent21 + pa_p1_2_0*tangent27 + pa_p1_2_1*tangent32 + pa_p1_2_2*tangent36 + pa_p2_2_0*tangent39 + pa_p2_2_1*tangent40 + pa_p2_2_2*tangent41;
          const s_t pa_y2_2_1 = pa_p0_2_0*tangent7 + pa_p0_2_1*tangent15 + pa_p0_2_2*tangent22 + pa_p1_2_0*tangent28 + pa_p1_2_1*tangent33 + pa_p1_2_2*tangent37 + pa_p2_2_0*tangent40 + pa_p2_2_1*tangent42 + pa_p2_2_2*tangent43;
          const s_t pa_y2_2_2 = pa_p0_2_0*tangent8 + pa_p0_2_1*tangent16 + pa_p0_2_2*tangent23 + pa_p1_2_0*tangent29 + pa_p1_2_1*tangent34 + pa_p1_2_2*tangent38 + pa_p2_2_0*tangent41 + pa_p2_2_1*tangent43 + pa_p2_2_2*tangent44;
          const s_t pa_y2_3_0 = pa_p0_3_0*tangent6 + pa_p0_3_1*tangent14 + pa_p0_3_2*tangent21 + pa_p1_3_0*tangent27 + pa_p1_3_1*tangent32 + pa_p1_3_2*tangent36 + pa_p2_3_0*tangent39 + pa_p2_3_1*tangent40 + pa_p2_3_2*tangent41;
          const s_t pa_y2_3_1 = pa_p0_3_0*tangent7 + pa_p0_3_1*tangent15 + pa_p0_3_2*tangent22 + pa_p1_3_0*tangent28 + pa_p1_3_1*tangent33 + pa_p1_3_2*tangent37 + pa_p2_3_0*tangent40 + pa_p2_3_1*tangent42 + pa_p2_3_2*tangent43;
          const s_t pa_y2_3_2 = pa_p0_3_0*tangent8 + pa_p0_3_1*tangent16 + pa_p0_3_2*tangent23 + pa_p1_3_0*tangent29 + pa_p1_3_1*tangent34 + pa_p1_3_2*tangent38 + pa_p2_3_0*tangent41 + pa_p2_3_1*tangent43 + pa_p2_3_2*tangent44;
          const s_t mixed_t0 = ((s_t(15) / s_t(2)))*pa_y0_1_0;
          const s_t mixed_t1 = ((s_t(15) / s_t(2)))*pa_y0_2_0;
          const s_t mixed_t2 = ((s_t(15) / s_t(2)))*pa_y0_1_1;
          const s_t mixed_t3 = ((s_t(15) / s_t(2)))*pa_y0_2_1;
          const s_t mixed_t4 = ((s_t(15) / s_t(2)))*pa_y0_1_2;
          const s_t mixed_t5 = ((s_t(15) / s_t(2)))*pa_y0_2_2;
          const s_t mixed_t6 = -(s_t(15) / s_t(2))*pa_y0_0_0;
          const s_t mixed_t7 = s_t(6)*pa_y0_3_0;
          const s_t mixed_t8 = -(s_t(15) / s_t(2))*pa_y0_0_1;
          const s_t mixed_t9 = s_t(6)*pa_y0_3_1;
          const s_t mixed_t10 = -(s_t(15) / s_t(2))*pa_y0_0_2;
          const s_t mixed_t11 = s_t(6)*pa_y0_3_2;
          const s_t mixed_t12 = ((s_t(15) / s_t(2)))*pa_y1_1_0;
          const s_t mixed_t13 = ((s_t(15) / s_t(2)))*pa_y1_2_0;
          const s_t mixed_t14 = ((s_t(15) / s_t(2)))*pa_y1_1_1;
          const s_t mixed_t15 = ((s_t(15) / s_t(2)))*pa_y1_2_1;
          const s_t mixed_t16 = ((s_t(15) / s_t(2)))*pa_y1_1_2;
          const s_t mixed_t17 = ((s_t(15) / s_t(2)))*pa_y1_2_2;
          const s_t mixed_t18 = -(s_t(15) / s_t(2))*pa_y1_0_0;
          const s_t mixed_t19 = s_t(6)*pa_y1_3_0;
          const s_t mixed_t20 = -(s_t(15) / s_t(2))*pa_y1_0_1;
          const s_t mixed_t21 = s_t(6)*pa_y1_3_1;
          const s_t mixed_t22 = -(s_t(15) / s_t(2))*pa_y1_0_2;
          const s_t mixed_t23 = s_t(6)*pa_y1_3_2;
          const s_t mixed_t24 = ((s_t(15) / s_t(2)))*pa_y2_1_0;
          const s_t mixed_t25 = ((s_t(15) / s_t(2)))*pa_y2_2_0;
          const s_t mixed_t26 = ((s_t(15) / s_t(2)))*pa_y2_1_1;
          const s_t mixed_t27 = ((s_t(15) / s_t(2)))*pa_y2_2_1;
          const s_t mixed_t28 = ((s_t(15) / s_t(2)))*pa_y2_1_2;
          const s_t mixed_t29 = ((s_t(15) / s_t(2)))*pa_y2_2_2;
          const s_t mixed_t30 = -(s_t(15) / s_t(2))*pa_y2_0_0;
          const s_t mixed_t31 = s_t(6)*pa_y2_3_0;
          const s_t mixed_t32 = -(s_t(15) / s_t(2))*pa_y2_0_1;
          const s_t mixed_t33 = s_t(6)*pa_y2_3_1;
          const s_t mixed_t34 = -(s_t(15) / s_t(2))*pa_y2_0_2;
          const s_t mixed_t35 = s_t(6)*pa_y2_3_2;
          const s_t pa_q0_0_0 = -mixed_t0 - mixed_t1 + s_t(15)*pa_y0_0_0;
          const s_t pa_q0_0_1 = -mixed_t2 - mixed_t3 + s_t(15)*pa_y0_0_1;
          const s_t pa_q0_0_2 = -mixed_t4 - mixed_t5 + s_t(15)*pa_y0_0_2;
          const s_t pa_q0_1_0 = mixed_t1 + mixed_t6 + mixed_t7 + s_t(21)*pa_y0_1_0;
          const s_t pa_q0_1_1 = mixed_t3 + mixed_t8 + mixed_t9 + s_t(21)*pa_y0_1_1;
          const s_t pa_q0_1_2 = mixed_t10 + mixed_t11 + mixed_t5 + s_t(21)*pa_y0_1_2;
          const s_t pa_q0_2_0 = mixed_t0 + mixed_t6 + s_t(15)*pa_y0_2_0;
          const s_t pa_q0_2_1 = mixed_t2 + mixed_t8 + s_t(15)*pa_y0_2_1;
          const s_t pa_q0_2_2 = mixed_t10 + mixed_t4 + s_t(15)*pa_y0_2_2;
          const s_t pa_q0_3_0 = mixed_t7 + s_t(6)*pa_y0_1_0;
          const s_t pa_q0_3_1 = mixed_t9 + s_t(6)*pa_y0_1_1;
          const s_t pa_q0_3_2 = mixed_t11 + s_t(6)*pa_y0_1_2;
          const s_t pa_q1_0_0 = -mixed_t12 - mixed_t13 + s_t(15)*pa_y1_0_0;
          const s_t pa_q1_0_1 = -mixed_t14 - mixed_t15 + s_t(15)*pa_y1_0_1;
          const s_t pa_q1_0_2 = -mixed_t16 - mixed_t17 + s_t(15)*pa_y1_0_2;
          const s_t pa_q1_1_0 = mixed_t13 + mixed_t18 + mixed_t19 + s_t(21)*pa_y1_1_0;
          const s_t pa_q1_1_1 = mixed_t15 + mixed_t20 + mixed_t21 + s_t(21)*pa_y1_1_1;
          const s_t pa_q1_1_2 = mixed_t17 + mixed_t22 + mixed_t23 + s_t(21)*pa_y1_1_2;
          const s_t pa_q1_2_0 = mixed_t12 + mixed_t18 + s_t(15)*pa_y1_2_0;
          const s_t pa_q1_2_1 = mixed_t14 + mixed_t20 + s_t(15)*pa_y1_2_1;
          const s_t pa_q1_2_2 = mixed_t16 + mixed_t22 + s_t(15)*pa_y1_2_2;
          const s_t pa_q1_3_0 = mixed_t19 + s_t(6)*pa_y1_1_0;
          const s_t pa_q1_3_1 = mixed_t21 + s_t(6)*pa_y1_1_1;
          const s_t pa_q1_3_2 = mixed_t23 + s_t(6)*pa_y1_1_2;
          const s_t pa_q2_0_0 = -mixed_t24 - mixed_t25 + s_t(15)*pa_y2_0_0;
          const s_t pa_q2_0_1 = -mixed_t26 - mixed_t27 + s_t(15)*pa_y2_0_1;
          const s_t pa_q2_0_2 = -mixed_t28 - mixed_t29 + s_t(15)*pa_y2_0_2;
          const s_t pa_q2_1_0 = mixed_t25 + mixed_t30 + mixed_t31 + s_t(21)*pa_y2_1_0;
          const s_t pa_q2_1_1 = mixed_t27 + mixed_t32 + mixed_t33 + s_t(21)*pa_y2_1_1;
          const s_t pa_q2_1_2 = mixed_t29 + mixed_t34 + mixed_t35 + s_t(21)*pa_y2_1_2;
          const s_t pa_q2_2_0 = mixed_t24 + mixed_t30 + s_t(15)*pa_y2_2_0;
          const s_t pa_q2_2_1 = mixed_t26 + mixed_t32 + s_t(15)*pa_y2_2_1;
          const s_t pa_q2_2_2 = mixed_t28 + mixed_t34 + s_t(15)*pa_y2_2_2;
          const s_t pa_q2_3_0 = mixed_t31 + s_t(6)*pa_y2_1_0;
          const s_t pa_q2_3_1 = mixed_t33 + s_t(6)*pa_y2_1_1;
          const s_t pa_q2_3_2 = mixed_t35 + s_t(6)*pa_y2_1_2;
          const s_t output_t0 = ((s_t(1) / s_t(30)))*pa_q0_3_1;
          const s_t output_t1 = ((s_t(1) / s_t(30)))*pa_q0_3_2;
          const s_t output_t2 = ((s_t(1) / s_t(30)))*pa_q0_1_0;
          const s_t output_t3 = ((s_t(1) / s_t(30)))*pa_q0_2_0;
          const s_t output_t4 = ((s_t(1) / s_t(30)))*pa_q0_2_2;
          const s_t output_t5 = output_t2 + output_t3 + output_t4;
          const s_t output_t6 = ((s_t(1) / s_t(30)))*pa_q0_1_1;
          const s_t output_t7 = ((s_t(1) / s_t(30)))*pa_q0_1_2;
          const s_t output_t8 = ((s_t(1) / s_t(30)))*pa_q0_2_1;
          const s_t output_t9 = output_t6 + output_t7 + output_t8;
          const s_t output_t10 = ((s_t(1) / s_t(30)))*pa_q0_0_0;
          const s_t output_t11 = ((s_t(1) / s_t(30)))*pa_q0_0_1;
          const s_t output_t12 = -output_t4;
          const s_t output_t13 = ((s_t(1) / s_t(30)))*pa_q0_0_2;
          const s_t output_t14 = output_t13 - output_t7;
          const s_t output_t15 = ((s_t(2) / s_t(15)))*pa_q0_3_0;
          const s_t output_t16 = -output_t15;
          const s_t output_t17 = ((s_t(4) / s_t(15)))*pa_q0_3_2;
          const s_t output_t18 = ((s_t(1) / s_t(10)))*pa_q0_1_2;
          const s_t output_t19 = output_t11 - output_t8 + ((s_t(1) / s_t(10)))*pa_q0_1_1 - (s_t(4) / s_t(15))*pa_q0_3_1;
          const s_t output_t20 = output_t10 + output_t16 - output_t2 + ((s_t(1) / s_t(10)))*pa_q0_2_0;
          const s_t output_t21 = ((s_t(2) / s_t(15)))*pa_q0_3_2;
          const s_t output_t22 = ((s_t(1) / s_t(10)))*pa_q0_2_2;
          const s_t output_t23 = -output_t10 + output_t15;
          const s_t output_t24 = -output_t11 + ((s_t(2) / s_t(15)))*pa_q0_3_1;
          const s_t output_t25 = -output_t13;
          const s_t output_t26 = ((s_t(1) / s_t(30)))*pa_q1_3_1;
          const s_t output_t27 = ((s_t(1) / s_t(30)))*pa_q1_3_2;
          const s_t output_t28 = ((s_t(1) / s_t(30)))*pa_q1_1_0;
          const s_t output_t29 = ((s_t(1) / s_t(30)))*pa_q1_2_0;
          const s_t output_t30 = ((s_t(1) / s_t(30)))*pa_q1_2_2;
          const s_t output_t31 = output_t28 + output_t29 + output_t30;
          const s_t output_t32 = ((s_t(1) / s_t(30)))*pa_q1_1_1;
          const s_t output_t33 = ((s_t(1) / s_t(30)))*pa_q1_1_2;
          const s_t output_t34 = ((s_t(1) / s_t(30)))*pa_q1_2_1;
          const s_t output_t35 = output_t32 + output_t33 + output_t34;
          const s_t output_t36 = ((s_t(1) / s_t(30)))*pa_q1_0_0;
          const s_t output_t37 = ((s_t(1) / s_t(30)))*pa_q1_0_1;
          const s_t output_t38 = -output_t30;
          const s_t output_t39 = ((s_t(1) / s_t(30)))*pa_q1_0_2;
          const s_t output_t40 = -output_t33 + output_t39;
          const s_t output_t41 = ((s_t(2) / s_t(15)))*pa_q1_3_0;
          const s_t output_t42 = -output_t41;
          const s_t output_t43 = ((s_t(4) / s_t(15)))*pa_q1_3_2;
          const s_t output_t44 = ((s_t(1) / s_t(10)))*pa_q1_1_2;
          const s_t output_t45 = -output_t34 + output_t37 + ((s_t(1) / s_t(10)))*pa_q1_1_1 - (s_t(4) / s_t(15))*pa_q1_3_1;
          const s_t output_t46 = -output_t28 + output_t36 + output_t42 + ((s_t(1) / s_t(10)))*pa_q1_2_0;
          const s_t output_t47 = ((s_t(2) / s_t(15)))*pa_q1_3_2;
          const s_t output_t48 = ((s_t(1) / s_t(10)))*pa_q1_2_2;
          const s_t output_t49 = -output_t36 + output_t41;
          const s_t output_t50 = -output_t37 + ((s_t(2) / s_t(15)))*pa_q1_3_1;
          const s_t output_t51 = -output_t39;
          const s_t output_t52 = ((s_t(1) / s_t(30)))*pa_q2_3_1;
          const s_t output_t53 = ((s_t(1) / s_t(30)))*pa_q2_3_2;
          const s_t output_t54 = ((s_t(1) / s_t(30)))*pa_q2_1_0;
          const s_t output_t55 = ((s_t(1) / s_t(30)))*pa_q2_2_0;
          const s_t output_t56 = ((s_t(1) / s_t(30)))*pa_q2_2_2;
          const s_t output_t57 = output_t54 + output_t55 + output_t56;
          const s_t output_t58 = ((s_t(1) / s_t(30)))*pa_q2_1_1;
          const s_t output_t59 = ((s_t(1) / s_t(30)))*pa_q2_1_2;
          const s_t output_t60 = ((s_t(1) / s_t(30)))*pa_q2_2_1;
          const s_t output_t61 = output_t58 + output_t59 + output_t60;
          const s_t output_t62 = ((s_t(1) / s_t(30)))*pa_q2_0_0;
          const s_t output_t63 = ((s_t(1) / s_t(30)))*pa_q2_0_1;
          const s_t output_t64 = -output_t56;
          const s_t output_t65 = ((s_t(1) / s_t(30)))*pa_q2_0_2;
          const s_t output_t66 = -output_t59 + output_t65;
          const s_t output_t67 = ((s_t(2) / s_t(15)))*pa_q2_3_0;
          const s_t output_t68 = -output_t67;
          const s_t output_t69 = ((s_t(4) / s_t(15)))*pa_q2_3_2;
          const s_t output_t70 = ((s_t(1) / s_t(10)))*pa_q2_1_2;
          const s_t output_t71 = -output_t60 + output_t63 + ((s_t(1) / s_t(10)))*pa_q2_1_1 - (s_t(4) / s_t(15))*pa_q2_3_1;
          const s_t output_t72 = -output_t54 + output_t62 + output_t68 + ((s_t(1) / s_t(10)))*pa_q2_2_0;
          const s_t output_t73 = ((s_t(2) / s_t(15)))*pa_q2_3_2;
          const s_t output_t74 = ((s_t(1) / s_t(10)))*pa_q2_2_2;
          const s_t output_t75 = -output_t62 + output_t67;
          const s_t output_t76 = -output_t63 + ((s_t(2) / s_t(15)))*pa_q2_3_1;
          const s_t output_t77 = -output_t65;
          const s_t element_out0_0 = -output_t0 - output_t1 + output_t5 + output_t9 + ((s_t(1) / s_t(10)))*pa_q0_0_0 + ((s_t(1) / s_t(10)))*pa_q0_0_1 + ((s_t(1) / s_t(10)))*pa_q0_0_2 - (s_t(1) / s_t(30))*pa_q0_3_0;
          const s_t element_out0_1 = output_t10 - output_t3 + ((s_t(1) / s_t(10)))*pa_q0_1_0 - (s_t(1) / s_t(10))*pa_q0_3_0;
          const s_t element_out0_2 = output_t0 + output_t11 - output_t6 + ((s_t(1) / s_t(10)))*pa_q0_2_1;
          const s_t element_out0_3 = output_t1 + output_t12 + output_t14;
          const s_t element_out0_4 = -output_t12 - output_t13 - output_t16 + output_t17 - output_t18 - output_t19 - (s_t(2) / s_t(15))*pa_q0_0_0 - (s_t(2) / s_t(15))*pa_q0_1_0;
          const s_t element_out0_5 = output_t19 + output_t20;
          const s_t element_out0_6 = -output_t14 - output_t20 + output_t21 - output_t22 - (s_t(2) / s_t(15))*pa_q0_0_1 - (s_t(2) / s_t(15))*pa_q0_2_1;
          const s_t element_out0_7 = output_t2 + output_t23 + output_t24 + output_t3 + output_t6 + output_t8 - (s_t(2) / s_t(15))*pa_q0_0_2;
          const s_t element_out0_8 = -output_t17 + output_t18 - output_t23 - output_t25 - output_t5;
          const s_t element_out0_9 = -output_t21 + output_t22 - output_t24 - output_t25 - output_t9;
          const s_t element_out1_0 = -output_t26 - output_t27 + output_t31 + output_t35 + ((s_t(1) / s_t(10)))*pa_q1_0_0 + ((s_t(1) / s_t(10)))*pa_q1_0_1 + ((s_t(1) / s_t(10)))*pa_q1_0_2 - (s_t(1) / s_t(30))*pa_q1_3_0;
          const s_t element_out1_1 = -output_t29 + output_t36 + ((s_t(1) / s_t(10)))*pa_q1_1_0 - (s_t(1) / s_t(10))*pa_q1_3_0;
          const s_t element_out1_2 = output_t26 - output_t32 + output_t37 + ((s_t(1) / s_t(10)))*pa_q1_2_1;
          const s_t element_out1_3 = output_t27 + output_t38 + output_t40;
          const s_t element_out1_4 = -output_t38 - output_t39 - output_t42 + output_t43 - output_t44 - output_t45 - (s_t(2) / s_t(15))*pa_q1_0_0 - (s_t(2) / s_t(15))*pa_q1_1_0;
          const s_t element_out1_5 = output_t45 + output_t46;
          const s_t element_out1_6 = -output_t40 - output_t46 + output_t47 - output_t48 - (s_t(2) / s_t(15))*pa_q1_0_1 - (s_t(2) / s_t(15))*pa_q1_2_1;
          const s_t element_out1_7 = output_t28 + output_t29 + output_t32 + output_t34 + output_t49 + output_t50 - (s_t(2) / s_t(15))*pa_q1_0_2;
          const s_t element_out1_8 = -output_t31 - output_t43 + output_t44 - output_t49 - output_t51;
          const s_t element_out1_9 = -output_t35 - output_t47 + output_t48 - output_t50 - output_t51;
          const s_t element_out2_0 = -output_t52 - output_t53 + output_t57 + output_t61 + ((s_t(1) / s_t(10)))*pa_q2_0_0 + ((s_t(1) / s_t(10)))*pa_q2_0_1 + ((s_t(1) / s_t(10)))*pa_q2_0_2 - (s_t(1) / s_t(30))*pa_q2_3_0;
          const s_t element_out2_1 = -output_t55 + output_t62 + ((s_t(1) / s_t(10)))*pa_q2_1_0 - (s_t(1) / s_t(10))*pa_q2_3_0;
          const s_t element_out2_2 = output_t52 - output_t58 + output_t63 + ((s_t(1) / s_t(10)))*pa_q2_2_1;
          const s_t element_out2_3 = output_t53 + output_t64 + output_t66;
          const s_t element_out2_4 = -output_t64 - output_t65 - output_t68 + output_t69 - output_t70 - output_t71 - (s_t(2) / s_t(15))*pa_q2_0_0 - (s_t(2) / s_t(15))*pa_q2_1_0;
          const s_t element_out2_5 = output_t71 + output_t72;
          const s_t element_out2_6 = -output_t66 - output_t72 + output_t73 - output_t74 - (s_t(2) / s_t(15))*pa_q2_0_1 - (s_t(2) / s_t(15))*pa_q2_2_1;
          const s_t element_out2_7 = output_t54 + output_t55 + output_t58 + output_t60 + output_t75 + output_t76 - (s_t(2) / s_t(15))*pa_q2_0_2;
          const s_t element_out2_8 = -output_t57 - output_t69 + output_t70 - output_t75 - output_t77;
          const s_t element_out2_9 = -output_t61 - output_t73 + output_t74 - output_t76 - output_t77;
          bout0_0[lane] = element_out0_0;
          bout0_1[lane] = element_out0_1;
          bout0_2[lane] = element_out0_2;
          bout0_3[lane] = element_out0_3;
          bout0_4[lane] = element_out0_4;
          bout0_5[lane] = element_out0_5;
          bout0_6[lane] = element_out0_6;
          bout0_7[lane] = element_out0_7;
          bout0_8[lane] = element_out0_8;
          bout0_9[lane] = element_out0_9;
          bout1_0[lane] = element_out1_0;
          bout1_1[lane] = element_out1_1;
          bout1_2[lane] = element_out1_2;
          bout1_3[lane] = element_out1_3;
          bout1_4[lane] = element_out1_4;
          bout1_5[lane] = element_out1_5;
          bout1_6[lane] = element_out1_6;
          bout1_7[lane] = element_out1_7;
          bout1_8[lane] = element_out1_8;
          bout1_9[lane] = element_out1_9;
          bout2_0[lane] = element_out2_0;
          bout2_1[lane] = element_out2_1;
          bout2_2[lane] = element_out2_2;
          bout2_3[lane] = element_out2_3;
          bout2_4[lane] = element_out2_4;
          bout2_5[lane] = element_out2_5;
          bout2_6[lane] = element_out2_6;
          bout2_7[lane] = element_out2_7;
          bout2_8[lane] = element_out2_8;
          bout2_9[lane] = element_out2_9;
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
          pk_out[0 * max_nodes_per_pack + bev4[lane]] += bout0_4[lane];
        }
        for (int lane = 0; lane < ne; ++lane) {
          pk_out[0 * max_nodes_per_pack + bev5[lane]] += bout0_5[lane];
        }
        for (int lane = 0; lane < ne; ++lane) {
          pk_out[0 * max_nodes_per_pack + bev6[lane]] += bout0_6[lane];
        }
        for (int lane = 0; lane < ne; ++lane) {
          pk_out[0 * max_nodes_per_pack + bev7[lane]] += bout0_7[lane];
        }
        for (int lane = 0; lane < ne; ++lane) {
          pk_out[0 * max_nodes_per_pack + bev8[lane]] += bout0_8[lane];
        }
        for (int lane = 0; lane < ne; ++lane) {
          pk_out[0 * max_nodes_per_pack + bev9[lane]] += bout0_9[lane];
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
          pk_out[1 * max_nodes_per_pack + bev4[lane]] += bout1_4[lane];
        }
        for (int lane = 0; lane < ne; ++lane) {
          pk_out[1 * max_nodes_per_pack + bev5[lane]] += bout1_5[lane];
        }
        for (int lane = 0; lane < ne; ++lane) {
          pk_out[1 * max_nodes_per_pack + bev6[lane]] += bout1_6[lane];
        }
        for (int lane = 0; lane < ne; ++lane) {
          pk_out[1 * max_nodes_per_pack + bev7[lane]] += bout1_7[lane];
        }
        for (int lane = 0; lane < ne; ++lane) {
          pk_out[1 * max_nodes_per_pack + bev8[lane]] += bout1_8[lane];
        }
        for (int lane = 0; lane < ne; ++lane) {
          pk_out[1 * max_nodes_per_pack + bev9[lane]] += bout1_9[lane];
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
        for (int lane = 0; lane < ne; ++lane) {
          pk_out[2 * max_nodes_per_pack + bev4[lane]] += bout2_4[lane];
        }
        for (int lane = 0; lane < ne; ++lane) {
          pk_out[2 * max_nodes_per_pack + bev5[lane]] += bout2_5[lane];
        }
        for (int lane = 0; lane < ne; ++lane) {
          pk_out[2 * max_nodes_per_pack + bev6[lane]] += bout2_6[lane];
        }
        for (int lane = 0; lane < ne; ++lane) {
          pk_out[2 * max_nodes_per_pack + bev7[lane]] += bout2_7[lane];
        }
        for (int lane = 0; lane < ne; ++lane) {
          pk_out[2 * max_nodes_per_pack + bev8[lane]] += bout2_8[lane];
        }
        for (int lane = 0; lane < ne; ++lane) {
          pk_out[2 * max_nodes_per_pack + bev9[lane]] += bout2_9[lane];
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
static SFEM_INLINE int neohookean_ogden_tet10_inexact_apply_compressed_a_msoa_impl(
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
    const idx_t ev4 = elements[4][element];
    const idx_t ev5 = elements[5][element];
    const idx_t ev6 = elements[6][element];
    const idx_t ev7 = elements[7][element];
    const idx_t ev8 = elements[8][element];
    const idx_t ev9 = elements[9][element];
    const s_t hx_0 = hx[ev0 * h_stride];
    const s_t hx_1 = hx[ev1 * h_stride];
    const s_t hx_2 = hx[ev2 * h_stride];
    const s_t hx_3 = hx[ev3 * h_stride];
    const s_t hx_4 = hx[ev4 * h_stride];
    const s_t hx_5 = hx[ev5 * h_stride];
    const s_t hx_6 = hx[ev6 * h_stride];
    const s_t hx_7 = hx[ev7 * h_stride];
    const s_t hx_8 = hx[ev8 * h_stride];
    const s_t hx_9 = hx[ev9 * h_stride];
    const s_t hy_0 = hy[ev0 * h_stride];
    const s_t hy_1 = hy[ev1 * h_stride];
    const s_t hy_2 = hy[ev2 * h_stride];
    const s_t hy_3 = hy[ev3 * h_stride];
    const s_t hy_4 = hy[ev4 * h_stride];
    const s_t hy_5 = hy[ev5 * h_stride];
    const s_t hy_6 = hy[ev6 * h_stride];
    const s_t hy_7 = hy[ev7 * h_stride];
    const s_t hy_8 = hy[ev8 * h_stride];
    const s_t hy_9 = hy[ev9 * h_stride];
    const s_t hz_0 = hz[ev0 * h_stride];
    const s_t hz_1 = hz[ev1 * h_stride];
    const s_t hz_2 = hz[ev2 * h_stride];
    const s_t hz_3 = hz[ev3 * h_stride];
    const s_t hz_4 = hz[ev4 * h_stride];
    const s_t hz_5 = hz[ev5 * h_stride];
    const s_t hz_6 = hz[ev6 * h_stride];
    const s_t hz_7 = hz[ev7 * h_stride];
    const s_t hz_8 = hz[ev8 * h_stride];
    const s_t hz_9 = hz[ev9 * h_stride];
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
    const s_t compressed_increment_t0 = -(s_t(2) / s_t(15))*hx_4;
    const s_t compressed_increment_t1 = ((s_t(1) / s_t(30)))*hx_1;
    const s_t compressed_increment_t2 = ((s_t(1) / s_t(30)))*hx_7;
    const s_t compressed_increment_t3 = ((s_t(1) / s_t(10)))*hx_0;
    const s_t compressed_increment_t4 = ((s_t(1) / s_t(30)))*hx_5;
    const s_t compressed_increment_t5 = -compressed_increment_t2 + compressed_increment_t3 + compressed_increment_t4;
    const s_t compressed_increment_t6 = ((s_t(1) / s_t(30)))*hx_6;
    const s_t compressed_increment_t7 = ((s_t(1) / s_t(30)))*hx_8;
    const s_t compressed_increment_t8 = -compressed_increment_t6 + compressed_increment_t7;
    const s_t compressed_increment_t9 = -(s_t(2) / s_t(15))*hx_6;
    const s_t compressed_increment_t10 = ((s_t(1) / s_t(30)))*hx_2;
    const s_t compressed_increment_t11 = ((s_t(1) / s_t(30)))*hx_4;
    const s_t compressed_increment_t12 = ((s_t(1) / s_t(30)))*hx_9;
    const s_t compressed_increment_t13 = -compressed_increment_t11 + compressed_increment_t12;
    const s_t compressed_increment_t14 = -(s_t(2) / s_t(15))*hx_7;
    const s_t compressed_increment_t15 = ((s_t(1) / s_t(30)))*hx_3;
    const s_t compressed_increment_t16 = -compressed_increment_t7;
    const s_t compressed_increment_t17 = ((s_t(1) / s_t(30)))*hx_0;
    const s_t compressed_increment_t18 = compressed_increment_t17 + compressed_increment_t2 - compressed_increment_t4;
    const s_t compressed_increment_t19 = compressed_increment_t0 + ((s_t(1) / s_t(10)))*hx_1;
    const s_t compressed_increment_t20 = -compressed_increment_t10 + compressed_increment_t17;
    const s_t compressed_increment_t21 = -compressed_increment_t12;
    const s_t compressed_increment_t22 = compressed_increment_t21 - (s_t(1) / s_t(10))*hx_4;
    const s_t compressed_increment_t23 = compressed_increment_t2 + ((s_t(1) / s_t(10)))*hx_5;
    const s_t compressed_increment_t24 = -compressed_increment_t15 + compressed_increment_t17;
    const s_t compressed_increment_t25 = compressed_increment_t16 - (s_t(1) / s_t(10))*hx_6;
    const s_t compressed_increment_t26 = -(s_t(4) / s_t(15))*hx_4 + ((s_t(2) / s_t(15)))*hx_9;
    const s_t compressed_increment_t27 = -(s_t(2) / s_t(15))*hy_4;
    const s_t compressed_increment_t28 = ((s_t(1) / s_t(30)))*hy_1;
    const s_t compressed_increment_t29 = ((s_t(1) / s_t(30)))*hy_7;
    const s_t compressed_increment_t30 = ((s_t(1) / s_t(10)))*hy_0;
    const s_t compressed_increment_t31 = ((s_t(1) / s_t(30)))*hy_5;
    const s_t compressed_increment_t32 = -compressed_increment_t29 + compressed_increment_t30 + compressed_increment_t31;
    const s_t compressed_increment_t33 = ((s_t(1) / s_t(30)))*hy_6;
    const s_t compressed_increment_t34 = ((s_t(1) / s_t(30)))*hy_8;
    const s_t compressed_increment_t35 = -compressed_increment_t33 + compressed_increment_t34;
    const s_t compressed_increment_t36 = -(s_t(2) / s_t(15))*hy_6;
    const s_t compressed_increment_t37 = ((s_t(1) / s_t(30)))*hy_2;
    const s_t compressed_increment_t38 = ((s_t(1) / s_t(30)))*hy_4;
    const s_t compressed_increment_t39 = ((s_t(1) / s_t(30)))*hy_9;
    const s_t compressed_increment_t40 = -compressed_increment_t38 + compressed_increment_t39;
    const s_t compressed_increment_t41 = -(s_t(2) / s_t(15))*hy_7;
    const s_t compressed_increment_t42 = ((s_t(1) / s_t(30)))*hy_3;
    const s_t compressed_increment_t43 = -compressed_increment_t34;
    const s_t compressed_increment_t44 = ((s_t(1) / s_t(30)))*hy_0;
    const s_t compressed_increment_t45 = compressed_increment_t29 - compressed_increment_t31 + compressed_increment_t44;
    const s_t compressed_increment_t46 = compressed_increment_t27 + ((s_t(1) / s_t(10)))*hy_1;
    const s_t compressed_increment_t47 = -compressed_increment_t37 + compressed_increment_t44;
    const s_t compressed_increment_t48 = -compressed_increment_t39;
    const s_t compressed_increment_t49 = compressed_increment_t48 - (s_t(1) / s_t(10))*hy_4;
    const s_t compressed_increment_t50 = compressed_increment_t29 + ((s_t(1) / s_t(10)))*hy_5;
    const s_t compressed_increment_t51 = -compressed_increment_t42 + compressed_increment_t44;
    const s_t compressed_increment_t52 = compressed_increment_t43 - (s_t(1) / s_t(10))*hy_6;
    const s_t compressed_increment_t53 = -(s_t(4) / s_t(15))*hy_4 + ((s_t(2) / s_t(15)))*hy_9;
    const s_t compressed_increment_t54 = -(s_t(2) / s_t(15))*hz_4;
    const s_t compressed_increment_t55 = ((s_t(1) / s_t(30)))*hz_1;
    const s_t compressed_increment_t56 = ((s_t(1) / s_t(30)))*hz_7;
    const s_t compressed_increment_t57 = ((s_t(1) / s_t(10)))*hz_0;
    const s_t compressed_increment_t58 = ((s_t(1) / s_t(30)))*hz_5;
    const s_t compressed_increment_t59 = -compressed_increment_t56 + compressed_increment_t57 + compressed_increment_t58;
    const s_t compressed_increment_t60 = ((s_t(1) / s_t(30)))*hz_6;
    const s_t compressed_increment_t61 = ((s_t(1) / s_t(30)))*hz_8;
    const s_t compressed_increment_t62 = -compressed_increment_t60 + compressed_increment_t61;
    const s_t compressed_increment_t63 = -(s_t(2) / s_t(15))*hz_6;
    const s_t compressed_increment_t64 = ((s_t(1) / s_t(30)))*hz_2;
    const s_t compressed_increment_t65 = ((s_t(1) / s_t(30)))*hz_4;
    const s_t compressed_increment_t66 = ((s_t(1) / s_t(30)))*hz_9;
    const s_t compressed_increment_t67 = -compressed_increment_t65 + compressed_increment_t66;
    const s_t compressed_increment_t68 = -(s_t(2) / s_t(15))*hz_7;
    const s_t compressed_increment_t69 = ((s_t(1) / s_t(30)))*hz_3;
    const s_t compressed_increment_t70 = -compressed_increment_t61;
    const s_t compressed_increment_t71 = ((s_t(1) / s_t(30)))*hz_0;
    const s_t compressed_increment_t72 = compressed_increment_t56 - compressed_increment_t58 + compressed_increment_t71;
    const s_t compressed_increment_t73 = compressed_increment_t54 + ((s_t(1) / s_t(10)))*hz_1;
    const s_t compressed_increment_t74 = -compressed_increment_t64 + compressed_increment_t71;
    const s_t compressed_increment_t75 = -compressed_increment_t66;
    const s_t compressed_increment_t76 = compressed_increment_t75 - (s_t(1) / s_t(10))*hz_4;
    const s_t compressed_increment_t77 = compressed_increment_t56 + ((s_t(1) / s_t(10)))*hz_5;
    const s_t compressed_increment_t78 = -compressed_increment_t69 + compressed_increment_t71;
    const s_t compressed_increment_t79 = compressed_increment_t70 - (s_t(1) / s_t(10))*hz_6;
    const s_t compressed_increment_t80 = -(s_t(4) / s_t(15))*hz_4 + ((s_t(2) / s_t(15)))*hz_9;
    const s_t pa_p0_0_0 = compressed_increment_t0 + compressed_increment_t1 + compressed_increment_t5 + compressed_increment_t8;
    const s_t pa_p0_0_1 = compressed_increment_t10 + compressed_increment_t13 + compressed_increment_t5 + compressed_increment_t9;
    const s_t pa_p0_0_2 = compressed_increment_t13 + compressed_increment_t14 + compressed_increment_t15 + compressed_increment_t3 + compressed_increment_t8;
    const s_t pa_p0_1_0 = compressed_increment_t16 + compressed_increment_t18 + compressed_increment_t19 + compressed_increment_t6;
    const s_t pa_p0_1_1 = compressed_increment_t20 + compressed_increment_t22 + compressed_increment_t23;
    const s_t pa_p0_1_2 = compressed_increment_t22 + compressed_increment_t24 + compressed_increment_t6 + ((s_t(1) / s_t(10)))*hx_8;
    const s_t pa_p0_2_0 = -compressed_increment_t1 + compressed_increment_t17 + compressed_increment_t23 + compressed_increment_t25;
    const s_t pa_p0_2_1 = compressed_increment_t11 + compressed_increment_t18 + compressed_increment_t21 + compressed_increment_t9 + ((s_t(1) / s_t(10)))*hx_2;
    const s_t pa_p0_2_2 = compressed_increment_t11 + compressed_increment_t24 + compressed_increment_t25 + ((s_t(1) / s_t(10)))*hx_9;
    const s_t pa_p0_3_0 = -compressed_increment_t14 - compressed_increment_t17 - compressed_increment_t19 - compressed_increment_t9 - (s_t(2) / s_t(15))*hx_5 - (s_t(2) / s_t(15))*hx_8;
    const s_t pa_p0_3_1 = -compressed_increment_t14 - compressed_increment_t20 - compressed_increment_t26 - (s_t(4) / s_t(15))*hx_5;
    const s_t pa_p0_3_2 = -compressed_increment_t24 - compressed_increment_t26 - compressed_increment_t9 - (s_t(4) / s_t(15))*hx_8;
    const s_t pa_p1_0_0 = compressed_increment_t27 + compressed_increment_t28 + compressed_increment_t32 + compressed_increment_t35;
    const s_t pa_p1_0_1 = compressed_increment_t32 + compressed_increment_t36 + compressed_increment_t37 + compressed_increment_t40;
    const s_t pa_p1_0_2 = compressed_increment_t30 + compressed_increment_t35 + compressed_increment_t40 + compressed_increment_t41 + compressed_increment_t42;
    const s_t pa_p1_1_0 = compressed_increment_t33 + compressed_increment_t43 + compressed_increment_t45 + compressed_increment_t46;
    const s_t pa_p1_1_1 = compressed_increment_t47 + compressed_increment_t49 + compressed_increment_t50;
    const s_t pa_p1_1_2 = compressed_increment_t33 + compressed_increment_t49 + compressed_increment_t51 + ((s_t(1) / s_t(10)))*hy_8;
    const s_t pa_p1_2_0 = -compressed_increment_t28 + compressed_increment_t44 + compressed_increment_t50 + compressed_increment_t52;
    const s_t pa_p1_2_1 = compressed_increment_t36 + compressed_increment_t38 + compressed_increment_t45 + compressed_increment_t48 + ((s_t(1) / s_t(10)))*hy_2;
    const s_t pa_p1_2_2 = compressed_increment_t38 + compressed_increment_t51 + compressed_increment_t52 + ((s_t(1) / s_t(10)))*hy_9;
    const s_t pa_p1_3_0 = -compressed_increment_t36 - compressed_increment_t41 - compressed_increment_t44 - compressed_increment_t46 - (s_t(2) / s_t(15))*hy_5 - (s_t(2) / s_t(15))*hy_8;
    const s_t pa_p1_3_1 = -compressed_increment_t41 - compressed_increment_t47 - compressed_increment_t53 - (s_t(4) / s_t(15))*hy_5;
    const s_t pa_p1_3_2 = -compressed_increment_t36 - compressed_increment_t51 - compressed_increment_t53 - (s_t(4) / s_t(15))*hy_8;
    const s_t pa_p2_0_0 = compressed_increment_t54 + compressed_increment_t55 + compressed_increment_t59 + compressed_increment_t62;
    const s_t pa_p2_0_1 = compressed_increment_t59 + compressed_increment_t63 + compressed_increment_t64 + compressed_increment_t67;
    const s_t pa_p2_0_2 = compressed_increment_t57 + compressed_increment_t62 + compressed_increment_t67 + compressed_increment_t68 + compressed_increment_t69;
    const s_t pa_p2_1_0 = compressed_increment_t60 + compressed_increment_t70 + compressed_increment_t72 + compressed_increment_t73;
    const s_t pa_p2_1_1 = compressed_increment_t74 + compressed_increment_t76 + compressed_increment_t77;
    const s_t pa_p2_1_2 = compressed_increment_t60 + compressed_increment_t76 + compressed_increment_t78 + ((s_t(1) / s_t(10)))*hz_8;
    const s_t pa_p2_2_0 = -compressed_increment_t55 + compressed_increment_t71 + compressed_increment_t77 + compressed_increment_t79;
    const s_t pa_p2_2_1 = compressed_increment_t63 + compressed_increment_t65 + compressed_increment_t72 + compressed_increment_t75 + ((s_t(1) / s_t(10)))*hz_2;
    const s_t pa_p2_2_2 = compressed_increment_t65 + compressed_increment_t78 + compressed_increment_t79 + ((s_t(1) / s_t(10)))*hz_9;
    const s_t pa_p2_3_0 = -compressed_increment_t63 - compressed_increment_t68 - compressed_increment_t71 - compressed_increment_t73 - (s_t(2) / s_t(15))*hz_5 - (s_t(2) / s_t(15))*hz_8;
    const s_t pa_p2_3_1 = -compressed_increment_t68 - compressed_increment_t74 - compressed_increment_t80 - (s_t(4) / s_t(15))*hz_5;
    const s_t pa_p2_3_2 = -compressed_increment_t63 - compressed_increment_t78 - compressed_increment_t80 - (s_t(4) / s_t(15))*hz_8;
    const s_t pa_y0_0_0 = pa_p0_0_0*tangent0 + pa_p0_0_1*tangent1 + pa_p0_0_2*tangent2 + pa_p1_0_0*tangent3 + pa_p1_0_1*tangent4 + pa_p1_0_2*tangent5 + pa_p2_0_0*tangent6 + pa_p2_0_1*tangent7 + pa_p2_0_2*tangent8;
    const s_t pa_y0_0_1 = pa_p0_0_0*tangent1 + pa_p0_0_1*tangent9 + pa_p0_0_2*tangent10 + pa_p1_0_0*tangent11 + pa_p1_0_1*tangent12 + pa_p1_0_2*tangent13 + pa_p2_0_0*tangent14 + pa_p2_0_1*tangent15 + pa_p2_0_2*tangent16;
    const s_t pa_y0_0_2 = pa_p0_0_0*tangent2 + pa_p0_0_1*tangent10 + pa_p0_0_2*tangent17 + pa_p1_0_0*tangent18 + pa_p1_0_1*tangent19 + pa_p1_0_2*tangent20 + pa_p2_0_0*tangent21 + pa_p2_0_1*tangent22 + pa_p2_0_2*tangent23;
    const s_t pa_y0_1_0 = pa_p0_1_0*tangent0 + pa_p0_1_1*tangent1 + pa_p0_1_2*tangent2 + pa_p1_1_0*tangent3 + pa_p1_1_1*tangent4 + pa_p1_1_2*tangent5 + pa_p2_1_0*tangent6 + pa_p2_1_1*tangent7 + pa_p2_1_2*tangent8;
    const s_t pa_y0_1_1 = pa_p0_1_0*tangent1 + pa_p0_1_1*tangent9 + pa_p0_1_2*tangent10 + pa_p1_1_0*tangent11 + pa_p1_1_1*tangent12 + pa_p1_1_2*tangent13 + pa_p2_1_0*tangent14 + pa_p2_1_1*tangent15 + pa_p2_1_2*tangent16;
    const s_t pa_y0_1_2 = pa_p0_1_0*tangent2 + pa_p0_1_1*tangent10 + pa_p0_1_2*tangent17 + pa_p1_1_0*tangent18 + pa_p1_1_1*tangent19 + pa_p1_1_2*tangent20 + pa_p2_1_0*tangent21 + pa_p2_1_1*tangent22 + pa_p2_1_2*tangent23;
    const s_t pa_y0_2_0 = pa_p0_2_0*tangent0 + pa_p0_2_1*tangent1 + pa_p0_2_2*tangent2 + pa_p1_2_0*tangent3 + pa_p1_2_1*tangent4 + pa_p1_2_2*tangent5 + pa_p2_2_0*tangent6 + pa_p2_2_1*tangent7 + pa_p2_2_2*tangent8;
    const s_t pa_y0_2_1 = pa_p0_2_0*tangent1 + pa_p0_2_1*tangent9 + pa_p0_2_2*tangent10 + pa_p1_2_0*tangent11 + pa_p1_2_1*tangent12 + pa_p1_2_2*tangent13 + pa_p2_2_0*tangent14 + pa_p2_2_1*tangent15 + pa_p2_2_2*tangent16;
    const s_t pa_y0_2_2 = pa_p0_2_0*tangent2 + pa_p0_2_1*tangent10 + pa_p0_2_2*tangent17 + pa_p1_2_0*tangent18 + pa_p1_2_1*tangent19 + pa_p1_2_2*tangent20 + pa_p2_2_0*tangent21 + pa_p2_2_1*tangent22 + pa_p2_2_2*tangent23;
    const s_t pa_y0_3_0 = pa_p0_3_0*tangent0 + pa_p0_3_1*tangent1 + pa_p0_3_2*tangent2 + pa_p1_3_0*tangent3 + pa_p1_3_1*tangent4 + pa_p1_3_2*tangent5 + pa_p2_3_0*tangent6 + pa_p2_3_1*tangent7 + pa_p2_3_2*tangent8;
    const s_t pa_y0_3_1 = pa_p0_3_0*tangent1 + pa_p0_3_1*tangent9 + pa_p0_3_2*tangent10 + pa_p1_3_0*tangent11 + pa_p1_3_1*tangent12 + pa_p1_3_2*tangent13 + pa_p2_3_0*tangent14 + pa_p2_3_1*tangent15 + pa_p2_3_2*tangent16;
    const s_t pa_y0_3_2 = pa_p0_3_0*tangent2 + pa_p0_3_1*tangent10 + pa_p0_3_2*tangent17 + pa_p1_3_0*tangent18 + pa_p1_3_1*tangent19 + pa_p1_3_2*tangent20 + pa_p2_3_0*tangent21 + pa_p2_3_1*tangent22 + pa_p2_3_2*tangent23;
    const s_t pa_y1_0_0 = pa_p0_0_0*tangent3 + pa_p0_0_1*tangent11 + pa_p0_0_2*tangent18 + pa_p1_0_0*tangent24 + pa_p1_0_1*tangent25 + pa_p1_0_2*tangent26 + pa_p2_0_0*tangent27 + pa_p2_0_1*tangent28 + pa_p2_0_2*tangent29;
    const s_t pa_y1_0_1 = pa_p0_0_0*tangent4 + pa_p0_0_1*tangent12 + pa_p0_0_2*tangent19 + pa_p1_0_0*tangent25 + pa_p1_0_1*tangent30 + pa_p1_0_2*tangent31 + pa_p2_0_0*tangent32 + pa_p2_0_1*tangent33 + pa_p2_0_2*tangent34;
    const s_t pa_y1_0_2 = pa_p0_0_0*tangent5 + pa_p0_0_1*tangent13 + pa_p0_0_2*tangent20 + pa_p1_0_0*tangent26 + pa_p1_0_1*tangent31 + pa_p1_0_2*tangent35 + pa_p2_0_0*tangent36 + pa_p2_0_1*tangent37 + pa_p2_0_2*tangent38;
    const s_t pa_y1_1_0 = pa_p0_1_0*tangent3 + pa_p0_1_1*tangent11 + pa_p0_1_2*tangent18 + pa_p1_1_0*tangent24 + pa_p1_1_1*tangent25 + pa_p1_1_2*tangent26 + pa_p2_1_0*tangent27 + pa_p2_1_1*tangent28 + pa_p2_1_2*tangent29;
    const s_t pa_y1_1_1 = pa_p0_1_0*tangent4 + pa_p0_1_1*tangent12 + pa_p0_1_2*tangent19 + pa_p1_1_0*tangent25 + pa_p1_1_1*tangent30 + pa_p1_1_2*tangent31 + pa_p2_1_0*tangent32 + pa_p2_1_1*tangent33 + pa_p2_1_2*tangent34;
    const s_t pa_y1_1_2 = pa_p0_1_0*tangent5 + pa_p0_1_1*tangent13 + pa_p0_1_2*tangent20 + pa_p1_1_0*tangent26 + pa_p1_1_1*tangent31 + pa_p1_1_2*tangent35 + pa_p2_1_0*tangent36 + pa_p2_1_1*tangent37 + pa_p2_1_2*tangent38;
    const s_t pa_y1_2_0 = pa_p0_2_0*tangent3 + pa_p0_2_1*tangent11 + pa_p0_2_2*tangent18 + pa_p1_2_0*tangent24 + pa_p1_2_1*tangent25 + pa_p1_2_2*tangent26 + pa_p2_2_0*tangent27 + pa_p2_2_1*tangent28 + pa_p2_2_2*tangent29;
    const s_t pa_y1_2_1 = pa_p0_2_0*tangent4 + pa_p0_2_1*tangent12 + pa_p0_2_2*tangent19 + pa_p1_2_0*tangent25 + pa_p1_2_1*tangent30 + pa_p1_2_2*tangent31 + pa_p2_2_0*tangent32 + pa_p2_2_1*tangent33 + pa_p2_2_2*tangent34;
    const s_t pa_y1_2_2 = pa_p0_2_0*tangent5 + pa_p0_2_1*tangent13 + pa_p0_2_2*tangent20 + pa_p1_2_0*tangent26 + pa_p1_2_1*tangent31 + pa_p1_2_2*tangent35 + pa_p2_2_0*tangent36 + pa_p2_2_1*tangent37 + pa_p2_2_2*tangent38;
    const s_t pa_y1_3_0 = pa_p0_3_0*tangent3 + pa_p0_3_1*tangent11 + pa_p0_3_2*tangent18 + pa_p1_3_0*tangent24 + pa_p1_3_1*tangent25 + pa_p1_3_2*tangent26 + pa_p2_3_0*tangent27 + pa_p2_3_1*tangent28 + pa_p2_3_2*tangent29;
    const s_t pa_y1_3_1 = pa_p0_3_0*tangent4 + pa_p0_3_1*tangent12 + pa_p0_3_2*tangent19 + pa_p1_3_0*tangent25 + pa_p1_3_1*tangent30 + pa_p1_3_2*tangent31 + pa_p2_3_0*tangent32 + pa_p2_3_1*tangent33 + pa_p2_3_2*tangent34;
    const s_t pa_y1_3_2 = pa_p0_3_0*tangent5 + pa_p0_3_1*tangent13 + pa_p0_3_2*tangent20 + pa_p1_3_0*tangent26 + pa_p1_3_1*tangent31 + pa_p1_3_2*tangent35 + pa_p2_3_0*tangent36 + pa_p2_3_1*tangent37 + pa_p2_3_2*tangent38;
    const s_t pa_y2_0_0 = pa_p0_0_0*tangent6 + pa_p0_0_1*tangent14 + pa_p0_0_2*tangent21 + pa_p1_0_0*tangent27 + pa_p1_0_1*tangent32 + pa_p1_0_2*tangent36 + pa_p2_0_0*tangent39 + pa_p2_0_1*tangent40 + pa_p2_0_2*tangent41;
    const s_t pa_y2_0_1 = pa_p0_0_0*tangent7 + pa_p0_0_1*tangent15 + pa_p0_0_2*tangent22 + pa_p1_0_0*tangent28 + pa_p1_0_1*tangent33 + pa_p1_0_2*tangent37 + pa_p2_0_0*tangent40 + pa_p2_0_1*tangent42 + pa_p2_0_2*tangent43;
    const s_t pa_y2_0_2 = pa_p0_0_0*tangent8 + pa_p0_0_1*tangent16 + pa_p0_0_2*tangent23 + pa_p1_0_0*tangent29 + pa_p1_0_1*tangent34 + pa_p1_0_2*tangent38 + pa_p2_0_0*tangent41 + pa_p2_0_1*tangent43 + pa_p2_0_2*tangent44;
    const s_t pa_y2_1_0 = pa_p0_1_0*tangent6 + pa_p0_1_1*tangent14 + pa_p0_1_2*tangent21 + pa_p1_1_0*tangent27 + pa_p1_1_1*tangent32 + pa_p1_1_2*tangent36 + pa_p2_1_0*tangent39 + pa_p2_1_1*tangent40 + pa_p2_1_2*tangent41;
    const s_t pa_y2_1_1 = pa_p0_1_0*tangent7 + pa_p0_1_1*tangent15 + pa_p0_1_2*tangent22 + pa_p1_1_0*tangent28 + pa_p1_1_1*tangent33 + pa_p1_1_2*tangent37 + pa_p2_1_0*tangent40 + pa_p2_1_1*tangent42 + pa_p2_1_2*tangent43;
    const s_t pa_y2_1_2 = pa_p0_1_0*tangent8 + pa_p0_1_1*tangent16 + pa_p0_1_2*tangent23 + pa_p1_1_0*tangent29 + pa_p1_1_1*tangent34 + pa_p1_1_2*tangent38 + pa_p2_1_0*tangent41 + pa_p2_1_1*tangent43 + pa_p2_1_2*tangent44;
    const s_t pa_y2_2_0 = pa_p0_2_0*tangent6 + pa_p0_2_1*tangent14 + pa_p0_2_2*tangent21 + pa_p1_2_0*tangent27 + pa_p1_2_1*tangent32 + pa_p1_2_2*tangent36 + pa_p2_2_0*tangent39 + pa_p2_2_1*tangent40 + pa_p2_2_2*tangent41;
    const s_t pa_y2_2_1 = pa_p0_2_0*tangent7 + pa_p0_2_1*tangent15 + pa_p0_2_2*tangent22 + pa_p1_2_0*tangent28 + pa_p1_2_1*tangent33 + pa_p1_2_2*tangent37 + pa_p2_2_0*tangent40 + pa_p2_2_1*tangent42 + pa_p2_2_2*tangent43;
    const s_t pa_y2_2_2 = pa_p0_2_0*tangent8 + pa_p0_2_1*tangent16 + pa_p0_2_2*tangent23 + pa_p1_2_0*tangent29 + pa_p1_2_1*tangent34 + pa_p1_2_2*tangent38 + pa_p2_2_0*tangent41 + pa_p2_2_1*tangent43 + pa_p2_2_2*tangent44;
    const s_t pa_y2_3_0 = pa_p0_3_0*tangent6 + pa_p0_3_1*tangent14 + pa_p0_3_2*tangent21 + pa_p1_3_0*tangent27 + pa_p1_3_1*tangent32 + pa_p1_3_2*tangent36 + pa_p2_3_0*tangent39 + pa_p2_3_1*tangent40 + pa_p2_3_2*tangent41;
    const s_t pa_y2_3_1 = pa_p0_3_0*tangent7 + pa_p0_3_1*tangent15 + pa_p0_3_2*tangent22 + pa_p1_3_0*tangent28 + pa_p1_3_1*tangent33 + pa_p1_3_2*tangent37 + pa_p2_3_0*tangent40 + pa_p2_3_1*tangent42 + pa_p2_3_2*tangent43;
    const s_t pa_y2_3_2 = pa_p0_3_0*tangent8 + pa_p0_3_1*tangent16 + pa_p0_3_2*tangent23 + pa_p1_3_0*tangent29 + pa_p1_3_1*tangent34 + pa_p1_3_2*tangent38 + pa_p2_3_0*tangent41 + pa_p2_3_1*tangent43 + pa_p2_3_2*tangent44;
    const s_t mixed_t0 = ((s_t(15) / s_t(2)))*pa_y0_1_0;
    const s_t mixed_t1 = ((s_t(15) / s_t(2)))*pa_y0_2_0;
    const s_t mixed_t2 = ((s_t(15) / s_t(2)))*pa_y0_1_1;
    const s_t mixed_t3 = ((s_t(15) / s_t(2)))*pa_y0_2_1;
    const s_t mixed_t4 = ((s_t(15) / s_t(2)))*pa_y0_1_2;
    const s_t mixed_t5 = ((s_t(15) / s_t(2)))*pa_y0_2_2;
    const s_t mixed_t6 = -(s_t(15) / s_t(2))*pa_y0_0_0;
    const s_t mixed_t7 = s_t(6)*pa_y0_3_0;
    const s_t mixed_t8 = -(s_t(15) / s_t(2))*pa_y0_0_1;
    const s_t mixed_t9 = s_t(6)*pa_y0_3_1;
    const s_t mixed_t10 = -(s_t(15) / s_t(2))*pa_y0_0_2;
    const s_t mixed_t11 = s_t(6)*pa_y0_3_2;
    const s_t mixed_t12 = ((s_t(15) / s_t(2)))*pa_y1_1_0;
    const s_t mixed_t13 = ((s_t(15) / s_t(2)))*pa_y1_2_0;
    const s_t mixed_t14 = ((s_t(15) / s_t(2)))*pa_y1_1_1;
    const s_t mixed_t15 = ((s_t(15) / s_t(2)))*pa_y1_2_1;
    const s_t mixed_t16 = ((s_t(15) / s_t(2)))*pa_y1_1_2;
    const s_t mixed_t17 = ((s_t(15) / s_t(2)))*pa_y1_2_2;
    const s_t mixed_t18 = -(s_t(15) / s_t(2))*pa_y1_0_0;
    const s_t mixed_t19 = s_t(6)*pa_y1_3_0;
    const s_t mixed_t20 = -(s_t(15) / s_t(2))*pa_y1_0_1;
    const s_t mixed_t21 = s_t(6)*pa_y1_3_1;
    const s_t mixed_t22 = -(s_t(15) / s_t(2))*pa_y1_0_2;
    const s_t mixed_t23 = s_t(6)*pa_y1_3_2;
    const s_t mixed_t24 = ((s_t(15) / s_t(2)))*pa_y2_1_0;
    const s_t mixed_t25 = ((s_t(15) / s_t(2)))*pa_y2_2_0;
    const s_t mixed_t26 = ((s_t(15) / s_t(2)))*pa_y2_1_1;
    const s_t mixed_t27 = ((s_t(15) / s_t(2)))*pa_y2_2_1;
    const s_t mixed_t28 = ((s_t(15) / s_t(2)))*pa_y2_1_2;
    const s_t mixed_t29 = ((s_t(15) / s_t(2)))*pa_y2_2_2;
    const s_t mixed_t30 = -(s_t(15) / s_t(2))*pa_y2_0_0;
    const s_t mixed_t31 = s_t(6)*pa_y2_3_0;
    const s_t mixed_t32 = -(s_t(15) / s_t(2))*pa_y2_0_1;
    const s_t mixed_t33 = s_t(6)*pa_y2_3_1;
    const s_t mixed_t34 = -(s_t(15) / s_t(2))*pa_y2_0_2;
    const s_t mixed_t35 = s_t(6)*pa_y2_3_2;
    const s_t pa_q0_0_0 = -mixed_t0 - mixed_t1 + s_t(15)*pa_y0_0_0;
    const s_t pa_q0_0_1 = -mixed_t2 - mixed_t3 + s_t(15)*pa_y0_0_1;
    const s_t pa_q0_0_2 = -mixed_t4 - mixed_t5 + s_t(15)*pa_y0_0_2;
    const s_t pa_q0_1_0 = mixed_t1 + mixed_t6 + mixed_t7 + s_t(21)*pa_y0_1_0;
    const s_t pa_q0_1_1 = mixed_t3 + mixed_t8 + mixed_t9 + s_t(21)*pa_y0_1_1;
    const s_t pa_q0_1_2 = mixed_t10 + mixed_t11 + mixed_t5 + s_t(21)*pa_y0_1_2;
    const s_t pa_q0_2_0 = mixed_t0 + mixed_t6 + s_t(15)*pa_y0_2_0;
    const s_t pa_q0_2_1 = mixed_t2 + mixed_t8 + s_t(15)*pa_y0_2_1;
    const s_t pa_q0_2_2 = mixed_t10 + mixed_t4 + s_t(15)*pa_y0_2_2;
    const s_t pa_q0_3_0 = mixed_t7 + s_t(6)*pa_y0_1_0;
    const s_t pa_q0_3_1 = mixed_t9 + s_t(6)*pa_y0_1_1;
    const s_t pa_q0_3_2 = mixed_t11 + s_t(6)*pa_y0_1_2;
    const s_t pa_q1_0_0 = -mixed_t12 - mixed_t13 + s_t(15)*pa_y1_0_0;
    const s_t pa_q1_0_1 = -mixed_t14 - mixed_t15 + s_t(15)*pa_y1_0_1;
    const s_t pa_q1_0_2 = -mixed_t16 - mixed_t17 + s_t(15)*pa_y1_0_2;
    const s_t pa_q1_1_0 = mixed_t13 + mixed_t18 + mixed_t19 + s_t(21)*pa_y1_1_0;
    const s_t pa_q1_1_1 = mixed_t15 + mixed_t20 + mixed_t21 + s_t(21)*pa_y1_1_1;
    const s_t pa_q1_1_2 = mixed_t17 + mixed_t22 + mixed_t23 + s_t(21)*pa_y1_1_2;
    const s_t pa_q1_2_0 = mixed_t12 + mixed_t18 + s_t(15)*pa_y1_2_0;
    const s_t pa_q1_2_1 = mixed_t14 + mixed_t20 + s_t(15)*pa_y1_2_1;
    const s_t pa_q1_2_2 = mixed_t16 + mixed_t22 + s_t(15)*pa_y1_2_2;
    const s_t pa_q1_3_0 = mixed_t19 + s_t(6)*pa_y1_1_0;
    const s_t pa_q1_3_1 = mixed_t21 + s_t(6)*pa_y1_1_1;
    const s_t pa_q1_3_2 = mixed_t23 + s_t(6)*pa_y1_1_2;
    const s_t pa_q2_0_0 = -mixed_t24 - mixed_t25 + s_t(15)*pa_y2_0_0;
    const s_t pa_q2_0_1 = -mixed_t26 - mixed_t27 + s_t(15)*pa_y2_0_1;
    const s_t pa_q2_0_2 = -mixed_t28 - mixed_t29 + s_t(15)*pa_y2_0_2;
    const s_t pa_q2_1_0 = mixed_t25 + mixed_t30 + mixed_t31 + s_t(21)*pa_y2_1_0;
    const s_t pa_q2_1_1 = mixed_t27 + mixed_t32 + mixed_t33 + s_t(21)*pa_y2_1_1;
    const s_t pa_q2_1_2 = mixed_t29 + mixed_t34 + mixed_t35 + s_t(21)*pa_y2_1_2;
    const s_t pa_q2_2_0 = mixed_t24 + mixed_t30 + s_t(15)*pa_y2_2_0;
    const s_t pa_q2_2_1 = mixed_t26 + mixed_t32 + s_t(15)*pa_y2_2_1;
    const s_t pa_q2_2_2 = mixed_t28 + mixed_t34 + s_t(15)*pa_y2_2_2;
    const s_t pa_q2_3_0 = mixed_t31 + s_t(6)*pa_y2_1_0;
    const s_t pa_q2_3_1 = mixed_t33 + s_t(6)*pa_y2_1_1;
    const s_t pa_q2_3_2 = mixed_t35 + s_t(6)*pa_y2_1_2;
    const s_t output_t0 = ((s_t(1) / s_t(30)))*pa_q0_3_1;
    const s_t output_t1 = ((s_t(1) / s_t(30)))*pa_q0_3_2;
    const s_t output_t2 = ((s_t(1) / s_t(30)))*pa_q0_1_0;
    const s_t output_t3 = ((s_t(1) / s_t(30)))*pa_q0_2_0;
    const s_t output_t4 = ((s_t(1) / s_t(30)))*pa_q0_2_2;
    const s_t output_t5 = output_t2 + output_t3 + output_t4;
    const s_t output_t6 = ((s_t(1) / s_t(30)))*pa_q0_1_1;
    const s_t output_t7 = ((s_t(1) / s_t(30)))*pa_q0_1_2;
    const s_t output_t8 = ((s_t(1) / s_t(30)))*pa_q0_2_1;
    const s_t output_t9 = output_t6 + output_t7 + output_t8;
    const s_t output_t10 = ((s_t(1) / s_t(30)))*pa_q0_0_0;
    const s_t output_t11 = ((s_t(1) / s_t(30)))*pa_q0_0_1;
    const s_t output_t12 = -output_t4;
    const s_t output_t13 = ((s_t(1) / s_t(30)))*pa_q0_0_2;
    const s_t output_t14 = output_t13 - output_t7;
    const s_t output_t15 = ((s_t(2) / s_t(15)))*pa_q0_3_0;
    const s_t output_t16 = -output_t15;
    const s_t output_t17 = ((s_t(4) / s_t(15)))*pa_q0_3_2;
    const s_t output_t18 = ((s_t(1) / s_t(10)))*pa_q0_1_2;
    const s_t output_t19 = output_t11 - output_t8 + ((s_t(1) / s_t(10)))*pa_q0_1_1 - (s_t(4) / s_t(15))*pa_q0_3_1;
    const s_t output_t20 = output_t10 + output_t16 - output_t2 + ((s_t(1) / s_t(10)))*pa_q0_2_0;
    const s_t output_t21 = ((s_t(2) / s_t(15)))*pa_q0_3_2;
    const s_t output_t22 = ((s_t(1) / s_t(10)))*pa_q0_2_2;
    const s_t output_t23 = -output_t10 + output_t15;
    const s_t output_t24 = -output_t11 + ((s_t(2) / s_t(15)))*pa_q0_3_1;
    const s_t output_t25 = -output_t13;
    const s_t output_t26 = ((s_t(1) / s_t(30)))*pa_q1_3_1;
    const s_t output_t27 = ((s_t(1) / s_t(30)))*pa_q1_3_2;
    const s_t output_t28 = ((s_t(1) / s_t(30)))*pa_q1_1_0;
    const s_t output_t29 = ((s_t(1) / s_t(30)))*pa_q1_2_0;
    const s_t output_t30 = ((s_t(1) / s_t(30)))*pa_q1_2_2;
    const s_t output_t31 = output_t28 + output_t29 + output_t30;
    const s_t output_t32 = ((s_t(1) / s_t(30)))*pa_q1_1_1;
    const s_t output_t33 = ((s_t(1) / s_t(30)))*pa_q1_1_2;
    const s_t output_t34 = ((s_t(1) / s_t(30)))*pa_q1_2_1;
    const s_t output_t35 = output_t32 + output_t33 + output_t34;
    const s_t output_t36 = ((s_t(1) / s_t(30)))*pa_q1_0_0;
    const s_t output_t37 = ((s_t(1) / s_t(30)))*pa_q1_0_1;
    const s_t output_t38 = -output_t30;
    const s_t output_t39 = ((s_t(1) / s_t(30)))*pa_q1_0_2;
    const s_t output_t40 = -output_t33 + output_t39;
    const s_t output_t41 = ((s_t(2) / s_t(15)))*pa_q1_3_0;
    const s_t output_t42 = -output_t41;
    const s_t output_t43 = ((s_t(4) / s_t(15)))*pa_q1_3_2;
    const s_t output_t44 = ((s_t(1) / s_t(10)))*pa_q1_1_2;
    const s_t output_t45 = -output_t34 + output_t37 + ((s_t(1) / s_t(10)))*pa_q1_1_1 - (s_t(4) / s_t(15))*pa_q1_3_1;
    const s_t output_t46 = -output_t28 + output_t36 + output_t42 + ((s_t(1) / s_t(10)))*pa_q1_2_0;
    const s_t output_t47 = ((s_t(2) / s_t(15)))*pa_q1_3_2;
    const s_t output_t48 = ((s_t(1) / s_t(10)))*pa_q1_2_2;
    const s_t output_t49 = -output_t36 + output_t41;
    const s_t output_t50 = -output_t37 + ((s_t(2) / s_t(15)))*pa_q1_3_1;
    const s_t output_t51 = -output_t39;
    const s_t output_t52 = ((s_t(1) / s_t(30)))*pa_q2_3_1;
    const s_t output_t53 = ((s_t(1) / s_t(30)))*pa_q2_3_2;
    const s_t output_t54 = ((s_t(1) / s_t(30)))*pa_q2_1_0;
    const s_t output_t55 = ((s_t(1) / s_t(30)))*pa_q2_2_0;
    const s_t output_t56 = ((s_t(1) / s_t(30)))*pa_q2_2_2;
    const s_t output_t57 = output_t54 + output_t55 + output_t56;
    const s_t output_t58 = ((s_t(1) / s_t(30)))*pa_q2_1_1;
    const s_t output_t59 = ((s_t(1) / s_t(30)))*pa_q2_1_2;
    const s_t output_t60 = ((s_t(1) / s_t(30)))*pa_q2_2_1;
    const s_t output_t61 = output_t58 + output_t59 + output_t60;
    const s_t output_t62 = ((s_t(1) / s_t(30)))*pa_q2_0_0;
    const s_t output_t63 = ((s_t(1) / s_t(30)))*pa_q2_0_1;
    const s_t output_t64 = -output_t56;
    const s_t output_t65 = ((s_t(1) / s_t(30)))*pa_q2_0_2;
    const s_t output_t66 = -output_t59 + output_t65;
    const s_t output_t67 = ((s_t(2) / s_t(15)))*pa_q2_3_0;
    const s_t output_t68 = -output_t67;
    const s_t output_t69 = ((s_t(4) / s_t(15)))*pa_q2_3_2;
    const s_t output_t70 = ((s_t(1) / s_t(10)))*pa_q2_1_2;
    const s_t output_t71 = -output_t60 + output_t63 + ((s_t(1) / s_t(10)))*pa_q2_1_1 - (s_t(4) / s_t(15))*pa_q2_3_1;
    const s_t output_t72 = -output_t54 + output_t62 + output_t68 + ((s_t(1) / s_t(10)))*pa_q2_2_0;
    const s_t output_t73 = ((s_t(2) / s_t(15)))*pa_q2_3_2;
    const s_t output_t74 = ((s_t(1) / s_t(10)))*pa_q2_2_2;
    const s_t output_t75 = -output_t62 + output_t67;
    const s_t output_t76 = -output_t63 + ((s_t(2) / s_t(15)))*pa_q2_3_1;
    const s_t output_t77 = -output_t65;
    const s_t element_out0_0 = -output_t0 - output_t1 + output_t5 + output_t9 + ((s_t(1) / s_t(10)))*pa_q0_0_0 + ((s_t(1) / s_t(10)))*pa_q0_0_1 + ((s_t(1) / s_t(10)))*pa_q0_0_2 - (s_t(1) / s_t(30))*pa_q0_3_0;
    const s_t element_out0_1 = output_t10 - output_t3 + ((s_t(1) / s_t(10)))*pa_q0_1_0 - (s_t(1) / s_t(10))*pa_q0_3_0;
    const s_t element_out0_2 = output_t0 + output_t11 - output_t6 + ((s_t(1) / s_t(10)))*pa_q0_2_1;
    const s_t element_out0_3 = output_t1 + output_t12 + output_t14;
    const s_t element_out0_4 = -output_t12 - output_t13 - output_t16 + output_t17 - output_t18 - output_t19 - (s_t(2) / s_t(15))*pa_q0_0_0 - (s_t(2) / s_t(15))*pa_q0_1_0;
    const s_t element_out0_5 = output_t19 + output_t20;
    const s_t element_out0_6 = -output_t14 - output_t20 + output_t21 - output_t22 - (s_t(2) / s_t(15))*pa_q0_0_1 - (s_t(2) / s_t(15))*pa_q0_2_1;
    const s_t element_out0_7 = output_t2 + output_t23 + output_t24 + output_t3 + output_t6 + output_t8 - (s_t(2) / s_t(15))*pa_q0_0_2;
    const s_t element_out0_8 = -output_t17 + output_t18 - output_t23 - output_t25 - output_t5;
    const s_t element_out0_9 = -output_t21 + output_t22 - output_t24 - output_t25 - output_t9;
    const s_t element_out1_0 = -output_t26 - output_t27 + output_t31 + output_t35 + ((s_t(1) / s_t(10)))*pa_q1_0_0 + ((s_t(1) / s_t(10)))*pa_q1_0_1 + ((s_t(1) / s_t(10)))*pa_q1_0_2 - (s_t(1) / s_t(30))*pa_q1_3_0;
    const s_t element_out1_1 = -output_t29 + output_t36 + ((s_t(1) / s_t(10)))*pa_q1_1_0 - (s_t(1) / s_t(10))*pa_q1_3_0;
    const s_t element_out1_2 = output_t26 - output_t32 + output_t37 + ((s_t(1) / s_t(10)))*pa_q1_2_1;
    const s_t element_out1_3 = output_t27 + output_t38 + output_t40;
    const s_t element_out1_4 = -output_t38 - output_t39 - output_t42 + output_t43 - output_t44 - output_t45 - (s_t(2) / s_t(15))*pa_q1_0_0 - (s_t(2) / s_t(15))*pa_q1_1_0;
    const s_t element_out1_5 = output_t45 + output_t46;
    const s_t element_out1_6 = -output_t40 - output_t46 + output_t47 - output_t48 - (s_t(2) / s_t(15))*pa_q1_0_1 - (s_t(2) / s_t(15))*pa_q1_2_1;
    const s_t element_out1_7 = output_t28 + output_t29 + output_t32 + output_t34 + output_t49 + output_t50 - (s_t(2) / s_t(15))*pa_q1_0_2;
    const s_t element_out1_8 = -output_t31 - output_t43 + output_t44 - output_t49 - output_t51;
    const s_t element_out1_9 = -output_t35 - output_t47 + output_t48 - output_t50 - output_t51;
    const s_t element_out2_0 = -output_t52 - output_t53 + output_t57 + output_t61 + ((s_t(1) / s_t(10)))*pa_q2_0_0 + ((s_t(1) / s_t(10)))*pa_q2_0_1 + ((s_t(1) / s_t(10)))*pa_q2_0_2 - (s_t(1) / s_t(30))*pa_q2_3_0;
    const s_t element_out2_1 = -output_t55 + output_t62 + ((s_t(1) / s_t(10)))*pa_q2_1_0 - (s_t(1) / s_t(10))*pa_q2_3_0;
    const s_t element_out2_2 = output_t52 - output_t58 + output_t63 + ((s_t(1) / s_t(10)))*pa_q2_2_1;
    const s_t element_out2_3 = output_t53 + output_t64 + output_t66;
    const s_t element_out2_4 = -output_t64 - output_t65 - output_t68 + output_t69 - output_t70 - output_t71 - (s_t(2) / s_t(15))*pa_q2_0_0 - (s_t(2) / s_t(15))*pa_q2_1_0;
    const s_t element_out2_5 = output_t71 + output_t72;
    const s_t element_out2_6 = -output_t66 - output_t72 + output_t73 - output_t74 - (s_t(2) / s_t(15))*pa_q2_0_1 - (s_t(2) / s_t(15))*pa_q2_2_1;
    const s_t element_out2_7 = output_t54 + output_t55 + output_t58 + output_t60 + output_t75 + output_t76 - (s_t(2) / s_t(15))*pa_q2_0_2;
    const s_t element_out2_8 = -output_t57 - output_t69 + output_t70 - output_t75 - output_t77;
    const s_t element_out2_9 = -output_t61 - output_t73 + output_t74 - output_t76 - output_t77;
    #pragma omp atomic update
    outx[ev0 * out_stride] += scale * element_out0_0;
    #pragma omp atomic update
    outx[ev1 * out_stride] += scale * element_out0_1;
    #pragma omp atomic update
    outx[ev2 * out_stride] += scale * element_out0_2;
    #pragma omp atomic update
    outx[ev3 * out_stride] += scale * element_out0_3;
    #pragma omp atomic update
    outx[ev4 * out_stride] += scale * element_out0_4;
    #pragma omp atomic update
    outx[ev5 * out_stride] += scale * element_out0_5;
    #pragma omp atomic update
    outx[ev6 * out_stride] += scale * element_out0_6;
    #pragma omp atomic update
    outx[ev7 * out_stride] += scale * element_out0_7;
    #pragma omp atomic update
    outx[ev8 * out_stride] += scale * element_out0_8;
    #pragma omp atomic update
    outx[ev9 * out_stride] += scale * element_out0_9;
    #pragma omp atomic update
    outy[ev0 * out_stride] += scale * element_out1_0;
    #pragma omp atomic update
    outy[ev1 * out_stride] += scale * element_out1_1;
    #pragma omp atomic update
    outy[ev2 * out_stride] += scale * element_out1_2;
    #pragma omp atomic update
    outy[ev3 * out_stride] += scale * element_out1_3;
    #pragma omp atomic update
    outy[ev4 * out_stride] += scale * element_out1_4;
    #pragma omp atomic update
    outy[ev5 * out_stride] += scale * element_out1_5;
    #pragma omp atomic update
    outy[ev6 * out_stride] += scale * element_out1_6;
    #pragma omp atomic update
    outy[ev7 * out_stride] += scale * element_out1_7;
    #pragma omp atomic update
    outy[ev8 * out_stride] += scale * element_out1_8;
    #pragma omp atomic update
    outy[ev9 * out_stride] += scale * element_out1_9;
    #pragma omp atomic update
    outz[ev0 * out_stride] += scale * element_out2_0;
    #pragma omp atomic update
    outz[ev1 * out_stride] += scale * element_out2_1;
    #pragma omp atomic update
    outz[ev2 * out_stride] += scale * element_out2_2;
    #pragma omp atomic update
    outz[ev3 * out_stride] += scale * element_out2_3;
    #pragma omp atomic update
    outz[ev4 * out_stride] += scale * element_out2_4;
    #pragma omp atomic update
    outz[ev5 * out_stride] += scale * element_out2_5;
    #pragma omp atomic update
    outz[ev6 * out_stride] += scale * element_out2_6;
    #pragma omp atomic update
    outz[ev7 * out_stride] += scale * element_out2_7;
    #pragma omp atomic update
    outz[ev8 * out_stride] += scale * element_out2_8;
    #pragma omp atomic update
    outz[ev9 * out_stride] += scale * element_out2_9;
  }

  return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem
