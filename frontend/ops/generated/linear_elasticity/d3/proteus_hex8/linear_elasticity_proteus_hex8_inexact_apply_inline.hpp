#pragma once
#include "../../../kernel_math.hpp"
#include "../../../packed_thread_scratch.hpp"
#include "../../../reference/line_p1_q2.hpp"
#include "../../../reference/quad_line_q2.hpp"
#include "../../../tensor_product_kernels.hpp"

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t, typename tangent_t, int VS>
static SFEM_INLINE int linear_elasticity_proteus_hex8_inexact_apply_tangent_a_msoa_impl(
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
    static constexpr int NQ = 8;
    static constexpr int NQ1 = 2;
    const s_t *const RSTR q_weight_1d = sfem::codegen::quad_line_q2<s_t>::q_weight_1d();
    static constexpr s_t QMEASURE = s_t(1);
    s_t btangent_acc[45][VS];
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
      const int qx = q % NQ1;
      const int qy = (q / NQ1) % NQ1;
      const int qz = q / (NQ1 * NQ1);
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
        const s_t qw = q_weight_1d[qx] * q_weight_1d[qy] * q_weight_1d[qz] * QMEASURE;
            const s_t integrand_t0 = pow_m1(determinant);
            const s_t integrand_t1 = pow_2(adjugate1);
            const s_t integrand_t2 = integrand_t1*mu;
            const s_t integrand_t3 = pow_2(adjugate2);
            const s_t integrand_t4 = integrand_t3*mu;
            const s_t integrand_t5 = pow_2(adjugate0);
            const s_t integrand_t6 = lmbda + s_t(2)*mu;
            const s_t integrand_t7 = adjugate1*mu;
            const s_t integrand_t8 = adjugate4*integrand_t7;
            const s_t integrand_t9 = adjugate2*mu;
            const s_t integrand_t10 = adjugate5*integrand_t9;
            const s_t integrand_t11 = adjugate0*integrand_t6;
            const s_t integrand_t12 = adjugate7*integrand_t7;
            const s_t integrand_t13 = adjugate8*integrand_t9;
            const s_t integrand_t14 = adjugate0*lmbda;
            const s_t integrand_t15 = pow_2(adjugate4);
            const s_t integrand_t16 = integrand_t15*mu;
            const s_t integrand_t17 = pow_2(adjugate5);
            const s_t integrand_t18 = integrand_t17*mu;
            const s_t integrand_t19 = pow_2(adjugate3);
            const s_t integrand_t20 = adjugate4*mu;
            const s_t integrand_t21 = adjugate7*integrand_t20;
            const s_t integrand_t22 = adjugate5*mu;
            const s_t integrand_t23 = adjugate8*integrand_t22;
            const s_t integrand_t24 = adjugate3*adjugate6;
            const s_t integrand_t25 = adjugate3*lmbda;
            const s_t integrand_t26 = pow_2(adjugate7);
            const s_t integrand_t27 = integrand_t26*mu;
            const s_t integrand_t28 = pow_2(adjugate8);
            const s_t integrand_t29 = integrand_t28*mu;
            const s_t integrand_t30 = pow_2(adjugate6);
            const s_t integrand_t31 = adjugate7*mu;
            const s_t integrand_t32 = adjugate6*lmbda;
            const s_t integrand_t33 = adjugate8*mu;
            const s_t integrand_t34 = integrand_t5*mu;
            const s_t integrand_t35 = adjugate0*mu;
            const s_t integrand_t36 = adjugate3*integrand_t35;
            const s_t integrand_t37 = adjugate1*integrand_t6;
            const s_t integrand_t38 = adjugate6*integrand_t35;
            const s_t integrand_t39 = adjugate1*lmbda;
            const s_t integrand_t40 = integrand_t19*mu;
            const s_t integrand_t41 = integrand_t24*mu;
            const s_t integrand_t42 = adjugate4*lmbda;
            const s_t integrand_t43 = integrand_t30*mu;
            const s_t integrand_t44 = adjugate7*lmbda;
            const s_t integrand_t45 = adjugate2*integrand_t6;
            const s_t integrand0 = integrand_t0*(integrand_t2 + integrand_t4 + integrand_t5*integrand_t6);
            const s_t integrand1 = integrand_t0*(adjugate3*integrand_t11 + integrand_t10 + integrand_t8);
            const s_t integrand2 = integrand_t0*(adjugate6*integrand_t11 + integrand_t12 + integrand_t13);
            const s_t integrand3 = integrand_t0*(adjugate0*integrand_t7 + adjugate1*integrand_t14);
            const s_t integrand4 = integrand_t0*(adjugate3*integrand_t7 + adjugate4*integrand_t14);
            const s_t integrand5 = integrand_t0*(adjugate6*integrand_t7 + adjugate7*integrand_t14);
            const s_t integrand6 = integrand_t0*(adjugate0*integrand_t9 + adjugate2*integrand_t14);
            const s_t integrand7 = integrand_t0*(adjugate3*integrand_t9 + adjugate5*integrand_t14);
            const s_t integrand8 = integrand_t0*(adjugate6*integrand_t9 + adjugate8*integrand_t14);
            const s_t integrand9 = integrand_t0*(integrand_t16 + integrand_t18 + integrand_t19*integrand_t6);
            const s_t integrand10 = integrand_t0*(integrand_t21 + integrand_t23 + integrand_t24*integrand_t6);
            const s_t integrand11 = integrand_t0*(adjugate0*integrand_t20 + adjugate1*integrand_t25);
            const s_t integrand12 = integrand_t0*(adjugate3*integrand_t20 + adjugate4*integrand_t25);
            const s_t integrand13 = integrand_t0*(adjugate6*integrand_t20 + adjugate7*integrand_t25);
            const s_t integrand14 = integrand_t0*(adjugate0*integrand_t22 + adjugate2*integrand_t25);
            const s_t integrand15 = integrand_t0*(adjugate3*integrand_t22 + adjugate5*integrand_t25);
            const s_t integrand16 = integrand_t0*(adjugate6*integrand_t22 + adjugate8*integrand_t25);
            const s_t integrand17 = integrand_t0*(integrand_t27 + integrand_t29 + integrand_t30*integrand_t6);
            const s_t integrand18 = integrand_t0*(adjugate0*integrand_t31 + adjugate1*integrand_t32);
            const s_t integrand19 = integrand_t0*(adjugate3*integrand_t31 + adjugate4*integrand_t32);
            const s_t integrand20 = integrand_t0*(adjugate6*integrand_t31 + adjugate7*integrand_t32);
            const s_t integrand21 = integrand_t0*(adjugate0*integrand_t33 + adjugate2*integrand_t32);
            const s_t integrand22 = integrand_t0*(adjugate3*integrand_t33 + adjugate5*integrand_t32);
            const s_t integrand23 = integrand_t0*(adjugate6*integrand_t33 + adjugate8*integrand_t32);
            const s_t integrand24 = integrand_t0*(integrand_t1*integrand_t6 + integrand_t34 + integrand_t4);
            const s_t integrand25 = integrand_t0*(adjugate4*integrand_t37 + integrand_t10 + integrand_t36);
            const s_t integrand26 = integrand_t0*(adjugate7*integrand_t37 + integrand_t13 + integrand_t38);
            const s_t integrand27 = integrand_t0*(adjugate2*integrand_t39 + adjugate2*integrand_t7);
            const s_t integrand28 = integrand_t0*(adjugate4*integrand_t9 + adjugate5*integrand_t39);
            const s_t integrand29 = integrand_t0*(adjugate7*integrand_t9 + adjugate8*integrand_t39);
            const s_t integrand30 = integrand_t0*(integrand_t15*integrand_t6 + integrand_t18 + integrand_t40);
            const s_t integrand31 = integrand_t0*(adjugate4*adjugate7*integrand_t6 + integrand_t23 + integrand_t41);
            const s_t integrand32 = integrand_t0*(adjugate2*integrand_t42 + adjugate5*integrand_t7);
            const s_t integrand33 = integrand_t0*(adjugate5*integrand_t20 + adjugate5*integrand_t42);
            const s_t integrand34 = integrand_t0*(adjugate7*integrand_t22 + adjugate8*integrand_t42);
            const s_t integrand35 = integrand_t0*(integrand_t26*integrand_t6 + integrand_t29 + integrand_t43);
            const s_t integrand36 = integrand_t0*(adjugate2*integrand_t44 + adjugate8*integrand_t7);
            const s_t integrand37 = integrand_t0*(adjugate5*integrand_t44 + adjugate8*integrand_t20);
            const s_t integrand38 = integrand_t0*(adjugate8*integrand_t31 + adjugate8*integrand_t44);
            const s_t integrand39 = integrand_t0*(integrand_t2 + integrand_t3*integrand_t6 + integrand_t34);
            const s_t integrand40 = integrand_t0*(adjugate5*integrand_t45 + integrand_t36 + integrand_t8);
            const s_t integrand41 = integrand_t0*(adjugate8*integrand_t45 + integrand_t12 + integrand_t38);
            const s_t integrand42 = integrand_t0*(integrand_t16 + integrand_t17*integrand_t6 + integrand_t40);
            const s_t integrand43 = integrand_t0*(adjugate5*adjugate8*integrand_t6 + integrand_t21 + integrand_t41);
            const s_t integrand44 = integrand_t0*(integrand_t27 + integrand_t28*integrand_t6 + integrand_t43);
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
static SFEM_INLINE int linear_elasticity_proteus_hex8_inexact_apply_stored_a_msoa_impl(
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
    }
    s_t bhx_0[VS];
    s_t bhx_1[VS];
    s_t bhx_2[VS];
    s_t bhx_3[VS];
    s_t bhx_4[VS];
    s_t bhx_5[VS];
    s_t bhx_6[VS];
    s_t bhx_7[VS];
    s_t bhy_0[VS];
    s_t bhy_1[VS];
    s_t bhy_2[VS];
    s_t bhy_3[VS];
    s_t bhy_4[VS];
    s_t bhy_5[VS];
    s_t bhy_6[VS];
    s_t bhy_7[VS];
    s_t bhz_0[VS];
    s_t bhz_1[VS];
    s_t bhz_2[VS];
    s_t bhz_3[VS];
    s_t bhz_4[VS];
    s_t bhz_5[VS];
    s_t bhz_6[VS];
    s_t bhz_7[VS];
    s_t bout0_0[VS];
    s_t bout0_1[VS];
    s_t bout0_2[VS];
    s_t bout0_3[VS];
    s_t bout0_4[VS];
    s_t bout0_5[VS];
    s_t bout0_6[VS];
    s_t bout0_7[VS];
    s_t bout1_0[VS];
    s_t bout1_1[VS];
    s_t bout1_2[VS];
    s_t bout1_3[VS];
    s_t bout1_4[VS];
    s_t bout1_5[VS];
    s_t bout1_6[VS];
    s_t bout1_7[VS];
    s_t bout2_0[VS];
    s_t bout2_1[VS];
    s_t bout2_2[VS];
    s_t bout2_3[VS];
    s_t bout2_4[VS];
    s_t bout2_5[VS];
    s_t bout2_6[VS];
    s_t bout2_7[VS];
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
      bhy_0[lane] = hy[bev0[lane] * h_stride];
      bhy_1[lane] = hy[bev1[lane] * h_stride];
      bhy_2[lane] = hy[bev2[lane] * h_stride];
      bhy_3[lane] = hy[bev3[lane] * h_stride];
      bhy_4[lane] = hy[bev4[lane] * h_stride];
      bhy_5[lane] = hy[bev5[lane] * h_stride];
      bhy_6[lane] = hy[bev6[lane] * h_stride];
      bhy_7[lane] = hy[bev7[lane] * h_stride];
      bhz_0[lane] = hz[bev0[lane] * h_stride];
      bhz_1[lane] = hz[bev1[lane] * h_stride];
      bhz_2[lane] = hz[bev2[lane] * h_stride];
      bhz_3[lane] = hz[bev3[lane] * h_stride];
      bhz_4[lane] = hz[bev4[lane] * h_stride];
      bhz_5[lane] = hz[bev5[lane] * h_stride];
      bhz_6[lane] = hz[bev6[lane] * h_stride];
      bhz_7[lane] = hz[bev7[lane] * h_stride];
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
      const s_t hy_0 = bhy_0[lane];
      const s_t hy_1 = bhy_1[lane];
      const s_t hy_2 = bhy_2[lane];
      const s_t hy_3 = bhy_3[lane];
      const s_t hy_4 = bhy_4[lane];
      const s_t hy_5 = bhy_5[lane];
      const s_t hy_6 = bhy_6[lane];
      const s_t hy_7 = bhy_7[lane];
      const s_t hz_0 = bhz_0[lane];
      const s_t hz_1 = bhz_1[lane];
      const s_t hz_2 = bhz_2[lane];
      const s_t hz_3 = bhz_3[lane];
      const s_t hz_4 = bhz_4[lane];
      const s_t hz_5 = bhz_5[lane];
      const s_t hz_6 = bhz_6[lane];
      const s_t hz_7 = bhz_7[lane];
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
      const s_t reference_product_t0 = ((s_t(1) / s_t(9)))*hx_1;
      const s_t reference_product_t1 = ((s_t(1) / s_t(36)))*hx_6;
      const s_t reference_product_t2 = ((s_t(1) / s_t(18)))*hx_3;
      const s_t reference_product_t3 = ((s_t(1) / s_t(18)))*hx_4;
      const s_t reference_product_t4 = -reference_product_t2 + reference_product_t3;
      const s_t reference_product_t5 = ((s_t(1) / s_t(18)))*hx_5;
      const s_t reference_product_t6 = ((s_t(1) / s_t(18)))*hx_2;
      const s_t reference_product_t7 = -reference_product_t5 + reference_product_t6;
      const s_t reference_product_t8 = reference_product_t4 + reference_product_t7;
      const s_t reference_product_t9 = ((s_t(1) / s_t(9)))*hx_0 - (s_t(1) / s_t(36))*hx_7;
      const s_t reference_product_t10 = -reference_product_t0 + reference_product_t1 + reference_product_t8 + reference_product_t9;
      const s_t reference_product_t11 = ((s_t(1) / s_t(12)))*hx_1;
      const s_t reference_product_t12 = ((s_t(1) / s_t(24)))*hx_6;
      const s_t reference_product_t13 = ((s_t(1) / s_t(12)))*hx_0 - (s_t(1) / s_t(24))*hx_7;
      const s_t reference_product_t14 = -reference_product_t11 + reference_product_t12 + reference_product_t13;
      const s_t reference_product_t15 = ((s_t(1) / s_t(12)))*hx_3;
      const s_t reference_product_t16 = ((s_t(1) / s_t(24)))*hx_4;
      const s_t reference_product_t17 = -reference_product_t15 + reference_product_t16;
      const s_t reference_product_t18 = ((s_t(1) / s_t(24)))*hx_5;
      const s_t reference_product_t19 = ((s_t(1) / s_t(12)))*hx_2;
      const s_t reference_product_t20 = -reference_product_t18 + reference_product_t19;
      const s_t reference_product_t21 = reference_product_t17 + reference_product_t20;
      const s_t reference_product_t22 = reference_product_t14 + reference_product_t21;
      const s_t reference_product_t23 = ((s_t(1) / s_t(24)))*hx_3;
      const s_t reference_product_t24 = ((s_t(1) / s_t(12)))*hx_4;
      const s_t reference_product_t25 = -reference_product_t23 + reference_product_t24;
      const s_t reference_product_t26 = ((s_t(1) / s_t(12)))*hx_5;
      const s_t reference_product_t27 = ((s_t(1) / s_t(24)))*hx_2;
      const s_t reference_product_t28 = -reference_product_t26 + reference_product_t27;
      const s_t reference_product_t29 = reference_product_t25 + reference_product_t28;
      const s_t reference_product_t30 = reference_product_t14 + reference_product_t29;
      const s_t reference_product_t31 = ((s_t(1) / s_t(18)))*hx_1;
      const s_t reference_product_t32 = ((s_t(1) / s_t(18)))*hx_6;
      const s_t reference_product_t33 = ((s_t(1) / s_t(18)))*hx_0 - (s_t(1) / s_t(18))*hx_7;
      const s_t reference_product_t34 = -reference_product_t31 + reference_product_t32 + reference_product_t33;
      const s_t reference_product_t35 = ((s_t(1) / s_t(9)))*hx_3;
      const s_t reference_product_t36 = ((s_t(1) / s_t(36)))*hx_4;
      const s_t reference_product_t37 = -reference_product_t35 + reference_product_t36;
      const s_t reference_product_t38 = ((s_t(1) / s_t(36)))*hx_5;
      const s_t reference_product_t39 = ((s_t(1) / s_t(9)))*hx_2;
      const s_t reference_product_t40 = -reference_product_t38 + reference_product_t39;
      const s_t reference_product_t41 = reference_product_t34 + reference_product_t37 + reference_product_t40;
      const s_t reference_product_t42 = -reference_product_t22;
      const s_t reference_product_t43 = ((s_t(1) / s_t(24)))*hx_1;
      const s_t reference_product_t44 = ((s_t(1) / s_t(12)))*hx_6;
      const s_t reference_product_t45 = ((s_t(1) / s_t(24)))*hx_0 - (s_t(1) / s_t(12))*hx_7;
      const s_t reference_product_t46 = -reference_product_t43 + reference_product_t44 + reference_product_t45;
      const s_t reference_product_t47 = reference_product_t21 + reference_product_t46;
      const s_t reference_product_t48 = ((s_t(1) / s_t(36)))*hx_3;
      const s_t reference_product_t49 = ((s_t(1) / s_t(9)))*hx_4;
      const s_t reference_product_t50 = -reference_product_t48 + reference_product_t49;
      const s_t reference_product_t51 = ((s_t(1) / s_t(9)))*hx_5;
      const s_t reference_product_t52 = ((s_t(1) / s_t(36)))*hx_2;
      const s_t reference_product_t53 = -reference_product_t51 + reference_product_t52;
      const s_t reference_product_t54 = reference_product_t34 + reference_product_t50 + reference_product_t53;
      const s_t reference_product_t55 = reference_product_t29 + reference_product_t46;
      const s_t reference_product_t56 = -reference_product_t30;
      const s_t reference_product_t57 = ((s_t(1) / s_t(36)))*hx_1;
      const s_t reference_product_t58 = ((s_t(1) / s_t(9)))*hx_6;
      const s_t reference_product_t59 = ((s_t(1) / s_t(36)))*hx_0 - (s_t(1) / s_t(9))*hx_7;
      const s_t reference_product_t60 = -reference_product_t57 + reference_product_t58 + reference_product_t59 + reference_product_t8;
      const s_t reference_product_t61 = -reference_product_t55;
      const s_t reference_product_t62 = -reference_product_t47;
      const s_t reference_product_t63 = reference_product_t13 + reference_product_t18 - reference_product_t19;
      const s_t reference_product_t64 = reference_product_t11 - reference_product_t12;
      const s_t reference_product_t65 = reference_product_t17 + reference_product_t64;
      const s_t reference_product_t66 = reference_product_t63 + reference_product_t65;
      const s_t reference_product_t67 = reference_product_t31 - reference_product_t32;
      const s_t reference_product_t68 = reference_product_t4 + reference_product_t67;
      const s_t reference_product_t69 = reference_product_t38 - reference_product_t39 + reference_product_t68 + reference_product_t9;
      const s_t reference_product_t70 = reference_product_t43 - reference_product_t44;
      const s_t reference_product_t71 = reference_product_t25 + reference_product_t70;
      const s_t reference_product_t72 = reference_product_t63 + reference_product_t71;
      const s_t reference_product_t73 = -reference_product_t66;
      const s_t reference_product_t74 = reference_product_t33 + reference_product_t5 - reference_product_t6;
      const s_t reference_product_t75 = reference_product_t0 - reference_product_t1;
      const s_t reference_product_t76 = reference_product_t37 + reference_product_t74 + reference_product_t75;
      const s_t reference_product_t77 = reference_product_t26 - reference_product_t27 + reference_product_t45;
      const s_t reference_product_t78 = reference_product_t65 + reference_product_t77;
      const s_t reference_product_t79 = reference_product_t71 + reference_product_t77;
      const s_t reference_product_t80 = reference_product_t57 - reference_product_t58;
      const s_t reference_product_t81 = reference_product_t50 + reference_product_t74 + reference_product_t80;
      const s_t reference_product_t82 = -reference_product_t72;
      const s_t reference_product_t83 = -reference_product_t79;
      const s_t reference_product_t84 = reference_product_t51 - reference_product_t52 + reference_product_t59 + reference_product_t68;
      const s_t reference_product_t85 = -reference_product_t78;
      const s_t reference_product_t86 = reference_product_t28 + reference_product_t64;
      const s_t reference_product_t87 = reference_product_t13 + reference_product_t23 - reference_product_t24;
      const s_t reference_product_t88 = reference_product_t86 + reference_product_t87;
      const s_t reference_product_t89 = reference_product_t20 + reference_product_t70;
      const s_t reference_product_t90 = reference_product_t87 + reference_product_t89;
      const s_t reference_product_t91 = reference_product_t67 + reference_product_t7;
      const s_t reference_product_t92 = reference_product_t48 - reference_product_t49 + reference_product_t9 + reference_product_t91;
      const s_t reference_product_t93 = -reference_product_t88;
      const s_t reference_product_t94 = reference_product_t15 - reference_product_t16 + reference_product_t45;
      const s_t reference_product_t95 = reference_product_t86 + reference_product_t94;
      const s_t reference_product_t96 = reference_product_t2 - reference_product_t3 + reference_product_t33;
      const s_t reference_product_t97 = reference_product_t53 + reference_product_t75 + reference_product_t96;
      const s_t reference_product_t98 = reference_product_t89 + reference_product_t94;
      const s_t reference_product_t99 = -reference_product_t90;
      const s_t reference_product_t100 = reference_product_t40 + reference_product_t80 + reference_product_t96;
      const s_t reference_product_t101 = -reference_product_t98;
      const s_t reference_product_t102 = -reference_product_t95;
      const s_t reference_product_t103 = reference_product_t35 - reference_product_t36 + reference_product_t59 + reference_product_t91;
      const s_t reference_product_t104 = ((s_t(1) / s_t(9)))*hy_1;
      const s_t reference_product_t105 = ((s_t(1) / s_t(36)))*hy_6;
      const s_t reference_product_t106 = ((s_t(1) / s_t(18)))*hy_3;
      const s_t reference_product_t107 = ((s_t(1) / s_t(18)))*hy_4;
      const s_t reference_product_t108 = -reference_product_t106 + reference_product_t107;
      const s_t reference_product_t109 = ((s_t(1) / s_t(18)))*hy_5;
      const s_t reference_product_t110 = ((s_t(1) / s_t(18)))*hy_2;
      const s_t reference_product_t111 = -reference_product_t109 + reference_product_t110;
      const s_t reference_product_t112 = reference_product_t108 + reference_product_t111;
      const s_t reference_product_t113 = ((s_t(1) / s_t(9)))*hy_0 - (s_t(1) / s_t(36))*hy_7;
      const s_t reference_product_t114 = -reference_product_t104 + reference_product_t105 + reference_product_t112 + reference_product_t113;
      const s_t reference_product_t115 = ((s_t(1) / s_t(12)))*hy_1;
      const s_t reference_product_t116 = ((s_t(1) / s_t(24)))*hy_6;
      const s_t reference_product_t117 = ((s_t(1) / s_t(12)))*hy_0 - (s_t(1) / s_t(24))*hy_7;
      const s_t reference_product_t118 = -reference_product_t115 + reference_product_t116 + reference_product_t117;
      const s_t reference_product_t119 = ((s_t(1) / s_t(12)))*hy_3;
      const s_t reference_product_t120 = ((s_t(1) / s_t(24)))*hy_4;
      const s_t reference_product_t121 = -reference_product_t119 + reference_product_t120;
      const s_t reference_product_t122 = ((s_t(1) / s_t(24)))*hy_5;
      const s_t reference_product_t123 = ((s_t(1) / s_t(12)))*hy_2;
      const s_t reference_product_t124 = -reference_product_t122 + reference_product_t123;
      const s_t reference_product_t125 = reference_product_t121 + reference_product_t124;
      const s_t reference_product_t126 = reference_product_t118 + reference_product_t125;
      const s_t reference_product_t127 = ((s_t(1) / s_t(24)))*hy_3;
      const s_t reference_product_t128 = ((s_t(1) / s_t(12)))*hy_4;
      const s_t reference_product_t129 = -reference_product_t127 + reference_product_t128;
      const s_t reference_product_t130 = ((s_t(1) / s_t(12)))*hy_5;
      const s_t reference_product_t131 = ((s_t(1) / s_t(24)))*hy_2;
      const s_t reference_product_t132 = -reference_product_t130 + reference_product_t131;
      const s_t reference_product_t133 = reference_product_t129 + reference_product_t132;
      const s_t reference_product_t134 = reference_product_t118 + reference_product_t133;
      const s_t reference_product_t135 = ((s_t(1) / s_t(18)))*hy_1;
      const s_t reference_product_t136 = ((s_t(1) / s_t(18)))*hy_6;
      const s_t reference_product_t137 = ((s_t(1) / s_t(18)))*hy_0 - (s_t(1) / s_t(18))*hy_7;
      const s_t reference_product_t138 = -reference_product_t135 + reference_product_t136 + reference_product_t137;
      const s_t reference_product_t139 = ((s_t(1) / s_t(9)))*hy_3;
      const s_t reference_product_t140 = ((s_t(1) / s_t(36)))*hy_4;
      const s_t reference_product_t141 = -reference_product_t139 + reference_product_t140;
      const s_t reference_product_t142 = ((s_t(1) / s_t(36)))*hy_5;
      const s_t reference_product_t143 = ((s_t(1) / s_t(9)))*hy_2;
      const s_t reference_product_t144 = -reference_product_t142 + reference_product_t143;
      const s_t reference_product_t145 = reference_product_t138 + reference_product_t141 + reference_product_t144;
      const s_t reference_product_t146 = -reference_product_t126;
      const s_t reference_product_t147 = ((s_t(1) / s_t(24)))*hy_1;
      const s_t reference_product_t148 = ((s_t(1) / s_t(12)))*hy_6;
      const s_t reference_product_t149 = ((s_t(1) / s_t(24)))*hy_0 - (s_t(1) / s_t(12))*hy_7;
      const s_t reference_product_t150 = -reference_product_t147 + reference_product_t148 + reference_product_t149;
      const s_t reference_product_t151 = reference_product_t125 + reference_product_t150;
      const s_t reference_product_t152 = ((s_t(1) / s_t(36)))*hy_3;
      const s_t reference_product_t153 = ((s_t(1) / s_t(9)))*hy_4;
      const s_t reference_product_t154 = -reference_product_t152 + reference_product_t153;
      const s_t reference_product_t155 = ((s_t(1) / s_t(9)))*hy_5;
      const s_t reference_product_t156 = ((s_t(1) / s_t(36)))*hy_2;
      const s_t reference_product_t157 = -reference_product_t155 + reference_product_t156;
      const s_t reference_product_t158 = reference_product_t138 + reference_product_t154 + reference_product_t157;
      const s_t reference_product_t159 = reference_product_t133 + reference_product_t150;
      const s_t reference_product_t160 = -reference_product_t134;
      const s_t reference_product_t161 = ((s_t(1) / s_t(36)))*hy_1;
      const s_t reference_product_t162 = ((s_t(1) / s_t(9)))*hy_6;
      const s_t reference_product_t163 = ((s_t(1) / s_t(36)))*hy_0 - (s_t(1) / s_t(9))*hy_7;
      const s_t reference_product_t164 = reference_product_t112 - reference_product_t161 + reference_product_t162 + reference_product_t163;
      const s_t reference_product_t165 = -reference_product_t159;
      const s_t reference_product_t166 = -reference_product_t151;
      const s_t reference_product_t167 = reference_product_t117 + reference_product_t122 - reference_product_t123;
      const s_t reference_product_t168 = reference_product_t115 - reference_product_t116;
      const s_t reference_product_t169 = reference_product_t121 + reference_product_t168;
      const s_t reference_product_t170 = reference_product_t167 + reference_product_t169;
      const s_t reference_product_t171 = reference_product_t135 - reference_product_t136;
      const s_t reference_product_t172 = reference_product_t108 + reference_product_t171;
      const s_t reference_product_t173 = reference_product_t113 + reference_product_t142 - reference_product_t143 + reference_product_t172;
      const s_t reference_product_t174 = reference_product_t147 - reference_product_t148;
      const s_t reference_product_t175 = reference_product_t129 + reference_product_t174;
      const s_t reference_product_t176 = reference_product_t167 + reference_product_t175;
      const s_t reference_product_t177 = -reference_product_t170;
      const s_t reference_product_t178 = reference_product_t109 - reference_product_t110 + reference_product_t137;
      const s_t reference_product_t179 = reference_product_t104 - reference_product_t105;
      const s_t reference_product_t180 = reference_product_t141 + reference_product_t178 + reference_product_t179;
      const s_t reference_product_t181 = reference_product_t130 - reference_product_t131 + reference_product_t149;
      const s_t reference_product_t182 = reference_product_t169 + reference_product_t181;
      const s_t reference_product_t183 = reference_product_t175 + reference_product_t181;
      const s_t reference_product_t184 = reference_product_t161 - reference_product_t162;
      const s_t reference_product_t185 = reference_product_t154 + reference_product_t178 + reference_product_t184;
      const s_t reference_product_t186 = -reference_product_t176;
      const s_t reference_product_t187 = -reference_product_t183;
      const s_t reference_product_t188 = reference_product_t155 - reference_product_t156 + reference_product_t163 + reference_product_t172;
      const s_t reference_product_t189 = -reference_product_t182;
      const s_t reference_product_t190 = reference_product_t132 + reference_product_t168;
      const s_t reference_product_t191 = reference_product_t117 + reference_product_t127 - reference_product_t128;
      const s_t reference_product_t192 = reference_product_t190 + reference_product_t191;
      const s_t reference_product_t193 = reference_product_t124 + reference_product_t174;
      const s_t reference_product_t194 = reference_product_t191 + reference_product_t193;
      const s_t reference_product_t195 = reference_product_t111 + reference_product_t171;
      const s_t reference_product_t196 = reference_product_t113 + reference_product_t152 - reference_product_t153 + reference_product_t195;
      const s_t reference_product_t197 = -reference_product_t192;
      const s_t reference_product_t198 = reference_product_t119 - reference_product_t120 + reference_product_t149;
      const s_t reference_product_t199 = reference_product_t190 + reference_product_t198;
      const s_t reference_product_t200 = reference_product_t106 - reference_product_t107 + reference_product_t137;
      const s_t reference_product_t201 = reference_product_t157 + reference_product_t179 + reference_product_t200;
      const s_t reference_product_t202 = reference_product_t193 + reference_product_t198;
      const s_t reference_product_t203 = -reference_product_t194;
      const s_t reference_product_t204 = reference_product_t144 + reference_product_t184 + reference_product_t200;
      const s_t reference_product_t205 = -reference_product_t202;
      const s_t reference_product_t206 = -reference_product_t199;
      const s_t reference_product_t207 = reference_product_t139 - reference_product_t140 + reference_product_t163 + reference_product_t195;
      const s_t reference_product_t208 = ((s_t(1) / s_t(9)))*hz_1;
      const s_t reference_product_t209 = ((s_t(1) / s_t(36)))*hz_6;
      const s_t reference_product_t210 = ((s_t(1) / s_t(18)))*hz_3;
      const s_t reference_product_t211 = ((s_t(1) / s_t(18)))*hz_4;
      const s_t reference_product_t212 = -reference_product_t210 + reference_product_t211;
      const s_t reference_product_t213 = ((s_t(1) / s_t(18)))*hz_5;
      const s_t reference_product_t214 = ((s_t(1) / s_t(18)))*hz_2;
      const s_t reference_product_t215 = -reference_product_t213 + reference_product_t214;
      const s_t reference_product_t216 = reference_product_t212 + reference_product_t215;
      const s_t reference_product_t217 = ((s_t(1) / s_t(9)))*hz_0 - (s_t(1) / s_t(36))*hz_7;
      const s_t reference_product_t218 = -reference_product_t208 + reference_product_t209 + reference_product_t216 + reference_product_t217;
      const s_t reference_product_t219 = ((s_t(1) / s_t(12)))*hz_1;
      const s_t reference_product_t220 = ((s_t(1) / s_t(24)))*hz_6;
      const s_t reference_product_t221 = ((s_t(1) / s_t(12)))*hz_0 - (s_t(1) / s_t(24))*hz_7;
      const s_t reference_product_t222 = -reference_product_t219 + reference_product_t220 + reference_product_t221;
      const s_t reference_product_t223 = ((s_t(1) / s_t(12)))*hz_3;
      const s_t reference_product_t224 = ((s_t(1) / s_t(24)))*hz_4;
      const s_t reference_product_t225 = -reference_product_t223 + reference_product_t224;
      const s_t reference_product_t226 = ((s_t(1) / s_t(24)))*hz_5;
      const s_t reference_product_t227 = ((s_t(1) / s_t(12)))*hz_2;
      const s_t reference_product_t228 = -reference_product_t226 + reference_product_t227;
      const s_t reference_product_t229 = reference_product_t225 + reference_product_t228;
      const s_t reference_product_t230 = reference_product_t222 + reference_product_t229;
      const s_t reference_product_t231 = ((s_t(1) / s_t(24)))*hz_3;
      const s_t reference_product_t232 = ((s_t(1) / s_t(12)))*hz_4;
      const s_t reference_product_t233 = -reference_product_t231 + reference_product_t232;
      const s_t reference_product_t234 = ((s_t(1) / s_t(12)))*hz_5;
      const s_t reference_product_t235 = ((s_t(1) / s_t(24)))*hz_2;
      const s_t reference_product_t236 = -reference_product_t234 + reference_product_t235;
      const s_t reference_product_t237 = reference_product_t233 + reference_product_t236;
      const s_t reference_product_t238 = reference_product_t222 + reference_product_t237;
      const s_t reference_product_t239 = ((s_t(1) / s_t(18)))*hz_1;
      const s_t reference_product_t240 = ((s_t(1) / s_t(18)))*hz_6;
      const s_t reference_product_t241 = ((s_t(1) / s_t(18)))*hz_0 - (s_t(1) / s_t(18))*hz_7;
      const s_t reference_product_t242 = -reference_product_t239 + reference_product_t240 + reference_product_t241;
      const s_t reference_product_t243 = ((s_t(1) / s_t(9)))*hz_3;
      const s_t reference_product_t244 = ((s_t(1) / s_t(36)))*hz_4;
      const s_t reference_product_t245 = -reference_product_t243 + reference_product_t244;
      const s_t reference_product_t246 = ((s_t(1) / s_t(36)))*hz_5;
      const s_t reference_product_t247 = ((s_t(1) / s_t(9)))*hz_2;
      const s_t reference_product_t248 = -reference_product_t246 + reference_product_t247;
      const s_t reference_product_t249 = reference_product_t242 + reference_product_t245 + reference_product_t248;
      const s_t reference_product_t250 = -reference_product_t230;
      const s_t reference_product_t251 = ((s_t(1) / s_t(24)))*hz_1;
      const s_t reference_product_t252 = ((s_t(1) / s_t(12)))*hz_6;
      const s_t reference_product_t253 = ((s_t(1) / s_t(24)))*hz_0 - (s_t(1) / s_t(12))*hz_7;
      const s_t reference_product_t254 = -reference_product_t251 + reference_product_t252 + reference_product_t253;
      const s_t reference_product_t255 = reference_product_t229 + reference_product_t254;
      const s_t reference_product_t256 = ((s_t(1) / s_t(36)))*hz_3;
      const s_t reference_product_t257 = ((s_t(1) / s_t(9)))*hz_4;
      const s_t reference_product_t258 = -reference_product_t256 + reference_product_t257;
      const s_t reference_product_t259 = ((s_t(1) / s_t(9)))*hz_5;
      const s_t reference_product_t260 = ((s_t(1) / s_t(36)))*hz_2;
      const s_t reference_product_t261 = -reference_product_t259 + reference_product_t260;
      const s_t reference_product_t262 = reference_product_t242 + reference_product_t258 + reference_product_t261;
      const s_t reference_product_t263 = reference_product_t237 + reference_product_t254;
      const s_t reference_product_t264 = -reference_product_t238;
      const s_t reference_product_t265 = ((s_t(1) / s_t(36)))*hz_1;
      const s_t reference_product_t266 = ((s_t(1) / s_t(9)))*hz_6;
      const s_t reference_product_t267 = ((s_t(1) / s_t(36)))*hz_0 - (s_t(1) / s_t(9))*hz_7;
      const s_t reference_product_t268 = reference_product_t216 - reference_product_t265 + reference_product_t266 + reference_product_t267;
      const s_t reference_product_t269 = -reference_product_t263;
      const s_t reference_product_t270 = -reference_product_t255;
      const s_t reference_product_t271 = reference_product_t221 + reference_product_t226 - reference_product_t227;
      const s_t reference_product_t272 = reference_product_t219 - reference_product_t220;
      const s_t reference_product_t273 = reference_product_t225 + reference_product_t272;
      const s_t reference_product_t274 = reference_product_t271 + reference_product_t273;
      const s_t reference_product_t275 = reference_product_t239 - reference_product_t240;
      const s_t reference_product_t276 = reference_product_t212 + reference_product_t275;
      const s_t reference_product_t277 = reference_product_t217 + reference_product_t246 - reference_product_t247 + reference_product_t276;
      const s_t reference_product_t278 = reference_product_t251 - reference_product_t252;
      const s_t reference_product_t279 = reference_product_t233 + reference_product_t278;
      const s_t reference_product_t280 = reference_product_t271 + reference_product_t279;
      const s_t reference_product_t281 = -reference_product_t274;
      const s_t reference_product_t282 = reference_product_t213 - reference_product_t214 + reference_product_t241;
      const s_t reference_product_t283 = reference_product_t208 - reference_product_t209;
      const s_t reference_product_t284 = reference_product_t245 + reference_product_t282 + reference_product_t283;
      const s_t reference_product_t285 = reference_product_t234 - reference_product_t235 + reference_product_t253;
      const s_t reference_product_t286 = reference_product_t273 + reference_product_t285;
      const s_t reference_product_t287 = reference_product_t279 + reference_product_t285;
      const s_t reference_product_t288 = reference_product_t265 - reference_product_t266;
      const s_t reference_product_t289 = reference_product_t258 + reference_product_t282 + reference_product_t288;
      const s_t reference_product_t290 = -reference_product_t280;
      const s_t reference_product_t291 = -reference_product_t287;
      const s_t reference_product_t292 = reference_product_t259 - reference_product_t260 + reference_product_t267 + reference_product_t276;
      const s_t reference_product_t293 = -reference_product_t286;
      const s_t reference_product_t294 = reference_product_t236 + reference_product_t272;
      const s_t reference_product_t295 = reference_product_t221 + reference_product_t231 - reference_product_t232;
      const s_t reference_product_t296 = reference_product_t294 + reference_product_t295;
      const s_t reference_product_t297 = reference_product_t228 + reference_product_t278;
      const s_t reference_product_t298 = reference_product_t295 + reference_product_t297;
      const s_t reference_product_t299 = reference_product_t215 + reference_product_t275;
      const s_t reference_product_t300 = reference_product_t217 + reference_product_t256 - reference_product_t257 + reference_product_t299;
      const s_t reference_product_t301 = -reference_product_t296;
      const s_t reference_product_t302 = reference_product_t223 - reference_product_t224 + reference_product_t253;
      const s_t reference_product_t303 = reference_product_t294 + reference_product_t302;
      const s_t reference_product_t304 = reference_product_t210 - reference_product_t211 + reference_product_t241;
      const s_t reference_product_t305 = reference_product_t261 + reference_product_t283 + reference_product_t304;
      const s_t reference_product_t306 = reference_product_t297 + reference_product_t302;
      const s_t reference_product_t307 = -reference_product_t298;
      const s_t reference_product_t308 = reference_product_t248 + reference_product_t288 + reference_product_t304;
      const s_t reference_product_t309 = -reference_product_t306;
      const s_t reference_product_t310 = -reference_product_t303;
      const s_t reference_product_t311 = reference_product_t243 - reference_product_t244 + reference_product_t267 + reference_product_t299;
      const s_t output_t0 = reference_product_t10*tangent0;
      const s_t output_t1 = reference_product_t114*tangent3;
      const s_t output_t2 = reference_product_t173*tangent12;
      const s_t output_t3 = reference_product_t196*tangent20;
      const s_t output_t4 = reference_product_t218*tangent6;
      const s_t output_t5 = reference_product_t277*tangent15;
      const s_t output_t6 = reference_product_t300*tangent23;
      const s_t output_t7 = reference_product_t69*tangent9;
      const s_t output_t8 = reference_product_t92*tangent17;
      const s_t output_t9 = reference_product_t126*tangent11 + reference_product_t134*tangent18 + reference_product_t22*tangent1 + reference_product_t230*tangent14 + reference_product_t238*tangent21 + reference_product_t30*tangent2;
      const s_t output_t10 = reference_product_t170*tangent4 + reference_product_t176*tangent19 + reference_product_t274*tangent7 + reference_product_t280*tangent22 + reference_product_t66*tangent1 + reference_product_t72*tangent10;
      const s_t output_t11 = reference_product_t192*tangent5 + reference_product_t194*tangent13 + reference_product_t296*tangent8 + reference_product_t298*tangent16 + reference_product_t88*tangent2 + reference_product_t90*tangent10;
      const s_t output_t12 = reference_product_t180*tangent12;
      const s_t output_t13 = reference_product_t201*tangent20;
      const s_t output_t14 = reference_product_t284*tangent15;
      const s_t output_t15 = reference_product_t305*tangent23;
      const s_t output_t16 = reference_product_t76*tangent9;
      const s_t output_t17 = reference_product_t97*tangent17;
      const s_t output_t18 = reference_product_t177*tangent4 + reference_product_t182*tangent19 + reference_product_t281*tangent7 + reference_product_t286*tangent22 + reference_product_t73*tangent1 + reference_product_t78*tangent10;
      const s_t output_t19 = reference_product_t197*tangent5 + reference_product_t199*tangent13 + reference_product_t301*tangent8 + reference_product_t303*tangent16 + reference_product_t93*tangent2 + reference_product_t95*tangent10;
      const s_t output_t20 = reference_product_t100*tangent17;
      const s_t output_t21 = reference_product_t145*tangent3;
      const s_t output_t22 = reference_product_t204*tangent20;
      const s_t output_t23 = reference_product_t249*tangent6;
      const s_t output_t24 = reference_product_t308*tangent23;
      const s_t output_t25 = reference_product_t41*tangent0;
      const s_t output_t26 = reference_product_t146*tangent11 + reference_product_t151*tangent18 + reference_product_t250*tangent14 + reference_product_t255*tangent21 + reference_product_t42*tangent1 + reference_product_t47*tangent2;
      const s_t output_t27 = reference_product_t202*tangent5 + reference_product_t203*tangent13 + reference_product_t306*tangent8 + reference_product_t307*tangent16 + reference_product_t98*tangent2 + reference_product_t99*tangent10;
      const s_t output_t28 = reference_product_t103*tangent17;
      const s_t output_t29 = reference_product_t207*tangent20;
      const s_t output_t30 = reference_product_t311*tangent23;
      const s_t output_t31 = reference_product_t101*tangent2 + reference_product_t102*tangent10 + reference_product_t205*tangent5 + reference_product_t206*tangent13 + reference_product_t309*tangent8 + reference_product_t310*tangent16;
      const s_t output_t32 = reference_product_t158*tangent3;
      const s_t output_t33 = reference_product_t185*tangent12;
      const s_t output_t34 = reference_product_t262*tangent6;
      const s_t output_t35 = reference_product_t289*tangent15;
      const s_t output_t36 = reference_product_t54*tangent0;
      const s_t output_t37 = reference_product_t81*tangent9;
      const s_t output_t38 = reference_product_t159*tangent11 + reference_product_t160*tangent18 + reference_product_t263*tangent14 + reference_product_t264*tangent21 + reference_product_t55*tangent1 + reference_product_t56*tangent2;
      const s_t output_t39 = reference_product_t183*tangent4 + reference_product_t186*tangent19 + reference_product_t287*tangent7 + reference_product_t290*tangent22 + reference_product_t79*tangent1 + reference_product_t82*tangent10;
      const s_t output_t40 = reference_product_t188*tangent12;
      const s_t output_t41 = reference_product_t292*tangent15;
      const s_t output_t42 = reference_product_t84*tangent9;
      const s_t output_t43 = reference_product_t187*tangent4 + reference_product_t189*tangent19 + reference_product_t291*tangent7 + reference_product_t293*tangent22 + reference_product_t83*tangent1 + reference_product_t85*tangent10;
      const s_t output_t44 = reference_product_t164*tangent3;
      const s_t output_t45 = reference_product_t268*tangent6;
      const s_t output_t46 = reference_product_t60*tangent0;
      const s_t output_t47 = reference_product_t165*tangent11 + reference_product_t166*tangent18 + reference_product_t269*tangent14 + reference_product_t270*tangent21 + reference_product_t61*tangent1 + reference_product_t62*tangent2;
      const s_t output_t48 = reference_product_t10*tangent3;
      const s_t output_t49 = reference_product_t114*tangent24;
      const s_t output_t50 = reference_product_t173*tangent30;
      const s_t output_t51 = reference_product_t196*tangent35;
      const s_t output_t52 = reference_product_t218*tangent27;
      const s_t output_t53 = reference_product_t277*tangent33;
      const s_t output_t54 = reference_product_t300*tangent38;
      const s_t output_t55 = reference_product_t69*tangent12;
      const s_t output_t56 = reference_product_t92*tangent20;
      const s_t output_t57 = reference_product_t126*tangent25 + reference_product_t134*tangent26 + reference_product_t22*tangent4 + reference_product_t230*tangent32 + reference_product_t238*tangent36 + reference_product_t30*tangent5;
      const s_t output_t58 = reference_product_t170*tangent25 + reference_product_t176*tangent31 + reference_product_t274*tangent28 + reference_product_t280*tangent37 + reference_product_t66*tangent11 + reference_product_t72*tangent13;
      const s_t output_t59 = reference_product_t192*tangent26 + reference_product_t194*tangent31 + reference_product_t296*tangent29 + reference_product_t298*tangent34 + reference_product_t88*tangent18 + reference_product_t90*tangent19;
      const s_t output_t60 = reference_product_t180*tangent30;
      const s_t output_t61 = reference_product_t201*tangent35;
      const s_t output_t62 = reference_product_t284*tangent33;
      const s_t output_t63 = reference_product_t305*tangent38;
      const s_t output_t64 = reference_product_t76*tangent12;
      const s_t output_t65 = reference_product_t97*tangent20;
      const s_t output_t66 = reference_product_t177*tangent25 + reference_product_t182*tangent31 + reference_product_t281*tangent28 + reference_product_t286*tangent37 + reference_product_t73*tangent11 + reference_product_t78*tangent13;
      const s_t output_t67 = reference_product_t197*tangent26 + reference_product_t199*tangent31 + reference_product_t301*tangent29 + reference_product_t303*tangent34 + reference_product_t93*tangent18 + reference_product_t95*tangent19;
      const s_t output_t68 = reference_product_t100*tangent20;
      const s_t output_t69 = reference_product_t145*tangent24;
      const s_t output_t70 = reference_product_t204*tangent35;
      const s_t output_t71 = reference_product_t249*tangent27;
      const s_t output_t72 = reference_product_t308*tangent38;
      const s_t output_t73 = reference_product_t41*tangent3;
      const s_t output_t74 = reference_product_t146*tangent25 + reference_product_t151*tangent26 + reference_product_t250*tangent32 + reference_product_t255*tangent36 + reference_product_t42*tangent4 + reference_product_t47*tangent5;
      const s_t output_t75 = reference_product_t202*tangent26 + reference_product_t203*tangent31 + reference_product_t306*tangent29 + reference_product_t307*tangent34 + reference_product_t98*tangent18 + reference_product_t99*tangent19;
      const s_t output_t76 = reference_product_t103*tangent20;
      const s_t output_t77 = reference_product_t207*tangent35;
      const s_t output_t78 = reference_product_t311*tangent38;
      const s_t output_t79 = reference_product_t101*tangent18 + reference_product_t102*tangent19 + reference_product_t205*tangent26 + reference_product_t206*tangent31 + reference_product_t309*tangent29 + reference_product_t310*tangent34;
      const s_t output_t80 = reference_product_t158*tangent24;
      const s_t output_t81 = reference_product_t185*tangent30;
      const s_t output_t82 = reference_product_t262*tangent27;
      const s_t output_t83 = reference_product_t289*tangent33;
      const s_t output_t84 = reference_product_t54*tangent3;
      const s_t output_t85 = reference_product_t81*tangent12;
      const s_t output_t86 = reference_product_t159*tangent25 + reference_product_t160*tangent26 + reference_product_t263*tangent32 + reference_product_t264*tangent36 + reference_product_t55*tangent4 + reference_product_t56*tangent5;
      const s_t output_t87 = reference_product_t183*tangent25 + reference_product_t186*tangent31 + reference_product_t287*tangent28 + reference_product_t290*tangent37 + reference_product_t79*tangent11 + reference_product_t82*tangent13;
      const s_t output_t88 = reference_product_t188*tangent30;
      const s_t output_t89 = reference_product_t292*tangent33;
      const s_t output_t90 = reference_product_t84*tangent12;
      const s_t output_t91 = reference_product_t187*tangent25 + reference_product_t189*tangent31 + reference_product_t291*tangent28 + reference_product_t293*tangent37 + reference_product_t83*tangent11 + reference_product_t85*tangent13;
      const s_t output_t92 = reference_product_t164*tangent24;
      const s_t output_t93 = reference_product_t268*tangent27;
      const s_t output_t94 = reference_product_t60*tangent3;
      const s_t output_t95 = reference_product_t165*tangent25 + reference_product_t166*tangent26 + reference_product_t269*tangent32 + reference_product_t270*tangent36 + reference_product_t61*tangent4 + reference_product_t62*tangent5;
      const s_t output_t96 = reference_product_t10*tangent6;
      const s_t output_t97 = reference_product_t114*tangent27;
      const s_t output_t98 = reference_product_t173*tangent33;
      const s_t output_t99 = reference_product_t196*tangent38;
      const s_t output_t100 = reference_product_t218*tangent39;
      const s_t output_t101 = reference_product_t277*tangent42;
      const s_t output_t102 = reference_product_t300*tangent44;
      const s_t output_t103 = reference_product_t69*tangent15;
      const s_t output_t104 = reference_product_t92*tangent23;
      const s_t output_t105 = reference_product_t126*tangent28 + reference_product_t134*tangent29 + reference_product_t22*tangent7 + reference_product_t230*tangent40 + reference_product_t238*tangent41 + reference_product_t30*tangent8;
      const s_t output_t106 = reference_product_t170*tangent32 + reference_product_t176*tangent34 + reference_product_t274*tangent40 + reference_product_t280*tangent43 + reference_product_t66*tangent14 + reference_product_t72*tangent16;
      const s_t output_t107 = reference_product_t192*tangent36 + reference_product_t194*tangent37 + reference_product_t296*tangent41 + reference_product_t298*tangent43 + reference_product_t88*tangent21 + reference_product_t90*tangent22;
      const s_t output_t108 = reference_product_t180*tangent33;
      const s_t output_t109 = reference_product_t201*tangent38;
      const s_t output_t110 = reference_product_t284*tangent42;
      const s_t output_t111 = reference_product_t305*tangent44;
      const s_t output_t112 = reference_product_t76*tangent15;
      const s_t output_t113 = reference_product_t97*tangent23;
      const s_t output_t114 = reference_product_t177*tangent32 + reference_product_t182*tangent34 + reference_product_t281*tangent40 + reference_product_t286*tangent43 + reference_product_t73*tangent14 + reference_product_t78*tangent16;
      const s_t output_t115 = reference_product_t197*tangent36 + reference_product_t199*tangent37 + reference_product_t301*tangent41 + reference_product_t303*tangent43 + reference_product_t93*tangent21 + reference_product_t95*tangent22;
      const s_t output_t116 = reference_product_t100*tangent23;
      const s_t output_t117 = reference_product_t145*tangent27;
      const s_t output_t118 = reference_product_t204*tangent38;
      const s_t output_t119 = reference_product_t249*tangent39;
      const s_t output_t120 = reference_product_t308*tangent44;
      const s_t output_t121 = reference_product_t41*tangent6;
      const s_t output_t122 = reference_product_t146*tangent28 + reference_product_t151*tangent29 + reference_product_t250*tangent40 + reference_product_t255*tangent41 + reference_product_t42*tangent7 + reference_product_t47*tangent8;
      const s_t output_t123 = reference_product_t202*tangent36 + reference_product_t203*tangent37 + reference_product_t306*tangent41 + reference_product_t307*tangent43 + reference_product_t98*tangent21 + reference_product_t99*tangent22;
      const s_t output_t124 = reference_product_t103*tangent23;
      const s_t output_t125 = reference_product_t207*tangent38;
      const s_t output_t126 = reference_product_t311*tangent44;
      const s_t output_t127 = reference_product_t101*tangent21 + reference_product_t102*tangent22 + reference_product_t205*tangent36 + reference_product_t206*tangent37 + reference_product_t309*tangent41 + reference_product_t310*tangent43;
      const s_t output_t128 = reference_product_t158*tangent27;
      const s_t output_t129 = reference_product_t185*tangent33;
      const s_t output_t130 = reference_product_t262*tangent39;
      const s_t output_t131 = reference_product_t289*tangent42;
      const s_t output_t132 = reference_product_t54*tangent6;
      const s_t output_t133 = reference_product_t81*tangent15;
      const s_t output_t134 = reference_product_t159*tangent28 + reference_product_t160*tangent29 + reference_product_t263*tangent40 + reference_product_t264*tangent41 + reference_product_t55*tangent7 + reference_product_t56*tangent8;
      const s_t output_t135 = reference_product_t183*tangent32 + reference_product_t186*tangent34 + reference_product_t287*tangent40 + reference_product_t290*tangent43 + reference_product_t79*tangent14 + reference_product_t82*tangent16;
      const s_t output_t136 = reference_product_t188*tangent33;
      const s_t output_t137 = reference_product_t292*tangent42;
      const s_t output_t138 = reference_product_t84*tangent15;
      const s_t output_t139 = reference_product_t187*tangent32 + reference_product_t189*tangent34 + reference_product_t291*tangent40 + reference_product_t293*tangent43 + reference_product_t83*tangent14 + reference_product_t85*tangent16;
      const s_t output_t140 = reference_product_t164*tangent27;
      const s_t output_t141 = reference_product_t268*tangent39;
      const s_t output_t142 = reference_product_t60*tangent6;
      const s_t output_t143 = reference_product_t165*tangent28 + reference_product_t166*tangent29 + reference_product_t269*tangent40 + reference_product_t270*tangent41 + reference_product_t61*tangent7 + reference_product_t62*tangent8;
      const s_t element_out0_0 = output_t0 + output_t1 + output_t10 + output_t11 + output_t2 + output_t3 + output_t4 + output_t5 + output_t6 + output_t7 + output_t8 + output_t9;
      const s_t element_out0_1 = -output_t0 - output_t1 + output_t12 + output_t13 + output_t14 + output_t15 + output_t16 + output_t17 + output_t18 + output_t19 - output_t4 + output_t9;
      const s_t element_out0_2 = output_t10 - output_t2 + output_t20 + output_t21 + output_t22 + output_t23 + output_t24 + output_t25 + output_t26 + output_t27 - output_t5 - output_t7;
      const s_t element_out0_3 = -output_t12 - output_t14 - output_t16 + output_t18 - output_t21 - output_t23 - output_t25 + output_t26 + output_t28 + output_t29 + output_t30 + output_t31;
      const s_t element_out0_4 = output_t11 - output_t3 + output_t32 + output_t33 + output_t34 + output_t35 + output_t36 + output_t37 + output_t38 + output_t39 - output_t6 - output_t8;
      const s_t element_out0_5 = -output_t13 - output_t15 - output_t17 + output_t19 - output_t32 - output_t34 - output_t36 + output_t38 + output_t40 + output_t41 + output_t42 + output_t43;
      const s_t element_out0_6 = -output_t20 - output_t22 - output_t24 + output_t27 - output_t33 - output_t35 - output_t37 + output_t39 + output_t44 + output_t45 + output_t46 + output_t47;
      const s_t element_out0_7 = -output_t28 - output_t29 - output_t30 + output_t31 - output_t40 - output_t41 - output_t42 + output_t43 - output_t44 - output_t45 - output_t46 + output_t47;
      const s_t element_out1_0 = output_t48 + output_t49 + output_t50 + output_t51 + output_t52 + output_t53 + output_t54 + output_t55 + output_t56 + output_t57 + output_t58 + output_t59;
      const s_t element_out1_1 = -output_t48 - output_t49 - output_t52 + output_t57 + output_t60 + output_t61 + output_t62 + output_t63 + output_t64 + output_t65 + output_t66 + output_t67;
      const s_t element_out1_2 = -output_t50 - output_t53 - output_t55 + output_t58 + output_t68 + output_t69 + output_t70 + output_t71 + output_t72 + output_t73 + output_t74 + output_t75;
      const s_t element_out1_3 = -output_t60 - output_t62 - output_t64 + output_t66 - output_t69 - output_t71 - output_t73 + output_t74 + output_t76 + output_t77 + output_t78 + output_t79;
      const s_t element_out1_4 = -output_t51 - output_t54 - output_t56 + output_t59 + output_t80 + output_t81 + output_t82 + output_t83 + output_t84 + output_t85 + output_t86 + output_t87;
      const s_t element_out1_5 = -output_t61 - output_t63 - output_t65 + output_t67 - output_t80 - output_t82 - output_t84 + output_t86 + output_t88 + output_t89 + output_t90 + output_t91;
      const s_t element_out1_6 = -output_t68 - output_t70 - output_t72 + output_t75 - output_t81 - output_t83 - output_t85 + output_t87 + output_t92 + output_t93 + output_t94 + output_t95;
      const s_t element_out1_7 = -output_t76 - output_t77 - output_t78 + output_t79 - output_t88 - output_t89 - output_t90 + output_t91 - output_t92 - output_t93 - output_t94 + output_t95;
      const s_t element_out2_0 = output_t100 + output_t101 + output_t102 + output_t103 + output_t104 + output_t105 + output_t106 + output_t107 + output_t96 + output_t97 + output_t98 + output_t99;
      const s_t element_out2_1 = -output_t100 + output_t105 + output_t108 + output_t109 + output_t110 + output_t111 + output_t112 + output_t113 + output_t114 + output_t115 - output_t96 - output_t97;
      const s_t element_out2_2 = -output_t101 - output_t103 + output_t106 + output_t116 + output_t117 + output_t118 + output_t119 + output_t120 + output_t121 + output_t122 + output_t123 - output_t98;
      const s_t element_out2_3 = -output_t108 - output_t110 - output_t112 + output_t114 - output_t117 - output_t119 - output_t121 + output_t122 + output_t124 + output_t125 + output_t126 + output_t127;
      const s_t element_out2_4 = -output_t102 - output_t104 + output_t107 + output_t128 + output_t129 + output_t130 + output_t131 + output_t132 + output_t133 + output_t134 + output_t135 - output_t99;
      const s_t element_out2_5 = -output_t109 - output_t111 - output_t113 + output_t115 - output_t128 - output_t130 - output_t132 + output_t134 + output_t136 + output_t137 + output_t138 + output_t139;
      const s_t element_out2_6 = -output_t116 - output_t118 - output_t120 + output_t123 - output_t129 - output_t131 - output_t133 + output_t135 + output_t140 + output_t141 + output_t142 + output_t143;
      const s_t element_out2_7 = -output_t124 - output_t125 - output_t126 + output_t127 - output_t136 - output_t137 - output_t138 + output_t139 - output_t140 - output_t141 - output_t142 + output_t143;
      bout0_0[lane] = element_out0_0;
      bout0_1[lane] = element_out0_1;
      bout0_2[lane] = element_out0_2;
      bout0_3[lane] = element_out0_3;
      bout0_4[lane] = element_out0_4;
      bout0_5[lane] = element_out0_5;
      bout0_6[lane] = element_out0_6;
      bout0_7[lane] = element_out0_7;
      bout1_0[lane] = element_out1_0;
      bout1_1[lane] = element_out1_1;
      bout1_2[lane] = element_out1_2;
      bout1_3[lane] = element_out1_3;
      bout1_4[lane] = element_out1_4;
      bout1_5[lane] = element_out1_5;
      bout1_6[lane] = element_out1_6;
      bout1_7[lane] = element_out1_7;
      bout2_0[lane] = element_out2_0;
      bout2_1[lane] = element_out2_1;
      bout2_2[lane] = element_out2_2;
      bout2_3[lane] = element_out2_3;
      bout2_4[lane] = element_out2_4;
      bout2_5[lane] = element_out2_5;
      bout2_6[lane] = element_out2_6;
      bout2_7[lane] = element_out2_7;
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
  }

  return SFEM_SUCCESS;
}

template <typename s_t, typename tangent_t, int VS>
static SFEM_INLINE int linear_elasticity_proteus_hex8_inexact_apply_stored_packed_two_pass_a_msoa_impl(
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
        }
        s_t bhx_0[VS];
        s_t bhx_1[VS];
        s_t bhx_2[VS];
        s_t bhx_3[VS];
        s_t bhx_4[VS];
        s_t bhx_5[VS];
        s_t bhx_6[VS];
        s_t bhx_7[VS];
        s_t bhy_0[VS];
        s_t bhy_1[VS];
        s_t bhy_2[VS];
        s_t bhy_3[VS];
        s_t bhy_4[VS];
        s_t bhy_5[VS];
        s_t bhy_6[VS];
        s_t bhy_7[VS];
        s_t bhz_0[VS];
        s_t bhz_1[VS];
        s_t bhz_2[VS];
        s_t bhz_3[VS];
        s_t bhz_4[VS];
        s_t bhz_5[VS];
        s_t bhz_6[VS];
        s_t bhz_7[VS];
        s_t bout0_0[VS];
        s_t bout0_1[VS];
        s_t bout0_2[VS];
        s_t bout0_3[VS];
        s_t bout0_4[VS];
        s_t bout0_5[VS];
        s_t bout0_6[VS];
        s_t bout0_7[VS];
        s_t bout1_0[VS];
        s_t bout1_1[VS];
        s_t bout1_2[VS];
        s_t bout1_3[VS];
        s_t bout1_4[VS];
        s_t bout1_5[VS];
        s_t bout1_6[VS];
        s_t bout1_7[VS];
        s_t bout2_0[VS];
        s_t bout2_1[VS];
        s_t bout2_2[VS];
        s_t bout2_3[VS];
        s_t bout2_4[VS];
        s_t bout2_5[VS];
        s_t bout2_6[VS];
        s_t bout2_7[VS];
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
          bhy_0[lane] = pk_h[1 * max_nodes_per_pack + bev0[lane]];
          bhy_1[lane] = pk_h[1 * max_nodes_per_pack + bev1[lane]];
          bhy_2[lane] = pk_h[1 * max_nodes_per_pack + bev2[lane]];
          bhy_3[lane] = pk_h[1 * max_nodes_per_pack + bev3[lane]];
          bhy_4[lane] = pk_h[1 * max_nodes_per_pack + bev4[lane]];
          bhy_5[lane] = pk_h[1 * max_nodes_per_pack + bev5[lane]];
          bhy_6[lane] = pk_h[1 * max_nodes_per_pack + bev6[lane]];
          bhy_7[lane] = pk_h[1 * max_nodes_per_pack + bev7[lane]];
          bhz_0[lane] = pk_h[2 * max_nodes_per_pack + bev0[lane]];
          bhz_1[lane] = pk_h[2 * max_nodes_per_pack + bev1[lane]];
          bhz_2[lane] = pk_h[2 * max_nodes_per_pack + bev2[lane]];
          bhz_3[lane] = pk_h[2 * max_nodes_per_pack + bev3[lane]];
          bhz_4[lane] = pk_h[2 * max_nodes_per_pack + bev4[lane]];
          bhz_5[lane] = pk_h[2 * max_nodes_per_pack + bev5[lane]];
          bhz_6[lane] = pk_h[2 * max_nodes_per_pack + bev6[lane]];
          bhz_7[lane] = pk_h[2 * max_nodes_per_pack + bev7[lane]];
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
          const s_t hy_0 = bhy_0[lane];
          const s_t hy_1 = bhy_1[lane];
          const s_t hy_2 = bhy_2[lane];
          const s_t hy_3 = bhy_3[lane];
          const s_t hy_4 = bhy_4[lane];
          const s_t hy_5 = bhy_5[lane];
          const s_t hy_6 = bhy_6[lane];
          const s_t hy_7 = bhy_7[lane];
          const s_t hz_0 = bhz_0[lane];
          const s_t hz_1 = bhz_1[lane];
          const s_t hz_2 = bhz_2[lane];
          const s_t hz_3 = bhz_3[lane];
          const s_t hz_4 = bhz_4[lane];
          const s_t hz_5 = bhz_5[lane];
          const s_t hz_6 = bhz_6[lane];
          const s_t hz_7 = bhz_7[lane];
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
          const s_t reference_product_t0 = ((s_t(1) / s_t(9)))*hx_1;
          const s_t reference_product_t1 = ((s_t(1) / s_t(36)))*hx_6;
          const s_t reference_product_t2 = ((s_t(1) / s_t(18)))*hx_3;
          const s_t reference_product_t3 = ((s_t(1) / s_t(18)))*hx_4;
          const s_t reference_product_t4 = -reference_product_t2 + reference_product_t3;
          const s_t reference_product_t5 = ((s_t(1) / s_t(18)))*hx_5;
          const s_t reference_product_t6 = ((s_t(1) / s_t(18)))*hx_2;
          const s_t reference_product_t7 = -reference_product_t5 + reference_product_t6;
          const s_t reference_product_t8 = reference_product_t4 + reference_product_t7;
          const s_t reference_product_t9 = ((s_t(1) / s_t(9)))*hx_0 - (s_t(1) / s_t(36))*hx_7;
          const s_t reference_product_t10 = -reference_product_t0 + reference_product_t1 + reference_product_t8 + reference_product_t9;
          const s_t reference_product_t11 = ((s_t(1) / s_t(12)))*hx_1;
          const s_t reference_product_t12 = ((s_t(1) / s_t(24)))*hx_6;
          const s_t reference_product_t13 = ((s_t(1) / s_t(12)))*hx_0 - (s_t(1) / s_t(24))*hx_7;
          const s_t reference_product_t14 = -reference_product_t11 + reference_product_t12 + reference_product_t13;
          const s_t reference_product_t15 = ((s_t(1) / s_t(12)))*hx_3;
          const s_t reference_product_t16 = ((s_t(1) / s_t(24)))*hx_4;
          const s_t reference_product_t17 = -reference_product_t15 + reference_product_t16;
          const s_t reference_product_t18 = ((s_t(1) / s_t(24)))*hx_5;
          const s_t reference_product_t19 = ((s_t(1) / s_t(12)))*hx_2;
          const s_t reference_product_t20 = -reference_product_t18 + reference_product_t19;
          const s_t reference_product_t21 = reference_product_t17 + reference_product_t20;
          const s_t reference_product_t22 = reference_product_t14 + reference_product_t21;
          const s_t reference_product_t23 = ((s_t(1) / s_t(24)))*hx_3;
          const s_t reference_product_t24 = ((s_t(1) / s_t(12)))*hx_4;
          const s_t reference_product_t25 = -reference_product_t23 + reference_product_t24;
          const s_t reference_product_t26 = ((s_t(1) / s_t(12)))*hx_5;
          const s_t reference_product_t27 = ((s_t(1) / s_t(24)))*hx_2;
          const s_t reference_product_t28 = -reference_product_t26 + reference_product_t27;
          const s_t reference_product_t29 = reference_product_t25 + reference_product_t28;
          const s_t reference_product_t30 = reference_product_t14 + reference_product_t29;
          const s_t reference_product_t31 = ((s_t(1) / s_t(18)))*hx_1;
          const s_t reference_product_t32 = ((s_t(1) / s_t(18)))*hx_6;
          const s_t reference_product_t33 = ((s_t(1) / s_t(18)))*hx_0 - (s_t(1) / s_t(18))*hx_7;
          const s_t reference_product_t34 = -reference_product_t31 + reference_product_t32 + reference_product_t33;
          const s_t reference_product_t35 = ((s_t(1) / s_t(9)))*hx_3;
          const s_t reference_product_t36 = ((s_t(1) / s_t(36)))*hx_4;
          const s_t reference_product_t37 = -reference_product_t35 + reference_product_t36;
          const s_t reference_product_t38 = ((s_t(1) / s_t(36)))*hx_5;
          const s_t reference_product_t39 = ((s_t(1) / s_t(9)))*hx_2;
          const s_t reference_product_t40 = -reference_product_t38 + reference_product_t39;
          const s_t reference_product_t41 = reference_product_t34 + reference_product_t37 + reference_product_t40;
          const s_t reference_product_t42 = -reference_product_t22;
          const s_t reference_product_t43 = ((s_t(1) / s_t(24)))*hx_1;
          const s_t reference_product_t44 = ((s_t(1) / s_t(12)))*hx_6;
          const s_t reference_product_t45 = ((s_t(1) / s_t(24)))*hx_0 - (s_t(1) / s_t(12))*hx_7;
          const s_t reference_product_t46 = -reference_product_t43 + reference_product_t44 + reference_product_t45;
          const s_t reference_product_t47 = reference_product_t21 + reference_product_t46;
          const s_t reference_product_t48 = ((s_t(1) / s_t(36)))*hx_3;
          const s_t reference_product_t49 = ((s_t(1) / s_t(9)))*hx_4;
          const s_t reference_product_t50 = -reference_product_t48 + reference_product_t49;
          const s_t reference_product_t51 = ((s_t(1) / s_t(9)))*hx_5;
          const s_t reference_product_t52 = ((s_t(1) / s_t(36)))*hx_2;
          const s_t reference_product_t53 = -reference_product_t51 + reference_product_t52;
          const s_t reference_product_t54 = reference_product_t34 + reference_product_t50 + reference_product_t53;
          const s_t reference_product_t55 = reference_product_t29 + reference_product_t46;
          const s_t reference_product_t56 = -reference_product_t30;
          const s_t reference_product_t57 = ((s_t(1) / s_t(36)))*hx_1;
          const s_t reference_product_t58 = ((s_t(1) / s_t(9)))*hx_6;
          const s_t reference_product_t59 = ((s_t(1) / s_t(36)))*hx_0 - (s_t(1) / s_t(9))*hx_7;
          const s_t reference_product_t60 = -reference_product_t57 + reference_product_t58 + reference_product_t59 + reference_product_t8;
          const s_t reference_product_t61 = -reference_product_t55;
          const s_t reference_product_t62 = -reference_product_t47;
          const s_t reference_product_t63 = reference_product_t13 + reference_product_t18 - reference_product_t19;
          const s_t reference_product_t64 = reference_product_t11 - reference_product_t12;
          const s_t reference_product_t65 = reference_product_t17 + reference_product_t64;
          const s_t reference_product_t66 = reference_product_t63 + reference_product_t65;
          const s_t reference_product_t67 = reference_product_t31 - reference_product_t32;
          const s_t reference_product_t68 = reference_product_t4 + reference_product_t67;
          const s_t reference_product_t69 = reference_product_t38 - reference_product_t39 + reference_product_t68 + reference_product_t9;
          const s_t reference_product_t70 = reference_product_t43 - reference_product_t44;
          const s_t reference_product_t71 = reference_product_t25 + reference_product_t70;
          const s_t reference_product_t72 = reference_product_t63 + reference_product_t71;
          const s_t reference_product_t73 = -reference_product_t66;
          const s_t reference_product_t74 = reference_product_t33 + reference_product_t5 - reference_product_t6;
          const s_t reference_product_t75 = reference_product_t0 - reference_product_t1;
          const s_t reference_product_t76 = reference_product_t37 + reference_product_t74 + reference_product_t75;
          const s_t reference_product_t77 = reference_product_t26 - reference_product_t27 + reference_product_t45;
          const s_t reference_product_t78 = reference_product_t65 + reference_product_t77;
          const s_t reference_product_t79 = reference_product_t71 + reference_product_t77;
          const s_t reference_product_t80 = reference_product_t57 - reference_product_t58;
          const s_t reference_product_t81 = reference_product_t50 + reference_product_t74 + reference_product_t80;
          const s_t reference_product_t82 = -reference_product_t72;
          const s_t reference_product_t83 = -reference_product_t79;
          const s_t reference_product_t84 = reference_product_t51 - reference_product_t52 + reference_product_t59 + reference_product_t68;
          const s_t reference_product_t85 = -reference_product_t78;
          const s_t reference_product_t86 = reference_product_t28 + reference_product_t64;
          const s_t reference_product_t87 = reference_product_t13 + reference_product_t23 - reference_product_t24;
          const s_t reference_product_t88 = reference_product_t86 + reference_product_t87;
          const s_t reference_product_t89 = reference_product_t20 + reference_product_t70;
          const s_t reference_product_t90 = reference_product_t87 + reference_product_t89;
          const s_t reference_product_t91 = reference_product_t67 + reference_product_t7;
          const s_t reference_product_t92 = reference_product_t48 - reference_product_t49 + reference_product_t9 + reference_product_t91;
          const s_t reference_product_t93 = -reference_product_t88;
          const s_t reference_product_t94 = reference_product_t15 - reference_product_t16 + reference_product_t45;
          const s_t reference_product_t95 = reference_product_t86 + reference_product_t94;
          const s_t reference_product_t96 = reference_product_t2 - reference_product_t3 + reference_product_t33;
          const s_t reference_product_t97 = reference_product_t53 + reference_product_t75 + reference_product_t96;
          const s_t reference_product_t98 = reference_product_t89 + reference_product_t94;
          const s_t reference_product_t99 = -reference_product_t90;
          const s_t reference_product_t100 = reference_product_t40 + reference_product_t80 + reference_product_t96;
          const s_t reference_product_t101 = -reference_product_t98;
          const s_t reference_product_t102 = -reference_product_t95;
          const s_t reference_product_t103 = reference_product_t35 - reference_product_t36 + reference_product_t59 + reference_product_t91;
          const s_t reference_product_t104 = ((s_t(1) / s_t(9)))*hy_1;
          const s_t reference_product_t105 = ((s_t(1) / s_t(36)))*hy_6;
          const s_t reference_product_t106 = ((s_t(1) / s_t(18)))*hy_3;
          const s_t reference_product_t107 = ((s_t(1) / s_t(18)))*hy_4;
          const s_t reference_product_t108 = -reference_product_t106 + reference_product_t107;
          const s_t reference_product_t109 = ((s_t(1) / s_t(18)))*hy_5;
          const s_t reference_product_t110 = ((s_t(1) / s_t(18)))*hy_2;
          const s_t reference_product_t111 = -reference_product_t109 + reference_product_t110;
          const s_t reference_product_t112 = reference_product_t108 + reference_product_t111;
          const s_t reference_product_t113 = ((s_t(1) / s_t(9)))*hy_0 - (s_t(1) / s_t(36))*hy_7;
          const s_t reference_product_t114 = -reference_product_t104 + reference_product_t105 + reference_product_t112 + reference_product_t113;
          const s_t reference_product_t115 = ((s_t(1) / s_t(12)))*hy_1;
          const s_t reference_product_t116 = ((s_t(1) / s_t(24)))*hy_6;
          const s_t reference_product_t117 = ((s_t(1) / s_t(12)))*hy_0 - (s_t(1) / s_t(24))*hy_7;
          const s_t reference_product_t118 = -reference_product_t115 + reference_product_t116 + reference_product_t117;
          const s_t reference_product_t119 = ((s_t(1) / s_t(12)))*hy_3;
          const s_t reference_product_t120 = ((s_t(1) / s_t(24)))*hy_4;
          const s_t reference_product_t121 = -reference_product_t119 + reference_product_t120;
          const s_t reference_product_t122 = ((s_t(1) / s_t(24)))*hy_5;
          const s_t reference_product_t123 = ((s_t(1) / s_t(12)))*hy_2;
          const s_t reference_product_t124 = -reference_product_t122 + reference_product_t123;
          const s_t reference_product_t125 = reference_product_t121 + reference_product_t124;
          const s_t reference_product_t126 = reference_product_t118 + reference_product_t125;
          const s_t reference_product_t127 = ((s_t(1) / s_t(24)))*hy_3;
          const s_t reference_product_t128 = ((s_t(1) / s_t(12)))*hy_4;
          const s_t reference_product_t129 = -reference_product_t127 + reference_product_t128;
          const s_t reference_product_t130 = ((s_t(1) / s_t(12)))*hy_5;
          const s_t reference_product_t131 = ((s_t(1) / s_t(24)))*hy_2;
          const s_t reference_product_t132 = -reference_product_t130 + reference_product_t131;
          const s_t reference_product_t133 = reference_product_t129 + reference_product_t132;
          const s_t reference_product_t134 = reference_product_t118 + reference_product_t133;
          const s_t reference_product_t135 = ((s_t(1) / s_t(18)))*hy_1;
          const s_t reference_product_t136 = ((s_t(1) / s_t(18)))*hy_6;
          const s_t reference_product_t137 = ((s_t(1) / s_t(18)))*hy_0 - (s_t(1) / s_t(18))*hy_7;
          const s_t reference_product_t138 = -reference_product_t135 + reference_product_t136 + reference_product_t137;
          const s_t reference_product_t139 = ((s_t(1) / s_t(9)))*hy_3;
          const s_t reference_product_t140 = ((s_t(1) / s_t(36)))*hy_4;
          const s_t reference_product_t141 = -reference_product_t139 + reference_product_t140;
          const s_t reference_product_t142 = ((s_t(1) / s_t(36)))*hy_5;
          const s_t reference_product_t143 = ((s_t(1) / s_t(9)))*hy_2;
          const s_t reference_product_t144 = -reference_product_t142 + reference_product_t143;
          const s_t reference_product_t145 = reference_product_t138 + reference_product_t141 + reference_product_t144;
          const s_t reference_product_t146 = -reference_product_t126;
          const s_t reference_product_t147 = ((s_t(1) / s_t(24)))*hy_1;
          const s_t reference_product_t148 = ((s_t(1) / s_t(12)))*hy_6;
          const s_t reference_product_t149 = ((s_t(1) / s_t(24)))*hy_0 - (s_t(1) / s_t(12))*hy_7;
          const s_t reference_product_t150 = -reference_product_t147 + reference_product_t148 + reference_product_t149;
          const s_t reference_product_t151 = reference_product_t125 + reference_product_t150;
          const s_t reference_product_t152 = ((s_t(1) / s_t(36)))*hy_3;
          const s_t reference_product_t153 = ((s_t(1) / s_t(9)))*hy_4;
          const s_t reference_product_t154 = -reference_product_t152 + reference_product_t153;
          const s_t reference_product_t155 = ((s_t(1) / s_t(9)))*hy_5;
          const s_t reference_product_t156 = ((s_t(1) / s_t(36)))*hy_2;
          const s_t reference_product_t157 = -reference_product_t155 + reference_product_t156;
          const s_t reference_product_t158 = reference_product_t138 + reference_product_t154 + reference_product_t157;
          const s_t reference_product_t159 = reference_product_t133 + reference_product_t150;
          const s_t reference_product_t160 = -reference_product_t134;
          const s_t reference_product_t161 = ((s_t(1) / s_t(36)))*hy_1;
          const s_t reference_product_t162 = ((s_t(1) / s_t(9)))*hy_6;
          const s_t reference_product_t163 = ((s_t(1) / s_t(36)))*hy_0 - (s_t(1) / s_t(9))*hy_7;
          const s_t reference_product_t164 = reference_product_t112 - reference_product_t161 + reference_product_t162 + reference_product_t163;
          const s_t reference_product_t165 = -reference_product_t159;
          const s_t reference_product_t166 = -reference_product_t151;
          const s_t reference_product_t167 = reference_product_t117 + reference_product_t122 - reference_product_t123;
          const s_t reference_product_t168 = reference_product_t115 - reference_product_t116;
          const s_t reference_product_t169 = reference_product_t121 + reference_product_t168;
          const s_t reference_product_t170 = reference_product_t167 + reference_product_t169;
          const s_t reference_product_t171 = reference_product_t135 - reference_product_t136;
          const s_t reference_product_t172 = reference_product_t108 + reference_product_t171;
          const s_t reference_product_t173 = reference_product_t113 + reference_product_t142 - reference_product_t143 + reference_product_t172;
          const s_t reference_product_t174 = reference_product_t147 - reference_product_t148;
          const s_t reference_product_t175 = reference_product_t129 + reference_product_t174;
          const s_t reference_product_t176 = reference_product_t167 + reference_product_t175;
          const s_t reference_product_t177 = -reference_product_t170;
          const s_t reference_product_t178 = reference_product_t109 - reference_product_t110 + reference_product_t137;
          const s_t reference_product_t179 = reference_product_t104 - reference_product_t105;
          const s_t reference_product_t180 = reference_product_t141 + reference_product_t178 + reference_product_t179;
          const s_t reference_product_t181 = reference_product_t130 - reference_product_t131 + reference_product_t149;
          const s_t reference_product_t182 = reference_product_t169 + reference_product_t181;
          const s_t reference_product_t183 = reference_product_t175 + reference_product_t181;
          const s_t reference_product_t184 = reference_product_t161 - reference_product_t162;
          const s_t reference_product_t185 = reference_product_t154 + reference_product_t178 + reference_product_t184;
          const s_t reference_product_t186 = -reference_product_t176;
          const s_t reference_product_t187 = -reference_product_t183;
          const s_t reference_product_t188 = reference_product_t155 - reference_product_t156 + reference_product_t163 + reference_product_t172;
          const s_t reference_product_t189 = -reference_product_t182;
          const s_t reference_product_t190 = reference_product_t132 + reference_product_t168;
          const s_t reference_product_t191 = reference_product_t117 + reference_product_t127 - reference_product_t128;
          const s_t reference_product_t192 = reference_product_t190 + reference_product_t191;
          const s_t reference_product_t193 = reference_product_t124 + reference_product_t174;
          const s_t reference_product_t194 = reference_product_t191 + reference_product_t193;
          const s_t reference_product_t195 = reference_product_t111 + reference_product_t171;
          const s_t reference_product_t196 = reference_product_t113 + reference_product_t152 - reference_product_t153 + reference_product_t195;
          const s_t reference_product_t197 = -reference_product_t192;
          const s_t reference_product_t198 = reference_product_t119 - reference_product_t120 + reference_product_t149;
          const s_t reference_product_t199 = reference_product_t190 + reference_product_t198;
          const s_t reference_product_t200 = reference_product_t106 - reference_product_t107 + reference_product_t137;
          const s_t reference_product_t201 = reference_product_t157 + reference_product_t179 + reference_product_t200;
          const s_t reference_product_t202 = reference_product_t193 + reference_product_t198;
          const s_t reference_product_t203 = -reference_product_t194;
          const s_t reference_product_t204 = reference_product_t144 + reference_product_t184 + reference_product_t200;
          const s_t reference_product_t205 = -reference_product_t202;
          const s_t reference_product_t206 = -reference_product_t199;
          const s_t reference_product_t207 = reference_product_t139 - reference_product_t140 + reference_product_t163 + reference_product_t195;
          const s_t reference_product_t208 = ((s_t(1) / s_t(9)))*hz_1;
          const s_t reference_product_t209 = ((s_t(1) / s_t(36)))*hz_6;
          const s_t reference_product_t210 = ((s_t(1) / s_t(18)))*hz_3;
          const s_t reference_product_t211 = ((s_t(1) / s_t(18)))*hz_4;
          const s_t reference_product_t212 = -reference_product_t210 + reference_product_t211;
          const s_t reference_product_t213 = ((s_t(1) / s_t(18)))*hz_5;
          const s_t reference_product_t214 = ((s_t(1) / s_t(18)))*hz_2;
          const s_t reference_product_t215 = -reference_product_t213 + reference_product_t214;
          const s_t reference_product_t216 = reference_product_t212 + reference_product_t215;
          const s_t reference_product_t217 = ((s_t(1) / s_t(9)))*hz_0 - (s_t(1) / s_t(36))*hz_7;
          const s_t reference_product_t218 = -reference_product_t208 + reference_product_t209 + reference_product_t216 + reference_product_t217;
          const s_t reference_product_t219 = ((s_t(1) / s_t(12)))*hz_1;
          const s_t reference_product_t220 = ((s_t(1) / s_t(24)))*hz_6;
          const s_t reference_product_t221 = ((s_t(1) / s_t(12)))*hz_0 - (s_t(1) / s_t(24))*hz_7;
          const s_t reference_product_t222 = -reference_product_t219 + reference_product_t220 + reference_product_t221;
          const s_t reference_product_t223 = ((s_t(1) / s_t(12)))*hz_3;
          const s_t reference_product_t224 = ((s_t(1) / s_t(24)))*hz_4;
          const s_t reference_product_t225 = -reference_product_t223 + reference_product_t224;
          const s_t reference_product_t226 = ((s_t(1) / s_t(24)))*hz_5;
          const s_t reference_product_t227 = ((s_t(1) / s_t(12)))*hz_2;
          const s_t reference_product_t228 = -reference_product_t226 + reference_product_t227;
          const s_t reference_product_t229 = reference_product_t225 + reference_product_t228;
          const s_t reference_product_t230 = reference_product_t222 + reference_product_t229;
          const s_t reference_product_t231 = ((s_t(1) / s_t(24)))*hz_3;
          const s_t reference_product_t232 = ((s_t(1) / s_t(12)))*hz_4;
          const s_t reference_product_t233 = -reference_product_t231 + reference_product_t232;
          const s_t reference_product_t234 = ((s_t(1) / s_t(12)))*hz_5;
          const s_t reference_product_t235 = ((s_t(1) / s_t(24)))*hz_2;
          const s_t reference_product_t236 = -reference_product_t234 + reference_product_t235;
          const s_t reference_product_t237 = reference_product_t233 + reference_product_t236;
          const s_t reference_product_t238 = reference_product_t222 + reference_product_t237;
          const s_t reference_product_t239 = ((s_t(1) / s_t(18)))*hz_1;
          const s_t reference_product_t240 = ((s_t(1) / s_t(18)))*hz_6;
          const s_t reference_product_t241 = ((s_t(1) / s_t(18)))*hz_0 - (s_t(1) / s_t(18))*hz_7;
          const s_t reference_product_t242 = -reference_product_t239 + reference_product_t240 + reference_product_t241;
          const s_t reference_product_t243 = ((s_t(1) / s_t(9)))*hz_3;
          const s_t reference_product_t244 = ((s_t(1) / s_t(36)))*hz_4;
          const s_t reference_product_t245 = -reference_product_t243 + reference_product_t244;
          const s_t reference_product_t246 = ((s_t(1) / s_t(36)))*hz_5;
          const s_t reference_product_t247 = ((s_t(1) / s_t(9)))*hz_2;
          const s_t reference_product_t248 = -reference_product_t246 + reference_product_t247;
          const s_t reference_product_t249 = reference_product_t242 + reference_product_t245 + reference_product_t248;
          const s_t reference_product_t250 = -reference_product_t230;
          const s_t reference_product_t251 = ((s_t(1) / s_t(24)))*hz_1;
          const s_t reference_product_t252 = ((s_t(1) / s_t(12)))*hz_6;
          const s_t reference_product_t253 = ((s_t(1) / s_t(24)))*hz_0 - (s_t(1) / s_t(12))*hz_7;
          const s_t reference_product_t254 = -reference_product_t251 + reference_product_t252 + reference_product_t253;
          const s_t reference_product_t255 = reference_product_t229 + reference_product_t254;
          const s_t reference_product_t256 = ((s_t(1) / s_t(36)))*hz_3;
          const s_t reference_product_t257 = ((s_t(1) / s_t(9)))*hz_4;
          const s_t reference_product_t258 = -reference_product_t256 + reference_product_t257;
          const s_t reference_product_t259 = ((s_t(1) / s_t(9)))*hz_5;
          const s_t reference_product_t260 = ((s_t(1) / s_t(36)))*hz_2;
          const s_t reference_product_t261 = -reference_product_t259 + reference_product_t260;
          const s_t reference_product_t262 = reference_product_t242 + reference_product_t258 + reference_product_t261;
          const s_t reference_product_t263 = reference_product_t237 + reference_product_t254;
          const s_t reference_product_t264 = -reference_product_t238;
          const s_t reference_product_t265 = ((s_t(1) / s_t(36)))*hz_1;
          const s_t reference_product_t266 = ((s_t(1) / s_t(9)))*hz_6;
          const s_t reference_product_t267 = ((s_t(1) / s_t(36)))*hz_0 - (s_t(1) / s_t(9))*hz_7;
          const s_t reference_product_t268 = reference_product_t216 - reference_product_t265 + reference_product_t266 + reference_product_t267;
          const s_t reference_product_t269 = -reference_product_t263;
          const s_t reference_product_t270 = -reference_product_t255;
          const s_t reference_product_t271 = reference_product_t221 + reference_product_t226 - reference_product_t227;
          const s_t reference_product_t272 = reference_product_t219 - reference_product_t220;
          const s_t reference_product_t273 = reference_product_t225 + reference_product_t272;
          const s_t reference_product_t274 = reference_product_t271 + reference_product_t273;
          const s_t reference_product_t275 = reference_product_t239 - reference_product_t240;
          const s_t reference_product_t276 = reference_product_t212 + reference_product_t275;
          const s_t reference_product_t277 = reference_product_t217 + reference_product_t246 - reference_product_t247 + reference_product_t276;
          const s_t reference_product_t278 = reference_product_t251 - reference_product_t252;
          const s_t reference_product_t279 = reference_product_t233 + reference_product_t278;
          const s_t reference_product_t280 = reference_product_t271 + reference_product_t279;
          const s_t reference_product_t281 = -reference_product_t274;
          const s_t reference_product_t282 = reference_product_t213 - reference_product_t214 + reference_product_t241;
          const s_t reference_product_t283 = reference_product_t208 - reference_product_t209;
          const s_t reference_product_t284 = reference_product_t245 + reference_product_t282 + reference_product_t283;
          const s_t reference_product_t285 = reference_product_t234 - reference_product_t235 + reference_product_t253;
          const s_t reference_product_t286 = reference_product_t273 + reference_product_t285;
          const s_t reference_product_t287 = reference_product_t279 + reference_product_t285;
          const s_t reference_product_t288 = reference_product_t265 - reference_product_t266;
          const s_t reference_product_t289 = reference_product_t258 + reference_product_t282 + reference_product_t288;
          const s_t reference_product_t290 = -reference_product_t280;
          const s_t reference_product_t291 = -reference_product_t287;
          const s_t reference_product_t292 = reference_product_t259 - reference_product_t260 + reference_product_t267 + reference_product_t276;
          const s_t reference_product_t293 = -reference_product_t286;
          const s_t reference_product_t294 = reference_product_t236 + reference_product_t272;
          const s_t reference_product_t295 = reference_product_t221 + reference_product_t231 - reference_product_t232;
          const s_t reference_product_t296 = reference_product_t294 + reference_product_t295;
          const s_t reference_product_t297 = reference_product_t228 + reference_product_t278;
          const s_t reference_product_t298 = reference_product_t295 + reference_product_t297;
          const s_t reference_product_t299 = reference_product_t215 + reference_product_t275;
          const s_t reference_product_t300 = reference_product_t217 + reference_product_t256 - reference_product_t257 + reference_product_t299;
          const s_t reference_product_t301 = -reference_product_t296;
          const s_t reference_product_t302 = reference_product_t223 - reference_product_t224 + reference_product_t253;
          const s_t reference_product_t303 = reference_product_t294 + reference_product_t302;
          const s_t reference_product_t304 = reference_product_t210 - reference_product_t211 + reference_product_t241;
          const s_t reference_product_t305 = reference_product_t261 + reference_product_t283 + reference_product_t304;
          const s_t reference_product_t306 = reference_product_t297 + reference_product_t302;
          const s_t reference_product_t307 = -reference_product_t298;
          const s_t reference_product_t308 = reference_product_t248 + reference_product_t288 + reference_product_t304;
          const s_t reference_product_t309 = -reference_product_t306;
          const s_t reference_product_t310 = -reference_product_t303;
          const s_t reference_product_t311 = reference_product_t243 - reference_product_t244 + reference_product_t267 + reference_product_t299;
          const s_t output_t0 = reference_product_t10*tangent0;
          const s_t output_t1 = reference_product_t114*tangent3;
          const s_t output_t2 = reference_product_t173*tangent12;
          const s_t output_t3 = reference_product_t196*tangent20;
          const s_t output_t4 = reference_product_t218*tangent6;
          const s_t output_t5 = reference_product_t277*tangent15;
          const s_t output_t6 = reference_product_t300*tangent23;
          const s_t output_t7 = reference_product_t69*tangent9;
          const s_t output_t8 = reference_product_t92*tangent17;
          const s_t output_t9 = reference_product_t126*tangent11 + reference_product_t134*tangent18 + reference_product_t22*tangent1 + reference_product_t230*tangent14 + reference_product_t238*tangent21 + reference_product_t30*tangent2;
          const s_t output_t10 = reference_product_t170*tangent4 + reference_product_t176*tangent19 + reference_product_t274*tangent7 + reference_product_t280*tangent22 + reference_product_t66*tangent1 + reference_product_t72*tangent10;
          const s_t output_t11 = reference_product_t192*tangent5 + reference_product_t194*tangent13 + reference_product_t296*tangent8 + reference_product_t298*tangent16 + reference_product_t88*tangent2 + reference_product_t90*tangent10;
          const s_t output_t12 = reference_product_t180*tangent12;
          const s_t output_t13 = reference_product_t201*tangent20;
          const s_t output_t14 = reference_product_t284*tangent15;
          const s_t output_t15 = reference_product_t305*tangent23;
          const s_t output_t16 = reference_product_t76*tangent9;
          const s_t output_t17 = reference_product_t97*tangent17;
          const s_t output_t18 = reference_product_t177*tangent4 + reference_product_t182*tangent19 + reference_product_t281*tangent7 + reference_product_t286*tangent22 + reference_product_t73*tangent1 + reference_product_t78*tangent10;
          const s_t output_t19 = reference_product_t197*tangent5 + reference_product_t199*tangent13 + reference_product_t301*tangent8 + reference_product_t303*tangent16 + reference_product_t93*tangent2 + reference_product_t95*tangent10;
          const s_t output_t20 = reference_product_t100*tangent17;
          const s_t output_t21 = reference_product_t145*tangent3;
          const s_t output_t22 = reference_product_t204*tangent20;
          const s_t output_t23 = reference_product_t249*tangent6;
          const s_t output_t24 = reference_product_t308*tangent23;
          const s_t output_t25 = reference_product_t41*tangent0;
          const s_t output_t26 = reference_product_t146*tangent11 + reference_product_t151*tangent18 + reference_product_t250*tangent14 + reference_product_t255*tangent21 + reference_product_t42*tangent1 + reference_product_t47*tangent2;
          const s_t output_t27 = reference_product_t202*tangent5 + reference_product_t203*tangent13 + reference_product_t306*tangent8 + reference_product_t307*tangent16 + reference_product_t98*tangent2 + reference_product_t99*tangent10;
          const s_t output_t28 = reference_product_t103*tangent17;
          const s_t output_t29 = reference_product_t207*tangent20;
          const s_t output_t30 = reference_product_t311*tangent23;
          const s_t output_t31 = reference_product_t101*tangent2 + reference_product_t102*tangent10 + reference_product_t205*tangent5 + reference_product_t206*tangent13 + reference_product_t309*tangent8 + reference_product_t310*tangent16;
          const s_t output_t32 = reference_product_t158*tangent3;
          const s_t output_t33 = reference_product_t185*tangent12;
          const s_t output_t34 = reference_product_t262*tangent6;
          const s_t output_t35 = reference_product_t289*tangent15;
          const s_t output_t36 = reference_product_t54*tangent0;
          const s_t output_t37 = reference_product_t81*tangent9;
          const s_t output_t38 = reference_product_t159*tangent11 + reference_product_t160*tangent18 + reference_product_t263*tangent14 + reference_product_t264*tangent21 + reference_product_t55*tangent1 + reference_product_t56*tangent2;
          const s_t output_t39 = reference_product_t183*tangent4 + reference_product_t186*tangent19 + reference_product_t287*tangent7 + reference_product_t290*tangent22 + reference_product_t79*tangent1 + reference_product_t82*tangent10;
          const s_t output_t40 = reference_product_t188*tangent12;
          const s_t output_t41 = reference_product_t292*tangent15;
          const s_t output_t42 = reference_product_t84*tangent9;
          const s_t output_t43 = reference_product_t187*tangent4 + reference_product_t189*tangent19 + reference_product_t291*tangent7 + reference_product_t293*tangent22 + reference_product_t83*tangent1 + reference_product_t85*tangent10;
          const s_t output_t44 = reference_product_t164*tangent3;
          const s_t output_t45 = reference_product_t268*tangent6;
          const s_t output_t46 = reference_product_t60*tangent0;
          const s_t output_t47 = reference_product_t165*tangent11 + reference_product_t166*tangent18 + reference_product_t269*tangent14 + reference_product_t270*tangent21 + reference_product_t61*tangent1 + reference_product_t62*tangent2;
          const s_t output_t48 = reference_product_t10*tangent3;
          const s_t output_t49 = reference_product_t114*tangent24;
          const s_t output_t50 = reference_product_t173*tangent30;
          const s_t output_t51 = reference_product_t196*tangent35;
          const s_t output_t52 = reference_product_t218*tangent27;
          const s_t output_t53 = reference_product_t277*tangent33;
          const s_t output_t54 = reference_product_t300*tangent38;
          const s_t output_t55 = reference_product_t69*tangent12;
          const s_t output_t56 = reference_product_t92*tangent20;
          const s_t output_t57 = reference_product_t126*tangent25 + reference_product_t134*tangent26 + reference_product_t22*tangent4 + reference_product_t230*tangent32 + reference_product_t238*tangent36 + reference_product_t30*tangent5;
          const s_t output_t58 = reference_product_t170*tangent25 + reference_product_t176*tangent31 + reference_product_t274*tangent28 + reference_product_t280*tangent37 + reference_product_t66*tangent11 + reference_product_t72*tangent13;
          const s_t output_t59 = reference_product_t192*tangent26 + reference_product_t194*tangent31 + reference_product_t296*tangent29 + reference_product_t298*tangent34 + reference_product_t88*tangent18 + reference_product_t90*tangent19;
          const s_t output_t60 = reference_product_t180*tangent30;
          const s_t output_t61 = reference_product_t201*tangent35;
          const s_t output_t62 = reference_product_t284*tangent33;
          const s_t output_t63 = reference_product_t305*tangent38;
          const s_t output_t64 = reference_product_t76*tangent12;
          const s_t output_t65 = reference_product_t97*tangent20;
          const s_t output_t66 = reference_product_t177*tangent25 + reference_product_t182*tangent31 + reference_product_t281*tangent28 + reference_product_t286*tangent37 + reference_product_t73*tangent11 + reference_product_t78*tangent13;
          const s_t output_t67 = reference_product_t197*tangent26 + reference_product_t199*tangent31 + reference_product_t301*tangent29 + reference_product_t303*tangent34 + reference_product_t93*tangent18 + reference_product_t95*tangent19;
          const s_t output_t68 = reference_product_t100*tangent20;
          const s_t output_t69 = reference_product_t145*tangent24;
          const s_t output_t70 = reference_product_t204*tangent35;
          const s_t output_t71 = reference_product_t249*tangent27;
          const s_t output_t72 = reference_product_t308*tangent38;
          const s_t output_t73 = reference_product_t41*tangent3;
          const s_t output_t74 = reference_product_t146*tangent25 + reference_product_t151*tangent26 + reference_product_t250*tangent32 + reference_product_t255*tangent36 + reference_product_t42*tangent4 + reference_product_t47*tangent5;
          const s_t output_t75 = reference_product_t202*tangent26 + reference_product_t203*tangent31 + reference_product_t306*tangent29 + reference_product_t307*tangent34 + reference_product_t98*tangent18 + reference_product_t99*tangent19;
          const s_t output_t76 = reference_product_t103*tangent20;
          const s_t output_t77 = reference_product_t207*tangent35;
          const s_t output_t78 = reference_product_t311*tangent38;
          const s_t output_t79 = reference_product_t101*tangent18 + reference_product_t102*tangent19 + reference_product_t205*tangent26 + reference_product_t206*tangent31 + reference_product_t309*tangent29 + reference_product_t310*tangent34;
          const s_t output_t80 = reference_product_t158*tangent24;
          const s_t output_t81 = reference_product_t185*tangent30;
          const s_t output_t82 = reference_product_t262*tangent27;
          const s_t output_t83 = reference_product_t289*tangent33;
          const s_t output_t84 = reference_product_t54*tangent3;
          const s_t output_t85 = reference_product_t81*tangent12;
          const s_t output_t86 = reference_product_t159*tangent25 + reference_product_t160*tangent26 + reference_product_t263*tangent32 + reference_product_t264*tangent36 + reference_product_t55*tangent4 + reference_product_t56*tangent5;
          const s_t output_t87 = reference_product_t183*tangent25 + reference_product_t186*tangent31 + reference_product_t287*tangent28 + reference_product_t290*tangent37 + reference_product_t79*tangent11 + reference_product_t82*tangent13;
          const s_t output_t88 = reference_product_t188*tangent30;
          const s_t output_t89 = reference_product_t292*tangent33;
          const s_t output_t90 = reference_product_t84*tangent12;
          const s_t output_t91 = reference_product_t187*tangent25 + reference_product_t189*tangent31 + reference_product_t291*tangent28 + reference_product_t293*tangent37 + reference_product_t83*tangent11 + reference_product_t85*tangent13;
          const s_t output_t92 = reference_product_t164*tangent24;
          const s_t output_t93 = reference_product_t268*tangent27;
          const s_t output_t94 = reference_product_t60*tangent3;
          const s_t output_t95 = reference_product_t165*tangent25 + reference_product_t166*tangent26 + reference_product_t269*tangent32 + reference_product_t270*tangent36 + reference_product_t61*tangent4 + reference_product_t62*tangent5;
          const s_t output_t96 = reference_product_t10*tangent6;
          const s_t output_t97 = reference_product_t114*tangent27;
          const s_t output_t98 = reference_product_t173*tangent33;
          const s_t output_t99 = reference_product_t196*tangent38;
          const s_t output_t100 = reference_product_t218*tangent39;
          const s_t output_t101 = reference_product_t277*tangent42;
          const s_t output_t102 = reference_product_t300*tangent44;
          const s_t output_t103 = reference_product_t69*tangent15;
          const s_t output_t104 = reference_product_t92*tangent23;
          const s_t output_t105 = reference_product_t126*tangent28 + reference_product_t134*tangent29 + reference_product_t22*tangent7 + reference_product_t230*tangent40 + reference_product_t238*tangent41 + reference_product_t30*tangent8;
          const s_t output_t106 = reference_product_t170*tangent32 + reference_product_t176*tangent34 + reference_product_t274*tangent40 + reference_product_t280*tangent43 + reference_product_t66*tangent14 + reference_product_t72*tangent16;
          const s_t output_t107 = reference_product_t192*tangent36 + reference_product_t194*tangent37 + reference_product_t296*tangent41 + reference_product_t298*tangent43 + reference_product_t88*tangent21 + reference_product_t90*tangent22;
          const s_t output_t108 = reference_product_t180*tangent33;
          const s_t output_t109 = reference_product_t201*tangent38;
          const s_t output_t110 = reference_product_t284*tangent42;
          const s_t output_t111 = reference_product_t305*tangent44;
          const s_t output_t112 = reference_product_t76*tangent15;
          const s_t output_t113 = reference_product_t97*tangent23;
          const s_t output_t114 = reference_product_t177*tangent32 + reference_product_t182*tangent34 + reference_product_t281*tangent40 + reference_product_t286*tangent43 + reference_product_t73*tangent14 + reference_product_t78*tangent16;
          const s_t output_t115 = reference_product_t197*tangent36 + reference_product_t199*tangent37 + reference_product_t301*tangent41 + reference_product_t303*tangent43 + reference_product_t93*tangent21 + reference_product_t95*tangent22;
          const s_t output_t116 = reference_product_t100*tangent23;
          const s_t output_t117 = reference_product_t145*tangent27;
          const s_t output_t118 = reference_product_t204*tangent38;
          const s_t output_t119 = reference_product_t249*tangent39;
          const s_t output_t120 = reference_product_t308*tangent44;
          const s_t output_t121 = reference_product_t41*tangent6;
          const s_t output_t122 = reference_product_t146*tangent28 + reference_product_t151*tangent29 + reference_product_t250*tangent40 + reference_product_t255*tangent41 + reference_product_t42*tangent7 + reference_product_t47*tangent8;
          const s_t output_t123 = reference_product_t202*tangent36 + reference_product_t203*tangent37 + reference_product_t306*tangent41 + reference_product_t307*tangent43 + reference_product_t98*tangent21 + reference_product_t99*tangent22;
          const s_t output_t124 = reference_product_t103*tangent23;
          const s_t output_t125 = reference_product_t207*tangent38;
          const s_t output_t126 = reference_product_t311*tangent44;
          const s_t output_t127 = reference_product_t101*tangent21 + reference_product_t102*tangent22 + reference_product_t205*tangent36 + reference_product_t206*tangent37 + reference_product_t309*tangent41 + reference_product_t310*tangent43;
          const s_t output_t128 = reference_product_t158*tangent27;
          const s_t output_t129 = reference_product_t185*tangent33;
          const s_t output_t130 = reference_product_t262*tangent39;
          const s_t output_t131 = reference_product_t289*tangent42;
          const s_t output_t132 = reference_product_t54*tangent6;
          const s_t output_t133 = reference_product_t81*tangent15;
          const s_t output_t134 = reference_product_t159*tangent28 + reference_product_t160*tangent29 + reference_product_t263*tangent40 + reference_product_t264*tangent41 + reference_product_t55*tangent7 + reference_product_t56*tangent8;
          const s_t output_t135 = reference_product_t183*tangent32 + reference_product_t186*tangent34 + reference_product_t287*tangent40 + reference_product_t290*tangent43 + reference_product_t79*tangent14 + reference_product_t82*tangent16;
          const s_t output_t136 = reference_product_t188*tangent33;
          const s_t output_t137 = reference_product_t292*tangent42;
          const s_t output_t138 = reference_product_t84*tangent15;
          const s_t output_t139 = reference_product_t187*tangent32 + reference_product_t189*tangent34 + reference_product_t291*tangent40 + reference_product_t293*tangent43 + reference_product_t83*tangent14 + reference_product_t85*tangent16;
          const s_t output_t140 = reference_product_t164*tangent27;
          const s_t output_t141 = reference_product_t268*tangent39;
          const s_t output_t142 = reference_product_t60*tangent6;
          const s_t output_t143 = reference_product_t165*tangent28 + reference_product_t166*tangent29 + reference_product_t269*tangent40 + reference_product_t270*tangent41 + reference_product_t61*tangent7 + reference_product_t62*tangent8;
          const s_t element_out0_0 = output_t0 + output_t1 + output_t10 + output_t11 + output_t2 + output_t3 + output_t4 + output_t5 + output_t6 + output_t7 + output_t8 + output_t9;
          const s_t element_out0_1 = -output_t0 - output_t1 + output_t12 + output_t13 + output_t14 + output_t15 + output_t16 + output_t17 + output_t18 + output_t19 - output_t4 + output_t9;
          const s_t element_out0_2 = output_t10 - output_t2 + output_t20 + output_t21 + output_t22 + output_t23 + output_t24 + output_t25 + output_t26 + output_t27 - output_t5 - output_t7;
          const s_t element_out0_3 = -output_t12 - output_t14 - output_t16 + output_t18 - output_t21 - output_t23 - output_t25 + output_t26 + output_t28 + output_t29 + output_t30 + output_t31;
          const s_t element_out0_4 = output_t11 - output_t3 + output_t32 + output_t33 + output_t34 + output_t35 + output_t36 + output_t37 + output_t38 + output_t39 - output_t6 - output_t8;
          const s_t element_out0_5 = -output_t13 - output_t15 - output_t17 + output_t19 - output_t32 - output_t34 - output_t36 + output_t38 + output_t40 + output_t41 + output_t42 + output_t43;
          const s_t element_out0_6 = -output_t20 - output_t22 - output_t24 + output_t27 - output_t33 - output_t35 - output_t37 + output_t39 + output_t44 + output_t45 + output_t46 + output_t47;
          const s_t element_out0_7 = -output_t28 - output_t29 - output_t30 + output_t31 - output_t40 - output_t41 - output_t42 + output_t43 - output_t44 - output_t45 - output_t46 + output_t47;
          const s_t element_out1_0 = output_t48 + output_t49 + output_t50 + output_t51 + output_t52 + output_t53 + output_t54 + output_t55 + output_t56 + output_t57 + output_t58 + output_t59;
          const s_t element_out1_1 = -output_t48 - output_t49 - output_t52 + output_t57 + output_t60 + output_t61 + output_t62 + output_t63 + output_t64 + output_t65 + output_t66 + output_t67;
          const s_t element_out1_2 = -output_t50 - output_t53 - output_t55 + output_t58 + output_t68 + output_t69 + output_t70 + output_t71 + output_t72 + output_t73 + output_t74 + output_t75;
          const s_t element_out1_3 = -output_t60 - output_t62 - output_t64 + output_t66 - output_t69 - output_t71 - output_t73 + output_t74 + output_t76 + output_t77 + output_t78 + output_t79;
          const s_t element_out1_4 = -output_t51 - output_t54 - output_t56 + output_t59 + output_t80 + output_t81 + output_t82 + output_t83 + output_t84 + output_t85 + output_t86 + output_t87;
          const s_t element_out1_5 = -output_t61 - output_t63 - output_t65 + output_t67 - output_t80 - output_t82 - output_t84 + output_t86 + output_t88 + output_t89 + output_t90 + output_t91;
          const s_t element_out1_6 = -output_t68 - output_t70 - output_t72 + output_t75 - output_t81 - output_t83 - output_t85 + output_t87 + output_t92 + output_t93 + output_t94 + output_t95;
          const s_t element_out1_7 = -output_t76 - output_t77 - output_t78 + output_t79 - output_t88 - output_t89 - output_t90 + output_t91 - output_t92 - output_t93 - output_t94 + output_t95;
          const s_t element_out2_0 = output_t100 + output_t101 + output_t102 + output_t103 + output_t104 + output_t105 + output_t106 + output_t107 + output_t96 + output_t97 + output_t98 + output_t99;
          const s_t element_out2_1 = -output_t100 + output_t105 + output_t108 + output_t109 + output_t110 + output_t111 + output_t112 + output_t113 + output_t114 + output_t115 - output_t96 - output_t97;
          const s_t element_out2_2 = -output_t101 - output_t103 + output_t106 + output_t116 + output_t117 + output_t118 + output_t119 + output_t120 + output_t121 + output_t122 + output_t123 - output_t98;
          const s_t element_out2_3 = -output_t108 - output_t110 - output_t112 + output_t114 - output_t117 - output_t119 - output_t121 + output_t122 + output_t124 + output_t125 + output_t126 + output_t127;
          const s_t element_out2_4 = -output_t102 - output_t104 + output_t107 + output_t128 + output_t129 + output_t130 + output_t131 + output_t132 + output_t133 + output_t134 + output_t135 - output_t99;
          const s_t element_out2_5 = -output_t109 - output_t111 - output_t113 + output_t115 - output_t128 - output_t130 - output_t132 + output_t134 + output_t136 + output_t137 + output_t138 + output_t139;
          const s_t element_out2_6 = -output_t116 - output_t118 - output_t120 + output_t123 - output_t129 - output_t131 - output_t133 + output_t135 + output_t140 + output_t141 + output_t142 + output_t143;
          const s_t element_out2_7 = -output_t124 - output_t125 - output_t126 + output_t127 - output_t136 - output_t137 - output_t138 + output_t139 - output_t140 - output_t141 - output_t142 + output_t143;
          bout0_0[lane] = element_out0_0;
          bout0_1[lane] = element_out0_1;
          bout0_2[lane] = element_out0_2;
          bout0_3[lane] = element_out0_3;
          bout0_4[lane] = element_out0_4;
          bout0_5[lane] = element_out0_5;
          bout0_6[lane] = element_out0_6;
          bout0_7[lane] = element_out0_7;
          bout1_0[lane] = element_out1_0;
          bout1_1[lane] = element_out1_1;
          bout1_2[lane] = element_out1_2;
          bout1_3[lane] = element_out1_3;
          bout1_4[lane] = element_out1_4;
          bout1_5[lane] = element_out1_5;
          bout1_6[lane] = element_out1_6;
          bout1_7[lane] = element_out1_7;
          bout2_0[lane] = element_out2_0;
          bout2_1[lane] = element_out2_1;
          bout2_2[lane] = element_out2_2;
          bout2_3[lane] = element_out2_3;
          bout2_4[lane] = element_out2_4;
          bout2_5[lane] = element_out2_5;
          bout2_6[lane] = element_out2_6;
          bout2_7[lane] = element_out2_7;
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
static SFEM_INLINE int linear_elasticity_proteus_hex8_inexact_apply_compressed_a_msoa_impl(
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
    const s_t hx_0 = hx[ev0 * h_stride];
    const s_t hx_1 = hx[ev1 * h_stride];
    const s_t hx_2 = hx[ev2 * h_stride];
    const s_t hx_3 = hx[ev3 * h_stride];
    const s_t hx_4 = hx[ev4 * h_stride];
    const s_t hx_5 = hx[ev5 * h_stride];
    const s_t hx_6 = hx[ev6 * h_stride];
    const s_t hx_7 = hx[ev7 * h_stride];
    const s_t hy_0 = hy[ev0 * h_stride];
    const s_t hy_1 = hy[ev1 * h_stride];
    const s_t hy_2 = hy[ev2 * h_stride];
    const s_t hy_3 = hy[ev3 * h_stride];
    const s_t hy_4 = hy[ev4 * h_stride];
    const s_t hy_5 = hy[ev5 * h_stride];
    const s_t hy_6 = hy[ev6 * h_stride];
    const s_t hy_7 = hy[ev7 * h_stride];
    const s_t hz_0 = hz[ev0 * h_stride];
    const s_t hz_1 = hz[ev1 * h_stride];
    const s_t hz_2 = hz[ev2 * h_stride];
    const s_t hz_3 = hz[ev3 * h_stride];
    const s_t hz_4 = hz[ev4 * h_stride];
    const s_t hz_5 = hz[ev5 * h_stride];
    const s_t hz_6 = hz[ev6 * h_stride];
    const s_t hz_7 = hz[ev7 * h_stride];
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
    const s_t reference_product_t0 = ((s_t(1) / s_t(9)))*hx_1;
    const s_t reference_product_t1 = ((s_t(1) / s_t(36)))*hx_6;
    const s_t reference_product_t2 = ((s_t(1) / s_t(18)))*hx_3;
    const s_t reference_product_t3 = ((s_t(1) / s_t(18)))*hx_4;
    const s_t reference_product_t4 = -reference_product_t2 + reference_product_t3;
    const s_t reference_product_t5 = ((s_t(1) / s_t(18)))*hx_5;
    const s_t reference_product_t6 = ((s_t(1) / s_t(18)))*hx_2;
    const s_t reference_product_t7 = -reference_product_t5 + reference_product_t6;
    const s_t reference_product_t8 = reference_product_t4 + reference_product_t7;
    const s_t reference_product_t9 = ((s_t(1) / s_t(9)))*hx_0 - (s_t(1) / s_t(36))*hx_7;
    const s_t reference_product_t10 = -reference_product_t0 + reference_product_t1 + reference_product_t8 + reference_product_t9;
    const s_t reference_product_t11 = ((s_t(1) / s_t(12)))*hx_1;
    const s_t reference_product_t12 = ((s_t(1) / s_t(24)))*hx_6;
    const s_t reference_product_t13 = ((s_t(1) / s_t(12)))*hx_0 - (s_t(1) / s_t(24))*hx_7;
    const s_t reference_product_t14 = -reference_product_t11 + reference_product_t12 + reference_product_t13;
    const s_t reference_product_t15 = ((s_t(1) / s_t(12)))*hx_3;
    const s_t reference_product_t16 = ((s_t(1) / s_t(24)))*hx_4;
    const s_t reference_product_t17 = -reference_product_t15 + reference_product_t16;
    const s_t reference_product_t18 = ((s_t(1) / s_t(24)))*hx_5;
    const s_t reference_product_t19 = ((s_t(1) / s_t(12)))*hx_2;
    const s_t reference_product_t20 = -reference_product_t18 + reference_product_t19;
    const s_t reference_product_t21 = reference_product_t17 + reference_product_t20;
    const s_t reference_product_t22 = reference_product_t14 + reference_product_t21;
    const s_t reference_product_t23 = ((s_t(1) / s_t(24)))*hx_3;
    const s_t reference_product_t24 = ((s_t(1) / s_t(12)))*hx_4;
    const s_t reference_product_t25 = -reference_product_t23 + reference_product_t24;
    const s_t reference_product_t26 = ((s_t(1) / s_t(12)))*hx_5;
    const s_t reference_product_t27 = ((s_t(1) / s_t(24)))*hx_2;
    const s_t reference_product_t28 = -reference_product_t26 + reference_product_t27;
    const s_t reference_product_t29 = reference_product_t25 + reference_product_t28;
    const s_t reference_product_t30 = reference_product_t14 + reference_product_t29;
    const s_t reference_product_t31 = ((s_t(1) / s_t(18)))*hx_1;
    const s_t reference_product_t32 = ((s_t(1) / s_t(18)))*hx_6;
    const s_t reference_product_t33 = ((s_t(1) / s_t(18)))*hx_0 - (s_t(1) / s_t(18))*hx_7;
    const s_t reference_product_t34 = -reference_product_t31 + reference_product_t32 + reference_product_t33;
    const s_t reference_product_t35 = ((s_t(1) / s_t(9)))*hx_3;
    const s_t reference_product_t36 = ((s_t(1) / s_t(36)))*hx_4;
    const s_t reference_product_t37 = -reference_product_t35 + reference_product_t36;
    const s_t reference_product_t38 = ((s_t(1) / s_t(36)))*hx_5;
    const s_t reference_product_t39 = ((s_t(1) / s_t(9)))*hx_2;
    const s_t reference_product_t40 = -reference_product_t38 + reference_product_t39;
    const s_t reference_product_t41 = reference_product_t34 + reference_product_t37 + reference_product_t40;
    const s_t reference_product_t42 = -reference_product_t22;
    const s_t reference_product_t43 = ((s_t(1) / s_t(24)))*hx_1;
    const s_t reference_product_t44 = ((s_t(1) / s_t(12)))*hx_6;
    const s_t reference_product_t45 = ((s_t(1) / s_t(24)))*hx_0 - (s_t(1) / s_t(12))*hx_7;
    const s_t reference_product_t46 = -reference_product_t43 + reference_product_t44 + reference_product_t45;
    const s_t reference_product_t47 = reference_product_t21 + reference_product_t46;
    const s_t reference_product_t48 = ((s_t(1) / s_t(36)))*hx_3;
    const s_t reference_product_t49 = ((s_t(1) / s_t(9)))*hx_4;
    const s_t reference_product_t50 = -reference_product_t48 + reference_product_t49;
    const s_t reference_product_t51 = ((s_t(1) / s_t(9)))*hx_5;
    const s_t reference_product_t52 = ((s_t(1) / s_t(36)))*hx_2;
    const s_t reference_product_t53 = -reference_product_t51 + reference_product_t52;
    const s_t reference_product_t54 = reference_product_t34 + reference_product_t50 + reference_product_t53;
    const s_t reference_product_t55 = reference_product_t29 + reference_product_t46;
    const s_t reference_product_t56 = -reference_product_t30;
    const s_t reference_product_t57 = ((s_t(1) / s_t(36)))*hx_1;
    const s_t reference_product_t58 = ((s_t(1) / s_t(9)))*hx_6;
    const s_t reference_product_t59 = ((s_t(1) / s_t(36)))*hx_0 - (s_t(1) / s_t(9))*hx_7;
    const s_t reference_product_t60 = -reference_product_t57 + reference_product_t58 + reference_product_t59 + reference_product_t8;
    const s_t reference_product_t61 = -reference_product_t55;
    const s_t reference_product_t62 = -reference_product_t47;
    const s_t reference_product_t63 = reference_product_t13 + reference_product_t18 - reference_product_t19;
    const s_t reference_product_t64 = reference_product_t11 - reference_product_t12;
    const s_t reference_product_t65 = reference_product_t17 + reference_product_t64;
    const s_t reference_product_t66 = reference_product_t63 + reference_product_t65;
    const s_t reference_product_t67 = reference_product_t31 - reference_product_t32;
    const s_t reference_product_t68 = reference_product_t4 + reference_product_t67;
    const s_t reference_product_t69 = reference_product_t38 - reference_product_t39 + reference_product_t68 + reference_product_t9;
    const s_t reference_product_t70 = reference_product_t43 - reference_product_t44;
    const s_t reference_product_t71 = reference_product_t25 + reference_product_t70;
    const s_t reference_product_t72 = reference_product_t63 + reference_product_t71;
    const s_t reference_product_t73 = -reference_product_t66;
    const s_t reference_product_t74 = reference_product_t33 + reference_product_t5 - reference_product_t6;
    const s_t reference_product_t75 = reference_product_t0 - reference_product_t1;
    const s_t reference_product_t76 = reference_product_t37 + reference_product_t74 + reference_product_t75;
    const s_t reference_product_t77 = reference_product_t26 - reference_product_t27 + reference_product_t45;
    const s_t reference_product_t78 = reference_product_t65 + reference_product_t77;
    const s_t reference_product_t79 = reference_product_t71 + reference_product_t77;
    const s_t reference_product_t80 = reference_product_t57 - reference_product_t58;
    const s_t reference_product_t81 = reference_product_t50 + reference_product_t74 + reference_product_t80;
    const s_t reference_product_t82 = -reference_product_t72;
    const s_t reference_product_t83 = -reference_product_t79;
    const s_t reference_product_t84 = reference_product_t51 - reference_product_t52 + reference_product_t59 + reference_product_t68;
    const s_t reference_product_t85 = -reference_product_t78;
    const s_t reference_product_t86 = reference_product_t28 + reference_product_t64;
    const s_t reference_product_t87 = reference_product_t13 + reference_product_t23 - reference_product_t24;
    const s_t reference_product_t88 = reference_product_t86 + reference_product_t87;
    const s_t reference_product_t89 = reference_product_t20 + reference_product_t70;
    const s_t reference_product_t90 = reference_product_t87 + reference_product_t89;
    const s_t reference_product_t91 = reference_product_t67 + reference_product_t7;
    const s_t reference_product_t92 = reference_product_t48 - reference_product_t49 + reference_product_t9 + reference_product_t91;
    const s_t reference_product_t93 = -reference_product_t88;
    const s_t reference_product_t94 = reference_product_t15 - reference_product_t16 + reference_product_t45;
    const s_t reference_product_t95 = reference_product_t86 + reference_product_t94;
    const s_t reference_product_t96 = reference_product_t2 - reference_product_t3 + reference_product_t33;
    const s_t reference_product_t97 = reference_product_t53 + reference_product_t75 + reference_product_t96;
    const s_t reference_product_t98 = reference_product_t89 + reference_product_t94;
    const s_t reference_product_t99 = -reference_product_t90;
    const s_t reference_product_t100 = reference_product_t40 + reference_product_t80 + reference_product_t96;
    const s_t reference_product_t101 = -reference_product_t98;
    const s_t reference_product_t102 = -reference_product_t95;
    const s_t reference_product_t103 = reference_product_t35 - reference_product_t36 + reference_product_t59 + reference_product_t91;
    const s_t reference_product_t104 = ((s_t(1) / s_t(9)))*hy_1;
    const s_t reference_product_t105 = ((s_t(1) / s_t(36)))*hy_6;
    const s_t reference_product_t106 = ((s_t(1) / s_t(18)))*hy_3;
    const s_t reference_product_t107 = ((s_t(1) / s_t(18)))*hy_4;
    const s_t reference_product_t108 = -reference_product_t106 + reference_product_t107;
    const s_t reference_product_t109 = ((s_t(1) / s_t(18)))*hy_5;
    const s_t reference_product_t110 = ((s_t(1) / s_t(18)))*hy_2;
    const s_t reference_product_t111 = -reference_product_t109 + reference_product_t110;
    const s_t reference_product_t112 = reference_product_t108 + reference_product_t111;
    const s_t reference_product_t113 = ((s_t(1) / s_t(9)))*hy_0 - (s_t(1) / s_t(36))*hy_7;
    const s_t reference_product_t114 = -reference_product_t104 + reference_product_t105 + reference_product_t112 + reference_product_t113;
    const s_t reference_product_t115 = ((s_t(1) / s_t(12)))*hy_1;
    const s_t reference_product_t116 = ((s_t(1) / s_t(24)))*hy_6;
    const s_t reference_product_t117 = ((s_t(1) / s_t(12)))*hy_0 - (s_t(1) / s_t(24))*hy_7;
    const s_t reference_product_t118 = -reference_product_t115 + reference_product_t116 + reference_product_t117;
    const s_t reference_product_t119 = ((s_t(1) / s_t(12)))*hy_3;
    const s_t reference_product_t120 = ((s_t(1) / s_t(24)))*hy_4;
    const s_t reference_product_t121 = -reference_product_t119 + reference_product_t120;
    const s_t reference_product_t122 = ((s_t(1) / s_t(24)))*hy_5;
    const s_t reference_product_t123 = ((s_t(1) / s_t(12)))*hy_2;
    const s_t reference_product_t124 = -reference_product_t122 + reference_product_t123;
    const s_t reference_product_t125 = reference_product_t121 + reference_product_t124;
    const s_t reference_product_t126 = reference_product_t118 + reference_product_t125;
    const s_t reference_product_t127 = ((s_t(1) / s_t(24)))*hy_3;
    const s_t reference_product_t128 = ((s_t(1) / s_t(12)))*hy_4;
    const s_t reference_product_t129 = -reference_product_t127 + reference_product_t128;
    const s_t reference_product_t130 = ((s_t(1) / s_t(12)))*hy_5;
    const s_t reference_product_t131 = ((s_t(1) / s_t(24)))*hy_2;
    const s_t reference_product_t132 = -reference_product_t130 + reference_product_t131;
    const s_t reference_product_t133 = reference_product_t129 + reference_product_t132;
    const s_t reference_product_t134 = reference_product_t118 + reference_product_t133;
    const s_t reference_product_t135 = ((s_t(1) / s_t(18)))*hy_1;
    const s_t reference_product_t136 = ((s_t(1) / s_t(18)))*hy_6;
    const s_t reference_product_t137 = ((s_t(1) / s_t(18)))*hy_0 - (s_t(1) / s_t(18))*hy_7;
    const s_t reference_product_t138 = -reference_product_t135 + reference_product_t136 + reference_product_t137;
    const s_t reference_product_t139 = ((s_t(1) / s_t(9)))*hy_3;
    const s_t reference_product_t140 = ((s_t(1) / s_t(36)))*hy_4;
    const s_t reference_product_t141 = -reference_product_t139 + reference_product_t140;
    const s_t reference_product_t142 = ((s_t(1) / s_t(36)))*hy_5;
    const s_t reference_product_t143 = ((s_t(1) / s_t(9)))*hy_2;
    const s_t reference_product_t144 = -reference_product_t142 + reference_product_t143;
    const s_t reference_product_t145 = reference_product_t138 + reference_product_t141 + reference_product_t144;
    const s_t reference_product_t146 = -reference_product_t126;
    const s_t reference_product_t147 = ((s_t(1) / s_t(24)))*hy_1;
    const s_t reference_product_t148 = ((s_t(1) / s_t(12)))*hy_6;
    const s_t reference_product_t149 = ((s_t(1) / s_t(24)))*hy_0 - (s_t(1) / s_t(12))*hy_7;
    const s_t reference_product_t150 = -reference_product_t147 + reference_product_t148 + reference_product_t149;
    const s_t reference_product_t151 = reference_product_t125 + reference_product_t150;
    const s_t reference_product_t152 = ((s_t(1) / s_t(36)))*hy_3;
    const s_t reference_product_t153 = ((s_t(1) / s_t(9)))*hy_4;
    const s_t reference_product_t154 = -reference_product_t152 + reference_product_t153;
    const s_t reference_product_t155 = ((s_t(1) / s_t(9)))*hy_5;
    const s_t reference_product_t156 = ((s_t(1) / s_t(36)))*hy_2;
    const s_t reference_product_t157 = -reference_product_t155 + reference_product_t156;
    const s_t reference_product_t158 = reference_product_t138 + reference_product_t154 + reference_product_t157;
    const s_t reference_product_t159 = reference_product_t133 + reference_product_t150;
    const s_t reference_product_t160 = -reference_product_t134;
    const s_t reference_product_t161 = ((s_t(1) / s_t(36)))*hy_1;
    const s_t reference_product_t162 = ((s_t(1) / s_t(9)))*hy_6;
    const s_t reference_product_t163 = ((s_t(1) / s_t(36)))*hy_0 - (s_t(1) / s_t(9))*hy_7;
    const s_t reference_product_t164 = reference_product_t112 - reference_product_t161 + reference_product_t162 + reference_product_t163;
    const s_t reference_product_t165 = -reference_product_t159;
    const s_t reference_product_t166 = -reference_product_t151;
    const s_t reference_product_t167 = reference_product_t117 + reference_product_t122 - reference_product_t123;
    const s_t reference_product_t168 = reference_product_t115 - reference_product_t116;
    const s_t reference_product_t169 = reference_product_t121 + reference_product_t168;
    const s_t reference_product_t170 = reference_product_t167 + reference_product_t169;
    const s_t reference_product_t171 = reference_product_t135 - reference_product_t136;
    const s_t reference_product_t172 = reference_product_t108 + reference_product_t171;
    const s_t reference_product_t173 = reference_product_t113 + reference_product_t142 - reference_product_t143 + reference_product_t172;
    const s_t reference_product_t174 = reference_product_t147 - reference_product_t148;
    const s_t reference_product_t175 = reference_product_t129 + reference_product_t174;
    const s_t reference_product_t176 = reference_product_t167 + reference_product_t175;
    const s_t reference_product_t177 = -reference_product_t170;
    const s_t reference_product_t178 = reference_product_t109 - reference_product_t110 + reference_product_t137;
    const s_t reference_product_t179 = reference_product_t104 - reference_product_t105;
    const s_t reference_product_t180 = reference_product_t141 + reference_product_t178 + reference_product_t179;
    const s_t reference_product_t181 = reference_product_t130 - reference_product_t131 + reference_product_t149;
    const s_t reference_product_t182 = reference_product_t169 + reference_product_t181;
    const s_t reference_product_t183 = reference_product_t175 + reference_product_t181;
    const s_t reference_product_t184 = reference_product_t161 - reference_product_t162;
    const s_t reference_product_t185 = reference_product_t154 + reference_product_t178 + reference_product_t184;
    const s_t reference_product_t186 = -reference_product_t176;
    const s_t reference_product_t187 = -reference_product_t183;
    const s_t reference_product_t188 = reference_product_t155 - reference_product_t156 + reference_product_t163 + reference_product_t172;
    const s_t reference_product_t189 = -reference_product_t182;
    const s_t reference_product_t190 = reference_product_t132 + reference_product_t168;
    const s_t reference_product_t191 = reference_product_t117 + reference_product_t127 - reference_product_t128;
    const s_t reference_product_t192 = reference_product_t190 + reference_product_t191;
    const s_t reference_product_t193 = reference_product_t124 + reference_product_t174;
    const s_t reference_product_t194 = reference_product_t191 + reference_product_t193;
    const s_t reference_product_t195 = reference_product_t111 + reference_product_t171;
    const s_t reference_product_t196 = reference_product_t113 + reference_product_t152 - reference_product_t153 + reference_product_t195;
    const s_t reference_product_t197 = -reference_product_t192;
    const s_t reference_product_t198 = reference_product_t119 - reference_product_t120 + reference_product_t149;
    const s_t reference_product_t199 = reference_product_t190 + reference_product_t198;
    const s_t reference_product_t200 = reference_product_t106 - reference_product_t107 + reference_product_t137;
    const s_t reference_product_t201 = reference_product_t157 + reference_product_t179 + reference_product_t200;
    const s_t reference_product_t202 = reference_product_t193 + reference_product_t198;
    const s_t reference_product_t203 = -reference_product_t194;
    const s_t reference_product_t204 = reference_product_t144 + reference_product_t184 + reference_product_t200;
    const s_t reference_product_t205 = -reference_product_t202;
    const s_t reference_product_t206 = -reference_product_t199;
    const s_t reference_product_t207 = reference_product_t139 - reference_product_t140 + reference_product_t163 + reference_product_t195;
    const s_t reference_product_t208 = ((s_t(1) / s_t(9)))*hz_1;
    const s_t reference_product_t209 = ((s_t(1) / s_t(36)))*hz_6;
    const s_t reference_product_t210 = ((s_t(1) / s_t(18)))*hz_3;
    const s_t reference_product_t211 = ((s_t(1) / s_t(18)))*hz_4;
    const s_t reference_product_t212 = -reference_product_t210 + reference_product_t211;
    const s_t reference_product_t213 = ((s_t(1) / s_t(18)))*hz_5;
    const s_t reference_product_t214 = ((s_t(1) / s_t(18)))*hz_2;
    const s_t reference_product_t215 = -reference_product_t213 + reference_product_t214;
    const s_t reference_product_t216 = reference_product_t212 + reference_product_t215;
    const s_t reference_product_t217 = ((s_t(1) / s_t(9)))*hz_0 - (s_t(1) / s_t(36))*hz_7;
    const s_t reference_product_t218 = -reference_product_t208 + reference_product_t209 + reference_product_t216 + reference_product_t217;
    const s_t reference_product_t219 = ((s_t(1) / s_t(12)))*hz_1;
    const s_t reference_product_t220 = ((s_t(1) / s_t(24)))*hz_6;
    const s_t reference_product_t221 = ((s_t(1) / s_t(12)))*hz_0 - (s_t(1) / s_t(24))*hz_7;
    const s_t reference_product_t222 = -reference_product_t219 + reference_product_t220 + reference_product_t221;
    const s_t reference_product_t223 = ((s_t(1) / s_t(12)))*hz_3;
    const s_t reference_product_t224 = ((s_t(1) / s_t(24)))*hz_4;
    const s_t reference_product_t225 = -reference_product_t223 + reference_product_t224;
    const s_t reference_product_t226 = ((s_t(1) / s_t(24)))*hz_5;
    const s_t reference_product_t227 = ((s_t(1) / s_t(12)))*hz_2;
    const s_t reference_product_t228 = -reference_product_t226 + reference_product_t227;
    const s_t reference_product_t229 = reference_product_t225 + reference_product_t228;
    const s_t reference_product_t230 = reference_product_t222 + reference_product_t229;
    const s_t reference_product_t231 = ((s_t(1) / s_t(24)))*hz_3;
    const s_t reference_product_t232 = ((s_t(1) / s_t(12)))*hz_4;
    const s_t reference_product_t233 = -reference_product_t231 + reference_product_t232;
    const s_t reference_product_t234 = ((s_t(1) / s_t(12)))*hz_5;
    const s_t reference_product_t235 = ((s_t(1) / s_t(24)))*hz_2;
    const s_t reference_product_t236 = -reference_product_t234 + reference_product_t235;
    const s_t reference_product_t237 = reference_product_t233 + reference_product_t236;
    const s_t reference_product_t238 = reference_product_t222 + reference_product_t237;
    const s_t reference_product_t239 = ((s_t(1) / s_t(18)))*hz_1;
    const s_t reference_product_t240 = ((s_t(1) / s_t(18)))*hz_6;
    const s_t reference_product_t241 = ((s_t(1) / s_t(18)))*hz_0 - (s_t(1) / s_t(18))*hz_7;
    const s_t reference_product_t242 = -reference_product_t239 + reference_product_t240 + reference_product_t241;
    const s_t reference_product_t243 = ((s_t(1) / s_t(9)))*hz_3;
    const s_t reference_product_t244 = ((s_t(1) / s_t(36)))*hz_4;
    const s_t reference_product_t245 = -reference_product_t243 + reference_product_t244;
    const s_t reference_product_t246 = ((s_t(1) / s_t(36)))*hz_5;
    const s_t reference_product_t247 = ((s_t(1) / s_t(9)))*hz_2;
    const s_t reference_product_t248 = -reference_product_t246 + reference_product_t247;
    const s_t reference_product_t249 = reference_product_t242 + reference_product_t245 + reference_product_t248;
    const s_t reference_product_t250 = -reference_product_t230;
    const s_t reference_product_t251 = ((s_t(1) / s_t(24)))*hz_1;
    const s_t reference_product_t252 = ((s_t(1) / s_t(12)))*hz_6;
    const s_t reference_product_t253 = ((s_t(1) / s_t(24)))*hz_0 - (s_t(1) / s_t(12))*hz_7;
    const s_t reference_product_t254 = -reference_product_t251 + reference_product_t252 + reference_product_t253;
    const s_t reference_product_t255 = reference_product_t229 + reference_product_t254;
    const s_t reference_product_t256 = ((s_t(1) / s_t(36)))*hz_3;
    const s_t reference_product_t257 = ((s_t(1) / s_t(9)))*hz_4;
    const s_t reference_product_t258 = -reference_product_t256 + reference_product_t257;
    const s_t reference_product_t259 = ((s_t(1) / s_t(9)))*hz_5;
    const s_t reference_product_t260 = ((s_t(1) / s_t(36)))*hz_2;
    const s_t reference_product_t261 = -reference_product_t259 + reference_product_t260;
    const s_t reference_product_t262 = reference_product_t242 + reference_product_t258 + reference_product_t261;
    const s_t reference_product_t263 = reference_product_t237 + reference_product_t254;
    const s_t reference_product_t264 = -reference_product_t238;
    const s_t reference_product_t265 = ((s_t(1) / s_t(36)))*hz_1;
    const s_t reference_product_t266 = ((s_t(1) / s_t(9)))*hz_6;
    const s_t reference_product_t267 = ((s_t(1) / s_t(36)))*hz_0 - (s_t(1) / s_t(9))*hz_7;
    const s_t reference_product_t268 = reference_product_t216 - reference_product_t265 + reference_product_t266 + reference_product_t267;
    const s_t reference_product_t269 = -reference_product_t263;
    const s_t reference_product_t270 = -reference_product_t255;
    const s_t reference_product_t271 = reference_product_t221 + reference_product_t226 - reference_product_t227;
    const s_t reference_product_t272 = reference_product_t219 - reference_product_t220;
    const s_t reference_product_t273 = reference_product_t225 + reference_product_t272;
    const s_t reference_product_t274 = reference_product_t271 + reference_product_t273;
    const s_t reference_product_t275 = reference_product_t239 - reference_product_t240;
    const s_t reference_product_t276 = reference_product_t212 + reference_product_t275;
    const s_t reference_product_t277 = reference_product_t217 + reference_product_t246 - reference_product_t247 + reference_product_t276;
    const s_t reference_product_t278 = reference_product_t251 - reference_product_t252;
    const s_t reference_product_t279 = reference_product_t233 + reference_product_t278;
    const s_t reference_product_t280 = reference_product_t271 + reference_product_t279;
    const s_t reference_product_t281 = -reference_product_t274;
    const s_t reference_product_t282 = reference_product_t213 - reference_product_t214 + reference_product_t241;
    const s_t reference_product_t283 = reference_product_t208 - reference_product_t209;
    const s_t reference_product_t284 = reference_product_t245 + reference_product_t282 + reference_product_t283;
    const s_t reference_product_t285 = reference_product_t234 - reference_product_t235 + reference_product_t253;
    const s_t reference_product_t286 = reference_product_t273 + reference_product_t285;
    const s_t reference_product_t287 = reference_product_t279 + reference_product_t285;
    const s_t reference_product_t288 = reference_product_t265 - reference_product_t266;
    const s_t reference_product_t289 = reference_product_t258 + reference_product_t282 + reference_product_t288;
    const s_t reference_product_t290 = -reference_product_t280;
    const s_t reference_product_t291 = -reference_product_t287;
    const s_t reference_product_t292 = reference_product_t259 - reference_product_t260 + reference_product_t267 + reference_product_t276;
    const s_t reference_product_t293 = -reference_product_t286;
    const s_t reference_product_t294 = reference_product_t236 + reference_product_t272;
    const s_t reference_product_t295 = reference_product_t221 + reference_product_t231 - reference_product_t232;
    const s_t reference_product_t296 = reference_product_t294 + reference_product_t295;
    const s_t reference_product_t297 = reference_product_t228 + reference_product_t278;
    const s_t reference_product_t298 = reference_product_t295 + reference_product_t297;
    const s_t reference_product_t299 = reference_product_t215 + reference_product_t275;
    const s_t reference_product_t300 = reference_product_t217 + reference_product_t256 - reference_product_t257 + reference_product_t299;
    const s_t reference_product_t301 = -reference_product_t296;
    const s_t reference_product_t302 = reference_product_t223 - reference_product_t224 + reference_product_t253;
    const s_t reference_product_t303 = reference_product_t294 + reference_product_t302;
    const s_t reference_product_t304 = reference_product_t210 - reference_product_t211 + reference_product_t241;
    const s_t reference_product_t305 = reference_product_t261 + reference_product_t283 + reference_product_t304;
    const s_t reference_product_t306 = reference_product_t297 + reference_product_t302;
    const s_t reference_product_t307 = -reference_product_t298;
    const s_t reference_product_t308 = reference_product_t248 + reference_product_t288 + reference_product_t304;
    const s_t reference_product_t309 = -reference_product_t306;
    const s_t reference_product_t310 = -reference_product_t303;
    const s_t reference_product_t311 = reference_product_t243 - reference_product_t244 + reference_product_t267 + reference_product_t299;
    const s_t output_t0 = reference_product_t10*tangent0;
    const s_t output_t1 = reference_product_t114*tangent3;
    const s_t output_t2 = reference_product_t173*tangent12;
    const s_t output_t3 = reference_product_t196*tangent20;
    const s_t output_t4 = reference_product_t218*tangent6;
    const s_t output_t5 = reference_product_t277*tangent15;
    const s_t output_t6 = reference_product_t300*tangent23;
    const s_t output_t7 = reference_product_t69*tangent9;
    const s_t output_t8 = reference_product_t92*tangent17;
    const s_t output_t9 = reference_product_t126*tangent11 + reference_product_t134*tangent18 + reference_product_t22*tangent1 + reference_product_t230*tangent14 + reference_product_t238*tangent21 + reference_product_t30*tangent2;
    const s_t output_t10 = reference_product_t170*tangent4 + reference_product_t176*tangent19 + reference_product_t274*tangent7 + reference_product_t280*tangent22 + reference_product_t66*tangent1 + reference_product_t72*tangent10;
    const s_t output_t11 = reference_product_t192*tangent5 + reference_product_t194*tangent13 + reference_product_t296*tangent8 + reference_product_t298*tangent16 + reference_product_t88*tangent2 + reference_product_t90*tangent10;
    const s_t output_t12 = reference_product_t180*tangent12;
    const s_t output_t13 = reference_product_t201*tangent20;
    const s_t output_t14 = reference_product_t284*tangent15;
    const s_t output_t15 = reference_product_t305*tangent23;
    const s_t output_t16 = reference_product_t76*tangent9;
    const s_t output_t17 = reference_product_t97*tangent17;
    const s_t output_t18 = reference_product_t177*tangent4 + reference_product_t182*tangent19 + reference_product_t281*tangent7 + reference_product_t286*tangent22 + reference_product_t73*tangent1 + reference_product_t78*tangent10;
    const s_t output_t19 = reference_product_t197*tangent5 + reference_product_t199*tangent13 + reference_product_t301*tangent8 + reference_product_t303*tangent16 + reference_product_t93*tangent2 + reference_product_t95*tangent10;
    const s_t output_t20 = reference_product_t100*tangent17;
    const s_t output_t21 = reference_product_t145*tangent3;
    const s_t output_t22 = reference_product_t204*tangent20;
    const s_t output_t23 = reference_product_t249*tangent6;
    const s_t output_t24 = reference_product_t308*tangent23;
    const s_t output_t25 = reference_product_t41*tangent0;
    const s_t output_t26 = reference_product_t146*tangent11 + reference_product_t151*tangent18 + reference_product_t250*tangent14 + reference_product_t255*tangent21 + reference_product_t42*tangent1 + reference_product_t47*tangent2;
    const s_t output_t27 = reference_product_t202*tangent5 + reference_product_t203*tangent13 + reference_product_t306*tangent8 + reference_product_t307*tangent16 + reference_product_t98*tangent2 + reference_product_t99*tangent10;
    const s_t output_t28 = reference_product_t103*tangent17;
    const s_t output_t29 = reference_product_t207*tangent20;
    const s_t output_t30 = reference_product_t311*tangent23;
    const s_t output_t31 = reference_product_t101*tangent2 + reference_product_t102*tangent10 + reference_product_t205*tangent5 + reference_product_t206*tangent13 + reference_product_t309*tangent8 + reference_product_t310*tangent16;
    const s_t output_t32 = reference_product_t158*tangent3;
    const s_t output_t33 = reference_product_t185*tangent12;
    const s_t output_t34 = reference_product_t262*tangent6;
    const s_t output_t35 = reference_product_t289*tangent15;
    const s_t output_t36 = reference_product_t54*tangent0;
    const s_t output_t37 = reference_product_t81*tangent9;
    const s_t output_t38 = reference_product_t159*tangent11 + reference_product_t160*tangent18 + reference_product_t263*tangent14 + reference_product_t264*tangent21 + reference_product_t55*tangent1 + reference_product_t56*tangent2;
    const s_t output_t39 = reference_product_t183*tangent4 + reference_product_t186*tangent19 + reference_product_t287*tangent7 + reference_product_t290*tangent22 + reference_product_t79*tangent1 + reference_product_t82*tangent10;
    const s_t output_t40 = reference_product_t188*tangent12;
    const s_t output_t41 = reference_product_t292*tangent15;
    const s_t output_t42 = reference_product_t84*tangent9;
    const s_t output_t43 = reference_product_t187*tangent4 + reference_product_t189*tangent19 + reference_product_t291*tangent7 + reference_product_t293*tangent22 + reference_product_t83*tangent1 + reference_product_t85*tangent10;
    const s_t output_t44 = reference_product_t164*tangent3;
    const s_t output_t45 = reference_product_t268*tangent6;
    const s_t output_t46 = reference_product_t60*tangent0;
    const s_t output_t47 = reference_product_t165*tangent11 + reference_product_t166*tangent18 + reference_product_t269*tangent14 + reference_product_t270*tangent21 + reference_product_t61*tangent1 + reference_product_t62*tangent2;
    const s_t output_t48 = reference_product_t10*tangent3;
    const s_t output_t49 = reference_product_t114*tangent24;
    const s_t output_t50 = reference_product_t173*tangent30;
    const s_t output_t51 = reference_product_t196*tangent35;
    const s_t output_t52 = reference_product_t218*tangent27;
    const s_t output_t53 = reference_product_t277*tangent33;
    const s_t output_t54 = reference_product_t300*tangent38;
    const s_t output_t55 = reference_product_t69*tangent12;
    const s_t output_t56 = reference_product_t92*tangent20;
    const s_t output_t57 = reference_product_t126*tangent25 + reference_product_t134*tangent26 + reference_product_t22*tangent4 + reference_product_t230*tangent32 + reference_product_t238*tangent36 + reference_product_t30*tangent5;
    const s_t output_t58 = reference_product_t170*tangent25 + reference_product_t176*tangent31 + reference_product_t274*tangent28 + reference_product_t280*tangent37 + reference_product_t66*tangent11 + reference_product_t72*tangent13;
    const s_t output_t59 = reference_product_t192*tangent26 + reference_product_t194*tangent31 + reference_product_t296*tangent29 + reference_product_t298*tangent34 + reference_product_t88*tangent18 + reference_product_t90*tangent19;
    const s_t output_t60 = reference_product_t180*tangent30;
    const s_t output_t61 = reference_product_t201*tangent35;
    const s_t output_t62 = reference_product_t284*tangent33;
    const s_t output_t63 = reference_product_t305*tangent38;
    const s_t output_t64 = reference_product_t76*tangent12;
    const s_t output_t65 = reference_product_t97*tangent20;
    const s_t output_t66 = reference_product_t177*tangent25 + reference_product_t182*tangent31 + reference_product_t281*tangent28 + reference_product_t286*tangent37 + reference_product_t73*tangent11 + reference_product_t78*tangent13;
    const s_t output_t67 = reference_product_t197*tangent26 + reference_product_t199*tangent31 + reference_product_t301*tangent29 + reference_product_t303*tangent34 + reference_product_t93*tangent18 + reference_product_t95*tangent19;
    const s_t output_t68 = reference_product_t100*tangent20;
    const s_t output_t69 = reference_product_t145*tangent24;
    const s_t output_t70 = reference_product_t204*tangent35;
    const s_t output_t71 = reference_product_t249*tangent27;
    const s_t output_t72 = reference_product_t308*tangent38;
    const s_t output_t73 = reference_product_t41*tangent3;
    const s_t output_t74 = reference_product_t146*tangent25 + reference_product_t151*tangent26 + reference_product_t250*tangent32 + reference_product_t255*tangent36 + reference_product_t42*tangent4 + reference_product_t47*tangent5;
    const s_t output_t75 = reference_product_t202*tangent26 + reference_product_t203*tangent31 + reference_product_t306*tangent29 + reference_product_t307*tangent34 + reference_product_t98*tangent18 + reference_product_t99*tangent19;
    const s_t output_t76 = reference_product_t103*tangent20;
    const s_t output_t77 = reference_product_t207*tangent35;
    const s_t output_t78 = reference_product_t311*tangent38;
    const s_t output_t79 = reference_product_t101*tangent18 + reference_product_t102*tangent19 + reference_product_t205*tangent26 + reference_product_t206*tangent31 + reference_product_t309*tangent29 + reference_product_t310*tangent34;
    const s_t output_t80 = reference_product_t158*tangent24;
    const s_t output_t81 = reference_product_t185*tangent30;
    const s_t output_t82 = reference_product_t262*tangent27;
    const s_t output_t83 = reference_product_t289*tangent33;
    const s_t output_t84 = reference_product_t54*tangent3;
    const s_t output_t85 = reference_product_t81*tangent12;
    const s_t output_t86 = reference_product_t159*tangent25 + reference_product_t160*tangent26 + reference_product_t263*tangent32 + reference_product_t264*tangent36 + reference_product_t55*tangent4 + reference_product_t56*tangent5;
    const s_t output_t87 = reference_product_t183*tangent25 + reference_product_t186*tangent31 + reference_product_t287*tangent28 + reference_product_t290*tangent37 + reference_product_t79*tangent11 + reference_product_t82*tangent13;
    const s_t output_t88 = reference_product_t188*tangent30;
    const s_t output_t89 = reference_product_t292*tangent33;
    const s_t output_t90 = reference_product_t84*tangent12;
    const s_t output_t91 = reference_product_t187*tangent25 + reference_product_t189*tangent31 + reference_product_t291*tangent28 + reference_product_t293*tangent37 + reference_product_t83*tangent11 + reference_product_t85*tangent13;
    const s_t output_t92 = reference_product_t164*tangent24;
    const s_t output_t93 = reference_product_t268*tangent27;
    const s_t output_t94 = reference_product_t60*tangent3;
    const s_t output_t95 = reference_product_t165*tangent25 + reference_product_t166*tangent26 + reference_product_t269*tangent32 + reference_product_t270*tangent36 + reference_product_t61*tangent4 + reference_product_t62*tangent5;
    const s_t output_t96 = reference_product_t10*tangent6;
    const s_t output_t97 = reference_product_t114*tangent27;
    const s_t output_t98 = reference_product_t173*tangent33;
    const s_t output_t99 = reference_product_t196*tangent38;
    const s_t output_t100 = reference_product_t218*tangent39;
    const s_t output_t101 = reference_product_t277*tangent42;
    const s_t output_t102 = reference_product_t300*tangent44;
    const s_t output_t103 = reference_product_t69*tangent15;
    const s_t output_t104 = reference_product_t92*tangent23;
    const s_t output_t105 = reference_product_t126*tangent28 + reference_product_t134*tangent29 + reference_product_t22*tangent7 + reference_product_t230*tangent40 + reference_product_t238*tangent41 + reference_product_t30*tangent8;
    const s_t output_t106 = reference_product_t170*tangent32 + reference_product_t176*tangent34 + reference_product_t274*tangent40 + reference_product_t280*tangent43 + reference_product_t66*tangent14 + reference_product_t72*tangent16;
    const s_t output_t107 = reference_product_t192*tangent36 + reference_product_t194*tangent37 + reference_product_t296*tangent41 + reference_product_t298*tangent43 + reference_product_t88*tangent21 + reference_product_t90*tangent22;
    const s_t output_t108 = reference_product_t180*tangent33;
    const s_t output_t109 = reference_product_t201*tangent38;
    const s_t output_t110 = reference_product_t284*tangent42;
    const s_t output_t111 = reference_product_t305*tangent44;
    const s_t output_t112 = reference_product_t76*tangent15;
    const s_t output_t113 = reference_product_t97*tangent23;
    const s_t output_t114 = reference_product_t177*tangent32 + reference_product_t182*tangent34 + reference_product_t281*tangent40 + reference_product_t286*tangent43 + reference_product_t73*tangent14 + reference_product_t78*tangent16;
    const s_t output_t115 = reference_product_t197*tangent36 + reference_product_t199*tangent37 + reference_product_t301*tangent41 + reference_product_t303*tangent43 + reference_product_t93*tangent21 + reference_product_t95*tangent22;
    const s_t output_t116 = reference_product_t100*tangent23;
    const s_t output_t117 = reference_product_t145*tangent27;
    const s_t output_t118 = reference_product_t204*tangent38;
    const s_t output_t119 = reference_product_t249*tangent39;
    const s_t output_t120 = reference_product_t308*tangent44;
    const s_t output_t121 = reference_product_t41*tangent6;
    const s_t output_t122 = reference_product_t146*tangent28 + reference_product_t151*tangent29 + reference_product_t250*tangent40 + reference_product_t255*tangent41 + reference_product_t42*tangent7 + reference_product_t47*tangent8;
    const s_t output_t123 = reference_product_t202*tangent36 + reference_product_t203*tangent37 + reference_product_t306*tangent41 + reference_product_t307*tangent43 + reference_product_t98*tangent21 + reference_product_t99*tangent22;
    const s_t output_t124 = reference_product_t103*tangent23;
    const s_t output_t125 = reference_product_t207*tangent38;
    const s_t output_t126 = reference_product_t311*tangent44;
    const s_t output_t127 = reference_product_t101*tangent21 + reference_product_t102*tangent22 + reference_product_t205*tangent36 + reference_product_t206*tangent37 + reference_product_t309*tangent41 + reference_product_t310*tangent43;
    const s_t output_t128 = reference_product_t158*tangent27;
    const s_t output_t129 = reference_product_t185*tangent33;
    const s_t output_t130 = reference_product_t262*tangent39;
    const s_t output_t131 = reference_product_t289*tangent42;
    const s_t output_t132 = reference_product_t54*tangent6;
    const s_t output_t133 = reference_product_t81*tangent15;
    const s_t output_t134 = reference_product_t159*tangent28 + reference_product_t160*tangent29 + reference_product_t263*tangent40 + reference_product_t264*tangent41 + reference_product_t55*tangent7 + reference_product_t56*tangent8;
    const s_t output_t135 = reference_product_t183*tangent32 + reference_product_t186*tangent34 + reference_product_t287*tangent40 + reference_product_t290*tangent43 + reference_product_t79*tangent14 + reference_product_t82*tangent16;
    const s_t output_t136 = reference_product_t188*tangent33;
    const s_t output_t137 = reference_product_t292*tangent42;
    const s_t output_t138 = reference_product_t84*tangent15;
    const s_t output_t139 = reference_product_t187*tangent32 + reference_product_t189*tangent34 + reference_product_t291*tangent40 + reference_product_t293*tangent43 + reference_product_t83*tangent14 + reference_product_t85*tangent16;
    const s_t output_t140 = reference_product_t164*tangent27;
    const s_t output_t141 = reference_product_t268*tangent39;
    const s_t output_t142 = reference_product_t60*tangent6;
    const s_t output_t143 = reference_product_t165*tangent28 + reference_product_t166*tangent29 + reference_product_t269*tangent40 + reference_product_t270*tangent41 + reference_product_t61*tangent7 + reference_product_t62*tangent8;
    const s_t element_out0_0 = output_t0 + output_t1 + output_t10 + output_t11 + output_t2 + output_t3 + output_t4 + output_t5 + output_t6 + output_t7 + output_t8 + output_t9;
    const s_t element_out0_1 = -output_t0 - output_t1 + output_t12 + output_t13 + output_t14 + output_t15 + output_t16 + output_t17 + output_t18 + output_t19 - output_t4 + output_t9;
    const s_t element_out0_2 = output_t10 - output_t2 + output_t20 + output_t21 + output_t22 + output_t23 + output_t24 + output_t25 + output_t26 + output_t27 - output_t5 - output_t7;
    const s_t element_out0_3 = -output_t12 - output_t14 - output_t16 + output_t18 - output_t21 - output_t23 - output_t25 + output_t26 + output_t28 + output_t29 + output_t30 + output_t31;
    const s_t element_out0_4 = output_t11 - output_t3 + output_t32 + output_t33 + output_t34 + output_t35 + output_t36 + output_t37 + output_t38 + output_t39 - output_t6 - output_t8;
    const s_t element_out0_5 = -output_t13 - output_t15 - output_t17 + output_t19 - output_t32 - output_t34 - output_t36 + output_t38 + output_t40 + output_t41 + output_t42 + output_t43;
    const s_t element_out0_6 = -output_t20 - output_t22 - output_t24 + output_t27 - output_t33 - output_t35 - output_t37 + output_t39 + output_t44 + output_t45 + output_t46 + output_t47;
    const s_t element_out0_7 = -output_t28 - output_t29 - output_t30 + output_t31 - output_t40 - output_t41 - output_t42 + output_t43 - output_t44 - output_t45 - output_t46 + output_t47;
    const s_t element_out1_0 = output_t48 + output_t49 + output_t50 + output_t51 + output_t52 + output_t53 + output_t54 + output_t55 + output_t56 + output_t57 + output_t58 + output_t59;
    const s_t element_out1_1 = -output_t48 - output_t49 - output_t52 + output_t57 + output_t60 + output_t61 + output_t62 + output_t63 + output_t64 + output_t65 + output_t66 + output_t67;
    const s_t element_out1_2 = -output_t50 - output_t53 - output_t55 + output_t58 + output_t68 + output_t69 + output_t70 + output_t71 + output_t72 + output_t73 + output_t74 + output_t75;
    const s_t element_out1_3 = -output_t60 - output_t62 - output_t64 + output_t66 - output_t69 - output_t71 - output_t73 + output_t74 + output_t76 + output_t77 + output_t78 + output_t79;
    const s_t element_out1_4 = -output_t51 - output_t54 - output_t56 + output_t59 + output_t80 + output_t81 + output_t82 + output_t83 + output_t84 + output_t85 + output_t86 + output_t87;
    const s_t element_out1_5 = -output_t61 - output_t63 - output_t65 + output_t67 - output_t80 - output_t82 - output_t84 + output_t86 + output_t88 + output_t89 + output_t90 + output_t91;
    const s_t element_out1_6 = -output_t68 - output_t70 - output_t72 + output_t75 - output_t81 - output_t83 - output_t85 + output_t87 + output_t92 + output_t93 + output_t94 + output_t95;
    const s_t element_out1_7 = -output_t76 - output_t77 - output_t78 + output_t79 - output_t88 - output_t89 - output_t90 + output_t91 - output_t92 - output_t93 - output_t94 + output_t95;
    const s_t element_out2_0 = output_t100 + output_t101 + output_t102 + output_t103 + output_t104 + output_t105 + output_t106 + output_t107 + output_t96 + output_t97 + output_t98 + output_t99;
    const s_t element_out2_1 = -output_t100 + output_t105 + output_t108 + output_t109 + output_t110 + output_t111 + output_t112 + output_t113 + output_t114 + output_t115 - output_t96 - output_t97;
    const s_t element_out2_2 = -output_t101 - output_t103 + output_t106 + output_t116 + output_t117 + output_t118 + output_t119 + output_t120 + output_t121 + output_t122 + output_t123 - output_t98;
    const s_t element_out2_3 = -output_t108 - output_t110 - output_t112 + output_t114 - output_t117 - output_t119 - output_t121 + output_t122 + output_t124 + output_t125 + output_t126 + output_t127;
    const s_t element_out2_4 = -output_t102 - output_t104 + output_t107 + output_t128 + output_t129 + output_t130 + output_t131 + output_t132 + output_t133 + output_t134 + output_t135 - output_t99;
    const s_t element_out2_5 = -output_t109 - output_t111 - output_t113 + output_t115 - output_t128 - output_t130 - output_t132 + output_t134 + output_t136 + output_t137 + output_t138 + output_t139;
    const s_t element_out2_6 = -output_t116 - output_t118 - output_t120 + output_t123 - output_t129 - output_t131 - output_t133 + output_t135 + output_t140 + output_t141 + output_t142 + output_t143;
    const s_t element_out2_7 = -output_t124 - output_t125 - output_t126 + output_t127 - output_t136 - output_t137 - output_t138 + output_t139 - output_t140 - output_t141 - output_t142 + output_t143;
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
  }

  return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem
