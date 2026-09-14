#pragma once
#include "../../../kernel_math.hpp"
#include "../../../reference/line_p1_q2.hpp"
#include "../../../reference/quad_line_q2.hpp"
#include "../../../tensor_product_kernels.hpp"

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t, typename tangent_t, int VS>
static SFEM_INLINE int neohookean_ogden_proteus_quad4_inexact_apply_tangent_a_msoa_impl(
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
    static constexpr int NQ1 = 2;
    static constexpr int NS = 4;
    const s_t *const RSTR shape_1d = sfem::codegen::ref_line_p1_q2<s_t>::shape_1d();
    const s_t *const RSTR grad_1d = sfem::codegen::ref_line_p1_q2<s_t>::grad_1d();
    const s_t *const RSTR q_weight_1d = sfem::codegen::quad_line_q2<s_t>::q_weight_1d();
    static constexpr s_t QMEASURE = s_t(1);
    s_t btangent_acc[10][VS];
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
    s_t bu_data[NS * 2][VS];
    s_t gu_ref_q[NQ * 4 * VS];
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      bu_data[0][lane] = ux[bev0[lane] * u_stride];
      bu_data[1][lane] = uy[bev0[lane] * u_stride];
      bu_data[2][lane] = ux[bev1[lane] * u_stride];
      bu_data[3][lane] = uy[bev1[lane] * u_stride];
      bu_data[4][lane] = ux[bev2[lane] * u_stride];
      bu_data[5][lane] = uy[bev2[lane] * u_stride];
      bu_data[6][lane] = ux[bev3[lane] * u_stride];
      bu_data[7][lane] = uy[bev3[lane] * u_stride];
    }
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 2, 2>(ne, shape_1d, grad_1d, bu_data, 0, &gu_ref_q[0 * VS]);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 2, 2>(ne, shape_1d, grad_1d, bu_data, 1, &gu_ref_q[8 * VS]);
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
      const s_t *const RSTR gu_ref0 = &gu_ref_q[(0 + q * 2) * VS];
      const s_t *const RSTR gu_ref1 = &gu_ref_q[(1 + q * 2) * VS];
      const s_t *const RSTR gu_ref2 = &gu_ref_q[(8 + q * 2) * VS];
      const s_t *const RSTR gu_ref3 = &gu_ref_q[(9 + q * 2) * VS];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
      const s_t adjugate0 = s_t(bg_adj0[lane]);
      const s_t adjugate1 = s_t(bg_adj1[lane]);
      const s_t adjugate2 = s_t(bg_adj2[lane]);
      const s_t adjugate3 = s_t(bg_adj3[lane]);
      const s_t determinant = s_t(bg_det0[lane]);
      const s_t gu_ref_0_0 = gu_ref0[lane];
      const s_t gu_ref_0_1 = gu_ref1[lane];
      const s_t gu_ref_1_0 = gu_ref2[lane];
      const s_t gu_ref_1_1 = gu_ref3[lane];
        const s_t qw = q_weight_1d[qx] * q_weight_1d[qy] * QMEASURE;
            const s_t gradient0_t0 = pow_m1(determinant);
            const s_t gradient0_t1 = gradient0_t0*gu_ref_0_0;
            const s_t gradient0_t2 = gradient0_t0*gu_ref_0_1;
            const s_t gradient0_t3 = gradient0_t0*gu_ref_1_0;
            const s_t gradient0_t4 = gradient0_t0*gu_ref_1_1;
            const s_t gu_0_0 = adjugate0*gradient0_t1 + adjugate2*gradient0_t2;
            const s_t gu_0_1 = adjugate1*gradient0_t1 + adjugate3*gradient0_t2;
            const s_t gu_1_0 = adjugate0*gradient0_t3 + adjugate2*gradient0_t4;
            const s_t gu_1_1 = adjugate1*gradient0_t3 + adjugate3*gradient0_t4;
            const s_t integrand_t0 = pow_m1(determinant);
            const s_t integrand_t1 = pow_2(adjugate1);
            const s_t integrand_t2 = gu_0_1*gu_1_0;
            const s_t integrand_t3 = gu_0_0 + s_t(1);
            const s_t integrand_t4 = gu_1_1 + s_t(1);
            const s_t integrand_t5 = -integrand_t2 + integrand_t3*integrand_t4;
            const s_t integrand_t6 = pow_m2(integrand_t5);
            const s_t integrand_t7 = pow_2(gu_1_0)*integrand_t6;
            const s_t integrand_t8 = integrand_t7*lmbda;
            const s_t integrand_t9 = log(integrand_t5);
            const s_t integrand_t10 = integrand_t7*mu - integrand_t8*integrand_t9 + integrand_t8 + mu;
            const s_t integrand_t11 = adjugate0*adjugate1;
            const s_t integrand_t12 = integrand_t4*integrand_t6;
            const s_t integrand_t13 = gu_1_0*integrand_t12;
            const s_t integrand_t14 = gu_1_0*integrand_t4*integrand_t6*integrand_t9*lmbda - integrand_t13*lmbda - integrand_t13*mu;
            const s_t integrand_t15 = s_t(2)*integrand_t14;
            const s_t integrand_t16 = pow_2(adjugate0);
            const s_t integrand_t17 = pow_2(integrand_t4)*integrand_t6;
            const s_t integrand_t18 = integrand_t17*lmbda;
            const s_t integrand_t19 = integrand_t17*mu - integrand_t18*integrand_t9 + integrand_t18 + mu;
            const s_t integrand_t20 = adjugate1*adjugate3;
            const s_t integrand_t21 = adjugate0*adjugate3;
            const s_t integrand_t22 = adjugate1*adjugate2;
            const s_t integrand_t23 = adjugate0*adjugate2;
            const s_t integrand_t24 = gu_0_1*integrand_t12;
            const s_t integrand_t25 = gu_0_1*integrand_t4*integrand_t6*integrand_t9*lmbda - integrand_t24*lmbda - integrand_t24*mu;
            const s_t integrand_t26 = integrand_t6*lmbda;
            const s_t integrand_t27 = gu_1_0*integrand_t3;
            const s_t integrand_t28 = integrand_t6*mu;
            const s_t integrand_t29 = gu_1_0*integrand_t3*integrand_t6*integrand_t9*lmbda - integrand_t26*integrand_t27 - integrand_t27*integrand_t28;
            const s_t integrand_t30 = pow_m1(integrand_t5);
            const s_t integrand_t31 = integrand_t30*mu;
            const s_t integrand_t32 = integrand_t9*lmbda;
            const s_t integrand_t33 = integrand_t30*integrand_t32;
            const s_t integrand_t34 = integrand_t32*integrand_t6;
            const s_t integrand_t35 = integrand_t2*integrand_t26 + integrand_t2*integrand_t28 - integrand_t2*integrand_t34 + integrand_t31 - integrand_t33;
            const s_t integrand_t36 = integrand_t3*integrand_t4;
            const s_t integrand_t37 = integrand_t26*integrand_t36 + integrand_t28*integrand_t36 - integrand_t31 + integrand_t33 - integrand_t34*integrand_t36;
            const s_t integrand_t38 = integrand_t20*integrand_t29 + integrand_t23*integrand_t25;
            const s_t integrand_t39 = pow_2(adjugate3);
            const s_t integrand_t40 = adjugate2*adjugate3;
            const s_t integrand_t41 = pow_2(adjugate2);
            const s_t integrand_t42 = pow_2(gu_0_1)*integrand_t6;
            const s_t integrand_t43 = -integrand_t32*integrand_t42 + integrand_t42*lmbda + integrand_t42*mu + mu;
            const s_t integrand_t44 = gu_0_1*integrand_t3;
            const s_t integrand_t45 = gu_0_1*integrand_t3*integrand_t6*integrand_t9*lmbda - integrand_t26*integrand_t44 - integrand_t28*integrand_t44;
            const s_t integrand_t46 = s_t(2)*integrand_t45;
            const s_t integrand_t47 = pow_2(integrand_t3)*integrand_t6;
            const s_t integrand_t48 = -integrand_t32*integrand_t47 + integrand_t47*lmbda + integrand_t47*mu + mu;
            const s_t integrand0 = integrand_t0*(integrand_t1*integrand_t10 + integrand_t11*integrand_t15 + integrand_t16*integrand_t19);
            const s_t integrand1 = integrand_t0*(integrand_t10*integrand_t20 + integrand_t14*integrand_t21 + integrand_t14*integrand_t22 + integrand_t19*integrand_t23);
            const s_t integrand2 = integrand_t0*(integrand_t1*integrand_t29 + integrand_t11*integrand_t35 + integrand_t11*integrand_t37 + integrand_t16*integrand_t25);
            const s_t integrand3 = integrand_t0*(integrand_t21*integrand_t37 + integrand_t22*integrand_t35 + integrand_t38);
            const s_t integrand4 = integrand_t0*(integrand_t10*integrand_t39 + integrand_t15*integrand_t40 + integrand_t19*integrand_t41);
            const s_t integrand5 = integrand_t0*(integrand_t21*integrand_t35 + integrand_t22*integrand_t37 + integrand_t38);
            const s_t integrand6 = integrand_t0*(integrand_t25*integrand_t41 + integrand_t29*integrand_t39 + integrand_t35*integrand_t40 + integrand_t37*integrand_t40);
            const s_t integrand7 = integrand_t0*(integrand_t1*integrand_t48 + integrand_t11*integrand_t46 + integrand_t16*integrand_t43);
            const s_t integrand8 = integrand_t0*(integrand_t20*integrand_t48 + integrand_t21*integrand_t45 + integrand_t22*integrand_t45 + integrand_t23*integrand_t43);
            const s_t integrand9 = integrand_t0*(integrand_t39*integrand_t48 + integrand_t40*integrand_t46 + integrand_t41*integrand_t43);
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
static SFEM_INLINE int neohookean_ogden_proteus_quad4_inexact_apply_stored_a_msoa_impl(
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
      const s_t output_t0 = reference_product_t14*tangent4;
      const s_t output_t1 = reference_product_t20*tangent2;
      const s_t output_t2 = reference_product_t3*tangent0;
      const s_t output_t3 = reference_product_t31*tangent6;
      const s_t output_t4 = reference_product_t24*tangent5 + reference_product_t7*tangent1;
      const s_t output_t5 = reference_product_t13*tangent1 + reference_product_t30*tangent3;
      const s_t output_t6 = reference_product_t16*tangent4 + reference_product_t33*tangent6;
      const s_t output_t7 = reference_product_t11*tangent0 + reference_product_t28*tangent2;
      const s_t output_t8 = reference_product_t14*tangent6;
      const s_t output_t9 = reference_product_t20*tangent7;
      const s_t output_t10 = reference_product_t3*tangent2;
      const s_t output_t11 = reference_product_t31*tangent9;
      const s_t output_t12 = reference_product_t24*tangent8 + reference_product_t7*tangent3;
      const s_t output_t13 = reference_product_t13*tangent5 + reference_product_t30*tangent8;
      const s_t output_t14 = reference_product_t16*tangent6 + reference_product_t33*tangent9;
      const s_t output_t15 = reference_product_t11*tangent2 + reference_product_t28*tangent7;
      const s_t element_out0_0 = output_t0 + output_t1 + output_t2 + output_t3 + output_t4 + output_t5;
      const s_t element_out0_1 = -output_t1 - output_t2 + output_t4 + output_t6 + reference_product_t15*tangent1 + reference_product_t32*tangent3;
      const s_t element_out0_2 = -output_t0 - output_t3 + output_t5 + output_t7 + reference_product_t12*tangent1 + reference_product_t29*tangent5;
      const s_t element_out0_3 = -output_t6 - output_t7 + reference_product_t12*tangent1 + reference_product_t15*tangent1 + reference_product_t29*tangent5 + reference_product_t32*tangent3;
      const s_t element_out1_0 = output_t10 + output_t11 + output_t12 + output_t13 + output_t8 + output_t9;
      const s_t element_out1_1 = -output_t10 + output_t12 + output_t14 - output_t9 + reference_product_t15*tangent5 + reference_product_t32*tangent8;
      const s_t element_out1_2 = -output_t11 + output_t13 + output_t15 - output_t8 + reference_product_t12*tangent3 + reference_product_t29*tangent8;
      const s_t element_out1_3 = -output_t14 - output_t15 + reference_product_t12*tangent3 + reference_product_t15*tangent5 + reference_product_t29*tangent8 + reference_product_t32*tangent8;
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
static SFEM_INLINE int neohookean_ogden_proteus_quad4_inexact_apply_compressed_a_msoa_impl(
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
    const s_t output_t0 = reference_product_t14*tangent4;
    const s_t output_t1 = reference_product_t20*tangent2;
    const s_t output_t2 = reference_product_t3*tangent0;
    const s_t output_t3 = reference_product_t31*tangent6;
    const s_t output_t4 = reference_product_t24*tangent5 + reference_product_t7*tangent1;
    const s_t output_t5 = reference_product_t13*tangent1 + reference_product_t30*tangent3;
    const s_t output_t6 = reference_product_t16*tangent4 + reference_product_t33*tangent6;
    const s_t output_t7 = reference_product_t11*tangent0 + reference_product_t28*tangent2;
    const s_t output_t8 = reference_product_t14*tangent6;
    const s_t output_t9 = reference_product_t20*tangent7;
    const s_t output_t10 = reference_product_t3*tangent2;
    const s_t output_t11 = reference_product_t31*tangent9;
    const s_t output_t12 = reference_product_t24*tangent8 + reference_product_t7*tangent3;
    const s_t output_t13 = reference_product_t13*tangent5 + reference_product_t30*tangent8;
    const s_t output_t14 = reference_product_t16*tangent6 + reference_product_t33*tangent9;
    const s_t output_t15 = reference_product_t11*tangent2 + reference_product_t28*tangent7;
    const s_t element_out0_0 = output_t0 + output_t1 + output_t2 + output_t3 + output_t4 + output_t5;
    const s_t element_out0_1 = -output_t1 - output_t2 + output_t4 + output_t6 + reference_product_t15*tangent1 + reference_product_t32*tangent3;
    const s_t element_out0_2 = -output_t0 - output_t3 + output_t5 + output_t7 + reference_product_t12*tangent1 + reference_product_t29*tangent5;
    const s_t element_out0_3 = -output_t6 - output_t7 + reference_product_t12*tangent1 + reference_product_t15*tangent1 + reference_product_t29*tangent5 + reference_product_t32*tangent3;
    const s_t element_out1_0 = output_t10 + output_t11 + output_t12 + output_t13 + output_t8 + output_t9;
    const s_t element_out1_1 = -output_t10 + output_t12 + output_t14 - output_t9 + reference_product_t15*tangent5 + reference_product_t32*tangent8;
    const s_t element_out1_2 = -output_t11 + output_t13 + output_t15 - output_t8 + reference_product_t12*tangent3 + reference_product_t29*tangent8;
    const s_t element_out1_3 = -output_t14 - output_t15 + reference_product_t12*tangent3 + reference_product_t15*tangent5 + reference_product_t29*tangent8 + reference_product_t32*tangent8;
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
