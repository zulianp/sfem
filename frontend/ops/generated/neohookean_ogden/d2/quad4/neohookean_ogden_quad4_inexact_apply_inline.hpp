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
      const s_t tangent_t0 = pow_2(adjugate1);
      const s_t tangent_t1 = pow_m1(determinant);
      const s_t tangent_t2 = ((s_t(1) / s_t(6)))*sqrt(s_t(3));
      const s_t tangent_t3 = tangent_t2 + (s_t(1) / s_t(2));
      const s_t tangent_t4 = -tangent_t1*tangent_t3;
      const s_t tangent_t5 = adjugate1*tangent_t4;
      const s_t tangent_t6 = adjugate3*tangent_t4;
      const s_t tangent_t7 = tangent_t5 + tangent_t6;
      const s_t tangent_t8 = adjugate1*tangent_t1;
      const s_t tangent_t9 = tangent_t3*tangent_t8;
      const s_t tangent_t10 = (s_t(1) / s_t(2)) - tangent_t2;
      const s_t tangent_t11 = -tangent_t10;
      const s_t tangent_t12 = adjugate3*tangent_t1;
      const s_t tangent_t13 = tangent_t11*tangent_t12;
      const s_t tangent_t14 = tangent_t13 + tangent_t9;
      const s_t tangent_t15 = tangent_t10*tangent_t8;
      const s_t tangent_t16 = tangent_t10*tangent_t12;
      const s_t tangent_t17 = tangent_t15 + tangent_t16;
      const s_t tangent_t18 = tangent_t11*tangent_t8;
      const s_t tangent_t19 = tangent_t12*tangent_t3;
      const s_t tangent_t20 = tangent_t18 + tangent_t19;
      const s_t tangent_t21 = tangent_t14*ux_1 + tangent_t17*ux_2 + tangent_t20*ux_3 + tangent_t7*ux_0;
      const s_t tangent_t22 = adjugate0*tangent_t4;
      const s_t tangent_t23 = adjugate2*tangent_t4;
      const s_t tangent_t24 = tangent_t22 + tangent_t23;
      const s_t tangent_t25 = adjugate0*tangent_t1;
      const s_t tangent_t26 = tangent_t25*tangent_t3;
      const s_t tangent_t27 = adjugate2*tangent_t1;
      const s_t tangent_t28 = tangent_t11*tangent_t27;
      const s_t tangent_t29 = tangent_t26 + tangent_t28;
      const s_t tangent_t30 = tangent_t10*tangent_t25;
      const s_t tangent_t31 = tangent_t10*tangent_t27;
      const s_t tangent_t32 = tangent_t30 + tangent_t31;
      const s_t tangent_t33 = tangent_t11*tangent_t25;
      const s_t tangent_t34 = tangent_t27*tangent_t3;
      const s_t tangent_t35 = tangent_t33 + tangent_t34;
      const s_t tangent_t36 = tangent_t24*uy_0 + tangent_t29*uy_1 + tangent_t32*uy_2 + tangent_t35*uy_3;
      const s_t tangent_t37 = tangent_t21*tangent_t36;
      const s_t tangent_t38 = tangent_t24*ux_0 + tangent_t29*ux_1 + tangent_t32*ux_2 + tangent_t35*ux_3 + s_t(1);
      const s_t tangent_t39 = tangent_t14*uy_1 + tangent_t17*uy_2 + tangent_t20*uy_3 + tangent_t7*uy_0 + s_t(1);
      const s_t tangent_t40 = -tangent_t37 + tangent_t38*tangent_t39;
      const s_t tangent_t41 = pow_m2(tangent_t40);
      const s_t tangent_t42 = pow_2(tangent_t36)*tangent_t41;
      const s_t tangent_t43 = lmbda*tangent_t42;
      const s_t tangent_t44 = log(tangent_t40);
      const s_t tangent_t45 = mu*tangent_t42 + mu - tangent_t43*tangent_t44 + tangent_t43;
      const s_t tangent_t46 = pow_2(adjugate0);
      const s_t tangent_t47 = pow_2(tangent_t39)*tangent_t41;
      const s_t tangent_t48 = lmbda*tangent_t47;
      const s_t tangent_t49 = mu*tangent_t47 + mu - tangent_t44*tangent_t48 + tangent_t48;
      const s_t tangent_t50 = tangent_t39*tangent_t41;
      const s_t tangent_t51 = tangent_t36*tangent_t50;
      const s_t tangent_t52 = lmbda*tangent_t36*tangent_t39*tangent_t41*tangent_t44 - lmbda*tangent_t51 - mu*tangent_t51;
      const s_t tangent_t53 = adjugate0*adjugate1;
      const s_t tangent_t54 = s_t(2)*tangent_t53;
      const s_t tangent_t55 = ((s_t(1) / s_t(4)))*tangent_t1;
      const s_t tangent_t56 = tangent_t13 + tangent_t5;
      const s_t tangent_t57 = tangent_t6 + tangent_t9;
      const s_t tangent_t58 = tangent_t15 + tangent_t19;
      const s_t tangent_t59 = tangent_t16 + tangent_t18;
      const s_t tangent_t60 = tangent_t56*ux_0 + tangent_t57*ux_1 + tangent_t58*ux_2 + tangent_t59*ux_3;
      const s_t tangent_t61 = tangent_t22 + tangent_t28;
      const s_t tangent_t62 = tangent_t23 + tangent_t26;
      const s_t tangent_t63 = tangent_t30 + tangent_t34;
      const s_t tangent_t64 = tangent_t31 + tangent_t33;
      const s_t tangent_t65 = tangent_t61*uy_0 + tangent_t62*uy_1 + tangent_t63*uy_2 + tangent_t64*uy_3;
      const s_t tangent_t66 = tangent_t60*tangent_t65;
      const s_t tangent_t67 = tangent_t61*ux_0 + tangent_t62*ux_1 + tangent_t63*ux_2 + tangent_t64*ux_3 + s_t(1);
      const s_t tangent_t68 = tangent_t56*uy_0 + tangent_t57*uy_1 + tangent_t58*uy_2 + tangent_t59*uy_3 + s_t(1);
      const s_t tangent_t69 = -tangent_t66 + tangent_t67*tangent_t68;
      const s_t tangent_t70 = pow_m2(tangent_t69);
      const s_t tangent_t71 = pow_2(tangent_t65)*tangent_t70;
      const s_t tangent_t72 = lmbda*tangent_t71;
      const s_t tangent_t73 = log(tangent_t69);
      const s_t tangent_t74 = mu*tangent_t71 + mu - tangent_t72*tangent_t73 + tangent_t72;
      const s_t tangent_t75 = pow_2(tangent_t68)*tangent_t70;
      const s_t tangent_t76 = lmbda*tangent_t75;
      const s_t tangent_t77 = mu*tangent_t75 + mu - tangent_t73*tangent_t76 + tangent_t76;
      const s_t tangent_t78 = tangent_t68*tangent_t70;
      const s_t tangent_t79 = tangent_t65*tangent_t78;
      const s_t tangent_t80 = lmbda*tangent_t65*tangent_t68*tangent_t70*tangent_t73 - lmbda*tangent_t79 - mu*tangent_t79;
      const s_t tangent_t81 = tangent_t18 + tangent_t6;
      const s_t tangent_t82 = tangent_t13 + tangent_t15;
      const s_t tangent_t83 = tangent_t16 + tangent_t9;
      const s_t tangent_t84 = tangent_t19 + tangent_t5;
      const s_t tangent_t85 = tangent_t81*ux_0 + tangent_t82*ux_1 + tangent_t83*ux_2 + tangent_t84*ux_3;
      const s_t tangent_t86 = tangent_t23 + tangent_t33;
      const s_t tangent_t87 = tangent_t28 + tangent_t30;
      const s_t tangent_t88 = tangent_t26 + tangent_t31;
      const s_t tangent_t89 = tangent_t22 + tangent_t34;
      const s_t tangent_t90 = tangent_t86*uy_0 + tangent_t87*uy_1 + tangent_t88*uy_2 + tangent_t89*uy_3;
      const s_t tangent_t91 = tangent_t85*tangent_t90;
      const s_t tangent_t92 = tangent_t86*ux_0 + tangent_t87*ux_1 + tangent_t88*ux_2 + tangent_t89*ux_3 + s_t(1);
      const s_t tangent_t93 = tangent_t81*uy_0 + tangent_t82*uy_1 + tangent_t83*uy_2 + tangent_t84*uy_3 + s_t(1);
      const s_t tangent_t94 = -tangent_t91 + tangent_t92*tangent_t93;
      const s_t tangent_t95 = pow_m2(tangent_t94);
      const s_t tangent_t96 = pow_2(tangent_t90)*tangent_t95;
      const s_t tangent_t97 = lmbda*tangent_t96;
      const s_t tangent_t98 = log(tangent_t94);
      const s_t tangent_t99 = mu*tangent_t96 + mu - tangent_t97*tangent_t98 + tangent_t97;
      const s_t tangent_t100 = pow_2(tangent_t93)*tangent_t95;
      const s_t tangent_t101 = lmbda*tangent_t100;
      const s_t tangent_t102 = mu*tangent_t100 + mu - tangent_t101*tangent_t98 + tangent_t101;
      const s_t tangent_t103 = tangent_t93*tangent_t95;
      const s_t tangent_t104 = tangent_t103*tangent_t90;
      const s_t tangent_t105 = -lmbda*tangent_t104 + lmbda*tangent_t90*tangent_t93*tangent_t95*tangent_t98 - mu*tangent_t104;
      const s_t tangent_t106 = tangent_t13 + tangent_t18;
      const s_t tangent_t107 = tangent_t15 + tangent_t6;
      const s_t tangent_t108 = tangent_t19 + tangent_t9;
      const s_t tangent_t109 = tangent_t16 + tangent_t5;
      const s_t tangent_t110 = tangent_t106*ux_0 + tangent_t107*ux_1 + tangent_t108*ux_2 + tangent_t109*ux_3;
      const s_t tangent_t111 = tangent_t28 + tangent_t33;
      const s_t tangent_t112 = tangent_t23 + tangent_t30;
      const s_t tangent_t113 = tangent_t26 + tangent_t34;
      const s_t tangent_t114 = tangent_t22 + tangent_t31;
      const s_t tangent_t115 = tangent_t111*uy_0 + tangent_t112*uy_1 + tangent_t113*uy_2 + tangent_t114*uy_3;
      const s_t tangent_t116 = tangent_t110*tangent_t115;
      const s_t tangent_t117 = tangent_t111*ux_0 + tangent_t112*ux_1 + tangent_t113*ux_2 + tangent_t114*ux_3 + s_t(1);
      const s_t tangent_t118 = tangent_t106*uy_0 + tangent_t107*uy_1 + tangent_t108*uy_2 + tangent_t109*uy_3 + s_t(1);
      const s_t tangent_t119 = -tangent_t116 + tangent_t117*tangent_t118;
      const s_t tangent_t120 = pow_m2(tangent_t119);
      const s_t tangent_t121 = pow_2(tangent_t115)*tangent_t120;
      const s_t tangent_t122 = lmbda*tangent_t121;
      const s_t tangent_t123 = log(tangent_t119);
      const s_t tangent_t124 = mu*tangent_t121 + mu - tangent_t122*tangent_t123 + tangent_t122;
      const s_t tangent_t125 = pow_2(tangent_t118)*tangent_t120;
      const s_t tangent_t126 = lmbda*tangent_t125;
      const s_t tangent_t127 = mu*tangent_t125 + mu - tangent_t123*tangent_t126 + tangent_t126;
      const s_t tangent_t128 = tangent_t118*tangent_t120;
      const s_t tangent_t129 = tangent_t115*tangent_t128;
      const s_t tangent_t130 = lmbda*tangent_t115*tangent_t118*tangent_t120*tangent_t123 - lmbda*tangent_t129 - mu*tangent_t129;
      const s_t tangent_t131 = adjugate1*adjugate3;
      const s_t tangent_t132 = adjugate0*adjugate2;
      const s_t tangent_t133 = adjugate0*adjugate3;
      const s_t tangent_t134 = adjugate1*adjugate2;
      const s_t tangent_t135 = tangent_t21*tangent_t50;
      const s_t tangent_t136 = -lmbda*tangent_t135 + lmbda*tangent_t21*tangent_t39*tangent_t41*tangent_t44 - mu*tangent_t135;
      const s_t tangent_t137 = lmbda*tangent_t41;
      const s_t tangent_t138 = tangent_t36*tangent_t38;
      const s_t tangent_t139 = mu*tangent_t41;
      const s_t tangent_t140 = lmbda*tangent_t36*tangent_t38*tangent_t41*tangent_t44 - tangent_t137*tangent_t138 - tangent_t138*tangent_t139;
      const s_t tangent_t141 = pow_m1(tangent_t40);
      const s_t tangent_t142 = mu*tangent_t141;
      const s_t tangent_t143 = lmbda*tangent_t44;
      const s_t tangent_t144 = tangent_t141*tangent_t143;
      const s_t tangent_t145 = tangent_t143*tangent_t41;
      const s_t tangent_t146 = tangent_t137*tangent_t37 + tangent_t139*tangent_t37 + tangent_t142 - tangent_t144 - tangent_t145*tangent_t37;
      const s_t tangent_t147 = tangent_t38*tangent_t39;
      const s_t tangent_t148 = tangent_t137*tangent_t147 + tangent_t139*tangent_t147 - tangent_t142 + tangent_t144 - tangent_t145*tangent_t147;
      const s_t tangent_t149 = tangent_t60*tangent_t78;
      const s_t tangent_t150 = -lmbda*tangent_t149 + lmbda*tangent_t60*tangent_t68*tangent_t70*tangent_t73 - mu*tangent_t149;
      const s_t tangent_t151 = lmbda*tangent_t70;
      const s_t tangent_t152 = tangent_t65*tangent_t67;
      const s_t tangent_t153 = mu*tangent_t70;
      const s_t tangent_t154 = lmbda*tangent_t65*tangent_t67*tangent_t70*tangent_t73 - tangent_t151*tangent_t152 - tangent_t152*tangent_t153;
      const s_t tangent_t155 = pow_m1(tangent_t69);
      const s_t tangent_t156 = mu*tangent_t155;
      const s_t tangent_t157 = lmbda*tangent_t73;
      const s_t tangent_t158 = tangent_t155*tangent_t157;
      const s_t tangent_t159 = tangent_t157*tangent_t70;
      const s_t tangent_t160 = tangent_t151*tangent_t66 + tangent_t153*tangent_t66 + tangent_t156 - tangent_t158 - tangent_t159*tangent_t66;
      const s_t tangent_t161 = tangent_t67*tangent_t68;
      const s_t tangent_t162 = tangent_t151*tangent_t161 + tangent_t153*tangent_t161 - tangent_t156 + tangent_t158 - tangent_t159*tangent_t161;
      const s_t tangent_t163 = tangent_t103*tangent_t85;
      const s_t tangent_t164 = -lmbda*tangent_t163 + lmbda*tangent_t85*tangent_t93*tangent_t95*tangent_t98 - mu*tangent_t163;
      const s_t tangent_t165 = lmbda*tangent_t95;
      const s_t tangent_t166 = tangent_t90*tangent_t92;
      const s_t tangent_t167 = mu*tangent_t95;
      const s_t tangent_t168 = lmbda*tangent_t90*tangent_t92*tangent_t95*tangent_t98 - tangent_t165*tangent_t166 - tangent_t166*tangent_t167;
      const s_t tangent_t169 = pow_m1(tangent_t94);
      const s_t tangent_t170 = mu*tangent_t169;
      const s_t tangent_t171 = lmbda*tangent_t98;
      const s_t tangent_t172 = tangent_t169*tangent_t171;
      const s_t tangent_t173 = tangent_t171*tangent_t95;
      const s_t tangent_t174 = tangent_t165*tangent_t91 + tangent_t167*tangent_t91 + tangent_t170 - tangent_t172 - tangent_t173*tangent_t91;
      const s_t tangent_t175 = tangent_t92*tangent_t93;
      const s_t tangent_t176 = tangent_t165*tangent_t175 + tangent_t167*tangent_t175 - tangent_t170 + tangent_t172 - tangent_t173*tangent_t175;
      const s_t tangent_t177 = tangent_t110*tangent_t128;
      const s_t tangent_t178 = lmbda*tangent_t110*tangent_t118*tangent_t120*tangent_t123 - lmbda*tangent_t177 - mu*tangent_t177;
      const s_t tangent_t179 = lmbda*tangent_t120;
      const s_t tangent_t180 = tangent_t115*tangent_t117;
      const s_t tangent_t181 = mu*tangent_t120;
      const s_t tangent_t182 = lmbda*tangent_t115*tangent_t117*tangent_t120*tangent_t123 - tangent_t179*tangent_t180 - tangent_t180*tangent_t181;
      const s_t tangent_t183 = pow_m1(tangent_t119);
      const s_t tangent_t184 = mu*tangent_t183;
      const s_t tangent_t185 = lmbda*tangent_t123;
      const s_t tangent_t186 = tangent_t183*tangent_t185;
      const s_t tangent_t187 = tangent_t120*tangent_t185;
      const s_t tangent_t188 = tangent_t116*tangent_t179 + tangent_t116*tangent_t181 - tangent_t116*tangent_t187 + tangent_t184 - tangent_t186;
      const s_t tangent_t189 = tangent_t117*tangent_t118;
      const s_t tangent_t190 = tangent_t179*tangent_t189 + tangent_t181*tangent_t189 - tangent_t184 + tangent_t186 - tangent_t187*tangent_t189;
      const s_t tangent_t191 = tangent_t131*tangent_t140 + tangent_t132*tangent_t136;
      const s_t tangent_t192 = tangent_t131*tangent_t154 + tangent_t132*tangent_t150;
      const s_t tangent_t193 = tangent_t131*tangent_t168 + tangent_t132*tangent_t164;
      const s_t tangent_t194 = tangent_t131*tangent_t182 + tangent_t132*tangent_t178;
      const s_t tangent_t195 = pow_2(adjugate3);
      const s_t tangent_t196 = pow_2(adjugate2);
      const s_t tangent_t197 = adjugate2*adjugate3;
      const s_t tangent_t198 = s_t(2)*tangent_t197;
      const s_t tangent_t199 = pow_2(tangent_t21)*tangent_t41;
      const s_t tangent_t200 = lmbda*tangent_t199 + mu*tangent_t199 + mu - tangent_t143*tangent_t199;
      const s_t tangent_t201 = pow_2(tangent_t38)*tangent_t41;
      const s_t tangent_t202 = lmbda*tangent_t201 + mu*tangent_t201 + mu - tangent_t143*tangent_t201;
      const s_t tangent_t203 = tangent_t21*tangent_t38;
      const s_t tangent_t204 = lmbda*tangent_t21*tangent_t38*tangent_t41*tangent_t44 - tangent_t137*tangent_t203 - tangent_t139*tangent_t203;
      const s_t tangent_t205 = pow_2(tangent_t60)*tangent_t70;
      const s_t tangent_t206 = lmbda*tangent_t205 + mu*tangent_t205 + mu - tangent_t157*tangent_t205;
      const s_t tangent_t207 = pow_2(tangent_t67)*tangent_t70;
      const s_t tangent_t208 = lmbda*tangent_t207 + mu*tangent_t207 + mu - tangent_t157*tangent_t207;
      const s_t tangent_t209 = tangent_t60*tangent_t67;
      const s_t tangent_t210 = lmbda*tangent_t60*tangent_t67*tangent_t70*tangent_t73 - tangent_t151*tangent_t209 - tangent_t153*tangent_t209;
      const s_t tangent_t211 = pow_2(tangent_t85)*tangent_t95;
      const s_t tangent_t212 = lmbda*tangent_t211 + mu*tangent_t211 + mu - tangent_t171*tangent_t211;
      const s_t tangent_t213 = pow_2(tangent_t92)*tangent_t95;
      const s_t tangent_t214 = lmbda*tangent_t213 + mu*tangent_t213 + mu - tangent_t171*tangent_t213;
      const s_t tangent_t215 = tangent_t85*tangent_t92;
      const s_t tangent_t216 = lmbda*tangent_t85*tangent_t92*tangent_t95*tangent_t98 - tangent_t165*tangent_t215 - tangent_t167*tangent_t215;
      const s_t tangent_t217 = pow_2(tangent_t110)*tangent_t120;
      const s_t tangent_t218 = lmbda*tangent_t217 + mu*tangent_t217 + mu - tangent_t185*tangent_t217;
      const s_t tangent_t219 = pow_2(tangent_t117)*tangent_t120;
      const s_t tangent_t220 = lmbda*tangent_t219 + mu*tangent_t219 + mu - tangent_t185*tangent_t219;
      const s_t tangent_t221 = tangent_t110*tangent_t117;
      const s_t tangent_t222 = lmbda*tangent_t110*tangent_t117*tangent_t120*tangent_t123 - tangent_t179*tangent_t221 - tangent_t181*tangent_t221;
      const s_t tangent0 = tangent_t55*(tangent_t0*tangent_t124 + tangent_t127*tangent_t46 + tangent_t130*tangent_t54) + tangent_t55*(tangent_t0*tangent_t45 + tangent_t46*tangent_t49 + tangent_t52*tangent_t54) + tangent_t55*(tangent_t0*tangent_t74 + tangent_t46*tangent_t77 + tangent_t54*tangent_t80) + tangent_t55*(tangent_t0*tangent_t99 + tangent_t102*tangent_t46 + tangent_t105*tangent_t54);
      const s_t tangent1 = tangent_t55*(tangent_t102*tangent_t132 + tangent_t105*tangent_t133 + tangent_t105*tangent_t134 + tangent_t131*tangent_t99) + tangent_t55*(tangent_t124*tangent_t131 + tangent_t127*tangent_t132 + tangent_t130*tangent_t133 + tangent_t130*tangent_t134) + tangent_t55*(tangent_t131*tangent_t45 + tangent_t132*tangent_t49 + tangent_t133*tangent_t52 + tangent_t134*tangent_t52) + tangent_t55*(tangent_t131*tangent_t74 + tangent_t132*tangent_t77 + tangent_t133*tangent_t80 + tangent_t134*tangent_t80);
      const s_t tangent2 = tangent_t55*(tangent_t0*tangent_t140 + tangent_t136*tangent_t46 + tangent_t146*tangent_t53 + tangent_t148*tangent_t53) + tangent_t55*(tangent_t0*tangent_t154 + tangent_t150*tangent_t46 + tangent_t160*tangent_t53 + tangent_t162*tangent_t53) + tangent_t55*(tangent_t0*tangent_t168 + tangent_t164*tangent_t46 + tangent_t174*tangent_t53 + tangent_t176*tangent_t53) + tangent_t55*(tangent_t0*tangent_t182 + tangent_t178*tangent_t46 + tangent_t188*tangent_t53 + tangent_t190*tangent_t53);
      const s_t tangent3 = tangent_t55*(tangent_t133*tangent_t148 + tangent_t134*tangent_t146 + tangent_t191) + tangent_t55*(tangent_t133*tangent_t162 + tangent_t134*tangent_t160 + tangent_t192) + tangent_t55*(tangent_t133*tangent_t176 + tangent_t134*tangent_t174 + tangent_t193) + tangent_t55*(tangent_t133*tangent_t190 + tangent_t134*tangent_t188 + tangent_t194);
      const s_t tangent4 = tangent_t55*(tangent_t102*tangent_t196 + tangent_t105*tangent_t198 + tangent_t195*tangent_t99) + tangent_t55*(tangent_t124*tangent_t195 + tangent_t127*tangent_t196 + tangent_t130*tangent_t198) + tangent_t55*(tangent_t195*tangent_t45 + tangent_t196*tangent_t49 + tangent_t198*tangent_t52) + tangent_t55*(tangent_t195*tangent_t74 + tangent_t196*tangent_t77 + tangent_t198*tangent_t80);
      const s_t tangent5 = tangent_t55*(tangent_t133*tangent_t146 + tangent_t134*tangent_t148 + tangent_t191) + tangent_t55*(tangent_t133*tangent_t160 + tangent_t134*tangent_t162 + tangent_t192) + tangent_t55*(tangent_t133*tangent_t174 + tangent_t134*tangent_t176 + tangent_t193) + tangent_t55*(tangent_t133*tangent_t188 + tangent_t134*tangent_t190 + tangent_t194);
      const s_t tangent6 = tangent_t55*(tangent_t136*tangent_t196 + tangent_t140*tangent_t195 + tangent_t146*tangent_t197 + tangent_t148*tangent_t197) + tangent_t55*(tangent_t150*tangent_t196 + tangent_t154*tangent_t195 + tangent_t160*tangent_t197 + tangent_t162*tangent_t197) + tangent_t55*(tangent_t164*tangent_t196 + tangent_t168*tangent_t195 + tangent_t174*tangent_t197 + tangent_t176*tangent_t197) + tangent_t55*(tangent_t178*tangent_t196 + tangent_t182*tangent_t195 + tangent_t188*tangent_t197 + tangent_t190*tangent_t197);
      const s_t tangent7 = tangent_t55*(tangent_t0*tangent_t202 + tangent_t200*tangent_t46 + tangent_t204*tangent_t54) + tangent_t55*(tangent_t0*tangent_t208 + tangent_t206*tangent_t46 + tangent_t210*tangent_t54) + tangent_t55*(tangent_t0*tangent_t214 + tangent_t212*tangent_t46 + tangent_t216*tangent_t54) + tangent_t55*(tangent_t0*tangent_t220 + tangent_t218*tangent_t46 + tangent_t222*tangent_t54);
      const s_t tangent8 = tangent_t55*(tangent_t131*tangent_t202 + tangent_t132*tangent_t200 + tangent_t133*tangent_t204 + tangent_t134*tangent_t204) + tangent_t55*(tangent_t131*tangent_t208 + tangent_t132*tangent_t206 + tangent_t133*tangent_t210 + tangent_t134*tangent_t210) + tangent_t55*(tangent_t131*tangent_t214 + tangent_t132*tangent_t212 + tangent_t133*tangent_t216 + tangent_t134*tangent_t216) + tangent_t55*(tangent_t131*tangent_t220 + tangent_t132*tangent_t218 + tangent_t133*tangent_t222 + tangent_t134*tangent_t222);
      const s_t tangent9 = tangent_t55*(tangent_t195*tangent_t202 + tangent_t196*tangent_t200 + tangent_t198*tangent_t204) + tangent_t55*(tangent_t195*tangent_t208 + tangent_t196*tangent_t206 + tangent_t198*tangent_t210) + tangent_t55*(tangent_t195*tangent_t214 + tangent_t196*tangent_t212 + tangent_t198*tangent_t216) + tangent_t55*(tangent_t195*tangent_t220 + tangent_t196*tangent_t218 + tangent_t198*tangent_t222);
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
