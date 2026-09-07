#include "../../../kernel_math.hpp"

namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t, typename tangent_t>
static SFEM_INLINE int linear_elasticity_tet4_inexact_apply_tangent_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        idx_t **const SFEM_RESTRICT elements,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const scalar_t lmbda,
        const scalar_t mu,
        const ptrdiff_t u_stride,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const scalar_t *const SFEM_RESTRICT uz,
        const ptrdiff_t tangent_element_stride,
        const ptrdiff_t tangent_component_stride,
        tangent_t *const SFEM_RESTRICT tangent
) {
    #pragma omp parallel for schedule(static)
    for (ptrdiff_t element = 0; element < nelements; ++element) {
        const idx_t ev0 = elements[0][element];
        const idx_t ev1 = elements[1][element];
        const idx_t ev2 = elements[2][element];
        const idx_t ev3 = elements[3][element];
        const scalar_t adjugate0 = scalar_t(g_jacobian_adjugate0[element]);
        const scalar_t adjugate1 = scalar_t(g_jacobian_adjugate1[element]);
        const scalar_t adjugate2 = scalar_t(g_jacobian_adjugate2[element]);
        const scalar_t adjugate3 = scalar_t(g_jacobian_adjugate3[element]);
        const scalar_t adjugate4 = scalar_t(g_jacobian_adjugate4[element]);
        const scalar_t adjugate5 = scalar_t(g_jacobian_adjugate5[element]);
        const scalar_t adjugate6 = scalar_t(g_jacobian_adjugate6[element]);
        const scalar_t adjugate7 = scalar_t(g_jacobian_adjugate7[element]);
        const scalar_t adjugate8 = scalar_t(g_jacobian_adjugate8[element]);
        const scalar_t determinant = scalar_t(g_jacobian_determinant0[element]);
        const scalar_t tangent_t0 = pow_m1(determinant);
        const scalar_t tangent_t1 = pow_2(adjugate1);
        const scalar_t tangent_t2 = mu*tangent_t1;
        const scalar_t tangent_t3 = pow_2(adjugate2);
        const scalar_t tangent_t4 = mu*tangent_t3;
        const scalar_t tangent_t5 = pow_2(adjugate0);
        const scalar_t tangent_t6 = lmbda + scalar_t(2)*mu;
        const scalar_t tangent_t7 = adjugate1*mu;
        const scalar_t tangent_t8 = adjugate4*tangent_t7;
        const scalar_t tangent_t9 = adjugate2*mu;
        const scalar_t tangent_t10 = adjugate5*tangent_t9;
        const scalar_t tangent_t11 = adjugate0*tangent_t6;
        const scalar_t tangent_t12 = adjugate7*tangent_t7;
        const scalar_t tangent_t13 = adjugate8*tangent_t9;
        const scalar_t tangent_t14 = adjugate0*lmbda;
        const scalar_t tangent_t15 = pow_2(adjugate4);
        const scalar_t tangent_t16 = mu*tangent_t15;
        const scalar_t tangent_t17 = pow_2(adjugate5);
        const scalar_t tangent_t18 = mu*tangent_t17;
        const scalar_t tangent_t19 = pow_2(adjugate3);
        const scalar_t tangent_t20 = adjugate4*mu;
        const scalar_t tangent_t21 = adjugate7*tangent_t20;
        const scalar_t tangent_t22 = adjugate5*mu;
        const scalar_t tangent_t23 = adjugate8*tangent_t22;
        const scalar_t tangent_t24 = adjugate3*adjugate6;
        const scalar_t tangent_t25 = adjugate3*lmbda;
        const scalar_t tangent_t26 = pow_2(adjugate7);
        const scalar_t tangent_t27 = mu*tangent_t26;
        const scalar_t tangent_t28 = pow_2(adjugate8);
        const scalar_t tangent_t29 = mu*tangent_t28;
        const scalar_t tangent_t30 = pow_2(adjugate6);
        const scalar_t tangent_t31 = adjugate7*mu;
        const scalar_t tangent_t32 = adjugate6*lmbda;
        const scalar_t tangent_t33 = adjugate8*mu;
        const scalar_t tangent_t34 = mu*tangent_t5;
        const scalar_t tangent_t35 = adjugate0*mu;
        const scalar_t tangent_t36 = adjugate3*tangent_t35;
        const scalar_t tangent_t37 = adjugate1*tangent_t6;
        const scalar_t tangent_t38 = adjugate6*tangent_t35;
        const scalar_t tangent_t39 = adjugate1*lmbda;
        const scalar_t tangent_t40 = mu*tangent_t19;
        const scalar_t tangent_t41 = mu*tangent_t24;
        const scalar_t tangent_t42 = adjugate4*lmbda;
        const scalar_t tangent_t43 = mu*tangent_t30;
        const scalar_t tangent_t44 = adjugate7*lmbda;
        const scalar_t tangent_t45 = adjugate2*tangent_t6;
        const scalar_t tangent0 = tangent_t0*(tangent_t2 + tangent_t4 + tangent_t5*tangent_t6);
        const scalar_t tangent1 = tangent_t0*(adjugate3*tangent_t11 + tangent_t10 + tangent_t8);
        const scalar_t tangent2 = tangent_t0*(adjugate6*tangent_t11 + tangent_t12 + tangent_t13);
        const scalar_t tangent3 = tangent_t0*(adjugate0*tangent_t7 + adjugate1*tangent_t14);
        const scalar_t tangent4 = tangent_t0*(adjugate3*tangent_t7 + adjugate4*tangent_t14);
        const scalar_t tangent5 = tangent_t0*(adjugate6*tangent_t7 + adjugate7*tangent_t14);
        const scalar_t tangent6 = tangent_t0*(adjugate0*tangent_t9 + adjugate2*tangent_t14);
        const scalar_t tangent7 = tangent_t0*(adjugate3*tangent_t9 + adjugate5*tangent_t14);
        const scalar_t tangent8 = tangent_t0*(adjugate6*tangent_t9 + adjugate8*tangent_t14);
        const scalar_t tangent9 = tangent_t0*(tangent_t16 + tangent_t18 + tangent_t19*tangent_t6);
        const scalar_t tangent10 = tangent_t0*(tangent_t21 + tangent_t23 + tangent_t24*tangent_t6);
        const scalar_t tangent11 = tangent_t0*(adjugate0*tangent_t20 + adjugate1*tangent_t25);
        const scalar_t tangent12 = tangent_t0*(adjugate3*tangent_t20 + adjugate4*tangent_t25);
        const scalar_t tangent13 = tangent_t0*(adjugate6*tangent_t20 + adjugate7*tangent_t25);
        const scalar_t tangent14 = tangent_t0*(adjugate0*tangent_t22 + adjugate2*tangent_t25);
        const scalar_t tangent15 = tangent_t0*(adjugate3*tangent_t22 + adjugate5*tangent_t25);
        const scalar_t tangent16 = tangent_t0*(adjugate6*tangent_t22 + adjugate8*tangent_t25);
        const scalar_t tangent17 = tangent_t0*(tangent_t27 + tangent_t29 + tangent_t30*tangent_t6);
        const scalar_t tangent18 = tangent_t0*(adjugate0*tangent_t31 + adjugate1*tangent_t32);
        const scalar_t tangent19 = tangent_t0*(adjugate3*tangent_t31 + adjugate4*tangent_t32);
        const scalar_t tangent20 = tangent_t0*(adjugate6*tangent_t31 + adjugate7*tangent_t32);
        const scalar_t tangent21 = tangent_t0*(adjugate0*tangent_t33 + adjugate2*tangent_t32);
        const scalar_t tangent22 = tangent_t0*(adjugate3*tangent_t33 + adjugate5*tangent_t32);
        const scalar_t tangent23 = tangent_t0*(adjugate6*tangent_t33 + adjugate8*tangent_t32);
        const scalar_t tangent24 = tangent_t0*(tangent_t1*tangent_t6 + tangent_t34 + tangent_t4);
        const scalar_t tangent25 = tangent_t0*(adjugate4*tangent_t37 + tangent_t10 + tangent_t36);
        const scalar_t tangent26 = tangent_t0*(adjugate7*tangent_t37 + tangent_t13 + tangent_t38);
        const scalar_t tangent27 = tangent_t0*(adjugate2*tangent_t39 + adjugate2*tangent_t7);
        const scalar_t tangent28 = tangent_t0*(adjugate4*tangent_t9 + adjugate5*tangent_t39);
        const scalar_t tangent29 = tangent_t0*(adjugate7*tangent_t9 + adjugate8*tangent_t39);
        const scalar_t tangent30 = tangent_t0*(tangent_t15*tangent_t6 + tangent_t18 + tangent_t40);
        const scalar_t tangent31 = tangent_t0*(adjugate4*adjugate7*tangent_t6 + tangent_t23 + tangent_t41);
        const scalar_t tangent32 = tangent_t0*(adjugate2*tangent_t42 + adjugate5*tangent_t7);
        const scalar_t tangent33 = tangent_t0*(adjugate5*tangent_t20 + adjugate5*tangent_t42);
        const scalar_t tangent34 = tangent_t0*(adjugate7*tangent_t22 + adjugate8*tangent_t42);
        const scalar_t tangent35 = tangent_t0*(tangent_t26*tangent_t6 + tangent_t29 + tangent_t43);
        const scalar_t tangent36 = tangent_t0*(adjugate2*tangent_t44 + adjugate8*tangent_t7);
        const scalar_t tangent37 = tangent_t0*(adjugate5*tangent_t44 + adjugate8*tangent_t20);
        const scalar_t tangent38 = tangent_t0*(adjugate8*tangent_t31 + adjugate8*tangent_t44);
        const scalar_t tangent39 = tangent_t0*(tangent_t2 + tangent_t3*tangent_t6 + tangent_t34);
        const scalar_t tangent40 = tangent_t0*(adjugate5*tangent_t45 + tangent_t36 + tangent_t8);
        const scalar_t tangent41 = tangent_t0*(adjugate8*tangent_t45 + tangent_t12 + tangent_t38);
        const scalar_t tangent42 = tangent_t0*(tangent_t16 + tangent_t17*tangent_t6 + tangent_t40);
        const scalar_t tangent43 = tangent_t0*(adjugate5*adjugate8*tangent_t6 + tangent_t21 + tangent_t41);
        const scalar_t tangent44 = tangent_t0*(tangent_t27 + tangent_t28*tangent_t6 + tangent_t43);
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
        tangent[element * tangent_element_stride + 10 * tangent_component_stride] = tangent_t(tangent10);
        tangent[element * tangent_element_stride + 11 * tangent_component_stride] = tangent_t(tangent11);
        tangent[element * tangent_element_stride + 12 * tangent_component_stride] = tangent_t(tangent12);
        tangent[element * tangent_element_stride + 13 * tangent_component_stride] = tangent_t(tangent13);
        tangent[element * tangent_element_stride + 14 * tangent_component_stride] = tangent_t(tangent14);
        tangent[element * tangent_element_stride + 15 * tangent_component_stride] = tangent_t(tangent15);
        tangent[element * tangent_element_stride + 16 * tangent_component_stride] = tangent_t(tangent16);
        tangent[element * tangent_element_stride + 17 * tangent_component_stride] = tangent_t(tangent17);
        tangent[element * tangent_element_stride + 18 * tangent_component_stride] = tangent_t(tangent18);
        tangent[element * tangent_element_stride + 19 * tangent_component_stride] = tangent_t(tangent19);
        tangent[element * tangent_element_stride + 20 * tangent_component_stride] = tangent_t(tangent20);
        tangent[element * tangent_element_stride + 21 * tangent_component_stride] = tangent_t(tangent21);
        tangent[element * tangent_element_stride + 22 * tangent_component_stride] = tangent_t(tangent22);
        tangent[element * tangent_element_stride + 23 * tangent_component_stride] = tangent_t(tangent23);
        tangent[element * tangent_element_stride + 24 * tangent_component_stride] = tangent_t(tangent24);
        tangent[element * tangent_element_stride + 25 * tangent_component_stride] = tangent_t(tangent25);
        tangent[element * tangent_element_stride + 26 * tangent_component_stride] = tangent_t(tangent26);
        tangent[element * tangent_element_stride + 27 * tangent_component_stride] = tangent_t(tangent27);
        tangent[element * tangent_element_stride + 28 * tangent_component_stride] = tangent_t(tangent28);
        tangent[element * tangent_element_stride + 29 * tangent_component_stride] = tangent_t(tangent29);
        tangent[element * tangent_element_stride + 30 * tangent_component_stride] = tangent_t(tangent30);
        tangent[element * tangent_element_stride + 31 * tangent_component_stride] = tangent_t(tangent31);
        tangent[element * tangent_element_stride + 32 * tangent_component_stride] = tangent_t(tangent32);
        tangent[element * tangent_element_stride + 33 * tangent_component_stride] = tangent_t(tangent33);
        tangent[element * tangent_element_stride + 34 * tangent_component_stride] = tangent_t(tangent34);
        tangent[element * tangent_element_stride + 35 * tangent_component_stride] = tangent_t(tangent35);
        tangent[element * tangent_element_stride + 36 * tangent_component_stride] = tangent_t(tangent36);
        tangent[element * tangent_element_stride + 37 * tangent_component_stride] = tangent_t(tangent37);
        tangent[element * tangent_element_stride + 38 * tangent_component_stride] = tangent_t(tangent38);
        tangent[element * tangent_element_stride + 39 * tangent_component_stride] = tangent_t(tangent39);
        tangent[element * tangent_element_stride + 40 * tangent_component_stride] = tangent_t(tangent40);
        tangent[element * tangent_element_stride + 41 * tangent_component_stride] = tangent_t(tangent41);
        tangent[element * tangent_element_stride + 42 * tangent_component_stride] = tangent_t(tangent42);
        tangent[element * tangent_element_stride + 43 * tangent_component_stride] = tangent_t(tangent43);
        tangent[element * tangent_element_stride + 44 * tangent_component_stride] = tangent_t(tangent44);
    }

    return SFEM_SUCCESS;
}

template <typename scalar_t, typename tangent_t>
static SFEM_INLINE int linear_elasticity_tet4_inexact_apply_stored_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        idx_t **const SFEM_RESTRICT elements,
        const ptrdiff_t tangent_element_stride,
        const ptrdiff_t tangent_component_stride,
        const tangent_t *const SFEM_RESTRICT tangent,
        const ptrdiff_t h_stride,
        const scalar_t *const SFEM_RESTRICT hx,
        const scalar_t *const SFEM_RESTRICT hy,
        const scalar_t *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT outx,
        scalar_t *const SFEM_RESTRICT outy,
        scalar_t *const SFEM_RESTRICT outz
) {
    #pragma omp parallel for schedule(static)
    for (ptrdiff_t element = 0; element < nelements; ++element) {
        const idx_t ev0 = elements[0][element];
        const idx_t ev1 = elements[1][element];
        const idx_t ev2 = elements[2][element];
        const idx_t ev3 = elements[3][element];
        const scalar_t hx_0 = hx[ev0 * h_stride];
        const scalar_t hx_1 = hx[ev1 * h_stride];
        const scalar_t hx_2 = hx[ev2 * h_stride];
        const scalar_t hx_3 = hx[ev3 * h_stride];
        const scalar_t hy_0 = hy[ev0 * h_stride];
        const scalar_t hy_1 = hy[ev1 * h_stride];
        const scalar_t hy_2 = hy[ev2 * h_stride];
        const scalar_t hy_3 = hy[ev3 * h_stride];
        const scalar_t hz_0 = hz[ev0 * h_stride];
        const scalar_t hz_1 = hz[ev1 * h_stride];
        const scalar_t hz_2 = hz[ev2 * h_stride];
        const scalar_t hz_3 = hz[ev3 * h_stride];
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
        const scalar_t tangent10 = scalar_t(tangent[element * tangent_element_stride + 10 * tangent_component_stride]);
        const scalar_t tangent11 = scalar_t(tangent[element * tangent_element_stride + 11 * tangent_component_stride]);
        const scalar_t tangent12 = scalar_t(tangent[element * tangent_element_stride + 12 * tangent_component_stride]);
        const scalar_t tangent13 = scalar_t(tangent[element * tangent_element_stride + 13 * tangent_component_stride]);
        const scalar_t tangent14 = scalar_t(tangent[element * tangent_element_stride + 14 * tangent_component_stride]);
        const scalar_t tangent15 = scalar_t(tangent[element * tangent_element_stride + 15 * tangent_component_stride]);
        const scalar_t tangent16 = scalar_t(tangent[element * tangent_element_stride + 16 * tangent_component_stride]);
        const scalar_t tangent17 = scalar_t(tangent[element * tangent_element_stride + 17 * tangent_component_stride]);
        const scalar_t tangent18 = scalar_t(tangent[element * tangent_element_stride + 18 * tangent_component_stride]);
        const scalar_t tangent19 = scalar_t(tangent[element * tangent_element_stride + 19 * tangent_component_stride]);
        const scalar_t tangent20 = scalar_t(tangent[element * tangent_element_stride + 20 * tangent_component_stride]);
        const scalar_t tangent21 = scalar_t(tangent[element * tangent_element_stride + 21 * tangent_component_stride]);
        const scalar_t tangent22 = scalar_t(tangent[element * tangent_element_stride + 22 * tangent_component_stride]);
        const scalar_t tangent23 = scalar_t(tangent[element * tangent_element_stride + 23 * tangent_component_stride]);
        const scalar_t tangent24 = scalar_t(tangent[element * tangent_element_stride + 24 * tangent_component_stride]);
        const scalar_t tangent25 = scalar_t(tangent[element * tangent_element_stride + 25 * tangent_component_stride]);
        const scalar_t tangent26 = scalar_t(tangent[element * tangent_element_stride + 26 * tangent_component_stride]);
        const scalar_t tangent27 = scalar_t(tangent[element * tangent_element_stride + 27 * tangent_component_stride]);
        const scalar_t tangent28 = scalar_t(tangent[element * tangent_element_stride + 28 * tangent_component_stride]);
        const scalar_t tangent29 = scalar_t(tangent[element * tangent_element_stride + 29 * tangent_component_stride]);
        const scalar_t tangent30 = scalar_t(tangent[element * tangent_element_stride + 30 * tangent_component_stride]);
        const scalar_t tangent31 = scalar_t(tangent[element * tangent_element_stride + 31 * tangent_component_stride]);
        const scalar_t tangent32 = scalar_t(tangent[element * tangent_element_stride + 32 * tangent_component_stride]);
        const scalar_t tangent33 = scalar_t(tangent[element * tangent_element_stride + 33 * tangent_component_stride]);
        const scalar_t tangent34 = scalar_t(tangent[element * tangent_element_stride + 34 * tangent_component_stride]);
        const scalar_t tangent35 = scalar_t(tangent[element * tangent_element_stride + 35 * tangent_component_stride]);
        const scalar_t tangent36 = scalar_t(tangent[element * tangent_element_stride + 36 * tangent_component_stride]);
        const scalar_t tangent37 = scalar_t(tangent[element * tangent_element_stride + 37 * tangent_component_stride]);
        const scalar_t tangent38 = scalar_t(tangent[element * tangent_element_stride + 38 * tangent_component_stride]);
        const scalar_t tangent39 = scalar_t(tangent[element * tangent_element_stride + 39 * tangent_component_stride]);
        const scalar_t tangent40 = scalar_t(tangent[element * tangent_element_stride + 40 * tangent_component_stride]);
        const scalar_t tangent41 = scalar_t(tangent[element * tangent_element_stride + 41 * tangent_component_stride]);
        const scalar_t tangent42 = scalar_t(tangent[element * tangent_element_stride + 42 * tangent_component_stride]);
        const scalar_t tangent43 = scalar_t(tangent[element * tangent_element_stride + 43 * tangent_component_stride]);
        const scalar_t tangent44 = scalar_t(tangent[element * tangent_element_stride + 44 * tangent_component_stride]);
        const scalar_t compressed_increment_t0 = ((scalar_t(1) / scalar_t(6)))*hx_0;
        const scalar_t compressed_increment_t1 = ((scalar_t(1) / scalar_t(6)))*hy_0;
        const scalar_t compressed_increment_t2 = ((scalar_t(1) / scalar_t(6)))*hz_0;
        const scalar_t pa_p0_0_0 = compressed_increment_t0 - (scalar_t(1) / scalar_t(6))*hx_1;
        const scalar_t pa_p0_0_1 = compressed_increment_t0 - (scalar_t(1) / scalar_t(6))*hx_2;
        const scalar_t pa_p0_0_2 = compressed_increment_t0 - (scalar_t(1) / scalar_t(6))*hx_3;
        const scalar_t pa_p1_0_0 = compressed_increment_t1 - (scalar_t(1) / scalar_t(6))*hy_1;
        const scalar_t pa_p1_0_1 = compressed_increment_t1 - (scalar_t(1) / scalar_t(6))*hy_2;
        const scalar_t pa_p1_0_2 = compressed_increment_t1 - (scalar_t(1) / scalar_t(6))*hy_3;
        const scalar_t pa_p2_0_0 = compressed_increment_t2 - (scalar_t(1) / scalar_t(6))*hz_1;
        const scalar_t pa_p2_0_1 = compressed_increment_t2 - (scalar_t(1) / scalar_t(6))*hz_2;
        const scalar_t pa_p2_0_2 = compressed_increment_t2 - (scalar_t(1) / scalar_t(6))*hz_3;
        const scalar_t pa_y0_0_0 = pa_p0_0_0*tangent0 + pa_p0_0_1*tangent1 + pa_p0_0_2*tangent2 + pa_p1_0_0*tangent3 + pa_p1_0_1*tangent4 + pa_p1_0_2*tangent5 + pa_p2_0_0*tangent6 + pa_p2_0_1*tangent7 + pa_p2_0_2*tangent8;
        const scalar_t pa_y0_0_1 = pa_p0_0_0*tangent1 + pa_p0_0_1*tangent9 + pa_p0_0_2*tangent10 + pa_p1_0_0*tangent11 + pa_p1_0_1*tangent12 + pa_p1_0_2*tangent13 + pa_p2_0_0*tangent14 + pa_p2_0_1*tangent15 + pa_p2_0_2*tangent16;
        const scalar_t pa_y0_0_2 = pa_p0_0_0*tangent2 + pa_p0_0_1*tangent10 + pa_p0_0_2*tangent17 + pa_p1_0_0*tangent18 + pa_p1_0_1*tangent19 + pa_p1_0_2*tangent20 + pa_p2_0_0*tangent21 + pa_p2_0_1*tangent22 + pa_p2_0_2*tangent23;
        const scalar_t pa_y1_0_0 = pa_p0_0_0*tangent3 + pa_p0_0_1*tangent11 + pa_p0_0_2*tangent18 + pa_p1_0_0*tangent24 + pa_p1_0_1*tangent25 + pa_p1_0_2*tangent26 + pa_p2_0_0*tangent27 + pa_p2_0_1*tangent28 + pa_p2_0_2*tangent29;
        const scalar_t pa_y1_0_1 = pa_p0_0_0*tangent4 + pa_p0_0_1*tangent12 + pa_p0_0_2*tangent19 + pa_p1_0_0*tangent25 + pa_p1_0_1*tangent30 + pa_p1_0_2*tangent31 + pa_p2_0_0*tangent32 + pa_p2_0_1*tangent33 + pa_p2_0_2*tangent34;
        const scalar_t pa_y1_0_2 = pa_p0_0_0*tangent5 + pa_p0_0_1*tangent13 + pa_p0_0_2*tangent20 + pa_p1_0_0*tangent26 + pa_p1_0_1*tangent31 + pa_p1_0_2*tangent35 + pa_p2_0_0*tangent36 + pa_p2_0_1*tangent37 + pa_p2_0_2*tangent38;
        const scalar_t pa_y2_0_0 = pa_p0_0_0*tangent6 + pa_p0_0_1*tangent14 + pa_p0_0_2*tangent21 + pa_p1_0_0*tangent27 + pa_p1_0_1*tangent32 + pa_p1_0_2*tangent36 + pa_p2_0_0*tangent39 + pa_p2_0_1*tangent40 + pa_p2_0_2*tangent41;
        const scalar_t pa_y2_0_1 = pa_p0_0_0*tangent7 + pa_p0_0_1*tangent15 + pa_p0_0_2*tangent22 + pa_p1_0_0*tangent28 + pa_p1_0_1*tangent33 + pa_p1_0_2*tangent37 + pa_p2_0_0*tangent40 + pa_p2_0_1*tangent42 + pa_p2_0_2*tangent43;
        const scalar_t pa_y2_0_2 = pa_p0_0_0*tangent8 + pa_p0_0_1*tangent16 + pa_p0_0_2*tangent23 + pa_p1_0_0*tangent29 + pa_p1_0_1*tangent34 + pa_p1_0_2*tangent38 + pa_p2_0_0*tangent41 + pa_p2_0_1*tangent43 + pa_p2_0_2*tangent44;
        const scalar_t pa_q0_0_0 = scalar_t(6)*pa_y0_0_0;
        const scalar_t pa_q0_0_1 = scalar_t(6)*pa_y0_0_1;
        const scalar_t pa_q0_0_2 = scalar_t(6)*pa_y0_0_2;
        const scalar_t pa_q1_0_0 = scalar_t(6)*pa_y1_0_0;
        const scalar_t pa_q1_0_1 = scalar_t(6)*pa_y1_0_1;
        const scalar_t pa_q1_0_2 = scalar_t(6)*pa_y1_0_2;
        const scalar_t pa_q2_0_0 = scalar_t(6)*pa_y2_0_0;
        const scalar_t pa_q2_0_1 = scalar_t(6)*pa_y2_0_1;
        const scalar_t pa_q2_0_2 = scalar_t(6)*pa_y2_0_2;
        const scalar_t output_t0 = ((scalar_t(1) / scalar_t(6)))*pa_q0_0_0;
        const scalar_t output_t1 = ((scalar_t(1) / scalar_t(6)))*pa_q0_0_1;
        const scalar_t output_t2 = ((scalar_t(1) / scalar_t(6)))*pa_q0_0_2;
        const scalar_t output_t3 = ((scalar_t(1) / scalar_t(6)))*pa_q1_0_0;
        const scalar_t output_t4 = ((scalar_t(1) / scalar_t(6)))*pa_q1_0_1;
        const scalar_t output_t5 = ((scalar_t(1) / scalar_t(6)))*pa_q1_0_2;
        const scalar_t output_t6 = ((scalar_t(1) / scalar_t(6)))*pa_q2_0_0;
        const scalar_t output_t7 = ((scalar_t(1) / scalar_t(6)))*pa_q2_0_1;
        const scalar_t output_t8 = ((scalar_t(1) / scalar_t(6)))*pa_q2_0_2;
        const scalar_t element_out0_0 = output_t0 + output_t1 + output_t2;
        const scalar_t element_out0_1 = -output_t0;
        const scalar_t element_out0_2 = -output_t1;
        const scalar_t element_out0_3 = -output_t2;
        const scalar_t element_out1_0 = output_t3 + output_t4 + output_t5;
        const scalar_t element_out1_1 = -output_t3;
        const scalar_t element_out1_2 = -output_t4;
        const scalar_t element_out1_3 = -output_t5;
        const scalar_t element_out2_0 = output_t6 + output_t7 + output_t8;
        const scalar_t element_out2_1 = -output_t6;
        const scalar_t element_out2_2 = -output_t7;
        const scalar_t element_out2_3 = -output_t8;
        #pragma omp atomic update
        outx[ev0 * out_stride] += element_out0_0;
        #pragma omp atomic update
        outx[ev1 * out_stride] += element_out0_1;
        #pragma omp atomic update
        outx[ev2 * out_stride] += element_out0_2;
        #pragma omp atomic update
        outx[ev3 * out_stride] += element_out0_3;
        #pragma omp atomic update
        outy[ev0 * out_stride] += element_out1_0;
        #pragma omp atomic update
        outy[ev1 * out_stride] += element_out1_1;
        #pragma omp atomic update
        outy[ev2 * out_stride] += element_out1_2;
        #pragma omp atomic update
        outy[ev3 * out_stride] += element_out1_3;
        #pragma omp atomic update
        outz[ev0 * out_stride] += element_out2_0;
        #pragma omp atomic update
        outz[ev1 * out_stride] += element_out2_1;
        #pragma omp atomic update
        outz[ev2 * out_stride] += element_out2_2;
        #pragma omp atomic update
        outz[ev3 * out_stride] += element_out2_3;
    }

    return SFEM_SUCCESS;
}

template <typename scalar_t, typename tangent_t, typename scale_t>
static SFEM_INLINE int linear_elasticity_tet4_inexact_apply_compressed_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        idx_t **const SFEM_RESTRICT elements,
        const ptrdiff_t tangent_element_stride,
        const ptrdiff_t tangent_component_stride,
        const tangent_t *const SFEM_RESTRICT tangent,
        const scale_t *const SFEM_RESTRICT scaling,
        const ptrdiff_t h_stride,
        const scalar_t *const SFEM_RESTRICT hx,
        const scalar_t *const SFEM_RESTRICT hy,
        const scalar_t *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT outx,
        scalar_t *const SFEM_RESTRICT outy,
        scalar_t *const SFEM_RESTRICT outz
) {
    #pragma omp parallel for schedule(static)
    for (ptrdiff_t element = 0; element < nelements; ++element) {
        const idx_t ev0 = elements[0][element];
        const idx_t ev1 = elements[1][element];
        const idx_t ev2 = elements[2][element];
        const idx_t ev3 = elements[3][element];
        const scalar_t hx_0 = hx[ev0 * h_stride];
        const scalar_t hx_1 = hx[ev1 * h_stride];
        const scalar_t hx_2 = hx[ev2 * h_stride];
        const scalar_t hx_3 = hx[ev3 * h_stride];
        const scalar_t hy_0 = hy[ev0 * h_stride];
        const scalar_t hy_1 = hy[ev1 * h_stride];
        const scalar_t hy_2 = hy[ev2 * h_stride];
        const scalar_t hy_3 = hy[ev3 * h_stride];
        const scalar_t hz_0 = hz[ev0 * h_stride];
        const scalar_t hz_1 = hz[ev1 * h_stride];
        const scalar_t hz_2 = hz[ev2 * h_stride];
        const scalar_t hz_3 = hz[ev3 * h_stride];
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
        const scalar_t tangent10 = scalar_t(tangent[element * tangent_element_stride + 10 * tangent_component_stride]);
        const scalar_t tangent11 = scalar_t(tangent[element * tangent_element_stride + 11 * tangent_component_stride]);
        const scalar_t tangent12 = scalar_t(tangent[element * tangent_element_stride + 12 * tangent_component_stride]);
        const scalar_t tangent13 = scalar_t(tangent[element * tangent_element_stride + 13 * tangent_component_stride]);
        const scalar_t tangent14 = scalar_t(tangent[element * tangent_element_stride + 14 * tangent_component_stride]);
        const scalar_t tangent15 = scalar_t(tangent[element * tangent_element_stride + 15 * tangent_component_stride]);
        const scalar_t tangent16 = scalar_t(tangent[element * tangent_element_stride + 16 * tangent_component_stride]);
        const scalar_t tangent17 = scalar_t(tangent[element * tangent_element_stride + 17 * tangent_component_stride]);
        const scalar_t tangent18 = scalar_t(tangent[element * tangent_element_stride + 18 * tangent_component_stride]);
        const scalar_t tangent19 = scalar_t(tangent[element * tangent_element_stride + 19 * tangent_component_stride]);
        const scalar_t tangent20 = scalar_t(tangent[element * tangent_element_stride + 20 * tangent_component_stride]);
        const scalar_t tangent21 = scalar_t(tangent[element * tangent_element_stride + 21 * tangent_component_stride]);
        const scalar_t tangent22 = scalar_t(tangent[element * tangent_element_stride + 22 * tangent_component_stride]);
        const scalar_t tangent23 = scalar_t(tangent[element * tangent_element_stride + 23 * tangent_component_stride]);
        const scalar_t tangent24 = scalar_t(tangent[element * tangent_element_stride + 24 * tangent_component_stride]);
        const scalar_t tangent25 = scalar_t(tangent[element * tangent_element_stride + 25 * tangent_component_stride]);
        const scalar_t tangent26 = scalar_t(tangent[element * tangent_element_stride + 26 * tangent_component_stride]);
        const scalar_t tangent27 = scalar_t(tangent[element * tangent_element_stride + 27 * tangent_component_stride]);
        const scalar_t tangent28 = scalar_t(tangent[element * tangent_element_stride + 28 * tangent_component_stride]);
        const scalar_t tangent29 = scalar_t(tangent[element * tangent_element_stride + 29 * tangent_component_stride]);
        const scalar_t tangent30 = scalar_t(tangent[element * tangent_element_stride + 30 * tangent_component_stride]);
        const scalar_t tangent31 = scalar_t(tangent[element * tangent_element_stride + 31 * tangent_component_stride]);
        const scalar_t tangent32 = scalar_t(tangent[element * tangent_element_stride + 32 * tangent_component_stride]);
        const scalar_t tangent33 = scalar_t(tangent[element * tangent_element_stride + 33 * tangent_component_stride]);
        const scalar_t tangent34 = scalar_t(tangent[element * tangent_element_stride + 34 * tangent_component_stride]);
        const scalar_t tangent35 = scalar_t(tangent[element * tangent_element_stride + 35 * tangent_component_stride]);
        const scalar_t tangent36 = scalar_t(tangent[element * tangent_element_stride + 36 * tangent_component_stride]);
        const scalar_t tangent37 = scalar_t(tangent[element * tangent_element_stride + 37 * tangent_component_stride]);
        const scalar_t tangent38 = scalar_t(tangent[element * tangent_element_stride + 38 * tangent_component_stride]);
        const scalar_t tangent39 = scalar_t(tangent[element * tangent_element_stride + 39 * tangent_component_stride]);
        const scalar_t tangent40 = scalar_t(tangent[element * tangent_element_stride + 40 * tangent_component_stride]);
        const scalar_t tangent41 = scalar_t(tangent[element * tangent_element_stride + 41 * tangent_component_stride]);
        const scalar_t tangent42 = scalar_t(tangent[element * tangent_element_stride + 42 * tangent_component_stride]);
        const scalar_t tangent43 = scalar_t(tangent[element * tangent_element_stride + 43 * tangent_component_stride]);
        const scalar_t tangent44 = scalar_t(tangent[element * tangent_element_stride + 44 * tangent_component_stride]);
        const scalar_t compressed_increment_t0 = ((scalar_t(1) / scalar_t(6)))*hx_0;
        const scalar_t compressed_increment_t1 = ((scalar_t(1) / scalar_t(6)))*hy_0;
        const scalar_t compressed_increment_t2 = ((scalar_t(1) / scalar_t(6)))*hz_0;
        const scalar_t pa_p0_0_0 = compressed_increment_t0 - (scalar_t(1) / scalar_t(6))*hx_1;
        const scalar_t pa_p0_0_1 = compressed_increment_t0 - (scalar_t(1) / scalar_t(6))*hx_2;
        const scalar_t pa_p0_0_2 = compressed_increment_t0 - (scalar_t(1) / scalar_t(6))*hx_3;
        const scalar_t pa_p1_0_0 = compressed_increment_t1 - (scalar_t(1) / scalar_t(6))*hy_1;
        const scalar_t pa_p1_0_1 = compressed_increment_t1 - (scalar_t(1) / scalar_t(6))*hy_2;
        const scalar_t pa_p1_0_2 = compressed_increment_t1 - (scalar_t(1) / scalar_t(6))*hy_3;
        const scalar_t pa_p2_0_0 = compressed_increment_t2 - (scalar_t(1) / scalar_t(6))*hz_1;
        const scalar_t pa_p2_0_1 = compressed_increment_t2 - (scalar_t(1) / scalar_t(6))*hz_2;
        const scalar_t pa_p2_0_2 = compressed_increment_t2 - (scalar_t(1) / scalar_t(6))*hz_3;
        const scalar_t pa_y0_0_0 = pa_p0_0_0*tangent0 + pa_p0_0_1*tangent1 + pa_p0_0_2*tangent2 + pa_p1_0_0*tangent3 + pa_p1_0_1*tangent4 + pa_p1_0_2*tangent5 + pa_p2_0_0*tangent6 + pa_p2_0_1*tangent7 + pa_p2_0_2*tangent8;
        const scalar_t pa_y0_0_1 = pa_p0_0_0*tangent1 + pa_p0_0_1*tangent9 + pa_p0_0_2*tangent10 + pa_p1_0_0*tangent11 + pa_p1_0_1*tangent12 + pa_p1_0_2*tangent13 + pa_p2_0_0*tangent14 + pa_p2_0_1*tangent15 + pa_p2_0_2*tangent16;
        const scalar_t pa_y0_0_2 = pa_p0_0_0*tangent2 + pa_p0_0_1*tangent10 + pa_p0_0_2*tangent17 + pa_p1_0_0*tangent18 + pa_p1_0_1*tangent19 + pa_p1_0_2*tangent20 + pa_p2_0_0*tangent21 + pa_p2_0_1*tangent22 + pa_p2_0_2*tangent23;
        const scalar_t pa_y1_0_0 = pa_p0_0_0*tangent3 + pa_p0_0_1*tangent11 + pa_p0_0_2*tangent18 + pa_p1_0_0*tangent24 + pa_p1_0_1*tangent25 + pa_p1_0_2*tangent26 + pa_p2_0_0*tangent27 + pa_p2_0_1*tangent28 + pa_p2_0_2*tangent29;
        const scalar_t pa_y1_0_1 = pa_p0_0_0*tangent4 + pa_p0_0_1*tangent12 + pa_p0_0_2*tangent19 + pa_p1_0_0*tangent25 + pa_p1_0_1*tangent30 + pa_p1_0_2*tangent31 + pa_p2_0_0*tangent32 + pa_p2_0_1*tangent33 + pa_p2_0_2*tangent34;
        const scalar_t pa_y1_0_2 = pa_p0_0_0*tangent5 + pa_p0_0_1*tangent13 + pa_p0_0_2*tangent20 + pa_p1_0_0*tangent26 + pa_p1_0_1*tangent31 + pa_p1_0_2*tangent35 + pa_p2_0_0*tangent36 + pa_p2_0_1*tangent37 + pa_p2_0_2*tangent38;
        const scalar_t pa_y2_0_0 = pa_p0_0_0*tangent6 + pa_p0_0_1*tangent14 + pa_p0_0_2*tangent21 + pa_p1_0_0*tangent27 + pa_p1_0_1*tangent32 + pa_p1_0_2*tangent36 + pa_p2_0_0*tangent39 + pa_p2_0_1*tangent40 + pa_p2_0_2*tangent41;
        const scalar_t pa_y2_0_1 = pa_p0_0_0*tangent7 + pa_p0_0_1*tangent15 + pa_p0_0_2*tangent22 + pa_p1_0_0*tangent28 + pa_p1_0_1*tangent33 + pa_p1_0_2*tangent37 + pa_p2_0_0*tangent40 + pa_p2_0_1*tangent42 + pa_p2_0_2*tangent43;
        const scalar_t pa_y2_0_2 = pa_p0_0_0*tangent8 + pa_p0_0_1*tangent16 + pa_p0_0_2*tangent23 + pa_p1_0_0*tangent29 + pa_p1_0_1*tangent34 + pa_p1_0_2*tangent38 + pa_p2_0_0*tangent41 + pa_p2_0_1*tangent43 + pa_p2_0_2*tangent44;
        const scalar_t pa_q0_0_0 = scalar_t(6)*pa_y0_0_0;
        const scalar_t pa_q0_0_1 = scalar_t(6)*pa_y0_0_1;
        const scalar_t pa_q0_0_2 = scalar_t(6)*pa_y0_0_2;
        const scalar_t pa_q1_0_0 = scalar_t(6)*pa_y1_0_0;
        const scalar_t pa_q1_0_1 = scalar_t(6)*pa_y1_0_1;
        const scalar_t pa_q1_0_2 = scalar_t(6)*pa_y1_0_2;
        const scalar_t pa_q2_0_0 = scalar_t(6)*pa_y2_0_0;
        const scalar_t pa_q2_0_1 = scalar_t(6)*pa_y2_0_1;
        const scalar_t pa_q2_0_2 = scalar_t(6)*pa_y2_0_2;
        const scalar_t output_t0 = ((scalar_t(1) / scalar_t(6)))*pa_q0_0_0;
        const scalar_t output_t1 = ((scalar_t(1) / scalar_t(6)))*pa_q0_0_1;
        const scalar_t output_t2 = ((scalar_t(1) / scalar_t(6)))*pa_q0_0_2;
        const scalar_t output_t3 = ((scalar_t(1) / scalar_t(6)))*pa_q1_0_0;
        const scalar_t output_t4 = ((scalar_t(1) / scalar_t(6)))*pa_q1_0_1;
        const scalar_t output_t5 = ((scalar_t(1) / scalar_t(6)))*pa_q1_0_2;
        const scalar_t output_t6 = ((scalar_t(1) / scalar_t(6)))*pa_q2_0_0;
        const scalar_t output_t7 = ((scalar_t(1) / scalar_t(6)))*pa_q2_0_1;
        const scalar_t output_t8 = ((scalar_t(1) / scalar_t(6)))*pa_q2_0_2;
        const scalar_t element_out0_0 = output_t0 + output_t1 + output_t2;
        const scalar_t element_out0_1 = -output_t0;
        const scalar_t element_out0_2 = -output_t1;
        const scalar_t element_out0_3 = -output_t2;
        const scalar_t element_out1_0 = output_t3 + output_t4 + output_t5;
        const scalar_t element_out1_1 = -output_t3;
        const scalar_t element_out1_2 = -output_t4;
        const scalar_t element_out1_3 = -output_t5;
        const scalar_t element_out2_0 = output_t6 + output_t7 + output_t8;
        const scalar_t element_out2_1 = -output_t6;
        const scalar_t element_out2_2 = -output_t7;
        const scalar_t element_out2_3 = -output_t8;
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
