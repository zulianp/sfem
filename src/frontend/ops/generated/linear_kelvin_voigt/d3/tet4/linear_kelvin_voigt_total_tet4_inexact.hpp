#ifndef SFEM_CODEGEN_LINEAR_KELVIN_VOIGT_TOTAL_TET4_INEXACT_HPP
#define SFEM_CODEGEN_LINEAR_KELVIN_VOIGT_TOTAL_TET4_INEXACT_HPP

#include "kernel_math.hpp"
#include "inexact_tet4_q1.hpp"
#include "inexact_contract_d3.hpp"

namespace sfem {
namespace codegen {

// linear_kelvin_voigt_total on TET4: the weighted tangent the stored-tangent operator keeps at every point.
struct linear_kelvin_voigt_total_tet4_inexact {
  using element = inexact_tet4_q1;
  using contraction = inexact_contract_d3<true>;
  static constexpr int DIM = 3;
  static constexpr int NC = 45;
  static constexpr bool symmetric = true;
  static constexpr bool reads_state = false;
  static constexpr bool reads_previous = false;
  static constexpr int n_parameters = 5;
  // The integration-point fields the tangent reads, and their components.
  static constexpr int n_quadrature_fields = 0;
  static constexpr int max_quadrature_components = 1;

  // The tangent at one point, unweighted: `gu_ref` (and `gz_ref`, the previous state's) is the
  // reference gradient there, `adjugate` and `determinant` the geometry's.
  template <typename s_t>
  static SFEM_INLINE void point(const s_t *const RSTR parameters,
                    const s_t *const RSTR /*gu_ref*/,
                    const s_t *const RSTR /*gz_ref*/,
                    const s_t *const RSTR adjugate,
                    const s_t determinant,
                    const s_t *const *const RSTR /*quadrature*/,
                    s_t *const RSTR tangent) {
    const s_t eta_b = parameters[0];
    const s_t eta_s = parameters[1];
    const s_t lmbda = parameters[2];
    const s_t mu = parameters[3];
    const s_t u_dt_shift = parameters[4];
    const s_t adjugate0 = adjugate[0];
    const s_t adjugate1 = adjugate[1];
    const s_t adjugate2 = adjugate[2];
    const s_t adjugate3 = adjugate[3];
    const s_t adjugate4 = adjugate[4];
    const s_t adjugate5 = adjugate[5];
    const s_t adjugate6 = adjugate[6];
    const s_t adjugate7 = adjugate[7];
    const s_t adjugate8 = adjugate[8];
    const s_t tangent_t0 = pow_m1(determinant);
    const s_t tangent_t1 = pow_2(adjugate1);
    const s_t tangent_t2 = eta_s*u_dt_shift;
    const s_t tangent_t3 = mu + tangent_t2;
    const s_t tangent_t4 = tangent_t1*tangent_t3;
    const s_t tangent_t5 = pow_2(adjugate2);
    const s_t tangent_t6 = tangent_t3*tangent_t5;
    const s_t tangent_t7 = pow_2(adjugate0);
    const s_t tangent_t8 = eta_b*u_dt_shift + lmbda;
    const s_t tangent_t9 = s_t(2)*mu + ((s_t(4) / s_t(3)))*tangent_t2 + tangent_t8;
    const s_t tangent_t10 = adjugate1*tangent_t3;
    const s_t tangent_t11 = adjugate4*tangent_t10;
    const s_t tangent_t12 = adjugate2*tangent_t3;
    const s_t tangent_t13 = adjugate5*tangent_t12;
    const s_t tangent_t14 = adjugate0*tangent_t9;
    const s_t tangent_t15 = adjugate7*tangent_t10;
    const s_t tangent_t16 = adjugate8*tangent_t12;
    const s_t tangent_t17 = -(s_t(2) / s_t(3))*tangent_t2 + tangent_t8;
    const s_t tangent_t18 = adjugate0*tangent_t17;
    const s_t tangent_t19 = pow_2(adjugate4);
    const s_t tangent_t20 = tangent_t19*tangent_t3;
    const s_t tangent_t21 = pow_2(adjugate5);
    const s_t tangent_t22 = tangent_t21*tangent_t3;
    const s_t tangent_t23 = pow_2(adjugate3);
    const s_t tangent_t24 = adjugate4*tangent_t3;
    const s_t tangent_t25 = adjugate7*tangent_t24;
    const s_t tangent_t26 = adjugate5*tangent_t3;
    const s_t tangent_t27 = adjugate8*tangent_t26;
    const s_t tangent_t28 = adjugate3*adjugate6;
    const s_t tangent_t29 = adjugate3*tangent_t17;
    const s_t tangent_t30 = pow_2(adjugate7);
    const s_t tangent_t31 = tangent_t3*tangent_t30;
    const s_t tangent_t32 = pow_2(adjugate8);
    const s_t tangent_t33 = tangent_t3*tangent_t32;
    const s_t tangent_t34 = pow_2(adjugate6);
    const s_t tangent_t35 = adjugate7*tangent_t3;
    const s_t tangent_t36 = adjugate6*tangent_t17;
    const s_t tangent_t37 = adjugate8*tangent_t3;
    const s_t tangent_t38 = tangent_t3*tangent_t7;
    const s_t tangent_t39 = adjugate0*tangent_t3;
    const s_t tangent_t40 = adjugate3*tangent_t39;
    const s_t tangent_t41 = adjugate1*tangent_t9;
    const s_t tangent_t42 = adjugate6*tangent_t39;
    const s_t tangent_t43 = adjugate1*tangent_t17;
    const s_t tangent_t44 = tangent_t23*tangent_t3;
    const s_t tangent_t45 = tangent_t28*tangent_t3;
    const s_t tangent_t46 = adjugate4*tangent_t17;
    const s_t tangent_t47 = tangent_t3*tangent_t34;
    const s_t tangent_t48 = adjugate7*tangent_t17;
    const s_t tangent_t49 = adjugate2*tangent_t9;
    const s_t tangent0 = tangent_t0*(tangent_t4 + tangent_t6 + tangent_t7*tangent_t9);
    const s_t tangent1 = tangent_t0*(adjugate3*tangent_t14 + tangent_t11 + tangent_t13);
    const s_t tangent2 = tangent_t0*(adjugate6*tangent_t14 + tangent_t15 + tangent_t16);
    const s_t tangent3 = tangent_t0*(adjugate0*tangent_t10 + adjugate1*tangent_t18);
    const s_t tangent4 = tangent_t0*(adjugate3*tangent_t10 + adjugate4*tangent_t18);
    const s_t tangent5 = tangent_t0*(adjugate6*tangent_t10 + adjugate7*tangent_t18);
    const s_t tangent6 = tangent_t0*(adjugate0*tangent_t12 + adjugate2*tangent_t18);
    const s_t tangent7 = tangent_t0*(adjugate3*tangent_t12 + adjugate5*tangent_t18);
    const s_t tangent8 = tangent_t0*(adjugate6*tangent_t12 + adjugate8*tangent_t18);
    const s_t tangent9 = tangent_t0*(tangent_t20 + tangent_t22 + tangent_t23*tangent_t9);
    const s_t tangent10 = tangent_t0*(tangent_t25 + tangent_t27 + tangent_t28*tangent_t9);
    const s_t tangent11 = tangent_t0*(adjugate0*tangent_t24 + adjugate1*tangent_t29);
    const s_t tangent12 = tangent_t0*(adjugate3*tangent_t24 + adjugate4*tangent_t29);
    const s_t tangent13 = tangent_t0*(adjugate6*tangent_t24 + adjugate7*tangent_t29);
    const s_t tangent14 = tangent_t0*(adjugate0*tangent_t26 + adjugate2*tangent_t29);
    const s_t tangent15 = tangent_t0*(adjugate3*tangent_t26 + adjugate5*tangent_t29);
    const s_t tangent16 = tangent_t0*(adjugate6*tangent_t26 + adjugate8*tangent_t29);
    const s_t tangent17 = tangent_t0*(tangent_t31 + tangent_t33 + tangent_t34*tangent_t9);
    const s_t tangent18 = tangent_t0*(adjugate0*tangent_t35 + adjugate1*tangent_t36);
    const s_t tangent19 = tangent_t0*(adjugate3*tangent_t35 + adjugate4*tangent_t36);
    const s_t tangent20 = tangent_t0*(adjugate6*tangent_t35 + adjugate7*tangent_t36);
    const s_t tangent21 = tangent_t0*(adjugate0*tangent_t37 + adjugate2*tangent_t36);
    const s_t tangent22 = tangent_t0*(adjugate3*tangent_t37 + adjugate5*tangent_t36);
    const s_t tangent23 = tangent_t0*(adjugate6*tangent_t37 + adjugate8*tangent_t36);
    const s_t tangent24 = tangent_t0*(tangent_t1*tangent_t9 + tangent_t38 + tangent_t6);
    const s_t tangent25 = tangent_t0*(adjugate4*tangent_t41 + tangent_t13 + tangent_t40);
    const s_t tangent26 = tangent_t0*(adjugate7*tangent_t41 + tangent_t16 + tangent_t42);
    const s_t tangent27 = tangent_t0*(adjugate2*tangent_t10 + adjugate2*tangent_t43);
    const s_t tangent28 = tangent_t0*(adjugate4*tangent_t12 + adjugate5*tangent_t43);
    const s_t tangent29 = tangent_t0*(adjugate7*tangent_t12 + adjugate8*tangent_t43);
    const s_t tangent30 = tangent_t0*(tangent_t19*tangent_t9 + tangent_t22 + tangent_t44);
    const s_t tangent31 = tangent_t0*(adjugate4*adjugate7*tangent_t9 + tangent_t27 + tangent_t45);
    const s_t tangent32 = tangent_t0*(adjugate2*tangent_t46 + adjugate5*tangent_t10);
    const s_t tangent33 = tangent_t0*(adjugate5*tangent_t24 + adjugate5*tangent_t46);
    const s_t tangent34 = tangent_t0*(adjugate7*tangent_t26 + adjugate8*tangent_t46);
    const s_t tangent35 = tangent_t0*(tangent_t30*tangent_t9 + tangent_t33 + tangent_t47);
    const s_t tangent36 = tangent_t0*(adjugate2*tangent_t48 + adjugate8*tangent_t10);
    const s_t tangent37 = tangent_t0*(adjugate5*tangent_t48 + adjugate8*tangent_t24);
    const s_t tangent38 = tangent_t0*(adjugate8*tangent_t35 + adjugate8*tangent_t48);
    const s_t tangent39 = tangent_t0*(tangent_t38 + tangent_t4 + tangent_t5*tangent_t9);
    const s_t tangent40 = tangent_t0*(adjugate5*tangent_t49 + tangent_t11 + tangent_t40);
    const s_t tangent41 = tangent_t0*(adjugate8*tangent_t49 + tangent_t15 + tangent_t42);
    const s_t tangent42 = tangent_t0*(tangent_t20 + tangent_t21*tangent_t9 + tangent_t44);
    const s_t tangent43 = tangent_t0*(adjugate5*adjugate8*tangent_t9 + tangent_t25 + tangent_t45);
    const s_t tangent44 = tangent_t0*(tangent_t31 + tangent_t32*tangent_t9 + tangent_t47);
    tangent[0] = tangent0;
    tangent[1] = tangent1;
    tangent[2] = tangent2;
    tangent[3] = tangent3;
    tangent[4] = tangent4;
    tangent[5] = tangent5;
    tangent[6] = tangent6;
    tangent[7] = tangent7;
    tangent[8] = tangent8;
    tangent[9] = tangent9;
    tangent[10] = tangent10;
    tangent[11] = tangent11;
    tangent[12] = tangent12;
    tangent[13] = tangent13;
    tangent[14] = tangent14;
    tangent[15] = tangent15;
    tangent[16] = tangent16;
    tangent[17] = tangent17;
    tangent[18] = tangent18;
    tangent[19] = tangent19;
    tangent[20] = tangent20;
    tangent[21] = tangent21;
    tangent[22] = tangent22;
    tangent[23] = tangent23;
    tangent[24] = tangent24;
    tangent[25] = tangent25;
    tangent[26] = tangent26;
    tangent[27] = tangent27;
    tangent[28] = tangent28;
    tangent[29] = tangent29;
    tangent[30] = tangent30;
    tangent[31] = tangent31;
    tangent[32] = tangent32;
    tangent[33] = tangent33;
    tangent[34] = tangent34;
    tangent[35] = tangent35;
    tangent[36] = tangent36;
    tangent[37] = tangent37;
    tangent[38] = tangent38;
    tangent[39] = tangent39;
    tangent[40] = tangent40;
    tangent[41] = tangent41;
    tangent[42] = tangent42;
    tangent[43] = tangent43;
    tangent[44] = tangent44;
  }
};

}  // namespace codegen
}  // namespace sfem

#endif
