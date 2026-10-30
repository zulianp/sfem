#ifndef SFEM_CODEGEN_LINEAR_KELVIN_VOIGT_TOTAL_TET10_INEXACT_HPP
#define SFEM_CODEGEN_LINEAR_KELVIN_VOIGT_TOTAL_TET10_INEXACT_HPP

#include "kernel_math.hpp"
#include "inexact_tet10_q11.hpp"
#include "inexact_contract_d3.hpp"

namespace sfem {
namespace codegen {

// linear_kelvin_voigt_total on TET10: the weighted tangent the stored-tangent operator keeps at every point.
struct linear_kelvin_voigt_total_tet10_inexact {
  using element = inexact_tet10_q11;
  using contraction = inexact_contract_d3<true>;
  static constexpr int DIM = 3;
  static constexpr int NC = 45;
  static constexpr bool symmetric = true;
  static constexpr bool reads_state = false;
  static constexpr bool reads_previous = false;
  static constexpr int n_parameters = 5;
  // The integration-point fields the tangent reads, and their components.
  static constexpr int n_quadrature_fields = 0;
  static constexpr int quadrature_components[1] = {0};
  static constexpr int max_quadrature_components = 1;

  // The tangent at one point, unweighted: `gu_ref` (and `gz_ref`, the previous state's) is the
  // reference gradient there, `adjugate` and `determinant` the geometry's.
  template <typename s_t>
  static SFEM_INLINE void point(const s_t *const RSTR parameters,
                    const s_t *const RSTR gu_ref,
                    const s_t *const RSTR gz_ref,
                    const s_t *const RSTR adjugate,
                    const s_t determinant,
                    const s_t *const *const RSTR quadrature,
                    s_t *const RSTR tangent) {
    (void)quadrature;
    (void)gu_ref;
    (void)gz_ref;
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
    const s_t integrand_t0 = pow_m1(determinant);
    const s_t integrand_t1 = pow_2(adjugate1);
    const s_t integrand_t2 = eta_s*u_dt_shift;
    const s_t integrand_t3 = integrand_t2 + mu;
    const s_t integrand_t4 = integrand_t1*integrand_t3;
    const s_t integrand_t5 = pow_2(adjugate2);
    const s_t integrand_t6 = integrand_t3*integrand_t5;
    const s_t integrand_t7 = pow_2(adjugate0);
    const s_t integrand_t8 = eta_b*u_dt_shift + lmbda;
    const s_t integrand_t9 = ((s_t(4) / s_t(3)))*integrand_t2 + integrand_t8 + s_t(2)*mu;
    const s_t integrand_t10 = adjugate1*integrand_t3;
    const s_t integrand_t11 = adjugate4*integrand_t10;
    const s_t integrand_t12 = adjugate2*integrand_t3;
    const s_t integrand_t13 = adjugate5*integrand_t12;
    const s_t integrand_t14 = adjugate0*integrand_t9;
    const s_t integrand_t15 = adjugate7*integrand_t10;
    const s_t integrand_t16 = adjugate8*integrand_t12;
    const s_t integrand_t17 = -(s_t(2) / s_t(3))*integrand_t2 + integrand_t8;
    const s_t integrand_t18 = adjugate0*integrand_t17;
    const s_t integrand_t19 = pow_2(adjugate4);
    const s_t integrand_t20 = integrand_t19*integrand_t3;
    const s_t integrand_t21 = pow_2(adjugate5);
    const s_t integrand_t22 = integrand_t21*integrand_t3;
    const s_t integrand_t23 = pow_2(adjugate3);
    const s_t integrand_t24 = adjugate4*integrand_t3;
    const s_t integrand_t25 = adjugate7*integrand_t24;
    const s_t integrand_t26 = adjugate5*integrand_t3;
    const s_t integrand_t27 = adjugate8*integrand_t26;
    const s_t integrand_t28 = adjugate3*adjugate6;
    const s_t integrand_t29 = adjugate3*integrand_t17;
    const s_t integrand_t30 = pow_2(adjugate7);
    const s_t integrand_t31 = integrand_t3*integrand_t30;
    const s_t integrand_t32 = pow_2(adjugate8);
    const s_t integrand_t33 = integrand_t3*integrand_t32;
    const s_t integrand_t34 = pow_2(adjugate6);
    const s_t integrand_t35 = adjugate7*integrand_t3;
    const s_t integrand_t36 = adjugate6*integrand_t17;
    const s_t integrand_t37 = adjugate8*integrand_t3;
    const s_t integrand_t38 = integrand_t3*integrand_t7;
    const s_t integrand_t39 = adjugate0*integrand_t3;
    const s_t integrand_t40 = adjugate3*integrand_t39;
    const s_t integrand_t41 = adjugate1*integrand_t9;
    const s_t integrand_t42 = adjugate6*integrand_t39;
    const s_t integrand_t43 = adjugate1*integrand_t17;
    const s_t integrand_t44 = integrand_t23*integrand_t3;
    const s_t integrand_t45 = integrand_t28*integrand_t3;
    const s_t integrand_t46 = adjugate4*integrand_t17;
    const s_t integrand_t47 = integrand_t3*integrand_t34;
    const s_t integrand_t48 = adjugate7*integrand_t17;
    const s_t integrand_t49 = adjugate2*integrand_t9;
    const s_t integrand0 = integrand_t0*(integrand_t4 + integrand_t6 + integrand_t7*integrand_t9);
    const s_t integrand1 = integrand_t0*(adjugate3*integrand_t14 + integrand_t11 + integrand_t13);
    const s_t integrand2 = integrand_t0*(adjugate6*integrand_t14 + integrand_t15 + integrand_t16);
    const s_t integrand3 = integrand_t0*(adjugate0*integrand_t10 + adjugate1*integrand_t18);
    const s_t integrand4 = integrand_t0*(adjugate3*integrand_t10 + adjugate4*integrand_t18);
    const s_t integrand5 = integrand_t0*(adjugate6*integrand_t10 + adjugate7*integrand_t18);
    const s_t integrand6 = integrand_t0*(adjugate0*integrand_t12 + adjugate2*integrand_t18);
    const s_t integrand7 = integrand_t0*(adjugate3*integrand_t12 + adjugate5*integrand_t18);
    const s_t integrand8 = integrand_t0*(adjugate6*integrand_t12 + adjugate8*integrand_t18);
    const s_t integrand9 = integrand_t0*(integrand_t20 + integrand_t22 + integrand_t23*integrand_t9);
    const s_t integrand10 = integrand_t0*(integrand_t25 + integrand_t27 + integrand_t28*integrand_t9);
    const s_t integrand11 = integrand_t0*(adjugate0*integrand_t24 + adjugate1*integrand_t29);
    const s_t integrand12 = integrand_t0*(adjugate3*integrand_t24 + adjugate4*integrand_t29);
    const s_t integrand13 = integrand_t0*(adjugate6*integrand_t24 + adjugate7*integrand_t29);
    const s_t integrand14 = integrand_t0*(adjugate0*integrand_t26 + adjugate2*integrand_t29);
    const s_t integrand15 = integrand_t0*(adjugate3*integrand_t26 + adjugate5*integrand_t29);
    const s_t integrand16 = integrand_t0*(adjugate6*integrand_t26 + adjugate8*integrand_t29);
    const s_t integrand17 = integrand_t0*(integrand_t31 + integrand_t33 + integrand_t34*integrand_t9);
    const s_t integrand18 = integrand_t0*(adjugate0*integrand_t35 + adjugate1*integrand_t36);
    const s_t integrand19 = integrand_t0*(adjugate3*integrand_t35 + adjugate4*integrand_t36);
    const s_t integrand20 = integrand_t0*(adjugate6*integrand_t35 + adjugate7*integrand_t36);
    const s_t integrand21 = integrand_t0*(adjugate0*integrand_t37 + adjugate2*integrand_t36);
    const s_t integrand22 = integrand_t0*(adjugate3*integrand_t37 + adjugate5*integrand_t36);
    const s_t integrand23 = integrand_t0*(adjugate6*integrand_t37 + adjugate8*integrand_t36);
    const s_t integrand24 = integrand_t0*(integrand_t1*integrand_t9 + integrand_t38 + integrand_t6);
    const s_t integrand25 = integrand_t0*(adjugate4*integrand_t41 + integrand_t13 + integrand_t40);
    const s_t integrand26 = integrand_t0*(adjugate7*integrand_t41 + integrand_t16 + integrand_t42);
    const s_t integrand27 = integrand_t0*(adjugate2*integrand_t10 + adjugate2*integrand_t43);
    const s_t integrand28 = integrand_t0*(adjugate4*integrand_t12 + adjugate5*integrand_t43);
    const s_t integrand29 = integrand_t0*(adjugate7*integrand_t12 + adjugate8*integrand_t43);
    const s_t integrand30 = integrand_t0*(integrand_t19*integrand_t9 + integrand_t22 + integrand_t44);
    const s_t integrand31 = integrand_t0*(adjugate4*adjugate7*integrand_t9 + integrand_t27 + integrand_t45);
    const s_t integrand32 = integrand_t0*(adjugate2*integrand_t46 + adjugate5*integrand_t10);
    const s_t integrand33 = integrand_t0*(adjugate5*integrand_t24 + adjugate5*integrand_t46);
    const s_t integrand34 = integrand_t0*(adjugate7*integrand_t26 + adjugate8*integrand_t46);
    const s_t integrand35 = integrand_t0*(integrand_t30*integrand_t9 + integrand_t33 + integrand_t47);
    const s_t integrand36 = integrand_t0*(adjugate2*integrand_t48 + adjugate8*integrand_t10);
    const s_t integrand37 = integrand_t0*(adjugate5*integrand_t48 + adjugate8*integrand_t24);
    const s_t integrand38 = integrand_t0*(adjugate8*integrand_t35 + adjugate8*integrand_t48);
    const s_t integrand39 = integrand_t0*(integrand_t38 + integrand_t4 + integrand_t5*integrand_t9);
    const s_t integrand40 = integrand_t0*(adjugate5*integrand_t49 + integrand_t11 + integrand_t40);
    const s_t integrand41 = integrand_t0*(adjugate8*integrand_t49 + integrand_t15 + integrand_t42);
    const s_t integrand42 = integrand_t0*(integrand_t20 + integrand_t21*integrand_t9 + integrand_t44);
    const s_t integrand43 = integrand_t0*(adjugate5*adjugate8*integrand_t9 + integrand_t25 + integrand_t45);
    const s_t integrand44 = integrand_t0*(integrand_t31 + integrand_t32*integrand_t9 + integrand_t47);
    tangent[0] = integrand0;
    tangent[1] = integrand1;
    tangent[2] = integrand2;
    tangent[3] = integrand3;
    tangent[4] = integrand4;
    tangent[5] = integrand5;
    tangent[6] = integrand6;
    tangent[7] = integrand7;
    tangent[8] = integrand8;
    tangent[9] = integrand9;
    tangent[10] = integrand10;
    tangent[11] = integrand11;
    tangent[12] = integrand12;
    tangent[13] = integrand13;
    tangent[14] = integrand14;
    tangent[15] = integrand15;
    tangent[16] = integrand16;
    tangent[17] = integrand17;
    tangent[18] = integrand18;
    tangent[19] = integrand19;
    tangent[20] = integrand20;
    tangent[21] = integrand21;
    tangent[22] = integrand22;
    tangent[23] = integrand23;
    tangent[24] = integrand24;
    tangent[25] = integrand25;
    tangent[26] = integrand26;
    tangent[27] = integrand27;
    tangent[28] = integrand28;
    tangent[29] = integrand29;
    tangent[30] = integrand30;
    tangent[31] = integrand31;
    tangent[32] = integrand32;
    tangent[33] = integrand33;
    tangent[34] = integrand34;
    tangent[35] = integrand35;
    tangent[36] = integrand36;
    tangent[37] = integrand37;
    tangent[38] = integrand38;
    tangent[39] = integrand39;
    tangent[40] = integrand40;
    tangent[41] = integrand41;
    tangent[42] = integrand42;
    tangent[43] = integrand43;
    tangent[44] = integrand44;
  }
};

}  // namespace codegen
}  // namespace sfem

#endif
