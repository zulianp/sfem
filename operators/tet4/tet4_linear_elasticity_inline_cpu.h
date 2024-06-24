#ifndef TET4_LINEAR_ELASTICITY_INLINE_CPU_H
#define TET4_LINEAR_ELASTICITY_INLINE_CPU_H

static SFEM_INLINE void tet4_gradient_3(const jacobian_t *const SFEM_RESTRICT adjugate,
                                        const jacobian_t jacobian_determinant,
                                        const scalar_t *const SFEM_RESTRICT ux,
                                        const scalar_t *const SFEM_RESTRICT uy,
                                        const scalar_t *const SFEM_RESTRICT uz,
                                        scalar_t *const SFEM_RESTRICT disp_grad) {
    const real_t x0 = 1.0 / jacobian_determinant;
    const real_t x1 = ux[0] - ux[1];
    const real_t x2 = ux[0] - ux[2];
    const real_t x3 = ux[0] - ux[3];
    const real_t x4 = uy[0] - uy[1];
    const real_t x5 = uy[0] - uy[2];
    const real_t x6 = uy[0] - uy[3];
    const real_t x7 = uz[0] - uz[1];
    const real_t x8 = uz[0] - uz[2];
    const real_t x9 = uz[0] - uz[3];
    disp_grad[0] = x0 * (-adjugate[0] * x1 - adjugate[3] * x2 - adjugate[6] * x3);
    disp_grad[1] = x0 * (-adjugate[1] * x1 - adjugate[4] * x2 - adjugate[7] * x3);
    disp_grad[2] = x0 * (-adjugate[2] * x1 - adjugate[5] * x2 - adjugate[8] * x3);
    disp_grad[3] = x0 * (-adjugate[0] * x4 - adjugate[3] * x5 - adjugate[6] * x6);
    disp_grad[4] = x0 * (-adjugate[1] * x4 - adjugate[4] * x5 - adjugate[7] * x6);
    disp_grad[5] = x0 * (-adjugate[2] * x4 - adjugate[5] * x5 - adjugate[8] * x6);
    disp_grad[6] = x0 * (-adjugate[0] * x7 - adjugate[3] * x8 - adjugate[6] * x9);
    disp_grad[7] = x0 * (-adjugate[1] * x7 - adjugate[4] * x8 - adjugate[7] * x9);
    disp_grad[8] = x0 * (-adjugate[2] * x7 - adjugate[5] * x8 - adjugate[8] * x9);
}

static SFEM_INLINE void tet4_linear_elasticity_loperand(
        const scalar_t mu,
        const scalar_t lambda,
        const jacobian_t *const SFEM_RESTRICT adjugate,
        scalar_t *const SFEM_RESTRICT in_disp_grad_out_P_tXJinv_t) {
    // Shorter name
    real_t *const buff = in_disp_grad_out_P_tXJinv_t;

    const real_t x0 = (1.0 / 6.0) * mu;
    const real_t x1 = x0 * (buff[1] + buff[3]);
    const real_t x2 = x0 * (buff[2] + buff[6]);
    const real_t x3 = 2 * mu;
    const real_t x4 = lambda * (buff[0] + buff[4] + buff[8]);
    const real_t x5 = (1.0 / 6.0) * buff[0] * x3 + (1.0 / 6.0) * x4;
    const real_t x6 = x0 * (buff[5] + buff[7]);
    const real_t x7 = (1.0 / 6.0) * buff[4] * x3 + (1.0 / 6.0) * x4;
    const real_t x8 = (1.0 / 6.0) * buff[8] * x3 + (1.0 / 6.0) * x4;
    buff[0] = adjugate[0] * x5 + adjugate[1] * x1 + adjugate[2] * x2;
    buff[1] = adjugate[3] * x5 + adjugate[4] * x1 + adjugate[5] * x2;
    buff[2] = adjugate[6] * x5 + adjugate[7] * x1 + adjugate[8] * x2;
    buff[3] = adjugate[0] * x1 + adjugate[1] * x7 + adjugate[2] * x6;
    buff[4] = adjugate[3] * x1 + adjugate[4] * x7 + adjugate[5] * x6;
    buff[5] = adjugate[6] * x1 + adjugate[7] * x7 + adjugate[8] * x6;
    buff[6] = adjugate[0] * x2 + adjugate[1] * x6 + adjugate[2] * x8;
    buff[7] = adjugate[3] * x2 + adjugate[4] * x6 + adjugate[5] * x8;
    buff[8] = adjugate[6] * x2 + adjugate[7] * x6 + adjugate[8] * x8;
}

static SFEM_INLINE void tet4_linear_elasticity_apply_adj(const scalar_t mu,
                                                         const scalar_t lambda,
                                                         const jacobian_t *const SFEM_RESTRICT
                                                                 adjugate,
                                                         const jacobian_t jacobian_determinant,
                                                         const scalar_t *const SFEM_RESTRICT ux,
                                                         const scalar_t *const SFEM_RESTRICT uy,
                                                         const scalar_t *const SFEM_RESTRICT uz,
                                                         accumulator_t *const SFEM_RESTRICT outx,
                                                         accumulator_t *const SFEM_RESTRICT outy,
                                                         accumulator_t *const SFEM_RESTRICT outz) {
    scalar_t P_tXJinv_t[9];
    tet4_gradient_3(adjugate, jacobian_determinant, ux, uy, uz, P_tXJinv_t);
    tet4_linear_elasticity_loperand(mu, lambda, adjugate, P_tXJinv_t);

    outx[0] = -P_tXJinv_t[0] - P_tXJinv_t[1] - P_tXJinv_t[2];
    outx[1] = P_tXJinv_t[0];
    outx[2] = P_tXJinv_t[1];
    outx[3] = P_tXJinv_t[2];
    outy[0] = -P_tXJinv_t[3] - P_tXJinv_t[4] - P_tXJinv_t[5];
    outy[1] = P_tXJinv_t[3];
    outy[2] = P_tXJinv_t[4];
    outy[3] = P_tXJinv_t[5];
    outz[0] = -P_tXJinv_t[6] - P_tXJinv_t[7] - P_tXJinv_t[8];
    outz[1] = P_tXJinv_t[6];
    outz[2] = P_tXJinv_t[7];
    outz[3] = P_tXJinv_t[8];
}

static SFEM_INLINE void tet4_linear_elasticity_diag_adj(const scalar_t mu,
                                                        const scalar_t lambda,
                                                        const jacobian_t *const SFEM_RESTRICT
                                                                jacobian_adjugate,
                                                        const jacobian_t jacobian_determinant,
                                                        accumulator_t *const SFEM_RESTRICT diagx,
                                                        accumulator_t *const SFEM_RESTRICT diagy,
                                                        accumulator_t *const SFEM_RESTRICT diagz) {
    const real_t x0 = lambda + 2 * mu;
    const real_t x1 = adjugate[0] + adjugate[3] + adjugate[6];
    const real_t x2 = x0 * x1;
    const real_t x3 = adjugate[2] + adjugate[5] + adjugate[8];
    const real_t x4 = mu * x3;
    const real_t x5 = adjugate[2] * x4 + adjugate[5] * x4 + adjugate[8] * x4;
    const real_t x6 = adjugate[1] + adjugate[4] + adjugate[7];
    const real_t x7 = mu * x6;
    const real_t x8 = adjugate[1] * x7 + adjugate[4] * x7 + adjugate[7] * x7;
    const real_t x9 = (1.0 / 6.0) * qw / jacobian_determinant;
    const real_t x10 = POW2(adjugate[1]);
    const real_t x11 = mu * x10;
    const real_t x12 = POW2(adjugate[2]);
    const real_t x13 = mu * x12;
    const real_t x14 = POW2(adjugate[0]);
    const real_t x15 = POW2(adjugate[4]);
    const real_t x16 = mu * x15;
    const real_t x17 = POW2(adjugate[5]);
    const real_t x18 = mu * x17;
    const real_t x19 = POW2(adjugate[3]);
    const real_t x20 = POW2(adjugate[7]);
    const real_t x21 = mu * x20;
    const real_t x22 = POW2(adjugate[8]);
    const real_t x23 = mu * x22;
    const real_t x24 = POW2(adjugate[6]);
    const real_t x25 = x0 * x6;
    const real_t x26 = mu * x1;
    const real_t x27 = adjugate[0] * x26 + adjugate[3] * x26 + adjugate[6] * x26;
    const real_t x28 = mu * x14;
    const real_t x29 = mu * x19;
    const real_t x30 = mu * x24;
    const real_t x31 = x0 * x3;
    diagx[0] = x9 * (adjugate[0] * x2 + adjugate[3] * x2 + adjugate[6] * x2 + x5 + x8);
    diagx[1] = x9 * (x0 * x14 + x11 + x13);
    diagx[2] = x9 * (x0 * x19 + x16 + x18);
    diagx[3] = x9 * (x0 * x24 + x21 + x23);
    diagy[0] = x9 * (adjugate[1] * x25 + adjugate[4] * x25 + adjugate[7] * x25 + x27 + x5);
    diagy[1] = x9 * (x0 * x10 + x13 + x28);
    diagy[2] = x9 * (x0 * x15 + x18 + x29);
    diagy[3] = x9 * (x0 * x20 + x23 + x30);
    diagz[0] = x9 * (adjugate[2] * x31 + adjugate[5] * x31 + adjugate[8] * x31 + x27 + x8);
    diagz[1] = x9 * (x0 * x12 + x11 + x28);
    diagz[2] = x9 * (x0 * x17 + x16 + x29);
    diagz[3] = x9 * (x0 * x22 + x21 + x30);
}

#endif  // TET4_LINEAR_ELASTICITY_INLINE_CPU_H
