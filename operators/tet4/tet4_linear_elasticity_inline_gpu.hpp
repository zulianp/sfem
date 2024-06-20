
template <typename scalar_t,
          typename adjugate_t,
          typename jacobian_determinant_t,
          typename accumulator_t>
static /*inline*/ __device__ __host__ void apply_micro_kernel(
        const scalar_t mu,
        const scalar_t lambda,
        const adjugate_t *const SFEM_RESTRICT adjugate,
        const jacobian_determinant_t jacobian_determinant,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const scalar_t *const SFEM_RESTRICT uz,
        accumulator_t *const SFEM_RESTRICT outx,
        accumulator_t *const SFEM_RESTRICT outy,
        accumulator_t *const SFEM_RESTRICT outz) {
    scalar_t disp_grad[9];
    {
        const scalar_t x0 = (scalar_t)1.0 / jacobian_determinant;
        const scalar_t x1 = adjugate[0] * x0;
        const scalar_t x2 = adjugate[3] * x0;
        const scalar_t x3 = adjugate[6] * x0;
        const scalar_t x4 = -x1 - x2 - x3;
        const scalar_t x5 = adjugate[1] * x0;
        const scalar_t x6 = adjugate[4] * x0;
        const scalar_t x7 = adjugate[7] * x0;
        const scalar_t x8 = -x5 - x6 - x7;
        const scalar_t x9 = adjugate[2] * x0;
        const scalar_t x10 = adjugate[5] * x0;
        const scalar_t x11 = adjugate[8] * x0;
        const scalar_t x12 = -x10 - x11 - x9;
        // X
        disp_grad[0] = ux[0] * x4 + ux[1] * x1 + ux[2] * x2 + ux[3] * x3;
        disp_grad[1] = ux[0] * x8 + ux[1] * x5 + ux[2] * x6 + ux[3] * x7;
        disp_grad[2] = ux[0] * x12 + ux[1] * x9 + ux[2] * x10 + ux[3] * x11;

        // Y
        disp_grad[3] = uy[0] * x4 + uy[1] * x1 + uy[2] * x2 + uy[3] * x3;
        disp_grad[4] = uy[0] * x8 + uy[1] * x5 + uy[2] * x6 + uy[3] * x7;
        disp_grad[5] = uy[0] * x12 + uy[1] * x9 + uy[2] * x10 + uy[3] * x11;

        // Z
        disp_grad[6] = uz[2] * x2 + uz[3] * x3 + uz[0] * x4 + uz[1] * x1;
        disp_grad[7] = uz[2] * x6 + uz[3] * x7 + uz[0] * x8 + uz[1] * x5;
        disp_grad[8] = uz[2] * x10 + uz[3] * x11 + uz[0] * x12 + uz[1] * x9;
    }

    // We can reuse the buffer to avoid additional register usage
    scalar_t *P = disp_grad;
    {
        const scalar_t x0 = (scalar_t)(1.0 / 3.0) * mu;
        const scalar_t x1 = (scalar_t)(1.0 / 12.0) * lambda *
                            (2 * disp_grad[0] + 2 * disp_grad[4] + 2 * disp_grad[8]);
        const scalar_t x2 = (scalar_t)(1.0 / 6.0) * mu;
        const scalar_t x3 = x2 * (disp_grad[1] + disp_grad[3]);
        const scalar_t x4 = x2 * (disp_grad[2] + disp_grad[6]);
        const scalar_t x5 = x2 * (disp_grad[5] + disp_grad[7]);
        P[0] = disp_grad[0] * x0 + x1;
        P[1] = x3;
        P[2] = x4;
        P[3] = x3;
        P[4] = disp_grad[4] * x0 + x1;
        P[5] = x5;
        P[6] = x4;
        P[7] = x5;
        P[8] = disp_grad[8] * x0 + x1;
    }

    // Bilinear form
    {
        const scalar_t x0 = adjugate[0] + adjugate[3] + adjugate[6];
        const scalar_t x1 = adjugate[1] + adjugate[4] + adjugate[7];
        const scalar_t x2 = adjugate[2] + adjugate[5] + adjugate[8];
        // X
        outx[0] = -P[0] * x0 - P[1] * x1 - P[2] * x2;
        outx[1] = P[0] * adjugate[0] + P[1] * adjugate[1] + P[2] * adjugate[2];
        outx[2] = P[0] * adjugate[3] + P[1] * adjugate[4] + P[2] * adjugate[5];
        outx[3] = P[0] * adjugate[6] + P[1] * adjugate[7] + P[2] * adjugate[8];
        // Y
        outy[0] = -P[3] * x0 - P[4] * x1 - P[5] * x2;
        outy[1] = P[3] * adjugate[0] + P[4] * adjugate[1] + P[5] * adjugate[2];
        outy[2] = P[3] * adjugate[3] + P[4] * adjugate[4] + P[5] * adjugate[5];
        outy[3] = P[3] * adjugate[6] + P[4] * adjugate[7] + P[5] * adjugate[8];
        // Z
        outz[0] = -P[6] * x0 - P[7] * x1 - P[8] * x2;
        outz[1] = P[6] * adjugate[0] + P[7] * adjugate[1] + P[8] * adjugate[2];
        outz[2] = P[6] * adjugate[3] + P[7] * adjugate[4] + P[8] * adjugate[5];
        outz[3] = P[6] * adjugate[6] + P[7] * adjugate[7] + P[8] * adjugate[8];
    }
}
