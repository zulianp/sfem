#ifndef HEX8_SKELETON_OFFIDAG_STENCIL_H
#define HEX8_SKELETON_OFFIDAG_STENCIL_H

//===============
// stencil000)
//===============
static void sshex8_apply_offdiag_stencil000(const ptrdiff_t                     xc,
                                            const ptrdiff_t                     yc,
                                            const ptrdiff_t                     zc,
                                            const ptrdiff_t                     xstride,
                                            const ptrdiff_t                     ystride,
                                            const ptrdiff_t                     zstride,
                                            const scalar_t *const SFEM_RESTRICT A,
                                            const scalar_t *const SFEM_RESTRICT input,
                                            scalar_t *const SFEM_RESTRICT       output) {
    scalar_t stencil000[8];
    stencil000[0] = 0; // FIXME remove this component from computation
    stencil000[1] = A[1];
    stencil000[2] = A[3];
    stencil000[3] = A[2];
    stencil000[4] = A[4];
    stencil000[5] = A[5];
    stencil000[6] = A[7];
    stencil000[7] = A[6];
    // buffs
    const scalar_t *const in0 = &input[0];
    const scalar_t *const in1 = &input[xstride];
    const scalar_t *const in2 = &input[ystride];
    const scalar_t *const in3 = &input[xstride + ystride];
    const scalar_t *const in4 = &input[zstride];
    const scalar_t *const in5 = &input[xstride + zstride];
    const scalar_t *const in6 = &input[ystride + zstride];
    const scalar_t *const in7 = &input[xstride + ystride + zstride];
    scalar_t *const       out = &output[0];
    for (ptrdiff_t zi = 0; zi < (1); zi++) {
        for (ptrdiff_t yi = 0; yi < (1); yi++) {
            for (ptrdiff_t xi = 0; xi < (1); xi++) {
                const ptrdiff_t idx = xi * xstride + yi * ystride + zi * zstride;
                out[idx] += in0[idx] * stencil000[0];
                out[idx] += in1[idx] * stencil000[1];
                out[idx] += in2[idx] * stencil000[2];
                out[idx] += in3[idx] * stencil000[3];
                out[idx] += in4[idx] * stencil000[4];
                out[idx] += in5[idx] * stencil000[5];
                out[idx] += in6[idx] * stencil000[6];
                out[idx] += in7[idx] * stencil000[7];
            }
        }
    }
}

//===============
// stencil100)
//===============
static void sshex8_apply_offdiag_stencil100(const ptrdiff_t                     xc,
                                            const ptrdiff_t                     yc,
                                            const ptrdiff_t                     zc,
                                            const ptrdiff_t                     xstride,
                                            const ptrdiff_t                     ystride,
                                            const ptrdiff_t                     zstride,
                                            const scalar_t *const SFEM_RESTRICT A,
                                            const scalar_t *const SFEM_RESTRICT input,
                                            scalar_t *const SFEM_RESTRICT       output) {
    scalar_t stencil100[12];
    stencil100[0]  = A[8];
    stencil100[1]  = 0;
    stencil100[2]  = A[1];
    stencil100[3]  = A[11];
    stencil100[4]  = A[10] + A[3];
    stencil100[5]  = A[2];
    stencil100[6]  = A[12];
    stencil100[7]  = A[13] + A[4];
    stencil100[8]  = A[5];
    stencil100[9]  = A[15];
    stencil100[10] = A[14] + A[7];
    stencil100[11] = A[6];
    // buffs
    const scalar_t *const in0  = &input[0];
    const scalar_t *const in1  = &input[xstride];
    const scalar_t *const in2  = &input[2 * xstride];
    const scalar_t *const in3  = &input[ystride];
    const scalar_t *const in4  = &input[xstride + ystride];
    const scalar_t *const in5  = &input[2 * xstride + ystride];
    const scalar_t *const in6  = &input[zstride];
    const scalar_t *const in7  = &input[xstride + zstride];
    const scalar_t *const in8  = &input[2 * xstride + zstride];
    const scalar_t *const in9  = &input[ystride + zstride];
    const scalar_t *const in10 = &input[xstride + ystride + zstride];
    const scalar_t *const in11 = &input[2 * xstride + ystride + zstride];
    scalar_t *const       out  = &output[xstride];
    for (ptrdiff_t zi = 0; zi < (1); zi++) {
        for (ptrdiff_t yi = 0; yi < (1); yi++) {
            for (ptrdiff_t xi = 0; xi < (xc - 2); xi++) {
                const ptrdiff_t idx = xi * xstride + yi * ystride + zi * zstride;
                out[idx] += in0[idx] * stencil100[0];
                out[idx] += in1[idx] * stencil100[1];
                out[idx] += in2[idx] * stencil100[2];
                out[idx] += in3[idx] * stencil100[3];
                out[idx] += in4[idx] * stencil100[4];
                out[idx] += in5[idx] * stencil100[5];
                out[idx] += in6[idx] * stencil100[6];
                out[idx] += in7[idx] * stencil100[7];
                out[idx] += in8[idx] * stencil100[8];
                out[idx] += in9[idx] * stencil100[9];
                out[idx] += in10[idx] * stencil100[10];
                out[idx] += in11[idx] * stencil100[11];
            }
        }
    }
}

//===============
// stencil200)
//===============
static void sshex8_apply_offdiag_stencil200(const ptrdiff_t                     xc,
                                            const ptrdiff_t                     yc,
                                            const ptrdiff_t                     zc,
                                            const ptrdiff_t                     xstride,
                                            const ptrdiff_t                     ystride,
                                            const ptrdiff_t                     zstride,
                                            const scalar_t *const SFEM_RESTRICT A,
                                            const scalar_t *const SFEM_RESTRICT input,
                                            scalar_t *const SFEM_RESTRICT       output) {
    scalar_t stencil200[8];
    stencil200[0] = A[8];
    stencil200[1] = 0; // FIXME remove this component from computation
    stencil200[2] = A[11];
    stencil200[3] = A[10];
    stencil200[4] = A[12];
    stencil200[5] = A[13];
    stencil200[6] = A[15];
    stencil200[7] = A[14];
    // buffs
    const scalar_t *const in0 = &input[xstride * (xc - 2)];
    const scalar_t *const in1 = &input[xstride * (xc - 1)];
    const scalar_t *const in2 = &input[xstride * (xc - 2) + ystride];
    const scalar_t *const in3 = &input[xstride * (xc - 1) + ystride];
    const scalar_t *const in4 = &input[xstride * (xc - 2) + zstride];
    const scalar_t *const in5 = &input[xstride * (xc - 1) + zstride];
    const scalar_t *const in6 = &input[xstride * (xc - 2) + ystride + zstride];
    const scalar_t *const in7 = &input[xstride * (xc - 1) + ystride + zstride];
    scalar_t *const       out = &output[xstride * (xc - 1)];
    for (ptrdiff_t zi = 0; zi < (1); zi++) {
        for (ptrdiff_t yi = 0; yi < (1); yi++) {
            for (ptrdiff_t xi = 0; xi < (1); xi++) {
                const ptrdiff_t idx = xi * xstride + yi * ystride + zi * zstride;
                out[idx] += in0[idx] * stencil200[0];
                out[idx] += in1[idx] * stencil200[1];
                out[idx] += in2[idx] * stencil200[2];
                out[idx] += in3[idx] * stencil200[3];
                out[idx] += in4[idx] * stencil200[4];
                out[idx] += in5[idx] * stencil200[5];
                out[idx] += in6[idx] * stencil200[6];
                out[idx] += in7[idx] * stencil200[7];
            }
        }
    }
}

//===============
// stencil010)
//===============
static void sshex8_apply_offdiag_stencil010(const ptrdiff_t                     xc,
                                            const ptrdiff_t                     yc,
                                            const ptrdiff_t                     zc,
                                            const ptrdiff_t                     xstride,
                                            const ptrdiff_t                     ystride,
                                            const ptrdiff_t                     zstride,
                                            const scalar_t *const SFEM_RESTRICT A,
                                            const scalar_t *const SFEM_RESTRICT input,
                                            scalar_t *const SFEM_RESTRICT       output) {
    scalar_t stencil010[12];
    stencil010[0]  = A[24];
    stencil010[1]  = A[25];
    stencil010[2]  = 0;
    stencil010[3]  = A[1] + A[26];
    stencil010[4]  = A[3];
    stencil010[5]  = A[2];
    stencil010[6]  = A[28];
    stencil010[7]  = A[29];
    stencil010[8]  = A[31] + A[4];
    stencil010[9]  = A[30] + A[5];
    stencil010[10] = A[7];
    stencil010[11] = A[6];
    // buffs
    const scalar_t *const in0  = &input[0];
    const scalar_t *const in1  = &input[xstride];
    const scalar_t *const in2  = &input[ystride];
    const scalar_t *const in3  = &input[xstride + ystride];
    const scalar_t *const in4  = &input[2 * ystride];
    const scalar_t *const in5  = &input[xstride + 2 * ystride];
    const scalar_t *const in6  = &input[zstride];
    const scalar_t *const in7  = &input[xstride + zstride];
    const scalar_t *const in8  = &input[ystride + zstride];
    const scalar_t *const in9  = &input[xstride + ystride + zstride];
    const scalar_t *const in10 = &input[2 * ystride + zstride];
    const scalar_t *const in11 = &input[xstride + 2 * ystride + zstride];
    scalar_t *const       out  = &output[ystride];
    for (ptrdiff_t zi = 0; zi < (1); zi++) {
        for (ptrdiff_t yi = 0; yi < (yc - 2); yi++) {
            for (ptrdiff_t xi = 0; xi < (1); xi++) {
                const ptrdiff_t idx = xi * xstride + yi * ystride + zi * zstride;
                out[idx] += in0[idx] * stencil010[0];
                out[idx] += in1[idx] * stencil010[1];
                out[idx] += in2[idx] * stencil010[2];
                out[idx] += in3[idx] * stencil010[3];
                out[idx] += in4[idx] * stencil010[4];
                out[idx] += in5[idx] * stencil010[5];
                out[idx] += in6[idx] * stencil010[6];
                out[idx] += in7[idx] * stencil010[7];
                out[idx] += in8[idx] * stencil010[8];
                out[idx] += in9[idx] * stencil010[9];
                out[idx] += in10[idx] * stencil010[10];
                out[idx] += in11[idx] * stencil010[11];
            }
        }
    }
}

//===============
// stencil110)
//===============
static void sshex8_apply_offdiag_stencil110(const ptrdiff_t                     xc,
                                            const ptrdiff_t                     yc,
                                            const ptrdiff_t                     zc,
                                            const ptrdiff_t                     xstride,
                                            const ptrdiff_t                     ystride,
                                            const ptrdiff_t                     zstride,
                                            const scalar_t *const SFEM_RESTRICT A,
                                            const scalar_t *const SFEM_RESTRICT input,
                                            scalar_t *const SFEM_RESTRICT       output) {
    scalar_t stencil110[18];
    stencil110[0]  = A[16];
    stencil110[1]  = A[17] + A[24];
    stencil110[2]  = A[25];
    stencil110[3]  = A[19] + A[8];
    stencil110[4]  = 0;
    stencil110[5]  = A[1] + A[26];
    stencil110[6]  = A[11];
    stencil110[7]  = A[10] + A[3];
    stencil110[8]  = A[2];
    stencil110[9]  = A[20];
    stencil110[10] = A[21] + A[28];
    stencil110[11] = A[29];
    stencil110[12] = A[12] + A[23];
    stencil110[13] = A[13] + A[22] + A[31] + A[4];
    stencil110[14] = A[30] + A[5];
    stencil110[15] = A[15];
    stencil110[16] = A[14] + A[7];
    stencil110[17] = A[6];
    // buffs
    const scalar_t *const in0  = &input[0];
    const scalar_t *const in1  = &input[xstride];
    const scalar_t *const in2  = &input[2 * xstride];
    const scalar_t *const in3  = &input[ystride];
    const scalar_t *const in4  = &input[xstride + ystride];
    const scalar_t *const in5  = &input[2 * xstride + ystride];
    const scalar_t *const in6  = &input[2 * ystride];
    const scalar_t *const in7  = &input[xstride + 2 * ystride];
    const scalar_t *const in8  = &input[2 * xstride + 2 * ystride];
    const scalar_t *const in9  = &input[zstride];
    const scalar_t *const in10 = &input[xstride + zstride];
    const scalar_t *const in11 = &input[2 * xstride + zstride];
    const scalar_t *const in12 = &input[ystride + zstride];
    const scalar_t *const in13 = &input[xstride + ystride + zstride];
    const scalar_t *const in14 = &input[2 * xstride + ystride + zstride];
    const scalar_t *const in15 = &input[2 * ystride + zstride];
    const scalar_t *const in16 = &input[xstride + 2 * ystride + zstride];
    const scalar_t *const in17 = &input[2 * xstride + 2 * ystride + zstride];
    scalar_t *const       out  = &output[xstride + ystride];
    for (ptrdiff_t zi = 0; zi < (1); zi++) {
        for (ptrdiff_t yi = 0; yi < (yc - 2); yi++) {
            for (ptrdiff_t xi = 0; xi < (xc - 2); xi++) {
                const ptrdiff_t idx = xi * xstride + yi * ystride + zi * zstride;
                out[idx] += in0[idx] * stencil110[0];
                out[idx] += in1[idx] * stencil110[1];
                out[idx] += in2[idx] * stencil110[2];
                out[idx] += in3[idx] * stencil110[3];
                out[idx] += in4[idx] * stencil110[4];
                out[idx] += in5[idx] * stencil110[5];
                out[idx] += in6[idx] * stencil110[6];
                out[idx] += in7[idx] * stencil110[7];
                out[idx] += in8[idx] * stencil110[8];
                out[idx] += in9[idx] * stencil110[9];
                out[idx] += in10[idx] * stencil110[10];
                out[idx] += in11[idx] * stencil110[11];
                out[idx] += in12[idx] * stencil110[12];
                out[idx] += in13[idx] * stencil110[13];
                out[idx] += in14[idx] * stencil110[14];
                out[idx] += in15[idx] * stencil110[15];
                out[idx] += in16[idx] * stencil110[16];
                out[idx] += in17[idx] * stencil110[17];
            }
        }
    }
}

//===============
// stencil210)
//===============
static void sshex8_apply_offdiag_stencil210(const ptrdiff_t                     xc,
                                            const ptrdiff_t                     yc,
                                            const ptrdiff_t                     zc,
                                            const ptrdiff_t                     xstride,
                                            const ptrdiff_t                     ystride,
                                            const ptrdiff_t                     zstride,
                                            const scalar_t *const SFEM_RESTRICT A,
                                            const scalar_t *const SFEM_RESTRICT input,
                                            scalar_t *const SFEM_RESTRICT       output) {
    scalar_t stencil210[12];
    stencil210[0]  = A[16];
    stencil210[1]  = A[17];
    stencil210[2]  = A[19] + A[8];
    stencil210[3]  = 0;
    stencil210[4]  = A[11];
    stencil210[5]  = A[10];
    stencil210[6]  = A[20];
    stencil210[7]  = A[21];
    stencil210[8]  = A[12] + A[23];
    stencil210[9]  = A[13] + A[22];
    stencil210[10] = A[15];
    stencil210[11] = A[14];
    // buffs
    const scalar_t *const in0  = &input[xstride * (xc - 2)];
    const scalar_t *const in1  = &input[xstride * (xc - 1)];
    const scalar_t *const in2  = &input[xstride * (xc - 2) + ystride];
    const scalar_t *const in3  = &input[xstride * (xc - 1) + ystride];
    const scalar_t *const in4  = &input[xstride * (xc - 2) + 2 * ystride];
    const scalar_t *const in5  = &input[xstride * (xc - 1) + 2 * ystride];
    const scalar_t *const in6  = &input[xstride * (xc - 2) + zstride];
    const scalar_t *const in7  = &input[xstride * (xc - 1) + zstride];
    const scalar_t *const in8  = &input[xstride * (xc - 2) + ystride + zstride];
    const scalar_t *const in9  = &input[xstride * (xc - 1) + ystride + zstride];
    const scalar_t *const in10 = &input[xstride * (xc - 2) + 2 * ystride + zstride];
    const scalar_t *const in11 = &input[xstride * (xc - 1) + 2 * ystride + zstride];
    scalar_t *const       out  = &output[xstride * (xc - 1) + ystride];
    for (ptrdiff_t zi = 0; zi < (1); zi++) {
        for (ptrdiff_t yi = 0; yi < (yc - 2); yi++) {
            for (ptrdiff_t xi = 0; xi < (1); xi++) {
                const ptrdiff_t idx = xi * xstride + yi * ystride + zi * zstride;
                out[idx] += in0[idx] * stencil210[0];
                out[idx] += in1[idx] * stencil210[1];
                out[idx] += in2[idx] * stencil210[2];
                out[idx] += in3[idx] * stencil210[3];
                out[idx] += in4[idx] * stencil210[4];
                out[idx] += in5[idx] * stencil210[5];
                out[idx] += in6[idx] * stencil210[6];
                out[idx] += in7[idx] * stencil210[7];
                out[idx] += in8[idx] * stencil210[8];
                out[idx] += in9[idx] * stencil210[9];
                out[idx] += in10[idx] * stencil210[10];
                out[idx] += in11[idx] * stencil210[11];
            }
        }
    }
}

//===============
// stencil020)
//===============
static void sshex8_apply_offdiag_stencil020(const ptrdiff_t                     xc,
                                            const ptrdiff_t                     yc,
                                            const ptrdiff_t                     zc,
                                            const ptrdiff_t                     xstride,
                                            const ptrdiff_t                     ystride,
                                            const ptrdiff_t                     zstride,
                                            const scalar_t *const SFEM_RESTRICT A,
                                            const scalar_t *const SFEM_RESTRICT input,
                                            scalar_t *const SFEM_RESTRICT       output) {
    scalar_t stencil020[8];
    stencil020[0] = A[24];
    stencil020[1] = A[25];
    stencil020[2] = 0; // FIXME remove this component from computation
    stencil020[3] = A[26];
    stencil020[4] = A[28];
    stencil020[5] = A[29];
    stencil020[6] = A[31];
    stencil020[7] = A[30];
    // buffs
    const scalar_t *const in0 = &input[ystride * (yc - 2)];
    const scalar_t *const in1 = &input[xstride + ystride * (yc - 2)];
    const scalar_t *const in2 = &input[ystride * (yc - 1)];
    const scalar_t *const in3 = &input[xstride + ystride * (yc - 1)];
    const scalar_t *const in4 = &input[ystride * (yc - 2) + zstride];
    const scalar_t *const in5 = &input[xstride + ystride * (yc - 2) + zstride];
    const scalar_t *const in6 = &input[ystride * (yc - 1) + zstride];
    const scalar_t *const in7 = &input[xstride + ystride * (yc - 1) + zstride];
    scalar_t *const       out = &output[ystride * (yc - 1)];
    for (ptrdiff_t zi = 0; zi < (1); zi++) {
        for (ptrdiff_t yi = 0; yi < (1); yi++) {
            for (ptrdiff_t xi = 0; xi < (1); xi++) {
                const ptrdiff_t idx = xi * xstride + yi * ystride + zi * zstride;
                out[idx] += in0[idx] * stencil020[0];
                out[idx] += in1[idx] * stencil020[1];
                out[idx] += in2[idx] * stencil020[2];
                out[idx] += in3[idx] * stencil020[3];
                out[idx] += in4[idx] * stencil020[4];
                out[idx] += in5[idx] * stencil020[5];
                out[idx] += in6[idx] * stencil020[6];
                out[idx] += in7[idx] * stencil020[7];
            }
        }
    }
}

//===============
// stencil120)
//===============
static void sshex8_apply_offdiag_stencil120(const ptrdiff_t                     xc,
                                            const ptrdiff_t                     yc,
                                            const ptrdiff_t                     zc,
                                            const ptrdiff_t                     xstride,
                                            const ptrdiff_t                     ystride,
                                            const ptrdiff_t                     zstride,
                                            const scalar_t *const SFEM_RESTRICT A,
                                            const scalar_t *const SFEM_RESTRICT input,
                                            scalar_t *const SFEM_RESTRICT       output) {
    scalar_t stencil120[12];
    stencil120[0]  = A[16];
    stencil120[1]  = A[17] + A[24];
    stencil120[2]  = A[25];
    stencil120[3]  = A[19];
    stencil120[4]  = 0;
    stencil120[5]  = A[26];
    stencil120[6]  = A[20];
    stencil120[7]  = A[21] + A[28];
    stencil120[8]  = A[29];
    stencil120[9]  = A[23];
    stencil120[10] = A[22] + A[31];
    stencil120[11] = A[30];
    // buffs
    const scalar_t *const in0  = &input[ystride * (yc - 2)];
    const scalar_t *const in1  = &input[xstride + ystride * (yc - 2)];
    const scalar_t *const in2  = &input[2 * xstride + ystride * (yc - 2)];
    const scalar_t *const in3  = &input[ystride * (yc - 1)];
    const scalar_t *const in4  = &input[xstride + ystride * (yc - 1)];
    const scalar_t *const in5  = &input[2 * xstride + ystride * (yc - 1)];
    const scalar_t *const in6  = &input[ystride * (yc - 2) + zstride];
    const scalar_t *const in7  = &input[xstride + ystride * (yc - 2) + zstride];
    const scalar_t *const in8  = &input[2 * xstride + ystride * (yc - 2) + zstride];
    const scalar_t *const in9  = &input[ystride * (yc - 1) + zstride];
    const scalar_t *const in10 = &input[xstride + ystride * (yc - 1) + zstride];
    const scalar_t *const in11 = &input[2 * xstride + ystride * (yc - 1) + zstride];
    scalar_t *const       out  = &output[xstride + ystride * (yc - 1)];
    for (ptrdiff_t zi = 0; zi < (1); zi++) {
        for (ptrdiff_t yi = 0; yi < (1); yi++) {
            for (ptrdiff_t xi = 0; xi < (xc - 2); xi++) {
                const ptrdiff_t idx = xi * xstride + yi * ystride + zi * zstride;
                out[idx] += in0[idx] * stencil120[0];
                out[idx] += in1[idx] * stencil120[1];
                out[idx] += in2[idx] * stencil120[2];
                out[idx] += in3[idx] * stencil120[3];
                out[idx] += in4[idx] * stencil120[4];
                out[idx] += in5[idx] * stencil120[5];
                out[idx] += in6[idx] * stencil120[6];
                out[idx] += in7[idx] * stencil120[7];
                out[idx] += in8[idx] * stencil120[8];
                out[idx] += in9[idx] * stencil120[9];
                out[idx] += in10[idx] * stencil120[10];
                out[idx] += in11[idx] * stencil120[11];
            }
        }
    }
}

//===============
// stencil220)
//===============
static void sshex8_apply_offdiag_stencil220(const ptrdiff_t                     xc,
                                            const ptrdiff_t                     yc,
                                            const ptrdiff_t                     zc,
                                            const ptrdiff_t                     xstride,
                                            const ptrdiff_t                     ystride,
                                            const ptrdiff_t                     zstride,
                                            const scalar_t *const SFEM_RESTRICT A,
                                            const scalar_t *const SFEM_RESTRICT input,
                                            scalar_t *const SFEM_RESTRICT       output) {
    scalar_t stencil220[8];
    stencil220[0] = A[16];
    stencil220[1] = A[17];
    stencil220[2] = A[19];
    stencil220[3] = 0; // FIXME remove this component from computation
    stencil220[4] = A[20];
    stencil220[5] = A[21];
    stencil220[6] = A[23];
    stencil220[7] = A[22];
    // buffs
    const scalar_t *const in0 = &input[xstride * (xc - 2) + ystride * (yc - 2)];
    const scalar_t *const in1 = &input[xstride * (xc - 1) + ystride * (yc - 2)];
    const scalar_t *const in2 = &input[xstride * (xc - 2) + ystride * (yc - 1)];
    const scalar_t *const in3 = &input[xstride * (xc - 1) + ystride * (yc - 1)];
    const scalar_t *const in4 = &input[xstride * (xc - 2) + ystride * (yc - 2) + zstride];
    const scalar_t *const in5 = &input[xstride * (xc - 1) + ystride * (yc - 2) + zstride];
    const scalar_t *const in6 = &input[xstride * (xc - 2) + ystride * (yc - 1) + zstride];
    const scalar_t *const in7 = &input[xstride * (xc - 1) + ystride * (yc - 1) + zstride];
    scalar_t *const       out = &output[xstride * (xc - 1) + ystride * (yc - 1)];
    for (ptrdiff_t zi = 0; zi < (1); zi++) {
        for (ptrdiff_t yi = 0; yi < (1); yi++) {
            for (ptrdiff_t xi = 0; xi < (1); xi++) {
                const ptrdiff_t idx = xi * xstride + yi * ystride + zi * zstride;
                out[idx] += in0[idx] * stencil220[0];
                out[idx] += in1[idx] * stencil220[1];
                out[idx] += in2[idx] * stencil220[2];
                out[idx] += in3[idx] * stencil220[3];
                out[idx] += in4[idx] * stencil220[4];
                out[idx] += in5[idx] * stencil220[5];
                out[idx] += in6[idx] * stencil220[6];
                out[idx] += in7[idx] * stencil220[7];
            }
        }
    }
}

//===============
// stencil001)
//===============
static void sshex8_apply_offdiag_stencil001(const ptrdiff_t                     xc,
                                            const ptrdiff_t                     yc,
                                            const ptrdiff_t                     zc,
                                            const ptrdiff_t                     xstride,
                                            const ptrdiff_t                     ystride,
                                            const ptrdiff_t                     zstride,
                                            const scalar_t *const SFEM_RESTRICT A,
                                            const scalar_t *const SFEM_RESTRICT input,
                                            scalar_t *const SFEM_RESTRICT       output) {
    scalar_t stencil001[12];
    stencil001[0]  = A[32];
    stencil001[1]  = A[33];
    stencil001[2]  = A[35];
    stencil001[3]  = A[34];
    stencil001[4]  = 0;
    stencil001[5]  = A[1] + A[37];
    stencil001[6]  = A[39] + A[3];
    stencil001[7]  = A[2] + A[38];
    stencil001[8]  = A[4];
    stencil001[9]  = A[5];
    stencil001[10] = A[7];
    stencil001[11] = A[6];
    // buffs
    const scalar_t *const in0  = &input[0];
    const scalar_t *const in1  = &input[xstride];
    const scalar_t *const in2  = &input[ystride];
    const scalar_t *const in3  = &input[xstride + ystride];
    const scalar_t *const in4  = &input[zstride];
    const scalar_t *const in5  = &input[xstride + zstride];
    const scalar_t *const in6  = &input[ystride + zstride];
    const scalar_t *const in7  = &input[xstride + ystride + zstride];
    const scalar_t *const in8  = &input[2 * zstride];
    const scalar_t *const in9  = &input[xstride + 2 * zstride];
    const scalar_t *const in10 = &input[ystride + 2 * zstride];
    const scalar_t *const in11 = &input[xstride + ystride + 2 * zstride];
    scalar_t *const       out  = &output[zstride];
    for (ptrdiff_t zi = 0; zi < (zc - 2); zi++) {
        for (ptrdiff_t yi = 0; yi < (1); yi++) {
            for (ptrdiff_t xi = 0; xi < (1); xi++) {
                const ptrdiff_t idx = xi * xstride + yi * ystride + zi * zstride;
                out[idx] += in0[idx] * stencil001[0];
                out[idx] += in1[idx] * stencil001[1];
                out[idx] += in2[idx] * stencil001[2];
                out[idx] += in3[idx] * stencil001[3];
                out[idx] += in4[idx] * stencil001[4];
                out[idx] += in5[idx] * stencil001[5];
                out[idx] += in6[idx] * stencil001[6];
                out[idx] += in7[idx] * stencil001[7];
                out[idx] += in8[idx] * stencil001[8];
                out[idx] += in9[idx] * stencil001[9];
                out[idx] += in10[idx] * stencil001[10];
                out[idx] += in11[idx] * stencil001[11];
            }
        }
    }
}

//===============
// stencil101)
//===============
static void sshex8_apply_offdiag_stencil101(const ptrdiff_t                     xc,
                                            const ptrdiff_t                     yc,
                                            const ptrdiff_t                     zc,
                                            const ptrdiff_t                     xstride,
                                            const ptrdiff_t                     ystride,
                                            const ptrdiff_t                     zstride,
                                            const scalar_t *const SFEM_RESTRICT A,
                                            const scalar_t *const SFEM_RESTRICT input,
                                            scalar_t *const SFEM_RESTRICT       output) {
    scalar_t stencil101[18];
    stencil101[0]  = A[40];
    stencil101[1]  = A[32] + A[41];
    stencil101[2]  = A[33];
    stencil101[3]  = A[43];
    stencil101[4]  = A[35] + A[42];
    stencil101[5]  = A[34];
    stencil101[6]  = A[44] + A[8];
    stencil101[7]  = 0;
    stencil101[8]  = A[1] + A[37];
    stencil101[9]  = A[11] + A[47];
    stencil101[10] = A[10] + A[39] + A[3] + A[46];
    stencil101[11] = A[2] + A[38];
    stencil101[12] = A[12];
    stencil101[13] = A[13] + A[4];
    stencil101[14] = A[5];
    stencil101[15] = A[15];
    stencil101[16] = A[14] + A[7];
    stencil101[17] = A[6];
    // buffs
    const scalar_t *const in0  = &input[0];
    const scalar_t *const in1  = &input[xstride];
    const scalar_t *const in2  = &input[2 * xstride];
    const scalar_t *const in3  = &input[ystride];
    const scalar_t *const in4  = &input[xstride + ystride];
    const scalar_t *const in5  = &input[2 * xstride + ystride];
    const scalar_t *const in6  = &input[zstride];
    const scalar_t *const in7  = &input[xstride + zstride];
    const scalar_t *const in8  = &input[2 * xstride + zstride];
    const scalar_t *const in9  = &input[ystride + zstride];
    const scalar_t *const in10 = &input[xstride + ystride + zstride];
    const scalar_t *const in11 = &input[2 * xstride + ystride + zstride];
    const scalar_t *const in12 = &input[2 * zstride];
    const scalar_t *const in13 = &input[xstride + 2 * zstride];
    const scalar_t *const in14 = &input[2 * xstride + 2 * zstride];
    const scalar_t *const in15 = &input[ystride + 2 * zstride];
    const scalar_t *const in16 = &input[xstride + ystride + 2 * zstride];
    const scalar_t *const in17 = &input[2 * xstride + ystride + 2 * zstride];
    scalar_t *const       out  = &output[xstride + zstride];
    for (ptrdiff_t zi = 0; zi < (zc - 2); zi++) {
        for (ptrdiff_t yi = 0; yi < (1); yi++) {
            for (ptrdiff_t xi = 0; xi < (xc - 2); xi++) {
                const ptrdiff_t idx = xi * xstride + yi * ystride + zi * zstride;
                out[idx] += in0[idx] * stencil101[0];
                out[idx] += in1[idx] * stencil101[1];
                out[idx] += in2[idx] * stencil101[2];
                out[idx] += in3[idx] * stencil101[3];
                out[idx] += in4[idx] * stencil101[4];
                out[idx] += in5[idx] * stencil101[5];
                out[idx] += in6[idx] * stencil101[6];
                out[idx] += in7[idx] * stencil101[7];
                out[idx] += in8[idx] * stencil101[8];
                out[idx] += in9[idx] * stencil101[9];
                out[idx] += in10[idx] * stencil101[10];
                out[idx] += in11[idx] * stencil101[11];
                out[idx] += in12[idx] * stencil101[12];
                out[idx] += in13[idx] * stencil101[13];
                out[idx] += in14[idx] * stencil101[14];
                out[idx] += in15[idx] * stencil101[15];
                out[idx] += in16[idx] * stencil101[16];
                out[idx] += in17[idx] * stencil101[17];
            }
        }
    }
}

//===============
// stencil201)
//===============
static void sshex8_apply_offdiag_stencil201(const ptrdiff_t                     xc,
                                            const ptrdiff_t                     yc,
                                            const ptrdiff_t                     zc,
                                            const ptrdiff_t                     xstride,
                                            const ptrdiff_t                     ystride,
                                            const ptrdiff_t                     zstride,
                                            const scalar_t *const SFEM_RESTRICT A,
                                            const scalar_t *const SFEM_RESTRICT input,
                                            scalar_t *const SFEM_RESTRICT       output) {
    scalar_t stencil201[12];
    stencil201[0]  = A[40];
    stencil201[1]  = A[41];
    stencil201[2]  = A[43];
    stencil201[3]  = A[42];
    stencil201[4]  = A[44] + A[8];
    stencil201[5]  = 0;
    stencil201[6]  = A[11] + A[47];
    stencil201[7]  = A[10] + A[46];
    stencil201[8]  = A[12];
    stencil201[9]  = A[13];
    stencil201[10] = A[15];
    stencil201[11] = A[14];
    // buffs
    const scalar_t *const in0  = &input[xstride * (xc - 2)];
    const scalar_t *const in1  = &input[xstride * (xc - 1)];
    const scalar_t *const in2  = &input[xstride * (xc - 2) + ystride];
    const scalar_t *const in3  = &input[xstride * (xc - 1) + ystride];
    const scalar_t *const in4  = &input[xstride * (xc - 2) + zstride];
    const scalar_t *const in5  = &input[xstride * (xc - 1) + zstride];
    const scalar_t *const in6  = &input[xstride * (xc - 2) + ystride + zstride];
    const scalar_t *const in7  = &input[xstride * (xc - 1) + ystride + zstride];
    const scalar_t *const in8  = &input[xstride * (xc - 2) + 2 * zstride];
    const scalar_t *const in9  = &input[xstride * (xc - 1) + 2 * zstride];
    const scalar_t *const in10 = &input[xstride * (xc - 2) + ystride + 2 * zstride];
    const scalar_t *const in11 = &input[xstride * (xc - 1) + ystride + 2 * zstride];
    scalar_t *const       out  = &output[xstride * (xc - 1) + zstride];
    for (ptrdiff_t zi = 0; zi < (zc - 2); zi++) {
        for (ptrdiff_t yi = 0; yi < (1); yi++) {
            for (ptrdiff_t xi = 0; xi < (1); xi++) {
                const ptrdiff_t idx = xi * xstride + yi * ystride + zi * zstride;
                out[idx] += in0[idx] * stencil201[0];
                out[idx] += in1[idx] * stencil201[1];
                out[idx] += in2[idx] * stencil201[2];
                out[idx] += in3[idx] * stencil201[3];
                out[idx] += in4[idx] * stencil201[4];
                out[idx] += in5[idx] * stencil201[5];
                out[idx] += in6[idx] * stencil201[6];
                out[idx] += in7[idx] * stencil201[7];
                out[idx] += in8[idx] * stencil201[8];
                out[idx] += in9[idx] * stencil201[9];
                out[idx] += in10[idx] * stencil201[10];
                out[idx] += in11[idx] * stencil201[11];
            }
        }
    }
}

//===============
// stencil011)
//===============
static void sshex8_apply_offdiag_stencil011(const ptrdiff_t                     xc,
                                            const ptrdiff_t                     yc,
                                            const ptrdiff_t                     zc,
                                            const ptrdiff_t                     xstride,
                                            const ptrdiff_t                     ystride,
                                            const ptrdiff_t                     zstride,
                                            const scalar_t *const SFEM_RESTRICT A,
                                            const scalar_t *const SFEM_RESTRICT input,
                                            scalar_t *const SFEM_RESTRICT       output) {
    scalar_t stencil011[18];
    stencil011[0]  = A[56];
    stencil011[1]  = A[57];
    stencil011[2]  = A[32] + A[59];
    stencil011[3]  = A[33] + A[58];
    stencil011[4]  = A[35];
    stencil011[5]  = A[34];
    stencil011[6]  = A[24] + A[60];
    stencil011[7]  = A[25] + A[61];
    stencil011[8]  = 0;
    stencil011[9]  = A[1] + A[26] + A[37] + A[62];
    stencil011[10] = A[39] + A[3];
    stencil011[11] = A[2] + A[38];
    stencil011[12] = A[28];
    stencil011[13] = A[29];
    stencil011[14] = A[31] + A[4];
    stencil011[15] = A[30] + A[5];
    stencil011[16] = A[7];
    stencil011[17] = A[6];
    // buffs
    const scalar_t *const in0  = &input[0];
    const scalar_t *const in1  = &input[xstride];
    const scalar_t *const in2  = &input[ystride];
    const scalar_t *const in3  = &input[xstride + ystride];
    const scalar_t *const in4  = &input[2 * ystride];
    const scalar_t *const in5  = &input[xstride + 2 * ystride];
    const scalar_t *const in6  = &input[zstride];
    const scalar_t *const in7  = &input[xstride + zstride];
    const scalar_t *const in8  = &input[ystride + zstride];
    const scalar_t *const in9  = &input[xstride + ystride + zstride];
    const scalar_t *const in10 = &input[2 * ystride + zstride];
    const scalar_t *const in11 = &input[xstride + 2 * ystride + zstride];
    const scalar_t *const in12 = &input[2 * zstride];
    const scalar_t *const in13 = &input[xstride + 2 * zstride];
    const scalar_t *const in14 = &input[ystride + 2 * zstride];
    const scalar_t *const in15 = &input[xstride + ystride + 2 * zstride];
    const scalar_t *const in16 = &input[2 * ystride + 2 * zstride];
    const scalar_t *const in17 = &input[xstride + 2 * ystride + 2 * zstride];
    scalar_t *const       out  = &output[ystride + zstride];
    for (ptrdiff_t zi = 0; zi < (zc - 2); zi++) {
        for (ptrdiff_t yi = 0; yi < (yc - 2); yi++) {
            for (ptrdiff_t xi = 0; xi < (1); xi++) {
                const ptrdiff_t idx = xi * xstride + yi * ystride + zi * zstride;
                out[idx] += in0[idx] * stencil011[0];
                out[idx] += in1[idx] * stencil011[1];
                out[idx] += in2[idx] * stencil011[2];
                out[idx] += in3[idx] * stencil011[3];
                out[idx] += in4[idx] * stencil011[4];
                out[idx] += in5[idx] * stencil011[5];
                out[idx] += in6[idx] * stencil011[6];
                out[idx] += in7[idx] * stencil011[7];
                out[idx] += in8[idx] * stencil011[8];
                out[idx] += in9[idx] * stencil011[9];
                out[idx] += in10[idx] * stencil011[10];
                out[idx] += in11[idx] * stencil011[11];
                out[idx] += in12[idx] * stencil011[12];
                out[idx] += in13[idx] * stencil011[13];
                out[idx] += in14[idx] * stencil011[14];
                out[idx] += in15[idx] * stencil011[15];
                out[idx] += in16[idx] * stencil011[16];
                out[idx] += in17[idx] * stencil011[17];
            }
        }
    }
}

//===============
// stencil111)
//===============
static void sshex8_apply_offdiag_stencil111(const ptrdiff_t                     xc,
                                            const ptrdiff_t                     yc,
                                            const ptrdiff_t                     zc,
                                            const ptrdiff_t                     xstride,
                                            const ptrdiff_t                     ystride,
                                            const ptrdiff_t                     zstride,
                                            const scalar_t *const SFEM_RESTRICT A,
                                            const scalar_t *const SFEM_RESTRICT input,
                                            scalar_t *const SFEM_RESTRICT       output) {
    scalar_t stencil111[27];
    stencil111[0]  = A[48];
    stencil111[1]  = A[49] + A[56];
    stencil111[2]  = A[57];
    stencil111[3]  = A[40] + A[51];
    stencil111[4]  = A[32] + A[41] + A[50] + A[59];
    stencil111[5]  = A[33] + A[58];
    stencil111[6]  = A[43];
    stencil111[7]  = A[35] + A[42];
    stencil111[8]  = A[34];
    stencil111[9]  = A[16] + A[52];
    stencil111[10] = A[17] + A[24] + A[53] + A[60];
    stencil111[11] = A[25] + A[61];
    stencil111[12] = A[19] + A[44] + A[55] + A[8];
    stencil111[13] = 0; // FIXME remove this component from computation
    stencil111[14] = A[1] + A[26] + A[37] + A[62];
    stencil111[15] = A[11] + A[47];
    stencil111[16] = A[10] + A[39] + A[3] + A[46];
    stencil111[17] = A[2] + A[38];
    stencil111[18] = A[20];
    stencil111[19] = A[21] + A[28];
    stencil111[20] = A[29];
    stencil111[21] = A[12] + A[23];
    stencil111[22] = A[13] + A[22] + A[31] + A[4];
    stencil111[23] = A[30] + A[5];
    stencil111[24] = A[15];
    stencil111[25] = A[14] + A[7];
    stencil111[26] = A[6];
    // buffs
    const scalar_t *const in0  = &input[0];
    const scalar_t *const in1  = &input[xstride];
    const scalar_t *const in2  = &input[2 * xstride];
    const scalar_t *const in3  = &input[ystride];
    const scalar_t *const in4  = &input[xstride + ystride];
    const scalar_t *const in5  = &input[2 * xstride + ystride];
    const scalar_t *const in6  = &input[2 * ystride];
    const scalar_t *const in7  = &input[xstride + 2 * ystride];
    const scalar_t *const in8  = &input[2 * xstride + 2 * ystride];
    const scalar_t *const in9  = &input[zstride];
    const scalar_t *const in10 = &input[xstride + zstride];
    const scalar_t *const in11 = &input[2 * xstride + zstride];
    const scalar_t *const in12 = &input[ystride + zstride];
    const scalar_t *const in13 = &input[xstride + ystride + zstride];
    const scalar_t *const in14 = &input[2 * xstride + ystride + zstride];
    const scalar_t *const in15 = &input[2 * ystride + zstride];
    const scalar_t *const in16 = &input[xstride + 2 * ystride + zstride];
    const scalar_t *const in17 = &input[2 * xstride + 2 * ystride + zstride];
    const scalar_t *const in18 = &input[2 * zstride];
    const scalar_t *const in19 = &input[xstride + 2 * zstride];
    const scalar_t *const in20 = &input[2 * xstride + 2 * zstride];
    const scalar_t *const in21 = &input[ystride + 2 * zstride];
    const scalar_t *const in22 = &input[xstride + ystride + 2 * zstride];
    const scalar_t *const in23 = &input[2 * xstride + ystride + 2 * zstride];
    const scalar_t *const in24 = &input[2 * ystride + 2 * zstride];
    const scalar_t *const in25 = &input[xstride + 2 * ystride + 2 * zstride];
    const scalar_t *const in26 = &input[2 * xstride + 2 * ystride + 2 * zstride];
    scalar_t *const       out  = &output[xstride + ystride + zstride];
    for (ptrdiff_t zi = 0; zi < (zc - 2); zi++) {
        for (ptrdiff_t yi = 0; yi < (yc - 2); yi++) {
            for (ptrdiff_t xi = 0; xi < (xc - 2); xi++) {
                const ptrdiff_t idx = xi * xstride + yi * ystride + zi * zstride;
                out[idx] += in0[idx] * stencil111[0];
                out[idx] += in1[idx] * stencil111[1];
                out[idx] += in2[idx] * stencil111[2];
                out[idx] += in3[idx] * stencil111[3];
                out[idx] += in4[idx] * stencil111[4];
                out[idx] += in5[idx] * stencil111[5];
                out[idx] += in6[idx] * stencil111[6];
                out[idx] += in7[idx] * stencil111[7];
                out[idx] += in8[idx] * stencil111[8];
                out[idx] += in9[idx] * stencil111[9];
                out[idx] += in10[idx] * stencil111[10];
                out[idx] += in11[idx] * stencil111[11];
                out[idx] += in12[idx] * stencil111[12];
                out[idx] += in13[idx] * stencil111[13];
                out[idx] += in14[idx] * stencil111[14];
                out[idx] += in15[idx] * stencil111[15];
                out[idx] += in16[idx] * stencil111[16];
                out[idx] += in17[idx] * stencil111[17];
                out[idx] += in18[idx] * stencil111[18];
                out[idx] += in19[idx] * stencil111[19];
                out[idx] += in20[idx] * stencil111[20];
                out[idx] += in21[idx] * stencil111[21];
                out[idx] += in22[idx] * stencil111[22];
                out[idx] += in23[idx] * stencil111[23];
                out[idx] += in24[idx] * stencil111[24];
                out[idx] += in25[idx] * stencil111[25];
                out[idx] += in26[idx] * stencil111[26];
            }
        }
    }
}

//===============
// stencil211)
//===============
static void sshex8_apply_offdiag_stencil211(const ptrdiff_t                     xc,
                                            const ptrdiff_t                     yc,
                                            const ptrdiff_t                     zc,
                                            const ptrdiff_t                     xstride,
                                            const ptrdiff_t                     ystride,
                                            const ptrdiff_t                     zstride,
                                            const scalar_t *const SFEM_RESTRICT A,
                                            const scalar_t *const SFEM_RESTRICT input,
                                            scalar_t *const SFEM_RESTRICT       output) {
    scalar_t stencil211[18];
    stencil211[0]  = A[48];
    stencil211[1]  = A[49];
    stencil211[2]  = A[40] + A[51];
    stencil211[3]  = A[41] + A[50];
    stencil211[4]  = A[43];
    stencil211[5]  = A[42];
    stencil211[6]  = A[16] + A[52];
    stencil211[7]  = A[17] + A[53];
    stencil211[8]  = A[19] + A[44] + A[55] + A[8];
    stencil211[9]  = 0;
    stencil211[10] = A[11] + A[47];
    stencil211[11] = A[10] + A[46];
    stencil211[12] = A[20];
    stencil211[13] = A[21];
    stencil211[14] = A[12] + A[23];
    stencil211[15] = A[13] + A[22];
    stencil211[16] = A[15];
    stencil211[17] = A[14];
    // buffs
    const scalar_t *const in0  = &input[xstride * (xc - 2)];
    const scalar_t *const in1  = &input[xstride * (xc - 1)];
    const scalar_t *const in2  = &input[xstride * (xc - 2) + ystride];
    const scalar_t *const in3  = &input[xstride * (xc - 1) + ystride];
    const scalar_t *const in4  = &input[xstride * (xc - 2) + 2 * ystride];
    const scalar_t *const in5  = &input[xstride * (xc - 1) + 2 * ystride];
    const scalar_t *const in6  = &input[xstride * (xc - 2) + zstride];
    const scalar_t *const in7  = &input[xstride * (xc - 1) + zstride];
    const scalar_t *const in8  = &input[xstride * (xc - 2) + ystride + zstride];
    const scalar_t *const in9  = &input[xstride * (xc - 1) + ystride + zstride];
    const scalar_t *const in10 = &input[xstride * (xc - 2) + 2 * ystride + zstride];
    const scalar_t *const in11 = &input[xstride * (xc - 1) + 2 * ystride + zstride];
    const scalar_t *const in12 = &input[xstride * (xc - 2) + 2 * zstride];
    const scalar_t *const in13 = &input[xstride * (xc - 1) + 2 * zstride];
    const scalar_t *const in14 = &input[xstride * (xc - 2) + ystride + 2 * zstride];
    const scalar_t *const in15 = &input[xstride * (xc - 1) + ystride + 2 * zstride];
    const scalar_t *const in16 = &input[xstride * (xc - 2) + 2 * ystride + 2 * zstride];
    const scalar_t *const in17 = &input[xstride * (xc - 1) + 2 * ystride + 2 * zstride];
    scalar_t *const       out  = &output[xstride * (xc - 1) + ystride + zstride];
    for (ptrdiff_t zi = 0; zi < (zc - 2); zi++) {
        for (ptrdiff_t yi = 0; yi < (yc - 2); yi++) {
            for (ptrdiff_t xi = 0; xi < (1); xi++) {
                const ptrdiff_t idx = xi * xstride + yi * ystride + zi * zstride;
                out[idx] += in0[idx] * stencil211[0];
                out[idx] += in1[idx] * stencil211[1];
                out[idx] += in2[idx] * stencil211[2];
                out[idx] += in3[idx] * stencil211[3];
                out[idx] += in4[idx] * stencil211[4];
                out[idx] += in5[idx] * stencil211[5];
                out[idx] += in6[idx] * stencil211[6];
                out[idx] += in7[idx] * stencil211[7];
                out[idx] += in8[idx] * stencil211[8];
                out[idx] += in9[idx] * stencil211[9];
                out[idx] += in10[idx] * stencil211[10];
                out[idx] += in11[idx] * stencil211[11];
                out[idx] += in12[idx] * stencil211[12];
                out[idx] += in13[idx] * stencil211[13];
                out[idx] += in14[idx] * stencil211[14];
                out[idx] += in15[idx] * stencil211[15];
                out[idx] += in16[idx] * stencil211[16];
                out[idx] += in17[idx] * stencil211[17];
            }
        }
    }
}

//===============
// stencil021)
//===============
static void sshex8_apply_offdiag_stencil021(const ptrdiff_t                     xc,
                                            const ptrdiff_t                     yc,
                                            const ptrdiff_t                     zc,
                                            const ptrdiff_t                     xstride,
                                            const ptrdiff_t                     ystride,
                                            const ptrdiff_t                     zstride,
                                            const scalar_t *const SFEM_RESTRICT A,
                                            const scalar_t *const SFEM_RESTRICT input,
                                            scalar_t *const SFEM_RESTRICT       output) {
    scalar_t stencil021[12];
    stencil021[0]  = A[56];
    stencil021[1]  = A[57];
    stencil021[2]  = A[59];
    stencil021[3]  = A[58];
    stencil021[4]  = A[24] + A[60];
    stencil021[5]  = A[25] + A[61];
    stencil021[6]  = 0;
    stencil021[7]  = A[26] + A[62];
    stencil021[8]  = A[28];
    stencil021[9]  = A[29];
    stencil021[10] = A[31];
    stencil021[11] = A[30];
    // buffs
    const scalar_t *const in0  = &input[ystride * (yc - 2)];
    const scalar_t *const in1  = &input[xstride + ystride * (yc - 2)];
    const scalar_t *const in2  = &input[ystride * (yc - 1)];
    const scalar_t *const in3  = &input[xstride + ystride * (yc - 1)];
    const scalar_t *const in4  = &input[ystride * (yc - 2) + zstride];
    const scalar_t *const in5  = &input[xstride + ystride * (yc - 2) + zstride];
    const scalar_t *const in6  = &input[ystride * (yc - 1) + zstride];
    const scalar_t *const in7  = &input[xstride + ystride * (yc - 1) + zstride];
    const scalar_t *const in8  = &input[ystride * (yc - 2) + 2 * zstride];
    const scalar_t *const in9  = &input[xstride + ystride * (yc - 2) + 2 * zstride];
    const scalar_t *const in10 = &input[ystride * (yc - 1) + 2 * zstride];
    const scalar_t *const in11 = &input[xstride + ystride * (yc - 1) + 2 * zstride];
    scalar_t *const       out  = &output[ystride * (yc - 1) + zstride];
    for (ptrdiff_t zi = 0; zi < (zc - 2); zi++) {
        for (ptrdiff_t yi = 0; yi < (1); yi++) {
            for (ptrdiff_t xi = 0; xi < (1); xi++) {
                const ptrdiff_t idx = xi * xstride + yi * ystride + zi * zstride;
                out[idx] += in0[idx] * stencil021[0];
                out[idx] += in1[idx] * stencil021[1];
                out[idx] += in2[idx] * stencil021[2];
                out[idx] += in3[idx] * stencil021[3];
                out[idx] += in4[idx] * stencil021[4];
                out[idx] += in5[idx] * stencil021[5];
                out[idx] += in6[idx] * stencil021[6];
                out[idx] += in7[idx] * stencil021[7];
                out[idx] += in8[idx] * stencil021[8];
                out[idx] += in9[idx] * stencil021[9];
                out[idx] += in10[idx] * stencil021[10];
                out[idx] += in11[idx] * stencil021[11];
            }
        }
    }
}

//===============
// stencil121)
//===============
static void sshex8_apply_offdiag_stencil121(const ptrdiff_t                     xc,
                                            const ptrdiff_t                     yc,
                                            const ptrdiff_t                     zc,
                                            const ptrdiff_t                     xstride,
                                            const ptrdiff_t                     ystride,
                                            const ptrdiff_t                     zstride,
                                            const scalar_t *const SFEM_RESTRICT A,
                                            const scalar_t *const SFEM_RESTRICT input,
                                            scalar_t *const SFEM_RESTRICT       output) {
    scalar_t stencil121[18];
    stencil121[0]  = A[48];
    stencil121[1]  = A[49] + A[56];
    stencil121[2]  = A[57];
    stencil121[3]  = A[51];
    stencil121[4]  = A[50] + A[59];
    stencil121[5]  = A[58];
    stencil121[6]  = A[16] + A[52];
    stencil121[7]  = A[17] + A[24] + A[53] + A[60];
    stencil121[8]  = A[25] + A[61];
    stencil121[9]  = A[19] + A[55];
    stencil121[10] = 0; // FIXME remove this component from computation
    stencil121[11] = A[26] + A[62];
    stencil121[12] = A[20];
    stencil121[13] = A[21] + A[28];
    stencil121[14] = A[29];
    stencil121[15] = A[23];
    stencil121[16] = A[22] + A[31];
    stencil121[17] = A[30];
    // buffs
    const scalar_t *const in0  = &input[ystride * (yc - 2)];
    const scalar_t *const in1  = &input[xstride + ystride * (yc - 2)];
    const scalar_t *const in2  = &input[2 * xstride + ystride * (yc - 2)];
    const scalar_t *const in3  = &input[ystride * (yc - 1)];
    const scalar_t *const in4  = &input[xstride + ystride * (yc - 1)];
    const scalar_t *const in5  = &input[2 * xstride + ystride * (yc - 1)];
    const scalar_t *const in6  = &input[ystride * (yc - 2) + zstride];
    const scalar_t *const in7  = &input[xstride + ystride * (yc - 2) + zstride];
    const scalar_t *const in8  = &input[2 * xstride + ystride * (yc - 2) + zstride];
    const scalar_t *const in9  = &input[ystride * (yc - 1) + zstride];
    const scalar_t *const in10 = &input[xstride + ystride * (yc - 1) + zstride];
    const scalar_t *const in11 = &input[2 * xstride + ystride * (yc - 1) + zstride];
    const scalar_t *const in12 = &input[ystride * (yc - 2) + 2 * zstride];
    const scalar_t *const in13 = &input[xstride + ystride * (yc - 2) + 2 * zstride];
    const scalar_t *const in14 = &input[2 * xstride + ystride * (yc - 2) + 2 * zstride];
    const scalar_t *const in15 = &input[ystride * (yc - 1) + 2 * zstride];
    const scalar_t *const in16 = &input[xstride + ystride * (yc - 1) + 2 * zstride];
    const scalar_t *const in17 = &input[2 * xstride + ystride * (yc - 1) + 2 * zstride];
    scalar_t *const       out  = &output[xstride + ystride * (yc - 1) + zstride];
    for (ptrdiff_t zi = 0; zi < (zc - 2); zi++) {
        for (ptrdiff_t yi = 0; yi < (1); yi++) {
            for (ptrdiff_t xi = 0; xi < (xc - 2); xi++) {
                const ptrdiff_t idx = xi * xstride + yi * ystride + zi * zstride;
                out[idx] += in0[idx] * stencil121[0];
                out[idx] += in1[idx] * stencil121[1];
                out[idx] += in2[idx] * stencil121[2];
                out[idx] += in3[idx] * stencil121[3];
                out[idx] += in4[idx] * stencil121[4];
                out[idx] += in5[idx] * stencil121[5];
                out[idx] += in6[idx] * stencil121[6];
                out[idx] += in7[idx] * stencil121[7];
                out[idx] += in8[idx] * stencil121[8];
                out[idx] += in9[idx] * stencil121[9];
                out[idx] += in10[idx] * stencil121[10];
                out[idx] += in11[idx] * stencil121[11];
                out[idx] += in12[idx] * stencil121[12];
                out[idx] += in13[idx] * stencil121[13];
                out[idx] += in14[idx] * stencil121[14];
                out[idx] += in15[idx] * stencil121[15];
                out[idx] += in16[idx] * stencil121[16];
                out[idx] += in17[idx] * stencil121[17];
            }
        }
    }
}

//===============
// stencil221)
//===============
static void sshex8_apply_offdiag_stencil221(const ptrdiff_t                     xc,
                                            const ptrdiff_t                     yc,
                                            const ptrdiff_t                     zc,
                                            const ptrdiff_t                     xstride,
                                            const ptrdiff_t                     ystride,
                                            const ptrdiff_t                     zstride,
                                            const scalar_t *const SFEM_RESTRICT A,
                                            const scalar_t *const SFEM_RESTRICT input,
                                            scalar_t *const SFEM_RESTRICT       output) {
    scalar_t stencil221[12];
    stencil221[0]  = A[48];
    stencil221[1]  = A[49];
    stencil221[2]  = A[51];
    stencil221[3]  = A[50];
    stencil221[4]  = A[16] + A[52];
    stencil221[5]  = A[17] + A[53];
    stencil221[6]  = A[19] + A[55];
    stencil221[7]  = 0;
    stencil221[8]  = A[20];
    stencil221[9]  = A[21];
    stencil221[10] = A[23];
    stencil221[11] = A[22];
    // buffs
    const scalar_t *const in0  = &input[xstride * (xc - 2) + ystride * (yc - 2)];
    const scalar_t *const in1  = &input[xstride * (xc - 1) + ystride * (yc - 2)];
    const scalar_t *const in2  = &input[xstride * (xc - 2) + ystride * (yc - 1)];
    const scalar_t *const in3  = &input[xstride * (xc - 1) + ystride * (yc - 1)];
    const scalar_t *const in4  = &input[xstride * (xc - 2) + ystride * (yc - 2) + zstride];
    const scalar_t *const in5  = &input[xstride * (xc - 1) + ystride * (yc - 2) + zstride];
    const scalar_t *const in6  = &input[xstride * (xc - 2) + ystride * (yc - 1) + zstride];
    const scalar_t *const in7  = &input[xstride * (xc - 1) + ystride * (yc - 1) + zstride];
    const scalar_t *const in8  = &input[xstride * (xc - 2) + ystride * (yc - 2) + 2 * zstride];
    const scalar_t *const in9  = &input[xstride * (xc - 1) + ystride * (yc - 2) + 2 * zstride];
    const scalar_t *const in10 = &input[xstride * (xc - 2) + ystride * (yc - 1) + 2 * zstride];
    const scalar_t *const in11 = &input[xstride * (xc - 1) + ystride * (yc - 1) + 2 * zstride];
    scalar_t *const       out  = &output[xstride * (xc - 1) + ystride * (yc - 1) + zstride];
    for (ptrdiff_t zi = 0; zi < (zc - 2); zi++) {
        for (ptrdiff_t yi = 0; yi < (1); yi++) {
            for (ptrdiff_t xi = 0; xi < (1); xi++) {
                const ptrdiff_t idx = xi * xstride + yi * ystride + zi * zstride;
                out[idx] += in0[idx] * stencil221[0];
                out[idx] += in1[idx] * stencil221[1];
                out[idx] += in2[idx] * stencil221[2];
                out[idx] += in3[idx] * stencil221[3];
                out[idx] += in4[idx] * stencil221[4];
                out[idx] += in5[idx] * stencil221[5];
                out[idx] += in6[idx] * stencil221[6];
                out[idx] += in7[idx] * stencil221[7];
                out[idx] += in8[idx] * stencil221[8];
                out[idx] += in9[idx] * stencil221[9];
                out[idx] += in10[idx] * stencil221[10];
                out[idx] += in11[idx] * stencil221[11];
            }
        }
    }
}

//===============
// stencil002)
//===============
static void sshex8_apply_offdiag_stencil002(const ptrdiff_t                     xc,
                                            const ptrdiff_t                     yc,
                                            const ptrdiff_t                     zc,
                                            const ptrdiff_t                     xstride,
                                            const ptrdiff_t                     ystride,
                                            const ptrdiff_t                     zstride,
                                            const scalar_t *const SFEM_RESTRICT A,
                                            const scalar_t *const SFEM_RESTRICT input,
                                            scalar_t *const SFEM_RESTRICT       output) {
    scalar_t stencil002[8];
    stencil002[0] = A[32];
    stencil002[1] = A[33];
    stencil002[2] = A[35];
    stencil002[3] = A[34];
    stencil002[4] = 0; // FIXME remove this component from computation
    stencil002[5] = A[37];
    stencil002[6] = A[39];
    stencil002[7] = A[38];
    // buffs
    const scalar_t *const in0 = &input[zstride * (zc - 2)];
    const scalar_t *const in1 = &input[xstride + zstride * (zc - 2)];
    const scalar_t *const in2 = &input[ystride + zstride * (zc - 2)];
    const scalar_t *const in3 = &input[xstride + ystride + zstride * (zc - 2)];
    const scalar_t *const in4 = &input[zstride * (zc - 1)];
    const scalar_t *const in5 = &input[xstride + zstride * (zc - 1)];
    const scalar_t *const in6 = &input[ystride + zstride * (zc - 1)];
    const scalar_t *const in7 = &input[xstride + ystride + zstride * (zc - 1)];
    scalar_t *const       out = &output[zstride * (zc - 1)];
    for (ptrdiff_t zi = 0; zi < (1); zi++) {
        for (ptrdiff_t yi = 0; yi < (1); yi++) {
            for (ptrdiff_t xi = 0; xi < (1); xi++) {
                const ptrdiff_t idx = xi * xstride + yi * ystride + zi * zstride;
                out[idx] += in0[idx] * stencil002[0];
                out[idx] += in1[idx] * stencil002[1];
                out[idx] += in2[idx] * stencil002[2];
                out[idx] += in3[idx] * stencil002[3];
                out[idx] += in4[idx] * stencil002[4];
                out[idx] += in5[idx] * stencil002[5];
                out[idx] += in6[idx] * stencil002[6];
                out[idx] += in7[idx] * stencil002[7];
            }
        }
    }
}

//===============
// stencil102)
//===============
static void sshex8_apply_offdiag_stencil102(const ptrdiff_t                     xc,
                                            const ptrdiff_t                     yc,
                                            const ptrdiff_t                     zc,
                                            const ptrdiff_t                     xstride,
                                            const ptrdiff_t                     ystride,
                                            const ptrdiff_t                     zstride,
                                            const scalar_t *const SFEM_RESTRICT A,
                                            const scalar_t *const SFEM_RESTRICT input,
                                            scalar_t *const SFEM_RESTRICT       output) {
    scalar_t stencil102[12];
    stencil102[0]  = A[40];
    stencil102[1]  = A[32] + A[41];
    stencil102[2]  = A[33];
    stencil102[3]  = A[43];
    stencil102[4]  = A[35] + A[42];
    stencil102[5]  = A[34];
    stencil102[6]  = A[44];
    stencil102[7]  = 0;
    stencil102[8]  = A[37];
    stencil102[9]  = A[47];
    stencil102[10] = A[39] + A[46];
    stencil102[11] = A[38];
    // buffs
    const scalar_t *const in0  = &input[zstride * (zc - 2)];
    const scalar_t *const in1  = &input[xstride + zstride * (zc - 2)];
    const scalar_t *const in2  = &input[2 * xstride + zstride * (zc - 2)];
    const scalar_t *const in3  = &input[ystride + zstride * (zc - 2)];
    const scalar_t *const in4  = &input[xstride + ystride + zstride * (zc - 2)];
    const scalar_t *const in5  = &input[2 * xstride + ystride + zstride * (zc - 2)];
    const scalar_t *const in6  = &input[zstride * (zc - 1)];
    const scalar_t *const in7  = &input[xstride + zstride * (zc - 1)];
    const scalar_t *const in8  = &input[2 * xstride + zstride * (zc - 1)];
    const scalar_t *const in9  = &input[ystride + zstride * (zc - 1)];
    const scalar_t *const in10 = &input[xstride + ystride + zstride * (zc - 1)];
    const scalar_t *const in11 = &input[2 * xstride + ystride + zstride * (zc - 1)];
    scalar_t *const       out  = &output[xstride + zstride * (zc - 1)];
    for (ptrdiff_t zi = 0; zi < (1); zi++) {
        for (ptrdiff_t yi = 0; yi < (1); yi++) {
            for (ptrdiff_t xi = 0; xi < (xc - 2); xi++) {
                const ptrdiff_t idx = xi * xstride + yi * ystride + zi * zstride;
                out[idx] += in0[idx] * stencil102[0];
                out[idx] += in1[idx] * stencil102[1];
                out[idx] += in2[idx] * stencil102[2];
                out[idx] += in3[idx] * stencil102[3];
                out[idx] += in4[idx] * stencil102[4];
                out[idx] += in5[idx] * stencil102[5];
                out[idx] += in6[idx] * stencil102[6];
                out[idx] += in7[idx] * stencil102[7];
                out[idx] += in8[idx] * stencil102[8];
                out[idx] += in9[idx] * stencil102[9];
                out[idx] += in10[idx] * stencil102[10];
                out[idx] += in11[idx] * stencil102[11];
            }
        }
    }
}

//===============
// stencil202)
//===============
static void sshex8_apply_offdiag_stencil202(const ptrdiff_t                     xc,
                                            const ptrdiff_t                     yc,
                                            const ptrdiff_t                     zc,
                                            const ptrdiff_t                     xstride,
                                            const ptrdiff_t                     ystride,
                                            const ptrdiff_t                     zstride,
                                            const scalar_t *const SFEM_RESTRICT A,
                                            const scalar_t *const SFEM_RESTRICT input,
                                            scalar_t *const SFEM_RESTRICT       output) {
    scalar_t stencil202[8];
    stencil202[0] = A[40];
    stencil202[1] = A[41];
    stencil202[2] = A[43];
    stencil202[3] = A[42];
    stencil202[4] = A[44];
    stencil202[5] = 0; // FIXME remove this component from computation
    stencil202[6] = A[47];
    stencil202[7] = A[46];
    // buffs
    const scalar_t *const in0 = &input[xstride * (xc - 2) + zstride * (zc - 2)];
    const scalar_t *const in1 = &input[xstride * (xc - 1) + zstride * (zc - 2)];
    const scalar_t *const in2 = &input[xstride * (xc - 2) + ystride + zstride * (zc - 2)];
    const scalar_t *const in3 = &input[xstride * (xc - 1) + ystride + zstride * (zc - 2)];
    const scalar_t *const in4 = &input[xstride * (xc - 2) + zstride * (zc - 1)];
    const scalar_t *const in5 = &input[xstride * (xc - 1) + zstride * (zc - 1)];
    const scalar_t *const in6 = &input[xstride * (xc - 2) + ystride + zstride * (zc - 1)];
    const scalar_t *const in7 = &input[xstride * (xc - 1) + ystride + zstride * (zc - 1)];
    scalar_t *const       out = &output[xstride * (xc - 1) + zstride * (zc - 1)];
    for (ptrdiff_t zi = 0; zi < (1); zi++) {
        for (ptrdiff_t yi = 0; yi < (1); yi++) {
            for (ptrdiff_t xi = 0; xi < (1); xi++) {
                const ptrdiff_t idx = xi * xstride + yi * ystride + zi * zstride;
                out[idx] += in0[idx] * stencil202[0];
                out[idx] += in1[idx] * stencil202[1];
                out[idx] += in2[idx] * stencil202[2];
                out[idx] += in3[idx] * stencil202[3];
                out[idx] += in4[idx] * stencil202[4];
                out[idx] += in5[idx] * stencil202[5];
                out[idx] += in6[idx] * stencil202[6];
                out[idx] += in7[idx] * stencil202[7];
            }
        }
    }
}

//===============
// stencil012)
//===============
static void sshex8_apply_offdiag_stencil012(const ptrdiff_t                     xc,
                                            const ptrdiff_t                     yc,
                                            const ptrdiff_t                     zc,
                                            const ptrdiff_t                     xstride,
                                            const ptrdiff_t                     ystride,
                                            const ptrdiff_t                     zstride,
                                            const scalar_t *const SFEM_RESTRICT A,
                                            const scalar_t *const SFEM_RESTRICT input,
                                            scalar_t *const SFEM_RESTRICT       output) {
    scalar_t stencil012[12];
    stencil012[0]  = A[56];
    stencil012[1]  = A[57];
    stencil012[2]  = A[32] + A[59];
    stencil012[3]  = A[33] + A[58];
    stencil012[4]  = A[35];
    stencil012[5]  = A[34];
    stencil012[6]  = A[60];
    stencil012[7]  = A[61];
    stencil012[8]  = 0;
    stencil012[9]  = A[37] + A[62];
    stencil012[10] = A[39];
    stencil012[11] = A[38];
    // buffs
    const scalar_t *const in0  = &input[zstride * (zc - 2)];
    const scalar_t *const in1  = &input[xstride + zstride * (zc - 2)];
    const scalar_t *const in2  = &input[ystride + zstride * (zc - 2)];
    const scalar_t *const in3  = &input[xstride + ystride + zstride * (zc - 2)];
    const scalar_t *const in4  = &input[2 * ystride + zstride * (zc - 2)];
    const scalar_t *const in5  = &input[xstride + 2 * ystride + zstride * (zc - 2)];
    const scalar_t *const in6  = &input[zstride * (zc - 1)];
    const scalar_t *const in7  = &input[xstride + zstride * (zc - 1)];
    const scalar_t *const in8  = &input[ystride + zstride * (zc - 1)];
    const scalar_t *const in9  = &input[xstride + ystride + zstride * (zc - 1)];
    const scalar_t *const in10 = &input[2 * ystride + zstride * (zc - 1)];
    const scalar_t *const in11 = &input[xstride + 2 * ystride + zstride * (zc - 1)];
    scalar_t *const       out  = &output[ystride + zstride * (zc - 1)];
    for (ptrdiff_t zi = 0; zi < (1); zi++) {
        for (ptrdiff_t yi = 0; yi < (yc - 2); yi++) {
            for (ptrdiff_t xi = 0; xi < (1); xi++) {
                const ptrdiff_t idx = xi * xstride + yi * ystride + zi * zstride;
                out[idx] += in0[idx] * stencil012[0];
                out[idx] += in1[idx] * stencil012[1];
                out[idx] += in2[idx] * stencil012[2];
                out[idx] += in3[idx] * stencil012[3];
                out[idx] += in4[idx] * stencil012[4];
                out[idx] += in5[idx] * stencil012[5];
                out[idx] += in6[idx] * stencil012[6];
                out[idx] += in7[idx] * stencil012[7];
                out[idx] += in8[idx] * stencil012[8];
                out[idx] += in9[idx] * stencil012[9];
                out[idx] += in10[idx] * stencil012[10];
                out[idx] += in11[idx] * stencil012[11];
            }
        }
    }
}

//===============
// stencil112)
//===============
static void sshex8_apply_offdiag_stencil112(const ptrdiff_t                     xc,
                                            const ptrdiff_t                     yc,
                                            const ptrdiff_t                     zc,
                                            const ptrdiff_t                     xstride,
                                            const ptrdiff_t                     ystride,
                                            const ptrdiff_t                     zstride,
                                            const scalar_t *const SFEM_RESTRICT A,
                                            const scalar_t *const SFEM_RESTRICT input,
                                            scalar_t *const SFEM_RESTRICT       output) {
    scalar_t stencil112[18];
    stencil112[0]  = A[48];
    stencil112[1]  = A[49] + A[56];
    stencil112[2]  = A[57];
    stencil112[3]  = A[40] + A[51];
    stencil112[4]  = A[32] + A[41] + A[50] + A[59];
    stencil112[5]  = A[33] + A[58];
    stencil112[6]  = A[43];
    stencil112[7]  = A[35] + A[42];
    stencil112[8]  = A[34];
    stencil112[9]  = A[52];
    stencil112[10] = A[53] + A[60];
    stencil112[11] = A[61];
    stencil112[12] = A[44] + A[55];
    stencil112[13] = 0; // FIXME remove this component from computation
    stencil112[14] = A[37] + A[62];
    stencil112[15] = A[47];
    stencil112[16] = A[39] + A[46];
    stencil112[17] = A[38];
    // buffs
    const scalar_t *const in0  = &input[zstride * (zc - 2)];
    const scalar_t *const in1  = &input[xstride + zstride * (zc - 2)];
    const scalar_t *const in2  = &input[2 * xstride + zstride * (zc - 2)];
    const scalar_t *const in3  = &input[ystride + zstride * (zc - 2)];
    const scalar_t *const in4  = &input[xstride + ystride + zstride * (zc - 2)];
    const scalar_t *const in5  = &input[2 * xstride + ystride + zstride * (zc - 2)];
    const scalar_t *const in6  = &input[2 * ystride + zstride * (zc - 2)];
    const scalar_t *const in7  = &input[xstride + 2 * ystride + zstride * (zc - 2)];
    const scalar_t *const in8  = &input[2 * xstride + 2 * ystride + zstride * (zc - 2)];
    const scalar_t *const in9  = &input[zstride * (zc - 1)];
    const scalar_t *const in10 = &input[xstride + zstride * (zc - 1)];
    const scalar_t *const in11 = &input[2 * xstride + zstride * (zc - 1)];
    const scalar_t *const in12 = &input[ystride + zstride * (zc - 1)];
    const scalar_t *const in13 = &input[xstride + ystride + zstride * (zc - 1)];
    const scalar_t *const in14 = &input[2 * xstride + ystride + zstride * (zc - 1)];
    const scalar_t *const in15 = &input[2 * ystride + zstride * (zc - 1)];
    const scalar_t *const in16 = &input[xstride + 2 * ystride + zstride * (zc - 1)];
    const scalar_t *const in17 = &input[2 * xstride + 2 * ystride + zstride * (zc - 1)];
    scalar_t *const       out  = &output[xstride + ystride + zstride * (zc - 1)];
    for (ptrdiff_t zi = 0; zi < (1); zi++) {
        for (ptrdiff_t yi = 0; yi < (yc - 2); yi++) {
            for (ptrdiff_t xi = 0; xi < (xc - 2); xi++) {
                const ptrdiff_t idx = xi * xstride + yi * ystride + zi * zstride;
                out[idx] += in0[idx] * stencil112[0];
                out[idx] += in1[idx] * stencil112[1];
                out[idx] += in2[idx] * stencil112[2];
                out[idx] += in3[idx] * stencil112[3];
                out[idx] += in4[idx] * stencil112[4];
                out[idx] += in5[idx] * stencil112[5];
                out[idx] += in6[idx] * stencil112[6];
                out[idx] += in7[idx] * stencil112[7];
                out[idx] += in8[idx] * stencil112[8];
                out[idx] += in9[idx] * stencil112[9];
                out[idx] += in10[idx] * stencil112[10];
                out[idx] += in11[idx] * stencil112[11];
                out[idx] += in12[idx] * stencil112[12];
                out[idx] += in13[idx] * stencil112[13];
                out[idx] += in14[idx] * stencil112[14];
                out[idx] += in15[idx] * stencil112[15];
                out[idx] += in16[idx] * stencil112[16];
                out[idx] += in17[idx] * stencil112[17];
            }
        }
    }
}

//===============
// stencil212)
//===============
static void sshex8_apply_offdiag_stencil212(const ptrdiff_t                     xc,
                                            const ptrdiff_t                     yc,
                                            const ptrdiff_t                     zc,
                                            const ptrdiff_t                     xstride,
                                            const ptrdiff_t                     ystride,
                                            const ptrdiff_t                     zstride,
                                            const scalar_t *const SFEM_RESTRICT A,
                                            const scalar_t *const SFEM_RESTRICT input,
                                            scalar_t *const SFEM_RESTRICT       output) {
    scalar_t stencil212[12];
    stencil212[0]  = A[48];
    stencil212[1]  = A[49];
    stencil212[2]  = A[40] + A[51];
    stencil212[3]  = A[41] + A[50];
    stencil212[4]  = A[43];
    stencil212[5]  = A[42];
    stencil212[6]  = A[52];
    stencil212[7]  = A[53];
    stencil212[8]  = A[44] + A[55];
    stencil212[9]  = 0;
    stencil212[10] = A[47];
    stencil212[11] = A[46];
    // buffs
    const scalar_t *const in0  = &input[xstride * (xc - 2) + zstride * (zc - 2)];
    const scalar_t *const in1  = &input[xstride * (xc - 1) + zstride * (zc - 2)];
    const scalar_t *const in2  = &input[xstride * (xc - 2) + ystride + zstride * (zc - 2)];
    const scalar_t *const in3  = &input[xstride * (xc - 1) + ystride + zstride * (zc - 2)];
    const scalar_t *const in4  = &input[xstride * (xc - 2) + 2 * ystride + zstride * (zc - 2)];
    const scalar_t *const in5  = &input[xstride * (xc - 1) + 2 * ystride + zstride * (zc - 2)];
    const scalar_t *const in6  = &input[xstride * (xc - 2) + zstride * (zc - 1)];
    const scalar_t *const in7  = &input[xstride * (xc - 1) + zstride * (zc - 1)];
    const scalar_t *const in8  = &input[xstride * (xc - 2) + ystride + zstride * (zc - 1)];
    const scalar_t *const in9  = &input[xstride * (xc - 1) + ystride + zstride * (zc - 1)];
    const scalar_t *const in10 = &input[xstride * (xc - 2) + 2 * ystride + zstride * (zc - 1)];
    const scalar_t *const in11 = &input[xstride * (xc - 1) + 2 * ystride + zstride * (zc - 1)];
    scalar_t *const       out  = &output[xstride * (xc - 1) + ystride + zstride * (zc - 1)];
    for (ptrdiff_t zi = 0; zi < (1); zi++) {
        for (ptrdiff_t yi = 0; yi < (yc - 2); yi++) {
            for (ptrdiff_t xi = 0; xi < (1); xi++) {
                const ptrdiff_t idx = xi * xstride + yi * ystride + zi * zstride;
                out[idx] += in0[idx] * stencil212[0];
                out[idx] += in1[idx] * stencil212[1];
                out[idx] += in2[idx] * stencil212[2];
                out[idx] += in3[idx] * stencil212[3];
                out[idx] += in4[idx] * stencil212[4];
                out[idx] += in5[idx] * stencil212[5];
                out[idx] += in6[idx] * stencil212[6];
                out[idx] += in7[idx] * stencil212[7];
                out[idx] += in8[idx] * stencil212[8];
                out[idx] += in9[idx] * stencil212[9];
                out[idx] += in10[idx] * stencil212[10];
                out[idx] += in11[idx] * stencil212[11];
            }
        }
    }
}

//===============
// stencil022)
//===============
static void sshex8_apply_offdiag_stencil022(const ptrdiff_t                     xc,
                                            const ptrdiff_t                     yc,
                                            const ptrdiff_t                     zc,
                                            const ptrdiff_t                     xstride,
                                            const ptrdiff_t                     ystride,
                                            const ptrdiff_t                     zstride,
                                            const scalar_t *const SFEM_RESTRICT A,
                                            const scalar_t *const SFEM_RESTRICT input,
                                            scalar_t *const SFEM_RESTRICT       output) {
    scalar_t stencil022[8];
    stencil022[0] = A[56];
    stencil022[1] = A[57];
    stencil022[2] = A[59];
    stencil022[3] = A[58];
    stencil022[4] = A[60];
    stencil022[5] = A[61];
    stencil022[6] = 0; // FIXME remove this component from computation
    stencil022[7] = A[62];
    // buffs
    const scalar_t *const in0 = &input[ystride * (yc - 2) + zstride * (zc - 2)];
    const scalar_t *const in1 = &input[xstride + ystride * (yc - 2) + zstride * (zc - 2)];
    const scalar_t *const in2 = &input[ystride * (yc - 1) + zstride * (zc - 2)];
    const scalar_t *const in3 = &input[xstride + ystride * (yc - 1) + zstride * (zc - 2)];
    const scalar_t *const in4 = &input[ystride * (yc - 2) + zstride * (zc - 1)];
    const scalar_t *const in5 = &input[xstride + ystride * (yc - 2) + zstride * (zc - 1)];
    const scalar_t *const in6 = &input[ystride * (yc - 1) + zstride * (zc - 1)];
    const scalar_t *const in7 = &input[xstride + ystride * (yc - 1) + zstride * (zc - 1)];
    scalar_t *const       out = &output[ystride * (yc - 1) + zstride * (zc - 1)];
    for (ptrdiff_t zi = 0; zi < (1); zi++) {
        for (ptrdiff_t yi = 0; yi < (1); yi++) {
            for (ptrdiff_t xi = 0; xi < (1); xi++) {
                const ptrdiff_t idx = xi * xstride + yi * ystride + zi * zstride;
                out[idx] += in0[idx] * stencil022[0];
                out[idx] += in1[idx] * stencil022[1];
                out[idx] += in2[idx] * stencil022[2];
                out[idx] += in3[idx] * stencil022[3];
                out[idx] += in4[idx] * stencil022[4];
                out[idx] += in5[idx] * stencil022[5];
                out[idx] += in6[idx] * stencil022[6];
                out[idx] += in7[idx] * stencil022[7];
            }
        }
    }
}

//===============
// stencil122)
//===============
static void sshex8_apply_offdiag_stencil122(const ptrdiff_t                     xc,
                                            const ptrdiff_t                     yc,
                                            const ptrdiff_t                     zc,
                                            const ptrdiff_t                     xstride,
                                            const ptrdiff_t                     ystride,
                                            const ptrdiff_t                     zstride,
                                            const scalar_t *const SFEM_RESTRICT A,
                                            const scalar_t *const SFEM_RESTRICT input,
                                            scalar_t *const SFEM_RESTRICT       output) {
    scalar_t stencil122[12];
    stencil122[0]  = A[48];
    stencil122[1]  = A[49] + A[56];
    stencil122[2]  = A[57];
    stencil122[3]  = A[51];
    stencil122[4]  = A[50] + A[59];
    stencil122[5]  = A[58];
    stencil122[6]  = A[52];
    stencil122[7]  = A[53] + A[60];
    stencil122[8]  = A[61];
    stencil122[9]  = A[55];
    stencil122[10] = 0; // FIXME remove this component from computation
    stencil122[11] = A[62];
    // buffs
    const scalar_t *const in0  = &input[ystride * (yc - 2) + zstride * (zc - 2)];
    const scalar_t *const in1  = &input[xstride + ystride * (yc - 2) + zstride * (zc - 2)];
    const scalar_t *const in2  = &input[2 * xstride + ystride * (yc - 2) + zstride * (zc - 2)];
    const scalar_t *const in3  = &input[ystride * (yc - 1) + zstride * (zc - 2)];
    const scalar_t *const in4  = &input[xstride + ystride * (yc - 1) + zstride * (zc - 2)];
    const scalar_t *const in5  = &input[2 * xstride + ystride * (yc - 1) + zstride * (zc - 2)];
    const scalar_t *const in6  = &input[ystride * (yc - 2) + zstride * (zc - 1)];
    const scalar_t *const in7  = &input[xstride + ystride * (yc - 2) + zstride * (zc - 1)];
    const scalar_t *const in8  = &input[2 * xstride + ystride * (yc - 2) + zstride * (zc - 1)];
    const scalar_t *const in9  = &input[ystride * (yc - 1) + zstride * (zc - 1)];
    const scalar_t *const in10 = &input[xstride + ystride * (yc - 1) + zstride * (zc - 1)];
    const scalar_t *const in11 = &input[2 * xstride + ystride * (yc - 1) + zstride * (zc - 1)];
    scalar_t *const       out  = &output[xstride + ystride * (yc - 1) + zstride * (zc - 1)];
    for (ptrdiff_t zi = 0; zi < (1); zi++) {
        for (ptrdiff_t yi = 0; yi < (1); yi++) {
            for (ptrdiff_t xi = 0; xi < (xc - 2); xi++) {
                const ptrdiff_t idx = xi * xstride + yi * ystride + zi * zstride;
                out[idx] += in0[idx] * stencil122[0];
                out[idx] += in1[idx] * stencil122[1];
                out[idx] += in2[idx] * stencil122[2];
                out[idx] += in3[idx] * stencil122[3];
                out[idx] += in4[idx] * stencil122[4];
                out[idx] += in5[idx] * stencil122[5];
                out[idx] += in6[idx] * stencil122[6];
                out[idx] += in7[idx] * stencil122[7];
                out[idx] += in8[idx] * stencil122[8];
                out[idx] += in9[idx] * stencil122[9];
                out[idx] += in10[idx] * stencil122[10];
                out[idx] += in11[idx] * stencil122[11];
            }
        }
    }
}

//===============
// stencil222)
//===============
static void sshex8_apply_offdiag_stencil222(const ptrdiff_t                     xc,
                                            const ptrdiff_t                     yc,
                                            const ptrdiff_t                     zc,
                                            const ptrdiff_t                     xstride,
                                            const ptrdiff_t                     ystride,
                                            const ptrdiff_t                     zstride,
                                            const scalar_t *const SFEM_RESTRICT A,
                                            const scalar_t *const SFEM_RESTRICT input,
                                            scalar_t *const SFEM_RESTRICT       output) {
    scalar_t stencil222[8];
    stencil222[0] = A[48];
    stencil222[1] = A[49];
    stencil222[2] = A[51];
    stencil222[3] = A[50];
    stencil222[4] = A[52];
    stencil222[5] = A[53];
    stencil222[6] = A[55];
    stencil222[7] = 0; // FIXME remove this component from computation
    // buffs
    const scalar_t *const in0 = &input[xstride * (xc - 2) + ystride * (yc - 2) + zstride * (zc - 2)];
    const scalar_t *const in1 = &input[xstride * (xc - 1) + ystride * (yc - 2) + zstride * (zc - 2)];
    const scalar_t *const in2 = &input[xstride * (xc - 2) + ystride * (yc - 1) + zstride * (zc - 2)];
    const scalar_t *const in3 = &input[xstride * (xc - 1) + ystride * (yc - 1) + zstride * (zc - 2)];
    const scalar_t *const in4 = &input[xstride * (xc - 2) + ystride * (yc - 2) + zstride * (zc - 1)];
    const scalar_t *const in5 = &input[xstride * (xc - 1) + ystride * (yc - 2) + zstride * (zc - 1)];
    const scalar_t *const in6 = &input[xstride * (xc - 2) + ystride * (yc - 1) + zstride * (zc - 1)];
    const scalar_t *const in7 = &input[xstride * (xc - 1) + ystride * (yc - 1) + zstride * (zc - 1)];
    scalar_t *const       out = &output[xstride * (xc - 1) + ystride * (yc - 1) + zstride * (zc - 1)];
    for (ptrdiff_t zi = 0; zi < (1); zi++) {
        for (ptrdiff_t yi = 0; yi < (1); yi++) {
            for (ptrdiff_t xi = 0; xi < (1); xi++) {
                const ptrdiff_t idx = xi * xstride + yi * ystride + zi * zstride;
                out[idx] += in0[idx] * stencil222[0];
                out[idx] += in1[idx] * stencil222[1];
                out[idx] += in2[idx] * stencil222[2];
                out[idx] += in3[idx] * stencil222[3];
                out[idx] += in4[idx] * stencil222[4];
                out[idx] += in5[idx] * stencil222[5];
                out[idx] += in6[idx] * stencil222[6];
                out[idx] += in7[idx] * stencil222[7];
            }
        }
    }
}

static void sshex8_surface_offdiag_stencil(const ptrdiff_t                     xc,
                                           const ptrdiff_t                     yc,
                                           const ptrdiff_t                     zc,
                                           const ptrdiff_t                     xstride,
                                           const ptrdiff_t                     ystride,
                                           const ptrdiff_t                     zstride,
                                           const scalar_t *const SFEM_RESTRICT A,
                                           const scalar_t *const SFEM_RESTRICT input,
                                           scalar_t *const SFEM_RESTRICT       output) {
    sshex8_apply_offdiag_stencil000(xc, yc, zc, xstride, ystride, zstride, A, input, output);
    sshex8_apply_offdiag_stencil100(xc, yc, zc, xstride, ystride, zstride, A, input, output);
    sshex8_apply_offdiag_stencil200(xc, yc, zc, xstride, ystride, zstride, A, input, output);
    sshex8_apply_offdiag_stencil010(xc, yc, zc, xstride, ystride, zstride, A, input, output);
    sshex8_apply_offdiag_stencil110(xc, yc, zc, xstride, ystride, zstride, A, input, output);
    sshex8_apply_offdiag_stencil210(xc, yc, zc, xstride, ystride, zstride, A, input, output);
    sshex8_apply_offdiag_stencil020(xc, yc, zc, xstride, ystride, zstride, A, input, output);
    sshex8_apply_offdiag_stencil120(xc, yc, zc, xstride, ystride, zstride, A, input, output);
    sshex8_apply_offdiag_stencil220(xc, yc, zc, xstride, ystride, zstride, A, input, output);
    sshex8_apply_offdiag_stencil001(xc, yc, zc, xstride, ystride, zstride, A, input, output);
    sshex8_apply_offdiag_stencil101(xc, yc, zc, xstride, ystride, zstride, A, input, output);
    sshex8_apply_offdiag_stencil201(xc, yc, zc, xstride, ystride, zstride, A, input, output);
    sshex8_apply_offdiag_stencil011(xc, yc, zc, xstride, ystride, zstride, A, input, output);
    // sshex8_apply_offdiag_stencil111(xc, yc, zc, xstride, ystride, zstride, A, input, output);
    sshex8_apply_offdiag_stencil211(xc, yc, zc, xstride, ystride, zstride, A, input, output);
    sshex8_apply_offdiag_stencil021(xc, yc, zc, xstride, ystride, zstride, A, input, output);
    sshex8_apply_offdiag_stencil121(xc, yc, zc, xstride, ystride, zstride, A, input, output);
    sshex8_apply_offdiag_stencil221(xc, yc, zc, xstride, ystride, zstride, A, input, output);
    sshex8_apply_offdiag_stencil002(xc, yc, zc, xstride, ystride, zstride, A, input, output);
    sshex8_apply_offdiag_stencil102(xc, yc, zc, xstride, ystride, zstride, A, input, output);
    sshex8_apply_offdiag_stencil202(xc, yc, zc, xstride, ystride, zstride, A, input, output);
    sshex8_apply_offdiag_stencil012(xc, yc, zc, xstride, ystride, zstride, A, input, output);
    sshex8_apply_offdiag_stencil112(xc, yc, zc, xstride, ystride, zstride, A, input, output);
    sshex8_apply_offdiag_stencil212(xc, yc, zc, xstride, ystride, zstride, A, input, output);
    sshex8_apply_offdiag_stencil022(xc, yc, zc, xstride, ystride, zstride, A, input, output);
    sshex8_apply_offdiag_stencil122(xc, yc, zc, xstride, ystride, zstride, A, input, output);
    sshex8_apply_offdiag_stencil222(xc, yc, zc, xstride, ystride, zstride, A, input, output);
}

#endif