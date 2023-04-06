#include "p2_laplacian.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

#include <mpi.h>

#include "crs_graph.h"
#include "sortreduce.h"

#include "sfem_vec.h"

#define POW2(a) ((a) * (a))

static SFEM_INLINE void p2_laplacian_hessian(const real_t px0,
                                             const real_t px1,
                                             const real_t px2,
                                             const real_t px3,
                                             const real_t py0,
                                             const real_t py1,
                                             const real_t py2,
                                             const real_t py3,
                                             const real_t pz0,
                                             const real_t pz1,
                                             const real_t pz2,
                                             const real_t pz3,
                                             real_t *element_matrix) {
    static const int stride = 1;

    real_t fff[6];

    {
        //      - Result: 6*ADD + 6*ASSIGNMENT + 24*MUL + 9*POW
        //      - Subexpressions: 4*ADD + 6*DIV + 28*MUL + NEG + POW + 24*SUB
        const real_t x0 = -px0 + px1;
        const real_t x1 = -py0 + py2;
        const real_t x2 = -pz0 + pz3;
        const real_t x3 = x1 * x2;
        const real_t x4 = x0 * x3;
        const real_t x5 = -py0 + py3;
        const real_t x6 = -pz0 + pz2;
        const real_t x7 = x5 * x6;
        const real_t x8 = x0 * x7;
        const real_t x9 = -py0 + py1;
        const real_t x10 = -px0 + px2;
        const real_t x11 = x10 * x2;
        const real_t x12 = x11 * x9;
        const real_t x13 = -pz0 + pz1;
        const real_t x14 = x10 * x5;
        const real_t x15 = x13 * x14;
        const real_t x16 = -px0 + px3;
        const real_t x17 = x16 * x6 * x9;
        const real_t x18 = x1 * x16;
        const real_t x19 = x13 * x18;
        const real_t x20 = -1.0 / 6.0 * x12 + (1.0 / 6.0) * x15 + (1.0 / 6.0) * x17 - 1.0 / 6.0 * x19 +
                           (1.0 / 6.0) * x4 - 1.0 / 6.0 * x8;
        const real_t x21 = x14 - x18;
        const real_t x22 = 1. / POW2(-x12 + x15 + x17 - x19 + x4 - x8);
        const real_t x23 = -x11 + x16 * x6;
        const real_t x24 = x3 - x7;
        const real_t x25 = -x0 * x5 + x16 * x9;
        const real_t x26 = x21 * x22;
        const real_t x27 = x0 * x2 - x13 * x16;
        const real_t x28 = x22 * x23;
        const real_t x29 = x13 * x5 - x2 * x9;
        const real_t x30 = x22 * x24;
        const real_t x31 = x0 * x1 - x10 * x9;
        const real_t x32 = -x0 * x6 + x10 * x13;
        const real_t x33 = -x1 * x13 + x6 * x9;
        fff[0 * stride] = x20 * (POW2(x21) * x22 + x22 * POW2(x23) + x22 * POW2(x24));
        fff[1 * stride] = x20 * (x25 * x26 + x27 * x28 + x29 * x30);
        fff[2 * stride] = x20 * (x26 * x31 + x28 * x32 + x30 * x33);
        fff[3 * stride] = x20 * (x22 * POW2(x25) + x22 * POW2(x27) + x22 * POW2(x29));
        fff[4 * stride] = x20 * (x22 * x25 * x31 + x22 * x27 * x32 + x22 * x29 * x33);
        fff[5 * stride] = x20 * (x22 * POW2(x31) + x22 * POW2(x32) + x22 * POW2(x33));
    }

    {
        // FLOATING POINT OPS!
        //       - Result: 69*ADD + 100*ASSIGNMENT + 149*MUL
        //       - Subexpressions: 38*ADD + 69*MUL + 7*NEG + 13*SUB
        const real_t x0 = 0.033333333333333909 * fff[1 * stride];
        const real_t x1 = 0.033333333333333909 * fff[3 * stride];
        const real_t x2 = x0 + x1;
        const real_t x3 = 0.033333333333333326 * fff[5 * stride];
        const real_t x4 = 0.033333333333333326 * fff[3 * stride];
        const real_t x5 = 0.066666666666666652 * fff[4 * stride] + x3 + x4;
        const real_t x6 = 0.033333333333333909 * fff[0 * stride];
        const real_t x7 = 0.033333333333333909 * fff[2 * stride];
        const real_t x8 = 0.033333333333333326 * fff[4 * stride];
        const real_t x9 = x7 + x8;
        const real_t x10 = 0.066666666666667235 * fff[1 * stride] + x4 + x6 + x9;
        const real_t x11 = 0.033333333333333909 * fff[5 * stride];
        const real_t x12 = 0.066666666666667818 * fff[2 * stride] + x11 + x6;
        const real_t x13 = 0.033333333333333048 * fff[0 * stride];
        const real_t x14 = 0.033333333333333048 * fff[3 * stride];
        const real_t x15 = -0.066666666666666097 * fff[1 * stride] - 0.16666666666666557 * fff[2 * stride] -
                           0.16666666666666557 * fff[4 * stride] - 0.13333333333333253 * fff[5 * stride] - x13 - x14;
        const real_t x16 = 0.033333333333333048 * fff[1 * stride];
        const real_t x17 = x3 + x8;
        const real_t x18 = 0.066666666666666374 * fff[2 * stride] + x13 + x16 + x17;
        const real_t x19 = x11 + x7;
        const real_t x20 = -0.033333333333332771 * fff[1 * stride];
        const real_t x21 = 0.033333333333333881 * fff[2 * stride];
        const real_t x22 = -x21;
        const real_t x23 = 0.10000000000000009 * fff[1 * stride];
        const real_t x24 = 0.10000000000000009 * fff[2 * stride];
        const real_t x25 = x23 + x24;
        const real_t x26 = 0.033333333333332327 * fff[0 * stride];
        const real_t x27 = 0.033333333333332327 * fff[2 * stride];
        const real_t x28 = x26 + x27;
        const real_t x29 = 0.033333333333334547 * fff[0 * stride];
        const real_t x30 = 0.033333333333333881 * fff[4 * stride];
        const real_t x31 = 0.033333333333333881 * fff[1 * stride];
        const real_t x32 = 0.033333333333333881 * fff[3 * stride];
        const real_t x33 = x31 + x32;
        const real_t x34 = -x30;
        const real_t x35 = 0.033333333333333215 * fff[4 * stride];
        const real_t x36 = 0.033333333333333215 * fff[3 * stride];
        const real_t x37 = x35 + x36;
        const real_t x38 = 0.10000000000000053 * fff[1 * stride];
        const real_t x39 = -x36 + x38;
        const real_t x40 = 0.10000000000000053 * fff[4 * stride];
        const real_t x41 = x38 + x40;
        const real_t x42 = -x40;
        const real_t x43 = 0.033333333333333437 * fff[4 * stride];
        const real_t x44 = 0.033333333333333437 * fff[5 * stride];
        const real_t x45 = 0.033333333333333881 * fff[5 * stride];
        const real_t x46 = 9.9920072216264089e-16 * fff[1 * stride];
        const real_t x47 = 0.26666666666666661 * fff[3 * stride];
        const real_t x48 = 0.26666666666666661 * fff[5 * stride];
        const real_t x49 = 1.1102230246251565e-16 * fff[0 * stride];
        const real_t x50 = 0.26666666666666661 * fff[4 * stride];
        const real_t x51 = 0.13333333333333353 * fff[2 * stride] + x50;
        const real_t x52 = -0.26666666666666683 * fff[1 * stride] - x47 - x49 - x51;
        const real_t x53 = 0.13333333333333353 * fff[5 * stride];
        const real_t x54 = 0.13333333333333353 * fff[4 * stride];
        const real_t x55 = 0.13333333333333364 * fff[2 * stride] + x49 + x53 + x54;
        const real_t x56 = 9.9920072216264089e-16 * fff[0 * stride];
        const real_t x57 = 0.26666666666666583 * fff[2 * stride];
        const real_t x58 = 0.13333333333333341 * fff[3 * stride];
        const real_t x59 = 0.13333333333333341 * fff[4 * stride];
        const real_t x60 = 0.13333333333333341 * fff[1 * stride];
        const real_t x61 = -0.26666666666666672 * fff[2 * stride] - x48 - x50 - x56 - x60;
        const real_t x62 = 1.1102230246251565e-16 * fff[2 * stride] + x53;
        const real_t x63 = 0.26666666666666705 * fff[0 * stride];
        const real_t x64 = 0.26666666666666705 * fff[1 * stride];
        const real_t x65 = x63 + x64;
        const real_t x66 = 0.26666666666666705 * fff[2 * stride];
        const real_t x67 = x54 + x66;
        const real_t x68 = x63 + x67;
        const real_t x69 = 0.13333333333333272 * fff[0 * stride];
        const real_t x70 = x51 + x60 + x69;
        const real_t x71 = 0.26666666666666705 * fff[5 * stride];
        const real_t x72 = 0.13333333333333275 * fff[0 * stride];
        const real_t x73 = 0.13333333333333275 * fff[1 * stride];
        const real_t x74 = 0.13333333333333275 * fff[2 * stride];
        const real_t x75 = 0.26666666666666516 * fff[4 * stride] + x72 + x73 + x74;
        const real_t x76 = 0.26666666666666627 * fff[2 * stride];
        const real_t x77 = x53 + x76;
        const real_t x78 = x66 + x71 + x73;
        const real_t x79 = 0.033333333333333437 * fff[1 * stride];
        const real_t x80 = 0.1333333333333333 * fff[3 * stride];
        const real_t x81 = 0.1333333333333333 * fff[4 * stride];
        const real_t x82 = 0.26666666666666705 * fff[3 * stride];
        const real_t x83 = x64 + x74;
        const real_t x84 = x82 + x83;
        const real_t x85 = 0.26666666666666744 * fff[0 * stride];
        const real_t x86 = 0.26666666666666744 * fff[1 * stride] + x59;
        element_matrix[0 * stride] = (1.0 / 10.0) * fff[0 * stride] + (1.0 / 5.0) * fff[1 * stride] +
                                     (1.0 / 5.0) * fff[2 * stride] + (1.0 / 10.0) * fff[3 * stride] +
                                     (1.0 / 5.0) * fff[4 * stride] + (1.0 / 10.0) * fff[5 * stride];
        element_matrix[1 * stride] = 0.033333333333333659 * fff[0 * stride] + 0.033333333333333659 * fff[1 * stride] +
                                     0.033333333333333659 * fff[2 * stride];
        element_matrix[2 * stride] = 0.033333333333333909 * fff[4 * stride] + x2;
        element_matrix[3 * stride] = 0.033333333333332604 * fff[2 * stride] + 0.033333333333332604 * fff[4 * stride] +
                                     0.033333333333332604 * fff[5 * stride];
        element_matrix[4 * stride] = -0.13333333333333186 * fff[0 * stride] - 0.16666666666666519 * fff[1 * stride] -
                                     0.16666666666666519 * fff[2 * stride] - x5;
        element_matrix[5 * stride] = x10;
        element_matrix[6 * stride] = -0.1666666666666661 * fff[1 * stride] - 0.13333333333333219 * fff[3 * stride] -
                                     0.1666666666666661 * fff[4 * stride] - x12;
        element_matrix[7 * stride] = x15;
        element_matrix[8 * stride] = x18;
        element_matrix[9 * stride] = 0.066666666666666957 * fff[4 * stride] + x14 + x16 + x19;
        element_matrix[10 * stride] = 0.033333333333334103 * fff[0 * stride] + 0.033333333333334103 * fff[1 * stride] +
                                      0.033333333333334103 * fff[2 * stride];
        element_matrix[11 * stride] = 0.099999999999999645 * fff[0 * stride];
        element_matrix[12 * stride] = x20;
        element_matrix[13 * stride] = x22;
        element_matrix[14 * stride] = -0.13333333333333286 * fff[0 * stride] - x25;
        element_matrix[15 * stride] = 0.099999999999999645 * fff[1 * stride] - x26;
        element_matrix[16 * stride] = x28;
        element_matrix[17 * stride] = 0.033333333333334991 * fff[0 * stride] + 0.033333333333334991 * fff[1 * stride];
        element_matrix[18 * stride] = 0.099999999999999645 * fff[2 * stride] - x29;
        element_matrix[19 * stride] = -0.033333333333334547 * fff[1 * stride] - x27;
        element_matrix[20 * stride] = x30 + x33;
        element_matrix[21 * stride] = x20;
        element_matrix[22 * stride] = 0.10000000000000053 * fff[3 * stride];
        element_matrix[23 * stride] = x34;
        element_matrix[24 * stride] = -8.8817841970012523e-16 * fff[1 * stride] + x37;
        element_matrix[25 * stride] = x39;
        element_matrix[26 * stride] = -0.13333333333333441 * fff[3 * stride] - x41;
        element_matrix[27 * stride] = x33;
        element_matrix[28 * stride] = -x31 - x35;
        element_matrix[29 * stride] = -x32 - x42;
        element_matrix[30 * stride] = 0.033333333333332715 * fff[2 * stride] + 0.033333333333332715 * fff[4 * stride] +
                                      0.033333333333332715 * fff[5 * stride];
        element_matrix[31 * stride] = x22;
        element_matrix[32 * stride] = x34;
        element_matrix[33 * stride] = 0.10000000000000164 * fff[5 * stride];
        element_matrix[34 * stride] = -8.8817841970012523e-16 * fff[2 * stride] + x43 + x44;
        element_matrix[35 * stride] = -x21 - x43;
        element_matrix[36 * stride] = -7.1054273576010023e-16 * fff[4 * stride] + x21 + x45;
        element_matrix[37 * stride] = -0.10000000000000026 * fff[2 * stride] - 0.10000000000000026 * fff[4 * stride] -
                                      0.13333333333333264 * fff[5 * stride];
        element_matrix[38 * stride] = 0.10000000000000031 * fff[2 * stride] - x44;
        element_matrix[39 * stride] = 0.10000000000000031 * fff[4 * stride] - x45;
        element_matrix[40 * stride] = -0.13333333333333397 * fff[0 * stride] - 0.1666666666666673 * fff[1 * stride] -
                                      0.1666666666666673 * fff[2 * stride] - x5;
        element_matrix[41 * stride] = -0.13333333333333375 * fff[0 * stride] - x25;
        element_matrix[42 * stride] = x37 - x46;
        element_matrix[43 * stride] = 5.5511151231257827e-16 * fff[2 * stride] + x17;
        element_matrix[44 * stride] = 0.26666666666666305 * fff[0 * stride] + 0.26666666666666661 * fff[1 * stride] +
                                      0.26666666666666661 * fff[2 * stride] + 0.53333333333333321 * fff[4 * stride] +
                                      x47 + x48;
        element_matrix[45 * stride] = x52;
        element_matrix[46 * stride] = 0.26666666666667171 * fff[1 * stride] + x55;
        element_matrix[47 * stride] = 0.13333333333333441 * fff[1 * stride] + x56 + x57 + x58 + x59;
        element_matrix[48 * stride] = x61;
        element_matrix[49 * stride] = -0.26666666666666694 * fff[4 * stride] - x46 - x58 - x62;
        element_matrix[50 * stride] = x10;
        element_matrix[51 * stride] = x23 - x26;
        element_matrix[52 * stride] = x39;
        element_matrix[53 * stride] = -x9;
        element_matrix[54 * stride] = x52;
        element_matrix[55 * stride] = x47 + x65;
        element_matrix[56 * stride] = -0.26666666666666794 * fff[1 * stride] - x68;
        element_matrix[57 * stride] = -0.26666666666666616 * fff[1 * stride] - x58 - x69;
        element_matrix[58 * stride] = x70;
        element_matrix[59 * stride] = 0.13333333333333272 * fff[1 * stride] + x58 + x67;
        element_matrix[60 * stride] = -0.16666666666666824 * fff[1 * stride] - 0.13333333333333433 * fff[3 * stride] -
                                      0.16666666666666824 * fff[4 * stride] - x12;
        element_matrix[61 * stride] = x28;
        element_matrix[62 * stride] = -0.1333333333333343 * fff[3 * stride] - x41;
        element_matrix[63 * stride] = x19;
        element_matrix[64 * stride] = 0.26666666666666927 * fff[1 * stride] + x55;
        element_matrix[65 * stride] = -0.26666666666666783 * fff[1 * stride] - x68;
        element_matrix[66 * stride] = 0.26666666666666861 * fff[1 * stride] + 0.5333333333333341 * fff[2 * stride] +
                                      0.26666666666666861 * fff[3 * stride] + 0.26666666666666861 * fff[4 * stride] +
                                      x63 + x71;
        element_matrix[67 * stride] = x75;
        element_matrix[68 * stride] = -x72 - x77;
        element_matrix[69 * stride] = -0.26666666666666705 * fff[4 * stride] - x78;
        element_matrix[70 * stride] = x15;
        element_matrix[71 * stride] = 0.033333333333333437 * fff[0 * stride] + x79;
        element_matrix[72 * stride] = x2;
        element_matrix[73 * stride] =
            -0.10000000000000053 * fff[2 * stride] - 0.13333333333333297 * fff[5 * stride] - x40;
        element_matrix[74 * stride] =
            7.7715611723760958e-16 * fff[0 * stride] + 0.13333333333333408 * fff[1 * stride] + x57 + x80 + x81;
        element_matrix[75 * stride] = -0.26666666666666605 * fff[1 * stride] - x72 - x80;
        element_matrix[76 * stride] = x75;
        element_matrix[77 * stride] = 0.5333333333333341 * fff[1 * stride] + 0.26666666666666594 * fff[2 * stride] +
                                      0.26666666666666594 * fff[4 * stride] + 0.26666666666666505 * fff[5 * stride] +
                                      x63 + x82;
        element_matrix[78 * stride] = -x65 - x76 - x81;
        element_matrix[79 * stride] = -0.26666666666666572 * fff[4 * stride] - x84;
        element_matrix[80 * stride] = x18;
        element_matrix[81 * stride] = x24 - x29;
        element_matrix[82 * stride] = -x0 - x35;
        element_matrix[83 * stride] = 0.10000000000000092 * fff[2 * stride] - x3;
        element_matrix[84 * stride] = x61;
        element_matrix[85 * stride] = x70;
        element_matrix[86 * stride] = -x69 - x77;
        element_matrix[87 * stride] = -0.26666666666666605 * fff[2 * stride] - x85 - x86;
        element_matrix[88 * stride] = 0.26666666666666683 * fff[2 * stride] + x48 + x85;
        element_matrix[89 * stride] = 0.13333333333333272 * fff[2 * stride] + x53 + x86;
        element_matrix[90 * stride] = 0.033333333333333104 * fff[1 * stride] + 0.033333333333333104 * fff[3 * stride] +
                                      0.066666666666667013 * fff[4 * stride] + x19;
        element_matrix[91 * stride] = -x27 - x79;
        element_matrix[92 * stride] = -x1 - x42;
        element_matrix[93 * stride] = -x11 + x40;
        element_matrix[94 * stride] =
            -1.3322676295501878e-15 * fff[1 * stride] - 0.26666666666666683 * fff[4 * stride] - x62 - x80;
        element_matrix[95 * stride] = x67 + x73 + x80;
        element_matrix[96 * stride] = -0.26666666666666716 * fff[4 * stride] - x78;
        element_matrix[97 * stride] = -0.26666666666666539 * fff[4 * stride] - x84;
        element_matrix[98 * stride] = x53 + x81 + x83;
        element_matrix[99 * stride] = 0.2666666666666655 * fff[4 * stride] + x71 + x82;
    }
}

static SFEM_INLINE void p2_laplacian_gradient(const real_t px0,
                                              const real_t px1,
                                              const real_t px2,
                                              const real_t px3,
                                              const real_t py0,
                                              const real_t py1,
                                              const real_t py2,
                                              const real_t py3,
                                              const real_t pz0,
                                              const real_t pz1,
                                              const real_t pz2,
                                              const real_t pz3,
                                              const real_t *SFEM_RESTRICT u,
                                              real_t *SFEM_RESTRICT element_vector) {
    assert(false);
}

static SFEM_INLINE void p2_laplacian_value(const real_t px0,
                                           const real_t px1,
                                           const real_t px2,
                                           const real_t px3,
                                           const real_t py0,
                                           const real_t py1,
                                           const real_t py2,
                                           const real_t py3,
                                           const real_t pz0,
                                           const real_t pz1,
                                           const real_t pz2,
                                           const real_t pz3,
                                           const real_t *SFEM_RESTRICT u,
                                           real_t *SFEM_RESTRICT element_scalar) {
    assert(false);
}

static SFEM_INLINE int linear_search(const idx_t target, const idx_t *const arr, const int size) {
    int i;
    for (i = 0; i < size - SFEM_VECTOR_SIZE; i += SFEM_VECTOR_SIZE) {
        if (arr[i] == target) return i;
        if (arr[i + 1] == target) return i + 1;
        if (arr[i + 2] == target) return i + 2;
        if (arr[i + 3] == target) return i + 3;
    }
    for (; i < size; i++) {
        if (arr[i] == target) return i;
    }
    return -1;
}

static SFEM_INLINE int find_col(const idx_t key, const idx_t *const row, const int lenrow) {
    if (lenrow <= 32) {
        return linear_search(key, row, lenrow);

        // Using sentinel (potentially dangerous if matrix is buggy and column does not exist)
        // while (key > row[++k]) {
        //     // Hi
        // }
        // assert(k < lenrow);
        // assert(key == row[k]);
    } else {
        // Use this for larger number of dofs per row
        return find_idx_binary_search(key, row, lenrow);
    }
}

static SFEM_INLINE void find_cols10(const idx_t *targets, const idx_t *const row, const int lenrow, int *ks) {
    if (lenrow > 32) {
        for (int d = 0; d < 10; ++d) {
            ks[d] = find_col(targets[d], row, lenrow);
        }
    } else {
#pragma unroll(10)
        for (int d = 0; d < 10; ++d) {
            ks[d] = 0;
        }

        for (int i = 0; i < lenrow; ++i) {
#pragma unroll(10)
            for (int d = 0; d < 10; ++d) {
                ks[d] += row[i] < targets[d];
            }
        }
    }
}

void p2_laplacian_assemble_hessian(const ptrdiff_t nelements,
                                   const ptrdiff_t nnodes,
                                   idx_t **const SFEM_RESTRICT elems,
                                   geom_t **const SFEM_RESTRICT xyz,
                                   const count_t *const SFEM_RESTRICT rowptr,
                                   const idx_t *const SFEM_RESTRICT colidx,
                                   real_t *const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

    double tick = MPI_Wtime();

    idx_t ev[10];
    idx_t ks[10];

    real_t element_matrix[10 * 10];

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            ev[v] = elems[v][i];
        }

        // Element indices for affine coordinates
        const idx_t i0 = ev[0];
        const idx_t i1 = ev[1];
        const idx_t i2 = ev[2];
        const idx_t i3 = ev[3];

        p2_laplacian_hessian(
            // X-coordinates
            xyz[0][i0],
            xyz[0][i1],
            xyz[0][i2],
            xyz[0][i3],
            // Y-coordinates
            xyz[1][i0],
            xyz[1][i1],
            xyz[1][i2],
            xyz[1][i3],
            // Z-coordinates
            xyz[2][i0],
            xyz[2][i1],
            xyz[2][i2],
            xyz[2][i3],
            element_matrix);

        for (int edof_i = 0; edof_i < 10; ++edof_i) {
            const idx_t dof_i = elems[edof_i][i];
            const idx_t lenrow = rowptr[dof_i + 1] - rowptr[dof_i];

            const idx_t *row = &colidx[rowptr[dof_i]];

            find_cols10(ev, row, lenrow, ks);

            real_t *rowvalues = &values[rowptr[dof_i]];
            const real_t *element_row = &element_matrix[edof_i * 10];

#pragma unroll(10)
            for (int edof_j = 0; edof_j < 10; ++edof_j) {
                rowvalues[ks[edof_j]] += element_row[edof_j];
            }
        }
    }

    double tock = MPI_Wtime();
    printf("p2_laplacian.c: p2_laplacian_assemble_hessian\t%g seconds\n", tock - tick);
}

void p2_laplacian_assemble_gradient(const ptrdiff_t nelements,
                                    const ptrdiff_t nnodes,
                                    idx_t **const SFEM_RESTRICT elems,
                                    geom_t **const SFEM_RESTRICT xyz,
                                    const real_t *const SFEM_RESTRICT u,
                                    real_t *const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

    double tick = MPI_Wtime();

    idx_t ev[10];
    real_t element_vector[10 * 10];
    real_t element_u[10];

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            ev[v] = elems[v][i];
        }

        for (int v = 0; v < 10; ++v) {
            element_u[v] = u[ev[v]];
        }

        // Element indices
        const idx_t i0 = ev[0];
        const idx_t i1 = ev[1];
        const idx_t i2 = ev[2];
        const idx_t i3 = ev[3];

        p2_laplacian_gradient(
            // X-coordinates
            xyz[0][i0],
            xyz[0][i1],
            xyz[0][i2],
            xyz[0][i3],
            // Y-coordinates
            xyz[1][i0],
            xyz[1][i1],
            xyz[1][i2],
            xyz[1][i3],
            // Z-coordinates
            xyz[2][i0],
            xyz[2][i1],
            xyz[2][i2],
            xyz[2][i3],
            element_u,
            element_vector);

        for (int edof_i = 0; edof_i < 10; ++edof_i) {
            const idx_t dof_i = ev[edof_i];
            values[dof_i] += element_vector[edof_i];
        }
    }

    double tock = MPI_Wtime();
    printf("p2_laplacian.c: p2_laplacian_assemble_gradient\t%g seconds\n", tock - tick);
}

void p2_laplacian_assemble_value(const ptrdiff_t nelements,
                                 const ptrdiff_t nnodes,
                                 idx_t **const SFEM_RESTRICT elems,
                                 geom_t **const SFEM_RESTRICT xyz,
                                 const real_t *const SFEM_RESTRICT u,
                                 real_t *const SFEM_RESTRICT value) {
    SFEM_UNUSED(nnodes);

    double tick = MPI_Wtime();

    idx_t ev[10];
    real_t element_u[10];

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            ev[v] = elems[v][i];
        }

        for (int v = 0; v < 10; ++v) {
            element_u[v] = u[ev[v]];
        }

        // Element indices
        const idx_t i0 = ev[0];
        const idx_t i1 = ev[1];
        const idx_t i2 = ev[2];
        const idx_t i3 = ev[3];

        real_t element_scalar = 0;

        p2_laplacian_value(
            // X-coordinates
            xyz[0][i0],
            xyz[0][i1],
            xyz[0][i2],
            xyz[0][i3],
            // Y-coordinates
            xyz[1][i0],
            xyz[1][i1],
            xyz[1][i2],
            xyz[1][i3],
            // Z-coordinates
            xyz[2][i0],
            xyz[2][i1],
            xyz[2][i2],
            xyz[2][i3],
            element_u,
            &element_scalar);

        *value += element_scalar;
    }

    double tock = MPI_Wtime();
    printf("p2_laplacian.c: p2_laplacian_assemble_value\t%g seconds\n", tock - tick);
}

void p2_laplacian_apply(const ptrdiff_t nelements,
                        const ptrdiff_t nnodes,
                        idx_t **const SFEM_RESTRICT elems,
                        geom_t **const SFEM_RESTRICT xyz,
                        const real_t *const SFEM_RESTRICT u,
                        real_t *const SFEM_RESTRICT values) {
    p2_laplacian_assemble_gradient(nelements, nnodes, elems, xyz, u, values);
}
