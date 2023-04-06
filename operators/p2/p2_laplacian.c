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
        //      - Result: 5*ADD + 100*ASSIGNMENT + 6*MUL
        //      - Subexpressions: 34*ADD + 30*DIV + 12*MUL + 19*NEG + 34*SUB
        const real_t x0 = (1.0 / 10.0) * fff[0 * stride];
        const real_t x1 = (1.0 / 10.0) * fff[3 * stride];
        const real_t x2 = (1.0 / 10.0) * fff[5 * stride];
        const real_t x3 = (1.0 / 30.0) * fff[1 * stride];
        const real_t x4 = (1.0 / 30.0) * fff[0 * stride];
        const real_t x5 = (1.0 / 30.0) * fff[2 * stride];
        const real_t x6 = x4 + x5;
        const real_t x7 = x3 + x6;
        const real_t x8 = (1.0 / 30.0) * fff[3 * stride];
        const real_t x9 = (1.0 / 30.0) * fff[4 * stride];
        const real_t x10 = x8 + x9;
        const real_t x11 = x10 + x3;
        const real_t x12 = (1.0 / 30.0) * fff[5 * stride];
        const real_t x13 = x12 + x9;
        const real_t x14 = x13 + x5;
        const real_t x15 = (2.0 / 15.0) * fff[0 * stride];
        const real_t x16 = (1.0 / 6.0) * fff[1 * stride];
        const real_t x17 = (1.0 / 6.0) * fff[2 * stride];
        const real_t x18 = (1.0 / 15.0) * fff[4 * stride] + x12 + x8;
        const real_t x19 = -x15 - x16 - x17 - x18;
        const real_t x20 = (1.0 / 15.0) * fff[1 * stride];
        const real_t x21 = x10 + x20 + x6;
        const real_t x22 = (2.0 / 15.0) * fff[3 * stride];
        const real_t x23 = (1.0 / 15.0) * fff[2 * stride];
        const real_t x24 = (1.0 / 6.0) * fff[4 * stride] + x4;
        const real_t x25 = -x12 - x16 - x22 - x23 - x24;
        const real_t x26 = (2.0 / 15.0) * fff[5 * stride];
        const real_t x27 = -x17 - x20 - x24 - x26 - x8;
        const real_t x28 = x3 + x4;
        const real_t x29 = x13 + x23 + x28;
        const real_t x30 = x3 + x5;
        const real_t x31 = x18 + x30;
        const real_t x32 = -x3;
        const real_t x33 = -x5;
        const real_t x34 = (1.0 / 10.0) * fff[1 * stride];
        const real_t x35 = (1.0 / 10.0) * fff[2 * stride];
        const real_t x36 = -x15 - x34 - x35;
        const real_t x37 = x34 - x4;
        const real_t x38 = x35 - x4;
        const real_t x39 = -x30;
        const real_t x40 = -x9;
        const real_t x41 = x34 - x8;
        const real_t x42 = (1.0 / 10.0) * fff[4 * stride];
        const real_t x43 = -x22 - x34 - x42;
        const real_t x44 = x3 + x8;
        const real_t x45 = -x3 - x9;
        const real_t x46 = x42 - x8;
        const real_t x47 = -x5 - x9;
        const real_t x48 = x12 + x5;
        const real_t x49 = -x26 - x35 - x42;
        const real_t x50 = -x12;
        const real_t x51 = x35 + x50;
        const real_t x52 = x42 + x50;
        const real_t x53 = (4.0 / 15.0) * fff[0 * stride];
        const real_t x54 = (4.0 / 15.0) * fff[1 * stride];
        const real_t x55 = (4.0 / 15.0) * fff[3 * stride];
        const real_t x56 = x54 + x55;
        const real_t x57 = x53 + x56;
        const real_t x58 = (4.0 / 15.0) * fff[2 * stride];
        const real_t x59 = (4.0 / 15.0) * fff[5 * stride];
        const real_t x60 = x58 + x59;
        const real_t x61 = (4.0 / 15.0) * fff[4 * stride];
        const real_t x62 = (2.0 / 15.0) * fff[2 * stride];
        const real_t x63 = x61 + x62;
        const real_t x64 = -x56 - x63;
        const real_t x65 = (2.0 / 15.0) * fff[4 * stride];
        const real_t x66 = x54 + x65;
        const real_t x67 = x26 + x62 + x66;
        const real_t x68 = (2.0 / 15.0) * fff[1 * stride];
        const real_t x69 = x22 + x58 + x65 + x68;
        const real_t x70 = -x60 - x61 - x68;
        const real_t x71 = -x22 - x26 - x61;
        const real_t x72 = -x53 - x58 - x66;
        const real_t x73 = -x15 - x22 - x54;
        const real_t x74 = x15 + x63 + x68;
        const real_t x75 = x59 + x61;
        const real_t x76 = -x15 - x26 - x58;
        const real_t x77 = x53 + x60;
        element_matrix[0 * stride] = (1.0 / 5.0) * fff[1 * stride] + (1.0 / 5.0) * fff[2 * stride] +
                                     (1.0 / 5.0) * fff[4 * stride] + x0 + x1 + x2;
        element_matrix[1 * stride] = x7;
        element_matrix[2 * stride] = x11;
        element_matrix[3 * stride] = x14;
        element_matrix[4 * stride] = x19;
        element_matrix[5 * stride] = x21;
        element_matrix[6 * stride] = x25;
        element_matrix[7 * stride] = x27;
        element_matrix[8 * stride] = x29;
        element_matrix[9 * stride] = x31;
        element_matrix[10 * stride] = x7;
        element_matrix[11 * stride] = x0;
        element_matrix[12 * stride] = x32;
        element_matrix[13 * stride] = x33;
        element_matrix[14 * stride] = x36;
        element_matrix[15 * stride] = x37;
        element_matrix[16 * stride] = x6;
        element_matrix[17 * stride] = x28;
        element_matrix[18 * stride] = x38;
        element_matrix[19 * stride] = x39;
        element_matrix[20 * stride] = x11;
        element_matrix[21 * stride] = x32;
        element_matrix[22 * stride] = x1;
        element_matrix[23 * stride] = x40;
        element_matrix[24 * stride] = x10;
        element_matrix[25 * stride] = x41;
        element_matrix[26 * stride] = x43;
        element_matrix[27 * stride] = x44;
        element_matrix[28 * stride] = x45;
        element_matrix[29 * stride] = x46;
        element_matrix[30 * stride] = x14;
        element_matrix[31 * stride] = x33;
        element_matrix[32 * stride] = x40;
        element_matrix[33 * stride] = x2;
        element_matrix[34 * stride] = x13;
        element_matrix[35 * stride] = x47;
        element_matrix[36 * stride] = x48;
        element_matrix[37 * stride] = x49;
        element_matrix[38 * stride] = x51;
        element_matrix[39 * stride] = x52;
        element_matrix[40 * stride] = x19;
        element_matrix[41 * stride] = x36;
        element_matrix[42 * stride] = x10;
        element_matrix[43 * stride] = x13;
        element_matrix[44 * stride] = (8.0 / 15.0) * fff[4 * stride] + x57 + x60;
        element_matrix[45 * stride] = x64;
        element_matrix[46 * stride] = x67;
        element_matrix[47 * stride] = x69;
        element_matrix[48 * stride] = x70;
        element_matrix[49 * stride] = x71;
        element_matrix[50 * stride] = x21;
        element_matrix[51 * stride] = x37;
        element_matrix[52 * stride] = x41;
        element_matrix[53 * stride] = x47;
        element_matrix[54 * stride] = x64;
        element_matrix[55 * stride] = x57;
        element_matrix[56 * stride] = x72;
        element_matrix[57 * stride] = x73;
        element_matrix[58 * stride] = x74;
        element_matrix[59 * stride] = x69;
        element_matrix[60 * stride] = x25;
        element_matrix[61 * stride] = x6;
        element_matrix[62 * stride] = x43;
        element_matrix[63 * stride] = x48;
        element_matrix[64 * stride] = x67;
        element_matrix[65 * stride] = x72;
        element_matrix[66 * stride] = (8.0 / 15.0) * fff[2 * stride] + x57 + x75;
        element_matrix[67 * stride] = x74;
        element_matrix[68 * stride] = x76;
        element_matrix[69 * stride] = x70;
        element_matrix[70 * stride] = x27;
        element_matrix[71 * stride] = x28;
        element_matrix[72 * stride] = x44;
        element_matrix[73 * stride] = x49;
        element_matrix[74 * stride] = x69;
        element_matrix[75 * stride] = x73;
        element_matrix[76 * stride] = x74;
        element_matrix[77 * stride] = (8.0 / 15.0) * fff[1 * stride] + x55 + x61 + x77;
        element_matrix[78 * stride] = x72;
        element_matrix[79 * stride] = x64;
        element_matrix[80 * stride] = x29;
        element_matrix[81 * stride] = x38;
        element_matrix[82 * stride] = x45;
        element_matrix[83 * stride] = x51;
        element_matrix[84 * stride] = x70;
        element_matrix[85 * stride] = x74;
        element_matrix[86 * stride] = x76;
        element_matrix[87 * stride] = x72;
        element_matrix[88 * stride] = x77;
        element_matrix[89 * stride] = x67;
        element_matrix[90 * stride] = x31;
        element_matrix[91 * stride] = x39;
        element_matrix[92 * stride] = x46;
        element_matrix[93 * stride] = x52;
        element_matrix[94 * stride] = x71;
        element_matrix[95 * stride] = x69;
        element_matrix[96 * stride] = x70;
        element_matrix[97 * stride] = x64;
        element_matrix[98 * stride] = x67;
        element_matrix[99 * stride] = x55 + x75;
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
        //      - Result: 10*ADD + 10*ASSIGNMENT + 78*MUL
        //      - Subexpressions: 60*ADD + 35*DIV + 130*MUL + 18*NEG + 24*SUB
        const real_t x0 = (1.0 / 10.0) * u[0];
        const real_t x1 = (1.0 / 30.0) * fff[0 * stride];
        const real_t x2 = u[1] * x1;
        const real_t x3 = (2.0 / 15.0) * fff[0 * stride];
        const real_t x4 = -u[4] * x3;
        const real_t x5 = u[5] * x1;
        const real_t x6 = u[6] * x1;
        const real_t x7 = u[7] * x1;
        const real_t x8 = u[8] * x1;
        const real_t x9 = (1.0 / 5.0) * u[0];
        const real_t x10 = (1.0 / 30.0) * fff[1 * stride];
        const real_t x11 = u[1] * x10;
        const real_t x12 = u[2] * x10;
        const real_t x13 = (1.0 / 6.0) * fff[1 * stride];
        const real_t x14 = (1.0 / 15.0) * fff[1 * stride];
        const real_t x15 = u[8] * x10;
        const real_t x16 = u[9] * x10;
        const real_t x17 = (1.0 / 30.0) * fff[2 * stride];
        const real_t x18 = u[1] * x17;
        const real_t x19 = u[3] * x17;
        const real_t x20 = (1.0 / 6.0) * fff[2 * stride];
        const real_t x21 = u[5] * x17;
        const real_t x22 = (1.0 / 15.0) * fff[2 * stride];
        const real_t x23 = u[9] * x17;
        const real_t x24 = (1.0 / 30.0) * fff[3 * stride];
        const real_t x25 = u[2] * x24;
        const real_t x26 = u[4] * x24;
        const real_t x27 = u[5] * x24;
        const real_t x28 = (2.0 / 15.0) * fff[3 * stride];
        const real_t x29 = -u[6] * x28;
        const real_t x30 = u[7] * x24;
        const real_t x31 = u[9] * x24;
        const real_t x32 = (1.0 / 30.0) * fff[4 * stride];
        const real_t x33 = u[2] * x32;
        const real_t x34 = u[3] * x32;
        const real_t x35 = (1.0 / 15.0) * fff[4 * stride];
        const real_t x36 = u[5] * x32;
        const real_t x37 = (1.0 / 6.0) * fff[4 * stride];
        const real_t x38 = u[8] * x32;
        const real_t x39 = (1.0 / 30.0) * fff[5 * stride];
        const real_t x40 = u[3] * x39;
        const real_t x41 = u[4] * x39;
        const real_t x42 = u[6] * x39;
        const real_t x43 = (2.0 / 15.0) * fff[5 * stride];
        const real_t x44 = -u[7] * x43;
        const real_t x45 = u[8] * x39;
        const real_t x46 = u[9] * x39;
        const real_t x47 = (1.0 / 10.0) * u[4];
        const real_t x48 = -x12;
        const real_t x49 = (1.0 / 10.0) * u[1];
        const real_t x50 = u[0] * x10;
        const real_t x51 = u[0] * x17;
        const real_t x52 = x50 + x51;
        const real_t x53 = u[0] * x1;
        const real_t x54 = -x19 + x53;
        const real_t x55 = (1.0 / 10.0) * fff[1 * stride];
        const real_t x56 = u[5] * x55 + u[7] * x10;
        const real_t x57 = (1.0 / 10.0) * fff[2 * stride];
        const real_t x58 = u[6] * x17 + u[8] * x57;
        const real_t x59 = (1.0 / 10.0) * fff[4 * stride];
        const real_t x60 = -x34;
        const real_t x61 = u[0] * x32;
        const real_t x62 = x50 + x61;
        const real_t x63 = u[0] * x24;
        const real_t x64 = -x11 + x63;
        const real_t x65 = u[4] * x32 + u[9] * x59;
        const real_t x66 = -x18;
        const real_t x67 = x51 + x61;
        const real_t x68 = u[0] * x39;
        const real_t x69 = -x33 + x68;
        const real_t x70 = (4.0 / 15.0) * u[4];
        const real_t x71 = -fff[1 * stride] * x70;
        const real_t x72 = (4.0 / 15.0) * u[6];
        const real_t x73 = -fff[1 * stride] * x72;
        const real_t x74 = (2.0 / 15.0) * fff[4 * stride];
        const real_t x75 = (2.0 / 15.0) * fff[1 * stride];
        const real_t x76 = (4.0 / 15.0) * u[5];
        const real_t x77 = (4.0 / 15.0) * u[8];
        const real_t x78 = fff[1 * stride] * x49 - fff[3 * stride] * x70 + fff[3 * stride] * x76 +
                           fff[4 * stride] * x77 - u[6] * x74 - u[7] * x28 + u[8] * x75 + u[9] * x28 + x60 + x63;
        const real_t x79 = -fff[2 * stride] * x70;
        const real_t x80 = (4.0 / 15.0) * u[7];
        const real_t x81 = -fff[2 * stride] * x80;
        const real_t x82 = (2.0 / 15.0) * fff[2 * stride];
        const real_t x83 = fff[2 * stride] * x49 + fff[4 * stride] * x76 - fff[5 * stride] * x70 +
                           fff[5 * stride] * x77 + u[5] * x82 - u[6] * x43 - u[7] * x74 + u[9] * x43 + x69 + x79 + x81;
        const real_t x84 = -x25;
        const real_t x85 = (4.0 / 15.0) * fff[4 * stride];
        const real_t x86 = u[9] * x85;
        const real_t x87 = u[0] * x35 + x84 + x86;
        const real_t x88 = fff[2 * stride] * x77;
        const real_t x89 = -2.0 / 15.0 * fff[2 * stride] * u[6] + u[0] * x20 + x88;
        const real_t x90 = fff[1 * stride] * x76;
        const real_t x91 = -x40;
        const real_t x92 = -2.0 / 15.0 * fff[1 * stride] * u[7] + u[0] * x13 + x90 + x91;
        const real_t x93 = u[8] * x82;
        const real_t x94 = -x2;
        const real_t x95 = u[0] * x14 + x84 + x90 + x94;
        const real_t x96 = -fff[1 * stride] * x80;
        const real_t x97 = -fff[2 * stride] * x72;
        const real_t x98 = -u[4] * x85;
        const real_t x99 = u[9] * x74 + x96 + x97 + x98;
        const real_t x100 = (4.0 / 15.0) * u[9];
        const real_t x101 = -fff[0 * stride] * x72 + fff[0 * stride] * x76 + fff[2 * stride] * x100 + u[2] * x55 -
                            u[4] * x82 - u[7] * x3 + u[8] * x3 + u[9] * x75 + x54 + x71 + x73;
        const real_t x102 = -u[6] * x85;
        const real_t x103 = -u[7] * x85;
        const real_t x104 = u[0] * x22 + x88 + x94;
        const real_t x105 = fff[2 * stride] * x76 + fff[5 * stride] * x100 - fff[5 * stride] * x72 + u[2] * x59 -
                            u[4] * x43 + u[5] * x74 - u[7] * x82 + u[8] * x43 + x66 + x68;
        const real_t x106 = -2.0 / 15.0 * fff[4 * stride] * u[4] + u[0] * x37 + x86;
        const real_t x107 = fff[0 * stride] * x77 - fff[0 * stride] * x80 + fff[1 * stride] * x100 + u[3] * x57 -
                            u[4] * x75 + u[5] * x3 - u[6] * x3 + u[9] * x82 + x48 + x53;
        const real_t x108 = fff[1 * stride] * x77 + fff[3 * stride] * x100 - fff[3 * stride] * x80 + u[3] * x59 -
                            u[4] * x28 + u[5] * x28 - u[6] * x75 + u[8] * x74 + x102 + x103 + x64;
        const real_t x109 = u[5] * x75 + x91;
        element_vector[0 * stride] = fff[0 * stride] * x0 + fff[1 * stride] * x9 + fff[2 * stride] * x9 +
                                     fff[3 * stride] * x0 + fff[4 * stride] * x9 + fff[5 * stride] * x0 - u[4] * x13 -
                                     u[4] * x20 - u[4] * x35 + u[5] * x14 - u[6] * x13 - u[6] * x22 - u[6] * x37 -
                                     u[7] * x14 - u[7] * x20 - u[7] * x37 + u[8] * x22 + u[9] * x35 + x11 + x12 + x15 +
                                     x16 + x18 + x19 + x2 + x21 + x23 + x25 - x26 + x27 + x29 - x30 + x31 + x33 + x34 +
                                     x36 + x38 + x4 + x40 - x41 - x42 + x44 + x45 + x46 + x5 - x6 - x7 + x8;
        element_vector[1 * stride] = fff[0 * stride] * x49 - fff[1 * stride] * x47 - fff[2 * stride] * x47 - x16 - x23 +
                                     x4 + x48 - x5 + x52 + x54 + x56 + x58 + x6 + x7 - x8;
        element_vector[2 * stride] = (1.0 / 10.0) * fff[3 * stride] * u[2] - u[6] * x55 - u[6] * x59 - x15 + x26 - x27 +
                                     x29 + x30 - x31 - x38 + x56 + x60 + x62 + x64 + x65;
        element_vector[3 * stride] = (1.0 / 10.0) * fff[5 * stride] * u[3] - u[7] * x57 - u[7] * x59 - x21 - x36 + x41 +
                                     x42 + x44 - x45 - x46 + x58 + x65 + x66 + x67 + x69;
        element_vector[4 * stride] = (4.0 / 15.0) * fff[0 * stride] * u[4] + (8.0 / 15.0) * fff[4 * stride] * u[4] -
                                     u[0] * x3 - u[1] * x3 - x71 - x73 - x78 - x83 - x87 - x89 - x92;
        element_vector[5 * stride] = x101 + x67 + x78 + x93 + x95 + x99;
        element_vector[6 * stride] = (8.0 / 15.0) * fff[2 * stride] * u[6] + (4.0 / 15.0) * fff[3 * stride] * u[6] -
                                     u[0] * x28 - u[2] * x28 - x101 - x102 - x103 - x104 - x105 - x106 - x92;
        element_vector[7 * stride] = (8.0 / 15.0) * fff[1 * stride] * u[7] + (4.0 / 15.0) * fff[5 * stride] * u[7] -
                                     u[0] * x43 - u[3] * x43 - x106 - x107 - x108 - x79 - x81 - x89 - x95;
        element_vector[8 * stride] = x104 + x107 + x109 + x62 + x83 + x99;
        element_vector[9 * stride] = x105 + x108 + x109 + x52 + x87 + x93 + x96 + x97 + x98;
    }
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
    // FLOATING POINT OPS!
    //       - Result: ADD + ASSIGNMENT + 531*MUL
    //       - Subexpressions: 2*ADD + 7*DIV + 178*MUL + NEG + 20*POW + 21*SUB
    const real_t x0 = -py0 + py1;
    const real_t x1 = -pz0 + pz2;
    const real_t x2 = -py0 + py2;
    const real_t x3 = -pz0 + pz1;
    const real_t x4 = x0 * x1 - x2 * x3;
    const real_t x5 = pow(x4, 2);
    const real_t x6 = u[1] * x5;
    const real_t x7 = -pz0 + pz3;
    const real_t x8 = -px0 + px1;
    const real_t x9 = x2 * x8;
    const real_t x10 = -px0 + px2;
    const real_t x11 = -py0 + py3;
    const real_t x12 = x10 * x11;
    const real_t x13 = -px0 + px3;
    const real_t x14 = x1 * x8;
    const real_t x15 = x0 * x10;
    const real_t x16 = x13 * x3;
    const real_t x17 = pow(x0 * x1 * x13 - x11 * x14 + x12 * x3 - x15 * x7 - x16 * x2 + x7 * x9, -2);
    const real_t x18 = (1.0 / 30.0) * x17;
    const real_t x19 = u[0] * x18;
    const real_t x20 = -x0 * x7 + x11 * x3;
    const real_t x21 = pow(x20, 2);
    const real_t x22 = u[1] * x19;
    const real_t x23 = -x1 * x11 + x2 * x7;
    const real_t x24 = pow(x23, 2);
    const real_t x25 = x10 * x3 - x14;
    const real_t x26 = pow(x25, 2);
    const real_t x27 = u[2] * x19;
    const real_t x28 = -x16 + x7 * x8;
    const real_t x29 = pow(x28, 2);
    const real_t x30 = u[2] * x29;
    const real_t x31 = x1 * x13 - x10 * x7;
    const real_t x32 = pow(x31, 2);
    const real_t x33 = -x15 + x9;
    const real_t x34 = pow(x33, 2);
    const real_t x35 = u[3] * x34;
    const real_t x36 = x0 * x13 - x11 * x8;
    const real_t x37 = pow(x36, 2);
    const real_t x38 = u[3] * x19;
    const real_t x39 = x12 - x13 * x2;
    const real_t x40 = pow(x39, 2);
    const real_t x41 = u[4] * x19;
    const real_t x42 = u[4] * x29;
    const real_t x43 = (2.0 / 15.0) * x17;
    const real_t x44 = u[0] * x43;
    const real_t x45 = u[4] * x44;
    const real_t x46 = u[5] * x19;
    const real_t x47 = u[6] * x19;
    const real_t x48 = u[6] * x44;
    const real_t x49 = u[7] * x44;
    const real_t x50 = u[7] * x19;
    const real_t x51 = u[8] * x34;
    const real_t x52 = u[8] * x19;
    const real_t x53 = u[8] * x5;
    const real_t x54 = u[9] * x19;
    const real_t x55 = u[4] * x43;
    const real_t x56 = u[1] * x55;
    const real_t x57 = u[5] * x18;
    const real_t x58 = u[1] * x57;
    const real_t x59 = x18 * x6;
    const real_t x60 = u[1] * x18;
    const real_t x61 = u[6] * x60;
    const real_t x62 = u[7] * x60;
    const real_t x63 = u[8] * x60;
    const real_t x64 = u[2] * x18;
    const real_t x65 = u[4] * x64;
    const real_t x66 = u[2] * x57;
    const real_t x67 = u[2] * u[6];
    const real_t x68 = x43 * x67;
    const real_t x69 = x29 * x43;
    const real_t x70 = u[7] * x64;
    const real_t x71 = u[9] * x18;
    const real_t x72 = u[2] * x71;
    const real_t x73 = u[4] * x18;
    const real_t x74 = u[3] * x73;
    const real_t x75 = u[6] * x18;
    const real_t x76 = u[3] * x75;
    const real_t x77 = u[7] * x43;
    const real_t x78 = u[3] * x77;
    const real_t x79 = u[8] * x18;
    const real_t x80 = u[3] * x79;
    const real_t x81 = u[3] * x71;
    const real_t x82 = (4.0 / 15.0) * x17;
    const real_t x83 = u[5] * x82;
    const real_t x84 = u[4] * x83;
    const real_t x85 = u[6] * x55;
    const real_t x86 = u[7] * x55;
    const real_t x87 = x34 * x82;
    const real_t x88 = u[4] * u[8];
    const real_t x89 = x82 * x88;
    const real_t x90 = u[9] * x55;
    const real_t x91 = u[9] * x43;
    const real_t x92 = u[6] * x83;
    const real_t x93 = u[5] * x77;
    const real_t x94 = u[5] * x69;
    const real_t x95 = u[5] * x43;
    const real_t x96 = u[8] * x95;
    const real_t x97 = u[5] * x91;
    const real_t x98 = u[6] * x77;
    const real_t x99 = u[6] * x43;
    const real_t x100 = u[8] * x99;
    const real_t x101 = u[6] * u[9];
    const real_t x102 = x101 * x82;
    const real_t x103 = u[7] * x82;
    const real_t x104 = u[8] * x103;
    const real_t x105 = u[9] * x103;
    const real_t x106 = u[8] * x91;
    const real_t x107 = pow(u[0], 2);
    const real_t x108 = (1.0 / 20.0) * x17;
    const real_t x109 = x107 * x108;
    const real_t x110 = pow(u[1], 2) * x108;
    const real_t x111 = pow(u[2], 2) * x108;
    const real_t x112 = pow(u[3], 2) * x108;
    const real_t x113 = pow(u[4], 2);
    const real_t x114 = x113 * x43;
    const real_t x115 = pow(u[5], 2) * x43;
    const real_t x116 = pow(u[6], 2);
    const real_t x117 = x116 * x43;
    const real_t x118 = pow(u[7], 2);
    const real_t x119 = x118 * x43;
    const real_t x120 = pow(u[8], 2) * x43;
    const real_t x121 = pow(u[9], 2) * x43;
    const real_t x122 = x33 * x4;
    const real_t x123 = x20 * x36;
    const real_t x124 = x25 * x4;
    const real_t x125 = x20 * x28;
    const real_t x126 = x23 * x39;
    const real_t x127 = x23 * x31;
    const real_t x128 = x25 * x33;
    const real_t x129 = x28 * x36;
    const real_t x130 = x31 * x39;
    const real_t x131 = u[4] * x128;
    const real_t x132 = u[0] * x17;
    const real_t x133 = (1.0 / 15.0) * x132;
    const real_t x134 = u[4] * x122;
    const real_t x135 = (1.0 / 6.0) * x132;
    const real_t x136 = u[4] * x133;
    const real_t x137 = u[4] * x135;
    const real_t x138 = u[5] * x133;
    const real_t x139 = u[5] * x125;
    const real_t x140 = u[6] * x135;
    const real_t x141 = u[6] * x122;
    const real_t x142 = u[6] * x133;
    const real_t x143 = u[7] * x135;
    const real_t x144 = u[7] * x133;
    const real_t x145 = u[8] * x122;
    const real_t x146 = u[8] * x133;
    const real_t x147 = u[9] * x133;
    const real_t x148 = u[2] * x60;
    const real_t x149 = u[3] * x60;
    const real_t x150 = (1.0 / 10.0) * x17;
    const real_t x151 = u[1] * x150;
    const real_t x152 = u[4] * x151;
    const real_t x153 = u[5] * x151;
    const real_t x154 = u[8] * x151;
    const real_t x155 = u[9] * x60;
    const real_t x156 = u[3] * x128;
    const real_t x157 = u[3] * x64;
    const real_t x158 = u[2] * x150;
    const real_t x159 = u[5] * x158;
    const real_t x160 = x150 * x67;
    const real_t x161 = u[2] * x79;
    const real_t x162 = u[9] * x158;
    const real_t x163 = u[3] * x57;
    const real_t x164 = u[7] * x150;
    const real_t x165 = u[3] * x164;
    const real_t x166 = u[3] * x150;
    const real_t x167 = u[8] * x166;
    const real_t x168 = u[9] * x166;
    const real_t x169 = u[5] * x55;
    const real_t x170 = u[4] * u[6] * x82;
    const real_t x171 = u[4] * x103;
    const real_t x172 = x43 * x88;
    const real_t x173 = u[9] * x82;
    const real_t x174 = u[4] * x173;
    const real_t x175 = u[5] * x99;
    const real_t x176 = u[7] * x83;
    const real_t x177 = u[8] * x83;
    const real_t x178 = u[9] * x83;
    const real_t x179 = u[6] * x103;
    const real_t x180 = u[8] * x82;
    const real_t x181 = u[6] * x180;
    const real_t x182 = x101 * x43;
    const real_t x183 = u[8] * x77;
    const real_t x184 = u[9] * x77;
    const real_t x185 = u[8] * x173;
    const real_t x186 = x107 * x150;
    const real_t x187 = x186 * x4;
    const real_t x188 = x186 * x28;
    const real_t x189 = x186 * x39;
    const real_t x190 = x113 * x82;
    const real_t x191 = x114 * x4;
    const real_t x192 = x116 * x82;
    const real_t x193 = x118 * x82;
    element_scalar[0] =
        u[6] * x59 + u[7] * x18 * x30 + u[7] * x59 - u[7] * x94 + u[9] * x150 * x156 + u[9] * x94 - x100 * x21 -
        x100 * x24 - x100 * x37 - x100 * x40 - x101 * x87 - x102 * x122 - x102 * x123 - x102 * x126 - x102 * x128 -
        x102 * x129 - x102 * x130 - x102 * x37 - x102 * x40 + x103 * x134 - x103 * x53 - x104 * x122 - x104 * x123 -
        x104 * x124 - x104 * x125 - x104 * x126 - x104 * x127 - x104 * x21 - x104 * x24 - x105 * x124 - x105 * x125 -
        x105 * x127 - x105 * x128 - x105 * x129 - x105 * x130 - x105 * x26 - x105 * x29 - x105 * x32 + x106 * x122 +
        x106 * x123 + x106 * x126 + x106 * x128 + x106 * x129 + x106 * x130 + x106 * x37 + x106 * x40 + x109 * x21 +
        x109 * x24 + x109 * x26 + x109 * x29 + x109 * x32 + x109 * x34 + x109 * x37 + x109 * x40 + x109 * x5 +
        x110 * x21 + x110 * x24 + x110 * x5 + x111 * x26 + x111 * x29 + x111 * x32 + x112 * x34 + x112 * x37 +
        x112 * x40 + x114 * x123 + x114 * x125 + x114 * x126 + x114 * x127 + x114 * x21 + x114 * x24 + x114 * x26 +
        x114 * x29 + x114 * x32 + x114 * x34 + x114 * x37 + x114 * x40 + x114 * x5 + x115 * x124 + x115 * x125 +
        x115 * x127 + x115 * x21 + x115 * x24 + x115 * x26 + x115 * x29 + x115 * x32 + x115 * x5 + x117 * x124 +
        x117 * x125 + x117 * x127 + x117 * x128 + x117 * x129 + x117 * x130 + x117 * x21 + x117 * x24 + x117 * x26 +
        x117 * x29 + x117 * x32 + x117 * x34 + x117 * x37 + x117 * x40 + x117 * x5 + x119 * x122 + x119 * x123 +
        x119 * x126 + x119 * x128 + x119 * x129 + x119 * x130 + x119 * x21 + x119 * x24 + x119 * x26 + x119 * x29 +
        x119 * x32 + x119 * x34 + x119 * x37 + x119 * x40 + x119 * x5 + x120 * x122 + x120 * x123 + x120 * x126 +
        x120 * x21 + x120 * x24 + x120 * x34 + x120 * x37 + x120 * x40 + x120 * x5 + x121 * x128 + x121 * x129 +
        x121 * x130 + x121 * x26 + x121 * x29 + x121 * x32 + x121 * x34 + x121 * x37 + x121 * x40 - x122 * x143 -
        x122 * x149 + x122 * x154 - x122 * x155 - x122 * x163 - x122 * x165 - x122 * x169 + x122 * x178 - x122 * x184 +
        x122 * x192 + x122 * x22 + x122 * x38 + x122 * x46 + x122 * x54 + x122 * x61 + x122 * x76 + x122 * x85 -
        x122 * x89 - x122 * x92 + x122 * x96 + x122 * x98 - x123 * x137 - x123 * x142 - x123 * x143 + x123 * x146 -
        x123 * x149 - x123 * x152 + x123 * x154 - x123 * x155 - x123 * x163 - x123 * x165 + x123 * x167 - x123 * x169 +
        x123 * x171 + x123 * x178 - x123 * x181 - x123 * x184 + x123 * x186 + x123 * x192 + x123 * x22 + x123 * x38 +
        x123 * x46 + x123 * x54 + x123 * x61 + x123 * x76 + x123 * x85 - x123 * x89 - x123 * x92 + x123 * x96 +
        x123 * x98 - x124 * x137 + x124 * x138 - x124 * x140 - x124 * x144 - x124 * x148 - x124 * x152 + x124 * x153 -
        x124 * x155 + x124 * x159 - x124 * x160 - x124 * x161 + x124 * x170 - x124 * x172 - x124 * x176 - x124 * x182 +
        x124 * x185 + x124 * x193 + x124 * x22 + x124 * x27 + x124 * x52 + x124 * x54 + x124 * x62 + x124 * x70 -
        x124 * x84 + x124 * x86 - x124 * x92 + x124 * x96 + x124 * x97 + x124 * x98 - x125 * x137 - x125 * x140 -
        x125 * x144 - x125 * x148 - x125 * x152 + x125 * x153 - x125 * x155 - x125 * x160 - x125 * x161 + x125 * x170 -
        x125 * x172 - x125 * x176 - x125 * x182 + x125 * x185 + x125 * x193 + x125 * x22 + x125 * x27 + x125 * x52 +
        x125 * x54 + x125 * x62 + x125 * x70 - x125 * x84 + x125 * x86 - x125 * x92 + x125 * x96 + x125 * x97 +
        x125 * x98 - x126 * x137 - x126 * x142 - x126 * x143 + x126 * x146 - x126 * x149 - x126 * x152 + x126 * x154 -
        x126 * x155 - x126 * x163 - x126 * x165 + x126 * x167 - x126 * x169 + x126 * x171 + x126 * x178 - x126 * x181 -
        x126 * x184 + x126 * x192 + x126 * x22 + x126 * x38 + x126 * x46 + x126 * x54 + x126 * x61 + x126 * x76 +
        x126 * x85 - x126 * x89 - x126 * x92 + x126 * x96 + x126 * x98 - x127 * x137 + x127 * x138 - x127 * x140 -
        x127 * x144 - x127 * x148 - x127 * x152 + x127 * x153 - x127 * x155 + x127 * x159 - x127 * x160 - x127 * x161 +
        x127 * x170 - x127 * x172 - x127 * x176 - x127 * x182 + x127 * x185 + x127 * x186 + x127 * x193 + x127 * x22 +
        x127 * x27 + x127 * x52 + x127 * x54 + x127 * x62 + x127 * x70 - x127 * x84 + x127 * x86 - x127 * x92 +
        x127 * x96 + x127 * x97 + x127 * x98 - x128 * x140 - x128 * x143 + x128 * x147 - x128 * x160 - x128 * x161 +
        x128 * x162 - x128 * x175 + x128 * x177 + x128 * x179 - x128 * x183 + x128 * x186 + x128 * x190 + x128 * x27 +
        x128 * x38 + x128 * x46 + x128 * x52 + x128 * x65 + x128 * x74 - x128 * x84 + x128 * x85 + x128 * x86 -
        x128 * x89 + x128 * x97 - x129 * x136 - x129 * x140 - x129 * x143 + x129 * x147 - x129 * x157 - x129 * x160 -
        x129 * x161 + x129 * x162 - x129 * x163 - x129 * x165 + x129 * x168 - x129 * x174 - x129 * x175 + x129 * x177 +
        x129 * x179 - x129 * x183 + x129 * x190 + x129 * x27 + x129 * x38 + x129 * x46 + x129 * x52 + x129 * x65 +
        x129 * x74 - x129 * x84 + x129 * x85 + x129 * x86 - x129 * x89 + x129 * x97 - x130 * x136 - x130 * x140 -
        x130 * x143 + x130 * x147 - x130 * x157 - x130 * x160 - x130 * x161 + x130 * x162 - x130 * x163 - x130 * x165 +
        x130 * x168 - x130 * x174 - x130 * x175 + x130 * x177 + x130 * x179 - x130 * x183 + x130 * x190 + x130 * x27 +
        x130 * x38 + x130 * x46 + x130 * x52 + x130 * x65 + x130 * x74 - x130 * x84 + x130 * x85 + x130 * x86 -
        x130 * x89 + x130 * x97 - x131 * x133 - x131 * x173 + x133 * x139 - x133 * x141 + x133 * x145 - x134 * x135 -
        x134 * x151 + x139 * x158 - x141 * x180 + x145 * x166 - x156 * x164 - x156 * x57 - x156 * x64 + x187 * x25 +
        x187 * x33 + x188 * x20 + x188 * x36 + x189 * x23 + x189 * x31 + x19 * x30 + x19 * x35 - x19 * x42 + x19 * x51 +
        x19 * x53 + x19 * x6 + x191 * x25 + x191 * x33 + x21 * x22 - x21 * x45 + x21 * x46 - x21 * x47 - x21 * x50 +
        x21 * x52 - x21 * x56 - x21 * x58 + x21 * x61 + x21 * x62 - x21 * x63 - x21 * x92 - x21 * x93 + x21 * x96 +
        x21 * x98 + x22 * x24 - x24 * x45 + x24 * x46 - x24 * x47 - x24 * x50 + x24 * x52 - x24 * x56 - x24 * x58 +
        x24 * x61 + x24 * x62 - x24 * x63 - x24 * x92 - x24 * x93 + x24 * x96 + x24 * x98 + x26 * x27 - x26 * x41 +
        x26 * x46 - x26 * x48 - x26 * x50 + x26 * x54 + x26 * x65 - x26 * x66 - x26 * x68 + x26 * x70 - x26 * x72 -
        x26 * x84 + x26 * x86 - x26 * x90 - x26 * x93 + x26 * x97 + x27 * x32 + x29 * x46 - x29 * x48 - x29 * x50 +
        x29 * x54 - x29 * x66 - x30 * x71 - x32 * x41 + x32 * x46 - x32 * x48 - x32 * x50 + x32 * x54 + x32 * x65 -
        x32 * x66 - x32 * x68 + x32 * x70 - x32 * x72 - x32 * x84 + x32 * x86 - x32 * x90 - x32 * x93 + x32 * x97 -
        x34 * x41 - x34 * x47 - x34 * x49 + x34 * x54 + x34 * x85 - x34 * x90 - x35 * x71 + x35 * x73 + x35 * x75 -
        x35 * x77 - x35 * x79 + x37 * x38 - x37 * x41 - x37 * x47 - x37 * x49 + x37 * x52 + x37 * x54 + x37 * x74 +
        x37 * x76 - x37 * x78 - x37 * x80 - x37 * x81 + x37 * x85 - x37 * x89 - x37 * x90 + x38 * x40 - x40 * x41 -
        x40 * x47 - x40 * x49 + x40 * x52 + x40 * x54 + x40 * x74 + x40 * x76 - x40 * x78 - x40 * x80 - x40 * x81 +
        x40 * x85 - x40 * x89 - x40 * x90 + x42 * x64 + x42 * x77 - x42 * x83 - x42 * x91 - x45 * x5 + x46 * x5 -
        x47 * x5 - x5 * x50 - x5 * x92 - x5 * x93 + x5 * x98 + x51 * x91 - x51 * x99 - x53 * x60 + x53 * x95 -
        x53 * x99 - x55 * x6 - x57 * x6 - x67 * x69 - x87 * x88;
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
