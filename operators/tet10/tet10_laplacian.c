#include <assert.h>
#include <math.h>
#include <stdio.h>

#include <mpi.h>

#include "crs_graph.h"
#include "sortreduce.h"

#include "sfem_vec.h"

#define POW2(a) ((a) * (a))

static SFEM_INLINE void tet10_laplacian_hessian(const real_t px0,
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
        //       - Result: 5*ADD + 100*ASSIGNMENT + 6*MUL
        //       - Subexpressions: 34*ADD + 30*DIV + 12*MUL + 19*NEG + 34*SUB
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

static SFEM_INLINE void tet10_laplacian_gradient(const real_t px0,
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
        //       - Result: 10*ADD + 10*ASSIGNMENT + 78*MUL
        //       - Subexpressions: 60*ADD + 35*DIV + 130*MUL + 18*NEG + 24*SUB
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

static SFEM_INLINE void tet10_laplacian_value(const real_t px0,
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
    // FLOATING POINT OPS!
    //      - Result: 2*ADD + ASSIGNMENT + 538*MUL
    //      - Subexpressions: 2*ADD + 7*DIV + 178*MUL + NEG + 20*POW + 21*SUB
    const real_t x0 = -pz0 + pz3;
    const real_t x1 = -px0 + px1;
    const real_t x2 = -py0 + py2;
    const real_t x3 = x1 * x2;
    const real_t x4 = x0 * x3;
    const real_t x5 = -py0 + py3;
    const real_t x6 = -pz0 + pz2;
    const real_t x7 = x1 * x6;
    const real_t x8 = x5 * x7;
    const real_t x9 = -px0 + px2;
    const real_t x10 = -py0 + py1;
    const real_t x11 = x10 * x9;
    const real_t x12 = x0 * x11;
    const real_t x13 = -pz0 + pz1;
    const real_t x14 = x5 * x9;
    const real_t x15 = x13 * x14;
    const real_t x16 = -px0 + px3;
    const real_t x17 = x10 * x16 * x6;
    const real_t x18 = x13 * x16;
    const real_t x19 = x18 * x2;
    const real_t x20 = x10 * x6 - x13 * x2;
    const real_t x21 = pow(x20, 2);
    const real_t x22 = u[1] * x21;
    const real_t x23 = pow(-x12 + x15 + x17 - x19 + x4 - x8, -2);
    const real_t x24 = (1.0 / 30.0) * x23;
    const real_t x25 = u[0] * x24;
    const real_t x26 = -x0 * x10 + x13 * x5;
    const real_t x27 = pow(x26, 2);
    const real_t x28 = u[1] * x25;
    const real_t x29 = x0 * x2 - x5 * x6;
    const real_t x30 = pow(x29, 2);
    const real_t x31 = x13 * x9 - x7;
    const real_t x32 = pow(x31, 2);
    const real_t x33 = u[2] * x25;
    const real_t x34 = x0 * x1 - x18;
    const real_t x35 = pow(x34, 2);
    const real_t x36 = u[2] * x35;
    const real_t x37 = -x0 * x9 + x16 * x6;
    const real_t x38 = pow(x37, 2);
    const real_t x39 = -x11 + x3;
    const real_t x40 = pow(x39, 2);
    const real_t x41 = u[3] * x40;
    const real_t x42 = -x1 * x5 + x10 * x16;
    const real_t x43 = pow(x42, 2);
    const real_t x44 = u[3] * x25;
    const real_t x45 = x14 - x16 * x2;
    const real_t x46 = pow(x45, 2);
    const real_t x47 = u[4] * x25;
    const real_t x48 = u[4] * x35;
    const real_t x49 = (2.0 / 15.0) * x23;
    const real_t x50 = u[0] * x49;
    const real_t x51 = u[4] * x50;
    const real_t x52 = u[5] * x25;
    const real_t x53 = u[6] * x25;
    const real_t x54 = u[6] * x50;
    const real_t x55 = u[7] * x50;
    const real_t x56 = u[7] * x25;
    const real_t x57 = u[8] * x40;
    const real_t x58 = u[8] * x25;
    const real_t x59 = u[8] * x21;
    const real_t x60 = u[9] * x25;
    const real_t x61 = u[4] * x49;
    const real_t x62 = u[1] * x61;
    const real_t x63 = u[5] * x24;
    const real_t x64 = u[1] * x63;
    const real_t x65 = x22 * x24;
    const real_t x66 = u[1] * x24;
    const real_t x67 = u[6] * x66;
    const real_t x68 = u[7] * x66;
    const real_t x69 = u[8] * x66;
    const real_t x70 = u[2] * x24;
    const real_t x71 = u[4] * x70;
    const real_t x72 = u[2] * x63;
    const real_t x73 = u[2] * u[6];
    const real_t x74 = x49 * x73;
    const real_t x75 = x35 * x49;
    const real_t x76 = u[7] * x70;
    const real_t x77 = u[9] * x24;
    const real_t x78 = u[2] * x77;
    const real_t x79 = u[4] * x24;
    const real_t x80 = u[3] * x79;
    const real_t x81 = u[6] * x24;
    const real_t x82 = u[3] * x81;
    const real_t x83 = u[7] * x49;
    const real_t x84 = u[3] * x83;
    const real_t x85 = u[8] * x24;
    const real_t x86 = u[3] * x85;
    const real_t x87 = u[3] * x77;
    const real_t x88 = (4.0 / 15.0) * x23;
    const real_t x89 = u[5] * x88;
    const real_t x90 = u[4] * x89;
    const real_t x91 = u[6] * x61;
    const real_t x92 = u[7] * x61;
    const real_t x93 = x40 * x88;
    const real_t x94 = u[4] * u[8];
    const real_t x95 = x88 * x94;
    const real_t x96 = u[9] * x61;
    const real_t x97 = u[9] * x49;
    const real_t x98 = u[6] * x89;
    const real_t x99 = u[5] * x83;
    const real_t x100 = u[5] * x75;
    const real_t x101 = u[5] * x49;
    const real_t x102 = u[8] * x101;
    const real_t x103 = u[5] * x97;
    const real_t x104 = u[6] * x83;
    const real_t x105 = u[6] * x49;
    const real_t x106 = u[8] * x105;
    const real_t x107 = u[6] * u[9];
    const real_t x108 = x107 * x88;
    const real_t x109 = u[7] * x88;
    const real_t x110 = u[8] * x109;
    const real_t x111 = u[9] * x109;
    const real_t x112 = u[8] * x97;
    const real_t x113 = pow(u[0], 2);
    const real_t x114 = (1.0 / 20.0) * x23;
    const real_t x115 = x113 * x114;
    const real_t x116 = pow(u[1], 2) * x114;
    const real_t x117 = pow(u[2], 2) * x114;
    const real_t x118 = pow(u[3], 2) * x114;
    const real_t x119 = pow(u[4], 2);
    const real_t x120 = x119 * x49;
    const real_t x121 = pow(u[5], 2) * x49;
    const real_t x122 = pow(u[6], 2);
    const real_t x123 = x122 * x49;
    const real_t x124 = pow(u[7], 2);
    const real_t x125 = x124 * x49;
    const real_t x126 = pow(u[8], 2) * x49;
    const real_t x127 = pow(u[9], 2) * x49;
    const real_t x128 = x20 * x39;
    const real_t x129 = x26 * x42;
    const real_t x130 = x20 * x31;
    const real_t x131 = x26 * x34;
    const real_t x132 = x29 * x45;
    const real_t x133 = x29 * x37;
    const real_t x134 = x31 * x39;
    const real_t x135 = x34 * x42;
    const real_t x136 = x37 * x45;
    const real_t x137 = u[4] * x134;
    const real_t x138 = u[0] * x23;
    const real_t x139 = (1.0 / 15.0) * x138;
    const real_t x140 = u[4] * x128;
    const real_t x141 = (1.0 / 6.0) * x138;
    const real_t x142 = u[4] * x139;
    const real_t x143 = u[4] * x141;
    const real_t x144 = u[5] * x139;
    const real_t x145 = u[5] * x131;
    const real_t x146 = u[6] * x141;
    const real_t x147 = u[6] * x128;
    const real_t x148 = u[6] * x139;
    const real_t x149 = u[7] * x141;
    const real_t x150 = u[7] * x139;
    const real_t x151 = u[8] * x128;
    const real_t x152 = u[8] * x139;
    const real_t x153 = u[9] * x139;
    const real_t x154 = u[2] * x66;
    const real_t x155 = u[3] * x66;
    const real_t x156 = (1.0 / 10.0) * x23;
    const real_t x157 = u[1] * x156;
    const real_t x158 = u[4] * x157;
    const real_t x159 = u[5] * x157;
    const real_t x160 = u[8] * x157;
    const real_t x161 = u[9] * x66;
    const real_t x162 = u[3] * x134;
    const real_t x163 = u[3] * x70;
    const real_t x164 = u[2] * x156;
    const real_t x165 = u[5] * x164;
    const real_t x166 = x156 * x73;
    const real_t x167 = u[2] * x85;
    const real_t x168 = u[9] * x164;
    const real_t x169 = u[3] * x63;
    const real_t x170 = u[7] * x156;
    const real_t x171 = u[3] * x170;
    const real_t x172 = u[3] * x156;
    const real_t x173 = u[8] * x172;
    const real_t x174 = u[9] * x172;
    const real_t x175 = u[5] * x61;
    const real_t x176 = u[4] * u[6] * x88;
    const real_t x177 = u[4] * x109;
    const real_t x178 = x49 * x94;
    const real_t x179 = u[9] * x88;
    const real_t x180 = u[4] * x179;
    const real_t x181 = u[5] * x105;
    const real_t x182 = u[7] * x89;
    const real_t x183 = u[8] * x89;
    const real_t x184 = u[9] * x89;
    const real_t x185 = u[6] * x109;
    const real_t x186 = u[8] * x88;
    const real_t x187 = u[6] * x186;
    const real_t x188 = x107 * x49;
    const real_t x189 = u[8] * x83;
    const real_t x190 = u[9] * x83;
    const real_t x191 = u[8] * x179;
    const real_t x192 = x113 * x156;
    const real_t x193 = x192 * x20;
    const real_t x194 = x192 * x34;
    const real_t x195 = x192 * x45;
    const real_t x196 = x119 * x88;
    const real_t x197 = x120 * x20;
    const real_t x198 = x122 * x88;
    const real_t x199 = x124 * x88;
    element_scalar[0] =
        (-1.0 / 6.0 * x12 + (1.0 / 6.0) * x15 + (1.0 / 6.0) * x17 - 1.0 / 6.0 * x19 + (1.0 / 6.0) * x4 -
         1.0 / 6.0 * x8) *
        (u[6] * x65 - u[7] * x100 + u[7] * x24 * x36 + u[7] * x65 + u[9] * x100 + u[9] * x156 * x162 + x101 * x59 +
         x102 * x128 + x102 * x129 + x102 * x130 + x102 * x131 + x102 * x132 + x102 * x133 + x102 * x27 + x102 * x30 +
         x103 * x130 + x103 * x131 + x103 * x133 + x103 * x134 + x103 * x135 + x103 * x136 + x103 * x32 + x103 * x38 +
         x104 * x128 + x104 * x129 + x104 * x130 + x104 * x131 + x104 * x132 + x104 * x133 + x104 * x21 + x104 * x27 +
         x104 * x30 - x105 * x57 - x105 * x59 - x106 * x27 - x106 * x30 - x106 * x43 - x106 * x46 - x107 * x93 -
         x108 * x128 - x108 * x129 - x108 * x132 - x108 * x134 - x108 * x135 - x108 * x136 - x108 * x43 - x108 * x46 +
         x109 * x140 - x109 * x59 - x110 * x128 - x110 * x129 - x110 * x130 - x110 * x131 - x110 * x132 - x110 * x133 -
         x110 * x27 - x110 * x30 - x111 * x130 - x111 * x131 - x111 * x133 - x111 * x134 - x111 * x135 - x111 * x136 -
         x111 * x32 - x111 * x35 - x111 * x38 + x112 * x128 + x112 * x129 + x112 * x132 + x112 * x134 + x112 * x135 +
         x112 * x136 + x112 * x43 + x112 * x46 + x115 * x21 + x115 * x27 + x115 * x30 + x115 * x32 + x115 * x35 +
         x115 * x38 + x115 * x40 + x115 * x43 + x115 * x46 + x116 * x21 + x116 * x27 + x116 * x30 + x117 * x32 +
         x117 * x35 + x117 * x38 + x118 * x40 + x118 * x43 + x118 * x46 + x120 * x129 + x120 * x131 + x120 * x132 +
         x120 * x133 + x120 * x21 + x120 * x27 + x120 * x30 + x120 * x32 + x120 * x35 + x120 * x38 + x120 * x40 +
         x120 * x43 + x120 * x46 + x121 * x130 + x121 * x131 + x121 * x133 + x121 * x21 + x121 * x27 + x121 * x30 +
         x121 * x32 + x121 * x35 + x121 * x38 + x123 * x130 + x123 * x131 + x123 * x133 + x123 * x134 + x123 * x135 +
         x123 * x136 + x123 * x21 + x123 * x27 + x123 * x30 + x123 * x32 + x123 * x35 + x123 * x38 + x123 * x40 +
         x123 * x43 + x123 * x46 + x125 * x128 + x125 * x129 + x125 * x132 + x125 * x134 + x125 * x135 + x125 * x136 +
         x125 * x21 + x125 * x27 + x125 * x30 + x125 * x32 + x125 * x35 + x125 * x38 + x125 * x40 + x125 * x43 +
         x125 * x46 + x126 * x128 + x126 * x129 + x126 * x132 + x126 * x21 + x126 * x27 + x126 * x30 + x126 * x40 +
         x126 * x43 + x126 * x46 + x127 * x134 + x127 * x135 + x127 * x136 + x127 * x32 + x127 * x35 + x127 * x38 +
         x127 * x40 + x127 * x43 + x127 * x46 - x128 * x149 - x128 * x155 + x128 * x160 - x128 * x161 - x128 * x169 -
         x128 * x171 - x128 * x175 + x128 * x184 - x128 * x190 + x128 * x198 + x128 * x28 + x128 * x44 + x128 * x52 +
         x128 * x60 + x128 * x67 + x128 * x82 + x128 * x91 - x128 * x95 - x128 * x98 - x129 * x143 - x129 * x148 -
         x129 * x149 + x129 * x152 - x129 * x155 - x129 * x158 + x129 * x160 - x129 * x161 - x129 * x169 - x129 * x171 +
         x129 * x173 - x129 * x175 + x129 * x177 + x129 * x184 - x129 * x187 - x129 * x190 + x129 * x192 + x129 * x198 +
         x129 * x28 + x129 * x44 + x129 * x52 + x129 * x60 + x129 * x67 + x129 * x82 + x129 * x91 - x129 * x95 -
         x129 * x98 - x130 * x143 + x130 * x144 - x130 * x146 - x130 * x150 - x130 * x154 - x130 * x158 + x130 * x159 -
         x130 * x161 + x130 * x165 - x130 * x166 - x130 * x167 + x130 * x176 - x130 * x178 - x130 * x182 - x130 * x188 +
         x130 * x191 + x130 * x199 + x130 * x28 + x130 * x33 + x130 * x58 + x130 * x60 + x130 * x68 + x130 * x76 -
         x130 * x90 + x130 * x92 - x130 * x98 - x131 * x143 - x131 * x146 - x131 * x150 - x131 * x154 - x131 * x158 +
         x131 * x159 - x131 * x161 - x131 * x166 - x131 * x167 + x131 * x176 - x131 * x178 - x131 * x182 - x131 * x188 +
         x131 * x191 + x131 * x199 + x131 * x28 + x131 * x33 + x131 * x58 + x131 * x60 + x131 * x68 + x131 * x76 -
         x131 * x90 + x131 * x92 - x131 * x98 - x132 * x143 - x132 * x148 - x132 * x149 + x132 * x152 - x132 * x155 -
         x132 * x158 + x132 * x160 - x132 * x161 - x132 * x169 - x132 * x171 + x132 * x173 - x132 * x175 + x132 * x177 +
         x132 * x184 - x132 * x187 - x132 * x190 + x132 * x198 + x132 * x28 + x132 * x44 + x132 * x52 + x132 * x60 +
         x132 * x67 + x132 * x82 + x132 * x91 - x132 * x95 - x132 * x98 - x133 * x143 + x133 * x144 - x133 * x146 -
         x133 * x150 - x133 * x154 - x133 * x158 + x133 * x159 - x133 * x161 + x133 * x165 - x133 * x166 - x133 * x167 +
         x133 * x176 - x133 * x178 - x133 * x182 - x133 * x188 + x133 * x191 + x133 * x192 + x133 * x199 + x133 * x28 +
         x133 * x33 + x133 * x58 + x133 * x60 + x133 * x68 + x133 * x76 - x133 * x90 + x133 * x92 - x133 * x98 -
         x134 * x146 - x134 * x149 + x134 * x153 - x134 * x166 - x134 * x167 + x134 * x168 - x134 * x181 + x134 * x183 +
         x134 * x185 - x134 * x189 + x134 * x192 + x134 * x196 + x134 * x33 + x134 * x44 + x134 * x52 + x134 * x58 +
         x134 * x71 + x134 * x80 - x134 * x90 + x134 * x91 + x134 * x92 - x134 * x95 - x135 * x142 - x135 * x146 -
         x135 * x149 + x135 * x153 - x135 * x163 - x135 * x166 - x135 * x167 + x135 * x168 - x135 * x169 - x135 * x171 +
         x135 * x174 - x135 * x180 - x135 * x181 + x135 * x183 + x135 * x185 - x135 * x189 + x135 * x196 + x135 * x33 +
         x135 * x44 + x135 * x52 + x135 * x58 + x135 * x71 + x135 * x80 - x135 * x90 + x135 * x91 + x135 * x92 -
         x135 * x95 - x136 * x142 - x136 * x146 - x136 * x149 + x136 * x153 - x136 * x163 - x136 * x166 - x136 * x167 +
         x136 * x168 - x136 * x169 - x136 * x171 + x136 * x174 - x136 * x180 - x136 * x181 + x136 * x183 + x136 * x185 -
         x136 * x189 + x136 * x196 + x136 * x33 + x136 * x44 + x136 * x52 + x136 * x58 + x136 * x71 + x136 * x80 -
         x136 * x90 + x136 * x91 + x136 * x92 - x136 * x95 - x137 * x139 - x137 * x179 + x139 * x145 - x139 * x147 +
         x139 * x151 - x140 * x141 - x140 * x157 + x145 * x164 - x147 * x186 + x151 * x172 - x162 * x170 - x162 * x63 -
         x162 * x70 + x193 * x31 + x193 * x39 + x194 * x26 + x194 * x42 + x195 * x29 + x195 * x37 + x197 * x31 +
         x197 * x39 - x21 * x51 + x21 * x52 - x21 * x53 - x21 * x56 - x21 * x98 - x21 * x99 + x22 * x25 - x22 * x61 -
         x22 * x63 + x25 * x36 + x25 * x41 - x25 * x48 + x25 * x57 + x25 * x59 + x27 * x28 - x27 * x51 + x27 * x52 -
         x27 * x53 - x27 * x56 + x27 * x58 - x27 * x62 - x27 * x64 + x27 * x67 + x27 * x68 - x27 * x69 - x27 * x98 -
         x27 * x99 + x28 * x30 - x30 * x51 + x30 * x52 - x30 * x53 - x30 * x56 + x30 * x58 - x30 * x62 - x30 * x64 +
         x30 * x67 + x30 * x68 - x30 * x69 - x30 * x98 - x30 * x99 + x32 * x33 - x32 * x47 + x32 * x52 - x32 * x54 -
         x32 * x56 + x32 * x60 + x32 * x71 - x32 * x72 - x32 * x74 + x32 * x76 - x32 * x78 - x32 * x90 + x32 * x92 -
         x32 * x96 - x32 * x99 + x33 * x38 + x35 * x52 - x35 * x54 - x35 * x56 + x35 * x60 - x35 * x72 - x36 * x77 -
         x38 * x47 + x38 * x52 - x38 * x54 - x38 * x56 + x38 * x60 + x38 * x71 - x38 * x72 - x38 * x74 + x38 * x76 -
         x38 * x78 - x38 * x90 + x38 * x92 - x38 * x96 - x38 * x99 - x40 * x47 - x40 * x53 - x40 * x55 + x40 * x60 +
         x40 * x91 - x40 * x96 - x41 * x77 + x41 * x79 + x41 * x81 - x41 * x83 - x41 * x85 + x43 * x44 - x43 * x47 -
         x43 * x53 - x43 * x55 + x43 * x58 + x43 * x60 + x43 * x80 + x43 * x82 - x43 * x84 - x43 * x86 - x43 * x87 +
         x43 * x91 - x43 * x95 - x43 * x96 + x44 * x46 - x46 * x47 - x46 * x53 - x46 * x55 + x46 * x58 + x46 * x60 +
         x46 * x80 + x46 * x82 - x46 * x84 - x46 * x86 - x46 * x87 + x46 * x91 - x46 * x95 - x46 * x96 + x48 * x70 +
         x48 * x83 - x48 * x89 - x48 * x97 + x57 * x97 - x59 * x66 - x73 * x75 - x93 * x94);
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

void tet10_laplacian_assemble_hessian(const ptrdiff_t nelements,
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

        tet10_laplacian_hessian(
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
    printf("tet10_laplacian.c: tet10_laplacian_assemble_hessian\t%g seconds\n", tock - tick);
}

void tet10_laplacian_assemble_gradient(const ptrdiff_t nelements,
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

        tet10_laplacian_gradient(
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
    printf("tet10_laplacian.c: tet10_laplacian_assemble_gradient\t%g seconds\n", tock - tick);
}

void tet10_laplacian_assemble_value(const ptrdiff_t nelements,
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

        tet10_laplacian_value(
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
    printf("tet10_laplacian.c: tet10_laplacian_assemble_value\t%g seconds\n", tock - tick);
}

void tet10_laplacian_apply(const ptrdiff_t nelements,
                           const ptrdiff_t nnodes,
                           idx_t **const SFEM_RESTRICT elems,
                           geom_t **const SFEM_RESTRICT xyz,
                           const real_t *const SFEM_RESTRICT u,
                           real_t *const SFEM_RESTRICT values) {
    tet10_laplacian_assemble_gradient(nelements, nnodes, elems, xyz, u, values);
}
