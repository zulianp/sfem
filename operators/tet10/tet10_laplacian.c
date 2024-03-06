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
        // FLOATING POINT OPS!
        //       - Result: 6*ADD + 6*ASSIGNMENT + 24*MUL + 9*POW
        //       - Subexpressions: 2*ADD + 28*MUL + NEG + POW + 21*SUB
        const real_t x0 = -px0 + px1;
        const real_t x1 = -py0 + py2;
        const real_t x2 = -pz0 + pz3;
        const real_t x3 = x1 * x2;
        const real_t x4 = -pz0 + pz1;
        const real_t x5 = -px0 + px2;
        const real_t x6 = -py0 + py3;
        const real_t x7 = x5 * x6;
        const real_t x8 = -py0 + py1;
        const real_t x9 = -px0 + px3;
        const real_t x10 = -pz0 + pz2;
        const real_t x11 = x10 * x6;
        const real_t x12 = x2 * x5;
        const real_t x13 = x1 * x9;
        const real_t x14 = -x0 * x11 + x0 * x3 + x10 * x8 * x9 - x12 * x8 - x13 * x4 + x4 * x7;
        const real_t x15 = -x13 + x7;
        // const real_t x16 = pow(x14, -2);
        const real_t x16 = 1. / POW2(x14);
        const real_t x17 = x10 * x9 - x12;
        const real_t x18 = -x11 + x3;
        const real_t x19 = -x0 * x6 + x8 * x9;
        const real_t x20 = x15 * x16;
        const real_t x21 = x0 * x2 - x4 * x9;
        const real_t x22 = x16 * x17;
        const real_t x23 = -x2 * x8 + x4 * x6;
        const real_t x24 = x16 * x18;
        const real_t x25 = x0 * x1 - x5 * x8;
        const real_t x26 = -x0 * x10 + x4 * x5;
        const real_t x27 = -x1 * x4 + x10 * x8;
        fff[0 * stride] = x14 * (POW2(x15) * x16 + x16 * POW2(x17) + x16 * POW2(x18));
        fff[1 * stride] = x14 * (x19 * x20 + x21 * x22 + x23 * x24);
        fff[2 * stride] = x14 * (x20 * x25 + x22 * x26 + x24 * x27);
        fff[3 * stride] = x14 * (x16 * POW2(x19) + x16 * POW2(x21) + x16 * POW2(x23));
        fff[4 * stride] = x14 * (x16 * x19 * x25 + x16 * x21 * x26 + x16 * x23 * x27);
        fff[5 * stride] = x14 * (x16 * POW2(x25) + x16 * POW2(x26) + x16 * POW2(x27));
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
        // FLOATING POINT OPS!
        //       - Result: 6*ADD + 6*ASSIGNMENT + 24*MUL + 9*POW
        //       - Subexpressions: 2*ADD + 28*MUL + NEG + POW + 21*SUB
        const real_t x0 = -px0 + px1;
        const real_t x1 = -py0 + py2;
        const real_t x2 = -pz0 + pz3;
        const real_t x3 = x1 * x2;
        const real_t x4 = -pz0 + pz1;
        const real_t x5 = -px0 + px2;
        const real_t x6 = -py0 + py3;
        const real_t x7 = x5 * x6;
        const real_t x8 = -py0 + py1;
        const real_t x9 = -px0 + px3;
        const real_t x10 = -pz0 + pz2;
        const real_t x11 = x10 * x6;
        const real_t x12 = x2 * x5;
        const real_t x13 = x1 * x9;
        const real_t x14 = -x0 * x11 + x0 * x3 + x10 * x8 * x9 - x12 * x8 - x13 * x4 + x4 * x7;
        const real_t x15 = -x13 + x7;
        // const real_t x16 = pow(x14, -2);
        const real_t x16 = 1 / POW2(x14);
        const real_t x17 = x10 * x9 - x12;
        const real_t x18 = -x11 + x3;
        const real_t x19 = -x0 * x6 + x8 * x9;
        const real_t x20 = x15 * x16;
        const real_t x21 = x0 * x2 - x4 * x9;
        const real_t x22 = x16 * x17;
        const real_t x23 = -x2 * x8 + x4 * x6;
        const real_t x24 = x16 * x18;
        const real_t x25 = x0 * x1 - x5 * x8;
        const real_t x26 = -x0 * x10 + x4 * x5;
        const real_t x27 = -x1 * x4 + x10 * x8;
        fff[0 * stride] = x14 * (POW2(x15) * x16 + x16 * POW2(x17) + x16 * POW2(x18));
        fff[1 * stride] = x14 * (x19 * x20 + x21 * x22 + x23 * x24);
        fff[2 * stride] = x14 * (x20 * x25 + x22 * x26 + x24 * x27);
        fff[3 * stride] = x14 * (x16 * POW2(x19) + x16 * POW2(x21) + x16 * POW2(x23));
        fff[4 * stride] = x14 * (x16 * x19 * x25 + x16 * x21 * x26 + x16 * x23 * x27);
        fff[5 * stride] = x14 * (x16 * POW2(x25) + x16 * POW2(x26) + x16 * POW2(x27));
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
                           fff[4 * stride] * x77 - u[6] * x74 - u[7] * x28 + u[8] * x75 +
                           u[9] * x28 + x60 + x63;
        const real_t x79 = -fff[2 * stride] * x70;
        const real_t x80 = (4.0 / 15.0) * u[7];
        const real_t x81 = -fff[2 * stride] * x80;
        const real_t x82 = (2.0 / 15.0) * fff[2 * stride];
        const real_t x83 = fff[2 * stride] * x49 + fff[4 * stride] * x76 - fff[5 * stride] * x70 +
                           fff[5 * stride] * x77 + u[5] * x82 - u[6] * x43 - u[7] * x74 +
                           u[9] * x43 + x69 + x79 + x81;
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
        const real_t x101 = -fff[0 * stride] * x72 + fff[0 * stride] * x76 +
                            fff[2 * stride] * x100 + u[2] * x55 - u[4] * x82 - u[7] * x3 +
                            u[8] * x3 + u[9] * x75 + x54 + x71 + x73;
        const real_t x102 = -u[6] * x85;
        const real_t x103 = -u[7] * x85;
        const real_t x104 = u[0] * x22 + x88 + x94;
        const real_t x105 = fff[2 * stride] * x76 + fff[5 * stride] * x100 - fff[5 * stride] * x72 +
                            u[2] * x59 - u[4] * x43 + u[5] * x74 - u[7] * x82 + u[8] * x43 + x66 +
                            x68;
        const real_t x106 = -2.0 / 15.0 * fff[4 * stride] * u[4] + u[0] * x37 + x86;
        const real_t x107 = fff[0 * stride] * x77 - fff[0 * stride] * x80 + fff[1 * stride] * x100 +
                            u[3] * x57 - u[4] * x75 + u[5] * x3 - u[6] * x3 + u[9] * x82 + x48 +
                            x53;
        const real_t x108 = fff[1 * stride] * x77 + fff[3 * stride] * x100 - fff[3 * stride] * x80 +
                            u[3] * x59 - u[4] * x28 + u[5] * x28 - u[6] * x75 + u[8] * x74 + x102 +
                            x103 + x64;
        const real_t x109 = u[5] * x75 + x91;
        element_vector[0 * stride] =
            fff[0 * stride] * x0 + fff[1 * stride] * x9 + fff[2 * stride] * x9 +
            fff[3 * stride] * x0 + fff[4 * stride] * x9 + fff[5 * stride] * x0 - u[4] * x13 -
            u[4] * x20 - u[4] * x35 + u[5] * x14 - u[6] * x13 - u[6] * x22 - u[6] * x37 -
            u[7] * x14 - u[7] * x20 - u[7] * x37 + u[8] * x22 + u[9] * x35 + x11 + x12 + x15 + x16 +
            x18 + x19 + x2 + x21 + x23 + x25 - x26 + x27 + x29 - x30 + x31 + x33 + x34 + x36 + x38 +
            x4 + x40 - x41 - x42 + x44 + x45 + x46 + x5 - x6 - x7 + x8;
        element_vector[1 * stride] = fff[0 * stride] * x49 - fff[1 * stride] * x47 -
                                     fff[2 * stride] * x47 - x16 - x23 + x4 + x48 - x5 + x52 + x54 +
                                     x56 + x58 + x6 + x7 - x8;
        element_vector[2 * stride] = (1.0 / 10.0) * fff[3 * stride] * u[2] - u[6] * x55 -
                                     u[6] * x59 - x15 + x26 - x27 + x29 + x30 - x31 - x38 + x56 +
                                     x60 + x62 + x64 + x65;
        element_vector[3 * stride] = (1.0 / 10.0) * fff[5 * stride] * u[3] - u[7] * x57 -
                                     u[7] * x59 - x21 - x36 + x41 + x42 + x44 - x45 - x46 + x58 +
                                     x65 + x66 + x67 + x69;
        element_vector[4 * stride] = (4.0 / 15.0) * fff[0 * stride] * u[4] +
                                     (8.0 / 15.0) * fff[4 * stride] * u[4] - u[0] * x3 - u[1] * x3 -
                                     x71 - x73 - x78 - x83 - x87 - x89 - x92;
        element_vector[5 * stride] = x101 + x67 + x78 + x93 + x95 + x99;
        element_vector[6 * stride] = (8.0 / 15.0) * fff[2 * stride] * u[6] +
                                     (4.0 / 15.0) * fff[3 * stride] * u[6] - u[0] * x28 -
                                     u[2] * x28 - x101 - x102 - x103 - x104 - x105 - x106 - x92;
        element_vector[7 * stride] = (8.0 / 15.0) * fff[1 * stride] * u[7] +
                                     (4.0 / 15.0) * fff[5 * stride] * u[7] - u[0] * x43 -
                                     u[3] * x43 - x106 - x107 - x108 - x79 - x81 - x89 - x95;
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
    //       - Result: ADD + ASSIGNMENT + 532*MUL
    //       - Subexpressions: 2*ADD + 7*DIV + 178*MUL + NEG + 20*POW + 21*SUB
    const real_t x0 = -pz0 + pz3;
    const real_t x1 = -px0 + px1;
    const real_t x2 = -py0 + py2;
    const real_t x3 = x1 * x2;
    const real_t x4 = -pz0 + pz1;
    const real_t x5 = -px0 + px2;
    const real_t x6 = -py0 + py3;
    const real_t x7 = x5 * x6;
    const real_t x8 = -pz0 + pz2;
    const real_t x9 = -px0 + px3;
    const real_t x10 = -py0 + py1;
    const real_t x11 = x1 * x8;
    const real_t x12 = x10 * x5;
    const real_t x13 = x4 * x9;
    const real_t x14 = -x0 * x12 + x0 * x3 + x10 * x8 * x9 - x11 * x6 - x13 * x2 + x4 * x7;
    const real_t x15 = x10 * x8 - x2 * x4;
    const real_t x16 = POW2(x15);
    const real_t x17 = u[1] * x16;
    // const real_t x18 = pow(x14, -2);
    const real_t x18 = 1. / POW2(x14);
    const real_t x19 = (1.0 / 30.0) * x18;
    const real_t x20 = u[0] * x19;
    const real_t x21 = -x0 * x10 + x4 * x6;
    const real_t x22 = POW2(x21);
    const real_t x23 = u[1] * x20;
    const real_t x24 = x0 * x2 - x6 * x8;
    const real_t x25 = POW2(x24);
    const real_t x26 = -x11 + x4 * x5;
    const real_t x27 = POW2(x26);
    const real_t x28 = u[2] * x20;
    const real_t x29 = x0 * x1 - x13;
    const real_t x30 = POW2(x29);
    const real_t x31 = u[2] * x30;
    const real_t x32 = -x0 * x5 + x8 * x9;
    const real_t x33 = POW2(x32);
    const real_t x34 = -x12 + x3;
    const real_t x35 = POW2(x34);
    const real_t x36 = u[3] * x35;
    const real_t x37 = -x1 * x6 + x10 * x9;
    const real_t x38 = POW2(x37);
    const real_t x39 = u[3] * x20;
    const real_t x40 = -x2 * x9 + x7;
    const real_t x41 = POW2(x40);
    const real_t x42 = u[4] * x20;
    const real_t x43 = u[4] * x30;
    const real_t x44 = (2.0 / 15.0) * x18;
    const real_t x45 = u[0] * x44;
    const real_t x46 = u[4] * x45;
    const real_t x47 = u[5] * x20;
    const real_t x48 = u[6] * x20;
    const real_t x49 = u[6] * x45;
    const real_t x50 = u[7] * x45;
    const real_t x51 = u[7] * x20;
    const real_t x52 = u[8] * x35;
    const real_t x53 = u[8] * x20;
    const real_t x54 = u[8] * x16;
    const real_t x55 = u[9] * x20;
    const real_t x56 = u[4] * x44;
    const real_t x57 = u[1] * x56;
    const real_t x58 = u[5] * x19;
    const real_t x59 = u[1] * x58;
    const real_t x60 = x17 * x19;
    const real_t x61 = u[1] * x19;
    const real_t x62 = u[6] * x61;
    const real_t x63 = u[7] * x61;
    const real_t x64 = u[8] * x61;
    const real_t x65 = u[2] * x19;
    const real_t x66 = u[4] * x65;
    const real_t x67 = u[2] * x58;
    const real_t x68 = u[2] * u[6];
    const real_t x69 = x44 * x68;
    const real_t x70 = x30 * x44;
    const real_t x71 = u[7] * x65;
    const real_t x72 = u[9] * x19;
    const real_t x73 = u[2] * x72;
    const real_t x74 = u[4] * x19;
    const real_t x75 = u[3] * x74;
    const real_t x76 = u[6] * x19;
    const real_t x77 = u[3] * x76;
    const real_t x78 = u[7] * x44;
    const real_t x79 = u[3] * x78;
    const real_t x80 = u[8] * x19;
    const real_t x81 = u[3] * x80;
    const real_t x82 = u[3] * x72;
    const real_t x83 = (4.0 / 15.0) * x18;
    const real_t x84 = u[5] * x83;
    const real_t x85 = u[4] * x84;
    const real_t x86 = u[6] * x56;
    const real_t x87 = u[7] * x56;
    const real_t x88 = x35 * x83;
    const real_t x89 = u[4] * u[8];
    const real_t x90 = x83 * x89;
    const real_t x91 = u[9] * x56;
    const real_t x92 = u[9] * x44;
    const real_t x93 = u[6] * x84;
    const real_t x94 = u[5] * x78;
    const real_t x95 = u[5] * x70;
    const real_t x96 = u[5] * x44;
    const real_t x97 = u[8] * x96;
    const real_t x98 = u[5] * x92;
    const real_t x99 = u[6] * x78;
    const real_t x100 = u[6] * x44;
    const real_t x101 = u[8] * x100;
    const real_t x102 = u[6] * u[9];
    const real_t x103 = x102 * x83;
    const real_t x104 = u[7] * x83;
    const real_t x105 = u[8] * x104;
    const real_t x106 = u[9] * x104;
    const real_t x107 = u[8] * x92;
    const real_t x108 = POW2(u[0]);
    const real_t x109 = (1.0 / 20.0) * x18;
    const real_t x110 = x108 * x109;
    const real_t x111 = POW2(u[1]) * x109;
    const real_t x112 = POW2(u[2]) * x109;
    const real_t x113 = POW2(u[3]) * x109;
    const real_t x114 = POW2(u[4]);
    const real_t x115 = x114 * x44;
    const real_t x116 = POW2(u[5]) * x44;
    const real_t x117 = POW2(u[6]);
    const real_t x118 = x117 * x44;
    const real_t x119 = POW2(u[7]);
    const real_t x120 = x119 * x44;
    const real_t x121 = POW2(u[8]) * x44;
    const real_t x122 = POW2(u[9]) * x44;
    const real_t x123 = x15 * x34;
    const real_t x124 = x21 * x37;
    const real_t x125 = x15 * x26;
    const real_t x126 = x21 * x29;
    const real_t x127 = x24 * x40;
    const real_t x128 = x24 * x32;
    const real_t x129 = x26 * x34;
    const real_t x130 = x29 * x37;
    const real_t x131 = x32 * x40;
    const real_t x132 = u[4] * x129;
    const real_t x133 = u[0] * x18;
    const real_t x134 = (1.0 / 15.0) * x133;
    const real_t x135 = u[4] * x123;
    const real_t x136 = (1.0 / 6.0) * x133;
    const real_t x137 = u[4] * x134;
    const real_t x138 = u[4] * x136;
    const real_t x139 = u[5] * x134;
    const real_t x140 = u[5] * x126;
    const real_t x141 = u[6] * x136;
    const real_t x142 = u[6] * x123;
    const real_t x143 = u[6] * x134;
    const real_t x144 = u[7] * x136;
    const real_t x145 = u[7] * x134;
    const real_t x146 = u[8] * x123;
    const real_t x147 = u[8] * x134;
    const real_t x148 = u[9] * x134;
    const real_t x149 = u[2] * x61;
    const real_t x150 = u[3] * x61;
    const real_t x151 = (1.0 / 10.0) * x18;
    const real_t x152 = u[1] * x151;
    const real_t x153 = u[4] * x152;
    const real_t x154 = u[5] * x152;
    const real_t x155 = u[8] * x152;
    const real_t x156 = u[9] * x61;
    const real_t x157 = u[3] * x129;
    const real_t x158 = u[3] * x65;
    const real_t x159 = u[2] * x151;
    const real_t x160 = u[5] * x159;
    const real_t x161 = x151 * x68;
    const real_t x162 = u[2] * x80;
    const real_t x163 = u[9] * x159;
    const real_t x164 = u[3] * x58;
    const real_t x165 = u[7] * x151;
    const real_t x166 = u[3] * x165;
    const real_t x167 = u[3] * x151;
    const real_t x168 = u[8] * x167;
    const real_t x169 = u[9] * x167;
    const real_t x170 = u[5] * x56;
    const real_t x171 = u[4] * u[6] * x83;
    const real_t x172 = u[4] * x104;
    const real_t x173 = x44 * x89;
    const real_t x174 = u[9] * x83;
    const real_t x175 = u[4] * x174;
    const real_t x176 = u[5] * x100;
    const real_t x177 = u[7] * x84;
    const real_t x178 = u[8] * x84;
    const real_t x179 = u[9] * x84;
    const real_t x180 = u[6] * x104;
    const real_t x181 = u[8] * x83;
    const real_t x182 = u[6] * x181;
    const real_t x183 = x102 * x44;
    const real_t x184 = u[8] * x78;
    const real_t x185 = u[9] * x78;
    const real_t x186 = u[8] * x174;
    const real_t x187 = x108 * x151;
    const real_t x188 = x15 * x187;
    const real_t x189 = x187 * x29;
    const real_t x190 = x187 * x40;
    const real_t x191 = x114 * x83;
    const real_t x192 = x115 * x15;
    const real_t x193 = x117 * x83;
    const real_t x194 = x119 * x83;
    element_scalar[0] =
        x14 *
        (u[6] * x60 + u[7] * x19 * x31 + u[7] * x60 - u[7] * x95 + u[9] * x151 * x157 + u[9] * x95 -
         x100 * x52 - x100 * x54 - x101 * x22 - x101 * x25 - x101 * x38 - x101 * x41 - x102 * x88 -
         x103 * x123 - x103 * x124 - x103 * x127 - x103 * x129 - x103 * x130 - x103 * x131 -
         x103 * x38 - x103 * x41 + x104 * x135 - x104 * x54 - x105 * x123 - x105 * x124 -
         x105 * x125 - x105 * x126 - x105 * x127 - x105 * x128 - x105 * x22 - x105 * x25 -
         x106 * x125 - x106 * x126 - x106 * x128 - x106 * x129 - x106 * x130 - x106 * x131 -
         x106 * x27 - x106 * x30 - x106 * x33 + x107 * x123 + x107 * x124 + x107 * x127 +
         x107 * x129 + x107 * x130 + x107 * x131 + x107 * x38 + x107 * x41 + x110 * x16 +
         x110 * x22 + x110 * x25 + x110 * x27 + x110 * x30 + x110 * x33 + x110 * x35 + x110 * x38 +
         x110 * x41 + x111 * x16 + x111 * x22 + x111 * x25 + x112 * x27 + x112 * x30 + x112 * x33 +
         x113 * x35 + x113 * x38 + x113 * x41 + x115 * x124 + x115 * x126 + x115 * x127 +
         x115 * x128 + x115 * x16 + x115 * x22 + x115 * x25 + x115 * x27 + x115 * x30 + x115 * x33 +
         x115 * x35 + x115 * x38 + x115 * x41 + x116 * x125 + x116 * x126 + x116 * x128 +
         x116 * x16 + x116 * x22 + x116 * x25 + x116 * x27 + x116 * x30 + x116 * x33 + x118 * x125 +
         x118 * x126 + x118 * x128 + x118 * x129 + x118 * x130 + x118 * x131 + x118 * x16 +
         x118 * x22 + x118 * x25 + x118 * x27 + x118 * x30 + x118 * x33 + x118 * x35 + x118 * x38 +
         x118 * x41 + x120 * x123 + x120 * x124 + x120 * x127 + x120 * x129 + x120 * x130 +
         x120 * x131 + x120 * x16 + x120 * x22 + x120 * x25 + x120 * x27 + x120 * x30 + x120 * x33 +
         x120 * x35 + x120 * x38 + x120 * x41 + x121 * x123 + x121 * x124 + x121 * x127 +
         x121 * x16 + x121 * x22 + x121 * x25 + x121 * x35 + x121 * x38 + x121 * x41 + x122 * x129 +
         x122 * x130 + x122 * x131 + x122 * x27 + x122 * x30 + x122 * x33 + x122 * x35 +
         x122 * x38 + x122 * x41 - x123 * x144 - x123 * x150 + x123 * x155 - x123 * x156 -
         x123 * x164 - x123 * x166 - x123 * x170 + x123 * x179 - x123 * x185 + x123 * x193 +
         x123 * x23 + x123 * x39 + x123 * x47 + x123 * x55 + x123 * x62 + x123 * x77 + x123 * x86 -
         x123 * x90 - x123 * x93 + x123 * x97 + x123 * x99 - x124 * x138 - x124 * x143 -
         x124 * x144 + x124 * x147 - x124 * x150 - x124 * x153 + x124 * x155 - x124 * x156 -
         x124 * x164 - x124 * x166 + x124 * x168 - x124 * x170 + x124 * x172 + x124 * x179 -
         x124 * x182 - x124 * x185 + x124 * x187 + x124 * x193 + x124 * x23 + x124 * x39 +
         x124 * x47 + x124 * x55 + x124 * x62 + x124 * x77 + x124 * x86 - x124 * x90 - x124 * x93 +
         x124 * x97 + x124 * x99 - x125 * x138 + x125 * x139 - x125 * x141 - x125 * x145 -
         x125 * x149 - x125 * x153 + x125 * x154 - x125 * x156 + x125 * x160 - x125 * x161 -
         x125 * x162 + x125 * x171 - x125 * x173 - x125 * x177 - x125 * x183 + x125 * x186 +
         x125 * x194 + x125 * x23 + x125 * x28 + x125 * x53 + x125 * x55 + x125 * x63 + x125 * x71 -
         x125 * x85 + x125 * x87 - x125 * x93 + x125 * x97 + x125 * x98 + x125 * x99 - x126 * x138 -
         x126 * x141 - x126 * x145 - x126 * x149 - x126 * x153 + x126 * x154 - x126 * x156 -
         x126 * x161 - x126 * x162 + x126 * x171 - x126 * x173 - x126 * x177 - x126 * x183 +
         x126 * x186 + x126 * x194 + x126 * x23 + x126 * x28 + x126 * x53 + x126 * x55 +
         x126 * x63 + x126 * x71 - x126 * x85 + x126 * x87 - x126 * x93 + x126 * x97 + x126 * x98 +
         x126 * x99 - x127 * x138 - x127 * x143 - x127 * x144 + x127 * x147 - x127 * x150 -
         x127 * x153 + x127 * x155 - x127 * x156 - x127 * x164 - x127 * x166 + x127 * x168 -
         x127 * x170 + x127 * x172 + x127 * x179 - x127 * x182 - x127 * x185 + x127 * x193 +
         x127 * x23 + x127 * x39 + x127 * x47 + x127 * x55 + x127 * x62 + x127 * x77 + x127 * x86 -
         x127 * x90 - x127 * x93 + x127 * x97 + x127 * x99 - x128 * x138 + x128 * x139 -
         x128 * x141 - x128 * x145 - x128 * x149 - x128 * x153 + x128 * x154 - x128 * x156 +
         x128 * x160 - x128 * x161 - x128 * x162 + x128 * x171 - x128 * x173 - x128 * x177 -
         x128 * x183 + x128 * x186 + x128 * x187 + x128 * x194 + x128 * x23 + x128 * x28 +
         x128 * x53 + x128 * x55 + x128 * x63 + x128 * x71 - x128 * x85 + x128 * x87 - x128 * x93 +
         x128 * x97 + x128 * x98 + x128 * x99 - x129 * x141 - x129 * x144 + x129 * x148 -
         x129 * x161 - x129 * x162 + x129 * x163 - x129 * x176 + x129 * x178 + x129 * x180 -
         x129 * x184 + x129 * x187 + x129 * x191 + x129 * x28 + x129 * x39 + x129 * x47 +
         x129 * x53 + x129 * x66 + x129 * x75 - x129 * x85 + x129 * x86 + x129 * x87 - x129 * x90 +
         x129 * x98 - x130 * x137 - x130 * x141 - x130 * x144 + x130 * x148 - x130 * x158 -
         x130 * x161 - x130 * x162 + x130 * x163 - x130 * x164 - x130 * x166 + x130 * x169 -
         x130 * x175 - x130 * x176 + x130 * x178 + x130 * x180 - x130 * x184 + x130 * x191 +
         x130 * x28 + x130 * x39 + x130 * x47 + x130 * x53 + x130 * x66 + x130 * x75 - x130 * x85 +
         x130 * x86 + x130 * x87 - x130 * x90 + x130 * x98 - x131 * x137 - x131 * x141 -
         x131 * x144 + x131 * x148 - x131 * x158 - x131 * x161 - x131 * x162 + x131 * x163 -
         x131 * x164 - x131 * x166 + x131 * x169 - x131 * x175 - x131 * x176 + x131 * x178 +
         x131 * x180 - x131 * x184 + x131 * x191 + x131 * x28 + x131 * x39 + x131 * x47 +
         x131 * x53 + x131 * x66 + x131 * x75 - x131 * x85 + x131 * x86 + x131 * x87 - x131 * x90 +
         x131 * x98 - x132 * x134 - x132 * x174 + x134 * x140 - x134 * x142 + x134 * x146 -
         x135 * x136 - x135 * x152 + x140 * x159 - x142 * x181 + x146 * x167 - x157 * x165 -
         x157 * x58 - x157 * x65 - x16 * x46 + x16 * x47 - x16 * x48 - x16 * x51 - x16 * x93 -
         x16 * x94 + x16 * x99 + x17 * x20 - x17 * x56 - x17 * x58 + x188 * x26 + x188 * x34 +
         x189 * x21 + x189 * x37 + x190 * x24 + x190 * x32 + x192 * x26 + x192 * x34 + x20 * x31 +
         x20 * x36 - x20 * x43 + x20 * x52 + x20 * x54 + x22 * x23 - x22 * x46 + x22 * x47 -
         x22 * x48 - x22 * x51 + x22 * x53 - x22 * x57 - x22 * x59 + x22 * x62 + x22 * x63 -
         x22 * x64 - x22 * x93 - x22 * x94 + x22 * x97 + x22 * x99 + x23 * x25 - x25 * x46 +
         x25 * x47 - x25 * x48 - x25 * x51 + x25 * x53 - x25 * x57 - x25 * x59 + x25 * x62 +
         x25 * x63 - x25 * x64 - x25 * x93 - x25 * x94 + x25 * x97 + x25 * x99 + x27 * x28 -
         x27 * x42 + x27 * x47 - x27 * x49 - x27 * x51 + x27 * x55 + x27 * x66 - x27 * x67 -
         x27 * x69 + x27 * x71 - x27 * x73 - x27 * x85 + x27 * x87 - x27 * x91 - x27 * x94 +
         x27 * x98 + x28 * x33 + x30 * x47 - x30 * x49 - x30 * x51 + x30 * x55 - x30 * x67 -
         x31 * x72 - x33 * x42 + x33 * x47 - x33 * x49 - x33 * x51 + x33 * x55 + x33 * x66 -
         x33 * x67 - x33 * x69 + x33 * x71 - x33 * x73 - x33 * x85 + x33 * x87 - x33 * x91 -
         x33 * x94 + x33 * x98 - x35 * x42 - x35 * x48 - x35 * x50 + x35 * x55 + x35 * x86 -
         x35 * x91 - x36 * x72 + x36 * x74 + x36 * x76 - x36 * x78 - x36 * x80 + x38 * x39 -
         x38 * x42 - x38 * x48 - x38 * x50 + x38 * x53 + x38 * x55 + x38 * x75 + x38 * x77 -
         x38 * x79 - x38 * x81 - x38 * x82 + x38 * x86 - x38 * x90 - x38 * x91 + x39 * x41 -
         x41 * x42 - x41 * x48 - x41 * x50 + x41 * x53 + x41 * x55 + x41 * x75 + x41 * x77 -
         x41 * x79 - x41 * x81 - x41 * x82 + x41 * x86 - x41 * x90 - x41 * x91 + x43 * x65 +
         x43 * x78 - x43 * x84 - x43 * x92 + x52 * x92 - x54 * x61 + x54 * x96 - x68 * x70 -
         x88 * x89);
}

static SFEM_INLINE int linear_search(const idx_t target, const idx_t *const arr, const int size) {
    int i;
    for (i = 0; i < size - 4; i += 4) {
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

static SFEM_INLINE void find_cols10(const idx_t *targets,
                                    const idx_t *const row,
                                    const int lenrow,
                                    int *ks) {
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

#pragma omp parallel
    {
#pragma omp for //nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[10];
            idx_t ks[10];

            real_t element_matrix[10 * 10];

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
#pragma omp atomic update
                    rowvalues[ks[edof_j]] += element_row[edof_j];
                }
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

#pragma omp parallel
    {
#pragma omp for //nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[10];
            real_t element_vector[10 * 10];
            real_t element_u[10];

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

#pragma omp atomic update
                values[dof_i] += element_vector[edof_i];
            }
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

#pragma omp parallel
    {
#pragma omp for //nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[10];
            real_t element_u[10];

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

#pragma omp atomic update
            *value += element_scalar;
        }
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
