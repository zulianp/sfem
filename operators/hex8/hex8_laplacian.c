#include "hex8_laplacian.h"

#include "sfem_defs.h"

#include "tet4_inline_cpu.h"
#include "hex8_quadrature.h"

#define POW2(a) ((a) * (a))


static SFEM_INLINE void hex8_fff(const scalar_t *const SFEM_RESTRICT x,
                                 const scalar_t *const SFEM_RESTRICT y,
                                 const scalar_t *const SFEM_RESTRICT z,
                                 const scalar_t qx,
                                 const scalar_t qy,
                                 const scalar_t qz,
                                 scalar_t *const SFEM_RESTRICT fff) {
    const scalar_t x0 = qy*qz;
    const scalar_t x1 = 1 - qz;
    const scalar_t x2 = qy*x1;
    const scalar_t x3 = 1 - qy;
    const scalar_t x4 = qz*x3;
    const scalar_t x5 = x1*x3;
    const scalar_t x6 = x0*z[6] - x0*z[7] + x2*z[2] - x2*z[3] - x4*z[4] + x4*z[5] - x5*z[0] + x5*z[1];
    const scalar_t x7 = qx*qy;
    const scalar_t x8 = qx*x3;
    const scalar_t x9 = 1 - qx;
    const scalar_t x10 = qy*x9;
    const scalar_t x11 = x3*x9;
    const scalar_t x12 = qx*qy*x[6] + qx*x3*x[5] + qy*x9*x[7] - x10*x[3] - x11*x[0] + x3*x9*x[4] - x7*x[2] - x8*x[1];
    const scalar_t x13 = qx*qz;
    const scalar_t x14 = qx*x1;
    const scalar_t x15 = qz*x9;
    const scalar_t x16 = x1*x9;
    const scalar_t x17 = qx*qz*y[6] + qx*x1*y[2] + qz*x9*y[7] + x1*x9*y[3] - x13*y[5] - x14*y[1] - x15*y[4] - x16*y[0];
    const scalar_t x18 = x12*x17;
    const scalar_t x19 = x0*x[6] - x0*x[7] + x2*x[2] - x2*x[3] - x4*x[4] + x4*x[5] - x5*x[0] + x5*x[1];
    const scalar_t x20 = qx*qy*y[6] + qx*x3*y[5] + qy*x9*y[7] - x10*y[3] - x11*y[0] + x3*x9*y[4] - x7*y[2] - x8*y[1];
    const scalar_t x21 = qx*qz*z[6] + qx*x1*z[2] + qz*x9*z[7] + x1*x9*z[3] - x13*z[5] - x14*z[1] - x15*z[4] - x16*z[0];
    const scalar_t x22 = x20*x21;
    const scalar_t x23 = x0*y[6] - x0*y[7] + x2*y[2] - x2*y[3] - x4*y[4] + x4*y[5] - x5*y[0] + x5*y[1];
    const scalar_t x24 = qx*qy*z[6] + qx*x3*z[5] + qy*x9*z[7] - x10*z[3] - x11*z[0] + x3*x9*z[4] - x7*z[2] - x8*z[1];
    const scalar_t x25 = qx*qz*x[6] + qx*x1*x[2] + qz*x9*x[7] + x1*x9*x[3] - x13*x[5] - x14*x[1] - x15*x[4] - x16*x[0];
    const scalar_t x26 = x24*x25;
    const scalar_t x27 = x12*x21*x23 + x17*x19*x24 - x18*x6 - x19*x22 + x20*x25*x6 - x23*x26;
    const scalar_t x28 = -x18 + x20*x25;
    const scalar_t x29 = (1/POW2(x27));
    const scalar_t x30 = x12*x21 - x26;
    const scalar_t x31 = x17*x24 - x22;
    const scalar_t x32 = x12*x23 - x19*x20;
    const scalar_t x33 = x28*x29;
    const scalar_t x34 = -x12*x6 + x19*x24;
    const scalar_t x35 = x29*x30;
    const scalar_t x36 = x20*x6 - x23*x24;
    const scalar_t x37 = x29*x31;
    const scalar_t x38 = x17*x19 - x23*x25;
    const scalar_t x39 = -x19*x21 + x25*x6;
    const scalar_t x40 = -x17*x6 + x21*x23;
    fff[0] = x27*(POW2(x28)*x29 + x29*POW2(x30) + x29*POW2(x31));
    fff[1] = x27*(x32*x33 + x34*x35 + x36*x37);
    fff[2] = x27*(x33*x38 + x35*x39 + x37*x40);
    fff[3] = x27*(x29*POW2(x32) + x29*POW2(x34) + x29*POW2(x36));
    fff[4] = x27*(x29*x32*x38 + x29*x34*x39 + x29*x36*x40);
    fff[5] = x27*(x29*POW2(x38) + x29*POW2(x39) + x29*POW2(x40));
}

static SFEM_INLINE void hex8_laplacian_apply_fff_integral(const scalar_t *const SFEM_RESTRICT fff,
                                                          const scalar_t *SFEM_RESTRICT u,
                                                          accumulator_t *SFEM_RESTRICT
                                                                  element_vector) {
    const scalar_t x0 = (1.0/6.0)*fff[4];
    const scalar_t x1 = u[7]*x0;
    const scalar_t x2 = (1.0/9.0)*fff[3];
    const scalar_t x3 = u[3]*x2;
    const scalar_t x4 = (1.0/9.0)*fff[5];
    const scalar_t x5 = u[4]*x4;
    const scalar_t x6 = (1.0/12.0)*u[6];
    const scalar_t x7 = fff[4]*x6;
    const scalar_t x8 = (1.0/36.0)*u[6];
    const scalar_t x9 = fff[3]*x8;
    const scalar_t x10 = fff[5]*x8;
    const scalar_t x11 = u[0]*x0;
    const scalar_t x12 = u[0]*x2;
    const scalar_t x13 = u[0]*x4;
    const scalar_t x14 = (1.0/12.0)*fff[4];
    const scalar_t x15 = u[1]*x14;
    const scalar_t x16 = (1.0/36.0)*fff[3];
    const scalar_t x17 = u[5]*x16;
    const scalar_t x18 = (1.0/36.0)*fff[5];
    const scalar_t x19 = u[2]*x18;
    const scalar_t x20 = (1.0/6.0)*fff[1];
    const scalar_t x21 = (1.0/12.0)*fff[1];
    const scalar_t x22 = -fff[1]*x6 + u[0]*x20 - u[2]*x20 + u[4]*x21;
    const scalar_t x23 = (1.0/6.0)*fff[2];
    const scalar_t x24 = (1.0/12.0)*fff[2];
    const scalar_t x25 = -fff[2]*x6 + u[0]*x23 + u[3]*x24 - u[5]*x23;
    const scalar_t x26 = (1.0/9.0)*fff[0];
    const scalar_t x27 = (1.0/36.0)*u[7];
    const scalar_t x28 = (1.0/18.0)*fff[0];
    const scalar_t x29 = -u[2]*x28 + u[3]*x28 + u[4]*x28 - u[5]*x28;
    const scalar_t x30 = fff[0]*x27 - fff[0]*x8 + u[0]*x26 - u[1]*x26 + x29;
    const scalar_t x31 = (1.0/18.0)*fff[3];
    const scalar_t x32 = u[2]*x31;
    const scalar_t x33 = u[7]*x31;
    const scalar_t x34 = (1.0/18.0)*fff[5];
    const scalar_t x35 = u[5]*x34;
    const scalar_t x36 = u[7]*x34;
    const scalar_t x37 = u[1]*x31;
    const scalar_t x38 = u[4]*x31;
    const scalar_t x39 = u[1]*x34;
    const scalar_t x40 = u[3]*x34;
    const scalar_t x41 = -x32 - x33 - x35 - x36 + x37 + x38 + x39 + x40;
    const scalar_t x42 = u[1]*x0;
    const scalar_t x43 = u[1]*x2;
    const scalar_t x44 = u[1]*x4;
    const scalar_t x45 = u[0]*x14;
    const scalar_t x46 = u[4]*x16;
    const scalar_t x47 = u[3]*x18;
    const scalar_t x48 = u[6]*x0;
    const scalar_t x49 = u[2]*x2;
    const scalar_t x50 = u[5]*x4;
    const scalar_t x51 = u[7]*x14;
    const scalar_t x52 = fff[3]*x27;
    const scalar_t x53 = fff[5]*x27;
    const scalar_t x54 = u[1]*x20 - u[3]*x20 + u[5]*x21 - u[7]*x21;
    const scalar_t x55 = u[1]*x23 + u[2]*x24 - u[4]*x23 - u[7]*x24;
    const scalar_t x56 = u[0]*x31;
    const scalar_t x57 = u[5]*x31;
    const scalar_t x58 = u[0]*x34;
    const scalar_t x59 = u[2]*x34;
    const scalar_t x60 = u[3]*x31;
    const scalar_t x61 = u[6]*x31;
    const scalar_t x62 = u[4]*x34;
    const scalar_t x63 = u[6]*x34;
    const scalar_t x64 = -x56 - x57 - x58 - x59 + x60 + x61 + x62 + x63;
    const scalar_t x65 = u[5]*x0;
    const scalar_t x66 = u[2]*x4;
    const scalar_t x67 = u[4]*x14;
    const scalar_t x68 = u[0]*x18;
    const scalar_t x69 = u[2]*x0;
    const scalar_t x70 = u[6]*x4;
    const scalar_t x71 = u[3]*x14;
    const scalar_t x72 = u[4]*x18;
    const scalar_t x73 = u[1]*x24 + u[2]*x23 - u[4]*x24 - u[7]*x23;
    const scalar_t x74 = (1.0/36.0)*fff[0];
    const scalar_t x75 = u[0]*x28 - u[1]*x28 - u[6]*x28 + u[7]*x28;
    const scalar_t x76 = -u[2]*x26 + u[3]*x26 + u[4]*x74 - u[5]*x74 + x75;
    const scalar_t x77 = x35 + x36 - x39 - x40 + x56 + x57 - x60 - x61;
    const scalar_t x78 = u[3]*x0;
    const scalar_t x79 = u[7]*x4;
    const scalar_t x80 = u[2]*x14;
    const scalar_t x81 = u[5]*x18;
    const scalar_t x82 = u[4]*x0;
    const scalar_t x83 = u[3]*x4;
    const scalar_t x84 = u[5]*x14;
    const scalar_t x85 = u[1]*x18;
    const scalar_t x86 = u[0]*x24 + u[3]*x23 - u[5]*x24 - u[6]*x23;
    const scalar_t x87 = x32 + x33 - x37 - x38 + x58 + x59 - x62 - x63;
    const scalar_t x88 = u[7]*x2;
    const scalar_t x89 = u[2]*x16;
    const scalar_t x90 = u[4]*x2;
    const scalar_t x91 = u[1]*x16;
    const scalar_t x92 = u[0]*x21 - u[2]*x21 + u[4]*x20 - u[6]*x20;
    const scalar_t x93 = -u[2]*x74 + u[3]*x74 + u[4]*x26 - u[5]*x26 + x75;
    const scalar_t x94 = u[5]*x2;
    const scalar_t x95 = u[0]*x16;
    const scalar_t x96 = u[6]*x2;
    const scalar_t x97 = u[3]*x16;
    const scalar_t x98 = u[1]*x21 - u[3]*x21 + u[5]*x20 - u[7]*x20;
    const scalar_t x99 = u[0]*x74 - u[1]*x74 - u[6]*x26 + u[7]*x26 + x29;
    element_vector[0] = -x1 - x10 + x11 + x12 + x13 + x15 + x17 + x19 + x22 + x25 - x3 + x30 + x41 - x5 - x7 - x9;
    element_vector[1] = -x30 + x42 + x43 + x44 + x45 + x46 + x47 - x48 - x49 - x50 - x51 - x52 - x53 - x54 - x55 - x64;
    element_vector[2] = -x22 - x43 - x46 + x49 + x52 + x65 + x66 + x67 + x68 - x69 - x70 - x71 - x72 - x73 - x76 - x77;
    element_vector[3] = -x12 - x17 + x3 + x54 + x76 - x78 - x79 - x80 - x81 + x82 + x83 + x84 + x85 + x86 + x87 + x9;
    element_vector[4] = x10 - x13 - x19 + x5 + x55 + x77 + x78 + x80 - x82 - x84 - x88 - x89 + x90 + x91 + x92 + x93;
    element_vector[5] = -x25 - x44 - x47 + x50 + x53 - x65 - x67 + x69 + x71 - x87 - x93 + x94 + x95 - x96 - x97 - x98;
    element_vector[6] = -x41 - x42 - x45 + x48 + x51 - x66 - x68 + x70 + x72 - x86 - x92 - x94 - x95 + x96 + x97 - x99;
    element_vector[7] = x1 - x11 - x15 + x64 + x7 + x73 + x79 + x81 - x83 - x85 + x88 + x89 - x90 - x91 + x98 + x99;
}

static SFEM_INLINE void aahex8_jac_diag(const scalar_t px0,
                            const scalar_t px6,
                            const scalar_t py0,
                            const scalar_t py6,
                            const scalar_t pz0,
                            const scalar_t pz6,
                            scalar_t *const SFEM_RESTRICT jac_diag) {
    const scalar_t x0 = -py0 + py6;
    const scalar_t x1 = -pz0 + pz6;
    const scalar_t x2 = -px0 + px6;
    jac_diag[0] = x0 * x1 / x2;
    jac_diag[1] = x1 * x2 / x0;
    jac_diag[2] = x0 * x2 / x1;
}

static SFEM_INLINE void aahex8_laplacian_apply_integral(
        const scalar_t *const SFEM_RESTRICT jac_diag,
        const scalar_t *SFEM_RESTRICT u,
        accumulator_t *SFEM_RESTRICT element_vector) {
    const scalar_t x0 = (1.0 / 9.0) * jac_diag[1];
    const scalar_t x1 = u[3] * x0;
    const scalar_t x2 = (1.0 / 9.0) * jac_diag[2];
    const scalar_t x3 = u[4] * x2;
    const scalar_t x4 = (1.0 / 36.0) * u[6];
    const scalar_t x5 = jac_diag[1] * x4;
    const scalar_t x6 = jac_diag[2] * x4;
    const scalar_t x7 = u[0] * x0;
    const scalar_t x8 = u[0] * x2;
    const scalar_t x9 = (1.0 / 36.0) * jac_diag[1];
    const scalar_t x10 = u[5] * x9;
    const scalar_t x11 = (1.0 / 36.0) * jac_diag[2];
    const scalar_t x12 = u[2] * x11;
    const scalar_t x13 = (1.0 / 9.0) * jac_diag[0];
    const scalar_t x14 = (1.0 / 36.0) * u[7];
    const scalar_t x15 = (1.0 / 18.0) * jac_diag[0];
    const scalar_t x16 = -u[2] * x15 + u[3] * x15 + u[4] * x15 - u[5] * x15;
    const scalar_t x17 = jac_diag[0] * x14 - jac_diag[0] * x4 + u[0] * x13 - u[1] * x13 + x16;
    const scalar_t x18 = (1.0 / 18.0) * jac_diag[1];
    const scalar_t x19 = u[2] * x18;
    const scalar_t x20 = u[7] * x18;
    const scalar_t x21 = (1.0 / 18.0) * jac_diag[2];
    const scalar_t x22 = u[5] * x21;
    const scalar_t x23 = u[7] * x21;
    const scalar_t x24 = u[1] * x18;
    const scalar_t x25 = u[4] * x18;
    const scalar_t x26 = u[1] * x21;
    const scalar_t x27 = u[3] * x21;
    const scalar_t x28 = -x19 - x20 - x22 - x23 + x24 + x25 + x26 + x27;
    const scalar_t x29 = u[1] * x0;
    const scalar_t x30 = u[1] * x2;
    const scalar_t x31 = u[4] * x9;
    const scalar_t x32 = u[3] * x11;
    const scalar_t x33 = u[2] * x0;
    const scalar_t x34 = u[5] * x2;
    const scalar_t x35 = jac_diag[1] * x14;
    const scalar_t x36 = jac_diag[2] * x14;
    const scalar_t x37 = u[0] * x18;
    const scalar_t x38 = u[5] * x18;
    const scalar_t x39 = u[0] * x21;
    const scalar_t x40 = u[2] * x21;
    const scalar_t x41 = u[3] * x18;
    const scalar_t x42 = u[6] * x18;
    const scalar_t x43 = u[4] * x21;
    const scalar_t x44 = u[6] * x21;
    const scalar_t x45 = -x37 - x38 - x39 - x40 + x41 + x42 + x43 + x44;
    const scalar_t x46 = u[2] * x2;
    const scalar_t x47 = u[0] * x11;
    const scalar_t x48 = u[6] * x2;
    const scalar_t x49 = u[4] * x11;
    const scalar_t x50 = (1.0 / 36.0) * jac_diag[0];
    const scalar_t x51 = u[0] * x15 - u[1] * x15 - u[6] * x15 + u[7] * x15;
    const scalar_t x52 = -u[2] * x13 + u[3] * x13 + u[4] * x50 - u[5] * x50 + x51;
    const scalar_t x53 = x22 + x23 - x26 - x27 + x37 + x38 - x41 - x42;
    const scalar_t x54 = u[7] * x2;
    const scalar_t x55 = u[5] * x11;
    const scalar_t x56 = u[3] * x2;
    const scalar_t x57 = u[1] * x11;
    const scalar_t x58 = x19 + x20 - x24 - x25 + x39 + x40 - x43 - x44;
    const scalar_t x59 = u[7] * x0;
    const scalar_t x60 = u[2] * x9;
    const scalar_t x61 = u[4] * x0;
    const scalar_t x62 = u[1] * x9;
    const scalar_t x63 = -u[2] * x50 + u[3] * x50 + u[4] * x13 - u[5] * x13 + x51;
    const scalar_t x64 = u[5] * x0;
    const scalar_t x65 = u[0] * x9;
    const scalar_t x66 = u[6] * x0;
    const scalar_t x67 = u[3] * x9;
    const scalar_t x68 = u[0] * x50 - u[1] * x50 - u[6] * x13 + u[7] * x13 + x16;
    element_vector[0] = -x1 + x10 + x12 + x17 + x28 - x3 - x5 - x6 + x7 + x8;
    element_vector[1] = -x17 + x29 + x30 + x31 + x32 - x33 - x34 - x35 - x36 - x45;
    element_vector[2] = -x29 - x31 + x33 + x35 + x46 + x47 - x48 - x49 - x52 - x53;
    element_vector[3] = x1 - x10 + x5 + x52 - x54 - x55 + x56 + x57 + x58 - x7;
    element_vector[4] = -x12 + x3 + x53 - x59 + x6 - x60 + x61 + x62 + x63 - x8;
    element_vector[5] = -x30 - x32 + x34 + x36 - x58 - x63 + x64 + x65 - x66 - x67;
    element_vector[6] = -x28 - x46 - x47 + x48 + x49 - x64 - x65 + x66 + x67 - x68;
    element_vector[7] = x45 + x54 + x55 - x56 - x57 + x59 + x60 - x61 - x62 + x68;
}

static SFEM_INLINE void hex8_laplacian_apply_fff(const scalar_t *const SFEM_RESTRICT fff,
                                                 const scalar_t qx,
                                                 const scalar_t qy,
                                                 const scalar_t qz,
                                                 const scalar_t qw,
                                                 const scalar_t *SFEM_RESTRICT u,
                                                 accumulator_t *SFEM_RESTRICT element_vector) {
    scalar_t trial_operand[3];
    {
        const scalar_t x0 = qy * qz;
        const scalar_t x1 = 1 - qz;
        const scalar_t x2 = qy * x1;
        const scalar_t x3 = 1 - qy;
        const scalar_t x4 = qz * x3;
        const scalar_t x5 = x1 * x3;
        const scalar_t x6 = -u[0] * x5 + u[1] * x5 + u[2] * x2 - u[3] * x2 - u[4] * x4 + u[5] * x4 +
                            u[6] * x0 - u[7] * x0;
        const scalar_t x7 = 1 - qx;
        const scalar_t x8 = -qx * qz * u[5] + qx * qz * u[6] - qx * u[1] * x1 + qx * u[2] * x1 -
                            qz * u[4] * x7 + qz * u[7] * x7 - u[0] * x1 * x7 + u[3] * x1 * x7;
        const scalar_t x9 = -qx * qy * u[2] + qx * qy * u[6] - qx * u[1] * x3 + qx * u[5] * x3 -
                            qy * u[3] * x7 + qy * u[7] * x7 - u[0] * x3 * x7 + u[4] * x3 * x7;
        trial_operand[0] = qw * (fff[0] * x6 + fff[1] * x8 + fff[2] * x9);
        trial_operand[1] = qw * (fff[1] * x6 + fff[3] * x8 + fff[4] * x9);
        trial_operand[2] = qw * (fff[2] * x6 + fff[4] * x8 + fff[5] * x9);
    }

    // Dot product
    {
        const scalar_t x0 = 1 - qy;
        const scalar_t x1 = 1 - qz;
        const scalar_t x2 = trial_operand[0] * x1;
        const scalar_t x3 = x0 * x2;
        const scalar_t x4 = 1 - qx;
        const scalar_t x5 = trial_operand[1] * x1;
        const scalar_t x6 = x4 * x5;
        const scalar_t x7 = trial_operand[2] * x0;
        const scalar_t x8 = x4 * x7;
        const scalar_t x9 = qx * x5;
        const scalar_t x10 = qx * x7;
        const scalar_t x11 = qy * trial_operand[2];
        const scalar_t x12 = qx * x11;
        const scalar_t x13 = qy * x2;
        const scalar_t x14 = x11 * x4;
        const scalar_t x15 = qz * trial_operand[0];
        const scalar_t x16 = x0 * x15;
        const scalar_t x17 = qz * trial_operand[1];
        const scalar_t x18 = x17 * x4;
        const scalar_t x19 = qx * x17;
        const scalar_t x20 = qy * x15;
        element_vector[0] += -x3 - x6 - x8;
        element_vector[1] += -x10 + x3 - x9;
        element_vector[2] += -x12 + x13 + x9;
        element_vector[3] += -x13 - x14 + x6;
        element_vector[4] += -x16 - x18 + x8;
        element_vector[5] += x10 + x16 - x19;
        element_vector[6] += x12 + x19 + x20;
        element_vector[7] += x14 + x18 - x20;
    }
}

static SFEM_INLINE void hex8_laplacian_apply_points(const scalar_t *const SFEM_RESTRICT x,
                                                    const scalar_t *const SFEM_RESTRICT y,
                                                    const scalar_t *const SFEM_RESTRICT z,
                                                    const scalar_t qx,
                                                    const scalar_t qy,
                                                    const scalar_t qz,
                                                    const scalar_t qw,
                                                    const scalar_t *SFEM_RESTRICT u,
                                                    accumulator_t *SFEM_RESTRICT element_vector) {
    scalar_t trial_operand[3];
    {
        scalar_t fff[6];
        hex8_fff(x, y, z, qx, qy, qz, fff);

        const scalar_t x0 = qy * qz;
        const scalar_t x1 = 1 - qz;
        const scalar_t x2 = qy * x1;
        const scalar_t x3 = 1 - qy;
        const scalar_t x4 = qz * x3;
        const scalar_t x5 = x1 * x3;
        const scalar_t x6 = -u[0] * x5 + u[1] * x5 + u[2] * x2 - u[3] * x2 - u[4] * x4 + u[5] * x4 +
                            u[6] * x0 - u[7] * x0;
        const scalar_t x7 = 1 - qx;
        const scalar_t x8 = -qx * qz * u[5] + qx * qz * u[6] - qx * u[1] * x1 + qx * u[2] * x1 -
                            qz * u[4] * x7 + qz * u[7] * x7 - u[0] * x1 * x7 + u[3] * x1 * x7;
        const scalar_t x9 = -qx * qy * u[2] + qx * qy * u[6] - qx * u[1] * x3 + qx * u[5] * x3 -
                            qy * u[3] * x7 + qy * u[7] * x7 - u[0] * x3 * x7 + u[4] * x3 * x7;
        trial_operand[0] = qw * (fff[0] * x6 + fff[1] * x8 + fff[2] * x9);
        trial_operand[1] = qw * (fff[1] * x6 + fff[3] * x8 + fff[4] * x9);
        trial_operand[2] = qw * (fff[2] * x6 + fff[4] * x8 + fff[5] * x9);
    }

    // Dot product
    {
        const scalar_t x0 = 1 - qy;
        const scalar_t x1 = 1 - qz;
        const scalar_t x2 = trial_operand[0] * x1;
        const scalar_t x3 = x0 * x2;
        const scalar_t x4 = 1 - qx;
        const scalar_t x5 = trial_operand[1] * x1;
        const scalar_t x6 = x4 * x5;
        const scalar_t x7 = trial_operand[2] * x0;
        const scalar_t x8 = x4 * x7;
        const scalar_t x9 = qx * x5;
        const scalar_t x10 = qx * x7;
        const scalar_t x11 = qy * trial_operand[2];
        const scalar_t x12 = qx * x11;
        const scalar_t x13 = qy * x2;
        const scalar_t x14 = x11 * x4;
        const scalar_t x15 = qz * trial_operand[0];
        const scalar_t x16 = x0 * x15;
        const scalar_t x17 = qz * trial_operand[1];
        const scalar_t x18 = x17 * x4;
        const scalar_t x19 = qx * x17;
        const scalar_t x20 = qy * x15;
        element_vector[0] += -x3 - x6 - x8;
        element_vector[1] += -x10 + x3 - x9;
        element_vector[2] += -x12 + x13 + x9;
        element_vector[3] += -x13 - x14 + x6;
        element_vector[4] += -x16 - x18 + x8;
        element_vector[5] += x10 + x16 - x19;
        element_vector[6] += x12 + x19 + x20;
        element_vector[7] += x14 + x18 - x20;
    }
}

int hex8_laplacian_apply(const ptrdiff_t nelements,
                         const ptrdiff_t nnodes,
                         idx_t **const SFEM_RESTRICT elements,
                         geom_t **const SFEM_RESTRICT points,
                         const real_t *const SFEM_RESTRICT u,
                         real_t *const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

    int SFEM_HEX8_ASSUME_AXIS_ALIGNED = 0;
    int SFEM_HEX8_ASSUME_AFFINE = 0;
    SFEM_READ_ENV(SFEM_HEX8_ASSUME_AXIS_ALIGNED, atoi);
    SFEM_READ_ENV(SFEM_HEX8_ASSUME_AFFINE, atoi);

    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

    if (SFEM_HEX8_ASSUME_AXIS_ALIGNED) {
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[8];
            accumulator_t element_vector[8];
            scalar_t element_u[8];
            scalar_t jac_diag[3];

            for (int v = 0; v < 8; ++v) {
                ev[v] = elements[v][i];
            }

            for (int v = 0; v < 8; ++v) {
                element_u[v] = u[ev[v]];
            }

            // Assume affine here!
            aahex8_jac_diag(x[ev[0]], x[ev[6]], y[ev[0]], y[ev[6]], z[ev[0]], z[ev[6]], jac_diag);
            aahex8_laplacian_apply_integral(jac_diag, element_u, element_vector);

            for (int edof_i = 0; edof_i < 8; ++edof_i) {
                const idx_t dof_i = ev[edof_i];

#pragma omp atomic update
                values[dof_i] += element_vector[edof_i];
            }
        }
    } else if (SFEM_HEX8_ASSUME_AFFINE) {
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[8];
            accumulator_t element_vector[8];
            scalar_t element_u[8];
            scalar_t fff[6];

            for (int v = 0; v < 8; ++v) {
                ev[v] = elements[v][i];
            }

            for (int v = 0; v < 8; ++v) {
                element_u[v] = u[ev[v]];
            }

            // Assume affine here!
            tet4_fff_s(x[ev[0]],
                       x[ev[1]],
                       x[ev[3]],
                       x[ev[4]],
                       y[ev[0]],
                       y[ev[1]],
                       y[ev[3]],
                       y[ev[4]],
                       z[ev[0]],
                       z[ev[1]],
                       z[ev[3]],
                       z[ev[4]],
                       fff);

            hex8_laplacian_apply_fff_integral(fff, element_u, element_vector);

            for (int edof_i = 0; edof_i < 8; ++edof_i) {
                const idx_t dof_i = ev[edof_i];

#pragma omp atomic update
                values[dof_i] += element_vector[edof_i];
            }
        }

    } else {
        int SFEM_HEX8_QUADRATURE_ORDER = 27;
        SFEM_READ_ENV(SFEM_HEX8_QUADRATURE_ORDER, atoi);

        int n_qp = q27_n;
        const scalar_t *qx = q27_x;
        const scalar_t *qy = q27_y;
        const scalar_t *qz = q27_z;
        const scalar_t *qw = q27_w;

        if (SFEM_HEX8_QUADRATURE_ORDER == 58) {
            n_qp = q58_n;
            qx = q58_x;
            qy = q58_y;
            qz = q58_z;
            qw = q58_w;
        } else if (SFEM_HEX8_QUADRATURE_ORDER == 6) {
            n_qp = q6_n;
            qx = q6_x;
            qy = q6_y;
            qz = q6_z;
            qw = q6_w;
        }

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[8];
            accumulator_t element_vector[8] = {0};
            scalar_t element_u[8];

            for (int v = 0; v < 8; ++v) {
                ev[v] = elements[v][i];
            }

            for (int v = 0; v < 8; ++v) {
                element_u[v] = u[ev[v]];
            }

            const scalar_t lx[8] = {
                    x[ev[0]], x[ev[1]], x[ev[2]], x[ev[3]], x[ev[4]], x[ev[5]], x[ev[6]], x[ev[7]]};

            const scalar_t ly[8] = {
                    y[ev[0]], y[ev[1]], y[ev[2]], y[ev[3]], y[ev[4]], y[ev[5]], y[ev[6]], y[ev[7]]};

            const scalar_t lz[8] = {
                    z[ev[0]], z[ev[1]], z[ev[2]], z[ev[3]], z[ev[4]], z[ev[5]], z[ev[6]], z[ev[7]]};

            for (int k = 0; k < n_qp; k++) {
                hex8_laplacian_apply_points(
                        lx, ly, lz, qx[k], qy[k], qz[k], qw[k], element_u, element_vector);
            }

            for (int edof_i = 0; edof_i < 8; ++edof_i) {
                const idx_t dof_i = ev[edof_i];

#pragma omp atomic update
                values[dof_i] += element_vector[edof_i];
            }
        }
    }

    return SFEM_SUCCESS;
}
