#ifndef HEX8_NEOHOOKEAN_OGDEN_LOCAL_H
#define HEX8_NEOHOOKEAN_OGDEN_LOCAL_H

#include <math.h>
#include "sfem_base.h"

#include "hex8_partial_assembly_neohookean_inline.h"

#ifdef __cplusplus
extern "C" {
#endif

static SFEM_INLINE void hex8_neohookean_ogden_objective_at_qp(const scalar_t *const SFEM_RESTRICT adjugate,
                                                              const scalar_t                      jacobian_determinant,
                                                              const scalar_t                      qx,
                                                              const scalar_t                      qy,
                                                              const scalar_t                      qz,
                                                              const scalar_t                      qw,
                                                              const scalar_t                      mu,
                                                              const scalar_t                      lmbda,
                                                              const scalar_t *const SFEM_RESTRICT dispx,
                                                              const scalar_t *const SFEM_RESTRICT dispy,
                                                              const scalar_t *const SFEM_RESTRICT dispz,
                                                              scalar_t *const SFEM_RESTRICT       v) {
    scalar_t F[9];
    hex8_F(adjugate, jacobian_determinant, qx, qy, qz, dispx, dispy, dispz, F);

    // mundane ops: 49 divs: 0 sqrts: 0
    // total ops: 49
    const scalar_t x0 = log(F[0] * F[4] * F[8] - F[0] * F[5] * F[7] - F[1] * F[3] * F[8] + F[1] * F[5] * F[6] +
                            F[2] * F[3] * F[7] - F[2] * F[4] * F[6]);
    v[0] += jacobian_determinant * qw *
            ((1.0 / 2.0) * lmbda * POW2(x0) - mu * x0 +
             (1.0 / 2.0) * mu *
                     (POW2(F[0]) + POW2(F[1]) + POW2(F[2]) + POW2(F[3]) + POW2(F[4]) + POW2(F[5]) + POW2(F[6]) + POW2(F[7]) +
                      POW2(F[8]) - 3));
}

static SFEM_INLINE void hex8_neohookean_ogden_objective_integral(const scalar_t *const SFEM_RESTRICT lx,
                                                                 const scalar_t *const SFEM_RESTRICT ly,
                                                                 const scalar_t *const SFEM_RESTRICT lz,
                                                                 const int                           nqp,
                                                                 const scalar_t *const SFEM_RESTRICT qx,
                                                                 const scalar_t *const SFEM_RESTRICT qw,
                                                                 const scalar_t                      mu,
                                                                 const scalar_t                      lmbda,
                                                                 const scalar_t *const SFEM_RESTRICT dispx,
                                                                 const scalar_t *const SFEM_RESTRICT dispy,
                                                                 const scalar_t *const SFEM_RESTRICT dispz,
                                                                 scalar_t *const SFEM_RESTRICT       v) {
    scalar_t jacobian_adjugate[9];
    scalar_t jacobian_determinant;

    for (int kz = 0; kz < nqp; kz++) {
        for (int ky = 0; ky < nqp; ky++) {
            for (int kx = 0; kx < nqp; kx++) {
                hex8_adjugate_and_det(lx, ly, lz, qx[kx], qx[ky], qx[kz], jacobian_adjugate, &jacobian_determinant);
                hex8_neohookean_ogden_objective_at_qp(jacobian_adjugate,
                                                      jacobian_determinant,
                                                      qx[kx],
                                                      qx[ky],
                                                      qx[kz],
                                                      qw[kx] * qw[ky] * qw[kz],
                                                      mu,
                                                      lmbda,
                                                      dispx,
                                                      dispy,
                                                      dispz,
                                                      v);
            }
        }
    }
}

static SFEM_INLINE void hex8_neohookean_ogden_objective_steps_integral(const scalar_t *const SFEM_RESTRICT lx,
                                                                       const scalar_t *const SFEM_RESTRICT ly,
                                                                       const scalar_t *const SFEM_RESTRICT lz,
                                                                       const int                           nqp,
                                                                       const scalar_t *const SFEM_RESTRICT qx,
                                                                       const scalar_t *const SFEM_RESTRICT qw,
                                                                       const scalar_t                      mu,
                                                                       const scalar_t                      lmbda,
                                                                       const scalar_t *const SFEM_RESTRICT dispx,
                                                                       const scalar_t *const SFEM_RESTRICT dispy,
                                                                       const scalar_t *const SFEM_RESTRICT dispz,
                                                                       const scalar_t *const SFEM_RESTRICT incx,
                                                                       const scalar_t *const SFEM_RESTRICT incy,
                                                                       const scalar_t *const SFEM_RESTRICT incz,
                                                                       const int                           nsteps,
                                                                       const scalar_t *const SFEM_RESTRICT steps,
                                                                       scalar_t *const SFEM_RESTRICT       v) {
    scalar_t ux[8];
    scalar_t uy[8];
    scalar_t uz[8];

    for (int i = 0; i < nsteps; i++) {
        for (int j = 0; j < 8; j++) {
            ux[j] = dispx[j] + incx[j] * steps[i];
            uy[j] = dispy[j] + incy[j] * steps[i];
            uz[j] = dispz[j] + incz[j] * steps[i];
        }

        hex8_neohookean_ogden_objective_integral(lx, ly, lz, nqp, qx, qw, mu, lmbda, ux, uy, uz, &v[i]);
    }
}

static SFEM_INLINE void hex8_neohookean_grad(const scalar_t *const SFEM_RESTRICT adjugate,
                                             const scalar_t                      jacobian_determinant,
                                             const scalar_t                      qx,
                                             const scalar_t                      qy,
                                             const scalar_t                      qz,
                                             const scalar_t                      qw,
                                             const scalar_t                      mu,
                                             const scalar_t                      lmbda,
                                             const scalar_t *const SFEM_RESTRICT dispx,
                                             const scalar_t *const SFEM_RESTRICT dispy,
                                             const scalar_t *const SFEM_RESTRICT dispz,
                                             scalar_t *const SFEM_RESTRICT       gx,
                                             scalar_t *const SFEM_RESTRICT       gy,
                                             scalar_t *const SFEM_RESTRICT       gz) {
    scalar_t F[9];
    {
        // mundane ops: 267 divs: 1 sqrts: 0
        // total ops: 275
        const scalar_t x0 = 1.0 / jacobian_determinant;
        const scalar_t x1 = qy * qz;
        const scalar_t x2 = 1 - qz;
        const scalar_t x3 = qy * x2;
        const scalar_t x4 = 1 - qy;
        const scalar_t x5 = qz * x4;
        const scalar_t x6 = x2 * x4;
        const scalar_t x7 = dispx[0] * x6 - dispx[1] * x6 - dispx[2] * x3 + dispx[3] * x3 + dispx[4] * x5 - dispx[5] * x5 -
                            dispx[6] * x1 + dispx[7] * x1;
        const scalar_t x8  = qx * qz;
        const scalar_t x9  = qx * x2;
        const scalar_t x10 = 1 - qx;
        const scalar_t x11 = qz * x10;
        const scalar_t x12 = x10 * x2;
        const scalar_t x13 = dispx[0] * x12 + dispx[1] * x9 - dispx[2] * x9 - dispx[3] * x12 + dispx[4] * x11 + dispx[5] * x8 -
                             dispx[6] * x8 - dispx[7] * x11;
        const scalar_t x14 = qx * qy;
        const scalar_t x15 = qx * x4;
        const scalar_t x16 = qy * x10;
        const scalar_t x17 = x10 * x4;
        const scalar_t x18 = dispx[0] * x17 + dispx[1] * x15 + dispx[2] * x14 + dispx[3] * x16 - dispx[4] * x17 - dispx[5] * x15 -
                             dispx[6] * x14 - dispx[7] * x16;
        const scalar_t x19 = dispy[0] * x6 - dispy[1] * x6 - dispy[2] * x3 + dispy[3] * x3 + dispy[4] * x5 - dispy[5] * x5 -
                             dispy[6] * x1 + dispy[7] * x1;
        const scalar_t x20 = dispy[0] * x12 + dispy[1] * x9 - dispy[2] * x9 - dispy[3] * x12 + dispy[4] * x11 + dispy[5] * x8 -
                             dispy[6] * x8 - dispy[7] * x11;
        const scalar_t x21 = dispy[0] * x17 + dispy[1] * x15 + dispy[2] * x14 + dispy[3] * x16 - dispy[4] * x17 - dispy[5] * x15 -
                             dispy[6] * x14 - dispy[7] * x16;
        const scalar_t x22 = dispz[0] * x6 - dispz[1] * x6 - dispz[2] * x3 + dispz[3] * x3 + dispz[4] * x5 - dispz[5] * x5 -
                             dispz[6] * x1 + dispz[7] * x1;
        const scalar_t x23 = dispz[0] * x12 + dispz[1] * x9 - dispz[2] * x9 - dispz[3] * x12 + dispz[4] * x11 + dispz[5] * x8 -
                             dispz[6] * x8 - dispz[7] * x11;
        const scalar_t x24 = dispz[0] * x17 + dispz[1] * x15 + dispz[2] * x14 + dispz[3] * x16 - dispz[4] * x17 - dispz[5] * x15 -
                             dispz[6] * x14 - dispz[7] * x16;
        F[0] = -adjugate[0] * x0 * x7 - adjugate[3] * x0 * x13 - adjugate[6] * x0 * x18 + 1;
        F[1] = -x0 * (adjugate[1] * x7 + adjugate[4] * x13 + adjugate[7] * x18);
        F[2] = -x0 * (adjugate[2] * x7 + adjugate[5] * x13 + adjugate[8] * x18);
        F[3] = -x0 * (adjugate[0] * x19 + adjugate[3] * x20 + adjugate[6] * x21);
        F[4] = -adjugate[1] * x0 * x19 - adjugate[4] * x0 * x20 - adjugate[7] * x0 * x21 + 1;
        F[5] = -x0 * (adjugate[2] * x19 + adjugate[5] * x20 + adjugate[8] * x21);
        F[6] = -x0 * (adjugate[0] * x22 + adjugate[3] * x23 + adjugate[6] * x24);
        F[7] = -x0 * (adjugate[1] * x22 + adjugate[4] * x23 + adjugate[7] * x24);
        F[8] = -adjugate[2] * x0 * x22 - adjugate[5] * x0 * x23 - adjugate[8] * x0 * x24 + 1;
    }

    // mundane ops: 293 divs: 1 sqrts: 0
    // total ops: 301
    const scalar_t x0  = qx - 1;
    const scalar_t x1  = qy - 1;
    const scalar_t x2  = F[4] * F[8];
    const scalar_t x3  = F[5] * F[7];
    const scalar_t x4  = x2 - x3;
    const scalar_t x5  = F[5] * F[6];
    const scalar_t x6  = F[3] * F[7];
    const scalar_t x7  = F[3] * F[8];
    const scalar_t x8  = F[4] * F[6];
    const scalar_t x9  = F[0] * x2 - F[0] * x3 + F[1] * x5 - F[1] * x7 + F[2] * x6 - F[2] * x8;
    const scalar_t x10 = 1.0 / x9;
    const scalar_t x11 = mu * x10;
    const scalar_t x12 = lmbda * x10 * log(x9);
    const scalar_t x13 = F[0] * mu - x11 * x4 + x12 * x4;
    const scalar_t x14 = -x5 + x7;
    const scalar_t x15 = F[1] * mu + x11 * x14 - x12 * x14;
    const scalar_t x16 = x6 - x8;
    const scalar_t x17 = F[2] * mu - x11 * x16 + x12 * x16;
    const scalar_t x18 = adjugate[6] * x13 + adjugate[7] * x15 + adjugate[8] * x17;
    const scalar_t x19 = x1 * x18;
    const scalar_t x20 = x0 * x19;
    const scalar_t x21 = qz - 1;
    const scalar_t x22 = adjugate[3] * x13 + adjugate[4] * x15 + adjugate[5] * x17;
    const scalar_t x23 = x21 * x22;
    const scalar_t x24 = x0 * x23;
    const scalar_t x25 = adjugate[0] * x13 + adjugate[1] * x15 + adjugate[2] * x17;
    const scalar_t x26 = x21 * x25;
    const scalar_t x27 = x1 * x26;
    const scalar_t x28 = qx * x19;
    const scalar_t x29 = qx * x23;
    const scalar_t x30 = qy * x18;
    const scalar_t x31 = qx * x30;
    const scalar_t x32 = qy * x26;
    const scalar_t x33 = x0 * x30;
    const scalar_t x34 = qz * x22;
    const scalar_t x35 = x0 * x34;
    const scalar_t x36 = qz * x25;
    const scalar_t x37 = x1 * x36;
    const scalar_t x38 = qx * x34;
    const scalar_t x39 = qy * x36;
    const scalar_t x40 = F[1] * F[8] - F[2] * F[7];
    const scalar_t x41 = F[3] * mu + x11 * x40 - x12 * x40;
    const scalar_t x42 = F[0] * F[8] - F[2] * F[6];
    const scalar_t x43 = F[4] * mu - x11 * x42 + x12 * x42;
    const scalar_t x44 = F[0] * F[7] - F[1] * F[6];
    const scalar_t x45 = F[5] * mu + x11 * x44 - x12 * x44;
    const scalar_t x46 = adjugate[6] * x41 + adjugate[7] * x43 + adjugate[8] * x45;
    const scalar_t x47 = x1 * x46;
    const scalar_t x48 = x0 * x47;
    const scalar_t x49 = adjugate[3] * x41 + adjugate[4] * x43 + adjugate[5] * x45;
    const scalar_t x50 = x21 * x49;
    const scalar_t x51 = x0 * x50;
    const scalar_t x52 = adjugate[0] * x41 + adjugate[1] * x43 + adjugate[2] * x45;
    const scalar_t x53 = x21 * x52;
    const scalar_t x54 = x1 * x53;
    const scalar_t x55 = qx * x47;
    const scalar_t x56 = qx * x50;
    const scalar_t x57 = qy * x46;
    const scalar_t x58 = qx * x57;
    const scalar_t x59 = qy * x53;
    const scalar_t x60 = x0 * x57;
    const scalar_t x61 = qz * x49;
    const scalar_t x62 = x0 * x61;
    const scalar_t x63 = qz * x52;
    const scalar_t x64 = x1 * x63;
    const scalar_t x65 = qx * x61;
    const scalar_t x66 = qy * x63;
    const scalar_t x67 = F[1] * F[5] - F[2] * F[4];
    const scalar_t x68 = F[6] * mu - x11 * x67 + x12 * x67;
    const scalar_t x69 = F[0] * F[5] - F[2] * F[3];
    const scalar_t x70 = F[7] * mu + x11 * x69 - x12 * x69;
    const scalar_t x71 = F[0] * F[4] - F[1] * F[3];
    const scalar_t x72 = F[8] * mu - x11 * x71 + x12 * x71;
    const scalar_t x73 = adjugate[6] * x68 + adjugate[7] * x70 + adjugate[8] * x72;
    const scalar_t x74 = x1 * x73;
    const scalar_t x75 = x0 * x74;
    const scalar_t x76 = adjugate[3] * x68 + adjugate[4] * x70 + adjugate[5] * x72;
    const scalar_t x77 = x21 * x76;
    const scalar_t x78 = x0 * x77;
    const scalar_t x79 = adjugate[0] * x68 + adjugate[1] * x70 + adjugate[2] * x72;
    const scalar_t x80 = x21 * x79;
    const scalar_t x81 = x1 * x80;
    const scalar_t x82 = qx * x74;
    const scalar_t x83 = qx * x77;
    const scalar_t x84 = qy * x73;
    const scalar_t x85 = qx * x84;
    const scalar_t x86 = qy * x80;
    const scalar_t x87 = x0 * x84;
    const scalar_t x88 = qz * x76;
    const scalar_t x89 = x0 * x88;
    const scalar_t x90 = qz * x79;
    const scalar_t x91 = x1 * x90;
    const scalar_t x92 = qx * x88;
    const scalar_t x93 = qy * x90;
    gx[0] += -qw * (x20 + x24 + x27);
    gx[1] += qw * (x27 + x28 + x29);
    gx[2] += -qw * (x29 + x31 + x32);
    gx[3] += qw * (x24 + x32 + x33);
    gx[4] += qw * (x20 + x35 + x37);
    gx[5] += -qw * (x28 + x37 + x38);
    gx[6] += qw * (x31 + x38 + x39);
    gx[7] += -qw * (x33 + x35 + x39);
    gy[0] += -qw * (x48 + x51 + x54);
    gy[1] += qw * (x54 + x55 + x56);
    gy[2] += -qw * (x56 + x58 + x59);
    gy[3] += qw * (x51 + x59 + x60);
    gy[4] += qw * (x48 + x62 + x64);
    gy[5] += -qw * (x55 + x64 + x65);
    gy[6] += qw * (x58 + x65 + x66);
    gy[7] += -qw * (x60 + x62 + x66);
    gz[0] += -qw * (x75 + x78 + x81);
    gz[1] += qw * (x81 + x82 + x83);
    gz[2] += -qw * (x83 + x85 + x86);
    gz[3] += qw * (x78 + x86 + x87);
    gz[4] += qw * (x75 + x89 + x91);
    gz[5] += -qw * (x82 + x91 + x92);
    gz[6] += qw * (x85 + x92 + x93);
    gz[7] += -qw * (x87 + x89 + x93);
}

static SFEM_INLINE void hex8_neohookean_ogden_hessian_diag(const scalar_t *const SFEM_RESTRICT adjugate,
                                                           const scalar_t                      jacobian_determinant,
                                                           const scalar_t                      qx,
                                                           const scalar_t                      qy,
                                                           const scalar_t                      qz,
                                                           const scalar_t                      qw,
                                                           const scalar_t                      mu,
                                                           const scalar_t                      lmbda,
                                                           const scalar_t *const SFEM_RESTRICT dispx,
                                                           const scalar_t *const SFEM_RESTRICT dispy,
                                                           const scalar_t *const SFEM_RESTRICT dispz,
                                                           scalar_t *const SFEM_RESTRICT       H_diag) {
    scalar_t F[9];
    {
        // mundane ops: 267 divs: 1 sqrts: 0
        // total ops: 275
        const scalar_t x0 = 1.0 / jacobian_determinant;
        const scalar_t x1 = qy * qz;
        const scalar_t x2 = 1 - qz;
        const scalar_t x3 = qy * x2;
        const scalar_t x4 = 1 - qy;
        const scalar_t x5 = qz * x4;
        const scalar_t x6 = x2 * x4;
        const scalar_t x7 = dispx[0] * x6 - dispx[1] * x6 - dispx[2] * x3 + dispx[3] * x3 + dispx[4] * x5 - dispx[5] * x5 -
                            dispx[6] * x1 + dispx[7] * x1;
        const scalar_t x8  = qx * qz;
        const scalar_t x9  = qx * x2;
        const scalar_t x10 = 1 - qx;
        const scalar_t x11 = qz * x10;
        const scalar_t x12 = x10 * x2;
        const scalar_t x13 = dispx[0] * x12 + dispx[1] * x9 - dispx[2] * x9 - dispx[3] * x12 + dispx[4] * x11 + dispx[5] * x8 -
                             dispx[6] * x8 - dispx[7] * x11;
        const scalar_t x14 = qx * qy;
        const scalar_t x15 = qx * x4;
        const scalar_t x16 = qy * x10;
        const scalar_t x17 = x10 * x4;
        const scalar_t x18 = dispx[0] * x17 + dispx[1] * x15 + dispx[2] * x14 + dispx[3] * x16 - dispx[4] * x17 - dispx[5] * x15 -
                             dispx[6] * x14 - dispx[7] * x16;
        const scalar_t x19 = dispy[0] * x6 - dispy[1] * x6 - dispy[2] * x3 + dispy[3] * x3 + dispy[4] * x5 - dispy[5] * x5 -
                             dispy[6] * x1 + dispy[7] * x1;
        const scalar_t x20 = dispy[0] * x12 + dispy[1] * x9 - dispy[2] * x9 - dispy[3] * x12 + dispy[4] * x11 + dispy[5] * x8 -
                             dispy[6] * x8 - dispy[7] * x11;
        const scalar_t x21 = dispy[0] * x17 + dispy[1] * x15 + dispy[2] * x14 + dispy[3] * x16 - dispy[4] * x17 - dispy[5] * x15 -
                             dispy[6] * x14 - dispy[7] * x16;
        const scalar_t x22 = dispz[0] * x6 - dispz[1] * x6 - dispz[2] * x3 + dispz[3] * x3 + dispz[4] * x5 - dispz[5] * x5 -
                             dispz[6] * x1 + dispz[7] * x1;
        const scalar_t x23 = dispz[0] * x12 + dispz[1] * x9 - dispz[2] * x9 - dispz[3] * x12 + dispz[4] * x11 + dispz[5] * x8 -
                             dispz[6] * x8 - dispz[7] * x11;
        const scalar_t x24 = dispz[0] * x17 + dispz[1] * x15 + dispz[2] * x14 + dispz[3] * x16 - dispz[4] * x17 - dispz[5] * x15 -
                             dispz[6] * x14 - dispz[7] * x16;
        F[0] = -adjugate[0] * x0 * x7 - adjugate[3] * x0 * x13 - adjugate[6] * x0 * x18 + 1;
        F[1] = -x0 * (adjugate[1] * x7 + adjugate[4] * x13 + adjugate[7] * x18);
        F[2] = -x0 * (adjugate[2] * x7 + adjugate[5] * x13 + adjugate[8] * x18);
        F[3] = -x0 * (adjugate[0] * x19 + adjugate[3] * x20 + adjugate[6] * x21);
        F[4] = -adjugate[1] * x0 * x19 - adjugate[4] * x0 * x20 - adjugate[7] * x0 * x21 + 1;
        F[5] = -x0 * (adjugate[2] * x19 + adjugate[5] * x20 + adjugate[8] * x21);
        F[6] = -x0 * (adjugate[0] * x22 + adjugate[3] * x23 + adjugate[6] * x24);
        F[7] = -x0 * (adjugate[1] * x22 + adjugate[4] * x23 + adjugate[7] * x24);
        F[8] = -adjugate[2] * x0 * x22 - adjugate[5] * x0 * x23 - adjugate[8] * x0 * x24 + 1;
    }

    // mundane ops: 939 divs: 1 sqrts: 0
    // total ops: 947
    const scalar_t x0   = qx - 1;
    const scalar_t x1   = POW2(x0);
    const scalar_t x2   = qy - 1;
    const scalar_t x3   = POW2(x2);
    const scalar_t x4   = F[4] * F[8];
    const scalar_t x5   = F[5] * F[7];
    const scalar_t x6   = x4 - x5;
    const scalar_t x7   = F[5] * F[6];
    const scalar_t x8   = F[3] * F[7];
    const scalar_t x9   = F[3] * F[8];
    const scalar_t x10  = F[4] * F[6];
    const scalar_t x11  = F[0] * x4 - F[0] * x5 + F[1] * x7 - F[1] * x9 - F[2] * x10 + F[2] * x8;
    const scalar_t x12  = (1 / POW2(x11));
    const scalar_t x13  = lmbda * log(x11);
    const scalar_t x14  = x12 * (lmbda + mu - x13);
    const scalar_t x15  = x14 * x6;
    const scalar_t x16  = -x10 + x8;
    const scalar_t x17  = adjugate[8] * x16;
    const scalar_t x18  = -x7 + x9;
    const scalar_t x19  = x15 * x18;
    const scalar_t x20  = x12 * POW2(x6);
    const scalar_t x21  = lmbda * x20 + mu * x20 + mu - x13 * x20;
    const scalar_t x22  = adjugate[6] * x21 - adjugate[7] * x19 + x15 * x17;
    const scalar_t x23  = x15 * x16;
    const scalar_t x24  = x14 * x18;
    const scalar_t x25  = x16 * x24;
    const scalar_t x26  = x12 * POW2(x16);
    const scalar_t x27  = lmbda * x26 + mu * x26 + mu - x13 * x26;
    const scalar_t x28  = adjugate[6] * x23 - adjugate[7] * x25 + adjugate[8] * x27;
    const scalar_t x29  = x12 * POW2(x18);
    const scalar_t x30  = lmbda * x29 + mu * x29 + mu - x13 * x29;
    const scalar_t x31  = -adjugate[6] * x19 + adjugate[7] * x30 - x17 * x24;
    const scalar_t x32  = adjugate[6] * x22 + adjugate[7] * x31 + adjugate[8] * x28;
    const scalar_t x33  = x3 * x32;
    const scalar_t x34  = x1 * x33;
    const scalar_t x35  = qz - 1;
    const scalar_t x36  = POW2(x35);
    const scalar_t x37  = adjugate[3] * x21 - adjugate[4] * x19 + adjugate[5] * x23;
    const scalar_t x38  = adjugate[3] * x23 - adjugate[4] * x25 + adjugate[5] * x27;
    const scalar_t x39  = -adjugate[3] * x19 + adjugate[4] * x30 - adjugate[5] * x25;
    const scalar_t x40  = adjugate[3] * x37 + adjugate[4] * x39 + adjugate[5] * x38;
    const scalar_t x41  = x36 * x40;
    const scalar_t x42  = x1 * x41;
    const scalar_t x43  = adjugate[0] * x21 - adjugate[1] * x19 + adjugate[2] * x23;
    const scalar_t x44  = adjugate[0] * x23 - adjugate[1] * x25 + adjugate[2] * x27;
    const scalar_t x45  = -adjugate[0] * x19 + adjugate[1] * x30 - adjugate[2] * x25;
    const scalar_t x46  = adjugate[0] * x43 + adjugate[1] * x45 + adjugate[2] * x44;
    const scalar_t x47  = x36 * x46;
    const scalar_t x48  = x3 * x47;
    const scalar_t x49  = x2 * x36;
    const scalar_t x50  = adjugate[0] * x37 + adjugate[1] * x39 + adjugate[2] * x38;
    const scalar_t x51  = x0 * x50;
    const scalar_t x52  = adjugate[3] * x43 + adjugate[4] * x45 + adjugate[5] * x44;
    const scalar_t x53  = x49 * x52;
    const scalar_t x54  = x3 * x35;
    const scalar_t x55  = adjugate[0] * x22 + adjugate[1] * x31 + adjugate[2] * x28;
    const scalar_t x56  = x0 * x55;
    const scalar_t x57  = adjugate[6] * x43 + adjugate[7] * x45 + adjugate[8] * x44;
    const scalar_t x58  = x54 * x57;
    const scalar_t x59  = x2 * x35;
    const scalar_t x60  = adjugate[3] * x22 + adjugate[4] * x31 + adjugate[5] * x28;
    const scalar_t x61  = x1 * x60;
    const scalar_t x62  = adjugate[6] * x37 + adjugate[7] * x39 + adjugate[8] * x38;
    const scalar_t x63  = x59 * x62;
    const scalar_t x64  = qw / jacobian_determinant;
    const scalar_t x65  = POW2(qx);
    const scalar_t x66  = x33 * x65;
    const scalar_t x67  = x41 * x65;
    const scalar_t x68  = qx * x50;
    const scalar_t x69  = qx * x55;
    const scalar_t x70  = x60 * x65;
    const scalar_t x71  = POW2(qy);
    const scalar_t x72  = x32 * x71;
    const scalar_t x73  = x65 * x72;
    const scalar_t x74  = qy * x36;
    const scalar_t x75  = x52 * x74;
    const scalar_t x76  = x35 * x71;
    const scalar_t x77  = x57 * x76;
    const scalar_t x78  = qy * x35;
    const scalar_t x79  = x62 * x78;
    const scalar_t x80  = x47 * x71;
    const scalar_t x81  = x1 * x72;
    const scalar_t x82  = POW2(qz);
    const scalar_t x83  = x40 * x82;
    const scalar_t x84  = x1 * x83;
    const scalar_t x85  = x46 * x82;
    const scalar_t x86  = x3 * x85;
    const scalar_t x87  = qz * x3;
    const scalar_t x88  = x57 * x87;
    const scalar_t x89  = qz * x2;
    const scalar_t x90  = x62 * x89;
    const scalar_t x91  = x2 * x82;
    const scalar_t x92  = x52 * x91;
    const scalar_t x93  = x65 * x83;
    const scalar_t x94  = qy * x82;
    const scalar_t x95  = x52 * x94;
    const scalar_t x96  = qz * x71;
    const scalar_t x97  = x57 * x96;
    const scalar_t x98  = qy * qz;
    const scalar_t x99  = x62 * x98;
    const scalar_t x100 = x71 * x85;
    const scalar_t x101 = F[1] * F[8] - F[2] * F[7];
    const scalar_t x102 = x101 * x14;
    const scalar_t x103 = F[0] * F[7] - F[1] * F[6];
    const scalar_t x104 = adjugate[8] * x103;
    const scalar_t x105 = F[0] * F[8] - F[2] * F[6];
    const scalar_t x106 = x102 * x105;
    const scalar_t x107 = POW2(x101) * x12;
    const scalar_t x108 = lmbda * x107 + mu * x107 + mu - x107 * x13;
    const scalar_t x109 = adjugate[6] * x108 - adjugate[7] * x106 + x102 * x104;
    const scalar_t x110 = x102 * x103;
    const scalar_t x111 = x105 * x14;
    const scalar_t x112 = x103 * x111;
    const scalar_t x113 = POW2(x103) * x12;
    const scalar_t x114 = lmbda * x113 + mu * x113 + mu - x113 * x13;
    const scalar_t x115 = adjugate[6] * x110 - adjugate[7] * x112 + adjugate[8] * x114;
    const scalar_t x116 = POW2(x105) * x12;
    const scalar_t x117 = lmbda * x116 + mu * x116 + mu - x116 * x13;
    const scalar_t x118 = -adjugate[6] * x106 + adjugate[7] * x117 - x104 * x111;
    const scalar_t x119 = adjugate[6] * x109 + adjugate[7] * x118 + adjugate[8] * x115;
    const scalar_t x120 = x119 * x3;
    const scalar_t x121 = x1 * x120;
    const scalar_t x122 = adjugate[3] * x108 - adjugate[4] * x106 + adjugate[5] * x110;
    const scalar_t x123 = adjugate[3] * x110 - adjugate[4] * x112 + adjugate[5] * x114;
    const scalar_t x124 = -adjugate[3] * x106 + adjugate[4] * x117 - adjugate[5] * x112;
    const scalar_t x125 = adjugate[3] * x122 + adjugate[4] * x124 + adjugate[5] * x123;
    const scalar_t x126 = x125 * x36;
    const scalar_t x127 = x1 * x126;
    const scalar_t x128 = adjugate[0] * x108 - adjugate[1] * x106 + adjugate[2] * x110;
    const scalar_t x129 = adjugate[0] * x110 - adjugate[1] * x112 + adjugate[2] * x114;
    const scalar_t x130 = -adjugate[0] * x106 + adjugate[1] * x117 - adjugate[2] * x112;
    const scalar_t x131 = adjugate[0] * x128 + adjugate[1] * x130 + adjugate[2] * x129;
    const scalar_t x132 = x131 * x36;
    const scalar_t x133 = x132 * x3;
    const scalar_t x134 = adjugate[0] * x122 + adjugate[1] * x124 + adjugate[2] * x123;
    const scalar_t x135 = x0 * x49;
    const scalar_t x136 = adjugate[3] * x128 + adjugate[4] * x130 + adjugate[5] * x129;
    const scalar_t x137 = adjugate[0] * x109 + adjugate[1] * x118 + adjugate[2] * x115;
    const scalar_t x138 = x0 * x54;
    const scalar_t x139 = adjugate[6] * x128 + adjugate[7] * x130 + adjugate[8] * x129;
    const scalar_t x140 = adjugate[3] * x109 + adjugate[4] * x118 + adjugate[5] * x115;
    const scalar_t x141 = x1 * x59;
    const scalar_t x142 = adjugate[6] * x122 + adjugate[7] * x124 + adjugate[8] * x123;
    const scalar_t x143 = x120 * x65;
    const scalar_t x144 = x126 * x65;
    const scalar_t x145 = qx * x49;
    const scalar_t x146 = qx * x54;
    const scalar_t x147 = x59 * x65;
    const scalar_t x148 = x119 * x71;
    const scalar_t x149 = x148 * x65;
    const scalar_t x150 = qx * x74;
    const scalar_t x151 = qx * x76;
    const scalar_t x152 = x65 * x78;
    const scalar_t x153 = x132 * x71;
    const scalar_t x154 = x1 * x148;
    const scalar_t x155 = x0 * x74;
    const scalar_t x156 = x1 * x78;
    const scalar_t x157 = x0 * x76;
    const scalar_t x158 = x125 * x82;
    const scalar_t x159 = x1 * x158;
    const scalar_t x160 = x131 * x82;
    const scalar_t x161 = x160 * x3;
    const scalar_t x162 = x0 * x87;
    const scalar_t x163 = x1 * x89;
    const scalar_t x164 = x0 * x91;
    const scalar_t x165 = x158 * x65;
    const scalar_t x166 = qx * x87;
    const scalar_t x167 = qx * x91;
    const scalar_t x168 = x65 * x89;
    const scalar_t x169 = qx * x94;
    const scalar_t x170 = qx * x96;
    const scalar_t x171 = x65 * x98;
    const scalar_t x172 = x160 * x71;
    const scalar_t x173 = x1 * x98;
    const scalar_t x174 = x0 * x94;
    const scalar_t x175 = x0 * x96;
    const scalar_t x176 = F[1] * F[5] - F[2] * F[4];
    const scalar_t x177 = x14 * x176;
    const scalar_t x178 = F[0] * F[4] - F[1] * F[3];
    const scalar_t x179 = adjugate[8] * x178;
    const scalar_t x180 = F[0] * F[5] - F[2] * F[3];
    const scalar_t x181 = x177 * x180;
    const scalar_t x182 = x12 * POW2(x176);
    const scalar_t x183 = lmbda * x182 + mu * x182 + mu - x13 * x182;
    const scalar_t x184 = adjugate[6] * x183 - adjugate[7] * x181 + x177 * x179;
    const scalar_t x185 = x177 * x178;
    const scalar_t x186 = x14 * x180;
    const scalar_t x187 = x178 * x186;
    const scalar_t x188 = x12 * POW2(x178);
    const scalar_t x189 = lmbda * x188 + mu * x188 + mu - x13 * x188;
    const scalar_t x190 = adjugate[6] * x185 - adjugate[7] * x187 + adjugate[8] * x189;
    const scalar_t x191 = x12 * POW2(x180);
    const scalar_t x192 = lmbda * x191 + mu * x191 + mu - x13 * x191;
    const scalar_t x193 = -adjugate[6] * x181 + adjugate[7] * x192 - x179 * x186;
    const scalar_t x194 = adjugate[6] * x184 + adjugate[7] * x193 + adjugate[8] * x190;
    const scalar_t x195 = x194 * x3;
    const scalar_t x196 = x1 * x195;
    const scalar_t x197 = adjugate[3] * x183 - adjugate[4] * x181 + adjugate[5] * x185;
    const scalar_t x198 = adjugate[3] * x185 - adjugate[4] * x187 + adjugate[5] * x189;
    const scalar_t x199 = -adjugate[3] * x181 + adjugate[4] * x192 - adjugate[5] * x187;
    const scalar_t x200 = adjugate[3] * x197 + adjugate[4] * x199 + adjugate[5] * x198;
    const scalar_t x201 = x200 * x36;
    const scalar_t x202 = x1 * x201;
    const scalar_t x203 = adjugate[0] * x183 - adjugate[1] * x181 + adjugate[2] * x185;
    const scalar_t x204 = adjugate[0] * x185 - adjugate[1] * x187 + adjugate[2] * x189;
    const scalar_t x205 = -adjugate[0] * x181 + adjugate[1] * x192 - adjugate[2] * x187;
    const scalar_t x206 = adjugate[0] * x203 + adjugate[1] * x205 + adjugate[2] * x204;
    const scalar_t x207 = x206 * x36;
    const scalar_t x208 = x207 * x3;
    const scalar_t x209 = adjugate[0] * x197 + adjugate[1] * x199 + adjugate[2] * x198;
    const scalar_t x210 = adjugate[3] * x203 + adjugate[4] * x205 + adjugate[5] * x204;
    const scalar_t x211 = adjugate[0] * x184 + adjugate[1] * x193 + adjugate[2] * x190;
    const scalar_t x212 = adjugate[6] * x203 + adjugate[7] * x205 + adjugate[8] * x204;
    const scalar_t x213 = adjugate[3] * x184 + adjugate[4] * x193 + adjugate[5] * x190;
    const scalar_t x214 = adjugate[6] * x197 + adjugate[7] * x199 + adjugate[8] * x198;
    const scalar_t x215 = x195 * x65;
    const scalar_t x216 = x201 * x65;
    const scalar_t x217 = x194 * x71;
    const scalar_t x218 = x217 * x65;
    const scalar_t x219 = x207 * x71;
    const scalar_t x220 = x1 * x217;
    const scalar_t x221 = x200 * x82;
    const scalar_t x222 = x1 * x221;
    const scalar_t x223 = x206 * x82;
    const scalar_t x224 = x223 * x3;
    const scalar_t x225 = x221 * x65;
    const scalar_t x226 = x223 * x71;
    H_diag[0] += x64 * (x0 * x53 + x0 * x58 + x1 * x63 + x34 + x42 + x48 + x49 * x51 + x54 * x56 + x59 * x61);
    H_diag[1] += x64 * (qx * x53 + qx * x58 + x48 + x49 * x68 + x54 * x69 + x59 * x70 + x63 * x65 + x66 + x67);
    H_diag[2] += x64 * (qx * x75 + qx * x77 + x65 * x79 + x67 + x68 * x74 + x69 * x76 + x70 * x78 + x73 + x80);
    H_diag[3] += x64 * (x0 * x75 + x0 * x77 + x1 * x79 + x42 + x51 * x74 + x56 * x76 + x61 * x78 + x80 + x81);
    H_diag[4] += x64 * (x0 * x88 + x0 * x92 + x1 * x90 + x34 + x51 * x91 + x56 * x87 + x61 * x89 + x84 + x86);
    H_diag[5] += x64 * (qx * x88 + qx * x92 + x65 * x90 + x66 + x68 * x91 + x69 * x87 + x70 * x89 + x86 + x93);
    H_diag[6] += x64 * (qx * x95 + qx * x97 + x100 + x65 * x99 + x68 * x94 + x69 * x96 + x70 * x98 + x73 + x93);
    H_diag[7] += x64 * (x0 * x95 + x0 * x97 + x1 * x99 + x100 + x51 * x94 + x56 * x96 + x61 * x98 + x81 + x84);
    H_diag[8] += x64 * (x121 + x127 + x133 + x134 * x135 + x135 * x136 + x137 * x138 + x138 * x139 + x140 * x141 + x141 * x142);
    H_diag[9] += x64 * (x133 + x134 * x145 + x136 * x145 + x137 * x146 + x139 * x146 + x140 * x147 + x142 * x147 + x143 + x144);
    H_diag[10] += x64 * (x134 * x150 + x136 * x150 + x137 * x151 + x139 * x151 + x140 * x152 + x142 * x152 + x144 + x149 + x153);
    H_diag[11] += x64 * (x127 + x134 * x155 + x136 * x155 + x137 * x157 + x139 * x157 + x140 * x156 + x142 * x156 + x153 + x154);
    H_diag[12] += x64 * (x121 + x134 * x164 + x136 * x164 + x137 * x162 + x139 * x162 + x140 * x163 + x142 * x163 + x159 + x161);
    H_diag[13] += x64 * (x134 * x167 + x136 * x167 + x137 * x166 + x139 * x166 + x140 * x168 + x142 * x168 + x143 + x161 + x165);
    H_diag[14] += x64 * (x134 * x169 + x136 * x169 + x137 * x170 + x139 * x170 + x140 * x171 + x142 * x171 + x149 + x165 + x172);
    H_diag[15] += x64 * (x134 * x174 + x136 * x174 + x137 * x175 + x139 * x175 + x140 * x173 + x142 * x173 + x154 + x159 + x172);
    H_diag[16] += x64 * (x135 * x209 + x135 * x210 + x138 * x211 + x138 * x212 + x141 * x213 + x141 * x214 + x196 + x202 + x208);
    H_diag[17] += x64 * (x145 * x209 + x145 * x210 + x146 * x211 + x146 * x212 + x147 * x213 + x147 * x214 + x208 + x215 + x216);
    H_diag[18] += x64 * (x150 * x209 + x150 * x210 + x151 * x211 + x151 * x212 + x152 * x213 + x152 * x214 + x216 + x218 + x219);
    H_diag[19] += x64 * (x155 * x209 + x155 * x210 + x156 * x213 + x156 * x214 + x157 * x211 + x157 * x212 + x202 + x219 + x220);
    H_diag[20] += x64 * (x162 * x211 + x162 * x212 + x163 * x213 + x163 * x214 + x164 * x209 + x164 * x210 + x196 + x222 + x224);
    H_diag[21] += x64 * (x166 * x211 + x166 * x212 + x167 * x209 + x167 * x210 + x168 * x213 + x168 * x214 + x215 + x224 + x225);
    H_diag[22] += x64 * (x169 * x209 + x169 * x210 + x170 * x211 + x170 * x212 + x171 * x213 + x171 * x214 + x218 + x225 + x226);
    H_diag[23] += x64 * (x173 * x213 + x173 * x214 + x174 * x209 + x174 * x210 + x175 * x211 + x175 * x212 + x220 + x222 + x226);
}

#ifdef __cplusplus
}
#endif

#endif  // HEX8_NEOHOOKEAN_OGDEN_LOCAL_H
