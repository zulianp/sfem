#include "hex8_neohookean_ogden.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

#include <mpi.h>

#include "crs_graph.h"
#include "sfem_macros.h"
#include "sfem_vec.h"
#include "sortreduce.h"

#include "hex8_inline_cpu.h"
#include "line_quadrature.h"

#include "hex8_partial_assembly_neohookean_inline.h"

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

int hex8_neohookean_ogden_objective(const ptrdiff_t                   nelements,
                                    const ptrdiff_t                   stride,
                                    const ptrdiff_t                   nnodes,
                                    idx_t **const SFEM_RESTRICT       elements,
                                    geom_t **const SFEM_RESTRICT      points,
                                    const real_t                      mu,
                                    const real_t                      lambda,
                                    const ptrdiff_t                   u_stride,
                                    const real_t *const SFEM_RESTRICT ux,
                                    const real_t *const SFEM_RESTRICT uy,
                                    const real_t *const SFEM_RESTRICT uz,
                                    const int                         is_element_wise,
                                    real_t *const SFEM_RESTRICT       out) {
    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

    static const int       n_qp = line_q2_n;
    static const scalar_t *qx   = line_q2_x;
    static const scalar_t *qw   = line_q2_w;

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[8];

        scalar_t lx[8];
        scalar_t ly[8];
        scalar_t lz[8];

        scalar_t edispx[8];
        scalar_t edispy[8];
        scalar_t edispz[8];

        scalar_t jacobian_adjugate[9];
        scalar_t jacobian_determinant = 0;

        for (int v = 0; v < 8; ++v) {
            ev[v] = elements[v][i * stride];
        }

        for (int d = 0; d < 8; d++) {
            lx[d] = x[ev[d]];
            ly[d] = y[ev[d]];
            lz[d] = z[ev[d]];
        }

        for (int v = 0; v < 8; ++v) {
            const ptrdiff_t idx = ev[v] * u_stride;
            edispx[v]           = ux[idx];
            edispy[v]           = uy[idx];
            edispz[v]           = uz[idx];
        }

        scalar_t v = 0;
        hex8_neohookean_ogden_objective_integral(lx, ly, lz, n_qp, qx, qw, mu, lambda, edispx, edispy, edispz, &v);
        assert(v == v);

        if (is_element_wise) {
            out[i] = v;
        } else {
#pragma omp atomic update
            *out += v;
        }
    }

    if(*out != *out) {
        *out = 1e10;
    }

    return SFEM_SUCCESS;
}

int hex8_neohookean_ogden_objective_steps(const ptrdiff_t                   nelements,
                                          const ptrdiff_t                   stride,
                                          const ptrdiff_t                   nnodes,
                                          idx_t **const SFEM_RESTRICT       elements,
                                          geom_t **const SFEM_RESTRICT      points,
                                          const real_t                      mu,
                                          const real_t                      lambda,
                                          const ptrdiff_t                   u_stride,
                                          const real_t *const SFEM_RESTRICT ux,
                                          const real_t *const SFEM_RESTRICT uy,
                                          const real_t *const SFEM_RESTRICT uz,
                                          const ptrdiff_t                   inc_stride,
                                          const real_t *const SFEM_RESTRICT incx,
                                          const real_t *const SFEM_RESTRICT incy,
                                          const real_t *const SFEM_RESTRICT incz,
                                          const int                         nsteps,
                                          const real_t *const               steps,
                                          real_t *const SFEM_RESTRICT       out) {
    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

    static const int       n_qp = line_q2_n;
    static const scalar_t *qx   = line_q2_x;
    static const scalar_t *qw   = line_q2_w;

#pragma omp parallel
    {
        scalar_t *out_local = (scalar_t *)calloc(nsteps, sizeof(scalar_t));

#pragma omp for
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[8];

            scalar_t lx[8];
            scalar_t ly[8];
            scalar_t lz[8];

            scalar_t edispx[8];
            scalar_t edispy[8];
            scalar_t edispz[8];

            scalar_t eincx[8];
            scalar_t eincy[8];
            scalar_t eincz[8];

            for (int v = 0; v < 8; ++v) {
                ev[v] = elements[v][i * stride];
            }

            for (int d = 0; d < 8; d++) {
                lx[d] = x[ev[d]];
                ly[d] = y[ev[d]];
                lz[d] = z[ev[d]];
            }

            for (int v = 0; v < 8; ++v) {
                const ptrdiff_t idx = ev[v] * u_stride;
                edispx[v]           = ux[idx];
                edispy[v]           = uy[idx];
                edispz[v]           = uz[idx];
            }

            for (int v = 0; v < 8; ++v) {
                const ptrdiff_t idx = ev[v] * inc_stride;
                eincx[v]            = incx[idx];
                eincy[v]            = incy[idx];
                eincz[v]            = incz[idx];
            }

            hex8_neohookean_ogden_objective_steps_integral(
                    lx, ly, lz, n_qp, qx, qw, mu, lambda, edispx, edispy, edispz, eincx, eincy, eincz, nsteps, steps, out_local);
        }

        for (int s = 0; s < nsteps; s++) {
            if(out_local[s] != out_local[s]) {
                out_local[s] = 1e10;
            }
#pragma omp atomic update
            out[s] += out_local[s];
        }

        free(out_local);
    }

    return SFEM_SUCCESS;
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

int hex8_neohookean_ogden_gradient(const ptrdiff_t                   nelements,
                                   const ptrdiff_t                   stride,
                                   const ptrdiff_t                   nnodes,
                                   idx_t **const SFEM_RESTRICT       elements,
                                   geom_t **const SFEM_RESTRICT      points,
                                   const real_t                      mu,
                                   const real_t                      lambda,
                                   const ptrdiff_t                   u_stride,
                                   const real_t *const SFEM_RESTRICT ux,
                                   const real_t *const SFEM_RESTRICT uy,
                                   const real_t *const SFEM_RESTRICT uz,
                                   const ptrdiff_t                   out_stride,
                                   real_t *const SFEM_RESTRICT       outx,
                                   real_t *const SFEM_RESTRICT       outy,
                                   real_t *const SFEM_RESTRICT       outz) {
    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

    static const int       n_qp = line_q2_n;
    static const scalar_t *qx   = line_q2_x;
    static const scalar_t *qw   = line_q2_w;

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[8];

        scalar_t lx[8];
        scalar_t ly[8];
        scalar_t lz[8];

        scalar_t edispx[8];
        scalar_t edispy[8];
        scalar_t edispz[8];

        accumulator_t eoutx[8] = {0};
        accumulator_t eouty[8] = {0};
        accumulator_t eoutz[8] = {0};

        scalar_t jacobian_adjugate[9];
        scalar_t jacobian_determinant = 0;

        for (int v = 0; v < 8; ++v) {
            ev[v] = elements[v][i * stride];
        }

        for (int d = 0; d < 8; d++) {
            lx[d] = x[ev[d]];
            ly[d] = y[ev[d]];
            lz[d] = z[ev[d]];
        }

        for (int v = 0; v < 8; ++v) {
            const ptrdiff_t idx = ev[v] * u_stride;
            edispx[v]           = ux[idx];
            edispy[v]           = uy[idx];
            edispz[v]           = uz[idx];
        }

        for (int kz = 0; kz < n_qp; kz++) {
            for (int ky = 0; ky < n_qp; ky++) {
                for (int kx = 0; kx < n_qp; kx++) {
                    hex8_adjugate_and_det(lx, ly, lz, qx[kx], qx[ky], qx[kz], jacobian_adjugate, &jacobian_determinant);
                    assert(jacobian_determinant == jacobian_determinant);
                    assert(jacobian_determinant != 0);

                    hex8_neohookean_grad(jacobian_adjugate,
                                         jacobian_determinant,
                                         qx[kx],
                                         qx[ky],
                                         qx[kz],
                                         qw[kx] * qw[ky] * qw[kz],
                                         mu,
                                         lambda,
                                         edispx,
                                         edispy,
                                         edispz,
                                         eoutx,
                                         eouty,
                                         eoutz);
                }
            }
        }

        for (int edof_i = 0; edof_i < 8; edof_i++) {
            const ptrdiff_t idx = ev[edof_i] * out_stride;

            assert(eoutx[edof_i] == eoutx[edof_i]);
            assert(eouty[edof_i] == eouty[edof_i]);
            assert(eoutz[edof_i] == eoutz[edof_i]);

#pragma omp atomic update
            outx[idx] += eoutx[edof_i];

#pragma omp atomic update
            outy[idx] += eouty[edof_i];

#pragma omp atomic update
            outz[idx] += eoutz[edof_i];
        }
    }

    return SFEM_SUCCESS;
}

int hex8_neohookean_ogden_hessian_partial_assembly(const ptrdiff_t                      nelements,
                                                   const ptrdiff_t                      stride,
                                                   idx_t **const SFEM_RESTRICT          elements,
                                                   geom_t **const SFEM_RESTRICT         points,
                                                   const real_t                         mu,
                                                   const real_t                         lambda,
                                                   const ptrdiff_t                      u_stride,
                                                   const real_t *const SFEM_RESTRICT    ux,
                                                   const real_t *const SFEM_RESTRICT    uy,
                                                   const real_t *const SFEM_RESTRICT    uz,
                                                   metric_tensor_t *const SFEM_RESTRICT partial_assembly) {
    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t    ev[8];
        scalar_t element_ux[8];
        scalar_t element_uy[8];
        scalar_t element_uz[8];
        scalar_t lx[8];
        scalar_t ly[8];
        scalar_t lz[8];

        scalar_t jacobian_adjugate[9];
        scalar_t jacobian_determinant = 0;

        for (int v = 0; v < 8; ++v) {
            ev[v] = elements[v][i * stride];
        }

        for (int v = 0; v < 8; ++v) {
            lx[v] = x[ev[v]];
            ly[v] = y[ev[v]];
            lz[v] = z[ev[v]];
        }

        for (int v = 0; v < 8; ++v) {
            const ptrdiff_t idx = ev[v] * u_stride;
            element_ux[v]       = ux[idx];
            element_uy[v]       = uy[idx];
            element_uz[v]       = uz[idx];
        }

        static const scalar_t samplex = 0.5, sampley = 0.5, samplez = 0.5;
        hex8_adjugate_and_det(lx, ly, lz, samplex, sampley, samplez, jacobian_adjugate, &jacobian_determinant);

        // Sample at the centroid
        scalar_t F[9] = {0};
        hex8_F(jacobian_adjugate, jacobian_determinant, samplex, sampley, samplez, element_ux, element_uy, element_uz, F);
        scalar_t S_ikmn[HEX8_S_IKMN_SIZE] = {0};
        hex8_S_ikmn_neohookean(jacobian_adjugate, jacobian_determinant, samplex, sampley, samplez, F, mu, lambda, 1, S_ikmn);

        metric_tensor_t *const pai = &partial_assembly[i * HEX8_S_IKMN_SIZE];
        for (int k = 0; k < HEX8_S_IKMN_SIZE; k++) {
            assert(S_ikmn[k] == S_ikmn[k]);
            pai[k] = S_ikmn[k];
        }
    }

    return SFEM_SUCCESS;
}

int hex8_neohookean_ogden_partial_assembly_apply(const ptrdiff_t                            nelements,
                                                 const ptrdiff_t                            stride,
                                                 idx_t **const SFEM_RESTRICT                elements,
                                                 const metric_tensor_t *const SFEM_RESTRICT partial_assembly,
                                                 const ptrdiff_t                            h_stride,
                                                 const real_t *const                        hx,
                                                 const real_t *const                        hy,
                                                 const real_t *const                        hz,
                                                 const ptrdiff_t                            out_stride,
                                                 real_t *const                              outx,
                                                 real_t *const                              outy,
                                                 real_t *const                              outz) {
    scalar_t Wimpn_compressed[10];
    hex8_Wimpn_compressed(Wimpn_compressed);

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[8];

        scalar_t element_hx[8];
        scalar_t element_hy[8];
        scalar_t element_hz[8];

        accumulator_t eoutx[8];
        accumulator_t eouty[8];
        accumulator_t eoutz[8];

        for (int v = 0; v < 8; ++v) {
            ev[v] = elements[v][i * stride];
        }

        for (int v = 0; v < 8; ++v) {
            const ptrdiff_t idx = ev[v] * h_stride;
            element_hx[v]       = hx[idx];
            element_hy[v]       = hy[idx];
            element_hz[v]       = hz[idx];
        }

#if 0 
    // Slower than other variant
    scalar_t                     S_ikmn[3*3*3*3];
    hex8_expand_S(&partial_assembly[i * HEX8_S_IKMN_SIZE], S_ikmn);

    scalar_t                     Zpkmn[8*3*3*3];
    hex8_Zpkmn(Wimpn_compressed, element_hx, element_hy, element_hz, Zpkmn);
    hex8_SdotZ_expanded(S_ikmn, Zpkmn, eoutx, eouty, eoutz);
#else
        const metric_tensor_t *const pai = &partial_assembly[i * HEX8_S_IKMN_SIZE];
        scalar_t                     S_ikmn[HEX8_S_IKMN_SIZE];
        for (int k = 0; k < HEX8_S_IKMN_SIZE; k++) {
            S_ikmn[k] = pai[k];
            assert(S_ikmn[k] == S_ikmn[k]);
        }

        hex8_SdotHdotG(S_ikmn, Wimpn_compressed, element_hx, element_hy, element_hz, eoutx, eouty, eoutz);
#endif

        for (int edof_i = 0; edof_i < 8; edof_i++) {
            const ptrdiff_t idx = ev[edof_i] * out_stride;

            assert(eoutx[edof_i] == eoutx[edof_i]);
            assert(eouty[edof_i] == eouty[edof_i]);
            assert(eoutz[edof_i] == eoutz[edof_i]);

#pragma omp atomic update
            outx[idx] += eoutx[edof_i];

#pragma omp atomic update
            outy[idx] += eouty[edof_i];

#pragma omp atomic update
            outz[idx] += eoutz[edof_i];
        }
    }

    return SFEM_SUCCESS;
}

// Apply partially assembled operator
int hex8_neohookean_ogden_compressed_partial_assembly_apply(const ptrdiff_t                         nelements,
                                                            const ptrdiff_t                         stride,
                                                            idx_t **const SFEM_RESTRICT             elements,
                                                            const compressed_t *const SFEM_RESTRICT partial_assembly,
                                                            const scaling_t *const SFEM_RESTRICT    scaling,
                                                            const ptrdiff_t                         h_stride,
                                                            const real_t *const                     hx,
                                                            const real_t *const                     hy,
                                                            const real_t *const                     hz,
                                                            const ptrdiff_t                         out_stride,
                                                            real_t *const                           outx,
                                                            real_t *const                           outy,
                                                            real_t *const                           outz) {
    scalar_t Wimpn_compressed[10];
    hex8_Wimpn_compressed(Wimpn_compressed);

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[8];

        scalar_t element_hx[8];
        scalar_t element_hy[8];
        scalar_t element_hz[8];

        accumulator_t eoutx[8];
        accumulator_t eouty[8];
        accumulator_t eoutz[8];

        for (int v = 0; v < 8; ++v) {
            ev[v] = elements[v][i * stride];
        }

        for (int v = 0; v < 8; ++v) {
            const ptrdiff_t idx = ev[v] * h_stride;
            element_hx[v]       = hx[idx];
            element_hy[v]       = hy[idx];
            element_hz[v]       = hz[idx];
        }

        // Load and decompress low precision tensor
        const scalar_t            s   = scaling[i];
        const compressed_t *const pai = &partial_assembly[i * HEX8_S_IKMN_SIZE];
        scalar_t                  S_ikmn[HEX8_S_IKMN_SIZE];
        for (int k = 0; k < HEX8_S_IKMN_SIZE; k++) {
            S_ikmn[k] = s * (scalar_t)(pai[k]);
            assert(S_ikmn[k] == S_ikmn[k]);
        }

        hex8_SdotHdotG(S_ikmn, Wimpn_compressed, element_hx, element_hy, element_hz, eoutx, eouty, eoutz);

        for (int edof_i = 0; edof_i < 8; edof_i++) {
            const ptrdiff_t idx = ev[edof_i] * out_stride;

            assert(eoutx[edof_i] == eoutx[edof_i]);
            assert(eouty[edof_i] == eouty[edof_i]);
            assert(eoutz[edof_i] == eoutz[edof_i]);

#pragma omp atomic update
            outx[idx] += eoutx[edof_i];

#pragma omp atomic update
            outy[idx] += eouty[edof_i];

#pragma omp atomic update
            outz[idx] += eoutz[edof_i];
        }
    }

    return SFEM_SUCCESS;
}
