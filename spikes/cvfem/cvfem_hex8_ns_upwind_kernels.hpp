#ifndef CVFEM_HEX8_NS_UPWIND_KERNELS_HPP
#define CVFEM_HEX8_NS_UPWIND_KERNELS_HPP

#include <cstddef>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef SFEM_RESTRICT
#define SFEM_RESTRICT __restrict__
#endif

#ifndef SFEM_INLINE
#define SFEM_INLINE inline __attribute__((always_inline))
#endif

static constexpr int CVFEM_HEX8_N_FIELDS = 4;
static constexpr int CVFEM_HEX8_N_NODES  = 8;
static constexpr int CVFEM_HEX8_N_DOF    = CVFEM_HEX8_N_FIELDS * CVFEM_HEX8_N_NODES;
static constexpr int CVFEM_HEX8_N_SCS    = 12;

struct Hex8Geom {
    scalar_t cof[9];
    scalar_t det;
};

struct Hex8Face {
    int      i;
    int      j;
    scalar_t ar[3];
};

static constexpr Hex8Face CVFEM_HEX8_SCS[CVFEM_HEX8_N_SCS] = {
        {0, 1, {scalar_t(0.25), scalar_t(0), scalar_t(0)}},
        {3, 2, {scalar_t(0.25), scalar_t(0), scalar_t(0)}},
        {4, 5, {scalar_t(0.25), scalar_t(0), scalar_t(0)}},
        {7, 6, {scalar_t(0.25), scalar_t(0), scalar_t(0)}},
        {0, 3, {scalar_t(0), scalar_t(0.25), scalar_t(0)}},
        {1, 2, {scalar_t(0), scalar_t(0.25), scalar_t(0)}},
        {4, 7, {scalar_t(0), scalar_t(0.25), scalar_t(0)}},
        {5, 6, {scalar_t(0), scalar_t(0.25), scalar_t(0)}},
        {0, 4, {scalar_t(0), scalar_t(0), scalar_t(0.25)}},
        {1, 5, {scalar_t(0), scalar_t(0), scalar_t(0.25)}},
        {2, 6, {scalar_t(0), scalar_t(0), scalar_t(0.25)}},
        {3, 7, {scalar_t(0), scalar_t(0), scalar_t(0.25)}}};

static constexpr scalar_t CVFEM_HEX8_DN_REF[CVFEM_HEX8_N_NODES][3] = {
        {-scalar_t(0.25), -scalar_t(0.25), -scalar_t(0.25)},
        { scalar_t(0.25), -scalar_t(0.25), -scalar_t(0.25)},
        { scalar_t(0.25),  scalar_t(0.25), -scalar_t(0.25)},
        {-scalar_t(0.25),  scalar_t(0.25), -scalar_t(0.25)},
        {-scalar_t(0.25), -scalar_t(0.25),  scalar_t(0.25)},
        { scalar_t(0.25), -scalar_t(0.25),  scalar_t(0.25)},
        { scalar_t(0.25),  scalar_t(0.25),  scalar_t(0.25)},
        {-scalar_t(0.25),  scalar_t(0.25),  scalar_t(0.25)}};

static SFEM_INLINE void cvfem_zero_scalars(scalar_t *const SFEM_RESTRICT p, const ptrdiff_t n) {
#ifdef _OPENMP
#pragma omp parallel
    {
        const int       tid   = omp_get_thread_num();
        const int       nt    = omp_get_num_threads();
        const ptrdiff_t begin = (n * (ptrdiff_t)tid) / (ptrdiff_t)nt;
        const ptrdiff_t end   = (n * (ptrdiff_t)(tid + 1)) / (ptrdiff_t)nt;
        std::memset(p + begin, 0, (size_t)(end - begin) * sizeof(scalar_t));
    }
#else
    std::memset(p, 0, (size_t)n * sizeof(scalar_t));
#endif
}

static SFEM_INLINE void cvfem_hex8_area(const Hex8Geom &g,
                                        const scalar_t  ar0,
                                        const scalar_t  ar1,
                                        const scalar_t  ar2,
                                        scalar_t       &ax,
                                        scalar_t       &ay,
                                        scalar_t       &az) {
    ax = g.cof[0] * ar0 + g.cof[3] * ar1 + g.cof[6] * ar2;
    ay = g.cof[1] * ar0 + g.cof[4] * ar1 + g.cof[7] * ar2;
    az = g.cof[2] * ar0 + g.cof[5] * ar1 + g.cof[8] * ar2;
}

static SFEM_INLINE void cvfem_hex8_grad(const Hex8Geom                 &g,
                                        const scalar_t *const SFEM_RESTRICT ux,
                                        const scalar_t *const SFEM_RESTRICT uy,
                                        const scalar_t *const SFEM_RESTRICT uz,
                                        scalar_t *const SFEM_RESTRICT       grad) {
    scalar_t ur[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
        const scalar_t gx = CVFEM_HEX8_DN_REF[a][0];
        const scalar_t gy = CVFEM_HEX8_DN_REF[a][1];
        const scalar_t gz = CVFEM_HEX8_DN_REF[a][2];
        ur[0] += ux[a] * gx;
        ur[1] += ux[a] * gy;
        ur[2] += ux[a] * gz;
        ur[3] += uy[a] * gx;
        ur[4] += uy[a] * gy;
        ur[5] += uy[a] * gz;
        ur[6] += uz[a] * gx;
        ur[7] += uz[a] * gy;
        ur[8] += uz[a] * gz;
    }

    const scalar_t inv_det = scalar_t(1) / g.det;
    for (int c = 0; c < 3; ++c) {
        const scalar_t rx = ur[3 * c + 0];
        const scalar_t ry = ur[3 * c + 1];
        const scalar_t rz = ur[3 * c + 2];
        grad[3 * c + 0] = (g.cof[0] * rx + g.cof[3] * ry + g.cof[6] * rz) * inv_det;
        grad[3 * c + 1] = (g.cof[1] * rx + g.cof[4] * ry + g.cof[7] * rz) * inv_det;
        grad[3 * c + 2] = (g.cof[2] * rx + g.cof[5] * ry + g.cof[8] * rz) * inv_det;
    }
}

static SFEM_INLINE void cvfem_hex8_ns_upwind_residual(const scalar_t                        rho,
                                                      const scalar_t                        mu,
                                                      const Hex8Geom                       &geom,
                                                      const scalar_t *const SFEM_RESTRICT   ux,
                                                      const scalar_t *const SFEM_RESTRICT   uy,
                                                      const scalar_t *const SFEM_RESTRICT   uz,
                                                      const scalar_t *const SFEM_RESTRICT   p,
                                                      scalar_t *const SFEM_RESTRICT         r) {
    for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i) r[i] = scalar_t(0);

    scalar_t grad[9];
    cvfem_hex8_grad(geom, ux, uy, uz, grad);

    for (int s = 0; s < CVFEM_HEX8_N_SCS; ++s) {
        const int i = CVFEM_HEX8_SCS[s].i;
        const int j = CVFEM_HEX8_SCS[s].j;

        scalar_t ax, ay, az;
        cvfem_hex8_area(geom, CVFEM_HEX8_SCS[s].ar[0], CVFEM_HEX8_SCS[s].ar[1], CVFEM_HEX8_SCS[s].ar[2], ax, ay, az);

        const scalar_t adv_x = scalar_t(0.5) * (ux[i] + ux[j]);
        const scalar_t adv_y = scalar_t(0.5) * (uy[i] + uy[j]);
        const scalar_t adv_z = scalar_t(0.5) * (uz[i] + uz[j]);
        const scalar_t mdot  = rho * (adv_x * ax + adv_y * ay + adv_z * az);
        const scalar_t sgn   = mdot > scalar_t(0) ? scalar_t(1) : (mdot < scalar_t(0) ? scalar_t(-1) : scalar_t(0));
        const scalar_t mpos  = scalar_t(0.5) * (mdot + sgn * mdot);
        const scalar_t mneg  = scalar_t(0.5) * (mdot - sgn * mdot);
        const scalar_t pmid  = scalar_t(0.5) * (p[i] + p[j]);

        const scalar_t tau_x = mu * ((2 * grad[0]) * ax + (grad[1] + grad[3]) * ay + (grad[2] + grad[6]) * az);
        const scalar_t tau_y = mu * ((grad[3] + grad[1]) * ax + (2 * grad[4]) * ay + (grad[5] + grad[7]) * az);
        const scalar_t tau_z = mu * ((grad[6] + grad[2]) * ax + (grad[7] + grad[5]) * ay + (2 * grad[8]) * az);

        const scalar_t fx = mpos * ux[i] + mneg * ux[j] + pmid * ax - tau_x;
        const scalar_t fy = mpos * uy[i] + mneg * uy[j] + pmid * ay - tau_y;
        const scalar_t fz = mpos * uz[i] + mneg * uz[j] + pmid * az - tau_z;

        r[i * 4 + 0] += fx;
        r[i * 4 + 1] += fy;
        r[i * 4 + 2] += fz;
        r[i * 4 + 3] += mdot;
        r[j * 4 + 0] -= fx;
        r[j * 4 + 1] -= fy;
        r[j * 4 + 2] -= fz;
        r[j * 4 + 3] -= mdot;
    }
}

static SFEM_INLINE void cvfem_hex8_ns_upwind_jacobian_fd(const scalar_t                        rho,
                                                         const scalar_t                        mu,
                                                         const Hex8Geom                       &geom,
                                                         const scalar_t *const SFEM_RESTRICT   ux,
                                                         const scalar_t *const SFEM_RESTRICT   uy,
                                                         const scalar_t *const SFEM_RESTRICT   uz,
                                                         const scalar_t *const SFEM_RESTRICT   p,
                                                         scalar_t *const SFEM_RESTRICT         ke) {
    scalar_t q[CVFEM_HEX8_N_DOF];
    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
        q[a * 4 + 0] = ux[a];
        q[a * 4 + 1] = uy[a];
        q[a * 4 + 2] = uz[a];
        q[a * 4 + 3] = p[a];
    }

    scalar_t up[8], vp[8], wp[8], pp[8];
    scalar_t rm[CVFEM_HEX8_N_DOF], rp[CVFEM_HEX8_N_DOF];
    const scalar_t eps = scalar_t(1.0e-6);

    for (int col = 0; col < CVFEM_HEX8_N_DOF; ++col) {
        for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i) {
            const scalar_t delta = i == col ? eps : scalar_t(0);
            const int      a     = i / 4;
            const int      f     = i & 3;
            if (f == 0) up[a] = q[i] - delta;
            if (f == 1) vp[a] = q[i] - delta;
            if (f == 2) wp[a] = q[i] - delta;
            if (f == 3) pp[a] = q[i] - delta;
        }
        cvfem_hex8_ns_upwind_residual(rho, mu, geom, up, vp, wp, pp, rm);

        for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i) {
            const scalar_t delta = i == col ? eps : scalar_t(0);
            const int      a     = i / 4;
            const int      f     = i & 3;
            if (f == 0) up[a] = q[i] + delta;
            if (f == 1) vp[a] = q[i] + delta;
            if (f == 2) wp[a] = q[i] + delta;
            if (f == 3) pp[a] = q[i] + delta;
        }
        cvfem_hex8_ns_upwind_residual(rho, mu, geom, up, vp, wp, pp, rp);

        for (int row = 0; row < CVFEM_HEX8_N_DOF; ++row) {
            ke[row * CVFEM_HEX8_N_DOF + col] = (rp[row] - rm[row]) / (2 * eps);
        }
    }
}

#endif
