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

#ifndef VEC_BYTES
#define VEC_BYTES 128
#endif

#ifndef ALIGN_BYTES
#define ALIGN_BYTES 64
#endif

static constexpr int CVFEM_HEX8_N_FIELDS = 4;
static constexpr int CVFEM_HEX8_N_NODES  = 8;
static constexpr int CVFEM_HEX8_N_DOF    = CVFEM_HEX8_N_FIELDS * CVFEM_HEX8_N_NODES;
static constexpr int CVFEM_HEX8_N_SCS    = 12;
static constexpr int CVFEM_HEX8_VEC_SIZE = VEC_BYTES / int(sizeof(scalar_t));
static_assert(CVFEM_HEX8_VEC_SIZE >= 1, "invalid HEX8 vector size");

struct Hex8Geom {
    scalar_t cof[9];
    scalar_t det;
};

struct Hex8Face {
    int      i;
    int      j;
    scalar_t ar[3];
};

struct Hex8InputPack {
    alignas(ALIGN_BYTES) scalar_t ux[CVFEM_HEX8_N_NODES][CVFEM_HEX8_VEC_SIZE];
    alignas(ALIGN_BYTES) scalar_t uy[CVFEM_HEX8_N_NODES][CVFEM_HEX8_VEC_SIZE];
    alignas(ALIGN_BYTES) scalar_t uz[CVFEM_HEX8_N_NODES][CVFEM_HEX8_VEC_SIZE];
    alignas(ALIGN_BYTES) scalar_t p[CVFEM_HEX8_N_NODES][CVFEM_HEX8_VEC_SIZE];
};

struct Hex8ResidualPack {
    alignas(ALIGN_BYTES) scalar_t rx[CVFEM_HEX8_N_NODES][CVFEM_HEX8_VEC_SIZE];
    alignas(ALIGN_BYTES) scalar_t ry[CVFEM_HEX8_N_NODES][CVFEM_HEX8_VEC_SIZE];
    alignas(ALIGN_BYTES) scalar_t rz[CVFEM_HEX8_N_NODES][CVFEM_HEX8_VEC_SIZE];
    alignas(ALIGN_BYTES) scalar_t rc[CVFEM_HEX8_N_NODES][CVFEM_HEX8_VEC_SIZE];
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

static constexpr int CVFEM_HEX8_DIR_EDGES[3][4][2] = {{{0, 1}, {3, 2}, {4, 5}, {7, 6}},
                                                      {{0, 3}, {1, 2}, {4, 7}, {5, 6}},
                                                      {{0, 4}, {1, 5}, {2, 6}, {3, 7}}};

static constexpr scalar_t CVFEM_HEX8_SNET[CVFEM_HEX8_N_NODES][3] = {
        {scalar_t(1), scalar_t(1), scalar_t(1)},
        {scalar_t(-1), scalar_t(1), scalar_t(1)},
        {scalar_t(-1), scalar_t(-1), scalar_t(1)},
        {scalar_t(1), scalar_t(-1), scalar_t(1)},
        {scalar_t(1), scalar_t(1), scalar_t(-1)},
        {scalar_t(-1), scalar_t(1), scalar_t(-1)},
        {scalar_t(-1), scalar_t(-1), scalar_t(-1)},
        {scalar_t(1), scalar_t(-1), scalar_t(-1)}};

static constexpr scalar_t CVFEM_HEX8_DN_REF[CVFEM_HEX8_N_NODES][3] = {
        {-scalar_t(0.25), -scalar_t(0.25), -scalar_t(0.25)},
        {scalar_t(0.25), -scalar_t(0.25), -scalar_t(0.25)},
        {scalar_t(0.25), scalar_t(0.25), -scalar_t(0.25)},
        {-scalar_t(0.25), scalar_t(0.25), -scalar_t(0.25)},
        {-scalar_t(0.25), -scalar_t(0.25), scalar_t(0.25)},
        {scalar_t(0.25), -scalar_t(0.25), scalar_t(0.25)},
        {scalar_t(0.25), scalar_t(0.25), scalar_t(0.25)},
        {-scalar_t(0.25), scalar_t(0.25), scalar_t(0.25)}};

// Face-diff grad + 3-dir traction + 12-SCS convection (see kernel).
static constexpr double CVFEM_HEX8_RESIDUAL_FLOPS_PER_ELEMENT = 754.0;
static constexpr double CVFEM_HEX8_JAC_ACTION_FLOPS_PER_ELEMENT =
        CVFEM_HEX8_RESIDUAL_FLOPS_PER_ELEMENT + 12.0 * 8.0;
static constexpr double CVFEM_HEX8_ASSEMBLE_FLOPS_PER_ELEMENT = 2304.0;

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

static SFEM_INLINE void cvfem_hex8_face_diff(const scalar_t *const SFEM_RESTRICT u,
                                             scalar_t                           &dr,
                                             scalar_t                           &ds,
                                             scalar_t                           &dt) {
    dr = scalar_t(0.25) * ((u[1] + u[2] + u[5] + u[6]) - (u[0] + u[3] + u[4] + u[7]));
    ds = scalar_t(0.25) * ((u[2] + u[3] + u[6] + u[7]) - (u[0] + u[1] + u[4] + u[5]));
    dt = scalar_t(0.25) * ((u[4] + u[5] + u[6] + u[7]) - (u[0] + u[1] + u[2] + u[3]));
}

static SFEM_INLINE void cvfem_hex8_pushforward(const Hex8Geom &g,
                                               const scalar_t  dr,
                                               const scalar_t  ds,
                                               const scalar_t  dt,
                                               scalar_t       &gx,
                                               scalar_t       &gy,
                                               scalar_t       &gz) {
    const scalar_t inv_det = scalar_t(1) / g.det;
    gx                     = (g.cof[0] * dr + g.cof[3] * ds + g.cof[6] * dt) * inv_det;
    gy                     = (g.cof[1] * dr + g.cof[4] * ds + g.cof[7] * dt) * inv_det;
    gz                     = (g.cof[2] * dr + g.cof[5] * ds + g.cof[8] * dt) * inv_det;
}

static SFEM_INLINE void cvfem_hex8_dir_areas(const Hex8Geom &g, scalar_t A[3][3]) {
    A[0][0] = scalar_t(0.25) * g.cof[0];
    A[0][1] = scalar_t(0.25) * g.cof[1];
    A[0][2] = scalar_t(0.25) * g.cof[2];
    A[1][0] = scalar_t(0.25) * g.cof[3];
    A[1][1] = scalar_t(0.25) * g.cof[4];
    A[1][2] = scalar_t(0.25) * g.cof[5];
    A[2][0] = scalar_t(0.25) * g.cof[6];
    A[2][1] = scalar_t(0.25) * g.cof[7];
    A[2][2] = scalar_t(0.25) * g.cof[8];
}

static SFEM_INLINE void cvfem_hex8_traction(const scalar_t mu,
                                            const scalar_t g00,
                                            const scalar_t g01,
                                            const scalar_t g02,
                                            const scalar_t g10,
                                            const scalar_t g11,
                                            const scalar_t g12,
                                            const scalar_t g20,
                                            const scalar_t g21,
                                            const scalar_t g22,
                                            const scalar_t ax,
                                            const scalar_t ay,
                                            const scalar_t az,
                                            scalar_t      &tx,
                                            scalar_t      &ty,
                                            scalar_t      &tz) {
    tx = mu * ((scalar_t(2) * g00) * ax + (g01 + g10) * ay + (g02 + g20) * az);
    ty = mu * ((g10 + g01) * ax + (scalar_t(2) * g11) * ay + (g12 + g21) * az);
    tz = mu * ((g20 + g02) * ax + (g21 + g12) * ay + (scalar_t(2) * g22) * az);
}

static SFEM_INLINE void cvfem_hex8_grad_sumfact(const Hex8Geom                 &g,
                                                const scalar_t *const SFEM_RESTRICT ux,
                                                const scalar_t *const SFEM_RESTRICT uy,
                                                const scalar_t *const SFEM_RESTRICT uz,
                                                scalar_t *const SFEM_RESTRICT       grad) {
    scalar_t dr, ds, dt;
    cvfem_hex8_face_diff(ux, dr, ds, dt);
    cvfem_hex8_pushforward(g, dr, ds, dt, grad[0], grad[1], grad[2]);
    cvfem_hex8_face_diff(uy, dr, ds, dt);
    cvfem_hex8_pushforward(g, dr, ds, dt, grad[3], grad[4], grad[5]);
    cvfem_hex8_face_diff(uz, dr, ds, dt);
    cvfem_hex8_pushforward(g, dr, ds, dt, grad[6], grad[7], grad[8]);
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
        grad[3 * c + 0]   = (g.cof[0] * rx + g.cof[3] * ry + g.cof[6] * rz) * inv_det;
        grad[3 * c + 1]   = (g.cof[1] * rx + g.cof[4] * ry + g.cof[7] * rz) * inv_det;
        grad[3 * c + 2]   = (g.cof[2] * rx + g.cof[5] * ry + g.cof[8] * rz) * inv_det;
    }
}

static SFEM_INLINE void cvfem_hex8_scs_convection(const scalar_t rho,
                                                  const scalar_t ux_i,
                                                  const scalar_t ux_j,
                                                  const scalar_t uy_i,
                                                  const scalar_t uy_j,
                                                  const scalar_t uz_i,
                                                  const scalar_t uz_j,
                                                  const scalar_t p_i,
                                                  const scalar_t p_j,
                                                  const scalar_t ax,
                                                  const scalar_t ay,
                                                  const scalar_t az,
                                                  scalar_t      &fx,
                                                  scalar_t      &fy,
                                                  scalar_t      &fz,
                                                  scalar_t      &mdot) {
    const scalar_t adv_x = scalar_t(0.5) * (ux_i + ux_j);
    const scalar_t adv_y = scalar_t(0.5) * (uy_i + uy_j);
    const scalar_t adv_z = scalar_t(0.5) * (uz_i + uz_j);
    mdot                 = rho * (adv_x * ax + adv_y * ay + adv_z * az);
    const scalar_t sgn   = mdot > scalar_t(0) ? scalar_t(1) : (mdot < scalar_t(0) ? scalar_t(-1) : scalar_t(0));
    const scalar_t mpos  = scalar_t(0.5) * (mdot + sgn * mdot);
    const scalar_t mneg  = scalar_t(0.5) * (mdot - sgn * mdot);
    const scalar_t pmid  = scalar_t(0.5) * (p_i + p_j);
    fx                   = mpos * ux_i + mneg * ux_j + pmid * ax;
    fy                   = mpos * uy_i + mneg * uy_j + pmid * ay;
    fz                   = mpos * uz_i + mneg * uz_j + pmid * az;
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

static SFEM_INLINE void cvfem_hex8_ns_upwind_residual_sumfact(const scalar_t                        rho,
                                                              const scalar_t                        mu,
                                                              const Hex8Geom                       &geom,
                                                              const scalar_t *const SFEM_RESTRICT   ux,
                                                              const scalar_t *const SFEM_RESTRICT   uy,
                                                              const scalar_t *const SFEM_RESTRICT   uz,
                                                              const scalar_t *const SFEM_RESTRICT   p,
                                                              scalar_t *const SFEM_RESTRICT         r) {
    for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i) r[i] = scalar_t(0);

    scalar_t grad[9];
    cvfem_hex8_grad_sumfact(geom, ux, uy, uz, grad);

    scalar_t A[3][3];
    cvfem_hex8_dir_areas(geom, A);

    for (int d = 0; d < 3; ++d) {
        scalar_t tx, ty, tz;
        cvfem_hex8_traction(mu,
                            grad[0],
                            grad[1],
                            grad[2],
                            grad[3],
                            grad[4],
                            grad[5],
                            grad[6],
                            grad[7],
                            grad[8],
                            A[d][0],
                            A[d][1],
                            A[d][2],
                            tx,
                            ty,
                            tz);
        for (int e = 0; e < 4; ++e) {
            const int i = CVFEM_HEX8_DIR_EDGES[d][e][0];
            const int j = CVFEM_HEX8_DIR_EDGES[d][e][1];
            r[i * 4 + 0] -= tx;
            r[i * 4 + 1] -= ty;
            r[i * 4 + 2] -= tz;
            r[j * 4 + 0] += tx;
            r[j * 4 + 1] += ty;
            r[j * 4 + 2] += tz;
        }
    }

    for (int s = 0; s < CVFEM_HEX8_N_SCS; ++s) {
        const int i = CVFEM_HEX8_SCS[s].i;
        const int j = CVFEM_HEX8_SCS[s].j;
        const int d = s >> 2;
        scalar_t  fx, fy, fz, mdot;
        cvfem_hex8_scs_convection(rho,
                                  ux[i],
                                  ux[j],
                                  uy[i],
                                  uy[j],
                                  uz[i],
                                  uz[j],
                                  p[i],
                                  p[j],
                                  A[d][0],
                                  A[d][1],
                                  A[d][2],
                                  fx,
                                  fy,
                                  fz,
                                  mdot);
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

static SFEM_INLINE void cvfem_hex8_ns_upwind_jacobian_action(const scalar_t                        rho,
                                                             const scalar_t                        mu,
                                                             const Hex8Geom                       &geom,
                                                             const scalar_t *const SFEM_RESTRICT   ux,
                                                             const scalar_t *const SFEM_RESTRICT   uy,
                                                             const scalar_t *const SFEM_RESTRICT   uz,
                                                             const scalar_t *const SFEM_RESTRICT   vx,
                                                             const scalar_t *const SFEM_RESTRICT   vy,
                                                             const scalar_t *const SFEM_RESTRICT   vz,
                                                             const scalar_t *const SFEM_RESTRICT   q,
                                                             scalar_t *const SFEM_RESTRICT         r) {
    for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i) r[i] = scalar_t(0);

    scalar_t dgrad[9];
    cvfem_hex8_grad_sumfact(geom, vx, vy, vz, dgrad);

    scalar_t A[3][3];
    cvfem_hex8_dir_areas(geom, A);

    for (int d = 0; d < 3; ++d) {
        scalar_t tx, ty, tz;
        cvfem_hex8_traction(mu,
                            dgrad[0],
                            dgrad[1],
                            dgrad[2],
                            dgrad[3],
                            dgrad[4],
                            dgrad[5],
                            dgrad[6],
                            dgrad[7],
                            dgrad[8],
                            A[d][0],
                            A[d][1],
                            A[d][2],
                            tx,
                            ty,
                            tz);
        for (int e = 0; e < 4; ++e) {
            const int i = CVFEM_HEX8_DIR_EDGES[d][e][0];
            const int j = CVFEM_HEX8_DIR_EDGES[d][e][1];
            r[i * 4 + 0] -= tx;
            r[i * 4 + 1] -= ty;
            r[i * 4 + 2] -= tz;
            r[j * 4 + 0] += tx;
            r[j * 4 + 1] += ty;
            r[j * 4 + 2] += tz;
        }
    }

    const scalar_t half = scalar_t(0.5);
    const scalar_t one  = scalar_t(1);
    for (int s = 0; s < CVFEM_HEX8_N_SCS; ++s) {
        const int      i     = CVFEM_HEX8_SCS[s].i;
        const int      j     = CVFEM_HEX8_SCS[s].j;
        const int      d     = s >> 2;
        const scalar_t ax    = A[d][0];
        const scalar_t ay    = A[d][1];
        const scalar_t az    = A[d][2];
        const scalar_t adv_x = half * (ux[i] + ux[j]);
        const scalar_t adv_y = half * (uy[i] + uy[j]);
        const scalar_t adv_z = half * (uz[i] + uz[j]);
        const scalar_t mdot  = rho * (adv_x * ax + adv_y * ay + adv_z * az);
        const scalar_t sgn   = mdot > scalar_t(0) ? one : (mdot < scalar_t(0) ? -one : scalar_t(0));
        const scalar_t mpos  = half * (mdot + sgn * mdot);
        const scalar_t mneg  = half * (mdot - sgn * mdot);
        const scalar_t d_pos = half * (one + sgn);
        const scalar_t d_neg = half * (one - sgn);
        const scalar_t dmdot =
                rho * half * ((vx[i] + vx[j]) * ax + (vy[i] + vy[j]) * ay + (vz[i] + vz[j]) * az);
        const scalar_t dpos  = d_pos * dmdot;
        const scalar_t dneg  = d_neg * dmdot;
        const scalar_t qmid  = half * (q[i] + q[j]);
        const scalar_t fx    = dpos * ux[i] + mpos * vx[i] + dneg * ux[j] + mneg * vx[j] + qmid * ax;
        const scalar_t fy    = dpos * uy[i] + mpos * vy[i] + dneg * uy[j] + mneg * vy[j] + qmid * ay;
        const scalar_t fz    = dpos * uz[i] + mpos * vz[i] + dneg * uz[j] + mneg * vz[j] + qmid * az;
        r[i * 4 + 0] += fx;
        r[i * 4 + 1] += fy;
        r[i * 4 + 2] += fz;
        r[i * 4 + 3] += dmdot;
        r[j * 4 + 0] -= fx;
        r[j * 4 + 1] -= fy;
        r[j * 4 + 2] -= fz;
        r[j * 4 + 3] -= dmdot;
    }
}

template <bool Atomic>
static SFEM_INLINE void cvfem_hex8_acc(scalar_t &x, const scalar_t v) {
    if constexpr (Atomic) {
#pragma omp atomic update
        x += v;
    } else {
        x += v;
    }
}

template <bool Atomic, typename Slot>
static SFEM_INLINE void cvfem_hex8_bsr_acc(scalar_t *const SFEM_RESTRICT values,
                                           const Slot                   slot,
                                           const int                    rf,
                                           const int                    cf,
                                           const scalar_t               v) {
    cvfem_hex8_acc<Atomic>(values[(ptrdiff_t)slot * 16 + rf * 4 + cf], v);
}

template <bool Atomic, typename Slot>
static SFEM_INLINE void cvfem_hex8_ns_upwind_jacobian_add_slots(const scalar_t                        rho,
                                                                const scalar_t                        mu,
                                                                const Hex8Geom                       &geom,
                                                                const scalar_t *const SFEM_RESTRICT   ux,
                                                                const scalar_t *const SFEM_RESTRICT   uy,
                                                                const scalar_t *const SFEM_RESTRICT   uz,
                                                                const Slot *const SFEM_RESTRICT       slots,
                                                                scalar_t *const SFEM_RESTRICT         values) {
    scalar_t A[3][3];
    cvfem_hex8_dir_areas(geom, A);

    scalar_t w[CVFEM_HEX8_N_NODES][3];
    for (int k = 0; k < CVFEM_HEX8_N_NODES; ++k) {
        cvfem_hex8_pushforward(geom,
                               CVFEM_HEX8_DN_REF[k][0],
                               CVFEM_HEX8_DN_REF[k][1],
                               CVFEM_HEX8_DN_REF[k][2],
                               w[k][0],
                               w[k][1],
                               w[k][2]);
    }

    scalar_t Anet[CVFEM_HEX8_N_NODES][3];
    for (int i = 0; i < CVFEM_HEX8_N_NODES; ++i) {
        Anet[i][0] = CVFEM_HEX8_SNET[i][0] * A[0][0] + CVFEM_HEX8_SNET[i][1] * A[1][0] +
                     CVFEM_HEX8_SNET[i][2] * A[2][0];
        Anet[i][1] = CVFEM_HEX8_SNET[i][0] * A[0][1] + CVFEM_HEX8_SNET[i][1] * A[1][1] +
                     CVFEM_HEX8_SNET[i][2] * A[2][1];
        Anet[i][2] = CVFEM_HEX8_SNET[i][0] * A[0][2] + CVFEM_HEX8_SNET[i][1] * A[1][2] +
                     CVFEM_HEX8_SNET[i][2] * A[2][2];
    }

    for (int i = 0; i < CVFEM_HEX8_N_NODES; ++i) {
        const scalar_t Ax = Anet[i][0];
        const scalar_t Ay = Anet[i][1];
        const scalar_t Az = Anet[i][2];
        for (int k = 0; k < CVFEM_HEX8_N_NODES; ++k) {
            const scalar_t wx   = w[k][0];
            const scalar_t wy   = w[k][1];
            const scalar_t wz   = w[k][2];
            const scalar_t d00  = -(scalar_t(2) * wx * Ax + wy * Ay + wz * Az) * mu;
            const scalar_t d01  = -(wx * Ay) * mu;
            const scalar_t d02  = -(wx * Az) * mu;
            const scalar_t d10  = -(wy * Ax) * mu;
            const scalar_t d11  = -(wx * Ax + scalar_t(2) * wy * Ay + wz * Az) * mu;
            const scalar_t d12  = -(wy * Az) * mu;
            const scalar_t d20  = -(wz * Ax) * mu;
            const scalar_t d21  = -(wz * Ay) * mu;
            const scalar_t d22  = -(wx * Ax + wy * Ay + scalar_t(2) * wz * Az) * mu;
            const Slot     slot = slots[i * 8 + k];
            cvfem_hex8_bsr_acc<Atomic>(values, slot, 0, 0, d00);
            cvfem_hex8_bsr_acc<Atomic>(values, slot, 0, 1, d01);
            cvfem_hex8_bsr_acc<Atomic>(values, slot, 0, 2, d02);
            cvfem_hex8_bsr_acc<Atomic>(values, slot, 1, 0, d10);
            cvfem_hex8_bsr_acc<Atomic>(values, slot, 1, 1, d11);
            cvfem_hex8_bsr_acc<Atomic>(values, slot, 1, 2, d12);
            cvfem_hex8_bsr_acc<Atomic>(values, slot, 2, 0, d20);
            cvfem_hex8_bsr_acc<Atomic>(values, slot, 2, 1, d21);
            cvfem_hex8_bsr_acc<Atomic>(values, slot, 2, 2, d22);
        }
    }

    const scalar_t half  = scalar_t(0.5);
    const scalar_t one   = scalar_t(1);
    const scalar_t alpha = rho * half;
    for (int s = 0; s < CVFEM_HEX8_N_SCS; ++s) {
        const int      i     = CVFEM_HEX8_SCS[s].i;
        const int      j     = CVFEM_HEX8_SCS[s].j;
        const int      d     = s >> 2;
        const scalar_t ax    = A[d][0];
        const scalar_t ay    = A[d][1];
        const scalar_t az    = A[d][2];
        const scalar_t area[3] = {ax, ay, az};
        const scalar_t adv_x = half * (ux[i] + ux[j]);
        const scalar_t adv_y = half * (uy[i] + uy[j]);
        const scalar_t adv_z = half * (uz[i] + uz[j]);
        const scalar_t mdot  = rho * (adv_x * ax + adv_y * ay + adv_z * az);
        const scalar_t sgn   = mdot > scalar_t(0) ? one : (mdot < scalar_t(0) ? -one : scalar_t(0));
        const scalar_t mpos  = half * (mdot + sgn * mdot);
        const scalar_t mneg  = half * (mdot - sgn * mdot);
        const scalar_t d_pos = half * (one + sgn);
        const scalar_t d_neg = half * (one - sgn);
        const int      nodes[2] = {i, j};

        for (int nb = 0; nb < 2; ++nb) {
            const int b = nodes[nb];
            for (int c = 0; c < 3; ++c) {
                const scalar_t dmdot = alpha * area[c];
                const scalar_t dpos  = d_pos * dmdot;
                const scalar_t dneg  = d_neg * dmdot;
                scalar_t       dfx   = dpos * ux[i] + dneg * ux[j];
                scalar_t       dfy   = dpos * uy[i] + dneg * uy[j];
                scalar_t       dfz   = dpos * uz[i] + dneg * uz[j];
                const scalar_t mass  = (b == i) ? mpos : mneg;
                if (c == 0) dfx += mass;
                if (c == 1) dfy += mass;
                if (c == 2) dfz += mass;

                const Slot si = slots[i * 8 + b];
                const Slot sj = slots[j * 8 + b];
                cvfem_hex8_bsr_acc<Atomic>(values, si, 0, c, dfx);
                cvfem_hex8_bsr_acc<Atomic>(values, si, 1, c, dfy);
                cvfem_hex8_bsr_acc<Atomic>(values, si, 2, c, dfz);
                cvfem_hex8_bsr_acc<Atomic>(values, si, 3, c, dmdot);
                cvfem_hex8_bsr_acc<Atomic>(values, sj, 0, c, -dfx);
                cvfem_hex8_bsr_acc<Atomic>(values, sj, 1, c, -dfy);
                cvfem_hex8_bsr_acc<Atomic>(values, sj, 2, c, -dfz);
                cvfem_hex8_bsr_acc<Atomic>(values, sj, 3, c, -dmdot);
            }

            const scalar_t hax = half * ax;
            const scalar_t hay = half * ay;
            const scalar_t haz = half * az;
            const Slot     si  = slots[i * 8 + b];
            const Slot     sj  = slots[j * 8 + b];
            cvfem_hex8_bsr_acc<Atomic>(values, si, 0, 3, hax);
            cvfem_hex8_bsr_acc<Atomic>(values, si, 1, 3, hay);
            cvfem_hex8_bsr_acc<Atomic>(values, si, 2, 3, haz);
            cvfem_hex8_bsr_acc<Atomic>(values, sj, 0, 3, -hax);
            cvfem_hex8_bsr_acc<Atomic>(values, sj, 1, 3, -hay);
            cvfem_hex8_bsr_acc<Atomic>(values, sj, 2, 3, -haz);
        }
    }
}

static SFEM_INLINE void cvfem_hex8_ns_upwind_residual_sumfact_simd(
        const scalar_t                        rho_s,
        const scalar_t                        mu_s,
        const scalar_t *const SFEM_RESTRICT   cof0,
        const scalar_t *const SFEM_RESTRICT   cof1,
        const scalar_t *const SFEM_RESTRICT   cof2,
        const scalar_t *const SFEM_RESTRICT   cof3,
        const scalar_t *const SFEM_RESTRICT   cof4,
        const scalar_t *const SFEM_RESTRICT   cof5,
        const scalar_t *const SFEM_RESTRICT   cof6,
        const scalar_t *const SFEM_RESTRICT   cof7,
        const scalar_t *const SFEM_RESTRICT   cof8,
        const scalar_t *const SFEM_RESTRICT   det,
        const Hex8InputPack                  &in,
        Hex8ResidualPack                     &out) {
    const scalar_t rho  = rho_s;
    const scalar_t mu   = mu_s;
    const scalar_t half = scalar_t(0.5);
    const scalar_t qtr  = scalar_t(0.25);

    alignas(ALIGN_BYTES) scalar_t g00v[CVFEM_HEX8_VEC_SIZE], g01v[CVFEM_HEX8_VEC_SIZE], g02v[CVFEM_HEX8_VEC_SIZE];
    alignas(ALIGN_BYTES) scalar_t g10v[CVFEM_HEX8_VEC_SIZE], g11v[CVFEM_HEX8_VEC_SIZE], g12v[CVFEM_HEX8_VEC_SIZE];
    alignas(ALIGN_BYTES) scalar_t g20v[CVFEM_HEX8_VEC_SIZE], g21v[CVFEM_HEX8_VEC_SIZE], g22v[CVFEM_HEX8_VEC_SIZE];
    alignas(ALIGN_BYTES) scalar_t Ax0[CVFEM_HEX8_VEC_SIZE], Ay0[CVFEM_HEX8_VEC_SIZE], Az0[CVFEM_HEX8_VEC_SIZE];
    alignas(ALIGN_BYTES) scalar_t Ax1[CVFEM_HEX8_VEC_SIZE], Ay1[CVFEM_HEX8_VEC_SIZE], Az1[CVFEM_HEX8_VEC_SIZE];
    alignas(ALIGN_BYTES) scalar_t Ax2[CVFEM_HEX8_VEC_SIZE], Ay2[CVFEM_HEX8_VEC_SIZE], Az2[CVFEM_HEX8_VEC_SIZE];

#pragma omp simd aligned(cof0, cof1, cof2, cof3, cof4, cof5, cof6, cof7, cof8, det : 64)
    for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
        const scalar_t inv = scalar_t(1) / det[lane];
        const scalar_t c0  = cof0[lane];
        const scalar_t c1  = cof1[lane];
        const scalar_t c2  = cof2[lane];
        const scalar_t c3  = cof3[lane];
        const scalar_t c4  = cof4[lane];
        const scalar_t c5  = cof5[lane];
        const scalar_t c6  = cof6[lane];
        const scalar_t c7  = cof7[lane];
        const scalar_t c8  = cof8[lane];
        Ax0[lane]          = qtr * c0;
        Ay0[lane]          = qtr * c1;
        Az0[lane]          = qtr * c2;
        Ax1[lane]          = qtr * c3;
        Ay1[lane]          = qtr * c4;
        Az1[lane]          = qtr * c5;
        Ax2[lane]          = qtr * c6;
        Ay2[lane]          = qtr * c7;
        Az2[lane]          = qtr * c8;

        const scalar_t ux0 = in.ux[0][lane], ux1 = in.ux[1][lane], ux2 = in.ux[2][lane], ux3 = in.ux[3][lane];
        const scalar_t ux4 = in.ux[4][lane], ux5 = in.ux[5][lane], ux6 = in.ux[6][lane], ux7 = in.ux[7][lane];
        const scalar_t uy0 = in.uy[0][lane], uy1 = in.uy[1][lane], uy2 = in.uy[2][lane], uy3 = in.uy[3][lane];
        const scalar_t uy4 = in.uy[4][lane], uy5 = in.uy[5][lane], uy6 = in.uy[6][lane], uy7 = in.uy[7][lane];
        const scalar_t uz0 = in.uz[0][lane], uz1 = in.uz[1][lane], uz2 = in.uz[2][lane], uz3 = in.uz[3][lane];
        const scalar_t uz4 = in.uz[4][lane], uz5 = in.uz[5][lane], uz6 = in.uz[6][lane], uz7 = in.uz[7][lane];

        const scalar_t ur = qtr * ((ux1 + ux2 + ux5 + ux6) - (ux0 + ux3 + ux4 + ux7));
        const scalar_t us = qtr * ((ux2 + ux3 + ux6 + ux7) - (ux0 + ux1 + ux4 + ux5));
        const scalar_t ut = qtr * ((ux4 + ux5 + ux6 + ux7) - (ux0 + ux1 + ux2 + ux3));
        const scalar_t vr = qtr * ((uy1 + uy2 + uy5 + uy6) - (uy0 + uy3 + uy4 + uy7));
        const scalar_t vs = qtr * ((uy2 + uy3 + uy6 + uy7) - (uy0 + uy1 + uy4 + uy5));
        const scalar_t vt = qtr * ((uy4 + uy5 + uy6 + uy7) - (uy0 + uy1 + uy2 + uy3));
        const scalar_t wr = qtr * ((uz1 + uz2 + uz5 + uz6) - (uz0 + uz3 + uz4 + uz7));
        const scalar_t ws = qtr * ((uz2 + uz3 + uz6 + uz7) - (uz0 + uz1 + uz4 + uz5));
        const scalar_t wt = qtr * ((uz4 + uz5 + uz6 + uz7) - (uz0 + uz1 + uz2 + uz3));

        g00v[lane] = (c0 * ur + c3 * us + c6 * ut) * inv;
        g01v[lane] = (c1 * ur + c4 * us + c7 * ut) * inv;
        g02v[lane] = (c2 * ur + c5 * us + c8 * ut) * inv;
        g10v[lane] = (c0 * vr + c3 * vs + c6 * vt) * inv;
        g11v[lane] = (c1 * vr + c4 * vs + c7 * vt) * inv;
        g12v[lane] = (c2 * vr + c5 * vs + c8 * vt) * inv;
        g20v[lane] = (c0 * wr + c3 * ws + c6 * wt) * inv;
        g21v[lane] = (c1 * wr + c4 * ws + c7 * wt) * inv;
        g22v[lane] = (c2 * wr + c5 * ws + c8 * wt) * inv;

        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            out.rx[a][lane] = scalar_t(0);
            out.ry[a][lane] = scalar_t(0);
            out.rz[a][lane] = scalar_t(0);
            out.rc[a][lane] = scalar_t(0);
        }
    }

    const scalar_t *const Axv[3] = {Ax0, Ax1, Ax2};
    const scalar_t *const Ayv[3] = {Ay0, Ay1, Ay2};
    const scalar_t *const Azv[3] = {Az0, Az1, Az2};

    for (int d = 0; d < 3; ++d) {
#pragma omp simd
        for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
            scalar_t tx, ty, tz;
            cvfem_hex8_traction(mu,
                                g00v[lane],
                                g01v[lane],
                                g02v[lane],
                                g10v[lane],
                                g11v[lane],
                                g12v[lane],
                                g20v[lane],
                                g21v[lane],
                                g22v[lane],
                                Axv[d][lane],
                                Ayv[d][lane],
                                Azv[d][lane],
                                tx,
                                ty,
                                tz);
            for (int e = 0; e < 4; ++e) {
                const int i = CVFEM_HEX8_DIR_EDGES[d][e][0];
                const int j = CVFEM_HEX8_DIR_EDGES[d][e][1];
                out.rx[i][lane] -= tx;
                out.ry[i][lane] -= ty;
                out.rz[i][lane] -= tz;
                out.rx[j][lane] += tx;
                out.ry[j][lane] += ty;
                out.rz[j][lane] += tz;
            }
        }
    }

    for (int s = 0; s < CVFEM_HEX8_N_SCS; ++s) {
        const int i = CVFEM_HEX8_SCS[s].i;
        const int j = CVFEM_HEX8_SCS[s].j;
        const int d = s >> 2;
#pragma omp simd
        for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
            const scalar_t ax    = Axv[d][lane];
            const scalar_t ay    = Ayv[d][lane];
            const scalar_t az    = Azv[d][lane];
            const scalar_t adv_x = half * (in.ux[i][lane] + in.ux[j][lane]);
            const scalar_t adv_y = half * (in.uy[i][lane] + in.uy[j][lane]);
            const scalar_t adv_z = half * (in.uz[i][lane] + in.uz[j][lane]);
            const scalar_t mdot  = rho * (adv_x * ax + adv_y * ay + adv_z * az);
            const scalar_t sgn   = mdot > scalar_t(0) ? scalar_t(1) : (mdot < scalar_t(0) ? scalar_t(-1) : scalar_t(0));
            const scalar_t mpos  = half * (mdot + sgn * mdot);
            const scalar_t mneg  = half * (mdot - sgn * mdot);
            const scalar_t pmid  = half * (in.p[i][lane] + in.p[j][lane]);
            const scalar_t fx    = mpos * in.ux[i][lane] + mneg * in.ux[j][lane] + pmid * ax;
            const scalar_t fy    = mpos * in.uy[i][lane] + mneg * in.uy[j][lane] + pmid * ay;
            const scalar_t fz    = mpos * in.uz[i][lane] + mneg * in.uz[j][lane] + pmid * az;
            out.rx[i][lane] += fx;
            out.ry[i][lane] += fy;
            out.rz[i][lane] += fz;
            out.rc[i][lane] += mdot;
            out.rx[j][lane] -= fx;
            out.ry[j][lane] -= fy;
            out.rz[j][lane] -= fz;
            out.rc[j][lane] -= mdot;
        }
    }
}

static SFEM_INLINE void cvfem_hex8_ns_upwind_jacobian_action_simd(
        const scalar_t                        rho_s,
        const scalar_t                        mu_s,
        const scalar_t *const SFEM_RESTRICT   cof0,
        const scalar_t *const SFEM_RESTRICT   cof1,
        const scalar_t *const SFEM_RESTRICT   cof2,
        const scalar_t *const SFEM_RESTRICT   cof3,
        const scalar_t *const SFEM_RESTRICT   cof4,
        const scalar_t *const SFEM_RESTRICT   cof5,
        const scalar_t *const SFEM_RESTRICT   cof6,
        const scalar_t *const SFEM_RESTRICT   cof7,
        const scalar_t *const SFEM_RESTRICT   cof8,
        const scalar_t *const SFEM_RESTRICT   det,
        const Hex8InputPack                  &u,
        const Hex8InputPack                  &du,
        Hex8ResidualPack                     &out) {
    const scalar_t rho  = rho_s;
    const scalar_t mu   = mu_s;
    const scalar_t half = scalar_t(0.5);
    const scalar_t qtr  = scalar_t(0.25);
    const scalar_t one  = scalar_t(1);

    alignas(ALIGN_BYTES) scalar_t g00v[CVFEM_HEX8_VEC_SIZE], g01v[CVFEM_HEX8_VEC_SIZE], g02v[CVFEM_HEX8_VEC_SIZE];
    alignas(ALIGN_BYTES) scalar_t g10v[CVFEM_HEX8_VEC_SIZE], g11v[CVFEM_HEX8_VEC_SIZE], g12v[CVFEM_HEX8_VEC_SIZE];
    alignas(ALIGN_BYTES) scalar_t g20v[CVFEM_HEX8_VEC_SIZE], g21v[CVFEM_HEX8_VEC_SIZE], g22v[CVFEM_HEX8_VEC_SIZE];
    alignas(ALIGN_BYTES) scalar_t Ax0[CVFEM_HEX8_VEC_SIZE], Ay0[CVFEM_HEX8_VEC_SIZE], Az0[CVFEM_HEX8_VEC_SIZE];
    alignas(ALIGN_BYTES) scalar_t Ax1[CVFEM_HEX8_VEC_SIZE], Ay1[CVFEM_HEX8_VEC_SIZE], Az1[CVFEM_HEX8_VEC_SIZE];
    alignas(ALIGN_BYTES) scalar_t Ax2[CVFEM_HEX8_VEC_SIZE], Ay2[CVFEM_HEX8_VEC_SIZE], Az2[CVFEM_HEX8_VEC_SIZE];

#pragma omp simd aligned(cof0, cof1, cof2, cof3, cof4, cof5, cof6, cof7, cof8, det : 64)
    for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
        const scalar_t inv = scalar_t(1) / det[lane];
        const scalar_t c0  = cof0[lane];
        const scalar_t c1  = cof1[lane];
        const scalar_t c2  = cof2[lane];
        const scalar_t c3  = cof3[lane];
        const scalar_t c4  = cof4[lane];
        const scalar_t c5  = cof5[lane];
        const scalar_t c6  = cof6[lane];
        const scalar_t c7  = cof7[lane];
        const scalar_t c8  = cof8[lane];
        Ax0[lane]          = qtr * c0;
        Ay0[lane]          = qtr * c1;
        Az0[lane]          = qtr * c2;
        Ax1[lane]          = qtr * c3;
        Ay1[lane]          = qtr * c4;
        Az1[lane]          = qtr * c5;
        Ax2[lane]          = qtr * c6;
        Ay2[lane]          = qtr * c7;
        Az2[lane]          = qtr * c8;

        const scalar_t ux0 = du.ux[0][lane], ux1 = du.ux[1][lane], ux2 = du.ux[2][lane], ux3 = du.ux[3][lane];
        const scalar_t ux4 = du.ux[4][lane], ux5 = du.ux[5][lane], ux6 = du.ux[6][lane], ux7 = du.ux[7][lane];
        const scalar_t uy0 = du.uy[0][lane], uy1 = du.uy[1][lane], uy2 = du.uy[2][lane], uy3 = du.uy[3][lane];
        const scalar_t uy4 = du.uy[4][lane], uy5 = du.uy[5][lane], uy6 = du.uy[6][lane], uy7 = du.uy[7][lane];
        const scalar_t uz0 = du.uz[0][lane], uz1 = du.uz[1][lane], uz2 = du.uz[2][lane], uz3 = du.uz[3][lane];
        const scalar_t uz4 = du.uz[4][lane], uz5 = du.uz[5][lane], uz6 = du.uz[6][lane], uz7 = du.uz[7][lane];

        const scalar_t ur = qtr * ((ux1 + ux2 + ux5 + ux6) - (ux0 + ux3 + ux4 + ux7));
        const scalar_t us = qtr * ((ux2 + ux3 + ux6 + ux7) - (ux0 + ux1 + ux4 + ux5));
        const scalar_t ut = qtr * ((ux4 + ux5 + ux6 + ux7) - (ux0 + ux1 + ux2 + ux3));
        const scalar_t vr = qtr * ((uy1 + uy2 + uy5 + uy6) - (uy0 + uy3 + uy4 + uy7));
        const scalar_t vs = qtr * ((uy2 + uy3 + uy6 + uy7) - (uy0 + uy1 + uy4 + uy5));
        const scalar_t vt = qtr * ((uy4 + uy5 + uy6 + uy7) - (uy0 + uy1 + uy2 + uy3));
        const scalar_t wr = qtr * ((uz1 + uz2 + uz5 + uz6) - (uz0 + uz3 + uz4 + uz7));
        const scalar_t ws = qtr * ((uz2 + uz3 + uz6 + uz7) - (uz0 + uz1 + uz4 + uz5));
        const scalar_t wt = qtr * ((uz4 + uz5 + uz6 + uz7) - (uz0 + uz1 + uz2 + uz3));

        g00v[lane] = (c0 * ur + c3 * us + c6 * ut) * inv;
        g01v[lane] = (c1 * ur + c4 * us + c7 * ut) * inv;
        g02v[lane] = (c2 * ur + c5 * us + c8 * ut) * inv;
        g10v[lane] = (c0 * vr + c3 * vs + c6 * vt) * inv;
        g11v[lane] = (c1 * vr + c4 * vs + c7 * vt) * inv;
        g12v[lane] = (c2 * vr + c5 * vs + c8 * vt) * inv;
        g20v[lane] = (c0 * wr + c3 * ws + c6 * wt) * inv;
        g21v[lane] = (c1 * wr + c4 * ws + c7 * wt) * inv;
        g22v[lane] = (c2 * wr + c5 * ws + c8 * wt) * inv;

        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            out.rx[a][lane] = scalar_t(0);
            out.ry[a][lane] = scalar_t(0);
            out.rz[a][lane] = scalar_t(0);
            out.rc[a][lane] = scalar_t(0);
        }
    }

    const scalar_t *const Axv[3] = {Ax0, Ax1, Ax2};
    const scalar_t *const Ayv[3] = {Ay0, Ay1, Ay2};
    const scalar_t *const Azv[3] = {Az0, Az1, Az2};

    for (int d = 0; d < 3; ++d) {
#pragma omp simd
        for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
            scalar_t tx, ty, tz;
            cvfem_hex8_traction(mu,
                                g00v[lane],
                                g01v[lane],
                                g02v[lane],
                                g10v[lane],
                                g11v[lane],
                                g12v[lane],
                                g20v[lane],
                                g21v[lane],
                                g22v[lane],
                                Axv[d][lane],
                                Ayv[d][lane],
                                Azv[d][lane],
                                tx,
                                ty,
                                tz);
            for (int e = 0; e < 4; ++e) {
                const int i = CVFEM_HEX8_DIR_EDGES[d][e][0];
                const int j = CVFEM_HEX8_DIR_EDGES[d][e][1];
                out.rx[i][lane] -= tx;
                out.ry[i][lane] -= ty;
                out.rz[i][lane] -= tz;
                out.rx[j][lane] += tx;
                out.ry[j][lane] += ty;
                out.rz[j][lane] += tz;
            }
        }
    }

    for (int s = 0; s < CVFEM_HEX8_N_SCS; ++s) {
        const int i = CVFEM_HEX8_SCS[s].i;
        const int j = CVFEM_HEX8_SCS[s].j;
        const int d = s >> 2;
#pragma omp simd
        for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
            const scalar_t ax    = Axv[d][lane];
            const scalar_t ay    = Ayv[d][lane];
            const scalar_t az    = Azv[d][lane];
            const scalar_t adv_x = half * (u.ux[i][lane] + u.ux[j][lane]);
            const scalar_t adv_y = half * (u.uy[i][lane] + u.uy[j][lane]);
            const scalar_t adv_z = half * (u.uz[i][lane] + u.uz[j][lane]);
            const scalar_t mdot  = rho * (adv_x * ax + adv_y * ay + adv_z * az);
            const scalar_t sgn   = mdot > scalar_t(0) ? one : (mdot < scalar_t(0) ? -one : scalar_t(0));
            const scalar_t mpos  = half * (mdot + sgn * mdot);
            const scalar_t mneg  = half * (mdot - sgn * mdot);
            const scalar_t d_pos = half * (one + sgn);
            const scalar_t d_neg = half * (one - sgn);
            const scalar_t dmdot = rho * half *
                                   ((du.ux[i][lane] + du.ux[j][lane]) * ax + (du.uy[i][lane] + du.uy[j][lane]) * ay +
                                    (du.uz[i][lane] + du.uz[j][lane]) * az);
            const scalar_t dpos = d_pos * dmdot;
            const scalar_t dneg = d_neg * dmdot;
            const scalar_t qmid = half * (du.p[i][lane] + du.p[j][lane]);
            const scalar_t fx =
                    dpos * u.ux[i][lane] + mpos * du.ux[i][lane] + dneg * u.ux[j][lane] + mneg * du.ux[j][lane] + qmid * ax;
            const scalar_t fy =
                    dpos * u.uy[i][lane] + mpos * du.uy[i][lane] + dneg * u.uy[j][lane] + mneg * du.uy[j][lane] + qmid * ay;
            const scalar_t fz =
                    dpos * u.uz[i][lane] + mpos * du.uz[i][lane] + dneg * u.uz[j][lane] + mneg * du.uz[j][lane] + qmid * az;
            out.rx[i][lane] += fx;
            out.ry[i][lane] += fy;
            out.rz[i][lane] += fz;
            out.rc[i][lane] += dmdot;
            out.rx[j][lane] -= fx;
            out.ry[j][lane] -= fy;
            out.rz[j][lane] -= fz;
            out.rc[j][lane] -= dmdot;
        }
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

    scalar_t       up[8], vp[8], wp[8], pp[8];
    scalar_t       rm[CVFEM_HEX8_N_DOF], rp[CVFEM_HEX8_N_DOF];
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
