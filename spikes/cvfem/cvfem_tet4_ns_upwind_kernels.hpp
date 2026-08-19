#ifndef CVFEM_TET4_NS_UPWIND_KERNELS_HPP
#define CVFEM_TET4_NS_UPWIND_KERNELS_HPP

#include <cassert>
#include <cstddef>
#include <cstdint>
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

#ifndef CVFEM_IDX_INVALID
#define CVFEM_IDX_INVALID (-1)
#endif

static constexpr int CVFEM_N_FIELDS = 4;
static constexpr int CVFEM_N_NODES  = 4;
static constexpr int CVFEM_N_DOF    = CVFEM_N_NODES * CVFEM_N_FIELDS;

struct Tet4InputPack {
    scalar_t ux[4][VEC_SIZE];
    scalar_t uy[4][VEC_SIZE];
    scalar_t uz[4][VEC_SIZE];
    scalar_t p[4][VEC_SIZE];
};

struct Tet4ResidualPack {
    scalar_t rx[4][VEC_SIZE];
    scalar_t ry[4][VEC_SIZE];
    scalar_t rz[4][VEC_SIZE];
    scalar_t rc[4][VEC_SIZE];
};

static SFEM_INLINE const jacobian_t *cvfem_aligned_geom(const jacobian_t *p) {
    return static_cast<const jacobian_t *>(__builtin_assume_aligned(p, ALIGN_BYTES));
}

static constexpr double CVFEM_RESIDUAL_FLOPS_PER_ELEMENT = 562.0;

// Source add/mul/div in cvfem_tet4_ns_upwind_jacobian_dense. Not counted: abs/ternary,
// float→double casts, Ke zero-stores. Folded zero SCS areas: 3×15 + 3×9.
// Body per SCS: adv 6 + mdot 6 + pos/neg 4 + d_pos/d_neg 4 + dmdot 4
//   + 6 conv columns (14 mul + 17 add) + pressure (3+12) + viscous 4 nodes (27 mul + 24 add).
static constexpr double CVFEM_JACOBIAN_FLOPS_PER_ELEMENT = 1.0 + 9.0 + 6.0 + (3.0 * 15.0 + 3.0 * 9.0) +
                                                           6.0 * (6.0 + 6.0 + 4.0 + 4.0 + 4.0 + 6.0 * 31.0 + 15.0 + 4.0 * 51.0);

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

static SFEM_INLINE void cvfem_tet4_ns_upwind_simd_microkernel(const scalar_t                        rho_s,
                                                              const scalar_t                        mu_s,
                                                              const jacobian_t *const SFEM_RESTRICT adj0_ptr,
                                                              const jacobian_t *const SFEM_RESTRICT adj1_ptr,
                                                              const jacobian_t *const SFEM_RESTRICT adj2_ptr,
                                                              const jacobian_t *const SFEM_RESTRICT adj3_ptr,
                                                              const jacobian_t *const SFEM_RESTRICT adj4_ptr,
                                                              const jacobian_t *const SFEM_RESTRICT adj5_ptr,
                                                              const jacobian_t *const SFEM_RESTRICT adj6_ptr,
                                                              const jacobian_t *const SFEM_RESTRICT adj7_ptr,
                                                              const jacobian_t *const SFEM_RESTRICT adj8_ptr,
                                                              const jacobian_t *const SFEM_RESTRICT det_ptr,
                                                              const Tet4InputPack                  &in,
                                                              Tet4ResidualPack                     &out) {
    const scalar_t half = 0.5;
    const scalar_t two  = 2.0;
    const scalar_t rho  = rho_s;
    const scalar_t mu   = mu_s;

    const scalar_t c12 = 1.0 / 12.0;
    const scalar_t c24 = 1.0 / 24.0;

    alignas(ALIGN_BYTES) scalar_t g00v[VEC_SIZE], g01v[VEC_SIZE], g02v[VEC_SIZE];
    alignas(ALIGN_BYTES) scalar_t g10v[VEC_SIZE], g11v[VEC_SIZE], g12v[VEC_SIZE];
    alignas(ALIGN_BYTES) scalar_t g20v[VEC_SIZE], g21v[VEC_SIZE], g22v[VEC_SIZE];

#pragma omp simd aligned(adj0_ptr, adj1_ptr, adj2_ptr, adj3_ptr, adj4_ptr, adj5_ptr, adj6_ptr, adj7_ptr, adj8_ptr, det_ptr : 64)
    for (int lane = 0; lane < VEC_SIZE; ++lane) {
        const scalar_t adj0    = scalar_t(adj0_ptr[lane]);
        const scalar_t adj1    = scalar_t(adj1_ptr[lane]);
        const scalar_t adj2    = scalar_t(adj2_ptr[lane]);
        const scalar_t adj3    = scalar_t(adj3_ptr[lane]);
        const scalar_t adj4    = scalar_t(adj4_ptr[lane]);
        const scalar_t adj5    = scalar_t(adj5_ptr[lane]);
        const scalar_t adj6    = scalar_t(adj6_ptr[lane]);
        const scalar_t adj7    = scalar_t(adj7_ptr[lane]);
        const scalar_t adj8    = scalar_t(adj8_ptr[lane]);
        const scalar_t inv_det = 1.0 / scalar_t(det_ptr[lane]);
        const scalar_t ux0     = in.ux[0][lane];
        const scalar_t dux0    = in.ux[1][lane] - ux0;
        const scalar_t dux1    = in.ux[2][lane] - ux0;
        const scalar_t dux2    = in.ux[3][lane] - ux0;
        const scalar_t uy0     = in.uy[0][lane];
        const scalar_t duy0    = in.uy[1][lane] - uy0;
        const scalar_t duy1    = in.uy[2][lane] - uy0;
        const scalar_t duy2    = in.uy[3][lane] - uy0;
        const scalar_t uz0     = in.uz[0][lane];
        const scalar_t duz0    = in.uz[1][lane] - uz0;
        const scalar_t duz1    = in.uz[2][lane] - uz0;
        const scalar_t duz2    = in.uz[3][lane] - uz0;
        g00v[lane]             = (dux0 * adj0 + dux1 * adj3 + dux2 * adj6) * inv_det;
        g01v[lane]             = (dux0 * adj1 + dux1 * adj4 + dux2 * adj7) * inv_det;
        g02v[lane]             = (dux0 * adj2 + dux1 * adj5 + dux2 * adj8) * inv_det;
        g10v[lane]             = (duy0 * adj0 + duy1 * adj3 + duy2 * adj6) * inv_det;
        g11v[lane]             = (duy0 * adj1 + duy1 * adj4 + duy2 * adj7) * inv_det;
        g12v[lane]             = (duy0 * adj2 + duy1 * adj5 + duy2 * adj8) * inv_det;
        g20v[lane]             = (duz0 * adj0 + duz1 * adj3 + duz2 * adj6) * inv_det;
        g21v[lane]             = (duz0 * adj1 + duz1 * adj4 + duz2 * adj7) * inv_det;
        g22v[lane]             = (duz0 * adj2 + duz1 * adj5 + duz2 * adj8) * inv_det;
        for (int a = 0; a < 4; ++a) {
            out.rx[a][lane] = 0.0;
            out.ry[a][lane] = 0.0;
            out.rz[a][lane] = 0.0;
            out.rc[a][lane] = 0.0;
        }
    }

#define GEOM_SIMD_PRAGMA                                                                             \
    _Pragma(                                                                                         \
            "omp simd aligned(adj0_ptr, adj1_ptr, adj2_ptr, adj3_ptr, adj4_ptr, adj5_ptr, adj6_ptr, adj7_ptr, adj8_ptr, det_ptr: 64)")

#define SCS_AREA3(AR0, AR1, AR2)                                                                     \
    const scalar_t adj0 = scalar_t(adj0_ptr[lane]);                                                  \
    const scalar_t adj1 = scalar_t(adj1_ptr[lane]);                                                  \
    const scalar_t adj2 = scalar_t(adj2_ptr[lane]);                                                  \
    const scalar_t adj3 = scalar_t(adj3_ptr[lane]);                                                  \
    const scalar_t adj4 = scalar_t(adj4_ptr[lane]);                                                  \
    const scalar_t adj5 = scalar_t(adj5_ptr[lane]);                                                  \
    const scalar_t adj6 = scalar_t(adj6_ptr[lane]);                                                  \
    const scalar_t adj7 = scalar_t(adj7_ptr[lane]);                                                  \
    const scalar_t adj8 = scalar_t(adj8_ptr[lane]);                                                  \
    const scalar_t ax   = adj0 * (AR0) + adj3 * (AR1) + adj6 * (AR2);                                 \
    const scalar_t ay   = adj1 * (AR0) + adj4 * (AR1) + adj7 * (AR2);                                 \
    const scalar_t az   = adj2 * (AR0) + adj5 * (AR1) + adj8 * (AR2)

#define SCS_AREA_AR2_0(AR0, AR1)                                                                     \
    const scalar_t adj0 = scalar_t(adj0_ptr[lane]);                                                  \
    const scalar_t adj1 = scalar_t(adj1_ptr[lane]);                                                  \
    const scalar_t adj2 = scalar_t(adj2_ptr[lane]);                                                  \
    const scalar_t adj3 = scalar_t(adj3_ptr[lane]);                                                  \
    const scalar_t adj4 = scalar_t(adj4_ptr[lane]);                                                  \
    const scalar_t adj5 = scalar_t(adj5_ptr[lane]);                                                  \
    const scalar_t ax   = adj0 * (AR0) + adj3 * (AR1);                                                \
    const scalar_t ay   = adj1 * (AR0) + adj4 * (AR1);                                                \
    const scalar_t az   = adj2 * (AR0) + adj5 * (AR1)

#define SCS_AREA_AR1_0(AR0, AR2)                                                                     \
    const scalar_t adj0 = scalar_t(adj0_ptr[lane]);                                                  \
    const scalar_t adj1 = scalar_t(adj1_ptr[lane]);                                                  \
    const scalar_t adj2 = scalar_t(adj2_ptr[lane]);                                                  \
    const scalar_t adj6 = scalar_t(adj6_ptr[lane]);                                                  \
    const scalar_t adj7 = scalar_t(adj7_ptr[lane]);                                                  \
    const scalar_t adj8 = scalar_t(adj8_ptr[lane]);                                                  \
    const scalar_t ax   = adj0 * (AR0) + adj6 * (AR2);                                                \
    const scalar_t ay   = adj1 * (AR0) + adj7 * (AR2);                                                \
    const scalar_t az   = adj2 * (AR0) + adj8 * (AR2)

#define SCS_AREA_AR0_0(AR1, AR2)                                                                     \
    const scalar_t adj3 = scalar_t(adj3_ptr[lane]);                                                  \
    const scalar_t adj4 = scalar_t(adj4_ptr[lane]);                                                  \
    const scalar_t adj5 = scalar_t(adj5_ptr[lane]);                                                  \
    const scalar_t adj6 = scalar_t(adj6_ptr[lane]);                                                  \
    const scalar_t adj7 = scalar_t(adj7_ptr[lane]);                                                  \
    const scalar_t adj8 = scalar_t(adj8_ptr[lane]);                                                  \
    const scalar_t ax   = adj3 * (AR1) + adj6 * (AR2);                                                \
    const scalar_t ay   = adj4 * (AR1) + adj7 * (AR2);                                                \
    const scalar_t az   = adj5 * (AR1) + adj8 * (AR2)

#define SCS_FLUX_LANES(I, J, AREA)                                                                   \
    do {                                                                                             \
        GEOM_SIMD_PRAGMA for (int lane = 0; lane < VEC_SIZE; ++lane) {                               \
            AREA;                                                                                    \
            const scalar_t uxI      = in.ux[I][lane];                                                \
            const scalar_t uxJ      = in.ux[J][lane];                                                \
            const scalar_t uyI      = in.uy[I][lane];                                                \
            const scalar_t uyJ      = in.uy[J][lane];                                                \
            const scalar_t uzI      = in.uz[I][lane];                                                \
            const scalar_t uzJ      = in.uz[J][lane];                                                \
            const scalar_t adv_x    = half * (uxI + uxJ);                                            \
            const scalar_t adv_y    = half * (uyI + uyJ);                                            \
            const scalar_t adv_z    = half * (uzI + uzJ);                                            \
            const scalar_t mdot     = rho * (adv_x * ax + adv_y * ay + adv_z * az);                  \
            const scalar_t mdot_abs = mdot < scalar_t(0) ? -mdot : mdot;                             \
            const scalar_t mdot_pos = half * (mdot + mdot_abs);                                      \
            const scalar_t mdot_neg = half * (mdot - mdot_abs);                                      \
            const scalar_t p_mid    = half * (in.p[I][lane] + in.p[J][lane]);                        \
            const scalar_t g00      = g00v[lane];                                                    \
            const scalar_t g01      = g01v[lane];                                                    \
            const scalar_t g02      = g02v[lane];                                                    \
            const scalar_t g10      = g10v[lane];                                                    \
            const scalar_t g11      = g11v[lane];                                                    \
            const scalar_t g12      = g12v[lane];                                                    \
            const scalar_t g20      = g20v[lane];                                                    \
            const scalar_t g21      = g21v[lane];                                                    \
            const scalar_t g22      = g22v[lane];                                                    \
            const scalar_t tau_x    = mu * ((two * g00) * ax + (g01 + g10) * ay + (g02 + g20) * az); \
            const scalar_t tau_y    = mu * ((g10 + g01) * ax + (two * g11) * ay + (g12 + g21) * az); \
            const scalar_t tau_z    = mu * ((g20 + g02) * ax + (g21 + g12) * ay + (two * g22) * az); \
            const scalar_t fx       = mdot_pos * uxI + mdot_neg * uxJ + p_mid * ax - tau_x;          \
            const scalar_t fy       = mdot_pos * uyI + mdot_neg * uyJ + p_mid * ay - tau_y;          \
            const scalar_t fz       = mdot_pos * uzI + mdot_neg * uzJ + p_mid * az - tau_z;          \
            out.rx[I][lane] += fx;                                                                   \
            out.ry[I][lane] += fy;                                                                   \
            out.rz[I][lane] += fz;                                                                   \
            out.rc[I][lane] += mdot;                                                                 \
            out.rx[J][lane] -= fx;                                                                   \
            out.ry[J][lane] -= fy;                                                                   \
            out.rz[J][lane] -= fz;                                                                   \
            out.rc[J][lane] -= mdot;                                                                 \
        }                                                                                            \
    } while (0)

    SCS_FLUX_LANES(0, 1, SCS_AREA3(c12, c24, c24));
    SCS_FLUX_LANES(0, 2, SCS_AREA3(c24, c12, c24));
    SCS_FLUX_LANES(0, 3, SCS_AREA3(c24, c24, c12));
    SCS_FLUX_LANES(1, 2, SCS_AREA_AR2_0(-c24, c24));
    SCS_FLUX_LANES(1, 3, SCS_AREA_AR1_0(-c24, c24));
    SCS_FLUX_LANES(2, 3, SCS_AREA_AR0_0(-c24, c24));

#undef SCS_FLUX_LANES
#undef SCS_AREA3
#undef SCS_AREA_AR2_0
#undef SCS_AREA_AR1_0
#undef SCS_AREA_AR0_0
#undef GEOM_SIMD_PRAGMA
}

static SFEM_INLINE void cvfem_tet4_ns_upwind_jacobian_dense(const scalar_t                rho,
                                                            const scalar_t                mu,
                                                            const scalar_t                adj0,
                                                            const scalar_t                adj1,
                                                            const scalar_t                adj2,
                                                            const scalar_t                adj3,
                                                            const scalar_t                adj4,
                                                            const scalar_t                adj5,
                                                            const scalar_t                adj6,
                                                            const scalar_t                adj7,
                                                            const scalar_t                adj8,
                                                            const scalar_t                det,
                                                            const scalar_t                ux[4],
                                                            const scalar_t                uy[4],
                                                            const scalar_t                uz[4],
                                                            scalar_t *const SFEM_RESTRICT ke) {
    const scalar_t half = 0.5;
    const scalar_t two  = 2.0;
    const scalar_t one  = 1.0;
    const scalar_t c12  = 1.0 / 12.0;
    const scalar_t c24  = 1.0 / 24.0;
    const scalar_t inv_det = 1.0 / det;

    scalar_t gx[4], gy[4], gz[4];
    gx[1] = adj0 * inv_det;
    gy[1] = adj1 * inv_det;
    gz[1] = adj2 * inv_det;
    gx[2] = adj3 * inv_det;
    gy[2] = adj4 * inv_det;
    gz[2] = adj5 * inv_det;
    gx[3] = adj6 * inv_det;
    gy[3] = adj7 * inv_det;
    gz[3] = adj8 * inv_det;
    gx[0] = -(gx[1] + gx[2] + gx[3]);
    gy[0] = -(gy[1] + gy[2] + gy[3]);
    gz[0] = -(gz[1] + gz[2] + gz[3]);

#pragma omp simd aligned(ke : 64)
    for (int t = 0; t < CVFEM_N_DOF * CVFEM_N_DOF; ++t) ke[t] = 0.0;

#define KE(r, c) ke[(r) * CVFEM_N_DOF + (c)]

#define JAC_AREA3(AR0, AR1, AR2)                       \
    const scalar_t ax = adj0 * (AR0) + adj3 * (AR1) + adj6 * (AR2); \
    const scalar_t ay = adj1 * (AR0) + adj4 * (AR1) + adj7 * (AR2); \
    const scalar_t az = adj2 * (AR0) + adj5 * (AR1) + adj8 * (AR2)

#define JAC_AREA_AR2_0(AR0, AR1)                       \
    const scalar_t ax = adj0 * (AR0) + adj3 * (AR1);   \
    const scalar_t ay = adj1 * (AR0) + adj4 * (AR1);   \
    const scalar_t az = adj2 * (AR0) + adj5 * (AR1)

#define JAC_AREA_AR1_0(AR0, AR2)                       \
    const scalar_t ax = adj0 * (AR0) + adj6 * (AR2);   \
    const scalar_t ay = adj1 * (AR0) + adj7 * (AR2);   \
    const scalar_t az = adj2 * (AR0) + adj8 * (AR2)

#define JAC_AREA_AR0_0(AR1, AR2)                       \
    const scalar_t ax = adj3 * (AR1) + adj6 * (AR2);   \
    const scalar_t ay = adj4 * (AR1) + adj7 * (AR2);   \
    const scalar_t az = adj5 * (AR1) + adj8 * (AR2)

#define JAC_CONV_COL(I, J, K, FIELD, DMDOT, DUXI, DUXJ, DUYI, DUYJ, DUZI, DUZJ) \
    do {                                                                        \
        const scalar_t dpos = d_pos * (DMDOT);                                  \
        const scalar_t dneg = d_neg * (DMDOT);                                  \
        const scalar_t dcx  = dpos * uxI + mdot_pos * (DUXI) + dneg * uxJ + mdot_neg * (DUXJ); \
        const scalar_t dcy  = dpos * uyI + mdot_pos * (DUYI) + dneg * uyJ + mdot_neg * (DUYJ); \
        const scalar_t dcz  = dpos * uzI + mdot_pos * (DUZI) + dneg * uzJ + mdot_neg * (DUZJ); \
        const int      col  = (K) * 4 + (FIELD);                                \
        KE((I) * 4 + 0, col) += dcx;                                            \
        KE((I) * 4 + 1, col) += dcy;                                            \
        KE((I) * 4 + 2, col) += dcz;                                            \
        KE((J) * 4 + 0, col) -= dcx;                                            \
        KE((J) * 4 + 1, col) -= dcy;                                            \
        KE((J) * 4 + 2, col) -= dcz;                                            \
        KE((I) * 4 + 3, col) += (DMDOT);                                        \
        KE((J) * 4 + 3, col) -= (DMDOT);                                        \
    } while (0)

#define SCS_JAC(I, J, AREA)                                                                     \
    do {                                                                                        \
        AREA;                                                                                   \
        const scalar_t uxI      = ux[I];                                                        \
        const scalar_t uxJ      = ux[J];                                                        \
        const scalar_t uyI      = uy[I];                                                        \
        const scalar_t uyJ      = uy[J];                                                        \
        const scalar_t uzI      = uz[I];                                                        \
        const scalar_t uzJ      = uz[J];                                                        \
        const scalar_t adv_x    = half * (uxI + uxJ);                                           \
        const scalar_t adv_y    = half * (uyI + uyJ);                                           \
        const scalar_t adv_z    = half * (uzI + uzJ);                                           \
        const scalar_t mdot     = rho * (adv_x * ax + adv_y * ay + adv_z * az);                 \
        const scalar_t mdot_abs = mdot < scalar_t(0) ? -mdot : mdot;                            \
        const scalar_t mdot_pos = half * (mdot + mdot_abs);                                     \
        const scalar_t mdot_neg = half * (mdot - mdot_abs);                                     \
        const scalar_t sgn      = mdot > scalar_t(0) ? one : (mdot < scalar_t(0) ? -one : 0.0); \
        const scalar_t d_pos    = half * (one + sgn);                                           \
        const scalar_t d_neg    = half * (one - sgn);                                           \
        const scalar_t rh       = rho * half;                                                   \
        const scalar_t dmdot_dux = rh * ax;                                                     \
        const scalar_t dmdot_duy = rh * ay;                                                     \
        const scalar_t dmdot_duz = rh * az;                                                     \
        JAC_CONV_COL(I, J, I, 0, dmdot_dux, one, 0.0, 0.0, 0.0, 0.0, 0.0);                      \
        JAC_CONV_COL(I, J, I, 1, dmdot_duy, 0.0, 0.0, one, 0.0, 0.0, 0.0);                      \
        JAC_CONV_COL(I, J, I, 2, dmdot_duz, 0.0, 0.0, 0.0, 0.0, one, 0.0);                      \
        JAC_CONV_COL(I, J, J, 0, dmdot_dux, 0.0, one, 0.0, 0.0, 0.0, 0.0);                      \
        JAC_CONV_COL(I, J, J, 1, dmdot_duy, 0.0, 0.0, 0.0, one, 0.0, 0.0);                      \
        JAC_CONV_COL(I, J, J, 2, dmdot_duz, 0.0, 0.0, 0.0, 0.0, 0.0, one);                      \
        const scalar_t dpx = half * ax;                                                         \
        const scalar_t dpy = half * ay;                                                         \
        const scalar_t dpz = half * az;                                                         \
        KE((I) * 4 + 0, (I) * 4 + 3) += dpx;                                                    \
        KE((I) * 4 + 1, (I) * 4 + 3) += dpy;                                                    \
        KE((I) * 4 + 2, (I) * 4 + 3) += dpz;                                                    \
        KE((J) * 4 + 0, (I) * 4 + 3) -= dpx;                                                    \
        KE((J) * 4 + 1, (I) * 4 + 3) -= dpy;                                                    \
        KE((J) * 4 + 2, (I) * 4 + 3) -= dpz;                                                    \
        KE((I) * 4 + 0, (J) * 4 + 3) += dpx;                                                    \
        KE((I) * 4 + 1, (J) * 4 + 3) += dpy;                                                    \
        KE((I) * 4 + 2, (J) * 4 + 3) += dpz;                                                    \
        KE((J) * 4 + 0, (J) * 4 + 3) -= dpx;                                                    \
        KE((J) * 4 + 1, (J) * 4 + 3) -= dpy;                                                    \
        KE((J) * 4 + 2, (J) * 4 + 3) -= dpz;                                                    \
        for (int k = 0; k < 4; ++k) {                                                           \
            const scalar_t gk0    = gx[k];                                                      \
            const scalar_t gk1    = gy[k];                                                      \
            const scalar_t gk2    = gz[k];                                                      \
            const scalar_t dtx_ux = mu * (two * gk0 * ax + gk1 * ay + gk2 * az);                \
            const scalar_t dtx_uy = mu * (gk0 * ay);                                            \
            const scalar_t dtx_uz = mu * (gk0 * az);                                            \
            const scalar_t dty_ux = mu * (gk1 * ax);                                            \
            const scalar_t dty_uy = mu * (gk0 * ax + two * gk1 * ay + gk2 * az);                \
            const scalar_t dty_uz = mu * (gk1 * az);                                            \
            const scalar_t dtz_ux = mu * (gk2 * ax);                                            \
            const scalar_t dtz_uy = mu * (gk2 * ay);                                            \
            const scalar_t dtz_uz = mu * (gk0 * ax + gk1 * ay + two * gk2 * az);                \
            const int      col_x  = k * 4 + 0;                                                  \
            const int      col_y  = k * 4 + 1;                                                  \
            const int      col_z  = k * 4 + 2;                                                  \
            KE((I) * 4 + 0, col_x) -= dtx_ux;                                                   \
            KE((I) * 4 + 0, col_y) -= dtx_uy;                                                   \
            KE((I) * 4 + 0, col_z) -= dtx_uz;                                                   \
            KE((I) * 4 + 1, col_x) -= dty_ux;                                                   \
            KE((I) * 4 + 1, col_y) -= dty_uy;                                                   \
            KE((I) * 4 + 1, col_z) -= dty_uz;                                                   \
            KE((I) * 4 + 2, col_x) -= dtz_ux;                                                   \
            KE((I) * 4 + 2, col_y) -= dtz_uy;                                                   \
            KE((I) * 4 + 2, col_z) -= dtz_uz;                                                   \
            KE((J) * 4 + 0, col_x) += dtx_ux;                                                   \
            KE((J) * 4 + 0, col_y) += dtx_uy;                                                   \
            KE((J) * 4 + 0, col_z) += dtx_uz;                                                   \
            KE((J) * 4 + 1, col_x) += dty_ux;                                                   \
            KE((J) * 4 + 1, col_y) += dty_uy;                                                   \
            KE((J) * 4 + 1, col_z) += dty_uz;                                                   \
            KE((J) * 4 + 2, col_x) += dtz_ux;                                                   \
            KE((J) * 4 + 2, col_y) += dtz_uy;                                                   \
            KE((J) * 4 + 2, col_z) += dtz_uz;                                                   \
        }                                                                                       \
    } while (0)

    SCS_JAC(0, 1, JAC_AREA3(c12, c24, c24));
    SCS_JAC(0, 2, JAC_AREA3(c24, c12, c24));
    SCS_JAC(0, 3, JAC_AREA3(c24, c24, c12));
    SCS_JAC(1, 2, JAC_AREA_AR2_0(-c24, c24));
    SCS_JAC(1, 3, JAC_AREA_AR1_0(-c24, c24));
    SCS_JAC(2, 3, JAC_AREA_AR0_0(-c24, c24));

#undef SCS_JAC
#undef JAC_CONV_COL
#undef JAC_AREA3
#undef JAC_AREA_AR2_0
#undef JAC_AREA_AR1_0
#undef JAC_AREA_AR0_0
#undef KE
}

static SFEM_INLINE void cvfem_pad_geom_lanes(const jacobian_t *const SFEM_RESTRICT adj0,
                                             const jacobian_t *const SFEM_RESTRICT adj1,
                                             const jacobian_t *const SFEM_RESTRICT adj2,
                                             const jacobian_t *const SFEM_RESTRICT adj3,
                                             const jacobian_t *const SFEM_RESTRICT adj4,
                                             const jacobian_t *const SFEM_RESTRICT adj5,
                                             const jacobian_t *const SFEM_RESTRICT adj6,
                                             const jacobian_t *const SFEM_RESTRICT adj7,
                                             const jacobian_t *const SFEM_RESTRICT adj8,
                                             const jacobian_t *const SFEM_RESTRICT det,
                                             const int                             nlanes,
                                             jacobian_t *const SFEM_RESTRICT       a0,
                                             jacobian_t *const SFEM_RESTRICT       a1,
                                             jacobian_t *const SFEM_RESTRICT       a2,
                                             jacobian_t *const SFEM_RESTRICT       a3,
                                             jacobian_t *const SFEM_RESTRICT       a4,
                                             jacobian_t *const SFEM_RESTRICT       a5,
                                             jacobian_t *const SFEM_RESTRICT       a6,
                                             jacobian_t *const SFEM_RESTRICT       a7,
                                             jacobian_t *const SFEM_RESTRICT       a8,
                                             jacobian_t *const SFEM_RESTRICT       det_out) {
    const int last = nlanes - 1;
    for (int lane = 0; lane < VEC_SIZE; ++lane) {
        const int e = lane < nlanes ? lane : last;
        a0[lane]    = adj0[e];
        a1[lane]    = adj1[e];
        a2[lane]    = adj2[e];
        a3[lane]    = adj3[e];
        a4[lane]    = adj4[e];
        a5[lane]    = adj5[e];
        a6[lane]    = adj6[e];
        a7[lane]    = adj7[e];
        a8[lane]    = adj8[e];
        det_out[lane] = det[e];
    }
}

static SFEM_INLINE void cvfem_run_residual_kernel(const scalar_t                        rho,
                                                  const scalar_t                        mu,
                                                  const jacobian_t *const SFEM_RESTRICT adj0,
                                                  const jacobian_t *const SFEM_RESTRICT adj1,
                                                  const jacobian_t *const SFEM_RESTRICT adj2,
                                                  const jacobian_t *const SFEM_RESTRICT adj3,
                                                  const jacobian_t *const SFEM_RESTRICT adj4,
                                                  const jacobian_t *const SFEM_RESTRICT adj5,
                                                  const jacobian_t *const SFEM_RESTRICT adj6,
                                                  const jacobian_t *const SFEM_RESTRICT adj7,
                                                  const jacobian_t *const SFEM_RESTRICT adj8,
                                                  const jacobian_t *const SFEM_RESTRICT det,
                                                  const int                             nlanes,
                                                  const Tet4InputPack                  &in,
                                                  Tet4ResidualPack                     &out) {
    if (nlanes == VEC_SIZE) {
        cvfem_tet4_ns_upwind_simd_microkernel(rho,
                                              mu,
                                              cvfem_aligned_geom(adj0),
                                              cvfem_aligned_geom(adj1),
                                              cvfem_aligned_geom(adj2),
                                              cvfem_aligned_geom(adj3),
                                              cvfem_aligned_geom(adj4),
                                              cvfem_aligned_geom(adj5),
                                              cvfem_aligned_geom(adj6),
                                              cvfem_aligned_geom(adj7),
                                              cvfem_aligned_geom(adj8),
                                              cvfem_aligned_geom(det),
                                              in,
                                              out);
        return;
    }
    alignas(ALIGN_BYTES) jacobian_t a0[VEC_SIZE], a1[VEC_SIZE], a2[VEC_SIZE], a3[VEC_SIZE], a4[VEC_SIZE];
    alignas(ALIGN_BYTES) jacobian_t a5[VEC_SIZE], a6[VEC_SIZE], a7[VEC_SIZE], a8[VEC_SIZE], detp[VEC_SIZE];
    cvfem_pad_geom_lanes(adj0, adj1, adj2, adj3, adj4, adj5, adj6, adj7, adj8, det, nlanes, a0, a1, a2, a3, a4, a5, a6, a7, a8, detp);
    cvfem_tet4_ns_upwind_simd_microkernel(rho, mu, a0, a1, a2, a3, a4, a5, a6, a7, a8, detp, in, out);
}

template <typename Idx>
static SFEM_INLINE Idx cvfem_linear_search(const Idx target, const Idx *const arr, const int size) {
    int i = 0;
    for (; i < size - 4; i += 4) {
        if (arr[i] == target) return (Idx)i;
        if (arr[i + 1] == target) return (Idx)(i + 1);
        if (arr[i + 2] == target) return (Idx)(i + 2);
        if (arr[i + 3] == target) return (Idx)(i + 3);
    }
    for (; i < size; ++i) {
        if (arr[i] == target) return (Idx)i;
    }
    return (Idx)CVFEM_IDX_INVALID;
}

template <typename Idx>
static SFEM_INLINE void cvfem_find_cols4(const Idx *const SFEM_RESTRICT targets,
                                         const Idx *const SFEM_RESTRICT row,
                                         const int                      lenrow,
                                         Idx *const SFEM_RESTRICT       ks) {
    // if (lenrow > 32) {
    //     for (int d = 0; d < 4; ++d) {
    //         ks[d] = cvfem_linear_search(targets[d], row, lenrow);
    //     }
    //     return;
    // }
    for (int d = 0; d < 4; ++d) ks[d] = 0;
    for (int i = 0; i < lenrow; ++i) {
        for (int d = 0; d < 4; ++d) {
            ks[d] = (Idx)((int)ks[d] + (int)(row[i] < targets[d]));
        }
    }
}

template <bool Atomic>
static SFEM_INLINE void cvfem_bsr4_accum_dense_block(scalar_t *const SFEM_RESTRICT       block,
                                                     const scalar_t *const SFEM_RESTRICT src00) {
    if constexpr (Atomic) {
        for (int fi = 0; fi < 4; ++fi) {
            const scalar_t *const s = src00 + fi * 16;
            scalar_t *const       d = block + fi * 4;
#pragma omp atomic update
            d[0] += s[0];
#pragma omp atomic update
            d[1] += s[1];
#pragma omp atomic update
            d[2] += s[2];
#pragma omp atomic update
            d[3] += s[3];
        }
    } else {
#pragma unroll(4)
        for (int fi = 0; fi < 4; ++fi) {
            const scalar_t *const s = src00 + fi * 16;
            scalar_t *const       d = block + fi * 4;
            d[0] += s[0];
            d[1] += s[1];
            d[2] += s[2];
            d[3] += s[3];
        }
    }
}

static SFEM_INLINE void cvfem_bsr4_add16(scalar_t *const SFEM_RESTRICT       dst,
                                         const scalar_t *const SFEM_RESTRICT src) {
#pragma omp simd
    for (int t = 0; t < 16; ++t) dst[t] += src[t];
}

template <bool Atomic, typename Count, typename Idx>
static SFEM_INLINE void tet4_local_to_global_bsr4(const Idx *const SFEM_RESTRICT           ev,
                                                  const scalar_t *const SFEM_RESTRICT      element_matrix,
                                                  const Count *const SFEM_RESTRICT         rowptr,
                                                  const Idx *const SFEM_RESTRICT           colidx,
                                                  scalar_t *const SFEM_RESTRICT            values) {
    Idx ks[4];
    for (int edof_i = 0; edof_i < 4; ++edof_i) {
        const Idx  dof_i  = ev[edof_i];
        const int  lenrow = int(rowptr[dof_i + 1] - rowptr[dof_i]);
        const Idx *cols   = &colidx[rowptr[dof_i]];
        cvfem_find_cols4(ev, cols, lenrow, ks);

        const scalar_t *const ke_i = element_matrix + (edof_i * 4) * 16;
        for (int edof_j = 0; edof_j < 4; ++edof_j) {
            scalar_t *const SFEM_RESTRICT block = &values[(rowptr[dof_i] + ks[edof_j]) * 16];
            cvfem_bsr4_accum_dense_block<Atomic>(block, ke_i + edof_j * 4);
        }
    }
}

#include "cvfem_tet4_ns_upwind_sympy_kernels.hpp"

#endif
