#ifndef CVFEM_HEX8_NS_UPWIND_KERNELS_HPP
#define CVFEM_HEX8_NS_UPWIND_KERNELS_HPP

#include <cmath>
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

#include "cvfem_portability.hpp"

static constexpr int CVFEM_HEX8_N_FIELDS = 4;
static constexpr int CVFEM_HEX8_N_NODES  = 8;
static constexpr int CVFEM_HEX8_N_DOF    = CVFEM_HEX8_N_FIELDS * CVFEM_HEX8_N_NODES;
static constexpr int CVFEM_HEX8_N_SCS    = 12;
static constexpr int CVFEM_HEX8_VEC_SIZE = VEC_BYTES / int(sizeof(scalar_t));
static_assert(CVFEM_HEX8_VEC_SIZE >= 1, "invalid HEX8 vector size");

struct Hex8Face {
    int    i;
    int    j;
    double ar[3];
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

struct Hex8CoordPack {
    alignas(ALIGN_BYTES) scalar_t x[CVFEM_HEX8_N_NODES][CVFEM_HEX8_VEC_SIZE];
    alignas(ALIGN_BYTES) scalar_t y[CVFEM_HEX8_N_NODES][CVFEM_HEX8_VEC_SIZE];
    alignas(ALIGN_BYTES) scalar_t z[CVFEM_HEX8_N_NODES][CVFEM_HEX8_VEC_SIZE];
};

struct Hex8RhieChowPack {
    alignas(ALIGN_BYTES) scalar_t x[CVFEM_HEX8_N_NODES][CVFEM_HEX8_VEC_SIZE];
    alignas(ALIGN_BYTES) scalar_t y[CVFEM_HEX8_N_NODES][CVFEM_HEX8_VEC_SIZE];
    alignas(ALIGN_BYTES) scalar_t z[CVFEM_HEX8_N_NODES][CVFEM_HEX8_VEC_SIZE];
    alignas(ALIGN_BYTES) scalar_t pgx[CVFEM_HEX8_N_NODES][CVFEM_HEX8_VEC_SIZE];
    alignas(ALIGN_BYTES) scalar_t pgy[CVFEM_HEX8_N_NODES][CVFEM_HEX8_VEC_SIZE];
    alignas(ALIGN_BYTES) scalar_t pgz[CVFEM_HEX8_N_NODES][CVFEM_HEX8_VEC_SIZE];
};

/* Optional colocated Rhie–Chow: u_f = u_avg - D_f[(p_j-p_i)/h - 0.5(∇p_i+∇p_j)·e].
   scale=0 (default) keeps the original mdot = rho u_avg · A. Nodal ∇p is assembled
   outside the element kernel; globally linear p is unchanged. */
template <typename T>
struct Hex8RhieChowT {
    const T *x{};
    const T *y{};
    const T *z{};
    const T *pgx{};
    const T *pgy{};
    const T *pgz{};
    T        scale{0};
};

// Every existing host call site names `Hex8RhieChow`, so keep that spelling bound
// to the file-scope scalar type. Device code names Hex8RhieChowT<T> directly.
using Hex8RhieChow = Hex8RhieChowT<scalar_t>;

// The compile-time tables below must be readable from device code.
//
// A namespace-scope `static constexpr` array is host-only under nvcc -- it fails with
// "identifier ... is undefined in device code", and so do the plain-constexpr and
// static-data-member spellings (all three were tried). What nvcc does accept on both
// sides is a `static constexpr` array local to a __host__ __device__ function.
//
// The accessor form is therefore used only under __CUDACC__, and the plain array is
// kept everywhere else: routing the host build through the accessor changed its
// generated code (+20 bytes and a good deal of register reshuffling), and there is no
// reason to perturb the CPU path to solve a device problem. The initialiser is shared
// between the two forms, so the data is still written once. Every use site is
// unchanged in both cases, because the name is aliased to the accessor by macro.

#define CVFEM_HEX8_SCS_INIT { \
        {0, 1, {double(0.25), double(0), double(0)}}, \
        {3, 2, {double(0.25), double(0), double(0)}}, \
        {4, 5, {double(0.25), double(0), double(0)}}, \
        {7, 6, {double(0.25), double(0), double(0)}}, \
        {0, 3, {double(0), double(0.25), double(0)}}, \
        {1, 2, {double(0), double(0.25), double(0)}}, \
        {4, 7, {double(0), double(0.25), double(0)}}, \
        {5, 6, {double(0), double(0.25), double(0)}}, \
        {0, 4, {double(0), double(0), double(0.25)}}, \
        {1, 5, {double(0), double(0), double(0.25)}}, \
        {2, 6, {double(0), double(0), double(0.25)}}, \
        {3, 7, {double(0), double(0), double(0.25)}}}
#if defined(__CUDACC__)
static SFEM_INLINE SFEM_HOST_DEVICE const Hex8Face (&cvfem_hex8_scs_tbl())[CVFEM_HEX8_N_SCS] {
    static constexpr Hex8Face t[CVFEM_HEX8_N_SCS] = CVFEM_HEX8_SCS_INIT;
    return t;
}
#define CVFEM_HEX8_SCS cvfem_hex8_scs_tbl()
#else
static constexpr Hex8Face CVFEM_HEX8_SCS[CVFEM_HEX8_N_SCS] = CVFEM_HEX8_SCS_INIT;
#endif

#define CVFEM_HEX8_DIR_EDGES_INIT {{{0, 1}, {3, 2}, {4, 5}, {7, 6}}, \
                                   {{0, 3}, {1, 2}, {4, 7}, {5, 6}}, \
                                   {{0, 4}, {1, 5}, {2, 6}, {3, 7}}}
#if defined(__CUDACC__)
static SFEM_INLINE SFEM_HOST_DEVICE const int (&cvfem_hex8_dir_edges_tbl())[3][4][2] {
    static constexpr int t[3][4][2] = CVFEM_HEX8_DIR_EDGES_INIT;
    return t;
}
#define CVFEM_HEX8_DIR_EDGES cvfem_hex8_dir_edges_tbl()
#else
static constexpr int CVFEM_HEX8_DIR_EDGES[3][4][2] = CVFEM_HEX8_DIR_EDGES_INIT;
#endif

#define CVFEM_HEX8_SNET_INIT { \
        {double(1), double(1), double(1)}, \
        {double(-1), double(1), double(1)}, \
        {double(-1), double(-1), double(1)}, \
        {double(1), double(-1), double(1)}, \
        {double(1), double(1), double(-1)}, \
        {double(-1), double(1), double(-1)}, \
        {double(-1), double(-1), double(-1)}, \
        {double(1), double(-1), double(-1)}}
#if defined(__CUDACC__)
static SFEM_INLINE SFEM_HOST_DEVICE const double (&cvfem_hex8_snet_tbl())[CVFEM_HEX8_N_NODES][3] {
    static constexpr double t[CVFEM_HEX8_N_NODES][3] = CVFEM_HEX8_SNET_INIT;
    return t;
}
#define CVFEM_HEX8_SNET cvfem_hex8_snet_tbl()
#else
static constexpr double CVFEM_HEX8_SNET[CVFEM_HEX8_N_NODES][3] = CVFEM_HEX8_SNET_INIT;
#endif

#define CVFEM_HEX8_DN_REF_INIT { \
        {-double(0.25), -double(0.25), -double(0.25)}, \
        {double(0.25), -double(0.25), -double(0.25)}, \
        {double(0.25), double(0.25), -double(0.25)}, \
        {-double(0.25), double(0.25), -double(0.25)}, \
        {-double(0.25), -double(0.25), double(0.25)}, \
        {double(0.25), -double(0.25), double(0.25)}, \
        {double(0.25), double(0.25), double(0.25)}, \
        {-double(0.25), double(0.25), double(0.25)}}
#if defined(__CUDACC__)
static SFEM_INLINE SFEM_HOST_DEVICE const double (&cvfem_hex8_dn_ref_tbl())[CVFEM_HEX8_N_NODES][3] {
    static constexpr double t[CVFEM_HEX8_N_NODES][3] = CVFEM_HEX8_DN_REF_INIT;
    return t;
}
#define CVFEM_HEX8_DN_REF cvfem_hex8_dn_ref_tbl()
#else
static constexpr double CVFEM_HEX8_DN_REF[CVFEM_HEX8_N_NODES][3] = CVFEM_HEX8_DN_REF_INIT;
#endif

// Edge-associated SCS centroids in [0,1]^3, toward the element center.
#define CVFEM_HEX8_SCS_XI_INIT { \
        {double(0.5), double(0.25), double(0.25)}, \
        {double(0.5), double(0.75), double(0.25)}, \
        {double(0.5), double(0.25), double(0.75)}, \
        {double(0.5), double(0.75), double(0.75)}, \
        {double(0.25), double(0.5), double(0.25)}, \
        {double(0.75), double(0.5), double(0.25)}, \
        {double(0.25), double(0.5), double(0.75)}, \
        {double(0.75), double(0.5), double(0.75)}, \
        {double(0.25), double(0.25), double(0.5)}, \
        {double(0.75), double(0.25), double(0.5)}, \
        {double(0.75), double(0.75), double(0.5)}, \
        {double(0.25), double(0.75), double(0.5)}}
#if defined(__CUDACC__)
static SFEM_INLINE SFEM_HOST_DEVICE const double (&cvfem_hex8_scs_xi_tbl())[CVFEM_HEX8_N_SCS][3] {
    static constexpr double t[CVFEM_HEX8_N_SCS][3] = CVFEM_HEX8_SCS_XI_INIT;
    return t;
}
#define CVFEM_HEX8_SCS_XI cvfem_hex8_scs_xi_tbl()
#else
static constexpr double CVFEM_HEX8_SCS_XI[CVFEM_HEX8_N_SCS][3] = CVFEM_HEX8_SCS_XI_INIT;
#endif

// Face-diff grad + 3-dir traction + 12-SCS convection (see kernel).
static constexpr double CVFEM_HEX8_RESIDUAL_FLOPS_PER_ELEMENT = 754.0;
static constexpr double CVFEM_HEX8_JAC_ACTION_FLOPS_PER_ELEMENT =
        CVFEM_HEX8_RESIDUAL_FLOPS_PER_ELEMENT + 12.0 * 8.0;
// NOTE: this is an idealised work model, not the arithmetic the kernels actually
// issue. Counting operators in the emitted code gives ~2900 flops/element for
// cvfem_hex8_ns_upwind_jacobian_add_slots and ~5471 for the SymPy
// add_local_slots variant, so GFLOP/s_assemble_model understates the SymPy
// assembly rate by roughly 2.4x. Compare assembly variants with MDOF/s.
static constexpr double CVFEM_HEX8_ASSEMBLE_FLOPS_PER_ELEMENT = 2304.0;

// 12 SCS: dN (27) + J (144) + cof/det (33) + A (3) + ∇_ref u (144) + push (55)
// + traction (24) + convection (36) + scatter (8) = 474 * 12.
static constexpr double CVFEM_HEX8_ISOPARAM_RESIDUAL_FLOPS_PER_ELEMENT = 5688.0;
static constexpr double CVFEM_HEX8_ISOPARAM_JAC_ACTION_FLOPS_PER_ELEMENT =
        CVFEM_HEX8_ISOPARAM_RESIDUAL_FLOPS_PER_ELEMENT + 12.0 * (144.0 + 55.0 + 8.0);
static constexpr double CVFEM_HEX8_ISOPARAM_ASSEMBLE_FLOPS_PER_ELEMENT = 9216.0;

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

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_area(const scalar_t *const SFEM_RESTRICT adj,
                                        const scalar_t                      ar0,
                                        const scalar_t                      ar1,
                                        const scalar_t                      ar2,
                                        scalar_t                           &ax,
                                        scalar_t                           &ay,
                                        scalar_t                           &az) {
    ax = adj[0] * ar0 + adj[3] * ar1 + adj[6] * ar2;
    ay = adj[1] * ar0 + adj[4] * ar1 + adj[7] * ar2;
    az = adj[2] * ar0 + adj[5] * ar1 + adj[8] * ar2;
}

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_area_dir(const scalar_t *const SFEM_RESTRICT adj, const int d, scalar_t &ax,
                                            scalar_t &ay, scalar_t &az) {
    const scalar_t qtr = scalar_t(0.25);
    ax                 = qtr * adj[3 * d];
    ay                 = qtr * adj[3 * d + 1];
    az                 = qtr * adj[3 * d + 2];
}

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_face_diff(const scalar_t *const SFEM_RESTRICT u,
                                             scalar_t                           &dr,
                                             scalar_t                           &ds,
                                             scalar_t                           &dt) {
    dr = scalar_t(0.25) * ((u[1] + u[2] + u[5] + u[6]) - (u[0] + u[3] + u[4] + u[7]));
    ds = scalar_t(0.25) * ((u[2] + u[3] + u[6] + u[7]) - (u[0] + u[1] + u[4] + u[5]));
    dt = scalar_t(0.25) * ((u[4] + u[5] + u[6] + u[7]) - (u[0] + u[1] + u[2] + u[3]));
}

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_pushforward(const scalar_t *const SFEM_RESTRICT adj,
                                               const scalar_t                      inv_det,
                                               const scalar_t                      dr,
                                               const scalar_t                      ds,
                                               const scalar_t                      dt,
                                               scalar_t                           &gx,
                                               scalar_t                           &gy,
                                               scalar_t                           &gz) {
    gx = (adj[0] * dr + adj[3] * ds + adj[6] * dt) * inv_det;
    gy = (adj[1] * dr + adj[4] * ds + adj[7] * dt) * inv_det;
    gz = (adj[2] * dr + adj[5] * ds + adj[8] * dt) * inv_det;
}

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_dn_ref(const scalar_t xi,
                                          const scalar_t eta,
                                          const scalar_t zeta,
                                          scalar_t       dN[CVFEM_HEX8_N_NODES][3]) {
    const scalar_t x0 = scalar_t(1) - xi;
    const scalar_t y0 = scalar_t(1) - eta;
    const scalar_t z0 = scalar_t(1) - zeta;
    dN[0][0]          = -y0 * z0;
    dN[0][1]          = -x0 * z0;
    dN[0][2]          = -x0 * y0;
    dN[1][0]          = y0 * z0;
    dN[1][1]          = -xi * z0;
    dN[1][2]          = -xi * y0;
    dN[2][0]          = eta * z0;
    dN[2][1]          = xi * z0;
    dN[2][2]          = -xi * eta;
    dN[3][0]          = -eta * z0;
    dN[3][1]          = x0 * z0;
    dN[3][2]          = -x0 * eta;
    dN[4][0]          = -y0 * zeta;
    dN[4][1]          = -x0 * zeta;
    dN[4][2]          = x0 * y0;
    dN[5][0]          = y0 * zeta;
    dN[5][1]          = -xi * zeta;
    dN[5][2]          = xi * y0;
    dN[6][0]          = eta * zeta;
    dN[6][1]          = xi * zeta;
    dN[6][2]          = xi * eta;
    dN[7][0]          = -eta * zeta;
    dN[7][1]          = x0 * zeta;
    dN[7][2]          = x0 * eta;
}

// Vendored verbatim from operators/hex8/hex8_inline_cpu.hpp and templated on the
// scalar type. Vendored rather than included because that header is host-only:
// it pulls <stdio.h> under !NDEBUG. The arithmetic is unchanged.
template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_adjugate_and_det(const scalar_t *const SFEM_RESTRICT x,
                                              const scalar_t *const SFEM_RESTRICT y,
                                              const scalar_t *const SFEM_RESTRICT z,
                                              const scalar_t                      qx,
                                              const scalar_t                      qy,
                                              const scalar_t                      qz,
                                              scalar_t *const SFEM_RESTRICT       adjugate,
                                              scalar_t *const SFEM_RESTRICT       jacobian_determinant) {
    scalar_t jacobian[9];
    {
        const scalar_t x0  = qy * qz;
        const scalar_t x1  = 1 - qz;
        const scalar_t x2  = qy * x1;
        const scalar_t x3  = 1 - qy;
        const scalar_t x4  = qz * x3;
        const scalar_t x5  = x1 * x3;
        const scalar_t x6  = qx * qz;
        const scalar_t x7  = qx * x1;
        const scalar_t x8  = 1 - qx;
        const scalar_t x9  = qz * x8;
        const scalar_t x10 = x1 * x8;
        const scalar_t x11 = qx * qy;
        const scalar_t x12 = qx * x3;
        const scalar_t x13 = qy * x8;
        const scalar_t x14 = x3 * x8;

        jacobian[0] = x0 * x[6] - x0 * x[7] + x2 * x[2] - x2 * x[3] - x4 * x[4] + x4 * x[5] - x5 * x[0] + x5 * x[1];
        jacobian[1] = qx * qz * x[6] + qx * x1 * x[2] + qz * x8 * x[7] + x1 * x8 * x[3] - x10 * x[0] - x6 * x[5] - x7 * x[1] -
                      x9 * x[4];
        jacobian[2] = qx * qy * x[6] + qx * x3 * x[5] + qy * x8 * x[7] - x11 * x[2] - x12 * x[1] - x13 * x[3] - x14 * x[0] +
                      x3 * x8 * x[4];
        jacobian[3] = x0 * y[6] - x0 * y[7] + x2 * y[2] - x2 * y[3] - x4 * y[4] + x4 * y[5] - x5 * y[0] + x5 * y[1];
        jacobian[4] = qx * qz * y[6] + qx * x1 * y[2] + qz * x8 * y[7] + x1 * x8 * y[3] - x10 * y[0] - x6 * y[5] - x7 * y[1] -
                      x9 * y[4];
        jacobian[5] = qx * qy * y[6] + qx * x3 * y[5] + qy * x8 * y[7] - x11 * y[2] - x12 * y[1] - x13 * y[3] - x14 * y[0] +
                      x3 * x8 * y[4];
        jacobian[6] = x0 * z[6] - x0 * z[7] + x2 * z[2] - x2 * z[3] - x4 * z[4] + x4 * z[5] - x5 * z[0] + x5 * z[1];
        jacobian[7] = qx * qz * z[6] + qx * x1 * z[2] + qz * x8 * z[7] + x1 * x8 * z[3] - x10 * z[0] - x6 * z[5] - x7 * z[1] -
                      x9 * z[4];
        jacobian[8] = qx * qy * z[6] + qx * x3 * z[5] + qy * x8 * z[7] - x11 * z[2] - x12 * z[1] - x13 * z[3] - x14 * z[0] +
                      x3 * x8 * z[4];
    }

    const scalar_t x0 = jacobian[4] * jacobian[8];
    const scalar_t x1 = jacobian[5] * jacobian[7];
    const scalar_t x2 = jacobian[1] * jacobian[8];
    const scalar_t x3 = jacobian[1] * jacobian[5];
    const scalar_t x4 = jacobian[2] * jacobian[4];

    adjugate[0]           = x0 - x1;
    adjugate[1]           = jacobian[2] * jacobian[7] - x2;
    adjugate[2]           = x3 - x4;
    adjugate[3]           = -jacobian[3] * jacobian[8] + jacobian[5] * jacobian[6];
    adjugate[4]           = jacobian[0] * jacobian[8] - jacobian[2] * jacobian[6];
    adjugate[5]           = -jacobian[0] * jacobian[5] + jacobian[2] * jacobian[3];
    adjugate[6]           = jacobian[3] * jacobian[7] - jacobian[4] * jacobian[6];
    adjugate[7]           = -jacobian[0] * jacobian[7] + jacobian[1] * jacobian[6];
    adjugate[8]           = jacobian[0] * jacobian[4] - jacobian[1] * jacobian[3];
    *jacobian_determinant = jacobian[0] * x0 - jacobian[0] * x1 + jacobian[2] * jacobian[3] * jacobian[7] - jacobian[3] * x2 +
                            jacobian[6] * x3 - jacobian[6] * x4;
}

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_geom_at(const scalar_t *const SFEM_RESTRICT x,
                                           const scalar_t *const SFEM_RESTRICT y,
                                           const scalar_t *const SFEM_RESTRICT z,
                                           const scalar_t                      qx,
                                           const scalar_t                      qy,
                                           const scalar_t                      qz,
                                           scalar_t *const SFEM_RESTRICT       adj,
                                           scalar_t *const SFEM_RESTRICT       det) {
    cvfem_hex8_adjugate_and_det(x, y, z, qx, qy, qz, adj, det);
}

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_affine_adj(const scalar_t *const SFEM_RESTRICT x,
                                              const scalar_t *const SFEM_RESTRICT y,
                                              const scalar_t *const SFEM_RESTRICT z,
                                              scalar_t *const SFEM_RESTRICT       adj,
                                              scalar_t *const SFEM_RESTRICT       det) {
    cvfem_hex8_geom_at(x, y, z, scalar_t(0.5), scalar_t(0.5), scalar_t(0.5), adj, det);
}

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_grad_at(const scalar_t *const SFEM_RESTRICT adj, const scalar_t det,
                                           const scalar_t                      dN[CVFEM_HEX8_N_NODES][3],
                                           const scalar_t *const SFEM_RESTRICT ux,
                                           const scalar_t *const SFEM_RESTRICT uy,
                                           const scalar_t *const SFEM_RESTRICT uz,
                                           scalar_t *const SFEM_RESTRICT       grad) {
    scalar_t ur[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
        ur[0] += ux[a] * dN[a][0];
        ur[1] += ux[a] * dN[a][1];
        ur[2] += ux[a] * dN[a][2];
        ur[3] += uy[a] * dN[a][0];
        ur[4] += uy[a] * dN[a][1];
        ur[5] += uy[a] * dN[a][2];
        ur[6] += uz[a] * dN[a][0];
        ur[7] += uz[a] * dN[a][1];
        ur[8] += uz[a] * dN[a][2];
    }
    const scalar_t inv_det = scalar_t(1) / det;
    for (int c = 0; c < 3; ++c) {
        const scalar_t rx = ur[3 * c + 0];
        const scalar_t ry = ur[3 * c + 1];
        const scalar_t rz = ur[3 * c + 2];
        grad[3 * c + 0]   = (adj[0] * rx + adj[3] * ry + adj[6] * rz) * inv_det;
        grad[3 * c + 1]   = (adj[1] * rx + adj[4] * ry + adj[7] * rz) * inv_det;
        grad[3 * c + 2]   = (adj[2] * rx + adj[5] * ry + adj[8] * rz) * inv_det;
    }
}

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_dir_areas(const scalar_t *const SFEM_RESTRICT adj, scalar_t A[3][3]) {
    A[0][0] = scalar_t(0.25) * adj[0];
    A[0][1] = scalar_t(0.25) * adj[1];
    A[0][2] = scalar_t(0.25) * adj[2];
    A[1][0] = scalar_t(0.25) * adj[3];
    A[1][1] = scalar_t(0.25) * adj[4];
    A[1][2] = scalar_t(0.25) * adj[5];
    A[2][0] = scalar_t(0.25) * adj[6];
    A[2][1] = scalar_t(0.25) * adj[7];
    A[2][2] = scalar_t(0.25) * adj[8];
}

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_traction(const scalar_t mu,
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

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_grad_sumfact(const scalar_t *const SFEM_RESTRICT adj, const scalar_t det,
                                                const scalar_t *const SFEM_RESTRICT ux,
                                                const scalar_t *const SFEM_RESTRICT uy,
                                                const scalar_t *const SFEM_RESTRICT uz,
                                                scalar_t *const SFEM_RESTRICT       grad) {
    const scalar_t inv_det = scalar_t(1) / det;
    scalar_t       dr, ds, dt;
    cvfem_hex8_face_diff(ux, dr, ds, dt);
    cvfem_hex8_pushforward(adj, inv_det, dr, ds, dt, grad[0], grad[1], grad[2]);
    cvfem_hex8_face_diff(uy, dr, ds, dt);
    cvfem_hex8_pushforward(adj, inv_det, dr, ds, dt, grad[3], grad[4], grad[5]);
    cvfem_hex8_face_diff(uz, dr, ds, dt);
    cvfem_hex8_pushforward(adj, inv_det, dr, ds, dt, grad[6], grad[7], grad[8]);
}

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_grad(const scalar_t *const SFEM_RESTRICT adj, const scalar_t det,
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

    const scalar_t inv_det = scalar_t(1) / det;
    for (int c = 0; c < 3; ++c) {
        const scalar_t rx = ur[3 * c + 0];
        const scalar_t ry = ur[3 * c + 1];
        const scalar_t rz = ur[3 * c + 2];
        grad[3 * c + 0]   = (adj[0] * rx + adj[3] * ry + adj[6] * rz) * inv_det;
        grad[3 * c + 1]   = (adj[1] * rx + adj[4] * ry + adj[7] * rz) * inv_det;
        grad[3 * c + 2]   = (adj[2] * rx + adj[5] * ry + adj[8] * rz) * inv_det;
    }
}

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE int cvfem_hex8_rhie_chow_active(const Hex8RhieChowT<scalar_t> &rc) {
    return rc.scale != scalar_t(0) && rc.x && rc.y && rc.z && rc.pgx && rc.pgy && rc.pgz;
}

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE scalar_t cvfem_hex8_rhie_chow_mdot_coeff(const scalar_t rho, const scalar_t mu, const scalar_t rc_scale,
                                                            const scalar_t dx, const scalar_t dy, const scalar_t dz,
                                                            const scalar_t ax, const scalar_t ay, const scalar_t az) {
    if (rc_scale == scalar_t(0) || rho == scalar_t(0)) return scalar_t(0);
    const scalar_t h2    = dx * dx + dy * dy + dz * dz;
    const scalar_t Adotd = ax * dx + ay * dy + az * dz;
    const scalar_t A2    = ax * ax + ay * ay + az * az;
    const scalar_t lim   = scalar_t(1e-30) * (std::sqrt(A2 * h2) + scalar_t(1e-30));
    if (std::fabs(Adotd) < lim) return scalar_t(0);
    const scalar_t Df = rc_scale * h2 / (scalar_t(2) * (mu > scalar_t(1e-30) ? mu : scalar_t(1e-30)));
    return rho * Df * A2 / Adotd;
}

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE scalar_t cvfem_hex8_rhie_chow_mdotc(const scalar_t rho, const scalar_t mu, const Hex8RhieChowT<scalar_t> &rc, const int i,
                                                       const int j, const scalar_t ax, const scalar_t ay, const scalar_t az,
                                                       const scalar_t p_i, const scalar_t p_j) {
    if (!cvfem_hex8_rhie_chow_active(rc)) return scalar_t(0);
    const scalar_t dx    = rc.x[j] - rc.x[i];
    const scalar_t dy    = rc.y[j] - rc.y[i];
    const scalar_t dz    = rc.z[j] - rc.z[i];
    const scalar_t coeff = cvfem_hex8_rhie_chow_mdot_coeff(rho, mu, rc.scale, dx, dy, dz, ax, ay, az);
    const scalar_t half  = scalar_t(0.5);
    const scalar_t corr  = (p_j - p_i) - (half * (rc.pgx[i] + rc.pgx[j]) * dx + half * (rc.pgy[i] + rc.pgy[j]) * dy +
                                         half * (rc.pgz[i] + rc.pgz[j]) * dz);
    return -coeff * corr;
}

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE scalar_t cvfem_hex8_rhie_chow_dmdotc(const scalar_t rho, const scalar_t mu, const Hex8RhieChowT<scalar_t> &rc, const int i,
                                                        const int j, const scalar_t ax, const scalar_t ay, const scalar_t az,
                                                        const scalar_t q_i, const scalar_t q_j) {
    if (!cvfem_hex8_rhie_chow_active(rc)) return scalar_t(0);
    const scalar_t dx    = rc.x[j] - rc.x[i];
    const scalar_t dy    = rc.y[j] - rc.y[i];
    const scalar_t dz    = rc.z[j] - rc.z[i];
    const scalar_t coeff = cvfem_hex8_rhie_chow_mdot_coeff(rho, mu, rc.scale, dx, dy, dz, ax, ay, az);
    return coeff * (q_i - q_j);
}

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_scs_convection(const scalar_t rho,
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
                                                  scalar_t      &mdot,
                                                  const scalar_t mdot_rc = scalar_t(0)) {
    const scalar_t adv_x = scalar_t(0.5) * (ux_i + ux_j);
    const scalar_t adv_y = scalar_t(0.5) * (uy_i + uy_j);
    const scalar_t adv_z = scalar_t(0.5) * (uz_i + uz_j);
    mdot                 = rho * (adv_x * ax + adv_y * ay + adv_z * az) + mdot_rc;
    const scalar_t sgn   = mdot > scalar_t(0) ? scalar_t(1) : (mdot < scalar_t(0) ? scalar_t(-1) : scalar_t(0));
    const scalar_t mpos  = scalar_t(0.5) * (mdot + sgn * mdot);
    const scalar_t mneg  = scalar_t(0.5) * (mdot - sgn * mdot);
    const scalar_t pmid  = scalar_t(0.5) * (p_i + p_j);
    fx                   = mpos * ux_i + mneg * ux_j + pmid * ax;
    fy                   = mpos * uy_i + mneg * uy_j + pmid * ay;
    fz                   = mpos * uz_i + mneg * uz_j + pmid * az;
}

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_ns_upwind_residual(const scalar_t                        rho,
                                                      const scalar_t                        mu,
                                                      const scalar_t *const SFEM_RESTRICT adj, const scalar_t det,
                                                      const scalar_t *const SFEM_RESTRICT   ux,
                                                      const scalar_t *const SFEM_RESTRICT   uy,
                                                      const scalar_t *const SFEM_RESTRICT   uz,
                                                      const scalar_t *const SFEM_RESTRICT   p,
                                                      scalar_t *const SFEM_RESTRICT         r) {
    for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i) r[i] = scalar_t(0);

    scalar_t grad[9];
    cvfem_hex8_grad(adj, det, ux, uy, uz, grad);

    for (int s = 0; s < CVFEM_HEX8_N_SCS; ++s) {
        const int i = CVFEM_HEX8_SCS[s].i;
        const int j = CVFEM_HEX8_SCS[s].j;

        scalar_t ax, ay, az;
        cvfem_hex8_area_dir(adj, s >> 2, ax, ay, az);

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

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_ns_upwind_residual_sumfact(const scalar_t                        rho,
                                                              const scalar_t                        mu,
                                                              const scalar_t *const SFEM_RESTRICT adj, const scalar_t det,
                                                              const scalar_t *const SFEM_RESTRICT   ux,
                                                              const scalar_t *const SFEM_RESTRICT   uy,
                                                              const scalar_t *const SFEM_RESTRICT   uz,
                                                              const scalar_t *const SFEM_RESTRICT   p,
                                                              scalar_t *const SFEM_RESTRICT         r,
                                                              const Hex8RhieChowT<scalar_t>        &rc = {}) {
    for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i) r[i] = scalar_t(0);

    scalar_t grad[9];
    cvfem_hex8_grad_sumfact(adj, det, ux, uy, uz, grad);

    scalar_t A[3][3];
    cvfem_hex8_dir_areas(adj, A);

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
        const scalar_t mdot_rc =
                cvfem_hex8_rhie_chow_mdotc(rho, mu, rc, i, j, A[d][0], A[d][1], A[d][2], p[i], p[j]);
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
                                  mdot,
                                  mdot_rc);
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

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_ns_upwind_jacobian_action(const scalar_t                        rho,
                                                             const scalar_t                        mu,
                                                             const scalar_t *const SFEM_RESTRICT adj, const scalar_t det,
                                                             const scalar_t *const SFEM_RESTRICT   ux,
                                                             const scalar_t *const SFEM_RESTRICT   uy,
                                                             const scalar_t *const SFEM_RESTRICT   uz,
                                                             const scalar_t *const SFEM_RESTRICT   vx,
                                                             const scalar_t *const SFEM_RESTRICT   vy,
                                                             const scalar_t *const SFEM_RESTRICT   vz,
                                                             const scalar_t *const SFEM_RESTRICT   q,
                                                             scalar_t *const SFEM_RESTRICT         r,
                                                             const Hex8RhieChowT<scalar_t>        &rc = {},
                                                             const scalar_t *const SFEM_RESTRICT   p  = nullptr) {
    for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i) r[i] = scalar_t(0);

    scalar_t dgrad[9];
    cvfem_hex8_grad_sumfact(adj, det, vx, vy, vz, dgrad);

    scalar_t A[3][3];
    cvfem_hex8_dir_areas(adj, A);

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
        const scalar_t mdot_rc = p ? cvfem_hex8_rhie_chow_mdotc(rho, mu, rc, i, j, ax, ay, az, p[i], p[j]) : scalar_t(0);
        const scalar_t mdot    = rho * (adv_x * ax + adv_y * ay + adv_z * az) + mdot_rc;
        const scalar_t sgn   = mdot > scalar_t(0) ? one : (mdot < scalar_t(0) ? -one : scalar_t(0));
        const scalar_t mpos  = half * (mdot + sgn * mdot);
        const scalar_t mneg  = half * (mdot - sgn * mdot);
        const scalar_t d_pos = half * (one + sgn);
        const scalar_t d_neg = half * (one - sgn);
        const scalar_t dmdot =
                rho * half * ((vx[i] + vx[j]) * ax + (vy[i] + vy[j]) * ay + (vz[i] + vz[j]) * az) +
                cvfem_hex8_rhie_chow_dmdotc(rho, mu, rc, i, j, ax, ay, az, q[i], q[j]);
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

template <bool Atomic, typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_acc(scalar_t &x, const scalar_t v) {
    if constexpr (Atomic) {
        CVFEM_ATOMIC_ADD(x, v);
    } else {
        x += v;
    }
}

template <bool Atomic, typename Slot, typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_bsr_acc(scalar_t *const SFEM_RESTRICT values,
                                           const Slot                   slot,
                                           const int                    rf,
                                           const int                    cf,
                                           const scalar_t               v) {
    cvfem_hex8_acc<Atomic>(values[(ptrdiff_t)slot * 16 + rf * 4 + cf], v);
}

template <bool Atomic, typename Slot, typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_bsr_acc_mom(scalar_t *const SFEM_RESTRICT values,
                                               const Slot                   slot,
                                               const scalar_t               d00,
                                               const scalar_t               d01,
                                               const scalar_t               d02,
                                               const scalar_t               d10,
                                               const scalar_t               d11,
                                               const scalar_t               d12,
                                               const scalar_t               d20,
                                               const scalar_t               d21,
                                               const scalar_t               d22) {
    scalar_t *const SFEM_RESTRICT blk = values + (ptrdiff_t)slot * 16;
    cvfem_hex8_acc<Atomic>(blk[0], d00);
    cvfem_hex8_acc<Atomic>(blk[1], d01);
    cvfem_hex8_acc<Atomic>(blk[2], d02);
    cvfem_hex8_acc<Atomic>(blk[4], d10);
    cvfem_hex8_acc<Atomic>(blk[5], d11);
    cvfem_hex8_acc<Atomic>(blk[6], d12);
    cvfem_hex8_acc<Atomic>(blk[8], d20);
    cvfem_hex8_acc<Atomic>(blk[9], d21);
    cvfem_hex8_acc<Atomic>(blk[10], d22);
}

template <bool Atomic, typename Slot, typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_jac_rhie_chow_p(const scalar_t                        rho,
                                                   const scalar_t                        mu,
                                                   const Hex8RhieChowT<scalar_t>        &rc,
                                                   const scalar_t                        ax,
                                                   const scalar_t                        ay,
                                                   const scalar_t                        az,
                                                   const int                             i,
                                                   const int                             j,
                                                   const scalar_t *const SFEM_RESTRICT   ux,
                                                   const scalar_t *const SFEM_RESTRICT   uy,
                                                   const scalar_t *const SFEM_RESTRICT   uz,
                                                   const scalar_t *const SFEM_RESTRICT   p,
                                                   const Slot *const SFEM_RESTRICT       slots,
                                                   scalar_t *const SFEM_RESTRICT         values) {
    if (!p || !cvfem_hex8_rhie_chow_active(rc)) return;
    const scalar_t dx    = rc.x[j] - rc.x[i];
    const scalar_t dy    = rc.y[j] - rc.y[i];
    const scalar_t dz    = rc.z[j] - rc.z[i];
    const scalar_t coeff = cvfem_hex8_rhie_chow_mdot_coeff(rho, mu, rc.scale, dx, dy, dz, ax, ay, az);
    if (coeff == scalar_t(0)) return;

    const scalar_t half     = scalar_t(0.5);
    const scalar_t mdot_avg = rho * half * ((ux[i] + ux[j]) * ax + (uy[i] + uy[j]) * ay + (uz[i] + uz[j]) * az);
    const scalar_t corr     = (p[j] - p[i]) - (half * (rc.pgx[i] + rc.pgx[j]) * dx + half * (rc.pgy[i] + rc.pgy[j]) * dy +
                                           half * (rc.pgz[i] + rc.pgz[j]) * dz);
    const scalar_t mdot     = mdot_avg - coeff * corr;
    const scalar_t sgn      = mdot > scalar_t(0) ? scalar_t(1) : (mdot < scalar_t(0) ? scalar_t(-1) : scalar_t(0));
    const scalar_t d_pos    = half * (scalar_t(1) + sgn);
    const scalar_t d_neg    = half * (scalar_t(1) - sgn);
    const scalar_t uup_x    = d_pos * ux[i] + d_neg * ux[j];
    const scalar_t uup_y    = d_pos * uy[i] + d_neg * uy[j];
    const scalar_t uup_z    = d_pos * uz[i] + d_neg * uz[j];
    const scalar_t dmdot_i  = coeff;
    const scalar_t dmdot_j  = -coeff;
    cvfem_hex8_bsr_acc<Atomic>(values, slots[i * 8 + i], 3, 3, dmdot_i);
    cvfem_hex8_bsr_acc<Atomic>(values, slots[i * 8 + j], 3, 3, dmdot_j);
    cvfem_hex8_bsr_acc<Atomic>(values, slots[j * 8 + i], 3, 3, -dmdot_i);
    cvfem_hex8_bsr_acc<Atomic>(values, slots[j * 8 + j], 3, 3, -dmdot_j);
    cvfem_hex8_bsr_acc<Atomic>(values, slots[i * 8 + i], 0, 3, uup_x * dmdot_i);
    cvfem_hex8_bsr_acc<Atomic>(values, slots[i * 8 + j], 0, 3, uup_x * dmdot_j);
    cvfem_hex8_bsr_acc<Atomic>(values, slots[j * 8 + i], 0, 3, -uup_x * dmdot_i);
    cvfem_hex8_bsr_acc<Atomic>(values, slots[j * 8 + j], 0, 3, -uup_x * dmdot_j);
    cvfem_hex8_bsr_acc<Atomic>(values, slots[i * 8 + i], 1, 3, uup_y * dmdot_i);
    cvfem_hex8_bsr_acc<Atomic>(values, slots[i * 8 + j], 1, 3, uup_y * dmdot_j);
    cvfem_hex8_bsr_acc<Atomic>(values, slots[j * 8 + i], 1, 3, -uup_y * dmdot_i);
    cvfem_hex8_bsr_acc<Atomic>(values, slots[j * 8 + j], 1, 3, -uup_y * dmdot_j);
    cvfem_hex8_bsr_acc<Atomic>(values, slots[i * 8 + i], 2, 3, uup_z * dmdot_i);
    cvfem_hex8_bsr_acc<Atomic>(values, slots[i * 8 + j], 2, 3, uup_z * dmdot_j);
    cvfem_hex8_bsr_acc<Atomic>(values, slots[j * 8 + i], 2, 3, -uup_z * dmdot_i);
    cvfem_hex8_bsr_acc<Atomic>(values, slots[j * 8 + j], 2, 3, -uup_z * dmdot_j);
}

template <bool Atomic, typename Slot, typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_ns_upwind_jacobian_add_rhie_chow(const scalar_t                        rho,
                                                                    const scalar_t                        mu,
                                                                    const scalar_t *const SFEM_RESTRICT   adj,
                                                                    const Hex8RhieChowT<scalar_t>        &rc,
                                                                    const scalar_t *const SFEM_RESTRICT   ux,
                                                                    const scalar_t *const SFEM_RESTRICT   uy,
                                                                    const scalar_t *const SFEM_RESTRICT   uz,
                                                                    const scalar_t *const SFEM_RESTRICT   p,
                                                                    const Slot *const SFEM_RESTRICT       slots,
                                                                    scalar_t *const SFEM_RESTRICT         values) {
    if (!p || !cvfem_hex8_rhie_chow_active(rc)) return;
    scalar_t A[3][3];
    cvfem_hex8_dir_areas(adj, A);
    for (int s = 0; s < CVFEM_HEX8_N_SCS; ++s) {
        const int d = s >> 2;
        cvfem_hex8_jac_rhie_chow_p<Atomic>(rho,
                                           mu,
                                           rc,
                                           A[d][0],
                                           A[d][1],
                                           A[d][2],
                                           CVFEM_HEX8_SCS[s].i,
                                           CVFEM_HEX8_SCS[s].j,
                                           ux,
                                           uy,
                                           uz,
                                           p,
                                           slots,
                                           values);
    }
}

template <bool Atomic, typename Slot, typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_jac_conv_face(const scalar_t                        rho,
                                                 const scalar_t                        ax,
                                                 const scalar_t                        ay,
                                                 const scalar_t                        az,
                                                 const int                             i,
                                                 const int                             j,
                                                 const scalar_t *const SFEM_RESTRICT   ux,
                                                 const scalar_t *const SFEM_RESTRICT   uy,
                                                 const scalar_t *const SFEM_RESTRICT   uz,
                                                 const Slot *const SFEM_RESTRICT       slots,
                                                 scalar_t *const SFEM_RESTRICT         values,
                                                 const scalar_t                        mdot_rc = scalar_t(0));

template <bool Atomic, typename Slot, typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_ns_upwind_jacobian_add_slots(const scalar_t                        rho,
                                                                const scalar_t                        mu,
                                                                const scalar_t *const SFEM_RESTRICT adj, const scalar_t det,
                                                                const scalar_t *const SFEM_RESTRICT   ux,
                                                                const scalar_t *const SFEM_RESTRICT   uy,
                                                                const scalar_t *const SFEM_RESTRICT   uz,
                                                                const Slot *const SFEM_RESTRICT       slots,
                                                                scalar_t *const SFEM_RESTRICT         values,
                                                                const Hex8RhieChowT<scalar_t>        &rc = {},
                                                                const scalar_t *const SFEM_RESTRICT   p  = nullptr) {
    scalar_t A[3][3];
    cvfem_hex8_dir_areas(adj, A);

    scalar_t w[CVFEM_HEX8_N_NODES][3];
    const scalar_t inv_det = scalar_t(1) / det;
    for (int k = 0; k < CVFEM_HEX8_N_NODES; ++k) {
        cvfem_hex8_pushforward<scalar_t>(adj,
                               inv_det,
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
            cvfem_hex8_bsr_acc_mom<Atomic>(values, slot, d00, d01, d02, d10, d11, d12, d20, d21, d22);
        }
    }

    for (int s = 0; s < CVFEM_HEX8_N_SCS; ++s) {
        const int      d       = s >> 2;
        const int      i       = CVFEM_HEX8_SCS[s].i;
        const int      j       = CVFEM_HEX8_SCS[s].j;
        const scalar_t mdot_rc = p ? cvfem_hex8_rhie_chow_mdotc(rho, mu, rc, i, j, A[d][0], A[d][1], A[d][2], p[i], p[j])
                                   : scalar_t(0);
        cvfem_hex8_jac_conv_face<Atomic>(rho, A[d][0], A[d][1], A[d][2], i, j, ux, uy, uz, slots, values, mdot_rc);
        cvfem_hex8_jac_rhie_chow_p<Atomic>(rho, mu, rc, A[d][0], A[d][1], A[d][2], i, j, ux, uy, uz, p, slots, values);
    }
}

static SFEM_INLINE void cvfem_hex8_zero_residual_pack(Hex8ResidualPack &out) {
    std::memset(&out, 0, sizeof(out));
}

template <int I0, int J0, int I1, int J1, int I2, int J2, int I3, int J3>
static SFEM_INLINE void cvfem_hex8_visc_dir_simd(const scalar_t                      mu,
                                                 const scalar_t *const SFEM_RESTRICT g00,
                                                 const scalar_t *const SFEM_RESTRICT g01,
                                                 const scalar_t *const SFEM_RESTRICT g02,
                                                 const scalar_t *const SFEM_RESTRICT g10,
                                                 const scalar_t *const SFEM_RESTRICT g11,
                                                 const scalar_t *const SFEM_RESTRICT g12,
                                                 const scalar_t *const SFEM_RESTRICT g20,
                                                 const scalar_t *const SFEM_RESTRICT g21,
                                                 const scalar_t *const SFEM_RESTRICT g22,
                                                 const scalar_t *const SFEM_RESTRICT Ax,
                                                 const scalar_t *const SFEM_RESTRICT Ay,
                                                 const scalar_t *const SFEM_RESTRICT Az,
                                                 Hex8ResidualPack                   &out) {
#pragma omp simd
    for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
        const scalar_t ax = Ax[lane];
        const scalar_t ay = Ay[lane];
        const scalar_t az = Az[lane];
        const scalar_t tx = mu * ((scalar_t(2) * g00[lane]) * ax + (g01[lane] + g10[lane]) * ay +
                                  (g02[lane] + g20[lane]) * az);
        const scalar_t ty = mu * ((g10[lane] + g01[lane]) * ax + (scalar_t(2) * g11[lane]) * ay +
                                  (g12[lane] + g21[lane]) * az);
        const scalar_t tz = mu * ((g20[lane] + g02[lane]) * ax + (g21[lane] + g12[lane]) * ay +
                                  (scalar_t(2) * g22[lane]) * az);
        out.rx[I0][lane] -= tx;
        out.ry[I0][lane] -= ty;
        out.rz[I0][lane] -= tz;
        out.rx[J0][lane] += tx;
        out.ry[J0][lane] += ty;
        out.rz[J0][lane] += tz;
        out.rx[I1][lane] -= tx;
        out.ry[I1][lane] -= ty;
        out.rz[I1][lane] -= tz;
        out.rx[J1][lane] += tx;
        out.ry[J1][lane] += ty;
        out.rz[J1][lane] += tz;
        out.rx[I2][lane] -= tx;
        out.ry[I2][lane] -= ty;
        out.rz[I2][lane] -= tz;
        out.rx[J2][lane] += tx;
        out.ry[J2][lane] += ty;
        out.rz[J2][lane] += tz;
        out.rx[I3][lane] -= tx;
        out.ry[I3][lane] -= ty;
        out.rz[I3][lane] -= tz;
        out.rx[J3][lane] += tx;
        out.ry[J3][lane] += ty;
        out.rz[J3][lane] += tz;
    }
}

template <int I, int J, bool RC = false>
static SFEM_INLINE void cvfem_hex8_conv_face_simd(const scalar_t                      rho,
                                                  const scalar_t                      mu,
                                                  const scalar_t                      rc_scale,
                                                  const scalar_t                      half,
                                                  const scalar_t *const SFEM_RESTRICT Ax,
                                                  const scalar_t *const SFEM_RESTRICT Ay,
                                                  const scalar_t *const SFEM_RESTRICT Az,
                                                  const Hex8InputPack                &in,
                                                  const Hex8RhieChowPack             *rc,
                                                  Hex8ResidualPack                   &out) {
    if constexpr (!RC) {
        (void)mu;
        (void)rc_scale;
        (void)rc;
    }
#pragma omp simd
    for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
        const scalar_t ax    = Ax[lane];
        const scalar_t ay    = Ay[lane];
        const scalar_t az    = Az[lane];
        const scalar_t adv_x = half * (in.ux[I][lane] + in.ux[J][lane]);
        const scalar_t adv_y = half * (in.uy[I][lane] + in.uy[J][lane]);
        const scalar_t adv_z = half * (in.uz[I][lane] + in.uz[J][lane]);
        scalar_t       mdot  = rho * (adv_x * ax + adv_y * ay + adv_z * az);
        if constexpr (RC) {
            const scalar_t dx    = rc->x[J][lane] - rc->x[I][lane];
            const scalar_t dy    = rc->y[J][lane] - rc->y[I][lane];
            const scalar_t dz    = rc->z[J][lane] - rc->z[I][lane];
            const scalar_t coeff = cvfem_hex8_rhie_chow_mdot_coeff(rho, mu, rc_scale, dx, dy, dz, ax, ay, az);
            const scalar_t corr =
                    (in.p[J][lane] - in.p[I][lane]) -
                    (half * (rc->pgx[I][lane] + rc->pgx[J][lane]) * dx +
                     half * (rc->pgy[I][lane] + rc->pgy[J][lane]) * dy +
                     half * (rc->pgz[I][lane] + rc->pgz[J][lane]) * dz);
            mdot -= coeff * corr;
        }
        const scalar_t sgn  = mdot > scalar_t(0) ? scalar_t(1) : (mdot < scalar_t(0) ? scalar_t(-1) : scalar_t(0));
        const scalar_t mpos = half * (mdot + sgn * mdot);
        const scalar_t mneg = half * (mdot - sgn * mdot);
        const scalar_t pmid = half * (in.p[I][lane] + in.p[J][lane]);
        const scalar_t fx   = mpos * in.ux[I][lane] + mneg * in.ux[J][lane] + pmid * ax;
        const scalar_t fy   = mpos * in.uy[I][lane] + mneg * in.uy[J][lane] + pmid * ay;
        const scalar_t fz   = mpos * in.uz[I][lane] + mneg * in.uz[J][lane] + pmid * az;
        out.rx[I][lane] += fx;
        out.ry[I][lane] += fy;
        out.rz[I][lane] += fz;
        out.rc[I][lane] += mdot;
        out.rx[J][lane] -= fx;
        out.ry[J][lane] -= fy;
        out.rz[J][lane] -= fz;
        out.rc[J][lane] -= mdot;
    }
}

template <int I, int J, bool RC = false>
static SFEM_INLINE void cvfem_hex8_conv_face_jv_simd(const scalar_t                      rho,
                                                     const scalar_t                      mu,
                                                     const scalar_t                      rc_scale,
                                                     const scalar_t                      half,
                                                     const scalar_t                      one,
                                                     const scalar_t *const SFEM_RESTRICT Ax,
                                                     const scalar_t *const SFEM_RESTRICT Ay,
                                                     const scalar_t *const SFEM_RESTRICT Az,
                                                     const Hex8InputPack                &u,
                                                     const Hex8InputPack                &du,
                                                     const Hex8RhieChowPack             *rc,
                                                     Hex8ResidualPack                   &out) {
    if constexpr (!RC) {
        (void)mu;
        (void)rc_scale;
        (void)rc;
    }
#pragma omp simd
    for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
        const scalar_t ax    = Ax[lane];
        const scalar_t ay    = Ay[lane];
        const scalar_t az    = Az[lane];
        const scalar_t adv_x = half * (u.ux[I][lane] + u.ux[J][lane]);
        const scalar_t adv_y = half * (u.uy[I][lane] + u.uy[J][lane]);
        const scalar_t adv_z = half * (u.uz[I][lane] + u.uz[J][lane]);
        scalar_t       mdot  = rho * (adv_x * ax + adv_y * ay + adv_z * az);
        scalar_t       dmdot = rho * half *
                         ((du.ux[I][lane] + du.ux[J][lane]) * ax + (du.uy[I][lane] + du.uy[J][lane]) * ay +
                          (du.uz[I][lane] + du.uz[J][lane]) * az);
        if constexpr (RC) {
            const scalar_t dx    = rc->x[J][lane] - rc->x[I][lane];
            const scalar_t dy    = rc->y[J][lane] - rc->y[I][lane];
            const scalar_t dz    = rc->z[J][lane] - rc->z[I][lane];
            const scalar_t coeff = cvfem_hex8_rhie_chow_mdot_coeff(rho, mu, rc_scale, dx, dy, dz, ax, ay, az);
            const scalar_t corr =
                    (u.p[J][lane] - u.p[I][lane]) -
                    (half * (rc->pgx[I][lane] + rc->pgx[J][lane]) * dx +
                     half * (rc->pgy[I][lane] + rc->pgy[J][lane]) * dy +
                     half * (rc->pgz[I][lane] + rc->pgz[J][lane]) * dz);
            mdot -= coeff * corr;
            dmdot += coeff * (du.p[I][lane] - du.p[J][lane]);
        }
        const scalar_t sgn   = mdot > scalar_t(0) ? one : (mdot < scalar_t(0) ? -one : scalar_t(0));
        const scalar_t mpos  = half * (mdot + sgn * mdot);
        const scalar_t mneg  = half * (mdot - sgn * mdot);
        const scalar_t d_pos = half * (one + sgn);
        const scalar_t d_neg = half * (one - sgn);
        const scalar_t dpos  = d_pos * dmdot;
        const scalar_t dneg  = d_neg * dmdot;
        const scalar_t qmid  = half * (du.p[I][lane] + du.p[J][lane]);
        const scalar_t fx =
                dpos * u.ux[I][lane] + mpos * du.ux[I][lane] + dneg * u.ux[J][lane] + mneg * du.ux[J][lane] + qmid * ax;
        const scalar_t fy =
                dpos * u.uy[I][lane] + mpos * du.uy[I][lane] + dneg * u.uy[J][lane] + mneg * du.uy[J][lane] + qmid * ay;
        const scalar_t fz =
                dpos * u.uz[I][lane] + mpos * du.uz[I][lane] + dneg * u.uz[J][lane] + mneg * du.uz[J][lane] + qmid * az;
        out.rx[I][lane] += fx;
        out.ry[I][lane] += fy;
        out.rz[I][lane] += fz;
        out.rc[I][lane] += dmdot;
        out.rx[J][lane] -= fx;
        out.ry[J][lane] -= fy;
        out.rz[J][lane] -= fz;
        out.rc[J][lane] -= dmdot;
    }
}

template <bool RC = false>
static SFEM_INLINE void cvfem_hex8_conv_all_simd(const scalar_t                      rho,
                                                 const scalar_t                      mu,
                                                 const scalar_t                      rc_scale,
                                                 const scalar_t                      half,
                                                 const scalar_t *const SFEM_RESTRICT Ax0,
                                                 const scalar_t *const SFEM_RESTRICT Ay0,
                                                 const scalar_t *const SFEM_RESTRICT Az0,
                                                 const scalar_t *const SFEM_RESTRICT Ax1,
                                                 const scalar_t *const SFEM_RESTRICT Ay1,
                                                 const scalar_t *const SFEM_RESTRICT Az1,
                                                 const scalar_t *const SFEM_RESTRICT Ax2,
                                                 const scalar_t *const SFEM_RESTRICT Ay2,
                                                 const scalar_t *const SFEM_RESTRICT Az2,
                                                 const Hex8InputPack                &in,
                                                 const Hex8RhieChowPack             *rc,
                                                 Hex8ResidualPack                   &out) {
    cvfem_hex8_conv_face_simd<0, 1, RC>(rho, mu, rc_scale, half, Ax0, Ay0, Az0, in, rc, out);
    cvfem_hex8_conv_face_simd<3, 2, RC>(rho, mu, rc_scale, half, Ax0, Ay0, Az0, in, rc, out);
    cvfem_hex8_conv_face_simd<4, 5, RC>(rho, mu, rc_scale, half, Ax0, Ay0, Az0, in, rc, out);
    cvfem_hex8_conv_face_simd<7, 6, RC>(rho, mu, rc_scale, half, Ax0, Ay0, Az0, in, rc, out);
    cvfem_hex8_conv_face_simd<0, 3, RC>(rho, mu, rc_scale, half, Ax1, Ay1, Az1, in, rc, out);
    cvfem_hex8_conv_face_simd<1, 2, RC>(rho, mu, rc_scale, half, Ax1, Ay1, Az1, in, rc, out);
    cvfem_hex8_conv_face_simd<4, 7, RC>(rho, mu, rc_scale, half, Ax1, Ay1, Az1, in, rc, out);
    cvfem_hex8_conv_face_simd<5, 6, RC>(rho, mu, rc_scale, half, Ax1, Ay1, Az1, in, rc, out);
    cvfem_hex8_conv_face_simd<0, 4, RC>(rho, mu, rc_scale, half, Ax2, Ay2, Az2, in, rc, out);
    cvfem_hex8_conv_face_simd<1, 5, RC>(rho, mu, rc_scale, half, Ax2, Ay2, Az2, in, rc, out);
    cvfem_hex8_conv_face_simd<2, 6, RC>(rho, mu, rc_scale, half, Ax2, Ay2, Az2, in, rc, out);
    cvfem_hex8_conv_face_simd<3, 7, RC>(rho, mu, rc_scale, half, Ax2, Ay2, Az2, in, rc, out);
}

static SFEM_INLINE void cvfem_hex8_conv_all_simd(const scalar_t                      rho,
                                                 const scalar_t                      half,
                                                 const scalar_t *const SFEM_RESTRICT Ax0,
                                                 const scalar_t *const SFEM_RESTRICT Ay0,
                                                 const scalar_t *const SFEM_RESTRICT Az0,
                                                 const scalar_t *const SFEM_RESTRICT Ax1,
                                                 const scalar_t *const SFEM_RESTRICT Ay1,
                                                 const scalar_t *const SFEM_RESTRICT Az1,
                                                 const scalar_t *const SFEM_RESTRICT Ax2,
                                                 const scalar_t *const SFEM_RESTRICT Ay2,
                                                 const scalar_t *const SFEM_RESTRICT Az2,
                                                 const Hex8InputPack                &in,
                                                 Hex8ResidualPack                   &out) {
    cvfem_hex8_conv_all_simd<false>(rho,
                                    scalar_t(0),
                                    scalar_t(0),
                                    half,
                                    Ax0,
                                    Ay0,
                                    Az0,
                                    Ax1,
                                    Ay1,
                                    Az1,
                                    Ax2,
                                    Ay2,
                                    Az2,
                                    in,
                                    nullptr,
                                    out);
}

template <bool RC = false>
static SFEM_INLINE void cvfem_hex8_conv_all_jv_simd(const scalar_t                      rho,
                                                    const scalar_t                      mu,
                                                    const scalar_t                      rc_scale,
                                                    const scalar_t                      half,
                                                    const scalar_t                      one,
                                                    const scalar_t *const SFEM_RESTRICT Ax0,
                                                    const scalar_t *const SFEM_RESTRICT Ay0,
                                                    const scalar_t *const SFEM_RESTRICT Az0,
                                                    const scalar_t *const SFEM_RESTRICT Ax1,
                                                    const scalar_t *const SFEM_RESTRICT Ay1,
                                                    const scalar_t *const SFEM_RESTRICT Az1,
                                                    const scalar_t *const SFEM_RESTRICT Ax2,
                                                    const scalar_t *const SFEM_RESTRICT Ay2,
                                                    const scalar_t *const SFEM_RESTRICT Az2,
                                                    const Hex8InputPack                &u,
                                                    const Hex8InputPack                &du,
                                                    const Hex8RhieChowPack             *rc,
                                                    Hex8ResidualPack                   &out) {
    cvfem_hex8_conv_face_jv_simd<0, 1, RC>(rho, mu, rc_scale, half, one, Ax0, Ay0, Az0, u, du, rc, out);
    cvfem_hex8_conv_face_jv_simd<3, 2, RC>(rho, mu, rc_scale, half, one, Ax0, Ay0, Az0, u, du, rc, out);
    cvfem_hex8_conv_face_jv_simd<4, 5, RC>(rho, mu, rc_scale, half, one, Ax0, Ay0, Az0, u, du, rc, out);
    cvfem_hex8_conv_face_jv_simd<7, 6, RC>(rho, mu, rc_scale, half, one, Ax0, Ay0, Az0, u, du, rc, out);
    cvfem_hex8_conv_face_jv_simd<0, 3, RC>(rho, mu, rc_scale, half, one, Ax1, Ay1, Az1, u, du, rc, out);
    cvfem_hex8_conv_face_jv_simd<1, 2, RC>(rho, mu, rc_scale, half, one, Ax1, Ay1, Az1, u, du, rc, out);
    cvfem_hex8_conv_face_jv_simd<4, 7, RC>(rho, mu, rc_scale, half, one, Ax1, Ay1, Az1, u, du, rc, out);
    cvfem_hex8_conv_face_jv_simd<5, 6, RC>(rho, mu, rc_scale, half, one, Ax1, Ay1, Az1, u, du, rc, out);
    cvfem_hex8_conv_face_jv_simd<0, 4, RC>(rho, mu, rc_scale, half, one, Ax2, Ay2, Az2, u, du, rc, out);
    cvfem_hex8_conv_face_jv_simd<1, 5, RC>(rho, mu, rc_scale, half, one, Ax2, Ay2, Az2, u, du, rc, out);
    cvfem_hex8_conv_face_jv_simd<2, 6, RC>(rho, mu, rc_scale, half, one, Ax2, Ay2, Az2, u, du, rc, out);
    cvfem_hex8_conv_face_jv_simd<3, 7, RC>(rho, mu, rc_scale, half, one, Ax2, Ay2, Az2, u, du, rc, out);
}

static SFEM_INLINE void cvfem_hex8_conv_all_jv_simd(const scalar_t                      rho,
                                                    const scalar_t                      half,
                                                    const scalar_t                      one,
                                                    const scalar_t *const SFEM_RESTRICT Ax0,
                                                    const scalar_t *const SFEM_RESTRICT Ay0,
                                                    const scalar_t *const SFEM_RESTRICT Az0,
                                                    const scalar_t *const SFEM_RESTRICT Ax1,
                                                    const scalar_t *const SFEM_RESTRICT Ay1,
                                                    const scalar_t *const SFEM_RESTRICT Az1,
                                                    const scalar_t *const SFEM_RESTRICT Ax2,
                                                    const scalar_t *const SFEM_RESTRICT Ay2,
                                                    const scalar_t *const SFEM_RESTRICT Az2,
                                                    const Hex8InputPack                &u,
                                                    const Hex8InputPack                &du,
                                                    Hex8ResidualPack                   &out) {
    cvfem_hex8_conv_all_jv_simd<false>(rho,
                                       scalar_t(0),
                                       scalar_t(0),
                                       half,
                                       one,
                                       Ax0,
                                       Ay0,
                                       Az0,
                                       Ax1,
                                       Ay1,
                                       Az1,
                                       Ax2,
                                       Ay2,
                                       Az2,
                                       u,
                                       du,
                                       nullptr,
                                       out);
}

template <bool Atomic, typename Slot, typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_jac_conv_face(const scalar_t                        rho,
                                                 const scalar_t                        ax,
                                                 const scalar_t                        ay,
                                                 const scalar_t                        az,
                                                 const int                             i,
                                                 const int                             j,
                                                 const scalar_t *const SFEM_RESTRICT   ux,
                                                 const scalar_t *const SFEM_RESTRICT   uy,
                                                 const scalar_t *const SFEM_RESTRICT   uz,
                                                 const Slot *const SFEM_RESTRICT       slots,
                                                 scalar_t *const SFEM_RESTRICT         values,
                                                 const scalar_t                        mdot_rc) {
    const scalar_t half  = scalar_t(0.5);
    const scalar_t one   = scalar_t(1);
    const scalar_t alpha = rho * half;
    const scalar_t adv_x = half * (ux[i] + ux[j]);
    const scalar_t adv_y = half * (uy[i] + uy[j]);
    const scalar_t adv_z = half * (uz[i] + uz[j]);
    const scalar_t mdot  = rho * (adv_x * ax + adv_y * ay + adv_z * az) + mdot_rc;
    const scalar_t sgn   = mdot > scalar_t(0) ? one : (mdot < scalar_t(0) ? -one : scalar_t(0));
    const scalar_t mpos  = half * (mdot + sgn * mdot);
    const scalar_t mneg  = half * (mdot - sgn * mdot);
    const scalar_t d_pos = half * (one + sgn);
    const scalar_t d_neg = half * (one - sgn);
    const scalar_t hax   = half * ax;
    const scalar_t hay   = half * ay;
    const scalar_t haz   = half * az;

    const int bnodes[2] = {i, j};
    const scalar_t mass[2] = {mpos, mneg};
    for (int nb = 0; nb < 2; ++nb) {
        const int      b  = bnodes[nb];
        const scalar_t m  = mass[nb];
        const Slot     si = slots[i * 8 + b];
        const Slot     sj = slots[j * 8 + b];

        {
            const scalar_t dmdot = alpha * ax;
            const scalar_t dpos  = d_pos * dmdot;
            const scalar_t dneg  = d_neg * dmdot;
            const scalar_t dfx   = dpos * ux[i] + dneg * ux[j] + m;
            const scalar_t dfy   = dpos * uy[i] + dneg * uy[j];
            const scalar_t dfz   = dpos * uz[i] + dneg * uz[j];
            cvfem_hex8_bsr_acc<Atomic>(values, si, 0, 0, dfx);
            cvfem_hex8_bsr_acc<Atomic>(values, si, 1, 0, dfy);
            cvfem_hex8_bsr_acc<Atomic>(values, si, 2, 0, dfz);
            cvfem_hex8_bsr_acc<Atomic>(values, si, 3, 0, dmdot);
            cvfem_hex8_bsr_acc<Atomic>(values, sj, 0, 0, -dfx);
            cvfem_hex8_bsr_acc<Atomic>(values, sj, 1, 0, -dfy);
            cvfem_hex8_bsr_acc<Atomic>(values, sj, 2, 0, -dfz);
            cvfem_hex8_bsr_acc<Atomic>(values, sj, 3, 0, -dmdot);
        }
        {
            const scalar_t dmdot = alpha * ay;
            const scalar_t dpos  = d_pos * dmdot;
            const scalar_t dneg  = d_neg * dmdot;
            const scalar_t dfx   = dpos * ux[i] + dneg * ux[j];
            const scalar_t dfy   = dpos * uy[i] + dneg * uy[j] + m;
            const scalar_t dfz   = dpos * uz[i] + dneg * uz[j];
            cvfem_hex8_bsr_acc<Atomic>(values, si, 0, 1, dfx);
            cvfem_hex8_bsr_acc<Atomic>(values, si, 1, 1, dfy);
            cvfem_hex8_bsr_acc<Atomic>(values, si, 2, 1, dfz);
            cvfem_hex8_bsr_acc<Atomic>(values, si, 3, 1, dmdot);
            cvfem_hex8_bsr_acc<Atomic>(values, sj, 0, 1, -dfx);
            cvfem_hex8_bsr_acc<Atomic>(values, sj, 1, 1, -dfy);
            cvfem_hex8_bsr_acc<Atomic>(values, sj, 2, 1, -dfz);
            cvfem_hex8_bsr_acc<Atomic>(values, sj, 3, 1, -dmdot);
        }
        {
            const scalar_t dmdot = alpha * az;
            const scalar_t dpos  = d_pos * dmdot;
            const scalar_t dneg  = d_neg * dmdot;
            const scalar_t dfx   = dpos * ux[i] + dneg * ux[j];
            const scalar_t dfy   = dpos * uy[i] + dneg * uy[j];
            const scalar_t dfz   = dpos * uz[i] + dneg * uz[j] + m;
            cvfem_hex8_bsr_acc<Atomic>(values, si, 0, 2, dfx);
            cvfem_hex8_bsr_acc<Atomic>(values, si, 1, 2, dfy);
            cvfem_hex8_bsr_acc<Atomic>(values, si, 2, 2, dfz);
            cvfem_hex8_bsr_acc<Atomic>(values, si, 3, 2, dmdot);
            cvfem_hex8_bsr_acc<Atomic>(values, sj, 0, 2, -dfx);
            cvfem_hex8_bsr_acc<Atomic>(values, sj, 1, 2, -dfy);
            cvfem_hex8_bsr_acc<Atomic>(values, sj, 2, 2, -dfz);
            cvfem_hex8_bsr_acc<Atomic>(values, sj, 3, 2, -dmdot);
        }
        cvfem_hex8_bsr_acc<Atomic>(values, si, 0, 3, hax);
        cvfem_hex8_bsr_acc<Atomic>(values, si, 1, 3, hay);
        cvfem_hex8_bsr_acc<Atomic>(values, si, 2, 3, haz);
        cvfem_hex8_bsr_acc<Atomic>(values, sj, 0, 3, -hax);
        cvfem_hex8_bsr_acc<Atomic>(values, sj, 1, 3, -hay);
        cvfem_hex8_bsr_acc<Atomic>(values, sj, 2, 3, -haz);
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
        Hex8ResidualPack                     &out,
        const Hex8RhieChowPack               *rc       = nullptr,
        const scalar_t                        rc_scale = scalar_t(0)) {
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
    }

    cvfem_hex8_zero_residual_pack(out);
    cvfem_hex8_visc_dir_simd<0, 1, 3, 2, 4, 5, 7, 6>(
            mu, g00v, g01v, g02v, g10v, g11v, g12v, g20v, g21v, g22v, Ax0, Ay0, Az0, out);
    cvfem_hex8_visc_dir_simd<0, 3, 1, 2, 4, 7, 5, 6>(
            mu, g00v, g01v, g02v, g10v, g11v, g12v, g20v, g21v, g22v, Ax1, Ay1, Az1, out);
    cvfem_hex8_visc_dir_simd<0, 4, 1, 5, 2, 6, 3, 7>(
            mu, g00v, g01v, g02v, g10v, g11v, g12v, g20v, g21v, g22v, Ax2, Ay2, Az2, out);

    if (rc && rc_scale != scalar_t(0)) {
        cvfem_hex8_conv_all_simd<true>(
                rho, mu, rc_scale, half, Ax0, Ay0, Az0, Ax1, Ay1, Az1, Ax2, Ay2, Az2, in, rc, out);
    } else {
        cvfem_hex8_conv_all_simd<false>(
                rho, mu, rc_scale, half, Ax0, Ay0, Az0, Ax1, Ay1, Az1, Ax2, Ay2, Az2, in, rc, out);
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
        Hex8ResidualPack                     &out,
        const Hex8RhieChowPack               *rc       = nullptr,
        const scalar_t                        rc_scale = scalar_t(0)) {
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
    }

    cvfem_hex8_zero_residual_pack(out);
    cvfem_hex8_visc_dir_simd<0, 1, 3, 2, 4, 5, 7, 6>(
            mu, g00v, g01v, g02v, g10v, g11v, g12v, g20v, g21v, g22v, Ax0, Ay0, Az0, out);
    cvfem_hex8_visc_dir_simd<0, 3, 1, 2, 4, 7, 5, 6>(
            mu, g00v, g01v, g02v, g10v, g11v, g12v, g20v, g21v, g22v, Ax1, Ay1, Az1, out);
    cvfem_hex8_visc_dir_simd<0, 4, 1, 5, 2, 6, 3, 7>(
            mu, g00v, g01v, g02v, g10v, g11v, g12v, g20v, g21v, g22v, Ax2, Ay2, Az2, out);

    if (rc && rc_scale != scalar_t(0)) {
        cvfem_hex8_conv_all_jv_simd<true>(
                rho, mu, rc_scale, half, one, Ax0, Ay0, Az0, Ax1, Ay1, Az1, Ax2, Ay2, Az2, u, du, rc, out);
    } else {
        cvfem_hex8_conv_all_jv_simd<false>(
                rho, mu, rc_scale, half, one, Ax0, Ay0, Az0, Ax1, Ay1, Az1, Ax2, Ay2, Az2, u, du, rc, out);
    }
}

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_ns_upwind_residual_isoparam(const scalar_t                        rho,
                                                              const scalar_t                        mu,
                                                              const scalar_t *const SFEM_RESTRICT   x,
                                                              const scalar_t *const SFEM_RESTRICT   y,
                                                              const scalar_t *const SFEM_RESTRICT   z,
                                                              const scalar_t *const SFEM_RESTRICT   ux,
                                                              const scalar_t *const SFEM_RESTRICT   uy,
                                                              const scalar_t *const SFEM_RESTRICT   uz,
                                                              const scalar_t *const SFEM_RESTRICT   p,
                                                              scalar_t *const SFEM_RESTRICT         r,
                                                              const Hex8RhieChowT<scalar_t>        &rc = {}) {
    for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i) r[i] = scalar_t(0);

    for (int s = 0; s < CVFEM_HEX8_N_SCS; ++s) {
        scalar_t dN[CVFEM_HEX8_N_NODES][3];
        cvfem_hex8_dn_ref<scalar_t>(CVFEM_HEX8_SCS_XI[s][0], CVFEM_HEX8_SCS_XI[s][1], CVFEM_HEX8_SCS_XI[s][2], dN);

        scalar_t adj[9], det;
        cvfem_hex8_geom_at<scalar_t>(x, y, z, CVFEM_HEX8_SCS_XI[s][0], CVFEM_HEX8_SCS_XI[s][1], CVFEM_HEX8_SCS_XI[s][2], adj, &det);

        scalar_t ax, ay, az;
        cvfem_hex8_area_dir(adj, s >> 2, ax, ay, az);

        scalar_t grad[9];
        cvfem_hex8_grad_at(adj, det, dN, ux, uy, uz, grad);

        const int i = CVFEM_HEX8_SCS[s].i;
        const int j = CVFEM_HEX8_SCS[s].j;

        scalar_t tau_x, tau_y, tau_z;
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
                            ax,
                            ay,
                            az,
                            tau_x,
                            tau_y,
                            tau_z);

        scalar_t fx, fy, fz, mdot;
        const scalar_t mdot_rc = cvfem_hex8_rhie_chow_mdotc(rho, mu, rc, i, j, ax, ay, az, p[i], p[j]);
        cvfem_hex8_scs_convection(rho, ux[i], ux[j], uy[i], uy[j], uz[i], uz[j], p[i], p[j], ax, ay, az, fx, fy, fz, mdot,
                                  mdot_rc);
        fx -= tau_x;
        fy -= tau_y;
        fz -= tau_z;

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

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_ns_upwind_jacobian_action_isoparam(const scalar_t                        rho,
                                                                     const scalar_t                        mu,
                                                                     const scalar_t *const SFEM_RESTRICT   x,
                                                                     const scalar_t *const SFEM_RESTRICT   y,
                                                                     const scalar_t *const SFEM_RESTRICT   z,
                                                                     const scalar_t *const SFEM_RESTRICT   ux,
                                                                     const scalar_t *const SFEM_RESTRICT   uy,
                                                                     const scalar_t *const SFEM_RESTRICT   uz,
                                                                     const scalar_t *const SFEM_RESTRICT   vx,
                                                                     const scalar_t *const SFEM_RESTRICT   vy,
                                                                     const scalar_t *const SFEM_RESTRICT   vz,
                                                                     const scalar_t *const SFEM_RESTRICT   q,
                                                                     scalar_t *const SFEM_RESTRICT         r,
                                                                     const Hex8RhieChowT<scalar_t>        &rc = {},
                                                                     const scalar_t *const SFEM_RESTRICT   p  = nullptr) {
    for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i) r[i] = scalar_t(0);

    const scalar_t half = scalar_t(0.5);
    const scalar_t one  = scalar_t(1);
    for (int s = 0; s < CVFEM_HEX8_N_SCS; ++s) {
        scalar_t dN[CVFEM_HEX8_N_NODES][3];
        cvfem_hex8_dn_ref<scalar_t>(CVFEM_HEX8_SCS_XI[s][0], CVFEM_HEX8_SCS_XI[s][1], CVFEM_HEX8_SCS_XI[s][2], dN);

        scalar_t adj[9], det;
        cvfem_hex8_geom_at<scalar_t>(x, y, z, CVFEM_HEX8_SCS_XI[s][0], CVFEM_HEX8_SCS_XI[s][1], CVFEM_HEX8_SCS_XI[s][2], adj, &det);

        scalar_t ax, ay, az;
        cvfem_hex8_area_dir(adj, s >> 2, ax, ay, az);

        scalar_t dgrad[9];
        cvfem_hex8_grad_at(adj, det, dN, vx, vy, vz, dgrad);

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
                            ax,
                            ay,
                            az,
                            tx,
                            ty,
                            tz);

        const int i = CVFEM_HEX8_SCS[s].i;
        const int j = CVFEM_HEX8_SCS[s].j;
        r[i * 4 + 0] -= tx;
        r[i * 4 + 1] -= ty;
        r[i * 4 + 2] -= tz;
        r[j * 4 + 0] += tx;
        r[j * 4 + 1] += ty;
        r[j * 4 + 2] += tz;

        const scalar_t adv_x = half * (ux[i] + ux[j]);
        const scalar_t adv_y = half * (uy[i] + uy[j]);
        const scalar_t adv_z = half * (uz[i] + uz[j]);
        const scalar_t mdot_rc = p ? cvfem_hex8_rhie_chow_mdotc(rho, mu, rc, i, j, ax, ay, az, p[i], p[j]) : scalar_t(0);
        const scalar_t mdot    = rho * (adv_x * ax + adv_y * ay + adv_z * az) + mdot_rc;
        const scalar_t sgn   = mdot > scalar_t(0) ? one : (mdot < scalar_t(0) ? -one : scalar_t(0));
        const scalar_t mpos  = half * (mdot + sgn * mdot);
        const scalar_t mneg  = half * (mdot - sgn * mdot);
        const scalar_t d_pos = half * (one + sgn);
        const scalar_t d_neg = half * (one - sgn);
        const scalar_t dmdot = rho * half * ((vx[i] + vx[j]) * ax + (vy[i] + vy[j]) * ay + (vz[i] + vz[j]) * az) +
                               cvfem_hex8_rhie_chow_dmdotc(rho, mu, rc, i, j, ax, ay, az, q[i], q[j]);
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

template <bool Atomic, typename Slot, typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_ns_upwind_jacobian_add_slots_isoparam(const scalar_t                        rho,
                                                                        const scalar_t                        mu,
                                                                        const scalar_t *const SFEM_RESTRICT   x,
                                                                        const scalar_t *const SFEM_RESTRICT   y,
                                                                        const scalar_t *const SFEM_RESTRICT   z,
                                                                        const scalar_t *const SFEM_RESTRICT   ux,
                                                                        const scalar_t *const SFEM_RESTRICT   uy,
                                                                        const scalar_t *const SFEM_RESTRICT   uz,
                                                                        const Slot *const SFEM_RESTRICT       slots,
                                                                        scalar_t *const SFEM_RESTRICT         values,
                                                                        const Hex8RhieChowT<scalar_t>        &rc = {},
                                                                        const scalar_t *const SFEM_RESTRICT   p  = nullptr) {
    for (int s = 0; s < CVFEM_HEX8_N_SCS; ++s) {
        scalar_t dN[CVFEM_HEX8_N_NODES][3];
        cvfem_hex8_dn_ref<scalar_t>(CVFEM_HEX8_SCS_XI[s][0], CVFEM_HEX8_SCS_XI[s][1], CVFEM_HEX8_SCS_XI[s][2], dN);

        scalar_t adj[9], det;
        cvfem_hex8_geom_at<scalar_t>(x, y, z, CVFEM_HEX8_SCS_XI[s][0], CVFEM_HEX8_SCS_XI[s][1], CVFEM_HEX8_SCS_XI[s][2], adj, &det);

        scalar_t ax, ay, az;
        cvfem_hex8_area_dir(adj, s >> 2, ax, ay, az);

        const scalar_t inv_det = scalar_t(1) / det;
        scalar_t       w[CVFEM_HEX8_N_NODES][3];
        for (int k = 0; k < CVFEM_HEX8_N_NODES; ++k) {
            cvfem_hex8_pushforward(adj, inv_det, dN[k][0], dN[k][1], dN[k][2], w[k][0], w[k][1], w[k][2]);
        }

        const int i = CVFEM_HEX8_SCS[s].i;
        const int j = CVFEM_HEX8_SCS[s].j;
        for (int k = 0; k < CVFEM_HEX8_N_NODES; ++k) {
            const scalar_t wx  = w[k][0];
            const scalar_t wy  = w[k][1];
            const scalar_t wz  = w[k][2];
            const scalar_t d00 = -(scalar_t(2) * wx * ax + wy * ay + wz * az) * mu;
            const scalar_t d01 = -(wx * ay) * mu;
            const scalar_t d02 = -(wx * az) * mu;
            const scalar_t d10 = -(wy * ax) * mu;
            const scalar_t d11 = -(wx * ax + scalar_t(2) * wy * ay + wz * az) * mu;
            const scalar_t d12 = -(wy * az) * mu;
            const scalar_t d20 = -(wz * ax) * mu;
            const scalar_t d21 = -(wz * ay) * mu;
            const scalar_t d22 = -(wx * ax + wy * ay + scalar_t(2) * wz * az) * mu;
            cvfem_hex8_bsr_acc_mom<Atomic>(values, slots[i * 8 + k], d00, d01, d02, d10, d11, d12, d20, d21, d22);
            cvfem_hex8_bsr_acc_mom<Atomic>(values,
                                           slots[j * 8 + k],
                                           -d00,
                                           -d01,
                                           -d02,
                                           -d10,
                                           -d11,
                                           -d12,
                                           -d20,
                                           -d21,
                                           -d22);
        }

        const scalar_t mdot_rc = p ? cvfem_hex8_rhie_chow_mdotc(rho, mu, rc, i, j, ax, ay, az, p[i], p[j]) : scalar_t(0);
        cvfem_hex8_jac_conv_face<Atomic>(rho, ax, ay, az, i, j, ux, uy, uz, slots, values, mdot_rc);
        cvfem_hex8_jac_rhie_chow_p<Atomic>(rho, mu, rc, ax, ay, az, i, j, ux, uy, uz, p, slots, values);
    }
}

#define CVFEM_HEX8_ISO_NODE(a, fld)                                                              \
    do {                                                                                         \
        const scalar_t d0 = dN[a][0];                                                            \
        const scalar_t d1 = dN[a][1];                                                            \
        const scalar_t d2 = dN[a][2];                                                            \
        const scalar_t xa = xyz.x[a][lane];                                                      \
        const scalar_t ya = xyz.y[a][lane];                                                      \
        const scalar_t za = xyz.z[a][lane];                                                      \
        jx0 += xa * d0;                                                                          \
        jy0 += xa * d1;                                                                          \
        jz0 += xa * d2;                                                                          \
        jx1 += ya * d0;                                                                          \
        jy1 += ya * d1;                                                                          \
        jz1 += ya * d2;                                                                          \
        jx2 += za * d0;                                                                          \
        jy2 += za * d1;                                                                          \
        jz2 += za * d2;                                                                          \
        ur0 += fld.ux[a][lane] * d0;                                                             \
        ur1 += fld.ux[a][lane] * d1;                                                             \
        ur2 += fld.ux[a][lane] * d2;                                                             \
        vr0 += fld.uy[a][lane] * d0;                                                             \
        vr1 += fld.uy[a][lane] * d1;                                                             \
        vr2 += fld.uy[a][lane] * d2;                                                             \
        wr0 += fld.uz[a][lane] * d0;                                                             \
        wr1 += fld.uz[a][lane] * d1;                                                             \
        wr2 += fld.uz[a][lane] * d2;                                                             \
    } while (0)

static SFEM_INLINE void cvfem_hex8_ns_upwind_residual_isoparam_simd(const scalar_t      rho_s,
                                                                   const scalar_t      mu_s,
                                                                   const Hex8CoordPack &xyz,
                                                                   const Hex8InputPack &in,
                                                                   Hex8ResidualPack    &out) {
    const scalar_t rho  = rho_s;
    const scalar_t mu   = mu_s;
    const scalar_t half = scalar_t(0.5);
    const scalar_t qtr  = scalar_t(0.25);

    cvfem_hex8_zero_residual_pack(out);

    for (int s = 0; s < CVFEM_HEX8_N_SCS; ++s) {
        scalar_t dN[CVFEM_HEX8_N_NODES][3];
        cvfem_hex8_dn_ref<scalar_t>(CVFEM_HEX8_SCS_XI[s][0], CVFEM_HEX8_SCS_XI[s][1], CVFEM_HEX8_SCS_XI[s][2], dN);
        const int i = CVFEM_HEX8_SCS[s].i;
        const int j = CVFEM_HEX8_SCS[s].j;
        const int d = s >> 2;

#pragma omp simd
        for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
            scalar_t jx0 = 0, jx1 = 0, jx2 = 0;
            scalar_t jy0 = 0, jy1 = 0, jy2 = 0;
            scalar_t jz0 = 0, jz1 = 0, jz2 = 0;
            scalar_t ur0 = 0, ur1 = 0, ur2 = 0;
            scalar_t vr0 = 0, vr1 = 0, vr2 = 0;
            scalar_t wr0 = 0, wr1 = 0, wr2 = 0;
            CVFEM_HEX8_ISO_NODE(0, in);
            CVFEM_HEX8_ISO_NODE(1, in);
            CVFEM_HEX8_ISO_NODE(2, in);
            CVFEM_HEX8_ISO_NODE(3, in);
            CVFEM_HEX8_ISO_NODE(4, in);
            CVFEM_HEX8_ISO_NODE(5, in);
            CVFEM_HEX8_ISO_NODE(6, in);
            CVFEM_HEX8_ISO_NODE(7, in);

            const scalar_t c0  = jy1 * jz2 - jy2 * jz1;
            const scalar_t c1  = jy2 * jz0 - jy0 * jz2;
            const scalar_t c2  = jy0 * jz1 - jy1 * jz0;
            const scalar_t c3  = jz1 * jx2 - jz2 * jx1;
            const scalar_t c4  = jz2 * jx0 - jz0 * jx2;
            const scalar_t c5  = jz0 * jx1 - jz1 * jx0;
            const scalar_t c6  = jx1 * jy2 - jx2 * jy1;
            const scalar_t c7  = jx2 * jy0 - jx0 * jy2;
            const scalar_t c8  = jx0 * jy1 - jx1 * jy0;
            const scalar_t det = jx0 * c0 + jx1 * c1 + jx2 * c2;
            const scalar_t inv = scalar_t(1) / det;
            scalar_t       ax, ay, az;
            if (d == 0) {
                ax = qtr * c0;
                ay = qtr * c1;
                az = qtr * c2;
            } else if (d == 1) {
                ax = qtr * c3;
                ay = qtr * c4;
                az = qtr * c5;
            } else {
                ax = qtr * c6;
                ay = qtr * c7;
                az = qtr * c8;
            }

            const scalar_t g00 = (c0 * ur0 + c3 * ur1 + c6 * ur2) * inv;
            const scalar_t g01 = (c1 * ur0 + c4 * ur1 + c7 * ur2) * inv;
            const scalar_t g02 = (c2 * ur0 + c5 * ur1 + c8 * ur2) * inv;
            const scalar_t g10 = (c0 * vr0 + c3 * vr1 + c6 * vr2) * inv;
            const scalar_t g11 = (c1 * vr0 + c4 * vr1 + c7 * vr2) * inv;
            const scalar_t g12 = (c2 * vr0 + c5 * vr1 + c8 * vr2) * inv;
            const scalar_t g20 = (c0 * wr0 + c3 * wr1 + c6 * wr2) * inv;
            const scalar_t g21 = (c1 * wr0 + c4 * wr1 + c7 * wr2) * inv;
            const scalar_t g22 = (c2 * wr0 + c5 * wr1 + c8 * wr2) * inv;

            const scalar_t tau_x =
                    mu * ((scalar_t(2) * g00) * ax + (g01 + g10) * ay + (g02 + g20) * az);
            const scalar_t tau_y =
                    mu * ((g10 + g01) * ax + (scalar_t(2) * g11) * ay + (g12 + g21) * az);
            const scalar_t tau_z =
                    mu * ((g20 + g02) * ax + (g21 + g12) * ay + (scalar_t(2) * g22) * az);

            const scalar_t adv_x = half * (in.ux[i][lane] + in.ux[j][lane]);
            const scalar_t adv_y = half * (in.uy[i][lane] + in.uy[j][lane]);
            const scalar_t adv_z = half * (in.uz[i][lane] + in.uz[j][lane]);
            const scalar_t mdot  = rho * (adv_x * ax + adv_y * ay + adv_z * az);
            const scalar_t sgn   = mdot > scalar_t(0) ? scalar_t(1) : (mdot < scalar_t(0) ? scalar_t(-1) : scalar_t(0));
            const scalar_t mpos  = half * (mdot + sgn * mdot);
            const scalar_t mneg  = half * (mdot - sgn * mdot);
            const scalar_t pmid  = half * (in.p[i][lane] + in.p[j][lane]);
            const scalar_t fx    = mpos * in.ux[i][lane] + mneg * in.ux[j][lane] + pmid * ax - tau_x;
            const scalar_t fy    = mpos * in.uy[i][lane] + mneg * in.uy[j][lane] + pmid * ay - tau_y;
            const scalar_t fz    = mpos * in.uz[i][lane] + mneg * in.uz[j][lane] + pmid * az - tau_z;
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

static SFEM_INLINE void cvfem_hex8_ns_upwind_jacobian_action_isoparam_simd(const scalar_t      rho_s,
                                                                          const scalar_t      mu_s,
                                                                          const Hex8CoordPack &xyz,
                                                                          const Hex8InputPack &u,
                                                                          const Hex8InputPack &du,
                                                                          Hex8ResidualPack    &out) {
    const scalar_t rho  = rho_s;
    const scalar_t mu   = mu_s;
    const scalar_t half = scalar_t(0.5);
    const scalar_t one  = scalar_t(1);
    const scalar_t qtr  = scalar_t(0.25);

    cvfem_hex8_zero_residual_pack(out);

    for (int s = 0; s < CVFEM_HEX8_N_SCS; ++s) {
        scalar_t dN[CVFEM_HEX8_N_NODES][3];
        cvfem_hex8_dn_ref<scalar_t>(CVFEM_HEX8_SCS_XI[s][0], CVFEM_HEX8_SCS_XI[s][1], CVFEM_HEX8_SCS_XI[s][2], dN);
        const int i = CVFEM_HEX8_SCS[s].i;
        const int j = CVFEM_HEX8_SCS[s].j;
        const int d = s >> 2;

#pragma omp simd
        for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
            scalar_t jx0 = 0, jx1 = 0, jx2 = 0;
            scalar_t jy0 = 0, jy1 = 0, jy2 = 0;
            scalar_t jz0 = 0, jz1 = 0, jz2 = 0;
            scalar_t ur0 = 0, ur1 = 0, ur2 = 0;
            scalar_t vr0 = 0, vr1 = 0, vr2 = 0;
            scalar_t wr0 = 0, wr1 = 0, wr2 = 0;
            CVFEM_HEX8_ISO_NODE(0, du);
            CVFEM_HEX8_ISO_NODE(1, du);
            CVFEM_HEX8_ISO_NODE(2, du);
            CVFEM_HEX8_ISO_NODE(3, du);
            CVFEM_HEX8_ISO_NODE(4, du);
            CVFEM_HEX8_ISO_NODE(5, du);
            CVFEM_HEX8_ISO_NODE(6, du);
            CVFEM_HEX8_ISO_NODE(7, du);

            const scalar_t c0  = jy1 * jz2 - jy2 * jz1;
            const scalar_t c1  = jy2 * jz0 - jy0 * jz2;
            const scalar_t c2  = jy0 * jz1 - jy1 * jz0;
            const scalar_t c3  = jz1 * jx2 - jz2 * jx1;
            const scalar_t c4  = jz2 * jx0 - jz0 * jx2;
            const scalar_t c5  = jz0 * jx1 - jz1 * jx0;
            const scalar_t c6  = jx1 * jy2 - jx2 * jy1;
            const scalar_t c7  = jx2 * jy0 - jx0 * jy2;
            const scalar_t c8  = jx0 * jy1 - jx1 * jy0;
            const scalar_t det = jx0 * c0 + jx1 * c1 + jx2 * c2;
            const scalar_t inv = scalar_t(1) / det;
            scalar_t       ax, ay, az;
            if (d == 0) {
                ax = qtr * c0;
                ay = qtr * c1;
                az = qtr * c2;
            } else if (d == 1) {
                ax = qtr * c3;
                ay = qtr * c4;
                az = qtr * c5;
            } else {
                ax = qtr * c6;
                ay = qtr * c7;
                az = qtr * c8;
            }

            const scalar_t g00 = (c0 * ur0 + c3 * ur1 + c6 * ur2) * inv;
            const scalar_t g01 = (c1 * ur0 + c4 * ur1 + c7 * ur2) * inv;
            const scalar_t g02 = (c2 * ur0 + c5 * ur1 + c8 * ur2) * inv;
            const scalar_t g10 = (c0 * vr0 + c3 * vr1 + c6 * vr2) * inv;
            const scalar_t g11 = (c1 * vr0 + c4 * vr1 + c7 * vr2) * inv;
            const scalar_t g12 = (c2 * vr0 + c5 * vr1 + c8 * vr2) * inv;
            const scalar_t g20 = (c0 * wr0 + c3 * wr1 + c6 * wr2) * inv;
            const scalar_t g21 = (c1 * wr0 + c4 * wr1 + c7 * wr2) * inv;
            const scalar_t g22 = (c2 * wr0 + c5 * wr1 + c8 * wr2) * inv;

            const scalar_t tx = mu * ((scalar_t(2) * g00) * ax + (g01 + g10) * ay + (g02 + g20) * az);
            const scalar_t ty = mu * ((g10 + g01) * ax + (scalar_t(2) * g11) * ay + (g12 + g21) * az);
            const scalar_t tz = mu * ((g20 + g02) * ax + (g21 + g12) * ay + (scalar_t(2) * g22) * az);

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
                    dpos * u.ux[i][lane] + mpos * du.ux[i][lane] + dneg * u.ux[j][lane] + mneg * du.ux[j][lane] + qmid * ax -
                    tx;
            const scalar_t fy =
                    dpos * u.uy[i][lane] + mpos * du.uy[i][lane] + dneg * u.uy[j][lane] + mneg * du.uy[j][lane] + qmid * ay -
                    ty;
            const scalar_t fz =
                    dpos * u.uz[i][lane] + mpos * du.uz[i][lane] + dneg * u.uz[j][lane] + mneg * du.uz[j][lane] + qmid * az -
                    tz;
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

#undef CVFEM_HEX8_ISO_NODE

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_ns_upwind_jacobian_fd(const scalar_t                        rho,
                                                         const scalar_t                        mu,
                                                         const scalar_t *const SFEM_RESTRICT adj, const scalar_t det,
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
        cvfem_hex8_ns_upwind_residual(rho, mu, adj, det, up, vp, wp, pp, rm);

        for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i) {
            const scalar_t delta = i == col ? eps : scalar_t(0);
            const int      a     = i / 4;
            const int      f     = i & 3;
            if (f == 0) up[a] = q[i] + delta;
            if (f == 1) vp[a] = q[i] + delta;
            if (f == 2) wp[a] = q[i] + delta;
            if (f == 3) pp[a] = q[i] + delta;
        }
        cvfem_hex8_ns_upwind_residual(rho, mu, adj, det, up, vp, wp, pp, rp);

        for (int row = 0; row < CVFEM_HEX8_N_DOF; ++row) {
            ke[row * CVFEM_HEX8_N_DOF + col] = (rp[row] - rm[row]) / (2 * eps);
        }
    }
}

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_ns_upwind_jacobian_fd_isoparam(const scalar_t                        rho,
                                                                  const scalar_t                        mu,
                                                                  const scalar_t *const SFEM_RESTRICT   x,
                                                                  const scalar_t *const SFEM_RESTRICT   y,
                                                                  const scalar_t *const SFEM_RESTRICT   z,
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
        cvfem_hex8_ns_upwind_residual_isoparam(rho, mu, x, y, z, up, vp, wp, pp, rm);

        for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i) {
            const scalar_t delta = i == col ? eps : scalar_t(0);
            const int      a     = i / 4;
            const int      f     = i & 3;
            if (f == 0) up[a] = q[i] + delta;
            if (f == 1) vp[a] = q[i] + delta;
            if (f == 2) wp[a] = q[i] + delta;
            if (f == 3) pp[a] = q[i] + delta;
        }
        cvfem_hex8_ns_upwind_residual_isoparam(rho, mu, x, y, z, up, vp, wp, pp, rp);

        for (int row = 0; row < CVFEM_HEX8_N_DOF; ++row) {
            ke[row * CVFEM_HEX8_N_DOF + col] = (rp[row] - rm[row]) / (2 * eps);
        }
    }
}

#endif

