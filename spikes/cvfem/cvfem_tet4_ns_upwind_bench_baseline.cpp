#include "smesh_mesh.hpp"

#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef SFEM_RESTRICT
#define SFEM_RESTRICT __restrict__
#endif

#ifndef SFEM_INLINE
#define SFEM_INLINE inline __attribute__((always_inline))
#endif

#ifndef SFEM_NOINLINE
#define SFEM_NOINLINE __attribute__((noinline))
#endif

using scalar_t = double;
using jacobian_t = smesh::jacobian_t;

#ifndef VEC_BYTES
#define VEC_BYTES 128
#endif

static constexpr int VEC_SIZE = VEC_BYTES / int(sizeof(scalar_t));
static_assert(VEC_SIZE >= 1, "invalid vector size");
static constexpr int ALIGN_BYTES = 64;

template <typename T>
struct AlignedBuffer {
  T *ptr{nullptr};
  size_t n{0};

  AlignedBuffer() = default;
  AlignedBuffer(const AlignedBuffer &) = delete;
  AlignedBuffer &operator=(const AlignedBuffer &) = delete;
  AlignedBuffer(AlignedBuffer &&o) noexcept : ptr(o.ptr), n(o.n) {
    o.ptr = nullptr;
    o.n = 0;
  }
  AlignedBuffer &operator=(AlignedBuffer &&o) noexcept {
    if (this != &o) {
      std::free(ptr);
      ptr = o.ptr;
      n = o.n;
      o.ptr = nullptr;
      o.n = 0;
    }
    return *this;
  }
  ~AlignedBuffer() { std::free(ptr); }

  void resize(const size_t count) {
    std::free(ptr);
    ptr = nullptr;
    n = 0;
    if (!count)
      return;
    const size_t bytes =
        ((count * sizeof(T) + (size_t)ALIGN_BYTES - 1) / (size_t)ALIGN_BYTES) *
        (size_t)ALIGN_BYTES;
    void *p = nullptr;
    if (posix_memalign(&p, (size_t)ALIGN_BYTES, bytes) != 0)
      return;
    std::memset(p, 0, bytes);
    ptr = static_cast<T *>(p);
    n = count;
  }

  T *data() { return ptr; }
  const T *data() const { return ptr; }
  T &operator[](ptrdiff_t i) { return ptr[i]; }
  const T &operator[](ptrdiff_t i) const { return ptr[i]; }
};

static SFEM_INLINE const jacobian_t *aligned_geom(const jacobian_t *p) {
  return static_cast<const jacobian_t *>(
      __builtin_assume_aligned(p, ALIGN_BYTES));
}

struct MeshData {
  std::shared_ptr<smesh::Mesh> mesh;
  ptrdiff_t nnodes{0};
  ptrdiff_t nelements{0};
  smesh::idx_t **elems{nullptr};
  smesh::geom_t **points{nullptr};

  std::vector<scalar_t> ux, uy, uz, p;
  std::vector<scalar_t> rx, ry, rz, rc;

  AlignedBuffer<jacobian_t> adj[9];
  AlignedBuffer<jacobian_t> det;
};

static ptrdiff_t parse_size(const char *s) {
  char *end = nullptr;
  const double v = std::strtod(s, &end);
  ptrdiff_t scale = 1;
  if (end && *end) {
    if (*end == 'k' || *end == 'K')
      scale = 1024LL;
    if (*end == 'm' || *end == 'M')
      scale = 1024LL * 1024LL;
  }
  return ptrdiff_t(v * double(scale));
}

static int threads_active() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

static double wall_time() {
  return std::chrono::duration<double>(
             std::chrono::high_resolution_clock::now().time_since_epoch())
      .count();
}

static void fill_fields(MeshData &d) {
  d.ux.resize(d.nnodes);
  d.uy.resize(d.nnodes);
  d.uz.resize(d.nnodes);
  d.p.resize(d.nnodes);
  d.rx.assign(d.nnodes, 0.0);
  d.ry.assign(d.nnodes, 0.0);
  d.rz.assign(d.nnodes, 0.0);
  d.rc.assign(d.nnodes, 0.0);

  const auto *const x = d.points[0];
  const auto *const y = d.points[1];
  const auto *const z = d.points[2];

#pragma omp parallel for schedule(static)
  for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
    d.ux[i] = 1.0 + 0.3 * x[i] - 0.2 * y[i] + 0.1 * z[i];
    d.uy[i] = -0.4 + 0.2 * x[i] + 0.5 * y[i] - 0.15 * z[i];
    d.uz[i] = 0.2 - 0.1 * x[i] + 0.25 * y[i] + 0.4 * z[i];
    d.p[i] = 1.0 + 0.1 * x[i] + 0.2 * y[i] - 0.05 * z[i];
  }
}

static void reset_residual(MeshData &d) {
#pragma omp parallel for schedule(static)
  for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
    d.rx[i] = 0.0;
    d.ry[i] = 0.0;
    d.rz[i] = 0.0;
    d.rc[i] = 0.0;
  }
}

static void precompute_affine_geometry(MeshData &d) {
  const ptrdiff_t padded_nelements =
      ((d.nelements + VEC_SIZE - 1) / VEC_SIZE) * VEC_SIZE;

  for (int k = 0; k < 9; ++k)
    d.adj[k].resize(padded_nelements);
  d.det.resize(padded_nelements);

  const auto *const x = d.points[0];
  const auto *const y = d.points[1];
  const auto *const z = d.points[2];
  smesh::idx_t **const ev = d.elems;

#pragma omp parallel for schedule(static)
  for (ptrdiff_t e = 0; e < d.nelements; ++e) {
    const smesh::idx_t i0 = ev[0][e];
    const smesh::idx_t i1 = ev[1][e];
    const smesh::idx_t i2 = ev[2][e];
    const smesh::idx_t i3 = ev[3][e];

    const scalar_t j00 = scalar_t(x[i1] - x[i0]);
    const scalar_t j10 = scalar_t(y[i1] - y[i0]);
    const scalar_t j20 = scalar_t(z[i1] - z[i0]);
    const scalar_t j01 = scalar_t(x[i2] - x[i0]);
    const scalar_t j11 = scalar_t(y[i2] - y[i0]);
    const scalar_t j21 = scalar_t(z[i2] - z[i0]);
    const scalar_t j02 = scalar_t(x[i3] - x[i0]);
    const scalar_t j12 = scalar_t(y[i3] - y[i0]);
    const scalar_t j22 = scalar_t(z[i3] - z[i0]);

    const scalar_t a00 = j11 * j22 - j12 * j21;
    const scalar_t a01 = -(j10 * j22 - j12 * j20);
    const scalar_t a02 = j10 * j21 - j11 * j20;
    const scalar_t a10 = -(j01 * j22 - j02 * j21);
    const scalar_t a11 = j00 * j22 - j02 * j20;
    const scalar_t a12 = -(j00 * j21 - j01 * j20);
    const scalar_t a20 = j01 * j12 - j02 * j11;
    const scalar_t a21 = -(j00 * j12 - j02 * j10);
    const scalar_t a22 = j00 * j11 - j01 * j10;

    const scalar_t determinant = j00 * a00 + j01 * a01 + j02 * a02;

    d.adj[0][e] = jacobian_t(a00);
    d.adj[1][e] = jacobian_t(a10);
    d.adj[2][e] = jacobian_t(a20);
    d.adj[3][e] = jacobian_t(a01);
    d.adj[4][e] = jacobian_t(a11);
    d.adj[5][e] = jacobian_t(a21);
    d.adj[6][e] = jacobian_t(a02);
    d.adj[7][e] = jacobian_t(a12);
    d.adj[8][e] = jacobian_t(a22);
    d.det[e] = jacobian_t(determinant);
  }

  if (padded_nelements > d.nelements) {
    const ptrdiff_t last = d.nelements - 1;
    for (ptrdiff_t e = d.nelements; e < padded_nelements; ++e) {
      for (int k = 0; k < 9; ++k)
        d.adj[k][e] = d.adj[k][last];
      d.det[e] = d.det[last];
    }
  }
}

static SFEM_INLINE void atomic_add(scalar_t *const SFEM_RESTRICT f,
                                   const smesh::idx_t id,
                                   const scalar_t value) {
#pragma omp atomic update
  f[id] += value;
}

struct Tet4InputPack {
  scalar_t ux0[VEC_SIZE], ux1[VEC_SIZE], ux2[VEC_SIZE], ux3[VEC_SIZE];
  scalar_t uy0[VEC_SIZE], uy1[VEC_SIZE], uy2[VEC_SIZE], uy3[VEC_SIZE];
  scalar_t uz0[VEC_SIZE], uz1[VEC_SIZE], uz2[VEC_SIZE], uz3[VEC_SIZE];
  scalar_t p0[VEC_SIZE], p1[VEC_SIZE], p2[VEC_SIZE], p3[VEC_SIZE];
};

struct Tet4ResidualPack {
  scalar_t rx0[VEC_SIZE], rx1[VEC_SIZE], rx2[VEC_SIZE], rx3[VEC_SIZE];
  scalar_t ry0[VEC_SIZE], ry1[VEC_SIZE], ry2[VEC_SIZE], ry3[VEC_SIZE];
  scalar_t rz0[VEC_SIZE], rz1[VEC_SIZE], rz2[VEC_SIZE], rz3[VEC_SIZE];
  scalar_t rc0[VEC_SIZE], rc1[VEC_SIZE], rc2[VEC_SIZE], rc3[VEC_SIZE];
};

static SFEM_INLINE void gather_tet4_pack(const MeshData &d,
                                         const ptrdiff_t begin,
                                         const int nlanes,
                                         Tet4InputPack &pack) {
  smesh::idx_t **const SFEM_RESTRICT elems = d.elems;
  const scalar_t *const SFEM_RESTRICT ux = d.ux.data();
  const scalar_t *const SFEM_RESTRICT uy = d.uy.data();
  const scalar_t *const SFEM_RESTRICT uz = d.uz.data();
  const scalar_t *const SFEM_RESTRICT p = d.p.data();

  const int last_active_lane = nlanes - 1;

  for (int lane = 0; lane < VEC_SIZE; ++lane) {
    const int active_lane = lane < nlanes ? lane : last_active_lane;
    const ptrdiff_t e = begin + active_lane;
    const smesh::idx_t n0 = elems[0][e];
    const smesh::idx_t n1 = elems[1][e];
    const smesh::idx_t n2 = elems[2][e];
    const smesh::idx_t n3 = elems[3][e];

    pack.ux0[lane] = ux[n0];
    pack.ux1[lane] = ux[n1];
    pack.ux2[lane] = ux[n2];
    pack.ux3[lane] = ux[n3];
    pack.uy0[lane] = uy[n0];
    pack.uy1[lane] = uy[n1];
    pack.uy2[lane] = uy[n2];
    pack.uy3[lane] = uy[n3];
    pack.uz0[lane] = uz[n0];
    pack.uz1[lane] = uz[n1];
    pack.uz2[lane] = uz[n2];
    pack.uz3[lane] = uz[n3];
    pack.p0[lane] = p[n0];
    pack.p1[lane] = p[n1];
    pack.p2[lane] = p[n2];
    pack.p3[lane] = p[n3];
  }
}

static SFEM_INLINE void cvfem_tet4_ns_upwind_simd_microkernel(
    const scalar_t rho_s, const scalar_t mu_s,
    const jacobian_t *const SFEM_RESTRICT adj0_ptr,
    const jacobian_t *const SFEM_RESTRICT adj1_ptr,
    const jacobian_t *const SFEM_RESTRICT adj2_ptr,
    const jacobian_t *const SFEM_RESTRICT adj3_ptr,
    const jacobian_t *const SFEM_RESTRICT adj4_ptr,
    const jacobian_t *const SFEM_RESTRICT adj5_ptr,
    const jacobian_t *const SFEM_RESTRICT adj6_ptr,
    const jacobian_t *const SFEM_RESTRICT adj7_ptr,
    const jacobian_t *const SFEM_RESTRICT adj8_ptr,
    const jacobian_t *const SFEM_RESTRICT det_ptr, const Tet4InputPack &in,
    Tet4ResidualPack &out) {
  const scalar_t half = 0.5;
  const scalar_t two = 2.0;
  const scalar_t rho = rho_s;
  const scalar_t mu = mu_s;
  const scalar_t c12 = 1.0 / 12.0;
  const scalar_t c24 = 1.0 / 24.0;

  alignas(ALIGN_BYTES) scalar_t g00v[VEC_SIZE], g01v[VEC_SIZE], g02v[VEC_SIZE];
  alignas(ALIGN_BYTES) scalar_t g10v[VEC_SIZE], g11v[VEC_SIZE], g12v[VEC_SIZE];
  alignas(ALIGN_BYTES) scalar_t g20v[VEC_SIZE], g21v[VEC_SIZE], g22v[VEC_SIZE];

#pragma omp simd aligned(adj0_ptr, adj1_ptr, adj2_ptr, adj3_ptr, adj4_ptr,      \
                             adj5_ptr, adj6_ptr, adj7_ptr, adj8_ptr, det_ptr   \
                         : 64)
  for (int lane = 0; lane < VEC_SIZE; ++lane) {
    const scalar_t adj0 = scalar_t(adj0_ptr[lane]);
    const scalar_t adj1 = scalar_t(adj1_ptr[lane]);
    const scalar_t adj2 = scalar_t(adj2_ptr[lane]);
    const scalar_t adj3 = scalar_t(adj3_ptr[lane]);
    const scalar_t adj4 = scalar_t(adj4_ptr[lane]);
    const scalar_t adj5 = scalar_t(adj5_ptr[lane]);
    const scalar_t adj6 = scalar_t(adj6_ptr[lane]);
    const scalar_t adj7 = scalar_t(adj7_ptr[lane]);
    const scalar_t adj8 = scalar_t(adj8_ptr[lane]);
    const scalar_t inv_det = 1.0 / scalar_t(det_ptr[lane]);
    const scalar_t ux0 = in.ux0[lane];
    const scalar_t dux0 = in.ux1[lane] - ux0;
    const scalar_t dux1 = in.ux2[lane] - ux0;
    const scalar_t dux2 = in.ux3[lane] - ux0;
    const scalar_t uy0 = in.uy0[lane];
    const scalar_t duy0 = in.uy1[lane] - uy0;
    const scalar_t duy1 = in.uy2[lane] - uy0;
    const scalar_t duy2 = in.uy3[lane] - uy0;
    const scalar_t uz0 = in.uz0[lane];
    const scalar_t duz0 = in.uz1[lane] - uz0;
    const scalar_t duz1 = in.uz2[lane] - uz0;
    const scalar_t duz2 = in.uz3[lane] - uz0;
    g00v[lane] = (dux0 * adj0 + dux1 * adj3 + dux2 * adj6) * inv_det;
    g01v[lane] = (dux0 * adj1 + dux1 * adj4 + dux2 * adj7) * inv_det;
    g02v[lane] = (dux0 * adj2 + dux1 * adj5 + dux2 * adj8) * inv_det;
    g10v[lane] = (duy0 * adj0 + duy1 * adj3 + duy2 * adj6) * inv_det;
    g11v[lane] = (duy0 * adj1 + duy1 * adj4 + duy2 * adj7) * inv_det;
    g12v[lane] = (duy0 * adj2 + duy1 * adj5 + duy2 * adj8) * inv_det;
    g20v[lane] = (duz0 * adj0 + duz1 * adj3 + duz2 * adj6) * inv_det;
    g21v[lane] = (duz0 * adj1 + duz1 * adj4 + duz2 * adj7) * inv_det;
    g22v[lane] = (duz0 * adj2 + duz1 * adj5 + duz2 * adj8) * inv_det;
    out.rx0[lane] = 0.0;
    out.rx1[lane] = 0.0;
    out.rx2[lane] = 0.0;
    out.rx3[lane] = 0.0;
    out.ry0[lane] = 0.0;
    out.ry1[lane] = 0.0;
    out.ry2[lane] = 0.0;
    out.ry3[lane] = 0.0;
    out.rz0[lane] = 0.0;
    out.rz1[lane] = 0.0;
    out.rz2[lane] = 0.0;
    out.rz3[lane] = 0.0;
    out.rc0[lane] = 0.0;
    out.rc1[lane] = 0.0;
    out.rc2[lane] = 0.0;
    out.rc3[lane] = 0.0;
  }

#define GEOM_SIMD_PRAGMA                                                       \
  _Pragma("omp simd aligned(adj0_ptr, adj1_ptr, adj2_ptr, adj3_ptr, adj4_ptr, adj5_ptr, adj6_ptr, adj7_ptr, adj8_ptr, det_ptr: 64)")

#define SCS_AREA3(AR0, AR1, AR2)                                               \
  const scalar_t adj0 = scalar_t(adj0_ptr[lane]);                              \
  const scalar_t adj1 = scalar_t(adj1_ptr[lane]);                              \
  const scalar_t adj2 = scalar_t(adj2_ptr[lane]);                              \
  const scalar_t adj3 = scalar_t(adj3_ptr[lane]);                              \
  const scalar_t adj4 = scalar_t(adj4_ptr[lane]);                              \
  const scalar_t adj5 = scalar_t(adj5_ptr[lane]);                              \
  const scalar_t adj6 = scalar_t(adj6_ptr[lane]);                              \
  const scalar_t adj7 = scalar_t(adj7_ptr[lane]);                              \
  const scalar_t adj8 = scalar_t(adj8_ptr[lane]);                              \
  const scalar_t ax = adj0 * (AR0) + adj3 * (AR1) + adj6 * (AR2);               \
  const scalar_t ay = adj1 * (AR0) + adj4 * (AR1) + adj7 * (AR2);               \
  const scalar_t az = adj2 * (AR0) + adj5 * (AR1) + adj8 * (AR2)

#define SCS_AREA_AR2_0(AR0, AR1)                                               \
  const scalar_t adj0 = scalar_t(adj0_ptr[lane]);                              \
  const scalar_t adj1 = scalar_t(adj1_ptr[lane]);                              \
  const scalar_t adj2 = scalar_t(adj2_ptr[lane]);                              \
  const scalar_t adj3 = scalar_t(adj3_ptr[lane]);                              \
  const scalar_t adj4 = scalar_t(adj4_ptr[lane]);                              \
  const scalar_t adj5 = scalar_t(adj5_ptr[lane]);                              \
  const scalar_t ax = adj0 * (AR0) + adj3 * (AR1);                              \
  const scalar_t ay = adj1 * (AR0) + adj4 * (AR1);                              \
  const scalar_t az = adj2 * (AR0) + adj5 * (AR1)

#define SCS_AREA_AR1_0(AR0, AR2)                                               \
  const scalar_t adj0 = scalar_t(adj0_ptr[lane]);                              \
  const scalar_t adj1 = scalar_t(adj1_ptr[lane]);                              \
  const scalar_t adj2 = scalar_t(adj2_ptr[lane]);                              \
  const scalar_t adj6 = scalar_t(adj6_ptr[lane]);                              \
  const scalar_t adj7 = scalar_t(adj7_ptr[lane]);                              \
  const scalar_t adj8 = scalar_t(adj8_ptr[lane]);                              \
  const scalar_t ax = adj0 * (AR0) + adj6 * (AR2);                              \
  const scalar_t ay = adj1 * (AR0) + adj7 * (AR2);                              \
  const scalar_t az = adj2 * (AR0) + adj8 * (AR2)

#define SCS_AREA_AR0_0(AR1, AR2)                                               \
  const scalar_t adj3 = scalar_t(adj3_ptr[lane]);                              \
  const scalar_t adj4 = scalar_t(adj4_ptr[lane]);                              \
  const scalar_t adj5 = scalar_t(adj5_ptr[lane]);                              \
  const scalar_t adj6 = scalar_t(adj6_ptr[lane]);                              \
  const scalar_t adj7 = scalar_t(adj7_ptr[lane]);                              \
  const scalar_t adj8 = scalar_t(adj8_ptr[lane]);                              \
  const scalar_t ax = adj3 * (AR1) + adj6 * (AR2);                              \
  const scalar_t ay = adj4 * (AR1) + adj7 * (AR2);                              \
  const scalar_t az = adj5 * (AR1) + adj8 * (AR2)

#define SCS_FLUX_LANES(I, J, AREA)                                             \
  do {                                                                         \
    GEOM_SIMD_PRAGMA for (int lane = 0; lane < VEC_SIZE; ++lane) {             \
      AREA;                                                                    \
      const scalar_t uxI = in.ux##I[lane];                                     \
      const scalar_t uxJ = in.ux##J[lane];                                     \
      const scalar_t uyI = in.uy##I[lane];                                     \
      const scalar_t uyJ = in.uy##J[lane];                                     \
      const scalar_t uzI = in.uz##I[lane];                                     \
      const scalar_t uzJ = in.uz##J[lane];                                     \
      const scalar_t adv_x = half * (uxI + uxJ);                               \
      const scalar_t adv_y = half * (uyI + uyJ);                               \
      const scalar_t adv_z = half * (uzI + uzJ);                               \
      const scalar_t mdot = rho * (adv_x * ax + adv_y * ay + adv_z * az);      \
      const scalar_t mdot_abs = mdot < scalar_t(0) ? -mdot : mdot;             \
      const scalar_t mdot_pos = half * (mdot + mdot_abs);                      \
      const scalar_t mdot_neg = half * (mdot - mdot_abs);                      \
      const scalar_t p_mid = half * (in.p##I[lane] + in.p##J[lane]);           \
      const scalar_t g00 = g00v[lane];                                         \
      const scalar_t g01 = g01v[lane];                                         \
      const scalar_t g02 = g02v[lane];                                         \
      const scalar_t g10 = g10v[lane];                                         \
      const scalar_t g11 = g11v[lane];                                         \
      const scalar_t g12 = g12v[lane];                                         \
      const scalar_t g20 = g20v[lane];                                         \
      const scalar_t g21 = g21v[lane];                                         \
      const scalar_t g22 = g22v[lane];                                         \
      const scalar_t tau_x =                                                   \
          mu * ((two * g00) * ax + (g01 + g10) * ay + (g02 + g20) * az);       \
      const scalar_t tau_y =                                                   \
          mu * ((g10 + g01) * ax + (two * g11) * ay + (g12 + g21) * az);       \
      const scalar_t tau_z =                                                   \
          mu * ((g20 + g02) * ax + (g21 + g12) * ay + (two * g22) * az);       \
      const scalar_t fx = mdot_pos * uxI + mdot_neg * uxJ + p_mid * ax - tau_x; \
      const scalar_t fy = mdot_pos * uyI + mdot_neg * uyJ + p_mid * ay - tau_y; \
      const scalar_t fz = mdot_pos * uzI + mdot_neg * uzJ + p_mid * az - tau_z; \
      out.rx##I[lane] += fx;                                                   \
      out.ry##I[lane] += fy;                                                   \
      out.rz##I[lane] += fz;                                                   \
      out.rc##I[lane] += mdot;                                                 \
      out.rx##J[lane] -= fx;                                                   \
      out.ry##J[lane] -= fy;                                                   \
      out.rz##J[lane] -= fz;                                                   \
      out.rc##J[lane] -= mdot;                                                 \
    }                                                                          \
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

static SFEM_INLINE void run_microkernel(const MeshData &d, const scalar_t rho,
                                        const scalar_t mu,
                                        const ptrdiff_t begin, const int nlanes,
                                        const Tet4InputPack &in,
                                        Tet4ResidualPack &out) {
  if (nlanes == VEC_SIZE) {
    cvfem_tet4_ns_upwind_simd_microkernel(
        rho, mu, aligned_geom(d.adj[0].data() + begin),
        aligned_geom(d.adj[1].data() + begin),
        aligned_geom(d.adj[2].data() + begin),
        aligned_geom(d.adj[3].data() + begin),
        aligned_geom(d.adj[4].data() + begin),
        aligned_geom(d.adj[5].data() + begin),
        aligned_geom(d.adj[6].data() + begin),
        aligned_geom(d.adj[7].data() + begin),
        aligned_geom(d.adj[8].data() + begin),
        aligned_geom(d.det.data() + begin), in, out);
    return;
  }

  alignas(ALIGN_BYTES) jacobian_t a0[VEC_SIZE], a1[VEC_SIZE], a2[VEC_SIZE],
      a3[VEC_SIZE], a4[VEC_SIZE];
  alignas(ALIGN_BYTES) jacobian_t a5[VEC_SIZE], a6[VEC_SIZE], a7[VEC_SIZE],
      a8[VEC_SIZE], det[VEC_SIZE];
  const int last = nlanes - 1;
  for (int lane = 0; lane < VEC_SIZE; ++lane) {
    const int e = lane < nlanes ? lane : last;
    a0[lane] = d.adj[0][begin + e];
    a1[lane] = d.adj[1][begin + e];
    a2[lane] = d.adj[2][begin + e];
    a3[lane] = d.adj[3][begin + e];
    a4[lane] = d.adj[4][begin + e];
    a5[lane] = d.adj[5][begin + e];
    a6[lane] = d.adj[6][begin + e];
    a7[lane] = d.adj[7][begin + e];
    a8[lane] = d.adj[8][begin + e];
    det[lane] = d.det[begin + e];
  }
  cvfem_tet4_ns_upwind_simd_microkernel(rho, mu, a0, a1, a2, a3, a4, a5, a6, a7,
                                        a8, det, in, out);
}

static SFEM_INLINE void scatter_tet4_pack(MeshData &d, const ptrdiff_t begin,
                                          const int nlanes,
                                          const Tet4ResidualPack &pack) {
  smesh::idx_t **const SFEM_RESTRICT elems = d.elems;
  scalar_t *const SFEM_RESTRICT rx_ptr = d.rx.data();
  scalar_t *const SFEM_RESTRICT ry_ptr = d.ry.data();
  scalar_t *const SFEM_RESTRICT rz_ptr = d.rz.data();
  scalar_t *const SFEM_RESTRICT rc_ptr = d.rc.data();

  for (int lane = 0; lane < nlanes; ++lane) {
    const ptrdiff_t e = begin + lane;
    const smesh::idx_t n0 = elems[0][e];
    const smesh::idx_t n1 = elems[1][e];
    const smesh::idx_t n2 = elems[2][e];
    const smesh::idx_t n3 = elems[3][e];

    atomic_add(rx_ptr, n0, pack.rx0[lane]);
    atomic_add(rx_ptr, n1, pack.rx1[lane]);
    atomic_add(rx_ptr, n2, pack.rx2[lane]);
    atomic_add(rx_ptr, n3, pack.rx3[lane]);
    atomic_add(ry_ptr, n0, pack.ry0[lane]);
    atomic_add(ry_ptr, n1, pack.ry1[lane]);
    atomic_add(ry_ptr, n2, pack.ry2[lane]);
    atomic_add(ry_ptr, n3, pack.ry3[lane]);
    atomic_add(rz_ptr, n0, pack.rz0[lane]);
    atomic_add(rz_ptr, n1, pack.rz1[lane]);
    atomic_add(rz_ptr, n2, pack.rz2[lane]);
    atomic_add(rz_ptr, n3, pack.rz3[lane]);
    atomic_add(rc_ptr, n0, pack.rc0[lane]);
    atomic_add(rc_ptr, n1, pack.rc1[lane]);
    atomic_add(rc_ptr, n2, pack.rc2[lane]);
    atomic_add(rc_ptr, n3, pack.rc3[lane]);
  }
}

static SFEM_NOINLINE void
cvfem_tet4_ns_upwind_apply(MeshData &d, const scalar_t rho, const scalar_t mu) {
  reset_residual(d);
  const ptrdiff_t ne = d.nelements;

#pragma omp parallel for schedule(static)
  for (ptrdiff_t begin = 0; begin < ne; begin += VEC_SIZE) {
    const int nlanes = int(std::min<ptrdiff_t>(ne - begin, VEC_SIZE));
    Tet4InputPack in;
    Tet4ResidualPack out;
    gather_tet4_pack(d, begin, nlanes, in);
    run_microkernel(d, rho, mu, begin, nlanes, in, out);
    scatter_tet4_pack(d, begin, nlanes, out);
  }
}

static scalar_t checksum(const MeshData &d) {
  scalar_t sum = 0.0;
  const ptrdiff_t stride = std::max<ptrdiff_t>(1, d.nnodes / 4096);
  for (ptrdiff_t i = 0; i < d.nnodes; i += stride) {
    const scalar_t w = 1.0 + scalar_t(i % 17) * 0.01;
    sum += w * (d.rx[i] + 1.3 * d.ry[i] + 1.7 * d.rz[i] + 2.1 * d.rc[i]);
  }
  return sum;
}

int main(int argc, char **argv) {
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  const int own_mpi = !mpi_initialized;
  if (own_mpi)
    MPI_Init(&argc, &argv);

  ptrdiff_t n = 48;
  int repeat = 20;
  int warmup = 3;
  scalar_t rho = 1.0;
  scalar_t mu = 0.01;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if ((arg == "--n" || arg == "--nx") && i + 1 < argc)
      n = parse_size(argv[++i]);
    else if (arg == "--repeat" && i + 1 < argc)
      repeat = std::atoi(argv[++i]);
    else if (arg == "--warmup" && i + 1 < argc)
      warmup = std::atoi(argv[++i]);
    else if (arg == "--rho" && i + 1 < argc)
      rho = std::atof(argv[++i]);
    else if (arg == "--mu" && i + 1 < argc)
      mu = std::atof(argv[++i]);
    else if (arg == "--help") {
      std::printf(
          "usage: %s [--n cube_cells_per_dim] [--repeat N] [--warmup N]\n",
          argv[0]);
      if (own_mpi)
        MPI_Finalize();
      return 0;
    }
  }

  MeshData d;
  d.mesh = smesh::Mesh::create_tet4_cube(smesh::Communicator::self(), n, n, n, 0,
                                         0, 0, 1, 1, 1);
  if (!d.mesh || d.mesh->element_type(0) != smesh::TET4) {
    std::fprintf(stderr, "failed to create TET4 mesh\n");
    d.mesh.reset();
    if (own_mpi)
      MPI_Finalize();
    return 1;
  }

  d.nnodes = d.mesh->n_nodes();
  d.nelements = d.mesh->n_elements(0);
  d.elems = d.mesh->elements(0)->data();
  d.points = d.mesh->points()->data();

  fill_fields(d);
  precompute_affine_geometry(d);

  for (int i = 0; i < warmup; ++i)
    cvfem_tet4_ns_upwind_apply(d, rho, mu);

  const double t0 = wall_time();
  for (int i = 0; i < repeat; ++i)
    cvfem_tet4_ns_upwind_apply(d, rho, mu);
  const double t1 = wall_time();

  const double seconds = t1 - t0;
  const double seconds_per_call = seconds / double(repeat);
  const double elem_apps = double(d.nelements) * double(repeat);

  constexpr double flops_inv_det = 1.0;
  constexpr double flops_ref_diff = 9.0;
  constexpr double flops_grad = 9.0 * 6.0;
  constexpr double flops_area = 3.0 * 15.0 + 3.0 * 9.0;
  constexpr double flops_scs_body = 6.0 + 6.0 + 4.0 + 2.0 + 27.0 + 18.0 + 8.0;
  constexpr double flops_per_element =
      flops_inv_det + flops_ref_diff + flops_grad + flops_area +
      6.0 * flops_scs_body;
  static_assert(flops_per_element == 562.0,
                "element flop model drifted from kernel");
  constexpr double bytes_per_element = 8.0 * double(sizeof(smesh::idx_t)) +
                                       10.0 * double(sizeof(jacobian_t)) +
                                       (16.0 + 32.0) * double(sizeof(scalar_t));
  constexpr double dofs_per_element_visit = 4.0 * 4.0;

  const double melems = double(d.nelements) / seconds_per_call / 1.0e6;
  const double mdofs =
      double(d.nelements) * dofs_per_element_visit / seconds_per_call / 1.0e6;
  const double unique_mdofs = double(d.nnodes) * 4.0 / seconds_per_call / 1.0e6;
  const double gflops = elem_apps * flops_per_element / seconds / 1.0e9;
  const double gbps = elem_apps * bytes_per_element / seconds / 1.0e9;

  std::printf("cvfem_tet4_ns_upwind_smesh_gather_simd_atomic_scatter\n");
  std::printf("  mesh_manager: smesh::Mesh::create_tet4_cube\n");
  std::printf("  OpenMP_threads: %d\n", threads_active());
  std::printf("  LANE_PACK_BYTES: %d\n", VEC_BYTES);
  std::printf("  LANES_PER_PACK: %d\n", VEC_SIZE);
  std::printf("  ALIGN_BYTES: %d\n", ALIGN_BYTES);
  std::printf("  geom_t_bytes: %zu\n", sizeof(smesh::geom_t));
  std::printf("  jacobian_t_bytes: %zu\n", sizeof(jacobian_t));
  std::printf("  cube_n: %td\n", n);
  std::printf("  nodes: %td\n", d.nnodes);
  std::printf("  elements: %td\n", d.nelements);
  std::printf("  repeat: %d\n", repeat);
  std::printf("  seconds_per_apply: %.6e\n", seconds_per_call);
  std::printf("  MELEM/s: %.3f\n", melems);
  std::printf("  MDOF/s_element_visits: %.3f\n", mdofs);
  std::printf("  MDOF/s_unique_mesh_dofs: %.3f\n", unique_mdofs);
  std::printf("  GFLOP/s_model: %.3f\n", gflops);
  std::printf("  GB/s_gather_scatter_model: %.3f\n", gbps);
  std::printf("  flops_per_element_model: %.1f\n", flops_per_element);
  std::printf("  bytes_per_element_model: %.1f\n", bytes_per_element);
  std::printf("  checksum: %.16e\n", checksum(d));

  d.mesh.reset();
  if (own_mpi)
    MPI_Finalize();
  return 0;
}
