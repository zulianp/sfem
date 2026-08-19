#include "smesh_mesh.hpp"
#include "smesh_buffer.hpp"
#include "sfem_BSR.hpp"

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

#include "cvfem_tet4_ns_upwind_kernels.hpp"

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

    pack.ux[0][lane] = ux[n0];
    pack.ux[1][lane] = ux[n1];
    pack.ux[2][lane] = ux[n2];
    pack.ux[3][lane] = ux[n3];
    pack.uy[0][lane] = uy[n0];
    pack.uy[1][lane] = uy[n1];
    pack.uy[2][lane] = uy[n2];
    pack.uy[3][lane] = uy[n3];
    pack.uz[0][lane] = uz[n0];
    pack.uz[1][lane] = uz[n1];
    pack.uz[2][lane] = uz[n2];
    pack.uz[3][lane] = uz[n3];
    pack.p[0][lane] = p[n0];
    pack.p[1][lane] = p[n1];
    pack.p[2][lane] = p[n2];
    pack.p[3][lane] = p[n3];
  }
}

static SFEM_INLINE void run_microkernel(const MeshData &d, const scalar_t rho,
                                        const scalar_t mu,
                                        const ptrdiff_t begin, const int nlanes,
                                        const Tet4InputPack &in,
                                        Tet4ResidualPack &out) {
  cvfem_run_residual_kernel(rho, mu, d.adj[0].data() + begin,
                            d.adj[1].data() + begin, d.adj[2].data() + begin,
                            d.adj[3].data() + begin, d.adj[4].data() + begin,
                            d.adj[5].data() + begin, d.adj[6].data() + begin,
                            d.adj[7].data() + begin, d.adj[8].data() + begin,
                            d.det.data() + begin, nlanes, in, out);
}

static SFEM_INLINE void run_jacobian_kernel(const MeshData &d, const scalar_t rho,
                                            const scalar_t mu,
                                            const ptrdiff_t begin,
                                            const int nlanes,
                                            const Tet4InputPack &in,
                                            scalar_t Ke[VEC_SIZE][CVFEM_N_DOF * CVFEM_N_DOF]) {
  cvfem_run_jacobian_kernel(rho, mu, d.adj[0].data() + begin,
                            d.adj[1].data() + begin, d.adj[2].data() + begin,
                            d.adj[3].data() + begin, d.adj[4].data() + begin,
                            d.adj[5].data() + begin, d.adj[6].data() + begin,
                            d.adj[7].data() + begin, d.adj[8].data() + begin,
                            d.det.data() + begin, nlanes, in, Ke);
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

    atomic_add(rx_ptr, n0, pack.rx[0][lane]);
    atomic_add(rx_ptr, n1, pack.rx[1][lane]);
    atomic_add(rx_ptr, n2, pack.rx[2][lane]);
    atomic_add(rx_ptr, n3, pack.rx[3][lane]);
    atomic_add(ry_ptr, n0, pack.ry[0][lane]);
    atomic_add(ry_ptr, n1, pack.ry[1][lane]);
    atomic_add(ry_ptr, n2, pack.ry[2][lane]);
    atomic_add(ry_ptr, n3, pack.ry[3][lane]);
    atomic_add(rz_ptr, n0, pack.rz[0][lane]);
    atomic_add(rz_ptr, n1, pack.rz[1][lane]);
    atomic_add(rz_ptr, n2, pack.rz[2][lane]);
    atomic_add(rz_ptr, n3, pack.rz[3][lane]);
    atomic_add(rc_ptr, n0, pack.rc[0][lane]);
    atomic_add(rc_ptr, n1, pack.rc[1][lane]);
    atomic_add(rc_ptr, n2, pack.rc[2][lane]);
    atomic_add(rc_ptr, n3, pack.rc[3][lane]);
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

struct BSR4 {
  std::shared_ptr<smesh::Mesh::NodeToNodeGraph> graph;
  const smesh::count_t *rowptr{nullptr};
  const smesh::idx_t *colidx{nullptr};
  smesh::SharedBuffer<scalar_t> values;
  ptrdiff_t nnz{0};
};

static BSR4 make_bsr4(const std::shared_ptr<smesh::Mesh> &mesh) {
  BSR4 b;
  b.graph = mesh->node_to_node_graph();
  b.rowptr = b.graph->rowptr()->data();
  b.colidx = b.graph->colidx()->data();
  b.nnz = b.graph->nnz();
  b.values = smesh::create_host_buffer<scalar_t>((size_t)b.nnz * 16);
  return b;
}

static void zero_bsr4(BSR4 &b) { cvfem_zero_scalars(b.values->data(), b.nnz * 16); }

static SFEM_NOINLINE void assemble_bsr4_atomic(MeshData &d, BSR4 &b,
                                               const scalar_t rho,
                                               const scalar_t mu) {
  zero_bsr4(b);
  const ptrdiff_t ne = d.nelements;
  smesh::idx_t **const SFEM_RESTRICT elems = d.elems;
  scalar_t *const SFEM_RESTRICT values = b.values->data();

#pragma omp parallel for schedule(static)
  for (ptrdiff_t begin = 0; begin < ne; begin += VEC_SIZE) {
    const int nlanes = int(std::min<ptrdiff_t>(ne - begin, VEC_SIZE));
    Tet4InputPack in;
    alignas(ALIGN_BYTES) scalar_t Ke[VEC_SIZE][CVFEM_N_DOF * CVFEM_N_DOF];
    gather_tet4_pack(d, begin, nlanes, in);
    run_jacobian_kernel(d, rho, mu, begin, nlanes, in, Ke);
    for (int lane = 0; lane < nlanes; ++lane) {
      const ptrdiff_t e = begin + lane;
      const smesh::idx_t ev[4] = {elems[0][e], elems[1][e], elems[2][e],
                                  elems[3][e]};
      tet4_local_to_global_bsr4<true>(ev, Ke[lane], b.rowptr, b.colidx, values);
    }
  }
}

static void pack_state(const MeshData &d, std::vector<scalar_t> &x) {
  x.resize((size_t)d.nnodes * 4);
  for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
    x[(size_t)i * 4 + 0] = d.ux[i];
    x[(size_t)i * 4 + 1] = d.uy[i];
    x[(size_t)i * 4 + 2] = d.uz[i];
    x[(size_t)i * 4 + 3] = d.p[i];
  }
}

static void unpack_residual(const MeshData &d, std::vector<scalar_t> &r) {
  r.resize((size_t)d.nnodes * 4);
  for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
    r[(size_t)i * 4 + 0] = d.rx[i];
    r[(size_t)i * 4 + 1] = d.ry[i];
    r[(size_t)i * 4 + 2] = d.rz[i];
    r[(size_t)i * 4 + 3] = d.rc[i];
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
  int assemble = 0;
  int verify_jac = 0;

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
    else if (arg == "--assemble")
      assemble = 1;
    else if (arg == "--verify-jac")
      verify_jac = 1;
    else if (arg == "--help") {
      std::printf("usage: %s [--n cube_cells_per_dim] [--repeat N] [--warmup N]\n"
                  "          [--assemble] [--verify-jac]\n",
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

  BSR4 bsr;
  if (assemble || verify_jac)
    bsr = make_bsr4(d.mesh);

  if (verify_jac) {
    assemble_bsr4_atomic(d, bsr, rho, mu);
    std::vector<scalar_t> x0, rm, rp, dir((size_t)d.nnodes * 4, 0.0),
        jv((size_t)d.nnodes * 4, 0.0);
    pack_state(d, x0);
    // Pressure probe: mixed velocity FD hits |mdot|=0 kinks in the upwind flux.
    for (ptrdiff_t i = 0; i < d.nnodes; ++i)
      dir[(size_t)i * 4 + 3] = 1.0;
    const scalar_t eps = 1.0e-6;
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
      d.ux[i] = x0[(size_t)i * 4 + 0] - eps * dir[(size_t)i * 4 + 0];
      d.uy[i] = x0[(size_t)i * 4 + 1] - eps * dir[(size_t)i * 4 + 1];
      d.uz[i] = x0[(size_t)i * 4 + 2] - eps * dir[(size_t)i * 4 + 2];
      d.p[i] = x0[(size_t)i * 4 + 3] - eps * dir[(size_t)i * 4 + 3];
    }
    cvfem_tet4_ns_upwind_apply(d, rho, mu);
    unpack_residual(d, rm);
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
      d.ux[i] = x0[(size_t)i * 4 + 0] + eps * dir[(size_t)i * 4 + 0];
      d.uy[i] = x0[(size_t)i * 4 + 1] + eps * dir[(size_t)i * 4 + 1];
      d.uz[i] = x0[(size_t)i * 4 + 2] + eps * dir[(size_t)i * 4 + 2];
      d.p[i] = x0[(size_t)i * 4 + 3] + eps * dir[(size_t)i * 4 + 3];
    }
    cvfem_tet4_ns_upwind_apply(d, rho, mu);
    unpack_residual(d, rp);
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
      d.ux[i] = x0[(size_t)i * 4 + 0];
      d.uy[i] = x0[(size_t)i * 4 + 1];
      d.uz[i] = x0[(size_t)i * 4 + 2];
      d.p[i] = x0[(size_t)i * 4 + 3];
    }
    auto spmv = sfem::h_bsr_spmv<smesh::count_t, smesh::idx_t, scalar_t>(
        d.nnodes, d.nnodes, 4, bsr.graph->rowptr(), bsr.graph->colidx(),
        bsr.values, scalar_t(0));
    spmv->apply(dir.data(), jv.data());
    scalar_t max_fd = 0.0, max_diff = 0.0;
    for (ptrdiff_t i = 0; i < d.nnodes * 4; ++i) {
      const scalar_t fd = (rp[(size_t)i] - rm[(size_t)i]) / (2.0 * eps);
      max_fd = std::max(max_fd, std::fabs(fd));
      max_diff = std::max(max_diff, std::fabs(jv[(size_t)i] - fd));
    }
    const scalar_t rel = max_diff / std::max(max_fd, 1.0e-30);
    std::printf("verify_jac_spmv_vs_fd_rel: %.6e\n", rel);
    std::printf("verify_jac_spmv_vs_fd_abs: %.6e\n", max_diff);
    if (rel > 1.0e-6) {
      std::fprintf(stderr, "SpMV J d vs FD residual mismatch\n");
      d.mesh.reset();
      if (own_mpi)
        MPI_Finalize();
      return 1;
    }
  }

  if (assemble) {
    for (int i = 0; i < warmup; ++i)
      assemble_bsr4_atomic(d, bsr, rho, mu);
  } else {
    for (int i = 0; i < warmup; ++i)
      cvfem_tet4_ns_upwind_apply(d, rho, mu);
  }

  const double t0 = wall_time();
  if (assemble) {
    for (int i = 0; i < repeat; ++i)
      assemble_bsr4_atomic(d, bsr, rho, mu);
  } else {
    for (int i = 0; i < repeat; ++i)
      cvfem_tet4_ns_upwind_apply(d, rho, mu);
  }
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
  if (assemble || verify_jac) {
    std::printf("  bsr_nnz: %td\n", bsr.nnz);
    std::printf("  flops_per_element_jacobian_model: %.1f\n",
                CVFEM_JACOBIAN_FLOPS_PER_ELEMENT);
  }
  if (assemble) {
    const double bytes_assemble =
        double(bsr.nnz) * 16.0 * 8.0 +
        double(d.nelements) * (8.0 * double(sizeof(smesh::idx_t)) +
                               10.0 * double(sizeof(jacobian_t)) +
                               16.0 * double(sizeof(scalar_t)));
    std::printf("  seconds_per_assemble: %.6e\n", seconds_per_call);
    std::printf("  MELEM/s_assemble: %.3f\n", melems);
    std::printf("  GB/s_assemble_model: %.3f\n",
                double(repeat) * bytes_assemble / seconds / 1.0e9);
    std::printf("  GFLOP/s_assemble_model: %.3f\n",
                elem_apps * CVFEM_JACOBIAN_FLOPS_PER_ELEMENT / seconds / 1.0e9);
  }

  d.mesh.reset();
  if (own_mpi)
    MPI_Finalize();
  return 0;
}


