// The generated kernels, host against device: one source, compiled twice.
//
// Built by `g++` against an OpenMP generation and by `nvcc` against a CUDA
// generation of the same materials.  Both link the same `extern "C"` entry
// points under the same names and run the same mesh, so what the two runs
// differ by is the generated kernel and nothing else.  The device arm is
// selected by `__CUDACC__` and is only memory plumbing -- allocate, copy,
// launch, copy back.
//
// Two things are measured, and they want different meshes:
//
//   * **Agreement.**  Every kernel whose device lowering is worth doubting,
//     run on a small mesh with *shared* connectivity so the scatter is
//     contended and `atomicAdd` is exercised rather than bypassed.  Each
//     output vector is written out by name; `compare.py` differences the two
//     files entry by entry.  Printing each run's own norms instead -- which
//     this driver used to do -- cannot see two answers that share an L2 and
//     differ entrywise.
//
//   * **Throughput.**  A structured lattice, because a timing on scrambled
//     connectivity is not a result.  Hexahedra for the tensor-product apply,
//     and the Kuhn subdivision of the same lattice for the simplex gradient.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

typedef ptrdiff_t idx_t;
typedef double real_t;
typedef double geom_t;
#define SFEM_GENERATED_SCALAR_T

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define CUDA_OK(call)                                                            \
  do {                                                                           \
    const cudaError_t status = (call);                                           \
    if (status != cudaSuccess) {                                                 \
      std::fprintf(stderr, "cuda: %s at %s:%d\n", cudaGetErrorString(status),    \
                   __FILE__, __LINE__);                                          \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)
static const char *WHERE = "gh200 (device)";
//: SFEM names every device entry point `cu_` and ends it with a stream --
//: `cu_laplacian_apply`, `cu_linear_elasticity_apply` -- because a host and a
//: device implementation of one operator are two symbols in one library.  The
//: generated tree follows that now, so the two arms of this driver call two
//: different names with two different arities, and saying so here is the whole
//: of the difference.
#define SFEM_KERNEL(name) cu_##name
//: the declaration takes a type, the call takes a value
#define SFEM_STREAM_PARAM , void *const
#define SFEM_STREAM_ARG , nullptr
template <class T> static T *upload(const std::vector<T> &host) {
  T *device = nullptr;
  CUDA_OK(cudaMalloc((void **)&device, host.size() * sizeof(T)));
  CUDA_OK(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
  return device;
}
template <class T> static void download(std::vector<T> &host, T *device) {
  CUDA_OK(cudaDeviceSynchronize());
  CUDA_OK(cudaMemcpy(host.data(), device, host.size() * sizeof(T), cudaMemcpyDeviceToHost));
}
template <class T> static void clear(T *device, size_t count) {
  CUDA_OK(cudaMemset(device, 0, count * sizeof(T)));
}
static void sync() { CUDA_OK(cudaDeviceSynchronize()); }
#else
static const char *WHERE = "host (OpenMP)";
#define SFEM_KERNEL(name) name
#define SFEM_STREAM_PARAM
#define SFEM_STREAM_ARG
template <class T> static T *upload(const std::vector<T> &host) {
  T *copy = (T *)std::malloc(host.size() * sizeof(T));
  std::memcpy(copy, host.data(), host.size() * sizeof(T));
  return copy;
}
template <class T> static void download(std::vector<T> &host, T *device) {
  std::memcpy(host.data(), device, host.size() * sizeof(T));
}
template <class T> static void clear(T *device, size_t count) {
  std::memset(device, 0, count * sizeof(T));
}
static void sync() {}
#endif

extern "C" {
int SFEM_KERNEL(neohookean_ogden_proteus_hex8_apply_i_msoa)(
    const int, const ptrdiff_t, const ptrdiff_t, idx_t **const, const geom_t *const *const,
    const real_t, const real_t, const ptrdiff_t, const void *const, const void *const,
    const void *const, const ptrdiff_t, const void *const, const void *const, const void *const,
    const ptrdiff_t, void *const, void *const, void *const SFEM_STREAM_PARAM);
int SFEM_KERNEL(laplace_tri3_gradient_a_msoa)(const int, const ptrdiff_t, const ptrdiff_t, idx_t **,
    const geom_t *, const geom_t *, const geom_t *, const real_t,
    const ptrdiff_t, const void *, const ptrdiff_t, void * SFEM_STREAM_PARAM);
int SFEM_KERNEL(laplace_tri3_apply_a_msoa)(const int, const ptrdiff_t, const ptrdiff_t, idx_t **,
    const geom_t *, const geom_t *, const geom_t *, const real_t,
    const ptrdiff_t, const void *, const ptrdiff_t, void * SFEM_STREAM_PARAM);
int SFEM_KERNEL(laplace_tet4_gradient_a_msoa)(const int, const ptrdiff_t, const ptrdiff_t, idx_t **,
    const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *,
    const geom_t *, const real_t, const ptrdiff_t, const void *, const ptrdiff_t, void * SFEM_STREAM_PARAM);
int SFEM_KERNEL(laplace_tet4_apply_a_msoa)(const int, const ptrdiff_t, const ptrdiff_t, idx_t **,
    const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *,
    const geom_t *, const real_t, const ptrdiff_t, const void *, const ptrdiff_t, void * SFEM_STREAM_PARAM);
int SFEM_KERNEL(linear_elasticity_tet4_gradient_a_msoa_aos_unit)(const int, const ptrdiff_t, const ptrdiff_t,
    idx_t **, const geom_t *, const geom_t *, const real_t, const real_t,
    const ptrdiff_t, const void *, const void *, const void *,
    const ptrdiff_t, void *, void *, void * SFEM_STREAM_PARAM);
int SFEM_KERNEL(laplace_quad4_gradient_i_msoa)(const int, const ptrdiff_t, const ptrdiff_t, idx_t **,
    const geom_t *const *, const real_t, const ptrdiff_t, const void *, const ptrdiff_t, void * SFEM_STREAM_PARAM);
int SFEM_KERNEL(laplace_hex8_gradient_i_msoa)(const int, const ptrdiff_t, const ptrdiff_t, idx_t **,
    const geom_t *const *, const real_t, const ptrdiff_t, const void *, const ptrdiff_t, void * SFEM_STREAM_PARAM);
int SFEM_KERNEL(mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_residual_a_msoa)(
    const int, const ptrdiff_t, const ptrdiff_t, idx_t **,
    const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *,
    const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *,
    const real_t, const real_t, const real_t,
    const ptrdiff_t, const void *, const void *, const void *,
    const ptrdiff_t, const void *, const void *, const void *,
    const ptrdiff_t, void *, void *, void * SFEM_STREAM_PARAM);
int SFEM_KERNEL(neumann_tet4_trishell3_boundary_residual_soa)(
    const ptrdiff_t, const ptrdiff_t, idx_t **, const geom_t *const *,
    const real_t, const real_t, const real_t, const int,
    real_t *, real_t *, real_t * SFEM_STREAM_PARAM);
}

static double seconds() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// One stream with a fixed seed: both binaries must see byte-identical inputs,
// so nothing here may depend on the compiler's random number generator.
static unsigned long long rng_state = 0x9e3779b97f4a7c15ull;
static double rnd() {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 7;
  rng_state ^= rng_state << 17;
  return (double)((rng_state >> 11) & ((1ull << 53) - 1)) / (double)(1ull << 53);
}
static void reseed() { rng_state = 0x9e3779b97f4a7c15ull; }

static FILE *record_file = nullptr;
static void record(const char *name, const std::vector<real_t> &values) {
  if (record_file == nullptr) return;
  std::fprintf(record_file, "# %s %zu\n", name, values.size());
  for (size_t i = 0; i < values.size(); ++i) std::fprintf(record_file, "%.17g\n", values[i]);
}

static std::vector<real_t> random_field(ptrdiff_t n) {
  std::vector<real_t> field(n);
  for (ptrdiff_t i = 0; i < n; ++i) field[i] = 2.0 * rnd() - 1.0;
  return field;
}

// Shared connectivity: several elements land on the same node, which is the
// case the scatter has to survive.
static idx_t **shared_element_table(int n_shape, ptrdiff_t nelements, ptrdiff_t nnodes) {
  std::vector<idx_t *> columns(n_shape);
  for (int shape = 0; shape < n_shape; ++shape) {
    std::vector<idx_t> column(nelements);
    for (ptrdiff_t e = 0; e < nelements; ++e) {
      idx_t node = (idx_t)(rnd() * (double)nnodes);
      column[e] = node >= nnodes ? nnodes - 1 : node;
    }
    columns[shape] = upload(column);
  }
  return upload(columns);
}

// A symmetric positive-definite metric per element, L L^T with a positive
// diagonal, so the kernels see numbers a real mesh could produce.
static std::vector<std::vector<geom_t>> spd_metric(int dim, ptrdiff_t nelements) {
  const int n_component = dim == 2 ? 3 : 6;
  std::vector<std::vector<geom_t>> metric(n_component, std::vector<geom_t>(nelements));
  for (ptrdiff_t e = 0; e < nelements; ++e) {
    double L[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    for (int i = 0; i < dim; ++i)
      for (int j = 0; j <= i; ++j) L[i][j] = (i == j) ? 0.5 + rnd() : 0.4 * (2.0 * rnd() - 1.0);
    double M[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    for (int i = 0; i < dim; ++i)
      for (int j = 0; j < dim; ++j)
        for (int k = 0; k < dim; ++k) M[i][j] += L[i][k] * L[j][k];
    int c = 0;
    for (int i = 0; i < dim; ++i)
      for (int j = i; j < dim; ++j) metric[c++][e] = M[i][j];
  }
  return metric;
}

// A structured lattice of side^dim cells, nodes numbered lexicographically with
// x fastest -- which is the order a PROTEUS element's micro-kernel is written
// in, so no permutation is needed here.
struct Lattice {
  ptrdiff_t side, np, nnodes, ncells;
  std::vector<geom_t> px[3];
  Lattice(ptrdiff_t n, int dim) : side(n), np(n + 1) {
    nnodes = 1;
    ncells = 1;
    for (int d = 0; d < dim; ++d) { nnodes *= np; ncells *= n; }
    for (int c = 0; c < dim; ++c) px[c].resize(nnodes);
    for (ptrdiff_t node = 0; node < nnodes; ++node) {
      ptrdiff_t rest = node;
      for (int d = 0; d < dim; ++d) { px[d][node] = (geom_t)(rest % np) / (geom_t)n; rest /= np; }
    }
  }
  ptrdiff_t corner(const ptrdiff_t base[3], int code, int dim) const {
    ptrdiff_t node = 0, stride = 1;
    for (int d = 0; d < dim; ++d) { node += (base[d] + ((code >> d) & 1)) * stride; stride *= np; }
    return node;
  }
  void cell_base(ptrdiff_t cell, int dim, ptrdiff_t base[3]) const {
    ptrdiff_t rest = cell;
    for (int d = 0; d < dim; ++d) { base[d] = rest % side; rest /= side; }
  }
};

// The mesh-order corner sequence: counter-clockwise on the bottom face, then
// the top, which is 0,1,3,2 in cube-corner bits.
static const int MESH_ORDER[8] = {0, 1, 3, 2, 4, 5, 7, 6};

static std::vector<std::vector<idx_t>> lattice_connectivity(const Lattice &lattice, int dim,
                                                            bool mesh_order) {
  const int n_shape = 1 << dim;
  std::vector<std::vector<idx_t>> connectivity(n_shape, std::vector<idx_t>(lattice.ncells));
  for (ptrdiff_t cell = 0; cell < lattice.ncells; ++cell) {
    ptrdiff_t base[3] = {0, 0, 0};
    lattice.cell_base(cell, dim, base);
    for (int shape = 0; shape < n_shape; ++shape) {
      const int code = mesh_order ? MESH_ORDER[shape] : shape;
      connectivity[shape][cell] = lattice.corner(base, code, dim);
    }
  }
  return connectivity;
}

// The Kuhn subdivision: the six paths 0 -> e_a -> e_a + e_b -> 7 through a
// cube's corners, which tile it with six tetrahedra and keep the lattice's
// locality.
static std::vector<std::vector<idx_t>> kuhn_connectivity(const Lattice &lattice) {
  static const int KUHN[6][4] = {{0, 1, 3, 7}, {0, 1, 5, 7}, {0, 2, 3, 7},
                                 {0, 2, 6, 7}, {0, 4, 5, 7}, {0, 4, 6, 7}};
  std::vector<std::vector<idx_t>> connectivity(4, std::vector<idx_t>(6 * lattice.ncells));
  for (ptrdiff_t cell = 0; cell < lattice.ncells; ++cell) {
    ptrdiff_t base[3] = {0, 0, 0};
    lattice.cell_base(cell, 3, base);
    for (int t = 0; t < 6; ++t)
      for (int shape = 0; shape < 4; ++shape)
        connectivity[shape][6 * cell + t] = lattice.corner(base, KUHN[t][shape], 3);
  }
  return connectivity;
}

template <class T> static T **upload_table(const std::vector<std::vector<T>> &columns) {
  std::vector<T *> uploaded(columns.size());
  for (size_t c = 0; c < columns.size(); ++c) uploaded[c] = upload(columns[c]);
  return upload(uploaded);
}

// ---------------------------------------------------------------- agreement

static void agree_simplex_metric(int dim) {
  const ptrdiff_t nelements = 4096, nnodes = 1024;
  reseed();
  idx_t **elements = shared_element_table(dim == 2 ? 3 : 4, nelements, nnodes);
  std::vector<std::vector<geom_t>> metric = spd_metric(dim, nelements);
  std::vector<geom_t *> met(metric.size());
  for (size_t c = 0; c < metric.size(); ++c) met[c] = upload(metric[c]);
  real_t *u = upload(random_field(nnodes));
  const std::vector<real_t> zero(nnodes, 0.0);

  char name[128];
  for (int is_apply = 0; is_apply < 2; ++is_apply) {
    real_t *out = upload(zero);
    if (dim == 2) {
      (is_apply ? SFEM_KERNEL(laplace_tri3_apply_a_msoa)
                 : SFEM_KERNEL(laplace_tri3_gradient_a_msoa))(
          (int)sizeof(real_t), nelements, nnodes, elements, met[0], met[1], met[2],
          1.7, 1, u, 1, out SFEM_STREAM_ARG);
    } else {
      (is_apply ? SFEM_KERNEL(laplace_tet4_apply_a_msoa)
                 : SFEM_KERNEL(laplace_tet4_gradient_a_msoa))(
          (int)sizeof(real_t), nelements, nnodes, elements,
          met[0], met[1], met[2], met[3], met[4], met[5], 1.7, 1, u, 1, out SFEM_STREAM_ARG);
    }
    sync();
    std::vector<real_t> result(nnodes);
    download(result, out);
    std::snprintf(name, sizeof(name), "laplace_%s_%s_a_msoa", dim == 2 ? "tri3" : "tet4",
                  is_apply ? "apply" : "gradient");
    record(name, result);
  }
}

static void agree_linear_elasticity_tet4() {
  const ptrdiff_t nelements = 4096, nnodes = 1024;
  reseed();
  idx_t **elements = shared_element_table(4, nelements, nnodes);

  std::vector<geom_t> adj_aos(9 * nelements), det(nelements);
  for (ptrdiff_t e = 0; e < nelements; ++e) {
    double J[3][3];
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) J[i][j] = (i == j ? 1.0 : 0.0) + 0.25 * (2.0 * rnd() - 1.0);
    det[e] = J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1]) -
             J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0]) +
             J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);
    // the adjugate: the transposed cofactor matrix
    adj_aos[9 * e + 0] =  (J[1][1] * J[2][2] - J[1][2] * J[2][1]);
    adj_aos[9 * e + 1] = -(J[0][1] * J[2][2] - J[0][2] * J[2][1]);
    adj_aos[9 * e + 2] =  (J[0][1] * J[1][2] - J[0][2] * J[1][1]);
    adj_aos[9 * e + 3] = -(J[1][0] * J[2][2] - J[1][2] * J[2][0]);
    adj_aos[9 * e + 4] =  (J[0][0] * J[2][2] - J[0][2] * J[2][0]);
    adj_aos[9 * e + 5] = -(J[0][0] * J[1][2] - J[0][2] * J[1][0]);
    adj_aos[9 * e + 6] =  (J[1][0] * J[2][1] - J[1][1] * J[2][0]);
    adj_aos[9 * e + 7] = -(J[0][0] * J[2][1] - J[0][1] * J[2][0]);
    adj_aos[9 * e + 8] =  (J[0][0] * J[1][1] - J[0][1] * J[1][0]);
  }
  geom_t *g_adj = upload(adj_aos), *g_det = upload(det);
  real_t *ux = upload(random_field(nnodes));
  real_t *uy = upload(random_field(nnodes));
  real_t *uz = upload(random_field(nnodes));
  const std::vector<real_t> zero(nnodes, 0.0);
  real_t *ox = upload(zero), *oy = upload(zero), *oz = upload(zero);

  SFEM_KERNEL(linear_elasticity_tet4_gradient_a_msoa_aos_unit)(
      (int)sizeof(real_t), nelements, nnodes, elements, g_adj, g_det, 0.31, 0.77,
      1, ux, uy, uz, 1, ox, oy, oz SFEM_STREAM_ARG);
  sync();
  std::vector<real_t> rx(nnodes), ry(nnodes), rz(nnodes);
  download(rx, ox); download(ry, oy); download(rz, oz);
  record("linear_elasticity_tet4_gradient_a_msoa_aos_unit_x", rx);
  record("linear_elasticity_tet4_gradient_a_msoa_aos_unit_y", ry);
  record("linear_elasticity_tet4_gradient_a_msoa_aos_unit_z", rz);
}

// The viscous Mooney-Rivlin residual: the residual family, which reached CUDA
// only when the mesh-kernel lowering moved onto the target.  It reads a current
// and a previous state, so both are supplied.
static void agree_residual_tet4() {
  const ptrdiff_t nelements = 4096, nnodes = 1024;
  reseed();
  idx_t **elements = shared_element_table(4, nelements, nnodes);

  std::vector<geom_t> adj[9], det(nelements);
  for (int c = 0; c < 9; ++c) adj[c].resize(nelements);
  for (ptrdiff_t e = 0; e < nelements; ++e) {
    double J[3][3];
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) J[i][j] = (i == j ? 1.0 : 0.0) + 0.25 * (2.0 * rnd() - 1.0);
    det[e] = J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1]) -
             J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0]) +
             J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);
    adj[0][e] =  (J[1][1] * J[2][2] - J[1][2] * J[2][1]);
    adj[1][e] = -(J[0][1] * J[2][2] - J[0][2] * J[2][1]);
    adj[2][e] =  (J[0][1] * J[1][2] - J[0][2] * J[1][1]);
    adj[3][e] = -(J[1][0] * J[2][2] - J[1][2] * J[2][0]);
    adj[4][e] =  (J[0][0] * J[2][2] - J[0][2] * J[2][0]);
    adj[5][e] = -(J[0][0] * J[1][2] - J[0][2] * J[1][0]);
    adj[6][e] =  (J[1][0] * J[2][1] - J[1][1] * J[2][0]);
    adj[7][e] = -(J[0][0] * J[2][1] - J[0][1] * J[2][0]);
    adj[8][e] =  (J[0][0] * J[1][1] - J[0][1] * J[1][0]);
  }
  geom_t *g_adj[9];
  for (int c = 0; c < 9; ++c) g_adj[c] = upload(adj[c]);
  geom_t *g_det = upload(det);

  real_t *u[3], *u_old[3], *out[3];
  const std::vector<real_t> zero(nnodes, 0.0);
  for (int c = 0; c < 3; ++c) {
    u[c] = upload(random_field(nnodes));
    u_old[c] = upload(random_field(nnodes));
    out[c] = upload(zero);
  }
  SFEM_KERNEL(mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_residual_a_msoa)(
      (int)sizeof(real_t), nelements, nnodes, elements,
      g_adj[0], g_adj[1], g_adj[2], g_adj[3], g_adj[4], g_adj[5], g_adj[6], g_adj[7], g_adj[8],
      g_det, 0.41, 0.73, 0.6,
      1, u[0], u[1], u[2], 1, u_old[0], u_old[1], u_old[2],
      1, out[0], out[1], out[2] SFEM_STREAM_ARG);
  sync();
  std::vector<real_t> result(nnodes);
  char name[128];
  for (int c = 0; c < 3; ++c) {
    download(result, out[c]);
    std::snprintf(name, sizeof(name),
                  "mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_residual_a_msoa_%c", "xyz"[c]);
    record(name, result);
  }
}

// The Neumann traction on a triangle shell: the boundary-residual family, whose
// emitter wrote its loop, signature, scatter, return and launch inline.  It
// takes points, so it gets a lattice's surface rather than random connectivity.
static void agree_boundary_tet4() {
  reseed();
  const Lattice lattice(12, 2);
  const ptrdiff_t nfaces = 2 * lattice.ncells;
  std::vector<std::vector<idx_t>> connectivity(3, std::vector<idx_t>(nfaces));
  for (ptrdiff_t cell = 0; cell < lattice.ncells; ++cell) {
    ptrdiff_t base[3] = {0, 0, 0};
    lattice.cell_base(cell, 2, base);
    // the two triangles of each quad, both counter-clockwise
    static const int TRIANGLES[2][3] = {{0, 1, 3}, {0, 3, 2}};
    for (int t = 0; t < 2; ++t)
      for (int shape = 0; shape < 3; ++shape)
        connectivity[shape][2 * cell + t] = lattice.corner(base, TRIANGLES[t][shape], 2);
  }
  idx_t **elements = upload_table(connectivity);
  // a flat shell in 3D: the lattice's two coordinates, and zero for the third
  std::vector<std::vector<geom_t>> point_columns = {
      lattice.px[0], lattice.px[1], std::vector<geom_t>(lattice.nnodes, 0.0)};
  geom_t **points = upload_table(point_columns);

  real_t *out[3];
  const std::vector<real_t> zero(lattice.nnodes, 0.0);
  for (int c = 0; c < 3; ++c) out[c] = upload(zero);
  SFEM_KERNEL(neumann_tet4_trishell3_boundary_residual_soa)(
      nfaces, lattice.nnodes, elements, points, 0.3, -0.7, 1.1, 1,
      out[0], out[1], out[2] SFEM_STREAM_ARG);
  sync();
  std::vector<real_t> result(lattice.nnodes);
  char name[128];
  for (int c = 0; c < 3; ++c) {
    download(result, out[c]);
    std::snprintf(name, sizeof(name),
                  "neumann_tet4_trishell3_boundary_residual_soa_%c", "xyz"[c]);
    record(name, result);
  }
}

// These two take points rather than a metric, so they get a lattice: random
// coordinates would give inverted or degenerate cells.
static void agree_mesh_order_tensor_product(int dim) {
  reseed();
  const Lattice lattice(dim == 2 ? 24 : 10, dim);
  idx_t **elements = upload_table(lattice_connectivity(lattice, dim, true));
  std::vector<std::vector<geom_t>> point_columns(lattice.px, lattice.px + dim);
  geom_t **points = upload_table(point_columns);
  real_t *u = upload(random_field(lattice.nnodes));
  real_t *out = upload(std::vector<real_t>(lattice.nnodes, 0.0));

  if (dim == 2) {
    SFEM_KERNEL(laplace_quad4_gradient_i_msoa)((int)sizeof(real_t), lattice.ncells, lattice.nnodes,
                                  elements, points, 1.3, 1, u, 1, out SFEM_STREAM_ARG);
  } else {
    SFEM_KERNEL(laplace_hex8_gradient_i_msoa)((int)sizeof(real_t), lattice.ncells, lattice.nnodes,
                                 elements, points, 1.3, 1, u, 1, out SFEM_STREAM_ARG);
  }
  sync();
  std::vector<real_t> result(lattice.nnodes);
  download(result, out);
  record(dim == 2 ? "laplace_quad4_gradient_i_msoa" : "laplace_hex8_gradient_i_msoa", result);
}

// --------------------------------------------------------------- throughput

static void bench_hex8_apply(ptrdiff_t side, int repeats, const char *out_path) {
  const Lattice lattice(side, 3);
  idx_t **elements = upload_table(lattice_connectivity(lattice, 3, false));
  std::vector<std::vector<geom_t>> point_columns(lattice.px, lattice.px + 3);
  geom_t **points = upload_table(point_columns);

  std::vector<real_t> u_host[3], h_host[3];
  for (int c = 0; c < 3; ++c) {
    u_host[c].resize(lattice.nnodes);
    h_host[c].resize(lattice.nnodes);
    for (ptrdiff_t v = 0; v < lattice.nnodes; ++v) {
      const double x = lattice.px[0][v], y = lattice.px[1][v], z = lattice.px[2][v];
      u_host[c][v] = 0.05 * std::sin((1.0 + c) * x + 2.0 * y + 0.5 * z);
      h_host[c][v] = 0.03 * std::sin(0.7 * x + (1.0 + c) * y + 1.3 * z);
    }
  }
  real_t *u[3], *h[3], *out[3];
  const std::vector<real_t> zero(lattice.nnodes, 0.0);
  for (int c = 0; c < 3; ++c) {
    u[c] = upload(u_host[c]); h[c] = upload(h_host[c]); out[c] = upload(zero);
  }
  const real_t lmbda = 2.2, mu = 2.3333333333333335;

  double best = 1e30;
  for (int r = 0; r <= repeats; ++r) {
    for (int c = 0; c < 3; ++c) clear(out[c], (size_t)lattice.nnodes);
    const double t0 = seconds();
    SFEM_KERNEL(neohookean_ogden_proteus_hex8_apply_i_msoa)(
        (int)sizeof(real_t), lattice.ncells, lattice.nnodes, elements, points, lmbda, mu,
        1, u[0], u[1], u[2], 1, h[0], h[1], h[2], 1, out[0], out[1], out[2] SFEM_STREAM_ARG);
    sync();
    const double elapsed = seconds() - t0;
    if (r > 0) best = std::min(best, elapsed);  // the first pass is the warm-up
  }
  const ptrdiff_t ndof = 3 * lattice.nnodes;
  std::printf("%-16s neohookean proteus_hex8 apply  side %4td  elements %10td  ndof %10td"
              "  %8.4f ms  %9.1f MDOF/s\n",
              WHERE, side, lattice.ncells, ndof, 1e3 * best, 1e-6 * (double)ndof / best);
  std::fflush(stdout);

  if (out_path != nullptr && std::strcmp(out_path, "-") != 0) {
    std::vector<real_t> result(lattice.nnodes);
    char name[128];
    for (int c = 0; c < 3; ++c) {
      download(result, out[c]);
      std::snprintf(name, sizeof(name), "neohookean_ogden_proteus_hex8_apply_i_msoa_%c", "xyz"[c]);
      record(name, result);
    }
  }
}

static void bench_tet4_gradient(ptrdiff_t side, int repeats) {
  reseed();
  const Lattice lattice(side, 3);
  idx_t **elements = upload_table(kuhn_connectivity(lattice));
  const ptrdiff_t nelements = 6 * lattice.ncells;
  std::vector<std::vector<geom_t>> metric = spd_metric(3, nelements);
  std::vector<geom_t *> met(6);
  for (int c = 0; c < 6; ++c) met[c] = upload(metric[c]);
  real_t *u = upload(random_field(lattice.nnodes));
  real_t *out = upload(std::vector<real_t>(lattice.nnodes, 0.0));

  double best = 1e30;
  for (int r = 0; r <= repeats; ++r) {
    clear(out, (size_t)lattice.nnodes);
    const double t0 = seconds();
    SFEM_KERNEL(laplace_tet4_gradient_a_msoa)((int)sizeof(real_t), nelements, lattice.nnodes, elements,
                                 met[0], met[1], met[2], met[3], met[4], met[5],
                                 1.7, 1, u, 1, out SFEM_STREAM_ARG);
    sync();
    const double elapsed = seconds() - t0;
    if (r > 0) best = std::min(best, elapsed);
  }
  std::printf("%-16s laplace tet4 gradient          side %4td  elements %10td  ndof %10td"
              "  %8.4f ms  %9.1f MDOF/s\n",
              WHERE, side, nelements, lattice.nnodes, 1e3 * best,
              1e-6 * (double)lattice.nnodes / best);
  std::fflush(stdout);
}

int main(int argc, char **argv) {
  const ptrdiff_t side = argc > 1 ? std::atol(argv[1]) : 32;
  const int repeats = argc > 2 ? std::atoi(argv[2]) : 5;
  const char *out_path = argc > 3 ? argv[3] : "-";

  if (std::strcmp(out_path, "-") != 0) {
    record_file = std::fopen(out_path, "w");
    if (record_file == nullptr) {
      std::fprintf(stderr, "cannot write %s\n", out_path);
      return 1;
    }
    std::fprintf(record_file, "# where %s\n", WHERE);
    agree_simplex_metric(2);
    agree_simplex_metric(3);
    agree_linear_elasticity_tet4();
    agree_residual_tet4();
    agree_boundary_tet4();
    agree_mesh_order_tensor_product(2);
    agree_mesh_order_tensor_product(3);
  }

  bench_hex8_apply(side, repeats, out_path);
  bench_tet4_gradient(side, repeats);

  if (record_file != nullptr) {
    std::fclose(record_file);
    std::printf("%-16s wrote %s\n", WHERE, out_path);
  }
  return 0;
}
