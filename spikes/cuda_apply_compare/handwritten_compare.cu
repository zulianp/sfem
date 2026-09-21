// The generated kernel against the one SFEM wrote by hand, same mesh, same
// geometry, same answer asked for.
//
// `cu_tet4_laplacian_apply` takes a flat `fff` with a stride; the generated
// `cu_laplace_tet4_gradient_a_msoa` takes the six symmetric-metric components
// as separate pointers.  One buffer serves both -- component `c` is
// `flat + c * nelements` -- so neither gets a layout the other does not, and
// the geometry is bit-identical between the two runs.
//
// With kappa = 1 the two compute the same thing: SFEM's Laplacian apply, and
// the gradient of the generated Laplacian energy, which for a linear operator
// is that same matrix-vector product.  So this is a correctness check against
// the reference implementation as well as a rate comparison.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

#include <cuda_runtime.h>

#include "cu_tet4_laplacian.hpp"
#include "cu_tet4_linear_elasticity.hpp"

extern "C" int cu_laplace_tet4_gradient_a_msoa(
    const int, const ptrdiff_t, const ptrdiff_t, idx_t **,
    const geom_t *, const geom_t *, const geom_t *,
    const geom_t *, const geom_t *, const geom_t *,
    const real_t, const ptrdiff_t, const void *, const ptrdiff_t, void *, void *const);

extern "C" int cu_linear_elasticity_tet4_apply_a_msoa(
    const int, const ptrdiff_t, const ptrdiff_t, idx_t **,
    const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *,
    const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *,
    const real_t, const real_t,
    const ptrdiff_t, const void *, const void *, const void *,
    const ptrdiff_t, void *, void *, void *, void *const);

#define CHECK(call)                                                            \
  do {                                                                         \
    const cudaError_t status = (call);                                         \
    if (status != cudaSuccess) {                                               \
      std::fprintf(stderr, "cuda: %s at %d\n", cudaGetErrorString(status),     \
                   __LINE__);                                                  \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

template <class T> static T *upload(const std::vector<T> &host) {
  T *device = nullptr;
  CHECK(cudaMalloc((void **)&device, host.size() * sizeof(T)));
  CHECK(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
  return device;
}
template <class T> static void download(std::vector<T> &host, const T *device) {
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(host.data(), device, host.size() * sizeof(T), cudaMemcpyDeviceToHost));
}

static double seconds() {
  return std::chrono::duration<double>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

static unsigned long long rng = 0x9e3779b97f4a7c15ull;
static double rnd() {
  rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
  return (double)((rng >> 11) & ((1ull << 53) - 1)) / (double)(1ull << 53);
}

//: One comparison's result, so the two cases report identically.
struct Comparison {
  double hand;
  double generated;
  double worst;
  ptrdiff_t nonzero;
};

static void report(const char *label, const Comparison &c, ptrdiff_t side,
                   ptrdiff_t nelements, ptrdiff_t ndof) {
  std::printf("%-22s hand-written  side %4td  elements %10td  ndof %10td  %8.4f ms  %9.1f MDOF/s\n",
              label, side, nelements, ndof, 1e3 * c.hand, 1e-6 * (double)ndof / c.hand);
  std::printf("%-22s generated     side %4td  elements %10td  ndof %10td  %8.4f ms  %9.1f MDOF/s\n",
              label, side, nelements, ndof, 1e3 * c.generated,
              1e-6 * (double)ndof / c.generated);
  std::printf("%-22s generated / hand-written %.3fx   nonzero %td/%td   worst rel diff %.3e\n",
              "", c.hand / c.generated, c.nonzero, ndof, c.worst);
}

static Comparison measure(const std::function<void()> &hand,
                          const std::function<void()> &generated,
                          int repeats) {
  Comparison c{1e30, 1e30, 0.0, 0};
  for (int r = 0; r <= repeats; ++r) {
    double t0 = seconds(); hand(); const double a = seconds() - t0;
    t0 = seconds(); generated(); const double b = seconds() - t0;
    if (r > 0) {  // the first pass is the warm-up
      c.hand = std::min(c.hand, a);
      c.generated = std::min(c.generated, b);
    }
  }
  return c;
}

static void difference(Comparison &c, const std::vector<real_t> &a,
                       const std::vector<real_t> &b) {
  double scale = 0.0, worst = 0.0;
  for (size_t v = 0; v < a.size(); ++v) {
    scale = std::max(scale, std::fabs(a[v]));
    worst = std::max(worst, std::fabs(a[v] - b[v]));
    c.nonzero += a[v] != 0.0;
  }
  c.worst = std::max(c.worst, worst / (scale == 0.0 ? 1.0 : scale));
}

int main(int argc, char **argv) {
  const ptrdiff_t side = argc > 1 ? std::atol(argv[1]) : 128;
  const int repeats = argc > 2 ? std::atoi(argv[2]) : 5;

  // the Kuhn subdivision of a structured lattice: six tetrahedra per cell,
  // lexicographic node numbering, which is the locality a real mesh has
  const ptrdiff_t np = side + 1;
  const ptrdiff_t nnodes = np * np * np;
  const ptrdiff_t ncells = side * side * side;
  const ptrdiff_t nelements = 6 * ncells;
  static const int KUHN[6][4] = {{0, 1, 3, 7}, {0, 1, 5, 7}, {0, 2, 3, 7},
                                 {0, 2, 6, 7}, {0, 4, 5, 7}, {0, 4, 6, 7}};
  std::vector<std::vector<idx_t>> connectivity(4, std::vector<idx_t>(nelements));
  for (ptrdiff_t cell = 0; cell < ncells; ++cell) {
    const ptrdiff_t i = cell % side, j = (cell / side) % side, k = cell / (side * side);
    for (int t = 0; t < 6; ++t)
      for (int s = 0; s < 4; ++s) {
        const int corner = KUHN[t][s];
        connectivity[s][6 * cell + t] =
            (i + (corner & 1)) + np * ((j + ((corner >> 1) & 1)) +
                                       np * (k + ((corner >> 2) & 1)));
      }
  }
  std::vector<idx_t *> columns(4);
  for (int s = 0; s < 4; ++s) columns[s] = upload(connectivity[s]);
  idx_t **elements = upload(columns);

  // one flat metric, component-major with an `nelements` stride: the
  // hand-written kernel reads it as `fff` with that stride, the generated one
  // as six pointers into it
  std::vector<geom_t> flat(6 * nelements);
  for (ptrdiff_t e = 0; e < nelements; ++e) {
    double L[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b <= a; ++b) L[a][b] = (a == b) ? 0.5 + rnd() : 0.4 * (2.0 * rnd() - 1.0);
    int c = 0;
    for (int a = 0; a < 3; ++a)
      for (int b = a; b < 3; ++b) {
        double m = 0.0;
        for (int q = 0; q < 3; ++q) m += L[a][q] * L[b][q];
        flat[(c++) * nelements + e] = (geom_t)m;
      }
  }
  geom_t *fff = upload(flat);

  std::vector<real_t> x_host(nnodes);
  for (ptrdiff_t v = 0; v < nnodes; ++v) x_host[v] = 2.0 * rnd() - 1.0;
  real_t *x = upload(x_host);
  const std::vector<real_t> zero(nnodes, 0.0);

  // ---------------------------------------------------------- Laplacian
  {
    real_t *y_hand = upload(zero);
    real_t *y_gen = upload(zero);
    Comparison c = measure(
        [&] {
          CHECK(cudaMemset(y_hand, 0, nnodes * sizeof(real_t)));
          cu_tet4_laplacian_apply(nelements, elements, nelements, fff,
                                  smesh::SMESH_FLOAT64, x, y_hand, nullptr);
          CHECK(cudaDeviceSynchronize());
        },
        [&] {
          CHECK(cudaMemset(y_gen, 0, nnodes * sizeof(real_t)));
          cu_laplace_tet4_gradient_a_msoa(
              (int)sizeof(real_t), nelements, nnodes, elements,
              fff + 0 * nelements, fff + 1 * nelements, fff + 2 * nelements,
              fff + 3 * nelements, fff + 4 * nelements, fff + 5 * nelements,
              1.0, 1, x, 1, y_gen, nullptr);
          CHECK(cudaDeviceSynchronize());
        },
        repeats);
    std::vector<real_t> a(nnodes), b(nnodes);
    download(a, y_hand);
    download(b, y_gen);
    difference(c, a, b);
    report("laplace tet4", c, side, nelements, nnodes);
  }

  // -------------------------------------------------- linear elasticity
  //
  // The adjugate is one flat buffer again -- the hand-written kernel reads it
  // with an `nelements` stride, the generated one as nine component pointers
  // into the same memory.  Note the two spell their material parameters in
  // opposite order: `(mu, lambda)` and `(lmbda, mu)`.
  {
    std::vector<geom_t> adj(9 * nelements), det(nelements);
    for (ptrdiff_t e = 0; e < nelements; ++e) {
      double J[3][3];
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
          J[a][b] = (a == b ? 1.0 : 0.0) + 0.25 * (2.0 * rnd() - 1.0);
      det[e] = (geom_t)(J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1]) -
                        J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0]) +
                        J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]));
      const double a_[9] = {
           (J[1][1] * J[2][2] - J[1][2] * J[2][1]), -(J[0][1] * J[2][2] - J[0][2] * J[2][1]),
           (J[0][1] * J[1][2] - J[0][2] * J[1][1]), -(J[1][0] * J[2][2] - J[1][2] * J[2][0]),
           (J[0][0] * J[2][2] - J[0][2] * J[2][0]), -(J[0][0] * J[1][2] - J[0][2] * J[1][0]),
           (J[1][0] * J[2][1] - J[1][1] * J[2][0]), -(J[0][0] * J[2][1] - J[0][1] * J[2][0]),
           (J[0][0] * J[1][1] - J[0][1] * J[1][0])};
      for (int c = 0; c < 9; ++c) adj[(ptrdiff_t)c * nelements + e] = (geom_t)a_[c];
    }
    geom_t *g_adj = upload(adj);
    geom_t *g_det = upload(det);
    real_t *u[3], *hand[3], *gen[3];
    for (int c = 0; c < 3; ++c) {
      std::vector<real_t> field(nnodes);
      for (ptrdiff_t v = 0; v < nnodes; ++v) field[v] = 2.0 * rnd() - 1.0;
      u[c] = upload(field);
      hand[c] = upload(zero);
      gen[c] = upload(zero);
    }
    const real_t mu = 0.31, lambda = 0.77;
    Comparison c = measure(
        [&] {
          for (int k = 0; k < 3; ++k) CHECK(cudaMemset(hand[k], 0, nnodes * sizeof(real_t)));
          cu_tet4_linear_elasticity_apply(nelements, elements, nelements, g_adj, g_det,
                                          mu, lambda, smesh::SMESH_FLOAT64,
                                          1, u[0], u[1], u[2],
                                          1, hand[0], hand[1], hand[2], nullptr);
          CHECK(cudaDeviceSynchronize());
        },
        [&] {
          for (int k = 0; k < 3; ++k) CHECK(cudaMemset(gen[k], 0, nnodes * sizeof(real_t)));
          cu_linear_elasticity_tet4_apply_a_msoa(
              (int)sizeof(real_t), nelements, nnodes, elements,
              g_adj + 0 * nelements, g_adj + 1 * nelements, g_adj + 2 * nelements,
              g_adj + 3 * nelements, g_adj + 4 * nelements, g_adj + 5 * nelements,
              g_adj + 6 * nelements, g_adj + 7 * nelements, g_adj + 8 * nelements,
              g_det, lambda, mu, 1, u[0], u[1], u[2],
              1, gen[0], gen[1], gen[2], nullptr);
          CHECK(cudaDeviceSynchronize());
        },
        repeats);
    std::vector<real_t> a(nnodes), b(nnodes);
    for (int k = 0; k < 3; ++k) {
      download(a, hand[k]);
      download(b, gen[k]);
      difference(c, a, b);
    }
    report("linear elasticity tet4", c, side, nelements, 3 * nnodes);
  }

  return 0;
}
