// One driver, two compilers.  Built by g++ against the OpenMP tree and by nvcc
// against the CUDA tree, it calls the same generated entry point by the same
// name and writes the same output file, so the two can be diffed.  Compiling it
// twice is what keeps the comparison honest: there is no second implementation
// of the operator anywhere in here.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cstddef>

// The same fallback typedefs the generated headers use when sfem_base.hpp is
// not on the include path, so the driver and the kernel agree on every type
// without dragging in the library build.
typedef double real_t;
typedef ptrdiff_t idx_t;
typedef ptrdiff_t count_t;
typedef double geom_t;
#define SFEM_GENERATED_SCALAR_T

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define CUDA_OK(call)                                                            \
  do {                                                                           \
    cudaError_t status = (call);                                                 \
    if (status != cudaSuccess) {                                                 \
      std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__,                     \
                   cudaGetErrorString(status));                                  \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)
#endif

extern "C" int neohookean_ogden_proteus_hex8_apply_i_msoa(
    const int scalar_bytes, const ptrdiff_t nelements, const ptrdiff_t nnodes,
    idx_t **const elements, const geom_t *const *const points,
    const real_t lmbda, const real_t mu,
    const ptrdiff_t u_stride, const void *const ux, const void *const uy, const void *const uz,
    const ptrdiff_t h_stride, const void *const hx, const void *const hy, const void *const hz,
    const ptrdiff_t out_stride, void *const outx, void *const outy, void *const outz);

static const int NS = 8, ND = 3;

// A structured lattice of n^3 hexahedra, nodes numbered lexicographically with x
// fastest -- which is the order a PROTEUS element's micro-kernel is written in,
// so no permutation is needed here.
struct Mesh {
  ptrdiff_t n, nnodes, nelements;
  std::vector<idx_t> ev[8];
  std::vector<geom_t> px[3];
  explicit Mesh(ptrdiff_t side) : n(side) {
    const ptrdiff_t np = n + 1;
    nnodes = np * np * np;
    nelements = n * n * n;
    for (int c = 0; c < 3; ++c) px[c].resize(nnodes);
    for (ptrdiff_t k = 0; k < np; ++k)
      for (ptrdiff_t j = 0; j < np; ++j)
        for (ptrdiff_t i = 0; i < np; ++i) {
          const ptrdiff_t v = i + np * (j + np * k);
          px[0][v] = (geom_t)i / (geom_t)n;
          px[1][v] = (geom_t)j / (geom_t)n;
          px[2][v] = (geom_t)k / (geom_t)n;
        }
    for (int s = 0; s < 8; ++s) ev[s].resize(nelements);
    for (ptrdiff_t k = 0; k < n; ++k)
      for (ptrdiff_t j = 0; j < n; ++j)
        for (ptrdiff_t i = 0; i < n; ++i) {
          const ptrdiff_t e = i + n * (j + n * k);
          for (int s = 0; s < 8; ++s) {
            const ptrdiff_t di = s & 1, dj = (s >> 1) & 1, dk = (s >> 2) & 1;
            ev[s][e] = (idx_t)((i + di) + np * ((j + dj) + np * (k + dk)));
          }
        }
  }
};

int main(int argc, char **argv) {
  const ptrdiff_t side = argc > 1 ? std::atol(argv[1]) : 32;
  const int repeats = argc > 2 ? std::atoi(argv[2]) : 5;
  const std::string out_path = argc > 3 ? argv[3] : "apply_out.bin";
  Mesh m(side);
  const ptrdiff_t ndof = 3 * m.nnodes;

  std::vector<real_t> u[3], h[3], out[3];
  for (int c = 0; c < 3; ++c) {
    u[c].resize(m.nnodes);
    h[c].resize(m.nnodes);
    out[c].assign(m.nnodes, 0.0);
    for (ptrdiff_t v = 0; v < m.nnodes; ++v) {
      const double x = m.px[0][v], y = m.px[1][v], z = m.px[2][v];
      u[c][v] = 0.05 * std::sin((1.0 + c) * x + 2.0 * y + 0.5 * z);
      h[c][v] = 0.03 * std::sin(0.7 * x + (1.0 + c) * y + 1.3 * z);
    }
  }
  const real_t lmbda = 2.2, mu = 2.3333333333333335;

#ifdef __CUDACC__
  const char *where = "gh200 (device)";
  idx_t *d_ev[8]; geom_t *d_px[3]; real_t *d_u[3], *d_h[3], *d_out[3];
  idx_t **d_elements; geom_t **d_points;
  for (int s = 0; s < 8; ++s) {
    CUDA_OK(cudaMalloc(&d_ev[s], m.nelements * sizeof(idx_t)));
    CUDA_OK(cudaMemcpy(d_ev[s], m.ev[s].data(), m.nelements * sizeof(idx_t), cudaMemcpyHostToDevice));
  }
  for (int c = 0; c < 3; ++c) {
    CUDA_OK(cudaMalloc(&d_px[c], m.nnodes * sizeof(geom_t)));
    CUDA_OK(cudaMemcpy(d_px[c], m.px[c].data(), m.nnodes * sizeof(geom_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMalloc(&d_u[c], m.nnodes * sizeof(real_t)));
    CUDA_OK(cudaMemcpy(d_u[c], u[c].data(), m.nnodes * sizeof(real_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMalloc(&d_h[c], m.nnodes * sizeof(real_t)));
    CUDA_OK(cudaMemcpy(d_h[c], h[c].data(), m.nnodes * sizeof(real_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMalloc(&d_out[c], m.nnodes * sizeof(real_t)));
  }
  CUDA_OK(cudaMalloc(&d_elements, 8 * sizeof(idx_t *)));
  CUDA_OK(cudaMemcpy(d_elements, d_ev, 8 * sizeof(idx_t *), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMalloc(&d_points, 3 * sizeof(geom_t *)));
  CUDA_OK(cudaMemcpy(d_points, d_px, 3 * sizeof(geom_t *), cudaMemcpyHostToDevice));
  auto run = [&] {
    for (int c = 0; c < 3; ++c) CUDA_OK(cudaMemset(d_out[c], 0, m.nnodes * sizeof(real_t)));
    neohookean_ogden_proteus_hex8_apply_i_msoa(
        (int)sizeof(real_t), m.nelements, m.nnodes, d_elements,
        (const geom_t *const *)d_points, lmbda, mu,
        1, d_u[0], d_u[1], d_u[2], 1, d_h[0], d_h[1], d_h[2],
        1, d_out[0], d_out[1], d_out[2]);
    CUDA_OK(cudaDeviceSynchronize());
  };
#else
  const char *where = "host (OpenMP)";
  idx_t *h_ev[8]; const geom_t *h_px[3];
  for (int s = 0; s < 8; ++s) h_ev[s] = m.ev[s].data();
  for (int c = 0; c < 3; ++c) h_px[c] = m.px[c].data();
  auto run = [&] {
    for (int c = 0; c < 3; ++c) std::memset(out[c].data(), 0, m.nnodes * sizeof(real_t));
    neohookean_ogden_proteus_hex8_apply_i_msoa(
        (int)sizeof(real_t), m.nelements, m.nnodes, h_ev, h_px, lmbda, mu,
        1, u[0].data(), u[1].data(), u[2].data(),
        1, h[0].data(), h[1].data(), h[2].data(),
        1, out[0].data(), out[1].data(), out[2].data());
  };
#endif

  run();  // warm up, and this is the answer that gets written
  double best = 1e30;
  for (int r = 0; r < repeats; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    run();
    const auto t1 = std::chrono::steady_clock::now();
    best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
  }

#ifdef __CUDACC__
  for (int c = 0; c < 3; ++c)
    CUDA_OK(cudaMemcpy(out[c].data(), d_out[c], m.nnodes * sizeof(real_t), cudaMemcpyDeviceToHost));
#endif

  if (!out_path.empty() && out_path != "-") {
    FILE *f = std::fopen(out_path.c_str(), "wb");
    for (int c = 0; c < 3; ++c) std::fwrite(out[c].data(), sizeof(real_t), m.nnodes, f);
    std::fclose(f);
  }

  // Norms rather than a byte compare: the device sums its contributions in a
  // different order and through atomics, so the two answers agree to round-off
  // and not to the bit.  Printing both lets the comparison state its tolerance.
  double l2 = 0.0, linf = 0.0;
  for (int c = 0; c < 3; ++c)
    for (ptrdiff_t v = 0; v < m.nnodes; ++v) {
      const double value = (double)out[c][v];
      l2 += value * value;
      linf = std::max(linf, std::fabs(value));
    }
  l2 = std::sqrt(l2);

  std::printf("%-16s side %3ld  nelements %9ld  ndof %9ld  best %8.4f s  %9.2f MDOF/s  l2 %.17g  linf %.17g\n",
              where, (long)side, (long)m.nelements, (long)ndof,
              best, (double)ndof / best / 1e6, l2, linf);
  return 0;
}
