// Host driver for the CUDA smoke test.
//
// Compiled by the ordinary C++ compiler, not nvcc -- it only sees the C ABI. It runs
// the CVFEM HEX8 residual on the host and on the device for the same element and
// compares, in double and in float.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using scalar_t = double;
#ifndef SFEM_RESTRICT
#define SFEM_RESTRICT __restrict__
#endif

#include "cvfem_hex8_ns_upwind_kernels.hpp"

#include "cvfem_cuda_smoke.hpp"

namespace {

// A mildly deformed unit cube, so the adjugate is not trivially diagonal and the
// convective terms are actually exercised.
template <typename T>
void make_element(T *adj, T &det, T *ux, T *uy, T *uz, T *p) {
    const T x[8] = {0, 1, 1, 0, 0, 1, 1, 0};
    const T y[8] = {0, 0, 1, 1, 0, 0, 1, 1};
    const T z[8] = {0, 0, 0, 0, 1, 1, 1, 1};
    cvfem_hex8_affine_adj(x, y, z, adj, &det);

    for (int a = 0; a < 8; ++a) {
        const T s = T(a + 1) / T(8);
        ux[a] = T(0.3) + T(0.7) * s;
        uy[a] = T(-0.2) + T(0.5) * s * s;
        uz[a] = T(0.1) * s - T(0.05);
        p[a]  = T(1.5) - T(0.25) * s;
    }
}

template <typename T>
double max_abs_diff(const T *a, const T *b, int n) {
    double m = 0;
    for (int i = 0; i < n; ++i) m = std::fmax(m, std::fabs(double(a[i]) - double(b[i])));
    return m;
}

template <typename T>
int check(const char *label, double tol) {
    T adj[9], det, ux[8], uy[8], uz[8], p[8];
    make_element<T>(adj, det, ux, uy, uz, p);

    T host_r[CVFEM_HEX8_N_DOF];
    cvfem_hex8_ns_upwind_residual(T(1.0), T(0.01), adj, det, ux, uy, uz, p, host_r);

    const size_t nelements = 1024;  // also checks that every thread gets the same answer
    std::vector<T> dev_r(nelements * CVFEM_HEX8_N_DOF, T(0));

    int rc;
    if (sizeof(T) == sizeof(double)) {
        rc = cvfem_cuda_smoke_residual(nelements, (double)1.0, (double)0.01,
                                       (const double *)adj, (double)det,
                                       (const double *)ux, (const double *)uy,
                                       (const double *)uz, (const double *)p,
                                       (double *)dev_r.data(), nullptr);
    } else {
        rc = cvfem_cuda_smoke_residual_f32(nelements, (float)1.0f, (float)0.01f,
                                           (const float *)adj, (float)det,
                                           (const float *)ux, (const float *)uy,
                                           (const float *)uz, (const float *)p,
                                           (float *)dev_r.data(), nullptr);
    }
    if (rc != 0) { std::printf("  %-8s FAILED: device call returned %d\n", label, rc); return 1; }

    double worst = 0;
    for (size_t e = 0; e < nelements; ++e)
        worst = std::fmax(worst, max_abs_diff(host_r, &dev_r[e * CVFEM_HEX8_N_DOF],
                                              CVFEM_HEX8_N_DOF));

    double norm = 0;
    for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i) norm = std::fmax(norm, std::fabs(double(host_r[i])));

    const bool ok = worst <= tol;
    std::printf("  %-8s host-vs-device max|diff| = %.3e  (|r|_inf = %.3e, tol %.1e)  %s\n",
                label, worst, norm, tol, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

}  // namespace

int main() {
    int sm = 0, shmem = 0, optin = 0, warp = 0;
    if (cvfem_cuda_smoke_device_info(&sm, &shmem, &optin, &warp) != 0) {
        std::printf("device query failed\n");
        return 1;
    }
    std::printf("device: %d SMs, warp %d, shared/block %d B, opt-in max %d B\n",
                sm, warp, shmem, optin);

    // What the opt-in limit means for the packed kernels: the residual stages 4 fields
    // in and 4 out per node, so 64 B/node in double.
    std::printf("  => at 64 B/node (residual, fp64) a block can stage %d nodes\n",
                optin / 64);

    int fail = 0;
    // fp64: the device evaluates exactly the same expression tree as the host. The
    // only permitted difference is FMA contraction, hence a small but nonzero tol.
    fail |= check<double>("fp64", 1e-12);
    // fp32: same kernel, different instantiation -- this is the device-side proof that
    // the Phase 1 templating is real.
    fail |= check<float>("fp32", 1e-4);

    std::printf("%s\n", fail ? "SMOKE TEST FAILED" : "smoke test passed");
    return fail;
}
