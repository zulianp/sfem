// THE TET4 MICRO-KERNELS, INSTANTIATED AND RUN AT SINGLE PRECISION.
//
// The third family DESIGN.md's correction reaches, after the flat packed layout
// (cvfem_mixed_precision_packs / _sweep) and the semi-structured one
// (cvfem_ss_mixed_precision). TET4 is where the correction's wording bites hardest, because
// this element's kernels are built on a SIMD vector type and a lane count, and both were one
// number per build:
//
//   VEC_SIZE  = VEC_BYTES / sizeof(the build's scalar)        the pack width
//   SIMD_SIZE = CVFEM_SIMD_BYTES / sizeof(the build's scalar) the vector lane count
//   scalar_v  = the build's scalar, vector_size(CVFEM_SIMD_BYTES)
//
// A kernel instantiated at float with those left alone compiles and is wrong in the way that
// is hardest to see: it strides a 32-lane pack by 16, so half the lanes are never written and
// every staged read runs past the end of its array. Nothing reports it. That is the same
// failure the HEX8 packs had, and the reason this gate compares answers rather than checking
// that the instantiation exists.
//
// It runs the SIMD residual micro-kernel over one full lane group of identical unit
// tetrahedra, at both precisions, and requires the lanes to agree with each other and the two
// precisions to agree with one another. Identical elements make every lane's answer the same,
// so a kernel that crosses lanes or stops short of them is caught as well -- which is what the
// HEX8 gate's lane-invariant input buys, and it costs nothing here.
#define CVFEM_PHASES 0

#include "smesh_types.hpp"
#include "kernels/cvfem_portability.hpp"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

#include "support/cvfem_default_types.hpp"
using idx_t      = smesh::idx_t;
using count_t    = smesh::count_t;
using jacobian_t = smesh::jacobian_t;

#include "kernels/microkernels/tet4/cvfem_tet4_ns_upwind_kernels.hpp"

// The build's lane width, which the kernels no longer take from the includer but the drivers
// still define for their own staging. Named here after the header so it is the header's own
// VEC_BYTES that sizes it.
static constexpr int VEC_SIZE = cvfem_tet4_vec_size<scalar_t>;

// The reference tetrahedron, whose adjugate and determinant are exact at either precision, so
// the comparison is about the kernel and not about the geometry's round-off.
template <typename S>
static void run(std::vector<S> &out) {
    constexpr int W = cvfem_tet4_vec_size<S>;
    Tet4InputPackT<S>    in{};
    Tet4ResidualPackT<S> res{};
    for (int a = 0; a < 4; ++a)
        for (int l = 0; l < W; ++l) {
            // A state off the upwind switch; a zero velocity sits on its non-differentiable
            // point. The SAME value in every lane, so every lane must produce the same answer.
            in.ux[a][l] = S(0.7) + S(0.13) * S(a);
            in.uy[a][l] = S(-0.3) + S(0.11) * S(a);
            in.uz[a][l] = S(0.2) - S(0.07) * S(a);
            in.p[a][l]  = S(1.0) + S(0.05) * S(a);
        }

    alignas(ALIGN_BYTES) jacobian_t a0[W], a1[W], a2[W], a3[W], a4[W], a5[W], a6[W], a7[W], a8[W], dt[W];
    for (int l = 0; l < W; ++l) {
        a0[l] = 1; a1[l] = 0; a2[l] = 0;
        a3[l] = 0; a4[l] = 1; a5[l] = 0;
        a6[l] = 0; a7[l] = 0; a8[l] = 1;
        dt[l] = 1;
    }

    cvfem_run_residual_kernel<S>(S(1), S(0.05), a0, a1, a2, a3, a4, a5, a6, a7, a8, dt, W, in, res);

    out.assign((size_t)4 * 4 * (size_t)W, S(0));
    for (int a = 0; a < 4; ++a)
        for (int l = 0; l < W; ++l) {
            out[(size_t)((a * 4 + 0) * W + l)] = res.rx[a][l];
            out[(size_t)((a * 4 + 1) * W + l)] = res.ry[a][l];
            out[(size_t)((a * 4 + 2) * W + l)] = res.rz[a][l];
            out[(size_t)((a * 4 + 3) * W + l)] = res.rc[a][l];
        }
}

// Every lane must equal lane 0: the elements are identical, so a kernel that crosses lanes or
// stops short of the group shows up here.
template <typename S>
static int lanes_agree(const char *what, const std::vector<S> &v) {
    constexpr int W     = cvfem_tet4_vec_size<S>;
    double        worst = 0, scale = 0;
    for (int k = 0; k < 16; ++k) {
        const double l0 = (double)v[(size_t)(k * W)];
        scale           = std::max(scale, std::fabs(l0));
        for (int l = 1; l < W; ++l)
            worst = std::max(worst, std::fabs((double)v[(size_t)(k * W + l)] - l0));
    }
    const bool ok = scale > 0 && worst == 0;
    std::printf("%-56s %d lanes, spread %.3e  %s\n", what, W, worst, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

int main() {
    std::vector<double> r64;
    std::vector<float>  r32;
    run<double>(r64);
    run<float>(r32);

    int bad = 0;
    bad += lanes_agree("the f64 residual is the same in every lane", r64);
    bad += lanes_agree("the f32 residual is the same in every lane", r32);

    // f32 must have twice the lanes, which is the whole point of the width being a property of
    // the scalar rather than of the build.
    const int w64 = cvfem_tet4_vec_size<double>, w32 = cvfem_tet4_vec_size<float>;
    const bool widths_ok = w32 == 2 * w64;
    std::printf("%-56s %d vs %d  %s\n", "the f32 pack has twice the lanes", w32, w64,
                widths_ok ? "OK" : "FAIL");
    if (!widths_ok) ++bad;

    double worst = 0, scale = 0;
    for (int k = 0; k < 16; ++k) {
        const double a = r64[(size_t)(k * w64)], b = (double)r32[(size_t)(k * w32)];
        worst = std::max(worst, std::fabs(a - b));
        scale = std::max(scale, std::fabs(a));
    }
    const double rel = scale > 0 ? worst / scale : worst;
    const bool   ok  = scale > 0 && rel < 1e-5 && std::isfinite(rel);
    std::printf("%-56s rel %.3e  %s\n", "the two precisions agree", rel, ok ? "OK" : "FAIL");
    if (!ok) ++bad;

    if (bad) {
        std::fprintf(stderr, "\ncvfem_tet4_mixed_precision: %d check(s) failed\n", bad);
        return 1;
    }
    std::printf("\nthe TET4 SIMD micro-kernel runs at single precision and agrees\n");
    return 0;
}
