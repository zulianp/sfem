// The split on a material with more than one unit.
//
// Mooney-Rivlin elasticity plus Kelvin-Voigt viscosity: an energy unit and a
// residual unit.  The operator's action is the sum of the two, so the split has
// to carry both -- two tangents assembled, two applies summed -- and the
// comparison is against the sum of the two exact kernels.
//
// The two units differ in a way that matters for the store.  The elastic
// tangent came from an energy, so it is a Hessian and its major symmetry folds
// 81 numbers into 45.  The viscous tangent came from a residual, is a Jacobian,
// and is genuinely unsymmetric, so all 81 are independent.  126 numbers per
// element in total, against 45 for a single hyperelastic material.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <type_traits>
#include "sfem_base.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif
#include "kernel_math.hpp"
#include "mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_inexact_apply_inline.hpp"
#include "mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_inexact_apply_inline.hpp"
#define ELEMENT_TET4
#include "element_mesh.inc"

#define TC_ELASTIC 45
#define TC_VISCOUS 81
typedef __fp16 half_t;

#define ELASTIC_TANGENT  sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_inexact_apply_tangent_affine_mesh_soa_impl
#define ELASTIC_STORED   sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_inexact_apply_stored_affine_mesh_soa_impl
#define ELASTIC_COMPRESS sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_inexact_apply_compressed_affine_mesh_soa_impl
#define VISCOUS_TANGENT  sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_inexact_apply_tangent_affine_mesh_soa_impl
#define VISCOUS_STORED   sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_inexact_apply_stored_affine_mesh_soa_impl
#define VISCOUS_COMPRESS sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_inexact_apply_compressed_affine_mesh_soa_impl

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_apply_affine_mesh_soa(
    const ptrdiff_t, const ptrdiff_t, idx_t **const,
    const geom_t *const, const geom_t *const, const geom_t *const, const geom_t *const,
    const geom_t *const, const geom_t *const, const geom_t *const, const geom_t *const,
    const geom_t *const, const geom_t *const, const double, const double,
    const ptrdiff_t, const double *const, const double *const, const double *const,
    const ptrdiff_t, const double *const, const double *const, const double *const,
    const ptrdiff_t, double *const, double *const, double *const);

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_jacobian_action_affine_mesh_soa(
    const ptrdiff_t, const ptrdiff_t, idx_t **const,
    const geom_t *const, const geom_t *const, const geom_t *const, const geom_t *const,
    const geom_t *const, const geom_t *const, const geom_t *const, const geom_t *const,
    const geom_t *const, const geom_t *const,
    const double, const double, const double,
    const ptrdiff_t, const double *const, const double *const, const double *const,
    const ptrdiff_t, const double *const, const double *const, const double *const,
    const ptrdiff_t, const double *const, const double *const, const double *const,
    const ptrdiff_t, double *const, double *const, double *const);

template <typename F> static double best_mdof(int repeats, ptrdiff_t ndof, F &&fn) {
    // The timed region is the kernel and nothing else.
    //
    // Two things used to sit inside it and both distorted the result, badly at
    // high thread counts.  Zeroing the output arrays is a serial `std::fill` of
    // several megabytes that is not part of the operator and does not
    // parallelise, so it charged Amdahl's tax to the kernel.  And timing a
    // single call charged that call for the OpenMP team startup and a cold
    // cache, which is a large fraction of a kernel that runs for a few
    // milliseconds on ten cores.  Together they understated throughput by three
    // to five times and compressed the measured scaling.
    //
    // So: the outputs are not cleared between repetitions.  The apply
    // accumulates, so the values grow -- which does not affect what is being
    // measured, and correctness is checked separately, with clearing, outside
    // any timed region.
    fn();
    double top = 0;
    for (int attempt = 0; attempt < 3; ++attempt) {
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < repeats; ++r) fn();
        auto t1 = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(t1 - t0).count();
        top = std::max(top, (double)ndof * repeats / seconds * 1e-6);
    }
    return top;
}

int main(int argc, char **argv) {
    const int repeats = argc > 1 ? std::atoi(argv[1]) : 7;
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    const double mu = 1.3, lmbda = 2.2, eta_s = 0.31, eta_b = 0.17, alpha = 0.9;
    std::printf("mooney_rivlin_kelvin_voigt_newmark (elastic + viscous), TET4, threads %d, best of %d\n\n",
                threads, repeats);
    std::printf("%10s %10s %12s | %8s %8s %8s %8s | %8s | %9s %9s\n",
                "elements", "nodes", "ndof", "exact", "st.f64", "st.f32", "st.f16",
                "assembly", "f32 diff", "f16 diff");
    std::printf("%10s %10s %12s | %s | %8s | %9s %9s\n", "", "", "",
                "          MDOF/s (apply)           ", "MDOF/s", "rel", "rel");

    for (int n : {8, 16, 24, 32, 40}) {
        Mesh m = build(n);
        const ptrdiff_t N = m.nnodes, EC = m.nelements, CS = EC + 64, ndof = 3 * N;
        std::vector<double> ux(N),uy(N),uz(N), zx(N),zy(N),zz(N), hx(N),hy(N),hz(N);
        for (ptrdiff_t v = 0; v < N; ++v) {
            const double x=m.px[v], y=m.py[v], z=m.pz[v];
            ux[v]=0.02*std::sin(3*x+y+0.5*z); uy[v]=0.02*std::sin(x+3*y+1.5*z); uz[v]=0.02*std::sin(0.5*x+1.5*y+3*z);
            zx[v]=0.013*std::sin(2*x+0.4*y+z); zy[v]=0.013*std::sin(0.4*x+2*y+z); zz[v]=0.013*std::sin(x+0.6*y+2*z);
            hx[v]=0.05*std::sin(2*x+0.7*y+1.1*z); hy[v]=0.05*std::sin(0.7*x+2*y+1.3*z); hz[v]=0.05*std::sin(1.1*x+1.3*y+2*z);
        }
        std::vector<double> E64((size_t)CS*TC_ELASTIC), V64((size_t)CS*TC_VISCOUS);
        std::vector<float>  E32((size_t)CS*TC_ELASTIC), V32((size_t)CS*TC_VISCOUS);
        std::vector<half_t> E16((size_t)CS*TC_ELASTIC), V16((size_t)CS*TC_VISCOUS);
        std::vector<float>  ES(EC, 1.0f), VS(EC, 1.0f);
        std::vector<double> ax(N,0),ay(N,0),az(N,0), bx(N,0),by(N,0),bz(N,0);
        const geom_t *A[9]; for (int i=0;i<9;++i) A[i]=m.adj[i].data();
        auto z3=[&](std::vector<double>&a,std::vector<double>&b,std::vector<double>&c){
            std::fill(a.begin(),a.end(),0.0);std::fill(b.begin(),b.end(),0.0);std::fill(c.begin(),c.end(),0.0);};

        // Both units, assembled.  This is the once-per-tangent cost.
        auto assemble = [&] {
            ELASTIC_TANGENT<double,geom_t,double>(EC, m.evp.data(),
                A[0],A[1],A[2],A[3],A[4],A[5],A[6],A[7],A[8], m.det.data(),
                lmbda, mu, 1, ux.data(),uy.data(),uz.data(), 1, CS, E64.data());
            VISCOUS_TANGENT<double,geom_t,double>(EC, m.evp.data(),
                A[0],A[1],A[2],A[3],A[4],A[5],A[6],A[7],A[8], m.det.data(),
                eta_b, eta_s, alpha, 1, ux.data(),uy.data(),uz.data(),
                1, zx.data(),zy.data(),zz.data(), 1, CS, V64.data());
        };
        assemble();
        auto narrow = [&](std::vector<double>&S64, std::vector<float>&S32,
                          std::vector<half_t>&S16, std::vector<float>&SC, int TC) {
            for (ptrdiff_t e = 0; e < EC; ++e) {
                double top = 0;
                for (int c = 0; c < TC; ++c) top = std::max(top, std::fabs(S64[(size_t)c*CS+e]));
                const double s = top > 65504.0 ? (top + 1e-8)/65504.0 : 1.0;
                SC[e] = (float)s;
                for (int c = 0; c < TC; ++c) {
                    const size_t at = (size_t)c*CS+e;
                    S32[at] = (float)S64[at]; S16[at] = (half_t)(S64[at]/s);
                }
            }
        };
        narrow(E64,E32,E16,ES,TC_ELASTIC); narrow(V64,V32,V16,VS,TC_VISCOUS);

        // The exact action is the sum of the two units' exact kernels.
        auto run_exact = [&] {
            mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_apply_affine_mesh_soa(
                EC, N, m.evp.data(), A[0],A[1],A[2],A[3],A[4],A[5],A[6],A[7],A[8], m.det.data(),
                lmbda, mu, 1, ux.data(),uy.data(),uz.data(), 1, hx.data(),hy.data(),hz.data(),
                1, ax.data(),ay.data(),az.data());
            mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_jacobian_action_affine_mesh_soa(
                EC, N, m.evp.data(), A[0],A[1],A[2],A[3],A[4],A[5],A[6],A[7],A[8], m.det.data(),
                eta_b, eta_s, alpha, 1, ux.data(),uy.data(),uz.data(),
                1, zx.data(),zy.data(),zz.data(), 1, hx.data(),hy.data(),hz.data(),
                1, ax.data(),ay.data(),az.data());
        };
        auto run_stored = [&](auto *ep, auto *vp) {
            ELASTIC_STORED<double, typename std::remove_const<typename std::remove_pointer<decltype(ep)>::type>::type>(
                EC, m.evp.data(), 1, CS, ep, 1, hx.data(),hy.data(),hz.data(),
                1, bx.data(),by.data(),bz.data());
            VISCOUS_STORED<double, typename std::remove_const<typename std::remove_pointer<decltype(vp)>::type>::type>(
                EC, m.evp.data(), 1, CS, vp, 1, hx.data(),hy.data(),hz.data(),
                1, bx.data(),by.data(),bz.data());
        };
        auto run_compressed = [&] {
            ELASTIC_COMPRESS<double, half_t, float>(EC, m.evp.data(), 1, CS, E16.data(), ES.data(),
                1, hx.data(),hy.data(),hz.data(), 1, bx.data(),by.data(),bz.data());
            VISCOUS_COMPRESS<double, half_t, float>(EC, m.evp.data(), 1, CS, V16.data(), VS.data(),
                1, hx.data(),hy.data(),hz.data(), 1, bx.data(),by.data(),bz.data());
        };
        auto rel=[&]{ double num=0,den=0; for(ptrdiff_t i=0;i<N;++i){
                num+=std::fabs(ax[i]-bx[i])+std::fabs(ay[i]-by[i])+std::fabs(az[i]-bz[i]);
                den+=std::fabs(ax[i])+std::fabs(ay[i])+std::fabs(az[i]); } return num/den; };

        // Correctness first, each from a cleared output.  Outside any timing.
        z3(ax,ay,az); run_exact();
        z3(bx,by,bz); run_stored(E64.data(), V64.data()); const double d64 = rel();
        z3(bx,by,bz); run_stored(E32.data(), V32.data()); const double d32 = rel();
        z3(bx,by,bz); run_compressed();                   const double d16 = rel();

        const double e   = best_mdof(repeats, ndof, run_exact);
        const double s64 = best_mdof(repeats, ndof, [&]{ run_stored(E64.data(), V64.data()); });
        const double s32 = best_mdof(repeats, ndof, [&]{ run_stored(E32.data(), V32.data()); });
        const double s16 = best_mdof(repeats, ndof, run_compressed);
        const double a   = best_mdof(repeats, ndof, assemble);
        std::printf("%10ld %10ld %12ld | %8.2f %8.2f %8.2f %8.2f | %8.2f | %9.1e %9.1e\n",
                    (long)EC, (long)N, (long)ndof, e, s64, s32, s16, a, d32, d16);
        if (n == 40) {
            std::printf("\n  stored-f64 vs exact rel diff %.2e\n", d64);
            auto be = [&](double s){ return s <= e ? -1.0 : (1.0/a)/(1.0/e - 1.0/s); };
            std::printf("  break-even applies per tangent: f64 %.1f  f32 %.1f  f16 %.1f\n",
                        be(s64), be(s32), be(s16));
            std::printf("  store bytes/element: f64 %d  f32 %d  f16+scale %d   (45 elastic + 81 viscous)\n",
                        (TC_ELASTIC+TC_VISCOUS)*8, (TC_ELASTIC+TC_VISCOUS)*4,
                        (TC_ELASTIC+TC_VISCOUS)*2 + 8);
        }
    }
    return 0;
}
