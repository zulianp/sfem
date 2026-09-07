// Scaling of one kernel, in isolation: no mesh build, no narrowing, no
// comparison inside the timed region, and the output zeroing hoisted out.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <type_traits>
#include "sfem_base.hpp"
#include <omp.h>
#include "kernel_math.hpp"
#include "mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_inexact_apply_inline.hpp"
#define ELEMENT_TET4
#include "element_mesh.inc"
#define TC 45

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_apply_affine_mesh_soa(
    const ptrdiff_t, const ptrdiff_t, idx_t **const,
    const geom_t *const, const geom_t *const, const geom_t *const, const geom_t *const,
    const geom_t *const, const geom_t *const, const geom_t *const, const geom_t *const,
    const geom_t *const, const geom_t *const, const double, const double,
    const ptrdiff_t, const double *const, const double *const, const double *const,
    const ptrdiff_t, const double *const, const double *const, const double *const,
    const ptrdiff_t, double *const, double *const, double *const);

int main(int argc, char **argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 40;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 20;
    Mesh m = build(n);
    const ptrdiff_t N = m.nnodes, EC = m.nelements, CS = EC + 64, ndof = 3*N;
    std::vector<double> ux(N),uy(N),uz(N),hx(N),hy(N),hz(N),ox(N,0),oy(N,0),oz(N,0);
    for (ptrdiff_t v=0; v<N; ++v){ const double x=m.px[v],y=m.py[v],z=m.pz[v];
        ux[v]=0.02*std::sin(3*x+y); uy[v]=0.02*std::sin(x+3*y); uz[v]=0.02*std::sin(y+3*z);
        hx[v]=0.05*std::sin(2*x+z); hy[v]=0.05*std::sin(y+2*z); hz[v]=0.05*std::sin(x+2*y); }
    std::vector<double> S((size_t)CS*TC);
    const geom_t *A[9]; for(int i=0;i<9;++i) A[i]=m.adj[i].data();
    sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_inexact_apply_tangent_affine_mesh_soa_impl<double,geom_t,double>(
        EC, m.evp.data(), A[0],A[1],A[2],A[3],A[4],A[5],A[6],A[7],A[8], m.det.data(),
        2.2, 1.3, 1, ux.data(),uy.data(),uz.data(), 1, CS, S.data());

    auto timed = [&](const char *label, auto &&fn) {
        fn();                                        // warm
        const double c0 = omp_get_wtime();
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) fn();
        auto t1 = std::chrono::steady_clock::now();
        const double wall = std::chrono::duration<double>(t1-t0).count();
        (void)c0;
        std::printf("  %-14s %8.2f MDOF/s   wall %6.3f s\n",
                    label, (double)ndof*reps/wall*1e-6, wall);
        return (double)ndof*reps/wall*1e-6;
    };
    std::printf("threads %d, %ld elements, ndof %ld, %d reps (timed region is the kernel only)\n",
                omp_get_max_threads(), (long)EC, (long)ndof, reps);
    timed("exact", [&]{
        mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_apply_affine_mesh_soa(
            EC, N, m.evp.data(), A[0],A[1],A[2],A[3],A[4],A[5],A[6],A[7],A[8], m.det.data(),
            2.2, 1.3, 1, ux.data(),uy.data(),uz.data(), 1, hx.data(),hy.data(),hz.data(),
            1, ox.data(),oy.data(),oz.data()); });
    timed("stored f64", [&]{
        sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_inexact_apply_stored_affine_mesh_soa_impl<double,double>(
            EC, m.evp.data(), 1, CS, S.data(), 1, hx.data(),hy.data(),hz.data(),
            1, ox.data(),oy.data(),oz.data()); });
    return 0;
}
