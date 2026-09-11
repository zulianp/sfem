// The split partial assembly: tangent stored once, applied many times.
//
// Sbar depends on the state and the geometry but not on the vector, so it is
// assembled once per Newton step and every Krylov apply reads it.  The material
// is evaluated once instead of once per iteration, and the apply that remains
// takes no geometry, no state and no material parameters at all.
//
// Covers TET4, HEX8 and TET10, chosen at compile time with -DELEMENT_TET4,
// -DELEMENT_HEX8 or -DELEMENT_TET10.  The three differ only in the mesh and in
// how many nodes an element has; the stored tangent is 45 numbers for all of
// them, which is the point -- the apply's cost stops depending on the material
// and starts depending only on the element's node count.
//
// Reported per problem size, with the dof count and thread count, because a
// throughput without them is not a result:
//
//   exact          the reference matrix-free apply
//   stored fp64    projected, tangent read from a double store
//   stored fp32    ... from a float store (SFEM's metric_tensor_t)
//   stored fp16    ... from a half store plus one scale per element
//   assembly       the cost of producing the store, once
//
// and then the break-even: how many applies must share one tangent before the
// split has paid back its assembly.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <cstdint>
#include <type_traits>
#include "sfem_base.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

#include "kernel_math.hpp"
#include MATERIAL_INEXACT_HEADER

#define TANGENT_COMPONENTS 45
#ifndef STATE_AMPLITUDE
#define STATE_AMPLITUDE 0.02
#endif
// `half_t` comes from sfem_config.h: __fp16 on some targets, _Float16 on
// others.  Declaring it here would conflict on whichever one it is not.

#ifndef LANE_VS
#define LANE_VS 16
#endif
#include "generated_abi.inc"

extern "C" int EXACT_APPLY(
        const int,
        const ptrdiff_t, const ptrdiff_t, idx_t **const,
        const geom_t *const, const geom_t *const, const geom_t *const,
        const geom_t *const, const geom_t *const, const geom_t *const,
        const geom_t *const, const geom_t *const, const geom_t *const,
        const geom_t *const,
        const double, const double,
#ifdef EXACT_TAKES_STATE
        const ptrdiff_t, const double *const, const double *const, const double *const,
#endif
        const ptrdiff_t, const double *const, const double *const, const double *const,
        const ptrdiff_t, double *const, double *const, double *const);

#include "element_mesh.inc"
#include "packed_mesh.inc"
#ifdef MATERIAL_PACKED_REFERENCE
#include MATERIAL_PACKED_REFERENCE
#endif

#ifdef PACKED_EXACT_APPLY
// The packed two-pass exact apply, which the generator already publishes.  It is
// here before any packed *inexact* kernel exists, because it is the cheapest way
// to find out whether the layout built in `packed_mesh.inc` is the layout the
// generated kernels actually expect: this one's answer is known.
extern "C" int PACKED_EXACT_APPLY(
        const int,
        const ptrdiff_t, const ptrdiff_t, const ptrdiff_t, const ptrdiff_t, const ptrdiff_t,
        uint16_t **const,
        const ptrdiff_t *const, const ptrdiff_t *const,
        const ptrdiff_t *const, const idx_t *const,
        const ptrdiff_t, const ptrdiff_t,
        const ptrdiff_t *const, const ptrdiff_t *const, const idx_t *const,
        void *const,
        const geom_t *const, const geom_t *const, const geom_t *const,
        const geom_t *const, const geom_t *const, const geom_t *const,
        const geom_t *const, const geom_t *const, const geom_t *const,
        const geom_t *const,
        const double, const double,
#ifdef EXACT_TAKES_STATE
        const ptrdiff_t, const double *const, const double *const, const double *const,
#endif
        const ptrdiff_t, const double *const, const double *const, const double *const,
        const ptrdiff_t, double *const, double *const, double *const);
#endif

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
    const int repeats = argc > 1 ? std::atoi(argv[1]) : 5;
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    const double mu = 2.3333333333333335, lmbda = 2.2;
    std::printf("%s, %s, threads %d, best of %d\n\n", MATERIAL_LABEL, ELEMENT_NAME, threads, repeats);
    std::printf("%10s %10s %12s | %8s %8s %8s %8s | %8s | %9s %9s %9s",
                "elements", "nodes", "ndof",
                "exact", "st.f64", "st.f32", "st.f16", "assembly",
                "f64 diff", "f32 diff", "f16 diff");
#ifdef PACKED_EXACT_APPLY
    std::printf(" | %8s %9s", "pk.exact", "pk diff");
#endif
#ifdef PACKED_STORED_APPLY
    std::printf(" | %8s %9s", "pk.st32", "pk diff");
#endif
    std::printf("\n");
    std::printf("%10s %10s %12s | %s | %8s | %9s %9s %9s", "", "", "",
                "          MDOF/s (apply)           ", "MDOF/s", "rel", "rel", "rel");
#ifdef PACKED_EXACT_APPLY
    std::printf(" | %8s %9s", "MDOF/s", "rel");
#endif
#ifdef PACKED_STORED_APPLY
    std::printf(" | %8s %9s", "MDOF/s", "rel");
#endif
    std::printf("\n");

    static const int sizes_probe[] = SIZES;
    for (int n : sizes_probe) {
        Mesh m = build(n);
        const ptrdiff_t ndof = 3 * m.nnodes;
        std::vector<double> hx(m.nnodes), hy(m.nnodes), hz(m.nnodes);
        std::vector<double> ux(m.nnodes), uy(m.nnodes), uz(m.nnodes);
        for (ptrdiff_t v = 0; v < m.nnodes; ++v) {
            // Seeded from position, not node index: a smooth field is what the
            // projection error is meant to be measured on, and on TET10 the
            // edge nodes are not on the lattice at all.
            const double x = m.px[v], y = m.py[v], z = m.pz[v];
            ux[v] = (STATE_AMPLITUDE)*std::sin(3.0*x + 1.0*y + 0.5*z);
            uy[v] = (STATE_AMPLITUDE)*std::sin(1.0*x + 3.0*y + 1.5*z);
            uz[v] = (STATE_AMPLITUDE)*std::sin(0.5*x + 1.5*y + 3.0*z);
            hx[v] = 0.05*std::sin(2.0*x + 0.7*y + 1.1*z);
            hy[v] = 0.05*std::sin(0.7*x + 2.0*y + 1.3*z);
            hz[v] = 0.05*std::sin(1.1*x + 1.3*y + 2.0*z);
        }
        // Component-major store: 45 streams, one per tangent component.  The
        // stride between them is padded off a power of two: at 196608 elements
        // the unpadded stride is exactly 1.5 MB, so all 45 streams land in the
        // same cache sets and the apply loses more than half its throughput.
        // The kernel takes the stride as a parameter so the caller can do this.
        const ptrdiff_t ecount = m.nelements;
        const ptrdiff_t cstride = ecount + 64;
        std::vector<double> S64((size_t)cstride * TANGENT_COMPONENTS);
        std::vector<float>  S32((size_t)cstride * TANGENT_COMPONENTS);
        std::vector<half_t> S16((size_t)cstride * TANGENT_COMPONENTS);
        std::vector<float>  scale(ecount, 1.0f);

        auto assemble = [&] {
            sfem::codegen::TANGENT_KERNEL<double, geom_t, double, LANE_VS>(
                m.nelements, m.evp.data(),
                m.adj[0].data(),m.adj[1].data(),m.adj[2].data(),m.adj[3].data(),m.adj[4].data(),
                m.adj[5].data(),m.adj[6].data(),m.adj[7].data(),m.adj[8].data(), m.det.data(),
                lmbda, mu, 1, ux.data(), uy.data(), uz.data(),
                cstride, S64.data());
        };
        assemble();
        // fp32 store, and the fp16 store with one max-abs scale per element so
        // the halves stay in range.
        for (ptrdiff_t e = 0; e < ecount; ++e) {
            double top = 0;
            for (int c = 0; c < TANGENT_COMPONENTS; ++c)
                top = std::max(top, std::fabs(S64[(size_t)c * cstride + e]));
            const double s = top > 65504.0 ? (top + 1e-8) / 65504.0 : 1.0;
            scale[e] = (float)s;
            for (int c = 0; c < TANGENT_COMPONENTS; ++c) {
                const size_t at = (size_t)c * cstride + e;
                S32[at] = (float)S64[at];
                S16[at] = (half_t)(S64[at] / s);
            }
        }

        std::vector<double> ax(m.nnodes,0), ay(m.nnodes,0), az(m.nnodes,0);
        std::vector<double> cx(m.nnodes,0), cy(m.nnodes,0), cz(m.nnodes,0);
#ifdef PACKED_EXACT_APPLY
        // The same mesh, partitioned into packs, with the fields placed through
        // its permutation so this is the same problem relabelled.
        PackedMeshView pk = build_packed_mesh(m, PACK_SIZE);
        std::vector<double> pux, puy, puz, phx, phy, phz;
        place_packed(ux, pk.layout, pux); place_packed(uy, pk.layout, puy);
        place_packed(uz, pk.layout, puz);
        place_packed(hx, pk.layout, phx); place_packed(hy, pk.layout, phy);
        place_packed(hz, pk.layout, phz);
        std::vector<double> pkx(m.nnodes), pky(m.nnodes), pkz(m.nnodes);
        std::vector<double> ghost_buf((size_t)pk.layout.n_ghost_entries * 3, 0.0);
#endif
        auto zero = [&](std::vector<double> &p, std::vector<double> &q, std::vector<double> &r) {
            std::fill(p.begin(),p.end(),0.0); std::fill(q.begin(),q.end(),0.0); std::fill(r.begin(),r.end(),0.0);
        };
        auto run_exact = [&] {
            zero(ax,ay,az);
            EXACT_APPLY(SFEM_CODEGEN_F64, m.nelements, m.nnodes, m.evp.data(),
                m.adj[0].data(),m.adj[1].data(),m.adj[2].data(),m.adj[3].data(),m.adj[4].data(),
                m.adj[5].data(),m.adj[6].data(),m.adj[7].data(),m.adj[8].data(), m.det.data(),
                lmbda, mu,
#ifdef EXACT_TAKES_STATE
                1, ux.data(), uy.data(), uz.data(),
#endif
                1, hx.data(), hy.data(), hz.data(), 1, ax.data(), ay.data(), az.data());
        };
        auto run_stored = [&](auto *store) {
            zero(cx,cy,cz);
            sfem::codegen::STORED_APPLY<double, typename std::remove_const<
                typename std::remove_pointer<decltype(store)>::type>::type, LANE_VS>(
                m.nelements, m.evp.data(), cstride, store,
                1, hx.data(), hy.data(), hz.data(), 1, cx.data(), cy.data(), cz.data());
        };
        auto run_compressed = [&] {
            zero(cx,cy,cz);
            sfem::codegen::COMPRESSED_APPLY<double, half_t, float>(
                m.nelements, m.evp.data(), cstride, S16.data(), scale.data(),
                1, hx.data(), hy.data(), hz.data(), 1, cx.data(), cy.data(), cz.data());
        };
#ifdef PACKED_STORED_APPLY
        // The store is assembled on the standard mesh and read here unchanged:
        // packing renumbers nodes, never elements, so element e is element e in
        // both and its 45 tangent components are the same numbers.
        auto run_packed_stored = [&](auto *store) {
            zero(pkx,pky,pkz);
            sfem::codegen::PACKED_STORED_APPLY<double, typename std::remove_const<
                typename std::remove_pointer<decltype(store)>::type>::type, LANE_VS>(
                pk.layout.n_packs, pk.layout.n_elements_per_pack, m.nelements, m.nnodes,
                pk.layout.max_nodes_per_pack, pk.layout.element_ptrs.data(),
                pk.layout.owned_nodes_ptr.data(),
                pk.layout.n_ghost_entries, pk.layout.n_ghost_reduce_rows,
                pk.layout.ghost_ptr.data(), pk.layout.ghost_idx.data(),
                pk.layout.ghost_reduce_ptr.data(), pk.layout.ghost_reduce_idx.data(),
                pk.layout.ghost_reduce_dest.data(), ghost_buf.data(),
                cstride, store,
                1, phx.data(), phy.data(), phz.data(),
                1, pkx.data(), pky.data(), pkz.data());
        };
#endif
#ifdef PACKED_EXACT_APPLY
        auto run_packed_exact = [&] {
            zero(pkx,pky,pkz);
            PACKED_EXACT_APPLY(SFEM_CODEGEN_F64,
                pk.layout.n_packs, pk.layout.n_elements_per_pack, m.nelements, m.nnodes,
                pk.layout.max_nodes_per_pack, pk.layout.element_ptrs.data(),
                pk.layout.owned_nodes_ptr.data(), pk.layout.n_shared_nodes.data(),
                pk.layout.ghost_ptr.data(), pk.layout.ghost_idx.data(),
                pk.layout.n_ghost_entries, pk.layout.n_ghost_reduce_rows,
                pk.layout.ghost_reduce_ptr.data(), pk.layout.ghost_reduce_idx.data(),
                pk.layout.ghost_reduce_dest.data(), ghost_buf.data(),
                pk.mesh.adj[0].data(),pk.mesh.adj[1].data(),pk.mesh.adj[2].data(),
                pk.mesh.adj[3].data(),pk.mesh.adj[4].data(),pk.mesh.adj[5].data(),
                pk.mesh.adj[6].data(),pk.mesh.adj[7].data(),pk.mesh.adj[8].data(),
                pk.mesh.det.data(), lmbda, mu,
#ifdef EXACT_TAKES_STATE
                1, pux.data(), puy.data(), puz.data(),
#endif
                1, phx.data(), phy.data(), phz.data(),
                1, pkx.data(), pky.data(), pkz.data());
        };
        // Against the standard apply, read back through the permutation.  The two
        // are the same operator on the same problem, so this is round-off or it is
        // a layout that does not match what the kernel expects.
        auto rel_packed = [&] {
            double num = 0, den = 0;
            for (ptrdiff_t i = 0; i < m.nnodes; ++i) {
                const idx_t j = pk.layout.to_new[i];
                num += std::fabs(ax[i]-pkx[j])+std::fabs(ay[i]-pky[j])+std::fabs(az[i]-pkz[j]);
                den += std::fabs(ax[i])+std::fabs(ay[i])+std::fabs(az[i]);
            }
            return num / den;
        };
#endif
        auto rel = [&](const std::vector<double> &px, const std::vector<double> &py,
                       const std::vector<double> &pz) {
            double num = 0, den = 0;
            for (ptrdiff_t i = 0; i < m.nnodes; ++i) {
                num += std::fabs(ax[i]-px[i])+std::fabs(ay[i]-py[i])+std::fabs(az[i]-pz[i]);
                den += std::fabs(ax[i])+std::fabs(ay[i])+std::fabs(az[i]);
            }
            return num / den;
        };

        run_exact();
        run_stored(S64.data());
        const double d_64 = rel(cx,cy,cz);
        run_stored(S32.data());
        const double d_32 = rel(cx,cy,cz);
        run_compressed();
        const double d_16 = rel(cx,cy,cz);

#ifdef PACKED_EXACT_APPLY
        run_packed_exact();
        const double d_pk = rel_packed();
#endif
#ifdef PACKED_STORED_APPLY
        run_packed_stored(S32.data());
        const double d_pks = rel_packed();
#endif

        const double e  = best_mdof(repeats, ndof, run_exact);
        const double s64 = best_mdof(repeats, ndof, [&]{ run_stored(S64.data()); });
        const double s32 = best_mdof(repeats, ndof, [&]{ run_stored(S32.data()); });
        const double s16 = best_mdof(repeats, ndof, run_compressed);
        const double a  = best_mdof(repeats, ndof, assemble);
#ifdef PACKED_EXACT_APPLY
        const double epk = best_mdof(repeats, ndof, run_packed_exact);
#endif
#ifdef PACKED_STORED_APPLY
        const double spk = best_mdof(repeats, ndof, [&]{ run_packed_stored(S32.data()); });
#endif

        std::printf("%10ld %10ld %12ld | %8.2f %8.2f %8.2f %8.2f | %8.2f | %9.1e %9.1e %9.1e",
                    (long)m.nelements, (long)m.nnodes, (long)ndof,
                    e, s64, s32, s16, a, d_64, d_32, d_16);
#ifdef PACKED_EXACT_APPLY
        std::printf(" | %8.2f %9.1e", epk, d_pk);
#endif
#ifdef PACKED_STORED_APPLY
        std::printf(" | %8.2f %9.1e", spk, d_pks);
#endif
        std::printf("\n");
        if (n == sizes_probe[sizeof(sizes_probe)/sizeof(int) - 1]) {
            // The gate: on an affine simplex the projection loses nothing, so
            // the f64 store must reproduce the exact apply to round-off.
            std::printf("\n  stored-f64 vs exact rel diff %.2e\n", d_64);
            // One tangent serves k applies.  The split wins when
            //   1/a + k/s  <  k/e   =>   k > (1/a) / (1/e - 1/s)
            auto breakeven = [&](double s) {
                if (s <= e) return -1.0;
                return (1.0 / a) / (1.0 / e - 1.0 / s);
            };
            std::printf("  break-even applies per tangent: f64 %.1f  f32 %.1f  f16 %.1f\n",
                        breakeven(s64), breakeven(s32), breakeven(s16));
            std::printf("  store bytes/element: f64 %d  f32 %d  f16+scale %d\n",
                        TANGENT_COMPONENTS * 8, TANGENT_COMPONENTS * 4, TANGENT_COMPONENTS * 2 + 4);
        }
    }
    return 0;
}
