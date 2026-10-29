// Compressed quadrature-point partial assembly against BSR, the exact matrix-free apply and the
// projected (P0) store.  One element per build: -DELEMENT_TET4, -DELEMENT_TET10 or -DELEMENT_HEX8.
//
//   bench_qp [n] [repeats]
//
// Environment:
//   QP_NU       Poisson ratio (E = 1); unset keeps the spike's mu = 2.3333, lambda = 2.2
//   QP_WARP     amplitude w of the nonlinear part of the state u = L x + w N(x)   (default 0.4)
//   QP_ONLY     substring filters on the variant names, separated by '|'
//   QP_PACK     elements per pack, or auto                (default auto: about 20 packs per thread)
//   QP_BREAKDOWN  also time the traversal alone and the element kernel alone, per packed variant
//
// Every row reports the problem size, the thread count and BSR: the comparison is always against
// the assembled operator, in double and single precision values.  Errors are relative to the exact
// matrix-free apply with a white-noise increment, in the l2 norm and in the energy h^T H h.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "sfem_base.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

#include "kernel_math.hpp"
#ifndef NO_P0
#include MATERIAL_INEXACT_HEADER
#endif

#include "qp_setup.hpp"
#include "qp_apply.hpp"

#include "generated_abi.inc"
#include "element_mesh.inc"
#include "packed_mesh.inc"
#include "bsr_matrix.inc"
#include "bsr_quant.hpp"

#include "bench_problem.inc"

template <class St, typename tau, qp::Form FORM = qp::Form::Lanes>
static void run_variant(const char *name, Problem &P, int repeats) {
    if (!selected(name)) return;
    Mesh &m = P.m;
    St st(m.nelements);
    auto setup = [&] {
        qp::qp_setup_impl<double, geom_t, St, St::VS>(
            m.nelements, m.kevp.data(), m.adj[0].data(), m.adj[1].data(), m.adj[2].data(), m.adj[3].data(),
            m.adj[4].data(), m.adj[5].data(), m.adj[6].data(), m.adj[7].data(), m.adj[8].data(), m.det.data(),
            P.lmbda, P.mu, 1, P.ux.data(), P.uy.data(), P.uz.data(), st);
    };
    std::vector<double> ox(m.nnodes), oy(m.nnodes), oz(m.nnodes);
    auto apply = [&] {
        qp::apply<St, tau, FORM>(st, m.nelements, m.kevp.data(), 1, P.hx.data(), P.hy.data(), P.hz.data(), 1,
                                 ox.data(), oy.data(), oz.data());
    };
    Row row;
    row.name = name;
    row.setup_s = best_seconds(std::max(1, repeats / 4), setup);
    std::fill(ox.begin(), ox.end(), 0.0);
    std::fill(oy.begin(), oy.end(), 0.0);
    std::fill(oz.begin(), oz.end(), 0.0);
    apply();
    errors(P, ox, oy, oz, row);
    row.apply_s = best_seconds(repeats, apply);
    row.bytes = (double)St::bytes_per_element() * m.nelements;
    print_row(P, row);
}

// QP_BREAKDOWN: the traversal with an identity element kernel (gather, two-pass scatter, ghost
// reduction; no store read), and the element kernel alone (decode and contraction over every
// block, fixed increment, no gather or scatter).  Their sum against the full apply shows where
// the time goes.
template <class El_, int VS_>
struct IdentityStore {
    using Element = El_;
    static constexpr int VS = VS_;
    static constexpr bool is_element_matrix = true;
    template <typename tau>
    void block(const ptrdiff_t, const int ne, const double bh[3][El_::N][VS_], double out[3][El_::N][VS_]) const {
        for (int i = 0; i < 3; ++i)
            for (int p = 0; p < El_::N; ++p)
#pragma omp simd
                for (int lane = 0; lane < ne; ++lane) out[i][p][lane] = bh[i][p][lane];
    }
};

template <class St, typename tau, qp::Form FORM>
static void breakdown_rows(const char *name, const St &st, Problem &P, PackedMeshView &pk, int repeats) {
    Mesh &m = P.m;
    static constexpr int N = El::N, VS = St::VS;
    std::vector<double> phx(P.hx), phy(P.hy), phz(P.hz);
    std::vector<double> ox(m.nnodes, 0), oy(m.nnodes, 0), oz(m.nnodes, 0);
    std::vector<double> ghost((size_t)pk.layout.n_ghost_entries * 3, 0.0);
    IdentityStore<El, VS> id;
    Row row;
    row.name = std::string(name) + " [traverse]";
    row.apply_s = best_seconds(repeats, [&] {
        qp::apply_packed<IdentityStore<El, VS>, tau, FORM>(id, pk.layout, pk.kernel_element_ptrs.data(), m.nelements,
                                                           1, phx.data(), phy.data(), phz.data(), 1, ox.data(),
                                                           oy.data(), oz.data(), ghost.data());
    });
    print_row(P, row);
    double sink = 0;
    row.name = std::string(name) + " [kernel]";
    row.apply_s = best_seconds(repeats, [&] {
#pragma omp parallel reduction(+ : sink)
        {
            double bh[3][N][VS], out[3][N][VS];
            for (int i = 0; i < 3; ++i)
                for (int p = 0; p < N; ++p)
                    for (int lane = 0; lane < VS; ++lane) bh[i][p][lane] = P.hx[(size_t)(p * VS + lane) % P.hx.size()] * (i + 1);
#pragma omp for schedule(static)
            for (ptrdiff_t evb = 0; evb < m.nelements; evb += VS) {
                const int ne = (int)std::min<ptrdiff_t>(VS, m.nelements - evb);
                qp::any_block<St, tau, FORM>(st, evb, ne, bh, out);
                sink += out[0][0][0] + out[2][N - 1][ne - 1];
            }
        }
    });
    print_row(P, row);
    if (sink == 12345.678) std::printf("%g\n", sink);
}

// The same store applied through the packed layout (two-pass, no atomics).  Packing renumbers
// nodes, never elements, so the store is set up on the standard mesh and read unchanged; the
// increment goes in and the result comes out through the permutation.
template <class St, typename tau, qp::Form FORM = qp::Form::Lanes>
static void run_variant_packed(const char *name, Problem &P, PackedMeshView &pk, int repeats) {
    if (!selected(name)) return;
    Mesh &m = P.m;
    St st(m.nelements);
    qp::qp_setup_impl<double, geom_t, St, St::VS>(
        m.nelements, m.kevp.data(), m.adj[0].data(), m.adj[1].data(), m.adj[2].data(), m.adj[3].data(),
        m.adj[4].data(), m.adj[5].data(), m.adj[6].data(), m.adj[7].data(), m.adj[8].data(), m.det.data(), P.lmbda,
        P.mu, 1, P.ux.data(), P.uy.data(), P.uz.data(), st);
    std::vector<double> phx, phy, phz;
    place_packed(P.hx, pk.layout, phx);
    place_packed(P.hy, pk.layout, phy);
    place_packed(P.hz, pk.layout, phz);
    std::vector<double> ox(m.nnodes, 0), oy(m.nnodes, 0), oz(m.nnodes, 0);
    std::vector<double> ghost((size_t)pk.layout.n_ghost_entries * 3, 0.0);
    auto apply = [&] {
        qp::apply_packed<St, tau, FORM>(st, pk.layout, pk.kernel_element_ptrs.data(), m.nelements, 1, phx.data(),
                                        phy.data(), phz.data(), 1, ox.data(), oy.data(), oz.data(), ghost.data());
    };
    apply();
    std::vector<double> x(m.nnodes), y(m.nnodes), z(m.nnodes);
    for (ptrdiff_t i = 0; i < m.nnodes; ++i) {
        const idx_t j = pk.layout.to_new[i];
        x[i] = ox[j];
        y[i] = oy[j];
        z[i] = oz[j];
    }
    Row row;
    row.name = name;
    errors(P, x, y, z, row);
    row.apply_s = best_seconds(repeats, apply);
    row.bytes = (double)St::bytes_per_element() * m.nelements;
    print_row(P, row);
    if (std::getenv("QP_BREAKDOWN")) breakdown_rows<St, tau, FORM>(name, st, P, pk, repeats);
}

template <typename code_t, qp::Scheme S, qp::Layout L>
using StoreT = qp::Store<El, code_t, S, L, LANE_VS>;
using qp::Layout;
using qp::Scheme;

int main(int argc, char **argv) {
    static const int sizes[] = SIZES;
    const int n = argc > 1 ? std::atoi(argv[1]) : sizes[2];
    const int repeats = argc > 2 ? std::atoi(argv[2]) : 10;
    const double warp = std::getenv("QP_WARP") ? std::atof(std::getenv("QP_WARP")) : 0.4;
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif

    Problem P;
    make_problem(P, n, warp);
    Mesh &m = P.m;
    auto exact = [&](std::vector<double> &x, std::vector<double> &y, std::vector<double> &z) { exact_apply(P, x, y, z); };

    std::printf("%s %s: n=%d, %ld elements, %ld nodes, %ld dof, %d threads, mu=%g lambda=%g, warp %g\n",
                MATERIAL_LABEL, ELEMENT_NAME, n, (long)m.nelements, (long)m.nnodes, (long)P.ndof, threads, P.mu,
                P.lmbda, warp);
    std::printf("  %-34s %10s %10s %10s %10s %10s\n", "operator", "setup ms", "MDOF/s", "B/dof", "rel l2",
                "rel energy");

    {  // exact matrix-free
        Row row;
        row.name = "exact matrix-free";
        std::vector<double> x(m.nnodes), y(m.nnodes), z(m.nnodes);
        row.apply_s = best_seconds(repeats, [&] { exact(x, y, z); });
        std::fill(x.begin(), x.end(), 0.0);
        std::fill(y.begin(), y.end(), 0.0);
        std::fill(z.begin(), z.end(), 0.0);
        exact(x, y, z);
        errors(P, x, y, z, row);
        print_row(P, row);
    }

    {  // BSR, double and single precision values, assembled by the generated kernel
        const BsrGraph g = build_bsr_graph(m);
        std::vector<geom_t> gx(m.nnodes), gy(m.nnodes), gz(m.nnodes);
        for (ptrdiff_t v = 0; v < m.nnodes; ++v) {
            gx[v] = (geom_t)m.px[v];
            gy[v] = (geom_t)m.py[v];
            gz[v] = (geom_t)m.pz[v];
        }
        const geom_t *const pts[3] = {gx.data(), gy.data(), gz.data()};
        std::vector<double> xi((size_t)m.nnodes * 3), yi((size_t)m.nnodes * 3);
        for (ptrdiff_t v = 0; v < m.nnodes; ++v) {
            xi[(size_t)v * 3 + 0] = P.hx[v];
            xi[(size_t)v * 3 + 1] = P.hy[v];
            xi[(size_t)v * 3 + 2] = P.hz[v];
        }
        // The generated assembly into values of type V (zeroed first).
        auto assemble_into = [&](auto &values) {
            using V = typename std::decay_t<decltype(values)>::value_type;
            std::fill(values.begin(), values.end(), V(0));
            MATERIAL_HESSIAN_BSR(SFEM_CODEGEN_F64, (int)sizeof(V), m.nelements, m.nnodes, m.evp.data(),
#ifdef BSR_AFFINE
                                 m.adj[0].data(), m.adj[1].data(), m.adj[2].data(), m.adj[3].data(),
                                 m.adj[4].data(), m.adj[5].data(), m.adj[6].data(), m.adj[7].data(),
                                 m.adj[8].data(), m.det.data(),
#else
                                 pts,
#endif
                                 P.lmbda, P.mu, 1, P.ux.data(), P.uy.data(), P.uz.data(), g.rowptr.data(),
                                 g.colidx.data(), values.data());
        };
        auto bsr = [&](auto value) {
            using V = decltype(value);
            const char *name = sizeof(V) == 8 ? "BSR f64" : "BSR f32";
            if (!selected(name)) return;
            std::vector<V> values((size_t)g.nnz() * 9);
            auto assemble = [&] { assemble_into(values); };
            Row row;
            row.name = name;
            row.setup_s = best_seconds(1, assemble);
            bsr_apply_typed<V>(g, values.data(), xi.data(), yi.data());
            std::vector<double> x(m.nnodes), y(m.nnodes), z(m.nnodes);
            for (ptrdiff_t v = 0; v < m.nnodes; ++v) {
                x[v] = yi[(size_t)v * 3 + 0];
                y[v] = yi[(size_t)v * 3 + 1];
                z[v] = yi[(size_t)v * 3 + 2];
            }
            errors(P, x, y, z, row);
            row.apply_s = best_seconds(repeats, [&] { bsr_apply_typed<V>(g, values.data(), xi.data(), yi.data()); });
            row.bytes = (double)values.size() * sizeof(V) + g.colidx.size() * sizeof(idx_t) +
                        g.rowptr.size() * sizeof(count_t);
            print_row(P, row);
        };
        bsr(double(0));
        bsr(float(0));

        // Quantized BSR (bsr_quant.hpp): the f32 assembly, then the encoding; set-up is both.
        // In place, the records are written over the values, which then stay allocated.
        auto qbsr = [&](auto store, const char *name, const bool in_place = false) {
            using Q = decltype(store);
            if (!selected(name)) return;
            std::vector<float> values((size_t)g.nnz() * 9);
            Q q(m.nnodes, g.rowptr.data(), g.colidx.data());
            auto setup = [&] {
                assemble_into(values);
                if constexpr (Q::upper) {
                    if (in_place) {
                        q.encode_in_place(g.rowptr.data(), g.colidx.data(), values.data());
                        return;
                    }
                }
                q.encode(g.rowptr.data(), g.colidx.data(), values.data());
            };
            Row row;
            row.name = name;
            row.setup_s = best_seconds(1, setup);
            if (!in_place) std::vector<float>().swap(values);
            q.template apply<float, false>(xi.data(), yi.data());
            std::vector<double> x(m.nnodes), y(m.nnodes), z(m.nnodes);
            for (ptrdiff_t v = 0; v < m.nnodes; ++v) {
                x[v] = yi[(size_t)v * 3 + 0];
                y[v] = yi[(size_t)v * 3 + 1];
                z[v] = yi[(size_t)v * 3 + 2];
            }
            errors(P, x, y, z, row);
            row.apply_s = best_seconds(repeats, [&] { q.template apply<float, false>(xi.data(), yi.data()); });
            row.bytes = (double)q.bytes();
            print_row(P, row);
        };
        qbsr(qp::QuantBsr<int8_t, qp::BsrForm::Difference, LANE_VS>(0, nullptr, nullptr), "BSR i8/blk diff");
        qbsr(qp::QuantBsr<int8_t, qp::BsrForm::Full, LANE_VS>(0, nullptr, nullptr), "BSR i8/blk full");
        qbsr(qp::QuantBsr<half_t, qp::BsrForm::Difference, LANE_VS>(0, nullptr, nullptr), "BSR f16 diff");
        qbsr(qp::QuantBsr<half_t, qp::BsrForm::Full, LANE_VS>(0, nullptr, nullptr), "BSR f16 full");
        qbsr(qp::QuantBsr<int8_t, qp::BsrForm::Upper, LANE_VS>(0, nullptr, nullptr), "BSR i8/blk upper");
        qbsr(qp::QuantBsr<half_t, qp::BsrForm::Upper, LANE_VS>(0, nullptr, nullptr), "BSR f16 upper");
        qbsr(qp::QuantBsr<int8_t, qp::BsrForm::Upper, LANE_VS, uint16_t>(0, nullptr, nullptr), "BSR i8/blk upper u16");
        qbsr(qp::QuantBsr<half_t, qp::BsrForm::Upper, LANE_VS, uint16_t>(0, nullptr, nullptr), "BSR f16 upper u16");
        qbsr(qp::QuantBsr<int8_t, qp::BsrForm::Upper, LANE_VS, uint16_t>(0, nullptr, nullptr), "BSR i8/blk upper u16 in place", true);
        qbsr(qp::QuantBsr<half_t, qp::BsrForm::Upper, LANE_VS, uint16_t>(0, nullptr, nullptr), "BSR f16 upper u16 in place", true);
    }

#ifndef NO_P0
    {  // the projected (P0) store of the companion paper, f32, as the library publishes it
        if (selected("projected P0 f32")) {
            const ptrdiff_t cstride = m.nelements + 64;
            std::vector<float> S32((size_t)cstride * 45);
            auto setup = [&] {
                sfem::codegen::TANGENT_KERNEL<double, geom_t, float, LANE_VS>(
                    m.nelements, m.kevp.data(), m.adj[0].data(), m.adj[1].data(), m.adj[2].data(), m.adj[3].data(),
                    m.adj[4].data(), m.adj[5].data(), m.adj[6].data(), m.adj[7].data(), m.adj[8].data(),
                    m.det.data(), P.lmbda, P.mu, 1, P.ux.data(), P.uy.data(), P.uz.data(), cstride, S32.data());
            };
            std::vector<double> x(m.nnodes), y(m.nnodes), z(m.nnodes);
            auto apply = [&] {
                sfem::codegen::STORED_APPLY<double, float, LANE_VS>(m.nelements, m.kevp.data(), cstride, S32.data(), 1,
                                                                    P.hx.data(), P.hy.data(), P.hz.data(), 1, x.data(),
                                                                    y.data(), z.data());
            };
            Row row;
            row.name = "projected P0 f32";
            row.setup_s = best_seconds(std::max(1, repeats / 4), setup);
            apply();
            errors(P, x, y, z, row);
            row.apply_s = best_seconds(repeats, apply);
            row.bytes = 45.0 * 4 * m.nelements;
            print_row(P, row);
        }
    }
#endif

    // The compressed quadrature-point store.  Name: code, scheme, layout, arithmetic, loop form
    // (lanes: lane loop innermost; regs: one lane loop around decode and contraction).
    // Uncompressed references and the SoA layout first, then the candidates.
    using qp::Form;
    run_variant<StoreT<double, Scheme::Plain, Layout::SoA>, double>("qp f64 soa a64", P, repeats);
    run_variant<StoreT<float, Scheme::Plain, Layout::Blocked>, float>("qp f32 blk a32", P, repeats);
    run_variant<StoreT<half_t, Scheme::Plain, Layout::SoA>, float>("qp f16 soa a32", P, repeats);
    run_variant<StoreT<int8_t, Scheme::Plain, Layout::SoA>, float>("qp i8/el soa a32", P, repeats);
    run_variant<StoreT<half_t, Scheme::Plain, Layout::Blocked>, double>("qp f16 blk a64", P, repeats);
    run_variant<StoreT<half_t, Scheme::Plain, Layout::Blocked>, float>("qp f16 blk a32 lanes", P, repeats);
    run_variant<StoreT<half_t, Scheme::Plain, Layout::Blocked>, float, Form::Registers>("qp f16 blk a32 regs", P, repeats);
    run_variant<StoreT<int8_t, Scheme::Plain, Layout::Blocked>, float>("qp i8/el blk a32 lanes", P, repeats);
    run_variant<StoreT<int8_t, Scheme::Plain, Layout::Blocked>, float, Form::Registers>("qp i8/el blk a32 regs", P, repeats);
    if constexpr (El::NV > 1) {
        run_variant<StoreT<int8_t, Scheme::PerComponent, Layout::Blocked>, float>("qp i8/comp blk a32 lanes", P, repeats);
        run_variant<StoreT<int8_t, Scheme::MeanDeviation, Layout::Blocked>, double>("qp i8 mean+dev blk a64 lanes", P, repeats);
        run_variant<StoreT<int8_t, Scheme::MeanDeviation, Layout::Blocked>, float>("qp i8 mean+dev blk a32 lanes", P, repeats);
        run_variant<StoreT<int8_t, Scheme::MeanDeviation, Layout::Blocked>, float, Form::Registers>("qp i8 mean+dev blk a32 regs", P, repeats);
        run_variant<StoreT<half_t, Scheme::MeanDeviation, Layout::Blocked>, float>("qp f16 mean+dev blk a32 lanes", P, repeats);
    }
    // Packed two-pass: the scatter the library's fastest operators use.
    // QP_PACK overrides the pack size: the library sizes packs to its 16-bit node indices, not
    // to the lane width, so unaligned packs are the case to test.
    // The default (and QP_PACK=auto) gives every thread about 20 packs, the best measured on Grace
    // for HEX8, TET10 and HEX27 at 10^7 dof (fewer, larger packs leave threads idle; more, smaller
    // ones pay the per-pack gather and write-back more often): at least 16 elements, at most 8192.
    ptrdiff_t pack_size = std::max<ptrdiff_t>(16, std::min<ptrdiff_t>(8192, (m.nelements / (20 * threads)) / 16 * 16));
    if (const char *qp_pack = std::getenv("QP_PACK")) {
        if (std::strcmp(qp_pack, "auto")) pack_size = std::atol(qp_pack);
    }
    PackedMeshView pk = build_packed_mesh(m, pack_size);
    std::printf("  packs of %ld elements: %ld packs\n", (long)pack_size, (long)pk.layout.n_packs);
#ifndef NO_P0
    if (selected("projected P0 f32 packed")) {
        const ptrdiff_t cstride = m.nelements + 64;
        std::vector<float> S32((size_t)cstride * 45);
        sfem::codegen::TANGENT_KERNEL<double, geom_t, float, LANE_VS>(
            m.nelements, m.kevp.data(), m.adj[0].data(), m.adj[1].data(), m.adj[2].data(), m.adj[3].data(),
            m.adj[4].data(), m.adj[5].data(), m.adj[6].data(), m.adj[7].data(), m.adj[8].data(), m.det.data(),
            P.lmbda, P.mu, 1, P.ux.data(), P.uy.data(), P.uz.data(), cstride, S32.data());
        std::vector<double> phx, phy, phz;
        place_packed(P.hx, pk.layout, phx);
        place_packed(P.hy, pk.layout, phy);
        place_packed(P.hz, pk.layout, phz);
        std::vector<double> ox(m.nnodes, 0), oy(m.nnodes, 0), oz(m.nnodes, 0);
        std::vector<double> ghost((size_t)pk.layout.n_ghost_entries * 3, 0.0);
        auto apply = [&] {
            sfem::codegen::PACKED_STORED_APPLY<double, float, LANE_VS>(
                pk.layout.n_packs, pk.layout.n_elements_per_pack, m.nelements, pk.layout.max_nodes_per_pack,
                pk.kernel_element_ptrs.data(), pk.layout.owned_nodes_ptr.data(), pk.layout.n_ghost_entries,
                pk.layout.n_ghost_reduce_rows, pk.layout.ghost_ptr.data(), pk.layout.ghost_idx.data(),
                pk.layout.ghost_reduce_ptr.data(), pk.layout.ghost_reduce_idx.data(), pk.layout.ghost_reduce_dest.data(),
                ghost.data(), cstride, S32.data(), 1, phx.data(), phy.data(), phz.data(), 1, ox.data(), oy.data(),
                oz.data());
        };
        apply();
        std::vector<double> x(m.nnodes), y(m.nnodes), z(m.nnodes);
        for (ptrdiff_t i = 0; i < m.nnodes; ++i) {
            const idx_t j = pk.layout.to_new[i];
            x[i] = ox[j];
            y[i] = oy[j];
            z[i] = oz[j];
        }
        Row row;
        row.name = "projected P0 f32 packed";
        errors(P, x, y, z, row);
        row.apply_s = best_seconds(repeats, apply);
        row.bytes = 45.0 * 4 * m.nelements;
        print_row(P, row);
    }
#endif
    if (selected("exact matrix-free packed")) {
        // The packed mesh renumbers nodes; adjugates and determinants are per element and unchanged.
        std::vector<double> pux, puy, puz, phx, phy, phz;
        place_packed(P.ux, pk.layout, pux);
        place_packed(P.uy, pk.layout, puy);
        place_packed(P.uz, pk.layout, puz);
        place_packed(P.hx, pk.layout, phx);
        place_packed(P.hy, pk.layout, phy);
        place_packed(P.hz, pk.layout, phz);
        std::vector<double> ox(m.nnodes, 0), oy(m.nnodes, 0), oz(m.nnodes, 0);
        std::vector<double> ghost((size_t)pk.layout.n_ghost_entries * 3, 0.0);
        std::vector<geom_t> qx(m.nnodes), qy(m.nnodes), qz(m.nnodes);
        for (ptrdiff_t v = 0; v < m.nnodes; ++v) {
            qx[v] = (geom_t)pk.mesh.px[v];
            qy[v] = (geom_t)pk.mesh.py[v];
            qz[v] = (geom_t)pk.mesh.pz[v];
        }
        const geom_t *const pk_points[3] = {qx.data(), qy.data(), qz.data()};
        (void)pk_points;
        auto apply = [&] {
            PACKED_EXACT_APPLY(SFEM_CODEGEN_F64, pk.layout.n_packs, pk.layout.n_elements_per_pack, m.nelements,
                               m.nnodes, pk.layout.max_nodes_per_pack, pk.layout.element_ptrs.data(),
                               pk.layout.owned_nodes_ptr.data(), pk.layout.n_shared_nodes.data(),
                               pk.layout.ghost_ptr.data(), pk.layout.ghost_idx.data(), pk.layout.n_ghost_entries,
                               pk.layout.n_ghost_reduce_rows, pk.layout.ghost_reduce_ptr.data(),
                               pk.layout.ghost_reduce_idx.data(), pk.layout.ghost_reduce_dest.data(), ghost.data(),
#ifdef EXACT_ISO
                               pk_points,
#else
                               pk.mesh.adj[0].data(), pk.mesh.adj[1].data(), pk.mesh.adj[2].data(),
                               pk.mesh.adj[3].data(), pk.mesh.adj[4].data(), pk.mesh.adj[5].data(),
                               pk.mesh.adj[6].data(), pk.mesh.adj[7].data(), pk.mesh.adj[8].data(),
                               pk.mesh.det.data(),
#endif
                               P.lmbda, P.mu, 1, pux.data(), puy.data(), puz.data(), 1,
                               phx.data(), phy.data(), phz.data(), 1, ox.data(), oy.data(), oz.data());
        };
        apply();
        std::vector<double> x(m.nnodes), y(m.nnodes), z(m.nnodes);
        for (ptrdiff_t i = 0; i < m.nnodes; ++i) {
            const idx_t j = pk.layout.to_new[i];
            x[i] = ox[j];
            y[i] = oy[j];
            z[i] = oz[j];
        }
        Row row;
        row.name = "exact matrix-free packed two-pass";
        errors(P, x, y, z, row);
        row.apply_s = best_seconds(repeats, apply);
        print_row(P, row);
    }
    run_variant_packed<StoreT<float, Scheme::Plain, Layout::Blocked>, float>("qp f32 blk a32 packed", P, pk, repeats);
    run_variant_packed<StoreT<half_t, Scheme::Plain, Layout::Blocked>, float>("qp f16 blk a32 lanes packed", P, pk, repeats);
    run_variant_packed<StoreT<half_t, Scheme::Plain, Layout::Blocked>, float, Form::Registers>("qp f16 blk a32 regs packed", P, pk, repeats);
    run_variant_packed<StoreT<int8_t, Scheme::Plain, Layout::Blocked>, float>("qp i8/el blk a32 lanes packed", P, pk, repeats);
    if constexpr (El::NV > 1)
    {
        run_variant_packed<StoreT<int8_t, Scheme::MeanDeviation, Layout::Blocked>, float>("qp i8 mean+dev blk a32 lanes packed", P, pk, repeats);
        run_variant_packed<StoreT<half_t, Scheme::MeanDeviation, Layout::Blocked>, float>("qp f16 mean+dev blk a32 lanes packed", P, pk, repeats);
    }

    // Element matrices, from the same set-up kernel, quantized like the point store.
    using qp::EaForm;
    run_variant_packed<qp::EaStore<El, double, EaForm::Full, LANE_VS>, double>("ea full f64 packed", P, pk, repeats);
    run_variant_packed<qp::EaStore<El, float, EaForm::Full, LANE_VS>, float>("ea full f32 packed", P, pk, repeats);
    run_variant_packed<qp::EaStore<El, half_t, EaForm::Full, LANE_VS>, float>("ea full f16 packed", P, pk, repeats);
    run_variant_packed<qp::EaStore<El, int8_t, EaForm::Full, LANE_VS>, float>("ea full i8 packed", P, pk, repeats);
    run_variant_packed<qp::EaStore<El, double, EaForm::Difference, LANE_VS>, double>("ea diff f64 packed", P, pk, repeats);
    run_variant_packed<qp::EaStore<El, float, EaForm::Difference, LANE_VS>, float>("ea diff f32 packed", P, pk, repeats);
    run_variant_packed<qp::EaStore<El, half_t, EaForm::Difference, LANE_VS>, float>("ea diff f16 packed", P, pk, repeats);
    run_variant_packed<qp::EaStore<El, int8_t, EaForm::Difference, LANE_VS>, float>("ea diff i8 packed", P, pk, repeats);
    run_variant_packed<qp::EaStore<El, int8_t, EaForm::PairScale, LANE_VS>, float>("ea diff i8 pair packed", P, pk, repeats);
    run_variant<qp::EaStore<El, half_t, EaForm::Difference, LANE_VS>, float>("ea diff f16 atomic", P, repeats);

    return 0;
}
