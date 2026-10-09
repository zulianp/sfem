// The curvature partition the semi-structured geometry split runs on.
//
// Whether a macro element is curved is mesh data, not a configuration: one mesh carries curved
// and straight macro elements side by side and which is which is known only per element. So the
// affine and isoparametric sweeps cannot be selected by a template parameter; they are two
// sweeps over two ranges of SSMeshData::macro_order, which sscvfem_classify_macros partitions
// once per level. Everything that split buys rests on that array being right.
//
// A wrong partition is quiet in the worst way. Put a curved macro element in the straight range
// and its micro cells get a hoisted geometry -- the bug cvfem_flat_vs_ss_test's warped arms
// exist to catch. But merely PERMUTING the order, or getting n_straight off by one in the
// direction that puts a straight element in the curved range, changes no answer at all: the
// isoparametric sweep is correct for a straight element too, just slower. Nothing downstream
// would ever report it. So the invariants are asserted here directly.

#include "frontend/ss/cvfem_sshex8_ns.hpp"

#include "sfem_context.hpp"
#include "smesh_mesh.hpp"
#include "smesh_semistructured.hpp"

#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

static int g_failures = 0;

static void check(const bool ok, const char *what) {
    std::printf("%-64s %s\n", what, ok ? "OK" : "FAIL");
    if (!ok) ++g_failures;
}

// The same warp cvfem_flat_vs_ss_test and cvfem_sshex8_bench use: a product of sines, which
// vanishes on all six faces of the box so the boundary stays where the coordinate-selected
// closure expects it, confined to X < L/2 so the mesh carries both kinds of macro element.
static void warp(const std::shared_ptr<smesh::Mesh> &m, const double L, const double amp) {
    auto *const px = m->points()->data()[0];
    auto *const py = m->points()->data()[1];
    auto *const pz = m->points()->data()[2];
    for (ptrdiff_t i = 0; i < m->n_nodes(); ++i) {
        const double X = (double)px[i];
        const double w = (X < 0.5 * L ? std::sin(2 * M_PI * X / L) : 0.0) *
                         std::sin(M_PI * (double)py[i] / L) * std::sin(M_PI * (double)pz[i] / L);
        px[i] = (geom_t)((double)px[i] + amp * w);
        py[i] = (geom_t)((double)py[i] + amp * w * 0.7);
        pz[i] = (geom_t)((double)pz[i] - amp * w * 0.4);
    }
}

static void partition_of(sfem::Context &ctx, const int macro, const int level, const double warp_frac,
                         const char *what) {
    const double L    = 1;
    auto         mesh = smesh::Mesh::create_hex8_cube(ctx.communicator(), macro, macro, macro, 0, 0, 0, L, L, L);
    mesh              = smesh::to_semistructured(level, mesh, true, false);
    if (warp_frac > 0) warp(mesh, L, warp_frac * L / (macro * level));

    SSMeshData d;
    sscvfem_init(d, mesh, level);

    char msg[200];
    const ptrdiff_t n_curved = (ptrdiff_t)d.macro_curved.size() == d.nmacro
                                       ? [&] {
                                             ptrdiff_t n = 0;
                                             for (ptrdiff_t e = 0; e < d.nmacro; ++e) n += sscvfem_macro_curved(d.macro_curved.empty() ? nullptr : d.macro_curved.data(), e) ? 1 : 0;
                                             return n;
                                         }()
                                       : 0;

    // An all-straight mesh takes the identity path: no order array at all, so the sweeps index
    // the element directly and nothing about a box changes.
    if (n_curved == 0) {
        std::snprintf(msg, sizeof(msg), "%s: nothing curved, so no order array", what);
        check(d.macro_order.empty(), msg);
        std::snprintf(msg, sizeof(msg), "%s: the whole range is straight", what);
        check(d.n_straight == d.nmacro, msg);
        std::printf("  %s: %td macro elements, none curved\n", what, d.nmacro);
        return;
    }

    std::snprintf(msg, sizeof(msg), "%s: the mesh carries both kinds", what);
    check(n_curved > 0 && n_curved < d.nmacro, msg);

    std::snprintf(msg, sizeof(msg), "%s: the order array covers every macro element", what);
    check((ptrdiff_t)d.macro_order.size() == d.nmacro, msg);

    std::snprintf(msg, sizeof(msg), "%s: n_straight accounts for every element", what);
    check(d.n_straight == d.nmacro - n_curved, msg);

    // A permutation, not merely a list of valid indices: every element exactly once.
    std::vector<int> seen((size_t)d.nmacro, 0);
    bool             in_range = true;
    for (ptrdiff_t i = 0; i < (ptrdiff_t)d.macro_order.size(); ++i) {
        const ptrdiff_t e = d.macro_order[(size_t)i];
        if (e < 0 || e >= d.nmacro) { in_range = false; break; }
        ++seen[(size_t)e];
    }
    std::snprintf(msg, sizeof(msg), "%s: every entry is a macro element", what);
    check(in_range, msg);
    bool once = in_range;
    for (ptrdiff_t e = 0; e < d.nmacro && once; ++e) once = seen[(size_t)e] == 1;
    std::snprintf(msg, sizeof(msg), "%s: it is a permutation", what);
    check(once, msg);

    // The two ranges hold what their sweeps assume, which is the invariant the split rests on.
    bool straight_clean = true, curved_clean = true, ascending = true;
    for (ptrdiff_t i = 0; i < d.n_straight; ++i)
        if (sscvfem_macro_curved(d.macro_curved.empty() ? nullptr : d.macro_curved.data(), d.macro_order[(size_t)i])) straight_clean = false;
    for (ptrdiff_t i = d.n_straight; i < d.nmacro; ++i)
        if (!sscvfem_macro_curved(d.macro_curved.empty() ? nullptr : d.macro_curved.data(), d.macro_order[(size_t)i])) curved_clean = false;
    for (ptrdiff_t i = 1; i < d.n_straight; ++i)
        if (d.macro_order[(size_t)i] <= d.macro_order[(size_t)i - 1]) ascending = false;
    for (ptrdiff_t i = d.n_straight + 1; i < d.nmacro; ++i)
        if (d.macro_order[(size_t)i] <= d.macro_order[(size_t)i - 1]) ascending = false;

    std::snprintf(msg, sizeof(msg), "%s: the affine range holds no curved element", what);
    check(straight_clean, msg);
    std::snprintf(msg, sizeof(msg), "%s: the isoparametric range holds only curved ones", what);
    check(curved_clean, msg);
    // Ascending within each part, which is what keeps a mesh with nothing curved identical and
    // keeps the element order a sweep visits as close to the original as the partition allows.
    std::snprintf(msg, sizeof(msg), "%s: each part is in ascending element order", what);
    check(ascending, msg);

    std::printf("  %s: %td macro elements, %td straight then %td curved\n", what, d.nmacro,
                d.n_straight, d.nmacro - d.n_straight);

    // THE PARTIAL RANGE, which is the subtle part. A sweep pair is usually launched over the
    // whole mesh, but the nodal gradient hands the scatter sweep the element tail the packs do
    // not reach on a distributed mesh. sscvfem_order_positions restricts each half of the
    // partition to that element range with a binary search, which is only correct because each
    // half is ascending -- so this checks the two position ranges together cover exactly the
    // elements in [b, e), each once, and nothing outside it.
    const ptrdiff_t cuts[] = {0, 1, d.nmacro / 3, d.n_straight, d.nmacro - 1, d.nmacro};
    bool            exact = true;
    for (const ptrdiff_t b : cuts)
        for (const ptrdiff_t e : cuts) {
            if (e < b) continue;
            std::vector<int> hit((size_t)d.nmacro, 0);
            for (int half = 0; half < 2 && exact; ++half) {
                const cvfem_range p = sscvfem_order_positions(d, half == 1, b, e);
                if (p.begin > p.end) { exact = false; break; }
                for (ptrdiff_t i = p.begin; i < p.end; ++i) {
                    const ptrdiff_t el = d.macro_order[(size_t)i];
                    if (el < b || el >= e) { exact = false; break; }
                    ++hit[(size_t)el];
                }
            }
            for (ptrdiff_t el = 0; el < d.nmacro && exact; ++el)
                if (hit[(size_t)el] != (el >= b && el < e ? 1 : 0)) exact = false;
            if (!exact) {
                std::printf("    restricted to [%td, %td) is wrong\n", b, e);
                break;
            }
        }
    std::snprintf(msg, sizeof(msg), "%s: a partial element range restricts exactly", what);
    check(exact, msg);
}

int main(int argc, char **argv) {
    auto ctx = sfem::initialize(argc, argv);
    partition_of(*ctx, 3, 2, 0.0, "box, level 2");
    partition_of(*ctx, 3, 4, 0.0, "box, level 4");
    partition_of(*ctx, 3, 2, 0.25, "warped, level 2");
    partition_of(*ctx, 3, 4, 0.25, "warped, level 4");
    partition_of(*ctx, 6, 2, 0.25, "warped, level 2, finer");

    if (g_failures) {
        std::fprintf(stderr, "\ncvfem_ss_curvature_partition_test: %d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("\nthe curvature partition is a permutation and each range holds its own kind\n");
    return 0;
}
