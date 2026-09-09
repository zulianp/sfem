// The packed and atomic residuals must be the same operator, on a non-box domain too.
//
// They agree to every digit on a box, and cvfem_ns_op_gate checks that. Nothing checked a
// domain with a re-entrant face -- and that is where the two paths differ in structure:
// the atomic sweep closes each element's boundary control volumes inline, while the packed
// sweep does the volume term over packs and closes the boundary in a separate pass
// afterwards. Anything the second arrangement gets wrong is invisible on a box, because
// there a coordinate test and the real boundary agree by construction.
//
// The comparison is keyed by node COORDINATE, not by index: creating a PackedMesh
// renumbers the mesh nodes in place, so the two runs need separate mesh objects and index
// equality means nothing between them.
//
// The measures are chosen so a difference cannot hide. The continuity sum is the net mass
// flux through the domain and is renumbering-invariant; the per-node worst case says
// whether a disagreement is everywhere or on a handful of faces.

#include "cvfem_hex8_ns_core.hpp"

#include "sfem_context.hpp"
#include "smesh_mesh.hpp"
#include "smesh_sideset.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <map>
#include <vector>

static int g_failures = 0;

static void check(const bool ok, const char *what) {
    std::printf("%-56s %s\n", what, ok ? "OK" : "FAIL");
    if (!ok) ++g_failures;
}

struct Run {
    std::shared_ptr<smesh::Mesh> mesh;
    MeshData                     d;
    PackedData                   packed;
};

// The same problem set up twice: once with the packed decomposition, once without.
static void build(Run &r, const int pack_size, const bool lshape, sfem::Context &ctx,
                  const bool skin_before_pack = false) {
    if (lshape) {
        // 40 x 8 x 4 with a 4 x 4 notch: the backward-facing step's geometry, and the only
        // non-box domain the spike has.
        r.mesh = smesh::Mesh::create_hex8_lshape(ctx.communicator(), 40, 8, 4, 10, 2, 1, 1, 1);
    } else {
        r.mesh = smesh::Mesh::create_hex8_cube(ctx.communicator(), 8, 4, 4, 0, 0, 0, 10, 2, 1);
    }
    MeshData &d = r.d;
    d.mesh      = r.mesh;
    d.Lx = 10; d.Ly = 2; d.Lz = 1;
    // Rhie-Chow on. With it off the Jacobian comparison below is vacuous, because the term
    // whose derivative went missing is not evaluated at all.
    d.rhie_chow_scale = 1;
    // The step driver names its sidesets before the operator is initialized, so a skin has
    // already been extracted by the time the packed layout renumbers the nodes. Reproduce
    // that order: it is the order under which the two paths disagreed.
    if (skin_before_pack) (void)smesh::skin_sideset(d.mesh);
    if (pack_size > 0) {
        r.packed = make_packed(d.mesh, pack_size);  // renumbers d.mesh's nodes in place
        d.packed = &r.packed;
    }
    d.nnodes    = r.mesh->n_nodes();
    d.nelements = r.mesh->n_elements(0);
    d.elems     = r.mesh->elements(0)->data();
    d.points    = r.mesh->points()->data();
    cvfem_hex8_precompute_affine_geometry(d);

    const auto *const px = d.points[0];
    const auto *const py = d.points[1];
    const auto *const pz = d.points[2];
    d.ux.resize((size_t)d.nnodes);
    d.uy.resize((size_t)d.nnodes);
    d.uz.resize((size_t)d.nnodes);
    d.p.resize((size_t)d.nnodes);
    d.rx.assign((size_t)d.nnodes, 0);
    d.ry.assign((size_t)d.nnodes, 0);
    d.rz.assign((size_t)d.nnodes, 0);
    d.rc.assign((size_t)d.nnodes, 0);
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        // A function of position, so the state is identical between the two runs even
        // though their node numbering is not.
        const scalar_t x = px[i], y = py[i], z = pz[i];
        d.ux[(size_t)i] = 0.7 + 0.13 * x - 0.21 * y + 0.05 * z;
        d.uy[(size_t)i] = -0.3 + 0.11 * x + 0.17 * y - 0.04 * z;
        d.uz[(size_t)i] = 0.2 - 0.07 * x + 0.09 * y + 0.12 * z;
        d.p[(size_t)i]  = 1.0 + 0.05 * x + 0.03 * y;
    }
}

// Residual keyed by rounded coordinate, so the two numberings can be compared.
static std::map<std::array<long long, 3>, std::array<scalar_t, 4>> keyed_residual(Run &r) {
    apply_residual(r.d, scalar_t(1), scalar_t(0.05), GeomKind::Affine);
    std::map<std::array<long long, 3>, std::array<scalar_t, 4>> out;
    const auto *const px = r.d.points[0];
    const auto *const py = r.d.points[1];
    const auto *const pz = r.d.points[2];
    for (ptrdiff_t i = 0; i < r.d.nnodes; ++i) {
        const std::array<long long, 3> key{(long long)std::llround(px[i] * 1e6),
                                           (long long)std::llround(py[i] * 1e6),
                                           (long long)std::llround(pz[i] * 1e6)};
        out[key] = {r.d.rx[(size_t)i], r.d.ry[(size_t)i], r.d.rz[(size_t)i], r.d.rc[(size_t)i]};
    }
    return out;
}

static void compare(sfem::Context &ctx, const bool lshape, const char *name) {
    Run a, b;
    build(a, 0, lshape, ctx);     // atomic
    build(b, 2048, lshape, ctx);  // packed
    const auto ra = keyed_residual(a);
    const auto rb = keyed_residual(b);

    std::printf("\n--- %s: %td nodes, %td elements ---\n", name, a.d.nnodes, a.d.nelements);
    char msg[192];
    std::snprintf(msg, sizeof(msg), "%s: both runs see the same node set", name);
    check(ra.size() == rb.size() && ra.size() == (size_t)a.d.nnodes, msg);

    scalar_t worst = 0, sum_a = 0, sum_b = 0;
    ptrdiff_t n_bad = 0;
    for (const auto &kv : ra) {
        const auto it = rb.find(kv.first);
        if (it == rb.end()) continue;
        for (int c = 0; c < 4; ++c) {
            const scalar_t d = std::fabs(kv.second[c] - it->second[c]);
            if (d > 1e-12) ++n_bad;
            worst = std::max(worst, d);
        }
        sum_a += kv.second[3];
        sum_b += it->second[3];
    }
    std::printf("  worst |packed - atomic| per dof : %.6e  (%td dofs over 1e-12)\n", (double)worst, n_bad);
    std::printf("  net mass flux  atomic %.9e   packed %.9e\n", (double)sum_a, (double)sum_b);

    std::snprintf(msg, sizeof(msg), "%s: packed and atomic residuals agree", name);
    check(worst < 1e-10, msg);
    std::snprintf(msg, sizeof(msg), "%s: net mass flux agrees", name);
    check(std::fabs(sum_a - sum_b) < 1e-10, msg);
}

// The Dirichlet set must not depend on the layout.
//
// The step driver takes a skin sideset before the operator is initialized and another one
// after, and constrains the second. Packing renumbers the mesh nodes in between. If any
// node-indexed structure cached by the first call survives the renumbering, the second call
// describes the old numbering and the constrained set is wrong -- silently, because the
// solve still runs and still converges, to a different problem. That is exactly what
// happened: the step case constrained 1647 of 1765 nodes packed against 949 unpacked, and
// the residuals differed by 14x while the operator itself was identical to 5e-17.
//
// Keyed by coordinate, like the residual comparison, and checked against the closed form
// so a failure says which of the two is wrong rather than only that they differ.
static void compare_skin(sfem::Context &ctx, const ptrdiff_t expect) {
    Run a, b;
    build(a, 0, true, ctx, /*skin_before_pack=*/true);     // atomic
    build(b, 2048, true, ctx, /*skin_before_pack=*/true);  // packed

    auto skin_coords = [](Run &r) {
        std::vector<std::array<long long, 3>> out;
        auto skin = smesh::skin_sideset(r.d.mesh);
        if (!skin) return out;
        auto ns = smesh::create_nodeset_from_sideset(r.d.mesh, skin);
        if (!ns) return out;
        const auto *const px = r.d.points[0];
        const auto *const py = r.d.points[1];
        const auto *const pz = r.d.points[2];
        for (ptrdiff_t k = 0; k < ns->size(); ++k) {
            const idx_t i = ns->data()[k];
            out.push_back({(long long)std::llround(px[i] * 1e6),
                           (long long)std::llround(py[i] * 1e6),
                           (long long)std::llround(pz[i] * 1e6)});
        }
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        return out;
    };

    const auto sa = skin_coords(a);
    const auto sb = skin_coords(b);
    std::printf("\n--- L-shape skin after a pre-pack skin call ---\n");
    std::printf("  skin nodes  atomic %zu   packed %zu   (closed form %td)\n", sa.size(), sb.size(), expect);

    check(sa.size() == (size_t)expect, "the unpacked skin matches the closed form");
    check(sb.size() == (size_t)expect, "the packed skin matches the closed form");
    check(sa == sb, "packed and unpacked select the same skin nodes");
}

// The Jacobian ACTION, which is where the two paths actually diverged.
//
// The residual comparison above passed for months while the packed Jacobian was missing the
// derivative of the Rhie-Chow pressure-gradient reconstruction -- the residual carries the
// term, so the residuals agreed; only the Jacobian did not. It cost the backward-facing step
// its convergence: an 11.7% error in the continuity rows against a finite difference, which
// does not make any single Newton step fail, it just caps Newton at a linear rate, which
// looks like a hard problem rather than a bug.
//
// Nothing here checked it. tests/cvfem_ns_op_gate.cpp runs the whole gate with
// SFEM_RC_EXACT_JAC=0 on purpose, so the exact term is off there by construction.
static void compare_jacobian(sfem::Context &ctx, const bool lshape, const char *name) {
    Run a, b;
    build(a, 0, lshape, ctx);     // atomic
    build(b, 2048, lshape, ctx);  // packed

    auto keyed_jv = [](Run &r) {
        MeshData &d = r.d;
        assemble_nodal_p_grad(d, GeomKind::Affine);
        // A direction that is a function of position, so it is the same field on both
        // numberings even though the node indices are not.
        const auto *const px = d.points[0];
        const auto *const py = d.points[1];
        const auto *const pz = d.points[2];
        std::vector<scalar_t> dir((size_t)d.nnodes * N_FIELDS, 0), jv((size_t)d.nnodes * N_FIELDS, 0);
        for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
            const scalar_t x = px[i], y = py[i], z = pz[i];
            dir[(size_t)i * 4 + 0] = 0.31 - 0.12 * x + 0.07 * y;
            dir[(size_t)i * 4 + 1] = -0.17 + 0.09 * x - 0.05 * z;
            dir[(size_t)i * 4 + 2] = 0.23 + 0.04 * y - 0.11 * z;
            dir[(size_t)i * 4 + 3] = 0.5 - 0.13 * x + 0.21 * y - 0.03 * z;
        }
        apply_jacobian_action_accumulate(d, scalar_t(1), scalar_t(0.05), GeomKind::Affine,
                                         dir.data(), jv.data());
        std::map<std::array<long long, 3>, std::array<scalar_t, 4>> out;
        for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
            const std::array<long long, 3> key{(long long)std::llround(px[i] * 1e6),
                                               (long long)std::llround(py[i] * 1e6),
                                               (long long)std::llround(pz[i] * 1e6)};
            out[key] = {jv[(size_t)i * 4 + 0], jv[(size_t)i * 4 + 1],
                        jv[(size_t)i * 4 + 2], jv[(size_t)i * 4 + 3]};
        }
        return out;
    };

    const auto ja = keyed_jv(a);
    const auto jb = keyed_jv(b);
    scalar_t worst = 0, scale = 0;
    for (const auto &kv : ja) {
        const auto it = jb.find(kv.first);
        if (it == jb.end()) continue;
        for (int c = 0; c < 4; ++c) {
            worst = std::max(worst, std::fabs(kv.second[c] - it->second[c]));
            scale = std::max(scale, std::fabs(kv.second[c]));
        }
    }
    const scalar_t rel = scale > 0 ? worst / scale : worst;
    std::printf("\n--- %s Jacobian action (Rhie-Chow on) ---\n", name);
    std::printf("  worst |packed - atomic| = %.6e   relative %.6e\n", (double)worst, (double)rel);
    char msg[192];
    std::snprintf(msg, sizeof(msg), "%s: packed and atomic Jacobian actions agree", name);
    check(rel < 1e-12, msg);
}

int main(int argc, char **argv) {
    auto ctx = sfem::initialize(argc, argv);
    // The box first: it is the case that already works, so if it fails the harness is at
    // fault rather than the code under test.
    compare(*ctx, false, "box");
    compare(*ctx, true, "L-shape");
    // 1765 nodes on the 40x8x4 L-shape, 771 of them strictly interior.
    compare_skin(*ctx, 994);
    compare_jacobian(*ctx, false, "box");
    compare_jacobian(*ctx, true, "L-shape");

    if (g_failures) {
        std::fprintf(stderr, "\ncvfem_packed_vs_atomic_test: %d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("\npacked and atomic agree on both domains\n");
    return 0;
}
