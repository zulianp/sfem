// The flat and semi-structured operators, on the same mesh, must be the same operator.
//
// A semi-structured mesh at internal level L whose macro count is n/L describes exactly the
// same lattice of hexahedra as a flat mesh of n cells. The discretisation does not know the
// difference -- a micro cell is a cell -- so the residual must agree to round-off.
//
// It does not. Poiseuille, whose exact solution the scheme represents essentially exactly,
// converges to u_linf 1.18e-11 flat and 5.62e-07 at level 2 and 7.95e-02 at level 3 on the
// identical 33,124-dof mesh, with no boundary condition involved and independently of the
// linear tolerance. Nothing caught it: the manufactured solution cannot see it, its own
// discretisation error being three orders larger, and the multigrid work was measured on
// convergence rates rather than against an exact solution.
//
// This compares the two operators directly rather than through a solve, so a difference is
// attributed to the operator and not to a solver, and reports it per field so the term is
// identifiable. Keyed by node COORDINATE: the two meshes number their nodes differently.

#include "cvfem_hex8_ns_op.hpp"

#include "sfem_Function.hpp"
#include "sfem_context.hpp"
#include "smesh_mesh.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <vector>

static int g_failures = 0;

static void check(const bool ok, const char *what) {
    std::printf("%-62s %s\n", what, ok ? "OK" : "FAIL");
    if (!ok) ++g_failures;
}

static constexpr int    NF  = 4;
static constexpr double LX = 1, LY = 1, LZ = 1;

// A state that is a function of position, so it is the same field on both numberings.
static std::vector<real_t> state_of(const std::shared_ptr<smesh::Mesh> &m) {
    const ptrdiff_t   n  = m->n_nodes();
    const auto *const px = m->points()->data()[0];
    const auto *const py = m->points()->data()[1];
    const auto *const pz = m->points()->data()[2];
    std::vector<real_t> x((size_t)n * NF, 0);
    for (ptrdiff_t i = 0; i < n; ++i) {
        const real_t a = px[i], b = py[i], c = pz[i];
        x[(size_t)i * NF + 0] = 0.7 + 0.13 * a - 0.21 * b + 0.05 * c;
        x[(size_t)i * NF + 1] = -0.3 + 0.11 * a + 0.17 * b - 0.04 * c;
        x[(size_t)i * NF + 2] = 0.2 - 0.07 * a + 0.09 * b + 0.12 * c;
        x[(size_t)i * NF + 3] = 1.0 + 0.05 * a + 0.03 * b;
    }
    return x;
}

// Residual keyed by rounded coordinate.
static std::map<std::array<long long, 3>, std::array<real_t, NF>> residual_of(
        sfem::Context &ctx, const int cells, const int level, const real_t rc_scale) {
    const int macro = cells / level;
    auto      mesh  = smesh::Mesh::create_hex8_cube(ctx.communicator(), macro, macro, macro,
                                                    0, 0, 0, LX, LY, LZ);
    if (level > 1) mesh = smesh::to_semistructured(level, mesh, true, false);

    auto fs = sfem::FunctionSpace::create(mesh, NF);
    auto op = std::make_shared<sfem::CVFEMNavierStokes>(fs);
    op->rho             = 1;
    op->mu              = 0.05;
    op->rhie_chow_scale = rc_scale;
    op->pack_size       = 0;  // the atomic layout, so the packed renumbering is not in play
    op->initialize();

    auto f = sfem::Function::create(fs);
    f->add_operator(op);
    const auto          x = state_of(mesh);
    std::vector<real_t> r((size_t)mesh->n_nodes() * NF, 0);
    f->gradient(x.data(), r.data());

    std::map<std::array<long long, 3>, std::array<real_t, NF>> out;
    const auto *const px = mesh->points()->data()[0];
    const auto *const py = mesh->points()->data()[1];
    const auto *const pz = mesh->points()->data()[2];
    for (ptrdiff_t i = 0; i < mesh->n_nodes(); ++i) {
        out[{(long long)std::llround(px[i] * 1e6), (long long)std::llround(py[i] * 1e6),
             (long long)std::llround(pz[i] * 1e6)}] = {r[(size_t)i * NF + 0], r[(size_t)i * NF + 1],
                                                       r[(size_t)i * NF + 2], r[(size_t)i * NF + 3]};
    }
    return out;
}

static void compare(sfem::Context &ctx, const int cells, const int level, const real_t rc) {
    const auto flat = residual_of(ctx, cells, 1, rc);
    const auto ss   = residual_of(ctx, cells, level, rc);

    char msg[160];
    std::snprintf(msg, sizeof(msg), "level %d, rc %.1f: both see the same node set", level, (double)rc);
    check(flat.size() == ss.size(), msg);

    real_t                  worst[NF] = {0, 0, 0, 0}, scale[NF] = {0, 0, 0, 0};
    ptrdiff_t               matched = 0, n_bad = 0, n_bad_bnd = 0, n_bad_int = 0;
    std::array<long long, 3> worst_at{0, 0, 0};
    real_t                  worst_any = 0;
    for (const auto &kv : flat) {
        const auto it = ss.find(kv.first);
        if (it == ss.end()) continue;
        ++matched;
        real_t here = 0;
        for (int c = 0; c < NF; ++c) {
            const real_t dc = (real_t)std::fabs(kv.second[c] - it->second[c]);
            here     = std::max(here, dc);
            worst[c] = std::max(worst[c], dc);
            scale[c] = std::max(scale[c], (real_t)std::fabs(kv.second[c]));
        }
        if (here > worst_any) {
            worst_any = here;
            worst_at  = kv.first;
        }
        // Where the difference lives says which term it is. A node on the bounding box is
        // reached by the boundary closure; one strictly inside is only ever touched by the
        // volume kernel, so a difference there cannot be a boundary-condition question.
        if (here > 1e-12) {
            ++n_bad;
            const bool bnd = kv.first[0] == 0 || kv.first[1] == 0 || kv.first[2] == 0 ||
                             kv.first[0] == (long long)std::llround(LX * 1e6) ||
                             kv.first[1] == (long long)std::llround(LY * 1e6) ||
                             kv.first[2] == (long long)std::llround(LZ * 1e6);
            if (bnd) ++n_bad_bnd; else ++n_bad_int;
        }
    }
    real_t rel = 0;
    std::printf("  cells %d, level %d, rhie_chow %.1f: %td flat nodes, %td ss nodes, %td at the same place\n",
                cells, level, (double)rc, (ptrdiff_t)flat.size(), (ptrdiff_t)ss.size(), matched);
    static const char *nm[NF] = {"ux", "uy", "uz", "p "};
    for (int c = 0; c < NF; ++c) {
        const real_t rc_ = scale[c] > 0 ? worst[c] / scale[c] : worst[c];
        rel              = std::max(rel, rc_);
        std::printf("    %s  |flat - ss| %.6e   relative %.6e\n", nm[c], (double)worst[c], (double)rc_);
    }
    std::printf("    worst at (%.4f %.4f %.4f);  %td node(s) differ by >1e-12: %td on the boundary, "
                "%td strictly interior\n",
                worst_at[0] * 1e-6, worst_at[1] * 1e-6, worst_at[2] * 1e-6, n_bad, n_bad_bnd, n_bad_int);
    // 1e-3 and not round-off, deliberately.
    //
    // The two meshes are built independently and geom_t is single precision, so their node
    // coordinates differ by about 6e-08 -- measured, and not a defect of either. This
    // operator differentiates, so it divides by the cell size and amplifies that by roughly
    // 35x, landing at 1e-05 relative. Demanding 1e-14 here would be demanding that two
    // float32 meshes be bit-identical, which they are not and need not be.
    //
    // The threshold still has teeth for the thing it is here to catch: the hoisted geometry
    // at a non-power-of-two level reads 37 -- seven orders the wrong side of this line.
    std::snprintf(msg, sizeof(msg), "level %d, rc %.1f: the residuals agree", level, (double)rc);
    check(rel < 1e-3, msg);
}

int main(int argc, char **argv) {
    auto ctx = sfem::initialize(argc, argv);
    // Rhie-Chow on, then off. The term is the one thing in this operator that is built from
    // a length scale rather than from the state, so if the difference is in it, turning it
    // off removes the difference and names the culprit in one line.
    // Powers of two, which is what the operator accepts and what it is exact for. The
    // levels it refuses -- 3 and 6, where the hoisted geometry is not congruent in single
    // precision -- are checked by cvfem_ss_level_refused below rather than here, because
    // this test would have to assert a known-wrong answer to cover them.
    compare(*ctx, 12, 2, 1);
    compare(*ctx, 12, 4, 1);
    compare(*ctx, 12, 2, 0);
    compare(*ctx, 12, 4, 0);

    if (g_failures) {
        std::fprintf(stderr, "\ncvfem_flat_vs_ss_test: %d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("\nthe flat and semi-structured operators agree\n");
    return 0;
}
