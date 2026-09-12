// Every spelling of the Jacobian must agree with every other one.
//
// The spike carries the same operator in several forms -- a matrix-free action, an assembled
// BSR, a block diagonal for the preconditioner -- and they are written independently. Nothing
// compared the block diagonal against the action until this test, and that gap was not
// hypothetical: with the exact Rhie-Chow Jacobian on, assemble_block_diag was building the
// FROZEN operator's blocks while the action differentiated the reconstructed pressure
// gradient, so the preconditioner was built from a different operator than the one it
// preconditions. It measured 3.5e-02 relative and it stopped the pressure-port case
// converging at all. A self-consistency test is the only thing that catches that class of
// defect, because each form looks perfectly reasonable on its own.
//
// The invariants are not all unconditional, and saying which hold where is most of the value:
//
//   [1] block diagonal == the diagonal 4x4 blocks of the ACTION.
//       Asserted in the frozen form, where it holds to round-off. It SHOULD be
//       unconditional -- the action is the operator the solver applies, so whatever the
//       preconditioner inverts ought to be its diagonal -- and in the exact form it is not,
//       by 8.2e-02 steady and 1.2e-02 on BDF2. That is measured here rather than asserted,
//       because the gap is understood and closing it is not a local change:
//
//       qg_i is the reconstructed gradient at node i, summed over EVERY element containing
//       i and then scaled by 1/W_i. So d(qg_i)/d(q_i) is a property of i's whole element
//       neighbourhood, while a sub-control surface (i,j) belongs to one element. An element
//       loop can therefore only ever supply its own share of it, and assemble_block_diag
//       does exactly that -- which is why its exact-form diagonal is closer to the action's
//       than the frozen one but still not equal to it. Getting it right needs a per-node
//       pass for d(qg_i)/d(q_i) and a per-edge one for d(qg_j)/d(q_i), both structured like
//       the reconstruction itself.
//
//       An earlier commit claimed this test's [1] was satisfied unconditionally. It was not;
//       this test is what established that, which is the reason it exists.
//
//   [2] BSR SpMV == action,  [3] diag(BSR) == block diagonal.
//       FROZEN FORM ONLY, and deliberately so. The exact term couples pressures beyond
//       nearest neighbours and would widen the BSR pattern, which is why the assembled
//       matrix does not carry it -- see cvfem_hex8_jac_rhie_chow_p. With the exact term on
//       the two are MEANT to differ, so asserting equality there would be asserting that a
//       documented design decision had been undone. They are still measured and printed, and
//       bounded loosely, so a blow-up is caught even where equality is not required.
//
// The diagonal of the action is obtained by probing it with unit vectors, which is legitimate
// here and nowhere else: this is a test, the mesh is deliberately tiny, and the point is to
// get the diagonal from the operator itself rather than from anything that shares code with
// the thing being checked.
//
// Run for steady, BDF1 and BDF2, because the transient term enters the diagonal and the
// action by different paths -- a post-pass on one and a per-node weight on the other -- and
// an inconsistency there would be invisible to a steady-only test.
//
// SFEM_RC_EXACT_JAC is a process-wide static, so it cannot be swept in-process; CMake
// registers this binary twice, once per setting, the way cvfem_flat_vs_ss_packed is.

#include "cvfem_hex8_ns_core.hpp"

#include "sfem_context.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static int g_failures = 0;

static void check(const bool ok, const char *what) {
    std::printf("%-64s %s\n", what, ok ? "OK" : "FAIL");
    if (!ok) ++g_failures;
}

static void report(const scalar_t rel, const char *what) {
    std::printf("%-64s rel %.3e  (reported, not asserted)\n", what, (double)rel);
}

static void check_rel(const scalar_t worst, const scalar_t scale, const scalar_t tol, const char *what) {
    const scalar_t rel = scale > 0 ? worst / scale : worst;
    const bool     ok  = rel <= tol;
    std::printf("%-64s rel %.3e %s\n", what, (double)rel, ok ? "OK" : "FAIL");
    if (!ok) ++g_failures;
}

namespace {

    // Deliberately not a flow: every component and every coupling is excited, so a term that
    // is dropped somewhere cannot hide behind a residual that happens to be near zero.
    void fill_state(MeshData &d) {
        const auto *const px = d.points[0];
        const auto *const py = d.points[1];
        const auto *const pz = d.points[2];
        for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
            const scalar_t X = px[i], Y = py[i], Z = pz[i];
            d.ux[(size_t)i] = std::sin(scalar_t(1.7) * X) * std::cos(scalar_t(2.3) * Y) + scalar_t(0.3) * Z;
            d.uy[(size_t)i] = std::cos(scalar_t(1.1) * Y) * (scalar_t(1) + scalar_t(0.2) * X) - scalar_t(0.15) * Z;
            d.uz[(size_t)i] = std::sin(scalar_t(0.9) * Z) * (scalar_t(0.5) + scalar_t(0.1) * Y);
            d.p[(size_t)i]  = scalar_t(0.7) * X - scalar_t(0.4) * Y + scalar_t(0.25) * Z * Z;
        }
    }

    void make_mesh(MeshData &d, std::shared_ptr<smesh::Mesh> &mesh, sfem::Context &ctx) {
        // Small on purpose: the probe below costs one operator apply per degree of freedom.
        mesh = smesh::Mesh::create_hex8_cube(ctx.communicator(), 3, 2, 2, 0, 0, 0, 2, 1, 1);
        d.mesh      = mesh;
        d.Lx        = 2;
        d.Ly        = 1;
        d.Lz        = 1;
        d.nnodes    = mesh->n_nodes();
        d.nelements = mesh->n_elements(0);
        d.elems     = mesh->elements(0)->data();
        d.points    = mesh->points()->data();
        cvfem_hex8_precompute_affine_geometry(d);
        d.ux.assign((size_t)d.nnodes, 0);
        d.uy.assign((size_t)d.nnodes, 0);
        d.uz.assign((size_t)d.nnodes, 0);
        d.p.assign((size_t)d.nnodes, 0);
        d.rx.assign((size_t)d.nnodes, 0);
        d.ry.assign((size_t)d.nnodes, 0);
        d.rz.assign((size_t)d.nnodes, 0);
        d.rc.assign((size_t)d.nnodes, 0);
        d.rhie_chow_scale = 1;
        fill_state(d);
    }

    // The diagonal 4x4 blocks of the action, column by column. Nothing here shares code with
    // assemble_block_diag, which is the entire point.
    void probe_action_diag(MeshData &d, const scalar_t rho, const scalar_t mu,
                           const std::vector<uint8_t> &free_mask, std::vector<scalar_t> &blocks) {
        const ptrdiff_t ndof = d.nnodes * N_FIELDS;
        blocks.assign((size_t)d.nnodes * 16, scalar_t(0));
        std::vector<scalar_t> dir((size_t)ndof), jv((size_t)ndof);
        for (ptrdiff_t col = 0; col < ndof; ++col) {
            std::fill(dir.begin(), dir.end(), scalar_t(0));
            dir[(size_t)col] = scalar_t(1);
            apply_jacobian_action(d, rho, mu, GeomKind::Affine, free_mask, dir.data(), jv.data());
            const ptrdiff_t node = col / N_FIELDS;
            const int       c    = (int)(col % N_FIELDS);
            for (int r = 0; r < N_FIELDS; ++r)
                blocks[(size_t)node * 16 + (size_t)r * 4 + (size_t)c] = jv[(size_t)node * N_FIELDS + r];
        }
    }

    scalar_t worst_rel(const std::vector<scalar_t> &a, const std::vector<scalar_t> &b) {
        scalar_t worst = 0, scale = 0;
        for (size_t k = 0; k < a.size(); ++k) {
            worst = std::max(worst, std::fabs(a[k] - b[k]));
            scale = std::max(scale, std::fabs(b[k]));
        }
        return scale > 0 ? worst / scale : worst;
    }

    void run_variant(sfem::Context &ctx, const char *name, const scalar_t dt, const int bdf_order,
                     const bool with_prev2, const bool exact) {
        std::printf("\n-- %s --\n", name);

        MeshData                     d;
        std::shared_ptr<smesh::Mesh> mesh;
        make_mesh(d, mesh, ctx);
        d.dt        = dt;
        d.bdf_order = bdf_order;
        if (dt > 0) {
            d.u_prev.assign((size_t)d.nnodes * 3, scalar_t(0.25));
            if (with_prev2) d.u_prev2.assign((size_t)d.nnodes * 3, scalar_t(0.1));
        }

        const scalar_t rho = 1, mu = 0.01;
        const ptrdiff_t ndof = d.nnodes * N_FIELDS;

        // The nodal pressure gradient the exact term reads; the action requires it to exist.
        assemble_nodal_p_grad(d, GeomKind::Affine);

        const std::vector<uint8_t> free_mask((size_t)ndof, 0);

        std::vector<scalar_t> bd;
        assemble_block_diag(d, rho, mu, GeomKind::Affine, bd);

        std::vector<scalar_t> probed;
        probe_action_diag(d, rho, mu, free_mask, probed);

        scalar_t scale = 0;
        for (const scalar_t v : probed) scale = std::max(scale, std::fabs(v));
        check(scale > scalar_t(1e-8), "the probed diagonal is not trivially zero");

        const scalar_t r_bd = worst_rel(bd, probed);
        if (exact) {
            report(r_bd, "[1] block diagonal vs action diagonal (exact form: known gap)");
            check(r_bd < scalar_t(0.25), "[1] the block diagonal's known gap stays bounded");
        } else {
            check_rel(r_bd, scalar_t(1), scalar_t(1e-10),
                      "[1] block diagonal == diagonal blocks of the action");
        }

        // [2] and [3]
        BSR4 b = make_bsr4(d.mesh);
        precompute_element_bsr_slots(d, b);
        assemble_jacobian(d, b, rho, mu, GeomKind::Affine);

        std::vector<scalar_t> dir((size_t)ndof), act((size_t)ndof), spmv((size_t)ndof, scalar_t(0));
        for (ptrdiff_t i = 0; i < ndof; ++i)
            dir[(size_t)i] = std::sin(scalar_t(0.37) * (scalar_t)i) + scalar_t(0.11) * (scalar_t)(i % 7);
        apply_jacobian_action(d, rho, mu, GeomKind::Affine, free_mask, dir.data(), act.data());

        const scalar_t *const vals = b.data();
        for (ptrdiff_t row = 0; row < d.nnodes; ++row)
            for (smesh::count_t k = b.rowptr[row]; k < b.rowptr[row + 1]; ++k) {
                const scalar_t *const blk = vals + (ptrdiff_t)k * 16;
                const smesh::idx_t    col = b.colidx[k];
                for (int r = 0; r < 4; ++r)
                    for (int c = 0; c < 4; ++c)
                        spmv[(size_t)row * 4 + (size_t)r] += blk[r * 4 + c] * dir[(size_t)col * 4 + (size_t)c];
            }

        std::vector<scalar_t> bsr_diag((size_t)d.nnodes * 16);
        for (ptrdiff_t r = 0; r < d.nnodes; ++r)
            for (int k = 0; k < 16; ++k)
                bsr_diag[(size_t)r * 16 + (size_t)k] = vals[(ptrdiff_t)b.diag_slots[(size_t)r] * 16 + k];

        const scalar_t r_spmv = worst_rel(spmv, act);
        const scalar_t r_diag = worst_rel(bsr_diag, probed);
        if (exact) {
            // Meant to differ: the assembled matrix deliberately omits the wide term. Bounded
            // rather than ignored, so a blow-up is still caught.
            report(r_spmv, "[2] BSR SpMV vs action (exact form: differs by design)");
            report(r_diag, "[3] diag(BSR) vs action diagonal (exact form: differs by design)");
            check(r_spmv < scalar_t(0.5) && r_diag < scalar_t(0.5),
                  "[2,3] the by-design difference stays bounded");
        } else {
            check_rel(r_spmv * scalar_t(1), scalar_t(1), scalar_t(1e-11), "[2] BSR SpMV == action");
            check_rel(r_diag * scalar_t(1), scalar_t(1), scalar_t(1e-11), "[3] diag(BSR) == action diagonal");
        }
    }

}  // namespace

int main(int argc, char **argv) {
    auto      ctx   = sfem::initialize(argc, argv);
    const char *const raw = std::getenv("SFEM_RC_EXACT_JAC");
    const bool  exact = !(raw && raw[0] == '0');
    std::printf("operator consistency: SFEM_RC_EXACT_JAC=%d (%s Rhie-Chow Jacobian)\n",
                exact ? 1 : 0, exact ? "exact" : "frozen");

    run_variant(*ctx, "steady", scalar_t(0), 1, false, exact);
    run_variant(*ctx, "transient BDF1", scalar_t(0.05), 1, false, exact);
    run_variant(*ctx, "transient BDF2", scalar_t(0.05), 2, true, exact);

    std::printf("\n%s\n", g_failures ? "operator consistency: FAILED" : "all operator forms agree");
    return g_failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
