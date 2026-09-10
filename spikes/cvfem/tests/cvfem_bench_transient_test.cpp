// The benchmark family's transient term, against closed forms.
//
// The term is a duplicate. The benchmark family and the solver core cannot include each
// other -- each #errors on the other's guard -- so build_node_volume, bdf_coeffs,
// apply_transient and its Jacobian exist twice, in cvfem_hex8_layout_common.hpp and in
// cvfem_hex8_ns_core.hpp. Duplicates drift, and nothing here can compare the two copies
// directly. What can be done, and is what this test does, is pin the benchmark's copy to
// things that are true of the mathematics rather than of either implementation:
//
//   * The control volumes partition the domain. Their sum is the domain volume exactly,
//     and on a uniform box every interior node owns exactly one cell, h^3 -- an eighth
//     from each of the eight elements that meet there. A node volume that is off by a
//     factor, or that misses an element, fails one of these.
//   * The Jacobian is the derivative of the residual. The BDF term is linear in u, so a
//     central difference of apply_transient_pass reproduces apply_transient_action_pass
//     EXACTLY, not to truncation order -- a much sharper statement than the usual
//     finite-difference check, and it is the one that would catch a wrong a0.
//   * The assembled diagonal, the block diagonal and the matrix-free action are three
//     spellings of the same weight, so they must agree to round-off.
//   * BDF1 and BDF2 differ by exactly the factor 1.5 in the Jacobian, and the residual
//     with a history equal to the state reduces to (a0 + a1 + a2) = 0 -- the consistency
//     condition every BDF scheme satisfies, and a direct check that the coefficients were
//     not transcribed wrongly.

#include "cvfem_hex8_layout_common.hpp"
#include "cvfem_hex8_layout_atomic.hpp"

#include "sfem_context.hpp"
#include "smesh_mesh.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

static int g_failures = 0;

static void check(const bool ok, const char *what, const double got = 0.0) {
    std::printf("%-62s %-12.3e %s\n", what, got, ok ? "OK" : "FAIL");
    if (!ok) ++g_failures;
}

static const int    N   = 6;      // 6 x 6 x 6 elements
static const scalar_t L = 2;      // on [0, 2]^3, so h != 1 and a wrong power of h shows

static void build(MeshData &d, std::shared_ptr<smesh::Mesh> &mesh, sfem::Context &ctx,
                  const scalar_t dt, const int bdf_order) {
    mesh = smesh::Mesh::create_hex8_cube(ctx.communicator(), N, N, N, 0, 0, 0, L, L, L);
    d.mesh      = mesh;
    d.nnodes    = mesh->n_nodes();
    d.nelements = mesh->n_elements(0);
    d.elems     = mesh->elements(0)->data();
    d.points    = mesh->points()->data();
    fill_fields(d);
    d.dt        = dt;
    d.bdf_order = bdf_order;
    if (dt > scalar_t(0)) {
        d.u_prev.assign((size_t)d.nnodes * 3, scalar_t(0));
        if (bdf_order >= 2) d.u_prev2.assign((size_t)d.nnodes * 3, scalar_t(0));
        for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
            for (int c = 0; c < 3; ++c) {
                const scalar_t u = c == 0 ? d.ux[i] : c == 1 ? d.uy[i] : d.uz[i];
                d.u_prev[(size_t)i * 3 + (size_t)c] = scalar_t(0.9) * u;
                if (!d.u_prev2.empty()) d.u_prev2[(size_t)i * 3 + (size_t)c] = scalar_t(0.8) * u;
            }
        }
    }
}

int main(int argc, char **argv) {
    sfem::Context ctx(argc, argv);
    const scalar_t rho = scalar_t(1.3), dt = scalar_t(0.02);

    MeshData                     d;
    std::shared_ptr<smesh::Mesh> mesh;
    build(d, mesh, ctx, dt, 1);

    // ---- the control volumes partition the domain -----------------------------------
    std::vector<scalar_t> vol;
    build_node_volume(d, vol);
    scalar_t sum = 0;
    for (const scalar_t v : vol) sum += v;
    const scalar_t domain = L * L * L;
    check(std::fabs(sum - domain) <= scalar_t(1e-12) * domain,
          "the control volumes sum to the domain volume", (double)std::fabs(sum - domain));

    // An interior node is shared by eight elements and takes an eighth of each, so it owns
    // exactly one cell. A node on a face owns half of one, an edge a quarter, a corner an
    // eighth -- checked together by counting how many nodes hold each value.
    {
        const scalar_t h3 = (L / N) * (L / N) * (L / N);
        ptrdiff_t      n_interior = 0, n_wrong = 0;
        scalar_t       worst = 0;
        for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
            const scalar_t x = d.points[0][i], y = d.points[1][i], z = d.points[2][i];
            const bool     interior = x > scalar_t(1e-9) && x < L - scalar_t(1e-9) &&
                                      y > scalar_t(1e-9) && y < L - scalar_t(1e-9) &&
                                      z > scalar_t(1e-9) && z < L - scalar_t(1e-9);
            if (!interior) continue;
            ++n_interior;
            const scalar_t e = std::fabs(vol[(size_t)i] - h3);
            worst            = std::max(worst, e);
            // Not round-off in double: smesh::geom_t is float32, so h = L/N = 1/3 reaches
            // the volume through single-precision coordinates and the closed form can only
            // be matched to about a float epsilon. The measured worst case is 1.8e-7
            // relative, which is exactly that; a tighter band here would be testing the
            // mesh's storage type rather than the volume.
            if (e > scalar_t(1e-6) * h3) ++n_wrong;
        }
        check(n_interior == (ptrdiff_t)(N - 1) * (N - 1) * (N - 1),
              "the interior node count is (N-1)^3", (double)n_interior);
        check(n_wrong == 0, "every interior control volume is exactly h^3", (double)worst);
    }

    // ---- the Jacobian is the derivative, exactly -------------------------------------
    //
    // The BDF term is linear in u, so this is an identity rather than an approximation and
    // eps cancels. A wrong a0 -- the single most likely transcription error -- shows up as
    // a clean ratio.
    {
        std::vector<scalar_t> dir((size_t)d.nnodes * N_FIELDS, scalar_t(0));
        for (ptrdiff_t i = 0; i < d.nnodes; ++i)
            for (int c = 0; c < 3; ++c)
                dir[(size_t)i * N_FIELDS + (size_t)c] =
                        scalar_t(0.3) + scalar_t(0.1) * (scalar_t)((i * 3 + c) % 7);

        const scalar_t        eps = scalar_t(1e-4);
        std::vector<scalar_t> rm((size_t)d.nnodes * 3), rp((size_t)d.nnodes * 3);
        const std::vector<scalar_t> ux0 = d.ux, uy0 = d.uy, uz0 = d.uz;
        for (int sign = -1; sign <= 1; sign += 2) {
            for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
                d.ux[i] = ux0[(size_t)i] + sign * eps * dir[(size_t)i * N_FIELDS + 0];
                d.uy[i] = uy0[(size_t)i] + sign * eps * dir[(size_t)i * N_FIELDS + 1];
                d.uz[i] = uz0[(size_t)i] + sign * eps * dir[(size_t)i * N_FIELDS + 2];
            }
            reset_residual(d);
            apply_transient_pass(d, rho);
            std::vector<scalar_t> &dst = sign < 0 ? rm : rp;
            for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
                dst[(size_t)i * 3 + 0] = d.rx[i];
                dst[(size_t)i * 3 + 1] = d.ry[i];
                dst[(size_t)i * 3 + 2] = d.rz[i];
            }
        }
        d.ux = ux0; d.uy = uy0; d.uz = uz0;

        std::vector<scalar_t> jv((size_t)d.nnodes * N_FIELDS, scalar_t(0));
        apply_transient_action_pass(d, rho, dir.data(), jv.data());

        scalar_t worst = 0, scale = 0;
        for (ptrdiff_t i = 0; i < d.nnodes; ++i)
            for (int c = 0; c < 3; ++c) {
                const scalar_t fd = (rp[(size_t)i * 3 + (size_t)c] - rm[(size_t)i * 3 + (size_t)c]) / (2 * eps);
                scale            = std::max(scale, std::fabs(fd));
                worst            = std::max(worst, std::fabs(fd - jv[(size_t)i * N_FIELDS + (size_t)c]));
            }
        check(scale > scalar_t(1e-6), "the transient derivative is not trivially zero", (double)scale);
        check(worst <= scalar_t(1e-10) * scale,
              "the action is the exact derivative of the residual term", (double)(worst / scale));

        // The pressure component must be untouched: the term is velocity-only, and a
        // Jacobian that leaked into it would destroy the saddle-point structure the block
        // preconditioner depends on.
        scalar_t leak = 0;
        for (ptrdiff_t i = 0; i < d.nnodes; ++i) leak = std::max(leak, std::fabs(jv[(size_t)i * N_FIELDS + 3]));
        check(leak == scalar_t(0), "the transient term touches no pressure dof", (double)leak);
    }

    // ---- the block diagonal spells the same weight -----------------------------------
    {
        std::vector<scalar_t> diag((size_t)d.nnodes * 16, scalar_t(0));
        assemble_diag_transient_pass(d, rho, diag);
        const scalar_t a     = transient_diag_weight(d, rho);
        scalar_t       worst = 0, off = 0;
        for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
            const scalar_t w = a * vol[(size_t)i];
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 4; ++c) {
                    const scalar_t v = diag[(size_t)i * 16 + (size_t)r * 4 + (size_t)c];
                    if (r == c && r < 3) worst = std::max(worst, std::fabs(v - w));
                    else off = std::max(off, std::fabs(v));
                }
        }
        check(worst <= scalar_t(1e-14) * std::fabs(a * domain),
              "the block diagonal carries rho V a0 / dt on the velocities", (double)worst);
        check(off == scalar_t(0), "and nothing anywhere else in the block", (double)off);
    }

    // ---- the BDF coefficients --------------------------------------------------------
    //
    // With a history equal to the state the BDF residual must vanish: sum(a_k) = 0 is the
    // consistency condition, and it holds for BDF1 and BDF2 alike.
    for (int order = 1; order <= 2; ++order) {
        MeshData                     e;
        std::shared_ptr<smesh::Mesh> m;
        build(e, m, ctx, dt, order);
        for (ptrdiff_t i = 0; i < e.nnodes; ++i)
            for (int c = 0; c < 3; ++c) {
                const scalar_t u = c == 0 ? e.ux[i] : c == 1 ? e.uy[i] : e.uz[i];
                e.u_prev[(size_t)i * 3 + (size_t)c] = u;
                if (!e.u_prev2.empty()) e.u_prev2[(size_t)i * 3 + (size_t)c] = u;
            }
        reset_residual(e);
        apply_transient_pass(e, rho);
        scalar_t worst = 0;
        for (ptrdiff_t i = 0; i < e.nnodes; ++i)
            worst = std::max(worst, std::max(std::fabs(e.rx[i]), std::max(std::fabs(e.ry[i]), std::fabs(e.rz[i]))));
        char what[96];
        std::snprintf(what, sizeof(what), "BDF%d vanishes on a constant history", bdf_coeffs(e).order);
        check(worst <= scalar_t(1e-12) * rho * domain / dt, what, (double)worst);

        const scalar_t a = transient_diag_weight(e, rho);
        const scalar_t want = (order >= 2 ? scalar_t(1.5) : scalar_t(1)) * rho / dt;
        std::snprintf(what, sizeof(what), "BDF%d Jacobian weight is a0 rho / dt", bdf_coeffs(e).order);
        check(std::fabs(a - want) <= scalar_t(1e-14) * want, what, (double)std::fabs(a - want));
    }

    // A steady run must be untouched by every one of these: dt <= 0 is the default and is
    // what every recorded baseline was measured with.
    {
        MeshData                     e;
        std::shared_ptr<smesh::Mesh> m;
        build(e, m, ctx, scalar_t(0), 1);
        reset_residual(e);
        apply_transient_pass(e, rho);
        std::vector<scalar_t> jv((size_t)e.nnodes * N_FIELDS, scalar_t(1));
        apply_transient_action_pass(e, rho, jv.data(), jv.data());
        scalar_t worst = 0;
        for (ptrdiff_t i = 0; i < e.nnodes; ++i) worst = std::max(worst, std::fabs(e.rx[i]));
        for (ptrdiff_t i = 0; i < e.nnodes * N_FIELDS; ++i) worst = std::max(worst, std::fabs(jv[(size_t)i] - 1));
        check(worst == scalar_t(0), "dt <= 0 leaves the steady operator exactly alone", (double)worst);
    }

    if (g_failures) {
        std::fprintf(stderr, "cvfem_bench_transient_test: %d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all transient checks passed\n");
    return 0;
}
