// The BDF transient term on the semi-structured operator.
//
// tests/cvfem_transient_test.cpp asserts all of this for the flat path and none of it here,
// and the difference was not academic: sscvfem_apply, the Jacobian action a Krylov solver
// applies, was missing the term entirely. Every steady result was correct, so nothing
// noticed until a transient pump stalled.
//
// The pump's swept-volume identity cannot cover this either. It is a statement about mass
// flux with a Dirichlet diaphragm and continuity enforced at each step, so it holds whatever
// the time discretisation's coefficients are -- a BDF2 with the wrong a0 would pass it. The
// order of accuracy has to be measured directly, which is what this does.
//
// Mirrors the flat test's structure deliberately, so the two can be read against each other
// and a divergence between the paths shows up as a divergence between the files.

#include "cvfem_sshex8_ns.hpp"

#include "sfem_context.hpp"
#include "smesh_mesh.hpp"
#include "smesh_semistructured.hpp"

#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

static int g_failures = 0;

static void check(const bool ok, const char *what) {
    std::printf("%-58s %s\n", what, ok ? "OK" : "FAIL");
    if (!ok) ++g_failures;
}

static void check_close(const scalar_t got, const scalar_t want, const scalar_t tol, const char *what) {
    const scalar_t err = std::fabs(got - want);
    std::printf("%-58s got %+.12e  %s\n", what, (double)got, err <= tol ? "OK" : "FAIL");
    if (!(err <= tol)) ++g_failures;
}

// u(t) on every node: a polynomial in time with a space-dependent coefficient, so the
// answer is not uniform and a per-node error shows rather than cancelling.
static void set_state_at(SSMeshData &d, const scalar_t t, const int power, std::vector<scalar_t> *into) {
    const auto *const px = d.points[0];
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        const scalar_t s = 0.5 + 0.25 * (scalar_t)px[i];
        const scalar_t v = power == 1 ? s * t : power == 2 ? s * t * t : s * t * t * t;
        if (into) {
            (*into)[(size_t)i * 3 + 0] = v;
            (*into)[(size_t)i * 3 + 1] = 2 * v;
            (*into)[(size_t)i * 3 + 2] = -v;
        } else {
            d.ux[(size_t)i] = v;
            d.uy[(size_t)i] = 2 * v;
            d.uz[(size_t)i] = -v;
        }
    }
}

// Worst |transient residual / (rho V) - du/dt| over the nodes. The term divided by its own
// weight approximates du/dt, so this is the truncation error of the scheme itself.
static scalar_t bdf_error(SSMeshData &d, const scalar_t rho, const scalar_t dt, const int order,
                          const int power, const scalar_t t_new, std::vector<scalar_t> &res) {
    d.dt        = dt;
    d.bdf_order = order;
    d.u_prev.assign((size_t)d.nnodes * 3, 0);
    if (order >= 2) d.u_prev2.assign((size_t)d.nnodes * 3, 0);
    else d.u_prev2.clear();

    set_state_at(d, t_new - dt, power, &d.u_prev);
    if (order >= 2) set_state_at(d, t_new - 2 * dt, power, &d.u_prev2);
    set_state_at(d, t_new, power, nullptr);

    res.assign((size_t)d.nnodes * N_FIELDS, 0);
    sscvfem_apply_transient(d, rho, res.data());

    if ((ptrdiff_t)d.node_vol.size() != d.nnodes) sscvfem_node_volume(d, d.node_vol);
    const auto *const px    = d.points[0];
    scalar_t          worst = 0;
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        const scalar_t w = rho * d.node_vol[(size_t)i];
        if (w == scalar_t(0)) continue;
        const scalar_t s     = 0.5 + 0.25 * (scalar_t)px[i];
        const scalar_t exact = power == 1 ? s : power == 2 ? 2 * s * t_new : 3 * s * t_new * t_new;
        worst = std::max(worst, std::fabs(res[(size_t)i * N_FIELDS + 0] / w - exact));
    }
    return worst;
}

int main(int argc, char **argv) {
    auto      ctx   = sfem::initialize(argc, argv);
    const int level = 2;  // a power of two; the operator refuses the others
    auto coarse = smesh::Mesh::create_hex8_cube(ctx->communicator(), 2, 2, 2, 0, 0, 0, 2, 1, 1);
    auto mesh   = smesh::to_semistructured(level, coarse, true, false);
    if (!mesh) {
        std::fprintf(stderr, "to_semistructured failed\n");
        return 1;
    }
    SSMeshData d;
    sscvfem_init(d, mesh, level);

    const scalar_t      rho = 1.25;
    std::vector<scalar_t> res;

    // 1. dt <= 0 is not "small", it is not evaluated. Every steady number recorded in docs/
    //    was measured on this operator and has to stay comparable.
    {
        d.dt        = 0;
        d.bdf_order = 2;
        d.u_prev.assign((size_t)d.nnodes * 3, 1);
        d.u_prev2.assign((size_t)d.nnodes * 3, 1);
        set_state_at(d, 1.0, 1, nullptr);
        res.assign((size_t)d.nnodes * N_FIELDS, 0);
        sscvfem_apply_transient(d, rho, res.data());
        scalar_t touched = 0;
        for (const auto v : res) touched = std::max(touched, std::fabs(v));
        check(touched == scalar_t(0), "dt = 0: the transient pass writes nothing at all");
        check(sscvfem_transient_diag_weight(d, rho) == scalar_t(0), "dt = 0: no Jacobian contribution");
    }

    // 2. Exactness, both ways round. Checking only that the error is "small" would pass a
    //    scheme of entirely the wrong order, so the inexactness is asserted too.
    check_close(bdf_error(d, rho, 0.1, 1, 1, 0.7, res), 0.0, 1e-12, "BDF1 is exact for u linear in t");
    check_close(bdf_error(d, rho, 0.1, 2, 2, 0.7, res), 0.0, 1e-12, "BDF2 is exact for u quadratic in t");
    check(bdf_error(d, rho, 0.1, 1, 2, 0.7, res) > 1e-3, "BDF1 is not exact for u quadratic in t");

    // 3. The observed order over a halved timestep.
    {
        struct Cfg { int order, power; scalar_t lo, hi; const char *name; };
        const Cfg cfgs[] = {{1, 2, 0.9, 1.1, "BDF1 observed order is 1"},
                            {2, 3, 1.9, 2.1, "BDF2 observed order is 2"}};
        for (const auto &c : cfgs) {
            const scalar_t e_coarse = bdf_error(d, rho, 0.1, c.order, c.power, 0.7, res);
            const scalar_t e_fine   = bdf_error(d, rho, 0.05, c.order, c.power, 0.7, res);
            const scalar_t p        = std::log(e_coarse / e_fine) / std::log(scalar_t(2));
            std::printf("  order %d: %.6e -> %.6e   observed %.4f\n", c.order, (double)e_coarse,
                        (double)e_fine, (double)p);
            check(p > c.lo && p < c.hi, c.name);
        }
    }

    // 4. Pressure carries no time term. A mass term on the continuity row would be a
    //    different set of equations, and it is what reusing SFEM's BDF2 operator would have
    //    produced -- it loops over every component of the block.
    {
        d.dt        = 0.1;
        d.bdf_order = 1;
        d.u_prev.assign((size_t)d.nnodes * 3, 0);
        d.u_prev2.clear();
        set_state_at(d, 1.0, 1, nullptr);
        res.assign((size_t)d.nnodes * N_FIELDS, 0);
        sscvfem_apply_transient(d, rho, res.data());
        scalar_t worst = 0;
        for (ptrdiff_t i = 0; i < d.nnodes; ++i)
            worst = std::max(worst, std::fabs(res[(size_t)i * N_FIELDS + 3]));
        check(worst == scalar_t(0), "the continuity row gets no transient term");
    }

    // 5. The Jacobian action must carry what the residual carries. This is the property that
    //    was missing -- sscvfem_apply never called it -- and it is checked here on the term
    //    itself, and in cvfem_flat_vs_ss_test against the flat operator end to end.
    {
        d.dt        = 0.1;
        d.bdf_order = 1;
        d.u_prev.assign((size_t)d.nnodes * 3, 0);
        d.u_prev2.clear();
        if ((ptrdiff_t)d.node_vol.size() != d.nnodes) sscvfem_node_volume(d, d.node_vol);
        const scalar_t a = sscvfem_transient_diag_weight(d, rho);
        check_close(a, rho / scalar_t(0.1), 1e-12, "BDF1 diagonal weight is rho a0 / dt");

        std::vector<scalar_t> dir((size_t)d.nnodes * N_FIELDS, 0), jv((size_t)d.nnodes * N_FIELDS, 0);
        for (ptrdiff_t i = 0; i < d.nnodes; ++i)
            for (int c = 0; c < N_FIELDS; ++c) dir[(size_t)i * N_FIELDS + (size_t)c] = 1;
        sscvfem_apply_transient_action(d, rho, dir.data(), jv.data());
        scalar_t worst_v = 0, worst_p = 0;
        for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
            const scalar_t want = a * d.node_vol[(size_t)i];
            for (int c = 0; c < 3; ++c)
                worst_v = std::max(worst_v, std::fabs(jv[(size_t)i * N_FIELDS + (size_t)c] - want));
            worst_p = std::max(worst_p, std::fabs(jv[(size_t)i * N_FIELDS + 3]));
        }
        check(worst_v < 1e-12, "the Jacobian action applies rho V a0 / dt to each velocity");
        check(worst_p == scalar_t(0), "and nothing to pressure");

        // 6. And it does not need a history to do it. A coarse level in a multigrid
        //    hierarchy is built by clone_onto, is applied only to a correction, and never
        //    receives one -- so a weight that demanded a history handed every coarse
        //    operator a STEADY Jacobian while the fine one carried a mass term that, for a
        //    small timestep, dominates its diagonal.
        d.u_prev.clear();
        d.u_prev2.clear();
        check_close(sscvfem_transient_diag_weight(d, rho), rho / scalar_t(0.1), 1e-12,
                    "the weight survives an empty history (the coarse-level case)");
        d.bdf_order = 2;
        check_close(sscvfem_transient_diag_weight(d, rho), scalar_t(1.5) * rho / scalar_t(0.1), 1e-12,
                    "and takes a0 from the requested order when there is none");
        d.dt = 0;
        check(sscvfem_transient_diag_weight(d, rho) == scalar_t(0),
              "but dt = 0 is still no term at all");
    }

    if (g_failures) {
        std::fprintf(stderr, "\ncvfem_ss_transient_test: %d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("\nall semi-structured transient checks passed\n");
    return 0;
}
