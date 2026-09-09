// The BDF transient term.
//
// Four things are checked, and the first is the one that protects everything already in
// the repository: with dt <= 0 the term is not merely small, it is not evaluated, and the
// residual is bit-identical to what it was before the term existed. Every recorded number
// in docs/ was measured on the steady operator and must stay comparable.
//
// The rest is order of accuracy and consistency. In a control-volume scheme the mass
// matrix is the control volume -- diagonal, and already built for the body force -- so the
// term is rho * V_i * (a0 u^{n+1} + a1 u^n + a2 u^{n-1}) / dt on the three velocity
// components and nothing on pressure. That makes it cheap to test exactly:
//
//   * BDF1 differentiates a linear-in-time field exactly, BDF2 a quadratic one, and
//     neither should be exact beyond its order. Checking only that the error is "small"
//     would pass a scheme of the wrong order entirely, so both the exactness and the
//     inexactness are asserted.
//   * The observed order over a dt sequence is 1 and 2 respectively.
//   * The Jacobian's contribution is d/du of the residual's, checked against a finite
//     difference rather than against the formula it was derived from.
//   * Pressure has no time derivative and must not acquire one.

#include "cvfem_hex8_ns_core.hpp"

#include "sfem_context.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

static int g_failures = 0;

static void check(const bool ok, const char *what) {
    std::printf("%-58s %s\n", what, ok ? "OK" : "FAIL");
    if (!ok) ++g_failures;
}

static void check_close(const scalar_t got, const scalar_t want, const scalar_t tol, const char *what) {
    const bool ok = std::fabs(got - want) <= tol;
    std::printf("%-58s got %+.10e want %+.10e %s\n", what, (double)got, (double)want, ok ? "OK" : "FAIL");
    if (!ok) ++g_failures;
}

// A small box, set up the way the gate does.
static void make_mesh(MeshData &d, std::shared_ptr<smesh::Mesh> &mesh, sfem::Context &ctx) {
    mesh = smesh::Mesh::create_hex8_cube(ctx.communicator(), 4, 3, 3, 0, 0, 0, 2, 1, 1);
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
}

// u(t) evaluated on every node, for a field that is a polynomial in time with a
// space-dependent coefficient -- so the answer is not uniform and a per-node error shows.
static void set_state_at(MeshData &d, const scalar_t t, const int power, std::vector<scalar_t> *into) {
    const auto *const px = d.points[0];
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        const scalar_t s  = 0.5 + 0.25 * (scalar_t)px[i];
        const scalar_t v  = power == 1 ? s * t : power == 2 ? s * t * t : s * t * t * t;
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

// Worst |transient residual / (rho V) - du/dt| over the nodes: the term divided by its own
// weight is an approximation of du/dt, so this is the truncation error of the scheme.
static scalar_t bdf_error(MeshData &d, const scalar_t rho, const scalar_t dt, const int order,
                          const int power, const scalar_t t_new) {
    d.dt        = dt;
    d.bdf_order = order;
    d.u_prev.assign((size_t)d.nnodes * 3, 0);
    if (order >= 2) d.u_prev2.assign((size_t)d.nnodes * 3, 0);
    else d.u_prev2.clear();

    set_state_at(d, t_new - dt, power, &d.u_prev);
    if (order >= 2) set_state_at(d, t_new - 2 * dt, power, &d.u_prev2);
    set_state_at(d, t_new, power, nullptr);

    std::fill(d.rx.begin(), d.rx.end(), scalar_t(0));
    std::fill(d.ry.begin(), d.ry.end(), scalar_t(0));
    std::fill(d.rz.begin(), d.rz.end(), scalar_t(0));
    apply_transient(d, rho);

    if ((ptrdiff_t)d.node_vol.size() != d.nnodes) build_node_volume(d, d.node_vol);
    const auto *const px    = d.points[0];
    scalar_t          worst = 0;
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        const scalar_t w = rho * d.node_vol[(size_t)i];
        if (w == scalar_t(0)) continue;
        const scalar_t s     = 0.5 + 0.25 * (scalar_t)px[i];
        // d/dt of s t, s t^2, s t^3
        const scalar_t exact = power == 1 ? s : power == 2 ? 2 * s * t_new : 3 * s * t_new * t_new;
        worst                = std::max(worst, std::fabs(d.rx[(size_t)i] / w - exact));
    }
    return worst;
}

int main(int argc, char **argv) {
    auto ctx = sfem::initialize(argc, argv);
    MeshData                     d;
    std::shared_ptr<smesh::Mesh> mesh;
    make_mesh(d, mesh, *ctx);

    const scalar_t rho = 1.25, mu = 0.01;

    // 1. Steady is untouched.
    //
    // Note what is NOT asserted here. Comparing two full apply_residual calls for
    // bit-equality does not work: the atomic layout accumulates through #pragma omp atomic,
    // so its result is not reproducible bit-for-bit between runs under threads at all --
    // an earlier version of this test failed for that reason and passed single-threaded.
    // That would have been a test of determinism, not of this term. So the exact assertion
    // is made where it is meaningful -- on the transient pass itself, which is a plain
    // per-node loop with no atomics -- and the full residual is compared to round-off.
    {
        for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
            d.ux[(size_t)i] = 0.3 + 0.1 * (scalar_t)d.points[0][i];
            d.uy[(size_t)i] = -0.2 + 0.05 * (scalar_t)d.points[1][i];
            d.uz[(size_t)i] = 0.1;
            d.p[(size_t)i]  = 0.7 * (scalar_t)d.points[0][i];
        }
        // History deliberately present and non-zero, so a term that ignored dt would fire.
        d.dt = 0;
        d.u_prev.assign((size_t)d.nnodes * 3, 1.0);
        d.u_prev2.assign((size_t)d.nnodes * 3, 2.0);

        std::fill(d.rx.begin(), d.rx.end(), scalar_t(0));
        std::fill(d.ry.begin(), d.ry.end(), scalar_t(0));
        std::fill(d.rz.begin(), d.rz.end(), scalar_t(0));
        apply_transient(d, rho);
        scalar_t touched = 0;
        for (ptrdiff_t i = 0; i < d.nnodes; ++i)
            touched = std::max(touched, std::fabs(d.rx[(size_t)i]) + std::fabs(d.ry[(size_t)i]) +
                                                std::fabs(d.rz[(size_t)i]));
        check(touched == scalar_t(0), "dt = 0: the transient pass writes nothing at all");
        check(transient_diag_weight(d, rho) == scalar_t(0), "dt = 0: no Jacobian contribution");

        // And the whole residual is unchanged by the presence of history, to round-off.
        apply_residual(d, rho, mu, GeomKind::Affine);
        std::vector<scalar_t> with_history = d.rx;
        d.u_prev.clear();
        d.u_prev2.clear();
        apply_residual(d, rho, mu, GeomKind::Affine);
        scalar_t worst = 0;
        for (ptrdiff_t i = 0; i < d.nnodes; ++i)
            worst = std::max(worst, std::fabs(d.rx[(size_t)i] - with_history[(size_t)i]));
        check(worst < 1e-14, "dt = 0: the full residual is unchanged by stored history");
    }

    // 2. BDF1 is exact for a field linear in time; BDF2 for a quadratic one. And BDF1 is
    //    NOT exact for the quadratic -- without that half, a first-order scheme mislabelled
    //    second-order would pass.
    check_close(bdf_error(d, rho, 0.1, 1, 1, 0.7), 0.0, 1e-12, "BDF1 is exact for u linear in t");
    check_close(bdf_error(d, rho, 0.1, 2, 2, 0.7), 0.0, 1e-12, "BDF2 is exact for u quadratic in t");
    check(bdf_error(d, rho, 0.1, 1, 2, 0.7) > 1e-3, "BDF1 is not exact for u quadratic in t");

    // 3. Observed order over a dt sequence. Each scheme is measured on a field one degree
    //    beyond what it integrates exactly, so there is a truncation error to see at all:
    //    BDF1 on a quadratic, BDF2 on a cubic. Halving dt must divide the error by 2 and
    //    by 4 respectively.
    {
        struct Case { int order, power; const char *name; double lo, hi; };
        const Case cases[] = {{1, 2, "BDF1 observed order (on u ~ t^2)", 0.9, 1.1},
                              {2, 3, "BDF2 observed order (on u ~ t^3)", 1.9, 2.1}};
        for (const Case &c : cases) {
            const scalar_t e_coarse = bdf_error(d, rho, 0.1, c.order, c.power, 0.7);
            const scalar_t e_fine   = bdf_error(d, rho, 0.05, c.order, c.power, 0.7);
            const double   p        = std::log((double)(e_coarse / e_fine)) / std::log(2.0);
            std::printf("%-58s %.3f  (%.3e -> %.3e)\n", c.name, p, (double)e_coarse, (double)e_fine);
            check(p > c.lo && p < c.hi, c.order == 1 ? "BDF1 observed order is 1" : "BDF2 observed order is 2");
        }
    }

    // 4. The Jacobian's contribution is the derivative of the residual's. Checked by a
    //    finite difference on the residual rather than against the formula it came from.
    {
        d.dt        = 0.05;
        d.bdf_order = 1;
        d.u_prev.assign((size_t)d.nnodes * 3, 0.25);
        d.u_prev2.clear();
        if ((ptrdiff_t)d.node_vol.size() != d.nnodes) build_node_volume(d, d.node_vol);

        const scalar_t a   = transient_diag_weight(d, rho);
        const ptrdiff_t i0 = d.nnodes / 2;
        const scalar_t eps = 1e-6;

        auto transient_rx = [&](const scalar_t ux_i) {
            const scalar_t save = d.ux[(size_t)i0];
            d.ux[(size_t)i0]    = ux_i;
            std::fill(d.rx.begin(), d.rx.end(), scalar_t(0));
            apply_transient(d, rho);
            const scalar_t out = d.rx[(size_t)i0];
            d.ux[(size_t)i0]   = save;
            return out;
        };
        const scalar_t base = d.ux[(size_t)i0];
        const scalar_t fd   = (transient_rx(base + eps) - transient_rx(base - eps)) / (2 * eps);
        check_close(fd, a * d.node_vol[(size_t)i0], 1e-6 * std::fabs(a * d.node_vol[(size_t)i0]) + 1e-9,
                    "the Jacobian diagonal is d(residual)/du");
    }

    // 5. Pressure acquires no time derivative. The continuity equation is a constraint,
    //    not an evolution equation, and a mass term on it would be a different system.
    {
        d.dt        = 0.05;
        d.bdf_order = 1;
        d.u_prev.assign((size_t)d.nnodes * 3, 0.25);
        std::fill(d.rc.begin(), d.rc.end(), scalar_t(0));
        apply_transient(d, rho);
        scalar_t worst = 0;
        for (ptrdiff_t i = 0; i < d.nnodes; ++i) worst = std::max(worst, std::fabs(d.rc[(size_t)i]));
        check(worst == scalar_t(0), "the continuity row gets no transient term");

        std::vector<scalar_t> diag;
        assemble_block_diag(d, rho, mu, GeomKind::Affine, diag);
        std::vector<scalar_t> diag_steady;
        const scalar_t        dt_save = d.dt;
        d.dt                          = 0;
        assemble_block_diag(d, rho, mu, GeomKind::Affine, diag_steady);
        d.dt = dt_save;
        bool pressure_same = true, velocity_grew = false;
        for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
            pressure_same = pressure_same && diag[(size_t)i * 16 + 15] == diag_steady[(size_t)i * 16 + 15];
            if (diag[(size_t)i * 16 + 0] > diag_steady[(size_t)i * 16 + 0]) velocity_grew = true;
        }
        check(pressure_same, "the block diagonal's pressure entry is unchanged");
        check(velocity_grew, "the block diagonal's velocity entries grow with the time term");
    }

    if (g_failures) {
        std::fprintf(stderr, "cvfem_transient_test: %d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all transient checks passed\n");
    return 0;
}
