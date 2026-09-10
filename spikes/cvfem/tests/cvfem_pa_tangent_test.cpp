// The partially assembled Jacobian action, against the one it replaces.
//
// For a fixed Newton iterate the flux through a sub-control surface is bilinear -- state
// scalars times direction vectors -- and the partial assembly is the observation that the
// state enters through only five numbers per surface. The step it rests on is
//
//     dpos * u_I + dneg * u_J  =  dmdot * (d_pos * u_I + d_neg * u_J)  =  dmdot * uup
//
// using dpos = d_pos * dmdot, dneg = d_neg * dmdot. If that is right, storing (mpos, mneg,
// uup) per surface loses nothing; if it is wrong, or if the tangent builder ever drifts
// from the kernel whose arithmetic it mirrors, the operator is wrong rather than slow --
// and a wrong Jacobian does not fail loudly, it caps Newton at a linear rate.
//
// So the check is the operator itself: the same direction through both applies, on a warped
// mesh with Rhie-Chow and the direction's reconstructed gradient both on, which is the
// configuration where the two paths share the least code.
//
// It comes out BIT-identical, which is more than the reassociation promises and worth
// knowing why: with the hard upwind switch sgn is exactly +-1, so d_pos and d_neg are
// exactly 1 and 0 and both spellings reduce to dmdot times one nodal velocity. With the
// smoothing band on they are strictly between 0 and 1 and the two would differ by a
// rounding. The band is therefore where to look first if this check ever starts failing
// marginally, and the tolerance below is relative to the size of the answer rather than
// zero so that it does not have to be revisited when that day comes.
//
// The cache is checked too. The tangent is a function of the STATE, which no key made of
// rho, mu and the mesh can see, so it is invalidated by hand; a cache that fails to
// invalidate would serve a tangent from the previous Newton iterate, which is exactly the
// silent-wrong-operator failure this test exists to prevent.

#include "cvfem_hex8_layout_common.hpp"
#include "cvfem_hex8_layout_atomic.hpp"
#include "cvfem_hex8_layout_packed.hpp"

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

static scalar_t worst(const std::vector<scalar_t> &a, const std::vector<scalar_t> &b) {
    scalar_t m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}
static scalar_t norm(const std::vector<scalar_t> &a) {
    scalar_t m = 0;
    for (const scalar_t v : a) m = std::max(m, std::fabs(v));
    return m;
}

int main(int argc, char **argv) {
    sfem::Context ctx(argc, argv);
    const scalar_t rho = 1.0, mu = 0.013;

    auto mesh = smesh::Mesh::create_hex8_cube(ctx.communicator(), 8, 6, 6, 0, 0, 0, 2, 1.5, 1);
    // Four packs, not one: with a single pack there are no ghost rows and the scatter the
    // two applies share is only half exercised.
    PackedData packed = make_packed(mesh, 64);

    MeshData d;
    d.mesh      = mesh;
    d.nnodes    = mesh->n_nodes();
    d.nelements = mesh->n_elements(0);
    d.elems     = mesh->elements(0)->data();
    d.points    = mesh->points()->data();
    // Warped, so no cofactor is zero and a surface whose area vector went missing would
    // change the answer rather than cancel.
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        const scalar_t y = d.points[1][i], z = d.points[2][i];
        d.points[0][i] += smesh::geom_t(scalar_t(0.11) * y * (scalar_t(1.5) - y) * z);
    }
    precompute_affine_geometry(d);
    fill_fields(d);
    d.rhie_chow_scale = 1;
    cvfem_hex8_assemble_nodal_grad_packed(d, packed, 0, d.p.data(), 1, d.pgx, d.pgy, d.pgz);

    // A direction unrelated to the state, and its reconstructed pressure gradient, so the
    // exact Rhie-Chow term is live in both applies.
    std::vector<scalar_t> dir((size_t)d.nnodes * N_FIELDS);
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        const scalar_t x = d.points[0][i], y = d.points[1][i], z = d.points[2][i];
        dir[(size_t)i * N_FIELDS + 0] = scalar_t(0.31) - scalar_t(0.2) * y + scalar_t(0.4) * z;
        dir[(size_t)i * N_FIELDS + 1] = scalar_t(-0.17) + scalar_t(0.29) * x - scalar_t(0.11) * z;
        dir[(size_t)i * N_FIELDS + 2] = scalar_t(0.23) + scalar_t(0.13) * y - scalar_t(0.19) * x;
        dir[(size_t)i * N_FIELDS + 3] = scalar_t(0.7) + std::sin(scalar_t(1.7) * x) * (scalar_t(1) + y);
    }
    cvfem_hex8_assemble_nodal_grad_packed(d, packed, 0, dir.data() + 3, N_FIELDS, d.qgx, d.qgy, d.qgz);

    std::vector<scalar_t> jv_direct((size_t)d.nnodes * N_FIELDS, 0);
    std::vector<scalar_t> jv_pa((size_t)d.nnodes * N_FIELDS, 0);

    apply_jacobian_action_packed(d, packed, rho, mu, dir.data(), jv_direct.data(), GeomKind::Affine);

    cvfem_hex8_build_pa_tangent(d, rho, mu, scalar_t(0));
    apply_jacobian_action_packed_pa(d, packed, rho, mu, dir.data(), jv_pa.data());

    const scalar_t scale = norm(jv_direct);
    check(scale > scalar_t(1e-6), "the direct action is not trivially zero", (double)scale);
    check(worst(jv_direct, jv_pa) <= scalar_t(1e-13) * scale,
          "the partially assembled action agrees with the direct one", (double)(worst(jv_direct, jv_pa) / scale));

    // The store is what it claims to be, and small.
    {
        const bool sized = (ptrdiff_t)d.pa_tangent.size() == (ptrdiff_t)CVFEM_HEX8_PA_PER_ELEM * d.nelements;
        check(sized, "the store is 60 scalars per element", (double)d.pa_tangent.size());
        const double bpd = cvfem_hex8_pa_bytes_per_dof(d);
        check(bpd > 0 && bpd < 130.0, "and under 130 bytes per degree of freedom", bpd);
    }

    // Linearity in the direction. The stored tangent is the state half of a bilinear form,
    // so doubling the direction must double the result exactly -- and a store that had
    // absorbed some of the direction by mistake would fail this while passing the above.
    {
        std::vector<scalar_t> dir2 = dir, jv2((size_t)d.nnodes * N_FIELDS, 0);
        for (scalar_t &v : dir2) v *= 2;
        cvfem_hex8_assemble_nodal_grad_packed(d, packed, 0, dir2.data() + 3, N_FIELDS, d.qgx, d.qgy, d.qgz);
        apply_jacobian_action_packed_pa(d, packed, rho, mu, dir2.data(), jv2.data());
        scalar_t w = 0;
        for (size_t i = 0; i < jv2.size(); ++i) w = std::max(w, std::fabs(jv2[i] - 2 * jv_pa[i]));
        check(w <= scalar_t(1e-13) * scale, "the action is linear in the direction", (double)(w / scale));
        cvfem_hex8_assemble_nodal_grad_packed(d, packed, 0, dir.data() + 3, N_FIELDS, d.qgx, d.qgy, d.qgz);
    }

    // ---- the cache -----------------------------------------------------------------
    //
    // The tangent depends on the state, which the key cannot see. A cache that does not
    // invalidate serves the previous Newton iterate's tangent: a wrong operator, and one
    // that shows up as a lost Newton rate rather than as a failure.
    {
        const scalar_t before = d.pa_tangent[0];
        cvfem_hex8_build_pa_tangent(d, rho, mu, scalar_t(0));
        check(d.pa_tangent[0] == before, "a rebuild with the same key and state is a no-op");

        // Move the state and clear the flag, as the caller must.
        for (ptrdiff_t i = 0; i < d.nnodes; ++i) d.ux[(size_t)i] += scalar_t(0.37);
        d.pa_valid = false;
        cvfem_hex8_build_pa_tangent(d, rho, mu, scalar_t(0));
        check(d.pa_tangent[0] != before, "clearing pa_valid rebuilds it");

        // And the parameter key still works on its own: mu moves under Reynolds
        // continuation, and the Rhie-Chow coefficient inside the tangent moves with it.
        const scalar_t after_state = d.pa_tangent[0];
        cvfem_hex8_build_pa_tangent(d, rho, mu * scalar_t(0.5), scalar_t(0));
        check(d.pa_tangent[0] != after_state, "changing mu rebuilds it");
    }

    if (g_failures) {
        std::fprintf(stderr, "cvfem_pa_tangent_test: %d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all partial-assembly checks passed\n");
    return 0;
}
