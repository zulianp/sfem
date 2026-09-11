// The boundary closure must give the same answer whatever the thread count.
//
// It did not. The closure used to scatter each element's contribution into the shared node
// arrays with `atomic_add` under `#pragma omp parallel for schedule(static)`. Static
// scheduling fixes which thread owns which face, but not the order in which two threads
// holding faces that meet at a node commit their updates, and floating-point addition is
// not associative -- so the operator was not bit-reproducible with itself. Measured on the
// pressure port at 33,124 dof: five runs at one thread all took 922 linear iterations and
// printed the same residual to every digit at linear iteration 100, while five at 72
// threads took 900, 839, 1595, 943 and 742, and one earlier run diverged outright.
//
// That is not a tolerance question. A Krylov method amplifies a last-bit difference, so
// "the same binary gave a different answer" becomes indistinguishable from a regression --
// which is exactly the trap a verification A/B walked into, where one case appeared to
// break and turned out to be fragile in both binaries.
//
// Two things are asserted here. That repeated applies at 1, 2, 4 and 8 threads are BIT
// identical, which is the property the gather exists to provide; and that the answer still
// agrees with a serial scatter over the same elements to round-off, which is what says the
// operator did not change.
//
// The second is a tolerance and not a bitwise check, deliberately. The map lists a node's
// slots in element order, so a strictly sequential sum would reproduce a serial scatter
// exactly -- but these kernels are built with -ffast-math, which lets the compiler
// reassociate the gather's inner reduction into partial accumulators. That reassociation is
// chosen at compile time and is identical on every call, so it costs nothing here: what
// determinism requires is a fixed order, not a particular one.

#include "cvfem_hex8_ns_core.hpp"

#include "sfem_context.hpp"
#include "smesh_mesh.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

static int g_failures = 0;

static void check(const bool ok, const char *what) {
    std::printf("%-60s %s\n", what, ok ? "OK" : "FAIL");
    if (!ok) ++g_failures;
}

static constexpr scalar_t RHO = 1.0, MU = 0.01;
static constexpr scalar_t LX = 4, LY = 2, LZ = 1;

struct Run {
    std::shared_ptr<smesh::Mesh> mesh;
    MeshData                     d;
    PackedData                   packed;
    std::vector<scalar_t>        dir;
};

static void build(Run &r, sfem::Context &ctx, const int pack_size = 0) {
    // Big enough that the shell spans many threads' chunks -- a shell that fits in one
    // chunk cannot show a race even when there is one.
    r.mesh = smesh::Mesh::create_hex8_cube(ctx.communicator(), 24, 12, 6, 0, 0, 0, LX, LY, LZ);
    MeshData &d = r.d;
    d.mesh      = r.mesh;
    d.Lx = LX; d.Ly = LY; d.Lz = LZ;
    // Rhie-Chow on, so the full matvec below exercises the reconstruction of the
    // direction's pressure gradient as well as the element sweep and the closure.
    d.rhie_chow_scale = 1;
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
    r.dir.resize((size_t)d.nnodes * N_FIELDS);
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        const scalar_t x = px[i], y = py[i], z = pz[i];
        d.ux[(size_t)i] = 0.7 + 0.13 * x - 0.21 * y + 0.05 * z;
        d.uy[(size_t)i] = -0.3 + 0.11 * x + 0.17 * y - 0.04 * z;
        d.uz[(size_t)i] = 0.2 - 0.07 * x + 0.09 * y + 0.12 * z;
        d.p[(size_t)i]  = 1.0 + 0.05 * x + 0.03 * y;
        // Node-varying and not a low-order polynomial in the same basis as the state, so
        // cancellations cannot make a summation-order difference invisible.
        r.dir[(size_t)i * 4 + 0] = 0.31 * std::sin(3.1 * x + 1.7 * y) + 0.11 * z;
        r.dir[(size_t)i * 4 + 1] = 0.27 * std::cos(2.3 * y - 0.9 * z) - 0.07 * x;
        r.dir[(size_t)i * 4 + 2] = 0.19 * std::sin(1.3 * z + 2.1 * x) + 0.05 * y;
        r.dir[(size_t)i * 4 + 3] = 0.23 * std::cos(1.9 * x + 0.7 * z) + 0.09 * y;
    }
}

// One apply at a given thread count, into a fresh destination.
static std::vector<scalar_t> apply_at(Run &r, const int threads) {
#ifdef _OPENMP
    omp_set_num_threads(threads);
#else
    (void)threads;
#endif
    std::vector<scalar_t> jv((size_t)r.d.nnodes * N_FIELDS, scalar_t(0));
    apply_boundary_scs_jacobian_action(r.d, RHO, MU, 0, r.dir.data(), jv.data());
    return jv;
}

// The same closure, scattered serially in element order -- the order the gather map claims
// to reproduce. Deliberately written out rather than routed through the pass being tested.
static std::vector<scalar_t> serial_reference(Run &r) {
    MeshData &d = r.d;
    cvfem_hex8_build_face_mask_eff(d);
    std::vector<scalar_t> jv((size_t)d.nnodes * N_FIELDS, scalar_t(0));
    const ptrdiff_t n_bnd = (ptrdiff_t)d.bnd_elems.size();
    for (ptrdiff_t i = 0; i < n_bnd; ++i) {
        const ptrdiff_t e     = d.bnd_elems[(size_t)i];
        const int       fmask = (int)d.face_mask_eff[(size_t)e];
        scalar_t x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8], vx[8], vy[8], vz[8], q[8];
        scalar_t r_e[CVFEM_HEX8_N_DOF];
        gather_element_coords(d, e, x, y, z);
        gather_element_fields(d, e, ux, uy, uz, p);
        gather_element_dir(d, e, r.dir.data(), vx, vy, vz, q);
        std::memset(r_e, 0, sizeof(r_e));
        scalar_t adj[9], det = scalar_t(0);
        cvfem_hex8_load_adj(d, e, adj, &det);
        boundary_scs_add_jacobian_action(RHO, MU, 0, adj, det, d.Lx, d.Ly, d.Lz, x, y, z, ux, uy, uz,
                                         vx, vy, vz, q, r_e, fmask,
                                         d.natural_mask.empty() ? 0 : (int)d.natural_mask[(size_t)e], hex8_bd(d, e));
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const ptrdiff_t g = (ptrdiff_t)d.elems[a][e] * N_FIELDS;
            for (int c = 0; c < 4; ++c) jv[(size_t)g + c] += r_e[a * 4 + c];
        }
    }
    return jv;
}

// Bitwise, not to a tolerance: the claim is reproducibility, and a tolerance would pass on
// a result that still wandered.
static ptrdiff_t differing(const std::vector<scalar_t> &a, const std::vector<scalar_t> &b) {
    ptrdiff_t n = 0;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(&a[i], &b[i], sizeof(scalar_t)) != 0) ++n;
    return n;
}

int main(int argc, char **argv) {
    sfem::Context ctx(argc, argv);
    Run r;
    build(r, ctx);
    const std::vector<scalar_t> ref = apply_at(r, 1);
    // After the first apply, which is what builds the shell.
    std::printf("cube %td nodes, %td elements, %zu boundary elements\n",
                r.d.nnodes, r.d.nelements, r.d.bnd_elems.size());
    ptrdiff_t nz = 0;
    for (const scalar_t v : ref)
        if (v != scalar_t(0)) ++nz;
    // A closure that contributed nothing would be trivially reproducible.
    check(nz > 0, "the closure contributes something at all");

    char msg[128];
    for (const int t : {2, 4, 8}) {
        const std::vector<scalar_t> got = apply_at(r, t);
        const ptrdiff_t             n   = differing(ref, got);
        std::snprintf(msg, sizeof(msg), "%d threads: bit identical to 1 thread (%td differ)", t, n);
        check(n == 0, msg);
    }

    // Repeat at the widest count: a race that lands the same way twice by luck would
    // otherwise be reported as determinism.
    for (int rep = 0; rep < 4; ++rep) {
        const std::vector<scalar_t> got = apply_at(r, 8);
        std::snprintf(msg, sizeof(msg), "8 threads, repeat %d: bit identical", rep + 1);
        check(differing(ref, got) == 0, msg);
    }

    const std::vector<scalar_t> ser = serial_reference(r);
    scalar_t worst = 0, scale = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        worst = std::fmax(worst, std::fabs(ref[i] - ser[i]));
        scale = std::fmax(scale, std::fabs(ser[i]));
    }
    const scalar_t rel = scale > 0 ? worst / scale : worst;
    std::snprintf(msg, sizeof(msg), "agrees with a serial scatter (rel %.3e, %td bits differ)", rel,
                  differing(ref, ser));
    check(rel < 1e-13, msg);

    // ---- the whole matvec, not just the closure ----
    //
    // The closure was the pass that had the race, but what the solver applies is the
    // element sweep, the direction's gradient reconstruction, the closure and the transient
    // term together. Asserting the composite is what makes "the operator is deterministic"
    // a statement about the operator rather than about one of its passes.
    Run pk;
    // A pack size well below the element count, so the sweep really does span many packs
    // and close their ghosts -- a single pack has no ghosts and could not show a race.
    build(pk, ctx, 256);
    assemble_nodal_p_grad(pk.d, GeomKind::Affine);
    auto matvec_at = [&pk](const int threads) {
#ifdef _OPENMP
        omp_set_num_threads(threads);
#else
        (void)threads;
#endif
        std::vector<scalar_t> jv((size_t)pk.d.nnodes * N_FIELDS, scalar_t(0));
        apply_jacobian_action_accumulate(pk.d, RHO, MU, GeomKind::Affine, pk.dir.data(), jv.data());
        return jv;
    };
    const std::vector<scalar_t> mref = matvec_at(1);
    ptrdiff_t                   mnz  = 0;
    for (const scalar_t v : mref)
        if (v != scalar_t(0)) ++mnz;
    check(mnz > 0, "full matvec: contributes something at all");
    for (const int t : {2, 4, 8}) {
        const ptrdiff_t n = differing(mref, matvec_at(t));
        std::snprintf(msg, sizeof(msg), "full matvec, %d threads: bit identical (%td differ)", t, n);
        check(n == 0, msg);
    }
    for (int rep = 0; rep < 3; ++rep) {
        std::snprintf(msg, sizeof(msg), "full matvec, 8 threads, repeat %d: bit identical", rep + 1);
        check(differing(mref, matvec_at(8)) == 0, msg);
    }

    std::printf("\n%s\n", g_failures ? "FAILED" : "PASSED");
    return g_failures ? 1 : 0;
}
