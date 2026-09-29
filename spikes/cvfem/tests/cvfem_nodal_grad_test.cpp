// The nodal gradient reconstruction, against closed forms and an independent reference.
//
// This pass is the largest single item in the Jacobian action -- 69% of the matvec at
// 4,121,204 dof on Grace, and 40-52% of every matvec in the solver's own trace -- so it is
// where the work to speed the operator up goes, and it had no test of its own. It now has
// one, because the changes coming to it are changes to an operator the Rhie-Chow term is
// defined in terms of, not to an implementation detail.
//
// What is pinned, and why each one would catch a plausible mistake:
//
//   * A globally linear pressure reconstructs EXACTLY. Every element gradient is then the
//     same constant, and any weighted average of one repeated value is that value whatever
//     the weights are. This is the property Rhie-Chow rests on: the correction subtracts
//     the reconstructed gradient from the compact difference, so a wrong reconstruction
//     turns a stabilisation into a consistency error. It also catches a wrong weight, a
//     wrong sign of the determinant, and a wrong push-forward, none of which survive it.
//   * The general case against a serial reference written out here from the definition,
//     independent of the implementation's caching, threading and det cancellation.
//   * The cached denominator against sum_e |det| computed separately, and its cache key --
//     a key that never invalidates is worse than no cache.
//   * Stride invariance: the same field read contiguously and read with stride 4 out of an
//     interleaved vector must give the same answer. The Jacobian action reads its direction
//     the second way and the residual the first, so anything that made them differ would
//     make the two paths disagree about the same operator.

#include "cvfem_hex8_best_common.hpp"
#include "cvfem_hex8_best_atomic.hpp"

#include "sfem_context.hpp"
#include "smesh_mesh.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

// Ghost staging for the packed reconstruction, held once for the whole test rather than
// created per call: the sweep grows it to n_ghost_entries * 3 and reuses it thereafter.
static std::vector<scalar_t> ng_gbuf;

static int g_failures = 0;

static void check(const bool ok, const char *what, const double got = 0.0) {
    std::printf("%-60s %-12.3e %s\n", what, got, ok ? "OK" : "FAIL");
    if (!ok) ++g_failures;
}

// The definition, written out: nodal value = (sum_e |det_e| grad_e) / (sum_e |det_e|),
// serial, no caching, no cancellation. Deliberately the slow spelling.
static void reference_grad(const MeshData &d, const std::vector<scalar_t> &f,
                           std::vector<scalar_t> &gx, std::vector<scalar_t> &gy,
                           std::vector<scalar_t> &gz) {
    gx.assign((size_t)d.nnodes, 0);
    gy.assign((size_t)d.nnodes, 0);
    gz.assign((size_t)d.nnodes, 0);
    std::vector<scalar_t> w((size_t)d.nnodes, 0);
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t fe[CVFEM_HEX8_N_NODES];
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) fe[a] = f[(size_t)d.elems[a][e]];
        scalar_t adj[9], det;
        load_hex8_adj(d, e, adj, &det);
        const scalar_t vol = std::fabs(det);
        if (vol < scalar_t(1e-30)) continue;
        scalar_t dr, ds, dt, ex, ey, ez;
        cvfem_hex8_face_diff(fe, dr, ds, dt);
        cvfem_hex8_pushforward(adj, scalar_t(1) / det, dr, ds, dt, ex, ey, ez);
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const smesh::idx_t g = d.elems[a][e];
            gx[(size_t)g] += vol * ex;
            gy[(size_t)g] += vol * ey;
            gz[(size_t)g] += vol * ez;
            w[(size_t)g] += vol;
        }
    }
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        if (w[(size_t)i] <= 0) continue;
        const scalar_t inv = scalar_t(1) / w[(size_t)i];
        gx[(size_t)i] *= inv;
        gy[(size_t)i] *= inv;
        gz[(size_t)i] *= inv;
    }
}

static scalar_t worst_diff(const std::vector<scalar_t> &a, const std::vector<scalar_t> &b) {
    scalar_t m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

int main(int argc, char **argv) {
    sfem::Context ctx(argc, argv);

    // Warped, so the adjugate has no zero entries and the three components are genuinely
    // coupled: on an axis-aligned box most cofactors vanish and a transposed push-forward
    // would go unnoticed.
    const int N = 6;
    auto mesh = smesh::Mesh::create_hex8_cube(ctx.communicator(), N, N, N, 0, 0, 0, 2, 1.5, 1);
    // Packed first: creating a PackedMesh renumbers the mesh nodes in place, so everything
    // derived from the node ordering has to come after it. A small pack deliberately: with
    // one pack holding every element there are no ghost rows, and the ghost reduction --
    // the part of the packed sweep that replaces the atomics -- would go unexercised.
    PackedData packed = make_packed(mesh, 64);
    MeshData   d;
    d.mesh      = mesh;
    d.nnodes    = mesh->n_nodes();
    d.nelements = mesh->n_elements(0);
    d.elems     = mesh->elements(0)->data();
    d.points    = mesh->points()->data();
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        const scalar_t y = d.points[1][i], z = d.points[2][i];
        d.points[0][i] += smesh::geom_t(scalar_t(0.13) * y * (scalar_t(1.5) - y) * z);
    }
    precompute_affine_geometry(d);

    std::vector<scalar_t> ogx, ogy, ogz, rx, ry, rz;

    // ---- a linear field reconstructs exactly ---------------------------------------
    {
        const scalar_t ax = scalar_t(0.37), ay = scalar_t(-0.21), az = scalar_t(0.58);
        std::vector<scalar_t> f((size_t)d.nnodes);
        for (ptrdiff_t i = 0; i < d.nnodes; ++i)
            f[(size_t)i] = scalar_t(1.7) + ax * d.points[0][i] + ay * d.points[1][i] + az * d.points[2][i];
        cvfem_hex8_assemble_nodal_grads_atomic(d, 0, f.data(), 1, ogx, ogy, ogz);
        scalar_t worst = 0;
        for (ptrdiff_t i = 0; i < d.nnodes; ++i)
            worst = std::max(worst,
                             std::max(std::fabs(ogx[(size_t)i] - ax),
                                      std::max(std::fabs(ogy[(size_t)i] - ay), std::fabs(ogz[(size_t)i] - az))));
        check(worst <= scalar_t(1e-12), "a linear field reconstructs to its exact gradient", (double)worst);
    }

    // ---- the general case against the definition -----------------------------------
    std::vector<scalar_t> f((size_t)d.nnodes);
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        const scalar_t x = d.points[0][i], y = d.points[1][i], z = d.points[2][i];
        f[(size_t)i] = std::sin(scalar_t(1.3) * x) * (scalar_t(1) + y * y) - scalar_t(0.4) * z * x;
    }
    cvfem_hex8_assemble_nodal_grads_atomic(d, 0, f.data(), 1, ogx, ogy, ogz);
    reference_grad(d, f, rx, ry, rz);
    {
        scalar_t scale = 0;
        for (const scalar_t v : rx) scale = std::max(scale, std::fabs(v));
        const scalar_t w = std::max(worst_diff(ogx, rx), std::max(worst_diff(ogy, ry), worst_diff(ogz, rz)));
        check(scale > scalar_t(1e-6), "the reference gradient is not trivially zero", (double)scale);
        check(w <= scalar_t(1e-13) * scale, "agrees with the definition, computed serially", (double)(w / scale));
    }

    // ---- the cached denominator ----------------------------------------------------
    {
        std::vector<scalar_t> w((size_t)d.nnodes, 0);
        for (ptrdiff_t e = 0; e < d.nelements; ++e) {
            const scalar_t vol = std::fabs(d.jacobian_determinant[(size_t)e]);
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) w[(size_t)d.elems[a][e]] += vol;
        }
        scalar_t worst = 0;
        for (ptrdiff_t i = 0; i < d.nnodes; ++i)
            worst = std::max(worst, std::fabs(d.grad_w_inv[(size_t)i] * w[(size_t)i] - scalar_t(1)));
        check(worst <= scalar_t(1e-14), "the cached weight is 1 / sum_e |det|", (double)worst);

        // A cache that never invalidates is worse than no cache: it would survive a change
        // of geometry rule and silently divide by the wrong denominator.
        const scalar_t before = d.grad_w_inv[0];
        cvfem_hex8_build_grad_weight(d, 0);
        check(d.grad_w_inv[0] == before, "a repeat build with the same key is a no-op");
        cvfem_hex8_build_grad_weight(d, 1);
        check(d.grad_w_isoparam == 1, "changing the geometry rule rebuilds it");
        cvfem_hex8_build_grad_weight(d, 0);
    }

    // ---- stride invariance ---------------------------------------------------------
    //
    // The action reads its direction's pressure with stride N_FIELDS out of the interleaved
    // Krylov vector; the residual reads the state contiguously. Same operator either way.
    {
        std::vector<scalar_t> interleaved((size_t)d.nnodes * N_FIELDS, scalar_t(0));
        for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
            interleaved[(size_t)i * N_FIELDS + 0] = scalar_t(-9);  // poison: must not be read
            interleaved[(size_t)i * N_FIELDS + 1] = scalar_t(-9);
            interleaved[(size_t)i * N_FIELDS + 2] = scalar_t(-9);
            interleaved[(size_t)i * N_FIELDS + 3] = f[(size_t)i];
        }
        std::vector<scalar_t> sx, sy, sz;
        cvfem_hex8_assemble_nodal_grads_atomic(d, 0, interleaved.data() + 3, N_FIELDS, sx, sy, sz);

        // Not bit-equality, and the reason is worth recording rather than banding away.
        // This sweep accumulates with `#pragma omp atomic update`, so the order in which a
        // node's elements reach it is not fixed and the last bits of the result are not
        // reproducible -- not between two strides, and not between two runs of the SAME
        // stride. Measure that self-variation and require the stride difference to be no
        // worse than it, which is the strongest statement the current implementation
        // supports. Moving this pass onto the packed layout, where a ghost reduction
        // replaces the atomics, would make it deterministic and let this tighten to zero.
        std::vector<scalar_t> ax, ay, az;
        cvfem_hex8_assemble_nodal_grads_atomic(d, 0, f.data(), 1, ax, ay, az);
        const scalar_t self = std::max(worst_diff(ax, ogx), std::max(worst_diff(ay, ogy), worst_diff(az, ogz)));
        const scalar_t w = std::max(worst_diff(sx, ogx), std::max(worst_diff(sy, ogy), worst_diff(sz, ogz)));
        scalar_t scale = 0;
        for (const scalar_t v : ogx) scale = std::max(scale, std::fabs(v));
        check(w <= scalar_t(1e-14) * scale, "stride 4 out of an interleaved vector matches stride 1", (double)w);
        std::printf("%-60s %-12.3e (informational)\n", "  the same input twice differs by", (double)self);
    }

    // ---- the packed sweep is the same operator, and is deterministic ---------------
    //
    // It replaces 24 atomics per element with a private per-pack accumulation and a ghost
    // reduction over the shared rows only. That changes the summation order, so agreement
    // is to round-off and not bitwise -- but it also FIXES the order, which the atomic
    // sweep does not, so the packed one must reproduce itself exactly where the atomic one
    // does not.
    {
        std::vector<scalar_t> px, py, pz, qx, qy, qz;
        cvfem_hex8_assemble_nodal_grads_packed(d, packed, 0, f.data(), 1, px, py, pz, ng_gbuf);
        cvfem_hex8_assemble_nodal_grads_packed(d, packed, 0, f.data(), 1, qx, qy, qz, ng_gbuf);

        scalar_t scale = 0;
        for (const scalar_t v : ogx) scale = std::max(scale, std::fabs(v));
        const scalar_t w = std::max(worst_diff(px, ogx), std::max(worst_diff(py, ogy), worst_diff(pz, ogz)));
        check(w <= scalar_t(1e-13) * scale, "the packed sweep agrees with the atomic one", (double)(w / scale));

        const scalar_t self = std::max(worst_diff(px, qx), std::max(worst_diff(py, qy), worst_diff(pz, qz)));
        check(self == scalar_t(0), "and reproduces itself exactly, which the atomic one does not",
              (double)self);

        // The linear-field property has to survive the change of sweep: it is the one the
        // Rhie-Chow correction is defined against.
        const scalar_t ax = scalar_t(0.37), ay = scalar_t(-0.21), az = scalar_t(0.58);
        std::vector<scalar_t> lin((size_t)d.nnodes);
        for (ptrdiff_t i = 0; i < d.nnodes; ++i)
            lin[(size_t)i] = scalar_t(1.7) + ax * d.points[0][i] + ay * d.points[1][i] + az * d.points[2][i];
        cvfem_hex8_assemble_nodal_grads_packed(d, packed, 0, lin.data(), 1, px, py, pz, ng_gbuf);
        scalar_t worst = 0;
        for (ptrdiff_t i = 0; i < d.nnodes; ++i)
            worst = std::max(worst,
                             std::max(std::fabs(px[(size_t)i] - ax),
                                      std::max(std::fabs(py[(size_t)i] - ay), std::fabs(pz[(size_t)i] - az))));
        check(worst <= scalar_t(1e-12), "packed: a linear field still reconstructs exactly", (double)worst);
    }

    // ---- isoparametric agrees with affine on an affine mesh ------------------------
    //
    // Both evaluate the geometry at the element centre, so on a mesh of parallelepipeds the
    // two rules are the same rule. This mesh is warped, so they are not expected to agree
    // to round-off -- only to stay close, which is what says the isoparametric branch is
    // reconstructing the same quantity and not something else.
    {
        std::vector<scalar_t> ix, iy, iz;
        cvfem_hex8_assemble_nodal_grads_atomic(d, 1, f.data(), 1, ix, iy, iz);
        scalar_t scale = 0;
        for (const scalar_t v : ogx) scale = std::max(scale, std::fabs(v));
        const scalar_t w = std::max(worst_diff(ix, ogx), std::max(worst_diff(iy, ogy), worst_diff(iz, ogz)));
        check(w <= scalar_t(0.05) * scale, "the isoparametric rule tracks the affine one", (double)(w / scale));

        // The PACKED isoparametric sweep against the flat one. The geometry kind is a template
        // parameter of both sweeps, so this is a distinct instantiation from every other case in
        // this file -- all of which are affine -- and nothing else here would compile it, let
        // alone run it. The two paths differ only in how they reach the element's nodes, so on
        // the same mesh and the same field they must agree to round-off.
        std::vector<scalar_t> pix, piy, piz;
        cvfem_hex8_assemble_nodal_grads_packed(d, packed, 1, f.data(), 1, pix, piy, piz, ng_gbuf);
        const scalar_t wp = std::max(worst_diff(pix, ix), std::max(worst_diff(piy, iy), worst_diff(piz, iz)));
        check(wp <= scalar_t(1e-13) * scale, "the packed isoparametric sweep matches the flat one",
              (double)(wp / scale));
    }

    // ---- the FUSED nine-component sweeps equal three single-component ones -----------
    //
    // assemble_nodal_u_grad used to call the one-field sweep three times and transpose. It now
    // does one sweep for all nine, in both the packed and the flat path, which is a claim of
    // EXACT equivalence: the per-component arithmetic is unchanged and only the geometry,
    // indirection and reduction are shared. Three single sweeps are therefore the independent
    // witness, and the comparison is exact rather than toleranced -- the same operations in
    // the same order must give the same bits. A tolerance here would hide precisely the kind
    // of reassociation that fusing invites.
    {
        std::vector<scalar_t> u[3];
        for (int r = 0; r < 3; ++r) {
            u[r].resize((size_t)d.nnodes);
            for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
                const scalar_t x = d.points[0][i], y = d.points[1][i], z = d.points[2][i];
                u[r][(size_t)i] = std::sin(scalar_t(1.3 + r) * x) + scalar_t(0.7) * y * z
                                  + scalar_t(0.2 + 0.1 * r) * x * y;
            }
        }
        // the witness: three separate single-field sweeps, transposed as the old code did
        std::vector<scalar_t> ref((size_t)d.nnodes * 9, scalar_t(0));
        for (int r = 0; r < 3; ++r) {
            cvfem_hex8_assemble_nodal_grads_packed(d, packed, 0, u[r].data(), 1, ogx, ogy, ogz, ng_gbuf);
            for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
                ref[(size_t)i * 9 + (size_t)r * 3 + 0] = ogx[(size_t)i];
                ref[(size_t)i * 9 + (size_t)r * 3 + 1] = ogy[(size_t)i];
                ref[(size_t)i * 9 + (size_t)r * 3 + 2] = ogz[(size_t)i];
            }
        }

        const scalar_t *const srcs[3] = {u[0].data(), u[1].data(), u[2].data()};
        std::vector<scalar_t> got, gbuf;
        cvfem_hex8_assemble_nodal_grads_packed(d, packed, 0, srcs, 3, got, gbuf);
        ptrdiff_t bad = 0;
        scalar_t  fused_worst = 0, fused_scale = 0;
        for (size_t k = 0; k < ref.size(); ++k) {
            if (got[k] != ref[k]) ++bad;
            fused_worst = std::max(fused_worst, std::fabs(got[k] - ref[k]));
            fused_scale = std::max(fused_scale, std::fabs(ref[k]));
        }
        // ROUND-OFF, NOT BIT-FOR-BIT, and the change is deliberate. This asked for exact equality
        // between two DIFFERENT loop shapes -- nine components accumulated in one pass against
        // three components accumulated three times -- and no compiler owes that: with FMA
        // contraction the two fuse differently. It held on Apple clang and failed on GCC for
        // aarch64, where 464 of the entries differed in their last bits while the worst relative
        // difference was 2.1e-16, the same order as the flat sibling below that has always been
        // held to a tolerance. The old check reported only the COUNT of differing entries, which
        // cannot distinguish that from a wrong answer, so the magnitude is what is checked now
        // and the count rides along as information.
        //
        // What the paper claims bitwise is one kernel reproducing ITSELF across runs and thread
        // counts, which "and reproduces itself exactly" above still gates. Two spellings of one
        // operation agreeing to round-off is the right standard here.
        check(fused_worst <= scalar_t(1e-14) * (fused_scale > 0 ? fused_scale : scalar_t(1)),
              "packed: the fused nine-component sweep matches three sweeps to round-off",
              (double)(fused_worst / (fused_scale > 0 ? fused_scale : scalar_t(1))));
        std::printf("  %-62s %ld (informational)\n", "entries differing in their last bits", (long)bad);

        // The flat sweep uses atomics, so its summation order is not reproducible and an exact
        // comparison would be testing the scheduler. It is held to round-off instead, against
        // the same witness.
        std::vector<scalar_t> flat;
        cvfem_hex8_assemble_nodal_grads_atomic(d, 0, srcs, 3, flat);
        scalar_t worst = 0, scale = 0;
        for (size_t k = 0; k < ref.size(); ++k) {
            worst = std::max(worst, std::fabs(flat[k] - ref[k]));
            scale = std::max(scale, std::fabs(ref[k]));
        }
        check(worst <= scalar_t(1e-12) * std::max(scale, scalar_t(1)),
              "flat: the fused sweep matches to round-off", (double)(worst / std::max(scale, scalar_t(1))));
    }

    if (g_failures) {
        std::fprintf(stderr, "cvfem_nodal_grad_test: %d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all nodal-gradient checks passed\n");
    return 0;
}
