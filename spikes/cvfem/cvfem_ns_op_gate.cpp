// Gate: sfem::CVFEMNavierStokes must reproduce the standalone driver.
//
// The Op delegates to the same kernels, so this is not testing the physics -- it is
// testing the wiring around them: that the FunctionSpace-derived mesh state, the domain
// extents the boundary sub-control-surfaces key off, the packed/coloring setup and the
// BSR slot caches all land where the driver puts them. Any of those being subtly wrong
// would still produce a plausible-looking residual, which is exactly why this compares
// against the reference path rather than against a tolerance on the physics.
//
// Both paths run in one process on one mesh, so the only admissible difference is the
// order of floating-point accumulation in the parallel scatter.

#include "cvfem_hex8_ns_op.hpp"

// The gate needs the reference path too, so unlike a driver it includes the core
// deliberately. Anything that only wants the operator includes the header above alone.
#include "cvfem_hex8_ns_core.hpp"

#include "sfem_context.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

    // A deterministic, non-symmetric state. Something like the exact Poiseuille profile
    // would leave the continuity residual near zero and hide a scatter that drops terms,
    // so this deliberately excites every component and every coupling.
    void fill_state(const MeshData &d, std::vector<scalar_t> &x) {
        const auto *const px = d.points[0];
        const auto *const py = d.points[1];
        const auto *const pz = d.points[2];
        for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
            const scalar_t X = px[i], Y = py[i], Z = pz[i];
            x[(size_t)i * 4 + 0] = std::sin(scalar_t(1.7) * X) * std::cos(scalar_t(2.3) * Y) + scalar_t(0.3) * Z;
            x[(size_t)i * 4 + 1] = std::cos(scalar_t(1.1) * Y) * (scalar_t(1) + scalar_t(0.2) * X) - scalar_t(0.15) * Z;
            x[(size_t)i * 4 + 2] = std::sin(scalar_t(0.9) * Z) * (scalar_t(0.5) + scalar_t(0.1) * Y);
            x[(size_t)i * 4 + 3] = scalar_t(0.7) * X - scalar_t(0.4) * Y + scalar_t(0.25) * Z * Z;
        }
    }

    struct Diff {
        double max_abs{0};
        double max_rel{0};
        double ref_inf{0};
    };

    Diff compare(const std::vector<scalar_t> &ref, const std::vector<real_t> &got) {
        Diff r;
        for (size_t i = 0; i < ref.size(); ++i) {
            const double a = (double)ref[i];
            const double b = (double)got[i];
            const double e = std::fabs(a - b);
            r.ref_inf = std::max(r.ref_inf, std::fabs(a));
            r.max_abs = std::max(r.max_abs, e);
        }
        r.max_rel = (r.ref_inf > 0) ? r.max_abs / r.ref_inf : r.max_abs;
        return r;
    }

}  // namespace

int main(int argc, char **argv) {
    auto ctx = sfem::initialize(argc, argv);

    const int      nx = 8, ny = 4, nz = 4;
    const scalar_t Lx = 4, Ly = 1, Lz = 1;
    const scalar_t rho = 1, mu = 0.01, rc_scale = 1;
    // Loose enough to admit a different parallel accumulation order, tight enough that
    // any real wiring mistake -- a missed element, a wrong slot, a stale gradient --
    // shows up as a relative error many orders of magnitude larger.
    const double   tol = 1e-11;

    int failures = 0;

    for (const auto geom : {GeomKind::Affine, GeomKind::Isoparam}) {
        for (const int pack_size : {2048, 0}) {
            if (geom == GeomKind::Isoparam && pack_size == 0) continue;  // same path either way

            const char *gname = (geom == GeomKind::Affine) ? "affine" : "isoparam";

            auto mesh = smesh::Mesh::create_hex8_cube(ctx->communicator(), nx, ny, nz, 0, 0, 0, Lx, Ly, Lz);

            // The Op is built and initialized FIRST, before the reference path touches the
            // mesh. That ordering is the whole point: initialize() renumbers the nodes for
            // the packed layout, and a gate that let the reference pack the mesh first
            // would find the mesh already renumbered and never exercise the case a real
            // driver hits. This gate did exactly that once, and passed while the frontend
            // driver segfaulted on the same code.
            auto fs = sfem::FunctionSpace::create(mesh, N_FIELDS);
            auto op = std::make_shared<sfem::CVFEMNavierStokes>(fs);
            op->rho             = rho;
            op->mu              = mu;
            op->rhie_chow_scale = rc_scale;
            op->geom            = (geom == GeomKind::Isoparam) ? sfem::CVFEMGeometry::Isoparam
                                                               : sfem::CVFEMGeometry::Affine;
            op->pack_size       = pack_size;
            op->initialize();

            // ---- reference: exactly what the driver sets up, on the mesh as the Op left it ----
            MeshData     d;
            PackedData   packed;
            PackColoring coloring;
            d.Lx              = Lx;
            d.Ly              = Ly;
            d.Lz              = Lz;
            d.rhie_chow_scale = rc_scale;
            d.mesh            = mesh;
            if (geom == GeomKind::Affine && pack_size > 0) {
                packed     = make_packed(d.mesh, pack_size);
                d.packed   = &packed;
                coloring   = cvfem_build_pack_coloring(packed.n_packs, packed.owned_nodes_ptr, packed.ghost_ptr, packed.ghost_idx);
                d.coloring = &coloring;
            }
            d.nnodes    = mesh->n_nodes();
            d.nelements = mesh->n_elements(0);
            d.elems     = mesh->elements(0)->data();
            d.points    = mesh->points()->data();
            if (geom == GeomKind::Affine) cvfem_hex8_precompute_affine_geometry(d);

            d.ux.assign((size_t)d.nnodes, 0);
            d.uy.assign((size_t)d.nnodes, 0);
            d.uz.assign((size_t)d.nnodes, 0);
            d.p.assign((size_t)d.nnodes, 0);
            d.rx.assign((size_t)d.nnodes, 0);
            d.ry.assign((size_t)d.nnodes, 0);
            d.rz.assign((size_t)d.nnodes, 0);
            d.rc.assign((size_t)d.nnodes, 0);

            const ptrdiff_t       ndof = d.nnodes * N_FIELDS;
            std::vector<scalar_t> x((size_t)ndof);
            fill_state(d, x);
            unpack_fields(d, x.data());

            apply_residual(d, rho, mu, geom);
            std::vector<scalar_t> ref_r((size_t)ndof);
            pack_residual(d, ref_r.data());

            BSR4 b = make_bsr4(mesh);
            precompute_element_bsr_slots(d, b);
            assemble_jacobian(d, b, rho, mu, geom);
            std::vector<scalar_t> ref_h((size_t)b.nnz * 16);
            for (size_t i = 0; i < ref_h.size(); ++i) ref_h[i] = b.data()[i];

            // A direction for the matrix-free action, reusing the state generator so it
            // is just as non-trivial.
            std::vector<scalar_t> dir((size_t)ndof);
            fill_state(d, dir);
            for (size_t i = 0; i < dir.size(); ++i) dir[i] *= scalar_t(0.37);
            std::vector<scalar_t> ref_jv((size_t)ndof, 0);
            assemble_nodal_p_grad(d, geom);
            apply_jacobian_action_accumulate(d, rho, mu, geom, dir.data(), ref_jv.data());

            std::vector<real_t> op_r((size_t)ndof, 0);
            op->gradient(x.data(), op_r.data());

            std::vector<real_t> op_h((size_t)b.nnz * 16, 0);
            op->hessian_bsr(x.data(), b.rowptr, b.colidx, op_h.data());

            std::vector<real_t> op_jv((size_t)ndof, 0);
            op->apply(x.data(), dir.data(), op_jv.data());

            // Block diagonal against the diagonal blocks of the assembled Jacobian. This
            // is the check the bench's own diag verification cannot make: it compares
            // against a boundary-free assembly from its own header, so it would pass with
            // the Rhie-Chow and boundary terms missing.
            std::vector<scalar_t> ref_bd((size_t)d.nnodes * 16, scalar_t(0));
            for (ptrdiff_t r = 0; r < d.nnodes; ++r) {
                for (smesh::count_t j = b.rowptr[r]; j < b.rowptr[r + 1]; ++j) {
                    if (b.colidx[j] != (smesh::idx_t)r) continue;
                    for (int k = 0; k < 16; ++k) ref_bd[(size_t)r * 16 + k] = ref_h[(size_t)j * 16 + k];
                }
            }
            std::vector<real_t> op_bd((size_t)d.nnodes * 16, 0);
            op->hessian_block_diag(x.data(), op_bd.data());

            // And the scalar diagonal, which is what the base-class virtual exposes.
            std::vector<scalar_t> ref_sd((size_t)ndof, scalar_t(0));
            for (ptrdiff_t r = 0; r < d.nnodes; ++r)
                for (int c = 0; c < N_FIELDS; ++c)
                    ref_sd[(size_t)r * N_FIELDS + c] = ref_bd[(size_t)r * 16 + c * 4 + c];
            std::vector<real_t> op_sd((size_t)ndof, 0);
            op->hessian_diag(x.data(), op_sd.data());

            const Diff dr = compare(ref_r, op_r);
            const Diff dh = compare(ref_h, op_h);
            const Diff dj = compare(ref_jv, op_jv);
            const Diff db = compare(ref_bd, op_bd);
            const Diff ds = compare(ref_sd, op_sd);

            std::printf("%-9s pack=%-5d  grad=%.3e  bsr=%.3e  apply=%.3e  blockdiag=%.3e  diag=%.3e\n",
                        gname,
                        pack_size,
                        dr.max_rel,
                        dh.max_rel,
                        dj.max_rel,
                        db.max_rel,
                        ds.max_rel);

            const bool ok = dr.max_rel < tol && dh.max_rel < tol && dj.max_rel < tol && db.max_rel < tol &&
                            ds.max_rel < tol;
            if (!ok) {
                std::printf("  FAIL (tol %.1e)  |ref|_inf: grad %.6e  hess %.6e  apply %.6e\n",
                            tol,
                            dr.ref_inf,
                            dh.ref_inf,
                            dj.ref_inf);
                ++failures;
            }
        }
    }

    if (failures) {
        std::printf("cvfem_ns_op_gate: FAILED (%d configuration(s))\n", failures);
        return EXIT_FAILURE;
    }
    std::printf("cvfem_ns_op_gate: OK\n");
    return EXIT_SUCCESS;
}
