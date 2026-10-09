#pragma once

// THE STANDARD LAYOUT'S RETIRED SWEEPS.
//
// DESIGN.md's correction: "the micro-kernel selector must be removed. Only the best
// micro-kernels need to be used (given the results in Grace), so there should be only one per
// kernel. The rest is moved to subpar". These nine had no caller left once --kernel was gone.
//
// FOUR GENERATED ASSEMBLY ARRANGEMENTS -- flat, blockwise, rowwise, facewise. Grace at 28,756k
// dof: the generated arrangement assembles the atomic layout at 14 MELEM/s against the
// sum-factored kernel's 10, its one win anywhere -- and that 14 only ties the PACKED
// sum-factored rate and is half the coloured one (28), which is the fastest assembly in the
// spike. Keeping the sum-factored kernel therefore costs no configuration anyone would run.
// The rowwise and facewise arrangements had already lost the saturated evaluation.
//
// TWO FINITE-DIFFERENCE ASSEMBLIES, affine and isoparametric. These difference the residual
// with eps = 1e-6 and were never a performance arm: they are the correctness reference, and
// `--kernel current --assemble` silently measured the affine one because there is no
// hand-written affine assembly kernel for `current` to mean. A row reporting a
// finite-difference matrix as an assembly rate is one of the reports that made that flag a
// liability. The Jacobian-action gate's tolerance went back to 1e-8 with them: it had been
// loosened to 1e-3 to accommodate the truncation error of the differencing.
//
// THE ISOPARAMETRIC SPLIT, linear plus nonlinear. It assembles the geometry-only half once and
// restores it per iteration, adding only the velocity-dependent half. Grace job 4982167,
// 8,586,756 dof, 72 threads: 14.1 MELEM/s against the generated isoparametric assembly's 47.7
// -- 3.4x SLOWER -- and the affine form is 14.3 against 39.0, 2.7x slower. The half it saves is
// the cheap one, and restoring it costs a full pass over the values. It was never measured
// before being built.
//
// THE GENERATED AFFINE RESIDUAL'S SWEEP. Its micro-kernel was already quarantined, so this
// sweep called a stub that aborts -- which made `--verify --layout atomic` abort too, since the
// verification block reached it without going through the flag's refusal. That was true at
// d0eb4c53e as well, with -DCVFEM_ENABLE_SUBPAR=OFF, so the oracle it fed
// (verify_sympy_residual_vs_current_abs) had been dead rather than passing. On the packed
// layout the same oracle compared sumfact against current, which
// verify_packed_sumfact_residual_vs_current_abs already does.
//
// Build with -DCVFEM_ENABLE_SUBPAR to reach them.

#include "kernels/standard/affine/cvfem_hex8_best_atomic_affine.hpp"
#include "kernels/standard/isoparametric/cvfem_hex8_best_atomic_isoparam.hpp"


static SFEM_NOINLINE void assemble_jacobian_atomic_fd(
        // The staging objects are gone; what this sweep reads out of them is what it takes.
        const scalar_t *const *const SFEM_RESTRICT adj_ptr,
        const scalar_t *const SFEM_RESTRICT det_ptr,
        idx_t **const SFEM_RESTRICT mesh_elems,
        const ptrdiff_t nelements,
        const scalar_t *const SFEM_RESTRICT pres,
        const scalar_t *const SFEM_RESTRICT ux_src,
        const scalar_t *const SFEM_RESTRICT uy_src,
        const scalar_t *const SFEM_RESTRICT uz_src, 
        const idx_t *const SFEM_RESTRICT bsr_colidx,
        const count_t *const SFEM_RESTRICT slots,
        const ptrdiff_t bsr_nnz,
        const count_t *const SFEM_RESTRICT bsr_rowptr,
        scalar_t *const SFEM_RESTRICT values, const scalar_t rho, const scalar_t mu) {
    // zero_bsr4 inlined, so that this sweep names the matrix arrays it already writes rather
    // than the BSR4 object that owns them. Identical work: that function is this zeroing plus
    // the phase probe, which the macros carry here.
    CVFEM_PHASE_CLOCK(_tz);
    cvfem_zero_scalars(values, bsr_nnz * 16);
    CVFEM_PHASE_GLOBAL(_tz, PH_ZERO);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        scalar_t ux[8], uy[8], uz[8], p[8], ke[CVFEM_HEX8_N_DOF * CVFEM_HEX8_N_DOF];
        gather_element_fields(mesh_elems, ux_src, uy_src, uz_src, pres, e, ux, uy, uz, p);
        scalar_t adj[9], det;
        load_hex8_adj(adj_ptr, det_ptr, e, adj, &det);
        cvfem_hex8_ns_upwind_jacobian_fd(rho, mu, adj, det, ux, uy, uz, p, ke);

        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const idx_t row = mesh_elems[a][e];
            for (int bnode = 0; bnode < CVFEM_HEX8_N_NODES; ++bnode) {
                const count_t slot =
                        slots ? slots[(size_t)e * 64 + a * 8 + bnode] : find_bsr_slot(bsr_rowptr, bsr_colidx, row, mesh_elems[bnode][e]);
                scalar_t *const      blk  = values + (ptrdiff_t)slot * 16;
                for (int rf = 0; rf < 4; ++rf) {
                    for (int cf = 0; cf < 4; ++cf) {
                        const scalar_t v = ke[(a * 4 + rf) * CVFEM_HEX8_N_DOF + (bnode * 4 + cf)];
                        CVFEM_ATOMIC_ADD(blk[rf * 4 + cf], v);
                    }
                }
            }
        }
    }
}


static SFEM_NOINLINE void assemble_jacobian_atomic_sympy(
        // The staging objects are gone; what this sweep reads out of them is what it takes.
        const scalar_t *const *const SFEM_RESTRICT adj_ptr,
        const scalar_t *const SFEM_RESTRICT det_ptr,
        idx_t **const SFEM_RESTRICT mesh_elems,
        const ptrdiff_t nelements,
        const scalar_t *const SFEM_RESTRICT pres,
        const scalar_t *const SFEM_RESTRICT ux_src,
        const scalar_t *const SFEM_RESTRICT uy_src,
        const scalar_t *const SFEM_RESTRICT uz_src, 
        const count_t *const SFEM_RESTRICT slots,
        const ptrdiff_t bsr_nnz,
        scalar_t *const SFEM_RESTRICT values, const scalar_t rho, const scalar_t mu) {
    // zero_bsr4 inlined, so that this sweep names the matrix arrays it already writes rather
    // than the BSR4 object that owns them. Identical work: that function is this zeroing plus
    // the phase probe, which the macros carry here.
    CVFEM_PHASE_CLOCK(_tz);
    cvfem_zero_scalars(values, bsr_nnz * 16);
    CVFEM_PHASE_GLOBAL(_tz, PH_ZERO);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        scalar_t ux[8], uy[8], uz[8], p[8];
        gather_element_fields(mesh_elems, ux_src, uy_src, uz_src, pres, e, ux, uy, uz, p);
        scalar_t adj[9], det;
        load_hex8_adj(adj_ptr, det_ptr, e, adj, &det);
        cvfem_hex8_ns_upwind_sympy_jacobian_add_bsr_slots(rho, mu, adj, det, ux, uy, uz, slots + (size_t)e * 64, values);
    }
}


static SFEM_NOINLINE void assemble_jacobian_atomic_sympy_block(
        // The staging objects are gone; what this sweep reads out of them is what it takes.
        const scalar_t *const *const SFEM_RESTRICT adj_ptr,
        const scalar_t *const SFEM_RESTRICT det_ptr,
        idx_t **const SFEM_RESTRICT mesh_elems,
        const ptrdiff_t nelements,
        const scalar_t *const SFEM_RESTRICT pres,
        const scalar_t *const SFEM_RESTRICT ux_src,
        const scalar_t *const SFEM_RESTRICT uy_src,
        const scalar_t *const SFEM_RESTRICT uz_src, 
        const count_t *const SFEM_RESTRICT slots,
        const ptrdiff_t bsr_nnz,
        scalar_t *const SFEM_RESTRICT values, const scalar_t rho, const scalar_t mu) {
    // zero_bsr4 inlined, so that this sweep names the matrix arrays it already writes rather
    // than the BSR4 object that owns them. Identical work: that function is this zeroing plus
    // the phase probe, which the macros carry here.
    CVFEM_PHASE_CLOCK(_tz);
    cvfem_zero_scalars(values, bsr_nnz * 16);
    CVFEM_PHASE_GLOBAL(_tz, PH_ZERO);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        scalar_t ux[8], uy[8], uz[8], p[8];
        gather_element_fields(mesh_elems, ux_src, uy_src, uz_src, pres, e, ux, uy, uz, p);
        scalar_t adj[9], det;
        load_hex8_adj(adj_ptr, det_ptr, e, adj, &det);
        cvfem_hex8_ns_upwind_sympy_jacobian_add_bsr_slots_blockwise(
                rho, mu, adj, det, ux, uy, uz, slots + (size_t)e * 64, values);
    }
}


static SFEM_NOINLINE void assemble_jacobian_atomic_sympy_row(
        // The staging objects are gone; what this sweep reads out of them is what it takes.
        const scalar_t *const *const SFEM_RESTRICT adj_ptr,
        const scalar_t *const SFEM_RESTRICT det_ptr,
        idx_t **const SFEM_RESTRICT mesh_elems,
        const ptrdiff_t nelements,
        const scalar_t *const SFEM_RESTRICT pres,
        const scalar_t *const SFEM_RESTRICT ux_src,
        const scalar_t *const SFEM_RESTRICT uy_src,
        const scalar_t *const SFEM_RESTRICT uz_src, 
        const count_t *const SFEM_RESTRICT slots,
        const ptrdiff_t bsr_nnz,
        scalar_t *const SFEM_RESTRICT values, const scalar_t rho, const scalar_t mu) {
    // zero_bsr4 inlined, so that this sweep names the matrix arrays it already writes rather
    // than the BSR4 object that owns them. Identical work: that function is this zeroing plus
    // the phase probe, which the macros carry here.
    CVFEM_PHASE_CLOCK(_tz);
    cvfem_zero_scalars(values, bsr_nnz * 16);
    CVFEM_PHASE_GLOBAL(_tz, PH_ZERO);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        scalar_t ux[8], uy[8], uz[8], p[8];
        gather_element_fields(mesh_elems, ux_src, uy_src, uz_src, pres, e, ux, uy, uz, p);
        scalar_t adj[9], det;
        load_hex8_adj(adj_ptr, det_ptr, e, adj, &det);
        cvfem_hex8_ns_upwind_sympy_jacobian_add_bsr_slots_rowwise(
                rho, mu, adj, det, ux, uy, uz, slots + (size_t)e * 64, values);
    }
}


static SFEM_NOINLINE void assemble_jacobian_atomic_sympy_face(
        // The staging objects are gone; what this sweep reads out of them is what it takes.
        const scalar_t *const *const SFEM_RESTRICT adj_ptr,
        const scalar_t *const SFEM_RESTRICT det_ptr,
        idx_t **const SFEM_RESTRICT mesh_elems,
        const ptrdiff_t nelements,
        const scalar_t *const SFEM_RESTRICT pres,
        const scalar_t *const SFEM_RESTRICT ux_src,
        const scalar_t *const SFEM_RESTRICT uy_src,
        const scalar_t *const SFEM_RESTRICT uz_src, 
        const count_t *const SFEM_RESTRICT slots,
        const ptrdiff_t bsr_nnz,
        scalar_t *const SFEM_RESTRICT values, const scalar_t rho, const scalar_t mu) {
    // zero_bsr4 inlined, so that this sweep names the matrix arrays it already writes rather
    // than the BSR4 object that owns them. Identical work: that function is this zeroing plus
    // the phase probe, which the macros carry here.
    CVFEM_PHASE_CLOCK(_tz);
    cvfem_zero_scalars(values, bsr_nnz * 16);
    CVFEM_PHASE_GLOBAL(_tz, PH_ZERO);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        scalar_t ux[8], uy[8], uz[8], p[8];
        gather_element_fields(mesh_elems, ux_src, uy_src, uz_src, pres, e, ux, uy, uz, p);
        scalar_t adj[9], det;
        load_hex8_adj(adj_ptr, det_ptr, e, adj, &det);
        cvfem_hex8_ns_upwind_sympy_jacobian_add_bsr_slots_facewise(
                rho, mu, adj, det, ux, uy, uz, slots + (size_t)e * 64, values);
    }
}


static SFEM_NOINLINE void apply_residual_atomic_sympy(
        // The staging objects are gone; what this sweep reads out of them is what it takes.
        const scalar_t *const *const SFEM_RESTRICT adj_ptr,
        const scalar_t *const SFEM_RESTRICT det_ptr,
        idx_t **const SFEM_RESTRICT mesh_elems,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const scalar_t *const SFEM_RESTRICT pres,
        scalar_t *const SFEM_RESTRICT rc_out,
        scalar_t *const SFEM_RESTRICT rx,
        scalar_t *const SFEM_RESTRICT ry,
        scalar_t *const SFEM_RESTRICT rz,
        const scalar_t *const SFEM_RESTRICT ux_src,
        const scalar_t *const SFEM_RESTRICT uy_src,
        const scalar_t *const SFEM_RESTRICT uz_src, const scalar_t rho, const scalar_t mu) {
    reset_residual(nnodes, rx, ry, rz, rc_out);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        scalar_t ux[8], uy[8], uz[8], p[8], r[CVFEM_HEX8_N_DOF];
        gather_element_fields(mesh_elems, ux_src, uy_src, uz_src, pres, e, ux, uy, uz, p);
        scalar_t adj[9], det;
        load_hex8_adj(adj_ptr, det_ptr, e, adj, &det);
        cvfem_hex8_ns_upwind_sympy_residual(rho, mu, adj, det, ux, uy, uz, p, r);

        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const idx_t g = mesh_elems[a][e];
            atomic_add(rx, g, r[a * 4 + 0]);
            atomic_add(ry, g, r[a * 4 + 1]);
            atomic_add(rz, g, r[a * 4 + 2]);
            atomic_add(rc_out, g, r[a * 4 + 3]);
        }
    }
}


// Finite-difference Jacobian on isoparametric geometry. The affine layout has had this
// since the beginning; the isoparametric atomic path did not, so `--kernel fd --geom
// isoparam --layout atomic` silently ran the hand-written kernel and reported its speed
// under the name `fd`. The kernel it needs already existed and was already used by the
// packed layout (cvfem_hex8_best_packed.hpp), which is why that layout reported the
// honest -- and much slower, as a finite-difference Jacobian should be -- figure.
static SFEM_NOINLINE void assemble_jacobian_atomic_fd_isoparam(
        // The staging objects are gone; what this sweep reads out of them is what it takes.
        idx_t **const SFEM_RESTRICT mesh_elems,
        const ptrdiff_t nelements,
        const scalar_t *const SFEM_RESTRICT pres,
        geom_t **const SFEM_RESTRICT points,
        const scalar_t *const SFEM_RESTRICT ux_src,
        const scalar_t *const SFEM_RESTRICT uy_src,
        const scalar_t *const SFEM_RESTRICT uz_src,
                                                               
        const idx_t *const SFEM_RESTRICT bsr_colidx,
        const count_t *const SFEM_RESTRICT slots,
        const ptrdiff_t bsr_nnz,
        const count_t *const SFEM_RESTRICT bsr_rowptr,
        scalar_t *const SFEM_RESTRICT values,
                                                               const scalar_t rho,
                                                               const scalar_t mu) {
    // zero_bsr4 inlined, so that this sweep names the matrix arrays it already writes rather
    // than the BSR4 object that owns them. Identical work: that function is this zeroing plus
    // the phase probe, which the macros carry here.
    CVFEM_PHASE_CLOCK(_tz);
    cvfem_zero_scalars(values, bsr_nnz * 16);
    CVFEM_PHASE_GLOBAL(_tz, PH_ZERO);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        scalar_t x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8];
        scalar_t ke[CVFEM_HEX8_N_DOF * CVFEM_HEX8_N_DOF];
        gather_element_coords(mesh_elems, points, e, x, y, z);
        gather_element_fields(mesh_elems, ux_src, uy_src, uz_src, pres, e, ux, uy, uz, p);
        cvfem_hex8_ns_upwind_jacobian_fd_isoparam(rho, mu, x, y, z, ux, uy, uz, p, ke);

        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const idx_t row = mesh_elems[a][e];
            for (int bnode = 0; bnode < CVFEM_HEX8_N_NODES; ++bnode) {
                const count_t slot =
                        slots ? slots[(size_t)e * 64 + a * 8 + bnode]
                              : find_bsr_slot(bsr_rowptr, bsr_colidx, row, mesh_elems[bnode][e]);
                scalar_t *const blk = values + (ptrdiff_t)slot * 16;
                for (int rf = 0; rf < 4; ++rf)
                    for (int cf = 0; cf < 4; ++cf)
                        CVFEM_ATOMIC_ADD(blk[rf * 4 + cf],
                                         ke[(a * 4 + rf) * CVFEM_HEX8_N_DOF + (bnode * 4 + cf)]);
            }
        }
    }
}


// ---------------------------------------------------------------------------
// Split assembly on isoparametric geometry.
//
// The viscous block depends on geometry and mu only, so it is constant across Newton
// iterations even though the geometry is rebuilt at each sub-control surface. The two
// halves are selected out of one kernel body by the Part parameter, so linear +
// nonlinear reproduces the full assembly by construction.
static SFEM_NOINLINE void assemble_jacobian_atomic_linear_isoparam(
        // The staging objects are gone; what this sweep reads out of them is what it takes.
        idx_t **const SFEM_RESTRICT mesh_elems,
        const ptrdiff_t nelements,
        const scalar_t *const SFEM_RESTRICT pres,
        geom_t **const SFEM_RESTRICT points,
        const scalar_t *const SFEM_RESTRICT ux_src,
        const scalar_t *const SFEM_RESTRICT uy_src,
        const scalar_t *const SFEM_RESTRICT uz_src,
                                                                   
        const count_t *const SFEM_RESTRICT slots,
                                                                   const scalar_t        mu,
                                                                   scalar_t *const SFEM_RESTRICT linear) {
    // The buffer arrives sized. It used to be a std::vector& that this sweep called .assign() on,
    // which is an allocation inside a kernel -- and a kernel that allocates cannot be handed a
    // device buffer or a sub-range. The caller sizes and zeroes it.
    scalar_t *const SFEM_RESTRICT             values = linear;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        scalar_t x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8];
        gather_element_coords(mesh_elems, points, e, x, y, z);
        gather_element_fields(mesh_elems, ux_src, uy_src, uz_src, pres, e, ux, uy, uz, p);
        cvfem_hex8_ns_upwind_jacobian_add_slots_isoparam<true, CVFEM_HEX8_PART_LINEAR>(
                scalar_t(0), mu, x, y, z, ux, uy, uz, slots + (size_t)e * 64, values);
        (void)p;
    }
}


static SFEM_NOINLINE void assemble_jacobian_atomic_nonlinear_isoparam(
        
        // The staging objects are gone; what this sweep reads out of them is what it takes.
        const scalar_t *const *const SFEM_RESTRICT adj_ptr,
        const scalar_t *const SFEM_RESTRICT det_ptr,
        idx_t **const SFEM_RESTRICT mesh_elems,
        const uint8_t *const SFEM_RESTRICT face_mask,
        const ptrdiff_t nelements,
        const scalar_t *const SFEM_RESTRICT pres,
        const scalar_t *const SFEM_RESTRICT pgx,
        const scalar_t *const SFEM_RESTRICT pgy,
        const scalar_t *const SFEM_RESTRICT pgz,
        geom_t **const SFEM_RESTRICT points,
        const scalar_t *const SFEM_RESTRICT qgx,
        const scalar_t *const SFEM_RESTRICT qgy,
        const scalar_t *const SFEM_RESTRICT qgz,
        const scalar_t *const SFEM_RESTRICT ux_src,
        const scalar_t *const SFEM_RESTRICT uy_src,
        const scalar_t *const SFEM_RESTRICT uz_src,
                                              // Which optional terms are on, resolved once per
                                              // solve by the caller rather than per sweep here:
                                              // cvfem_hex8_extras_of reads the mesh, which this
                                              // kernel is not meant to name.
                                              const Hex8Extras &opt, 
        const count_t *const SFEM_RESTRICT slots,
        const ptrdiff_t bsr_nnz,
        scalar_t *const SFEM_RESTRICT values, const scalar_t rho, const scalar_t mu,
        const scalar_t *const SFEM_RESTRICT linear) {

    // Restore the constant part, then add only what the velocity changes.
    std::memcpy(values, linear, (size_t)bsr_nnz * 16 * sizeof(scalar_t));


#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        scalar_t         ux[8], uy[8], uz[8], p[8];
        Hex8ExtraScratch ex;
        ex.load(mesh_elems, points, face_mask, pgx, pgy, pgz, qgx, qgy, qgz, ux_src, uy_src, uz_src, adj_ptr, det_ptr, opt, e);
        if (!opt.with_rc && !opt.with_bnd) gather_element_coords(mesh_elems, points, e, ex.x, ex.y, ex.z);
        gather_element_fields(mesh_elems, ux_src, uy_src, uz_src, pres, e, ux, uy, uz, p);
        cvfem_hex8_ns_upwind_jacobian_add_slots_isoparam<true, CVFEM_HEX8_PART_NONLINEAR>(
                rho, mu, ex.x, ex.y, ex.z, ux, uy, uz, slots + (size_t)e * 64, values, ex.rc, p);
    }
}
