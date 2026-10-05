#pragma once

// The standard layout's isoparametric sweeps.
//
// Separated by geometry as DESIGN.md asks, by MOVING whole functions: these two sets were always
// distinct sweeps, differing in where the Jacobian comes from -- one adjugate per element read
// from a table, against one derived per sub-control volume from the node coordinates. Nothing is
// duplicated to achieve the split.
//
// Where a format's two geometries are ONE sweep templated on `bool ISO` -- the packed and store
// layouts -- they stay that way. DESIGN.md's clause is "logically separated (now they are mixed
// in with enum and booleans)"; the enum and the booleans are gone and the choice is made at
// compile time, which is the separation it asks for. Splitting those physically would mean two
// copies of the pack staging, the drain and the ghost reduction.

#include "kernels/standard/cvfem_hex8_best_atomic.hpp"


static SFEM_NOINLINE void apply_jacobian_action_atomic_isoparam(
        // The staging objects are gone; what this sweep reads out of them is what it takes.
        const scalar_t *const *const SFEM_RESTRICT adj_ptr,
        const scalar_t *const SFEM_RESTRICT det_ptr,
        idx_t **const SFEM_RESTRICT mesh_elems,
        const uint8_t *const SFEM_RESTRICT face_mask,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
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
                                                                const scalar_t        rho,
                                                                const scalar_t        mu,
                                                                const scalar_t *const dir,
                                                                scalar_t *const       jv) {
    cvfem_zero_scalars(jv, nnodes * CVFEM_HEX8_N_FIELDS);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        scalar_t         ux[8], uy[8], uz[8], p[8], vx[8], vy[8], vz[8], q[8], r[CVFEM_HEX8_N_DOF];
        Hex8ExtraScratch ex;
        ex.load(mesh_elems, points, face_mask, pgx, pgy, pgz, qgx, qgy, qgz, ux_src, uy_src, uz_src, adj_ptr, det_ptr, opt, e);
        if (!opt.with_rc && !opt.with_bnd) gather_element_coords(mesh_elems, points, e, ex.x, ex.y, ex.z);
        gather_element_fields(mesh_elems, ux_src, uy_src, uz_src, pres, e, ux, uy, uz, p);
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const idx_t                  g  = mesh_elems[a][e];
            const scalar_t *const SFEM_RESTRICT dv = dir + (ptrdiff_t)g * CVFEM_HEX8_N_FIELDS;
            vx[a]                                  = dv[0];
            vy[a]                                  = dv[1];
            vz[a]                                  = dv[2];
            q[a]                                   = dv[3];
        }
        cvfem_hex8_ns_upwind_jacobian_action_isoparam(rho, mu, ex.x, ex.y, ex.z, ux, uy, uz, vx, vy, vz, q, r,
                                                      ex.rc, p);
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const idx_t g = mesh_elems[a][e];
            atomic_add(jv + (ptrdiff_t)g * CVFEM_HEX8_N_FIELDS + 0, 0, r[a * 4 + 0]);
            atomic_add(jv + (ptrdiff_t)g * CVFEM_HEX8_N_FIELDS + 1, 0, r[a * 4 + 1]);
            atomic_add(jv + (ptrdiff_t)g * CVFEM_HEX8_N_FIELDS + 2, 0, r[a * 4 + 2]);
            atomic_add(jv + (ptrdiff_t)g * CVFEM_HEX8_N_FIELDS + 3, 0, r[a * 4 + 3]);
        }
    }
}

static SFEM_NOINLINE void apply_residual_atomic_isoparam(
        // The staging objects are gone; what this sweep reads out of them is what it takes.
        const scalar_t *const *const SFEM_RESTRICT adj_ptr,
        const scalar_t *const SFEM_RESTRICT det_ptr,
        idx_t **const SFEM_RESTRICT mesh_elems,
        const uint8_t *const SFEM_RESTRICT face_mask,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const scalar_t *const SFEM_RESTRICT pres,
        const scalar_t *const SFEM_RESTRICT pgx,
        const scalar_t *const SFEM_RESTRICT pgy,
        const scalar_t *const SFEM_RESTRICT pgz,
        geom_t **const SFEM_RESTRICT points,
        const scalar_t *const SFEM_RESTRICT qgx,
        const scalar_t *const SFEM_RESTRICT qgy,
        const scalar_t *const SFEM_RESTRICT qgz,
        scalar_t *const SFEM_RESTRICT rc_out,
        scalar_t *const SFEM_RESTRICT rx,
        scalar_t *const SFEM_RESTRICT ry,
        scalar_t *const SFEM_RESTRICT rz,
        const scalar_t *const SFEM_RESTRICT ux_src,
        const scalar_t *const SFEM_RESTRICT uy_src,
        const scalar_t *const SFEM_RESTRICT uz_src,
                                              // Which optional terms are on, resolved once per
                                              // solve by the caller rather than per sweep here:
                                              // cvfem_hex8_extras_of reads the mesh, which this
                                              // kernel is not meant to name.
                                              const Hex8Extras &opt, const scalar_t rho, const scalar_t mu) {
    reset_residual(nnodes, rx, ry, rz, rc_out);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        scalar_t         ux[8], uy[8], uz[8], p[8], r[CVFEM_HEX8_N_DOF];
        Hex8ExtraScratch ex;
        ex.load(mesh_elems, points, face_mask, pgx, pgy, pgz, qgx, qgy, qgz, ux_src, uy_src, uz_src, adj_ptr, det_ptr, opt, e);
        if (!opt.with_rc && !opt.with_bnd) gather_element_coords(mesh_elems, points, e, ex.x, ex.y, ex.z);
        gather_element_fields(mesh_elems, ux_src, uy_src, uz_src, pres, e, ux, uy, uz, p);
        cvfem_hex8_ns_upwind_residual_isoparam(rho, mu, ex.x, ex.y, ex.z, ux, uy, uz, p, r, ex.rc);

        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const idx_t g = mesh_elems[a][e];
            atomic_add(rx, g, r[a * 4 + 0]);
            atomic_add(ry, g, r[a * 4 + 1]);
            atomic_add(rz, g, r[a * 4 + 2]);
            atomic_add(rc_out, g, r[a * 4 + 3]);
        }
    }
}

static SFEM_NOINLINE void assemble_jacobian_atomic_isoparam(
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
        scalar_t *const SFEM_RESTRICT values, const scalar_t rho, const scalar_t mu) {
    // zero_bsr4 inlined, so that this sweep names the matrix arrays it already writes rather
    // than the BSR4 object that owns them. Identical work: that function is this zeroing plus
    // the phase probe, which the macros carry here.
    CVFEM_PHASE_CLOCK(_tz);
    cvfem_zero_scalars(values, bsr_nnz * 16);
    CVFEM_PHASE_GLOBAL(_tz, PH_ZERO);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < nelements; ++e)
        cvfem_hex8_assemble_element_isoparam<true>(mesh_elems, points, face_mask, adj_ptr, det_ptr,
                                                   pres, pgx, pgy, pgz, qgx, qgy, qgz, ux_src,
                                                   uy_src, uz_src, opt, slots, e, rho, mu, values);
}

// Generated (CSE) kernels on isoparametric geometry. The affine SymPy kernels beat the
// hand-written ones because all twelve faces share one adjugate, so CSE has a great deal
// to factor out. Isoparametrically each face carries its own geometry and there is much
// less to share -- these exist to measure how much of the advantage survives.
static SFEM_NOINLINE void apply_residual_atomic_isoparam_sympy(
        // The staging objects are gone; what this sweep reads out of them is what it takes.
        idx_t **const SFEM_RESTRICT mesh_elems,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const scalar_t *const SFEM_RESTRICT pres,
        geom_t **const SFEM_RESTRICT points,
        scalar_t *const SFEM_RESTRICT rc_out,
        scalar_t *const SFEM_RESTRICT rx,
        scalar_t *const SFEM_RESTRICT ry,
        scalar_t *const SFEM_RESTRICT rz,
        const scalar_t *const SFEM_RESTRICT ux_src,
        const scalar_t *const SFEM_RESTRICT uy_src,
        const scalar_t *const SFEM_RESTRICT uz_src,
                                                               const scalar_t rho,
                                                               const scalar_t mu) {
    reset_residual(nnodes, rx, ry, rz, rc_out);
#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        scalar_t x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8], r[CVFEM_HEX8_N_DOF];
        gather_element_coords(mesh_elems, points, e, x, y, z);
        gather_element_fields(mesh_elems, ux_src, uy_src, uz_src, pres, e, ux, uy, uz, p);
        cvfem_hex8_ns_upwind_sympy_residual_isoparam(rho, mu, x, y, z, ux, uy, uz, p, r);
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const idx_t g = mesh_elems[a][e];
            atomic_add(rx, g, r[a * 4 + 0]);
            atomic_add(ry, g, r[a * 4 + 1]);
            atomic_add(rz, g, r[a * 4 + 2]);
            atomic_add(rc_out, g, r[a * 4 + 3]);
        }
    }
}

static SFEM_NOINLINE void assemble_jacobian_atomic_isoparam_sympy(
        // The staging objects are gone; what this sweep reads out of them is what it takes.
        idx_t **const SFEM_RESTRICT mesh_elems,
        const ptrdiff_t nelements,
        const scalar_t *const SFEM_RESTRICT pres,
        geom_t **const SFEM_RESTRICT points,
        const scalar_t *const SFEM_RESTRICT ux_src,
        const scalar_t *const SFEM_RESTRICT uy_src,
        const scalar_t *const SFEM_RESTRICT uz_src,
                                                                  
        const count_t *const SFEM_RESTRICT slots,
        const ptrdiff_t bsr_nnz,
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
        gather_element_coords(mesh_elems, points, e, x, y, z);
        gather_element_fields(mesh_elems, ux_src, uy_src, uz_src, pres, e, ux, uy, uz, p);
        cvfem_hex8_ns_upwind_sympy_jacobian_add_bsr_slots_isoparam(
                rho, mu, x, y, z, ux, uy, uz, slots + (size_t)e * 64, values);
        (void)p;
    }
}

static SFEM_NOINLINE void assemble_diag_atomic_isoparam(
        // The staging objects are gone; what this sweep reads out of them is what it takes.
        const scalar_t box_lx,
        const scalar_t box_ly,
        const scalar_t box_lz,
        const scalar_t *const *const SFEM_RESTRICT adj_ptr,
        const ptrdiff_t *const SFEM_RESTRICT bnd_elems,
        const scalar_t *const SFEM_RESTRICT det_ptr,
        idx_t **const SFEM_RESTRICT mesh_elems,
        const uint8_t *const SFEM_RESTRICT face_mask,
        const uint8_t *const SFEM_RESTRICT face_mask_eff,
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
                                              // How many boundary elements bnd_elems was
                                              // compacted to. opt.with_bnd already answers the
                                              // other half -- it IS !face_mask.empty() -- so only
                                              // the count has to be told.
                                              const ptrdiff_t n_bnd,
                                              // Which optional terms are on, resolved once per
                                              // solve by the caller rather than per sweep here:
                                              // cvfem_hex8_extras_of reads the mesh, which this
                                              // kernel is not meant to name.
                                              const Hex8Extras &opt,
                                                        const scalar_t        rho,
                                                        const scalar_t        mu,
                                                        scalar_t *const SFEM_RESTRICT diag) {
    // The buffer arrives sized. It used to be a std::vector& that this sweep called .assign() on,
    // which is an allocation inside a kernel -- and a kernel that allocates cannot be handed a
    // device buffer or a sub-range. The caller sizes and zeroes it.
    scalar_t *const SFEM_RESTRICT values = diag;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        ptrdiff_t        sl[64];
        scalar_t         ux[8], uy[8], uz[8], p[8];
        Hex8ExtraScratch ex;
        ex.load(mesh_elems, points, face_mask, pgx, pgy, pgz, qgx, qgy, qgz, ux_src, uy_src, uz_src, adj_ptr, det_ptr, opt, e);
        if (!opt.with_rc && !opt.with_bnd) gather_element_coords(mesh_elems, points, e, ex.x, ex.y, ex.z);
        diag_node_slots(mesh_elems, e, sl);
        gather_element_fields(mesh_elems, ux_src, uy_src, uz_src, pres, e, ux, uy, uz, p);
        cvfem_hex8_ns_upwind_jacobian_add_slots_isoparam<true>(
                rho, mu, ex.x, ex.y, ex.z, ux, uy, uz, sl, values, ex.rc, p);
    }
    assemble_diag_boundary_scs_pass(box_lx, box_ly, box_lz, adj_ptr, bnd_elems, det_ptr, mesh_elems, face_mask_eff, pres, points, ux_src, uy_src, uz_src, opt.with_bnd, n_bnd, rho, mu, 1, diag);
}
